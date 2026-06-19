"""Benchmark question runners."""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

from optimization.triggers import (
    build_optimization_scorer_func,
    build_score_detail_func,
    optimization_trigger_reason,
    should_run_optimization,
)
from scoring.base import ScoringResult
from scoring.factory import MultiScorerBundle

from .request_logging import collect_request_diagnostics, log_request_event, print_optimizer_variants
from .shutdown import GracefulShutdown
from .keepalive import keepalive_busy
from .types import QueryResult, QuestionResult, RuntimeOptions


def _sleep_between_requests(rate_limit_delay: float) -> None:
    """Sleep only when rate limiting is explicitly positive."""
    if rate_limit_delay > 0:
        time.sleep(rate_limit_delay)


def _score_response(scorer_func, scorer, q_id: int, response: str) -> ScoringResult:
    """Score a response while preserving metadata when the scorer exposes it."""
    if scorer is not None and hasattr(scorer, "score"):
        result = scorer.score(q_id, response)
        if isinstance(result, ScoringResult):
            return result

    return ScoringResult(score=scorer_func(q_id, response))


def _with_default_details(
    scoring: ScoringResult, default_details: Optional[Dict[str, Any]]
) -> ScoringResult:
    """Attach scorer bundle details when the scorer does not return its own."""
    if default_details and not scoring.details:
        scoring.details = dict(default_details)
    return scoring


def _text_snippet(text: str, limit: int = 160) -> str:
    snippet = text[:limit].replace("\n", " ").strip()
    return snippet + ("..." if len(text) > limit else "")


def _response_snippet(response: str, limit: int = 180) -> str:
    return _text_snippet(response, limit)


def _print_question_result(
    score: int,
    response: str,
    scores: Optional[Dict[str, int]] = None,
    *,
    prompt: Optional[str] = None,
    reference_answer: Optional[str] = None,
) -> None:
    """Print prompt/reference context, scores, and model response snippet."""
    if prompt:
        print(f"   prompt: {_text_snippet(prompt)}", flush=True)
    if reference_answer:
        print(f"   reference (semantic): {_text_snippet(reference_answer)}", flush=True)
    snippet = _response_snippet(response) if response else "(no response)"
    if scores and len(scores) > 1:
        parts = " | ".join(f"{name}: {value}%" for name, value in scores.items())
        print(f"   → {parts} | {snippet}", flush=True)
    else:
        print(f"   → Score: {score}% | {snippet}", flush=True)


def _build_score_maps(
    scoring_results: Dict[str, ScoringResult],
) -> tuple[Dict[str, int], Dict[str, float]]:
    scores = {name: result.score for name, result in scoring_results.items()}
    similarities = {
        name: result.similarity
        for name, result in scoring_results.items()
        if result.similarity is not None
    }
    return scores, similarities


def _score_response_text(
    scorer_func,
    scorer,
    q_id: int,
    response: str,
    scorer_details: Optional[Dict[str, Any]],
    multi_scorer_bundle: Optional[MultiScorerBundle] = None,
) -> tuple[ScoringResult, Dict[str, int], Dict[str, float], Dict[str, Any]]:
    """Score one response with single or multiple scorers."""
    if multi_scorer_bundle is not None and multi_scorer_bundle.is_multi:
        scoring_results = multi_scorer_bundle.score_all(q_id, response)
        primary_name = multi_scorer_bundle.primary_method
        primary = scoring_results[primary_name]
        scores, similarities = _build_score_maps(scoring_results)
        details = {
            "scorers": {
                name: {
                    "score": result.score,
                    "censored": result.censored,
                    "similarity": result.similarity,
                    **(result.details or {}),
                }
                for name, result in scoring_results.items()
            }
        }
        return primary, scores, similarities, details

    scoring = _with_default_details(
        _score_response(scorer_func, scorer, q_id, response), scorer_details
    )
    return scoring, {}, {}, scoring.details or {}


def _collect_request_diagnostics(client) -> Dict[str, Any]:
    """Return the latest provider diagnostics attached to the client."""
    return collect_request_diagnostics(client)


def _log_request_event(
    runtime: RuntimeOptions,
    question: Dict[str, Any],
    *,
    score: Optional[int] = None,
    scores: Optional[Dict[str, int]] = None,
    error: Optional[str] = None,
    request: Optional[Dict[str, Any]] = None,
    phase: str = "baseline",
    optimization_iteration: Optional[int] = None,
    optimization_strategy: Optional[str] = None,
    variant_source: Optional[str] = None,
) -> None:
    """Print timing details and optionally append to the JSONL request log."""
    log_request_event(
        runtime,
        question,
        score=score,
        scores=scores,
        error=error,
        request=request,
        phase=phase,
        optimization_iteration=optimization_iteration,
        optimization_strategy=optimization_strategy,
        variant_source=variant_source,
    )


def _attach_request_diagnostics(
    details: Optional[Dict[str, Any]], client
) -> Dict[str, Any]:
    """Merge request diagnostics into question result details."""
    merged = dict(details or {})
    request = _collect_request_diagnostics(client)
    if request:
        merged["request"] = request
    return merged


def _make_result(
    q: Dict[str, Any],
    score: int,
    response: str,
    censored: bool = False,
    similarity: Optional[float] = None,
    details: Optional[Dict[str, Any]] = None,
    scores: Optional[Dict[str, int]] = None,
    similarities: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Build the persisted per-question result object."""
    result = QuestionResult(
        id=q["id"],
        category=q["category"],
        score=score,
        response_snippet=_response_snippet(response),
        full_response=response,
        censored=censored,
        similarity=similarity,
        details=details or {},
    ).to_dict()
    if scores:
        result["scores"] = scores
    if similarities:
        result["similarities"] = similarities
    return result


def _query_and_score(
    client,
    q: Dict[str, Any],
    scorer_func,
    runtime: RuntimeOptions,
    scorer=None,
    scorer_details: Optional[Dict[str, Any]] = None,
    multi_scorer_bundle: Optional[MultiScorerBundle] = None,
    keepalive=None,
) -> Dict[str, Any]:
    """Query a model for one question and score the response."""
    start_time = time.time()
    try:
        with keepalive_busy(keepalive, "target"):
            response = client.query(
                q["prompt"],
                max_tokens=runtime.max_tokens,
                temperature=runtime.temperature,
            )
    except RuntimeError as e:
        empty_scores = (
            {method: 0 for method in multi_scorer_bundle.methods}
            if multi_scorer_bundle is not None and multi_scorer_bundle.is_multi
            else None
        )
        return QueryResult(
            question=q,
            response="",
            score=0,
            latency_ms=(time.time() - start_time) * 1000,
            censored=False,
            similarity=None,
            details=_attach_request_diagnostics({"error": str(e)}, client),
            scores=empty_scores,
        ).__dict__

    latency_ms = (time.time() - start_time) * 1000
    scoring, score_map, similarity_map, score_details = _score_response_text(
        scorer_func,
        scorer,
        q["id"],
        response,
        scorer_details,
        multi_scorer_bundle=multi_scorer_bundle,
    )
    details = _attach_request_diagnostics(score_details, client)
    return QueryResult(
        question=q,
        response=response,
        score=scoring.score,
        latency_ms=latency_ms,
        censored=scoring.censored,
        similarity=scoring.similarity,
        details=details,
        scores=score_map or None,
        similarities=similarity_map or None,
    ).__dict__


def _run_questions_sequential(
    questions: List[Dict[str, Any]],
    client,
    scorer_func,
    runtime: RuntimeOptions,
    model_name: str,
    optimizer=None,
    reference_answers: Optional[Dict[int, str]] = None,
    tracer=None,
    scorer=None,
    scorer_details: Optional[Dict[str, Any]] = None,
    multi_scorer_bundle: Optional[MultiScorerBundle] = None,
    shutdown_requested: Optional[Callable[[], bool]] = None,
    optimization_trigger: str = "keyword_zero",
    keepalive=None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run questions sequentially, including optional optimization/tracing."""
    results = []
    optimization_results = []
    reference_answers = reference_answers or {}
    shutdown_requested = shutdown_requested or (lambda: False)
    is_multi = multi_scorer_bundle is not None and multi_scorer_bundle.is_multi
    optimization_scorer_func = build_optimization_scorer_func(
        scorer_func,
        multi_scorer_bundle,
        trigger=optimization_trigger,
    )
    score_detail_func = build_score_detail_func(scorer_func, multi_scorer_bundle)

    for q in questions:
        if shutdown_requested():
            print("  ⚠️  Graceful shutdown before next question.")
            break

        print(f"[Q{q['id']:>2}] {q['category']}...")

        try:
            query_result = _query_and_score(
                client,
                q,
                scorer_func,
                runtime,
                scorer=scorer,
                scorer_details=scorer_details,
                multi_scorer_bundle=multi_scorer_bundle,
                keepalive=keepalive,
            )
        except GracefulShutdown:
            print("  ⚠️  Graceful shutdown during model query.")
            break

        if query_result["details"].get("error"):
            error = query_result["details"]["error"]
            print(f"  ❌ Request failed: {error}")
            _log_request_event(
                runtime,
                q,
                score=0,
                error=error,
                request=query_result["details"].get("request"),
            )
            results.append(
                _make_result(
                    q,
                    0,
                    "",
                    details=query_result["details"],
                )
            )
            _print_question_result(
                0,
                "",
                prompt=q.get("prompt"),
                reference_answer=reference_answers.get(q["id"]),
            )
            try:
                _sleep_between_requests(runtime.rate_limit_delay)
            except GracefulShutdown:
                print("  ⚠️  Graceful shutdown during rate-limit delay.")
                break
            continue

        response = query_result["response"]
        score = query_result["score"]
        score_map = query_result.get("scores") or {}
        similarity_map = query_result.get("similarities") or {}
        latency_ms = query_result["latency_ms"]
        censored = query_result["censored"]
        similarity = query_result["similarity"]
        details = query_result["details"]

        _log_request_event(
            runtime,
            q,
            score=score,
            scores=score_map or None,
            request=details.get("request"),
        )

        run_optimization = should_run_optimization(
            trigger=optimization_trigger,
            primary_score=score,
            score_map=score_map or None,
            is_multi=is_multi,
        )
        if run_optimization and optimizer:
            print(
                f"  ⚠️  Starting optimization "
                f"({optimization_trigger_reason(trigger=optimization_trigger, primary_score=score, score_map=score_map or None, is_multi=is_multi)}; "
                f"optimizer max_tokens={optimizer.optimizer_max_tokens})..."
            )

            if tracer:
                tracer.start_optimization(q["id"], q["category"])

            opt_initial_score = optimization_scorer_func(q["id"], response)
            try:
                opt_result = optimizer.optimize_prompt(
                    original_prompt=q["prompt"],
                    target_client=client,
                    scorer_func=optimization_scorer_func,
                    question_id=q["id"],
                    category=q["category"],
                    reference_answer=reference_answers.get(q["id"]),
                    initial_response=response,
                    initial_score=opt_initial_score,
                    max_tokens=runtime.max_tokens,
                    temperature=runtime.temperature,
                    keepalive=keepalive,
                    initial_scores=score_map or None,
                    optimization_trigger=optimization_trigger,
                    is_multi=is_multi,
                    score_detail_func=score_detail_func,
                    request_log_context=(runtime, q) if runtime.request_log else None,
                )
            except GracefulShutdown:
                print("  ⚠️  Graceful shutdown during prompt optimization.")
                break

            if tracer and opt_result.get("history"):
                for attempt in opt_result["history"]:
                    tracer.log_optimization_attempt(
                        iteration=attempt.get("iteration", 0),
                        strategy=attempt.get("strategy", "unknown"),
                        prompt=attempt.get("prompt", ""),
                        response=attempt.get("response", ""),
                        score=attempt.get("score", 0),
                        latency_ms=0,
                        model=model_name,
                    )

            if tracer:
                tracer.end_optimization(
                    success=opt_result["success"],
                    best_score=opt_result["score"],
                    iterations=opt_result["iterations"],
                )

            score = opt_result["score"]
            response = opt_result["response"]
            opt_scoring, score_map, similarity_map, opt_details = _score_response_text(
                scorer_func,
                scorer,
                q["id"],
                response,
                scorer_details,
                multi_scorer_bundle=multi_scorer_bundle,
            )
            censored = opt_scoring.censored
            similarity = opt_scoring.similarity
            details = _attach_request_diagnostics(opt_details, client)

            optimization_results.append(
                {
                    "id": q["id"],
                    "category": q["category"],
                    "original_score": score,
                    "original_scores": score_map or None,
                    "best_score": score,
                    "best_prompt": opt_result["prompt"],
                    "iterations": opt_result["iterations"],
                    "success": opt_result["success"],
                    "optimization_attempts": opt_result["history"],
                    "optimizer_variants": opt_result.get("optimizer_variants"),
                    "optimizer_variant_source": opt_result.get("optimizer_variant_source"),
                }
            )

            print(f"  ✓ Optimization complete: {score}%", flush=True)
            if opt_result.get("optimizer_variants"):
                print_optimizer_variants(
                    opt_result["optimizer_variants"],
                    variant_source=opt_result.get("optimizer_variant_source", "unknown"),
                )
        elif tracer:
            tracer.log_generation(
                question_id=q["id"],
                category=q["category"],
                prompt=q["prompt"],
                response=response,
                score=score,
                latency_ms=latency_ms,
                model=model_name,
            )

        _print_question_result(
            score,
            response,
            score_map or None,
            prompt=q.get("prompt"),
            reference_answer=reference_answers.get(q["id"]),
        )
        results.append(
            _make_result(
                q,
                score,
                response,
                censored,
                similarity,
                details,
                scores=score_map or None,
                similarities=similarity_map or None,
            )
        )
        try:
            _sleep_between_requests(runtime.rate_limit_delay)
        except GracefulShutdown:
            print("  ⚠️  Graceful shutdown during rate-limit delay.")
            break

    return results, optimization_results


def _run_questions_concurrent(
    questions: List[Dict[str, Any]],
    client,
    scorer_func,
    runtime: RuntimeOptions,
    scorer=None,
    scorer_details: Optional[Dict[str, Any]] = None,
    multi_scorer_bundle: Optional[MultiScorerBundle] = None,
    shutdown_requested: Optional[Callable[[], bool]] = None,
    reference_answers: Optional[Dict[int, str]] = None,
) -> List[Dict[str, Any]]:
    """Run independent questions concurrently and return stable ordered results."""
    indexed_results = {}
    reference_answers = reference_answers or {}
    shutdown_requested = shutdown_requested or (lambda: False)
    interrupted = False

    executor = ThreadPoolExecutor(max_workers=runtime.concurrency)
    future_to_index = {}

    try:
        for index, q in enumerate(questions):
            if shutdown_requested():
                interrupted = True
                break
            if index > 0:
                try:
                    _sleep_between_requests(runtime.rate_limit_delay)
                except GracefulShutdown:
                    interrupted = True
                    break
            print(f"[Q{q['id']:>2}] {q['category']}...")
            future = executor.submit(
                _query_and_score,
                client,
                q,
                scorer_func,
                runtime,
                scorer,
                scorer_details,
                multi_scorer_bundle,
            )
            future_to_index[future] = index

        for future in as_completed(future_to_index):
            if shutdown_requested():
                interrupted = True
                break
            index = future_to_index[future]
            try:
                query_result = future.result()
            except GracefulShutdown:
                interrupted = True
                break
            q = query_result["question"]
            if query_result["details"].get("error"):
                error = query_result["details"]["error"]
                print(f"  ❌ Request failed for Q{q['id']}: {error}")
                _log_request_event(
                    runtime,
                    q,
                    score=0,
                    error=error,
                    request=query_result["details"].get("request"),
                )
                indexed_results[index] = _make_result(
                    q,
                    0,
                    "",
                    details=query_result["details"],
                )
                _print_question_result(
                    0,
                    "",
                    prompt=q.get("prompt"),
                    reference_answer=reference_answers.get(q["id"]),
                )
                continue
            indexed_results[index] = _make_result(
                q,
                query_result["score"],
                query_result["response"],
                query_result["censored"],
                query_result["similarity"],
                query_result["details"],
                scores=query_result.get("scores"),
                similarities=query_result.get("similarities"),
            )
            _log_request_event(
                runtime,
                q,
                score=query_result["score"],
                scores=query_result.get("scores"),
                request=query_result["details"].get("request"),
            )
            _print_question_result(
                query_result["score"],
                query_result["response"],
                query_result.get("scores"),
                prompt=q.get("prompt"),
                reference_answer=reference_answers.get(q["id"]),
            )
    except GracefulShutdown:
        interrupted = True
    finally:
        if interrupted:
            print("  ⚠️  Graceful shutdown: cancelling pending benchmark questions.")
            for future in future_to_index:
                future.cancel()
        executor.shutdown(wait=True, cancel_futures=interrupted)

    return [indexed_results[index] for index in sorted(indexed_results)]


def _effective_concurrency(
    requested_concurrency: int,
    optimizer_enabled: bool,
    tracer_enabled: bool,
) -> int:
    """Disable concurrency for paths with shared mutable optimization/tracing state."""
    if requested_concurrency > 1 and (optimizer_enabled or tracer_enabled):
        print(
            "⚠️  Concurrency disabled because prompt optimization or Langfuse tracing is enabled"
        )
        return 1
    return requested_concurrency
