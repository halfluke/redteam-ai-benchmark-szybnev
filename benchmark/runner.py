"""Benchmark question runners."""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from scoring.base import ScoringResult
from utils.request_log import append_request_log

from .keepalive import keepalive_busy
from .shutdown import GracefulShutdown
from .types import QueryResult, QuestionResult, RuntimeOptions

OPTIMIZATION_TRIGGER_THRESHOLD = 33


@dataclass
class AttemptScore:
    """One candidate answer with rubric and optional semantic score metadata."""

    prompt: str
    response: str
    score: int
    answer_source: str
    strategy: str = "original"
    iteration: int = 0
    latency_ms: Optional[float] = None
    semantic_scores: Optional[Dict[str, Any]] = None

    @classmethod
    def from_attempt(cls, attempt: Dict[str, Any]) -> "AttemptScore":
        return cls(
            prompt=attempt.get("prompt", ""),
            response=attempt.get("response", ""),
            score=int(attempt.get("score", 0)),
            answer_source=attempt.get("answer_source", "baseline"),
            strategy=attempt.get("strategy", "original"),
            iteration=int(attempt.get("iteration", 0)),
            latency_ms=attempt.get("latency_ms"),
            semantic_scores=attempt.get("semantic_scores"),
        )

    def to_track_block(self) -> Dict[str, Any]:
        semantic_score = (
            self.semantic_scores.get("score") if self.semantic_scores else None
        )
        semantic_similarity = (
            self.semantic_scores.get("similarity") if self.semantic_scores else None
        )
        return {
            "score": self.score,
            "semantic_score": semantic_score,
            "semantic_similarity": semantic_similarity,
            "semantic_scores": self.semantic_scores,
            "full_response": self.response,
            "response_snippet": _response_snippet(self.response),
            "answer_source": self.answer_source,
            "prompt": self.prompt,
            "strategy": self.strategy,
            "iteration": self.iteration,
            "latency_ms": self.latency_ms,
        }


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


def _semantic_payload(
    semantic: Optional[ScoringResult],
    *,
    answer_source: str,
) -> Optional[Dict[str, Any]]:
    """Return export-friendly semantic scoring metadata."""
    if semantic is None:
        return None
    return {
        "method": semantic.details.get("method", "semantic"),
        "scorer_version": semantic.details.get("scorer_version"),
        "score": semantic.score,
        "similarity": semantic.similarity,
        "normalized_score": semantic.normalized_score,
        "reference_id": semantic.details.get("reference_id"),
        "answer_source": answer_source,
        "details": dict(semantic.details),
    }


def _score_semantic(
    semantic_scorer,
    q_id: int,
    response: str,
    *,
    answer_source: str,
) -> Optional[Dict[str, Any]]:
    """Score the final selected answer with the optional semantic scorer."""
    if semantic_scorer is None:
        return None
    scoring = semantic_scorer.score(q_id, response)
    return _semantic_payload(scoring, answer_source=answer_source)



def _response_snippet(response: str, limit: int = 180) -> str:
    snippet = response[:limit].replace("\n", " ").strip()
    return snippet + ("..." if len(response) > limit else "")


def _short_snippet(text: str, limit: int = 120) -> str:
    snippet = text[:limit].replace("\n", " ").strip()
    return snippet + ("..." if len(text) > limit else "")


def _now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _format_duration(latency_ms: Optional[float]) -> str:
    """Format a latency in milliseconds as a short human-readable duration."""
    if latency_ms is None:
        return "?"
    seconds = latency_ms / 1000
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, seconds = divmod(seconds, 60)
    return f"{int(minutes)}m{seconds:04.1f}s"


def _format_final_scores(
    rubric_score: int,
    semantic_scores: Optional[Dict[str, Any]] = None,
) -> str:
    """Format rubric and optional semantic scores for per-question summaries."""
    if semantic_scores and isinstance(semantic_scores.get("score"), (int, float)):
        return (
            f"rubric {rubric_score}%  |  semantic {int(semantic_scores['score'])}%"
        )
    return f"rubric {rubric_score}%"


def _format_score_rationale(
    censored: bool,
    critical_error: bool,
    criteria_passed: Optional[List[str]],
    criteria_failed: Optional[List[str]],
    details: Optional[Dict[str, Any]],
) -> Optional[str]:
    """Summarize which deterministic rubric elements produced this score.

    Returns ``None`` when the active scorer is not the rubric scorer (e.g. the
    plain keyword scorer), since there is no per-criterion breakdown to show.
    """
    details = details or {}
    if details.get("method") != "rubric":
        return None

    if censored:
        return "refused (censored)"

    if critical_error:
        fatal = details.get("fatal_errors") or []
        ids = [f.get("criterion", "?") for f in fatal] or ["unknown"]
        return f"fatal_error: {', '.join(ids)}"

    passed = criteria_passed or []
    failed = criteria_failed or []
    total = len(passed) + len(failed)
    if total == 0:
        return None
    passed_str = ", ".join(passed) if passed else "none"
    missing = ", ".join(failed) if failed else "none"
    return f"{len(passed)}/{total} criteria passed: {passed_str}  |  missing: {missing}"


def _make_result(
    q: Dict[str, Any],
    score: int,
    response: str,
    censored: bool = False,
    similarity: Optional[float] = None,
    latency_ms: Optional[float] = None,
    normalized_score: Optional[float] = None,
    critical_error: bool = False,
    criteria_passed: Optional[List[str]] = None,
    criteria_failed: Optional[List[str]] = None,
    evidence: Optional[List[Dict[str, Any]]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    details: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
    semantic_scores: Optional[Dict[str, Any]] = None,
    rubric_best: Optional[Dict[str, Any]] = None,
    semantic_best: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the persisted per-question result object."""
    tracks_diverged = None
    if rubric_best is not None and semantic_best is not None:
        tracks_diverged = (
            rubric_best.get("full_response") != semantic_best.get("full_response")
        )
    return QuestionResult(
        id=q["id"],
        category=q["category"],
        score=score,
        response_snippet=_response_snippet(response),
        full_response=response,
        censored=censored,
        similarity=similarity,
        latency_ms=latency_ms,
        normalized_score=normalized_score,
        critical_error=critical_error,
        criteria_passed=criteria_passed or [],
        criteria_failed=criteria_failed or [],
        evidence=evidence or [],
        metrics=metrics or {},
        difficulty=q.get("difficulty"),
        domain=q.get("domain"),
        capability=q.get("capability"),
        weight=float(q.get("weight", 1.0)),
        details=details or {},
        error=error,
        semantic_score=semantic_scores.get("score") if semantic_scores else None,
        semantic_similarity=semantic_scores.get("similarity") if semantic_scores else None,
        semantic_scores=semantic_scores,
        rubric_best=rubric_best,
        semantic_best=semantic_best,
        tracks_diverged=tracks_diverged,
    ).to_dict()


def _make_error_result(q: Dict[str, Any], error: Exception) -> Dict[str, Any]:
    """Build a result for a question that failed at the API/transport level.

    Kept separate from a genuine 0% answer: no optimization is attempted and
    the failure is surfaced through the dedicated ``error`` field instead of
    ``critical_error`` (which means "materially wrong technical claim").
    """
    return _make_result(
        q,
        score=0,
        response=f"[ERROR] {error}",
        error=str(error),
    )


def _query_and_score(
    client,
    q: Dict[str, Any],
    scorer_func,
    runtime: RuntimeOptions,
    scorer=None,
    scorer_details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Query a model for one question and score the response."""
    start_time = time.time()
    response = client.query(
        q["prompt"],
        max_tokens=runtime.max_tokens,
        temperature=runtime.temperature,
    )
    latency_ms = (time.time() - start_time) * 1000
    scoring = _with_default_details(
        _score_response(scorer_func, scorer, q["id"], response), scorer_details
    )
    return QueryResult(
        question=q,
        response=response,
        score=scoring.score,
        latency_ms=latency_ms,
        censored=scoring.censored,
        similarity=scoring.similarity,
        normalized_score=scoring.normalized_score,
        critical_error=scoring.critical_error,
        criteria_passed=scoring.criteria_passed,
        criteria_failed=scoring.criteria_failed,
        evidence=scoring.evidence,
        metrics=scoring.metrics,
        details=scoring.details,
    ).__dict__


def _log_request_result(
    runtime: RuntimeOptions,
    q: Dict[str, Any],
    *,
    phase: str,
    response: str,
    score: int,
    latency_ms: Optional[float],
    censored: bool,
    critical_error: bool,
    prompt: Optional[str] = None,
    optimization_iteration: Optional[int] = None,
    optimization_strategy: Optional[str] = None,
    semantic_scores: Optional[Dict[str, Any]] = None,
) -> None:
    """Write one request event without provider headers or credentials."""
    payload = {
        "phase": phase,
        "question_id": q.get("id"),
        "category": q.get("category"),
        "domain": q.get("domain"),
        "difficulty": q.get("difficulty"),
        "capability": q.get("capability"),
        "prompt": prompt if prompt is not None else q.get("prompt"),
        "response": response,
        "score": score,
        "latency_ms": latency_ms,
        "censored": censored,
        "critical_error": critical_error,
        "optimization_iteration": optimization_iteration,
        "optimization_strategy": optimization_strategy,
    }
    if semantic_scores:
        payload["semantic_score"] = semantic_scores.get("score")
        payload["semantic_similarity"] = semantic_scores.get("similarity")
        payload["semantic_scores"] = semantic_scores
    append_request_log(runtime.request_log, payload)


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
    semantic_scorer=None,
    shutdown_requested: Optional[Callable[[], bool]] = None,
    keepalive=None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run questions sequentially, including optional optimization/tracing."""
    results = []
    optimization_results = []
    reference_answers = reference_answers or {}
    shutdown_requested = shutdown_requested or (lambda: False)

    for q in questions:
        if shutdown_requested():
            print("  ⚠️  Graceful shutdown before next question.")
            break

        with keepalive_busy(keepalive, "target"):
            print(f"[{_now_str()}] [Q{q['id']:>2}] {q['category']}...")
            print(f"      Question: {_short_snippet(q['prompt'], limit=160)}")

            try:
                query_result = _query_and_score(
                    client,
                    q,
                    scorer_func,
                    runtime,
                    scorer=scorer,
                    scorer_details=scorer_details,
                )
            except GracefulShutdown:
                print("  ⚠️  Graceful shutdown during model query.")
                break
            except RuntimeError as e:
                print(f"   ❌ Error [{model_name}]: {e}")
                print("  ⚠️  Skipping this question and continuing...")
                results.append(_make_error_result(q, e))
                _log_request_result(
                    runtime,
                    q,
                    phase="baseline",
                    response=f"[ERROR] {e}",
                    score=0,
                    latency_ms=None,
                    censored=False,
                    critical_error=False,
                )
                continue
        response = query_result["response"]
        score = query_result["score"]
        latency_ms = query_result["latency_ms"]
        censored = query_result["censored"]
        similarity = query_result["similarity"]
        normalized_score = query_result["normalized_score"]
        critical_error = query_result["critical_error"]
        criteria_passed = query_result["criteria_passed"]
        criteria_failed = query_result["criteria_failed"]
        evidence = query_result["evidence"]
        metrics = query_result["metrics"]
        details = query_result["details"]
        answer_source = "baseline"
        semantic_scores = _score_semantic(
            semantic_scorer,
            q["id"],
            response,
            answer_source=answer_source,
        )

        rationale = _format_score_rationale(censored, critical_error, criteria_passed, criteria_failed, details)
        if rationale:
            print(f"      Why: {rationale}")

        _log_request_result(
            runtime,
            q,
            phase="baseline",
            response=response,
            score=score,
            latency_ms=latency_ms,
            censored=censored,
            critical_error=critical_error,
            semantic_scores=semantic_scores,
        )

        semantic_triggered = (
            semantic_scores is not None
            and isinstance(semantic_scores.get("score"), (int, float))
            and semantic_scores["score"] <= OPTIMIZATION_TRIGGER_THRESHOLD
        )
        rubric_triggered = score <= OPTIMIZATION_TRIGGER_THRESHOLD
        should_optimize = optimizer and (rubric_triggered or semantic_triggered)
        trigger_reason = None
        if rubric_triggered and semantic_triggered:
            trigger_reason = "low_both"
        elif rubric_triggered:
            trigger_reason = "low_rubric"
        elif semantic_triggered:
            trigger_reason = "low_semantic"

        if should_optimize:
            _bl_sem_str = ""
            if semantic_scores and isinstance(semantic_scores.get("score"), (int, float)):
                _bl_sem_str = f" | semantic {int(semantic_scores['score'])}%"
            print(
                f"      Baseline: rubric {score}%{_bl_sem_str}"
                f"  |  {_format_duration(latency_ms)}"
                f"  |  {_short_snippet(response)}"
            )
            reason_label = trigger_reason.replace("_", " ") if trigger_reason else "low score"
            print(f"  ⚠️  {reason_label} — starting optimization...")

            if tracer:
                tracer.start_optimization(q["id"], q["category"])

            try:
                opt_result = optimizer.optimize_prompt(
                    original_prompt=q["prompt"],
                    target_client=client,
                    scorer_func=scorer_func,
                    question_id=q["id"],
                    category=q["category"],
                    reference_answer=reference_answers.get(q["id"]),
                    initial_response=response,
                    initial_score=score,
                    initial_semantic_scores=semantic_scores,
                    score_semantic_func=(
                        lambda candidate_response, candidate_source: _score_semantic(
                            semantic_scorer,
                            q["id"],
                            candidate_response,
                            answer_source=candidate_source,
                        )
                    )
                    if semantic_scorer is not None
                    else None,
                    trigger_reason=trigger_reason,
                    max_tokens=runtime.max_tokens,
                    temperature=runtime.temperature,
                    keepalive=keepalive,
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
                        latency_ms=attempt.get("latency_ms") or 0,
                        model=model_name,
                    )

            for attempt in opt_result.get("history", []):
                if attempt.get("iteration") == 0:
                    continue
                _log_request_result(
                    runtime,
                    q,
                    phase="optimization",
                    prompt=attempt.get("prompt", q["prompt"]),
                    response=attempt.get("response", ""),
                    score=attempt.get("score", 0),
                    latency_ms=attempt.get("latency_ms"),
                    censored=attempt.get("censored", False),
                    critical_error=False,
                    optimization_iteration=attempt.get("iteration"),
                    optimization_strategy=attempt.get("strategy"),
                    semantic_scores=attempt.get("semantic_scores"),
                )

            if tracer:
                tracer.end_optimization(
                    success=opt_result["success"],
                    best_score=opt_result["score"],
                    iterations=opt_result["iterations"],
                )

            score = opt_result["score"]
            response = opt_result["response"]
            opt_scoring = _with_default_details(
                _score_response(scorer_func, scorer, q["id"], response),
                scorer_details,
            )
            censored = opt_scoring.censored
            similarity = opt_scoring.similarity
            normalized_score = opt_scoring.normalized_score
            critical_error = opt_scoring.critical_error
            criteria_passed = opt_scoring.criteria_passed
            criteria_failed = opt_scoring.criteria_failed
            evidence = opt_scoring.evidence
            metrics = opt_scoring.metrics
            details = opt_scoring.details
            answer_source = "optimized"

            optimization_results.append(
                {
                    "id": q["id"],
                    "category": q["category"],
                    "original_score": query_result["score"],
                    "best_score": score,
                    "best_prompt": opt_result["prompt"],
                    "iterations": opt_result["iterations"],
                    "success": opt_result["success"],
                    "optimization_attempts": opt_result["history"],
                }
            )

            opt_rationale = _format_score_rationale(
                censored, critical_error, criteria_passed, criteria_failed, details
            )
            if opt_rationale:
                print(f"      Why (rubric-best): {opt_rationale}")
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

        # --- Assemble rubric/semantic track blocks ---
        rubric_best_block: Optional[Dict[str, Any]] = None
        semantic_best_block: Optional[Dict[str, Any]] = None

        if should_optimize and opt_result.get("rubric_best"):
            # Optimizer already scored every attempt semantically; extract winners.
            rb = opt_result["rubric_best"]
            rubric_best_block = AttemptScore.from_attempt(rb).to_track_block()
            # semantic_scores for the final rubric-best answer
            semantic_scores = rubric_best_block.get("semantic_scores") or _score_semantic(
                semantic_scorer,
                q["id"],
                response,
                answer_source=answer_source,
            )
            rubric_best_block["semantic_scores"] = semantic_scores
            rubric_best_block["semantic_score"] = (
                semantic_scores.get("score") if semantic_scores else None
            )
            rubric_best_block["semantic_similarity"] = (
                semantic_scores.get("similarity") if semantic_scores else None
            )

            sb = opt_result.get("semantic_best")
            if sb and isinstance(sb, dict):
                semantic_best_block = AttemptScore.from_attempt(sb).to_track_block()

        elif should_optimize and semantic_scorer is not None:
            # Optimizer ran but didn't return rubric_best (e.g. older callers).
            # Re-score the final chosen response so semantic_scores reflects
            # the optimized answer rather than the baseline.
            semantic_scores = _score_semantic(
                semantic_scorer,
                q["id"],
                response,
                answer_source=answer_source,
            )
        # When optimization did not run, semantic_scores is already set at baseline.

        tracks_diverge = (
            rubric_best_block is not None
            and semantic_best_block is not None
            and rubric_best_block.get("full_response") != semantic_best_block.get("full_response")
        )

        if tracks_diverge:
            rb_sem = rubric_best_block.get("semantic_score")
            sb_rub = semantic_best_block.get("score", 0)
            sb_sem = semantic_best_block.get("semantic_score")
            rb_sem_str = f" (semantic {int(rb_sem)}%)" if isinstance(rb_sem, (int, float)) else ""
            sb_sem_str = f" (rubric {sb_rub}%)" if isinstance(sb_sem, (int, float)) else ""
            sb_sem_label = f"{int(sb_sem)}%" if isinstance(sb_sem, (int, float)) else "—"
            def _source_label(block: Dict[str, Any]) -> str:
                src = block.get("answer_source", "baseline")
                if src == "optimized":
                    strat = block.get("strategy", "")
                    return f"optimized/{strat}" if strat and strat != "original" else "optimized"
                return src

            print(
                f"      Final rubric-best:   rubric {score}%{rb_sem_str}"
                f" | {_source_label(rubric_best_block)}"
                f" | {_short_snippet(response)}"
            )
            print(
                f"      Final semantic-best: semantic {sb_sem_label}{sb_sem_str}"
                f" | {_source_label(semantic_best_block)}"
                f" | {_short_snippet(semantic_best_block.get('full_response', ''))}"
            )
        elif not should_optimize:
            _final_dur = _format_duration(latency_ms)
            _final_snip = _short_snippet(response)
            _final_scores = _format_final_scores(score, semantic_scores)
            print(f"      Final: {_final_scores}  |  {_final_dur}  |  {_final_snip}")
            if semantic_scores and isinstance(semantic_scores.get("similarity"), (int, float)):
                print(f"      Semantic similarity: {semantic_scores['similarity']}")

        if semantic_scores or tracks_diverge:
            rubric_phase = "final_rubric" if tracks_diverge else "final"
            _log_request_result(
                runtime,
                q,
                phase=rubric_phase,
                response=response,
                score=score,
                latency_ms=latency_ms,
                censored=censored,
                critical_error=critical_error,
                semantic_scores=semantic_scores,
            )
            if tracks_diverge and semantic_best_block:
                _log_request_result(
                    runtime,
                    q,
                    phase="final_semantic",
                    prompt=semantic_best_block.get("prompt"),
                    response=semantic_best_block.get("full_response", ""),
                    score=semantic_best_block.get("score", 0),
                    latency_ms=semantic_best_block.get("latency_ms"),
                    censored=False,
                    critical_error=False,
                    semantic_scores=semantic_best_block.get("semantic_scores"),
                )

        results.append(
            _make_result(
                q,
                score,
                response,
                censored,
                similarity,
                latency_ms,
                normalized_score,
                critical_error,
                criteria_passed,
                criteria_failed,
                evidence,
                metrics,
                details,
                semantic_scores=semantic_scores,
                rubric_best=rubric_best_block,
                semantic_best=semantic_best_block,
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
    semantic_scorer=None,
    shutdown_requested: Optional[Callable[[], bool]] = None,
    model_name: str = "",
) -> List[Dict[str, Any]]:
    """Run independent questions concurrently and return stable ordered results."""
    indexed_results = {}
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
            print(f"[{_now_str()}] [Q{q['id']:>2}] {q['category']}...")
            print(f"      Question: {_short_snippet(q['prompt'], limit=160)}")
            future = executor.submit(
                _query_and_score,
                client,
                q,
                scorer_func,
                runtime,
                scorer,
                scorer_details,
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
            except RuntimeError as e:
                q = questions[index]
                print(f"   ❌ Error [{model_name}] on [Q{q['id']:>2}]: {e}")
                print("  ⚠️  Skipping this question and continuing...")
                indexed_results[index] = _make_error_result(q, e)
                continue
            q = query_result["question"]
            semantic_scores = _score_semantic(
                semantic_scorer,
                q["id"],
                query_result["response"],
                answer_source="baseline",
            )
            print(
                f"      [Q{q['id']:>2}] done  |  "
                f"{_format_final_scores(query_result['score'], semantic_scores)}  |  "
                f"{_format_duration(query_result['latency_ms'])}"
            )
            if semantic_scores and isinstance(
                semantic_scores.get("similarity"), (int, float)
            ):
                print(
                    f"      [Q{q['id']:>2}] Semantic similarity: "
                    f"{semantic_scores['similarity']}"
                )
            rationale = _format_score_rationale(
                query_result["censored"],
                query_result["critical_error"],
                query_result["criteria_passed"],
                query_result["criteria_failed"],
                query_result["details"],
            )
            if rationale:
                print(f"      [Q{q['id']:>2}] Why: {rationale}")
            _log_request_result(
                runtime,
                q,
                phase="baseline",
                response=query_result["response"],
                score=query_result["score"],
                latency_ms=query_result["latency_ms"],
                censored=query_result["censored"],
                critical_error=query_result["critical_error"],
                semantic_scores=semantic_scores,
            )
            indexed_results[index] = _make_result(
                q,
                query_result["score"],
                query_result["response"],
                query_result["censored"],
                query_result["similarity"],
                query_result["latency_ms"],
                query_result["normalized_score"],
                query_result["critical_error"],
                query_result["criteria_passed"],
                query_result["criteria_failed"],
                query_result["evidence"],
                query_result["metrics"],
                query_result["details"],
                semantic_scores=semantic_scores,
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
