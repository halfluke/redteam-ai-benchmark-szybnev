"""Single-model benchmark orchestration."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from utils.export import get_interpretation

from .metrics import (
    build_track_results,
    summarize_results,
    summarize_semantic_results,
    summarize_track,
    weighted_score,
)
from .runner import (
    _effective_concurrency,
    _run_questions_concurrent,
    _run_questions_sequential,
)
from .types import RuntimeOptions


@dataclass
class SingleModelBenchmarkResult:
    """Completed benchmark state for one model."""

    model_name: str
    results: List[Dict[str, Any]]
    total_score: float
    interpretation: str
    optimization_results: List[Dict[str, Any]] = field(default_factory=list)
    exported: Dict[str, str] = field(default_factory=dict)
    tracer_failed: bool = False
    interrupted: bool = False


def run_single_model_benchmark(
    *,
    questions: List[Dict[str, Any]],
    client,
    model_name: str,
    scorer_bundle,
    runtime: RuntimeOptions,
    optimizer=None,
    semantic_scorer=None,
    reference_answers: Optional[Dict[int, str]] = None,
    tracer_config=None,
    tracer_factory: Optional[Callable[[Any], Any]] = None,
    export_callback: Optional[Callable[..., Dict[str, str]]] = None,
    export_kwargs: Optional[Dict[str, Any]] = None,
    shutdown_requested: Optional[Callable[[], bool]] = None,
    keepalive=None,
) -> SingleModelBenchmarkResult:
    """Run, score, trace, and optionally export one model benchmark."""
    scorer_func = scorer_bundle.score_func
    scorer = getattr(scorer_bundle, "scorer", None)
    scorer_details = getattr(scorer_bundle, "details", {})
    scoring_method = scorer_bundle.method_label
    tracer = None
    tracer_failed = False
    shutdown_requested = shutdown_requested or (lambda: False)

    if tracer_config and tracer_factory:
        try:
            tracer = tracer_factory(tracer_config)
            tracer.start_benchmark(model_name, scoring_method)
        except Exception as e:
            tracer_failed = True
            print(f"⚠️  Warning: Failed to start Langfuse trace: {e}")

    effective_concurrency = _effective_concurrency(
        runtime.concurrency,
        optimizer_enabled=optimizer is not None,
        tracer_enabled=tracer is not None,
    )
    runtime.concurrency = effective_concurrency

    if effective_concurrency > 1:
        results = _run_questions_concurrent(
            questions,
            client,
            scorer_func,
            runtime,
            scorer=scorer,
            scorer_details=scorer_details,
            semantic_scorer=semantic_scorer,
            shutdown_requested=shutdown_requested,
            model_name=model_name,
        )
        optimization_results = []
    else:
        results, optimization_results = _run_questions_sequential(
            questions=questions,
            client=client,
            scorer_func=scorer_func,
            runtime=runtime,
            model_name=model_name,
            optimizer=optimizer,
            reference_answers=reference_answers,
            tracer=tracer,
            scorer=scorer,
            scorer_details=scorer_details,
            semantic_scorer=semantic_scorer,
            shutdown_requested=shutdown_requested,
            keepalive=keepalive,
        )

    interrupted = shutdown_requested()
    total_score = weighted_score(results) if results else 0.0
    interpretation = get_interpretation(total_score)
    summary = summarize_results(results)

    has_dual_track = any(r.get("semantic_best") for r in results)
    if has_dual_track:
        rubric_track = build_track_results(results, track="rubric")
        semantic_track = build_track_results(results, track="semantic")
        summary["rubric_track"] = summarize_track(rubric_track, primary="score")
        if semantic_track:
            summary["semantic_track"] = summarize_track(
                semantic_track, primary="semantic_score"
            )
            # Align summary.semantic with the semantic track (semantic-best),
            # not the rubric-best answer's sibling semantic score.
            semantic_summary = summarize_semantic_results(results)
            if semantic_summary:
                summary["semantic"] = semantic_summary
    else:
        semantic_summary = summarize_semantic_results(results)
        if semantic_summary:
            summary["semantic"] = semantic_summary

    if tracer:
        tracer.end_benchmark(total_score, interpretation)

    exported = {}
    if export_callback and results:
        metadata = {}
        if interrupted:
            metadata = {
                "interrupted": True,
                "completed_questions": len(results),
                "total_questions": len(questions),
            }
        exported = export_callback(
            results=results,
            model_name=model_name,
            total_score=total_score,
            interpretation=interpretation,
            scoring_method=scoring_method,
            summary=summary,
            metadata=metadata or None,
            **(export_kwargs or {}),
        )

    return SingleModelBenchmarkResult(
        model_name=model_name,
        results=results,
        total_score=total_score,
        interpretation=interpretation,
        optimization_results=optimization_results,
        exported=exported,
        tracer_failed=tracer_failed,
        interrupted=interrupted,
    )
