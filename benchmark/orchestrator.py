"""Single-model benchmark orchestration."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from utils.export import get_interpretation

from .runner import (
    _effective_concurrency,
    _run_questions_concurrent,
    _run_questions_sequential,
)
from .scoring_summary import compute_total_scores, primary_interpretation, primary_total_score
from .types import RuntimeOptions
from scoring.factory import MultiScorerBundle


@dataclass
class SingleModelBenchmarkResult:
    """Completed benchmark state for one model."""

    model_name: str
    results: List[Dict[str, Any]]
    total_score: float
    interpretation: str
    total_scores: Dict[str, float] = field(default_factory=dict)
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
    reference_answers: Optional[Dict[int, str]] = None,
    tracer_config=None,
    tracer_factory: Optional[Callable[[Any], Any]] = None,
    export_callback: Optional[Callable[..., Dict[str, str]]] = None,
    export_kwargs: Optional[Dict[str, Any]] = None,
    shutdown_requested: Optional[Callable[[], bool]] = None,
    optimization_trigger: str = "keyword_zero",
    keepalive=None,
) -> SingleModelBenchmarkResult:
    """Run, score, trace, and optionally export one model benchmark."""
    scorer_func = scorer_bundle.score_func
    scorer = getattr(scorer_bundle, "scorer", None)
    scorer_details = getattr(scorer_bundle, "details", {})
    scoring_method = scorer_bundle.method_label
    multi_scorer_bundle = scorer_bundle if isinstance(scorer_bundle, MultiScorerBundle) and scorer_bundle.is_multi else None
    score_methods = multi_scorer_bundle.methods if multi_scorer_bundle else None
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
            multi_scorer_bundle=multi_scorer_bundle,
            shutdown_requested=shutdown_requested,
            reference_answers=reference_answers,
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
            multi_scorer_bundle=multi_scorer_bundle,
            shutdown_requested=shutdown_requested,
            optimization_trigger=optimization_trigger,
            keepalive=keepalive,
        )

    interrupted = shutdown_requested()
    total_scores = compute_total_scores(results, score_methods)
    total_score = primary_total_score(total_scores)
    interpretation = primary_interpretation(total_scores)

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
            total_scores=total_scores or None,
            score_methods=score_methods,
            metadata=metadata or None,
            **(export_kwargs or {}),
        )

    return SingleModelBenchmarkResult(
        model_name=model_name,
        results=results,
        total_score=total_score,
        interpretation=interpretation,
        total_scores=total_scores,
        optimization_results=optimization_results,
        exported=exported,
        tracer_failed=tracer_failed,
        interrupted=interrupted,
    )
