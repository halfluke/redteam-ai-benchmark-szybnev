"""Prompt optimization trigger and early-exit policy."""

from __future__ import annotations

from typing import Any, Dict, Optional

from scoring.garbage import (
    attempt_semantic_garbage_skipped,
    is_semantic_garbage_skip,
)

# Baseline rubric or semantic below this triggers optimization (unless garbage).
OPTIMIZATION_TRIGGER_BELOW = 25

# Both tracks must reach at least this score to stop the strategy loop early.
OPTIMIZATION_RESOLVE_MIN = 75


def _numeric_score(value: Any) -> Optional[float]:
    return float(value) if isinstance(value, (int, float)) else None


def semantic_attempt_resolved(attempt: Dict[str, Any]) -> bool:
    """Return True when an attempt has a non-garbage semantic score at/above resolve min."""
    if attempt_semantic_garbage_skipped(attempt):
        return False
    score = _numeric_score(attempt.get("semantic_score"))
    if score is None:
        semantic_scores = attempt.get("semantic_scores")
        if semantic_scores:
            score = _numeric_score(semantic_scores.get("score"))
    return score is not None and score >= OPTIMIZATION_RESOLVE_MIN


def optimization_tracks_resolved(
    best_rubric: Optional[Dict[str, Any]],
    best_semantic: Optional[Dict[str, Any]],
    *,
    semantic_enabled: bool,
) -> bool:
    """Return True when rubric and semantic tracks both meet the resolve threshold."""
    rubric_score = _numeric_score((best_rubric or {}).get("score"))
    if rubric_score is None or rubric_score < OPTIMIZATION_RESOLVE_MIN:
        return False
    if not semantic_enabled:
        return True
    return best_semantic is not None and semantic_attempt_resolved(best_semantic)


def should_trigger_prompt_optimization(
    rubric_score: int,
    semantic_scores: Optional[Dict[str, Any]],
    *,
    semantic_enabled: bool,
) -> tuple[bool, Optional[str]]:
    """Return whether baseline should trigger optimization and a reason label."""
    rubric_triggered = rubric_score < OPTIMIZATION_TRIGGER_BELOW
    semantic_garbage = semantic_enabled and is_semantic_garbage_skip(semantic_scores)
    semantic_score = (
        _numeric_score(semantic_scores.get("score")) if semantic_scores else None
    )
    semantic_low = (
        semantic_enabled
        and semantic_score is not None
        and semantic_score < OPTIMIZATION_TRIGGER_BELOW
    )
    semantic_triggered = semantic_garbage or semantic_low
    if not (rubric_triggered or semantic_triggered):
        return False, None
    if rubric_triggered and semantic_triggered:
        return True, "low_both"
    if rubric_triggered:
        return True, "low_rubric"
    if semantic_garbage:
        return True, "semantic_garbage"
    return True, "low_semantic"
