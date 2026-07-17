"""Tests for optimization trigger and early-exit policy."""

from optimization.policy import (
    OPTIMIZATION_RESOLVE_MIN,
    OPTIMIZATION_TRIGGER_BELOW,
    optimization_tracks_resolved,
    semantic_attempt_resolved,
    should_trigger_prompt_optimization,
)
from scoring.garbage import semantic_garbage_skip_payload


def test_semantic_attempt_resolved_requires_resolve_threshold():
    assert not semantic_attempt_resolved(
        {"semantic_score": 0, "semantic_scores": {"score": 0}}
    )
    assert not semantic_attempt_resolved({"semantic_score": 74})
    assert not semantic_attempt_resolved(
        {
            "semantic_skipped": "garbage_response",
            "semantic_scores": semantic_garbage_skip_payload(answer_source="baseline"),
        }
    )
    assert semantic_attempt_resolved({"semantic_score": OPTIMIZATION_RESOLVE_MIN})
    assert semantic_attempt_resolved({"semantic_score": 90})


def test_tracks_resolved_requires_both_thresholds_when_semantic_enabled():
    assert optimization_tracks_resolved(
        {"score": 90},
        {"semantic_score": 40},
        semantic_enabled=True,
    ) is False
    assert optimization_tracks_resolved(
        {"score": OPTIMIZATION_RESOLVE_MIN},
        {"semantic_score": OPTIMIZATION_RESOLVE_MIN},
        semantic_enabled=True,
    )
    assert not optimization_tracks_resolved(
        {"score": 90},
        None,
        semantic_enabled=True,
    )


def test_trigger_low_rubric_with_nonzero_semantic_only():
    assert should_trigger_prompt_optimization(
        0,
        {"score": 40},
        semantic_enabled=True,
    ) == (True, "low_rubric")


def test_trigger_garbage_only_with_high_rubric():
    payload = semantic_garbage_skip_payload(answer_source="baseline")
    assert should_trigger_prompt_optimization(
        90,
        payload,
        semantic_enabled=True,
    ) == (True, "semantic_garbage")


def test_trigger_skips_scores_at_or_above_threshold():
    assert should_trigger_prompt_optimization(
        OPTIMIZATION_TRIGGER_BELOW,
        {"score": OPTIMIZATION_TRIGGER_BELOW},
        semantic_enabled=True,
    ) == (False, None)
