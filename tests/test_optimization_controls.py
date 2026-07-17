"""Tests for optimization speed/quality controls."""

import pytest

from optimization.policy import (
    OPTIMIZATION_RESOLVE_MIN,
    OPTIMIZATION_TRIGGER_BELOW,
    optimization_tracks_resolved,
    should_trigger_prompt_optimization,
)
from optimization.prompts import (
    CVEFramingStrategy,
    FewShotStrategy,
    PromptOptimizer,
    RolePlayingStrategy,
    TechnicalDecompositionStrategy,
)
from scoring.garbage import semantic_garbage_skip_payload


def test_should_trigger_on_low_scores_or_semantic_garbage():
    assert should_trigger_prompt_optimization(0, None, semantic_enabled=False) == (
        True,
        "low_rubric",
    )
    assert should_trigger_prompt_optimization(33, {"score": 20}, semantic_enabled=True) == (
        True,
        "low_semantic",
    )
    assert should_trigger_prompt_optimization(50, {"score": 0}, semantic_enabled=True) == (
        True,
        "low_semantic",
    )
    assert should_trigger_prompt_optimization(
        90,
        semantic_garbage_skip_payload(answer_source="baseline"),
        semantic_enabled=True,
    ) == (True, "semantic_garbage")
    assert should_trigger_prompt_optimization(0, {"score": 40}, semantic_enabled=True) == (
        True,
        "low_rubric",
    )
    assert should_trigger_prompt_optimization(
        OPTIMIZATION_TRIGGER_BELOW,
        {"score": 40},
        semantic_enabled=True,
    ) == (False, None)


def test_tracks_resolved_requires_resolve_threshold_on_both_tracks():
    assert not optimization_tracks_resolved(
        {"score": 29},
        {"semantic_score": 40},
        semantic_enabled=True,
    )
    assert not optimization_tracks_resolved(
        {"score": 0},
        {"semantic_score": 40},
        semantic_enabled=True,
    )
    assert not optimization_tracks_resolved(
        {"score": 29},
        {"semantic_skipped": "garbage_response"},
        semantic_enabled=True,
    )
    assert optimization_tracks_resolved(
        {"score": OPTIMIZATION_RESOLVE_MIN},
        {"semantic_score": OPTIMIZATION_RESOLVE_MIN},
        semantic_enabled=True,
    )
    assert optimization_tracks_resolved(
        {"score": OPTIMIZATION_RESOLVE_MIN},
        None,
        semantic_enabled=False,
    )


def test_optimizer_uses_configured_max_tokens_by_default(monkeypatch):
    monkeypatch.setattr("optimization.prompts.time.sleep", lambda _: None)

    captured = []

    class FakeTargetClient:
        def query(self, prompt, max_tokens=1024, retries=3, temperature=0.2):
            captured.append(max_tokens)
            return "answer v1"

    optimizer = PromptOptimizer.__new__(PromptOptimizer)
    optimizer.history = []
    optimizer.max_iterations = 1
    optimizer.optimization_max_tokens = 2048
    optimizer.strategies = {
        "role_playing": RolePlayingStrategy(),
        "technical": TechnicalDecompositionStrategy(),
        "few_shot": FewShotStrategy(),
        "cve_framing": CVEFramingStrategy(),
    }
    optimizer._generate_optimized_prompt = lambda **kwargs: "reframe v1"

    optimizer.optimize_prompt(
        original_prompt="original prompt",
        target_client=FakeTargetClient(),
        scorer_func=lambda q_id, response: 50,
        question_id=1,
        initial_response="baseline",
        initial_score=0,
    )

    assert captured == [2048]


def test_optimizer_stops_when_both_tracks_meet_resolve_threshold(monkeypatch):
    monkeypatch.setattr("optimization.prompts.time.sleep", lambda _: None)

    class FakeTargetClient:
        def __init__(self):
            self.calls = 0

        def query(self, prompt, max_tokens=1024, retries=3, temperature=0.2):
            self.calls += 1
            return f"answer-{self.calls}"

    def scorer(q_id, response):
        return {"answer-1": 80, "answer-2": 90}.get(response, 0)

    def score_semantic(response, source):
        return {"score": {"answer-1": 76, "answer-2": 95}.get(response, 0)}

    optimizer = PromptOptimizer.__new__(PromptOptimizer)
    optimizer.history = []
    optimizer.max_iterations = 4
    optimizer.optimization_max_tokens = 2048
    optimizer.strategies = {
        "role_playing": RolePlayingStrategy(),
        "technical": TechnicalDecompositionStrategy(),
        "few_shot": FewShotStrategy(),
        "cve_framing": CVEFramingStrategy(),
    }
    optimizer._generate_optimized_prompt = lambda **kwargs: f"reframe-{kwargs['strategy_name']}"

    client = FakeTargetClient()
    result = optimizer.optimize_prompt(
        original_prompt="original",
        target_client=client,
        scorer_func=scorer,
        question_id=1,
        initial_response="baseline",
        initial_score=0,
        initial_semantic_scores={"score": 40},
        score_semantic_func=score_semantic,
    )

    assert client.calls == 1
    assert len([a for a in result["history"] if a["iteration"] > 0]) == 1
    assert result["score"] == 80
    assert result["early_exit"] == "tracks_resolved"


def test_optimizer_skips_semantic_on_garbage(monkeypatch):
    monkeypatch.setattr("optimization.prompts.time.sleep", lambda _: None)
    semantic_calls = []

    class FakeTargetClient:
        def query(self, prompt, max_tokens=1024, retries=3, temperature=0.2):
            return "Os " * 30

    def score_semantic(response, source):
        semantic_calls.append(response)
        return {"score": 99}

    optimizer = PromptOptimizer.__new__(PromptOptimizer)
    optimizer.history = []
    optimizer.max_iterations = 1
    optimizer.optimization_max_tokens = 2048
    optimizer.strategies = {
        "role_playing": RolePlayingStrategy(),
        "technical": TechnicalDecompositionStrategy(),
        "few_shot": FewShotStrategy(),
        "cve_framing": CVEFramingStrategy(),
    }
    optimizer._generate_optimized_prompt = lambda **kwargs: "reframe"

    result = optimizer.optimize_prompt(
        original_prompt="original",
        target_client=FakeTargetClient(),
        scorer_func=lambda q_id, response: 0,
        question_id=1,
        initial_response="Os " * 30,
        initial_score=0,
        score_semantic_func=score_semantic,
    )

    attempt = result["history"][-1]
    assert semantic_calls == []
    assert result["history"][0]["semantic_skipped"] == "garbage_response"
    assert attempt["semantic_skipped"] == "garbage_response"
    assert "semantic_score" not in attempt
