"""Tests for cooperative graceful shutdown."""

import pytest

from benchmark.shutdown import GracefulShutdown, ShutdownState, check_shutdown
from optimization.prompts import PromptOptimizer


def test_check_shutdown_raises_when_requested():
    state = ShutdownState()
    state.requested = True
    with pytest.raises(GracefulShutdown):
        check_shutdown(state.is_requested)


def test_shutdown_state_second_signal_raises_keyboard_interrupt():
    state = ShutdownState()
    state.request(2)
    assert state.requested is True
    with pytest.raises(KeyboardInterrupt):
        state.request(2)


def test_optimize_prompt_stops_when_shutdown_requested():
    class FakeTarget:
        def query(self, prompt, max_tokens=1024, retries=3, temperature=0.2):
            return "target answer"

    class FakeOptimizerClient:
        def query(self, prompt, max_tokens=1024, retries=3, temperature=0.2):
            return "---VARIANT: ROLE-PLAYING---\nreframed prompt"

        def close(self):
            return None

    optimizer = PromptOptimizer(
        optimizer_model="opt",
        optimizer_provider="ollama",
        optimizer_endpoint="http://localhost:11434",
        max_iterations=4,
    )
    optimizer.optimizer_client = FakeOptimizerClient()

    with pytest.raises(GracefulShutdown):
        optimizer.optimize_prompt(
            original_prompt="question",
            target_client=FakeTarget(),
            scorer_func=lambda q_id, response: 0,
            question_id=1,
            initial_response="baseline",
            initial_score=0,
            shutdown_requested=lambda: True,
        )
