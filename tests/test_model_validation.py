"""Tests for pre-flight model availability validation."""

from types import SimpleNamespace

import pytest

import run_benchmark
from models.validation import (
    ModelListError,
    ModelNotFoundError,
    is_model_listed,
    validate_model_available,
)


class _FakeClient:
    base_url = "http://provider.local"
    model_name = "target-model"

    def __init__(self, models):
        self._models = models

    def list_models(self):
        return self._models


def test_is_model_listed_exact_match():
    models = [{"id": "anthropic/claude-3.5-sonnet"}]
    assert is_model_listed("anthropic/claude-3.5-sonnet", models, provider="openrouter")


def test_is_model_listed_ollama_tag_equivalence():
    models = [{"name": "llama3.1:8b", "size": 1}]
    assert is_model_listed("llama3.1:8b", models, provider="ollama")
    assert is_model_listed("llama3.1", models, provider="ollama")


def test_validate_model_available_passes_when_listed():
    client = _FakeClient([{"name": "target-model"}])
    validate_model_available(
        client,
        "target-model",
        provider="ollama",
        role="target",
    )


def test_validate_model_available_raises_when_missing():
    client = _FakeClient([{"name": "other-model"}])
    with pytest.raises(ModelNotFoundError, match="Target model 'target-model' not found"):
        validate_model_available(
            client,
            "target-model",
            provider="ollama",
            role="target",
        )


def test_validate_model_available_raises_on_empty_model_list():
    client = _FakeClient([])
    with pytest.raises(ModelListError, match="empty model list"):
        validate_model_available(
            client,
            "target-model",
            provider="ollama",
            role="target",
        )


def test_initialize_optimizer_aborts_when_optimizer_model_missing(monkeypatch):
    class FakeOptimizer:
        def __init__(self, **kwargs):
            self.optimizer_client = _FakeClient([{"name": "other-model"}])

        def close(self):
            return None

    monkeypatch.setattr(run_benchmark, "PromptOptimizer", FakeOptimizer)

    args = SimpleNamespace(
        no_optimize=False,
        optimizer_provider="ollama",
        optimizer_model="optimizer-model",
        optimizer_api_key=None,
        optimizer_endpoint=None,
        max_optimization_iterations=4,
    )

    with pytest.raises(SystemExit, match="Optimizer model 'optimizer-model' not found"):
        run_benchmark._initialize_optimizer(args, None, "http://localhost:11434")
