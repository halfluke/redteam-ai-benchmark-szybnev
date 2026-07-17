"""Tests for optimizer provider selection and pre-flight validation."""

import os
from types import SimpleNamespace

import pytest

import run_benchmark
from models import create_client
from models.openrouter import OpenRouterClient
from optimization.prompts import PromptOptimizer
from utils.config import BenchmarkConfig, OptimizationConfig, ProviderConfig


def _make_config(
    *,
    provider=None,
    model=None,
    api_key=None,
):
    return BenchmarkConfig(
        provider=ProviderConfig(name="ollama", endpoint="http://localhost:11434"),
        optimization=OptimizationConfig(
            optimizer_provider=provider,
            optimizer_model=model,
            optimizer_api_key=api_key,
        ),
    )


def test_create_client_deepinfra_uses_openrouter_client():
    client = create_client(
        "deepinfra",
        None,
        "deepseek-ai/DeepSeek-V3",
        api_key="test-token",
    )

    assert isinstance(client, OpenRouterClient)
    assert client.base_url == "https://api.deepinfra.com/v1/openai"
    assert client.model_name == "deepseek-ai/DeepSeek-V3"
    assert client.api_key == "test-token"


def test_prompt_optimizer_deepinfra_builds_openrouter_client():
    optimizer = PromptOptimizer(
        optimizer_model="deepseek-ai/DeepSeek-V3",
        optimizer_provider="deepinfra",
        optimizer_api_key="test-token",
    )

    assert isinstance(optimizer.optimizer_client, OpenRouterClient)
    assert optimizer.optimizer_client.base_url == "https://api.deepinfra.com/v1/openai"
    optimizer.close()


def test_initialize_optimizer_returns_none_when_unconfigured():
    args = SimpleNamespace(
        no_optimize=False,
        optimizer_provider=None,
        optimizer_model=None,
        optimizer_api_key=None,
        optimizer_endpoint=None,
        max_optimization_iterations=4,
    )

    assert run_benchmark._initialize_optimizer(args, None, "http://localhost:11434") is None


def test_initialize_optimizer_only_provider_aborts():
    args = SimpleNamespace(
        no_optimize=False,
        optimizer_provider="ollama",
        optimizer_model=None,
        optimizer_api_key=None,
        optimizer_endpoint=None,
        max_optimization_iterations=4,
    )

    with pytest.raises(SystemExit, match="must be used together"):
        run_benchmark._initialize_optimizer(args, None, "http://localhost:11434")


def test_initialize_optimizer_only_model_aborts():
    args = SimpleNamespace(
        no_optimize=False,
        optimizer_provider=None,
        optimizer_model="llama3.1:8b",
        optimizer_api_key=None,
        optimizer_endpoint=None,
        max_optimization_iterations=4,
    )

    with pytest.raises(SystemExit, match="must be used together"):
        run_benchmark._initialize_optimizer(args, None, "http://localhost:11434")


def test_initialize_optimizer_cloud_provider_without_key_aborts(monkeypatch):
    monkeypatch.delenv("DEEPINFRA_TOKEN", raising=False)

    args = SimpleNamespace(
        no_optimize=False,
        optimizer_provider="deepinfra",
        optimizer_model="deepseek-ai/DeepSeek-V3",
        optimizer_api_key=None,
        optimizer_endpoint=None,
        max_optimization_iterations=4,
    )

    with pytest.raises(SystemExit, match="requires an API key"):
        run_benchmark._initialize_optimizer(args, None, "http://localhost:11434")


def test_initialize_optimizer_cloud_provider_uses_env_key(monkeypatch):
    monkeypatch.setenv("DEEPINFRA_TOKEN", "env-token")
    monkeypatch.setattr(run_benchmark, "_validate_model_available", lambda *a, **k: None)

    args = SimpleNamespace(
        no_optimize=False,
        optimizer_provider="deepinfra",
        optimizer_model="deepseek-ai/DeepSeek-V3",
        optimizer_api_key=None,
        optimizer_endpoint=None,
        max_optimization_iterations=4,
    )

    optimizer = run_benchmark._initialize_optimizer(args, None, "http://localhost:11434")
    assert optimizer is not None
    assert isinstance(optimizer.optimizer_client, OpenRouterClient)
    assert optimizer.optimizer_client.api_key == "env-token"
    optimizer.close()


def test_initialize_optimizer_no_optimize_overrides_config(monkeypatch):
    monkeypatch.setattr(
        run_benchmark,
        "PromptOptimizer",
        lambda **kwargs: pytest.fail("PromptOptimizer should not be constructed"),
    )

    args = SimpleNamespace(
        no_optimize=True,
        optimizer_provider=None,
        optimizer_model=None,
        optimizer_api_key=None,
        optimizer_endpoint=None,
        max_optimization_iterations=4,
    )
    config = _make_config(provider="ollama", model="llama3.1:8b")

    assert run_benchmark._initialize_optimizer(args, config, "http://localhost:11434") is None


def test_initialize_optimizer_deepinfra_does_not_inherit_target_endpoint(monkeypatch):
    created = {}

    class FakeOptimizer:
        def __init__(self, **kwargs):
            created.update(kwargs)
            self.optimizer_client = SimpleNamespace(
                base_url="https://api.deepinfra.com/v1/openai",
                list_models=lambda: [{"id": "zai-org/GLM-5.2"}],
            )

        def close(self):
            return None

    monkeypatch.setattr(run_benchmark, "PromptOptimizer", FakeOptimizer)
    monkeypatch.setenv("DEEPINFRA_TOKEN", "env-token")

    args = SimpleNamespace(
        no_optimize=False,
        optimizer_provider="deepinfra",
        optimizer_model="zai-org/GLM-5.2",
        optimizer_api_key=None,
        optimizer_endpoint=None,
        max_optimization_iterations=4,
    )

    optimizer = run_benchmark._initialize_optimizer(
        args,
        None,
        "https://bugtrace-apex-26b.example.a.run.app",
    )
    assert optimizer is not None
    assert created["optimizer_endpoint"] is None
    assert created["optimizer_provider"] == "deepinfra"


def test_initialize_optimizer_reads_provider_model_from_config(monkeypatch):
    created = {}

    class FakeOptimizer:
        def __init__(self, **kwargs):
            created.update(kwargs)
            self.optimizer_client = SimpleNamespace(
                base_url="http://localhost:11434",
                list_models=lambda: [{"name": "llama3.1:8b"}],
            )

        def close(self):
            return None

    monkeypatch.setattr(run_benchmark, "PromptOptimizer", FakeOptimizer)

    args = SimpleNamespace(
        no_optimize=False,
        optimizer_provider=None,
        optimizer_model=None,
        optimizer_api_key=None,
        optimizer_endpoint=None,
        max_optimization_iterations=4,
    )
    config = _make_config(provider="ollama", model="llama3.1:8b")

    optimizer = run_benchmark._initialize_optimizer(args, config, "http://localhost:11434")
    assert optimizer is not None
    assert created["optimizer_provider"] == "ollama"
    assert created["optimizer_model"] == "llama3.1:8b"
