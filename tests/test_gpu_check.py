from types import SimpleNamespace

import pytest
import requests

import run_benchmark
from benchmark.gpu_check import (
    GpuCheckFailed,
    get_vram_residency,
    run_gpu_check,
)
from models.ollama import OllamaClient
from utils.config import GpuCheckConfig


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code
        self.text = ""

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self.payload


class FakeSession:
    """Fake requests.Session serving a fixed /api/chat and /api/ps payload."""

    def __init__(self, chat_payload=None, ps_payload=None, raise_on_post=False):
        self.chat_payload = chat_payload or {"message": {"content": "hi"}}
        self.ps_payload = ps_payload or {"models": []}
        self.raise_on_post = raise_on_post
        self.posts = []
        self.gets = []

    def post(self, url, headers=None, json=None, timeout=None):
        if self.raise_on_post:
            raise RuntimeError("boom")
        self.posts.append({"url": url, "json": json, "timeout": timeout})
        return FakeResponse(self.chat_payload)

    def get(self, url, headers=None, timeout=None):
        self.gets.append({"url": url, "timeout": timeout})
        return FakeResponse(self.ps_payload)


def _client(chat_payload=None, ps_payload=None, raise_on_post=False, **kwargs):
    client = OllamaClient("http://ollama.local", "test-model", timeout=77, **kwargs)
    client.session = FakeSession(chat_payload, ps_payload, raise_on_post)
    return client


def _ps_payload(model_name, size, size_vram):
    return {
        "models": [
            {"name": model_name, "model": model_name, "size": size, "size_vram": size_vram}
        ]
    }


def test_get_vram_residency_finds_matching_model():
    client = _client(ps_payload=_ps_payload("test-model", 100, 90))

    residency = get_vram_residency(client, timeout_s=30)

    assert residency == (100, 90)


def test_get_vram_residency_no_matching_model_returns_none():
    client = _client(ps_payload=_ps_payload("other-model", 100, 90))

    assert get_vram_residency(client, timeout_s=30) is None


def test_get_vram_residency_non_ollama_client_returns_none():
    assert get_vram_residency(object(), timeout_s=30) is None


def test_get_vram_residency_request_failure_returns_none():
    client = _client(raise_on_post=True)

    class RaisingSession:
        def get(self, *a, **k):
            raise RuntimeError("boom")

    client.session = RaisingSession()
    assert get_vram_residency(client, timeout_s=30) is None


def test_run_gpu_check_disabled_when_threshold_zero():
    client = _client(ps_payload=_ps_payload("test-model", 100, 0))
    assert run_gpu_check(client, min_vram_fraction=0, timeout_s=10) == 0.0


def test_run_gpu_check_passes_when_fully_resident():
    client = _client(ps_payload=_ps_payload("test-model", 100, 100))

    fraction = run_gpu_check(client, min_vram_fraction=0.9, timeout_s=10)

    assert fraction == pytest.approx(1.0)


def test_run_gpu_check_raises_when_mostly_on_cpu():
    client = _client(ps_payload=_ps_payload("test-model", 100, 0))

    with pytest.raises(GpuCheckFailed, match="0% of the model"):
        run_gpu_check(client, min_vram_fraction=0.9, timeout_s=10)


def test_run_gpu_check_raises_on_partial_offload_below_threshold():
    client = _client(ps_payload=_ps_payload("test-model", 100, 50))

    with pytest.raises(GpuCheckFailed, match="50% of the model"):
        run_gpu_check(client, min_vram_fraction=0.9, timeout_s=10)


def test_run_gpu_check_raises_when_unmeasurable():
    client = _client(ps_payload={"models": []})
    with pytest.raises(GpuCheckFailed, match="/api/ps did not report VRAM residency"):
        run_gpu_check(client, min_vram_fraction=0.9, timeout_s=10)


def test_run_gpu_check_raises_when_load_request_fails():
    client = _client(raise_on_post=True, ps_payload=_ps_payload("test-model", 100, 100))
    with pytest.raises(GpuCheckFailed, match="could not load the model"):
        run_gpu_check(client, min_vram_fraction=0.9, timeout_s=10)


def test_run_gpu_check_retries_load_after_timeout():
    class TimeoutThenOkSession:
        def __init__(self):
            self.post_calls = 0

        def post(self, url, headers=None, json=None, timeout=None):
            self.post_calls += 1
            if self.post_calls == 1:
                raise requests.exceptions.Timeout()
            return FakeResponse({"message": {"content": "hi"}})

        def get(self, url, headers=None, timeout=None):
            return FakeResponse(_ps_payload("test-model", 100, 100))

    client = _client()
    client.session = TimeoutThenOkSession()

    fraction = run_gpu_check(client, min_vram_fraction=0.9, timeout_s=10)

    assert fraction == pytest.approx(1.0)
    assert client.session.post_calls == 2


def test_run_gpu_check_raises_for_non_ollama_client():
    with pytest.raises(GpuCheckFailed, match="provider is not Ollama"):
        run_gpu_check(object(), min_vram_fraction=0.9, timeout_s=10)


def test_resolve_gpu_check_config_cli_overrides_disabled_config():
    args = SimpleNamespace(min_vram_fraction=0.9)
    config = SimpleNamespace(gpu_check=GpuCheckConfig(enabled=False, min_vram_fraction=0))

    resolved = run_benchmark._resolve_gpu_check_config(args, config)

    assert resolved.enabled is True
    assert resolved.min_vram_fraction == 0.9


def test_resolve_gpu_check_config_cli_zero_disables():
    args = SimpleNamespace(min_vram_fraction=0.0)
    config = SimpleNamespace(gpu_check=GpuCheckConfig(enabled=True, min_vram_fraction=0.9))

    resolved = run_benchmark._resolve_gpu_check_config(args, config)

    assert resolved.enabled is False
    assert resolved.min_vram_fraction == 0.0


def test_resolve_gpu_check_config_falls_back_to_config_when_cli_unset():
    args = SimpleNamespace(min_vram_fraction=None)
    config = SimpleNamespace(gpu_check=GpuCheckConfig(enabled=True, min_vram_fraction=0.8))

    resolved = run_benchmark._resolve_gpu_check_config(args, config)

    assert resolved.enabled is True
    assert resolved.min_vram_fraction == 0.8


def test_resolve_gpu_check_config_defaults_when_no_config():
    args = SimpleNamespace(min_vram_fraction=None)

    resolved = run_benchmark._resolve_gpu_check_config(args, None)

    assert resolved.enabled is False
    assert resolved.min_vram_fraction == 0.0


def test_run_gpu_check_or_exit_prints_ok_and_returns(capsys):
    client = _client(ps_payload=_ps_payload("test-model", 100, 100))
    args = SimpleNamespace(min_vram_fraction=0.9)

    run_benchmark._run_gpu_check_or_exit(client, "test-model", args, None)

    captured = capsys.readouterr()
    assert "ok (100% in VRAM)" in captured.out


def test_run_gpu_check_or_exit_raises_gpu_check_failed():
    client = _client(ps_payload=_ps_payload("test-model", 100, 0))
    args = SimpleNamespace(min_vram_fraction=0.9)

    with pytest.raises(GpuCheckFailed):
        run_benchmark._run_gpu_check_or_exit(client, "test-model", args, None)


def test_run_gpu_check_or_exit_noop_when_disabled(capsys):
    client = _client(ps_payload=_ps_payload("test-model", 100, 0))
    args = SimpleNamespace(min_vram_fraction=None)

    run_benchmark._run_gpu_check_or_exit(client, "test-model", args, None)

    captured = capsys.readouterr()
    assert captured.out == ""


class _FakeTargetClient:
    base_url = "http://target.local"
    model_name = "target-model"

    def test_connection(self):
        return True

    def list_models(self):
        return [{"name": "target-model", "size": 1}]

    def query(self, prompt, max_tokens=1024, retries=3, temperature=0.2):
        return "response"

    def close(self):
        return None


class _FakeOptimizerClient:
    base_url = "http://optimizer.local"
    model_name = "optimizer-model"

    def list_models(self):
        return [{"name": "optimizer-model", "size": 1}]


class _FakeOptimizer:
    def __init__(
        self,
        optimizer_model,
        optimizer_provider,
        optimizer_endpoint=None,
        optimizer_api_key=None,
        max_iterations=1,
        optimization_max_tokens=2048,
        optimizer_timeout=300,
    ):
        self.optimizer_client = _FakeOptimizerClient()

    def close(self):
        return None


def _fake_run_single_model_benchmark(**kwargs):
    return SimpleNamespace(
        results=[{"id": 1, "category": "AMSI", "score": 50, "response_snippet": "s"}],
        total_score=50.0,
        interpretation="not-suitable",
        optimization_results=[],
    )


def _fake_load_dataset(filepath="benchmark.json"):
    return run_benchmark.BenchmarkDataset(
        questions=[{"id": 1, "category": "AMSI", "prompt": "prompt"}],
        path=filepath,
        content_hash="hash",
    )


def test_gpu_check_only_applies_to_target_not_optimizer_in_run_command(monkeypatch):
    target_client = _FakeTargetClient()
    gpu_check_calls = []

    def fake_run_gpu_check(client, **kwargs):
        gpu_check_calls.append(client)
        return 1.0

    args = SimpleNamespace(
        provider="ollama",
        endpoint=None,
        api_key=None,
        config=None,
        model="target-model",
        rate_limit_delay=0,
        max_tokens=32,
        temperature=0.2,
        concurrency=1,
        no_optimize=False,
        optimizer_provider="ollama",
        optimizer_model="optimizer-model",
        optimizer_api_key=None,
        optimizer_endpoint="http://optimizer.local",
        max_optimization_iterations=1,
        export_csv=False,
        output=None,
        min_vram_fraction=0.9,
    )

    monkeypatch.setattr(run_benchmark, "create_client", lambda *a, **k: target_client)
    monkeypatch.setattr(run_benchmark, "load_dataset", _fake_load_dataset)
    monkeypatch.setattr(
        run_benchmark, "run_single_model_benchmark", _fake_run_single_model_benchmark
    )
    monkeypatch.setattr(run_benchmark, "run_gpu_check", fake_run_gpu_check)
    monkeypatch.setattr(run_benchmark, "PromptOptimizer", _FakeOptimizer)

    run_benchmark.cmd_run_benchmark(args)

    assert gpu_check_calls == [target_client]


def test_gpu_check_only_applies_to_target_not_optimizer_in_interactive(monkeypatch):
    target_client = _FakeTargetClient()
    gpu_check_calls = []

    def fake_run_gpu_check(client, **kwargs):
        gpu_check_calls.append(client)
        return 1.0

    args = SimpleNamespace(
        provider="ollama",
        endpoint=None,
        api_key=None,
        config=None,
        rate_limit_delay=0,
        max_tokens=32,
        temperature=0.2,
        concurrency=1,
        no_optimize=False,
        optimizer_provider="ollama",
        optimizer_model="optimizer-model",
        optimizer_api_key=None,
        optimizer_endpoint="http://optimizer.local",
        max_optimization_iterations=1,
        export_csv=False,
        output=None,
        min_vram_fraction=0.9,
    )

    monkeypatch.setattr(run_benchmark, "create_client", lambda *a, **k: target_client)
    monkeypatch.setattr(run_benchmark, "pick", lambda *a, **k: [("target-model", 0)])
    monkeypatch.setattr(run_benchmark, "load_dataset", _fake_load_dataset)
    monkeypatch.setattr(
        run_benchmark, "run_single_model_benchmark", _fake_run_single_model_benchmark
    )
    monkeypatch.setattr(run_benchmark, "run_gpu_check", fake_run_gpu_check)
    monkeypatch.setattr(run_benchmark, "PromptOptimizer", _FakeOptimizer)

    run_benchmark.cmd_interactive(args)

    assert gpu_check_calls == [target_client]
