"""Regression tests for P0/P1 audit fixes (excluding garbage-strip and GPU tags)."""

from argparse import Namespace
from types import SimpleNamespace

import pytest

import run_benchmark
from benchmark.metrics import (
    build_track_results,
    summarize_semantic_results,
    weighted_primary_score,
)
from models.base import normalize_message_content
from scoring.garbage import semantic_garbage_skip_payload


def test_normalize_message_content_handles_null():
    assert normalize_message_content(None) == ""
    assert normalize_message_content("ok") == "ok"


def test_build_track_results_keeps_garbage_semantic_rows():
    results = [
        {
            "id": 1,
            "weight": 1.0,
            "score": 80,
            "semantic_score": None,
            "semantic_scores": semantic_garbage_skip_payload(answer_source="baseline"),
            "semantic_best": None,
            "rubric_best": {
                "score": 80,
                "semantic_score": None,
                "full_response": "Os " * 30,
                "answer_source": "baseline",
            },
        },
        {
            "id": 2,
            "weight": 1.0,
            "score": 40,
            "semantic_score": 60,
            "semantic_best": {
                "score": 30,
                "semantic_score": 90,
                "semantic_similarity": 0.9,
                "full_response": "good",
                "answer_source": "optimized",
                "strategy": "role_playing",
            },
            "rubric_best": {
                "score": 40,
                "semantic_score": 60,
                "full_response": "rubric winner",
                "answer_source": "baseline",
            },
            "tracks_diverged": True,
        },
    ]

    semantic_track = build_track_results(results, track="semantic")
    assert len(semantic_track) == 2
    assert semantic_track[0]["semantic_score"] == 0
    assert semantic_track[1]["semantic_score"] == 90
    assert weighted_primary_score(semantic_track, "semantic_score") == 45.0


def test_summarize_semantic_uses_semantic_best_not_rubric_sibling():
    results = [
        {
            "id": 1,
            "weight": 1.0,
            "score": 80,
            "semantic_score": 20,
            "semantic_similarity": 0.2,
            "semantic_best": {
                "score": 40,
                "semantic_score": 90,
                "semantic_similarity": 0.9,
            },
            "rubric_best": {
                "score": 80,
                "semantic_score": 20,
            },
        }
    ]
    summary = summarize_semantic_results(results)
    assert summary is not None
    assert summary["weighted_score"] == 90.0
    assert summary["questions"] == 1


def test_config_load_failure_exits(monkeypatch, tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("provider: [", encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        run_benchmark._load_optional_config(Namespace(config=str(bad)))
    assert exc.value.code == 1


def test_cli_max_optimization_iterations_not_overwritten_by_config():
    args = Namespace(
        endpoint=None,
        api_key=None,
        optimizer_provider=None,
        optimizer_model=None,
        optimizer_api_key=None,
        optimizer_endpoint=None,
        max_optimization_iterations=1,
        semantic=False,
    )
    config = SimpleNamespace(
        provider=SimpleNamespace(endpoint=None, api_key=None, api_key_env=None),
        optimization=SimpleNamespace(
            optimizer_provider=None,
            optimizer_model=None,
            optimizer_api_key=None,
            optimizer_endpoint=None,
            max_iterations=4,
        ),
        scoring=SimpleNamespace(semantic=None),
    )
    run_benchmark._apply_config_defaults(args, config)
    assert args.max_optimization_iterations == 1


def test_cli_max_optimization_iterations_defaults_from_config_when_unset():
    args = Namespace(
        endpoint=None,
        api_key=None,
        optimizer_provider=None,
        optimizer_model=None,
        optimizer_api_key=None,
        optimizer_endpoint=None,
        max_optimization_iterations=None,
        semantic=False,
    )
    config = SimpleNamespace(
        provider=SimpleNamespace(endpoint=None, api_key=None, api_key_env=None),
        optimization=SimpleNamespace(
            optimizer_provider=None,
            optimizer_model=None,
            optimizer_api_key=None,
            optimizer_endpoint=None,
            max_iterations=2,
        ),
        scoring=SimpleNamespace(semantic=None),
    )
    run_benchmark._apply_config_defaults(args, config)
    assert args.max_optimization_iterations == 2


def test_deepinfra_is_cli_provider_choice():
    import argparse

    parser = argparse.ArgumentParser()
    run_benchmark._add_provider_arg(parser)
    parsed = parser.parse_args(["deepinfra"])
    assert parsed.provider == "deepinfra"


def test_shutdown_during_optimization_keeps_baseline(monkeypatch):
    class FakeClient:
        def query(self, prompt, max_tokens=1024, retries=3, temperature=0.2):
            return "baseline answer"

    class FakeOptimizer:
        def optimize_prompt(self, **kwargs):
            raise run_benchmark.GracefulShutdown

    monkeypatch.setattr(run_benchmark.time, "sleep", lambda _: None)
    results, _ = run_benchmark._run_questions_sequential(
        questions=[{"id": 1, "category": "cat", "prompt": "prompt"}],
        client=FakeClient(),
        scorer_func=lambda q_id, response: 0,
        runtime=run_benchmark.RuntimeOptions(rate_limit_delay=0),
        model_name="model",
        optimizer=FakeOptimizer(),
    )
    assert len(results) == 1
    assert results[0]["id"] == 1
    assert results[0]["score"] == 0
    assert results[0]["full_response"] == "baseline answer"
    assert results[0]["error"] == "interrupted_during_optimization"


def test_judge_default_results_include_exporter_layout():
    from benchmark.offline_judge import DEFAULT_RESULTS

    assert "results/*.json" in DEFAULT_RESULTS


def test_openrouter_timeout_wrapped_as_runtime_error(monkeypatch):
    httpx = pytest.importorskip("httpx")
    from models.openrouter import OpenRouterClient

    client = OpenRouterClient(
        base_url="https://openrouter.ai/api/v1",
        model_name="test/model",
        api_key="test-key",
        timeout=1,
    )

    def boom(*args, **kwargs):
        raise httpx.TimeoutException("timed out")

    monkeypatch.setattr(client, "_make_request", boom)
    with pytest.raises(RuntimeError, match="timeout"):
        client.query("hello", retries=1)
    client.close()


def test_401_refresh_retries_once(monkeypatch):
    import requests

    from models.lmstudio import LMStudioClient

    client = LMStudioClient("http://localhost:1234", "model", api_key="stale")
    calls = {"n": 0}
    refreshed = {"ok": False}

    class FakeResponse:
        status_code = 401
        text = "unauthorized"

        def raise_for_status(self):
            if self.status_code >= 400:
                err = requests.exceptions.HTTPError(response=self)
                raise err

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    def fake_post(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return FakeResponse()
        ok = FakeResponse()
        ok.status_code = 200

        def raise_ok():
            return None

        ok.raise_for_status = raise_ok  # type: ignore[method-assign]
        return ok

    def refresh(headers):
        refreshed["ok"] = True
        headers["Authorization"] = "Bearer fresh"
        return True

    monkeypatch.setattr(client.session, "post", fake_post)
    monkeypatch.setattr(client, "_refresh_auth_headers", refresh)
    client.auth_token_getter = lambda: "fresh"

    assert client.query("hello", retries=2) == "ok"
    assert calls["n"] == 2
    assert refreshed["ok"] is True
    client.close()
