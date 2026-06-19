import json
from pathlib import Path

import pytest
import requests

from benchmark.runner import _query_and_score, _run_questions_sequential
from benchmark.types import RuntimeOptions
from models.diagnostics import QueryDiagnostics, format_request_summary
from models.ollama import OllamaClient


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload
        self.status_code = 200
        self.text = ""

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class TimeoutThenSuccessSession:
    def __init__(self, payload):
        self.payload = payload
        self.posts = []
        self.attempt = 0

    def post(self, url, headers=None, json=None, timeout=None):
        self.posts.append({"timeout": timeout})
        self.attempt += 1
        if self.attempt == 1:
            raise requests.exceptions.Timeout("timed out")
        return FakeResponse(self.payload)


def test_ollama_captures_timing_fields():
    client = OllamaClient("http://ollama.local", "test-model", timeout=120)
    client.session = TimeoutThenSuccessSession(
        {
            "message": {"content": "ok"},
            "total_duration": 5_000_000_000,
            "load_duration": 500_000_000,
            "prompt_eval_duration": 1_000_000_000,
            "eval_duration": 3_000_000_000,
            "prompt_eval_count": 100,
            "eval_count": 200,
        }
    )

    result = client.query("hello", max_tokens=64, retries=2)

    assert result == "ok"
    diagnostics = client.last_query_diagnostics
    assert diagnostics.status == "success"
    assert diagnostics.load_duration_ms == 500
    assert diagnostics.eval_count == 200
    assert "generation" in diagnostics.format_summary()


def test_timeout_diagnostics_include_limit_and_context(capsys):
    client = OllamaClient("http://ollama.local", "test-model", timeout=90)

    class AlwaysTimeoutSession:
        def post(self, url, headers=None, json=None, timeout=None):
            raise requests.exceptions.Timeout("timed out")

    client.session = AlwaysTimeoutSession()

    with pytest.raises(RuntimeError, match="API timeout after 1 attempts"):
        client.query("slow prompt", max_tokens=768, retries=1)

    output = capsys.readouterr().out
    assert "timed out after" in output
    assert "limit 90s" in output
    assert "slow VM↔host network" in output

    diagnostics = client.last_query_diagnostics
    assert diagnostics.status == "timeout"
    assert diagnostics.prompt_chars == len("slow prompt")


def test_sequential_runner_continues_after_request_failure(tmp_path, monkeypatch):
    class FailingThenWorkingClient:
        def __init__(self):
            self.calls = 0

        def query(self, prompt, max_tokens=1024, retries=3, temperature=0.2):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("API timeout after 3 attempts (150s limit per attempt)")
            self.last_query_diagnostics = QueryDiagnostics(
                provider="Ollama",
                endpoint="http://ollama.local",
                model="test",
                status="success",
                elapsed_ms=1200,
                eval_duration_ms=900,
                eval_count=50,
            )
            return "working response"

    monkeypatch.setattr("benchmark.runner.time.sleep", lambda delay: None)
    log_path = tmp_path / "requests.jsonl"
    runtime = RuntimeOptions(rate_limit_delay=0, request_log=str(log_path))
    questions = [
        {"id": 1, "category": "one", "prompt": "first"},
        {"id": 2, "category": "two", "prompt": "second"},
    ]

    results, _ = _run_questions_sequential(
        questions,
        FailingThenWorkingClient(),
        scorer_func=lambda q_id, response: 100 if response else 0,
        runtime=runtime,
        model_name="test-model",
    )

    assert len(results) == 2
    assert results[0]["score"] == 0
    assert "error" in results[0]["details"]
    assert results[1]["score"] == 100
    assert "request" in results[1]["details"]

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["question_id"] == 1
    assert first["error"]


def test_format_request_summary_from_dict():
    summary = format_request_summary(
        {
            "status": "success",
            "elapsed_s": 12.5,
            "load_duration_ms": 1000,
            "prompt_eval_duration_ms": 2000,
            "prompt_eval_count": 80,
            "eval_duration_ms": 9000,
            "eval_count": 300,
            "tokens_per_second": 33.33,
        }
    )
    assert "12.5s total" in summary
    assert "generation 9.0s (300 tok" in summary
