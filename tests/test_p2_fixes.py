"""Regression tests for audit items #14–25 (selected)."""

import json
import threading
from argparse import Namespace
from types import SimpleNamespace

import pytest
import requests

import run_benchmark
from benchmark.keepalive import ModelKeepalive
from benchmark.metrics import summarize_results
from models import create_client
from models.base import RequestsRetryMixin
from models.lmstudio import LMStudioClient
from models.openrouter import OpenRouterClient
from scoring.semantic_scorer import (
    SemanticScorer,
    parse_answers_sections,
    parse_semantic_references_content,
)


def test_import_utils_alone_succeeds():
    import importlib
    import sys

    for name in list(sys.modules):
        if name == "utils" or name.startswith("utils."):
            del sys.modules[name]
    utils = importlib.import_module("utils")
    assert utils.get_interpretation(85) == "strong-candidate"


def test_answers_parser_ignores_inline_delimiter_and_rejects_duplicates():
    content = (
        "=== Q1: One ===\n"
        "Body with inline === Q99: fake === text\n"
        "still Q1.\n"
        "\n"
        "=== Q2: Two ===\n"
        "Second answer.\n"
    )
    refs = parse_semantic_references_content(content)
    assert set(refs) == {1, 2}
    assert "=== Q99: fake ===" in refs[1]

    with pytest.raises(ValueError, match="Duplicate question id Q1"):
        parse_answers_sections(
            "=== Q1: A ===\nfirst\n=== Q1: B ===\nsecond\n",
            source="dup.txt",
        )


def test_cache_digest_includes_max_seq_length(tmp_path):
    answers = tmp_path / "answers.txt"
    answers.write_text("=== Q1: T ===\nok\n", encoding="utf-8")
    questions = [{"id": 1, "prompt": "p"}]

    class FakeEncoder:
        max_seq_length = 2048

        def encode(self, texts, **kwargs):
            return [[1.0, 0.0] for _ in texts]

    a = SemanticScorer(
        questions,
        answers_file=str(answers),
        max_seq_length=2048,
        encoder=FakeEncoder(),
        cache_dir=tmp_path / "cache",
    )
    b = SemanticScorer(
        questions,
        answers_file=str(answers),
        max_seq_length=512,
        encoder=FakeEncoder(),
        cache_dir=tmp_path / "cache",
    )
    assert a._reference_cache_file() != b._reference_cache_file()


def test_reference_cache_merge_is_thread_safe(tmp_path, monkeypatch):
    answers = tmp_path / "answers.txt"
    answers.write_text(
        "=== Q1: A ===\none\n\n=== Q2: B ===\ntwo\n",
        encoding="utf-8",
    )
    questions = [{"id": 1, "prompt": "p"}, {"id": 2, "prompt": "p"}]
    encode_lock = threading.Lock()
    encode_calls = []

    class FakeEncoder:
        max_seq_length = 2048

        def encode(self, texts, **kwargs):
            with encode_lock:
                encode_calls.append(list(texts))
            return [[1.0, 0.0] for _ in texts]

    from scoring.semantic_embedder import InjectedEmbedderAdapter

    scorer = SemanticScorer(
        questions,
        answers_file=str(answers),
        encoder=None,
        cache_dir=tmp_path / "cache",
    )
    monkeypatch.setattr(
        scorer,
        "_load_embedder",
        lambda: InjectedEmbedderAdapter(
            FakeEncoder(),
            model_name=scorer.model_name,
            max_seq_length=scorer.max_seq_length,
        ),
    )

    errors = []

    def worker(q_id):
        try:
            scorer._ensure_reference_embedding(q_id)
        except Exception as exc:  # pragma: no cover - surfaced via errors
            errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=(1,)),
        threading.Thread(target=worker, args=(2,)),
        threading.Thread(target=worker, args=(1,)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    cache_file = scorer._reference_cache_file()
    payload = json.loads(cache_file.read_text(encoding="utf-8"))
    assert payload["max_seq_length"] == scorer.max_seq_length
    assert set(payload["embeddings"]) == {"1", "2"}
    json.loads(cache_file.read_text(encoding="utf-8"))  # still valid JSON


def test_openwebui_cloudrun_identity_is_rejected(monkeypatch):
    args = Namespace()
    config = SimpleNamespace(
        provider=SimpleNamespace(
            timeout=None,
            auth="cloudrun_identity",
            cloudrun_audience=None,
            cloudrun_impersonate_service_account=None,
        )
    )

    monkeypatch.setattr(
        run_benchmark,
        "provider_auth_kwargs",
        lambda **kwargs: {
            "auth_token_getter": lambda: "tok",
            "invalidate_auth_token": lambda: None,
        },
    )
    with pytest.raises(SystemExit, match="openwebui"):
        run_benchmark._create_configured_client(
            "openwebui",
            "https://example.run.app",
            "model",
            api_key=None,
            config=config,
            args=args,
        )


def test_local_optimizer_does_not_inherit_cloud_target_endpoint():
    args = SimpleNamespace(optimizer_endpoint=None)
    config = SimpleNamespace(
        optimization=SimpleNamespace(optimizer_endpoint=None)
    )
    resolved = run_benchmark._resolve_optimizer_endpoint(
        "ollama",
        args,
        "https://bugtrace.example.a.run.app",
        config,
    )
    assert resolved is None


def test_deepinfra_does_not_use_openrouter_env(monkeypatch):
    monkeypatch.delenv("DEEPINFRA_TOKEN", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key-should-not-be-used")
    with pytest.raises(RuntimeError, match="DEEPINFRA_TOKEN"):
        OpenRouterClient(
            base_url="https://api.deepinfra.com/v1/openai",
            model_name="m",
            api_key=None,
            api_key_env="DEEPINFRA_TOKEN",
        )
    with pytest.raises(RuntimeError, match="DEEPINFRA_TOKEN"):
        create_client("deepinfra", None, "m", api_key=None)


def test_connection_error_is_retried(monkeypatch):
    client = LMStudioClient("http://localhost:1234", "model")
    calls = {"n": 0}

    def fake_post(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise requests.exceptions.ConnectionError("boom")
        response = SimpleNamespace()
        response.raise_for_status = lambda: None
        response.json = lambda: {"choices": [{"message": {"content": "ok"}}]}
        return response

    monkeypatch.setattr(client.session, "post", fake_post)
    assert client.query("hi", retries=3) == "ok"
    assert calls["n"] == 3
    client.close()


def test_summary_lexical_metrics_are_weighted():
    results = [
        {
            "score": 100,
            "weight": 3.0,
            "metrics": {"technical_accuracy": 1.0, "completeness": 1.0, "specificity": 1.0},
        },
        {
            "score": 0,
            "weight": 1.0,
            "metrics": {"technical_accuracy": 0.0, "completeness": 0.0, "specificity": 0.0},
        },
    ]
    summary = summarize_results(results)
    # (1*3 + 0*1) / 4 * 100 = 75
    assert summary["metrics"]["technical_accuracy"] == 75.0
    assert summary["weighted_score"] == 75.0


def test_keepalive_stop_join_covers_pingable_roles(monkeypatch):
    joined = {}

    class FakeThread:
        def __init__(self):
            self._alive = True

        def is_alive(self):
            return self._alive

        def join(self, timeout=None):
            joined["timeout"] = timeout
            self._alive = False

    keepalive = ModelKeepalive(
        [("target", SimpleNamespace()), ("optimizer", SimpleNamespace())],
        timeout_s=10,
    )
    keepalive._skip_ping_roles = {"optimizer"}
    keepalive._thread = FakeThread()
    keepalive._stop.set()
    keepalive.stop()
    assert joined["timeout"] == 10 * 1 + 5


def test_requests_retry_mixin_is_used_by_lmstudio():
    assert issubclass(LMStudioClient, RequestsRetryMixin)


def test_parse_reference_answers_uses_strict_parser(tmp_path):
    path = tmp_path / "answers_all.txt"
    path.write_text(
        "=== Q1: A ===\n"
        "has === Q9: not a header === inside\n"
        "=== Q2: B ===\ntwo\n",
        encoding="utf-8",
    )
    refs = run_benchmark.parse_reference_answers(str(path))
    assert set(refs) == {1, 2}
    assert "=== Q9: not a header ===" in refs[1]
