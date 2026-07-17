"""Tests for resilient HTTPS connectivity probes."""

from types import SimpleNamespace

import pytest
import requests

from models.lmstudio import LMStudioClient
from models.ollama import OllamaClient


def _response(status_code: int, text: str = "") -> SimpleNamespace:
    return SimpleNamespace(status_code=status_code, text=text)


def test_remote_probe_retries_503_then_succeeds(monkeypatch):
    client = OllamaClient("https://example.run.app", "model", timeout=30)
    calls = {"n": 0}
    sleeps: list[float] = []

    def fake_get(url, headers, timeout):
        calls["n"] += 1
        if calls["n"] < 3:
            return _response(503, "Service Unavailable")
        return _response(200, '{"models":[]}')

    monkeypatch.setattr(client.session, "get", fake_get)
    monkeypatch.setattr("models.base.time.sleep", lambda seconds: sleeps.append(seconds))

    assert client.test_connection() is True
    assert calls["n"] == 3
    assert sleeps == [1, 2]
    assert client.last_probe_error is None
    client.close()


def test_remote_probe_reports_http_error(monkeypatch):
    client = LMStudioClient("https://example.run.app", "model", timeout=30)

    monkeypatch.setattr(
        client.session,
        "get",
        lambda url, headers, timeout: _response(403, "Forbidden"),
    )
    monkeypatch.setattr("models.base.time.sleep", lambda _seconds: None)

    assert client.test_connection() is False
    assert client.last_probe_error == "HTTP 403: Forbidden"
    client.close()


def test_remote_probe_retries_connection_error(monkeypatch):
    client = OllamaClient("https://example.run.app", "model", timeout=30)
    calls = {"n": 0}

    def fake_get(url, headers, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            raise requests.exceptions.ConnectionError("reset by peer")
        return _response(200, '{"models":[]}')

    monkeypatch.setattr(client.session, "get", fake_get)
    monkeypatch.setattr("models.base.time.sleep", lambda _seconds: None)

    assert client.test_connection() is True
    assert calls["n"] == 2
    client.close()


def test_local_probe_does_not_retry(monkeypatch):
    client = OllamaClient("http://localhost:11434", "model", timeout=30)
    calls = {"n": 0}

    def fake_get(url, headers, timeout):
        calls["n"] += 1
        return _response(503, "Service Unavailable")

    monkeypatch.setattr(client.session, "get", fake_get)
    monkeypatch.setattr("models.base.time.sleep", lambda _seconds: pytest.fail("local probe retried"))

    assert client.test_connection() is False
    assert calls["n"] == 1
    assert client.last_probe_error == "HTTP 503: Service Unavailable"
    client.close()


def test_remote_probe_refreshes_auth_on_401(monkeypatch):
    client = OllamaClient("https://example.run.app", "model", timeout=30)
    calls = {"n": 0}
    tokens = iter(["old-token", "new-token"])

    client.auth_token_getter = lambda: next(tokens)
    client.invalidate_auth_token = lambda: None

    def fake_get(url, headers, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            return _response(401, "Unauthorized")
        assert headers["Authorization"] == "Bearer new-token"
        return _response(200, '{"models":[]}')

    monkeypatch.setattr(client.session, "get", fake_get)
    monkeypatch.setattr("models.base.time.sleep", lambda _seconds: None)

    assert client.test_connection() is True
    assert calls["n"] == 2
    client.close()
