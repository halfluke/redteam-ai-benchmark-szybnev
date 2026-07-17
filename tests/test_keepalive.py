"""Tests for model keepalive ping behavior."""

from types import SimpleNamespace

from benchmark.keepalive import ModelKeepalive, _is_cloud_optimizer_client, _skip_ping_for_role
from models.openrouter import OpenRouterClient


def test_cloud_optimizer_clients_are_detected():
    deepinfra = OpenRouterClient(
        base_url="https://api.deepinfra.com/v1/openai",
        model_name="zai-org/GLM-5.2",
        api_key="test",
    )
    openrouter = OpenRouterClient(
        base_url="https://openrouter.ai/api/v1",
        model_name="anthropic/claude-3.5-sonnet",
        api_key="test",
    )
    assert _is_cloud_optimizer_client(deepinfra)
    assert _is_cloud_optimizer_client(openrouter)


def test_local_optimizer_clients_are_not_skipped():
    local = OpenRouterClient(
        base_url="http://localhost:1234/v1",
        model_name="local-model",
        api_key="test",
    )
    assert not _is_cloud_optimizer_client(local)
    assert not _skip_ping_for_role("optimizer", local)
    assert not _skip_ping_for_role("target", local)


def test_model_keepalive_skips_cloud_optimizer_pings(monkeypatch):
    pinged = []

    def fake_ping(client, **kwargs):
        pinged.append(client)
        return True

    monkeypatch.setattr("benchmark.keepalive.ping_model_client", fake_ping)

    target = SimpleNamespace()
    optimizer = OpenRouterClient(
        base_url="https://api.deepinfra.com/v1/openai",
        model_name="zai-org/GLM-5.2",
        api_key="test",
    )
    keepalive = ModelKeepalive([("target", target), ("optimizer", optimizer)])

    assert keepalive.ping_role("optimizer") is True
    assert pinged == []

    assert keepalive.ping_role("target") is True
    assert pinged == [target]

    keepalive.ping_idle_endpoints()
    assert pinged == [target, target]
