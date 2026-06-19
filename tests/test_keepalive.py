"""Tests for model keepalive behavior."""

from types import SimpleNamespace
from unittest.mock import patch

from benchmark.keepalive import ModelKeepalive, keepalive_busy, ping_model_client
import run_benchmark
from utils.config import BenchmarkConfig, KeepaliveConfig, ProviderConfig, load_config


class FakeClient:
    def __init__(self, name: str):
        self.name = name
        self.calls = 0

    def query(self, prompt, max_tokens=16, retries=1, temperature=0):
        self.calls += 1
        return "OK"


def test_warmup_pings_all_endpoints():
    target = FakeClient("target")
    optimizer = FakeClient("optimizer")

    with patch("benchmark.keepalive.ping_model_client", side_effect=[True, True]) as ping:
        keepalive = ModelKeepalive([("target", target), ("optimizer", optimizer)])
        results = keepalive.warmup()

    assert results == {"target": True, "optimizer": True}
    assert ping.call_count == 2


def test_ping_idle_endpoints_skips_busy_role():
    target = FakeClient("target")
    optimizer = FakeClient("optimizer")
    keepalive = ModelKeepalive([("target", target), ("optimizer", optimizer)])

    with patch("benchmark.keepalive.ping_model_client", return_value=True) as ping:
        keepalive.mark_busy("target")
        results = keepalive.ping_idle_endpoints()

    assert results == {"optimizer": True}
    ping.assert_called_once()
    assert ping.call_args[0][0] is optimizer


def test_ping_idle_endpoints_pings_both_when_nobody_busy():
    target = FakeClient("target")
    optimizer = FakeClient("optimizer")
    keepalive = ModelKeepalive([("target", target), ("optimizer", optimizer)])

    with patch("benchmark.keepalive.ping_model_client", return_value=True) as ping:
        results = keepalive.ping_idle_endpoints()

    assert results == {"target": True, "optimizer": True}
    assert ping.call_count == 2


def test_keepalive_busy_context_marks_roles():
    keepalive = ModelKeepalive([("target", object()), ("optimizer", object())])

    assert keepalive.idle_roles() == {"target", "optimizer"}
    with keepalive_busy(keepalive, "target"):
        assert keepalive.idle_roles() == {"optimizer"}
    assert keepalive.idle_roles() == {"target", "optimizer"}


def test_keepalive_busy_noop_when_disabled():
    with keepalive_busy(None, "target"):
        pass


def test_ping_model_client_fallback_uses_query():
    client = FakeClient("fallback")
    assert ping_model_client(client) is True
    assert client.calls == 1


def test_load_config_parses_keepalive():
    config = load_config("configs/cloudrun_ollama_optimize.yaml")
    assert config.keepalive.enabled is True
    assert config.keepalive.interval_s == 60
    assert config.optimization.optimizer_timeout == 600


def test_cloud_run_config_auto_keepalive_without_optimizer():
    config = load_config("configs/cloudrun_vllm_deephat.yaml")
    assert config.provider.auth == "cloudrun_identity"
    assert run_benchmark._should_run_keepalive(config, optimizer=None) is True


def test_keepalive_disabled_explicitly_in_yaml():
    config = BenchmarkConfig(
        provider=ProviderConfig(
            name="ollama",
            endpoint="https://example.run.app",
            auth="cloudrun_identity",
        ),
        keepalive=KeepaliveConfig(enabled=False),
        keepalive_in_yaml=True,
    )
    assert run_benchmark._should_run_keepalive(config, optimizer=None) is False


def test_create_keepalive_target_only_for_cloud_run():
    client = FakeClient("target")
    config = load_config("configs/cloudrun_ollama.yaml")
    keepalive = run_benchmark._create_keepalive(client, None, config)
    assert keepalive is not None
    assert keepalive.idle_roles() == {"target"}


def test_create_keepalive_includes_optimizer_when_present():
    client = FakeClient("target")
    optimizer = SimpleNamespace(
        optimizer_client=FakeClient("optimizer"),
    )
    config = load_config("configs/cloudrun_ollama_optimize.yaml")
    keepalive = run_benchmark._create_keepalive(client, optimizer, config)
    assert keepalive is not None
    assert keepalive.idle_roles() == {"target", "optimizer"}


def test_ping_role_uses_keep_alive_only_for_optimizer():
    target = object()
    optimizer = object()
    keepalive = ModelKeepalive(
        [("target", target), ("optimizer", optimizer)],
        optimizer_ollama_keep_alive="30m",
    )

    with patch("benchmark.keepalive.ping_model_client", return_value=True) as ping:
        keepalive.ping_role("target")
        assert ping.call_args.kwargs.get("keep_alive") is None

        keepalive.ping_role("optimizer")
        assert ping.call_args.kwargs.get("keep_alive") == "30m"


def test_connection_probe_timeout_local_vs_remote():
    from models.base import (
        DEFAULT_LOCAL_CONNECTION_PROBE_TIMEOUT,
        DEFAULT_REMOTE_CONNECTION_PROBE_TIMEOUT,
        connection_probe_timeout,
    )

    assert (
        connection_probe_timeout(900, remote=False)
        == DEFAULT_LOCAL_CONNECTION_PROBE_TIMEOUT
    )
    assert (
        connection_probe_timeout(900, remote=True)
        == DEFAULT_REMOTE_CONNECTION_PROBE_TIMEOUT
    )
    assert connection_probe_timeout(15, remote=False) == 15

