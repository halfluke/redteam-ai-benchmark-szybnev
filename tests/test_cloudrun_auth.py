"""Tests for Cloud Run identity token auth."""

import base64
import json
import subprocess
import time
from unittest.mock import MagicMock, patch

import pytest
import requests

from models.cloudrun_auth import (
    CloudRunIdentityTokenProvider,
    fetch_cloudrun_identity_token,
    normalize_cloudrun_audience,
)
from models.lmstudio import LMStudioClient
from models.ollama import OllamaClient
from utils.config import load_config


def _make_jwt(exp: int) -> str:
    header = base64.urlsafe_b64encode(b'{"alg":"none"}').decode().rstrip("=")
    payload = base64.urlsafe_b64encode(
        json.dumps({"exp": exp}).encode()
    ).decode().rstrip("=")
    return f"{header}.{payload}.sig"


def test_normalize_cloudrun_audience_strips_path():
    url = "https://svc-abc.run.app/api/chat"
    assert normalize_cloudrun_audience(url) == "https://svc-abc.run.app"


def test_fetch_cloudrun_identity_token_uses_audiences_for_service_account():
    with patch("models.cloudrun_auth.subprocess.run") as run:
        run.return_value = MagicMock(stdout="token-abc\n", returncode=0)
        token = fetch_cloudrun_identity_token("https://svc.run.app/v1/chat")
    assert token == "token-abc"
    args = run.call_args[0][0]
    assert args[0:3] == ["gcloud", "auth", "print-identity-token"]
    assert args[3] == "--audiences=https://svc.run.app"


def test_fetch_cloudrun_identity_token_falls_back_for_user_account():
    sa_error = subprocess.CalledProcessError(
        1,
        "gcloud",
        stderr="Invalid account type for `--audiences`. Requires valid service account.",
    )

    def side_effect(cmd, **kwargs):
        if any(str(arg).startswith("--audiences=") for arg in cmd):
            raise sa_error
        return MagicMock(stdout="user-token\n", returncode=0)

    with patch("models.cloudrun_auth.subprocess.run", side_effect=side_effect) as run:
        token = fetch_cloudrun_identity_token("https://svc.run.app")
    assert token == "user-token"
    assert run.call_count == 2


def test_provider_refreshes_before_expiry():
    now = int(time.time())
    first = _make_jwt(now + 3600)
    second = _make_jwt(now + 7200)
    provider = CloudRunIdentityTokenProvider(
        "https://svc.run.app",
        refresh_margin_s=600,
    )
    with patch(
        "models.cloudrun_auth.fetch_cloudrun_identity_token",
        side_effect=[first, second],
    ) as fetch:
        assert provider.get_token() == first
        assert provider.get_token() == first
        provider._expires_at = now + 100
        assert provider.get_token() == second
    assert fetch.call_count == 2


def test_provider_invalidate_forces_refresh():
    provider = CloudRunIdentityTokenProvider("https://svc.run.app")
    with patch(
        "models.cloudrun_auth.fetch_cloudrun_identity_token",
        side_effect=["a", "b"],
    ) as fetch:
        assert provider.get_token() == "a"
        provider.invalidate()
        assert provider.get_token() == "b"
    assert fetch.call_count == 2


def test_cloudrun_ollama_config_loads_auth():
    config = load_config("configs/cloudrun_ollama.yaml")
    assert config.provider.auth == "cloudrun_identity"
    assert config.provider.endpoint.startswith("https://")


def test_ollama_client_retries_on_401_with_token_refresh():
    client = OllamaClient(
        "https://svc.run.app",
        "model",
        auth_token_getter=lambda: "fresh-token",
        invalidate_auth_token=lambda: None,
    )
    unauthorized = MagicMock()
    unauthorized.status_code = 401
    unauthorized.text = "Unauthorized"
    unauthorized.raise_for_status.side_effect = requests.exceptions.HTTPError(
        response=unauthorized
    )
    ok = MagicMock()
    ok.status_code = 200
    ok.json.return_value = {"message": {"content": "ok"}}
    ok.raise_for_status.return_value = None
    client.session.post = MagicMock(side_effect=[unauthorized, ok])

    result = client.query("hi", max_tokens=8, retries=2)

    assert result == "ok"
    assert client.session.post.call_count == 2
    second_headers = client.session.post.call_args_list[1].kwargs["headers"]
    assert second_headers["Authorization"] == "Bearer fresh-token"


def test_lmstudio_client_sends_bearer_token():
    client = LMStudioClient(
        "https://svc.run.app",
        "model",
        auth_token_getter=lambda: "tok",
    )
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"choices": [{"message": {"content": "hi"}}]}
    response.raise_for_status.return_value = None
    client.session.post = MagicMock(return_value=response)

    assert client.query("prompt", max_tokens=8) == "hi"
    headers = client.session.post.call_args.kwargs["headers"]
    assert headers["Authorization"] == "Bearer tok"
