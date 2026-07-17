"""LLM API client implementations."""

import importlib.util
import os
from typing import Callable, Optional

from .base import APIClient
from .validation import (
    ModelListError,
    ModelNotFoundError,
    is_model_listed,
    validate_model_available,
)
from .bearer_auth import BearerAuthMixin
from .cloudrun_auth import create_cloudrun_identity_auth
from .lmstudio import LMStudioClient
from .ollama import OllamaClient
from .openwebui import OpenWebUIClient

_httpx_available = importlib.util.find_spec("httpx") is not None
_tenacity_available = importlib.util.find_spec("tenacity") is not None

if _httpx_available and _tenacity_available:
    from .openrouter import OpenRouterClient

    OPENROUTER_AVAILABLE = True
else:
    OpenRouterClient = None  # type: ignore
    OPENROUTER_AVAILABLE = False

__all__ = [
    "APIClient",
    "BearerAuthMixin",
    "LMStudioClient",
    "OllamaClient",
    "OpenWebUIClient",
    "OpenRouterClient",
    "OPENROUTER_AVAILABLE",
    "create_client",
    "provider_auth_kwargs",
    "is_model_listed",
    "validate_model_available",
    "ModelListError",
    "ModelNotFoundError",
]


def provider_auth_kwargs(
    *,
    auth: Optional[str] = None,
    endpoint: Optional[str] = None,
    cloudrun_audience: Optional[str] = None,
    cloudrun_impersonate_service_account: Optional[str] = None,
) -> dict:
    """Build optional auth kwargs for ``create_client`` from provider config."""
    if auth != "cloudrun_identity":
        return {}
    if not endpoint:
        raise ValueError("endpoint is required for cloudrun_identity auth")
    audience = cloudrun_audience or endpoint
    getter, invalidate = create_cloudrun_identity_auth(
        audience,
        impersonate_service_account=cloudrun_impersonate_service_account,
    )
    return {
        "auth_token_getter": getter,
        "invalidate_auth_token": invalidate,
    }


def create_client(
    provider: str,
    endpoint: Optional[str],
    model: str,
    api_key: Optional[str] = None,
    timeout: Optional[int] = None,
    keep_alive: Optional[str] = None,
    auth_token_getter: Optional[Callable[[], str]] = None,
    invalidate_auth_token: Optional[Callable[[], None]] = None,
) -> APIClient:
    """
    Create appropriate API client based on provider.

    Args:
        provider: Provider name ("lmstudio", "ollama", "openwebui", "openrouter", "deepinfra")
        endpoint: Custom endpoint URL (optional)
        model: Model name/ID
        api_key: API key (static bearer token for reverse-proxy or OpenRouter)
        timeout: Optional request timeout in seconds
        keep_alive: Optional Ollama keep_alive value
        auth_token_getter: Callable returning a refreshed bearer token (Cloud Run)
        invalidate_auth_token: Callable to invalidate the cached token on 401

    Returns:
        Configured APIClient instance
    """
    if endpoint is None:
        if provider == "lmstudio":
            endpoint = "http://localhost:1234"
        elif provider == "ollama":
            endpoint = "http://localhost:11434"
        elif provider == "openwebui":
            endpoint = "http://localhost:3000"
        elif provider == "openrouter":
            endpoint = "https://openrouter.ai/api/v1"
        elif provider == "deepinfra":
            endpoint = "https://api.deepinfra.com/v1/openai"
        else:
            raise ValueError(f"Unknown provider: {provider}")

    client_timeout = timeout if timeout is not None else 150

    if provider == "lmstudio":
        return LMStudioClient(
            endpoint,
            model,
            timeout=client_timeout,
            api_key=api_key,
            auth_token_getter=auth_token_getter,
            invalidate_auth_token=invalidate_auth_token,
        )
    elif provider == "ollama":
        return OllamaClient(
            endpoint,
            model,
            timeout=client_timeout,
            api_key=api_key,
            auth_token_getter=auth_token_getter,
            invalidate_auth_token=invalidate_auth_token,
            keep_alive=keep_alive,
        )
    elif provider == "openwebui":
        return OpenWebUIClient(
            endpoint,
            model,
            api_key=api_key,
            timeout=client_timeout,
        )
    elif provider == "openrouter":
        if not OPENROUTER_AVAILABLE:
            raise RuntimeError(
                "OpenRouter requires httpx and tenacity. "
                "Install with: pip install httpx tenacity"
            )
        return OpenRouterClient(
            endpoint,
            model,
            api_key=api_key,
            timeout=timeout if timeout is not None else 120,
        )
    elif provider == "deepinfra":
        if not OPENROUTER_AVAILABLE:
            raise RuntimeError(
                "DeepInfra requires httpx and tenacity. "
                "Install with: pip install httpx tenacity"
            )
        return OpenRouterClient(
            endpoint,
            model,
            api_key=api_key or os.environ.get("DEEPINFRA_TOKEN"),
            timeout=timeout if timeout is not None else 120,
            api_key_env="DEEPINFRA_TOKEN",
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")
