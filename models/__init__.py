"""LLM API client implementations."""

import importlib.util
from typing import Callable, Optional

from .base import APIClient
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
    "LMStudioClient",
    "OllamaClient",
    "OpenWebUIClient",
    "OpenRouterClient",
    "OPENROUTER_AVAILABLE",
    "create_client",
    "provider_auth_kwargs",
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
    auth_token_getter: Optional[Callable[[], str]] = None,
    invalidate_auth_token: Optional[Callable[[], None]] = None,
) -> APIClient:
    """
    Create appropriate API client based on provider.

    Args:
        provider: Provider name ("lmstudio", "ollama", "openwebui", "openrouter")
        endpoint: Custom endpoint URL (optional)
        model: Model name/ID
        api_key: API key for providers that require it (e.g., OpenRouter)
        timeout: Optional request timeout in seconds

    Returns:
        Configured APIClient instance
    """
    # Set default endpoints
    if endpoint is None:
        if provider == "lmstudio":
            endpoint = "http://localhost:1234"
        elif provider == "ollama":
            endpoint = "http://localhost:11434"
        elif provider == "openwebui":
            endpoint = "http://localhost:3000"
        elif provider == "openrouter":
            endpoint = "https://openrouter.ai/api/v1"
        else:
            raise ValueError(f"Unknown provider: {provider}")

    # Create client
    client_timeout = timeout if timeout is not None else 150
    auth_kwargs = {
        "auth_token_getter": auth_token_getter,
        "invalidate_auth_token": invalidate_auth_token,
    }
    if provider == "lmstudio":
        return LMStudioClient(
            endpoint,
            model,
            timeout=client_timeout,
            api_key=api_key,
            **auth_kwargs,
        )
    elif provider == "ollama":
        return OllamaClient(
            endpoint,
            model,
            timeout=client_timeout,
            api_key=None if auth_token_getter else api_key,
            **auth_kwargs,
        )
    elif provider == "openwebui":
        return OpenWebUIClient(
            endpoint,
            model,
            api_key=api_key,
            timeout=timeout if timeout is not None else 150,
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
    else:
        raise ValueError(f"Unknown provider: {provider}")
