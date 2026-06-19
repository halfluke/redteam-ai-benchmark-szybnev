"""Base classes for LLM API clients."""

import json
import threading
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import requests

if TYPE_CHECKING:
    from .diagnostics import QueryDiagnostics

# Health-check probes (/api/tags, /v1/models).
DEFAULT_LOCAL_CONNECTION_PROBE_TIMEOUT = 30
DEFAULT_REMOTE_CONNECTION_PROBE_TIMEOUT = 180


def connection_probe_timeout(query_timeout: int, *, remote: bool = False) -> int:
    """Cap probe wait time separately from long-running generation requests."""
    cap = (
        DEFAULT_REMOTE_CONNECTION_PROBE_TIMEOUT
        if remote
        else DEFAULT_LOCAL_CONNECTION_PROBE_TIMEOUT
    )
    return min(query_timeout, cap)


class APIClient(ABC):
    """Abstract base class for LLM API clients."""

    def __init__(self, base_url: str, model_name: str):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name

    @abstractmethod
    def query(
        self,
        prompt: str,
        max_tokens: int = 1024,
        retries: int = 3,
        temperature: float = 0.2,
    ) -> str:
        """Query the LLM API with retry logic."""
        pass

    @abstractmethod
    def list_models(self) -> List[Dict]:
        """List available models."""
        pass

    @abstractmethod
    def test_connection(self) -> bool:
        """Test if API is accessible."""
        pass

    def close(self) -> None:
        """Close any persistent client resources."""
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


class RequestsRetryMixin:
    """Shared retry/error handling for requests-based model clients."""

    provider_name: str
    base_url: str
    session: requests.Session
    timeout: int

    def _init_diagnostics_storage(self) -> None:
        if not hasattr(self, "_diagnostics_local"):
            self._diagnostics_local = threading.local()

    @property
    def last_query_diagnostics(self) -> Optional["QueryDiagnostics"]:
        self._init_diagnostics_storage()
        return getattr(self._diagnostics_local, "value", None)

    @last_query_diagnostics.setter
    def last_query_diagnostics(self, value: Optional["QueryDiagnostics"]) -> None:
        self._init_diagnostics_storage()
        self._diagnostics_local.value = value

    def _post_json_with_retries(
        self,
        *,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        retries: int,
        diagnostics: Optional["QueryDiagnostics"] = None,
    ) -> Dict[str, Any]:
        """POST JSON with retry behavior used by local providers."""
        if diagnostics is not None:
            diagnostics.retries = retries
            diagnostics.timeout_s = self.timeout

        for attempt in range(retries):
            started = time.time()
            if diagnostics is not None:
                diagnostics.attempt = attempt + 1

            try:
                response = self.session.post(
                    url, headers=headers, json=payload, timeout=self.timeout
                )
                response.raise_for_status()
                if diagnostics is not None:
                    diagnostics.elapsed_ms = (time.time() - started) * 1000
                    diagnostics.status = "success"
                return response.json()

            except requests.exceptions.Timeout:
                elapsed_ms = (time.time() - started) * 1000
                if diagnostics is not None:
                    diagnostics.elapsed_ms = elapsed_ms
                    diagnostics.status = "timeout"
                    diagnostics.error = (
                        f"HTTP read timeout after {elapsed_ms / 1000:.1f}s "
                        f"(limit {self.timeout}s)"
                    )
                    print(f"   {diagnostics.format_summary()}")
                    print(f"      {diagnostics.format_details()}")
                else:
                    print(f"   Timeout on attempt {attempt + 1}/{retries}")
                if attempt == retries - 1:
                    raise RuntimeError(
                        f"API timeout after {retries} attempts "
                        f"({self.timeout}s limit per attempt)"
                    ) from None
                time.sleep(2**attempt)

            except requests.exceptions.ConnectionError as e:
                raise RuntimeError(
                    f"Cannot connect to {self.provider_name} at {self.base_url}. "
                    "Is it running?"
                ) from e

            except requests.exceptions.HTTPError as e:
                if (
                    e.response.status_code == 401
                    and attempt < retries - 1
                    and hasattr(self, "_refresh_auth_headers")
                    and self._refresh_auth_headers(headers)
                ):
                    continue
                if e.response.status_code == 429:
                    print("   Rate limited, waiting...")
                    time.sleep(5)
                    continue
                raise RuntimeError(
                    f"API error {e.response.status_code}: {e.response.text}"
                ) from e

            except (KeyError, json.JSONDecodeError) as e:
                raise RuntimeError(f"Invalid API response format: {e}") from e

        raise RuntimeError("Max retries exceeded")
