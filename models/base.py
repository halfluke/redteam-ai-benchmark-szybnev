"""Base classes for LLM API clients."""

import json
import threading
import time
from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import Any, Dict, List, Optional

import requests


def normalize_message_content(content: Any) -> str:
    """Coerce chat ``content`` fields to a string (providers may return null)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return str(content)


class APIClient(ABC):
    """Abstract base class for LLM API clients."""

    def __init__(self, base_url: str, model_name: str):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self._request_lock = threading.Lock()

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
    _request_lock: threading.Lock
    last_probe_error: Optional[str]

    def _probe_attempts(self) -> int:
        """Remote HTTPS probes retry through cold-start blips; local probes fail fast."""
        return 10 if self.base_url.startswith("https://") else 1

    def _probe_get_with_retries(
        self,
        url: str,
        headers: Dict[str, str],
        *,
        timeout: int,
        retries: Optional[int] = None,
    ) -> bool:
        """GET connectivity probe with retries for Cloud Run cold starts."""
        if retries is None:
            retries = self._probe_attempts()
        self.last_probe_error = None
        lock = getattr(self, "_request_lock", None)
        lock_ctx = lock if lock is not None else nullcontext()

        for attempt in range(retries):
            try:
                with lock_ctx:
                    response = self.session.get(url, headers=headers, timeout=timeout)
                if response.status_code == 200:
                    return True

                status = response.status_code
                if status == 401:
                    refresh = getattr(self, "_refresh_auth_headers", None)
                    if (
                        callable(refresh)
                        and attempt < retries - 1
                        and refresh(headers)
                    ):
                        print(
                            f"   Probe auth expired on attempt {attempt + 1}/{retries}, "
                            "refreshing token..."
                        )
                        continue
                    self.last_probe_error = f"HTTP {status} (authentication failed)"
                    return False

                if status in {502, 503, 504} and attempt < retries - 1:
                    delay = min(2**attempt, 30)
                    print(
                        f"   Probe got HTTP {status} on attempt {attempt + 1}/{retries}, "
                        f"retrying in {delay}s (cold start?)"
                    )
                    time.sleep(delay)
                    continue

                body_preview = (response.text or "").strip().replace("\n", " ")[:120]
                if body_preview:
                    self.last_probe_error = f"HTTP {status}: {body_preview}"
                else:
                    self.last_probe_error = f"HTTP {status}"
                return False

            except requests.exceptions.Timeout:
                if attempt < retries - 1:
                    delay = min(2**attempt, 30)
                    print(
                        f"   Probe timeout on attempt {attempt + 1}/{retries}, "
                        f"retrying in {delay}s..."
                    )
                    time.sleep(delay)
                    continue
                self.last_probe_error = (
                    f"timed out after {retries} attempt(s) ({timeout}s each)"
                )
                return False

            except requests.exceptions.ConnectionError as e:
                if attempt < retries - 1:
                    delay = min(2**attempt, 30)
                    print(
                        f"   Probe connection error on attempt {attempt + 1}/{retries}, "
                        f"retrying in {delay}s..."
                    )
                    time.sleep(delay)
                    continue
                self.last_probe_error = f"connection error: {e}"
                return False

            except Exception as e:
                self.last_probe_error = str(e)
                return False

        self.last_probe_error = f"probe failed after {retries} attempt(s)"
        return False

    def _post_json_with_retries(
        self,
        *,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        retries: int,
    ) -> Dict[str, Any]:
        """POST JSON with retry behavior used by local providers."""
        lock = getattr(self, "_request_lock", None)
        lock_ctx = lock if lock is not None else nullcontext()
        for attempt in range(retries):
            try:
                with lock_ctx:
                    response = self.session.post(
                        url, headers=headers, json=payload, timeout=self.timeout
                    )
                response.raise_for_status()
                return response.json()

            except requests.exceptions.Timeout:
                print(f"   Timeout on attempt {attempt + 1}/{retries}")
                if attempt == retries - 1:
                    raise RuntimeError(
                        f"API timeout after {retries} attempts"
                    ) from None
                time.sleep(2**attempt)

            except requests.exceptions.ConnectionError as e:
                print(f"   Connection error on attempt {attempt + 1}/{retries}")
                if attempt == retries - 1:
                    raise RuntimeError(
                        f"Cannot connect to {self.provider_name} at {self.base_url}. "
                        "Is it running?"
                    ) from e
                time.sleep(2**attempt)

            except requests.exceptions.HTTPError as e:
                status = e.response.status_code
                if status == 401:
                    refresh = getattr(self, "_refresh_auth_headers", None)
                    if (
                        callable(refresh)
                        and attempt < retries - 1
                        and refresh(headers)
                    ):
                        print(
                            f"   Auth expired on attempt {attempt + 1}/{retries}, "
                            "refreshing token..."
                        )
                        continue
                    raise RuntimeError(
                        f"API error {status}: {e.response.text}"
                    ) from e
                if status == 429:
                    print("   Rate limited, waiting...")
                    time.sleep(5)
                    continue
                if status >= 500:
                    if attempt < retries - 1:
                        print(
                            f"   Server error {status} on attempt {attempt + 1}/{retries}, retrying..."
                        )
                        time.sleep(2**attempt)
                        continue
                raise RuntimeError(
                    f"API error {status}: {e.response.text}"
                ) from e

            except (KeyError, json.JSONDecodeError) as e:
                raise RuntimeError(f"Invalid API response format: {e}") from e

        raise RuntimeError("Max retries exceeded")
