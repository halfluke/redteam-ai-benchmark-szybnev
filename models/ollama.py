"""Ollama API client."""

import os
from typing import Callable, Dict, List, Optional

import requests

from .base import APIClient, RequestsRetryMixin, connection_probe_timeout
from .bearer_auth import BearerAuthMixin
from .diagnostics import QueryDiagnostics, ns_to_ms


class OllamaClient(BearerAuthMixin, RequestsRetryMixin, APIClient):
    """Ollama API client."""

    provider_name = "Ollama"

    def __init__(
        self,
        base_url: str,
        model_name: str,
        timeout: int = 150,
        api_key: Optional[str] = None,
        auth_token_getter: Optional[Callable[[], str]] = None,
        invalidate_auth_token: Optional[Callable[[], None]] = None,
        keep_alive: Optional[str] = None,
    ):
        super().__init__(base_url, model_name)
        self.timeout = timeout
        self.api_key = api_key or os.environ.get("OLLAMA_API_KEY")
        self.auth_token_getter = auth_token_getter
        self.invalidate_auth_token = invalidate_auth_token
        self.keep_alive = keep_alive
        self.session = requests.Session()

    def _get_headers(self) -> Dict[str, str]:
        """Request headers, including optional Cloud Run / reverse-proxy auth."""
        return self._auth_headers()

    def _begin_query_diagnostics(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> QueryDiagnostics:
        diagnostics = QueryDiagnostics(
            provider=self.provider_name,
            endpoint=self.base_url,
            model=self.model_name,
            prompt_chars=len(prompt),
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_s=self.timeout,
        )
        self.last_query_diagnostics = diagnostics
        return diagnostics

    @staticmethod
    def _apply_ollama_timing(diagnostics: QueryDiagnostics, data: dict) -> None:
        """Attach Ollama timing fields when the API returns them."""
        diagnostics.total_duration_ms = ns_to_ms(data.get("total_duration"))
        diagnostics.load_duration_ms = ns_to_ms(data.get("load_duration"))
        diagnostics.prompt_eval_duration_ms = ns_to_ms(data.get("prompt_eval_duration"))
        diagnostics.eval_duration_ms = ns_to_ms(data.get("eval_duration"))
        diagnostics.prompt_eval_count = data.get("prompt_eval_count")
        diagnostics.eval_count = data.get("eval_count")

    @staticmethod
    def _extract_assistant_text(message: dict) -> str:
        """Return assistant text from Ollama chat response.

        Some reasoning models (e.g. Tongyi DeepResearch) emit chain-of-thought in
        ``thinking`` while leaving ``content`` empty when generation hits the token
        limit during the think phase.
        """
        content = message.get("content") or ""
        if content.strip():
            return content
        thinking = message.get("thinking") or ""
        if thinking.strip():
            return thinking
        return content

    def query(
        self,
        prompt: str,
        max_tokens: int = 768,
        retries: int = 3,
        temperature: float = 0.2,
        system: Optional[str] = None,
    ) -> str:
        """Query Ollama API with retry logic."""
        url = f"{self.base_url}/api/chat"
        headers = self._get_headers()
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive

        prompt_chars = len(prompt) + (len(system) if system else 0)
        diagnostics = self._begin_query_diagnostics(
            prompt if not system else f"{system}\n\n{prompt}",
            max_tokens,
            temperature,
        )
        diagnostics.prompt_chars = prompt_chars
        data = self._post_json_with_retries(
            url=url,
            headers=headers,
            payload=payload,
            retries=retries,
            diagnostics=diagnostics,
        )
        self._apply_ollama_timing(diagnostics, data)
        try:
            return self._extract_assistant_text(data["message"])
        except KeyError as e:
            diagnostics.status = "error"
            diagnostics.error = f"Invalid API response format: {e}"
            raise RuntimeError(f"Invalid API response format: {e}") from e

    def list_models(self) -> List[Dict]:
        """List available models from Ollama."""
        try:
            url = f"{self.base_url}/api/tags"
            response = self.session.get(url, headers=self._get_headers(), timeout=5)
            response.raise_for_status()
            data = response.json()
            return data.get("models", [])
        except Exception as e:
            raise RuntimeError(f"Failed to list models: {e}") from e

    def test_connection(self) -> bool:
        """Test Ollama connection."""
        try:
            url = f"{self.base_url}/api/tags"
            response = self.session.get(
                url,
                headers=self._get_headers(),
                timeout=connection_probe_timeout(
                    self.timeout, remote=self.auth_token_getter is not None
                ),
            )
            return response.status_code == 200
        except Exception:
            return False

    def close(self) -> None:
        """Close the persistent HTTP session."""
        self.session.close()
