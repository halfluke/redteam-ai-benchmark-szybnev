"""LM Studio API client (OpenAI-compatible)."""

from typing import Callable, Dict, List, Optional

import requests

from .base import APIClient, RequestsRetryMixin
from .bearer_auth import BearerAuthMixin


class LMStudioClient(BearerAuthMixin, RequestsRetryMixin, APIClient):
    """LM Studio API client (OpenAI-compatible)."""

    provider_name = "LM Studio"

    def __init__(
        self,
        base_url: str,
        model_name: str,
        timeout: int = 150,
        api_key: Optional[str] = None,
        auth_token_getter: Optional[Callable[[], str]] = None,
        invalidate_auth_token: Optional[Callable[[], None]] = None,
    ):
        super().__init__(base_url, model_name)
        self.timeout = timeout
        self.api_key = api_key
        self.auth_token_getter = auth_token_getter
        self.invalidate_auth_token = invalidate_auth_token
        self.session = requests.Session()

    def query(
        self,
        prompt: str,
        max_tokens: int = 768,
        retries: int = 3,
        temperature: float = 0.2,
    ) -> str:
        """Query LM Studio API with retry logic."""
        url = f"{self.base_url}/v1/chat/completions"
        headers = self._auth_headers()
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        data = self._post_json_with_retries(
            url=url, headers=headers, payload=payload, retries=retries
        )
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Invalid API response format: {e}") from e

    def list_models(self) -> List[Dict]:
        """List available models from LM Studio."""
        try:
            url = f"{self.base_url}/v1/models"
            response = self.session.get(url, headers=self._auth_headers(), timeout=5)
            response.raise_for_status()
            data = response.json()
            return data.get("data", [])
        except Exception as e:
            raise RuntimeError(f"Failed to list models: {e}") from e

    def test_connection(self) -> bool:
        """Test LM Studio connection."""
        try:
            url = f"{self.base_url}/v1/models"
            response = self.session.get(url, headers=self._auth_headers(), timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def close(self) -> None:
        """Close the persistent HTTP session."""
        self.session.close()
