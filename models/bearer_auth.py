"""Shared Bearer-token header helpers for HTTP API clients."""

from typing import Callable, Dict, Optional


class BearerAuthMixin:
    """Attach static or dynamically refreshed Bearer tokens to requests."""

    api_key: Optional[str]
    auth_token_getter: Optional[Callable[[], str]]
    invalidate_auth_token: Optional[Callable[[], None]]

    def _resolve_bearer_token(self) -> Optional[str]:
        if self.auth_token_getter is not None:
            return self.auth_token_getter()
        return self.api_key

    def _auth_headers(self, headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        merged = {"Content-Type": "application/json"}
        if headers:
            merged.update(headers)
        token = self._resolve_bearer_token()
        if token:
            merged["Authorization"] = f"Bearer {token}"
        return merged

    def _refresh_auth_headers(self, headers: Dict[str, str]) -> bool:
        """Invalidate cached token and rebuild Authorization header."""
        if self.auth_token_getter is None:
            return False
        if self.invalidate_auth_token is not None:
            self.invalidate_auth_token()
        token = self._resolve_bearer_token()
        if not token:
            return False
        headers["Authorization"] = f"Bearer {token}"
        return True
