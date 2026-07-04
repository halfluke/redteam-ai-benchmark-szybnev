"""Google Cloud Run identity token helpers for direct HTTPS access."""

from __future__ import annotations

import base64
import json
import subprocess
import time
from typing import Callable, Optional, Tuple
from urllib.parse import urlparse


def normalize_cloudrun_audience(url: str) -> str:
    """Return the Cloud Run JWT audience (service origin, no path)."""
    parsed = urlparse(url.strip())
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid Cloud Run URL for audience: {url!r}")
    return f"{parsed.scheme}://{parsed.netloc}".rstrip("/")


def _jwt_exp_unix(token: str) -> Optional[int]:
    try:
        payload_segment = token.split(".")[1]
        padding = "=" * (-len(payload_segment) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_segment + padding))
        exp = payload.get("exp")
        return int(exp) if exp is not None else None
    except (IndexError, ValueError, TypeError, json.JSONDecodeError):
        return None


def fetch_cloudrun_identity_token(
    audience: str,
    *,
    gcloud_bin: str = "gcloud",
    timeout_s: int = 30,
    impersonate_service_account: Optional[str] = None,
) -> str:
    """Fetch a fresh identity token via ``gcloud auth print-identity-token``.

    Service accounts use ``--audiences=<service URL>``. User accounts (``gcloud auth login``)
    cannot pass ``--audiences``; we fall back to a plain identity token, which Cloud Run
    accepts when the user has ``roles/run.invoker``.
    """
    audience = normalize_cloudrun_audience(audience)
    attempts: list[list[str]] = []

    if impersonate_service_account:
        attempts.append(
            [
                gcloud_bin,
                "auth",
                "print-identity-token",
                f"--impersonate-service-account={impersonate_service_account}",
                f"--audiences={audience}",
            ]
        )
    else:
        attempts.append(
            [
                gcloud_bin,
                "auth",
                "print-identity-token",
                f"--audiences={audience}",
            ]
        )
        attempts.append([gcloud_bin, "auth", "print-identity-token"])

    last_error = ""
    for cmd in attempts:
        try:
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=timeout_s,
            )
        except FileNotFoundError as e:
            raise RuntimeError(
                "gcloud CLI not found. Install Google Cloud SDK or use "
                "'gcloud run services proxy' instead."
            ) from e
        except subprocess.CalledProcessError as e:
            stderr = (e.stderr or e.stdout or "").strip()
            last_error = stderr or str(e)
            if "Invalid account type" in last_error and len(attempts) > 1:
                continue
            raise RuntimeError(
                f"Failed to fetch Cloud Run identity token: {last_error}"
            ) from e
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                f"gcloud identity token request timed out after {timeout_s}s"
            ) from e
        else:
            token = completed.stdout.strip()
            if not token:
                last_error = "gcloud returned an empty identity token"
                continue
            return token

    raise RuntimeError(
        f"Failed to fetch Cloud Run identity token: {last_error or 'unknown error'}"
    )


class CloudRunIdentityTokenProvider:
    """Cache and refresh Cloud Run identity tokens for long benchmark runs."""

    def __init__(
        self,
        audience: str,
        *,
        refresh_margin_s: int = 300,
        gcloud_bin: str = "gcloud",
        impersonate_service_account: Optional[str] = None,
    ):
        self.audience = normalize_cloudrun_audience(audience)
        self.refresh_margin_s = refresh_margin_s
        self.gcloud_bin = gcloud_bin
        self.impersonate_service_account = impersonate_service_account
        self._token: Optional[str] = None
        self._expires_at: Optional[int] = None

    def invalidate(self) -> None:
        """Drop the cached token so the next call refreshes."""
        self._token = None
        self._expires_at = None

    def _needs_refresh(self) -> bool:
        if not self._token:
            return True
        if self._expires_at is None:
            return True
        return time.time() >= self._expires_at - self.refresh_margin_s

    def get_token(self) -> str:
        """Return a valid identity token, refreshing when near expiry."""
        if not self._needs_refresh():
            assert self._token is not None
            return self._token

        token = fetch_cloudrun_identity_token(
            self.audience,
            gcloud_bin=self.gcloud_bin,
            impersonate_service_account=self.impersonate_service_account,
        )
        self._token = token
        self._expires_at = _jwt_exp_unix(token)
        return token


def create_cloudrun_identity_auth(
    audience: str,
    *,
    impersonate_service_account: Optional[str] = None,
) -> Tuple[Callable[[], str], Callable[[], None]]:
    """Return (token_getter, invalidate) for API clients."""
    provider = CloudRunIdentityTokenProvider(
        audience,
        impersonate_service_account=impersonate_service_account,
    )
    return provider.get_token, provider.invalidate
