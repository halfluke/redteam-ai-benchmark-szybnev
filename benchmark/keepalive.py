"""Background keepalive pings to keep target and optimizer models warm."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Callable, Dict, List, Optional, Tuple

from models.lmstudio import LMStudioClient
from models.ollama import OllamaClient


def ping_model_client(
    client,
    *,
    prompt: str = "Say OK",
    max_tokens: int = 16,
    timeout_s: int = 90,
    keep_alive: Optional[str] = None,
) -> bool:
    """Send a minimal generation request without touching main query diagnostics."""
    if isinstance(client, OllamaClient):
        return _ping_ollama(client, prompt, max_tokens, timeout_s, keep_alive)
    if isinstance(client, LMStudioClient):
        return _ping_lmstudio(client, prompt, max_tokens, timeout_s)
    try:
        client.query(prompt, max_tokens=max_tokens, retries=1, temperature=0)
        return True
    except Exception:
        return False


def _ping_ollama(
    client: OllamaClient,
    prompt: str,
    max_tokens: int,
    timeout_s: int,
    keep_alive: Optional[str] = None,
) -> bool:
    url = f"{client.base_url}/api/chat"
    headers = client._get_headers()
    payload = {
        "model": client.model_name,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": max_tokens,
        },
    }
    ollama_keep_alive = keep_alive if keep_alive is not None else getattr(client, "keep_alive", None)
    if ollama_keep_alive is not None:
        payload["keep_alive"] = ollama_keep_alive
    try:
        response = client.session.post(
            url, json=payload, headers=headers, timeout=timeout_s
        )
        response.raise_for_status()
        return True
    except Exception:
        return False


def _ping_lmstudio(
    client: LMStudioClient,
    prompt: str,
    max_tokens: int,
    timeout_s: int,
) -> bool:
    url = f"{client.base_url}/v1/chat/completions"
    headers = client._auth_headers()
    payload = {
        "model": client.model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    try:
        response = client.session.post(
            url, json=payload, headers=headers, timeout=timeout_s
        )
        response.raise_for_status()
        return True
    except Exception:
        return False


@contextmanager
def keepalive_busy(keepalive: Optional["ModelKeepalive"], role: str):
    """Mark a model busy for the duration of a main benchmark query."""
    if keepalive is None:
        yield
        return
    keepalive.mark_busy(role)
    try:
        yield
    finally:
        keepalive.mark_idle(role)


class ModelKeepalive:
    """Ping idle models on an interval so they stay loaded between long requests."""

    def __init__(
        self,
        endpoints: List[Tuple[str, object]],
        *,
        interval_s: float = 60,
        prompt: str = "Say OK",
        max_tokens: int = 16,
        timeout_s: int = 90,
        optimizer_ollama_keep_alive: Optional[str] = "30m",
        on_ping: Optional[Callable[[str, bool], None]] = None,
    ):
        self._endpoints: List[Tuple[str, object]] = list(endpoints)
        self.interval_s = interval_s
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.optimizer_ollama_keep_alive = optimizer_ollama_keep_alive
        self._on_ping = on_ping
        self._busy: set[str] = set()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def mark_busy(self, role: str) -> None:
        with self._lock:
            self._busy.add(role)

    def mark_idle(self, role: str) -> None:
        with self._lock:
            self._busy.discard(role)

    def idle_roles(self) -> set[str]:
        with self._lock:
            busy = set(self._busy)
        return {name for name, _ in self._endpoints if name not in busy}

    def ping_role(self, role: str) -> bool:
        client = self._client_for(role)
        if client is None:
            return False
        # Target (Cloud Run / vLLM): plain traffic ping only — no Ollama keep_alive.
        # Optimizer (local Ollama): keep_alive extends model load between long target calls.
        keep_alive = self.optimizer_ollama_keep_alive if role == "optimizer" else None
        ok = ping_model_client(
            client,
            prompt=self.prompt,
            max_tokens=self.max_tokens,
            timeout_s=self.timeout_s,
            keep_alive=keep_alive,
        )
        if self._on_ping:
            self._on_ping(role, ok)
        return ok

    def warmup(self) -> Dict[str, bool]:
        """Synchronously ping every endpoint once before the benchmark starts."""
        results = {}
        for role, _ in self._endpoints:
            results[role] = self.ping_role(role)
        return results

    def ping_idle_endpoints(self) -> Dict[str, bool]:
        """Ping endpoints that are not currently handling a main request."""
        idle = self.idle_roles()
        results = {}
        for role in idle:
            results[role] = self.ping_role(role)
        return results

    def start(self) -> None:
        """Warm up all models and start the background keepalive loop."""
        warmup = self.warmup()
        for role, ok in warmup.items():
            status = "ok" if ok else "failed"
            print(f"   Keepalive warmup ({role}): {status}")
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="model-keepalive",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the background keepalive loop."""
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=self.timeout_s + 5)
        self._thread = None

    def __enter__(self) -> "ModelKeepalive":
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop()

    def _client_for(self, role: str):
        for name, client in self._endpoints:
            if name == role:
                return client
        return None

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.ping_idle_endpoints()
            if self._stop.wait(self.interval_s):
                break
