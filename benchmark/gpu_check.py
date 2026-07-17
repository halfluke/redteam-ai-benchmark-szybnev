"""Pre-flight GPU residency check.

Runs before the paid benchmark starts and aborts early if Ollama itself
reports the target model is not actually resident in GPU VRAM. This catches
silent CPU fallback (e.g. a broken Ollama GPU backend on Cloud Run -- see
GCP-CLOUDRUN-AImodels/README.md for the underlying incident this guards
against) before it burns a full run's worth of time and money.

This intentionally does NOT try to parse Ollama server logs or query GCP
Cloud Logging: the benchmark client only ever talks to the model's HTTP API
(the same code path works for local Ollama, LM Studio, OpenWebUI, and Cloud
Run), and neither of those log sources is reachable that way -- Cloud
Logging would additionally need a GCP project/service id (the client only
has the HTTPS URL), separate IAM permissions, and has ingestion delay.
Instead, Ollama's own `/api/ps` endpoint reports exactly how many bytes of
the loaded model are resident in GPU VRAM (`size_vram`) versus its total
size (`size`) -- an authoritative, provider-native signal that needs no
per-model speed calibration.
"""

from __future__ import annotations

from typing import Optional, Tuple

from models.ollama import OllamaClient


class GpuCheckFailed(RuntimeError):
    """Raised when the GPU residency check fails or cannot be completed."""


def _load_model(client: OllamaClient, *, timeout_s: int) -> None:
    """Send a minimal request so Ollama loads the model into memory."""
    url = f"{client.base_url}/api/chat"
    payload = {
        "model": client.model_name,
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": 1,
        },
    }
    if client.keep_alive:
        payload["keep_alive"] = client.keep_alive

    original_timeout = client.timeout
    client.timeout = timeout_s
    try:
        client._post_json_with_retries(
            url=url, headers=client._get_headers(), payload=payload, retries=2
        )
    finally:
        client.timeout = original_timeout


def get_vram_residency(
    client: OllamaClient, *, timeout_s: int
) -> Optional[Tuple[int, int]]:
    """Return (size, size_vram) in bytes for the running model, or None if unavailable."""
    if not isinstance(client, OllamaClient):
        return None

    url = f"{client.base_url}/api/ps"
    try:
        response = client.session.get(
            url, headers=client._get_headers(), timeout=timeout_s
        )
        response.raise_for_status()
        data = response.json()
    except Exception:
        return None

    for model in data.get("models", []):
        if client.model_name in (model.get("model"), model.get("name")):
            size = model.get("size")
            size_vram = model.get("size_vram")
            if not isinstance(size, int) or not isinstance(size_vram, int) or size <= 0:
                return None
            return size, size_vram
    return None


def run_gpu_check(
    client,
    *,
    min_vram_fraction: float,
    timeout_s: int,
) -> float:
    """Load the model, check VRAM residency, and raise if below threshold.

    Returns the resident fraction (0.0-1.0) on success. When
    ``min_vram_fraction > 0``, any probe failure aborts via ``GpuCheckFailed``.
    """
    if min_vram_fraction <= 0:
        return 0.0

    if not isinstance(client, OllamaClient):
        raise GpuCheckFailed(
            "GPU check failed: provider is not Ollama, so /api/ps VRAM residency "
            "cannot be measured. Disable gpu_check or use an Ollama target."
        )

    try:
        _load_model(client, timeout_s=timeout_s)
    except Exception as e:
        raise GpuCheckFailed(
            f"GPU check failed: could not load the model within {timeout_s}s "
            f"({e}). Aborting before the benchmark run."
        ) from e

    residency = get_vram_residency(client, timeout_s=timeout_s)
    if residency is None:
        raise GpuCheckFailed(
            "GPU check failed: Ollama /api/ps did not report VRAM residency for "
            f"{client.model_name!r}. Aborting before the benchmark run."
        )
    size, size_vram = residency

    fraction = size_vram / size
    if fraction < min_vram_fraction:
        raise GpuCheckFailed(
            f"GPU check failed: only {fraction:.0%} of the model is resident in "
            f"GPU VRAM ({size_vram / 1e9:.1f} GB of {size / 1e9:.1f} GB total), "
            f"below the required {min_vram_fraction:.0%}. Ollama's own /api/ps "
            "accounting shows this is running mostly or entirely on CPU. "
            "Aborting before running the full (paid) benchmark. "
            "Fix the deployment, or override with --min-vram-fraction 0 "
            "(or gpu_check.enabled: false in config) to disable this check."
        )
    return fraction
