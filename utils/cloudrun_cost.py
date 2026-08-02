"""Estimated Cloud Run GPU-instance cost tracking.

This is NOT the authoritative GCP invoice: Cloud Billing data (BigQuery
export, Cost Management UI) typically lags actual usage by ~24 hours and has
no real-time API. Instead, this mirrors Cloud Run's *instance-based billing*
model, which GPU-attached services must use: the full CPU + memory + GPU
allocation is billed continuously for every second an instance is up,
regardless of whether it is actively serving a request or idling between
them. Multiplying observed wall-clock instance uptime by the published
per-second rate therefore tracks the real bill closely, modulo committed-use
discounts, free-tier credits, or since-changed list prices.

Rates below are Tier 1 region on-demand instance-based list prices
(europe-west1 / Belgium is Tier 1). Verify against PRICING_URL for your
region and current pricing before relying on this for budget decisions.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from utils.request_log import append_request_log

PRICING_URL = "https://cloud.google.com/run/pricing"
RATES_AS_OF = "2026-08"
REGION_NOTE = (
    "Tier 1 region on-demand instance-based billing "
    "(e.g. europe-west1); GPU services must use instance-based billing"
)

CPU_PER_VCPU_SECOND = 0.000018
MEMORY_PER_GIB_SECOND = 0.000002

# Keyed by (gpu_type, zonal_redundancy).
GPU_PER_SECOND = {
    ("nvidia-l4", False): 0.0001867,
    ("nvidia-l4", True): 0.0002909,
    ("nvidia-rtx-pro-6000", False): 0.00036522,
    ("nvidia-rtx-pro-6000", True): 0.00056913,
}

SESSION_START_ENV = "CLOUDRUN_COST_SESSION_START"
WARMUP_SECONDS_ENV = "CLOUDRUN_COST_WARMUP_SECONDS"
WARMUP_ENV_FILE = Path(".cache/redteam/cloudrun_cost_warmup.env")

DEFAULT_CURRENCY = "GBP"
DEFAULT_USD_PER_UNIT = 1.33


class CloudRunCostLimitExceeded(RuntimeError):
    """Raised when estimated display-currency cost reaches the configured cap."""


@dataclass
class CloudRunCostEstimate:
    """Estimated USD cost breakdown for a Cloud Run instance-based-billing service."""

    elapsed_seconds: float
    cpu_cost: float
    memory_cost: float
    gpu_cost: float

    @property
    def total_cost(self) -> float:
        return self.cpu_cost + self.memory_cost + self.gpu_cost


def gpu_per_second_rate(gpu_type: str, gpu_zonal_redundancy: bool) -> float:
    """Return the USD/second rate for a supported Cloud Run GPU type."""
    key = (gpu_type.strip().lower(), gpu_zonal_redundancy)
    if key not in GPU_PER_SECOND:
        supported = ", ".join(sorted({name for name, _ in GPU_PER_SECOND}))
        raise ValueError(f"Unknown Cloud Run GPU type {gpu_type!r}. Supported: {supported}")
    return GPU_PER_SECOND[key]


def estimate_cost(
    elapsed_seconds: float,
    *,
    cpu: float,
    memory_gib: float,
    gpu_type: Optional[str] = None,
    gpu_zonal_redundancy: bool = False,
) -> CloudRunCostEstimate:
    """Estimate instance-based-billing cost for `elapsed_seconds` of uptime."""
    cpu_cost = elapsed_seconds * cpu * CPU_PER_VCPU_SECOND
    memory_cost = elapsed_seconds * memory_gib * MEMORY_PER_GIB_SECOND
    gpu_cost = 0.0
    if gpu_type:
        gpu_cost = elapsed_seconds * gpu_per_second_rate(gpu_type, gpu_zonal_redundancy)
    return CloudRunCostEstimate(
        elapsed_seconds=elapsed_seconds,
        cpu_cost=cpu_cost,
        memory_cost=memory_cost,
        gpu_cost=gpu_cost,
    )


def to_display(usd: float, usd_per_unit: float) -> float:
    """Convert a USD amount into the configured display currency."""
    if usd_per_unit <= 0:
        raise ValueError("usd_per_unit must be > 0")
    return usd / usd_per_unit


def currency_symbol(currency: str) -> str:
    """Return a compact currency prefix for console output."""
    code = (currency or "").strip().upper()
    return {"GBP": "£", "USD": "$", "EUR": "€"}.get(code, f"{code} ")


def resolve_session_started_at(fallback: float) -> float:
    """Prefer CLOUDRUN_COST_SESSION_START (unix seconds) over a Python fallback."""
    raw = os.environ.get(SESSION_START_ENV)
    if raw is None or raw.strip() == "":
        return fallback
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(
            f"{SESSION_START_ENV} must be a unix timestamp, got {raw!r}"
        ) from exc


def resolve_warmup_seconds(*, env_file: Path = WARMUP_ENV_FILE) -> float:
    """Read warmup seconds from env, falling back to the warmup sidecar file."""
    raw = os.environ.get(WARMUP_SECONDS_ENV)
    if raw is not None and raw.strip() != "":
        try:
            return max(0.0, float(raw))
        except ValueError as exc:
            raise ValueError(
                f"{WARMUP_SECONDS_ENV} must be a number of seconds, got {raw!r}"
            ) from exc

    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            if key.strip() == WARMUP_SECONDS_ENV:
                try:
                    return max(0.0, float(value.strip().strip("\"'")))
                except ValueError as exc:
                    raise ValueError(
                        f"{env_file}: {WARMUP_SECONDS_ENV} must be numeric, got {value!r}"
                    ) from exc
    return 0.0


def format_cost_estimate(
    estimate: CloudRunCostEstimate,
    *,
    currency: str = DEFAULT_CURRENCY,
    usd_per_unit: float = DEFAULT_USD_PER_UNIT,
    warmup_seconds: float = 0.0,
) -> str:
    """Format an estimate for console output (USD + display currency)."""
    hours = estimate.elapsed_seconds / 3600
    hourly = estimate.total_cost / hours if hours else 0.0
    display_total = to_display(estimate.total_cost, usd_per_unit)
    symbol = currency_symbol(currency)
    warmup_note = ""
    if warmup_seconds > 0:
        warmup_note = f"; warmup {warmup_seconds:.0f}s included"
    return (
        f"~${estimate.total_cost:.2f} / {symbol}{display_total:.2f} estimated Cloud Run cost "
        f"(instance uptime: {hours:.2f}h @ ~${hourly:.2f}/h{warmup_note}; "
        f"{REGION_NOTE}; rates as of {RATES_AS_OF}, {PRICING_URL}; "
        f"not the official GCP invoice, see docs/agent-reference.md)"
    )


def estimate_to_dict(
    estimate: CloudRunCostEstimate,
    *,
    cpu: float,
    memory_gib: float,
    gpu_type: Optional[str],
    gpu_zonal_redundancy: bool,
    currency: str,
    usd_per_unit: float,
    warmup_seconds: float = 0.0,
    benchmark_seconds: Optional[float] = None,
    projected_elapsed_seconds: Optional[float] = None,
    projected_usd: Optional[float] = None,
    projected_display: Optional[float] = None,
    completed_questions: Optional[int] = None,
    total_questions: Optional[int] = None,
    cost_limit_exceeded: bool = False,
    max_cost: Optional[float] = None,
) -> Dict[str, Any]:
    """Serialize an estimate for result JSON / request-log summary."""
    total_seconds = estimate.elapsed_seconds
    if benchmark_seconds is None:
        benchmark_seconds = max(0.0, total_seconds - warmup_seconds)
    display_total = to_display(estimate.total_cost, usd_per_unit)
    payload: Dict[str, Any] = {
        "elapsed_seconds": round(total_seconds, 3),
        "warmup_seconds": round(warmup_seconds, 3),
        "benchmark_seconds": round(benchmark_seconds, 3),
        "cpu": cpu,
        "memory_gib": memory_gib,
        "gpu_type": gpu_type,
        "gpu_zonal_redundancy": gpu_zonal_redundancy,
        "usd": {
            "cpu": round(estimate.cpu_cost, 6),
            "memory": round(estimate.memory_cost, 6),
            "gpu": round(estimate.gpu_cost, 6),
            "total": round(estimate.total_cost, 6),
        },
        "currency": currency.upper(),
        "usd_per_unit": usd_per_unit,
        "display_total": round(display_total, 6),
        "rates_as_of": RATES_AS_OF,
        "pricing_url": PRICING_URL,
        "region_note": REGION_NOTE,
        "cost_limit_exceeded": cost_limit_exceeded,
        "max_cost": max_cost,
    }
    if completed_questions is not None:
        payload["completed_questions"] = completed_questions
    if total_questions is not None:
        payload["total_questions"] = total_questions
    if projected_elapsed_seconds is not None:
        payload["projected_elapsed_seconds"] = round(projected_elapsed_seconds, 3)
    if projected_usd is not None:
        payload["projected_usd"] = round(projected_usd, 6)
    if projected_display is not None:
        payload["projected_display"] = round(projected_display, 6)
    return payload


class CloudRunCostTracker:
    """Track estimated Cloud Run cost across a benchmark session."""

    def __init__(
        self,
        *,
        cpu: float,
        memory_gib: float,
        gpu_type: Optional[str] = None,
        gpu_zonal_redundancy: bool = False,
        currency: str = DEFAULT_CURRENCY,
        usd_per_unit: float = DEFAULT_USD_PER_UNIT,
        progress_every: int = 5,
        max_cost: Optional[float] = None,
        session_started_at: float,
        warmup_seconds: float = 0.0,
        total_questions: int = 0,
        request_log: Optional[str] = None,
    ) -> None:
        if usd_per_unit <= 0:
            raise ValueError("usd_per_unit must be > 0")
        if progress_every < 0:
            raise ValueError("progress_every must be >= 0")
        if max_cost is not None and max_cost <= 0:
            raise ValueError("max_cost must be > 0 when set")

        self.cpu = cpu
        self.memory_gib = memory_gib
        self.gpu_type = gpu_type
        self.gpu_zonal_redundancy = gpu_zonal_redundancy
        self.currency = (currency or DEFAULT_CURRENCY).upper()
        self.usd_per_unit = float(usd_per_unit)
        self.progress_every = int(progress_every)
        self.max_cost = float(max_cost) if max_cost is not None else None
        self.session_started_at = float(session_started_at)
        self.warmup_seconds = max(0.0, float(warmup_seconds))
        self.total_questions = int(total_questions)
        self.request_log = request_log
        self.cost_limit_exceeded = False

    @classmethod
    def from_config(
        cls,
        cost_config,
        *,
        session_started_at: float,
        total_questions: int = 0,
        request_log: Optional[str] = None,
        max_cost_override: Optional[float] = None,
        progress_every_override: Optional[int] = None,
    ) -> "CloudRunCostTracker":
        """Build a tracker from CloudRunCostConfig plus env session/warmup."""
        max_cost = (
            max_cost_override
            if max_cost_override is not None
            else getattr(cost_config, "max_cost", None)
        )
        progress_every = (
            progress_every_override
            if progress_every_override is not None
            else getattr(cost_config, "progress_every", 5)
        )
        return cls(
            cpu=cost_config.cpu,
            memory_gib=cost_config.memory_gib,
            gpu_type=cost_config.gpu_type,
            gpu_zonal_redundancy=cost_config.gpu_zonal_redundancy,
            currency=getattr(cost_config, "currency", DEFAULT_CURRENCY),
            usd_per_unit=getattr(cost_config, "usd_per_unit", DEFAULT_USD_PER_UNIT),
            progress_every=progress_every,
            max_cost=max_cost,
            session_started_at=resolve_session_started_at(session_started_at),
            warmup_seconds=resolve_warmup_seconds(),
            total_questions=total_questions,
            request_log=request_log,
        )

    def elapsed_seconds(self, *, now: Optional[float] = None) -> float:
        """Wall-clock seconds since session start (includes warmup when set)."""
        current = time.time() if now is None else now
        return max(0.0, current - self.session_started_at)

    def _estimate(self, elapsed_seconds: float) -> CloudRunCostEstimate:
        return estimate_cost(
            elapsed_seconds,
            cpu=self.cpu,
            memory_gib=self.memory_gib,
            gpu_type=self.gpu_type,
            gpu_zonal_redundancy=self.gpu_zonal_redundancy,
        )

    def snapshot(
        self,
        completed: int,
        total: Optional[int] = None,
        *,
        now: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Current estimate plus linear projection to `total` questions."""
        elapsed = self.elapsed_seconds(now=now)
        estimate = self._estimate(elapsed)
        total_q = self.total_questions if total is None else total
        projected_elapsed = None
        projected_usd = None
        projected_display = None
        if completed > 0 and total_q > 0:
            projected_elapsed = elapsed / completed * total_q
            projected_estimate = self._estimate(projected_elapsed)
            projected_usd = projected_estimate.total_cost
            projected_display = to_display(projected_usd, self.usd_per_unit)
        return estimate_to_dict(
            estimate,
            cpu=self.cpu,
            memory_gib=self.memory_gib,
            gpu_type=self.gpu_type,
            gpu_zonal_redundancy=self.gpu_zonal_redundancy,
            currency=self.currency,
            usd_per_unit=self.usd_per_unit,
            warmup_seconds=self.warmup_seconds,
            projected_elapsed_seconds=projected_elapsed,
            projected_usd=projected_usd,
            projected_display=projected_display,
            completed_questions=completed,
            total_questions=total_q,
            cost_limit_exceeded=self.cost_limit_exceeded,
            max_cost=self.max_cost,
        )

    def maybe_report(self, completed: int, *, now: Optional[float] = None) -> None:
        """Print spent + projected cost every N completed questions."""
        if self.progress_every <= 0 or completed <= 0:
            return
        if completed % self.progress_every != 0:
            return
        snap = self.snapshot(completed, now=now)
        symbol = currency_symbol(self.currency)
        spent_usd = snap["usd"]["total"]
        spent_display = snap["display_total"]
        line = (
            f"💰 Cloud Run cost after {completed}/{snap.get('total_questions') or '?'} q: "
            f"~${spent_usd:.2f} / {symbol}{spent_display:.2f}"
        )
        if snap.get("projected_usd") is not None and snap.get("projected_display") is not None:
            line += (
                f" | projected full run ~${snap['projected_usd']:.2f} / "
                f"{symbol}{snap['projected_display']:.2f}"
            )
        if self.max_cost is not None:
            line += f" | cap {symbol}{self.max_cost:.2f}"
        print(line)

    def check_limit(self, *, now: Optional[float] = None) -> None:
        """Abort when display-currency spend reaches max_cost."""
        if self.max_cost is None:
            return
        estimate = self._estimate(self.elapsed_seconds(now=now))
        display_total = to_display(estimate.total_cost, self.usd_per_unit)
        if display_total >= self.max_cost:
            self.cost_limit_exceeded = True
            symbol = currency_symbol(self.currency)
            raise CloudRunCostLimitExceeded(
                f"Estimated Cloud Run cost {symbol}{display_total:.2f} reached "
                f"cap {symbol}{self.max_cost:.2f} "
                f"(~${estimate.total_cost:.2f} USD); aborting run"
            )

    def final_payload(
        self,
        *,
        completed: Optional[int] = None,
        now: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Serializable summary for JSON export and request-log."""
        completed_n = 0 if completed is None else completed
        return self.snapshot(completed_n, now=now)

    def format_final(self, *, completed: Optional[int] = None, now: Optional[float] = None) -> str:
        """Console line for end-of-run cost."""
        elapsed = self.elapsed_seconds(now=now)
        estimate = self._estimate(elapsed)
        return format_cost_estimate(
            estimate,
            currency=self.currency,
            usd_per_unit=self.usd_per_unit,
            warmup_seconds=self.warmup_seconds,
        )

    def write_request_log(
        self,
        *,
        completed: Optional[int] = None,
        now: Optional[float] = None,
    ) -> None:
        """Append a cost_summary row when request logging is configured."""
        if not self.request_log:
            return
        payload = self.final_payload(completed=completed, now=now)
        append_request_log(self.request_log, {"phase": "cost_summary", **payload})
