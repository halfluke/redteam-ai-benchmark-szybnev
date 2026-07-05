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

Rates are Tier 1 region on-demand list prices as of mid-2026. Verify against
https://cloud.google.com/run/pricing for your region and current pricing.
"""

from dataclasses import dataclass
from typing import Optional

CPU_PER_VCPU_SECOND = 0.000018
MEMORY_PER_GIB_SECOND = 0.000002

# Keyed by (gpu_type, zonal_redundancy).
GPU_PER_SECOND = {
    ("nvidia-l4", False): 0.0001867,
    ("nvidia-l4", True): 0.0002909,
    ("nvidia-rtx-pro-6000", False): 0.00036522,
    ("nvidia-rtx-pro-6000", True): 0.00056913,
}


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


def format_cost_estimate(estimate: CloudRunCostEstimate) -> str:
    """Format an estimate for console output."""
    hours = estimate.elapsed_seconds / 3600
    return (
        f"~${estimate.total_cost:.2f} estimated Cloud Run cost "
        f"(instance uptime: {hours:.2f}h @ ~${estimate.total_cost / hours if hours else 0:.2f}/h; "
        f"not the official GCP invoice, see docs/agent-reference.md)"
    )
