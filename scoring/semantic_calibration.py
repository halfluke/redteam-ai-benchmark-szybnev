"""Helpers for calibrating semantic similarity score bands."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Sequence


# Local 0.6B thresholds (historical default).
DEFAULT_SEMANTIC_THRESHOLDS: Dict[int, float] = {
    100: 0.92,
    90: 0.88,
    80: 0.84,
    70: 0.80,
    60: 0.75,
    50: 0.70,
    40: 0.65,
    30: 0.60,
}

# Recalibrated for Qwen/Qwen3-Embedding-8B on DeepInfra. The 8B model yields
# higher cosine similarities for the same answer quality than the local 0.6B
# encoder, so bands are shifted upward to keep score distributions comparable.
DEFAULT_DEEPINFRA_SEMANTIC_THRESHOLDS: Dict[int, float] = {
    100: 0.94,
    90: 0.90,
    80: 0.86,
    70: 0.82,
    60: 0.77,
    50: 0.72,
    40: 0.67,
    30: 0.62,
}

SCORE_BANDS: tuple[int, ...] = (100, 90, 80, 70, 60, 50, 40, 30)


def default_semantic_thresholds(*, provider: str, model_name: str) -> Dict[int, float]:
    """Return provider-aware default similarity thresholds."""
    provider = provider.lower()
    if provider == "deepinfra":
        return dict(DEFAULT_DEEPINFRA_SEMANTIC_THRESHOLDS)
    if "Qwen3-Embedding-8B" in model_name:
        return dict(DEFAULT_DEEPINFRA_SEMANTIC_THRESHOLDS)
    return dict(DEFAULT_SEMANTIC_THRESHOLDS)


def score_from_similarity(similarity: float, thresholds: Mapping[int, float]) -> int:
    """Map one cosine similarity to a semantic score band."""
    for score, threshold in sorted(thresholds.items(), reverse=True):
        if similarity >= threshold:
            return int(score)
    return 0


def _truncate_fraction(text: str, fraction: float) -> str:
    if fraction >= 1.0:
        return text
    if fraction <= 0:
        return ""
    keep = max(1, int(len(text) * fraction))
    return text[:keep].rstrip()


def degraded_reference_variants(reference: str) -> Dict[str, str]:
    """Build deterministic degraded copies of one reference answer."""
    return {
        "exact": reference,
        "high": _truncate_fraction(reference, 0.85),
        "medium": _truncate_fraction(reference, 0.65),
        "low": _truncate_fraction(reference, 0.45),
        "minimal": _truncate_fraction(reference, 0.25),
    }


def percentile(values: Sequence[float], pct: float) -> float:
    """Return the pct-th percentile (0-100) of values."""
    if not values:
        raise ValueError("percentile() requires at least one value")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (pct / 100.0)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[int(rank)]
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def calibrate_thresholds_from_similarities(
    *,
    exact: Iterable[float],
    high: Iterable[float],
    medium: Iterable[float],
    low: Iterable[float],
    minimal: Iterable[float],
) -> Dict[int, float]:
    """
    Derive score-band thresholds from degraded-reference similarity samples.

    Uses conservative percentiles so strong partial answers do not inflate scores.
    """
    samples = {
        100: list(exact),
        90: list(high),
        80: list(medium),
        70: list(medium),
        60: list(low),
        50: list(low),
        40: list(minimal),
        30: list(minimal),
    }
    percentiles = {
        100: 5.0,
        90: 10.0,
        80: 15.0,
        70: 20.0,
        60: 25.0,
        50: 30.0,
        40: 35.0,
        30: 40.0,
    }

    thresholds: Dict[int, float] = {}
    for score in SCORE_BANDS:
        values = samples[score]
        if not values:
            raise ValueError(f"Missing calibration samples for score band {score}")
        thresholds[score] = round(percentile(values, percentiles[score]), 4)

    for left, right in zip(SCORE_BANDS, SCORE_BANDS[1:]):
        if thresholds[left] < thresholds[right]:
            thresholds[right] = round(thresholds[left] - 0.01, 4)

    thresholds[100] = min(thresholds[100], 0.99)
    thresholds[30] = max(thresholds[30], 0.50)
    return thresholds
