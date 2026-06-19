"""Compute aggregate scores for benchmark results."""

from typing import Any, Dict, List, Optional

from utils.export import get_interpretation


def compute_total_scores(
    results: List[Dict[str, Any]],
    methods: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Return average score per method for completed questions."""
    if not results:
        return {}

    if methods:
        totals = {method: 0.0 for method in methods}
        for row in results:
            row_scores = row.get("scores") or {}
            for method in methods:
                totals[method] += float(row_scores.get(method, row.get("score", 0)))
        return {method: total / len(results) for method, total in totals.items()}

    return {"primary": sum(r.get("score", 0) for r in results) / len(results)}


def primary_total_score(total_scores: Dict[str, float]) -> float:
    """Pick the headline score for legacy report/export fields."""
    if "keyword" in total_scores:
        return total_scores["keyword"]
    if "primary" in total_scores:
        return total_scores["primary"]
    return next(iter(total_scores.values())) if total_scores else 0.0


def primary_interpretation(total_scores: Dict[str, float]) -> str:
    """Interpret the primary benchmark score."""
    return get_interpretation(primary_total_score(total_scores))
