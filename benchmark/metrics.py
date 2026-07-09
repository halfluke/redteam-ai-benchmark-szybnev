"""Aggregate benchmark result metrics."""

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Literal


def _question_weight(result: Dict[str, Any]) -> float:
    weight = result.get("weight", 1.0)
    return float(weight) if isinstance(weight, (int, float)) and weight > 0 else 1.0


def weighted_score(results: Iterable[Dict[str, Any]]) -> float:
    """Return weighted average score across question results."""
    total_weight = 0.0
    weighted_total = 0.0
    for result in results:
        weight = _question_weight(result)
        total_weight += weight
        weighted_total += float(result.get("score", 0)) * weight
    return weighted_total / total_weight if total_weight else 0.0


def weighted_semantic_score(results: Iterable[Dict[str, Any]]) -> float | None:
    """Return weighted average semantic score when semantic scoring is present."""
    total_weight = 0.0
    weighted_total = 0.0
    seen = False
    for result in results:
        semantic_score = result.get("semantic_score")
        if not isinstance(semantic_score, (int, float)):
            continue
        seen = True
        weight = _question_weight(result)
        total_weight += weight
        weighted_total += float(semantic_score) * weight
    if not seen or not total_weight:
        return None
    return weighted_total / total_weight


def _primary_score(result: Dict[str, Any], primary: str) -> float:
    value = result.get(primary)
    return float(value) if isinstance(value, (int, float)) else 0.0


def weighted_primary_score(results: Iterable[Dict[str, Any]], primary: str) -> float:
    """Return weighted average for an arbitrary numeric score field."""
    total_weight = 0.0
    weighted_total = 0.0
    for result in results:
        weight = _question_weight(result)
        total_weight += weight
        weighted_total += _primary_score(result, primary) * weight
    return weighted_total / total_weight if total_weight else 0.0


def _breakdown(
    results: List[Dict[str, Any]],
    field_name: str,
    *,
    primary: str = "score",
) -> Dict[str, Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for result in results:
        value = result.get(field_name) or "unclassified"
        groups[str(value)].append(result)

    return {
        name: {
            "score": round(weighted_primary_score(items, primary), 2),
            "questions": len(items),
        }
        for name, items in sorted(groups.items())
    }


def _average_metric(results: List[Dict[str, Any]], metric: str) -> float | None:
    values = []
    for result in results:
        metrics = result.get("metrics") or {}
        value = metrics.get(metric)
        if isinstance(value, (int, float)):
            values.append(float(value))
    if not values:
        return None
    return round(sum(values) / len(values) * 100, 2)


def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build enterprise-facing aggregate metrics and breakdowns."""
    total = len(results)
    if not total:
        return {
            "weighted_score": 0.0,
            "metrics": {},
            "breakdown": {},
        }

    refusal_count = sum(1 for result in results if result.get("censored"))
    critical_count = sum(1 for result in results if result.get("critical_error"))
    latencies = [
        float(result["latency_ms"])
        for result in results
        if isinstance(result.get("latency_ms"), (int, float))
    ]

    metrics = {
        "refusal_rate": round(refusal_count / total * 100, 2),
        "technical_accuracy": _average_metric(results, "technical_accuracy"),
        "critical_error_rate": round(critical_count / total * 100, 2),
        "completeness": _average_metric(results, "completeness"),
        "specificity": _average_metric(results, "specificity"),
        "hallucination_rate": None,
        "latency_ms_avg": round(sum(latencies) / len(latencies), 2) if latencies else None,
        "tokens_cost": None,
        "stability": None,
    }

    return {
        "weighted_score": round(weighted_score(results), 2),
        "metrics": metrics,
        "breakdown": {
            "difficulty": _breakdown(results, "difficulty"),
            "domain": _breakdown(results, "domain"),
            "capability": _breakdown(results, "capability"),
        },
    }


def build_track_results(
    results: Iterable[Dict[str, Any]],
    *,
    track: Literal["rubric", "semantic"],
) -> List[Dict[str, Any]]:
    """Project results onto one track's selected answer.

    The top-level result stays backward-compatible and represents rubric-best.
    This helper exposes a flat shape for aggregate scoring and final tables.
    """
    projected = []
    key = "rubric_best" if track == "rubric" else "semantic_best"
    for result in results:
        best = result.get(key)
        if not isinstance(best, dict):
            if track == "semantic" and not isinstance(
                result.get("semantic_score"), (int, float)
            ):
                continue
            projected.append(dict(result))
            continue

        item = dict(result)
        item.update(
            {
                "score": best.get("score", result.get("score", 0)),
                "response_snippet": best.get(
                    "response_snippet", result.get("response_snippet", "")
                ),
                "full_response": best.get(
                    "full_response", result.get("full_response", "")
                ),
                "semantic_score": best.get(
                    "semantic_score", result.get("semantic_score")
                ),
                "semantic_similarity": best.get(
                    "semantic_similarity", result.get("semantic_similarity")
                ),
                "semantic_scores": best.get(
                    "semantic_scores", result.get("semantic_scores")
                ),
                "answer_source": best.get("answer_source"),
                "optimization_strategy": best.get("strategy"),
            }
        )
        projected.append(item)
    return projected


def summarize_track(
    results: List[Dict[str, Any]],
    *,
    primary: Literal["score", "semantic_score"],
) -> Dict[str, Any]:
    """Build a compact weighted summary for one reporting track."""
    total = weighted_primary_score(results, primary) if results else 0.0
    return {
        "weighted_score": round(total, 2),
        "questions": len(results),
        "breakdown": {
            "difficulty": _breakdown(results, "difficulty", primary=primary),
            "domain": _breakdown(results, "domain", primary=primary),
            "capability": _breakdown(results, "capability", primary=primary),
        },
    }


def summarize_semantic_results(results: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    """Build aggregate metrics for optional semantic scoring."""
    semantic_results = [
        result
        for result in results
        if isinstance(result.get("semantic_score"), (int, float))
    ]
    if not semantic_results:
        return None

    similarities = [
        float(result["semantic_similarity"])
        for result in semantic_results
        if isinstance(result.get("semantic_similarity"), (int, float))
    ]
    weighted = weighted_semantic_score(semantic_results)
    return {
        "enabled": True,
        "weighted_score": round(weighted, 2) if weighted is not None else None,
        "similarity_avg": (
            round(sum(similarities) / len(similarities), 6) if similarities else None
        ),
        "questions": len(semantic_results),
        "scored_questions": len(semantic_results),
    }
