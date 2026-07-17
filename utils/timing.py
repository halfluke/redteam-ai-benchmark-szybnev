"""Shared duration formatting for console and export metadata."""

from typing import Any, Dict, Optional


def format_duration(latency_ms: Optional[float]) -> str:
    """Format milliseconds as a short human-readable duration."""
    if latency_ms is None:
        return "?"
    seconds = latency_ms / 1000
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, seconds = divmod(seconds, 60)
    return f"{int(minutes)}m{seconds:04.1f}s"


def format_semantic_timing_line(semantic_scores: Optional[Dict[str, Any]]) -> Optional[str]:
    """Describe reference/response embed time for one semantic score."""
    if not semantic_scores:
        return None
    total = semantic_scores.get("semantic_elapsed_ms")
    if not isinstance(total, (int, float)):
        return None

    if semantic_scores.get("skipped") or semantic_scores.get("skip_reason"):
        return f"Semantic check ({format_duration(total)})"

    ref = semantic_scores.get("reference_embed_ms")
    resp = semantic_scores.get("response_embed_ms")
    ref_part = (
        f"reference embedding {format_duration(ref)}"
        if isinstance(ref, (int, float)) and ref > 0
        else "reference embedding cached"
    )
    resp_part = (
        f"response embedding {format_duration(resp)}"
        if isinstance(resp, (int, float)) and resp > 0
        else "response embedding 0.0s"
    )
    return (
        f"Semantic scoring ({format_duration(total)}: "
        f"{ref_part}, {resp_part})"
    )
