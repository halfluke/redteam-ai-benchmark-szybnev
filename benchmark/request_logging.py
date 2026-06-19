"""Shared JSONL request logging for baseline and optimization queries."""

from typing import Any, Dict, Optional

from models.diagnostics import format_request_details, format_request_summary
from utils.request_log import append_request_log

from .types import RuntimeOptions


def collect_request_diagnostics(client) -> Dict[str, Any]:
    """Return the latest provider diagnostics attached to the client."""
    diagnostics = getattr(client, "last_query_diagnostics", None)
    if diagnostics is None:
        return {}
    if hasattr(diagnostics, "to_dict"):
        return diagnostics.to_dict()
    return dict(diagnostics)


def log_request_event(
    runtime: RuntimeOptions,
    question: Dict[str, Any],
    *,
    score: Optional[int] = None,
    scores: Optional[Dict[str, int]] = None,
    error: Optional[str] = None,
    request: Optional[Dict[str, Any]] = None,
    phase: str = "baseline",
    optimization_iteration: Optional[int] = None,
    optimization_strategy: Optional[str] = None,
    variant_source: Optional[str] = None,
    optimization_prompt: Optional[str] = None,
    optimizer_variants: Optional[Dict[str, str]] = None,
    optimizer_raw_response: Optional[str] = None,
    print_summary: bool = True,
) -> None:
    """Print timing details and optionally append to the JSONL request log."""
    if print_summary and request:
        summary = format_request_summary(request)
        if summary:
            prefix = "   " if phase == "baseline" else "      "
            print(f"{prefix}{summary}")
        if request.get("status") != "success":
            details = format_request_details(request)
            if details:
                print(f"      {details}")

    payload: Dict[str, Any] = {
        "question_id": question.get("id"),
        "category": question.get("category"),
        "phase": phase,
        "score": score,
        "error": error,
        "request": request or {},
    }
    if scores:
        payload["scores"] = scores
    if optimization_iteration is not None:
        payload["optimization_iteration"] = optimization_iteration
    if optimization_strategy:
        payload["optimization_strategy"] = optimization_strategy
    if variant_source:
        payload["variant_source"] = variant_source
    if optimization_prompt is not None:
        payload["optimization_prompt"] = optimization_prompt
        payload["optimization_prompt_chars"] = len(optimization_prompt)
    if optimizer_variants:
        payload["optimizer_variants"] = optimizer_variants
        payload["optimizer_variant_chars"] = {
            name: len(text) for name, text in optimizer_variants.items()
        }
    if optimizer_raw_response is not None:
        payload["optimizer_raw_response"] = optimizer_raw_response
    append_request_log(runtime.request_log, payload)


_VARIANT_DISPLAY_ORDER = (
    ("role_playing", ("role_playing", "role")),
    ("technical_decomposition", ("technical_decomposition", "technical")),
    ("few_shot", ("few_shot", "few")),
)


def _ordered_optimizer_variants(variants: Dict[str, str]) -> Dict[str, str]:
    """Return canonical variant keys in stable display order."""
    ordered: Dict[str, str] = {}
    for label, aliases in _VARIANT_DISPLAY_ORDER:
        for alias in aliases:
            if alias in variants:
                ordered[label] = variants[alias]
                break
    for name, text in variants.items():
        if name not in ordered and name != "first":
            ordered[name] = text
    return ordered


def _format_variant_preview(text: str, *, max_chars: int = 160) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3] + "..."


def print_optimizer_variants(
    variants: Dict[str, str],
    *,
    variant_source: str,
    console_preview: int = 160,
    prefix: str = "   ",
) -> None:
    """Print optimizer prompt variants to the console."""
    ordered = _ordered_optimizer_variants(variants)
    if not ordered:
        return

    source_labels = {
        "optimizer_llm": "optimizer LLM",
        "optimizer_llm_repaired": "optimizer LLM + rule-based fill-ins",
        "rule_fallback": "rule-based fallback (optimizer LLM failed)",
        "parse_fallback": "parse fallback (optimizer output malformed)",
    }
    source_label = source_labels.get(variant_source, variant_source)

    print(f"{prefix}Optimizer variants ({source_label}):", flush=True)
    for name, text in ordered.items():
        if name == "technical" and "technical_decomposition" in ordered:
            continue
        print(
            f"{prefix}   {name} ({len(text)} chars): "
            f"{_format_variant_preview(text, max_chars=console_preview)}",
            flush=True,
        )


def log_optimizer_variants(
    runtime: RuntimeOptions,
    question: Dict[str, Any],
    *,
    variants: Dict[str, str],
    variant_source: str,
    optimizer_request: Optional[Dict[str, Any]] = None,
    optimizer_raw_response: Optional[str] = None,
) -> None:
    """JSONL-log all optimizer-generated prompt variants for one question."""
    ordered = _ordered_optimizer_variants(variants)
    if not ordered:
        return

    if not runtime.request_log:
        return

    log_request_event(
        runtime,
        question,
        phase="optimization_variants",
        variant_source=variant_source,
        optimizer_variants=ordered,
        optimizer_raw_response=optimizer_raw_response,
        request=optimizer_request or {},
        print_summary=False,
    )
