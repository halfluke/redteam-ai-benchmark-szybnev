"""Detect degenerate model output unsuitable for semantic scoring."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple


GARBAGE_SKIP_LABEL = "semantic skipped (garbage)"
GARBAGE_SKIP_REASON = "garbage_response"

_GARBAGE_MIN_WORDS = 24
_GARBAGE_MAX_UNIQUE_RATIO = 0.12


def garbage_diagnostics(response: str) -> Dict[str, Any]:
    """Return word-diversity stats explaining garbage classification."""
    text = response.strip()
    words = re.findall(r"\b[\w-]+\b", text)
    word_count = len(words)
    unique_word_count = len(Counter(words)) if words else 0
    unique_ratio = unique_word_count / word_count if word_count else None
    return {
        "garbage_word_count": word_count,
        "garbage_unique_word_count": unique_word_count,
        "garbage_unique_ratio": (
            round(unique_ratio, 4) if unique_ratio is not None else None
        ),
    }


def is_garbage_response(response: str) -> bool:
    """Return True when a response is repetitive junk unsuitable for semantic scoring."""
    text = response.strip()
    if not text:
        return True

    words = re.findall(r"\b[\w-]+\b", text)
    if len(words) >= _GARBAGE_MIN_WORDS:
        unique_ratio = len(Counter(words)) / len(words)
        if unique_ratio < _GARBAGE_MAX_UNIQUE_RATIO:
            return True

    return False


def semantic_garbage_skip_payload(
    *,
    answer_source: str = "baseline",
    response: str = "",
) -> Dict[str, Any]:
    """Return export-friendly metadata when semantic scoring is skipped."""
    return {
        "method": "semantic",
        "score": None,
        "similarity": None,
        "skipped": True,
        "skip_reason": GARBAGE_SKIP_REASON,
        "answer_source": answer_source,
        **garbage_diagnostics(response),
    }


def is_semantic_garbage_skip(semantic_scores: Optional[Dict[str, Any]]) -> bool:
    return bool(
        semantic_scores and semantic_scores.get("skip_reason") == GARBAGE_SKIP_REASON
    )


def attempt_semantic_garbage_skipped(attempt: Dict[str, Any]) -> bool:
    return (
        attempt.get("semantic_skipped") == GARBAGE_SKIP_REASON
        or is_semantic_garbage_skip(attempt.get("semantic_scores"))
    )


def format_semantic_attempt_suffix(
    *,
    semantic_scores: Optional[Dict[str, Any]] = None,
    attempt: Optional[Dict[str, Any]] = None,
    prefix: str = "  |  ",
) -> str:
    """Format inline semantic score or garbage-skip label for console output."""
    if attempt is not None and attempt_semantic_garbage_skipped(attempt):
        return f"{prefix}{GARBAGE_SKIP_LABEL}"
    if is_semantic_garbage_skip(semantic_scores):
        return f"{prefix}{GARBAGE_SKIP_LABEL}"

    score = None
    if attempt is not None:
        score = attempt.get("semantic_score")
    if score is None and semantic_scores is not None:
        score = semantic_scores.get("score")
    if isinstance(score, (int, float)):
        return f"{prefix}semantic {int(score)}%"
    return ""


def format_garbage_diagnostics_line(
    semantic_scores: Optional[Dict[str, Any]] = None,
    *,
    prefix: str = "      ",
) -> Optional[str]:
    """Format inline garbage word-diversity stats for console output."""
    if not is_semantic_garbage_skip(semantic_scores):
        return None

    word_count = semantic_scores.get("garbage_word_count")
    unique_ratio = semantic_scores.get("garbage_unique_ratio")

    if word_count == 0 and unique_ratio is None:
        return f"{prefix}Garbage: empty response"

    if isinstance(unique_ratio, (int, float)):
        return (
            f"{prefix}Garbage: {word_count} words, unique ratio {unique_ratio:.3f} "
            f"(threshold: >={_GARBAGE_MIN_WORDS} words, ratio >= {_GARBAGE_MAX_UNIQUE_RATIO})"
        )

    if isinstance(word_count, int):
        return (
            f"{prefix}Garbage: {word_count} words "
            f"(below {_GARBAGE_MIN_WORDS}-word repetition threshold)"
        )

    return None


def resolve_question_semantic_display(
    semantic_enabled: bool,
    *,
    baseline_semantic: Optional[Dict[str, Any]] = None,
    opt_history: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Optional[int], bool]:
    """Return (best_semantic_score, all_responses_garbage)."""
    if not semantic_enabled:
        return None, False

    best: Optional[int] = None
    saw_garbage = False
    saw_semantic_path = False

    def ingest(
        *,
        semantic_scores: Optional[Dict[str, Any]] = None,
        attempt: Optional[Dict[str, Any]] = None,
    ) -> None:
        nonlocal best, saw_garbage, saw_semantic_path
        saw_semantic_path = True
        if attempt is not None and attempt_semantic_garbage_skipped(attempt):
            saw_garbage = True
            return
        if is_semantic_garbage_skip(semantic_scores):
            saw_garbage = True
            return

        score = attempt.get("semantic_score") if attempt is not None else None
        if score is None and semantic_scores is not None:
            score = semantic_scores.get("score")
        if isinstance(score, (int, float)):
            value = int(score)
            best = value if best is None else max(best, value)

    ingest(semantic_scores=baseline_semantic)
    if opt_history:
        for attempt in opt_history:
            ingest(attempt=attempt, semantic_scores=attempt.get("semantic_scores"))

    if best is not None:
        return best, False
    if saw_semantic_path and saw_garbage:
        return None, True
    return None, False


def format_final_semantic_scores(
    rubric_score: int,
    semantic_enabled: bool,
    *,
    baseline_semantic: Optional[Dict[str, Any]] = None,
    opt_history: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Format the Final line score summary for rubric + semantic tracks."""
    best, all_garbage = resolve_question_semantic_display(
        semantic_enabled,
        baseline_semantic=baseline_semantic,
        opt_history=opt_history,
    )
    if best is not None:
        return f"rubric {rubric_score}%  |  semantic {best}%"
    if all_garbage:
        return f"rubric {rubric_score}%  |  {GARBAGE_SKIP_LABEL}"
    return f"rubric {rubric_score}%"
