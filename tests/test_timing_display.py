"""Tests for shared timing display helpers."""

from utils.timing import format_duration, format_semantic_timing_line


def test_format_duration_seconds_and_minutes():
    assert format_duration(19200) == "19.2s"
    assert format_duration(76500) == "1m16.5s"


def test_format_semantic_timing_line_for_scored_response():
    line = format_semantic_timing_line(
        {
            "semantic_elapsed_ms": 4100,
            "reference_embed_ms": 3200,
            "response_embed_ms": 900,
            "score": 50,
        }
    )
    assert line == (
        "Semantic scoring (4.1s: reference embedding 3.2s, response embedding 0.9s)"
    )


def test_format_semantic_timing_line_for_garbage_skip():
    line = format_semantic_timing_line(
        {
            "semantic_elapsed_ms": 12,
            "skipped": True,
            "skip_reason": "garbage",
        }
    )
    assert line == "Semantic check (0.0s)"
