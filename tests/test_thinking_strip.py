"""Regression tests for thinking-block strip regexes.

Tag names are built at runtime so tests stay stable even when source filters
rewrite literal XML tag spellings.
"""

from __future__ import annotations

import pytest

from scoring.semantic_scorer import _THINKING_STRIP_PATTERNS, strip_thinking_blocks


def _tag(name: str, *, close: bool = False) -> str:
    return f"</{name}>" if close else f"<{name}>"


THINK = "think"
REDACTED = "redacted_thinking"

# Expected UTF-8 close-tag hex (see scoring/semantic_scorer.py header comment).
REDACTED_THINKING_CLOSE_HEX = "3c2f72656461637465645f7468696e6b696e673e"
XML_THINK_CLOSE_HEX = "3c2f7468696e6b3e"


def _pattern_close_hex(pattern_name: str) -> str:
    _, pattern = next(item for item in _THINKING_STRIP_PATTERNS if item[0] == pattern_name)
    source = pattern.pattern
    start = source.index("</")
    end = source.index(">", start)
    return source[start : end + 1].encode("utf-8").hex()


@pytest.mark.parametrize(
    ("name", "sample", "expected_text", "expected_patterns"),
    [
        (
            "bugtrace_channel",
            "<|channel>thought internal trace <channel|>Use fodhelper under HKCU.",
            "Use fodhelper under HKCU.",
            ["bugtrace_channel_thought"],
        ),
        (
            "bugtrace_channel_pipe",
            "<|channel|>thought trace here <channel|>Registry hijack answer.",
            "Registry hijack answer.",
            ["bugtrace_channel_thought"],
        ),
        (
            "redacted_thinking",
            _tag(REDACTED) + "internal plan" + _tag(REDACTED, close=True) + "Use certutil.",
            "Use certutil.",
            ["redacted_thinking"],
        ),
        (
            "xml_think",
            _tag(THINK) + "step one" + _tag(THINK, close=True) + "Final AMSI answer.",
            "Final AMSI answer.",
            ["xml_think"],
        ),
        (
            "xml_thinking",
            _tag("thinking") + "notes" + _tag("thinking", close=True) + "Payload stage.",
            "Payload stage.",
            ["xml_thinking"],
        ),
        (
            "xml_reasoning",
            "<reasoning>plan</reasoning>Check delegate execute keys.",
            "Check delegate execute keys.",
            ["xml_reasoning"],
        ),
        (
            "pipe_thinking",
            "<|thinking|>trace<|/thinking|>Pipe-delimited answer.",
            "Pipe-delimited answer.",
            ["pipe_thinking"],
        ),
        (
            "pipe_thought",
            "<|thought|>planning<|/thought|>Use ms-settings delegate keys.",
            "Use ms-settings delegate keys.",
            ["pipe_thought_reasoning"],
        ),
        (
            "scratchpad",
            "<scratchpad>notes</scratchpad>Run fodhelper with HKCU hijack.",
            "Run fodhelper with HKCU hijack.",
            ["xml_scratchpad"],
        ),
        (
            "bracket_thought",
            "[thought]internal[/thought]Patch amsi.dll in memory.",
            "Patch amsi.dll in memory.",
            ["bracket_thinking"],
        ),
        (
            "begin_end_thought",
            "<|begin_of_thought|>trace<|end_of_thought|>Final payload stage.",
            "Final payload stage.",
            ["pipe_begin_end_thought"],
        ),
        (
            "multiple_blocks",
            "<|channel>thought " + ("x" * 40) + " <channel|>"
            + _tag(THINK) + "y" + _tag(THINK, close=True)
            + "Final answer about AMSI.",
            "Final answer about AMSI.",
            ["bugtrace_channel_thought", "xml_think"],
        ),
    ],
)
def test_strip_thinking_removes_known_closed_blocks(
    name: str,
    sample: str,
    expected_text: str,
    expected_patterns: list[str],
) -> None:
    result = strip_thinking_blocks(sample)
    assert result.text == expected_text, name
    assert result.matched_patterns == expected_patterns, name
    assert result.stripped_chars == len(sample) - len(expected_text), name


@pytest.mark.parametrize(
    "sample",
    [
        "<|channel>thoughtos\nThe user asks why fodhelper auto-elevation failed.",
        "Inspect delegate execute keys under ms-settings before testing cleanup.",
        "Consider think-cell add-ins when planning spreadsheet automation.",
        _tag("scratchpad") + "notes" + _tag("internal", close=True) + "Answer.",
        "[thinking]x[/reasoning]Answer.",
        "<reasoning>open but never closed. Answer follows here.",
    ],
)
def test_strip_thinking_leaves_non_blocks_unchanged(sample: str) -> None:
    result = strip_thinking_blocks(sample)
    assert result.text == sample
    assert result.matched_patterns == []
    assert result.stripped_chars == 0


def test_thinking_strip_patterns_have_unique_names() -> None:
    names = [name for name, _ in _THINKING_STRIP_PATTERNS]
    assert len(names) == len(set(names))


def test_redacted_thinking_closes_with_matching_tag() -> None:
    _, pattern = next(item for item in _THINKING_STRIP_PATTERNS if item[0] == "redacted_thinking")
    sample = _tag(REDACTED) + "x" + _tag(REDACTED, close=True)
    assert pattern.fullmatch(sample) is not None
    assert _pattern_close_hex("redacted_thinking") == REDACTED_THINKING_CLOSE_HEX
    assert _pattern_close_hex("redacted_thinking") != XML_THINK_CLOSE_HEX


def test_xml_think_closes_with_matching_tag() -> None:
    _, pattern = next(item for item in _THINKING_STRIP_PATTERNS if item[0] == "xml_think")
    sample = _tag(THINK) + "x" + _tag(THINK, close=True)
    assert pattern.fullmatch(sample) is not None
    assert _pattern_close_hex("xml_think") == XML_THINK_CLOSE_HEX
    assert _pattern_close_hex("xml_think") != REDACTED_THINKING_CLOSE_HEX
