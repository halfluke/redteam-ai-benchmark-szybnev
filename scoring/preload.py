"""Warm up semantic scoring assets on the local machine."""

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .semantic_scorer import (
    DEFAULT_SEMANTIC_ANSWERS_FILE,
    DEFAULT_SEMANTIC_MAX_SEQ_LENGTH,
    DEFAULT_SEMANTIC_MODEL,
    SemanticScorer,
    parse_semantic_references,
)


@dataclass
class SemanticPreloadResult:
    """Summary of a semantic preload run."""

    model_name: str
    answers_file: str
    elapsed_s: float
    reference_count: int
    encoded_count: int
    embedding_cache: Path
    huggingface_cache: Path


def huggingface_cache_dir() -> Path:
    """Return the Hugging Face hub cache directory."""
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))


def _require_semantic_dependencies() -> None:
    try:
        import sentence_transformers  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "sentence-transformers not installed. Install with: uv sync --extra semantic"
        ) from e


def preload_semantic_scorer(
    *,
    model_name: str = DEFAULT_SEMANTIC_MODEL,
    answers_file: str = DEFAULT_SEMANTIC_ANSWERS_FILE,
    cache_dir: Optional[str | Path] = None,
    device: Optional[str] = None,
    max_seq_length: Optional[int] = None,
    force: bool = False,
) -> SemanticPreloadResult:
    """
    Download/load the embedding model and warm the reference-answer cache.

    This does not keep a daemon running; it prepares on-disk caches so later
    benchmark runs with ``--semantic`` start faster.
    """
    _require_semantic_dependencies()

    references = parse_semantic_references(answers_file)
    questions = [{"id": q_id} for q_id in sorted(references)]
    scorer = SemanticScorer(
        questions,
        answers_file=answers_file,
        model_name=model_name,
        device=device,
        max_seq_length=max_seq_length or DEFAULT_SEMANTIC_MAX_SEQ_LENGTH,
        cache_dir=cache_dir,
    )

    started = time.time()
    warmup = scorer.warm_reference_cache(force=force)
    elapsed_s = time.time() - started

    return SemanticPreloadResult(
        model_name=model_name,
        answers_file=answers_file,
        elapsed_s=elapsed_s,
        reference_count=warmup["reference_count"],
        encoded_count=warmup["encoded_count"],
        embedding_cache=warmup["cache_file"],
        huggingface_cache=huggingface_cache_dir(),
    )
