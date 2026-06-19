"""Warm up semantic scoring assets on the local machine."""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .constants import DEFAULT_SEMANTIC_MODEL
from .semantic_cache import (
    DEFAULT_CACHE_DIR,
    embedding_cache_path,
    huggingface_cache_dir,
)
from .semantic_scorer import SEMANTIC_AVAILABLE, SemanticScorer


@dataclass
class SemanticPreloadResult:
    """Summary of a semantic preload run."""

    model_name: str
    answers_file: str
    elapsed_s: float
    reference_count: int
    embedding_cache: Path
    huggingface_cache: Path


def preload_semantic_scorer(
    *,
    model_name: str = DEFAULT_SEMANTIC_MODEL,
    answers_file: str = "answers_all.txt",
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> SemanticPreloadResult:
    """
    Download/load the embedding model and warm the reference-answer cache.

    This does not keep a daemon running; it prepares on-disk caches so later
    benchmark runs start faster.
    """
    if not SEMANTIC_AVAILABLE:
        raise RuntimeError(
            "sentence-transformers not installed. Install with: uv sync --extra semantic"
        )

    started = time.time()
    scorer = SemanticScorer(model_name)
    scorer.load_reference_answers(answers_file, cache_dir=cache_dir, use_cache=False)
    elapsed_s = time.time() - started

    return SemanticPreloadResult(
        model_name=model_name,
        answers_file=answers_file,
        elapsed_s=elapsed_s,
        reference_count=len(scorer.reference_embeddings),
        embedding_cache=embedding_cache_path(model_name, answers_file, cache_dir),
        huggingface_cache=huggingface_cache_dir(),
    )
