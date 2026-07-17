"""Warm up semantic scoring assets."""

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .semantic_embedder import DEFAULT_DEEPINFRA_SEMANTIC_MODEL
from .semantic_scorer import (
    DEFAULT_SEMANTIC_ANSWERS_FILE,
    DEFAULT_SEMANTIC_MODEL,
    DEFAULT_SEMANTIC_PROVIDER,
    SemanticScorer,
    default_semantic_max_seq_length,
    parse_semantic_references,
)


@dataclass
class SemanticPreloadResult:
    """Summary of a semantic preload run."""

    provider: str
    model_name: str
    answers_file: str
    elapsed_s: float
    reference_count: int
    encoded_count: int
    embedding_cache: Path
    huggingface_cache: Optional[Path]


def huggingface_cache_dir() -> Path:
    """Return the Hugging Face hub cache directory."""
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))


def _require_local_semantic_dependencies() -> None:
    try:
        import sentence_transformers  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "sentence-transformers not installed. Install with: uv sync --extra semantic"
        ) from e


def preload_semantic_scorer(
    *,
    provider: str = DEFAULT_SEMANTIC_PROVIDER,
    model_name: str = DEFAULT_SEMANTIC_MODEL,
    answers_file: str = DEFAULT_SEMANTIC_ANSWERS_FILE,
    cache_dir: Optional[str | Path] = None,
    device: Optional[str] = None,
    max_seq_length: Optional[int] = None,
    endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
    api_key_env: str = "DEEPINFRA_TOKEN",
    force: bool = False,
) -> SemanticPreloadResult:
    """
    Warm the reference-answer embedding cache for later ``--semantic`` runs.

    Local provider: downloads/loads SentenceTransformer weights.
    DeepInfra provider: encodes references through the remote embeddings API.
    """
    provider = provider.lower()
    if provider == "local":
        _require_local_semantic_dependencies()
    if provider == "deepinfra" and model_name == DEFAULT_SEMANTIC_MODEL:
        model_name = DEFAULT_DEEPINFRA_SEMANTIC_MODEL

    references = parse_semantic_references(answers_file)
    questions = [{"id": q_id} for q_id in sorted(references)]
    scorer = SemanticScorer(
        questions,
        answers_file=answers_file,
        model_name=model_name,
        provider=provider,
        device=device,
        max_seq_length=max_seq_length,
        endpoint=endpoint,
        api_key=api_key,
        api_key_env=api_key_env,
        cache_dir=cache_dir,
    )

    started = time.time()
    warmup = scorer.warm_reference_cache(force=force)
    elapsed_s = time.time() - started

    return SemanticPreloadResult(
        provider=provider,
        model_name=model_name,
        answers_file=answers_file,
        elapsed_s=elapsed_s,
        reference_count=warmup["reference_count"],
        encoded_count=warmup["encoded_count"],
        embedding_cache=warmup["cache_file"],
        huggingface_cache=huggingface_cache_dir() if provider == "local" else None,
    )
