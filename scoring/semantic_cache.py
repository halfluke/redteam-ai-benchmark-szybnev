"""On-disk cache for precomputed semantic reference embeddings."""

import hashlib
import os
from pathlib import Path
from typing import Dict, Optional

DEFAULT_CACHE_DIR = Path(
    os.environ.get("REDTEAM_SEMANTIC_CACHE_DIR", ".cache/redteam/semantic")
)


def _answers_digest(answers_path: Path) -> str:
    if not answers_path.is_file():
        return "missing"
    return hashlib.sha256(answers_path.read_bytes()).hexdigest()


def embedding_cache_path(
    model_name: str,
    answers_file: str,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> Path:
    """Return the cache file path for a model + answers file pair."""
    answers_digest = _answers_digest(Path(answers_file))
    model_slug = model_name.replace("/", "_").replace(":", "_")
    filename = f"{model_slug}_{answers_digest[:16]}.pt"
    return cache_dir / filename


def huggingface_cache_dir() -> Path:
    """Return the Hugging Face hub cache directory."""
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))


def save_reference_embeddings(
    embeddings: Dict[int, object],
    *,
    model_name: str,
    answers_file: str,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> Path:
    """Persist reference embeddings for faster benchmark startup."""
    import torch

    path = embedding_cache_path(model_name, answers_file, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": model_name,
            "answers_file": str(Path(answers_file)),
            "answers_digest": _answers_digest(Path(answers_file)),
            "embeddings": embeddings,
        },
        path,
    )
    return path


def load_reference_embeddings(
    *,
    model_name: str,
    answers_file: str,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> Optional[Dict[int, object]]:
    """Load cached reference embeddings when model and answers match."""
    import torch

    path = embedding_cache_path(model_name, answers_file, cache_dir)
    if not path.is_file():
        return None

    payload = torch.load(path, weights_only=False)
    if payload.get("model_name") != model_name:
        return None
    if payload.get("answers_digest") != _answers_digest(Path(answers_file)):
        return None
    return payload.get("embeddings", {})
