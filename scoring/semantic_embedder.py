"""Embedding backends for optional semantic scoring."""

from __future__ import annotations

import math
import os
from typing import Any, List, Optional, Protocol, runtime_checkable

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

DEFAULT_DEEPINFRA_EMBEDDINGS_ENDPOINT = "https://api.deepinfra.com/v1/openai/embeddings"
DEFAULT_DEEPINFRA_SEMANTIC_MODEL = "Qwen/Qwen3-Embedding-8B"
DEFAULT_DEEPINFRA_API_KEY_ENV = "DEEPINFRA_TOKEN"
DEEPINFRA_ENCODE_BATCH_SIZE = 8


def _truncate_text_for_max_seq_length(text: str, max_seq_length: int) -> str:
    """Approximate token cap with a character budget (4 chars per token)."""
    max_chars = max(1, max_seq_length * 4)
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def _normalize_vector(vector: List[float]) -> List[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def _to_float_list(vector: Any) -> List[float]:
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    return [float(value) for value in vector]


@runtime_checkable
class SemanticEmbedder(Protocol):
    """Minimal interface used by ``SemanticScorer``."""

    model_name: str
    provider: str

    def encode(
        self,
        texts: List[str],
        *,
        batch_size: Optional[int] = None,
        show_progress: bool = False,
    ) -> List[List[float]]: ...

    def warm(self) -> None: ...


class LocalSentenceTransformerEmbedder:
    """Local SentenceTransformer backend (``uv sync --extra semantic``)."""

    provider = "local"

    def __init__(
        self,
        model_name: str,
        *,
        device: Optional[str] = None,
        max_seq_length: int,
    ):
        self.model_name = model_name
        self.device = device
        self.max_seq_length = max_seq_length
        self._encoder: Any = None

    def warm(self) -> None:
        self._load_encoder()

    def _load_encoder(self) -> Any:
        if self._encoder is not None:
            return self._encoder

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError(
                "Semantic scoring requires optional dependencies. "
                "Install them with: uv sync --extra semantic"
            ) from e

        kwargs = {}
        if self.device and self.device != "auto":
            kwargs["device"] = self.device
        self._encoder = SentenceTransformer(self.model_name, **kwargs)
        self._encoder.max_seq_length = self.max_seq_length
        return self._encoder

    def encode(
        self,
        texts: List[str],
        *,
        batch_size: Optional[int] = None,
        show_progress: bool = False,
    ) -> List[List[float]]:
        encoder = self._load_encoder()
        encode_kwargs = {
            "convert_to_numpy": True,
            "normalize_embeddings": True,
            "show_progress_bar": show_progress,
        }
        if batch_size is not None:
            encode_kwargs["batch_size"] = batch_size
        try:
            encoded = encoder.encode(texts, **encode_kwargs)
        except TypeError:
            encoded = encoder.encode(texts)
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        if texts and encoded and not isinstance(encoded[0], (list, tuple)):
            encoded = [encoded]
        return [_to_float_list(vector) for vector in encoded]


class DeepInfraEmbeddingEmbedder:
    """DeepInfra OpenAI-compatible embeddings API backend."""

    provider = "deepinfra"

    def __init__(
        self,
        model_name: str,
        *,
        max_seq_length: int,
        endpoint: str = DEFAULT_DEEPINFRA_EMBEDDINGS_ENDPOINT,
        api_key: Optional[str] = None,
        api_key_env: str = DEFAULT_DEEPINFRA_API_KEY_ENV,
        timeout: int = 120,
        client: Optional[httpx.Client] = None,
    ):
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.endpoint = endpoint.rstrip("/")
        if self.endpoint.endswith("/embeddings"):
            self.embeddings_url = self.endpoint
        else:
            self.embeddings_url = f"{self.endpoint}/embeddings"
        self.api_key_env = api_key_env
        self.api_key = api_key or os.environ.get(api_key_env)
        self.timeout = timeout
        self._client = client

        if not self.api_key:
            raise RuntimeError(
                f"DeepInfra semantic scoring requires an API key. "
                f"Set {api_key_env} or pass api_key."
            )

    def _http_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def warm(self) -> None:
        self.encode(["semantic warmup"], batch_size=1)

    def encode(
        self,
        texts: List[str],
        *,
        batch_size: Optional[int] = None,
        show_progress: bool = False,
    ) -> List[List[float]]:
        if not texts:
            return []

        chunk_size = batch_size or DEEPINFRA_ENCODE_BATCH_SIZE
        vectors: List[List[float]] = []
        use_progress = show_progress and len(texts) > 1
        progress = None

        if use_progress:
            from tqdm import tqdm

            progress = tqdm(total=len(texts), unit="answer", desc="Batches")

        try:
            for start in range(0, len(texts), chunk_size):
                chunk = [
                    _truncate_text_for_max_seq_length(text, self.max_seq_length)
                    for text in texts[start : start + chunk_size]
                ]
                vectors.extend(self._encode_batch(chunk))
                if progress is not None:
                    progress.update(len(chunk))
        finally:
            if progress is not None:
                progress.close()

        return vectors

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def _encode_batch(self, texts: List[str]) -> List[List[float]]:
        payload = {
            "model": self.model_name,
            "input": texts if len(texts) > 1 else texts[0],
            "encoding_format": "float",
        }
        response = self._http_client().post(
            self.embeddings_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        response.raise_for_status()
        body = response.json()
        data = body.get("data") or []
        ordered = sorted(data, key=lambda item: int(item.get("index", 0)))
        if len(ordered) != len(texts):
            raise RuntimeError(
                f"DeepInfra embeddings returned {len(ordered)} vectors for "
                f"{len(texts)} input(s)"
            )
        return [_normalize_vector(_to_float_list(item["embedding"])) for item in ordered]


class InjectedEmbedderAdapter:
    """Wrap a legacy test ``encoder`` object for ``SemanticScorer`` injection."""

    provider = "local"

    def __init__(self, encoder: Any, *, model_name: str, max_seq_length: int):
        self._encoder = encoder
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        if hasattr(encoder, "max_seq_length"):
            encoder.max_seq_length = max_seq_length

    def warm(self) -> None:
        return None

    def encode(
        self,
        texts: List[str],
        *,
        batch_size: Optional[int] = None,
        show_progress: bool = False,
    ) -> List[List[float]]:
        encode_kwargs = {
            "convert_to_numpy": True,
            "normalize_embeddings": True,
            "show_progress_bar": show_progress,
        }
        if batch_size is not None:
            encode_kwargs["batch_size"] = batch_size
        try:
            encoded = self._encoder.encode(texts, **encode_kwargs)
        except TypeError:
            encoded = self._encoder.encode(texts)
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        if texts and encoded and not isinstance(encoded[0], (list, tuple)):
            encoded = [encoded]
        return [_to_float_list(vector) for vector in encoded]


def create_semantic_embedder(
    *,
    provider: str = "local",
    model_name: str,
    device: Optional[str] = None,
    max_seq_length: int,
    endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
    api_key_env: str = DEFAULT_DEEPINFRA_API_KEY_ENV,
    embedder: Any = None,
) -> SemanticEmbedder:
    """Create the configured semantic embedding backend."""
    if embedder is not None:
        if isinstance(embedder, (LocalSentenceTransformerEmbedder, DeepInfraEmbeddingEmbedder)):
            return embedder
        if hasattr(embedder, "encode"):
            return InjectedEmbedderAdapter(
                embedder,
                model_name=model_name,
                max_seq_length=max_seq_length,
            )
        raise TypeError(f"Unsupported semantic embedder: {type(embedder)!r}")

    provider = provider.lower()
    if provider == "local":
        return LocalSentenceTransformerEmbedder(
            model_name,
            device=device,
            max_seq_length=max_seq_length,
        )
    if provider == "deepinfra":
        return DeepInfraEmbeddingEmbedder(
            model_name,
            max_seq_length=max_seq_length,
            endpoint=endpoint or DEFAULT_DEEPINFRA_EMBEDDINGS_ENDPOINT,
            api_key=api_key,
            api_key_env=api_key_env,
        )
    raise ValueError(f"Unsupported semantic provider: {provider}")
