"""Tests for DeepInfra semantic embedding backend."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

import run_benchmark
from scoring.semantic_calibration import (
    DEFAULT_DEEPINFRA_SEMANTIC_THRESHOLDS,
    calibrate_thresholds_from_similarities,
    default_semantic_thresholds,
)
from scoring.semantic_embedder import (
    DeepInfraEmbeddingEmbedder,
    create_semantic_embedder,
)
from scoring.semantic_scorer import SemanticScorer


class _MockTransport(httpx.MockTransport):
    def __init__(self, handler):
        super().__init__(handler)
        self.calls = 0


def _embedding_handler(request: httpx.Request) -> httpx.Response:
    payload = json.loads(request.content.decode("utf-8"))
    inputs = payload["input"]
    if isinstance(inputs, str):
        inputs = [inputs]
    data = []
    for index, text in enumerate(inputs):
        data.append(
            {
                "object": "embedding",
                "index": index,
                "embedding": [float(len(text)), float(index), 1.0],
            }
        )
    return httpx.Response(
        200,
        json={"data": data, "model": payload["model"], "usage": {"total_tokens": 1}},
    )


def test_default_semantic_max_seq_length_for_deepinfra():
    from scoring.semantic_scorer import (
        DEFAULT_DEEPINFRA_SEMANTIC_MAX_SEQ_LENGTH,
        default_semantic_max_seq_length,
    )

    assert default_semantic_max_seq_length(provider="deepinfra") == 3072
    assert default_semantic_max_seq_length(provider="deepinfra") == (
        DEFAULT_DEEPINFRA_SEMANTIC_MAX_SEQ_LENGTH
    )
    assert default_semantic_max_seq_length(provider="local") == 2048


def test_default_optimization_max_tokens_for_deepinfra():
    from utils.config import default_optimization_max_tokens

    assert default_optimization_max_tokens(optimizer_provider="deepinfra") == 3072
    assert default_optimization_max_tokens(optimizer_provider="ollama") == 2048


def test_deepinfra_config_default_max_seq_length(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
scoring:
  semantic:
    provider: deepinfra
    model: Qwen/Qwen3-Embedding-8B
""",
        encoding="utf-8",
    )
    from utils.config import load_config

    config = load_config(str(config_path))
    assert config.scoring.semantic.max_seq_length == 3072
    assert config.scoring.semantic.max_seq_length_explicit is False


def test_run_bundle_uses_deepinfra_max_seq_length_with_default_config(monkeypatch):
    import run_benchmark

    class Args:
        semantic = True
        semantic_provider = "deepinfra"
        semantic_model = None
        semantic_answers = None
        semantic_api_key = None

    class SemanticConfig:
        answers_file = "answers_v2.txt"
        provider = "local"
        model = "Qwen/Qwen3-Embedding-0.6B"
        thresholds_explicit = False
        thresholds = {}
        device = "auto"
        max_seq_length = 2048
        max_seq_length_explicit = False
        endpoint = None
        api_key = None
        api_key_env = "DEEPINFRA_TOKEN"

    class ScoringConfig:
        semantic = SemanticConfig()

    class Config:
        scoring = ScoringConfig()

    captured = {}

    def fake_create_semantic_scorer(**kwargs):
        captured.update(kwargs)
        bundle = type("Bundle", (), {})()
        bundle.scorer = type(
            "Scorer",
            (),
            {"warm_encoder": lambda self: None},
        )()
        return bundle

    monkeypatch.setattr(run_benchmark, "create_semantic_scorer", fake_create_semantic_scorer)
    run_benchmark._create_semantic_scorer_bundle(
        Args(),
        Config(),
        [{"id": 1}],
    )
    assert captured["max_seq_length"] == 3072


def test_deepinfra_optimizer_config_default_max_tokens(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
optimization:
  optimizer_provider: deepinfra
  optimizer_model: deepseek-ai/DeepSeek-V4-Flash
""",
        encoding="utf-8",
    )
    from utils.config import load_config

    config = load_config(str(config_path))
    assert config.optimization.optimization_max_tokens == 3072
    assert config.optimization.optimization_max_tokens_explicit is False
    thresholds = default_semantic_thresholds(
        provider="deepinfra",
        model_name="Qwen/Qwen3-Embedding-8B",
    )
    assert thresholds == DEFAULT_DEEPINFRA_SEMANTIC_THRESHOLDS


def test_deepinfra_embedder_show_progress_uses_tqdm(monkeypatch):
    updates = []

    class FakeProgress:
        def __init__(self, *, total, unit, desc):
            self.total = total
            self.unit = unit
            self.desc = desc

        def update(self, amount):
            updates.append(amount)

        def close(self):
            return None

    monkeypatch.setattr("tqdm.tqdm", FakeProgress)

    transport = _MockTransport(_embedding_handler)
    client = httpx.Client(transport=transport)
    embedder = DeepInfraEmbeddingEmbedder(
        "Qwen/Qwen3-Embedding-8B",
        max_seq_length=3072,
        api_key="test-token",
        client=client,
    )

    vectors = embedder.encode(
        ["alpha", "beta", "gamma", "delta"],
        batch_size=2,
        show_progress=True,
    )

    assert len(vectors) == 4
    assert updates == [2, 2]
    assert sum(updates) == 4


def test_deepinfra_embedder_batches_and_normalizes(monkeypatch):
    transport = _MockTransport(_embedding_handler)
    client = httpx.Client(transport=transport)
    embedder = DeepInfraEmbeddingEmbedder(
        "Qwen/Qwen3-Embedding-8B",
        max_seq_length=3072,
        api_key="test-token",
        client=client,
    )

    vectors = embedder.encode(["alpha", "beta"], batch_size=1)

    assert len(vectors) == 2
    assert pytest.approx(sum(v * v for v in vectors[0])) == 1.0
    assert vectors[0][0] != vectors[1][0]


def test_semantic_scorer_uses_deepinfra_embedder(tmp_path):
    transport = _MockTransport(_embedding_handler)
    client = httpx.Client(transport=transport)
    embedder = DeepInfraEmbeddingEmbedder(
        "Qwen/Qwen3-Embedding-8B",
        max_seq_length=3072,
        api_key="test-token",
        client=client,
    )
    scorer = SemanticScorer(
        [{"id": 1}],
        answers_file="answers_v2.txt",
        provider="deepinfra",
        model_name="Qwen/Qwen3-Embedding-8B",
        embedder=embedder,
        cache_dir=tmp_path,
    )

    result = scorer.score(1, scorer.references[1])

    assert result.score == 100
    assert result.details["provider"] == "deepinfra"


def test_create_semantic_embedder_requires_deepinfra_token():
    with pytest.raises(RuntimeError, match="DEEPINFRA_TOKEN"):
        create_semantic_embedder(
            provider="deepinfra",
            model_name="Qwen/Qwen3-Embedding-8B",
            max_seq_length=2048,
            api_key=None,
        )


def test_deepinfra_config_uses_recalibrated_thresholds_when_omitted(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
scoring:
  semantic:
    provider: deepinfra
    model: Qwen/Qwen3-Embedding-8B
""",
        encoding="utf-8",
    )
    from utils.config import load_config

    config = load_config(str(config_path))
    assert config.scoring.semantic.provider == "deepinfra"
    assert config.scoring.semantic.thresholds_explicit is False
    assert config.scoring.semantic.thresholds == DEFAULT_DEEPINFRA_SEMANTIC_THRESHOLDS


def test_deepinfra_config_keeps_explicit_thresholds(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
scoring:
  semantic:
    provider: deepinfra
    thresholds:
      100: 0.99
      90: 0.95
      80: 0.90
      70: 0.85
      60: 0.80
      50: 0.75
      40: 0.70
      30: 0.65
""",
        encoding="utf-8",
    )
    from utils.config import load_config

    config = load_config(str(config_path))
    assert config.scoring.semantic.thresholds_explicit is True
    assert config.scoring.semantic.thresholds[100] == 0.99


def test_apply_config_sets_semantic_provider_from_yaml(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
provider:
  name: ollama
  endpoint: http://localhost:11434
scoring:
  semantic:
    enabled: true
    provider: deepinfra
    model: Qwen/Qwen3-Embedding-8B
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPINFRA_TOKEN", "test-token")

    class Args:
        endpoint = None
        api_key = None
        semantic = False
        semantic_provider = None
        semantic_model = None
        semantic_answers = None
        semantic_api_key = None
        max_optimization_iterations = None
        optimizer_provider = None
        optimizer_model = None
        optimizer_api_key = None
        optimizer_endpoint = None

    args = Args()
    config = run_benchmark.load_config(str(config_path))
    run_benchmark._apply_config_defaults(args, config)

    assert args.semantic is True
    assert args.semantic_provider == "deepinfra"
    assert args.semantic_model == "Qwen/Qwen3-Embedding-8B"


def test_calibrate_thresholds_from_similarities_is_monotonic():
    thresholds = calibrate_thresholds_from_similarities(
        exact=[0.99, 0.985, 0.992],
        high=[0.91, 0.89, 0.90],
        medium=[0.82, 0.80, 0.81],
        low=[0.72, 0.70, 0.71],
        minimal=[0.61, 0.59, 0.60],
    )

    ordered = [thresholds[score] for score in (100, 90, 80, 70, 60, 50, 40, 30)]
    assert ordered == sorted(ordered, reverse=True)
