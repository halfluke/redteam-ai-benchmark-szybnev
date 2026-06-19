from pathlib import Path

import pytest

from scoring.semantic_cache import (
    embedding_cache_path,
    load_reference_embeddings,
    save_reference_embeddings,
)


def test_embedding_cache_path_is_stable_for_same_inputs(tmp_path):
    answers = tmp_path / "answers_all.txt"
    answers.write_text("=== Q1: test ===\nanswer\n", encoding="utf-8")

    first = embedding_cache_path("Qwen/Qwen3-Embedding-0.6B", str(answers), tmp_path)
    second = embedding_cache_path("Qwen/Qwen3-Embedding-0.6B", str(answers), tmp_path)

    assert first == second
    assert first.name.startswith("Qwen_Qwen3-Embedding-0.6B_")


def test_embedding_cache_invalidates_when_answers_change(tmp_path):
    answers = tmp_path / "answers_all.txt"
    answers.write_text("=== Q1: test ===\nanswer\n", encoding="utf-8")
    first = embedding_cache_path("Qwen/Qwen3-Embedding-0.6B", str(answers), tmp_path)

    answers.write_text("=== Q1: test ===\nchanged\n", encoding="utf-8")
    second = embedding_cache_path("Qwen/Qwen3-Embedding-0.6B", str(answers), tmp_path)

    assert first != second


@pytest.mark.skipif(
    not Path(".venv").exists(),
    reason="torch only needed when semantic extra is installed",
)
def test_save_and_load_reference_embeddings_roundtrip(tmp_path):
    pytest.importorskip("torch")

    answers = tmp_path / "answers_all.txt"
    answers.write_text("=== Q1: test ===\nanswer\n", encoding="utf-8")
    embeddings = {1: "tensor-placeholder"}

    save_reference_embeddings(
        embeddings,
        model_name="Qwen/Qwen3-Embedding-0.6B",
        answers_file=str(answers),
        cache_dir=tmp_path,
    )

    loaded = load_reference_embeddings(
        model_name="Qwen/Qwen3-Embedding-0.6B",
        answers_file=str(answers),
        cache_dir=tmp_path,
    )

    assert loaded == embeddings
