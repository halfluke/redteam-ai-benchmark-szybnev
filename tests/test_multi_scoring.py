"""Tests for multi-scorer benchmark support."""

from scoring.base import ScoringResult
from scoring.factory import MultiScorerBundle, create_multi_scorer_bundle, parse_scorer_methods

from benchmark.runner import _make_result, _score_response_text
from benchmark.scoring_summary import compute_total_scores, primary_total_score


class FakeScorer:
    def __init__(self, score: int):
        self.score_value = score

    def score(self, q_id: int, response: str) -> ScoringResult:
        return ScoringResult(score=self.score_value, censored=False, similarity=0.5)


def test_create_multi_scorer_bundle_shares_semantic_assets(monkeypatch):
    """Semantic + hybrid should load the embedding model only once."""
    model_inits = []

    class FakeSentenceTransformer:
        def __init__(self, model_name, **kwargs):
            model_inits.append(model_name)

        def encode(self, texts, convert_to_tensor=True, show_progress_bar=False):
            count = len(texts) if isinstance(texts, list) else 1
            return [object()] * count

    monkeypatch.setattr(
        "scoring.semantic_scorer.SentenceTransformer", FakeSentenceTransformer
    )
    monkeypatch.setattr(
        "scoring.technical_scorer.SentenceTransformer", FakeSentenceTransformer
    )
    monkeypatch.setattr(
        "scoring.semantic_scorer.load_reference_embeddings",
        lambda **kwargs: {q_id: object() for q_id in range(1, 13)},
    )

    bundle = create_multi_scorer_bundle(
        ["keyword", "semantic", "hybrid"],
        semantic_model="test-model",
        answers_file="answers_all.txt",
        questions=[{"id": 1, "category": "test"}],
        use_llm_in_gray_zone=False,
    )

    assert bundle.is_multi
    assert (
        bundle.scorers["semantic"].model
        is bundle.scorers["hybrid"].technical_scorer.model
    )
    assert model_inits == ["test-model"]


def test_parse_scorer_methods_deduplicates():
    assert parse_scorer_methods("keyword,semantic,keyword") == ["keyword", "semantic"]


def test_score_response_text_multi():
    bundle = MultiScorerBundle(
        method_label="multi (keyword, semantic)",
        score_func=lambda q_id, response: 50,
        methods=["keyword", "semantic"],
        scorers={"keyword": FakeScorer(50), "semantic": FakeScorer(75)},
        is_multi=True,
    )
    primary, scores, similarities, details = _score_response_text(
        None,
        None,
        1,
        "example response",
        None,
        multi_scorer_bundle=bundle,
    )
    assert primary.score == 50
    assert scores == {"keyword": 50, "semantic": 75}
    assert "scorers" in details


def test_make_result_includes_score_map():
    result = _make_result(
        {"id": 1, "category": "test"},
        50,
        "hello",
        scores={"keyword": 50, "semantic": 75},
        similarities={"semantic": 0.8},
    )
    assert result["scores"]["semantic"] == 75
    assert result["similarities"]["semantic"] == 0.8


def test_compute_total_scores_multi():
    results = [
        {"score": 50, "scores": {"keyword": 50, "semantic": 75}},
        {"score": 100, "scores": {"keyword": 100, "semantic": 80}},
    ]
    totals = compute_total_scores(results, ["keyword", "semantic"])
    assert totals["keyword"] == 75.0
    assert totals["semantic"] == 77.5
    assert primary_total_score(totals) == 75.0
