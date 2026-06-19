"""Factory for benchmark scorers."""

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .base import BaseScorer, ScoringResult
from .hybrid_scorer import HybridScorer, create_hybrid_scorer
from .keyword_scorer import KeywordScorer
from .llm_judge import LLMJudge
from .semantic_scorer import (
    SEMANTIC_AVAILABLE,
    SemanticScorer,
    parse_reference_answers,
)

VALID_SCORER_METHODS = ("keyword", "semantic", "hybrid", "llm_judge")


@dataclass
class ScorerBundle:
    """Resolved scorer and metadata for benchmark orchestration."""

    method_label: str
    score_func: Callable[[int, str], int]
    details: Dict = field(default_factory=dict)
    scorer: Optional[BaseScorer] = None
    is_multi: bool = False


@dataclass
class MultiScorerBundle(ScorerBundle):
    """Score each response with several scorers after a single model query."""

    methods: List[str] = field(default_factory=list)
    scorers: Dict[str, BaseScorer] = field(default_factory=dict)
    is_multi: bool = True

    def score_all(self, q_id: int, response: str) -> Dict[str, ScoringResult]:
        """Run every configured scorer on the same response text."""
        return {method: self.scorers[method].score(q_id, response) for method in self.methods}

    @property
    def primary_method(self) -> str:
        """Primary score column for backward-compatible exports."""
        return "keyword" if "keyword" in self.methods else self.methods[0]


def _categories_by_id(questions: List[Dict]) -> Dict[int, str]:
    """Build q_id -> category mapping from benchmark questions."""
    return {q["id"]: q.get("category", f"Question {q['id']}") for q in questions}


def _score_value(scorer: BaseScorer) -> Callable[[int, str], int]:
    """Return a simple score function for BaseScorer instances."""
    return scorer.score_value


def _require_semantic_available() -> None:
    if not SEMANTIC_AVAILABLE:
        raise RuntimeError(
            "sentence-transformers not installed. "
            "Install with: uv sync --extra semantic"
        )


def _build_hybrid_scorer(
    *,
    semantic_model: str,
    answers_file: str,
    questions: List[Dict],
    openrouter_api_key: Optional[str] = None,
    llm_judge_model: str = "anthropic/claude-3.5-sonnet",
    semantic_weight: float = 0.7,
    keyword_weight: float = 0.3,
    use_llm_in_gray_zone: bool = True,
    shared_semantic: Optional[SemanticScorer] = None,
) -> HybridScorer:
    """Create a hybrid scorer, optionally reusing a shared semantic scorer."""
    _require_semantic_available()

    api_key = openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
    if use_llm_in_gray_zone and not api_key:
        raise RuntimeError(
            "--scorer hybrid with LLM gray-zone judging requires --api-key or "
            "OPENROUTER_API_KEY. Set use_llm_in_gray_zone: false in config "
            "to run technical-only hybrid scoring."
        )

    reference_answers = parse_reference_answers(answers_file)
    scorer = create_hybrid_scorer(
        model_name=semantic_model,
        openrouter_api_key=api_key,
        llm_model=llm_judge_model,
        reference_answers=reference_answers,
        categories=_categories_by_id(questions),
        use_llm=use_llm_in_gray_zone,
        answers_file=answers_file,
        shared_semantic=shared_semantic,
    )
    scorer.technical_scorer.semantic_weight = semantic_weight
    scorer.technical_scorer.keyword_weight = keyword_weight
    return scorer


def create_scorer(
    method: str,
    *,
    semantic_model: str,
    answers_file: str,
    questions: List[Dict],
    openrouter_api_key: Optional[str] = None,
    llm_judge_model: str = "anthropic/claude-3.5-sonnet",
    semantic_weight: float = 0.7,
    keyword_weight: float = 0.3,
    use_llm_in_gray_zone: bool = True,
) -> ScorerBundle:
    """
    Create a benchmark scorer.

    Raises RuntimeError with a user-facing message when an explicitly selected
    scorer cannot run because optional dependencies or credentials are missing.
    """
    method = method.lower()

    if method == "keyword":
        scorer = KeywordScorer()
        return ScorerBundle(
            method_label="keyword",
            score_func=_score_value(scorer),
            details={"method": "keyword"},
            scorer=scorer,
        )

    if method == "semantic":
        _require_semantic_available()

        scorer = SemanticScorer(semantic_model)
        scorer.load_reference_answers(answers_file)
        return ScorerBundle(
            method_label=f"semantic ({semantic_model})",
            score_func=scorer.score_response,
            details={"method": "semantic", "semantic_model": semantic_model},
            scorer=scorer,
        )

    if method == "hybrid":
        scorer = _build_hybrid_scorer(
            semantic_model=semantic_model,
            answers_file=answers_file,
            questions=questions,
            openrouter_api_key=openrouter_api_key,
            llm_judge_model=llm_judge_model,
            semantic_weight=semantic_weight,
            keyword_weight=keyword_weight,
            use_llm_in_gray_zone=use_llm_in_gray_zone,
        )
        return ScorerBundle(
            method_label=f"hybrid ({semantic_model})",
            score_func=_score_value(scorer),
            details={"method": "hybrid", "semantic_model": semantic_model},
            scorer=scorer,
        )

    if method == "llm_judge":
        api_key = openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "--scorer llm_judge requires --api-key or OPENROUTER_API_KEY"
            )

        scorer = LLMJudge(
            model=llm_judge_model,
            api_key=api_key,
            reference_answers=parse_reference_answers(answers_file),
            categories=_categories_by_id(questions),
        )
        if not scorer.is_available():
            raise RuntimeError(
                "--scorer llm_judge is unavailable. Ensure httpx/tenacity are installed "
                "and an OpenRouter API key is configured."
            )

        return ScorerBundle(
            method_label=f"llm_judge ({llm_judge_model})",
            score_func=_score_value(scorer),
            details={"method": "llm_judge", "llm_judge_model": llm_judge_model},
            scorer=scorer,
        )

    raise ValueError(f"Unknown scorer: {method}")


def parse_scorer_methods(method: str) -> List[str]:
    """Parse a scorer CLI/config value into one or more method names."""
    methods = [part.strip().lower() for part in method.split(",") if part.strip()]
    if not methods:
        raise ValueError("At least one scorer method is required")
    unknown = [m for m in methods if m not in VALID_SCORER_METHODS]
    if unknown:
        raise ValueError(f"Unknown scorer(s): {', '.join(unknown)}")
    deduped: List[str] = []
    for method_name in methods:
        if method_name not in deduped:
            deduped.append(method_name)
    return deduped


def create_multi_scorer_bundle(
    methods: List[str],
    *,
    semantic_model: str,
    answers_file: str,
    questions: List[Dict],
    openrouter_api_key: Optional[str] = None,
    llm_judge_model: str = "anthropic/claude-3.5-sonnet",
    semantic_weight: float = 0.7,
    keyword_weight: float = 0.3,
    use_llm_in_gray_zone: bool = True,
) -> MultiScorerBundle:
    """Create a bundle that applies multiple scorers to each response."""
    parsed = parse_scorer_methods(",".join(methods))
    if len(parsed) == 1:
        single = create_scorer(
            parsed[0],
            semantic_model=semantic_model,
            answers_file=answers_file,
            questions=questions,
            openrouter_api_key=openrouter_api_key,
            llm_judge_model=llm_judge_model,
            semantic_weight=semantic_weight,
            keyword_weight=keyword_weight,
            use_llm_in_gray_zone=use_llm_in_gray_zone,
        )
        return MultiScorerBundle(
            method_label=single.method_label,
            score_func=single.score_func,
            details=single.details,
            scorer=single.scorer,
            methods=parsed,
            scorers={parsed[0]: single.scorer},
            is_multi=False,
        )

    scorers: Dict[str, BaseScorer] = {}
    shared_semantic: Optional[SemanticScorer] = None

    if "semantic" in parsed and "hybrid" in parsed:
        _require_semantic_available()
        shared_semantic = SemanticScorer(semantic_model)
        shared_semantic.load_reference_answers(answers_file)

    for method in parsed:
        if method == "keyword":
            scorers[method] = KeywordScorer()
            continue

        if method == "semantic":
            if shared_semantic is not None:
                scorers[method] = shared_semantic
                continue
            bundle = create_scorer(
                method,
                semantic_model=semantic_model,
                answers_file=answers_file,
                questions=questions,
                openrouter_api_key=openrouter_api_key,
                llm_judge_model=llm_judge_model,
                semantic_weight=semantic_weight,
                keyword_weight=keyword_weight,
                use_llm_in_gray_zone=use_llm_in_gray_zone,
            )
            scorers[method] = bundle.scorer
            continue

        if method == "hybrid":
            scorers[method] = _build_hybrid_scorer(
                semantic_model=semantic_model,
                answers_file=answers_file,
                questions=questions,
                openrouter_api_key=openrouter_api_key,
                llm_judge_model=llm_judge_model,
                semantic_weight=semantic_weight,
                keyword_weight=keyword_weight,
                use_llm_in_gray_zone=use_llm_in_gray_zone,
                shared_semantic=shared_semantic,
            )
            continue

        bundle = create_scorer(
            method,
            semantic_model=semantic_model,
            answers_file=answers_file,
            questions=questions,
            openrouter_api_key=openrouter_api_key,
            llm_judge_model=llm_judge_model,
            semantic_weight=semantic_weight,
            keyword_weight=keyword_weight,
            use_llm_in_gray_zone=use_llm_in_gray_zone,
        )
        if bundle.scorer is None:
            raise RuntimeError(f"Scorer '{method}' did not expose a scorer instance")
        scorers[method] = bundle.scorer

    primary = "keyword" if "keyword" in parsed else parsed[0]
    primary_scorer = scorers[primary]
    return MultiScorerBundle(
        method_label=f"multi ({', '.join(parsed)})",
        score_func=_score_value(primary_scorer),
        details={"method": "multi", "methods": parsed},
        scorer=primary_scorer,
        methods=parsed,
        scorers=scorers,
        is_multi=True,
    )
