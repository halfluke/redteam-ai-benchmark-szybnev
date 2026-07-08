"""Factory for benchmark scorers."""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .base import BaseScorer
from .rubric_scorer import RubricScorer
from .semantic_scorer import SemanticScorer


@dataclass
class ScorerBundle:
    """Resolved scorer and metadata for benchmark orchestration."""

    method_label: str
    score_func: Callable[[int, str], int]
    details: Dict = field(default_factory=dict)
    scorer: Optional[BaseScorer] = None


def _score_value(scorer: BaseScorer) -> Callable[[int, str], int]:
    """Return a simple score function for BaseScorer instances."""
    return scorer.score_value


def create_scorer(
    method: str,
    *,
    questions: List[Dict],
) -> ScorerBundle:
    """Create the only supported runtime benchmark scorer."""
    method = method.lower()

    if method != "rubric":
        raise ValueError(f"Unsupported scorer: {method}")

    scorer = RubricScorer(questions)
    return ScorerBundle(
        method_label="rubric",
        score_func=_score_value(scorer),
        details={"method": "rubric", "scorer_version": RubricScorer.VERSION},
        scorer=scorer,
    )


def create_semantic_scorer(
    *,
    questions: List[Dict],
    answers_file: str,
    model_name: str,
    thresholds: Optional[Dict[int, float]] = None,
    device: Optional[str] = None,
    max_seq_length: Optional[int] = None,
) -> ScorerBundle:
    """Create the optional parallel semantic scorer."""
    scorer = SemanticScorer(
        questions,
        answers_file=answers_file,
        model_name=model_name,
        thresholds=thresholds,
        device=device,
        max_seq_length=max_seq_length,
    )
    return ScorerBundle(
        method_label="semantic",
        score_func=_score_value(scorer),
        details={"method": "semantic", "scorer_version": SemanticScorer.VERSION},
        scorer=scorer,
    )
