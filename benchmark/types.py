"""Typed benchmark result models."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

DEFAULT_RATE_LIMIT_DELAY = 1.5
DEFAULT_MAX_TOKENS = 3072
DEFAULT_TEMPERATURE = 0.2
DEFAULT_CONCURRENCY = 1


@dataclass
class RuntimeOptions:
    """Runtime settings that directly affect benchmark execution."""

    rate_limit_delay: float = DEFAULT_RATE_LIMIT_DELAY
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    concurrency: int = DEFAULT_CONCURRENCY
    request_log: Optional[str] = None


@dataclass
class QueryResult:
    """Raw model response and score metadata for one question."""

    question: Dict[str, Any]
    response: str
    score: int
    latency_ms: float
    censored: bool = False
    similarity: Optional[float] = None
    normalized_score: Optional[float] = None
    critical_error: bool = False
    criteria_passed: List[str] = field(default_factory=list)
    criteria_failed: List[str] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)
    semantic_score: Optional[int] = None
    semantic_similarity: Optional[float] = None
    semantic_scores: Optional[Dict[str, Any]] = None


@dataclass
class QuestionResult:
    """Persisted per-question benchmark result."""

    id: int
    category: str
    prompt: str
    score: int
    response_snippet: str
    full_response: str
    censored: bool = False
    similarity: Optional[float] = None
    latency_ms: Optional[float] = None
    normalized_score: Optional[float] = None
    critical_error: bool = False
    criteria_passed: List[str] = field(default_factory=list)
    criteria_failed: List[str] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    difficulty: Optional[str] = None
    domain: Optional[str] = None
    capability: Optional[str] = None
    weight: float = 1.0
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    semantic_score: Optional[int] = None
    semantic_similarity: Optional[float] = None
    semantic_scores: Optional[Dict[str, Any]] = None
    rubric_best: Optional[Dict[str, Any]] = None
    semantic_best: Optional[Dict[str, Any]] = None
    tracks_diverged: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return the backward-compatible export shape."""
        result: Dict[str, Any] = {
            "id": self.id,
            "category": self.category,
            "prompt": self.prompt,
            "score": self.score,
            "response_snippet": self.response_snippet,
            "full_response": self.full_response,
            "censored": self.censored,
            "latency_ms": self.latency_ms,
            "normalized_score": self.normalized_score,
            "critical_error": self.critical_error,
            "criteria_passed": self.criteria_passed,
            "criteria_failed": self.criteria_failed,
            "evidence": self.evidence,
            "metrics": self.metrics,
            "difficulty": self.difficulty,
            "domain": self.domain,
            "capability": self.capability,
            "weight": self.weight,
            "details": self.details,
        }
        if self.similarity is not None:
            result["similarity"] = self.similarity
        if self.semantic_score is not None:
            result["semantic_score"] = self.semantic_score
        if self.semantic_similarity is not None:
            result["semantic_similarity"] = self.semantic_similarity
        if self.semantic_scores is not None:
            result["semantic_scores"] = self.semantic_scores
        if self.rubric_best is not None:
            result["rubric_best"] = self.rubric_best
        if self.semantic_best is not None:
            result["semantic_best"] = self.semantic_best
        if self.tracks_diverged is not None:
            result["tracks_diverged"] = self.tracks_diverged
        if self.error is not None:
            result["error"] = self.error
        return result


@dataclass
class BenchmarkRunResult:
    """Complete benchmark result for one model."""

    model: str
    results: List[Dict[str, Any]]
    total_score: float
    interpretation: str
    optimization_results: List[Dict[str, Any]] = field(default_factory=list)
    exported: Dict[str, str] = field(default_factory=dict)
    interrupted: bool = False
