"""Request timing and provider diagnostics for model queries."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


def ns_to_ms(value: Optional[int]) -> Optional[float]:
    """Convert Ollama nanosecond durations to milliseconds."""
    if value is None:
        return None
    return value / 1_000_000


@dataclass
class QueryDiagnostics:
    """Timing and context captured for one model request."""

    provider: str = ""
    endpoint: str = ""
    model: str = ""
    question_id: Optional[int] = None
    category: str = ""
    prompt_chars: int = 0
    max_tokens: int = 0
    temperature: float = 0.0
    timeout_s: int = 0
    attempt: int = 0
    retries: int = 0
    elapsed_ms: float = 0.0
    status: str = "pending"  # pending | success | timeout | error
    error: Optional[str] = None
    total_duration_ms: Optional[float] = None
    load_duration_ms: Optional[float] = None
    prompt_eval_duration_ms: Optional[float] = None
    eval_duration_ms: Optional[float] = None
    prompt_eval_count: Optional[int] = None
    eval_count: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable diagnostics payload."""
        payload = asdict(self)
        payload["elapsed_s"] = round(self.elapsed_ms / 1000, 3)
        if self.status == "success" and self.eval_duration_ms is not None:
            payload["tokens_per_second"] = self._tokens_per_second()
        return payload

    def _tokens_per_second(self) -> Optional[float]:
        if not self.eval_count or not self.eval_duration_ms or self.eval_duration_ms <= 0:
            return None
        return round(self.eval_count / (self.eval_duration_ms / 1000), 2)

    def format_summary(self) -> str:
        """Human-readable one-line timing summary."""
        elapsed_s = self.elapsed_ms / 1000
        if self.status == "timeout":
            return (
                f"⏱ timed out after {elapsed_s:.1f}s "
                f"(limit {self.timeout_s}s, attempt {self.attempt}/{self.retries})"
            )
        if self.status == "error":
            return f"⏱ failed after {elapsed_s:.1f}s"

        parts = [f"⏱ {elapsed_s:.1f}s total"]
        if self.load_duration_ms is not None and self.load_duration_ms > 0:
            parts.append(f"load {self.load_duration_ms / 1000:.1f}s")
        if self.prompt_eval_duration_ms is not None:
            prompt_part = f"prompt {self.prompt_eval_duration_ms / 1000:.1f}s"
            if self.prompt_eval_count is not None:
                prompt_part += f" ({self.prompt_eval_count} tok)"
            parts.append(prompt_part)
        if self.eval_duration_ms is not None:
            gen_part = f"generation {self.eval_duration_ms / 1000:.1f}s"
            if self.eval_count is not None:
                gen_part += f" ({self.eval_count} tok)"
            tps = self._tokens_per_second()
            if tps is not None:
                gen_part += f", {tps} tok/s"
            parts.append(gen_part)
        return " | ".join(parts)

    def format_details(self) -> str:
        """Multi-line context useful when a request is slow or fails."""
        return format_request_details(self.to_dict())


def format_request_summary(request: Dict[str, Any]) -> str:
    """Format a persisted request diagnostics dictionary."""
    if not request:
        return ""

    elapsed_s = request.get("elapsed_s")
    if elapsed_s is None:
        elapsed_s = request.get("elapsed_ms", 0) / 1000

    status = request.get("status", "success")
    if status == "timeout":
        return (
            f"⏱ timed out after {elapsed_s:.1f}s "
            f"(limit {request.get('timeout_s', '?')}s, "
            f"attempt {request.get('attempt', '?')}/{request.get('retries', '?')})"
        )
    if status == "error":
        return f"⏱ failed after {elapsed_s:.1f}s"

    parts = [f"⏱ {elapsed_s:.1f}s total"]
    load_ms = request.get("load_duration_ms")
    if load_ms:
        parts.append(f"load {load_ms / 1000:.1f}s")
    prompt_ms = request.get("prompt_eval_duration_ms")
    if prompt_ms is not None:
        prompt_part = f"prompt {prompt_ms / 1000:.1f}s"
        prompt_count = request.get("prompt_eval_count")
        if prompt_count is not None:
            prompt_part += f" ({prompt_count} tok)"
        parts.append(prompt_part)
    eval_ms = request.get("eval_duration_ms")
    if eval_ms is not None:
        gen_part = f"generation {eval_ms / 1000:.1f}s"
        eval_count = request.get("eval_count")
        if eval_count is not None:
            gen_part += f" ({eval_count} tok)"
        tps = request.get("tokens_per_second")
        if tps is not None:
            gen_part += f", {tps} tok/s"
        parts.append(gen_part)
    return " | ".join(parts)


def format_request_details(request: Dict[str, Any]) -> str:
    """Format persisted diagnostics as multi-line context."""
    if not request:
        return ""

    lines = [
        f"endpoint: {request.get('endpoint', '')}",
        f"model: {request.get('model', '')}",
        (
            f"prompt: {request.get('prompt_chars', 0)} chars, "
            f"max_tokens={request.get('max_tokens', 0)}, "
            f"temperature={request.get('temperature', 0)}"
        ),
        (
            f"timeout: {request.get('timeout_s', '?')}s per attempt, "
            f"retries={request.get('retries', '?')}"
        ),
    ]
    if request.get("status") == "timeout":
        lines.append(
            "likely causes: model still loading, slow VM↔host network, "
            "or generation exceeded timeout — raise provider.timeout in config"
        )
    if request.get("error"):
        lines.append(f"error: {request['error']}")
    return "\n      ".join(lines)
