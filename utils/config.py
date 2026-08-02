"""YAML configuration management for benchmark."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from benchmark.types import DEFAULT_MAX_TOKENS
from scoring.semantic_calibration import (
    DEFAULT_SEMANTIC_THRESHOLDS,
    default_semantic_thresholds,
)
from scoring.semantic_embedder import DEFAULT_DEEPINFRA_SEMANTIC_MODEL
from scoring.semantic_scorer import (
    DEFAULT_SEMANTIC_ANSWERS_FILE,
    DEFAULT_SEMANTIC_MAX_SEQ_LENGTH,
    DEFAULT_SEMANTIC_MODEL,
    DEFAULT_SEMANTIC_PROVIDER,
    default_semantic_max_seq_length,
)

DEFAULT_QUESTIONS_FILE = "datasets/v2/benchmark.jsonl"
DEFAULT_ANSWERS_FILE = "answers_all.txt"
DEFAULT_SCORER = "rubric"
DEFAULT_OPTIMIZATION_MAX_TOKENS = 2048


def default_optimization_max_tokens(*, optimizer_provider: Optional[str]) -> int:
    """Return provider-aware optimization target max_tokens (matches benchmark default)."""
    if optimizer_provider and optimizer_provider.lower() == "deepinfra":
        return DEFAULT_MAX_TOKENS
    return DEFAULT_OPTIMIZATION_MAX_TOKENS


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""

    name: str
    endpoint: str
    api_key: Optional[str] = None
    api_key_env: Optional[str] = None
    default_model: Optional[str] = None
    timeout: int = 120
    keep_alive: Optional[str] = None
    auth: Optional[str] = None  # e.g. "cloudrun_identity"
    cloudrun_audience: Optional[str] = None
    cloudrun_impersonate_service_account: Optional[str] = None


@dataclass
class SemanticConfig:
    """Configuration for optional parallel semantic scoring."""

    enabled: bool = False
    provider: str = DEFAULT_SEMANTIC_PROVIDER
    answers_file: str = DEFAULT_SEMANTIC_ANSWERS_FILE
    model: str = DEFAULT_SEMANTIC_MODEL
    endpoint: Optional[str] = None
    api_key: Optional[str] = None
    api_key_env: str = "DEEPINFRA_TOKEN"
    thresholds: Dict[int, float] = field(
        default_factory=lambda: dict(DEFAULT_SEMANTIC_THRESHOLDS)
    )
    thresholds_explicit: bool = False
    device: str = "auto"
    max_seq_length: int = DEFAULT_SEMANTIC_MAX_SEQ_LENGTH
    max_seq_length_explicit: bool = False


@dataclass
class ScoringConfig:
    """Configuration for scoring system."""

    method: str = DEFAULT_SCORER
    semantic: SemanticConfig = field(default_factory=SemanticConfig)


@dataclass
class ExportConfig:
    """Configuration for export system."""

    formats: List[str] = field(default_factory=lambda: ["json"])
    output_dir: str = "."
    include_response: bool = True


@dataclass
class OptimizationConfig:
    """Configuration for prompt optimization."""

    optimizer_provider: Optional[str] = None
    optimizer_model: Optional[str] = None
    optimizer_endpoint: Optional[str] = None
    optimizer_api_key: Optional[str] = None
    max_iterations: int = 4
    optimization_max_tokens: int = DEFAULT_OPTIMIZATION_MAX_TOKENS
    optimization_max_tokens_explicit: bool = False
    strategies: List[str] = field(default_factory=lambda: [
        "role_playing",
        "technical",
        "few_shot",
        "cve_framing",
    ])


@dataclass
class KeepaliveConfig:
    """Configuration for background model keepalive pings."""

    enabled: bool = True
    interval_s: float = 60
    max_tokens: int = 16
    prompt: str = "Say OK"
    timeout_s: int = 90


@dataclass
class GpuCheckConfig:
    """Configuration for the pre-flight GPU residency sanity check.

    Loads the model with a minimal request, then asks Ollama's own /api/ps
    for size_vram vs size (bytes of the model actually resident in GPU
    VRAM) and aborts before the paid benchmark starts if too little of it
    is GPU-resident. This catches silent CPU fallback (e.g. a misconfigured
    or broken Ollama GPU backend on Cloud Run) before it burns a full run's
    worth of time and money. Only meaningful for Ollama-backed clients. When
    enabled, probe failures abort the run rather than being skipped.
    """

    enabled: bool = False
    min_vram_fraction: float = 0.0
    timeout_s: int = 210


@dataclass
class CloudRunCostConfig:
    """Configuration for the optional estimated Cloud Run cost printed at run end.

    This is an estimate derived from published on-demand instance-based-billing
    rates (CPU/memory/GPU per-second) multiplied by observed instance uptime,
    not the authoritative GCP invoice. See utils/cloudrun_cost.py.

    `usd_per_unit` is how many USD equal one unit of `currency` (e.g. 1.33 for GBP).
    `max_cost` is optional and expressed in display-currency units.
    """

    enabled: bool = False
    gpu_type: Optional[str] = "nvidia-l4"
    gpu_zonal_redundancy: bool = False
    cpu: float = 8
    memory_gib: float = 32
    currency: str = "GBP"
    usd_per_unit: float = 1.33
    progress_every: int = 5
    max_cost: Optional[float] = None


@dataclass
class LangfuseConfig:
    """Configuration for Langfuse observability."""

    enabled: bool = False
    secret_key: Optional[str] = None
    public_key: Optional[str] = None
    host: str = "http://localhost:3000"


@dataclass
class BenchmarkConfig:
    """Main benchmark configuration."""

    provider: ProviderConfig
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    keepalive: KeepaliveConfig = field(default_factory=KeepaliveConfig)
    keepalive_in_yaml: bool = False
    gpu_check: GpuCheckConfig = field(default_factory=GpuCheckConfig)
    cloudrun_cost: CloudRunCostConfig = field(default_factory=CloudRunCostConfig)
    langfuse: LangfuseConfig = field(default_factory=LangfuseConfig)
    questions_file: str = DEFAULT_QUESTIONS_FILE
    answers_file: str = DEFAULT_ANSWERS_FILE
    rate_limit_delay: float = 1.5
    max_tokens: int = 3072
    temperature: float = 0.2
    concurrency: int = 1
    request_log: Optional[str] = None


# Default providers configuration
DEFAULT_PROVIDERS = {
    "lmstudio": ProviderConfig(
        name="lmstudio",
        endpoint="http://localhost:1234",
    ),
    "ollama": ProviderConfig(
        name="ollama",
        endpoint="http://localhost:11434",
    ),
    "openwebui": ProviderConfig(
        name="openwebui",
        endpoint="http://localhost:3000",
        api_key_env="OPENWEBUI_API_KEY",
    ),
    "openrouter": ProviderConfig(
        name="openrouter",
        endpoint="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        default_model="anthropic/claude-4.5-haiku",
    ),
}


def _dict_to_provider_config(data: Dict[str, Any]) -> ProviderConfig:
    """Convert dict to ProviderConfig."""
    return ProviderConfig(
        name=data.get("name", "unknown"),
        endpoint=data.get("endpoint", ""),
        api_key=data.get("api_key"),
        api_key_env=data.get("api_key_env"),
        default_model=data.get("default_model"),
        timeout=data.get("timeout", 120),
        keep_alive=data.get("keep_alive"),
        auth=data.get("auth"),
        cloudrun_audience=data.get("cloudrun_audience"),
        cloudrun_impersonate_service_account=data.get("cloudrun_impersonate_service_account"),
    )


def _dict_to_keepalive_config(data: Dict[str, Any]) -> KeepaliveConfig:
    """Convert dict to KeepaliveConfig."""
    return KeepaliveConfig(
        enabled=data.get("enabled", True),
        interval_s=data.get("interval_s", 60),
        max_tokens=data.get("max_tokens", 16),
        prompt=data.get("prompt", "Say OK"),
        timeout_s=data.get("timeout_s", 90),
    )


def _dict_to_gpu_check_config(data: Dict[str, Any]) -> GpuCheckConfig:
    """Convert dict to GpuCheckConfig."""
    return GpuCheckConfig(
        enabled=data.get("enabled", False),
        min_vram_fraction=data.get("min_vram_fraction", 0.0),
        timeout_s=data.get("timeout_s", 210),
    )


def _dict_to_cloudrun_cost_config(data: Dict[str, Any]) -> CloudRunCostConfig:
    """Convert dict to CloudRunCostConfig."""
    max_cost = data.get("max_cost")
    return CloudRunCostConfig(
        enabled=data.get("enabled", False),
        gpu_type=data.get("gpu_type", "nvidia-l4"),
        gpu_zonal_redundancy=data.get("gpu_zonal_redundancy", False),
        cpu=data.get("cpu", 8),
        memory_gib=data.get("memory_gib", 32),
        currency=str(data.get("currency", "GBP")).upper(),
        usd_per_unit=float(data.get("usd_per_unit", 1.33)),
        progress_every=int(data.get("progress_every", 5)),
        max_cost=float(max_cost) if max_cost is not None else None,
    )


def _dict_to_scoring_config(data: Dict[str, Any]) -> ScoringConfig:
    """Convert dict to ScoringConfig."""
    return ScoringConfig(
        method=data.get("method", DEFAULT_SCORER),
        semantic=_dict_to_semantic_config(data.get("semantic", {})),
    )


def _dict_to_semantic_config(data: Dict[str, Any]) -> SemanticConfig:
    """Convert nested semantic scoring config."""
    provider = str(data.get("provider", DEFAULT_SEMANTIC_PROVIDER)).lower()
    model = data.get("model", DEFAULT_SEMANTIC_MODEL)
    if provider == "deepinfra" and model == DEFAULT_SEMANTIC_MODEL:
        model = DEFAULT_DEEPINFRA_SEMANTIC_MODEL
    thresholds_explicit = "thresholds" in data
    max_seq_length_explicit = "max_seq_length" in data
    raw_thresholds = data.get(
        "thresholds",
        default_semantic_thresholds(provider=provider, model_name=model),
    )
    thresholds = {
        int(score): float(threshold)
        for score, threshold in dict(raw_thresholds).items()
    }
    return SemanticConfig(
        enabled=bool(data.get("enabled", False)),
        provider=provider,
        answers_file=data.get("answers_file", DEFAULT_SEMANTIC_ANSWERS_FILE),
        model=model,
        endpoint=data.get("endpoint"),
        api_key=data.get("api_key"),
        api_key_env=data.get("api_key_env", "DEEPINFRA_TOKEN"),
        thresholds=thresholds,
        thresholds_explicit=thresholds_explicit,
        device=data.get("device", "auto"),
        max_seq_length=int(
            data.get(
                "max_seq_length",
                default_semantic_max_seq_length(provider=provider),
            )
        ),
        max_seq_length_explicit=max_seq_length_explicit,
    )


def _dict_to_export_config(data: Dict[str, Any]) -> ExportConfig:
    """Convert dict to ExportConfig."""
    return ExportConfig(
        formats=[fmt.lower() for fmt in data.get("formats", ["json"])],
        output_dir=data.get("output_dir", "."),
        include_response=data.get("include_response", True),
    )


def _dict_to_optimization_config(data: Dict[str, Any]) -> OptimizationConfig:
    """Convert dict to OptimizationConfig."""
    optimizer_provider = data.get("optimizer_provider")
    optimization_max_tokens_explicit = "optimization_max_tokens" in data
    return OptimizationConfig(
        optimizer_provider=optimizer_provider,
        optimizer_model=data.get("optimizer_model"),
        optimizer_endpoint=data.get("optimizer_endpoint"),
        optimizer_api_key=data.get("optimizer_api_key"),
        max_iterations=data.get("max_iterations", 4),
        optimization_max_tokens=int(
            data.get(
                "optimization_max_tokens",
                default_optimization_max_tokens(optimizer_provider=optimizer_provider),
            )
        ),
        optimization_max_tokens_explicit=optimization_max_tokens_explicit,
        strategies=data.get("strategies", [
            "role_playing", "technical", "few_shot", "cve_framing"
        ]),
    )


def _dict_to_langfuse_config(data: Dict[str, Any]) -> LangfuseConfig:
    """Convert dict to LangfuseConfig. Auto-enables if keys are present."""
    secret_key = data.get("secret_key")
    public_key = data.get("public_key")
    # Auto-enable if both keys are present
    auto_enabled = bool(secret_key and public_key)
    return LangfuseConfig(
        enabled=data.get("enabled", auto_enabled),
        secret_key=secret_key,
        public_key=public_key,
        host=data.get("host", "http://localhost:3000"),
    )


def load_config(config_path: str) -> BenchmarkConfig:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        BenchmarkConfig instance
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not data:
        raise ValueError(f"Empty config file: {config_path}")

    # Parse provider config
    provider_data = data.get("provider", {})
    provider_name = provider_data.get("name", "ollama")

    # Start with defaults if available
    if provider_name in DEFAULT_PROVIDERS:
        default_provider = DEFAULT_PROVIDERS[provider_name]
        provider = ProviderConfig(
            name=provider_name,
            endpoint=provider_data.get("endpoint", default_provider.endpoint),
            api_key=provider_data.get("api_key", default_provider.api_key),
            api_key_env=provider_data.get("api_key_env", default_provider.api_key_env),
            default_model=provider_data.get(
                "default_model", default_provider.default_model
            ),
            timeout=provider_data.get("timeout", default_provider.timeout),
            keep_alive=provider_data.get("keep_alive", default_provider.keep_alive),
            auth=provider_data.get("auth"),
            cloudrun_audience=provider_data.get("cloudrun_audience"),
            cloudrun_impersonate_service_account=provider_data.get(
                "cloudrun_impersonate_service_account"
            ),
        )
    else:
        provider = _dict_to_provider_config(provider_data)

    # Parse other configs
    scoring = _dict_to_scoring_config(data.get("scoring", {}))
    export = _dict_to_export_config(data.get("export", {}))
    optimization = _dict_to_optimization_config(data.get("optimization", {}))
    langfuse = _dict_to_langfuse_config(data.get("langfuse", {}))
    keepalive_in_yaml = "keepalive" in data
    keepalive = _dict_to_keepalive_config(data.get("keepalive", {}))
    gpu_check = _dict_to_gpu_check_config(data.get("gpu_check", {}))
    cloudrun_cost = _dict_to_cloudrun_cost_config(data.get("cloudrun_cost", {}))

    config = BenchmarkConfig(
        provider=provider,
        scoring=scoring,
        export=export,
        optimization=optimization,
        keepalive=keepalive,
        keepalive_in_yaml=keepalive_in_yaml,
        gpu_check=gpu_check,
        cloudrun_cost=cloudrun_cost,
        langfuse=langfuse,
        questions_file=data.get("questions_file", DEFAULT_QUESTIONS_FILE),
        answers_file=data.get("answers_file", DEFAULT_ANSWERS_FILE),
        rate_limit_delay=data.get("rate_limit_delay", 1.5),
        max_tokens=data.get("max_tokens", 3072),
        temperature=data.get("temperature", 0.2),
        concurrency=data.get("concurrency", 1),
        request_log=data.get("request_log"),
    )
    validate_config(config)
    return config


def validate_config(config: BenchmarkConfig) -> None:
    """Validate config values that affect runtime behavior."""
    if config.rate_limit_delay < 0:
        raise ValueError("rate_limit_delay must be >= 0")
    if config.max_tokens <= 0:
        raise ValueError("max_tokens must be > 0")
    if config.temperature < 0:
        raise ValueError("temperature must be >= 0")
    if config.concurrency <= 0:
        raise ValueError("concurrency must be > 0")
    if config.provider.timeout <= 0:
        raise ValueError("provider.timeout must be > 0")

    if config.scoring.method != DEFAULT_SCORER:
        raise ValueError(f"Unsupported scoring method: {config.scoring.method}")
    for score, threshold in config.scoring.semantic.thresholds.items():
        if not 0 <= score <= 100:
            raise ValueError("scoring.semantic.thresholds scores must be between 0 and 100")
        if not 0 <= threshold <= 1:
            raise ValueError(
                "scoring.semantic.thresholds similarity values must be between 0 and 1"
            )
    ordered_thresholds = [
        threshold
        for _, threshold in sorted(
            config.scoring.semantic.thresholds.items(), reverse=True
        )
    ]
    if ordered_thresholds != sorted(ordered_thresholds, reverse=True):
        raise ValueError("scoring.semantic.thresholds must decrease as scores decrease")
    if config.scoring.semantic.provider not in {"local", "deepinfra"}:
        raise ValueError(
            f"Unsupported scoring.semantic.provider: {config.scoring.semantic.provider}"
        )

    unsupported_formats = set(config.export.formats) - {"json", "csv", "criteria_csv"}
    if unsupported_formats:
        formatted = ", ".join(sorted(unsupported_formats))
        raise ValueError(f"Unsupported export format(s): {formatted}")

    if config.provider.auth == "cloudrun_identity" and not config.provider.endpoint:
        raise ValueError("provider.endpoint is required when auth is cloudrun_identity")
    if config.provider.auth and config.provider.auth != "cloudrun_identity":
        raise ValueError(f"Unsupported provider.auth: {config.provider.auth}")

    if config.keepalive.interval_s <= 0:
        raise ValueError("keepalive.interval_s must be > 0")
    if config.keepalive.max_tokens <= 0:
        raise ValueError("keepalive.max_tokens must be > 0")
    if config.keepalive.timeout_s <= 0:
        raise ValueError("keepalive.timeout_s must be > 0")

    if not 0 <= config.gpu_check.min_vram_fraction <= 1:
        raise ValueError("gpu_check.min_vram_fraction must be between 0 and 1")
    if config.gpu_check.timeout_s <= 0:
        raise ValueError("gpu_check.timeout_s must be > 0")

    if config.cloudrun_cost.usd_per_unit <= 0:
        raise ValueError("cloudrun_cost.usd_per_unit must be > 0")
    if config.cloudrun_cost.progress_every < 0:
        raise ValueError("cloudrun_cost.progress_every must be >= 0")
    if config.cloudrun_cost.max_cost is not None and config.cloudrun_cost.max_cost <= 0:
        raise ValueError("cloudrun_cost.max_cost must be > 0 when set")

    if config.cloudrun_cost.enabled:
        if config.cloudrun_cost.cpu <= 0:
            raise ValueError("cloudrun_cost.cpu must be > 0")
        if config.cloudrun_cost.memory_gib <= 0:
            raise ValueError("cloudrun_cost.memory_gib must be > 0")
        if config.cloudrun_cost.gpu_type:
            from utils.cloudrun_cost import gpu_per_second_rate

            gpu_per_second_rate(
                config.cloudrun_cost.gpu_type,
                config.cloudrun_cost.gpu_zonal_redundancy,
            )

    opt = config.optimization
    if bool(opt.optimizer_provider) != bool(opt.optimizer_model):
        raise ValueError(
            "optimization.optimizer_provider and optimization.optimizer_model "
            "must be set together"
        )
    if opt.optimization_max_tokens <= 0:
        raise ValueError("optimization.optimization_max_tokens must be > 0")


def create_default_config(
    provider: str = "ollama",
    model: Optional[str] = None,
) -> BenchmarkConfig:
    """
    Create a default configuration.

    Args:
        provider: Provider name
        model: Model name

    Returns:
        BenchmarkConfig with defaults
    """
    provider_config = DEFAULT_PROVIDERS.get(
        provider,
        ProviderConfig(name=provider, endpoint="http://localhost:11434"),
    )

    if model:
        provider_config = ProviderConfig(
            name=provider_config.name,
            endpoint=provider_config.endpoint,
            api_key_env=provider_config.api_key_env,
            default_model=model,
            timeout=provider_config.timeout,
            keep_alive=provider_config.keep_alive,
        )

    return BenchmarkConfig(provider=provider_config)


def save_config(config: BenchmarkConfig, config_path: str) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: BenchmarkConfig instance
        config_path: Path to save YAML file
    """
    data = {
        "provider": {
            "name": config.provider.name,
            "endpoint": config.provider.endpoint,
        },
        "scoring": {
            "method": config.scoring.method,
            "semantic": {
                "enabled": config.scoring.semantic.enabled,
                "provider": config.scoring.semantic.provider,
                "answers_file": config.scoring.semantic.answers_file,
                "model": config.scoring.semantic.model,
                "endpoint": config.scoring.semantic.endpoint,
                "api_key_env": config.scoring.semantic.api_key_env,
                "thresholds": config.scoring.semantic.thresholds,
                "device": config.scoring.semantic.device,
                "max_seq_length": config.scoring.semantic.max_seq_length,
            },
        },
        "export": {
            "formats": config.export.formats,
            "output_dir": config.export.output_dir,
        },
        "optimization": {
            "max_iterations": config.optimization.max_iterations,
            "optimization_max_tokens": config.optimization.optimization_max_tokens,
        },
        "questions_file": config.questions_file,
        "answers_file": config.answers_file,
        "rate_limit_delay": config.rate_limit_delay,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "concurrency": config.concurrency,
        "request_log": config.request_log,
    }

    # Add optional fields
    if config.provider.api_key_env:
        data["provider"]["api_key_env"] = config.provider.api_key_env
    if config.provider.default_model:
        data["provider"]["default_model"] = config.provider.default_model
    if config.provider.keep_alive:
        data["provider"]["keep_alive"] = config.provider.keep_alive
    if config.optimization.optimizer_provider:
        data["optimization"]["optimizer_provider"] = config.optimization.optimizer_provider
    if config.optimization.optimizer_model:
        data["optimization"]["optimizer_model"] = config.optimization.optimizer_model
    if config.optimization.optimizer_endpoint:
        data["optimization"]["optimizer_endpoint"] = config.optimization.optimizer_endpoint

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def get_api_key(provider_config: ProviderConfig) -> Optional[str]:
    """
    Get API key from environment variable.

    Args:
        provider_config: Provider configuration

    Returns:
        API key or None
    """
    if provider_config.api_key_env:
        return os.environ.get(provider_config.api_key_env)
    return None
