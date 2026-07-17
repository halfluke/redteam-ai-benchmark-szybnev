"""CLI entrypoint for Red Team AI Benchmark."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import time
from contextlib import nullcontext
from typing import Dict, List, Optional

from pick import pick

import tracing.langfuse as langfuse_module
from benchmark.metrics import (
    build_track_results,
    diverged_marker,
    weighted_primary_score,
    weighted_semantic_score,
)
from benchmark import (
    BenchmarkDataset,
    GracefulShutdown,
    QuestionLoadError,
    RuntimeOptions,
    _effective_concurrency,
    _make_result,
    _query_and_score,
    _run_questions_concurrent,
    _run_questions_sequential,
    _sleep_between_requests,
    install_signal_handlers,
    load_dataset,
    load_questions,
    run_single_model_benchmark,
)
from benchmark.offline_judge import add_judge_args
from benchmark.offline_judge import run as run_offline_judge
from benchmark.types import (
    DEFAULT_CONCURRENCY,
    DEFAULT_MAX_TOKENS,
    DEFAULT_RATE_LIMIT_DELAY,
    DEFAULT_TEMPERATURE,
)
from benchmark.gpu_check import GpuCheckFailed, run_gpu_check
from benchmark.keepalive import ModelKeepalive
from models import (
    ModelListError,
    ModelNotFoundError,
    create_client,
    provider_auth_kwargs,
    validate_model_available,
)
from optimization import PromptOptimizer, save_optimization_results
from scoring import create_scorer, create_semantic_scorer
from scoring.preload import preload_semantic_scorer
from scoring.semantic_calibration import default_semantic_thresholds
from scoring.semantic_embedder import DEFAULT_DEEPINFRA_SEMANTIC_MODEL
from scoring.semantic_scorer import (
    DEFAULT_SEMANTIC_ANSWERS_FILE,
    DEFAULT_SEMANTIC_MODEL,
    DEFAULT_SEMANTIC_PROVIDER,
    default_semantic_max_seq_length,
)
from scoring.refusal import is_censored_response
from tracing import LANGFUSE_AVAILABLE
from utils import load_config
from utils.cloudrun_cost import estimate_cost, format_cost_estimate
from utils.config import (
    DEFAULT_QUESTIONS_FILE,
    DEFAULT_SCORER,
    GpuCheckConfig,
    KeepaliveConfig,
    OptimizationConfig,
    default_optimization_max_tokens,
)
from utils.export import BenchmarkExporter, get_interpretation

Langfuse = langfuse_module.Langfuse
BENCHMARK_VERSION = "2.3.0"
DEFAULT_PROFILE = "standard"
PROFILE_DEFAULTS = {
    "quick": {"questions_file": DEFAULT_QUESTIONS_FILE},
    "standard": {"questions_file": DEFAULT_QUESTIONS_FILE},
    "enterprise": {"questions_file": DEFAULT_QUESTIONS_FILE},
    "local-only": {"questions_file": DEFAULT_QUESTIONS_FILE},
    "cloud-comparison": {"questions_file": DEFAULT_QUESTIONS_FILE},
}


class LangfuseTracer(langfuse_module.LangfuseTracer):
    """Compatibility wrapper that preserves run_benchmark.Langfuse monkeypatching."""

    def __init__(self, config):
        langfuse_module.Langfuse = Langfuse
        super().__init__(config)

__all__ = [
    "DEFAULT_CONCURRENCY",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_RATE_LIMIT_DELAY",
    "DEFAULT_TEMPERATURE",
    "BenchmarkDataset",
    "LANGFUSE_AVAILABLE",
    "GracefulShutdown",
    "Langfuse",
    "LangfuseTracer",
    "PromptOptimizer",
    "RuntimeOptions",
    "_effective_concurrency",
    "_make_result",
    "_query_and_score",
    "_run_questions_concurrent",
    "_run_questions_sequential",
    "_sleep_between_requests",
    "cmd_interactive",
    "cmd_judge",
    "cmd_list_models",
    "cmd_run_benchmark",
    "is_censored_response",
    "install_signal_handlers",
    "load_dataset",
    "load_questions",
    "main",
    "parse_reference_answers",
    "time",
]


def _load_optional_config(args):
    """Load YAML config once at command start.

    A missing or invalid ``--config`` aborts the run so Cloud Run auth,
    GPU checks, and other YAML settings cannot silently fall back to defaults.
    """
    if not getattr(args, "config", None):
        return None

    try:
        config = load_config(args.config)
        print(f"📄 Loaded configuration from {args.config}")
        return config
    except Exception as e:
        print(f"❌ Error: Failed to load config: {e}")
        sys.exit(1)


def _apply_config_defaults(args, config) -> None:
    """Apply config values when the corresponding CLI option was not explicit."""
    if not config:
        return

    if config.provider.endpoint and not args.endpoint:
        args.endpoint = config.provider.endpoint

    if not getattr(args, "api_key", None):
        if config.provider.api_key:
            args.api_key = config.provider.api_key
        elif config.provider.api_key_env:
            args.api_key = os.environ.get(config.provider.api_key_env)

    if hasattr(args, "optimizer_provider") and config.optimization.optimizer_provider and not args.optimizer_provider:
        args.optimizer_provider = config.optimization.optimizer_provider
    if hasattr(args, "optimizer_model") and config.optimization.optimizer_model and not args.optimizer_model:
        args.optimizer_model = config.optimization.optimizer_model
    if hasattr(args, "optimizer_api_key") and config.optimization.optimizer_api_key and not args.optimizer_api_key:
        args.optimizer_api_key = config.optimization.optimizer_api_key
    if (
        hasattr(args, "optimizer_endpoint")
        and config.optimization.optimizer_endpoint
        and not args.optimizer_endpoint
    ):
        args.optimizer_endpoint = config.optimization.optimizer_endpoint
    if (
        hasattr(args, "max_optimization_iterations")
        and args.max_optimization_iterations is None
        and config
    ):
        args.max_optimization_iterations = config.optimization.max_iterations
    if hasattr(args, "semantic"):
        semantic_config = getattr(config.scoring, "semantic", None)
        if semantic_config:
            if semantic_config.enabled and not args.semantic:
                args.semantic = True
            if not getattr(args, "semantic_provider", None):
                args.semantic_provider = semantic_config.provider
            if not getattr(args, "semantic_model", None):
                args.semantic_model = semantic_config.model
            if not getattr(args, "semantic_answers", None):
                args.semantic_answers = semantic_config.answers_file
            if (
                not getattr(args, "semantic_api_key", None)
                and semantic_config.api_key
            ):
                args.semantic_api_key = semantic_config.api_key


def _questions_file_for_args(args, config) -> str:
    """Resolve questions file through config or runtime profile."""
    if config:
        return config.questions_file
    profile = getattr(args, "profile", DEFAULT_PROFILE)
    return PROFILE_DEFAULTS.get(profile, PROFILE_DEFAULTS[DEFAULT_PROFILE])["questions_file"]


def _resolve_runtime_options(args, config) -> RuntimeOptions:
    """Resolve CLI > config > default runtime settings."""
    options = RuntimeOptions(
        rate_limit_delay=(
            args.rate_limit_delay
            if args.rate_limit_delay is not None
            else config.rate_limit_delay
            if config
            else DEFAULT_RATE_LIMIT_DELAY
        ),
        max_tokens=(
            args.max_tokens
            if args.max_tokens is not None
            else config.max_tokens
            if config
            else DEFAULT_MAX_TOKENS
        ),
        temperature=(
            args.temperature
            if args.temperature is not None
            else config.temperature
            if config
            else DEFAULT_TEMPERATURE
        ),
        concurrency=(
            args.concurrency
            if args.concurrency is not None
            else config.concurrency
            if config
            else DEFAULT_CONCURRENCY
        ),
        request_log=(
            args.request_log
            if getattr(args, "request_log", None)
            else getattr(config, "request_log", None)
            if config
            else None
        ),
    )
    _validate_runtime_options(options)
    return options


def _validate_runtime_options(options: RuntimeOptions) -> None:
    """Validate runtime options used by CLI commands."""
    if options.rate_limit_delay < 0:
        raise ValueError("--rate-limit-delay must be >= 0")
    if options.max_tokens <= 0:
        raise ValueError("--max-tokens must be > 0")
    if options.temperature < 0:
        raise ValueError("--temperature must be >= 0")
    if options.concurrency <= 0:
        raise ValueError("--concurrency must be > 0")


def _create_scorer_bundle(args, config, questions: List[Dict]):
    """Create the configured scorer bundle or exit with a clear CLI error."""
    try:
        return create_scorer(
            DEFAULT_SCORER,
            questions=questions,
        )
    except (RuntimeError, ValueError) as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


def _create_semantic_scorer_bundle(args, config, questions: List[Dict]):
    """Create the optional parallel semantic scorer bundle."""
    if not getattr(args, "semantic", False):
        return None

    semantic_config = getattr(getattr(config, "scoring", None), "semantic", None)
    answers_file = (
        getattr(args, "semantic_answers", None)
        or getattr(semantic_config, "answers_file", None)
        or DEFAULT_SEMANTIC_ANSWERS_FILE
    )
    provider = (
        getattr(args, "semantic_provider", None)
        or getattr(semantic_config, "provider", None)
        or DEFAULT_SEMANTIC_PROVIDER
    ).lower()
    model_name = (
        getattr(args, "semantic_model", None)
        or getattr(semantic_config, "model", None)
        or DEFAULT_SEMANTIC_MODEL
    )
    if provider == "deepinfra" and model_name == DEFAULT_SEMANTIC_MODEL:
        model_name = DEFAULT_DEEPINFRA_SEMANTIC_MODEL
    if semantic_config and semantic_config.thresholds_explicit:
        thresholds = semantic_config.thresholds
    else:
        thresholds = default_semantic_thresholds(
            provider=provider,
            model_name=model_name,
        )
    device = getattr(semantic_config, "device", "auto")
    if semantic_config and semantic_config.max_seq_length_explicit:
        max_seq_length = semantic_config.max_seq_length
    else:
        max_seq_length = default_semantic_max_seq_length(provider=provider)
    endpoint = getattr(semantic_config, "endpoint", None) if semantic_config else None
    api_key = (
        getattr(args, "semantic_api_key", None)
        or (getattr(semantic_config, "api_key", None) if semantic_config else None)
    )
    api_key_env = getattr(semantic_config, "api_key_env", "DEEPINFRA_TOKEN") if semantic_config else "DEEPINFRA_TOKEN"
    if provider == "deepinfra" and not api_key:
        api_key = os.environ.get(api_key_env)

    try:
        bundle = create_semantic_scorer(
            questions=questions,
            answers_file=answers_file,
            model_name=model_name,
            provider=provider,
            thresholds=thresholds,
            device=device,
            max_seq_length=max_seq_length,
            endpoint=endpoint,
            api_key=api_key,
            api_key_env=api_key_env,
        )
        bundle.scorer.warm_encoder()
        print(
            f"✓ Using parallel semantic scoring "
            f"(provider: {provider}, model: {model_name}, answers: {answers_file})\n"
        )
        return bundle
    except (RuntimeError, ValueError) as e:
        print(f"❌ Error initializing semantic scorer: {e}")
        sys.exit(1)


def parse_reference_answers(filepath: str = "answers_all.txt") -> Dict[int, str]:
    """Load legacy reference answers for prompt optimization context."""
    from scoring.semantic_scorer import parse_answers_sections

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        return {}

    return parse_answers_sections(
        content,
        source=filepath,
        require_nonempty=False,
    )


def _load_dataset_for_cli(filepath: str) -> BenchmarkDataset:
    """Load a dataset and convert loader errors into CLI exits."""
    try:
        return load_dataset(filepath)
    except QuestionLoadError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


def _filter_questions_by_profile(
    questions: List[Dict],
    profile: str,
) -> List[Dict]:
    """Filter v2 questions by runtime profile."""
    filtered = [
        question
        for question in questions
        if profile in question.get("profiles", [DEFAULT_PROFILE, "enterprise"])
    ]
    return filtered or questions


def _parse_question_ids(raw_ids: List[str] | None) -> List[int]:
    """Parse CLI question ids from repeated args and comma-separated chunks."""
    if not raw_ids:
        return []
    parsed = []
    for raw in raw_ids:
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                parsed.append(int(part))
            except ValueError as e:
                raise ValueError(f"Invalid question id: {part}") from e
    return parsed


def _filter_questions_by_ids(
    questions: List[Dict],
    question_ids: List[int],
) -> List[Dict]:
    """Filter by ids while preserving benchmark order."""
    if not question_ids:
        return questions

    available = {question["id"] for question in questions}
    missing = sorted(set(question_ids) - available)
    if missing:
        formatted = ", ".join(str(q_id) for q_id in missing)
        raise ValueError(f"Unknown question id(s) for selected profile: {formatted}")

    selected = set(question_ids)
    return [question for question in questions if question["id"] in selected]


def _select_questions_for_args(dataset: BenchmarkDataset, args) -> List[Dict]:
    """Apply profile and optional id filtering."""
    profile = getattr(args, "profile", DEFAULT_PROFILE)
    questions = _filter_questions_by_profile(dataset.questions, profile)
    question_ids = _parse_question_ids(getattr(args, "question_ids", None))
    return _filter_questions_by_ids(questions, question_ids)


def _ollama_keep_alive(provider, args=None, config=None) -> str | None:
    """Resolve optional Ollama keep_alive without affecting other providers."""
    if provider != "ollama":
        return None
    if args and getattr(args, "ollama_keep_alive", None):
        return args.ollama_keep_alive
    if config and getattr(config.provider, "keep_alive", None):
        return config.provider.keep_alive
    return os.environ.get("OLLAMA_KEEP_ALIVE")


def _create_configured_client(provider, endpoint, model_name, api_key, config, args=None):
    """Create a provider client, applying config timeout and Cloud Run auth when present."""
    timeout = config.provider.timeout if config else None
    keep_alive = _ollama_keep_alive(provider, args, config)
    extra_kwargs: dict = {}
    if keep_alive is not None:
        extra_kwargs["keep_alive"] = keep_alive

    if config and config.provider.auth:
        auth_kwargs = provider_auth_kwargs(
            auth=config.provider.auth,
            endpoint=endpoint,
            cloudrun_audience=config.provider.cloudrun_audience,
            cloudrun_impersonate_service_account=(
                config.provider.cloudrun_impersonate_service_account
            ),
        )
        if auth_kwargs:
            if provider == "openwebui":
                raise SystemExit(
                    "ERROR: provider 'openwebui' does not support "
                    f"auth={config.provider.auth!r} (e.g. cloudrun_identity). "
                    "Use ollama or lmstudio for Cloud Run identity tokens, "
                    "or remove provider.auth from the config."
                )
            api_key = None
            extra_kwargs.update(auth_kwargs)

    if timeout is not None:
        extra_kwargs["timeout"] = timeout

    return create_client(provider, endpoint, model_name, api_key, **extra_kwargs)


def _resolve_keepalive_config(config) -> KeepaliveConfig:
    """Return keepalive settings from config or defaults."""
    if config and getattr(config, "keepalive", None):
        return config.keepalive
    return KeepaliveConfig()


def _resolve_gpu_check_config(args, config) -> GpuCheckConfig:
    """Resolve CLI > config > default GPU residency check settings."""
    base = config.gpu_check if config and getattr(config, "gpu_check", None) else GpuCheckConfig()
    cli_min_fraction = getattr(args, "min_vram_fraction", None)
    if cli_min_fraction is None:
        return base
    return GpuCheckConfig(
        enabled=cli_min_fraction > 0,
        min_vram_fraction=cli_min_fraction,
        timeout_s=base.timeout_s,
    )


def _run_gpu_check_or_exit(client, model_name: str, args, config) -> None:
    """Run the pre-flight GPU residency check, printing status and raising on failure.

    Must only ever be called with the target model client, never an
    optimizer's client: the optimizer commonly runs against a separate
    (often local, non-Cloud-Run) endpoint with its own hardware profile, and
    should never be blocked by a GPU expectation that applies to the paid
    Cloud Run target.
    """
    gpu_check_cfg = _resolve_gpu_check_config(args, config)
    if not gpu_check_cfg.enabled or gpu_check_cfg.min_vram_fraction <= 0:
        return
    print(
        f"   Checking GPU residency for {model_name} "
        f"(min {gpu_check_cfg.min_vram_fraction:.0%} in VRAM)...",
        end=" ",
        flush=True,
    )
    fraction = run_gpu_check(
        client,
        min_vram_fraction=gpu_check_cfg.min_vram_fraction,
        timeout_s=gpu_check_cfg.timeout_s,
    )
    print(f"ok ({fraction:.0%} in VRAM)")


def _should_run_keepalive(config, optimizer) -> bool:
    """Whether background keepalive should run for this benchmark."""
    keepalive_cfg = _resolve_keepalive_config(config)
    is_cloud_run = bool(config and config.provider.auth == "cloudrun_identity")
    keepalive_in_yaml = bool(config and getattr(config, "keepalive_in_yaml", False))

    if keepalive_in_yaml and not keepalive_cfg.enabled:
        return False
    if is_cloud_run:
        return True
    if optimizer is not None:
        return True
    return keepalive_in_yaml and keepalive_cfg.enabled


def _create_keepalive(client, optimizer, config) -> ModelKeepalive | None:
    """Create keepalive for Cloud Run target and optional local optimizer."""
    if not _should_run_keepalive(config, optimizer):
        return None

    keepalive_cfg = _resolve_keepalive_config(config)

    def _on_ping(role: str, ok: bool) -> None:
        if not ok:
            print(f"   Keepalive ping ({role}): failed", flush=True)

    endpoints = [("target", client)]
    if optimizer is not None:
        endpoints.append(("optimizer", optimizer.optimizer_client))

    return ModelKeepalive(
        endpoints,
        interval_s=keepalive_cfg.interval_s,
        prompt=keepalive_cfg.prompt,
        max_tokens=keepalive_cfg.max_tokens,
        timeout_s=keepalive_cfg.timeout_s,
        on_ping=_on_ping,
    )


def _package_version() -> str:
    try:
        return importlib.metadata.version("redteam-ai-benchmark")
    except importlib.metadata.PackageNotFoundError:
        return BENCHMARK_VERSION


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def _stable_hash(payload: Dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _export_run_config(
    *,
    args,
    model_name: str | None,
    dataset: BenchmarkDataset | None,
    runtime: RuntimeOptions | None,
) -> Dict:
    """Build stable, non-secret run parameters for JSON exports."""
    return {
        "provider": getattr(args, "provider", None),
        "endpoint": getattr(args, "endpoint", None),
        "model": model_name,
        "profile": getattr(args, "profile", DEFAULT_PROFILE),
        "questions_file": dataset.path if dataset else None,
        "output": getattr(args, "output", None),
        "export_csv": getattr(args, "export_csv", None),
        "config_file": getattr(args, "config", None),
        "max_tokens": runtime.max_tokens if runtime else None,
        "temperature": runtime.temperature if runtime else None,
        "rate_limit_delay": runtime.rate_limit_delay if runtime else None,
        "concurrency": runtime.concurrency if runtime else None,
        "request_log": runtime.request_log if runtime else None,
        "question_ids": getattr(args, "question_ids", None),
        "no_optimize": getattr(args, "no_optimize", None),
        "optimizer_provider": getattr(args, "optimizer_provider", None),
        "optimizer_model": getattr(args, "optimizer_model", None),
        "optimizer_api_key": "set" if getattr(args, "optimizer_api_key", None) else None,
        "optimizer_endpoint": getattr(args, "optimizer_endpoint", None),
        "max_optimization_iterations": getattr(args, "max_optimization_iterations", None),
        "semantic": getattr(args, "semantic", None),
        "semantic_model": getattr(args, "semantic_model", None),
        "semantic_answers": getattr(args, "semantic_answers", None),
    }


def _export_metadata(
    *,
    args,
    model_name: str | None,
    config,
    dataset: BenchmarkDataset | None,
    runtime: RuntimeOptions | None,
    scoring_method: str,
    extra_metadata: Dict | None = None,
) -> Dict:
    """Build top-level audit provenance for exported benchmark results."""
    dataset_metadata = dataset.metadata if dataset else {}
    run_config = _export_run_config(
        args=args,
        model_name=model_name,
        dataset=dataset,
        runtime=runtime,
    )
    config_payload = {
        "run_config": run_config,
        "scorer": scoring_method,
    }
    metadata = {
        "benchmark_version": dataset_metadata.get("benchmark_version", BENCHMARK_VERSION),
        "dataset_id": dataset_metadata.get("dataset_id", "unknown"),
        "dataset_version": dataset_metadata.get("dataset_version", "1.0.0"),
        "dataset_hash": dataset.content_hash if dataset else None,
        "scorer_version": scoring_method,
        "config_hash": _stable_hash(config_payload),
        "run_config": run_config,
        "git_commit": _git_commit(),
        "package_version": _package_version(),
        "runtime_profile": getattr(args, "profile", DEFAULT_PROFILE),
    }
    if extra_metadata:
        metadata["metadata"] = extra_metadata
    return metadata


def _export_benchmark_results(
    results: List[Dict],
    model_name: str,
    total_score: float,
    interpretation: str,
    scoring_method: str,
    args,
    config,
    multi_model: bool = False,
    dataset: BenchmarkDataset | None = None,
    runtime: RuntimeOptions | None = None,
    summary: Dict | None = None,
    metadata: Dict | None = None,
) -> Dict[str, str]:
    """Export benchmark results according to CLI and config options."""
    export_config = config.export if config else None
    formats = list(export_config.formats if export_config else ["json"])

    if args.export_csv and "csv" not in formats:
        formats.append("csv")

    output_dir = export_config.output_dir if export_config else "."
    include_response = export_config.include_response if export_config else True

    filename = getattr(args, "output", None)
    if filename and multi_model:
        safe_model = BenchmarkExporter(model_name=model_name)._sanitize_filename(model_name)
        filename = f"{filename}_{safe_model}"

    exporter = BenchmarkExporter(output_dir=output_dir, model_name=model_name)
    exported = {}

    if "json" in formats:
        exported["json"] = exporter.export_json(
            results=results,
            total_score=total_score,
            interpretation=interpretation,
            scoring_method=scoring_method,
            summary=summary,
            metadata=_export_metadata(
                args=args,
                model_name=model_name,
                config=config,
                dataset=dataset,
                runtime=runtime,
                scoring_method=scoring_method,
                extra_metadata=metadata,
            ),
            filename=filename,
        )

    if "csv" in formats:
        exported["csv"] = exporter.export_csv(
            results=results,
            total_score=total_score,
            filename=filename,
            include_response=include_response,
        )

    if "criteria_csv" in formats:
        exported["criteria_csv"] = exporter.export_detailed_csv(
            results=results,
            filename=filename,
        )

    for path in exported.values():
        print(f"\n💾 Results saved to: {path}")

    return {fmt: str(path) for fmt, path in exported.items()}


def _abort_model_validation_error(exc: Exception) -> None:
    raise SystemExit(f"ERROR: {exc}") from exc


def _validate_model_available(client, model_name: str, *, provider: str, role: str) -> None:
    """Abort before benchmark questions if the model is not listed by the provider."""
    try:
        validate_model_available(
            client,
            model_name,
            provider=provider,
            role=role,
        )
    except (ModelListError, ModelNotFoundError) as e:
        _abort_model_validation_error(e)


_CLOUD_OPTIMIZER_PROVIDERS = {"deepinfra", "openrouter"}
_CLOUD_OPTIMIZER_ENV_KEYS = {
    "deepinfra": "DEEPINFRA_TOKEN",
    "openrouter": "OPENROUTER_API_KEY",
}


def _resolve_optimizer_endpoint(
    provider: str,
    args,
    target_endpoint: str,
    config,
) -> str | None:
    """Resolve optimizer endpoint.

    Explicit CLI/config wins. Cloud providers and local providers both default
    to ``None`` (provider localhost / public API defaults) so a Cloud Run
    target URL is never inherited by a local optimizer by accident.
    """
    del target_endpoint  # intentionally unused: never inherit the target URL
    explicit = getattr(args, "optimizer_endpoint", None)
    if not explicit and config:
        explicit = config.optimization.optimizer_endpoint
    if explicit:
        return explicit
    return None


def _initialize_optimizer(args, config, endpoint: str):
    """Create prompt optimizer if configured and not disabled."""
    if getattr(args, "no_optimize", False):
        return None

    provider = getattr(args, "optimizer_provider", None)
    model = getattr(args, "optimizer_model", None)
    api_key = getattr(args, "optimizer_api_key", None)

    if config:
        if not provider:
            provider = config.optimization.optimizer_provider
        if not model:
            model = config.optimization.optimizer_model
        if not api_key:
            api_key = config.optimization.optimizer_api_key

    if bool(provider) != bool(model):
        raise SystemExit(
            "ERROR: --optimizer-provider and --optimizer-model must be used together "
            f"(got provider={provider!r}, model={model!r}). "
            "Specify both, or neither (no optimization)."
        )

    if not provider and not model:
        return None

    if provider in _CLOUD_OPTIMIZER_PROVIDERS:
        env_var = _CLOUD_OPTIMIZER_ENV_KEYS[provider]
        effective_key = api_key or os.environ.get(env_var)
        if not effective_key:
            raise SystemExit(
                f"ERROR: optimizer provider '{provider}' requires an API key. "
                f"Pass --optimizer-api-key or set {env_var}."
            )
        api_key = effective_key

    optimizer_endpoint = _resolve_optimizer_endpoint(provider, args, endpoint, config)
    opt_cfg = config.optimization if config else OptimizationConfig()
    max_iterations = (
        args.max_optimization_iterations
        if getattr(args, "max_optimization_iterations", None) is not None
        else opt_cfg.max_iterations
    )
    if config and opt_cfg.optimization_max_tokens_explicit:
        optimization_max_tokens = opt_cfg.optimization_max_tokens
    else:
        optimization_max_tokens = default_optimization_max_tokens(
            optimizer_provider=provider,
        )
    try:
        optimizer = PromptOptimizer(
            optimizer_model=model,
            optimizer_provider=provider,
            optimizer_endpoint=optimizer_endpoint,
            optimizer_api_key=api_key,
            max_iterations=max_iterations,
            optimization_max_tokens=optimization_max_tokens,
        )
        _validate_model_available(
            optimizer.optimizer_client,
            model,
            provider=provider,
            role="optimizer",
        )
        print(f"✓ Prompt optimization enabled (optimizer: {provider}/{model})\n")
        return optimizer
    except Exception as e:
        print(f"❌ Error initializing optimizer: {e}")
        raise SystemExit(1) from e


def _langfuse_config_or_none(config):
    """Return active Langfuse config if tracing can be enabled."""
    if config and config.langfuse.enabled and LANGFUSE_AVAILABLE:
        print("✓ Langfuse tracing enabled\n")
        return config.langfuse
    return None


def _format_elapsed(seconds: float) -> str:
    """Format an elapsed duration in seconds as a short human-readable string."""
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _print_runtime(runtime: RuntimeOptions) -> None:
    print(
        f"Runtime: max_tokens={runtime.max_tokens}, "
        f"temperature={runtime.temperature}, "
        f"rate_limit_delay={runtime.rate_limit_delay}, "
        f"concurrency={runtime.concurrency}\n"
    )


def _interpretation_label(score: float) -> str:
    interp = get_interpretation(score)
    if interp == "strong-candidate":
        return "strong candidate — review breakdowns before production use"
    if interp == "requires-validation":
        return "requires RAG + manual validation before use"
    return "not suitable for offensive security tasks"


def _report_connection_failure(client, provider: str, *, is_remote: bool) -> None:
    """Print a connectivity failure with probe diagnostics when available."""
    print(f"❌ Cannot connect to {provider} at {client.base_url}")
    detail = getattr(client, "last_probe_error", None)
    if detail:
        print(f"   Reason: {detail}")
    if is_remote:
        print(
            "   Check: gcloud auth login, roles/run.invoker on the service, "
            "and that the URL is correct."
        )
    else:
        print(f"   Is {provider} running?")


def _winner_source_label(r: Dict) -> str:
    """Return a human-readable winner source for detail tables."""
    src = r.get("answer_source") or "baseline"
    if src == "optimized":
        strat = r.get("optimization_strategy") or r.get("strategy") or ""
        if strat and strat != "original":
            return f"optimization/{strat}"
        return "optimization"
    return "baseline"


def _winner_score_label(row: Dict, *, score_field: str) -> str:
    if score_field == "semantic_score":
        semantic_scores = row.get("semantic_scores") or {}
        if semantic_scores.get("skipped"):
            return "skipped"
    value = row.get(score_field)
    if isinstance(value, (int, float)):
        return f"{int(value)}%"
    return "—"


def _winner_question_label(row: Dict) -> str:
    """Return the prompt shown to the model for this track winner."""
    prompt = (row.get("prompt") or "").replace("\n", " ").strip()
    src = row.get("answer_source") or "baseline"
    if src == "optimized" and prompt:
        return f"(optimized) {prompt}"
    return prompt


def _print_winner_detail_table(
    rows: List[Dict],
    *,
    title: str,
    score_field: str,
    score_header: str,
    total_score: Optional[float] = None,
) -> None:
    """Print per-question winners with category, divergence, and prompt text."""
    if not rows:
        return

    print()
    print("=" * 100)
    print(title)
    print("=" * 100)
    print(
        f"{'Q#':<4} {'D':<2} {'Category':<22} {score_header:<10} "
        f"{'Source':<24} Question"
    )
    print("-" * 100)
    for row in sorted(rows, key=lambda item: item.get("id", 0)):
        q_id = row.get("id", "?")
        diverged = diverged_marker(row)
        category = row.get("category", "?")
        score_label = _winner_score_label(row, score_field=score_field)
        source = _winner_source_label(row)
        question = _winner_question_label(row)
        print(
            f"{q_id!s:<4} {diverged:<2} {category:<22} {score_label:<10} "
            f"{source:<24} {question}"
        )
    if total_score is not None:
        print("-" * 100)
        print(
            f"   Total: {total_score:.1f}%  →  {_interpretation_label(total_score)}"
        )


def _semantic_scoring_used(results: List[Dict]) -> bool:
    for result in results:
        if result.get("semantic_best"):
            return True
        if isinstance(result.get("semantic_score"), (int, float)):
            return True
        semantic_scores = result.get("semantic_scores") or {}
        if semantic_scores and not semantic_scores.get("skipped"):
            return True
    return False


def _print_winner_detail_tables(
    results: List[Dict],
    *,
    rubric_total: Optional[float] = None,
    semantic_total: Optional[float] = None,
) -> None:
    """Print rubric-best and semantic-best winner tables with full question text."""
    has_dual_track = any(result.get("semantic_best") for result in results)
    semantic_used = _semantic_scoring_used(results)

    if has_dual_track:
        rubric_rows = build_track_results(results, track="rubric")
        semantic_rows = build_track_results(results, track="semantic")
    else:
        rubric_rows = list(results)
        semantic_rows = (
            build_track_results(results, track="semantic") if semantic_used else []
        )

    _print_winner_detail_table(
        rubric_rows,
        title="RUBRIC WINNERS — best rubric-scoring answer per question",
        score_field="score",
        score_header="Rubric",
        total_score=rubric_total,
    )
    if semantic_used:
        _print_winner_detail_table(
            semantic_rows,
            title="SEMANTIC WINNERS — best semantic-scoring answer per question",
            score_field="semantic_score",
            score_header="Semantic",
            total_score=semantic_total,
        )


def _print_final_report(results: List[Dict], total_score: float) -> None:
    """Print final totals and per-track winner detail tables."""
    has_dual_track = any(result.get("semantic_best") for result in results)
    semantic_total = weighted_semantic_score(results)

    if has_dual_track:
        rubric_track = build_track_results(results, track="rubric")
        semantic_track = build_track_results(results, track="semantic")
        rubric_total = (
            weighted_primary_score(rubric_track, "score") if rubric_track else total_score
        )
        semantic_total = (
            weighted_primary_score(semantic_track, "semantic_score")
            if semantic_track
            else semantic_total
        )
    else:
        rubric_total = total_score

    print()
    print("=" * 70)
    if semantic_total is not None:
        print(
            f"📊 FINAL SCORE: rubric {rubric_total:.1f}%  |  "
            f"semantic {semantic_total:.1f}%"
        )
    else:
        print(f"📊 FINAL SCORE: rubric {rubric_total:.1f}%")
    print("=" * 70)
    print(f"\n✅ Interpretation (rubric): {_interpretation_label(rubric_total)}")
    if semantic_total is not None:
        print("\nℹ️  Semantic score is an independent audit metric vs answers_v2.txt.")

    _print_winner_detail_tables(
        results,
        rubric_total=rubric_total,
        semantic_total=semantic_total,
    )


def _run_model_with_export(
    *,
    questions,
    client,
    model_name,
    scorer_bundle,
    runtime,
    args,
    config,
    optimizer=None,
    semantic_scorer=None,
    reference_answers=None,
    langfuse_config=None,
    multi_model=False,
    dataset=None,
    shutdown_requested=None,
    keepalive=None,
):
    return run_single_model_benchmark(
        questions=questions,
        client=client,
        model_name=model_name,
        scorer_bundle=scorer_bundle,
        runtime=runtime,
        optimizer=optimizer,
        semantic_scorer=semantic_scorer,
        reference_answers=reference_answers,
        tracer_config=langfuse_config,
        tracer_factory=LangfuseTracer if langfuse_config else None,
        export_callback=_export_benchmark_results,
        export_kwargs={
            "args": args,
            "config": config,
            "multi_model": multi_model,
            "dataset": dataset,
            "runtime": runtime,
        },
        shutdown_requested=shutdown_requested,
        keepalive=keepalive,
    )


def cmd_list_models(args):
    """List available models from the provider."""
    try:
        config = _load_optional_config(args)
        _apply_config_defaults(args, config)
        api_key = getattr(args, "api_key", None)
        client = _create_configured_client(
            args.provider, args.endpoint, "temp", api_key, config, args
        )

        print(f"📋 Available models from {args.provider}:")
        print()

        models = client.list_models()
        if not models:
            print("   No models found")
            return

        if args.provider in {"lmstudio", "openwebui", "openrouter", "deepinfra"}:
            for model in models:
                model_id = model.get("id") or model.get("name", "unknown")
                print(f"   • {model_id}")
        else:
            for model in models:
                name = model.get("name") or model.get("id", "unknown")
                size_gb = model.get("size", 0) / (1024**3)
                print(f"   • {name} ({size_gb:.1f} GB)")

        print()
        print(f"💡 Use: uv run run_benchmark.py run {args.provider} -m <model_name>")

    except RuntimeError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    finally:
        if "client" in locals():
            client.close()


def cmd_judge(args) -> int:
    """Run offline LLM-as-Judge over saved v2 benchmark results."""
    return run_offline_judge(args)


def cmd_preload_semantic(args):
    """Download and warm the semantic scoring embedder and embedding cache."""
    config = _load_optional_config(args)
    semantic_config = getattr(getattr(config, "scoring", None), "semantic", None) if config else None

    provider = (
        getattr(args, "semantic_provider", None)
        or getattr(semantic_config, "provider", None)
        or DEFAULT_SEMANTIC_PROVIDER
    ).lower()
    model_name = (
        getattr(args, "semantic_model", None)
        or getattr(semantic_config, "model", None)
        or DEFAULT_SEMANTIC_MODEL
    )
    if provider == "deepinfra" and model_name == DEFAULT_SEMANTIC_MODEL:
        model_name = DEFAULT_DEEPINFRA_SEMANTIC_MODEL
    answers_file = (
        getattr(args, "semantic_answers", None)
        or getattr(semantic_config, "answers_file", None)
        or DEFAULT_SEMANTIC_ANSWERS_FILE
    )
    device = getattr(semantic_config, "device", "auto") if semantic_config else "auto"
    if semantic_config and semantic_config.max_seq_length_explicit:
        max_seq_length = semantic_config.max_seq_length
    else:
        max_seq_length = default_semantic_max_seq_length(provider=provider)
    endpoint = getattr(semantic_config, "endpoint", None) if semantic_config else None
    api_key = (
        getattr(args, "semantic_api_key", None)
        or (getattr(semantic_config, "api_key", None) if semantic_config else None)
    )
    api_key_env = getattr(semantic_config, "api_key_env", "DEEPINFRA_TOKEN") if semantic_config else "DEEPINFRA_TOKEN"
    if provider == "deepinfra" and not api_key:
        api_key = os.environ.get(api_key_env)

    try:
        result = preload_semantic_scorer(
            provider=provider,
            model_name=model_name,
            answers_file=answers_file,
            device=device if device != "auto" else None,
            max_seq_length=max_seq_length,
            endpoint=endpoint,
            api_key=api_key,
            api_key_env=api_key_env,
            force=getattr(args, "force", False),
        )
    except (RuntimeError, ValueError, FileNotFoundError) as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

    print("\n✅ Semantic scoring assets preloaded")
    print(f"   Provider: {result.provider}")
    print(f"   Model: {result.model_name}")
    print(f"   Reference answers: {result.reference_count} from {result.answers_file}")
    if result.encoded_count:
        print(f"   Newly encoded: {result.encoded_count}")
    print(f"   Warmup time: {result.elapsed_s:.1f}s")
    print(f"   Embedding cache: {result.embedding_cache}")
    if result.huggingface_cache is not None:
        print(f"   Hugging Face cache: {result.huggingface_cache}")
    print("\nLater runs with --semantic will reuse the embedding cache.")
    if result.provider == "local":
        print("The embedding model still loads into RAM each process; only download")
        print("and reference encoding are skipped after preload.")
    else:
        print("DeepInfra reference embeddings are cached locally; live API calls")
        print("still occur for each model response during the benchmark.")


def cmd_interactive(args):
    """Interactive TUI for selecting and testing multiple models."""
    try:
        config = _load_optional_config(args)
        _apply_config_defaults(args, config)
        try:
            runtime = _resolve_runtime_options(args, config)
        except ValueError as e:
            print(f"❌ Error: {e}")
            sys.exit(1)

        api_key = getattr(args, "api_key", None)
        client = _create_configured_client(
            args.provider, args.endpoint, "temp", api_key, config, args
        )

        if not client.test_connection():
            _report_connection_failure(
                client, args.provider, is_remote=client.base_url.startswith("https://")
            )
            sys.exit(1)

        print(f"🔍 Fetching available models from {args.provider}...\n")

        models = client.list_models()
        default_optimizer_endpoint = client.base_url
        client.close()
        if not models:
            print("   No models found")
            sys.exit(1)

        model_options = []
        model_names = []
        if args.provider in {"lmstudio", "openwebui", "openrouter", "deepinfra"}:
            for model in models:
                model_id = model.get("id") or model.get("name", "unknown")
                model_options.append(model_id)
                model_names.append(model_id)
        else:
            for model in models:
                name = model.get("name") or model.get("id", "unknown")
                size_gb = model.get("size", 0) / (1024**3)
                display_name = f"{name} ({size_gb:.1f} GB)"
                model_options.append(display_name)
                model_names.append(name)

        title = "Select models to benchmark (SPACE to select, ENTER to confirm, q to quit):"
        try:
            selected = pick(
                model_options,
                title,
                multiselect=True,
                min_selection_count=1,
                indicator="●",
                quit_keys=(ord("q"), ord("Q")),
            )
        except KeyboardInterrupt:
            print("\n\n❌ Cancelled by user")
            sys.exit(0)

        if not selected:
            print("\n❌ No models selected")
            sys.exit(0)

        selected_indices = [idx for _, idx in selected]
        selected_model_names = [model_names[idx] for idx in selected_indices]

        print(f"\n✅ Selected {len(selected_model_names)} model(s) for testing\n")
        _print_runtime(runtime)

        dataset = _load_dataset_for_cli(_questions_file_for_args(args, config))
        try:
            questions = _select_questions_for_args(dataset, args)
        except ValueError as e:
            print(f"❌ Error: {e}")
            sys.exit(1)
        scorer_bundle = _create_scorer_bundle(args, config, questions)
        semantic_scorer_bundle = _create_semantic_scorer_bundle(args, config, questions)
        optimizer = _initialize_optimizer(args, config, default_optimizer_endpoint)
        _sem = getattr(args, "semantic", False)
        _scoring_label = (
            f"{scorer_bundle.method_label} + semantic"
            if _sem
            else scorer_bundle.method_label
        )
        if optimizer and _sem:
            _scoring_label += " (dual-track)"
        print(f"✓ Using {_scoring_label} scoring\n")

        reference_answers = {}
        if optimizer:
            reference_answers = parse_reference_answers(
                config.answers_file if config else "answers_all.txt"
            )

        langfuse_config = _langfuse_config_or_none(config)
        all_results = []
        interrupted = False

        try:
            with install_signal_handlers() as shutdown:
                for i, model_name in enumerate(selected_model_names, 1):
                    if shutdown.is_requested():
                        interrupted = True
                        break

                    print("=" * 70)
                    print(f"Testing model [{i}/{len(selected_model_names)}]: {model_name}")
                    print("=" * 70)
                    model_started_at = time.time()
                    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(model_started_at))}")
                    print()

                    try:
                        model_client = _create_configured_client(
                            args.provider, args.endpoint, model_name, api_key, config, args
                        )
                    except (RuntimeError, ValueError) as e:
                        print(f"❌ Error creating client: {e}")
                        continue

                    try:
                        if not model_client.test_connection():
                            print(f"❌ Cannot connect to model {model_name}")
                            detail = getattr(model_client, "last_probe_error", None)
                            if detail:
                                print(f"   Reason: {detail}")
                            continue

                        try:
                            _validate_model_available(
                                model_client,
                                model_name,
                                provider=args.provider,
                                role="target",
                            )
                        except SystemExit as e:
                            print(f"❌ {e}")
                            print(f"   Skipping {model_name}")
                            continue

                        # Target only -- never optimizer.optimizer_client, even
                        # though the optimizer already exists at this point.
                        try:
                            _run_gpu_check_or_exit(model_client, model_name, args, config)
                        except GpuCheckFailed as e:
                            print(f"\n   ❌ {e}")
                            print(f"   Skipping {model_name}")
                            continue

                        try:
                            keepalive = _create_keepalive(model_client, optimizer, config)
                            keepalive_ctx = keepalive if keepalive else nullcontext()
                            with keepalive_ctx:
                                run_result = _run_model_with_export(
                                    questions=questions,
                                    client=model_client,
                                    model_name=model_name,
                                    scorer_bundle=scorer_bundle,
                                    runtime=RuntimeOptions(**runtime.__dict__),
                                    args=args,
                                    config=config,
                                    optimizer=optimizer,
                                    semantic_scorer=(
                                        semantic_scorer_bundle.scorer
                                        if semantic_scorer_bundle
                                        else None
                                    ),
                                    reference_answers=reference_answers,
                                    langfuse_config=langfuse_config,
                                    multi_model=len(selected_model_names) > 1,
                                    dataset=dataset,
                                    shutdown_requested=shutdown.is_requested,
                                    keepalive=keepalive,
                                )
                            print(
                                f"Finished {model_name}: {time.strftime('%Y-%m-%d %H:%M:%S')} "
                                f"(total {_format_elapsed(time.time() - model_started_at)})"
                            )
                            if config and config.cloudrun_cost.enabled:
                                estimate = estimate_cost(
                                    time.time() - model_started_at,
                                    cpu=config.cloudrun_cost.cpu,
                                    memory_gib=config.cloudrun_cost.memory_gib,
                                    gpu_type=config.cloudrun_cost.gpu_type,
                                    gpu_zonal_redundancy=config.cloudrun_cost.gpu_zonal_redundancy,
                                )
                                print(f"💰 {format_cost_estimate(estimate)}")
                        except RuntimeError as e:
                            print(f"   ❌ Error: {e}")
                            print(f"   Skipping remaining questions for {model_name}")
                            continue
                    finally:
                        model_client.close()

                    if run_result.results:
                        if run_result.optimization_results and optimizer:
                            save_optimization_results(
                                run_result.optimization_results,
                                model_name,
                                args.optimizer_model,
                            )

                        semantic_total = weighted_semantic_score(run_result.results)
                        all_results.append(
                            {
                                "model": model_name,
                                "score": run_result.total_score,
                                "semantic_score": semantic_total,
                                "interpretation": run_result.interpretation,
                            }
                        )
                        if semantic_total is not None:
                            print(
                                f"\n✅ {model_name}: rubric {run_result.total_score:.1f}%  |  "
                                f"semantic {semantic_total:.1f}%\n"
                            )
                        else:
                            print(f"\n✅ {model_name}: rubric {run_result.total_score:.1f}%\n")
                        _print_final_report(run_result.results, run_result.total_score)
                    else:
                        print(f"\n❌ No results for {model_name}\n")

                    if getattr(run_result, "interrupted", False):
                        interrupted = True
                        break
        except GracefulShutdown:
            interrupted = True
        finally:
            if optimizer:
                optimizer.close()

        if interrupted:
            print("\n⚠️  Benchmark interrupted. Partial results were saved when available.")

        if all_results:
            print("\n" + "=" * 70)
            print("📊 SUMMARY: ALL TESTED MODELS")
            print("=" * 70)
            show_semantic = any(
                isinstance(result.get("semantic_score"), (int, float))
                for result in all_results
            )
            if show_semantic:
                print(f"{'Model':<30} {'Rubric':<10} {'Semantic':<10} {'Interpretation'}")
            else:
                print(f"{'Model':<30} {'Rubric':<10} {'Interpretation'}")
            print("-" * 70)
            for result in all_results:
                if show_semantic:
                    semantic_score = result.get("semantic_score")
                    semantic_label = (
                        f"{semantic_score:.1f}%"
                        if isinstance(semantic_score, (int, float))
                        else "—"
                    )
                    print(
                        f"{result['model']:<30} "
                        f"{result['score']:<10.1f}% "
                        f"{semantic_label:<10} "
                        f"{result['interpretation']}"
                    )
                else:
                    print(
                        f"{result['model']:<30} "
                        f"{result['score']:<10.1f}% {result['interpretation']}"
                    )
            print("=" * 70)
        else:
            print("\n❌ No successful tests completed")

        if interrupted:
            sys.exit(130)

    except RuntimeError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


def cmd_run_benchmark(args):
    """Run the benchmark."""
    config = _load_optional_config(args)
    _apply_config_defaults(args, config)
    try:
        runtime = _resolve_runtime_options(args, config)
    except ValueError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

    api_key = getattr(args, "api_key", None)
    try:
        client = _create_configured_client(
            args.provider, args.endpoint, args.model, api_key, config, args
        )
    except (RuntimeError, ValueError) as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

    optimizer = None
    try:
        try:
            with install_signal_handlers() as shutdown:
                run_started_at = time.time()
                is_remote = client.base_url.startswith("https://")
                if is_remote:
                    print(
                        f"⏳ Probing {client.base_url} "
                        f"(Cloud Run cold start can take 1–3 min; "
                        f"{getattr(client, '_probe_attempts', lambda: 10)()} retries, "
                        f"{getattr(client, 'timeout', 150)}s per attempt)..."
                    )
                if not client.test_connection():
                    _report_connection_failure(client, args.provider, is_remote=is_remote)
                    sys.exit(1)

                print(f"🚀 Red Team AI Benchmark v{BENCHMARK_VERSION}")
                print(f"   Provider: {args.provider}")
                print(f"   Model: {args.model}")
                print(f"   Endpoint: {client.base_url}")
                profile = getattr(args, "profile", DEFAULT_PROFILE)
                print(f"   Profile: {profile}")
                print(
                    f"   Runtime: max_tokens={runtime.max_tokens}, "
                    f"temperature={runtime.temperature}, "
                    f"rate_limit_delay={runtime.rate_limit_delay}, "
                    f"concurrency={runtime.concurrency}"
                )
                print(f"   Started: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(run_started_at))}")
                print("   Evaluating uncensoredness & technical accuracy\n")

                _validate_model_available(
                    client,
                    args.model,
                    provider=args.provider,
                    role="target",
                )

                # Target only -- runs before the optimizer is created below, so
                # it structurally cannot touch the optimizer's client/model.
                try:
                    _run_gpu_check_or_exit(client, args.model, args, config)
                except GpuCheckFailed as e:
                    print(f"\n❌ {e}")
                    sys.exit(1)

                dataset = _load_dataset_for_cli(_questions_file_for_args(args, config))
                try:
                    questions = _select_questions_for_args(dataset, args)
                except ValueError as e:
                    print(f"❌ Error: {e}")
                    sys.exit(1)
                scorer_bundle = _create_scorer_bundle(args, config, questions)
                semantic_scorer_bundle = _create_semantic_scorer_bundle(
                    args, config, questions
                )
                optimizer = _initialize_optimizer(
                    args, config, args.optimizer_endpoint or client.base_url
                )
                _sem = getattr(args, "semantic", False)
                _scoring_label = (
                    f"{scorer_bundle.method_label} + semantic"
                    if _sem
                    else scorer_bundle.method_label
                )
                if optimizer and _sem:
                    _scoring_label += " (dual-track)"
                print(f"✓ Using {_scoring_label} scoring\n")

                reference_answers = {}
                if optimizer:
                    reference_answers = parse_reference_answers(
                        config.answers_file if config else "answers_all.txt"
                    )

                keepalive = _create_keepalive(client, optimizer, config)
                if keepalive:
                    keepalive_cfg = _resolve_keepalive_config(config)
                    roles = "target + optimizer" if optimizer else "target"
                    print(
                        f"✓ Model keepalive enabled ({roles}, "
                        f"every {keepalive_cfg.interval_s:.0f}s on idle model)\n"
                    )
                langfuse_config = _langfuse_config_or_none(config)

                try:
                    keepalive_ctx = keepalive if keepalive else nullcontext()
                    with keepalive_ctx:
                        run_result = _run_model_with_export(
                            questions=questions,
                            client=client,
                            model_name=args.model,
                            scorer_bundle=scorer_bundle,
                            runtime=runtime,
                            args=args,
                            config=config,
                            optimizer=optimizer,
                            semantic_scorer=(
                                semantic_scorer_bundle.scorer
                                if semantic_scorer_bundle
                                else None
                            ),
                            reference_answers=reference_answers,
                            langfuse_config=langfuse_config,
                            dataset=dataset,
                            shutdown_requested=shutdown.is_requested,
                            keepalive=keepalive,
                        )
                except RuntimeError as e:
                    print(f"   ❌ Error: {e}")
                    print("   Aborting benchmark.")
                    sys.exit(1)

                print(
                    f"\n   Finished: {time.strftime('%Y-%m-%d %H:%M:%S')} "
                    f"(total {_format_elapsed(time.time() - run_started_at)})"
                )
                if config and config.cloudrun_cost.enabled:
                    estimate = estimate_cost(
                        time.time() - run_started_at,
                        cpu=config.cloudrun_cost.cpu,
                        memory_gib=config.cloudrun_cost.memory_gib,
                        gpu_type=config.cloudrun_cost.gpu_type,
                        gpu_zonal_redundancy=config.cloudrun_cost.gpu_zonal_redundancy,
                    )
                    print(f"   💰 {format_cost_estimate(estimate)}")

                if run_result.optimization_results:
                    save_optimization_results(
                        run_result.optimization_results, args.model, args.optimizer_model
                    )

                if run_result.results:
                    _print_final_report(run_result.results, run_result.total_score)
                else:
                    print("\n⚠️  Benchmark interrupted before any question completed.")

                if getattr(run_result, "interrupted", False):
                    print("\n⚠️  Benchmark interrupted. Partial results were saved when available.")
                    sys.exit(130)
        except GracefulShutdown:
            print("\n⚠️  Benchmark interrupted before results could be saved.")
            sys.exit(130)
    finally:
        client.close()
        if optimizer:
            optimizer.close()


def _add_provider_arg(parser):
    parser.add_argument(
        "provider",
        choices=["lmstudio", "ollama", "openwebui", "openrouter", "deepinfra"],
        help="API provider",
    )


def _add_endpoint_arg(parser):
    parser.add_argument(
        "-e",
        "--endpoint",
        help=(
            "Custom endpoint URL (default: localhost:1234 for lmstudio, "
            "localhost:11434 for ollama, localhost:3000 for openwebui, "
            "OpenRouter/DeepInfra cloud defaults)"
        ),
    )


def _add_api_key_arg(parser):
    parser.add_argument(
        "--api-key",
        help="API key for providers or reverse proxies (OpenRouter, OpenWebUI, Ollama)",
    )


def _add_export_args(parser):
    parser.add_argument("-o", "--output", help="Custom output basename")
    parser.add_argument("--config", help="Load configuration from YAML file")
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Also export results to CSV format",
    )


def _add_profile_arg(parser):
    parser.add_argument(
        "--profile",
        choices=list(PROFILE_DEFAULTS.keys()),
        default=DEFAULT_PROFILE,
        help="Benchmark runtime profile (default: standard)",
    )


def _add_runtime_args(parser):
    parser.add_argument(
        "--rate-limit-delay",
        type=float,
        default=None,
        help="Delay between request starts in seconds (default: config or 1.5)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum response tokens per benchmark question (default: config or 3072)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Model temperature for benchmark questions (default: config or 0.2)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Number of concurrent benchmark questions (default: config or 1)",
    )
    parser.add_argument(
        "--question-ids",
        nargs="+",
        help="Run only selected v2 question IDs, preserving benchmark order",
    )
    parser.add_argument(
        "--request-log",
        help="Append per-question request diagnostics to a JSONL file",
    )
    parser.add_argument(
        "--ollama-keep-alive",
        help="Ollama keep_alive value for /api/chat, e.g. 30m or -1",
    )
    parser.add_argument(
        "--min-vram-fraction",
        type=float,
        default=None,
        help=(
            "Abort before the paid benchmark run if Ollama's /api/ps reports "
            "less than this fraction (0-1) of the model resident in GPU VRAM "
            "(catches silent CPU fallback). 0 disables the check. "
            "Default: config or disabled."
        ),
    )


def _add_optimization_args(parser):
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Disable optimization even if optimizer_provider/model are set in config",
    )
    parser.add_argument(
        "--optimizer-model",
        default=None,
        help="Model name for the optimizer LLM (required with --optimizer-provider)",
    )
    parser.add_argument(
        "--optimizer-provider",
        default=None,
        choices=["ollama", "lmstudio", "openwebui", "openrouter", "deepinfra"],
        help="Provider for the optimizer LLM (required with --optimizer-model)",
    )
    parser.add_argument(
        "--optimizer-api-key",
        default=None,
        help="API key for cloud optimizer providers (deepinfra, openrouter)",
    )
    parser.add_argument(
        "--optimizer-endpoint",
        help="Optimizer endpoint URL (default: same as main endpoint)",
    )
    parser.add_argument(
        "--max-optimization-iterations",
        type=int,
        default=None,
        help="Maximum optimization iterations per question (default: 4, or config value)",
    )


def _add_semantic_args(parser):
    parser.add_argument(
        "--semantic",
        action="store_true",
        help="Enable optional parallel semantic scoring",
    )
    parser.add_argument(
        "--semantic-provider",
        choices=["local", "deepinfra"],
        default=None,
        help=(
            "Semantic embedding provider "
            f"(default: {DEFAULT_SEMANTIC_PROVIDER}; deepinfra uses Qwen3-Embedding-8B)"
        ),
    )
    parser.add_argument(
        "--semantic-model",
        default=None,
        help=(
            "Embedding model for semantic scoring "
            f"(default: {DEFAULT_SEMANTIC_MODEL} local, "
            f"{DEFAULT_DEEPINFRA_SEMANTIC_MODEL} for deepinfra)"
        ),
    )
    parser.add_argument(
        "--semantic-api-key",
        default=None,
        help="API key for --semantic-provider deepinfra (default: DEEPINFRA_TOKEN env)",
    )
    parser.add_argument(
        "--semantic-answers",
        default=None,
        help=(
            "Reference answer file for semantic scoring "
            f"(default: {DEFAULT_SEMANTIC_ANSWERS_FILE})"
        ),
    )


def _add_benchmark_common_args(parser):
    _add_endpoint_arg(parser)
    _add_export_args(parser)
    _add_profile_arg(parser)
    _add_api_key_arg(parser)
    _add_runtime_args(parser)
    _add_optimization_args(parser)
    _add_semantic_args(parser)


def main():
    """Main entry point with CLI."""
    parser = argparse.ArgumentParser(
        description="Red Team AI Benchmark - Evaluate LLMs for offensive security",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available models
  uv run run_benchmark.py ls lmstudio
  uv run run_benchmark.py ls ollama

  # Interactive model selection (TUI)
  uv run run_benchmark.py interactive ollama
  uv run run_benchmark.py interactive lmstudio

  # Run default v2 benchmark (rubric scoring)
  uv run run_benchmark.py run lmstudio -m "mistral-7b"
  uv run run_benchmark.py run ollama -m "llama3.1:8b"

  # Run quick v2 smoke profile
  uv run run_benchmark.py run ollama -m "llama3.1:8b" --profile quick

  # Run post-hoc LLM-as-Judge over saved v2 results
  uv run run_benchmark.py judge --results "results/*.json" --mode disputed

  # Warm semantic embedding cache before a --semantic run
  uv sync --extra semantic
  uv run run_benchmark.py preload-semantic
  uv run run_benchmark.py preload-semantic --semantic-provider deepinfra

  # Custom endpoint
  uv run run_benchmark.py run ollama -e http://192.168.1.100:11434 -m "mistral"
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    parser_ls = subparsers.add_parser("ls", help="List available models")
    _add_provider_arg(parser_ls)
    _add_endpoint_arg(parser_ls)
    _add_api_key_arg(parser_ls)
    parser_ls.add_argument("--config", help="Load configuration from YAML file")

    parser_run = subparsers.add_parser("run", help="Run benchmark")
    _add_provider_arg(parser_run)
    parser_run.add_argument("-m", "--model", required=True, help="Model name")
    _add_benchmark_common_args(parser_run)

    parser_interactive = subparsers.add_parser(
        "interactive", help="Interactive TUI for selecting and testing multiple models"
    )
    _add_provider_arg(parser_interactive)
    _add_benchmark_common_args(parser_interactive)

    parser_judge = subparsers.add_parser(
        "judge",
        help="Run offline LLM-as-Judge over saved v2 benchmark results",
    )
    add_judge_args(parser_judge)

    parser_preload = subparsers.add_parser(
        "preload-semantic",
        help="Warm the semantic embedding cache (local model or DeepInfra API)",
    )
    parser_preload.add_argument("--config", help="Load configuration from YAML file")
    parser_preload.add_argument(
        "--semantic-provider",
        choices=["local", "deepinfra"],
        default=None,
        help=f"Embedding provider to preload (default: {DEFAULT_SEMANTIC_PROVIDER})",
    )
    parser_preload.add_argument(
        "--semantic-model",
        default=None,
        help=f"Embedding model to preload (default: {DEFAULT_SEMANTIC_MODEL})",
    )
    parser_preload.add_argument(
        "--semantic-answers",
        default=None,
        help=(
            "Reference answer file to preload "
            f"(default: {DEFAULT_SEMANTIC_ANSWERS_FILE})"
        ),
    )
    parser_preload.add_argument(
        "--semantic-api-key",
        default=None,
        help="API key for --semantic-provider deepinfra (default: DEEPINFRA_TOKEN env)",
    )
    parser_preload.add_argument(
        "--force",
        action="store_true",
        help="Re-encode all reference answers even when a cache file exists",
    )

    args = parser.parse_args()

    if args.command == "ls":
        cmd_list_models(args)
    elif args.command == "run":
        cmd_run_benchmark(args)
    elif args.command == "interactive":
        cmd_interactive(args)
    elif args.command == "judge":
        sys.exit(cmd_judge(args))
    elif args.command == "preload-semantic":
        cmd_preload_semantic(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
