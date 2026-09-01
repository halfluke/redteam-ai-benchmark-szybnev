# Agent Reference

This document keeps implementation details out of `AGENTS.md` and user-facing README files.

## Working directory

Run the CLI from the repository root only. Defaults such as `datasets/v2/benchmark.jsonl`, `answers_v2.txt`, `answers_all.txt`, `results/`, and `.cache/redteam/` are resolved relative to the process cwd. Packaged installs of the `redteam-benchmark` entrypoint do not ship those files; do not claim a cwd-independent installable CLI for full runs.

## v2 Dataset

Default dataset:

```text
datasets/v2/benchmark.jsonl
```

The first JSONL record is a manifest with `schema: rubric-v2`, `dataset_id`, `dataset_version`, and `benchmark_version`. Every question record must include:

- `id`
- `domain`
- `capability`
- `difficulty`
- `prompt`
- `expected_artifacts`
- `rubric`
- `fatal_errors`
- `acceptable_variants`
- `tags`
- `weight`

Allowed difficulty values:

- `L1 factual`
- `L2 procedure`
- `L3 troubleshooting`
- `L4 scenario reasoning`
- `L5 multi-step operator task`

Question loading is implemented in `benchmark/io.py`.

## Runtime Profiles

CLI profiles are defined in `run_benchmark.py`:

- `quick`: quick v2 smoke subset selected by question `profiles`.
- `standard`: default full v2 run.
- `enterprise`: full v2 run with audit-friendly export.
- `local-only`: full v2 run without LLM judge usage.
- `cloud-comparison`: full v2 run for fixed hosted-model comparisons.

If adding a profile, update `PROFILE_DEFAULTS`, CLI docs, README files, and tests.

`--question-ids` is an additional runtime filter. Apply it after profile filtering and preserve dataset order. Unknown IDs in the selected profile must fail before any model request.

## Scoring

Default scorer:

```text
rubric
```

`scoring/rubric_scorer.py` is deterministic and local. It checks each criterion pattern against the response, records passed and failed criteria, collects evidence, and applies fatal-error rules before normal scoring.

Do not add runtime scorer modes for keyword, semantic, hybrid, or online LLM judging. LLM-as-Judge belongs only in the offline `judge` command.

Optional embedded semantic scoring is allowed only as a parallel audit metric. It must not replace rubric scoring or change `score`, `total_score`, interpretation labels, optimizer triggers, or optimizer acceptance. The semantic scorer compares the final selected answer for each question (baseline unless prompt optimization replaces it) to the full-answer reference in `answers_v2.txt`, using local SentenceTransformer embeddings (`provider: local`, default `Qwen/Qwen3-Embedding-0.6B`) or DeepInfra API embeddings (`provider: deepinfra`, default `Qwen/Qwen3-Embedding-8B` with recalibrated cosine bands). Export semantic data as sibling fields such as `semantic_score`, `semantic_similarity`, and top-level `semantic_scoring`.

Reference embeddings are cached under `.cache/redteam/semantic/` in a file keyed by the full answers corpus digest and provider. The cache stores a per-answer SHA-256 hash alongside each embedding. When an older cache file lacks those hashes (legacy format), `preload-semantic` performs a one-time full re-encode into the new format. After that, only answers whose reference text hash changed are re-encoded; unchanged answers are reused from the current cache file.

YAML `scoring.semantic.thresholds` are optional. When omitted, provider-aware defaults apply (local 0.6B bands vs recalibrated DeepInfra 8B bands in `scoring/semantic_calibration.py`). When the block is present in YAML, values are used as-is.

### Semantic preprocessing

Implementation: `scoring/semantic_scorer.py` (`strip_thinking_blocks`, `_THINKING_STRIP_PATTERNS`), regression tests in `tests/test_thinking_strip.py`.

1. **Garbage check** (`scoring/garbage.py`) runs first. Responses with ≥24 words and unique-word ratio < 0.12 are treated as `semantic skipped (garbage)`; the embedder is not called. Typical BugTrace repetition leaks and MoE paragraph loops hit this rule. Skips export and log `garbage_word_count`, `garbage_unique_word_count`, and `garbage_unique_ratio`; the console prints a one-line summary when garbage is detected.
2. **Thinking strip** removes closed reasoning blocks before encoding. Named patterns include:
   - `bugtrace_channel_thought` — BugTrace/DeepHat `<|channel>thought…<channel|>`
   - `redacted_thinking` — Qwen3/DeepSeek `<think>…</think>` (close hex `3c2f72656461637465645f7468696e6b696e673e`, 20 B)
   - `xml_think` — DeepSeek-R1 `xml_think` blocks (close hex `3c2f7468696e6b3e`, 8 B; do not confuse with `redacted_thinking`)
   - `xml_thinking`, `xml_thought`, `xml_reasoning`, `xml_analysis`, `xml_scratchpad`, `bracket_thinking`, `pipe_thinking`, `pipe_thought_reasoning`, `chatml_think`, `pipe_begin_end_thought`
   Only full open+close blocks match; unclosed prefixes (e.g. `<|channel>thoughtos…`) stay intact.
3. **Encode** the stripped text. Local provider caps length at **2048** tokens by default (`scoring.semantic.max_seq_length`). DeepInfra defaults to **3072** (aligned with benchmark `max_tokens`) when omitted. Reference embeddings are cached under `.cache/redteam/semantic/` keyed by corpus digest and provider.

Recalibrate DeepInfra thresholds against your corpus with:

```bash
uv run scripts/calibrate_semantic_thresholds.py --provider deepinfra
```

Request logs (`--request-log` / `request_log` in config) append one JSONL row per model answer. Every question gets at least one row; when prompt optimization runs, the log keeps the full attempt history (baseline iteration 0 plus every optimized attempt) and separate final winner rows when optimization completes.

Phases:

- `baseline`: first target-model answer to the original question prompt (also used for iteration 0 inside optimizer history).
- `optimization`: reframed-prompt attempts (`optimization_iteration`, `optimization_strategy`, `optimizer_ms`, `latency_ms`).
- `final` / `final_rubric`: rubric-best winner after optimization (uses the winning attempt's reframed prompt when applicable).
- `final_semantic`: semantic-best winner whenever `--semantic` is active and optimization produced a semantic track (logged even when it matches rubric-best).

Each row may include the full prompt and response text, rubric score, `answer_source` (`baseline` / `optimized`), latency, refusal and critical-error flags, question metadata, and semantic strip/garbage diagnostics: `thinking_stripped_chars`, `thinking_stripped_tokens_est`, `strip_matched_pattern` (top-level and inside `semantic_scores`). Logs must not include provider headers or API keys.

A garbage baseline with a high rubric score still triggers prompt optimization when `--semantic` is active (`should_trigger_prompt_optimization()` in `optimization/policy.py`).

### Optimization policy

Implementation: `optimization/policy.py`.

- **Trigger** (`should_trigger_prompt_optimization()`): baseline rubric **< 25%**, semantic **< 25%**, or semantic **garbage**. Scores at/above 25% on both tracks do not trigger.
- **Early exit** (`optimization_tracks_resolved()`): stop the strategy loop once the best rubric score across all attempts is **≥ 75%** and the best semantic score is **≥ 75%** with a real embed score (not garbage). Without `--semantic`, rubric-best **≥ 75%** alone is enough.
- Rubric-best and semantic-best may come from **different** attempts; exit checks each track's best independently.
- Console trigger labels: `zero rubric`, `zero semantic`, `zero both`, `semantic garbage`.

## Offline LLM-as-Judge

Post-hoc judging is exposed through the main CLI:

```bash
uv run run_benchmark.py judge --results "results/*.json" --mode disputed
```

The implementation lives in `benchmark/offline_judge.py`; do not add a separate script entrypoint. Default `--results` globs are `results/*.json` and `results_*_v2/*.json` (exporter layout plus legacy folders). Zero matches is a hard error. It reads saved v2 result JSON files, does not rerun benchmark models, and writes sidecar audit artifacts under the configured output directory. Treat `judge_score` as the judged subset score and `judge_adjusted_score` as the comparison-friendly adjusted result.

`ScoringResult` now carries:

- `score`
- `normalized_score`
- `censored`
- `critical_error`
- `criteria_passed`
- `criteria_failed`
- `evidence`
- `metrics`
- `details`

## Aggregation

`benchmark/metrics.py` owns aggregate scoring:

- `weighted_score(results)`
- `summarize_results(results)`
- `summarize_semantic_results(results)` for the optional parallel semantic metric
- `build_track_results(results, track="rubric"|"semantic")` — projects each result onto one track's best-scoring answer for dual-track aggregation
- `summarize_track(results, primary="score"|"semantic_score")` — builds a compact weighted summary (score, question count, breakdowns) for one track
- `weighted_primary_score(results, primary)` — weighted average for any numeric score field

High scores are labeled `strong-candidate`, not `production-ready`.

### Dual-track reporting

When `--optimize-prompts` and `--semantic` are both active, the optimizer independently tracks the answer with the highest rubric score (`rubric_best`) and the answer with the highest semantic score (`semantic_best`) across all optimization attempts. These are stored on each `QuestionResult` via the `rubric_best`, `semantic_best`, and `tracks_diverged` fields.

The final report shows two separate tables and interpretations: one for the rubric track and one for the semantic track. JSON export gains a top-level `tracks` block with each track's weighted total score and interpretation. The `summary` dict gains `rubric_track` and `semantic_track` sub-keys when dual-track data is present.

The optimization trigger fires when rubric score **or** semantic score is **below 25%**, or when semantic scoring is skipped as garbage on the baseline (see `optimization/policy.py`). Early exit from the loop occurs when the best rubric score and best semantic score across all attempts are both **≥ 75%** (semantic must be a real score, not garbage) when `--semantic` is active. Without `--semantic`, early exit is rubric-only once the best rubric score is **≥ 75%**.

Semantic scoring for each optimization attempt is done synchronously after the target model returns: rubric score and semantic score are computed in sequence within the same iteration, then both are printed inline. Next-strategy generation by the optimizer LLM is still overlapped with the target model call using a background thread.

## Export

`utils/export.py` writes JSON, CSV, and `criteria_csv`.

JSON exports include top-level audit provenance:

- `benchmark_version`
- `dataset_id`
- `dataset_version`
- `dataset_hash`
- `scorer_version`
- `config_hash`
- `run_config`
- `git_commit`
- `package_version`
- `runtime_profile`

`criteria_csv` writes one row per passed or failed rubric criterion.

## Adding v2 Questions

When adding a question:

1. Add one JSON object to `datasets/v2/benchmark.jsonl`.
2. Keep `id` unique.
3. Use one allowed difficulty string.
4. Keep `rubric` non-empty and criteria weights positive.
5. Include fatal-error rules for common dangerous false claims.
6. Add or update calibration fixtures when scorer behavior changes.
7. Run `uv run pytest` and `python3 -m compileall -q run_benchmark.py benchmark models optimization scoring tracing utils`.

Do not add large batches of questions without rubric criteria.

## Optional Features

Prompt optimization remains separate from base-model scoring. It runs only after a baseline response scores **below 25% on rubric or semantic**, or when semantic scoring is skipped as garbage (see `optimization/policy.py`), tries up to every configured strategy (default: all four) from frozen baseline context, overlaps optimizer LLM generation of strategy *N+1* with target model execution of strategy *N*, scores rubric and semantic synchronously per iteration (printing both inline), independently tracks the rubric-best and semantic-best answers, and stops early once both best scores are **≥ 75%**. Do not mix optimized results into base model comparison tables.

When `--semantic` is enabled, console final summaries show rubric and semantic totals, then **Rubric Winners** and **Semantic Winners** detail tables (score, baseline vs optimization/strategy, full question prompt). With optimization active and diverged tracks, each table reflects that track's best answer independently.

Langfuse tracing is optional and should not be required for local or CI validation.

Ollama supports optional reverse-proxy Bearer auth through `--api-key`, config, or `OLLAMA_API_KEY`, and optional `keep_alive` through `--ollama-keep-alive`, `provider.keep_alive`, or `OLLAMA_KEEP_ALIVE`. Keep those options provider-local.

### Cloud Run cost estimate (`utils/cloudrun_cost.py`)

Optional, config-driven (`cloudrun_cost:` in YAML), disabled by default. When enabled:

- Prints USD **and** a display currency (default `currency: GBP`, `usd_per_unit: 1.33` = approximate USD per £1; no live FX API) at end of run / per model in `interactive`.
- Prints mid-run spent + linear full-run projection every `progress_every` questions (default 5; `0` disables). CLI: `--cloudrun-cost-progress-every`.
- Persists a top-level `cloudrun_cost` object in result JSON and appends a `phase: cost_summary` row to the request log when configured.
- Optional abort when display-currency spend reaches `max_cost` / `--max-cloudrun-cost` (exit code `2`); partial results are still exported when available.

This is an estimate, not the authoritative GCP invoice:

- Real Cloud Billing data (BigQuery export, Cost Management UI) lags actual usage by roughly 24 hours and has no real-time API, so a genuine live cost query is not possible during a run.
- The estimate multiplies observed wall-clock **session** uptime by published on-demand, Tier-1-region, instance-based-billing rates for CPU (`$/vCPU-second`), memory (`$/GiB-second`), and GPU (`$/second`).
- Session start defaults to Python `run_started_at` / `model_started_at`. Wrapper scripts (`scripts/run_*_baseline.sh`) set `CLOUDRUN_COST_SESSION_START` before warmup; `warmup_*.sh` write `CLOUDRUN_COST_WARMUP_SECONDS` to `.cache/redteam/cloudrun_cost_warmup.env` so warmup wall time is included in the billed elapsed window.
- This mirrors Cloud Run's actual billing model for GPU-attached services, which must use instance-based billing: the full CPU/memory/GPU allocation is billed continuously for every second the instance is up, whether idle or actively serving a request. That makes uptime × rate a close proxy for the real charge, modulo committed-use discounts, free-tier credits, or rates changing after this was written.
- Rates in code (`CPU_PER_VCPU_SECOND`, `MEMORY_PER_GIB_SECOND`, `GPU_PER_SECOND`) are Tier 1 on-demand instance-based list prices as of `RATES_AS_OF` (see constants in `utils/cloudrun_cost.py`), matching <https://cloud.google.com/run/pricing> for regions such as europe-west1. Re-verify and update those constants if list prices drift.
- `cpu`/`memory_gib`/`gpu_type`/`gpu_zonal_redundancy` in the config must match what the Cloud Run service was actually deployed with (see the corresponding `deploy-*.sh` script in `GCP-CLOUDRUN-AImodels/`); the tool has no way to introspect the live deployment's resource allocation.

### Pre-flight GPU residency check (`benchmark/gpu_check.py`)

Optional, config-driven (`gpu_check:` in YAML) or CLI-driven (`--min-vram-fraction`), disabled by default. Immediately after `test_connection()` succeeds and before any per-question benchmark call, keepalive thread start, or optimizer setup, sends a minimal request to load the model, then queries Ollama's own `GET /api/ps`, which reports `size` (total model bytes) and `size_vram` (bytes actually resident in GPU VRAM) for the running model. If `size_vram / size` is below the configured minimum fraction, raises `GpuCheckFailed` (a `RuntimeError` subclass) and the run aborts (`cmd_run_benchmark`: `sys.exit(1)`; `cmd_interactive`: skips that model and continues to the next).

Why this exists: a real incident where Cloud Run's L4 GPU driver (535) became incompatible with Ollama's CUDA backend from `v0.30.0` onward, causing silent CPU fallback (`inference compute id=cpu`) with no error — the benchmark just ran extremely slowly and every question timed out, burning both wall-clock time and Cloud Run GPU billing before anyone noticed. See `GCP-CLOUDRUN-AImodels/README.md` (the `v0.24.0` image pin) and the upstream root cause at <https://github.com/ollama/ollama/issues/16449>.

Why `/api/ps` instead of reading Ollama's server logs directly: the benchmark client only ever talks to the model's HTTP API — the same code path is shared across local Ollama, LM Studio, OpenWebUI, and Cloud Run. Actual Ollama server logs (which show explicit lines like `library=cuda_v12` or `no compatible GPUs were discovered`) aren't reachable through that API at all. Reading them would mean shelling out to `gcloud logging read`, which needs a GCP project/service id the client doesn't have (it only has the HTTPS URL), separate IAM permissions, has log-ingestion delay, and only works for this one Cloud Run deployment — not local/LM Studio/OpenWebUI targets this same tool supports. `/api/ps`'s `size_vram` field is Ollama's own authoritative, provider-native answer to "is this actually in GPU memory," reachable the same way for every deployment target, so an earlier tokens/sec-based heuristic (inferring GPU usage from generation speed, which needed per-model threshold guessing) was replaced with this direct check.

Implementation notes:

- Only measurable for `OllamaClient`. For any other provider, or if the load/`/api/ps` request itself raises or the model isn't found in the `/api/ps` list, `run_gpu_check()` returns `None` and the check is silently skipped — a missing signal must never block a run on its own.
- CLI `--min-vram-fraction` takes priority over `gpu_check:` in YAML when both are present (same precedence pattern as `--max-tokens`, `--concurrency`, etc. in `_resolve_runtime_options`). `--min-vram-fraction 0` explicitly disables the check even if YAML enables it.
- `configs/cloudrun_ollama.yaml` (generic Ollama Cloud Run) and `configs/cloudrun_ollama_bugtrace.yaml` (BugTraceAI) both ship with `min_vram_fraction: 0.9`, confirmed against live GPU deployments on `v0.24.0` (Qwen3 ~fully VRAM-resident; BugTraceAI ~63 tok/s, fully VRAM-resident).
- Scoped to the target model only, never the optimizer: `_run_gpu_check_or_exit()` is only ever called with the target `client`/`model_client`, never `optimizer.optimizer_client`. In `cmd_run_benchmark`, the check runs before `_initialize_optimizer()` is even called, so the optimizer object doesn't exist yet. In `cmd_interactive`, the optimizer is created once before the per-model loop, but the check is still called with `model_client` explicitly, never the optimizer. This matters because `--optimize-prompts` commonly points at a separate, often local, non-Cloud-Run optimizer endpoint with its own hardware profile that a Cloud Run GPU expectation should never gate. Locked in by `test_gpu_check_only_applies_to_target_not_optimizer_in_run_command` / `..._in_interactive` in `tests/test_gpu_check.py`.
