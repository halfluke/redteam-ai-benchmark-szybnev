# Red Team AI Benchmark

> **Branch note:** `feature/cloudrun-v2` has selectively merged the upstream v2.3.0 scoring layer (`rubric-v2.1.0` negation heuristics, refusal detector improvements, dataset 2.3.0 patterns). Runner, optimizer, repeats/seed, and leaderboard changes from upstream v2.3.0 have not been merged; this branch adds Cloud Run auth, keepalive, dual-track semantic scoring, and optimizer fixes not present in upstream.

**Russian version:** [README.ru.md](README.ru.md)

> **README sync:** Keep [README.md](README.md) and [README.ru.md](README.ru.md) in sync — same sections, features, and CLI examples; only the language differs. Update both in the same change when user-facing docs change.

Red Team AI Benchmark is a CLI benchmark for choosing base LLMs for authorized red-team and offensive-security work. Version 2 uses a rubric-based dataset instead of judging answers only against one golden response.

---

## Branch: `feature/cloudrun-v2`

This branch extends upstream `v2` with Cloud Run authentication, background keepalive, and several optimizer improvements. Changes are listed below with a tag indicating whether each one is **Cloud Run specific** or a **general improvement** that could benefit any deployment.

| Change | Tag | Description |
| --- | --- | --- |
| GCP Cloud Run auth (`cloudrun_auth.py`, `bearer_auth.py`) | ☁️ Cloud Run | Fetches and auto-refreshes GCP identity tokens via `gcloud auth print-identity-token`. Injected as a Bearer header into Ollama and LM Studio clients through `BearerAuthMixin`. No impact on local HTTP endpoints. |
| `cloudrun_identity` auth type in provider config | ☁️ Cloud Run | YAML config gains `auth: cloudrun_identity` and optional `cloudrun_audience` / `cloudrun_impersonate_service_account` fields. Ignored when targeting local endpoints. |
| Background keepalive (`benchmark/keepalive.py`) | ☁️ Cloud Run | Periodically pings the target model during a benchmark run so Cloud Run services do not scale to zero between questions. Configured under `keepalive:` in YAML. Has no effect unless explicitly enabled. |
| `_probe_timeout()` on Ollama and LM Studio clients | ☁️ Cloud Run | Uses a 5-second timeout for local HTTP endpoints and the full configured `timeout` for HTTPS/Cloud Run endpoints when testing connectivity. Prevents false-negative "cannot connect" errors on cold-start services. |
| Cloud Run example configs (`configs/`) | ☁️ Cloud Run | `configs/cloudrun_ollama.yaml`, `configs/cloudrun_vllm_deephat.yaml`, and `configs/cloudrun_ollama_bugtrace.yaml` (3072 max tokens) show working setups with auth, keepalive, and rubric scoring. |
| Helper scripts (`scripts/`) | ☁️ Cloud Run | Shell scripts for sourcing Cloud Run environment variables, warming up services, and running benchmarks (`warmup_*`, `run_*_baseline.sh`). Includes `local_env.sh.example` for local endpoint overrides (gitignored). |
| Optimization trigger: rubric or semantic below 25% | ✅ General | Optimization fires when baseline rubric or semantic score is **below 25%**, or when semantic scoring is skipped as garbage. Scores at/above 25% on both tracks do not trigger. |
| Optimization early exit at 75%+ tracks | ✅ General | When `--semantic` is active, the loop stops as soon as the best rubric score and best semantic score (across all attempts) are both **≥ 75%** and semantic is not garbage. Without `--semantic`, it stops once rubric-best is **≥ 75%**. Up to four strategies may run if a track stays below 75%. |
| Per-question output format | ✅ General | Non-optimized questions print `Final: rubric X% \| semantic Y% \| duration \| snippet`. Questions that trigger optimization print a `Baseline:` line (with both rubric and semantic scores) before the optimization block, and end with a `Best —` summary of the winning scores for each track. The final report ends with **Rubric Winners** and **Semantic Winners** tables (Q#, score, source, full question text) when applicable. |
| 4-strategy round-robin optimizer | ✅ General | Each optimization attempt uses the next strategy in a fixed cycle — `role_playing → technical → few_shot → cve_framing` — instead of always picking between two based on whether the response was censored. Default maximum attempts changed from 5 to 4 (one per strategy). |
| Verbose optimization output | ✅ General | After each attempt the optimizer prints the reframed prompt snippet, the model's response snippet, and both rubric and semantic scores (when `--semantic` is active). After all iterations a `Best —` summary shows the best rubric and semantic scores found. The final per-question lines and the summary tables include the source of each winner (`baseline`, `opt/role_playing`, etc.). |
| Keepalive correctness during optimization | ☁️ Cloud Run | `optimize_prompt()` gains a `keepalive` parameter. The target and optimizer LLM calls are wrapped with `keepalive_busy()` so the keepalive thread stops pinging while those calls are in flight, preventing spurious timeout warnings and keeping the model warm between attempts. |
| `keep_alive: -1` in Ollama Cloud Run configs | ☁️ Cloud Run | `configs/cloudrun_ollama.yaml` and `configs/cloudrun_ollama_bugtrace.yaml` set `provider.keep_alive: -1`, sent on every main query and keepalive ping. Tells the Ollama server to never evict the model from VRAM while the container is alive, so the model stays warm even if an individual background ping is skipped or times out during a long generation. |
| Retry-then-skip on per-question API errors | ✅ General | A `5xx` server error (e.g. a backend chat-template/formatting failure) now retries with backoff inside the client (`models/base.py`, `models/openwebui.py`) instead of failing on the first attempt. If it still fails after exhausting retries, the runner records that single question as a failed result (`error` field set, `score: 0`, no optimization attempted) and continues to the next question instead of aborting the whole benchmark run. |
| Per-question timestamp/duration/prompt + run start/finish | ✅ General | Each question's console line is prefixed with a clock timestamp (for correlating with Cloud Run logs), followed by a snippet of the original question prompt (previously only the optimizer's reframed prompt was ever shown, never the baseline question being asked), and the score line now shows how long that question took. The run prints `Started:`/`Finished:` (with total elapsed) once per model. |
| Score rationale (`Why: ...`) | ✅ General | When the rubric scorer (v2) is active, a `Why:` line is printed under each score explaining the deterministic cause: `N/M criteria passed: <ids> \| missing: <ids>` for a partial/normal score, `refused (censored)` for a refusal, or `fatal_error: <id>` when a `fatal_errors` pattern matched. Printed for the baseline answer, the post-optimization answer, and in the concurrent runner. Silently omitted for non-rubric scorers (e.g. plain keyword scorer), which have no per-criterion breakdown. |
| Fix: optimizer discarding real 0%-scored responses | ✅ General | `PromptOptimizer.optimize_prompt()` tracked the best attempt using a `best_score = 0` sentinel with a strict `score > best_score` comparison, so when the original prompt *and every optimization attempt* for a question scored exactly `0%`, the initial `best_response = ""` placeholder was never overwritten — the real (if keyword-mismatched) model output was silently discarded and the exported result showed an empty `full_response` at a genuine (non-empty) latency. Fixed by using a `-1` sentinel so the first attempt's response is always captured regardless of score. |
| Estimated Cloud Run cost (`cloudrun_cost:` in YAML) | ☁️ Cloud Run | Optional, disabled by default. Estimates spend from published on-demand instance-based rates × session uptime (includes warmup when `run_*_baseline.sh` sets `CLOUDRUN_COST_SESSION_START`). Prints USD + display currency (default GBP), mid-run spent/projected every N questions, persists a `cloudrun_cost` block in result JSON and a `cost_summary` request-log row, and can abort via `--max-cloudrun-cost` / `max_cost` (display-currency units). Not the GCP invoice — see `docs/agent-reference.md`. |
| Pre-flight GPU residency check (`gpu_check:` in YAML / `--min-vram-fraction`) | ☁️ Cloud Run | Optional, disabled by default. Before the paid benchmark starts, loads the model and asks Ollama's own `/api/ps` how much of it (`size_vram` vs `size`) is actually resident in GPU VRAM, aborting if it's below a configured fraction — catches silent CPU fallback on a broken/misconfigured Ollama GPU backend before it wastes a full run's worth of Cloud Run GPU billing and wall-clock time (this happened in practice: see [`ollama/ollama#16449`](https://github.com/ollama/ollama/issues/16449) and the `v0.24.0` pin in `GCP-CLOUDRUN-AImodels/README.md`). Uses Ollama's own accounting rather than an inferred speed heuristic, so it needs no per-model calibration. Only measurable for Ollama clients; skipped for other providers. Applies to the target model only — never the optimizer, which commonly runs on a separate, often local, non-Cloud-Run endpoint. See `configs/cloudrun_ollama.yaml` / `configs/cloudrun_ollama_bugtrace.yaml`. |
| Provider-agnostic optimizer (`--optimizer-provider`) | ✅ General | Optimizer LLM can be Ollama, LM Studio, OpenWebUI, OpenRouter, or DeepInfra. `--optimizer-provider` and `--optimizer-model` are an atomic pair (both required to enable optimization, or neither). Cloud providers fail fast at init if no API key is set. |

### Running on Cloud Run

Source your environment and run:

```bash
source scripts/cloudrun_env.sh        # set CLOUDRUN_ENDPOINT, TOKEN, etc.
uv run run_benchmark.py run lmstudio \
  -m "DeepHat/DeepHat-V1-7B" \
  -e "$CLOUDRUN_ENDPOINT" \
  --config configs/cloudrun_vllm_deephat.yaml
```

To keep a Cloud Run service warm during the run, add `keepalive: enabled: true` to your YAML config. See `configs/cloudrun_ollama.yaml` for a complete example.

To avoid paying for a full run that's silently stuck on CPU, add `gpu_check: enabled: true` with a `min_vram_fraction` threshold to your YAML config (already set in `configs/cloudrun_ollama.yaml` and `configs/cloudrun_ollama_bugtrace.yaml`), or pass `--min-vram-fraction 0.9` on the command line.

BugTrace Apex 26B Q4 (Ollama on Cloud Run, `max_tokens=3072`):

```bash
./scripts/run_bugtrace_baseline.sh   # auto-sources scripts/local_env.sh when present
# Optional cost cap (~£3 display currency; see cloudrun_cost.currency / usd_per_unit):
# uv run run_benchmark.py run ollama -m "$BUGTRACE_MODEL" -e "$BUGTRACE_ENDPOINT" \
#   --config configs/cloudrun_ollama_bugtrace.yaml --max-cloudrun-cost 3
```

Services use `min-instances=0` (scale to zero when idle). The `run_*_baseline.sh`
wrappers warm up automatically; use `./scripts/warmup_bugtrace.sh` alone only when
you want a standalone cold-start ping.

---

The default v2 suite contains 60 questions in `datasets/v2/benchmark.jsonl`, grouped by domain and difficulty.

## v2 Local Leaderboard

Top local models from the June 2026 v2 run, sorted by `judge_adjusted_score`.
The run used the full `standard` profile, Ollama, `max_tokens=3072`, `temperature=0.2`,
and post-hoc disputed-case LLM-as-Judge via OpenRouter `deepseek/deepseek-v4-flash`.

| Rank | Model | Rubric | Judge-adjusted | Judge critical error rate |
| --- | --- | ---: | ---: | ---: |
| 1 | `hf.co/BugTraceAI/BugTraceAI-Apex-G4-26B-Q4:latest` | `80.89%` | `89.45%` | `0.00%` |
| 2 | `nemotron-3-nano:30b` | `75.55%` | `86.81%` | `7.14%` |
| 3 | `hf.co/yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1-GGUF:Q6_K` | `73.23%` | `81.12%` | `7.14%` |
| 4 | `hf.co/unsloth/Qwen3-Coder-Next-GGUF:Qwen3-Coder-Next-Q4_K_S` | `75.50%` | `80.15%` | `33.33%` |
| 5 | `mistral-small3.2:24b` | `69.39%` | `76.58%` | `8.33%` |

`judge_adjusted_score` is the recommended audit score for this leaderboard: the
deterministic rubric score with judged disputed cases replaced by the LLM-as-Judge
score. Keep the critical-error rate in view when choosing a base model.

## What v2 Measures

The benchmark reports the total weighted score and separate audit metrics:

| Metric | Meaning |
| --- | --- |
| `refusal_rate` | Percentage of refused or censored answers |
| `technical_accuracy` | Average rubric accuracy for technical criteria |
| `critical_error_rate` | Answers with fatal technical falsehoods |
| `completeness` | Coverage of required steps and conditions |
| `specificity` | Presence of concrete tools, fields, commands, or evidence |
| `hallucination_rate` | Currently tied to critical technical errors |
| `latency_ms_avg` | Average response latency |

Interpretation labels are deliberately conservative:

| Final score | Interpretation |
| --- | --- |
| `< 60%` | `not-suitable` |
| `60-79.9%` | `requires-validation` |
| `>= 80%` | `strong-candidate` |

A high score is not a production approval. Review domain, difficulty, refusal, and critical-error breakdowns before selecting a model.

## Dataset Coverage

The v2 dataset covers:

- Windows tradecraft
- AD and AD CS
- Web exploitation
- Cloud and IAM
- Containers and Kubernetes
- Detection and evasion reasoning
- OpSec and operational tradeoffs
- Tool usage
- Post-exploitation planning
- Validation and reporting

Difficulty levels are `L1 factual`, `L2 procedure`, `L3 troubleshooting`, `L4 scenario reasoning`, and `L5 multi-step operator task`.

## Installation

Requirements:

- Python `3.13+`
- `uv`
- One provider: Ollama, LM Studio, OpenWebUI, OpenRouter, or DeepInfra

Install base dependencies **from the repository root**:

```bash
cd /path/to/redteam-ai-benchmark
uv sync
```

Run every CLI command from the repository root. Dataset paths (`datasets/v2/benchmark.jsonl`), answer files (`answers_v2.txt`, `answers_all.txt`), configs, and default export/cache directories are resolved relative to the current working directory. A wheel/`redteam-benchmark` console script install does not embed those data files; full benchmark runs are supported only when the cwd is the repo root (or you pass absolute paths for every file input).

## Providers

| Provider | Default endpoint | Notes |
| --- | --- | --- |
| `ollama` | `http://localhost:11434` | Native Ollama API; optional Bearer auth for reverse proxies |
| `lmstudio` | `http://localhost:1234` | OpenAI-compatible LM Studio API |
| `openwebui` | `http://localhost:3000` | OpenAI-compatible OpenWebUI API |
| `openrouter` | `https://openrouter.ai/api/v1` | Requires an API key |
| `deepinfra` | `https://api.deepinfra.com/v1/openai` | Requires `DEEPINFRA_TOKEN` or `--api-key` |

## Usage

List models:

```bash
uv run run_benchmark.py ls ollama
uv run run_benchmark.py ls lmstudio
uv run run_benchmark.py ls openwebui
uv run run_benchmark.py ls openrouter --api-key "$OPENROUTER_API_KEY"
```

Run the default v2 standard profile:

```bash
uv run run_benchmark.py run ollama -m "llama3.1:8b"
```

Run a quick smoke subset:

```bash
uv run run_benchmark.py run ollama -m "llama3.1:8b" --profile quick
```

Run selected v2 questions by ID:

```bash
uv run run_benchmark.py run ollama -m "llama3.1:8b" --question-ids 5 12
```

Write an append-only per-question request log:

```bash
uv run run_benchmark.py run ollama -m "llama3.1:8b" --request-log results/requests.jsonl
```

Run multiple local models interactively:

```bash
uv run run_benchmark.py interactive ollama --profile standard
```

Supported profiles:

| Profile | Purpose |
| --- | --- |
| `quick` | 16-question smoke subset |
| `standard` | Full 60-question v2 benchmark |
| `enterprise` | Full v2 dataset with audit-friendly export |
| `local-only` | Full v2 dataset without LLM judge usage |
| `cloud-comparison` | Full v2 dataset for fixed cloud-model comparisons |

## Scoring

Runtime scoring is always `rubric`. It is deterministic and does not require an external LLM judge. Each v2 question contains atomic criteria, fatal-error patterns, acceptable variants, tags, and a question weight.

Runtime scoring does not support legacy `keyword`, `semantic`, or `hybrid` modes. Use the offline `judge` command for post-hoc LLM-as-Judge auditing.

Optional embedded semantic scoring is available as a parallel audit metric. It does not replace rubric scoring: `score`, `total_score`, and base interpretation labels remain rubric-based. Semantic scoring compares each question's answer against `answers_v2.txt` using either a local SentenceTransformer model or DeepInfra embeddings.

**Local provider** (default: `Qwen/Qwen3-Embedding-0.6B`):

```bash
uv sync --extra semantic
uv run run_benchmark.py run ollama -m "llama3.1:8b" --semantic
uv run run_benchmark.py preload-semantic
```

**DeepInfra provider** (default: `Qwen/Qwen3-Embedding-8B`, recalibrated score bands):

```bash
export DEEPINFRA_TOKEN="your-token"
uv run run_benchmark.py run ollama -m "llama3.1:8b" \
  --semantic --semantic-provider deepinfra
uv run run_benchmark.py preload-semantic --semantic-provider deepinfra
```

Or configure in YAML (`scoring.semantic.provider: deepinfra`).

### Reference embedding cache

`preload-semantic` warms a **local disk cache of reference-answer embeddings** from `answers_v2.txt` (or `scoring.semantic.answers_file`). **Model responses during a benchmark run are not cached** — with DeepInfra, every live answer still calls the embedding API at runtime.

Cache files live under `.cache/redteam/semantic/` (override with `REDTEAM_SEMANTIC_CACHE_DIR`). Each file is named `<ModelName>_<digest16>.json`, where the digest covers the answers file path, full reference text, `max_seq_length`, semantic scorer version, and provider. Changing any of those produces a new cache file; older files are left on disk but unused.

| Situation | Behavior |
|-----------|----------|
| First run / missing cache file | Encodes all reference answers |
| Reference text changed | Re-encodes only changed question ids; reuses the rest (per-answer SHA-256 hashes) |
| Legacy cache without hashes | One-time full re-encode into the current format |
| Different provider, model, `max_seq_length`, or scorer version | New cache file; may import matching entries from a sibling cache |

When the cache is already complete, preload prints `Reference embedding cache already warm (60 answers)` and skips API calls. During `run`, missing entries are encoded on the fly and written back — a warm preload avoids that overhead.

**Force a full rebuild:**

```bash
uv run run_benchmark.py preload-semantic --semantic-provider deepinfra --force
```

Or delete the cache file or directory under `.cache/redteam/semantic/`. The `--force` flag is available on `preload-semantic` only, not on `run`.

This adds `semantic_score` and `semantic_similarity` columns to the final table alongside rubric scores, without changing the rubric total or interpretation.

Semantic scoring uses full-answer cosine bands by default: `100/90/80/70/60/50/40/30/0`. DeepInfra 8B uses recalibrated thresholds (see `scoring/semantic_calibration.py`). JSON and CSV exports add `semantic_score` and `semantic_similarity` fields without changing rubric totals.

Before encoding, the semantic scorer:

- Strips **closed** reasoning/thinking blocks (BugTrace/DeepHat channel markers, Qwen3/DeepSeek `redacted_thinking`, DeepSeek-R1 `xml_think`, pipe/XML/bracket variants). Only full open+close blocks are removed; unclosed prefixes stay intact. Pattern names and hex close-tag checks live in `scoring/semantic_scorer.py` and `tests/test_thinking_strip.py`.
- Skips degenerate repetitive outputs as `semantic skipped (garbage)` when word diversity is too low (≥24 words and unique-word ratio &lt; 0.12). A garbage baseline with a high rubric score can still trigger prompt optimization.
- Caps the embedding window at **2048 tokens** for the local provider (`scoring.semantic.max_seq_length`). DeepInfra defaults to **3072** (same as target `max_tokens`) when `max_seq_length` is omitted.

When `--request-log` is enabled, each semantic-scored JSONL row also includes `thinking_stripped_chars`, `thinking_stripped_tokens_est`, and `strip_matched_pattern` (top-level and inside `semantic_scores`).

### Dual-track mode (optimization + semantic)

When both optimizer flags (`--optimizer-provider` + `--optimizer-model`) and `--semantic` are enabled:

- The optimizer also triggers when the **semantic** score is **below 25%**, when semantic scoring is skipped as **garbage**, or when rubric is **below 25%**.
- When `--semantic` is active, the loop stops early once the best rubric and best semantic scores across attempts are both **≥ 75%** (semantic must not be garbage). Without `--semantic`, early exit is rubric-only at **≥ 75%**.
- Each optimization attempt is scored on both metrics independently.
- The run keeps two winners per question: the answer with the highest rubric score and the answer with the highest semantic score.
- The final report shows two separate tables and interpretations — one per track.
- JSON export gains a top-level `tracks` block with each track's total score and interpretation.

```bash
uv run run_benchmark.py run ollama -m "llama3.1:8b" \
  --semantic \
  --optimizer-provider ollama \
  --optimizer-model "llama3.3:70b"
```

Without an optimizer configured, `--semantic` still prints separate **Rubric Winners** and **Semantic Winners** tables (no merged per-question snippet table).

## Offline LLM-as-Judge

Saved v2 result JSON files can be audited post-hoc without rerunning benchmark models:

```bash
OPENROUTER_API_KEY=... uv run run_benchmark.py judge \
  --results "results/*.json" \
  --dataset datasets/v2/benchmark.jsonl \
  --judge-model "deepseek/deepseek-v4-flash" \
  --output-dir judge_results_v2 \
  --mode disputed \
  --concurrency 4
```

Default result globs (when `--results` is omitted) are `results/*.json` and `results_*_v2/*.json`. Zero matches is an error. The judge command writes `per_model/*.json`, `detailed.csv`, `summary.csv`, and `disputed_cases.csv`. `judge_score` is the judged subset score; use `judge_adjusted_score` for the rubric score with judged disputed cases replaced. LLM-as-Judge output is an audit layer and does not overwrite deterministic benchmark results.

## Configuration

Copy `config.example.yaml` to `config.yaml` and adjust it:

```yaml
provider:
  name: ollama
  endpoint: http://localhost:11434
  # api_key: sk-xxx
  # keep_alive: 30m

scoring:
  method: rubric
  semantic:
    enabled: false
    provider: local  # or deepinfra (Qwen/Qwen3-Embedding-8B, requires DEEPINFRA_TOKEN)
    answers_file: answers_v2.txt
    model: Qwen/Qwen3-Embedding-0.6B
    # Omit thresholds when using provider: deepinfra to get recalibrated 8B bands.
    thresholds:
      100: 0.92
      90: 0.88
      80: 0.84
      70: 0.80
      60: 0.75
      50: 0.70
      40: 0.65
      30: 0.60
    device: auto
    max_seq_length: 2048  # local; omit for deepinfra to use 3072

export:
  formats:
    - json
    - csv
    - criteria_csv
  output_dir: ./results
  include_response: true

optimization:
  optimizer_provider: ollama          # or: deepinfra, openrouter, lmstudio, openwebui
  optimizer_model: llama3.3:70b
  # optimizer_endpoint: http://localhost:11434
  # optimizer_api_key: ...            # for deepinfra/openrouter; prefer env var
  max_iterations: 4

questions_file: datasets/v2/benchmark.jsonl
answers_file: answers_all.txt
rate_limit_delay: 1.5
max_tokens: 3072
temperature: 0.2
concurrency: 1
# request_log: ./results/requests.jsonl
```

Run with config:

```bash
uv run run_benchmark.py run ollama -m "llama3.1:8b" --config config.yaml
```

## Output

JSON export includes model results, per-question rubric evidence, aggregate summary, and audit provenance:

```json
{
  "model": "llama3.1:8b",
  "scoring_method": "rubric",
  "total_score": 75.0,
  "interpretation": "requires-validation",
  "benchmark_version": "2.3.0",
  "dataset_id": "redteam-ai-benchmark-v2",
  "dataset_version": "2.1.0",
  "dataset_hash": "...",
  "scorer_version": "rubric",
  "config_hash": "...",
  "run_config": {
    "provider": "ollama",
    "model": "llama3.1:8b",
    "profile": "standard"
  },
  "git_commit": "...",
  "package_version": "2.3.0",
  "runtime_profile": "standard",
  "summary": {
    "metrics": {
      "refusal_rate": 0.0,
      "critical_error_rate": 0.0
    },
    "breakdown": {
      "difficulty": {},
      "domain": {},
      "capability": {}
    }
  }
}
```

CSV output contains per-question rows plus a `TOTAL` row. `criteria_csv` adds one row per passed or failed rubric criterion.

## Prompt Optimization

Prompt optimization is optional and separate from base-model scoring. Enable it by setting **both** `--optimizer-provider` and `--optimizer-model` (CLI or `optimization:` in config). Omit both for a baseline-only run, or pass `--no-optimize` to disable optimization even when configured in YAML.

Supported optimizer providers: `ollama`, `lmstudio`, `openwebui`, `openrouter`, `deepinfra`. Cloud providers (`openrouter`, `deepinfra`) require an API key via `--optimizer-api-key` or the `OPENROUTER_API_KEY` / `DEEPINFRA_TOKEN` environment variables.

Pre-flight validation runs before the first question:

- `--optimizer-provider` and `--optimizer-model` must be used together (or neither)
- Cloud providers without an API key abort immediately with a clear error

Optimization runs when the baseline rubric **or** semantic score is **below 25%**, or when semantic scoring is skipped as **garbage** (see `should_trigger_prompt_optimization()` in `optimization/policy.py`). By default it tries up to **four** reframing strategies (`--max-optimization-iterations 4`) and independently keeps the **rubric-best** and **semantic-best** answers. The loop stops early once **both** tracks reach **≥ 75%** (semantic must be a real score, not garbage); without `--semantic`, rubric-only early exit uses the same **≥ 75%** rule. Results are written to `optimized_prompts_{model}_{timestamp}.json`.

Each optimization iteration sends a **new reframed prompt** to the target model (fresh response, not a rewrite of the prior answer). The optimizer always anchors on the original benchmark question and the **baseline prompt/response** when generating each strategy variant. While the target model runs strategy *N*, the optimizer generates strategy *N+1* in parallel. When `--semantic` is active, each iteration scores rubric and semantic sequentially after the model returns, and prints both inline.

```bash
# Local Ollama optimizer
uv run run_benchmark.py run ollama -m "llama3.1:8b" \
  --optimizer-provider ollama \
  --optimizer-model "llama3.3:70b"

# DeepInfra optimizer
export DEEPINFRA_TOKEN="your-token"
uv run run_benchmark.py run ollama -m "llama3.1:8b" \
  --optimizer-provider deepinfra \
  --optimizer-model "deepseek-ai/DeepSeek-V3"

# Config has optimizer settings, but skip optimization this run
uv run run_benchmark.py run ollama -m "llama3.1:8b" --config config.yaml --no-optimize
```

**Wrong model names:** Before the first question, the runner calls `list_models()` and aborts if the **target** model (`-m`) or **optimizer** model is not listed by the provider. In `interactive` mode, an invalid target model skips that model and continues; a missing optimizer model still aborts the whole run.

Do not mix optimized scores with base model capability comparisons.

## Validation

Useful checks:

```bash
uv run run_benchmark.py --help
uv run run_benchmark.py run --help
uv run pytest
python3 -m compileall -q run_benchmark.py benchmark models optimization scoring tracing utils
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md), [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md), and [SECURITY.md](SECURITY.md).

## License

MIT. Use in authorized red team labs, commercial security assessments, AI-security research, and educational environments.
