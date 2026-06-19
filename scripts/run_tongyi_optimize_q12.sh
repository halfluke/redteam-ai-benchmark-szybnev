#!/usr/bin/env bash
# Tongyi Cloud Run + local Qwen optimizer — Q12 only (Phishing_Lure).
# Preflight: Cloud Run warmup + optimizer ping (keep_alive 30m).
# During run: benchmark keepalive pings idle model every 60s.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# shellcheck source=scripts/optimizer_env.sh
source "$ROOT/scripts/optimizer_env.sh"

if [[ "${SKIP_PREFLIGHT:-}" != "1" ]]; then
  "$ROOT/scripts/preflight_optimize.sh" tongyi
  echo
fi

export OPTIMIZER_ENDPOINT OPTIMIZER_MODEL

uv run run_benchmark.py run ollama \
  -m tongyi-deepresearch-iq2s \
  --config configs/cloudrun_ollama_optimize.yaml \
  --question-ids 12 \
  --optimizer-model "$OPTIMIZER_MODEL" \
  --request-log ./results/request_log_tongyi_optimize_q12.jsonl
