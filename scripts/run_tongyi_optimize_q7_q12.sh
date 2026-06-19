#!/usr/bin/env bash
# Tongyi Cloud Run + local Qwen optimizer — Q7 + Q12 subset.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
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
  --question-ids 7,12 \
  --optimizer-model "$OPTIMIZER_MODEL" \
  --request-log ./results/request_log_tongyi_optimize_q7_q12.jsonl
