#!/usr/bin/env bash
# DeepHat Cloud Run + local Qwen optimizer — Q7 + Q12 subset.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# shellcheck source=scripts/optimizer_env.sh
source "$ROOT/scripts/optimizer_env.sh"

if [[ "${SKIP_PREFLIGHT:-}" != "1" ]]; then
  "$ROOT/scripts/preflight_optimize.sh" deephat
  echo
fi

export OPTIMIZER_ENDPOINT OPTIMIZER_MODEL

uv run run_benchmark.py run lmstudio \
  -m "DeepHat/DeepHat-V1-7B" \
  --config configs/cloudrun_vllm_deephat_optimize.yaml \
  --question-ids 7,12 \
  --optimizer-model "$OPTIMIZER_MODEL" \
  --request-log ./results/request_log_deephat_optimize_q7_q12.jsonl
