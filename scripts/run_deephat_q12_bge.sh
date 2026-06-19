#!/usr/bin/env bash
# DeepHat Cloud Run — Q12 baseline with BAAI/bge-base-en-v1.5 semantic model.
# Purpose: compare semantic scores vs Qwen3-Embedding-0.6B on the same question.
# No optimizer; uses cloudrun_vllm_deephat.yaml (multi-score: keyword+semantic+hybrid).
# Preflight: ./scripts/warmup_deephat.sh  (optional; in-process keepalive still runs)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# shellcheck source=scripts/cloudrun_env.sh
source "$ROOT/scripts/cloudrun_env.sh"

if [[ "${SKIP_PREFLIGHT:-}" != "1" ]]; then
  "$ROOT/scripts/warmup_deephat.sh"
  echo
fi

uv run run_benchmark.py run lmstudio \
  -m "$DEEPHAT_MODEL" \
  --config configs/cloudrun_vllm_deephat.yaml \
  --question-ids 12 \
  --semantic-model "BAAI/bge-base-en-v1.5" \
  --request-log ./results/request_log_deephat_q12_bge_base.jsonl
