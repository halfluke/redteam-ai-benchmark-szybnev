#!/usr/bin/env bash
# Tongyi Cloud Run baseline multi-score (keyword + semantic + hybrid).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ "${SKIP_PREFLIGHT:-}" != "1" ]]; then
  "$ROOT/scripts/warmup_tongyi.sh"
  echo
fi

uv run run_benchmark.py run ollama \
  -m tongyi-deepresearch-iq2s \
  --config configs/cloudrun_ollama.yaml \
  --request-log ./results/request_log_tongyi_multi.jsonl
