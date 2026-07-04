#!/usr/bin/env bash
# DeepHat Cloud Run vLLM — full v2 rubric benchmark (60 questions).
# Auth: automatic Cloud Run identity token via gcloud.
# Keepalive pings the service every 60s between questions.
#
# Usage:
#   source scripts/local_env.sh   # optional: override endpoint/model
#   ./scripts/run_deephat_baseline.sh
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
  -e "$DEEPHAT_ENDPOINT" \
  --config configs/cloudrun_vllm_deephat.yaml
