#!/usr/bin/env bash
# Tongyi Cloud Run — full v2 rubric benchmark (60 questions).
# Auth: automatic Cloud Run identity token via gcloud.
# Keepalive pings the service every 60s between questions.
#
# Usage:
#   source scripts/local_env.sh   # optional: override endpoint/model
#   ./scripts/run_tongyi_baseline.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# shellcheck source=scripts/cloudrun_env.sh
source "$ROOT/scripts/cloudrun_env.sh"

if [[ "${SKIP_PREFLIGHT:-}" != "1" ]]; then
  "$ROOT/scripts/warmup_tongyi.sh"
  echo
fi

uv run run_benchmark.py run ollama \
  -m "$TONGYI_MODEL" \
  -e "$TONGYI_ENDPOINT" \
  --config configs/cloudrun_ollama.yaml
