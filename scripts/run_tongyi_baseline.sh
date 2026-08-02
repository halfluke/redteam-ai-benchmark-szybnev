#!/usr/bin/env bash
# Tongyi Cloud Run — full v2 rubric benchmark (60 questions).
# Auth: automatic Cloud Run identity token via gcloud.
# Keepalive pings the service every 60s between questions.
#
# Usage:
#   ./scripts/run_tongyi_baseline.sh
#   # optional: scripts/local_env.sh is auto-sourced when present
#   SKIP_PREFLIGHT=1 ./scripts/run_tongyi_baseline.sh   # skip built-in warmup
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# shellcheck source=lib_common.sh
source "$ROOT/scripts/lib_common.sh"
source_local_env_if_present "$ROOT"
# shellcheck source=cloudrun_env.sh
source "$ROOT/scripts/cloudrun_env.sh"

require_non_placeholder_endpoint "Tongyi" "$TONGYI_ENDPOINT"
ensure_cloudrun_cost_session_start

if [[ "${SKIP_PREFLIGHT:-}" != "1" ]]; then
  "$ROOT/scripts/warmup_tongyi.sh"
  load_cloudrun_cost_warmup_env "$ROOT"
  echo
fi

uv run run_benchmark.py run ollama \
  -m "$TONGYI_MODEL" \
  -e "$TONGYI_ENDPOINT" \
  --config configs/cloudrun_ollama.yaml
