#!/usr/bin/env bash
# Pre-benchmark checks: gcloud auth, Cloud Run target warmup, local Qwen optimizer warmup.
# Usage: ./scripts/preflight_optimize.sh tongyi|deephat
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/lib_common.sh
source "$ROOT/scripts/lib_common.sh"

TARGET="${1:-tongyi}"

case "$TARGET" in
  tongyi)
    "$ROOT/scripts/warmup_tongyi.sh"
    ;;
  deephat)
    "$ROOT/scripts/warmup_deephat.sh"
    ;;
  *)
    die "Usage: $0 tongyi|deephat"
    ;;
esac

"$ROOT/scripts/warmup_optimizer.sh"

echo
echo "Preflight complete — benchmark will run keepalive (warmup + every 60s on idle model)."
