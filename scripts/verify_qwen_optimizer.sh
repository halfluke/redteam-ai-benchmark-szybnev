#!/usr/bin/env bash
# Quick connectivity check for local Qwen optimizer (alias for warmup_optimizer).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT/scripts/warmup_optimizer.sh"
