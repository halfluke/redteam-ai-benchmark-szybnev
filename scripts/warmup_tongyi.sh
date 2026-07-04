#!/usr/bin/env bash
# Warm up Tongyi on Cloud Run (cold-start ping). Plain chat ping — no Ollama keep_alive on target.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/cloudrun_env.sh
source "$ROOT/scripts/cloudrun_env.sh"
# shellcheck source=scripts/lib_common.sh
source "$ROOT/scripts/lib_common.sh"

require_cmd curl python3 gcloud
check_gcloud_login

echo "== Warmup: Tongyi (Cloud Run) =="
echo "   endpoint: $TONGYI_ENDPOINT"
echo "   model:    $TONGYI_MODEL"

TOKEN="$(cloudrun_token)"
if ! resp="$(ping_ollama_chat \
  "$TONGYI_ENDPOINT" "$TONGYI_MODEL" "$TOKEN" "" \
  "$PING_PROMPT" "$PING_MAX_TOKENS" "$PING_TIMEOUT_S")"; then
  print_ping_fail "target"
  die "Tongyi warmup failed (timeout ${PING_TIMEOUT_S}s or HTTP error)"
fi

print_ping_ok "target"
python3 - <<PY "$resp"
import json, sys
data = json.loads(sys.argv[1])
msg = data.get("message", {}).get("content", "")
if msg:
    preview = msg.strip().replace("\n", " ")[:80]
    print(f"   reply: {preview}")
PY

echo "OK — Tongyi warm."
