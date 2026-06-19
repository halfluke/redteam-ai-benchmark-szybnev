#!/usr/bin/env bash
# Warm up DeepHat vLLM on Cloud Run (cold-start ping). Matches benchmark keepalive target role.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/cloudrun_env.sh
source "$ROOT/scripts/cloudrun_env.sh"
# shellcheck source=scripts/lib_common.sh
source "$ROOT/scripts/lib_common.sh"

require_cmd curl python3 gcloud
check_gcloud_login

echo "== Warmup: DeepHat (Cloud Run) =="
echo "   endpoint: $DEEPHAT_ENDPOINT"
echo "   model:    $DEEPHAT_MODEL"

TOKEN="$(cloudrun_token)"
if ! resp="$(ping_lmstudio_chat \
  "$DEEPHAT_ENDPOINT" "$DEEPHAT_MODEL" "$TOKEN" \
  "$PING_PROMPT" "$PING_MAX_TOKENS" "$PING_TIMEOUT_S")"; then
  print_ping_fail "target"
  die "DeepHat warmup failed (timeout ${PING_TIMEOUT_S}s or HTTP error)"
fi

print_ping_ok "target"
python3 - <<PY "$resp"
import json, sys
data = json.loads(sys.argv[1])
choices = data.get("choices") or []
if choices:
    msg = (choices[0].get("message") or {}).get("content", "")
    preview = msg.strip().replace("\n", " ")[:80]
    if preview:
        print(f"   reply: {preview}")
PY

echo "OK — DeepHat warm."
