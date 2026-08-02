#!/usr/bin/env bash
# Warm up local Windows Ollama optimizer (Qwen) with keep_alive. Matches benchmark keepalive optimizer role.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=lib_common.sh
source "$ROOT/scripts/lib_common.sh"
source_local_env_if_present "$ROOT"
# shellcheck source=cloudrun_env.sh
source "$ROOT/scripts/cloudrun_env.sh"
# shellcheck source=optimizer_env.sh
source "$ROOT/scripts/optimizer_env.sh"

require_cmd curl python3
require_non_placeholder_endpoint "Optimizer" "$OPTIMIZER_ENDPOINT"

echo "== Warmup: local optimizer (Ollama) =="
echo "   endpoint:   $OPTIMIZER_ENDPOINT"
echo "   model:      $OPTIMIZER_MODEL"
echo "   keep_alive: $OLLAMA_KEEP_ALIVE"

tags_json="$(curl -sf "$OPTIMIZER_ENDPOINT/api/tags")"
resolved="$(python3 - <<PY "$tags_json" "$OPTIMIZER_MODEL"
import json, sys
data = json.loads(sys.argv[1])
want = sys.argv[2].lower()
for m in data.get("models", []):
    n = m.get("name", "")
    base = n.split(":")[0].lower()
    if n.lower() == want or n.lower().startswith(want + ":") or base == want.split(":")[0]:
        print(n)
        break
PY
)"

if [[ -z "$resolved" ]]; then
  echo "   Installed models:"
  python3 - <<PY "$tags_json"
import json, sys
for m in json.loads(sys.argv[1]).get("models", []):
    print("    ", m.get("name", m))
PY
  die "'$OPTIMIZER_MODEL' not found. On Windows: ollama pull qwen2.5:7b"
fi

if [[ "$resolved" != "$OPTIMIZER_MODEL" ]]; then
  echo "   resolved tag: $resolved"
  OPTIMIZER_MODEL="$resolved"
fi

if ! resp="$(ping_ollama_chat \
  "$OPTIMIZER_ENDPOINT" "$OPTIMIZER_MODEL" "" "$OLLAMA_KEEP_ALIVE" \
  "$PING_PROMPT" "$PING_MAX_TOKENS" "$PING_TIMEOUT_S")"; then
  print_ping_fail "optimizer"
  die "Optimizer warmup failed (timeout ${PING_TIMEOUT_S}s or HTTP error)"
fi

print_ping_ok "optimizer"
python3 - <<PY "$resp"
import json, sys
data = json.loads(sys.argv[1])
msg = data.get("message", {}).get("content", "")
if msg:
    preview = msg.strip().replace("\n", " ")[:80]
    print(f"   reply: {preview}")
PY

export OPTIMIZER_MODEL
echo "OK — optimizer warm."
