# Shared shell helpers for Cloud Run + Ollama scripts.

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_cmd() {
  local cmd
  for cmd in "$@"; do
    command -v "$cmd" >/dev/null 2>&1 || die "Missing required command: $cmd"
  done
}

cloudrun_token() {
  require_cmd gcloud
  gcloud auth print-identity-token
}

check_gcloud_login() {
  require_cmd gcloud
  if ! gcloud auth print-identity-token >/dev/null 2>&1; then
    die "gcloud not logged in. Run: gcloud auth login"
  fi
}

curl_http_code() {
  curl -sS -o /dev/null -w "%{http_code}" "$@"
}

ping_ollama_chat() {
  local endpoint="$1"
  local model="$2"
  local token="${3:-}"
  local keep_alive="${4:-}"
  local prompt="${5:-Say OK}"
  local max_tokens="${6:-16}"
  local timeout_s="${7:-90}"

  local auth_header=()
  if [[ -n "$token" ]]; then
    auth_header=(-H "Authorization: Bearer $token")
  fi

  local payload
  payload="$(python3 - <<PY
import json
payload = {
    "model": "$model",
    "messages": [{"role": "user", "content": """$prompt"""}],
    "stream": False,
    "options": {"temperature": 0, "num_predict": int("$max_tokens")},
}
keep = """$keep_alive"""
if keep:
    payload["keep_alive"] = keep
print(json.dumps(payload))
PY
)"

  curl -sf --max-time "$timeout_s" \
    "${auth_header[@]}" \
    -H "Content-Type: application/json" \
    -X POST "$endpoint/api/chat" \
    -d "$payload"
}

ping_lmstudio_chat() {
  local endpoint="$1"
  local model="$2"
  local token="${3:-}"
  local prompt="${4:-Say OK}"
  local max_tokens="${5:-16}"
  local timeout_s="${6:-90}"

  local auth_header=()
  if [[ -n "$token" ]]; then
    auth_header=(-H "Authorization: Bearer $token")
  fi

  local payload
  payload="$(python3 - <<PY
import json
print(json.dumps({
    "model": """$model""",
    "messages": [{"role": "user", "content": """$prompt"""}],
    "temperature": 0,
    "max_tokens": int("$max_tokens"),
    "stream": False,
}))
PY
)"

  curl -sf --max-time "$timeout_s" \
    "${auth_header[@]}" \
    -H "Content-Type: application/json" \
    -X POST "$endpoint/v1/chat/completions" \
    -d "$payload"
}

print_ping_ok() {
  local role="$1"
  echo "   Warmup ping ($role): ok"
}

print_ping_fail() {
  local role="$1"
  echo "   Warmup ping ($role): failed" >&2
}
