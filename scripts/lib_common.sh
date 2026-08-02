# Shared shell helpers for Cloud Run + Ollama scripts.
# shellcheck shell=bash

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

# Source scripts/local_env.sh when present (gitignored overrides).
source_local_env_if_present() {
  local root="$1"
  local path="$root/scripts/local_env.sh"
  if [[ -f "$path" ]]; then
    # shellcheck source=/dev/null
    source "$path"
  fi
}

require_non_placeholder_endpoint() {
  local label="$1"
  local endpoint="$2"
  if [[ -z "$endpoint" ]] || cloudrun_is_placeholder_endpoint "$endpoint"; then
    die "$label endpoint is unset or still a placeholder ($endpoint). Set it in scripts/local_env.sh (from local_env.sh.example) or export it before running."
  fi
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

# --- Cloud Run endpoint helpers ---

# Map short Cloud Run URL region codes (legacy *.a.run.app) to region IDs.
cloudrun_region_from_code() {
  case "$1" in
    ew) echo "europe-west1" ;;
    ew2) echo "europe-west2" ;;
    ew3) echo "europe-west3" ;;
    ew4) echo "europe-west4" ;;
    uc) echo "us-central1" ;;
    ue) echo "us-east1" ;;
    uw) echo "us-west1" ;;
    as) echo "asia-southeast1" ;;
    an) echo "asia-northeast1" ;;
    *) return 1 ;;
  esac
}

# Parse service name (and optional region) from a Cloud Run HTTPS URL.
# Sets: _PARSED_SERVICE, _PARSED_REGION (region may be empty).
cloudrun_parse_endpoint() {
  local endpoint="$1"
  local host
  _PARSED_SERVICE=""
  _PARSED_REGION=""
  host="$(python3 - <<PY
from urllib.parse import urlparse
print(urlparse("""$endpoint""").hostname or "")
PY
)"
  [[ -n "$host" ]] || return 1

  # Legacy first: SERVICE-HASH-REGIONCODE.a.run.app
  # (Must precede the modern pattern — "*.a.run.app" also matches
  #  SERVICE-HASH.a.run.app with a false region of "a".)
  if [[ "$host" =~ ^(.+)-[a-z0-9]+-([a-z0-9]+)\.a\.run\.app$ ]]; then
    _PARSED_SERVICE="${BASH_REMATCH[1]}"
    _PARSED_REGION="$(cloudrun_region_from_code "${BASH_REMATCH[2]}" || true)"
    return 0
  fi

  # Modern: SERVICE-PROJECTHASH.REGION.run.app (region like europe-west1)
  if [[ "$host" =~ ^([a-z0-9-]+)-[a-z0-9]+\.([a-z0-9-]+)\.run\.app$ ]]; then
    _PARSED_SERVICE="${BASH_REMATCH[1]}"
    _PARSED_REGION="${BASH_REMATCH[2]}"
    return 0
  fi

  return 1
}

cloudrun_is_placeholder_endpoint() {
  local endpoint="$1"
  [[ "$endpoint" == *YOUR-* || "$endpoint" == *SERVICE-HASH* || "$endpoint" == *example* || "$endpoint" == *OPTIMIZER-LAN-IP* ]]
}

# Resolve service + region for a logical name (bugtrace|deephat|tongyi).
# Uses NAME_SERVICE / CLOUDRUN_REGION overrides, else parses NAME_ENDPOINT.
cloudrun_resolve_target() {
  local name="$1"
  local endpoint_var service_var
  local endpoint service region

  case "$name" in
    bugtrace)
      endpoint_var=BUGTRACE_ENDPOINT
      service_var=BUGTRACE_SERVICE
      ;;
    deephat)
      endpoint_var=DEEPHAT_ENDPOINT
      service_var=DEEPHAT_SERVICE
      ;;
    tongyi)
      endpoint_var=TONGYI_ENDPOINT
      service_var=TONGYI_SERVICE
      ;;
    *)
      die "Unknown Cloud Run target: $name"
      ;;
  esac

  endpoint="${!endpoint_var:-}"
  service="${!service_var:-}"
  region="${CLOUDRUN_REGION:-}"

  if [[ -z "$service" || -z "$region" ]]; then
    if [[ -n "$endpoint" ]] && ! cloudrun_is_placeholder_endpoint "$endpoint"; then
      if cloudrun_parse_endpoint "$endpoint"; then
        service="${service:-$_PARSED_SERVICE}"
        region="${region:-$_PARSED_REGION}"
      fi
    fi
  fi

  region="${region:-europe-west1}"

  if [[ -z "$service" ]]; then
    return 1
  fi
  if cloudrun_is_placeholder_endpoint "${endpoint:-}"; then
    return 1
  fi

  _RESOLVED_SERVICE="$service"
  _RESOLVED_REGION="$region"
  _RESOLVED_ENDPOINT="$endpoint"
  return 0
}

# Sidecar written by warmup_*.sh; sourced by run_*_baseline.sh for cost tracking.
cloudrun_cost_warmup_env_path() {
  local root="${1:-}"
  if [[ -z "$root" ]]; then
    root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  fi
  echo "$root/.cache/redteam/cloudrun_cost_warmup.env"
}

write_cloudrun_cost_warmup_seconds() {
  local root="$1"
  local seconds="$2"
  local env_path
  env_path="$(cloudrun_cost_warmup_env_path "$root")"
  mkdir -p "$(dirname "$env_path")"
  printf 'CLOUDRUN_COST_WARMUP_SECONDS=%s\n' "$seconds" >"$env_path"
  export CLOUDRUN_COST_WARMUP_SECONDS="$seconds"
  echo "   warmup duration: ${seconds}s (for Cloud Run cost estimate)"
}

load_cloudrun_cost_warmup_env() {
  local root="$1"
  local env_path
  env_path="$(cloudrun_cost_warmup_env_path "$root")"
  if [[ -f "$env_path" ]]; then
    # shellcheck source=/dev/null
    source "$env_path"
    export CLOUDRUN_COST_WARMUP_SECONDS
  fi
}

ensure_cloudrun_cost_session_start() {
  if [[ -z "${CLOUDRUN_COST_SESSION_START:-}" ]]; then
    CLOUDRUN_COST_SESSION_START="$(date +%s)"
  fi
  export CLOUDRUN_COST_SESSION_START
}
