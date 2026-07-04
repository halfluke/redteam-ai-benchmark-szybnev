# Cloud Run target endpoints (match configs/cloudrun_*.yaml).
# Source: source "$(dirname "$0")/cloudrun_env.sh"

TONGYI_ENDPOINT="${TONGYI_ENDPOINT:-https://YOUR-OLLAMA-SERVICE-HASH.a.run.app}"
TONGYI_MODEL="${TONGYI_MODEL:-your-ollama-model-id}"

DEEPHAT_ENDPOINT="${DEEPHAT_ENDPOINT:-https://YOUR-VLLM-SERVICE-HASH.a.run.app}"
DEEPHAT_MODEL="${DEEPHAT_MODEL:-YourOrg/YourModel}"

# Ping settings for keepalive / warmup scripts.
# OLLAMA_KEEP_ALIVE is used only for local optimizer warmup (not Cloud Run target).
PING_PROMPT="${PING_PROMPT:-Say OK}"
PING_MAX_TOKENS="${PING_MAX_TOKENS:-16}"
PING_TIMEOUT_S="${PING_TIMEOUT_S:-90}"
OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-30m}"
