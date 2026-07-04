# Shared local Ollama optimizer settings (Windows host).
# Source from other scripts:  source "$(dirname "$0")/optimizer_env.sh"

OPTIMIZER_ENDPOINT="${OPTIMIZER_ENDPOINT:-http://OPTIMIZER-LAN-IP:11434}"
# Match the name from `ollama list` on Windows after: ollama pull qwen2.5:7b
OPTIMIZER_MODEL="${OPTIMIZER_MODEL:-qwen2.5:7b}"
