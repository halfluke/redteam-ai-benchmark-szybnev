#!/usr/bin/env bash
# Shared local Ollama optimizer settings (Windows host).
# Usage: source scripts/optimizer_env.sh

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Source this file; do not execute it directly:" >&2
  echo "  source scripts/optimizer_env.sh" >&2
  exit 1
fi

OPTIMIZER_ENDPOINT="${OPTIMIZER_ENDPOINT:-http://OPTIMIZER-LAN-IP:11434}"
# Match the name from `ollama list` on Windows after: ollama pull qwen2.5:7b
OPTIMIZER_MODEL="${OPTIMIZER_MODEL:-qwen2.5:7b}"
