#!/usr/bin/env bash
# Cloud Run target endpoints (match configs/cloudrun_*.yaml).
# Usage: source scripts/cloudrun_env.sh

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Source this file; do not execute it directly:" >&2
  echo "  source scripts/cloudrun_env.sh" >&2
  exit 1
fi

# Cloud Run — vLLM target (DeepHat or equivalent)
DEEPHAT_MODEL="${DEEPHAT_MODEL:-YourOrg/YourModel}"

BUGTRACE_ENDPOINT="${BUGTRACE_ENDPOINT:-https://YOUR-BUGTRACE-SERVICE-HASH.a.run.app}"
BUGTRACE_MODEL="${BUGTRACE_MODEL:-hf.co/BugTraceAI/BugTraceAI-Apex-G4-26B-Q4:latest}"

# Ping settings for keepalive / warmup scripts.
# OLLAMA_KEEP_ALIVE is used only for local optimizer warmup (not Cloud Run target).
PING_PROMPT="${PING_PROMPT:-Say OK}"
PING_MAX_TOKENS="${PING_MAX_TOKENS:-16}"
PING_TIMEOUT_S="${PING_TIMEOUT_S:-90}"
OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-30m}"

# Export so child processes (e.g. warmup_*.sh spawned by run_*_baseline.sh) see
# the same values. Assignments above alone are not exported by default.
export DEEPHAT_ENDPOINT DEEPHAT_MODEL
export BUGTRACE_ENDPOINT BUGTRACE_MODEL
export PING_PROMPT PING_MAX_TOKENS PING_TIMEOUT_S OLLAMA_KEEP_ALIVE
