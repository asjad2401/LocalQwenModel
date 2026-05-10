#!/usr/bin/env bash
# Source this to activate the agent venv and add the CLI to your path:
#   source scripts/activate.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR/.."

source "$ROOT/.venv/bin/activate"
export AGENT_CONFIG="$ROOT/config.yaml"
echo "agent venv active — type 'agent --help' to get started"
