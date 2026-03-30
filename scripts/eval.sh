#!/usr/bin/env bash
# eval.sh — evaluate the fine-tuned model on held-out prompts
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

# Load .env if present
if [[ -f .env ]]; then
  set -a; source .env; set +a
fi

CONFIG="${1:-configs/evaluation/default.yaml}"

echo "=== Moonlake: Evaluation ==="
echo "Config:  $CONFIG"
echo "Prompts: $(grep -c '' data/eval/prompts.jsonl 2>/dev/null || echo '?') held-out prompts"
echo ""

python -m evaluation.pipeline --config "$CONFIG" "${@:2}"
