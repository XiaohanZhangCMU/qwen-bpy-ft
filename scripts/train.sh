#!/usr/bin/env bash
# train.sh — run SFT training via LLaMA-Factory
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

# Load .env if present
if [[ -f .env ]]; then
  set -a; source .env; set +a
fi

CONFIG="${1:-configs/training/qwen_sft.yaml}"

echo "=== Moonlake: SFT Training ==="
echo "Config: $CONFIG"
echo ""

python -m training.train --config "$CONFIG"
