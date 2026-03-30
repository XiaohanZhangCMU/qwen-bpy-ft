#!/usr/bin/env bash
# compare.sh — print comparison table from previously saved eval results
#
# Usage:
#   # By tag (recommended — finds latest run for each tag automatically):
#   bash scripts/compare.sh
#
#   # Quick run on fewer prompts:
#   bash scripts/compare.sh --quick 10
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

if [[ -f .env ]]; then set -a; source .env; set +a; fi

EXTRA_ARGS="${@}"

python -m evaluation.compare \
  --tags   base_qwen3b ft_qwen3b ft_qwen7b openai \
  --labels "Base Qwen-3B" "FT Qwen-3B" "FT Qwen-7B" "GPT-4o" \
  --save   "data/eval/comparison_$(date -u +%Y%m%dT%H%M%S).json" \
  $EXTRA_ARGS
