#!/usr/bin/env bash
# eval_all.sh — run all four evals (base, ft-3b, ft-7b, gpt-4o) then print comparison table
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

if [[ -f .env ]]; then set -a; source .env; set +a; fi

extract_path() {
  # Capture only the last line printed by the pipeline that contains the result path
  grep "Results written to" | tail -1 | awk '{print $NF}'
}

echo "=== Step 1/4: Base model (null benchmark) ==="
BASE_OUT=$(python -m evaluation.pipeline \
  --config configs/evaluation/base_model.yaml \
  2>&1 | tee /dev/stderr | extract_path)

echo ""
echo "=== Step 2/4: Fine-tuned Qwen-3B ==="
FT3B_OUT=$(python -m evaluation.pipeline \
  --config configs/evaluation/default.yaml \
  --checkpoint outputs/qwen2_5_coder_3b_lora \
  2>&1 | tee /dev/stderr | extract_path)

echo ""
echo "=== Step 3/4: Fine-tuned Qwen-7B ==="
FT7B_OUT=$(python -m evaluation.pipeline \
  --config configs/evaluation/finetuned_7b.yaml \
  2>&1 | tee /dev/stderr | extract_path)

echo ""
echo "=== Step 4/4: GPT-4o oracle ==="
OPENAI_OUT=$(python -m evaluation.pipeline \
  --config configs/evaluation/openai.yaml \
  2>&1 | tee /dev/stderr | extract_path)

echo ""
echo "=== Comparison table ==="
python -m evaluation.compare \
  --results "$BASE_OUT" "$FT3B_OUT" "$FT7B_OUT" "$OPENAI_OUT" \
  --labels  "Base Qwen-3B" "FT Qwen-3B" "FT Qwen-7B" "GPT-4o" \
  --save    "data/eval/comparison_$(date -u +%Y%m%dT%H%M%S).json"
