#!/usr/bin/env bash
# eval_all.sh — convenience reference: shows the four eval commands to run one by one.
# Run each independently; results are saved to disk. Then call compare.sh for the table.
#
# Usage:
#   bash scripts/eval_all.sh           # print the commands
#   bash scripts/eval_all.sh --run     # actually execute them sequentially

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

if [[ -f .env ]]; then set -a; source .env; set +a; fi

export CUDA_VISIBLE_DEVICES=4,5,6,7

NUM_PROMPTS_FLAG="${MOONLAKE_NUM_PROMPTS:+--num-prompts $MOONLAKE_NUM_PROMPTS}"

CMDS=(
  "CUDA_VISIBLE_DEVICES=4,5,6,7 python -m evaluation.pipeline --config configs/evaluation/base_model.yaml  --tag base_qwen3b  $NUM_PROMPTS_FLAG"
  "CUDA_VISIBLE_DEVICES=4,5,6,7 python -m evaluation.pipeline --config configs/evaluation/default.yaml      --tag ft_qwen3b    --checkpoint outputs/qwen2_5_coder_3b_lora  $NUM_PROMPTS_FLAG"
  "CUDA_VISIBLE_DEVICES=4,5,6,7 python -m evaluation.pipeline --config configs/evaluation/finetuned_7b.yaml --tag ft_qwen7b    $NUM_PROMPTS_FLAG"
  "python -m evaluation.pipeline --config configs/evaluation/openai.yaml       --tag openai       $NUM_PROMPTS_FLAG"
)

if [[ "${1:-}" != "--run" ]]; then
  echo "Run each eval independently (results are saved to data/eval/):"
  echo ""
  for cmd in "${CMDS[@]}"; do
    echo "  $cmd"
    echo ""
  done
  echo "Then print the comparison table:"
  echo "  bash scripts/compare.sh"
  exit 0
fi

# --run mode: execute sequentially
for cmd in "${CMDS[@]}"; do
  echo "=== Running: $cmd ==="
  eval "$cmd"
  echo ""
done

echo "=== All evals done. Printing table... ==="
bash "$SCRIPT_DIR/compare.sh"
