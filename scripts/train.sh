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

export CUDA_VISIBLE_DEVICES=4,5,6,7

# Workaround for NCCL P2P / SHM issues on Blackwell (RTX 5090, SM 12.0)
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1

echo "=== Moonlake: SFT Training ==="
echo "Config: $CONFIG"
echo "GPUs:   $CUDA_VISIBLE_DEVICES"
echo ""

python -m training.train --config "$CONFIG"
