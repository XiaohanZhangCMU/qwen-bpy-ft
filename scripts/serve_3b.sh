#!/usr/bin/env bash
# serve_3b.sh — start a vLLM server for Qwen-3B (base + fine-tuned adapter)
#
# Serves two model IDs on port 8000:
#   "Qwen/Qwen2.5-Coder-3B-Instruct" — base model (no adapter)
#   "ft_qwen3b"                       — fine-tuned LoRA adapter
#
# Usage:
#   bash scripts/serve_3b.sh          # run in foreground
#   bash scripts/serve_3b.sh &        # run in background
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

if [[ -f .env ]]; then set -a; source .env; set +a; fi

export CUDA_VISIBLE_DEVICES=4,5,6,7

# Ensure Python C headers are findable for vLLM's runtime CUDA extension compilation
PYTHON_INCLUDE="$(python -c "import sysconfig; print(sysconfig.get_path('include'))" 2>/dev/null || true)"
if [[ -n "$PYTHON_INCLUDE" && -f "$PYTHON_INCLUDE/Python.h" ]]; then
  export C_INCLUDE_PATH="${PYTHON_INCLUDE}${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
elif [[ -f "/usr/include/python3.10/Python.h" ]]; then
  export C_INCLUDE_PATH="/usr/include/python3.10${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
else
  echo "WARNING: Python.h not found; vLLM LoRA compilation may fail." >&2
  echo "  Fix: sudo apt-get install python3.10-dev" >&2
fi

echo "Starting vLLM server for Qwen-3B on port 8000..."
echo "  Base model : Qwen/Qwen2.5-Coder-3B-Instruct"
echo "  LoRA       : ft_qwen3b -> outputs/qwen2_5_coder_3b_lora"
echo "  GPUs       : $CUDA_VISIBLE_DEVICES"
echo ""

python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-Coder-3B-Instruct \
  --enable-lora \
  --lora-modules ft_qwen3b=outputs/qwen2_5_coder_3b_lora \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --enforce-eager \
  --port 8000 \
  --host 0.0.0.0
