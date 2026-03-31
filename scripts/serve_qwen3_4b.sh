#!/usr/bin/env bash
# serve_qwen3_4b.sh — start a vLLM server for Qwen3-4B (base + fine-tuned adapter)
#
# Serves two model IDs on port 8002:
#   "Qwen/Qwen3-4B"  — base model (no adapter)
#   "ft_qwen3_4b"    — fine-tuned LoRA adapter
#
# Usage:
#   bash scripts/serve_qwen3_4b.sh          # run in foreground
#   bash scripts/serve_qwen3_4b.sh &        # run in background
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

if [[ -f .env ]]; then set -a; source .env; set +a; fi

export CUDA_VISIBLE_DEVICES=4,5,6,7

# Use the dedicated serve venv (vllm needs newer transformers than llamafactory allows)
SERVE_VENV="${SERVE_VENV:-$ROOT/.venv-serve}"
PYTHON="${SERVE_VENV}/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: serve venv not found at $SERVE_VENV" >&2
  echo "  Create it with: bash scripts/setup_serve_venv.sh" >&2
  exit 1
fi

# Ensure Python C headers are findable for vLLM's runtime CUDA extension compilation
PYTHON_INCLUDE="$("$PYTHON" -c "import sysconfig; print(sysconfig.get_path('include'))" 2>/dev/null || true)"
if [[ -n "$PYTHON_INCLUDE" && -f "$PYTHON_INCLUDE/Python.h" ]]; then
  export C_INCLUDE_PATH="${PYTHON_INCLUDE}${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
fi

PORT="${VLLM_PORT:-8002}"

echo "Starting vLLM server for Qwen3-4B on port $PORT..."
echo "  Base model : Qwen/Qwen3-4B"
echo "  LoRA       : ft_qwen3_4b -> outputs/qwen3_4b_lora"
echo "  GPUs       : $CUDA_VISIBLE_DEVICES"
echo ""

"$PYTHON" -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-4B \
  --enable-lora \
  --lora-modules ft_qwen3_4b=outputs/qwen3_4b_lora \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --max-lora-rank 32 \
  --enforce-eager \
  --port "$PORT" \
  --host 0.0.0.0
