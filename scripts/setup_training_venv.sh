#!/usr/bin/env bash
# setup_training_venv.sh — create a dedicated venv for SFT training.
#
# vLLM requires a newer transformers than LLaMA-Factory allows, so serving
# and training use separate venvs:
#   .venv        — training (llamafactory + transformers<=4.52.4)
#   .venv-serve  — serving  (vllm + unconstrained transformers)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

VENV="$ROOT/.venv"

echo "Creating training venv at $VENV ..."
uv venv --python 3.10 "$VENV"

PYTHON="$VENV/bin/python"

echo "Installing PyTorch (cu128) ..."
uv pip install --python "$PYTHON" \
  torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128

echo "Installing LLaMA-Factory + pinned transformers ..."
uv pip install --python "$PYTHON" \
  "transformers>=4.45.0,<=4.52.4,!=4.52.0" \
  llamafactory \
  accelerate \
  datasets \
  peft

echo "Installing project dependencies ..."
uv pip install --python "$PYTHON" -e "$ROOT"

echo ""
echo "Done. Activate with:"
echo "  source .venv/bin/activate"
echo ""
echo "Or run training directly:"
echo "  bash scripts/train.sh configs/training/qwen_sft.yaml"
