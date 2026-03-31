#!/usr/bin/env bash
# setup_serve_venv.sh — create a dedicated venv for vLLM serving.
#
# vLLM requires a newer transformers than LLaMA-Factory allows, so serving
# and training use separate venvs:
#   .venv        — training (llamafactory + transformers<=4.52.4)
#   .venv-serve  — serving  (vllm + unconstrained transformers)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

VENV="$ROOT/.venv-serve"

echo "Creating serve venv at $VENV ..."
uv venv --python 3.10 "$VENV"

echo "Installing vLLM (cu128) ..."
uv pip install --python "$VENV/bin/python" \
  torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128

uv pip install --python "$VENV/bin/python" vllm

echo ""
echo "Done. Run the server with:"
echo "  bash scripts/serve_3b.sh"
