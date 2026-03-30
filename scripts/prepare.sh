#!/usr/bin/env bash
# prepare.sh — convert raw trajectories to LLaMA-Factory training format
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

echo "=== Moonlake: Prepare Dataset ==="

python -m training.prepare_dataset \
  --input  data/raw/ \
  --output data/processed/moonlake_sft.jsonl \
  --dataset-info data/processed/dataset_info.json \
  "$@"
