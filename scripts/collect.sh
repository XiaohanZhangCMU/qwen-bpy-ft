#!/usr/bin/env bash
# collect.sh — run the data collection pipeline
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

# Load .env if present
if [[ -f .env ]]; then
  set -a; source .env; set +a
fi

CONFIG="${1:-configs/data_collection/default.yaml}"

echo "=== Moonlake: Data Collection ==="
echo "Config:  $CONFIG"
echo "Output:  data/raw/"
echo ""

python -m data_collection.pipeline --config "$CONFIG" "${@:2}"
