#!/usr/bin/env bash
# render.sh — generate a scene from a prompt and render it
#
# Usage:
#   bash scripts/render.sh "a cozy bedroom with a bed and wardrobe"
#   bash scripts/render.sh "a night market" --backend openai
#   bash scripts/render.sh --script path/to/scene.py
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

if [[ -f .env ]]; then set -a; source .env; set +a; fi

python -m evaluation.render "$@"
