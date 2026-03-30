#!/usr/bin/env bash
# verify_env.sh — checks required tools and env vars before running any pipeline
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

ok()   { echo -e "${GREEN}[OK]${NC}    $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC}  $1"; }
fail() { echo -e "${RED}[FAIL]${NC}  $1"; ERRORS=$((ERRORS+1)); }

ERRORS=0

echo "=== Moonlake environment check ==="

# Python
if python --version &>/dev/null; then
  PY=$(python --version 2>&1)
  ok "Python: $PY"
else
  fail "Python not found"
fi

# Blender
BLENDER_BIN="${BLENDER_BIN:-blender}"
if "$BLENDER_BIN" --version &>/dev/null 2>&1; then
  BV=$("$BLENDER_BIN" --version 2>&1 | head -1)
  ok "Blender: $BV"
else
  fail "Blender not found at '$BLENDER_BIN'. Set BLENDER_BIN env var."
fi

# GPU
if command -v nvidia-smi &>/dev/null; then
  GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
  ok "GPU: $GPU"
else
  warn "nvidia-smi not found — GPU not detected (OK for data collection, needed for training)"
fi

# Python packages
for pkg in openai pydantic tqdm yaml dotenv; do
  if python -c "import $pkg" &>/dev/null 2>&1; then
    ok "Python package: $pkg"
  else
    fail "Python package missing: $pkg  (pip install $pkg)"
  fi
done

# Optional training packages
for pkg in torch transformers; do
  if python -c "import $pkg" &>/dev/null 2>&1; then
    ok "Training package: $pkg"
  else
    warn "Training package not found: $pkg (only needed for training/eval)"
  fi
done

# LLaMA-Factory CLI
if command -v llamafactory-cli &>/dev/null; then
  ok "LLaMA-Factory CLI found"
else
  warn "llamafactory-cli not found (only needed for training step)"
fi

# Env vars
for var in OPENAI_API_KEY BLENDER_BIN HF_TOKEN; do
  if [[ -n "${!var:-}" ]]; then
    ok "Env var set: $var"
  else
    if [[ "$var" == "OPENAI_API_KEY" ]]; then
      fail "Env var missing: $var (required for data collection)"
    else
      warn "Env var not set: $var"
    fi
  fi
done

echo ""
if [[ $ERRORS -eq 0 ]]; then
  echo -e "${GREEN}All checks passed.${NC}"
else
  echo -e "${RED}${ERRORS} check(s) failed.${NC}"
  exit 1
fi
