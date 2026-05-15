#!/usr/bin/env bash
# setup_env.sh
# ------------
# Creates a uv virtual environment for r2dreamer and installs all dependencies.
# Run from the TOP-LEVEL directory (one above r2dreamer/).
#
# Usage:
#   bash scripts/setup_env.sh
#
# After running, activate with:
#   source .venv/bin/activate

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
R2DREAMER_DIR="$REPO_DIR/r2dreamer"

echo "=== EventHorizon: r2dreamer environment setup ==="
echo "Repo root      : $REPO_DIR"
echo "r2dreamer dir  : $R2DREAMER_DIR"
echo ""

# ── 1. Check uv is available ──────────────────────────────────────────────────
if ! command -v uv &> /dev/null; then
    echo "[setup] uv not found. Installing via pip..."
    pip install uv
fi
echo "[setup] uv version: $(uv --version)"

# ── 2. Create virtual environment in repo root ────────────────────────────────
VENV_DIR="$REPO_DIR/.venv"
if [ -d "$VENV_DIR" ]; then
    echo "[setup] .venv already exists at $VENV_DIR — skipping creation."
else
    echo "[setup] Creating .venv with Python 3.11..."
    uv venv "$VENV_DIR" --python 3.11
fi

# ── 3. Install r2dreamer dependencies ─────────────────────────────────────────
echo "[setup] Installing r2dreamer/requirements.txt..."
uv pip install --python "$VENV_DIR/bin/python" \
    -r "$R2DREAMER_DIR/requirements.txt"

# ── 4. Install extra project-level tools ──────────────────────────────────────
echo "[setup] Installing project-level extras (wandb, pandas, seaborn)..."
uv pip install --python "$VENV_DIR/bin/python" \
    wandb pandas seaborn matplotlib

echo ""
echo "=== Setup complete ==="
echo "Activate with:  source .venv/bin/activate"
echo "Or prefix cmds: .venv/bin/python ..."
