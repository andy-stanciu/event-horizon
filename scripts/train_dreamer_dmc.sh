#!/usr/bin/env bash
# train_dreamer_dmc.sh
# --------------------
# Run DreamerV3 on DMC Proprio (walker_walk by default).
# All hyperparameters stay at published defaults except where noted.
# Run from the TOP-LEVEL directory (one above r2dreamer/).
#
# Usage:
#   bash scripts/train_dreamer_dmc.sh [TASK] [SEED] [HORIZON]
#
# Examples:
#   bash scripts/train_dreamer_dmc.sh                          # defaults
#   bash scripts/train_dreamer_dmc.sh dmc_walker_walk 0 15
#   bash scripts/train_dreamer_dmc.sh dmc_cheetah_run 1 30

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
R2DREAMER_DIR="$REPO_DIR/r2dreamer"
VENV_PYTHON="$REPO_DIR/.venv/bin/python"

# ── Arguments with defaults ───────────────────────────────────────────────────
TASK="${1:-dmc_walker_walk}"
SEED="${2:-0}"
HORIZON="${3:-15}"          # imagination horizon H (batch_length in r2dreamer)
STEPS="${4:-200000}"        # total environment steps (not gradient steps) — 500k in DreamerV3 paper

# ── Derived names ─────────────────────────────────────────────────────────────
ALGO="dreamer"              # model.rep_loss selects dreamer vs r2dreamer
RUN_NAME="${ALGO}_${TASK}_H${HORIZON}_seed${SEED}"
LOGDIR="$REPO_DIR/logdir/${RUN_NAME}"

echo "=== EventHorizon: DreamerV3 on DMC Proprio ==="
echo "Task           : $TASK"
echo "Seed           : $SEED"
echo "Horizon (H)    : $HORIZON"
echo "Logdir         : $LOGDIR"
echo "Run name       : $RUN_NAME"
echo ""

mkdir -p "$LOGDIR"

# ── MuJoCo headless rendering (EGL preferred on GPU nodes) ───────────────────
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"

# ── Launch training ───────────────────────────────────────────────────────────
cd "$R2DREAMER_DIR"

"$VENV_PYTHON" train.py \
    logdir="$LOGDIR" \
    model.rep_loss="$ALGO" \
    env=dmc_proprio \
    "env.task=$TASK" \
    "env.steps=$STEPS" \
    seed="$SEED" \
    batch_length="$HORIZON" \
    hydra.run.dir="$LOGDIR"

echo ""
echo "=== Training complete. Logdir: $LOGDIR ==="
