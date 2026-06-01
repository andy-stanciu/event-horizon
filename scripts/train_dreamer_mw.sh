#!/usr/bin/env bash
# train_dreamer_mw.sh
# -------------------
# Run DreamerV3 on MetaWorld (pick-place by default).
# Run from the TOP-LEVEL directory (one above r2dreamer/).
#
# Usage:
#   bash scripts/train_dreamer_mw.sh [TASK] [SEED] [HORIZON] [STEPS] [GPU]
#
# Examples:
#   bash scripts/train_dreamer_mw.sh
#   bash scripts/train_dreamer_mw.sh metaworld_pick-place 0 5
#   bash scripts/train_dreamer_mw.sh metaworld_assembly 1 10 1010000
#   bash scripts/train_dreamer_mw.sh metaworld_pick-place 0 5 1010000 3  # use GPU 3

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
R2DREAMER_DIR="$REPO_DIR/r2dreamer"
VENV_PYTHON="$REPO_DIR/.venv/bin/python"

# ── Arguments with defaults ───────────────────────────────────────────────────
TASK="${1:-metaworld_pick-place}"
SEED="${2:-0}"
HORIZON="${3:-15}"
STEPS="${4:-1010000}"
GPU="${5:-2}"

# ── Derived names ─────────────────────────────────────────────────────────────
ALGO="dreamer"
RUN_NAME="${ALGO}_${TASK}_H${HORIZON}_seed${SEED}"
LOGDIR="$REPO_DIR/logdir/${RUN_NAME}"

echo "=== EventHorizon: DreamerV3 on MetaWorld ==="
echo "Task           : $TASK"
echo "Seed           : $SEED"
echo "Horizon (H)    : $HORIZON"
echo "Steps          : $STEPS"
echo "GPU            : $GPU"
echo "Logdir         : $LOGDIR"
echo "Run name       : $RUN_NAME"
echo ""

mkdir -p "$LOGDIR"

# GPU: CUDA_VISIBLE_DEVICES remaps the chosen physical GPU to logical cuda:0
# MUJOCO_EGL_DEVICE_ID is always 0 (relative to CUDA_VISIBLE_DEVICES)
export CUDA_VISIBLE_DEVICES="$GPU"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export MUJOCO_EGL_DEVICE_ID="0"

cd "$R2DREAMER_DIR"

"$VENV_PYTHON" train.py \
    logdir="$LOGDIR" \
    model.rep_loss="$ALGO" \
    env=metaworld \
    "env.task=$TASK" \
    "env.steps=$STEPS" \
    seed="$SEED" \
    batch_length="$HORIZON" \
    hydra.run.dir="$LOGDIR"

echo ""
echo "=== Training complete. Logdir: $LOGDIR ==="