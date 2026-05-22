#!/usr/bin/env bash
# train_tdmpc2_dmc.sh
# -------------------
# Run TD-MPC2 on DMC proprioceptive tasks.
# Run from the TOP-LEVEL directory (one above tdmpc2/).
#
# Usage:
#   bash scripts/train_tdmpc2_dmc.sh [TASK] [SEED] [HORIZON] [STEPS]
#
# Examples:
#   bash scripts/train_tdmpc2_dmc.sh                   # defaults
#   bash scripts/train_tdmpc2_dmc.sh dmc_walker_walk 0 3
#   bash scripts/train_tdmpc2_dmc.sh dmc_cheetah_run 1 5 200000

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TDMPCC_DIR="$REPO_DIR/tdmpc2/tdmpc2/"
VENV_PYTHON="$REPO_DIR/.venv/bin/python"

TASK="${1:-dmc_walker_walk}"
SEED="${2:-0}"
HORIZON="${3:-3}"
STEPS="${4:-200000}"

TDMPCC_TASK="${TASK#dmc_}"
TDMPCC_TASK="${TDMPCC_TASK//_/-}"

RUN_NAME="tdmpc2_${TASK}_H${HORIZON}_seed${SEED}"
LOGDIR="$REPO_DIR/logdir/${RUN_NAME}"

echo "=== EventHorizon: TD-MPC2 on DMC Proprio ==="
echo "Task           : $TASK"
echo "TD-MPC2 task   : $TDMPCC_TASK"
echo "Seed           : $SEED"
echo "Horizon (H)    : $HORIZON"
echo "Steps          : $STEPS"
echo "Logdir         : $LOGDIR"
echo "Run name       : $RUN_NAME"
echo ""

mkdir -p "$LOGDIR"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"

cd "$TDMPCC_DIR"

"$VENV_PYTHON" train.py \
    task="$TDMPCC_TASK" \
    seed="$SEED" \
    steps="$STEPS" \
    horizon="$HORIZON" \
    save_video=false \
    enable_wandb=false \
    work_dir="$LOGDIR" \
    hydra/launcher=basic \
    hydra.run.dir="$LOGDIR"

echo ""
echo "=== Training complete. Logdir: $LOGDIR ==="
