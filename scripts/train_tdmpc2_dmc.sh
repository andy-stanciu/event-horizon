#!/usr/bin/env bash
# train_tdmpc2_dmc.sh
# -------------------
# Run TD-MPC2 on DMC proprioceptive tasks.
# Run from the TOP-LEVEL directory (one above tdmpc2/).
#
# Usage:
#   bash scripts/train_tdmpc2_dmc.sh [TASK] [SEED] [HORIZON] [STEPS] [GPU]
#
# Examples:
#   bash scripts/train_tdmpc2_dmc.sh                              # defaults
#   bash scripts/train_tdmpc2_dmc.sh dmc_walker_walk 0 3
#   bash scripts/train_tdmpc2_dmc.sh dmc_cheetah_run 1 5 200000
#   bash scripts/train_tdmpc2_dmc.sh dmc_walker_walk 0 3 200000 3  # use GPU 3

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TDMPCC_DIR="$REPO_DIR/tdmpc2/tdmpc2/"
VENV_PYTHON="$REPO_DIR/.venv/bin/python"

TASK="${1:-dmc_walker_walk}"
SEED="${2:-0}"
HORIZON="${3:-3}"
STEPS="${4:-200000}"
GPU="${5:-2}"

# Convert dmc_walker_walk → walker-walk (TD-MPC2 task format)
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

cd "$TDMPCC_DIR"

"$VENV_PYTHON" train.py \
    task="$TDMPCC_TASK" \
    seed="$SEED" \
    steps="$STEPS" \
    horizon="$HORIZON" \
    model_size=5 \
    data_dir="$LOGDIR/buffer" \
    save_video=false \
    save_agent=true \
    enable_wandb=false \
    wandb_project=none \
    wandb_entity=none \
    compile=false \
    work_dir="$LOGDIR" \
    exp_name="${RUN_NAME}" \
    hydra.run.dir="$LOGDIR"

echo ""

TDMPC2_OUTDIR="$TDMPCC_DIR/logs/${TDMPCC_TASK}/${SEED}/${RUN_NAME}"
if [ -d "$TDMPC2_OUTDIR" ]; then
    cp -r "$TDMPC2_OUTDIR/." "$LOGDIR/"
    echo "Logs moved to $LOGDIR"
fi

echo "=== Training complete. Logdir: $LOGDIR ==="

