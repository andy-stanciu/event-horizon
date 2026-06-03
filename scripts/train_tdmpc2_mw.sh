#!/usr/bin/env bash
# train_tdmpc2_metaworld.sh
# Run TD-MPC2 on MetaWorld tasks.
# Usage: bash scripts/train_tdmpc2_metaworld.sh [TASK] [SEED] [HORIZON] [STEPS] [GPU]

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TDMPCC_DIR="$REPO_DIR/tdmpc2/tdmpc2/"
VENV_PYTHON="$REPO_DIR/.venv/bin/python"

TASK="${1:-mw-pick-place}"
SEED="${2:-0}"
HORIZON="${3:-3}"
STEPS="${4:-500000}"
GPU="${5:-2}"

RUN_NAME="tdmpc2_${TASK//-/_}_H${HORIZON}_seed${SEED}"
LOGDIR="$REPO_DIR/logdir/${RUN_NAME}"

echo "=== EventHorizon: TD-MPC2 on MetaWorld ==="
echo "Task    : $TASK"
echo "Seed    : $SEED"
echo "Horizon : $HORIZON"
echo "Steps   : $STEPS"
echo "GPU     : $GPU"
echo "Logdir  : $LOGDIR"
echo ""

mkdir -p "$LOGDIR"

export CUDA_VISIBLE_DEVICES="$GPU"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export MUJOCO_EGL_DEVICE_ID="0"
export EGL_DEVICE_ID="0"

cd "$TDMPCC_DIR"

"$VENV_PYTHON" train.py \
    task="$TASK" \
    ~hydra/launcher \
    episodic=true \
    seed="$SEED" \
    steps="$STEPS" \
    horizon="$HORIZON" \
    model_size=5 \
    episode_length=100 \
    seed_steps=1000 \
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

TDMPC2_OUTDIR="$TDMPCC_DIR/logs/${TASK}/${SEED}/${RUN_NAME}"
if [ -d "$TDMPC2_OUTDIR" ]; then
    cp -r "$TDMPC2_OUTDIR/." "$LOGDIR/"
    echo "Logs moved to $LOGDIR"
fi

echo "=== Training complete. Logdir: $LOGDIR ==="