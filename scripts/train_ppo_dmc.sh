#!/usr/bin/env bash
# train_ppo_dmc.sh
# ----------------
# Run CleanRL PPO on DMC proprioceptive tasks.
# Run from the TOP-LEVEL event-horizon directory.
#
# Usage:
#   bash scripts/train_ppo_dmc.sh [TASK] [SEED] [STEPS] [GPU]
#
# Examples:
#   bash scripts/train_ppo_dmc.sh
#   bash scripts/train_ppo_dmc.sh walker-walk 0 3000000 2
#   bash scripts/train_ppo_dmc.sh cheetah-run 1 3000000 3

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLEANRL_DIR="$REPO_DIR/cleanrl"
VENV_PYTHON="$REPO_DIR/.venv/bin/python"

TASK="${1:-walker-walk}"
SEED="${2:-0}"
STEPS="${3:-200000}"
GPU="${4:-2}"

# Convert walker-walk → dm_control/walker-walk-v0
ENV_ID="dm_control/${TASK}-v0"

RUN_NAME="ppo_${TASK//-/_}_seed${SEED}"
LOGDIR="$REPO_DIR/logdir/${RUN_NAME}"

echo "=== EventHorizon: PPO on DMC ==="
echo "Task    : $TASK"
echo "Env ID  : $ENV_ID"
echo "Seed    : $SEED"
echo "Steps   : $STEPS"
echo "GPU     : $GPU"
echo "Logdir  : $LOGDIR"
echo ""

mkdir -p "$LOGDIR"

export CUDA_VISIBLE_DEVICES="$GPU"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export MUJOCO_EGL_DEVICE_ID="0"
export EGL_DEVICE_ID="0"

cd "$CLEANRL_DIR"

"$VENV_PYTHON" cleanrl/ppo_continuous_action.py \
    --env-id "$ENV_ID" \
    --seed "$SEED" \
    --total-timesteps "$STEPS" \
    --num-envs 4 \
    --num-steps 512 \
    --num-minibatches 32 \
    --update-epochs 10 \
    --gamma 0.99 \
    --gae-lambda 0.95 \
    --clip-coef 0.2 \
    --ent-coef 0.0 \
    --vf-coef 0.5 \
    --max-grad-norm 0.5 \
    --track false \
    --capture-video false \
    --exp-name "$RUN_NAME"

echo ""
echo "=== Training complete. Logdir: $LOGDIR ==="
