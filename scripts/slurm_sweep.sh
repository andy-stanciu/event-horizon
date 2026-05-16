#!/usr/bin/env bash
# slurm_sweep.sh
# --------------
# Submit one SLURM job per (horizon x seed) combination for the DMC sweep.
# Each job runs train_dreamer_dmc.sh on a single GPU.
# Run from the TOP-LEVEL directory (one above r2dreamer/).
#
# Usage:
#   bash scripts/slurm_sweep.sh [TASK] [SEEDS]
#
# Examples:
#   bash scripts/slurm_sweep.sh
#   bash scripts/slurm_sweep.sh dmc_cheetah_run "0 1 2"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

TASK="${1:-dmc_walker_walk}"
SEEDS="${2:-0}"
HORIZONS="30"
STEPS="200000"

# ── SLURM settings ────────────────────────────────────────────────────────────
ACCOUNT="${SLURM_ACCOUNT:-andys22}"
PARTITION="${SLURM_PARTITION:-gpu}"
NODES=1
NTASKS=1
GPUS=2
CPUS=8
MEM="32G"
TIME="2:00:00"

# ── Repo / environment ────────────────────────────────────────────────────────
VENV_PATH="$REPO_DIR/.venv"             # path to your virtualenv
PYTHONPATH_EXTRA="$REPO_DIR/r2dreamer"  # r2dreamer source lives here

echo "=== EventHorizon: SLURM Horizon Sweep ==="
echo "Task      : $TASK"
echo "Seeds     : $SEEDS"
echo "Horizons  : $HORIZONS"
echo "Steps     : $STEPS"
echo "Account   : $ACCOUNT"
echo "Partition : $PARTITION"
echo "Repo      : $REPO_DIR"
echo ""

for H in $HORIZONS; do
    for SEED in $SEEDS; do
        RUN_NAME="dreamer_${TASK}_H${H}_seed${SEED}"
        LOGDIR="$REPO_DIR/logdir/${RUN_NAME}"
        if [[ -f "$LOGDIR/metrics.jsonl" ]]; then
            echo "Skipping $RUN_NAME — already exists"
            continue
        fi
        mkdir -p "$LOGDIR"

        JOB_SCRIPT="$LOGDIR/job.sh"
        cat > "$JOB_SCRIPT" << SLURM
#!/bin/bash
#SBATCH --job-name=${RUN_NAME}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=${NODES}
#SBATCH --ntasks-per-node=${NTASKS}
#SBATCH --gpus=${GPUS}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --chdir=${REPO_DIR}
#SBATCH --export=ALL
#SBATCH --output=${LOGDIR}/slurm_%j.out
#SBATCH --error=${LOGDIR}/slurm_%j.err

echo "Job \$SLURM_JOB_ID starting on \$(hostname) at \$(date)"
echo "Run: ${RUN_NAME}"

# Activate virtualenv
source ${VENV_PATH}/bin/activate

# Set up MuJoCo EGL rendering on the allocated GPU
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=\$(echo \$CUDA_VISIBLE_DEVICES | cut -d',' -f1)
export PYTHONPATH="\${PYTHONPATH}:${PYTHONPATH_EXTRA}"

srun --gpus-per-node=${GPUS} bash ${SCRIPT_DIR}/train_dreamer_dmc.sh ${TASK} ${SEED} ${H} ${STEPS}

echo "Job finished at \$(date)"
SLURM

        JOB_ID=$(sbatch --parsable "$JOB_SCRIPT")
        echo "Submitted job $JOB_ID : $RUN_NAME"
    done
done

echo ""
echo "=== All jobs submitted ==="
echo "Monitor : squeue -u \$USER"
echo "Outputs : $REPO_DIR/logdir/"