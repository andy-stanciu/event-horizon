#!/usr/bin/env bash
# slurm_sweep.sh
# --------------
# Submit one SLURM job per (horizon x seed) combination for the DMC sweep.
# Each job runs train_dreamer_dmc.sh on a single Quadro RTX 6000.
# Run from the TOP-LEVEL directory (one above r2dreamer/).
#
# Usage:
#   bash scripts/slurm_sweep.sh [TASK] [SEEDS]
#
# Examples:
#   bash scripts/slurm_sweep.sh
#   bash scripts/slurm_sweep.sh dmc_cheetah_run "0 1 2"
#
# Hyak account/partition: edit ACCOUNT and PARTITION below to match your
# klone allocation (e.g. account=cse, partition=gpu-rtx6k).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

TASK="${1:-dmc_walker_walk}"
SEEDS="${2:-0 1 2}"
HORIZONS="5 15 30 60"

# ── Hyak SLURM settings — edit these ─────────────────────────────────────────
ACCOUNT="${SLURM_ACCOUNT:-cse}"
PARTITION="${SLURM_PARTITION:-gpu-rtx6k}"
GPUS=1
CPUS=8
MEM="32G"
TIME="12:00:00"       # 12 h per run — adjust based on your budget

echo "=== EventHorizon: SLURM Horizon Sweep ==="
echo "Task      : $TASK"
echo "Seeds     : $SEEDS"
echo "Horizons  : $HORIZONS"
echo "Account   : $ACCOUNT"
echo "Partition : $PARTITION"
echo ""

for H in $HORIZONS; do
    for SEED in $SEEDS; do
        RUN_NAME="dreamer_${TASK}_H${H}_seed${SEED}"
        LOGDIR="$REPO_DIR/logdir/${RUN_NAME}"
        mkdir -p "$LOGDIR"

        JOB_SCRIPT="$LOGDIR/job.sh"
        cat > "$JOB_SCRIPT" << SLURM
#!/bin/bash
#SBATCH --job-name=${RUN_NAME}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --gpus=${GPUS}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --output=${LOGDIR}/slurm_%j.out
#SBATCH --error=${LOGDIR}/slurm_%j.err

echo "Job \$SLURM_JOB_ID starting on \$(hostname) at \$(date)"

# Set up MuJoCo EGL rendering on the allocated GPU
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=\$(echo \$CUDA_VISIBLE_DEVICES | cut -d',' -f1)

bash ${SCRIPT_DIR}/train_dreamer_dmc.sh ${TASK} ${SEED} ${H}

echo "Job finished at \$(date)"
SLURM

        JOB_ID=$(sbatch --parsable "$JOB_SCRIPT")
        echo "Submitted job $JOB_ID : $RUN_NAME"
    done
done

echo ""
echo "=== All jobs submitted. Monitor with: squeue -u \$USER ==="
