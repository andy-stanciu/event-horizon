#!/usr/bin/env bash
# sweep_horizon_dmc.sh
# --------------------
# Run the full horizon sweep H ∈ {5, 15, 30, 60} for DreamerV3 on DMC.
# Launches one run per (seed x horizon) combination sequentially.
# For parallel SLURM submission, see scripts/slurm_sweep.sh.
# Run from the TOP-LEVEL directory (one above r2dreamer/).
#
# Usage:
#   bash scripts/sweep_horizon_dmc.sh [TASK] [SEEDS]
#
# Examples:
#   bash scripts/sweep_horizon_dmc.sh                              # walker_walk, seeds 0 1 2
#   bash scripts/sweep_horizon_dmc.sh dmc_cheetah_run "0 1 2"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TASK="${1:-dmc_walker_walk}"
SEEDS="${2:-0 1 2}"
HORIZONS="5 15 30 60"

echo "=== EventHorizon: Horizon Sweep on DMC Proprio ==="
echo "Task     : $TASK"
echo "Seeds    : $SEEDS"
echo "Horizons : $HORIZONS"
echo ""

for H in $HORIZONS; do
    for SEED in $SEEDS; do
        echo "--- Launching: task=$TASK  H=$H  seed=$SEED ---"
        bash "$SCRIPT_DIR/train_dreamer_dmc.sh" "$TASK" "$SEED" "$H"
        echo ""
    done
done

echo "=== Sweep complete ==="
