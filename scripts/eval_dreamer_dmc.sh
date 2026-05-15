#!/usr/bin/env bash
# eval_dreamer_dmc.sh
# -------------------
# Post-hoc: parse a finished run's metrics.jsonl and print episode
# return / success rate summary, then export episode_summary.csv.
# Run from the TOP-LEVEL directory (one above r2dreamer/).
#
# Usage:
#   bash scripts/eval_dreamer_dmc.sh <LOGDIR>
#
# Example:
#   bash scripts/eval_dreamer_dmc.sh logdir/dreamer_dmc_walker_walk_H15_seed0

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="$REPO_DIR/.venv/bin/python"

LOGDIR="${1:-}"
if [ -z "$LOGDIR" ]; then
    echo "Usage: bash scripts/eval_dreamer_dmc.sh <logdir>"
    exit 1
fi

# Resolve relative paths
if [[ "$LOGDIR" != /* ]]; then
    LOGDIR="$REPO_DIR/$LOGDIR"
fi

if [ ! -f "$LOGDIR/metrics.jsonl" ]; then
    echo "Error: no metrics.jsonl found at $LOGDIR"
    echo "Has the training run completed?"
    exit 1
fi

echo "=== EventHorizon: Post-hoc Episode Summary ==="
echo "Logdir: $LOGDIR"
echo ""

"$VENV_PYTHON" "$REPO_DIR/scripts/poc_eval_logger.py" \
    --logdir "$LOGDIR" \
    --csv "$LOGDIR/episode_summary.csv"

echo ""
echo "CSV saved to: $LOGDIR/episode_summary.csv"
