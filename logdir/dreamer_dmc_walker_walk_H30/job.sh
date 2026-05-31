#!/bin/bash
#SBATCH --job-name=dreamer_dmc_walker_walk_H30_seed0
#SBATCH --account=andys22
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --chdir=/homes/iws/andys22/579/event-horizon
#SBATCH --export=ALL
#SBATCH --output=/homes/iws/andys22/579/event-horizon/logdir/dreamer_dmc_walker_walk_H30_seed0/slurm_%j.out
#SBATCH --error=/homes/iws/andys22/579/event-horizon/logdir/dreamer_dmc_walker_walk_H30_seed0/slurm_%j.err

echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"
echo "Run: dreamer_dmc_walker_walk_H30_seed0"

# Activate virtualenv
source /homes/iws/andys22/579/event-horizon/.venv/bin/activate

# Set up MuJoCo EGL rendering on the allocated GPU
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f1)
export PYTHONPATH="${PYTHONPATH}:/homes/iws/andys22/579/event-horizon/r2dreamer"

srun --gpus-per-node=2 bash /homes/iws/andys22/579/event-horizon/scripts/train_dreamer_dmc.sh dmc_walker_walk 0 30 200000

echo "Job finished at $(date)"
