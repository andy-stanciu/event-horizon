#!/bin/bash
#SBATCH --job-name=tdmpc2_dmc_walker_walk_H30_seed0
#SBATCH --account=manfor
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --chdir=/homes/iws/manfor/cse579/event-horizon
#SBATCH --export=ALL
#SBATCH --output=/homes/iws/manfor/cse579/event-horizon/logdir/tdmpc2_dmc_walker_walk_H30_seed0/slurm_%j.out
#SBATCH --error=/homes/iws/manfor/cse579/event-horizon/logdir/tdmpc2_dmc_walker_walk_H30_seed0/slurm_%j.err

echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"
echo "Run: tdmpc2_dmc_walker_walk_H30_seed0"

# Activate virtualenv
source /homes/iws/manfor/cse579/event-horizon/.venv/bin/activate

# Set up MuJoCo EGL rendering on the allocated GPU
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f1)
export PYTHONPATH="${PYTHONPATH}:/homes/iws/manfor/cse579/event-horizon/tdmpc2"

srun --gpus-per-node=2 bash /homes/iws/manfor/cse579/event-horizon/scripts/train_tdmpc2_dmc.sh dmc_walker_walk 0 30 200000

echo "Job finished at $(date)"
