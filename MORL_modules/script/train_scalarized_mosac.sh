#!/bin/bash
#SBATCH --job-name=mosac_train
#SBATCH --output=mosac_%j.out
#SBATCH --error=mosac_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Load necessary modules (adjust based on your server)
module load python/3.9
module load cuda/11.8

# Activate your environment
source /path/to/your/venv/bin/activate

# Set PYTHONPATH
export PYTHONPATH=/path/to/your/energy-net-zoo:$PYTHONPATH

# Run training
python train_mosac_fixed.py