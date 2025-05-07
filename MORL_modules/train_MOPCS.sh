#!/bin/bash

# run_mo_battery_control.sh
# Script for training and evaluating a multi-objective battery controller
# Run it with ./run_mo_battery_control.sh

# ./train_MOPCS.sh --mode weighted_sum --timesteps 200000 --weights 0.7,0.1,0.1,0.1

set -e

# Create output directories
mkdir -p logs/mo_battery
mkdir -p results/plots
mkdir -p models

# Environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_VISIBLE_DEVICES=0  # Use first GPU if available

# Parse command line arguments
MODE="pareto"            # Default mode
TIMESTEPS=100000         # Default timesteps
EVAL_FREQ=5000           # Evaluation frequency
SEED=42                  # Random seed
WEIGHTS="0.4,0.2,0.1,0.3"  # Default weights

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --timesteps)
      TIMESTEPS="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --weights)
      WEIGHTS="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --mode VALUE       Optimization mode: 'weighted_sum' or 'pareto' (default: pareto)"
      echo "  --timesteps VALUE  Number of timesteps to train (default: 100000)"
      echo "  --seed VALUE       Random seed (default: 42)"
      echo "  --weights VALUE    Comma-separated preference weights (default: 0.4,0.2,0.1,0.3)"
      echo "                     Order: profit,lifecycle,stability,renewable"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

echo "========================================"
echo "Multi-Objective Battery Control Training"
echo "========================================"
echo "Mode: $MODE"
echo "Timesteps: $TIMESTEPS"
echo "Seed: $SEED"
echo "Weights: $WEIGHTS"
echo "========================================"

# Convert comma-separated weights to array format for Python
WEIGHTS_ARRAY="[${WEIGHTS}]"

# Run the training script
echo "Starting training..."
python -m mo_battery.train \
  --mode $MODE \
  --timesteps $TIMESTEPS \
  --seed $SEED \
  --weights "$WEIGHTS_ARRAY" \
  --log-dir logs/mo_battery \
  --save-dir models \
  --eval-freq $EVAL_FREQ

# If training successful, generate evaluation and visualizations
if [ $? -eq 0 ]; then
  echo "Training completed successfully"
  echo "Generating evaluation results..."

  # Run evaluation with different preference weights
  echo "Running evaluations with different preference combinations..."
  python -m mo_battery.evaluate \
    --model-dir models \
    --mode $MODE \
    --seed $SEED \
    --results-dir results

  # Generate visualizations
  echo "Generating visualization of Pareto front and trade-offs..."
  python -m mo_battery.visualize \
    --results-dir results \
    --plot-dir results/plots

  echo "All done! Results available in:"
  echo "- Training logs: logs/mo_battery"
  echo "- Trained models: models"
  echo "- Evaluation results: results"
  echo "- Visualizations: results/plots"
else
  echo "Training failed"
  exit 1
fi



