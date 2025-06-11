#!/bin/bash
# Script to train MOSAC agent on EnergyNet environment

# Set default values
TIMESTEPS=100000
LOG_FOLDER="logs/mosac_experiments"
PRICING="ONLINE"
DEMAND="SINUSOIDAL"
COST="CONSTANT"
WEIGHTS="0.25,0.25,0.25,0.25"
SEED=42

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --timesteps)
      TIMESTEPS="$2"
      shift 2
      ;;
    --log-folder)
      LOG_FOLDER="$2"
      shift 2
      ;;
    --weights)
      WEIGHTS="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Ensure PYTHONPATH is set correctly
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# Run the MOSAC training script
python MORL_modules/scripts/train_mosac.py \
  --n-timesteps "$TIMESTEPS" \
  --log-folder "$LOG_FOLDER" \
  --pricing-policy "$PRICING" \
  --demand-pattern "$DEMAND" \
  --cost-type "$COST" \
  --preference-weights "$WEIGHTS" \
  --seed "$SEED"

echo "Training complete! Results saved to $LOG_FOLDER"