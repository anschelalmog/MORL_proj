#!/bin/bash


# Check GPU nodes (optional)
sinfo -o "%20N %10c %10m %25f %10G" | grep gpu

# Start a new tmux session (detached) and run your training script in window 0
tmux new-session -d -s mosac_training_new 'source ../venv/bin/activate && python /script/train_mosac.py --timesteps 500000'

# Create a new tmux window for TensorBoard
tmux new-window -t mosac_training_new:1 -n tensorboard 'source ../venv/bin/activate && tensorboard --logdir=logs/mosac/tensorboard --port=6006 --bind_all'

# Attach to the tmux session (will not error because session is created above)
tmux attach-session -t mosac_training_new