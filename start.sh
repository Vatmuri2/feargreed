#!/bin/bash

# Ensure logs folder exists
mkdir -p logs

# Start tmux session for bots if not exists
tmux has-session -t feargreed 2>/dev/null || tmux new-session -d -s feargreed

# Start BOD bot in window 0
tmux send-keys -t feargreed:0 "cd /app && python ap_fgi_bod.py >> logs/bod.log 2>&1" C-m

# Create new window for EOD bot
tmux new-window -t feargreed:1 -n EOD
tmux send-keys -t feargreed:1 "cd /app && python alpaca_runEOD.py >> logs/eod.log 2>&1" C-m

# Optional: attach to session
tmux attach -t feargreed
