#!/bin/bash
cd /app
git reset --hard
git pull origin main

# Stop running bots
tmux kill-session -t feargreed 2>/dev/null

# Restart bots
bash start.sh
