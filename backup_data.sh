#!/bin/bash
cd /home/vikramatmuri/feargreed

# Commit any local uncommitted changes before pulling to avoid rebase conflict
git add -A
if ! git diff --cached --quiet; then
    git commit -m "Auto-stash before pull $(date +%Y-%m-%d_%H:%M)"
fi

# Pull any code changes
git pull origin main

# Generate dashboard and chart from latest trading data
.venv/bin/python generate_dashboard.py

# Stage data files, dashboard, and chart (silently skip missing files)
git add \
    trading_log_BOD.csv \
    trading_log_EOD.csv \
    datasets/fear_greed_forward_test_morning.csv \
    datasets/fear_greed_forward_test_afternoon.csv \
    datasets/fear_greed_combined_2011_2025.csv \
    README.md \
    assets/ 2>/dev/null || true

# Only commit if there are staged changes
if git diff --cached --quiet; then
    echo "$(date): No new data to push."
else
    git commit -m "Daily data backup $(date +%Y-%m-%d)"
    git push origin main
    echo "$(date): Data pushed to GitHub."
fi
