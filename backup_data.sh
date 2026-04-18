#!/bin/bash
cd /home/vikramatmuri/feargreed

# Pull any code changes first to avoid conflicts
git pull --rebase origin main

# Generate dashboard and chart from latest trading data
.venv/bin/python generate_dashboard.py

# Stage data files, dashboard, and chart
git add trading_log_BOD.csv trading_log_EOD.csv        datasets/fear_greed_forward_test_morning.csv        datasets/fear_greed_forward_test_afternoon.csv        datasets/fear_greed_combined_2011_2025.csv        README.md assets/

# Only commit if there are changes
if git diff --cached --quiet; then
    echo "$(date): No new data to push."
else
    git commit -m "Daily data backup $(date +%Y-%m-%d)"
    git push origin main
    echo "$(date): Data pushed to GitHub."
fi
