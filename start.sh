#!/bin/bash
cd /home/vikramatmuri/feargreed

# Pull latest code
git pull origin main

# Restart systemd services
sudo systemctl restart feargreed-bod.service
sudo systemctl restart feargreed-eod.service

# Show status
sudo systemctl status feargreed-bod.service --no-pager
sudo systemctl status feargreed-eod.service --no-pager

# Run daily backup
bash /home/vikramatmuri/feargreed/backup_data.sh
