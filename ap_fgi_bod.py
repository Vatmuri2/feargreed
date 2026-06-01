"""BOD ("beginning of day") Fear & Greed bot - thin entry point.

Execution time: 09:35 US/Eastern (regular session, 5 minutes after open).
This is the post-open replacement for the previous 09:20 ET pre-market run.
Pre-market execution was the primary driver of the exit failures (no market
orders, illiquid limits, partial-fill orphans) - see CHANGES.md.
"""
from __future__ import annotations

import datetime as dt
import os

from fgi_trading import BotConfig, TradingBot, run_forever


CONFIG = BotConfig(
    name="BOD",
    api_key=os.environ["ALPACA_BOD_API_KEY"],
    api_secret=os.environ["ALPACA_BOD_API_SECRET"],
    log_file="trading_log_BOD.csv",
    fg_path="datasets/fear_greed_forward_test_morning.csv",
    state_file="state_BOD.json",
    target_time_et=dt.time(9, 35),
    leverage=1.0,
)


if __name__ == "__main__":
    bot = TradingBot(CONFIG)
    run_forever(bot)
