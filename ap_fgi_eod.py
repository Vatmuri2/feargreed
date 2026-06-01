"""EOD ("end of day") Fear & Greed bot - thin entry point.

Execution time: 15:50 US/Eastern (regular session, 10 minutes before close).
This matches the backtest's stated execution assumption and replaces the
previous 16:10 ET after-hours run, which prevented market orders / proper
exit handling.
"""
from __future__ import annotations

import datetime as dt
import os

from fgi_trading import BotConfig, TradingBot, run_forever


CONFIG = BotConfig(
    name="EOD",
    api_key=os.environ["ALPACA_EOD_API_KEY"],
    api_secret=os.environ["ALPACA_EOD_API_SECRET"],
    log_file="trading_log_EOD.csv",
    fg_path="datasets/fear_greed_forward_test_afternoon.csv",
    state_file="state_EOD.json",
    target_time_et=dt.time(15, 50),
    leverage=1.0,
)


if __name__ == "__main__":
    bot = TradingBot(CONFIG)
    run_forever(bot)
