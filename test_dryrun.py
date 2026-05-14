"""Dry-run diagnostic for both BOD and EOD accounts. No orders placed."""
import os
import datetime
import pandas as pd
import numpy as np
import yfinance as yf
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide
from alpaca.trading.requests import GetOrdersRequest
from alpaca.common.exceptions import APIError

TRADE_SYMBOL = 'SPY'
LOOKBACK_DAYS = 3
MOMENTUM_THRESHOLD = 0.2
VELOCITY_THRESHOLD = 0.15
VOLATILITY_BUY_LIMIT = 0.6
VOLATILITY_SELL_LIMIT = 0.5
VOLATILITY_WINDOW = 20
MAX_DAYS_HELD = 8

def check_account(label, api_key_env, api_secret_env, fg_path, log_file):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    client = TradingClient(os.environ[api_key_env], os.environ[api_secret_env], paper=True)

    # --- Account state ---
    acct = client.get_account()
    print(f"Portfolio value : ${float(acct.portfolio_value):,.2f}")
    print(f"Buying power    : ${float(acct.buying_power):,.2f}")
    print(f"Cash            : ${float(acct.cash):,.2f}")

    # --- Position check (with short detection) ---
    try:
        pos = client.get_open_position(TRADE_SYMBOL)
        qty_raw = float(pos.qty)
        qty_int = int(qty_raw)
        print(f"\nPosition raw qty: {pos.qty} | side: {pos.side}")
        if qty_int < 0:
            print(f"  *** SHORT POSITION DETECTED - would be closed by close_unexpected_short() ***")
        elif qty_int == 0:
            print(f"  *** FRACTIONAL/ZERO qty - get_current_position() would return (False, 0) ***")
        else:
            print(f"  LONG position: {qty_int} shares (fix returns True, {qty_int})")
    except APIError:
        print("\nNo open position (clean state).")

    # --- Signal simulation ---
    fg_data = pd.read_csv(fg_path)
    fg_data['Date'] = pd.to_datetime(fg_data['Date']).dt.date
    fgi_column = 'fear_greed' if 'fear_greed' in fg_data.columns else ('Fear Greed' if 'Fear Greed' in fg_data.columns else 'Index')

    fg_data['fg_momentum'] = fg_data[fgi_column] - fg_data[fgi_column].rolling(LOOKBACK_DAYS, min_periods=1).mean()
    fg_data['fg_change'] = fg_data[fgi_column].diff().fillna(0)
    fg_data['fg_velocity'] = fg_data['fg_change'].rolling(LOOKBACK_DAYS, min_periods=1).mean()
    latest = fg_data.iloc[-1]

    end = datetime.date.today()
    start = end - datetime.timedelta(days=VOLATILITY_WINDOW * 3)
    ticker = yf.Ticker(TRADE_SYMBOL)
    df = ticker.history(start=start, end=end)
    df["returns"] = df["Close"].pct_change()
    df["volatility"] = df["returns"].rolling(VOLATILITY_WINDOW, min_periods=1).std() * np.sqrt(252)
    volatility = round(df["volatility"].iloc[-1], 4)

    print(f"\nLatest FGI      : {latest[fgi_column]:.2f}")
    print(f"FGI Momentum    : {latest['fg_momentum']:.4f}  (threshold {MOMENTUM_THRESHOLD})")
    print(f"FGI Velocity    : {latest['fg_velocity']:.4f}  (threshold {VELOCITY_THRESHOLD})")
    print(f"Volatility      : {volatility:.4f}  (buy limit {VOLATILITY_BUY_LIMIT}, sell limit {VOLATILITY_SELL_LIMIT})")

    # Simulate signal
    try:
        raw_qty = float(client.get_open_position(TRADE_SYMBOL).qty)
        has_position = int(raw_qty) > 0
        pos_qty = int(raw_qty) if has_position else 0
    except APIError:
        has_position = False
        pos_qty = 0

    days_held = 0
    if has_position:
        orders = client.get_orders(GetOrdersRequest(status="closed", direction="desc"))
        for o in orders:
            if o.symbol == TRADE_SYMBOL and o.side == OrderSide.BUY and o.filled_qty:
                days_held = (datetime.date.today() - o.filled_at.date()).days
                break

    if not has_position:
        if latest['fg_momentum'] > MOMENTUM_THRESHOLD and latest['fg_velocity'] > VELOCITY_THRESHOLD and volatility < VOLATILITY_BUY_LIMIT:
            signal, reason = "BUY", "Strong momentum/velocity, low volatility"
        elif latest['fg_momentum'] > MOMENTUM_THRESHOLD and latest['fg_velocity'] > VELOCITY_THRESHOLD:
            signal, reason = "HOLD", "Strong momentum/velocity but high volatility"
        elif volatility >= VOLATILITY_BUY_LIMIT:
            signal, reason = "HOLD", "Volatility too high for entry"
        else:
            signal, reason = "HOLD", "Insufficient momentum/velocity for entry"
    else:
        if (latest['fg_momentum'] < MOMENTUM_THRESHOLD or latest['fg_velocity'] < VELOCITY_THRESHOLD
                or volatility > VOLATILITY_SELL_LIMIT or days_held >= MAX_DAYS_HELD):
            signal = "SELL"
            reason = f"Max holding period ({days_held} days)" if days_held >= MAX_DAYS_HELD else "Momentum reversal or high volatility"
        else:
            signal, reason = "HOLD", f"Favorable indicators ({days_held}/{MAX_DAYS_HELD} days held)"

    print(f"\nSimulated signal: {signal}")
    print(f"Reason          : {reason}")
    if has_position:
        print(f"Days held       : {days_held} / {MAX_DAYS_HELD}")
        print(f"Position qty    : {pos_qty}")

    # --- Log health check ---
    try:
        log = pd.read_csv(log_file)
        log['Date'] = pd.to_datetime(log['Timestamp']).dt.date
        today = datetime.date.today()
        already_ran = today in log['Date'].values
        print(f"\nalready_executed_today: {already_ran}")
    except Exception as e:
        print(f"\nLog read error: {e}")

check_account(
    "BOD Account",
    "ALPACA_BOD_API_KEY", "ALPACA_BOD_API_SECRET",
    "datasets/fear_greed_forward_test_morning.csv",
    "trading_log_BOD.csv"
)

check_account(
    "EOD Account",
    "ALPACA_EOD_API_KEY", "ALPACA_EOD_API_SECRET",
    "datasets/fear_greed_forward_test_afternoon.csv",
    "trading_log_EOD.csv"
)
