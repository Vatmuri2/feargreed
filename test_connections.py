"""
test_connections.py — Smoke test for all external dependencies.
Run on the Pi before relying on the bots:
    python3 test_connections.py
"""

import datetime
import sys
import numpy as np
import pandas as pd

PASS = "PASS"
FAIL = "FAIL"
results = []

def check(label, fn):
    try:
        detail = fn()
        results.append((PASS, label, detail))
        print(f"  [{PASS}] {label}: {detail}")
    except Exception as e:
        results.append((FAIL, label, str(e)))
        print(f"  [{FAIL}] {label}: {e}")

# =============================================================================
# CONFIG — matches ap_fgi_bod.py and ap_fgi_eod.py
# =============================================================================
import os

# --- Check env vars first ---
print("\n--- Environment variables ---")
REQUIRED = ['ALPACA_BOD_API_KEY', 'ALPACA_BOD_API_SECRET', 'ALPACA_EOD_API_KEY', 'ALPACA_EOD_API_SECRET']
missing = [v for v in REQUIRED if not os.environ.get(v)]
if missing:
    for v in REQUIRED:
        status = PASS if os.environ.get(v) else FAIL
        results.append((status, v, "set" if status == PASS else "MISSING"))
        print(f"  [{status}] {v}: {'set' if status == PASS else 'MISSING'}")
    print(f"\nFATAL: Missing env vars: {missing}")
    print("Run: export ALPACA_BOD_API_KEY=... etc.")
    sys.exit(1)
else:
    for v in REQUIRED:
        results.append((PASS, v, "set"))
        print(f"  [{PASS}] {v}: set")

BOD_KEY    = os.environ['ALPACA_BOD_API_KEY']
BOD_SECRET = os.environ['ALPACA_BOD_API_SECRET']
EOD_KEY    = os.environ['ALPACA_EOD_API_KEY']
EOD_SECRET = os.environ['ALPACA_EOD_API_SECRET']

SYMBOL = 'SPY'


# =============================================================================
# 1. yfinance
# =============================================================================
print("\n--- yfinance ---")
import yfinance as yf

def test_yf_latest_price():
    t = yf.Ticker(SYMBOL)
    price = t.fast_info['lastPrice']
    assert price and price > 0
    return f"${price:.2f}"

def test_yf_history():
    end = datetime.date.today()
    start = end - datetime.timedelta(days=60)
    t = yf.Ticker(SYMBOL)
    df = t.history(start=start, end=end)
    assert not df.empty, "empty dataframe"
    return f"{len(df)} rows"

def test_yf_volatility():
    end = datetime.date.today()
    start = end - datetime.timedelta(days=90)
    t = yf.Ticker(SYMBOL)
    df = t.history(start=start, end=end)
    df["returns"] = df["Close"].pct_change()
    vol = df["returns"].rolling(20, min_periods=1).std().iloc[-1] * np.sqrt(252)
    assert not np.isnan(vol)
    return f"{vol:.4f} annualized"

check("Latest price", test_yf_latest_price)
check("Historical data (60d)", test_yf_history)
check("Volatility calc (20d rolling)", test_yf_volatility)


# =============================================================================
# 2. Fear & Greed Index
# =============================================================================
print("\n--- fear_and_greed ---")
import fear_and_greed as fg

def test_fg_fetch():
    idx = fg.get()
    assert idx.value is not None
    return f"{round(idx.value, 1)} ({idx.description}) — last updated {idx.last_update.date()}"

check("Fetch current index", test_fg_fetch)


# =============================================================================
# 3. Alpaca — BOD account
# =============================================================================
print(f"\n--- Alpaca BOD account ({BOD_KEY[:8]}...) ---")
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.common.exceptions import APIError

def test_bod_account():
    c = TradingClient(BOD_KEY, BOD_SECRET, paper=True)
    a = c.get_account()
    assert str(a.status) == "AccountStatus.ACTIVE"
    return f"status={a.status}  equity=${float(a.equity):,.2f}  buying_power=${float(a.buying_power):,.2f}"

def test_bod_position():
    c = TradingClient(BOD_KEY, BOD_SECRET, paper=True)
    try:
        p = c.get_open_position(SYMBOL)
        return f"qty={p.qty}  avg_entry=${float(p.avg_entry_price):.2f}"
    except APIError:
        return "no open position"

def test_bod_orders():
    c = TradingClient(BOD_KEY, BOD_SECRET, paper=True)
    orders = c.get_orders(GetOrdersRequest(status="closed", direction="desc"))
    n = len(orders)
    if n:
        last = orders[0]
        return f"{n} closed orders, latest: {last.side} {last.filled_qty} {last.symbol} @ {last.filled_avg_price}"
    return "0 closed orders"

check("Account status", test_bod_account)
check("Open position", test_bod_position)
check("Order history", test_bod_orders)


# =============================================================================
# 4. Alpaca — EOD account
# =============================================================================
print(f"\n--- Alpaca EOD account ({EOD_KEY[:8]}...) ---")

def test_eod_account():
    c = TradingClient(EOD_KEY, EOD_SECRET, paper=True)
    a = c.get_account()
    assert str(a.status) == "AccountStatus.ACTIVE"
    return f"status={a.status}  equity=${float(a.equity):,.2f}  buying_power=${float(a.buying_power):,.2f}"

def test_eod_position():
    c = TradingClient(EOD_KEY, EOD_SECRET, paper=True)
    try:
        p = c.get_open_position(SYMBOL)
        return f"qty={p.qty}  avg_entry=${float(p.avg_entry_price):.2f}"
    except APIError:
        return "no open position"

def test_eod_orders():
    c = TradingClient(EOD_KEY, EOD_SECRET, paper=True)
    orders = c.get_orders(GetOrdersRequest(status="closed", direction="desc"))
    n = len(orders)
    if n:
        last = orders[0]
        return f"{n} closed orders, latest: {last.side} {last.filled_qty} {last.symbol} @ {last.filled_avg_price}"
    return "0 closed orders"

check("Account status", test_eod_account)
check("Open position", test_eod_position)
check("Order history", test_eod_orders)


# =============================================================================
# 5. F&G dataset files
# =============================================================================
print("\n--- Dataset files ---")

def test_bod_dataset():
    df = pd.read_csv('datasets/fear_greed_forward_test_morning.csv')
    assert not df.empty
    return f"{len(df)} rows, latest date: {df['Date'].iloc[-1]}"

def test_eod_dataset():
    df = pd.read_csv('datasets/fear_greed_forward_test_afternoon.csv')
    assert not df.empty
    return f"{len(df)} rows, latest date: {df['Date'].iloc[-1]}"

check("BOD dataset (morning)", test_bod_dataset)
check("EOD dataset (afternoon)", test_eod_dataset)


# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 55)
passed = sum(1 for r in results if r[0] == PASS)
failed = sum(1 for r in results if r[0] == FAIL)
print(f"Results: {passed} passed, {failed} failed out of {len(results)} checks")
if failed:
    print("\nFailed checks:")
    for status, label, detail in results:
        if status == FAIL:
            print(f"  - {label}: {detail}")
    sys.exit(1)
else:
    print("All checks passed — bots should run correctly.")
