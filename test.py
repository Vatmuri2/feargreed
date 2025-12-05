from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestTradeRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from datetime import datetime, timedelta, timezone
import traceback

# ===========================================================
#  🔑 API KEYS
# ===========================================================
API_KEY = 'PK2FJH3CAEOSD2IPVIV3EVB3WD'
API_SECRET = 'FRKEVUMqGcZV7U6V9Q7yxnqqBM19MCYqewMLcPkEEb1p'

# ===========================================================
#  SETTINGS
# ===========================================================
SYMBOL = "SPY"
DO_TEST_ORDER = False  # set True to test a paper order
# ===========================================================

# Initialize clients
trading_client = TradingClient(API_KEY, API_SECRET, paper=True)
data_client = StockHistoricalDataClient(API_KEY, API_SECRET)

print("\n====================================")
print("TEST 1 — CHECK ACCOUNT")
print("====================================")
try:
    account = trading_client.get_account()
    print("✔ Account ID:", account.id)
    print("✔ Buying power:", account.buying_power)
except Exception:
    print("❌ FAILED: trading_client.get_account()")
    traceback.print_exc()

print("\n====================================")
print("TEST 2 — MARKET CLOCK")
print("====================================")
try:
    clock = trading_client.get_clock()
    print("✔ Market open:", clock.is_open)
    print("✔ Next open:", clock.next_open)
    print("✔ Next close:", clock.next_close)
except Exception:
    print("❌ FAILED: trading_client.get_clock()")
    traceback.print_exc()

print("\n====================================")
print("TEST 3 — LATEST TRADE")
print("====================================")
try:
    # ✅ FIX: symbol_or_symbols must be str or list[str], NOT dict
    req = StockLatestTradeRequest(symbol_or_symbols=SYMBOL)
    latest_trade_result = data_client.get_stock_latest_trade(req)
    trade = latest_trade_result[SYMBOL]
    print("✔ Latest trade price:", trade.price)
    print("✔ Timestamp:", trade.timestamp)
except Exception:
    print("❌ FAILED: data_client.get_stock_latest_trade()")
    traceback.print_exc()

print("\n====================================")
print("TEST 4 — HISTORICAL BARS")
print("====================================")
try:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=30)  # buffer for volatility calculation

    bars_req = StockBarsRequest(
        symbol_or_symbols=SYMBOL,
        start=start,
        end=end,
        timeframe=TimeFrame.Day  # ✅ Daily bars only
    )
    bars = data_client.get_stock_bars(bars_req).df

    if bars.empty:
        print("⚠ No historical bars returned!")
    else:
        print("✔ Retrieved bars:", len(bars))
        print(bars.tail())

except Exception:
    print("❌ FAILED: data_client.get_stock_bars()")
    traceback.print_exc()

print("\n====================================")
print("TEST 5 — PLACE PAPER ORDER (optional)")
print("====================================")
if DO_TEST_ORDER:
    try:
        order = MarketOrderRequest(
            symbol=SYMBOL,
            qty=1,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY
        )
        result = trading_client.submit_order(order)
        print("✔ Order submitted:", result.id)
    except Exception:
        print("❌ FAILED: trading_client.submit_order()")
        traceback.print_exc()
else:
    print("Skipping order test (DO_TEST_ORDER=False)")

print("\nALL TESTS COMPLETE")
