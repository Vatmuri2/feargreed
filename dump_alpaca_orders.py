"""Read-only dump of recent Alpaca orders + account state for both BOD and EOD paper accounts.

Run on the Pi:
    python dump_alpaca_orders.py

Requires ALPACA_BOD_API_KEY/SECRET and ALPACA_EOD_API_KEY/SECRET in env.
"""
import os
from datetime import datetime, timedelta, timezone

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus


def dump(label, key_env, secret_env):
    key = os.environ.get(key_env)
    secret = os.environ.get(secret_env)
    if not key or not secret:
        print(f"[{label}] missing {key_env}/{secret_env}, skipping")
        return

    client = TradingClient(key, secret, paper=True)

    print("=" * 70)
    print(f"{label} ACCOUNT")
    print("=" * 70)
    acct = client.get_account()
    print(f"  portfolio_value : {acct.portfolio_value}")
    print(f"  equity          : {acct.equity}")
    print(f"  cash            : {acct.cash}")
    print(f"  buying_power    : {acct.buying_power}")
    print(f"  status          : {acct.status}")

    try:
        pos = client.get_open_position("SPY")
        print(f"  SPY position    : qty={pos.qty} avg_entry={pos.avg_entry_price} "
              f"market_value={pos.market_value} unrealized_pl={pos.unrealized_pl}")
    except Exception:
        print("  SPY position    : none")

    after = datetime.now(timezone.utc) - timedelta(days=14)
    req = GetOrdersRequest(
        status=QueryOrderStatus.ALL,
        after=after,
        direction="desc",
        limit=100,
    )
    orders = client.get_orders(req)

    print()
    print(f"{label} ORDERS (last 14 days, newest first)")
    print("-" * 70)
    hdr = ("submitted_at", "side", "qty", "filled_qty", "limit", "filled_avg", "status", "filled_at")
    print("  " + " | ".join(f"{h:<20}" if i == 0 else f"{h:<12}" for i, h in enumerate(hdr)))
    for o in orders:
        if o.symbol != "SPY":
            continue
        row = (
            str(o.submitted_at)[:19] if o.submitted_at else "-",
            str(o.side).split(".")[-1],
            str(o.qty),
            str(o.filled_qty),
            str(o.limit_price) if o.limit_price else "-",
            str(o.filled_avg_price) if o.filled_avg_price else "-",
            str(o.status).split(".")[-1],
            str(o.filled_at)[:19] if o.filled_at else "-",
        )
        print("  " + " | ".join(f"{v:<20}" if i == 0 else f"{v:<12}" for i, v in enumerate(row)))
    print()


if __name__ == "__main__":
    dump("BOD", "ALPACA_BOD_API_KEY", "ALPACA_BOD_API_SECRET")
    dump("EOD", "ALPACA_EOD_API_KEY", "ALPACA_EOD_API_SECRET")
