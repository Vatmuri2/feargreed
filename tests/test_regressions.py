"""Regression tests for the three logged BOD failures:

  1. Orphan shares from partial fill   (e.g. BOUGHT 69 / SOLD 64, 5 orphaned)
  2. SELL insufficient buying power 40310000   (oversell because submitted qty
     exceeded owned shares)
  3. SELL qty must be > 0 40010001   (stale state: bot thought it held a
     position but the broker showed flat)

The tests have two layers:

* MOCK tests use a deterministic fake `TradingClient` to reproduce each
  failure condition exactly. They run anywhere and are the authoritative
  proof that the new reconciliation logic handles each case.

* LIVE tests run against the Alpaca paper API and verify the same behaviour
  end-to-end. They auto-skip if credentials aren't present or the market is
  closed (paper market orders only fill during regular session).

Usage:
    .venv/bin/python tests/test_regressions.py            # all available
    .venv/bin/python tests/test_regressions.py --mocks    # mock layer only
    .venv/bin/python tests/test_regressions.py --live     # live layer only
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Optional

# Ensure repo root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

# Load .env (the bots themselves read os.environ, so we need to populate it
# before importing them).
ENV_PATH = os.path.join(REPO_ROOT, ".env")
if os.path.exists(ENV_PATH):
    with open(ENV_PATH) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

# Map the generic Alpaca env vars onto the BOD/EOD names the bots expect,
# so that tests can run against a single shared paper account.
for prefix in ("ALPACA_BOD", "ALPACA_EOD"):
    if f"{prefix}_API_KEY" not in os.environ and "APCA_API_KEY_ID" in os.environ:
        os.environ[f"{prefix}_API_KEY"] = os.environ["APCA_API_KEY_ID"]
    if f"{prefix}_API_SECRET" not in os.environ and "APCA_API_SECRET_KEY" in os.environ:
        os.environ[f"{prefix}_API_SECRET"] = os.environ["APCA_API_SECRET_KEY"]

# Need placeholders so the modules import cleanly even if creds are absent.
os.environ.setdefault("ALPACA_BOD_API_KEY", "missing")
os.environ.setdefault("ALPACA_BOD_API_SECRET", "missing")
os.environ.setdefault("ALPACA_EOD_API_KEY", "missing")
os.environ.setdefault("ALPACA_EOD_API_SECRET", "missing")

from alpaca.common.exceptions import APIError
from alpaca.trading.enums import OrderSide, OrderStatus, QueryOrderStatus

import fgi_trading
from fgi_trading import BotConfig, TradingBot, PositionState, trading_days_between


# ---------------------------------------------------------------------------
# Mock TradingClient & DataClient
# ---------------------------------------------------------------------------


@dataclass
class _Order:
    id: str
    symbol: str
    qty: int
    side: OrderSide
    status: OrderStatus = OrderStatus.NEW
    filled_qty: float = 0.0
    filled_avg_price: float = 0.0
    filled_at: Optional[dt.datetime] = None


@dataclass
class _Position:
    symbol: str
    qty: int
    avg_entry_price: float


@dataclass
class _Account:
    cash: float = 50000.0
    portfolio_value: float = 50000.0
    buying_power: float = 50000.0
    equity: float = 50000.0


class FakeTradingClient:
    """Fakes alpaca-py's TradingClient with controllable fill behaviour."""

    def __init__(self, position_qty: int = 0):
        self.account = _Account()
        self.position_qty = position_qty
        self.orders: dict[str, _Order] = {}
        self.fill_plan: dict[str, list[int]] = {}     # order_id -> [fills...]
        self.error_plan: list[Exception] = []         # raised on next submit
        # Toggle: should close_position raise the qty-non-positive error
        # when we're already flat? (matches Alpaca behaviour)
        self.flat_close_raises = True

    # ------------------------------------------------ account / position

    def get_account(self):
        return self.account

    def get_open_position(self, symbol):
        if self.position_qty == 0:
            raise APIError(json.dumps({"code": 40410000, "message": "position does not exist"}))
        return _Position(symbol=symbol, qty=self.position_qty, avg_entry_price=700.0)

    # ------------------------------------------------ orders

    def get_orders(self, req):
        if req.status == QueryOrderStatus.OPEN:
            return [o for o in self.orders.values() if o.status not in fgi_trading._TERMINAL_ORDER_STATUSES]
        return list(self.orders.values())

    def get_order_by_id(self, oid):
        return self.orders[str(oid)]

    def submit_order(self, req):
        if self.error_plan:
            err = self.error_plan.pop(0)
            raise err
        oid = str(uuid.uuid4())
        side = req.side
        qty = int(req.qty)
        o = _Order(id=oid, symbol=req.symbol, qty=qty, side=side, status=OrderStatus.NEW)
        self.orders[oid] = o
        # Schedule fills per the plan: defaults to a full instant fill.
        plan = self.fill_plan.pop(oid, None)
        if plan is None:
            plan = [qty]   # full fill
        self._apply_fills(o, plan, side)
        return o

    def _apply_fills(self, order: _Order, plan: list[int], side: OrderSide):
        total_filled = sum(plan)
        order.filled_qty = float(total_filled)
        if total_filled > 0:
            order.filled_avg_price = 700.0
            order.filled_at = dt.datetime.now(fgi_trading.EASTERN)
            if side == OrderSide.BUY:
                self.position_qty += total_filled
            else:
                self.position_qty -= total_filled
        if total_filled >= order.qty:
            order.status = OrderStatus.FILLED
        elif total_filled > 0:
            order.status = OrderStatus.PARTIALLY_FILLED
        else:
            order.status = OrderStatus.NEW

    def cancel_order_by_id(self, oid):
        o = self.orders.get(str(oid))
        if o and o.status not in fgi_trading._TERMINAL_ORDER_STATUSES:
            o.status = OrderStatus.CANCELED

    def close_position(self, symbol):
        if self.position_qty == 0:
            if self.flat_close_raises:
                raise APIError(json.dumps({"code": 40010001, "message": "qty must be > 0"}))
            return _Order(id=str(uuid.uuid4()), symbol=symbol, qty=0, side=OrderSide.SELL,
                          status=OrderStatus.FILLED)
        qty = abs(self.position_qty)
        side = OrderSide.SELL if self.position_qty > 0 else OrderSide.BUY
        oid = str(uuid.uuid4())
        o = _Order(id=oid, symbol=symbol, qty=qty, side=side, status=OrderStatus.NEW)
        self.orders[oid] = o
        # Configurable: planned partial close, else full close
        plan = self.fill_plan.pop("__close__", None) or [qty]
        self._apply_fills(o, plan, side)
        return o


class FakeDataClient:
    """Returns a fixed quote/bar series so we can exercise the bot."""

    def get_stock_latest_quote(self, req):
        from types import SimpleNamespace
        symbol = (
            req.symbol_or_symbols if isinstance(req.symbol_or_symbols, str)
            else req.symbol_or_symbols[0]
        )
        return {symbol: SimpleNamespace(bid_price=700.0, ask_price=700.1)}

    def get_stock_bars(self, req):
        import pandas as pd
        # 30 days of dummy bars with stable close so vol ~ 0
        dates = pd.date_range(end=dt.datetime.now(fgi_trading.EASTERN).date(), periods=30, freq="B")
        df = pd.DataFrame({
            "open": [700.0] * 30,
            "high": [702.0] * 30,
            "low": [698.0] * 30,
            "close": [700.0 + (i % 3) * 0.1 for i in range(30)],
            "volume": [1e7] * 30,
        }, index=pd.MultiIndex.from_product(
            [["SPY"], dates], names=["symbol", "timestamp"]
        ))
        from types import SimpleNamespace
        return SimpleNamespace(df=df)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


PASS = 0
FAIL = 0
SKIP = 0


def _run(name, fn):
    global PASS, FAIL, SKIP
    print(f"\n=== {name} ===")
    try:
        result = fn()
        if result == "SKIP":
            SKIP += 1
            print(f"SKIP: {name}")
        else:
            PASS += 1
            print(f"PASS: {name}")
    except AssertionError as e:
        FAIL += 1
        print(f"FAIL: {name}\n  {e}")
        traceback.print_exc(limit=2)
    except Exception as e:
        FAIL += 1
        print(f"FAIL (exception): {name}\n  {e}")
        traceback.print_exc()


def _make_bot_with_fakes(tmpdir, position_qty=0) -> tuple[TradingBot, FakeTradingClient]:
    import shutil
    fg_src = os.path.join(REPO_ROOT, "datasets", "fear_greed_forward_test_morning.csv")
    fg_path = os.path.join(tmpdir, "fg.csv")
    shutil.copy(fg_src, fg_path)
    cfg = BotConfig(
        name="TEST",
        api_key="k", api_secret="s",
        log_file=os.path.join(tmpdir, "log.csv"),
        fg_path=fg_path,
        state_file=os.path.join(tmpdir, "state.json"),
        target_time_et=dt.time(9, 35),
        fill_poll_seconds=2,  # speed up tests
        sell_reconcile_sleep=0,
        fgi_max_retries=1,
        fgi_retry_delay=0,
    )
    tc = FakeTradingClient(position_qty=position_qty)
    dc = FakeDataClient()
    bot = TradingBot(cfg, trading_client=tc, data_client=dc)

    # Stub the FGI fetcher so tests don't hit the live CNN API.
    import pandas as pd

    def _fake_fetch(path, today_et, max_retries=1, retry_delay=0):
        df = pd.read_csv(path)
        df["Date"] = pd.to_datetime(df["Date"]).dt.date
        col = "fear_greed" if "fear_greed" in df.columns else (
            "Fear Greed" if "Fear Greed" in df.columns else "Index"
        )
        # Append today's row with a stable benign value
        if not (df["Date"] == today_et).any():
            row = {"Date": today_et, col: 50.0}
            if "rating" in df.columns:
                row["rating"] = "neutral"
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        return df, 50.0, col, "fresh"

    bot.fgi_fetcher = _fake_fetch
    return bot, tc


# ---------------------------------------------------------------------------
# MOCK regression tests - the three failure cases
# ---------------------------------------------------------------------------


def test_mock_partial_fill_orphans():
    """Reproduces the 2026-05-04 BOUGHT 69 / 2026-05-08 SOLD 64 incident.

    The old code returned 'SOLD' on a partial fill, leaving 5 orphan
    shares. The new code must sweep until flat (or fail loudly)."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        bot, tc = _make_bot_with_fakes(tmp, position_qty=69)
        bot.state.record_entry(dt.date.today() - dt.timedelta(days=4), 69)

        # Plan: the first close_position fills 64 of 69 (partial), then
        # subsequent attempts complete the remaining 5.
        tc.fill_plan["__close__"] = [64]

        action, qty, _, err = bot.execute_sell(700.0)

        # New code MUST end flat
        assert tc.position_qty == 0, (
            f"orphan shares! position_qty={tc.position_qty}, error={err}")
        assert action == "SOLD", f"expected SOLD, got {action} (err={err})"
        assert qty == 69, f"expected total filled 69, got {qty}"
        assert bot.state.get_entry_date() is None, "entry state must clear after flat"


def test_mock_oversell_blocked():
    """Reproduces the 2026-05-05 SELL ... insufficient buying power 40310000.

    The old code submitted a hard-coded `qty` even when it exceeded the
    live position. The new code reads the live position first and never
    requests more than is owned."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        bot, tc = _make_bot_with_fakes(tmp, position_qty=10)

        # If the bot tried to oversell (qty > position), our fake would
        # have submitted a SELL > 10. close_position scopes the order to
        # the actual position - verify it.
        action, qty, _, err = bot.execute_sell(700.0)
        assert action == "SOLD" and qty == 10, f"action={action}, qty={qty}, err={err}"
        assert tc.position_qty == 0
        # Inspect every SELL order submitted - none may exceed initial 10
        for o in tc.orders.values():
            if o.side == OrderSide.SELL:
                assert o.qty <= 10, f"oversell! order qty={o.qty}"


def test_mock_flat_treated_as_no_action():
    """Reproduces the 2026-05-11..14 SELL ... qty must be > 0 40010001 burn.

    The old code believed it held a position (stale `Days_Held`) and tried
    to sell 0 shares, hitting 40010001 four days in a row. The new code
    must recognise the flat state and return NO_ACTION with no error."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        bot, tc = _make_bot_with_fakes(tmp, position_qty=0)

        action, qty, _, err = bot.execute_sell(700.0)
        assert action == "NO_ACTION", f"expected NO_ACTION, got {action}"
        assert qty == 0
        # crucially, must not have raised 40010001 to the caller
        assert err is None or "qty must be > 0" not in str(err), f"err={err}"
        # Also: a full run_cycle on a flat account with no buy signal must
        # never submit a sell.
        bot.state.clear()
        summary = bot.run_cycle()
        assert summary["side_after"] == "flat"


def test_mock_run_cycle_idempotent_partial_recovery():
    """If a prior cycle left orphan shares, the next cycle's SELL signal
    must sweep them to flat - i.e. re-running recovers the failed exit."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        bot, tc = _make_bot_with_fakes(tmp, position_qty=5)
        # Force a sell-side state: entry date long enough ago to trigger
        # the max-hold exit regardless of momentum.
        bot.state.record_entry(
            dt.date.today() - dt.timedelta(days=30), 5
        )
        # Bias indicators to fail the buy criteria (we don't want a buy)
        summary = bot.run_cycle()
        assert tc.position_qty == 0, f"failed to sweep, position={tc.position_qty}"
        assert summary["side_after"] == "flat"


def test_mock_buy_never_oversizes_cash():
    """1x leverage / cash-only sizing: bot must never request more shares
    than cash * (1 - headroom) / price."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        bot, tc = _make_bot_with_fakes(tmp, position_qty=0)
        tc.account.cash = 7000.0  # exactly 10 shares at $700, headroom blocks 1
        target = bot._target_buy_qty(700.0)
        assert target <= 9, f"target {target} exceeds cash headroom"


def test_mock_short_detected_as_first_class():
    """A short position must be detected (returned as 'short') and not
    silently swallowed by `read_position`."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        bot, tc = _make_bot_with_fakes(tmp, position_qty=-3)
        side, qty = bot.read_position()
        assert side == "short" and qty == -3, f"side={side}, qty={qty}"


def test_mock_days_held_trading_days():
    """Days_Held should be measured in NYSE trading days, ET."""
    today = dt.date.today()
    # Friday -> Monday is 1 trading day
    friday = today
    while friday.weekday() != 4:
        friday -= dt.timedelta(days=1)
    monday = friday + dt.timedelta(days=3)
    n = trading_days_between(friday, monday)
    assert n == 1, f"Fri->Mon should be 1 trading day, got {n}"


def test_mock_api_error_code_via_attribute():
    """Error classification must use APIError.code, not substring matching."""
    err = APIError(json.dumps({"code": 40310000, "message": "insufficient buying power"}))
    assert fgi_trading._api_error_code(err) == 40310000


def test_mock_state_cleared_when_flat():
    """If broker says flat but state file claims a position, state must be cleared."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        bot, tc = _make_bot_with_fakes(tmp, position_qty=0)
        bot.state.record_entry(dt.date.today() - dt.timedelta(days=2), 5)
        bot.run_cycle()
        assert bot.state.get_entry_date() is None


# ---------------------------------------------------------------------------
# LIVE regression tests - against Alpaca paper
# ---------------------------------------------------------------------------


_LIVE_PREFLIGHT_CACHED: Optional[dict] = None


def _live_preflight() -> dict:
    """Cached preflight: confirms creds work and reports market status.
    Returns {'ok': bool, 'reason': str, 'market_open': bool}."""
    global _LIVE_PREFLIGHT_CACHED
    if _LIVE_PREFLIGHT_CACHED is not None:
        return _LIVE_PREFLIGHT_CACHED
    key = os.environ.get("ALPACA_BOD_API_KEY", "missing")
    sec = os.environ.get("ALPACA_BOD_API_SECRET", "missing")
    if key == "missing" or sec == "missing":
        _LIVE_PREFLIGHT_CACHED = {"ok": False, "reason": "no creds", "market_open": False}
        return _LIVE_PREFLIGHT_CACHED
    from alpaca.trading.client import TradingClient
    try:
        tc = TradingClient(key, sec, paper=True)
        clock = tc.get_clock()
        _LIVE_PREFLIGHT_CACHED = {
            "ok": True, "reason": "ok",
            "market_open": bool(clock.is_open),
        }
    except Exception as e:
        _LIVE_PREFLIGHT_CACHED = {"ok": False, "reason": f"auth failed: {e}",
                                  "market_open": False}
    return _LIVE_PREFLIGHT_CACHED


def _make_live_bot(tmpdir) -> TradingBot:
    import shutil
    fg_src = os.path.join(REPO_ROOT, "datasets", "fear_greed_forward_test_morning.csv")
    fg_path = os.path.join(tmpdir, "fg.csv")
    shutil.copy(fg_src, fg_path)
    cfg = BotConfig(
        name="LIVE_TEST",
        api_key=os.environ["ALPACA_BOD_API_KEY"],
        api_secret=os.environ["ALPACA_BOD_API_SECRET"],
        log_file=os.path.join(tmpdir, "log.csv"),
        fg_path=fg_path,
        state_file=os.path.join(tmpdir, "state.json"),
        target_time_et=dt.time(9, 35),
        symbol="SPY",
        sell_reconcile_sleep=2,
        fill_poll_seconds=20,
    )
    return TradingBot(cfg)


def _live_ensure_flat(bot: TradingBot, timeout: int = 30):
    """Best-effort: cancel open orders and close any SPY position. Skip if
    market is closed."""
    bot.cancel_open_orders()
    side, qty = bot.read_position()
    if side == "flat":
        return
    try:
        bot.tc.close_position(bot.cfg.symbol)
    except APIError as e:
        if "qty must be > 0" in str(e):
            return
        raise
    deadline = time.time() + timeout
    while time.time() < deadline:
        side, _ = bot.read_position()
        if side == "flat":
            return
        time.sleep(1)


def test_live_flat_sell_no_error():
    """LIVE: with a flat account, execute_sell must return NO_ACTION with no error.
    (Reproduces the 40010001 case, then verifies the new code doesn't repro.)"""
    pre = _live_preflight()
    if not pre["ok"]:
        print(f"  (skip: {pre['reason']})")
        return "SKIP"
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        bot = _make_live_bot(tmp)
        _live_ensure_flat(bot)
        action, qty, _, err = bot.execute_sell(0.0)
        assert action == "NO_ACTION", f"expected NO_ACTION, got {action}"
        assert qty == 0
        side, _ = bot.read_position()
        assert side == "flat"


def test_live_buy_then_sell_no_orphans():
    """LIVE: full buy/sell cycle on SPY must leave the account flat with
    no remaining position and no orphan shares, and never raise 40310000
    / 40010001."""
    pre = _live_preflight()
    if not pre["ok"]:
        print(f"  (skip: {pre['reason']})")
        return "SKIP"
    if not pre["market_open"]:
        print("  (skip: market closed - paper market orders only fill in regular session)")
        return "SKIP"
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        bot = _make_live_bot(tmp)
        _live_ensure_flat(bot)

        # Buy a tiny amount of SPY (cap target to 1 share for safety)
        price = bot.get_latest_price()
        assert price and price > 0, "could not fetch SPY price"

        # Temporarily override _target_buy_qty to just 1 share
        bot._target_buy_qty = lambda _p: 1
        action, qty, fill_price, err = bot.execute_buy(price)
        assert action == "BOUGHT", f"buy did not fill: action={action}, err={err}"
        assert qty == 1
        side, pos_qty = bot.read_position()
        assert side == "long" and pos_qty == 1
        assert bot.state.get_entry_date() is not None

        # Now sell - should sweep to flat with no errors
        action, qty, fill_price, err = bot.execute_sell(price)
        assert action == "SOLD" and qty == 1, f"sell failed: action={action} qty={qty} err={err}"
        side, pos_qty = bot.read_position()
        assert side == "flat" and pos_qty == 0, f"orphan! side={side} qty={pos_qty}"
        assert bot.state.get_entry_date() is None


def test_live_oversell_attempt_does_not_propagate():
    """LIVE: even if we *try* to oversell, the bot's execute_sell uses
    close_position which scopes to the actual qty - it cannot raise 40310000
    to the caller."""
    pre = _live_preflight()
    if not pre["ok"]:
        print(f"  (skip: {pre['reason']})")
        return "SKIP"
    if not pre["market_open"]:
        print("  (skip: market closed)")
        return "SKIP"
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        bot = _make_live_bot(tmp)
        _live_ensure_flat(bot)
        bot._target_buy_qty = lambda _p: 1
        price = bot.get_latest_price()
        action, qty, *_ = bot.execute_buy(price)
        assert action == "BOUGHT" and qty == 1

        # execute_sell must not raise even if the prior position view was stale
        action, qty, fill_price, err = bot.execute_sell(price)
        assert err is None or "40310000" not in str(err), f"oversell propagated: {err}"
        side, _ = bot.read_position()
        assert side == "flat"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def main():
    args = set(sys.argv[1:])
    run_mocks = ("--live" not in args)
    run_live = ("--mocks" not in args)

    if run_mocks:
        for name, fn in [
            ("mock: partial fill is swept to flat", test_mock_partial_fill_orphans),
            ("mock: oversell never submitted", test_mock_oversell_blocked),
            ("mock: flat -> NO_ACTION (no 40010001)", test_mock_flat_treated_as_no_action),
            ("mock: re-running recovers orphan shares", test_mock_run_cycle_idempotent_partial_recovery),
            ("mock: buy sizing respects 1x cash + headroom", test_mock_buy_never_oversizes_cash),
            ("mock: short detected first-class", test_mock_short_detected_as_first_class),
            ("mock: Days_Held counted in trading days", test_mock_days_held_trading_days),
            ("mock: APIError.code structured access", test_mock_api_error_code_via_attribute),
            ("mock: stale state cleared when broker flat", test_mock_state_cleared_when_flat),
        ]:
            _run(name, fn)

    if run_live:
        for name, fn in [
            ("live: flat SELL -> NO_ACTION", test_live_flat_sell_no_error),
            ("live: buy then sell leaves no orphans", test_live_buy_then_sell_no_orphans),
            ("live: oversell attempt does not propagate", test_live_oversell_attempt_does_not_propagate),
        ]:
            _run(name, fn)

    print(f"\nResults: PASS={PASS} FAIL={FAIL} SKIP={SKIP}")
    sys.exit(0 if FAIL == 0 else 1)


if __name__ == "__main__":
    main()
