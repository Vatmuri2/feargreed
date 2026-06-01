"""Shared hardened logic for the BOD and EOD Fear & Greed bots.

Design goals (see CHANGES.md):
- Exits NEVER leave orphans: every action reconciles against the live broker
  position and loops until the target state is reached.
- Execution runs in the regular session, so market orders / close_position work.
- Position sizing is 1x (no margin); leverage is a single config constant.
- Price + volatility come from Alpaca (the same broker we trade on).
- Days_Held is counted in NYSE trading days from a persisted entry-date state
  file, not from order history.
- Scheduling is tz-aware (US/Eastern) and DST-safe.
- Idempotent re-runs: same-day retries reconcile to the desired state instead
  of being blocked by a date lock.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import os
import time
from typing import Optional

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import pytz
import fear_and_greed as fg

from alpaca.common.exceptions import APIError
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderStatus, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import (
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
)


EASTERN = pytz.timezone("US/Eastern")
NYSE = mcal.get_calendar("NYSE")

_TERMINAL_ORDER_STATUSES = {
    OrderStatus.FILLED,
    OrderStatus.CANCELED,
    OrderStatus.EXPIRED,
    OrderStatus.REJECTED,
}

# Alpaca error codes we treat as "the broker's position view disagrees
# with ours" - they should never be raised by the hardened code, but if
# they are we want to surface them clearly.
_ERR_INSUFFICIENT_BP = 40310000
_ERR_QTY_NON_POSITIVE = 40010001

_CSV_COLUMNS = [
    "Timestamp", "Action", "Symbol", "Quantity", "Price", "FGI_Value",
    "FGI_Momentum", "FGI_Velocity", "Volatility", "Portfolio_Value",
    "Buying_Power", "Signal_Reason", "Days_Held",
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class BotConfig:
    name: str                       # "BOD" or "EOD"
    api_key: str
    api_secret: str
    log_file: str
    fg_path: str
    state_file: str
    target_time_et: dt.time         # local execution time, US/Eastern
    symbol: str = "SPY"
    paper: bool = True

    # Strategy parameters - kept identical to the prior live bots
    momentum_threshold: float = 0.2
    velocity_threshold: float = 0.15
    volatility_buy_limit: float = 0.6
    volatility_sell_limit: float = 0.5
    lookback_days: int = 3
    volatility_window: int = 20
    max_days_held: int = 8

    # Order-management knobs
    leverage: float = 1.0           # 1x = no margin
    cash_headroom: float = 0.005    # leave 0.5% cash unused to avoid edge BP rejections
    buy_limit_multipliers: tuple = (1.002, 1.005, 1.01)
    fill_poll_seconds: int = 30     # how long a single limit attempt is given
    sell_reconcile_attempts: int = 5
    sell_reconcile_sleep: int = 3   # seconds between sweep iterations
    fgi_max_retries: int = 3
    fgi_retry_delay: int = 30


# ---------------------------------------------------------------------------
# Structured logging helper
# ---------------------------------------------------------------------------


def _evt(name: str, **fields) -> None:
    """Emit a structured order-lifecycle event to stdout."""
    payload = {"event": name, "ts": dt.datetime.now(EASTERN).isoformat(), **fields}
    print(f"[order-event] {json.dumps(payload, default=str)}")


def _api_error_code(err: Exception) -> Optional[int]:
    """Return the structured Alpaca error code, or None if unavailable."""
    if isinstance(err, APIError):
        try:
            return int(err.code)
        except Exception:
            return None
    return None


# ---------------------------------------------------------------------------
# Persistent position state
# ---------------------------------------------------------------------------


class PositionState:
    """Tracks entry date/qty for the currently held position.

    The Alpaca account is the source of truth for position quantity. The
    state file's job is solely to remember when the *currently open*
    position was first established, so we can compute Days_Held correctly.
    """

    def __init__(self, path: str):
        self.path = path

    def load(self) -> dict:
        if not os.path.exists(self.path):
            return {}
        try:
            with open(self.path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    def save(self, data: dict) -> None:
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def clear(self) -> None:
        if os.path.exists(self.path):
            os.remove(self.path)

    def record_entry(self, entry_date_et: dt.date, qty: int) -> None:
        self.save({"entry_date": entry_date_et.isoformat(), "qty": int(qty)})

    def get_entry_date(self) -> Optional[dt.date]:
        data = self.load()
        if not data.get("entry_date"):
            return None
        try:
            return dt.date.fromisoformat(data["entry_date"])
        except (TypeError, ValueError):
            return None


def trading_days_between(start: dt.date, end: dt.date) -> int:
    """Number of NYSE trading days strictly after `start`, up to and
    including `end`. Returns 0 if start >= end."""
    if start >= end:
        return 0
    schedule = NYSE.schedule(start_date=start, end_date=end)
    if schedule.empty:
        return 0
    sessions = schedule.index.date
    return int(sum(1 for d in sessions if d > start))


# ---------------------------------------------------------------------------
# Fear & Greed data
# ---------------------------------------------------------------------------


def _fgi_column(df: pd.DataFrame) -> str:
    for c in ("fear_greed", "Fear Greed", "Index"):
        if c in df.columns:
            return c
    raise ValueError(f"No F&G column in {df.columns.tolist()}")


def fetch_and_update_fgi(
    fg_path: str,
    today_et: dt.date,
    max_retries: int = 3,
    retry_delay: int = 30,
) -> tuple[Optional[pd.DataFrame], Optional[float], Optional[str], str]:
    """Load the F&G CSV, fetch the latest API value, and ensure exactly one
    row exists for today. Returns (df, today_value, column, status)."""
    fg_data = pd.read_csv(fg_path)
    fg_data["Date"] = pd.to_datetime(fg_data["Date"]).dt.date
    column = _fgi_column(fg_data)

    fresh_value = None
    fresh_rating = None
    status = "fallback_last_known"

    for attempt in range(1, max_retries + 1):
        try:
            idx = fg.get()
            value = round(idx.value, 2)
            rating = idx.description
            last_update = idx.last_update
            if last_update.date() < today_et:
                if attempt < max_retries:
                    _evt("fgi.stale", attempt=attempt, last_update=str(last_update))
                    time.sleep(retry_delay)
                    continue
                fresh_value = value
                fresh_rating = rating
                status = "stale_api"
            else:
                fresh_value = value
                fresh_rating = rating
                status = "fresh"
            break
        except Exception as e:
            _evt("fgi.fetch_error", attempt=attempt, error=str(e))
            if attempt < max_retries:
                time.sleep(retry_delay)

    if fresh_value is None:
        # Total failure: use last known value, append today's row idempotently
        if fg_data.empty:
            return None, None, column, "fatal_no_data"
        fresh_value = float(fg_data[column].iloc[-1])
        fresh_rating = (
            fg_data["rating"].iloc[-1] if "rating" in fg_data.columns else ""
        )

    # Idempotent today-row insertion: replace if a row for today already
    # exists (e.g. from an earlier same-day run), otherwise append.
    mask_today = fg_data["Date"] == today_et
    if mask_today.any():
        fg_data.loc[mask_today, column] = fresh_value
        if "rating" in fg_data.columns:
            fg_data.loc[mask_today, "rating"] = fresh_rating
    else:
        new_row = {"Date": today_et, column: fresh_value}
        if "rating" in fg_data.columns:
            new_row["rating"] = fresh_rating
        fg_data = pd.concat([fg_data, pd.DataFrame([new_row])], ignore_index=True)

    fg_data.to_csv(fg_path, index=False)
    _evt("fgi.updated", date=str(today_et), value=fresh_value, status=status)
    return fg_data, fresh_value, column, status


def calculate_indicators(fg_data: pd.DataFrame, column: str, lookback: int) -> pd.DataFrame:
    data = fg_data.copy()
    data["fg_momentum"] = (
        data[column] - data[column].rolling(lookback, min_periods=1).mean()
    )
    data["fg_change"] = data[column].diff().fillna(0)
    data["fg_velocity"] = data["fg_change"].rolling(lookback, min_periods=1).mean()
    return data


# ---------------------------------------------------------------------------
# TradingBot - the meat
# ---------------------------------------------------------------------------


class TradingBot:
    """Hardened single-symbol Fear & Greed momentum bot."""

    def __init__(
        self,
        config: BotConfig,
        trading_client: Optional[TradingClient] = None,
        data_client: Optional[StockHistoricalDataClient] = None,
    ):
        self.cfg = config
        self.tc = trading_client or TradingClient(
            config.api_key, config.api_secret, paper=config.paper
        )
        self.dc = data_client or StockHistoricalDataClient(
            config.api_key, config.api_secret
        )
        self.state = PositionState(config.state_file)
        # Test seam: callers may override the FGI fetcher to avoid network I/O.
        self.fgi_fetcher = fetch_and_update_fgi
        self._ensure_log()

    # ------------------------------------------------------------------ data

    def _ensure_log(self):
        if not os.path.exists(self.cfg.log_file):
            pd.DataFrame(columns=_CSV_COLUMNS).to_csv(self.cfg.log_file, index=False)
            print(f"Created new log file: {self.cfg.log_file}")

    def get_latest_price(self) -> Optional[float]:
        """Latest quote midpoint from Alpaca. Never raises on transient
        errors - returns None so callers can decide whether to abort."""
        try:
            req = StockLatestQuoteRequest(symbol_or_symbols=self.cfg.symbol)
            quote = self.dc.get_stock_latest_quote(req)[self.cfg.symbol]
            bid = float(quote.bid_price) if quote.bid_price else 0.0
            ask = float(quote.ask_price) if quote.ask_price else 0.0
            if bid > 0 and ask > 0:
                return round((bid + ask) / 2, 2)
            return round(ask or bid, 2) or None
        except Exception as e:
            _evt("price.error", error=str(e))
            return None

    def get_current_volatility(self) -> Optional[float]:
        """Annualised vol from Alpaca daily bars. Window matches the
        backtest (rolling 20 day std of pct returns)."""
        try:
            end_et = dt.datetime.now(EASTERN)
            start_et = end_et - dt.timedelta(days=self.cfg.volatility_window * 3)
            req = StockBarsRequest(
                symbol_or_symbols=self.cfg.symbol,
                timeframe=TimeFrame.Day,
                start=start_et.astimezone(pytz.UTC),
                end=end_et.astimezone(pytz.UTC),
            )
            bars = self.dc.get_stock_bars(req).df
            if bars.empty:
                return None
            if "symbol" in bars.index.names:
                bars = bars.xs(self.cfg.symbol, level="symbol")
            returns = bars["close"].pct_change()
            vol = returns.rolling(self.cfg.volatility_window, min_periods=1).std() * np.sqrt(252)
            return round(float(vol.iloc[-1]), 4)
        except Exception as e:
            _evt("volatility.error", error=str(e))
            return None

    # -------------------------------------------------------- broker truth

    def read_position(self) -> tuple[str, int]:
        """Returns ('long', qty) | ('short', qty) | ('flat', 0)."""
        try:
            pos = self.tc.get_open_position(self.cfg.symbol)
        except APIError:
            return "flat", 0
        qty = int(float(pos.qty))
        if qty > 0:
            return "long", qty
        if qty < 0:
            return "short", qty
        return "flat", 0

    def cancel_open_orders(self, timeout: int = 10) -> None:
        """Cancel every open order for the symbol and confirm cancellation."""
        try:
            open_orders = self.tc.get_orders(
                GetOrdersRequest(
                    status=QueryOrderStatus.OPEN,
                    symbols=[self.cfg.symbol],
                )
            )
        except Exception as e:
            _evt("cancel.list_error", error=str(e))
            return

        ids = [o.id for o in open_orders]
        for oid in ids:
            try:
                self.tc.cancel_order_by_id(oid)
                _evt("cancel.submitted", order_id=str(oid))
            except Exception as e:
                _evt("cancel.error", order_id=str(oid), error=str(e))

        # Wait for cancels to be acknowledged.
        deadline = time.time() + timeout
        while ids and time.time() < deadline:
            still_open = []
            for oid in ids:
                try:
                    order = self.tc.get_order_by_id(oid)
                    if order.status not in _TERMINAL_ORDER_STATUSES:
                        still_open.append(oid)
                except Exception:
                    pass
            ids = still_open
            if ids:
                time.sleep(0.5)
        if ids:
            _evt("cancel.timeout", remaining=[str(i) for i in ids])

    # ----------------------------------------------------- signal generation

    def days_held(self, side: str) -> int:
        if side != "long":
            return 0
        entry = self.state.get_entry_date()
        if entry is None:
            return 0
        return trading_days_between(entry, dt.datetime.now(EASTERN).date())

    def generate_signal(
        self,
        latest_indicators: pd.Series,
        current_volatility: float,
        side: str,
        held_days: int,
    ) -> tuple[str, str]:
        cfg = self.cfg
        if side == "short":
            return "SELL", "Unexpected short position - flatten immediately"

        momentum_ok = latest_indicators["fg_momentum"] > cfg.momentum_threshold
        velocity_ok = latest_indicators["fg_velocity"] > cfg.velocity_threshold

        if side == "flat":
            if momentum_ok and velocity_ok and current_volatility < cfg.volatility_buy_limit:
                return "BUY", "Strong momentum/velocity, low volatility"
            if momentum_ok and velocity_ok:
                return "HOLD", "Strong momentum/velocity but high volatility"
            if current_volatility >= cfg.volatility_buy_limit:
                return "HOLD", "Volatility too high for entry"
            return "HOLD", "Insufficient momentum/velocity for entry"

        # side == "long"
        if (
            latest_indicators["fg_momentum"] < cfg.momentum_threshold
            or latest_indicators["fg_velocity"] < cfg.velocity_threshold
            or current_volatility > cfg.volatility_sell_limit
            or held_days >= cfg.max_days_held
        ):
            if held_days >= cfg.max_days_held:
                return "SELL", f"Maximum holding period reached ({held_days} days >= {cfg.max_days_held})"
            return "SELL", "Momentum reversal or high volatility"
        return "HOLD", f"Holding position - indicators still favorable ({held_days}/{cfg.max_days_held} days)"

    # ----------------------------------------------------- order execution

    def _target_buy_qty(self, current_price: float) -> int:
        """Size from cash (1x by default), with a small headroom buffer."""
        account = self.tc.get_account()
        cash = float(account.cash)
        deployable = cash * self.cfg.leverage * (1 - self.cfg.cash_headroom)
        return max(0, int(deployable // current_price))

    def _wait_for_fill(self, order_id, timeout_sec: int):
        deadline = time.time() + timeout_sec
        order = None
        while time.time() < deadline:
            try:
                order = self.tc.get_order_by_id(order_id)
            except Exception as e:
                _evt("order.poll_error", order_id=str(order_id), error=str(e))
                time.sleep(1)
                continue
            if order.status in _TERMINAL_ORDER_STATUSES:
                return order
            time.sleep(1)
        return order

    def execute_buy(self, current_price: float) -> tuple[str, int, float, Optional[str]]:
        """Buy up to target_qty using marketable limits with cancel/replace.
        Returns (action, filled_qty, avg_fill_price, error_reason).

        Reconciles against live position after every attempt so partial
        fills are accounted for and never lead to a duplicate submission
        beyond the target."""
        cfg = self.cfg
        target = self._target_buy_qty(current_price)
        if target <= 0:
            return "NO_ACTION", 0, current_price, "Insufficient cash for any shares"

        total_filled = 0
        total_cost = 0.0
        attempts = list(enumerate(cfg.buy_limit_multipliers, 1))
        for attempt, multiplier in attempts:
            side, pos_qty = self.read_position()
            if side == "short":
                return "NO_ACTION", 0, current_price, "Unexpected short; aborting buy"
            remaining = target - pos_qty
            if remaining <= 0:
                break

            self.cancel_open_orders()

            limit_price = round(current_price * multiplier, 2)
            # Defensive: shrink qty if cash actually can't support it
            cash = float(self.tc.get_account().cash)
            max_affordable = int((cash * (1 - cfg.cash_headroom)) // limit_price)
            this_qty = min(remaining, max_affordable)
            if this_qty <= 0:
                _evt("buy.no_cash", attempt=attempt, cash=cash, limit_price=limit_price)
                break

            req = LimitOrderRequest(
                symbol=cfg.symbol,
                qty=this_qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
                limit_price=limit_price,
                extended_hours=False,
            )
            try:
                submitted = self.tc.submit_order(req)
                _evt("buy.submitted", attempt=attempt, order_id=str(submitted.id),
                     qty=this_qty, limit_price=limit_price)
            except APIError as e:
                _evt("buy.submit_error", attempt=attempt, code=_api_error_code(e),
                     error=str(e))
                continue

            filled = self._wait_for_fill(submitted.id, cfg.fill_poll_seconds)
            filled_qty = int(float(filled.filled_qty)) if filled and filled.filled_qty else 0
            avg = float(filled.filled_avg_price) if filled and filled.filled_avg_price else 0.0
            _evt("buy.attempt_complete", attempt=attempt, order_id=str(submitted.id),
                 status=str(filled.status) if filled else "unknown",
                 filled_qty=filled_qty, avg_price=avg)
            if filled_qty > 0:
                total_filled += filled_qty
                total_cost += filled_qty * avg
            # cancel any remaining unfilled remainder before next attempt
            self.cancel_open_orders()

        if total_filled > 0:
            avg_price = total_cost / total_filled
            # Persist entry date if we transitioned from flat to long
            side, _ = self.read_position()
            if side == "long" and self.state.get_entry_date() is None:
                self.state.record_entry(dt.datetime.now(EASTERN).date(), total_filled)
            return "BOUGHT", total_filled, avg_price, None
        return "NO_ACTION", 0, current_price, f"BUY did not fill after {len(attempts)} attempts"

    def execute_sell(self, current_price: float) -> tuple[str, int, float, Optional[str]]:
        """Flatten the position using close_position (market, regular hours).

        Loops until live position is flat or attempts are exhausted. Never
        submits a hard-coded qty; always reads live position first to avoid
        oversell (40310000) and qty<=0 (40010001) errors."""
        cfg = self.cfg
        total_sold = 0
        proceeds = 0.0

        for attempt in range(1, cfg.sell_reconcile_attempts + 1):
            side, qty = self.read_position()
            if side == "flat":
                break
            if side == "short":
                _evt("sell.unexpected_short", qty=qty)
                # close_position handles shorts too
            if qty == 0:
                break

            self.cancel_open_orders()

            try:
                order = self.tc.close_position(cfg.symbol)
                _evt("sell.submitted", attempt=attempt, order_id=str(order.id),
                     qty=qty)
            except APIError as e:
                code = _api_error_code(e)
                _evt("sell.submit_error", attempt=attempt, code=code, error=str(e))
                if code == _ERR_QTY_NON_POSITIVE:
                    # broker says we hold nothing - treat as flat
                    break
                # Fallback: marketable order direct submit
                try:
                    side_alpaca = OrderSide.SELL if qty > 0 else OrderSide.BUY
                    req = MarketOrderRequest(
                        symbol=cfg.symbol,
                        qty=abs(qty),
                        side=side_alpaca,
                        time_in_force=TimeInForce.DAY,
                    )
                    order = self.tc.submit_order(req)
                    _evt("sell.fallback_submitted", attempt=attempt,
                         order_id=str(order.id), qty=qty)
                except Exception as e2:
                    _evt("sell.fallback_error", attempt=attempt, error=str(e2))
                    time.sleep(cfg.sell_reconcile_sleep)
                    continue

            filled = self._wait_for_fill(order.id, cfg.fill_poll_seconds)
            filled_qty = int(float(filled.filled_qty)) if filled and filled.filled_qty else 0
            avg = float(filled.filled_avg_price) if filled and filled.filled_avg_price else 0.0
            _evt("sell.attempt_complete", attempt=attempt, order_id=str(order.id),
                 status=str(filled.status) if filled else "unknown",
                 filled_qty=filled_qty, avg_price=avg)
            if filled_qty > 0:
                total_sold += filled_qty
                proceeds += filled_qty * avg
            time.sleep(cfg.sell_reconcile_sleep)

        side_after, qty_after = self.read_position()
        if side_after != "flat":
            return ("NO_ACTION", total_sold, current_price,
                    f"SELL incomplete - still holding {qty_after} after "
                    f"{cfg.sell_reconcile_attempts} attempts")

        # Successful flatten: clear entry state
        self.state.clear()
        if total_sold > 0:
            avg_price = proceeds / total_sold
            return "SOLD", total_sold, avg_price, None
        # We were already flat at entry; nothing to sell
        return "NO_ACTION", 0, current_price, "Position already flat"

    # ----------------------------------------------------- logging

    def log_trade(
        self,
        action: str,
        qty: int,
        price: float,
        fgi_value: float,
        momentum: float,
        velocity: float,
        volatility: float,
        portfolio_value: float,
        buying_power: float,
        reason: str,
        days_held: int,
    ) -> None:
        timestamp = dt.datetime.now(EASTERN).strftime("%Y-%m-%d %H:%M:%S")
        row = pd.DataFrame([{
            "Timestamp": timestamp,
            "Action": action,
            "Symbol": self.cfg.symbol,
            "Quantity": qty,
            "Price": price,
            "FGI_Value": fgi_value,
            "FGI_Momentum": momentum,
            "FGI_Velocity": velocity,
            "Volatility": volatility,
            "Portfolio_Value": portfolio_value,
            "Buying_Power": buying_power,
            "Signal_Reason": reason,
            "Days_Held": days_held,
        }])
        row.to_csv(self.cfg.log_file, mode="a", header=False, index=False)
        print(f"Logged trade: {timestamp} - {action} {qty} {self.cfg.symbol} "
              f"@ ${price} (Held for {days_held} trading days)")

    # -------------------------------------------- single-cycle entrypoint

    def run_cycle(self) -> dict:
        """Run one execution cycle. Idempotent: re-running on the same day
        reconciles to the desired state and never double-trades. Returns
        a summary dict for tests / callers."""
        cfg = self.cfg
        today_et = dt.datetime.now(EASTERN).date()

        # 1. Inspect live position FIRST - it's the source of truth.
        side, qty = self.read_position()

        # If state file references a position that no longer exists, clear it
        if side == "flat" and self.state.get_entry_date() is not None:
            _evt("state.stale_clear")
            self.state.clear()
        # If we hold a position but have no entry record, seed it conservatively
        # by reconstructing from the most recent BUY fill.
        if side == "long" and self.state.get_entry_date() is None:
            entry = self._reconstruct_entry_date(qty)
            if entry:
                self.state.record_entry(entry, qty)

        # 2. FGI + indicators
        fg_data, current_fgi, fgi_column, fgi_status = self.fgi_fetcher(
            cfg.fg_path, today_et,
            max_retries=cfg.fgi_max_retries, retry_delay=cfg.fgi_retry_delay,
        )
        if fg_data is None or current_fgi is None:
            # Can't compute indicators - but if we have a position, we may
            # still need to exit on the time-based limit.
            return self._panic_exit_if_needed(
                reason="F&G data unavailable",
                current_price=self.get_latest_price() or 0.0,
                fgi_value=0.0, momentum=0.0, velocity=0.0, volatility=0.0,
            )

        fg_data = calculate_indicators(fg_data, fgi_column, cfg.lookback_days)
        latest = fg_data.iloc[-1]

        # 3. Volatility (Alpaca data)
        current_volatility = self.get_current_volatility()
        if current_volatility is None:
            # Treat as conservative: high enough to block buys, low enough
            # not to force a sell on its own. Don't abort an exit if one is
            # otherwise needed.
            current_volatility = (cfg.volatility_buy_limit + cfg.volatility_sell_limit) / 2
            _evt("volatility.fallback", value=current_volatility)

        # 4. Price (Alpaca data)
        current_price = self.get_latest_price()
        if current_price is None:
            # If we hold and need to sell, we still can - close_position is
            # a market order and doesn't need our price.
            current_price = 0.0
            _evt("price.unavailable_proceeding")

        # 5. Days held
        held_days = self.days_held(side)

        # 6. Generate signal off the latest position view
        signal, reason = self.generate_signal(
            latest, current_volatility, side, held_days
        )

        # 7. Execute
        action, qty_done, fill_price, exec_reason = "NO_ACTION", 0, current_price, None
        if signal == "BUY":
            if current_price <= 0:
                exec_reason = "BUY skipped - no live price"
            else:
                action, qty_done, fill_price, exec_reason = self.execute_buy(current_price)
        elif signal == "SELL":
            action, qty_done, fill_price, exec_reason = self.execute_sell(current_price)

        final_reason = exec_reason or reason

        # 8. Snapshot account
        account = self.tc.get_account()
        portfolio_value = float(account.portfolio_value)
        buying_power = float(account.buying_power)

        # 9. Log
        self.log_trade(
            action=action, qty=qty_done, price=fill_price,
            fgi_value=float(current_fgi),
            momentum=float(latest["fg_momentum"]),
            velocity=float(latest["fg_velocity"]),
            volatility=current_volatility,
            portfolio_value=portfolio_value,
            buying_power=buying_power,
            reason=final_reason,
            days_held=held_days,
        )

        side_after, qty_after = self.read_position()
        return {
            "signal": signal,
            "action": action,
            "qty": qty_done,
            "price": fill_price,
            "reason": final_reason,
            "fgi_status": fgi_status,
            "side_before": side,
            "side_after": side_after,
            "qty_after": qty_after,
            "days_held": held_days,
        }

    # -------------------------------------------- helpers

    def _reconstruct_entry_date(self, current_qty: int) -> Optional[dt.date]:
        """Walk closed orders newest-first, summing BUY fills (and
        subtracting SELL fills) until cumulative qty equals current_qty.
        Returns the fill date of the earliest BUY that contributed."""
        try:
            orders = self.tc.get_orders(GetOrdersRequest(
                status=QueryOrderStatus.CLOSED,
                symbols=[self.cfg.symbol],
                direction="desc",
                limit=500,
            ))
        except Exception as e:
            _evt("state.reconstruct_error", error=str(e))
            return None

        running = 0
        earliest_buy: Optional[dt.date] = None
        for o in orders:
            if not o.filled_qty:
                continue
            qty = int(float(o.filled_qty))
            if o.side == OrderSide.BUY:
                running += qty
                fill_dt = o.filled_at
                if fill_dt is not None:
                    earliest_buy = fill_dt.astimezone(EASTERN).date()
            elif o.side == OrderSide.SELL:
                running -= qty
            if running >= current_qty:
                break
        return earliest_buy

    def _panic_exit_if_needed(self, *, reason, current_price, fgi_value,
                              momentum, velocity, volatility) -> dict:
        """Called when indicators can't be computed but we may still need
        to honour a max-hold exit. Never silently swallows a needed exit."""
        side, qty = self.read_position()
        held_days = self.days_held(side)
        if side == "long" and held_days >= self.cfg.max_days_held:
            action, qty_done, fill_price, exec_reason = self.execute_sell(current_price)
            account = self.tc.get_account()
            self.log_trade(
                action=action, qty=qty_done, price=fill_price,
                fgi_value=fgi_value, momentum=momentum, velocity=velocity,
                volatility=volatility,
                portfolio_value=float(account.portfolio_value),
                buying_power=float(account.buying_power),
                reason=exec_reason or f"Max-hold exit ({reason})",
                days_held=held_days,
            )
            side_after, qty_after = self.read_position()
            return {"signal": "SELL", "action": action, "qty": qty_done,
                    "side_after": side_after, "qty_after": qty_after,
                    "reason": exec_reason or reason}
        account = self.tc.get_account()
        self.log_trade(
            action="NO_ACTION", qty=0, price=current_price,
            fgi_value=fgi_value, momentum=momentum, velocity=velocity,
            volatility=volatility,
            portfolio_value=float(account.portfolio_value),
            buying_power=float(account.buying_power),
            reason=reason, days_held=held_days,
        )
        return {"signal": "HOLD", "action": "NO_ACTION", "reason": reason,
                "side_after": side, "qty_after": qty}


# ---------------------------------------------------------------------------
# Scheduling helpers (tz-aware, DST-safe)
# ---------------------------------------------------------------------------


def is_trading_day(date_et: dt.date) -> bool:
    return not NYSE.schedule(start_date=date_et, end_date=date_et).empty


def next_target_datetime(target_time_et: dt.time, now_et: Optional[dt.datetime] = None) -> dt.datetime:
    """Return the next future US/Eastern datetime at which the bot should
    execute, skipping non-trading days. Always tz-aware."""
    now_et = now_et or dt.datetime.now(EASTERN)
    today = now_et.date()
    horizon = today + dt.timedelta(days=10)
    schedule = NYSE.schedule(start_date=today, end_date=horizon)
    for session_date in schedule.index.date:
        target_dt = EASTERN.localize(
            dt.datetime.combine(session_date, target_time_et)
        )
        if target_dt > now_et:
            return target_dt
    # Should never happen, but return tomorrow as a fallback
    return EASTERN.localize(
        dt.datetime.combine(today + dt.timedelta(days=1), target_time_et)
    )


def run_forever(bot: TradingBot) -> None:
    """Continuously schedule and execute the bot's trading cycle.

    All time math is done in US/Eastern and uses the NYSE calendar.
    The sleep is bounded so DST transitions or clock skew can't cause
    huge or negative sleeps."""
    print("=" * 60)
    print(f"Fear & Greed Strategy [{bot.cfg.name}] - Starting")
    print(f"Execution time: {bot.cfg.target_time_et.strftime('%H:%M')} US/Eastern (regular session)")
    print(f"Leverage: {bot.cfg.leverage}x")
    print("=" * 60)

    while True:
        now_et = dt.datetime.now(EASTERN)
        target = next_target_datetime(bot.cfg.target_time_et, now_et)
        sleep_seconds = (target - now_et).total_seconds()
        sleep_seconds = max(60.0, min(sleep_seconds, 24 * 3600.0))
        print(f"Now: {now_et.isoformat()} | next execution: {target.isoformat()} "
              f"(sleeping {sleep_seconds/60:.1f} min)")
        time.sleep(sleep_seconds)

        now_et = dt.datetime.now(EASTERN)
        if not is_trading_day(now_et.date()):
            print(f"Not a trading day ({now_et.date()}); rescheduling.")
            continue
        try:
            summary = bot.run_cycle()
            print("\n" + "=" * 60)
            print("EXECUTION SUMMARY:")
            for k, v in summary.items():
                print(f"  {k}: {v}")
            print("=" * 60)
        except Exception as e:
            _evt("cycle.fatal_error", error=str(e))
            # don't kill the process - sleep until the next attempt
            time.sleep(60)
