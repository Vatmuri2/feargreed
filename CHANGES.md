# Hardening the BOD / EOD Fear & Greed Bots

This change set rewrites the two live bots around the principle that **the
broker is the source of truth**: every action reconciles against the live
Alpaca position rather than trusting a locally-computed quantity. Each fix
from the task spec is mapped to the new code below.

## File layout

Old (two near-identical 500-line scripts):

  ap_fgi_bod.py
  ap_fgi_eod.py

New:

  fgi_trading.py        # all shared logic - BotConfig, TradingBot, scheduler
  ap_fgi_bod.py         # 30-line entrypoint: builds a BOD BotConfig and runs
  ap_fgi_eod.py         # 30-line entrypoint: builds an EOD BotConfig and runs
  state_BOD.json        # persisted entry date/qty for the BOD position (created on first BUY)
  state_EOD.json        # persisted entry date/qty for the EOD position
  tests/test_regressions.py  # mock + live regression suite

Spec uses the names `trading_bot_BOD.py` / `trading_bot_EOD.py`; the actual
repo files are `ap_fgi_bod.py` / `ap_fgi_eod.py`, so those were edited
in place. Both bots are configured identically (same parameters, different
keys/log/state/FGI files/execution time).

## Section A - Execution / order management

### A1. Partial fills no longer orphan shares

* **Before:** `execute_trade` returned `"SOLD"` after a partial fill and
  abandoned the remainder; the resting order was sometimes not cancelled.
* **After:** [fgi_trading.py:execute_sell](fgi_trading.py) loops up to
  `sell_reconcile_attempts` times. Each iteration:
  1. reads the live position (`read_position`),
  2. cancels every open order for the symbol (`cancel_open_orders`,
     waits for cancellation to be acknowledged),
  3. submits `close_position(symbol)`,
  4. polls the order to terminal state,
  5. re-reads the position - exits only when `side == "flat"`.

  If the loop exits while shares are still held, the function returns
  `"NO_ACTION"` with an error reason. No code path now reports `"SOLD"`
  while shares remain.

### A2. Sells reconcile against the live position (no oversell, no qty<=0)

* **Before:** `execute_trade` sold a `qty` computed earlier in the cycle,
  even if it disagreed with the broker. This raised:
    - `40310000` ("insufficient buying power") when computed_qty > owned (2026-05-05)
    - `40010001` ("qty must be > 0") when computed_qty was stale (2026-05-11..14)
* **After:** `execute_sell` never accepts an external qty. It calls
  `close_position(symbol)`, which Alpaca scopes to whatever is owned. If the
  live position is flat, it returns `"NO_ACTION"` immediately. The
  `_ERR_QTY_NON_POSITIVE` (40010001) APIError is explicitly handled as
  "already flat".

### A3. Cancel/replace with confirmed cancellation

* **Before:** `wait_for_fill` waited 180s; cancels were fire-and-forget
  (`try ... except Exception: pass`), racing with potential fills.
* **After:** `cancel_open_orders` lists open orders for the symbol,
  submits a cancel for each, then **polls every order ID** until it
  reaches a terminal status (or a 10s timeout). After cancellation the
  bot always re-reads the live position before deciding what to do next.

### A4. Execution moved into the regular session

* **Before:** BOD ran at 09:20 ET (pre-market); EOD at 16:10 ET (after-hours).
  Market orders / `close_position` are rejected outside regular session,
  which forced the bot to use illiquid limit orders with retries.
* **After:**
  - BOD: `target_time_et=09:35` ET (5 minutes after open).
  - EOD: `target_time_et=15:50` ET (10 minutes before close - matches the backtest).
  - `extended_hours=False` everywhere.

  See [ap_fgi_bod.py](ap_fgi_bod.py) and [ap_fgi_eod.py](ap_fgi_eod.py).

### A5. 1x leverage, sized from cash

* **Before:** `qty_to_buy = int(buying_power / current_price)` deployed
  full 2x margin BP every entry.
* **After:** `_target_buy_qty` reads `account.cash`, multiplies by
  `cfg.leverage` (default `1.0`) and applies a `cfg.cash_headroom` buffer
  (default 0.5%). No margin is used. Leverage is a single named config
  constant. Each buy attempt also re-checks live cash before sizing the
  next limit slice.

### A6. Price & volatility come from Alpaca

* **Before:** `yfinance.Ticker(...).fast_info['lastPrice']` and
  `yfinance.Ticker(...).history(...)`. Flaky and could KeyError or return
  stale pre-market values, with errors causing the whole cycle (including
  exits) to be skipped.
* **After:** `get_latest_price` uses Alpaca `StockLatestQuoteRequest` and
  returns the bid/ask midpoint; `get_current_volatility` uses
  `StockBarsRequest` daily bars over 60 calendar days, computes a 20-day
  rolling pct-change std annualized by sqrt(252), identical to
  `backtest/main.py`. Both helpers return `None` on failure; the cycle
  proceeds with fallback values rather than aborting a needed exit
  (`_panic_exit_if_needed`).

## Section B - Position & holding-period state

### B7. Days_Held now from a state file, in NYSE trading days, ET

* **Before:** `get_position_entry_date` scanned closed orders descending
  and returned the first BUY date - including BUYs from previous round
  trips. UTC `.date()` and an implicit 50-order limit. This is what
  produced the 5/6/7/8 incrementing `Days_Held` while the account was flat.
* **After:**
  - `PositionState` persists `{entry_date, qty}` as JSON to
    `state_BOD.json` / `state_EOD.json`.
  - `record_entry` is called from `execute_buy` only when the bot
    transitions from flat to long.
  - `state.clear()` is called from `execute_sell` once the position is
    confirmed flat.
  - `_reconstruct_entry_date` is a one-shot recovery path: if the bot
    starts up holding a position with no state file, it walks closed
    orders newest-first, netting BUY-SELL fills, and persists the entry
    date of the earliest BUY that contributed to the current qty
    (pulls up to 500 orders, not 50).
  - `trading_days_between(start, end)` uses `pandas_market_calendars`
    NYSE schedule and counts sessions strictly after `start` up to `end`.
    Days_Held in `run_cycle` is computed from the persisted entry date
    in US/Eastern.
  - If the broker reports flat but state has an entry, the state file is
    cleared with a `state.stale_clear` event.

### B8. Position is read once per cycle and threaded through

* **Before:** `get_current_position` was called separately in
  `generate_signal` and again in `execute_trade`; results could disagree.
* **After:** `run_cycle` calls `read_position()` once and passes `side`
  through to `generate_signal`. `execute_sell` / `execute_buy` re-read
  the live position internally only to verify reconciliation, never to
  re-derive intent.

### B9. Shorts are first-class

* **Before:** `get_current_position` collapsed any non-positive qty to
  `(False, 0)`, hiding shorts from everything except the dedicated
  `close_unexpected_short` check.
* **After:** `read_position` returns one of `("flat", 0)`, `("long", qty>0)`,
  `("short", qty<0)`. `generate_signal` treats `short` as an unconditional
  SELL signal with reason "Unexpected short position - flatten immediately".
  `execute_sell` is short-aware (uses `close_position`, which handles both
  directions).

## Section C - Scheduling, idempotency, recovery

### C10. Same-day failed exits can be retried

* **Before:** `already_executed_today` checked for any row in the log for
  the current date and short-circuited the cycle. A failed SELL still
  wrote a row, so retries were blocked.
* **After:** `already_executed_today` is removed entirely. Idempotency
  comes from reconciliation:
  - Cycle reads live position.
  - If a BUY signal fires but the bot is already long at target qty, the
    reconciliation loop exits at attempt 1 with no order submitted.
  - If a SELL signal fires but the bot is already flat,
    `execute_sell` returns `"NO_ACTION"` without an error.
  - If a previous cycle left orphan shares (partial fill), the next
    cycle's signal logic sees a long position with a (now older)
    entry date - and the reconciliation loop sweeps it to flat.
  This is exercised by the mock test
  `mock: re-running recovers orphan shares`.

### C11. tz-aware, DST-safe scheduling

* **Before:** `datetime.datetime.now()` (naive local time) for scheduling
  and timestamps, while market checks used ET. Negative or huge sleeps
  possible at DST boundaries.
* **After:**
  - `EASTERN = pytz.timezone("US/Eastern")` is the single time zone for
    all market logic and log timestamps.
  - `next_target_datetime` walks the NYSE calendar forward up to 10 days
    to find the next session whose target time hasn't passed yet, and
    returns a tz-aware datetime.
  - `run_forever` clamps sleep to `[60s, 24h]` so DST jumps or clock skew
    can never produce negative or multi-day sleeps.
  - Log timestamps are produced via `dt.datetime.now(EASTERN).strftime(...)`.

## Section D - Indicators / data

### D12. FGI fetch + today's row insertion is consistent and idempotent

* **Before:** When `fg.get()` failed all retries, `fetch_and_update_fgi`
  returned the prior value without appending today's row, so indicators
  were computed off a series missing today. Stale-but-present API values
  were silently appended as "today".
* **After:** `fetch_and_update_fgi` always returns
  `(df, today_value, column, status)`. `status` is one of:
  `"fresh"`, `"stale_api"`, `"fallback_last_known"`, `"fatal_no_data"`.
  In every non-fatal path it inserts (or overwrites if same date) exactly
  one row for `today_et`, so `calculate_indicators` runs against a series
  that always includes today. The status is emitted as a structured event
  (`fgi.updated`) so future debugging is unambiguous.

### D13. Volatility from Alpaca, never blocks an exit

* **Before:** yfinance, with the bot aborting the entire cycle if vol
  couldn't be computed - including a required exit.
* **After:** Alpaca bars (same source as price). If the result is `None`,
  the bot substitutes a conservative midpoint between
  `volatility_buy_limit` and `volatility_sell_limit`, which neither
  forces a buy nor blocks an exit, and logs a `volatility.fallback`
  event. Thresholds and the 20-day annualized std computation are
  unchanged.

## Section E - Robustness / hygiene

### E14. APIError code via structured access

* **Before:** `if "40310000" in str(e) or "insufficient buying power" in err.lower()`.
* **After:** `_api_error_code(err)` returns `int(err.code)` via the
  structured `APIError.code` property. The two known error codes are
  named constants: `_ERR_INSUFFICIENT_BP = 40310000`,
  `_ERR_QTY_NON_POSITIVE = 40010001`.

### E15. Logged fill price is the real average

* **Before:** Market-fallback path logged `current_price` (the stale
  quote), not the actual fill.
* **After:** Both `execute_buy` and `execute_sell` track
  `proceeds / total_sold` (and `total_cost / total_filled` for buys)
  across every attempt; the CSV row's `Price` column is the true VWAP
  across all partial fills. Per-attempt `avg_price` is also emitted in
  the structured event log.

### E16. Structured order-lifecycle logging (CSV schema unchanged)

* `_evt(name, **fields)` emits one line of JSON per order event:
  submissions, cancels, fills, partial fills, fallbacks, stale state,
  and FGI/volatility/price fallbacks. Use these to debug future
  desyncs. Trading-log CSV columns (`Timestamp, Action, Symbol,
  Quantity, Price, FGI_Value, FGI_Momentum, FGI_Velocity, Volatility,
  Portfolio_Value, Buying_Power, Signal_Reason, Days_Held`) are
  unchanged - only the values are now accurate.

## State files

  state_BOD.json
  state_EOD.json

Format:

    {
      "entry_date": "YYYY-MM-DD",   # US/Eastern date of the BUY fill
      "qty": <int>                  # qty at entry (for reference; live qty is authoritative)
    }

* Created when the bot transitions flat -> long.
* Deleted when the bot confirms position is flat.
* Reconstructed via `_reconstruct_entry_date` if the bot finds a live
  position on startup with no state file present.
* `qty` is informational - all live decisions use the broker's reported
  position, never this value.

## Configuration confirmation

* Execution time, US/Eastern: BOD `09:35`, EOD `15:50` - both inside the
  regular session.
* `extended_hours=False` on every order submission.
* `leverage=1.0` in both `BotConfig` instances (no margin).
* `cash_headroom=0.005` leaves a 0.5% cash buffer to avoid 40310000 from
  rounding errors.

## Tests

`tests/test_regressions.py` has two layers:

* **Mock layer** (always runs). A `FakeTradingClient` simulates the
  exact failure conditions:
  1. `mock: partial fill is swept to flat` - the first close_position
     fills 64 of 69; the test asserts the loop sweeps the remaining 5
     and ends flat. Reproduces 2026-05-04 / 2026-05-08.
  2. `mock: oversell never submitted` - asserts no SELL order ever
     submitted with qty exceeding the live position. Reproduces the
     pattern behind 2026-05-05's 40310000.
  3. `mock: flat -> NO_ACTION (no 40010001)` - with `position_qty=0`,
     `execute_sell` returns `NO_ACTION` and `run_cycle` does not
     submit any order. Reproduces 2026-05-11..14.

  Plus six more (orphan-recovery, sizing, short detection, trading-day
  counting, structured error code, stale-state cleanup).

* **Live layer** (against Alpaca paper API). The three regression
  conditions are re-checked against the real broker. Tests auto-skip
  when:
    - `ALPACA_BOD_API_KEY` / `ALPACA_BOD_API_SECRET` are missing or
      the credentials return 401, **or**
    - the regular session is closed (paper market orders only fill
      during regular session).

  The current `.env` checked into the repo holds
  `APCA_API_KEY_ID=PKVVMQPZU4DBWK9DH51G` which the paper API now
  returns 401 for. The test runner maps `APCA_API_KEY_ID` ->
  `ALPACA_BOD_API_KEY` (and `_EOD`) as a fallback for single-account
  setups; to actually run the live layer you'll need to set valid
  `ALPACA_BOD_API_KEY/SECRET` + `ALPACA_EOD_API_KEY/SECRET` in the
  environment.

To run:

    .venv/bin/python tests/test_regressions.py           # mocks + live (skips if no creds)
    .venv/bin/python tests/test_regressions.py --mocks   # mocks only
    .venv/bin/python tests/test_regressions.py --live    # live only

Latest result: `PASS=9 FAIL=0 SKIP=3` (mocks all green; live skipped
due to 401 on the .env creds).
