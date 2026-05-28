"""
Daily Fear & Greed Index email alert with trading signal analysis.

Run via cron on the Pi at 1:05 PM PST:
    5 13 * * 1-5 cd /path/to/feargreed && /usr/bin/python3 fgi_email_alert.py

Gmail setup (one-time):
  1. Enable 2-Step Verification on your Google account
  2. Go to myaccount.google.com/apppasswords
  3. Generate an App Password for "Mail"
  4. Set env vars:
       export FGI_SENDER_EMAIL="your_gmail@gmail.com"
       export FGI_SENDER_PASSWORD="xxxx xxxx xxxx xxxx"  # 16-char app password
"""

import datetime
import os
import smtplib
import time
from email.message import EmailMessage
from pathlib import Path

import fear_and_greed as fg_lib
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import yfinance as yf
from scipy import stats as scipy_stats

# ── Config ───────────────────────────────────────────────────────────────────
RECIPIENT        = "Vikramatmuri01@gmail.com"
SENDER_EMAIL     = os.environ["FGI_SENDER_EMAIL"]
SENDER_PASSWORD  = os.environ["FGI_SENDER_PASSWORD"]

MAX_RETRIES  = 3
RETRY_DELAY  = 30
LOOKBACK     = 3       # rolling window for momentum/velocity (matches backtest)
SPY_START    = "2011-01-03"
MIN_SAMPLES  = 25      # minimum obs required to report a stat
FWD_DAYS     = [1, 5, 10, 20]
MAX_KELLY    = 0.50    # cap half-Kelly at 50%
MIN_KELLY    = 0.0     # floor -- no negative positions

BASE_DIR    = Path(__file__).parent
FG_COMBINED = BASE_DIR / "datasets" / "fear_greed_combined_2011_2025.csv"
FG_FORWARD  = BASE_DIR / "datasets" / "fear_greed_forward_test_afternoon.csv"


# ── FG Zone definitions ──────────────────────────────────────────────────────
ZONES = [
    ("Extreme Fear", 0,  25),
    ("Fear",        26,  45),
    ("Neutral",     46,  55),
    ("Greed",       56,  75),
    ("Extreme Greed", 76, 100),
]

def get_zone(value: float) -> tuple[str, int, int]:
    for name, lo, hi in ZONES:
        if lo <= value <= hi:
            return name, lo, hi
    return "Extreme Greed", 76, 100


# ── Data loading ─────────────────────────────────────────────────────────────
def load_fg_history() -> pd.DataFrame:
    """
    Merge combined (2011-2025) and forward-test CSVs into one sorted DataFrame.
    Normalises column names, deduplicates on Date, returns Date + fg columns.
    """
    frames = []
    for path in [FG_COMBINED, FG_FORWARD]:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["Date"] = pd.to_datetime(df.iloc[:, 0])
        # second column is the value regardless of name
        df = df.rename(columns={df.columns[1]: "fg"})
        frames.append(df[["Date", "fg"]])

    combined = pd.concat(frames, ignore_index=True)
    combined = (combined
                .drop_duplicates("Date", keep="last")
                .sort_values("Date")
                .reset_index(drop=True))
    combined["fg"] = pd.to_numeric(combined["fg"], errors="coerce")
    return combined.dropna(subset=["fg"])


def build_stats_dataset(fg_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load or download SPY, merge with FG history, compute momentum/velocity,
    market-structure features, and forward returns for every trading day.
    Returns (analysis_df, raw_spy_df). raw_spy_df retains all SPY rows so
    the caller can derive live market conditions for today.
    SPY data is cached as a dated CSV; files older than 3 days are removed.
    """
    today = datetime.date.today()
    cache_path = BASE_DIR / "datasets" / f"spy_{today.isoformat()}.csv"

    # Remove stale cache files (older than 3 days)
    cutoff = today - datetime.timedelta(days=3)
    for old in (BASE_DIR / "datasets").glob("spy_*.csv"):
        try:
            file_date = datetime.date.fromisoformat(old.stem[4:])
            if file_date < cutoff:
                old.unlink()
        except ValueError:
            pass

    if cache_path.exists():
        print("Loading SPY from cache ...")
        spy = pd.read_csv(cache_path, parse_dates=["Date"])
    else:
        end_date = (today + datetime.timedelta(days=1)).isoformat()
        spy = yf.download("SPY", start=SPY_START, end=end_date,
                          auto_adjust=True, progress=False)
        if spy.columns.nlevels > 1:
            spy.columns = spy.columns.droplevel(1)
        spy = spy.reset_index()[["Date", "Close"]]
        spy["Date"] = pd.to_datetime(spy["Date"])
        spy.to_csv(cache_path, index=False)
        print("SPY downloaded and cached.")

    spy_raw = spy.copy()

    data = spy.merge(fg_df, on="Date", how="left")
    data["fg"] = data["fg"].ffill()
    data = data.dropna(subset=["fg"]).reset_index(drop=True)

    # Signals -- identical to backtest strategy()
    data["fg_momentum"] = (data["fg"]
                           - data["fg"].rolling(LOOKBACK, min_periods=1).mean())
    data["fg_velocity"] = (data["fg"].diff().fillna(0)
                           .rolling(LOOKBACK, min_periods=1).mean())

    # Market-structure features -------------------------------------------------
    daily_ret = data["Close"].pct_change()

    # Consecutive red (non-positive) SPY days BEFORE each row
    # Build iteratively, then shift so each row reflects the streak ending yesterday.
    consec: list[int] = []
    count = 0
    for r in daily_ret:
        if pd.isna(r) or r > 0:
            count = 0
        else:
            count += 1
        consec.append(count)
    data["consec_red_before"] = (pd.Series(consec, index=data.index)
                                 .shift(1).fillna(0).astype(int))

    # 5-day SPY return ending yesterday (negative = recent pullback)
    data["spy_5d_ret_prior"] = data["Close"].pct_change(5).shift(1) * 100

    # SPY above 200-day MA (structural bull/bear regime)
    data["spy_above_200ma"] = (data["Close"]
                               > data["Close"].rolling(200, min_periods=100).mean())

    # Forward returns
    for d in FWD_DAYS:
        data[f"fwd_{d}d"] = data["Close"].pct_change(d).shift(-d) * 100

    # Drop rows where the longest horizon is still NaN
    data = data.dropna(subset=[f"fwd_{FWD_DAYS[-1]}d"]).reset_index(drop=True)
    return data, spy_raw


def compute_current_spy_conditions(spy_raw: pd.DataFrame) -> dict:
    """
    Derive today's live market-structure conditions from the raw SPY price series.
    Uses the last confirmed close (index -2 when today's close is present, or -1
    when it is not yet available) to avoid look-ahead.
    Returns: spy_5d_ret, consec_red_days, spy_above_200ma.
    """
    closes = spy_raw["Close"].dropna().values
    if len(closes) < 10:
        return {"spy_5d_ret": float("nan"), "consec_red_days": 0, "spy_above_200ma": True}

    # Use the last 2 closes to decide whether today's close is settled.
    # yfinance returns today if the session is over; treat it as available.
    # "yesterday" = closes[-2], "today" = closes[-1].
    daily_rets = np.diff(closes) / closes[:-1]   # len = n-1

    # Consecutive red days ending yesterday (exclude today's intraday move)
    consec_red = 0
    for r in reversed(daily_rets[:-1]):   # daily_rets[:-1] ends at yesterday's return
        if r <= 0:
            consec_red += 1
        else:
            break

    # 5-day return ending yesterday
    spy_5d_ret = float((closes[-2] / closes[-7] - 1) * 100) if len(closes) >= 7 else float("nan")

    # SPY above 200-day MA as of yesterday
    if len(closes) >= 201:
        ma200 = float(np.mean(closes[-201:-1]))
        spy_above_200ma = bool(closes[-2] > ma200)
    else:
        spy_above_200ma = True

    return {
        "spy_5d_ret":      spy_5d_ret,
        "consec_red_days": consec_red,
        "spy_above_200ma": spy_above_200ma,
    }


# ── Statistics helpers ────────────────────────────────────────────────────────
def ci95(series: pd.Series) -> float:
    if len(series) < 2:
        return float("nan")
    return float(scipy_stats.sem(series) * scipy_stats.t.ppf(0.975, len(series) - 1))


def _tier_stats(subset: pd.DataFrame, label: str) -> dict | None:
    """Compute win rates, returns, Kelly for one filtered subset."""
    if len(subset) < MIN_SAMPLES:
        return None

    result = {"label": label, "n": len(subset)}
    for d in FWD_DAYS:
        col  = f"fwd_{d}d"
        wins = subset[col][subset[col] > 0]
        losses = subset[col][subset[col] <= 0]
        wr   = (subset[col] > 0).mean()
        mu   = subset[col].mean()
        med  = subset[col].median()
        ci   = ci95(subset[col])
        sh   = (mu / subset[col].std() * np.sqrt(252 / d)
                if subset[col].std() > 0 else 0.0)

        avg_win  = wins.mean()  if len(wins)   > 0 else 0.0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 0.0

        # Kelly: f* = p - q/b  where b = avg_win / avg_loss
        b = avg_win / avg_loss if avg_loss > 0 else 0.0
        kelly_raw = (wr - (1 - wr) / b) if b > 0 else 0.0
        half_kelly = min(MAX_KELLY, max(MIN_KELLY, kelly_raw / 2))

        result[f"wr{d}d"]        = wr
        result[f"mu{d}d"]        = mu
        result[f"med{d}d"]       = med
        result[f"ci{d}d"]        = ci
        result[f"sh{d}d"]        = sh
        result[f"avg_win{d}d"]   = avg_win
        result[f"avg_loss{d}d"]  = avg_loss
        result[f"kelly{d}d"]     = half_kelly

    # Year breakdown for 5d
    subset2 = subset.copy()
    subset2["year"] = pd.to_datetime(subset2["Date"]).dt.year
    yby = {}
    for yr, grp in subset2.groupby("year"):
        if len(grp) >= 5:
            yby[int(yr)] = {
                "n":  len(grp),
                "wr": (grp["fwd_5d"] > 0).mean() * 100,
                "mu": grp["fwd_5d"].mean(),
            }
    result["year_by_year"] = yby
    return result


def analyse_conditions(data: pd.DataFrame,
                       current_fg: float,
                       momentum: float,
                       velocity: float,
                       spy_conds: dict | None = None) -> dict:
    """
    Return historical stats for today's conditions across two groups of tiers:

    FG signal tiers (existing):
      tier1 -- same FG zone only
      tier2 -- zone + momentum >= today's level
      tier3 -- zone + momentum >= today's level + velocity >= today's level
      exact_bin -- tightest 10-pt FG bin + both signal filters

    Market-structure tiers (new, conditioned on SPY data):
      trend   -- zone + same 200-day MA regime (bull vs bear)
      pullback -- zone + SPY 5d prior return < 0 (bought into weakness)
      redstreak -- zone + consec red days >= today's streak
      composite -- zone + all three market conditions match
    """
    _, zlo, zhi = get_zone(current_fg)
    zone_name, _, _ = get_zone(current_fg)
    zone_data = data[(data["fg"] >= zlo) & (data["fg"] <= zhi)]

    mom_data  = zone_data[zone_data["fg_momentum"] >= momentum]
    full_data = mom_data[mom_data["fg_velocity"]   >= velocity]

    t1 = _tier_stats(zone_data, f"{zone_name} zone only")
    t2 = _tier_stats(mom_data,  f"+ momentum >= {momentum:+.2f}")
    t3 = _tier_stats(full_data, f"+ velocity >= {velocity:+.2f}")

    fg_lo10 = int(current_fg // 10) * 10
    fg_hi10 = fg_lo10 + 10
    bin_data = data[(data["fg"] >= fg_lo10) & (data["fg"] <= fg_hi10)
                    & (data["fg_momentum"] >= momentum)
                    & (data["fg_velocity"]  >= velocity)]
    t_bin = _tier_stats(bin_data,
                        f"FG {fg_lo10}-{fg_hi10}, mom>={momentum:+.2f}, vel>={velocity:+.2f}")

    # Market-structure tiers
    t_trend = t_pullback = t_redstreak = t_composite = None
    if spy_conds is not None and "spy_above_200ma" in data.columns:
        above_200   = spy_conds["spy_above_200ma"]
        spy_5d      = spy_conds["spy_5d_ret"]
        consec_red  = spy_conds["consec_red_days"]

        regime_label = "above" if above_200 else "below"
        trend_data = zone_data[zone_data["spy_above_200ma"] == above_200]
        t_trend = _tier_stats(trend_data,
                              f"{zone_name} + SPY {regime_label} 200dMA")

        if not pd.isna(spy_5d) and spy_5d < 0:
            pb_data = zone_data[zone_data["spy_5d_ret_prior"] < 0]
            t_pullback = _tier_stats(pb_data,
                                     f"{zone_name} + SPY 5d ret < 0 (pullback)")
        else:
            pb_data = zone_data[zone_data["spy_5d_ret_prior"] >= 0]
            t_pullback = _tier_stats(pb_data,
                                     f"{zone_name} + SPY 5d ret >= 0 (no pullback)")

        if consec_red >= 1:
            rs_data = zone_data[zone_data["consec_red_before"] >= consec_red]
            t_redstreak = _tier_stats(rs_data,
                                      f"{zone_name} + >= {consec_red} consec red days")
        else:
            # no red streak today -- show what happens when entering on a green day
            rs_data = zone_data[zone_data["consec_red_before"] == 0]
            t_redstreak = _tier_stats(rs_data,
                                      f"{zone_name} + 0 consec red days (green entry)")

        # Composite: all three market conditions aligned
        comp = trend_data.copy()
        if not pd.isna(spy_5d):
            comp = comp[comp["spy_5d_ret_prior"] < 0] if spy_5d < 0 else comp[comp["spy_5d_ret_prior"] >= 0]
        if consec_red >= 1:
            comp = comp[comp["consec_red_before"] >= consec_red]
        t_composite = _tier_stats(comp,
                                  f"{zone_name} + regime + {'pullback' if spy_5d < 0 else 'no pullback'} + {consec_red}d red")

    return {
        "tier1": t1, "tier2": t2, "tier3": t3, "exact_bin": t_bin,
        "trend": t_trend, "pullback": t_pullback,
        "redstreak": t_redstreak, "composite": t_composite,
    }


# ── Signal label ─────────────────────────────────────────────────────────────
def signal_label(momentum: float, velocity: float) -> str:
    if momentum >= 2.0 and velocity >= 1.0:
        return "STRONG BUY"
    if momentum >= 0.2 and velocity >= 0.15:
        return "BUY"
    return "NO SIGNAL"


def kelly_label(half_kelly: float) -> str:
    if half_kelly <= 0.0:
        return "No position"
    if half_kelly < 0.08:
        return "Small  (< 8%)"
    if half_kelly < 0.15:
        return "Modest (8-15%)"
    if half_kelly < 0.25:
        return "Medium (15-25%)"
    if half_kelly < 0.35:
        return "Large  (25-35%)"
    return "Max    (35-50%)"


# ── Email builder ─────────────────────────────────────────────────────────────
def build_email(fg_value: float,
                description: str,
                last_update: datetime.datetime,
                momentum: float,
                velocity: float,
                analysis: dict,
                analysis_ok: bool = True,
                spy_conds: dict | None = None) -> EmailMessage:

    zone_name, zlo, zhi = get_zone(fg_value)
    today  = datetime.date.today().strftime("%B %d, %Y")
    signal = signal_label(momentum, velocity)

    # Most specific tier with enough data
    primary = analysis["exact_bin"] or analysis["tier3"] or analysis["tier2"] or analysis["tier1"]

    # Subject: include 10d win rate if available
    wr10_str = ""
    if primary:
        wr10_str = f" | 10d win rate {primary['wr10d']*100:.1f}%"
    subject = f"F&G Signal: {signal} | {fg_value} {zone_name}{wr10_str}"

    def fmt_tier(t: dict | None) -> str:
        if t is None:
            return f"    (fewer than {MIN_SAMPLES} observations -- not reported)\n"
        lines = [f"    Observations: {t['n']}  ({t['label']})"]
        header = f"    {'Horizon':<9} {'Win Rate':>9} {'Avg Ret':>9} {'Median':>9} {'95% CI':>10} {'Sharpe':>8}"
        lines.append(header)
        lines.append("    " + "-" * 58)
        for d in FWD_DAYS:
            lines.append(
                f"    {d:>2}d       "
                f"  {t[f'wr{d}d']*100:>7.1f}%"
                f"  {t[f'mu{d}d']:>+8.3f}%"
                f"  {t[f'med{d}d']:>+8.3f}%"
                f"  ±{t[f'ci{d}d']:>7.3f}%"
                f"  {t[f'sh{d}d']:>+7.2f}"
            )
        return "\n".join(lines) + "\n"

    def fmt_position(t: dict | None, horizon: int = 5) -> str:
        if t is None:
            return "    Insufficient data to size position.\n"
        wr      = t[f"wr{horizon}d"]
        avg_win = t[f"avg_win{horizon}d"]
        avg_loss= t[f"avg_loss{horizon}d"]
        hk      = t[f"kelly{horizon}d"]
        b       = avg_win / avg_loss if avg_loss > 0 else 0.0
        lines = [
            f"    Basis: {t['n']} historical observations ({horizon}-day horizon)",
            f"    Win rate:        {wr*100:.1f}%",
            f"    Avg gain (wins): {avg_win:+.3f}%",
            f"    Avg loss (loss): {avg_loss:.3f}%",
            f"    Win/Loss ratio:  {b:.2f}x",
            f"    Kelly fraction:  {hk*2*100:.1f}%  (full) --> {hk*100:.1f}%  (half-Kelly)",
            f"    Suggested size:  {kelly_label(hk)}  ({hk*100:.0f}% of risk capital)",
        ]
        return "\n".join(lines) + "\n"

    def fmt_year_by_year(t: dict | None) -> str:
        if t is None or not t.get("year_by_year"):
            return "    (no year breakdown available)\n"
        lines = [f"    {'Year':<6} {'N':>5} {'WR-5d':>8} {'Mean-5d':>10}"]
        lines.append("    " + "-" * 35)
        for yr, s in sorted(t["year_by_year"].items()):
            if s["n"] < 15:
                continue
            lines.append(f"    {yr:<6} {s['n']:>5} {s['wr']:>7.1f}%  {s['mu']:>+9.3f}%")
        return "\n".join(lines) + "\n"

    backtest_signal = (momentum >= 0.2 and velocity >= 0.15)

    # Format SPY condition summary line for CURRENT CONDITIONS block
    if spy_conds is not None:
        regime_str = "Above" if spy_conds["spy_above_200ma"] else "Below"
        spy_5d_val = spy_conds["spy_5d_ret"]
        spy_5d_str = f"{spy_5d_val:+.2f}%" if not pd.isna(spy_5d_val) else "n/a"
        spy_line = (
            f"  SPY 5d prior ret: {spy_5d_str}  |  "
            f"Consec red days: {spy_conds['consec_red_days']}  |  "
            f"200dMA regime: {regime_str}"
        )
    else:
        spy_line = "  SPY conditions: unavailable"

    if not analysis_ok:
        stats_section = "  Signal analysis unavailable -- check logs.\n"
    else:
        # Market-context section: composite is most informative; fall back through tiers
        mkt_primary = (analysis["composite"] or analysis["redstreak"]
                       or analysis["pullback"] or analysis["trend"])

        mkt_section = f"""\
MARKET CONTEXT ANALYSIS  (SPY-conditioned, same FG zone)

  200dMA regime (bull vs bear market):
{fmt_tier(analysis["trend"])}
  Recent pullback (SPY 5d prior return):
{fmt_tier(analysis["pullback"])}
  Consecutive red-day streak:
{fmt_tier(analysis["redstreak"])}
  Composite (all market conditions aligned):
{fmt_tier(analysis["composite"])}"""

        stats_section = f"""\
HISTORICAL PERFORMANCE  (2011-2026, most specific FG+signal conditions)

{fmt_tier(primary)}
{'=' * 65}

{mkt_section}
{'=' * 65}

POSITION SIZING  (Half-Kelly, 5-day horizon)
  Using most specific FG+signal tier as basis.

{fmt_position(primary, horizon=5)}  NOTE: Kelly assumes the historical sample is representative of future
  outcomes. Use half-Kelly (already applied) as the conservative estimate.
  Size further by your overall risk budget.

{'=' * 65}

YEAR-BY-YEAR BREAKDOWN (5d, years with n>=15)

{fmt_year_by_year(primary)}"""

    body = f"""\
FEAR & GREED INDEX  --  DAILY SIGNAL REPORT
Date:  {today}
Data:  Last updated {last_update.strftime("%Y-%m-%d %H:%M UTC")}
{'=' * 65}

CURRENT CONDITIONS
  F&G Value:   {fg_value:.2f} / 100  ({zone_name})
  Zone range:  {zlo}-{zhi}
  Momentum:    {momentum:+.3f}  (current FG vs {LOOKBACK}-day rolling avg)
  Velocity:    {velocity:+.3f}  ({LOOKBACK}-day avg of daily FG changes)
{spy_line}

  Signal:             {signal}
  Backtest entry:     {"YES" if backtest_signal else "NO"}  (mom>=0.2, vel>=0.15)
  High-conviction:    {"YES" if signal == "STRONG BUY" else "NO"}  (mom>=2.0, vel>=1.0)

{'=' * 65}

{stats_section}{'=' * 65}

F&G SCALE REFERENCE
  0-25   Extreme Fear  |  26-45  Fear  |  46-55  Neutral
  56-75  Greed         |  76-100 Extreme Greed
"""

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"]    = SENDER_EMAIL
    msg["To"]      = RECIPIENT
    msg.set_content(body)
    return msg


# ── Core helpers (unchanged) ──────────────────────────────────────────────────
def is_trading_day(date: datetime.date) -> bool:
    nyse = mcal.get_calendar("NYSE")
    schedule = nyse.schedule(start_date=date, end_date=date)
    return not schedule.empty


def fetch_fgi() -> tuple[float, str, datetime.datetime]:
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            data = fg_lib.get()
            return round(data.value, 2), data.description, data.last_update
        except Exception as exc:
            last_exc = exc
            print(f"Attempt {attempt}/{MAX_RETRIES} failed: {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
    raise RuntimeError(f"All FGI fetch attempts failed: {last_exc}") from last_exc


def send_email(msg: EmailMessage) -> None:
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(SENDER_EMAIL, SENDER_PASSWORD)
        smtp.send_message(msg)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    today = datetime.date.today()

    if not is_trading_day(today):
        print(f"{today} is not a trading day. No email sent.")
        return

    print(f"Fetching Fear & Greed Index for {today} ...")
    fg_value, description, last_update = fetch_fgi()
    print(f"FGI: {fg_value} ({description}), last updated {last_update}")

    momentum = velocity = None
    analysis = {
        "tier1": None, "tier2": None, "tier3": None, "exact_bin": None,
        "trend": None, "pullback": None, "redstreak": None, "composite": None,
    }
    spy_conds: dict | None = None
    analysis_ok = False

    try:
        print("Loading FG history and downloading SPY for signal analysis ...")
        fg_history = load_fg_history()

        # Append today's live value so momentum/velocity use the freshest point
        today_row = pd.DataFrame([{"Date": pd.Timestamp(today), "fg": fg_value}])
        fg_history = (pd.concat([fg_history, today_row], ignore_index=True)
                      .drop_duplicates("Date", keep="last")
                      .sort_values("Date")
                      .reset_index(drop=True))

        # Compute today's momentum and velocity from the last LOOKBACK days
        recent = fg_history["fg"].iloc[-10:]
        momentum = float(recent.iloc[-1] - recent.rolling(LOOKBACK, min_periods=1).mean().iloc[-1])
        velocity = float(recent.diff().fillna(0).rolling(LOOKBACK, min_periods=1).mean().iloc[-1])
        print(f"Signals -- momentum: {momentum:+.3f}, velocity: {velocity:+.3f}")

        data, spy_raw = build_stats_dataset(fg_history)
        spy_conds = compute_current_spy_conditions(spy_raw)
        print(f"SPY conditions -- 5d ret: {spy_conds['spy_5d_ret']:+.2f}%,"
              f" consec red: {spy_conds['consec_red_days']},"
              f" above 200dMA: {spy_conds['spy_above_200ma']}")

        analysis = analyse_conditions(data, fg_value, momentum, velocity, spy_conds)
        analysis_ok = True
        print("Signal analysis complete.")

    except Exception as exc:
        print(f"Signal analysis failed: {exc}")

    if momentum is None or velocity is None:
        print("Cannot compute signals -- skipping email.")
        return

    msg = build_email(fg_value, description, last_update,
                      momentum, velocity, analysis, analysis_ok=analysis_ok,
                      spy_conds=spy_conds)
    send_email(msg)
    print(f"Email sent to {RECIPIENT}.")


if __name__ == "__main__":
    main()
