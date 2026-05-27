"""
Fear & Greed Range Analysis: Velocity and Momentum for S&P 500

For each Fear & Greed index range (binned in 10-point intervals),
computes:
  - FG momentum    : fg - rolling(3).mean(fg)           [same as backtest]
  - FG velocity    : rolling(3).mean(fg.diff())          [same as backtest]
  - SPY velocity   : daily close-to-close return (%)
  - SPY momentum   : 5-day cumulative close return (%)
  - SPY fwd 1d     : next-day SPY return (forward-looking)
  - SPY fwd 5d     : next-5-day SPY cumulative return

Then ranks each range by each metric and prints a consolidated table.
"""

import pandas as pd
import numpy as np
import yfinance as yf

LOOKBACK = 3          # matches backtest lookback_days
FG_FILE  = "datasets/fear_greed_combined_2011_2025.csv"
START    = "2011-01-03"
END      = "2026-04-10"   # cover the forward-test range too

# ---------------------------------------------------------------------------
# 1. Load Fear & Greed data
# ---------------------------------------------------------------------------
fg = pd.read_csv(FG_FILE)
fg["Date"] = pd.to_datetime(fg["Date"])

# Normalise column name – the combined file uses 'fear_greed'
if "fear_greed" in fg.columns:
    fg = fg.rename(columns={"fear_greed": "fg"})
elif "Fear Greed" in fg.columns:
    fg = fg.rename(columns={"Fear Greed": "fg"})

fg = fg[["Date", "fg"]].drop_duplicates("Date").sort_values("Date").reset_index(drop=True)

# ---------------------------------------------------------------------------
# 2. Download SPY price data
# ---------------------------------------------------------------------------
print(f"Downloading SPY data {START} -> {END} ...")
spy_raw = yf.download("SPY", start=START, end=END, auto_adjust=True, progress=False)
if spy_raw.columns.nlevels > 1:
    spy_raw.columns = spy_raw.columns.droplevel(1)
spy_raw = spy_raw.reset_index()[["Date", "Close"]]
spy_raw["Date"] = pd.to_datetime(spy_raw["Date"])

# ---------------------------------------------------------------------------
# 3. Merge and forward-fill FG over weekends / holidays
# ---------------------------------------------------------------------------
data = pd.merge(spy_raw, fg, on="Date", how="left")
data["fg"] = data["fg"].ffill()
data = data.dropna(subset=["fg"]).reset_index(drop=True)

# ---------------------------------------------------------------------------
# 4. Compute indicators
# ---------------------------------------------------------------------------

# FG indicators (identical to backtest strategy())
data["fg_momentum"] = data["fg"] - data["fg"].rolling(LOOKBACK, min_periods=1).mean()
data["fg_change"]   = data["fg"].diff().fillna(0)
data["fg_velocity"] = data["fg_change"].rolling(LOOKBACK, min_periods=1).mean()

# SPY indicators
data["spy_daily_return"]  = data["Close"].pct_change().fillna(0) * 100          # %
data["spy_momentum_5d"]   = data["Close"].pct_change(5).shift(0).fillna(0) * 100  # 5-day trailing %

# Forward-looking SPY returns (what happens AFTER today's FG reading)
data["spy_fwd_1d"] = data["Close"].pct_change(1).shift(-1) * 100   # next session %
data["spy_fwd_5d"] = data["Close"].pct_change(5).shift(-5) * 100   # next-5-session %

# ---------------------------------------------------------------------------
# 5. Bin FG index into 10-point ranges
# ---------------------------------------------------------------------------
bins   = list(range(0, 101, 10))
labels = [f"{lo}-{lo+10}" for lo in range(0, 100, 10)]

data["fg_range"] = pd.cut(data["fg"], bins=bins, labels=labels,
                           include_lowest=True, right=True)

# Also add the classic CNN five-zone labels for reference
def zone(v):
    if v <= 25:  return "Extreme Fear (0-25)"
    if v <= 45:  return "Fear (26-45)"
    if v <= 55:  return "Neutral (46-55)"
    if v <= 75:  return "Greed (56-75)"
    return        "Extreme Greed (76-100)"

data["fg_zone"] = data["fg"].apply(zone)

# ---------------------------------------------------------------------------
# 6. Aggregate by range
# ---------------------------------------------------------------------------
metrics = ["fg_momentum", "fg_velocity",
           "spy_daily_return", "spy_momentum_5d",
           "spy_fwd_1d", "spy_fwd_5d"]

by_range = (
    data.groupby("fg_range", observed=False)[metrics]
    .agg(["mean", "median", "count"])
    .round(4)
)

# Flatten multi-level columns
by_range.columns = ["_".join(c) for c in by_range.columns]
by_range = by_range.reset_index()

# Drop rows with no data
by_range = by_range[by_range["fg_momentum_count"] > 0].copy()

# ---------------------------------------------------------------------------
# 7. Aggregate by CNN five-zone
# ---------------------------------------------------------------------------
by_zone = (
    data.groupby("fg_zone", observed=False)[metrics]
    .agg(["mean", "median", "count"])
    .round(4)
)
by_zone.columns = ["_".join(c) for c in by_zone.columns]
by_zone = by_zone.reset_index()
by_zone = by_zone[by_zone["fg_momentum_count"] > 0].copy()

# ---------------------------------------------------------------------------
# 8. Print results
# ---------------------------------------------------------------------------
DIVIDER = "=" * 110

print(f"\n{DIVIDER}")
print("FEAR & GREED RANGE ANALYSIS  --  Averages by 10-Point FG Bin")
print(DIVIDER)

header = (
    f"{'FG Range':<12} {'N':>5} "
    f"{'FG Mom':>9} {'FG Vel':>9} "
    f"{'SPY Ret%':>9} {'SPY Mom5d%':>11} "
    f"{'Fwd1d%':>8} {'Fwd5d%':>8}"
)
print(header)
print("-" * 110)

for _, row in by_range.iterrows():
    print(
        f"{str(row['fg_range']):<12} "
        f"{int(row['fg_momentum_count']):>5} "
        f"{row['fg_momentum_mean']:>+9.3f} "
        f"{row['fg_velocity_mean']:>+9.3f} "
        f"{row['spy_daily_return_mean']:>+9.3f} "
        f"{row['spy_momentum_5d_mean']:>+11.3f} "
        f"{row['spy_fwd_1d_mean']:>+8.3f} "
        f"{row['spy_fwd_5d_mean']:>+8.3f}"
    )

print(f"\n{DIVIDER}")
print("FEAR & GREED ZONE ANALYSIS  --  Averages by CNN Five-Zone Labels")
print(DIVIDER)

header2 = (
    f"{'FG Zone':<28} {'N':>5} "
    f"{'FG Mom':>9} {'FG Vel':>9} "
    f"{'SPY Ret%':>9} {'SPY Mom5d%':>11} "
    f"{'Fwd1d%':>8} {'Fwd5d%':>8}"
)
print(header2)
print("-" * 110)

for _, row in by_zone.sort_values("fg_momentum_mean", ascending=False).iterrows():
    print(
        f"{str(row['fg_zone']):<28} "
        f"{int(row['fg_momentum_count']):>5} "
        f"{row['fg_momentum_mean']:>+9.3f} "
        f"{row['fg_velocity_mean']:>+9.3f} "
        f"{row['spy_daily_return_mean']:>+9.3f} "
        f"{row['spy_momentum_5d_mean']:>+11.3f} "
        f"{row['spy_fwd_1d_mean']:>+8.3f} "
        f"{row['spy_fwd_5d_mean']:>+8.3f}"
    )

# ---------------------------------------------------------------------------
# 9. Identify top-ranked ranges per metric
# ---------------------------------------------------------------------------
rank_metrics = {
    "FG Momentum (mean)"          : "fg_momentum_mean",
    "FG Velocity (mean)"          : "fg_velocity_mean",
    "SPY Daily Return % (mean)"   : "spy_daily_return_mean",
    "SPY 5d Momentum % (mean)"    : "spy_momentum_5d_mean",
    "SPY Fwd 1d Return % (mean)"  : "spy_fwd_1d_mean",
    "SPY Fwd 5d Return % (mean)"  : "spy_fwd_5d_mean",
}

print(f"\n{DIVIDER}")
print("TOP-RANKED FG RANGES PER METRIC  (10-point bins)")
print(DIVIDER)

for label, col in rank_metrics.items():
    top_row = by_range.loc[by_range[col].idxmax()]
    print(f"  {label:<38}: FG {top_row['fg_range']}  =>  {top_row[col]:+.4f}")

print(f"\n{DIVIDER}")
print("TOP-RANKED FG ZONES PER METRIC  (CNN five-zone)")
print(DIVIDER)

for label, col in rank_metrics.items():
    top_row = by_zone.loc[by_zone[col].idxmax()]
    print(f"  {label:<38}: {top_row['fg_zone']}  =>  {top_row[col]:+.4f}")

# ---------------------------------------------------------------------------
# 10. Combined score: rank by average of normalised fwd1d + fwd5d + fg_momentum + fg_velocity
# ---------------------------------------------------------------------------
print(f"\n{DIVIDER}")
print("COMPOSITE SCORE  (norm-avg of FG Momentum + FG Velocity + Fwd1d + Fwd5d)")
print(DIVIDER)

score_cols = ["fg_momentum_mean", "fg_velocity_mean", "spy_fwd_1d_mean", "spy_fwd_5d_mean"]

# Min-max normalise each metric across the non-empty bins
norm = by_range[score_cols].copy()
for c in score_cols:
    mn, mx = norm[c].min(), norm[c].max()
    norm[c] = (norm[c] - mn) / (mx - mn) if mx > mn else 0.0

by_range["composite_score"] = norm[score_cols].mean(axis=1)
ranked = by_range.sort_values("composite_score", ascending=False).reset_index(drop=True)

print(f"{'Rank':<6} {'FG Range':<12} {'FG Mom':>9} {'FG Vel':>9} {'Fwd1d%':>8} {'Fwd5d%':>8} {'Score':>8}")
print("-" * 70)
for rank, row in ranked.iterrows():
    print(
        f"  #{rank+1:<4} {str(row['fg_range']):<12} "
        f"{row['fg_momentum_mean']:>+9.3f} "
        f"{row['fg_velocity_mean']:>+9.3f} "
        f"{row['spy_fwd_1d_mean']:>+8.3f} "
        f"{row['spy_fwd_5d_mean']:>+8.3f} "
        f"{row['composite_score']:>8.4f}"
    )

print(f"\nBest composite range: FG {ranked.iloc[0]['fg_range']}")
print(DIVIDER)
