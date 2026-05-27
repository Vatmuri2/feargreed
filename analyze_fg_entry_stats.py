"""
Fear & Greed Entry Condition Statistical Analysis

Answers: at what F&G value, momentum, and velocity should you enter,
and what are the historical win rates and return distributions?

Approach
--------
1. Build a daily dataset: SPY prices + F&G index + derived signals.
2. Compute forward returns at 1, 5, 10, 20 trading days.
3. Section A  – FG level alone: 5-point bins, historical rates at each level.
4. Section B  – Momentum grid: for each FG band, sweep momentum thresholds.
5. Section C  – Velocity grid: for each FG band, sweep velocity thresholds.
6. Section D  – Joint grid (FG bin × momentum × velocity): find the optimal
               entry condition with min sample-size guard.
7. Section E  – Best single entry condition full profile.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from scipy import stats
import yfinance as yf

# ── Config ──────────────────────────────────────────────────────────────────
LOOKBACK     = 3
FG_FILE      = "datasets/fear_greed_combined_2011_2025.csv"
START        = "2011-01-03"
END          = "2026-04-10"
MIN_SAMPLES  = 30          # minimum observations required for a result to count
FWD_DAYS     = [1, 5, 10, 20]

# ── 1. Load & merge data ─────────────────────────────────────────────────────
fg = pd.read_csv(FG_FILE)
fg["Date"] = pd.to_datetime(fg["Date"])
col = "fear_greed" if "fear_greed" in fg.columns else "Fear Greed"
fg = fg.rename(columns={col: "fg"})[["Date", "fg"]]
fg = fg.drop_duplicates("Date").sort_values("Date").reset_index(drop=True)

print("Downloading SPY …")
spy = yf.download("SPY", start=START, end=END, auto_adjust=True, progress=False)
if spy.columns.nlevels > 1:
    spy.columns = spy.columns.droplevel(1)
spy = spy.reset_index()[["Date", "Close"]]
spy["Date"] = pd.to_datetime(spy["Date"])

data = spy.merge(fg, on="Date", how="left")
data["fg"] = data["fg"].ffill()
data = data.dropna(subset=["fg"]).reset_index(drop=True)

# ── 2. Signals (matching backtest definitions) ───────────────────────────────
data["fg_momentum"] = data["fg"] - data["fg"].rolling(LOOKBACK, min_periods=1).mean()
data["fg_velocity"] = data["fg"].diff().fillna(0).rolling(LOOKBACK, min_periods=1).mean()

# ── 3. Forward returns ───────────────────────────────────────────────────────
for d in FWD_DAYS:
    data[f"fwd_{d}d"] = data["Close"].pct_change(d).shift(-d) * 100
    data[f"win_{d}d"]  = (data[f"fwd_{d}d"] > 0).astype(float)

data = data.dropna(subset=[f"fwd_{FWD_DAYS[-1]}d"]).reset_index(drop=True)

N_TOTAL = len(data)

# ── Helpers ──────────────────────────────────────────────────────────────────
DIV  = "=" * 110
SDIV = "-" * 110

def ci95(series):
    """95 % confidence interval half-width on the mean."""
    if len(series) < 2:
        return float("nan")
    return stats.sem(series) * stats.t.ppf(0.975, len(series) - 1)

def summarise(subset, tag=""):
    if len(subset) < MIN_SAMPLES:
        return None
    out = {"n": len(subset)}
    for d in FWD_DAYS:
        col = f"fwd_{d}d"
        wcol = f"win_{d}d"
        out[f"wr{d}d"]  = subset[wcol].mean() * 100
        out[f"mu{d}d"]  = subset[col].mean()
        out[f"ci{d}d"]  = ci95(subset[col])
        out[f"med{d}d"] = subset[col].median()
        out[f"sh{d}d"]  = (subset[col].mean() / subset[col].std()
                           * np.sqrt(252 / d) if subset[col].std() > 0 else 0)
    return out

def print_row(label, s, width=22):
    if s is None:
        return
    row = f"  {label:<{width}} n={s['n']:>4}"
    for d in FWD_DAYS:
        row += (f"  |  {s[f'wr{d}d']:>5.1f}% {s[f'mu{d}d']:>+6.2f}% "
                f"[±{s[f'ci{d}d']:.2f}]")
    print(row)

def print_header(title, note=""):
    print(f"\n{DIV}")
    print(title)
    if note:
        print(note)
    print(DIV)
    hdr = f"  {'Label':<22} {'N':>5}"
    for d in FWD_DAYS:
        hdr += f"  |  fwd-{d}d WinRate   Mean    95%CI "
    print(hdr)
    print(SDIV)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION A – FG level alone (5-point bins)
# ═══════════════════════════════════════════════════════════════════════════
print_header(
    "SECTION A  —  F&G VALUE ANALYSIS  (5-point bins, 2011–2026)",
    "  Each row = all days where FG fell in that 5-point range."
)

bins5   = list(range(0, 101, 5))
labels5 = [f"{lo:02d}-{lo+5:02d}" for lo in range(0, 100, 5)]
data["fg_bin5"] = pd.cut(data["fg"], bins=bins5, labels=labels5,
                          include_lowest=True, right=True)

best_a = {}
for lbl in labels5:
    sub = data[data["fg_bin5"] == lbl]
    s   = summarise(sub)
    print_row(f"FG {lbl}", s)
    if s:
        best_a[lbl] = s

# Rank by fwd-5d mean return
ranked_a = sorted(best_a.items(), key=lambda x: x[1]["mu5d"], reverse=True)
print(f"\n  Top-5 FG bins by mean fwd-5d return:")
for lbl, s in ranked_a[:5]:
    print(f"    FG {lbl}  ->  WR={s['wr5d']:.1f}%  Mean={s['mu5d']:+.3f}%  n={s['n']}")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION B – Momentum threshold analysis within FG zones
# ═══════════════════════════════════════════════════════════════════════════
print_header(
    "SECTION B  —  MOMENTUM THRESHOLD ANALYSIS  (within each FG zone)",
    "  Rows = days where FG is in zone AND fg_momentum >= threshold shown."
)

zones = {
    "Ext-Fear  (0-25)" : (0,  25),
    "Fear      (26-45)": (26, 45),
    "Neutral   (46-55)": (46, 55),
    "Greed     (56-75)": (56, 75),
    "Ext-Greed (76-100)":(76, 100),
}

mom_thresholds = [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3]

best_b = {}  # (zone, mom_thr) -> summary

for zone_lbl, (zlo, zhi) in zones.items():
    zone_data = data[(data["fg"] >= zlo) & (data["fg"] <= zhi)]
    print(f"\n  --- {zone_lbl}  (n={len(zone_data)}) ---")
    for thr in mom_thresholds:
        sub = zone_data[zone_data["fg_momentum"] >= thr]
        s   = summarise(sub)
        label = f"mom>={thr:+.1f}"
        print_row(label, s)
        if s:
            best_b[(zone_lbl, thr)] = s

# Best momentum threshold per zone (by fwd-5d mean)
print(f"\n  Best momentum threshold per zone (fwd-5d mean):")
for zone_lbl in zones:
    candidates = {k: v for k, v in best_b.items() if k[0] == zone_lbl}
    if candidates:
        best_k = max(candidates, key=lambda k: candidates[k]["mu5d"])
        s = candidates[best_k]
        print(f"    {zone_lbl}  mom>={best_k[1]:+.1f}  ->  WR={s['wr5d']:.1f}%  "
              f"Mean={s['mu5d']:+.3f}%  n={s['n']}")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION C – Velocity threshold analysis within FG zones
# ═══════════════════════════════════════════════════════════════════════════
print_header(
    "SECTION C  —  VELOCITY THRESHOLD ANALYSIS  (within each FG zone)",
    "  Rows = days where FG is in zone AND fg_velocity >= threshold shown."
)

vel_thresholds = [-3, -2, -1, -0.5, 0, 0.15, 0.3, 0.5, 1, 2, 3]

best_c = {}

for zone_lbl, (zlo, zhi) in zones.items():
    zone_data = data[(data["fg"] >= zlo) & (data["fg"] <= zhi)]
    print(f"\n  --- {zone_lbl}  (n={len(zone_data)}) ---")
    for thr in vel_thresholds:
        sub = zone_data[zone_data["fg_velocity"] >= thr]
        s   = summarise(sub)
        label = f"vel>={thr:+.2f}"
        print_row(label, s)
        if s:
            best_c[(zone_lbl, thr)] = s

print(f"\n  Best velocity threshold per zone (fwd-5d mean):")
for zone_lbl in zones:
    candidates = {k: v for k, v in best_c.items() if k[0] == zone_lbl}
    if candidates:
        best_k = max(candidates, key=lambda k: candidates[k]["mu5d"])
        s = candidates[best_k]
        print(f"    {zone_lbl}  vel>={best_k[1]:+.2f}  ->  WR={s['wr5d']:.1f}%  "
              f"Mean={s['mu5d']:+.3f}%  n={s['n']}")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION D – Joint grid: FG bin × momentum × velocity
# ═══════════════════════════════════════════════════════════════════════════
print_header(
    "SECTION D  —  JOINT GRID SEARCH  (FG 10-pt bin × momentum >= × velocity >=)",
    f"  Min {MIN_SAMPLES} samples required. Ranked by fwd-5d mean return."
)

fg_bands = {
    "FG 00-10": (0,  10),
    "FG 10-20": (10, 20),
    "FG 20-30": (20, 30),
    "FG 30-40": (30, 40),
    "FG 40-50": (40, 50),
    "FG 50-60": (50, 60),
    "FG 60-70": (60, 70),
    "FG 70-80": (70, 80),
    "FG 80-90": (80, 90),
    "FG 90-100":(90, 100),
}

grid_mom = [-1, -0.5, 0, 0.5, 1, 2]
grid_vel = [-1, -0.5, 0, 0.15, 0.3, 0.5, 1]

results_d = []

for band_lbl, (blo, bhi) in fg_bands.items():
    band_data = data[(data["fg"] >= blo) & (data["fg"] <= bhi)]
    for mthr in grid_mom:
        for vthr in grid_vel:
            sub = band_data[
                (band_data["fg_momentum"] >= mthr) &
                (band_data["fg_velocity"] >= vthr)
            ]
            s = summarise(sub)
            if s:
                results_d.append({
                    "fg_band": band_lbl,
                    "mom_thr": mthr,
                    "vel_thr": vthr,
                    **s
                })

df_d = pd.DataFrame(results_d)

if not df_d.empty:
    # Sort by fwd-5d mean, show top 20
    top20 = df_d.sort_values("mu5d", ascending=False).head(20)
    print(f"\n  {'FG Band':<12} {'mom>=':>7} {'vel>=':>7} {'n':>5}  "
          f"{'WR-1d':>7} {'WR-5d':>7} {'WR-10d':>8}  "
          f"{'Mu-1d':>7} {'Mu-5d':>7} {'Mu-10d':>8}  {'Sh-5d':>7}")
    print(SDIV)
    for _, r in top20.iterrows():
        print(f"  {r['fg_band']:<12} {r['mom_thr']:>+7.2f} {r['vel_thr']:>+7.2f} "
              f"{int(r['n']):>5}  "
              f"{r['wr1d']:>6.1f}% {r['wr5d']:>6.1f}% {r['wr10d']:>7.1f}%  "
              f"{r['mu1d']:>+7.3f} {r['mu5d']:>+7.3f} {r['mu10d']:>+8.3f}  "
              f"{r['sh5d']:>+7.3f}")

    # Also show top 10 by Sharpe-5d
    print(f"\n  Top 10 by Sharpe ratio (5d horizon):")
    top_sh = df_d.sort_values("sh5d", ascending=False).head(10)
    print(f"  {'FG Band':<12} {'mom>=':>7} {'vel>=':>7} {'n':>5}  "
          f"{'WR-5d':>7} {'Mu-5d':>7} {'Sh-5d':>7}")
    print(SDIV)
    for _, r in top_sh.iterrows():
        print(f"  {r['fg_band']:<12} {r['mom_thr']:>+7.2f} {r['vel_thr']:>+7.2f} "
              f"{int(r['n']):>5}  "
              f"{r['wr5d']:>6.1f}% {r['mu5d']:>+7.3f} {r['sh5d']:>+7.3f}")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION E – Full profile of the single best entry condition
# ═══════════════════════════════════════════════════════════════════════════
if not df_d.empty:
    best = df_d.sort_values("mu5d", ascending=False).iloc[0]

    print_header(
        "SECTION E  —  FULL PROFILE OF BEST ENTRY CONDITION",
        f"  Condition: {best['fg_band']}  AND  momentum >= {best['mom_thr']:+.2f}  "
        f"AND  velocity >= {best['vel_thr']:+.2f}"
    )

    blo, bhi = fg_bands[best["fg_band"]]
    sub = data[
        (data["fg"] >= blo) & (data["fg"] <= bhi) &
        (data["fg_momentum"] >= best["mom_thr"]) &
        (data["fg_velocity"] >= best["vel_thr"])
    ]

    print(f"\n  Matching days  : {len(sub)} / {N_TOTAL} ({len(sub)/N_TOTAL*100:.1f}% of trading days)")
    print(f"  FG value range : {sub['fg'].min():.1f} – {sub['fg'].max():.1f}  "
          f"(mean {sub['fg'].mean():.1f})")
    print(f"  Momentum range : {sub['fg_momentum'].min():.2f} – {sub['fg_momentum'].max():.2f}  "
          f"(mean {sub['fg_momentum'].mean():.2f})")
    print(f"  Velocity range : {sub['fg_velocity'].min():.2f} – {sub['fg_velocity'].max():.2f}  "
          f"(mean {sub['fg_velocity'].mean():.2f})")

    print(f"\n  {'Horizon':<10} {'Win Rate':>9} {'Mean Ret':>10} {'Median':>10} "
          f"{'95% CI':>12} {'Sharpe':>9} {'Worst':>9} {'Best':>9}")
    print(SDIV)
    for d in FWD_DAYS:
        col  = f"fwd_{d}d"
        wr   = sub[f"win_{d}d"].mean() * 100
        mu   = sub[col].mean()
        med  = sub[col].median()
        ci   = ci95(sub[col])
        sh   = mu / sub[col].std() * np.sqrt(252 / d) if sub[col].std() > 0 else 0
        worst= sub[col].min()
        best_= sub[col].max()
        print(f"  {d:>2}d         {wr:>8.1f}%  {mu:>+9.3f}%  {med:>+9.3f}%  "
              f"±{ci:>9.3f}%  {sh:>+8.3f}  {worst:>+8.2f}%  {best_:>+8.2f}%")

    # Year-by-year breakdown
    sub2 = sub.copy()
    sub2["year"] = sub2["Date"].dt.year
    print(f"\n  Year-by-year fwd-5d win rate and mean return:")
    print(f"  {'Year':<6} {'N':>5} {'WR-5d':>8} {'Mean-5d':>10}")
    print(SDIV)
    for yr, grp in sub2.groupby("year"):
        if len(grp) >= 5:
            wr = grp["win_5d"].mean() * 100
            mu = grp["fwd_5d"].mean()
            print(f"  {yr:<6} {len(grp):>5} {wr:>7.1f}%  {mu:>+9.3f}%")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION F – Backtest strategy-matching condition summary
# ═══════════════════════════════════════════════════════════════════════════
print_header(
    "SECTION F  —  CURRENT BACKTEST PARAMS vs. ALTERNATIVES",
    "  Compares the backtest defaults to neighbouring conditions."
)

# Backtest defaults
current_mom = 0.2
current_vel = 0.15

configs = [
    ("Backtest default", current_mom, current_vel, None),
    ("Tighter mom",      0.5,         current_vel, None),
    ("Tighter vel",      current_mom, 0.30,        None),
    ("Both tighter",     0.5,         0.30,        None),
    ("Relaxed both",     0.0,         0.0,         None),
    ("High mom+vel",     1.0,         0.50,        None),
    ("Very high m+v",    2.0,         1.00,        None),
]

print(f"\n  {'Config':<20} {'mom>=':>7} {'vel>=':>7} {'n':>5}  "
      f"{'WR-1d':>7} {'WR-5d':>7} {'WR-10d':>8}  "
      f"{'Mu-1d':>7} {'Mu-5d':>7} {'Mu-10d':>8}")
print(SDIV)

for name, mthr, vthr, _ in configs:
    sub = data[
        (data["fg_momentum"] >= mthr) &
        (data["fg_velocity"] >= vthr)
    ]
    s = summarise(sub)
    if s:
        print(f"  {name:<20} {mthr:>+7.2f} {vthr:>+7.2f} "
              f"{int(s['n']):>5}  "
              f"{s['wr1d']:>6.1f}% {s['wr5d']:>6.1f}% {s['wr10d']:>7.1f}%  "
              f"{s['mu1d']:>+7.3f} {s['mu5d']:>+7.3f} {s['mu10d']:>+8.3f}")

print(f"\n{DIV}\nDone.\n{DIV}\n")
