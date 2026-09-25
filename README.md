# Fear & Greed Index Trading Bot

> Dashboard auto-updated daily at market close | Last update: **2026-09-25 13:30 PST**

![Portfolio Performance](assets/portfolio_chart.png)

---

## BOD (Morning) Strategy

| Metric | Value |
|--------|-------|
| Portfolio Value | **$25,043.70** |
| Buying Power | $41,082.58 |
| Current FGI | 35.71 |
| Position | FLAT |
| Total P&L | **$+995** |
| Win Rate | 50% (1W / 1L) |
| Total Round Trips | 2 |
| Last Signal | NO_ACTION @ 2026-09-25 09:36 |

<details>
<summary>Trade History (2 trades)</summary>

| Buy Date | Sell Date | Buy Price | Sell Price | Qty | P&L | Return | Result |
|----------|-----------|-----------|------------|-----|-----|--------|--------|
| 2026-04-21 | 2026-04-22 | $710.20 | $709.24 | 70 | $-68 | -0.14% | LOSS |
| 2026-05-04 | 2026-05-08 | $719.65 | $735.05 | 69 | $+1,063 | +2.14% | WIN |

</details>

<details>
<summary>Recent Activity (last 5 entries)</summary>

| Time | Action | Price | FGI | Momentum | Velocity | Volatility | Reason |
|------|--------|-------|-----|----------|----------|------------|--------|
| 09-25 09:36 | NO_ACTION | $769.53 | 35.71 | 0.64 | 0.55 | 0.1065 | BUY did not fill after 3 attempts |
| 09-24 09:35 | NO_ACTION | $764.98 | 34.63 | 0.11 | 1.42 | 0.1099 | SELL incomplete - still holding 64 after 5 attempts |
| 09-23 09:35 | NO_ACTION | $772.33 | 34.86 | 1.76 | 2.16 | 0.1056 | Holding position - indicators still favorable (2/8 days) |
| 09-22 09:35 | NO_ACTION | $774.79 | 34.06 | 3.13 | 2.15 | 0.1058 | Holding position - indicators still favorable (1/8 days) |
| 09-21 09:36 | NO_ACTION | $766.19 | 30.37 | 1.59 | 0.89 | 0.0934 | BUY did not fill after 3 attempts |

</details>

---

## EOD (Afternoon) Strategy

| Metric | Value |
|--------|-------|
| Portfolio Value | **$26,314.30** |
| Buying Power | $44,110.44 |
| Current FGI | 36.57 |
| Position | IN POSITION |
| Total P&L | **$+302** |
| Win Rate | 50% (2W / 2L) |
| Total Round Trips | 4 |
| Last Signal | NO_ACTION @ 2026-09-25 15:50 |

<details>
<summary>Trade History (5 trades)</summary>

| Buy Date | Sell Date | Buy Price | Sell Price | Qty | P&L | Return | Result |
|----------|-----------|-----------|------------|-----|-----|--------|--------|
| 2026-04-20 | 2026-04-21 | $708.76 | $704.15 | 70 | $-323 | -0.65% | LOSS |
| 2026-05-01 | 2026-05-04 | $720.42 | $717.38 | 68 | $-207 | -0.42% | LOSS |
| 2026-05-05 | 2026-05-07 | $723.86 | $731.90 | 67 | $+539 | +1.11% | WIN |
| 2026-05-27 | 2026-05-28 | $750.78 | $755.22 | 66 | $+293 | +0.59% | WIN |
| 2026-09-24 | — | unknown (late/unlogged fill) | — | 66 | — | — | OPEN |

</details>

<details>
<summary>Recent Activity (last 5 entries)</summary>

| Time | Action | Price | FGI | Momentum | Velocity | Volatility | Reason |
|------|--------|-------|-----|----------|----------|------------|--------|
| 09-25 15:50 | NO_ACTION | $772.10 | 36.57 | 0.74 | 0.20 | 0.1081 | Holding position - indicators still favorable (1/8 days) |
| 09-24 15:51 | NO_ACTION | $767.69 | 36.0 | 0.37 | 0.48 | 0.1089 | BUY did not fill after 3 attempts |
| 09-23 15:50 | NO_ACTION | $767.52 | 34.91 | -0.24 | 1.95 | 0.1088 | SELL incomplete - still holding 66 after 5 attempts |
| 09-22 15:50 | NO_ACTION | $774.67 | 35.97 | 2.77 | 2.39 | 0.1058 | Holding position - indicators still favorable (2/8 days) |
| 09-21 15:50 | NO_ACTION | $773.92 | 34.57 | 3.76 | 3.16 | 0.1083 | Holding position - indicators still favorable (1/8 days) |

</details>

---

## Strategy

Momentum-based strategy using CNN Fear & Greed Index to trade SPY.

| Parameter | Value |
|-----------|-------|
| Momentum Threshold | 0.2 |
| Velocity Threshold | 0.15 |
| Volatility Buy Limit | 0.6 |
| Volatility Sell Limit | 0.5 |
| Max Days Held | 8 |
| Lookback Days | 3 |
| BOD Execution | 6:20 AM PST |
| EOD Execution | 1:10 PM PST |
