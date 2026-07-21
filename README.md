# Fear & Greed Index Trading Bot

> Dashboard auto-updated daily at market close | Last update: **2026-07-21 13:30 PST**

![Portfolio Performance](assets/portfolio_chart.png)

---

## BOD (Morning) Strategy

| Metric | Value |
|--------|-------|
| Portfolio Value | **$24,905.41** |
| Buying Power | $99,621.64 |
| Current FGI | 37.97 |
| Position | FLAT |
| Total P&L | **$+995** |
| Win Rate | 50% (1W / 1L) |
| Total Round Trips | 2 |
| Last Signal | NO_ACTION @ 2026-07-21 09:35 |

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
| 07-21 09:35 | NO_ACTION | $745.47 | 37.97 | -0.83 | -3.33 | 0.55 | Insufficient momentum/velocity for entry |
| 07-20 09:35 | NO_ACTION | $747.81 | 36.97 | -5.16 | -2.88 | 0.55 | Insufficient momentum/velocity for entry |
| 07-17 09:35 | NO_ACTION | $741.17 | 41.46 | -3.55 | -0.03 | 0.55 | Insufficient momentum/velocity for entry |
| 07-16 09:35 | NO_ACTION | $752.50 | 47.97 | 2.93 | -0.03 | 0.55 | Insufficient momentum/velocity for entry |
| 07-15 09:35 | NO_ACTION | $754.90 | 45.6 | 0.53 | -0.10 | 0.55 | Insufficient momentum/velocity for entry |

</details>

---

## EOD (Afternoon) Strategy

| Metric | Value |
|--------|-------|
| Portfolio Value | **$25,626.48** |
| Buying Power | $102,505.92 |
| Current FGI | 40.74 |
| Position | FLAT |
| Total P&L | **$+302** |
| Win Rate | 50% (2W / 2L) |
| Total Round Trips | 4 |
| Last Signal | NO_ACTION @ 2026-07-21 15:50 |

<details>
<summary>Trade History (4 trades)</summary>

| Buy Date | Sell Date | Buy Price | Sell Price | Qty | P&L | Return | Result |
|----------|-----------|-----------|------------|-----|-----|--------|--------|
| 2026-04-20 | 2026-04-21 | $708.76 | $704.15 | 70 | $-323 | -0.65% | LOSS |
| 2026-05-01 | 2026-05-04 | $720.42 | $717.38 | 68 | $-207 | -0.42% | LOSS |
| 2026-05-05 | 2026-05-07 | $723.86 | $731.90 | 67 | $+539 | +1.11% | WIN |
| 2026-05-27 | 2026-05-28 | $750.78 | $755.22 | 66 | $+293 | +0.59% | WIN |

</details>

<details>
<summary>Recent Activity (last 5 entries)</summary>

| Time | Action | Price | FGI | Momentum | Velocity | Volatility | Reason |
|------|--------|-------|-----|----------|----------|------------|--------|
| 07-21 15:50 | NO_ACTION | $748.32 | 40.74 | 1.95 | 0.19 | 0.55 | BUY did not fill after 3 attempts |
| 07-20 15:50 | NO_ACTION | $742.56 | 38.11 | -0.49 | -2.46 | 0.55 | Insufficient momentum/velocity for entry |
| 07-17 15:50 | NO_ACTION | $744.12 | 37.51 | -3.55 | -1.77 | 0.55 | Insufficient momentum/velocity for entry |
| 07-16 15:50 | NO_ACTION | $748.42 | 40.17 | -2.66 | -0.95 | 0.55 | Insufficient momentum/velocity for entry |
| 07-15 15:50 | NO_ACTION | $753.88 | 45.49 | 1.71 | -1.18 | 0.55 | Insufficient momentum/velocity for entry |

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
