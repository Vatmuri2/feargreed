# Fear & Greed Index Trading Bot

> Dashboard auto-updated daily at market close | Last update: **2026-07-16 13:30 PST**

![Portfolio Performance](assets/portfolio_chart.png)

---

## BOD (Morning) Strategy

| Metric | Value |
|--------|-------|
| Portfolio Value | **$24,905.41** |
| Buying Power | $99,621.64 |
| Current FGI | 47.97 |
| Position | FLAT |
| Total P&L | **$+995** |
| Win Rate | 50% (1W / 1L) |
| Total Round Trips | 2 |
| Last Signal | NO_ACTION @ 2026-07-16 09:35 |

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
| 07-16 09:35 | NO_ACTION | $752.50 | 47.97 | 2.93 | -0.03 | 0.55 | Insufficient momentum/velocity for entry |
| 07-15 09:35 | NO_ACTION | $754.90 | 45.6 | 0.53 | -0.10 | 0.55 | Insufficient momentum/velocity for entry |
| 07-14 09:35 | NO_ACTION | $749.51 | 41.54 | -3.62 | -0.73 | 0.55 | Insufficient momentum/velocity for entry |
| 07-13 09:35 | NO_ACTION | $753.58 | 48.06 | 2.16 | 1.45 | 0.55 | SELL incomplete - still holding 32 after 5 attempts |
| 07-10 09:35 | NO_ACTION | $752.62 | 45.89 | 1.44 | 0.53 | 0.55 | BUY did not fill after 3 attempts |

</details>

---

## EOD (Afternoon) Strategy

| Metric | Value |
|--------|-------|
| Portfolio Value | **$25,626.48** |
| Buying Power | $102,505.92 |
| Current FGI | 40.17 |
| Position | FLAT |
| Total P&L | **$+302** |
| Win Rate | 50% (2W / 2L) |
| Total Round Trips | 4 |
| Last Signal | NO_ACTION @ 2026-07-16 15:50 |

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
| 07-16 15:50 | NO_ACTION | $748.42 | 40.17 | -2.66 | -0.95 | 0.55 | Insufficient momentum/velocity for entry |
| 07-15 15:50 | NO_ACTION | $753.88 | 45.49 | 1.71 | -1.18 | 0.55 | Insufficient momentum/velocity for entry |
| 07-14 15:50 | NO_ACTION | $752.33 | 42.83 | -2.13 | -1.36 | 0.55 | Insufficient momentum/velocity for entry |
| 07-13 15:50 | NO_ACTION | $748.86 | 43.03 | -3.29 | 0.39 | 0.55 | Insufficient momentum/velocity for entry |
| 07-10 15:52 | NO_ACTION | $755.33 | 49.03 | 3.10 | 2.01 | 0.55 | SELL incomplete - still holding 33 after 5 attempts |

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
