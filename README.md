# Fear & Greed Index Trading Bot

> Dashboard auto-updated daily at market close | Last update: **2026-09-18 13:30 PST**

![Portfolio Performance](assets/portfolio_chart.png)

---

## BOD (Morning) Strategy

| Metric | Value |
|--------|-------|
| Portfolio Value | **$25,181.75** |
| Buying Power | $100,727.00 |
| Current FGI | 28.37 |
| Position | FLAT |
| Total P&L | **$+995** |
| Win Rate | 50% (1W / 1L) |
| Total Round Trips | 2 |
| Last Signal | NO_ACTION @ 2026-09-18 09:35 |

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
| 09-18 09:35 | NO_ACTION | $760.82 | 28.37 | 0.48 | -0.90 | 0.0917 | Insufficient momentum/velocity for entry |
| 09-17 09:35 | NO_ACTION | $761.73 | 27.6 | -1.19 | -1.69 | 0.0939 | Insufficient momentum/velocity for entry |
| 09-16 09:35 | NO_ACTION | $759.52 | 27.71 | -2.77 | -1.42 | 0.086 | Insufficient momentum/velocity for entry |
| 09-15 09:35 | NO_ACTION | $758.73 | 31.06 | -0.84 | -1.73 | 0.0873 | Insufficient momentum/velocity for entry |
| 09-14 09:35 | NO_ACTION | $758.93 | 32.66 | -0.97 | -2.05 | 0.0901 | Insufficient momentum/velocity for entry |

</details>

---

## EOD (Afternoon) Strategy

| Metric | Value |
|--------|-------|
| Portfolio Value | **$25,611.07** |
| Buying Power | $42,149.72 |
| Current FGI | 29.06 |
| Position | FLAT |
| Total P&L | **$+302** |
| Win Rate | 50% (2W / 2L) |
| Total Round Trips | 4 |
| Last Signal | NO_ACTION @ 2026-09-18 15:51 |

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
| 09-18 15:51 | NO_ACTION | $761.51 | 29.06 | 1.41 | 0.20 | 0.0916 | BUY did not fill after 3 attempts |
| 09-17 15:50 | NO_ACTION | $762.96 | 28.8 | 1.35 | -0.77 | 0.0966 | Insufficient momentum/velocity for entry |
| 09-16 15:50 | NO_ACTION | $753.39 | 25.08 | -3.14 | -2.80 | 0.0868 | Insufficient momentum/velocity for entry |
| 09-15 15:50 | NO_ACTION | $756.90 | 28.46 | -2.56 | -2.23 | 0.0883 | Insufficient momentum/velocity for entry |
| 09-14 15:50 | NO_ACTION | $761.63 | 31.11 | -2.14 | -2.81 | 0.0877 | Insufficient momentum/velocity for entry |

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
