# Fear & Greed Index Trading Bot

> Dashboard auto-updated daily at market close | Last update: **2026-09-14 13:30 PST**

![Portfolio Performance](assets/portfolio_chart.png)

---

## BOD (Morning) Strategy

| Metric | Value |
|--------|-------|
| Portfolio Value | **$25,181.75** |
| Buying Power | $100,727.00 |
| Current FGI | 32.66 |
| Position | FLAT |
| Total P&L | **$+995** |
| Win Rate | 50% (1W / 1L) |
| Total Round Trips | 2 |
| Last Signal | NO_ACTION @ 2026-09-14 09:35 |

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
| 09-14 09:35 | NO_ACTION | $758.93 | 32.66 | -0.97 | -2.05 | 0.0901 | Insufficient momentum/velocity for entry |
| 09-11 09:35 | NO_ACTION | $765.87 | 31.97 | -3.71 | -3.49 | 0.0906 | Insufficient momentum/velocity for entry |
| 09-10 09:35 | NO_ACTION | $757.96 | 36.26 | -2.90 | -2.64 | 0.0856 | Insufficient momentum/velocity for entry |
| 09-09 09:35 | NO_ACTION | $764.02 | 38.8 | -3.00 | 2.05 | 0.0831 | SELL incomplete - still holding 32 after 5 attempts |
| 09-08 09:36 | NO_ACTION | $768.76 | 42.43 | 2.68 | 3.86 | 0.0816 | BUY did not fill after 3 attempts |

</details>

---

## EOD (Afternoon) Strategy

| Metric | Value |
|--------|-------|
| Portfolio Value | **$25,613.62** |
| Buying Power | $102,454.48 |
| Current FGI | 31.11 |
| Position | FLAT |
| Total P&L | **$+302** |
| Win Rate | 50% (2W / 2L) |
| Total Round Trips | 4 |
| Last Signal | NO_ACTION @ 2026-09-14 15:50 |

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
| 09-14 15:50 | NO_ACTION | $761.63 | 31.11 | -2.14 | -2.81 | 0.0877 | Insufficient momentum/velocity for entry |
| 09-11 15:50 | NO_ACTION | $764.53 | 33.49 | -2.57 | -2.58 | 0.0885 | Insufficient momentum/velocity for entry |
| 09-10 15:50 | NO_ACTION | $758.26 | 35.14 | -3.50 | -2.22 | 0.0852 | Insufficient momentum/velocity for entry |
| 09-09 15:50 | NO_ACTION | $762.76 | 39.54 | -1.32 | 1.40 | 0.0839 | SELL incomplete - still holding 66 after 5 attempts |
| 09-08 15:51 | NO_ACTION | $766.33 | 41.23 | 1.77 | 2.49 | 0.083 | BUY did not fill after 3 attempts |

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
