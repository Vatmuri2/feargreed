# Fear & Greed Index Trading Bot

> Dashboard auto-updated daily at market close | Last update: **2026-08-18 13:30 PST**

![Portfolio Performance](assets/portfolio_chart.png)

---

## BOD (Morning) Strategy

| Metric | Value |
|--------|-------|
| Portfolio Value | **$25,258.57** |
| Buying Power | $101,034.28 |
| Current FGI | 58.69 |
| Position | FLAT |
| Total P&L | **$+995** |
| Win Rate | 50% (1W / 1L) |
| Total Round Trips | 2 |
| Last Signal | NO_ACTION @ 2026-08-18 09:35 |

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
| 08-18 09:35 | NO_ACTION | $768.53 | 58.69 | -4.45 | -1.41 | 0.55 | Insufficient momentum/velocity for entry |
| 08-17 09:35 | NO_ACTION | $775.69 | 64.06 | -0.48 | 0.87 | 0.55 | SELL incomplete - still holding 32 after 5 attempts |
| 08-14 09:35 | NO_ACTION | $777.74 | 66.66 | 2.98 | 0.62 | 0.55 | BUY did not fill after 3 attempts |
| 08-13 09:35 | NO_ACTION | $775.40 | 62.91 | -0.15 | -0.54 | 0.55 | Insufficient momentum/velocity for entry |
| 08-12 09:37 | NO_ACTION | $772.97 | 61.46 | -2.14 | 0.92 | 0.55 | SELL incomplete - still holding 32 after 5 attempts |

</details>

---

## EOD (Afternoon) Strategy

| Metric | Value |
|--------|-------|
| Portfolio Value | **$25,926.95** |
| Buying Power | $103,707.80 |
| Current FGI | 54.57 |
| Position | FLAT |
| Total P&L | **$+302** |
| Win Rate | 50% (2W / 2L) |
| Total Round Trips | 4 |
| Last Signal | NO_ACTION @ 2026-08-18 15:50 |

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
| 08-18 15:50 | NO_ACTION | $767.90 | 54.57 | -5.30 | -3.88 | 0.55 | Insufficient momentum/velocity for entry |
| 08-17 15:50 | NO_ACTION | $773.25 | 60.34 | -3.41 | -0.72 | 0.55 | Insufficient momentum/velocity for entry |
| 08-14 15:52 | NO_ACTION | $776.27 | 64.71 | 0.24 | 1.27 | 0.55 | SELL incomplete - still holding 33 after 5 attempts |
| 08-13 15:50 | NO_ACTION | $777.67 | 66.2 | 3.00 | 0.60 | 0.55 | BUY did not fill after 3 attempts |
| 08-12 15:50 | NO_ACTION | $772.51 | 62.51 | -0.09 | -0.41 | 0.55 | Insufficient momentum/velocity for entry |

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
