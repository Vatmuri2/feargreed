# Fear & Greed Index Trading Bot

> Dashboard auto-updated daily at market close | Last update: **2026-06-24 13:30 PST**

![Portfolio Performance](assets/portfolio_chart.png)

---

## BOD (Morning) Strategy

| Metric | Value |
|--------|-------|
| Portfolio Value | **$25,091.24** |
| Buying Power | $100,364.96 |
| Current FGI | 27.34 |
| Position | FLAT |
| Total P&L | **$+995** |
| Win Rate | 50% (1W / 1L) |
| Total Round Trips | 2 |
| Last Signal | NO_ACTION @ 2026-06-24 09:35 |

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
| 06-24 09:35 | NO_ACTION | $736.74 | 27.34 | -4.88 | -1.75 | 0.55 | Insufficient momentum/velocity for entry |
| 06-23 09:35 | NO_ACTION | $733.76 | 32.06 | -1.91 | -2.33 | 0.55 | Insufficient momentum/velocity for entry |
| 06-22 09:35 | NO_ACTION | $748.05 | 37.26 | 0.95 | -1.34 | 0.55 | SELL incomplete - still holding 33 after 5 attempts |
| 06-18 09:37 | NO_ACTION | $746.05 | 32.6 | -5.05 | -0.96 | 0.55 | SELL incomplete - still holding 33 after 5 attempts |
| 06-17 09:35 | NO_ACTION | $750.70 | 39.06 | 0.45 | 2.55 | 0.55 | BUY did not fill after 3 attempts |

</details>

---

## EOD (Afternoon) Strategy

| Metric | Value |
|--------|-------|
| Portfolio Value | **$25,512.23** |
| Buying Power | $102,048.92 |
| Current FGI | 25.1 |
| Position | FLAT |
| Total P&L | **$+302** |
| Win Rate | 50% (2W / 2L) |
| Total Round Trips | 4 |
| Last Signal | NO_ACTION @ 2026-06-24 15:50 |

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
| 06-24 15:50 | NO_ACTION | $732.08 | 25.1 | -4.13 | -4.05 | 0.55 | Insufficient momentum/velocity for entry |
| 06-23 15:50 | NO_ACTION | $734.80 | 28.09 | -5.20 | -1.69 | 0.55 | Insufficient momentum/velocity for entry |
| 06-22 15:50 | NO_ACTION | $743.81 | 34.51 | -0.47 | -1.49 | 0.55 | Insufficient momentum/velocity for entry |
| 06-18 15:50 | NO_ACTION | $746.81 | 37.26 | 0.79 | -1.20 | 0.55 | Insufficient momentum/velocity for entry |
| 06-17 15:52 | NO_ACTION | $739.73 | 33.17 | -4.50 | -0.23 | 0.55 | SELL incomplete - still holding 34 after 5 attempts |

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
