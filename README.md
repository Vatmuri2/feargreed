# Fear & Greed Index Trading Bot

> Dashboard auto-updated daily at market close | Last update: **2026-06-17 13:30 PST**

![Portfolio Performance](assets/portfolio_chart.png)

---

## BOD (Morning) Strategy

| Metric | Value |
|--------|-------|
| Portfolio Value | **$25,104.08** |
| Buying Power | $100,416.32 |
| Current FGI | 39.06 |
| Position | FLAT |
| Total P&L | **$+995** |
| Win Rate | 50% (1W / 1L) |
| Total Round Trips | 2 |
| Last Signal | NO_ACTION @ 2026-06-17 09:35 |

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
| 06-17 09:35 | NO_ACTION | $750.70 | 39.06 | 0.45 | 2.55 | 0.55 | BUY did not fill after 3 attempts |
| 06-16 09:35 | NO_ACTION | $755.20 | 41.29 | 5.23 | 4.78 | 0.55 | SELL incomplete - still holding 32 after 5 attempts |
| 06-15 09:36 | NO_ACTION | $752.84 | 35.49 | 4.21 | 1.04 | 0.55 | BUY did not fill after 3 attempts |
| 06-12 09:35 | NO_ACTION | $738.25 | 31.4 | 1.16 | -2.79 | 0.55 | Insufficient momentum/velocity for entry |
| 06-11 09:35 | NO_ACTION | $727.08 | 26.94 | -6.09 | -5.11 | 0.55 | Insufficient momentum/velocity for entry |

</details>

---

## EOD (Afternoon) Strategy

| Metric | Value |
|--------|-------|
| Portfolio Value | **$25,512.26** |
| Buying Power | $102,049.04 |
| Current FGI | 33.17 |
| Position | FLAT |
| Total P&L | **$+302** |
| Win Rate | 50% (2W / 2L) |
| Total Round Trips | 4 |
| Last Signal | NO_ACTION @ 2026-06-17 15:52 |

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
| 06-17 15:52 | NO_ACTION | $739.73 | 33.17 | -4.50 | -0.23 | 0.55 | SELL incomplete - still holding 34 after 5 attempts |
| 06-16 15:50 | NO_ACTION | $751.20 | 38.97 | 1.07 | 2.76 | 0.55 | BUY did not fill after 3 attempts |
| 06-15 15:52 | NO_ACTION | $754.49 | 40.86 | 5.72 | 4.27 | 0.55 | SELL incomplete - still holding 34 after 5 attempts |
| 06-12 15:50 | NO_ACTION | $741.06 | 33.86 | 2.99 | 0.75 | 0.55 | BUY did not fill after 3 attempts |
| 06-11 15:50 | NO_ACTION | $737.93 | 30.69 | 0.57 | -3.14 | 0.55 | Insufficient momentum/velocity for entry |

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
