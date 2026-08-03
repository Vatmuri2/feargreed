# Fear & Greed Index Trading Bot

> Dashboard auto-updated daily at market close | Last update: **2026-08-03 13:30 PST**

![Portfolio Performance](assets/portfolio_chart.png)

---

## BOD (Morning) Strategy

| Metric | Value |
|--------|-------|
| Portfolio Value | **$24,893.90** |
| Buying Power | $67,119.01 |
| Current FGI | 44.8 |
| Position | FLAT |
| Total P&L | **$+995** |
| Win Rate | 50% (1W / 1L) |
| Total Round Trips | 2 |
| Last Signal | NO_ACTION @ 2026-08-03 09:35 |

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
| 08-03 09:35 | NO_ACTION | $750.80 | 44.8 | 4.87 | 2.32 | 0.55 | SELL incomplete - still holding 36 after 5 attempts |
| 07-31 09:36 | NO_ACTION | $745.86 | 40.34 | 2.73 | 0.84 | 0.55 | BUY did not fill after 3 attempts |
| 07-30 09:35 | NO_ACTION | $735.64 | 34.66 | -2.11 | -1.59 | 0.55 | Insufficient momentum/velocity for entry |
| 07-29 09:35 | NO_ACTION | $739.73 | 37.83 | -0.53 | -0.34 | 0.55 | Insufficient momentum/velocity for entry |
| 07-28 09:35 | NO_ACTION | $738.66 | 37.83 | -0.88 | -1.78 | 0.55 | Insufficient momentum/velocity for entry |

</details>

---

## EOD (Afternoon) Strategy

| Metric | Value |
|--------|-------|
| Portfolio Value | **$25,956.17** |
| Buying Power | $103,824.68 |
| Current FGI | 45.91 |
| Position | FLAT |
| Total P&L | **$+302** |
| Win Rate | 50% (2W / 2L) |
| Total Round Trips | 4 |
| Last Signal | NO_ACTION @ 2026-08-03 15:52 |

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
| 08-03 15:52 | NO_ACTION | $758.15 | 45.91 | 2.83 | 3.30 | 0.55 | SELL incomplete - still holding 34 after 5 attempts |
| 07-31 15:51 | NO_ACTION | $748.53 | 44.06 | 4.29 | 1.92 | 0.55 | BUY did not fill after 3 attempts |
| 07-30 15:50 | NO_ACTION | $741.45 | 39.26 | 1.41 | -0.30 | 0.55 | Insufficient momentum/velocity for entry |
| 07-29 15:50 | NO_ACTION | $731.43 | 36.0 | -2.15 | -1.03 | 0.55 | Insufficient momentum/velocity for entry |
| 07-28 15:50 | NO_ACTION | $741.12 | 38.29 | -0.89 | -0.26 | 0.55 | Insufficient momentum/velocity for entry |

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
