# Fear & Greed Index Trading Bot

> Dashboard auto-updated daily at market close | Last update: **2026-10-09 13:30 PST**

![Portfolio Performance](assets/portfolio_chart.png)

---

## BOD (Morning) Strategy

| Metric | Value |
|--------|-------|
| Portfolio Value | **$25,048.90** |
| Buying Power | $100,195.60 |
| Current FGI | 39.74 |
| Position | FLAT |
| Total P&L | **$+995** |
| Win Rate | 50% (1W / 1L) |
| Total Round Trips | 2 |
| Last Signal | NO_ACTION @ 2026-10-09 09:35 |

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
| 10-09 09:35 | NO_ACTION | $776.56 | 39.74 | -3.23 | -1.32 | 0.0989 | Insufficient momentum/velocity for entry |
| 10-08 09:35 | NO_ACTION | $775.05 | 42.11 | -2.18 | 0.77 | 0.1011 | SELL incomplete - still holding 62 after 5 attempts |
| 10-07 09:35 | NO_ACTION | $775.14 | 47.06 | 3.54 | 6.63 | 0.1046 | Holding position - indicators still favorable (2/8 days) |
| 10-06 09:35 | NO_ACTION | $778.15 | 43.71 | 6.82 | 4.59 | 0.1041 | Holding position - indicators still favorable (1/8 days) |
| 10-05 09:36 | NO_ACTION | $770.80 | 39.8 | 7.50 | 3.70 | 0.1028 | BUY did not fill after 3 attempts |

</details>

---

## EOD (Afternoon) Strategy

| Metric | Value |
|--------|-------|
| Portfolio Value | **$25,951.15** |
| Buying Power | $103,804.60 |
| Current FGI | 45.66 |
| Position | FLAT |
| Total P&L | **$+302** |
| Win Rate | 50% (2W / 2L) |
| Total Round Trips | 4 |
| Last Signal | NO_ACTION @ 2026-10-09 15:50 |

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
| 10-09 15:50 | NO_ACTION | $778.54 | 45.66 | 3.12 | -0.68 | 0.1003 | Insufficient momentum/velocity for entry |
| 10-08 15:50 | NO_ACTION | $773.76 | 37.77 | -5.46 | -1.97 | 0.1023 | Insufficient momentum/velocity for entry |
| 10-07 15:50 | NO_ACTION | $777.29 | 44.2 | -1.00 | 4.30 | 0.1035 | SELL incomplete - still holding 65 after 5 attempts |
| 10-06 15:50 | NO_ACTION | $779.33 | 47.71 | 6.81 | 6.42 | 0.1051 | Holding position - indicators still favorable (1/8 days) |
| 10-05 15:51 | NO_ACTION | $774.85 | 43.69 | 9.20 | 3.59 | 0.1055 | BUY did not fill after 3 attempts |

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
