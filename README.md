# Fear & Greed Index Trading Bot

> Dashboard auto-updated daily at market close | Last update: **2026-05-26 13:30 PST**

![Portfolio Performance](assets/portfolio_chart.png)

---

## BOD (Morning) Strategy

| Metric | Value |
|--------|-------|
| Portfolio Value | **$24,673.63** |
| Buying Power | $49,347.26 |
| Current FGI | 58.57 |
| Position | FLAT |
| Total P&L | **$+995** |
| Win Rate | 50% (1W / 1L) |
| Total Round Trips | 2 |
| Last Signal | NO_ACTION @ 2026-05-26 06:30 |

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
| 05-26 06:30 | NO_ACTION | $750.05 | 58.57 | -0.57 | -0.23 | 0.103 | Insufficient momentum/velocity for entry |
| 05-22 06:30 | NO_ACTION | $746.10 | 58.03 | -1.34 | -1.38 | 0.1048 | Insufficient momentum/velocity for entry |
| 05-21 06:30 | NO_ACTION | $738.64 | 60.83 | 0.08 | -0.69 | 0.1071 | Insufficient momentum/velocity for entry |
| 05-20 06:30 | NO_ACTION | $735.64 | 59.26 | -2.19 | -2.04 | 0.107 | Insufficient momentum/velocity for entry |
| 05-19 06:30 | NO_ACTION | $734.86 | 62.17 | -1.31 | -0.73 | 0.1069 | Insufficient momentum/velocity for entry |

</details>

---

## EOD (Afternoon) Strategy

| Metric | Value |
|--------|-------|
| Portfolio Value | **$25,091.36** |
| Buying Power | $50,182.72 |
| Current FGI | 60.71 |
| Position | FLAT |
| Total P&L | **$+9** |
| Win Rate | 33% (1W / 2L) |
| Total Round Trips | 3 |
| Last Signal | NO_ACTION @ 2026-05-26 13:10 |

<details>
<summary>Trade History (3 trades)</summary>

| Buy Date | Sell Date | Buy Price | Sell Price | Qty | P&L | Return | Result |
|----------|-----------|-----------|------------|-----|-----|--------|--------|
| 2026-04-20 | 2026-04-21 | $708.76 | $704.15 | 70 | $-323 | -0.65% | LOSS |
| 2026-05-01 | 2026-05-04 | $720.42 | $717.38 | 68 | $-207 | -0.42% | LOSS |
| 2026-05-05 | 2026-05-07 | $723.86 | $731.90 | 67 | $+539 | +1.11% | WIN |

</details>

<details>
<summary>Recent Activity (last 5 entries)</summary>

| Time | Action | Price | FGI | Momentum | Velocity | Volatility | Reason |
|------|--------|-------|-----|----------|----------|------------|--------|
| 05-26 13:10 | NO_ACTION | $750.56 | 60.71 | 1.27 | -0.02 | 0.103 | Insufficient momentum/velocity for entry |
| 05-22 13:10 | NO_ACTION | $745.70 | 59.14 | -0.32 | -0.54 | 0.1048 | Insufficient momentum/velocity for entry |
| 05-21 13:10 | NO_ACTION | $742.77 | 58.46 | -1.54 | -1.31 | 0.1071 | Insufficient momentum/velocity for entry |
| 05-20 13:10 | NO_ACTION | $741.26 | 60.77 | -0.54 | -1.11 | 0.107 | Insufficient momentum/velocity for entry |
| 05-19 13:10 | NO_ACTION | $733.78 | 60.77 | -1.65 | -1.80 | 0.1069 | Insufficient momentum/velocity for entry |

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
