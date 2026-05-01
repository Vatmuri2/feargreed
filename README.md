# Fear & Greed Index Trading Bot

> Dashboard auto-updated daily at market close | Last update: **2026-05-01 13:30 PST**

![Portfolio Performance](assets/portfolio_chart.png)

---

## BOD (Morning) Strategy

| Metric | Value |
|--------|-------|
| Portfolio Value | **$24,932.04** |
| Buying Power | $49,864.08 |
| Current FGI | 66.77 |
| Position | FLAT |
| Total P&L | **$-68** |
| Win Rate | 0% (0W / 1L) |
| Total Round Trips | 1 |
| Last Signal | NO_ACTION @ 2026-05-01 06:30 |

<details>
<summary>Trade History (1 trades)</summary>

| Buy Date | Sell Date | Buy Price | Sell Price | Qty | P&L | Return | Result |
|----------|-----------|-----------|------------|-----|-----|--------|--------|
| 2026-04-21 | 2026-04-22 | $710.20 | $709.24 | 70 | $-68 | -0.14% | LOSS |

</details>

<details>
<summary>Recent Activity (last 5 entries)</summary>

| Time | Action | Price | FGI | Momentum | Velocity | Volatility | Reason |
|------|--------|-------|-----|----------|----------|------------|--------|
| 05-01 06:30 | NO_ACTION | $721.32 | 66.77 | 2.09 | -0.36 | 0.1197 | Insufficient momentum/velocity for entry |
| 04-30 06:30 | NO_ACTION | $714.69 | 63.43 | -1.61 | -0.87 | 0.1186 | Insufficient momentum/velocity for entry |
| 04-29 06:30 | NO_ACTION | $710.85 | 63.83 | -2.08 | -0.95 | 0.1456 | Insufficient momentum/velocity for entry |
| 04-28 06:30 | NO_ACTION | $711.82 | 67.86 | 1.00 | 0.52 | 0.1442 | Strong momentum/velocity, low volatility |
| 04-27 06:30 | NO_ACTION | $713.19 | 66.03 | -0.31 | -0.57 | 0.1655 | Insufficient momentum/velocity for entry |

</details>

---

## EOD (Afternoon) Strategy

| Metric | Value |
|--------|-------|
| Portfolio Value | **$24,757.40** |
| Buying Power | $528.28 |
| Current FGI | 66.6 |
| Position | IN POSITION |
| Total P&L | **$-323** |
| Win Rate | 0% (0W / 1L) |
| Total Round Trips | 1 |
| Last Signal | BOUGHT @ 2026-05-01 13:11 |

<details>
<summary>Trade History (2 trades)</summary>

| Buy Date | Sell Date | Buy Price | Sell Price | Qty | P&L | Return | Result |
|----------|-----------|-----------|------------|-----|-----|--------|--------|
| 2026-04-20 | 2026-04-21 | $708.76 | $704.15 | 70 | $-323 | -0.65% | LOSS |
| 2026-05-01 | — | $720.42 | — | 68 | — | — | OPEN |

</details>

<details>
<summary>Recent Activity (last 5 entries)</summary>

| Time | Action | Price | FGI | Momentum | Velocity | Volatility | Reason |
|------|--------|-------|-----|----------|----------|------------|--------|
| 05-01 13:11 | BOUGHT | $720.42 | 66.6 | 1.17 | 0.78 | 0.1197 | Strong momentum/velocity, low volatility |
| 04-30 13:10 | NO_ACTION | $718.66 | 66.46 | 1.81 | -0.29 | 0.1186 | Insufficient momentum/velocity for entry |
| 04-29 13:10 | NO_ACTION | $711.63 | 63.23 | -1.71 | -0.86 | 0.1456 | Insufficient momentum/velocity for entry |
| 04-28 13:10 | NO_ACTION | $711.54 | 64.26 | -1.54 | -0.68 | 0.1442 | Insufficient momentum/velocity for entry |
| 04-27 13:10 | NO_ACTION | $715.10 | 67.34 | 0.86 | -0.25 | 0.1655 | Insufficient momentum/velocity for entry |

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
