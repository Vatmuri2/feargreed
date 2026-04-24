# Fear & Greed Index Trading Bot

> Dashboard auto-updated daily at market close | Last update: **2026-04-24 13:30 PST**

![Portfolio Performance](assets/portfolio_chart.png)

---

## BOD (Morning) Strategy

| Metric | Value |
|--------|-------|
| Portfolio Value | **$24,932.04** |
| Buying Power | $49,864.08 |
| Current FGI | 66.69 |
| Position | FLAT |
| Total P&L | **$-68** |
| Win Rate | 0% (0W / 1L) |
| Total Round Trips | 1 |
| Last Signal | NO_ACTION @ 2026-04-24 06:30 |

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
| 04-24 06:30 | NO_ACTION | $710.99 | 66.69 | -0.22 | -1.41 | 0.1841 | Insufficient momentum/velocity for entry |
| 04-23 13:42 | NO_ACTION | $708.45 | 66.31 | -2.01 | -0.59 | 0.1818 | Insufficient momentum/velocity for entry |
| 04-22 06:30 | SOLD | $709.24 | 67.74 | -1.17 | -0.12 | 0.1824 | Momentum reversal or high volatility |
| 04-21 06:30 | BOUGHT | $710.20 | 70.91 | 1.88 | 0.94 | 0.1798 | Strong momentum/velocity, low volatility |
| 04-20 06:30 | NO_ACTION | $708.99 | 68.09 | 0.00 | 1.58 | 0.1907 | Insufficient momentum/velocity for entry |

</details>

---

## EOD (Afternoon) Strategy

| Metric | Value |
|--------|-------|
| Portfolio Value | **$24,759.44** |
| Buying Power | $49,518.88 |
| Current FGI | 65.8 |
| Position | FLAT |
| Total P&L | **$-323** |
| Win Rate | 0% (0W / 1L) |
| Total Round Trips | 1 |
| Last Signal | NO_ACTION @ 2026-04-24 13:10 |

<details>
<summary>Trade History (1 trades)</summary>

| Buy Date | Sell Date | Buy Price | Sell Price | Qty | P&L | Return | Result |
|----------|-----------|-----------|------------|-----|-----|--------|--------|
| 2026-04-20 | 2026-04-21 | $708.76 | $704.15 | 70 | $-323 | -0.65% | LOSS |

</details>

<details>
<summary>Recent Activity (last 5 entries)</summary>

| Time | Action | Price | FGI | Momentum | Velocity | Volatility | Reason |
|------|--------|-------|-----|----------|----------|------------|--------|
| 04-24 13:10 | NO_ACTION | $713.97 | 65.8 | -0.93 | -0.47 | 0.1841 | Insufficient momentum/velocity for entry |
| 04-23 13:42 | NO_ACTION | $708.45 | 66.31 | -0.89 | -1.07 | 0.1818 | Insufficient momentum/velocity for entry |
| 04-22 13:10 | NO_ACTION | $711.18 | 68.09 | -0.18 | 0.00 | 0.1824 | Insufficient momentum/velocity for entry |
| 04-21 13:10 | SOLD | $704.15 | 67.2 | -1.07 | -0.30 | 0.1798 | Momentum reversal or high volatility |
| 04-20 13:10 | BOUGHT | $708.76 | 69.51 | 0.95 | 2.06 | 0.1907 | Strong momentum/velocity, low volatility |

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
