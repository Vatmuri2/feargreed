# Fear & Greed Index Trading Bot

> Dashboard auto-updated daily at market close | Last update: **2026-04-22 13:30 PST**

![Portfolio Performance](assets/portfolio_chart.png)

---

## BOD (Morning) Strategy

| Metric | Value |
|--------|-------|
| Portfolio Value | **$24,941.16** |
| Buying Power | $258.62 |
| Current FGI | 67.74 |
| Position | IN POSITION |
| Total P&L | **$-254** |
| Win Rate | 0% (0W / 1L) |
| Total Round Trips | 1 |
| Last Signal | SOLD @ 2026-04-22 06:30 |

<details>
<summary>Trade History (2 trades)</summary>

| Buy Date | Sell Date | Buy Price | Sell Price | Qty | P&L | Return | Result |
|----------|-----------|-----------|------------|-----|-----|--------|--------|
| 2026-04-17 | 2026-04-22 | $710.14 | $709.24 | 281 | $-254 | -0.13% | LOSS |
| 2026-04-21 | — | $710.20 | — | 70 | — | — | OPEN |

</details>

<details>
<summary>Recent Activity (last 5 entries)</summary>

| Time | Action | Price | FGI | Momentum | Velocity | Volatility | Reason |
|------|--------|-------|-----|----------|----------|------------|--------|
| 04-22 06:30 | SOLD | $709.24 | 67.74 | -1.17 | -0.12 | 0.1824 | Momentum reversal or high volatility |
| 04-21 06:30 | BOUGHT | $710.20 | 70.91 | 1.88 | 0.94 | 0.1798 | Strong momentum/velocity, low volatility |
| 04-20 06:30 | NO_ACTION | $708.99 | 68.09 | 0.00 | 1.58 | 0.1907 | Insufficient momentum/velocity for entry |
| 04-17 20:41 | BOUGHT | $710.14 | 68.09 | 1.58 | 3.49 | 0.1894 | Strong momentum/velocity, low volatility |

</details>

---

## EOD (Afternoon) Strategy

| Metric | Value |
|--------|-------|
| Portfolio Value | **$24,759.44** |
| Buying Power | $49,518.88 |
| Current FGI | 68.09 |
| Position | IN POSITION |
| Total P&L | **$-1,683** |
| Win Rate | 0% (0W / 1L) |
| Total Round Trips | 1 |
| Last Signal | NO_ACTION @ 2026-04-22 13:10 |

<details>
<summary>Trade History (2 trades)</summary>

| Buy Date | Sell Date | Buy Price | Sell Price | Qty | P&L | Return | Result |
|----------|-----------|-----------|------------|-----|-----|--------|--------|
| 2026-04-17 | 2026-04-21 | $710.14 | $704.15 | 281 | $-1,683 | -0.84% | LOSS |
| 2026-04-20 | — | $708.76 | — | 70 | — | — | OPEN |

</details>

<details>
<summary>Recent Activity (last 5 entries)</summary>

| Time | Action | Price | FGI | Momentum | Velocity | Volatility | Reason |
|------|--------|-------|-----|----------|----------|------------|--------|
| 04-22 13:10 | NO_ACTION | $711.18 | 68.09 | -0.18 | 0.00 | 0.1824 | Insufficient momentum/velocity for entry |
| 04-21 13:10 | SOLD | $704.15 | 67.2 | -1.07 | -0.30 | 0.1798 | Momentum reversal or high volatility |
| 04-20 13:10 | BOUGHT | $708.76 | 69.51 | 0.95 | 2.06 | 0.1907 | Strong momentum/velocity, low volatility |
| 04-17 20:28 | BOUGHT | $710.14 | 68.09 | 1.58 | 3.49 | 0.1894 | Strong momentum/velocity, low volatility |
| 04-17 20:29 | NO_ACTION | $710.14 | 68.09 | 1.58 | 3.49 | 0.1894 | Strong momentum/velocity, low volatility |

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
