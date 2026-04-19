# Fear & Greed Index Trading Bot

> Dashboard auto-updated daily at market close | Last update: **2026-04-18 23:47 PST**

![Portfolio Performance](assets/portfolio_chart.png)

---

## BOD (Morning) Strategy

| Metric | Value |
|--------|-------|
| Portfolio Value | **$25,000.00** |
| Buying Power | $50,000.00 |
| Current FGI | 68.09 |
| Position | NO POSITION |
| Total P&L | **N/A** |
| Win Rate | N/A |
| Total Round Trips | 0 |
| Last Signal | BOUGHT @ 2026-04-17 20:41 |

<details>
<summary>Trade History (1 trades)</summary>

| Buy Date | Sell Date | Buy Price | Sell Price | Qty | P&L | Return | Result |
|----------|-----------|-----------|------------|-----|-----|--------|--------|
| 2026-04-17 | — | $710.14 | — | 281 | — | — | OPEN |

</details>

<details>
<summary>Recent Activity (last 5 entries)</summary>

| Time | Action | Price | FGI | Momentum | Velocity | Volatility | Reason |
|------|--------|-------|-----|----------|----------|------------|--------|
| 04-17 20:41 | BOUGHT | $710.14 | 68.09 | 1.58 | 3.49 | 0.1894 | Strong momentum/velocity, low volatility |

</details>

---

## EOD (Afternoon) Strategy

| Metric | Value |
|--------|-------|
| Portfolio Value | **$25,000.00** |
| Buying Power | $50,000.00 |
| Current FGI | 68.09 |
| Position | NO POSITION |
| Total P&L | **N/A** |
| Win Rate | N/A |
| Total Round Trips | 0 |
| Last Signal | NO_ACTION @ 2026-04-17 20:29 |

<details>
<summary>Trade History (1 trades)</summary>

| Buy Date | Sell Date | Buy Price | Sell Price | Qty | P&L | Return | Result |
|----------|-----------|-----------|------------|-----|-----|--------|--------|
| 2026-04-17 | — | $710.14 | — | 281 | — | — | OPEN |

</details>

<details>
<summary>Recent Activity (last 5 entries)</summary>

| Time | Action | Price | FGI | Momentum | Velocity | Volatility | Reason |
|------|--------|-------|-----|----------|----------|------------|--------|
| 04-17 20:29 | NO_ACTION | $710.14 | 68.09 | 1.58 | 3.49 | 0.1894 | Strong momentum/velocity, low volatility |
| 04-17 20:28 | BOUGHT | $710.14 | 68.09 | 1.58 | 3.49 | 0.1894 | Strong momentum/velocity, low volatility |

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
