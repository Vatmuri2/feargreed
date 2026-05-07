# Fear & Greed Index Trading Bot

> Dashboard auto-updated daily at market close | Last update: **2026-05-07 13:30 PST**

![Portfolio Performance](assets/portfolio_chart.png)

---

## BOD (Morning) Strategy

| Metric | Value |
|--------|-------|
| Portfolio Value | **$25,478.78** |
| Buying Power | $3,918.20 |
| Current FGI | 68.63 |
| Position | IN POSITION |
| Total P&L | **$-68** |
| Win Rate | 0% (0W / 1L) |
| Total Round Trips | 1 |
| Last Signal | NO_ACTION @ 2026-05-07 06:30 |

<details>
<summary>Trade History (2 trades)</summary>

| Buy Date | Sell Date | Buy Price | Sell Price | Qty | P&L | Return | Result |
|----------|-----------|-----------|------------|-----|-----|--------|--------|
| 2026-04-21 | 2026-04-22 | $710.20 | $709.24 | 70 | $-68 | -0.14% | LOSS |
| 2026-05-04 | — | $719.65 | — | 69 | — | — | OPEN |

</details>

<details>
<summary>Recent Activity (last 5 entries)</summary>

| Time | Action | Price | FGI | Momentum | Velocity | Volatility | Reason |
|------|--------|-------|-----|----------|----------|------------|--------|
| 05-07 06:30 | NO_ACTION | $735.07 | 68.63 | 2.61 | 0.84 | 0.1017 | Holding position - indicators still favorable (1/8 days) |
| 05-06 06:33 | NO_ACTION | $728.12 | 67.26 | 2.08 | 0.16 | 0.1226 | BUY order submission failed after 3 attempts: {"buying_power":"607.24","code":40310000,"cost_basis":"50007.2","message":"insufficient buying power"} |
| 05-05 06:33 | NO_ACTION | $721.80 | 62.17 | -2.85 | -0.42 | 0.1228 | SELL order submission failed after 3 attempts: {"buying_power":"50156.64","code":40310000,"cost_basis":"51313.96","message":"insufficient buying power"} |
| 05-04 06:32 | BOUGHT | $719.65 | 66.11 | 0.67 | 0.76 | 0.1191 | Strong momentum/velocity, low volatility |
| 05-01 06:30 | NO_ACTION | $721.32 | 66.77 | 2.09 | -0.36 | 0.1197 | Insufficient momentum/velocity for entry |

</details>

---

## EOD (Afternoon) Strategy

| Metric | Value |
|--------|-------|
| Portfolio Value | **$25,091.38** |
| Buying Power | $50,182.76 |
| Current FGI | 67.71 |
| Position | FLAT |
| Total P&L | **$+9** |
| Win Rate | 33% (1W / 2L) |
| Total Round Trips | 3 |
| Last Signal | SOLD @ 2026-05-07 13:11 |

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
| 05-07 13:11 | SOLD | $731.90 | 67.71 | -0.14 | 1.48 | 0.1017 | Momentum reversal or high volatility |
| 05-06 13:10 | NO_ACTION | $733.86 | 68.54 | 2.17 | 0.65 | 0.1226 | Holding position - indicators still favorable (1/8 days) |
| 05-05 13:11 | BOUGHT | $723.86 | 67.31 | 1.59 | 0.28 | 0.1228 | Strong momentum/velocity, low volatility |
| 05-04 13:11 | SOLD | $717.38 | 63.26 | -2.18 | 0.01 | 0.1191 | Momentum reversal or high volatility |
| 05-01 13:11 | BOUGHT | $720.42 | 66.6 | 1.17 | 0.78 | 0.1197 | Strong momentum/velocity, low volatility |

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
