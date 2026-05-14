# Fear & Greed Index Trading Bot

> Dashboard auto-updated daily at market close | Last update: **2026-05-14 13:30 PST**

![Portfolio Performance](assets/portfolio_chart.png)

---

## BOD (Morning) Strategy

| Metric | Value |
|--------|-------|
| Portfolio Value | **$24,937.95** |
| Buying Power | $2,281.66 |
| Current FGI | 64.37 |
| Position | FLAT |
| Total P&L | **$+995** |
| Win Rate | 50% (1W / 1L) |
| Total Round Trips | 2 |
| Last Signal | NO_ACTION @ 2026-05-14 06:30 |

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
| 05-14 06:30 | NO_ACTION | $743.72 | 64.37 | -0.90 | -0.85 | 0.0979 | SELL order submission failed after 3 attempts: {"code":40010001,"message":"qty must be \u003e 0"} |
| 05-13 06:30 | NO_ACTION | $738.30 | 65.54 | -0.58 | -0.41 | 0.099 | SELL order submission failed after 3 attempts: {"code":40010001,"message":"qty must be \u003e 0"} |
| 05-12 06:30 | NO_ACTION | $736.98 | 65.91 | -0.62 | -0.91 | 0.1025 | SELL order submission failed after 3 attempts: {"code":40010001,"message":"qty must be \u003e 0"} |
| 05-11 06:30 | NO_ACTION | $736.53 | 66.91 | -0.53 | -0.12 | 0.1045 | SELL order submission failed after 3 attempts: {"code":40010001,"message":"qty must be \u003e 0"} |
| 05-08 06:34 | SOLD | $735.05 | 66.77 | -0.78 | 1.53 | 0.1046 | Momentum reversal or high volatility |

</details>

---

## EOD (Afternoon) Strategy

| Metric | Value |
|--------|-------|
| Portfolio Value | **$25,091.36** |
| Buying Power | $50,182.72 |
| Current FGI | 66.17 |
| Position | FLAT |
| Total P&L | **$+9** |
| Win Rate | 33% (1W / 2L) |
| Total Round Trips | 3 |
| Last Signal | NO_ACTION @ 2026-05-14 13:10 |

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
| 05-14 13:10 | NO_ACTION | $748.17 | 66.17 | -0.06 | -0.39 | 0.0979 | Insufficient momentum/velocity for entry |
| 05-13 13:10 | NO_ACTION | $742.37 | 66.09 | -0.53 | -0.35 | 0.099 | Insufficient momentum/velocity for entry |
| 05-12 13:10 | NO_ACTION | $738.17 | 66.43 | -0.54 | -0.43 | 0.1025 | Insufficient momentum/velocity for entry |
| 05-11 13:10 | NO_ACTION | $739.26 | 67.34 | -0.06 | -0.40 | 0.1045 | Insufficient momentum/velocity for entry |
| 05-08 13:10 | NO_ACTION | $737.53 | 67.14 | -0.66 | -0.06 | 0.1046 | Insufficient momentum/velocity for entry |

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
