# Fear & Greed Index Trading Bot

> Dashboard auto-updated daily at market close | Last update: **2026-06-02 13:30 PST**

![Portfolio Performance](assets/portfolio_chart.png)

---

## BOD (Morning) Strategy

| Metric | Value |
|--------|-------|
| Portfolio Value | **$25,042.81** |
| Buying Power | $50,085.62 |
| Current FGI | 56.43 |
| Position | FLAT |
| Total P&L | **$+995** |
| Win Rate | 50% (1W / 1L) |
| Total Round Trips | 2 |
| Last Signal | NO_ACTION @ 2026-06-02 09:35 |

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
| 06-02 09:35 | NO_ACTION | $757.46 | 56.43 | -2.21 | -1.59 | 0.55 | Insufficient momentum/velocity for entry |
| 06-01 09:35 | NO_ACTION | $755.63 | 59.17 | -1.06 | -0.74 | 0.55 | Insufficient momentum/velocity for entry |
| 05-29 06:21 | SOLD | $755.68 | 60.31 | -0.66 | 0.58 | 0.101 | Momentum reversal or high volatility |
| 05-28 06:33 | NO_ACTION | $750.37 | 61.2 | 0.81 | 1.06 | 0.1011 | BUY order submission failed after 3 attempts |
| 05-27 06:33 | NO_ACTION | $750.90 | 61.4 | 2.07 | 0.19 | 0.1042 | BUY order submission failed after 3 attempts |

</details>

---

## EOD (Afternoon) Strategy

| Metric | Value |
|--------|-------|
| Portfolio Value | **$25,384.10** |
| Buying Power | $50,768.20 |
| Current FGI | 56.97 |
| Position | FLAT |
| Total P&L | **$+302** |
| Win Rate | 50% (2W / 2L) |
| Total Round Trips | 4 |
| Last Signal | NO_ACTION @ 2026-06-02 15:50 |

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
| 06-02 15:50 | NO_ACTION | $759.46 | 56.97 | -2.18 | -1.11 | 0.55 | Insufficient momentum/velocity for entry |
| 06-01 15:50 | NO_ACTION | $758.48 | 59.69 | -0.57 | -0.32 | 0.55 | Insufficient momentum/velocity for entry |
| 05-29 13:10 | NO_ACTION | $756.49 | 60.8 | 0.22 | 0.03 | 0.101 | Insufficient momentum/velocity for entry |
| 05-28 13:11 | SOLD | $755.22 | 60.29 | -0.26 | 0.38 | 0.1011 | Momentum reversal or high volatility |
| 05-27 13:11 | BOUGHT | $750.78 | 60.66 | 0.49 | 0.73 | 0.1042 | Strong momentum/velocity, low volatility |

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
