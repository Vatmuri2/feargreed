# Fear & Greed Index Trading Bot

> Dashboard auto-updated daily at market close | Last update: **2026-04-17 17:55 PST**

![Portfolio Performance](assets/portfolio_chart.png)

---

## BOD (Morning) Strategy

| Metric | Value |
|--------|-------|
| Portfolio Value | **$23,604.86** |
| Buying Power | $47,209.72 |
| Current FGI | 12.06 |
| Position | IN POSITION |
| Total P&L | **$-1,390** |
| Win Rate | 50% (3W / 3L) |
| Total Round Trips | 6 |
| Last Signal | BOUGHT @ 2025-11-24 06:40 |

<details>
<summary>Trade History (7 trades)</summary>

| Buy Date | Sell Date | Buy Price | Sell Price | Qty | P&L | Return | Result |
|----------|-----------|-----------|------------|-----|-----|--------|--------|
| 2025-09-18 | 2025-09-23 | $663.82 | $666.49 | 71 | $+190 | +0.40% | WIN |
| 2025-10-06 | 2025-10-08 | $670.82 | $669.71 | 71 | $-79 | -0.17% | LOSS |
| 2025-10-27 | 2025-10-31 | $682.83 | $684.12 | 69 | $+89 | +0.19% | WIN |
| 2025-11-03 | 2025-11-04 | $683.91 | $675.10 | 69 | $-608 | -1.29% | LOSS |
| 2025-11-10 | 2025-11-11 | $678.56 | $680.15 | 68 | $+108 | +0.23% | WIN |
| 2025-11-12 | 2025-11-14 | $682.60 | $666.57 | 68 | $-1,090 | -2.35% | LOSS |
| 2025-11-24 | — | $664.60 | — | 67 | — | — | OPEN |

</details>

<details>
<summary>Recent Activity (last 5 entries)</summary>

| Time | Action | Price | FGI | Momentum | Velocity | Volatility | Reason |
|------|--------|-------|-----|----------|----------|------------|--------|
| 11-24 06:40 | BOUGHT | $664.60 | 12.06 | 0.65 | 0.20 | 0.1431 | Strong momentum/velocity, low volatility |
| 11-21 06:40 | NO_ACTION | $655.22 | 7.3 | -3.91 | -1.77 | 0.1413 | Insufficient momentum/velocity for entry |
| 11-20 06:40 | NO_ACTION | $673.61 | 14.86 | 1.88 | -2.08 | 0.1336 | Insufficient momentum/velocity for entry |
| 11-19 06:40 | NO_ACTION | $661.87 | 11.47 | -3.60 | -1.87 | 0.1337 | Insufficient momentum/velocity for entry |
| 11-18 06:40 | NO_ACTION | $663.40 | 12.62 | -4.31 | -6.70 | 0.1307 | Insufficient momentum/velocity for entry |

</details>

---

## EOD (Afternoon) Strategy

| Metric | Value |
|--------|-------|
| Portfolio Value | **$24,495.54** |
| Buying Power | $3,368.80 |
| Current FGI | 18.74 |
| Position | IN POSITION |
| Total P&L | **$-1,319** |
| Win Rate | 33% (2W / 4L) |
| Total Round Trips | 6 |
| Last Signal | NO_ACTION @ 2025-11-26 12:50 |

<details>
<summary>Trade History (7 trades)</summary>

| Buy Date | Sell Date | Buy Price | Sell Price | Qty | P&L | Return | Result |
|----------|-----------|-----------|------------|-----|-----|--------|--------|
| 2025-09-18 | 2025-09-23 | $662.36 | $662.91 | 71 | $+39 | +0.08% | WIN |
| 2025-10-06 | 2025-10-07 | $671.71 | $669.27 | 70 | $-170 | -0.36% | LOSS |
| 2025-10-20 | 2025-10-22 | $671.93 | $667.71 | 70 | $-296 | -0.63% | LOSS |
| 2025-10-24 | 2025-10-30 | $677.64 | $680.53 | 68 | $+197 | +0.43% | WIN |
| 2025-11-03 | 2025-11-04 | $682.64 | $675.97 | 68 | $-454 | -0.98% | LOSS |
| 2025-11-10 | 2025-11-13 | $681.96 | $672.49 | 67 | $-634 | -1.39% | LOSS |
| 2025-11-24 | — | $668.99 | — | 67 | — | — | OPEN |

</details>

<details>
<summary>Recent Activity (last 5 entries)</summary>

| Time | Action | Price | FGI | Momentum | Velocity | Volatility | Reason |
|------|--------|-------|-----|----------|----------|------------|--------|
| 11-26 12:50 | NO_ACTION | $680.95 | 18.74 | 2.68 | 2.79 | 0.1512 | Holding position - indicators still favorable (2/8 days) |
| 11-25 12:50 | NO_ACTION | $675.65 | 15.54 | 2.26 | 2.53 | 0.147 | Holding position - indicators still favorable (1/8 days) |
| 11-24 12:50 | BOUGHT | $668.99 | 13.91 | 3.17 | 0.66 | 0.1431 | Strong momentum/velocity, low volatility |
| 11-21 12:50 | NO_ACTION | $660.44 | 10.38 | 0.30 | -0.41 | 0.1413 | Insufficient momentum/velocity for entry |
| 11-20 12:50 | NO_ACTION | $652.81 | 7.94 | -2.56 | -1.88 | 0.1336 | Insufficient momentum/velocity for entry |

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
