# Fear & Greed Index Trading Bot

> Dashboard auto-updated daily at market close | Last update: **2026-09-08 13:30 PST**

![Portfolio Performance](assets/portfolio_chart.png)

---

## BOD (Morning) Strategy

| Metric | Value |
|--------|-------|
| Portfolio Value | **$25,322.10** |
| Buying Power | $101,288.40 |
| Current FGI | 42.43 |
| Position | FLAT |
| Total P&L | **$+995** |
| Win Rate | 50% (1W / 1L) |
| Total Round Trips | 2 |
| Last Signal | NO_ACTION @ 2026-09-08 09:36 |

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
| 09-08 09:36 | NO_ACTION | $768.76 | 42.43 | 2.68 | 3.86 | 0.0816 | BUY did not fill after 3 attempts |
| 09-04 09:35 | NO_ACTION | $772.65 | 44.17 | 8.27 | -1.11 | 0.0803 | Insufficient momentum/velocity for entry |
| 09-03 09:35 | NO_ACTION | $768.75 | 32.66 | -4.35 | -7.26 | 0.0761 | Insufficient momentum/velocity for entry |
| 09-02 09:35 | NO_ACTION | $762.27 | 30.86 | -13.41 | -8.18 | 0.0718 | Insufficient momentum/velocity for entry |
| 09-01 09:35 | NO_ACTION | $761.95 | 47.51 | -4.94 | -2.16 | 0.0713 | Insufficient momentum/velocity for entry |

</details>

---

## EOD (Afternoon) Strategy

| Metric | Value |
|--------|-------|
| Portfolio Value | **$25,857.58** |
| Buying Power | $42,743.32 |
| Current FGI | 41.23 |
| Position | FLAT |
| Total P&L | **$+302** |
| Win Rate | 50% (2W / 2L) |
| Total Round Trips | 4 |
| Last Signal | NO_ACTION @ 2026-09-08 15:51 |

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
| 09-08 15:51 | NO_ACTION | $766.33 | 41.23 | 1.77 | 2.49 | 0.083 | BUY did not fill after 3 attempts |
| 09-04 15:50 | NO_ACTION | $770.33 | 41.8 | 4.83 | -0.81 | 0.0814 | Insufficient momentum/velocity for entry |
| 09-03 15:50 | NO_ACTION | $773.35 | 35.34 | -2.44 | -4.80 | 0.0832 | Insufficient momentum/velocity for entry |
| 09-02 15:50 | NO_ACTION | $765.32 | 33.77 | -8.81 | -6.83 | 0.0742 | Insufficient momentum/velocity for entry |
| 09-01 15:50 | NO_ACTION | $761.07 | 44.23 | -5.18 | -4.64 | 0.0723 | Insufficient momentum/velocity for entry |

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
