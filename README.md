# Fear & Greed Index Trading Bot

> Dashboard auto-updated daily at market close | Last update: **2026-07-08 13:30 PST**

![Portfolio Performance](assets/portfolio_chart.png)

---

## BOD (Morning) Strategy

| Metric | Value |
|--------|-------|
| Portfolio Value | **$24,880.22** |
| Buying Power | $70,092.93 |
| Current FGI | 43.71 |
| Position | FLAT |
| Total P&L | **$+995** |
| Win Rate | 50% (1W / 1L) |
| Total Round Trips | 2 |
| Last Signal | NO_ACTION @ 2026-07-08 09:35 |

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
| 07-08 09:35 | NO_ACTION | $743.50 | 43.71 | 1.87 | 4.46 | 0.55 | SELL incomplete - still holding 33 after 5 attempts |
| 07-07 09:37 | NO_ACTION | $750.05 | 44.29 | 6.91 | 4.44 | 0.55 | BUY did not fill after 3 attempts |
| 07-06 09:35 | NO_ACTION | $749.00 | 37.51 | 4.57 | 3.43 | 0.55 | SELL incomplete - still holding 44 after 5 attempts |
| 07-02 09:36 | NO_ACTION | $748.77 | 30.34 | 0.83 | 1.74 | 0.55 | BUY did not fill after 3 attempts |
| 07-01 09:38 | NO_ACTION | $742.63 | 30.97 | 3.20 | 2.22 | 0.55 | SELL incomplete - still holding 51 after 5 attempts |

</details>

---

## EOD (Afternoon) Strategy

| Metric | Value |
|--------|-------|
| Portfolio Value | **$25,490.79** |
| Buying Power | $101,963.16 |
| Current FGI | 41.86 |
| Position | FLAT |
| Total P&L | **$+302** |
| Win Rate | 50% (2W / 2L) |
| Total Round Trips | 4 |
| Last Signal | NO_ACTION @ 2026-07-08 15:50 |

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
| 07-08 15:50 | NO_ACTION | $744.75 | 41.86 | -1.40 | 3.69 | 0.55 | Insufficient momentum/velocity for entry |
| 07-07 15:52 | NO_ACTION | $747.28 | 43.0 | 3.43 | 3.21 | 0.55 | SELL incomplete - still holding 33 after 5 attempts |
| 07-06 15:50 | NO_ACTION | $751.47 | 44.91 | 8.55 | 4.53 | 0.55 | BUY did not fill after 3 attempts |
| 07-02 15:52 | NO_ACTION | $743.83 | 30.8 | -1.03 | 1.29 | 0.55 | SELL incomplete - still holding 34 after 5 attempts |
| 07-01 15:50 | NO_ACTION | $746.60 | 33.37 | 2.83 | 2.84 | 0.55 | BUY did not fill after 3 attempts |

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
