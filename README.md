# Fear & Greed Index Trading Bot

> Dashboard auto-updated daily at market close | Last update: **2026-08-07 13:30 PST**

![Portfolio Performance](assets/portfolio_chart.png)

---

## BOD (Morning) Strategy

| Metric | Value |
|--------|-------|
| Portfolio Value | **$25,420.25** |
| Buying Power | $72,078.03 |
| Current FGI | 58.71 |
| Position | FLAT |
| Total P&L | **$+995** |
| Win Rate | 50% (1W / 1L) |
| Total Round Trips | 2 |
| Last Signal | NO_ACTION @ 2026-08-07 09:36 |

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
| 08-07 09:36 | NO_ACTION | $771.71 | 58.71 | -0.76 | 2.65 | 0.55 | SELL incomplete - still holding -30 after 5 attempts |
| 08-06 09:35 | NO_ACTION | $769.78 | 59.77 | 2.94 | 4.99 | 0.55 | BUY did not fill after 3 attempts |
| 08-05 09:35 | NO_ACTION | $775.85 | 59.94 | 8.10 | 6.53 | 0.55 | SELL incomplete - still holding 34 after 5 attempts |
| 08-04 09:36 | NO_ACTION | $761.94 | 50.77 | 5.47 | 5.37 | 0.55 | BUY did not fill after 3 attempts |
| 08-03 09:35 | NO_ACTION | $750.80 | 44.8 | 4.87 | 2.32 | 0.55 | SELL incomplete - still holding 36 after 5 attempts |

</details>

---

## EOD (Afternoon) Strategy

| Metric | Value |
|--------|-------|
| Portfolio Value | **$26,073.85** |
| Buying Power | $104,295.40 |
| Current FGI | 63.74 |
| Position | FLAT |
| Total P&L | **$+302** |
| Win Rate | 50% (2W / 2L) |
| Total Round Trips | 4 |
| Last Signal | NO_ACTION @ 2026-08-07 15:52 |

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
| 08-07 15:52 | NO_ACTION | $772.84 | 63.74 | 2.48 | 1.67 | 0.55 | SELL incomplete - still holding 33 after 5 attempts |
| 08-06 15:50 | NO_ACTION | $767.96 | 60.0 | 0.41 | 4.70 | 0.55 | BUY did not fill after 3 attempts |
| 08-05 15:52 | NO_ACTION | $771.50 | 60.03 | 5.14 | 5.32 | 0.55 | SELL incomplete - still holding 33 after 5 attempts |
| 08-04 15:50 | NO_ACTION | $772.65 | 58.74 | 9.17 | 6.49 | 0.55 | BUY did not fill after 3 attempts |
| 08-03 15:52 | NO_ACTION | $758.15 | 45.91 | 2.83 | 3.30 | 0.55 | SELL incomplete - still holding 34 after 5 attempts |

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
