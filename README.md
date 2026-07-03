# Fear & Greed Index Trading Bot

> Dashboard auto-updated daily at market close | Last update: **2026-07-03 13:30 PST**

![Portfolio Performance](assets/portfolio_chart.png)

---

## BOD (Morning) Strategy

| Metric | Value |
|--------|-------|
| Portfolio Value | **$25,132.12** |
| Buying Power | $60,984.98 |
| Current FGI | 30.34 |
| Position | FLAT |
| Total P&L | **$+995** |
| Win Rate | 50% (1W / 1L) |
| Total Round Trips | 2 |
| Last Signal | NO_ACTION @ 2026-07-02 09:36 |

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
| 07-02 09:36 | NO_ACTION | $748.77 | 30.34 | 0.83 | 1.74 | 0.55 | BUY did not fill after 3 attempts |
| 07-01 09:38 | NO_ACTION | $742.63 | 30.97 | 3.20 | 2.22 | 0.55 | SELL incomplete - still holding 51 after 5 attempts |
| 06-30 09:36 | NO_ACTION | $741.14 | 27.23 | 1.68 | 0.38 | 0.55 | BUY did not fill after 3 attempts |
| 06-29 09:35 | NO_ACTION | $738.57 | 25.11 | -0.06 | -0.74 | 0.55 | Insufficient momentum/velocity for entry |
| 06-26 09:35 | NO_ACTION | $728.10 | 24.31 | -1.60 | -2.58 | 0.55 | Insufficient momentum/velocity for entry |

</details>

---

## EOD (Afternoon) Strategy

| Metric | Value |
|--------|-------|
| Portfolio Value | **$25,632.96** |
| Buying Power | $102,531.84 |
| Current FGI | 30.8 |
| Position | FLAT |
| Total P&L | **$+302** |
| Win Rate | 50% (2W / 2L) |
| Total Round Trips | 4 |
| Last Signal | NO_ACTION @ 2026-07-02 15:52 |

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
| 07-02 15:52 | NO_ACTION | $743.83 | 30.8 | -1.03 | 1.29 | 0.55 | SELL incomplete - still holding 34 after 5 attempts |
| 07-01 15:50 | NO_ACTION | $746.60 | 33.37 | 2.83 | 2.84 | 0.55 | BUY did not fill after 3 attempts |
| 06-30 15:52 | NO_ACTION | $746.88 | 31.31 | 3.61 | 2.03 | 0.55 | SELL incomplete - still holding 34 after 5 attempts |
| 06-29 15:50 | NO_ACTION | $740.70 | 26.94 | 1.26 | 0.61 | 0.55 | BUY did not fill after 3 attempts |
| 06-26 15:50 | NO_ACTION | $733.12 | 24.86 | -0.20 | -1.08 | 0.55 | Insufficient momentum/velocity for entry |

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
