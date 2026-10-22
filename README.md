# Fear & Greed Index Trading Bot

> Dashboard auto-updated daily at market close | Last update: **2026-10-22 13:30 PST**

![Portfolio Performance](assets/portfolio_chart.png)

---

## BOD (Morning) Strategy

| Metric | Value |
|--------|-------|
| Portfolio Value | **$24,855.38** |
| Buying Power | $40,532.82 |
| Current FGI | 27.52 |
| Position | FLAT |
| Total P&L | **$+995** |
| Win Rate | 50% (1W / 1L) |
| Total Round Trips | 2 |
| Last Signal | NO_ACTION @ 2026-10-22 09:36 |

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
| 10-22 09:36 | NO_ACTION | $767.18 | 27.52 | 1.15 | 0.59 | 0.0926 | BUY did not fill after 3 attempts |
| 10-21 09:35 | NO_ACTION | $770.34 | 30.34 | 4.55 | -0.12 | 0.0864 | Insufficient momentum/velocity for entry |
| 10-20 09:35 | NO_ACTION | $765.42 | 21.26 | -4.64 | -1.72 | 0.0891 | Insufficient momentum/velocity for entry |
| 10-19 09:35 | NO_ACTION | $767.38 | 25.76 | -1.87 | -6.18 | 0.0898 | Insufficient momentum/velocity for entry |
| 10-16 09:35 | NO_ACTION | $772.30 | 30.69 | -3.11 | -5.14 | 0.0963 | Insufficient momentum/velocity for entry |

</details>

---

## EOD (Afternoon) Strategy

| Metric | Value |
|--------|-------|
| Portfolio Value | **$25,226.50** |
| Buying Power | $100,906.00 |
| Current FGI | 18.8 |
| Position | IN POSITION |
| Total P&L | **$+302** |
| Win Rate | 50% (2W / 2L) |
| Total Round Trips | 4 |
| Last Signal | NO_ACTION @ 2026-10-22 15:50 |

<details>
<summary>Trade History (5 trades)</summary>

| Buy Date | Sell Date | Buy Price | Sell Price | Qty | P&L | Return | Result |
|----------|-----------|-----------|------------|-----|-----|--------|--------|
| 2026-04-20 | 2026-04-21 | $708.76 | $704.15 | 70 | $-323 | -0.65% | LOSS |
| 2026-05-01 | 2026-05-04 | $720.42 | $717.38 | 68 | $-207 | -0.42% | LOSS |
| 2026-05-05 | 2026-05-07 | $723.86 | $731.90 | 67 | $+539 | +1.11% | WIN |
| 2026-05-27 | 2026-05-28 | $750.78 | $755.22 | 66 | $+293 | +0.59% | WIN |
| 2026-10-21 | — | unknown (late/unlogged fill) | — | 66 | — | — | OPEN |

</details>

<details>
<summary>Recent Activity (last 5 entries)</summary>

| Time | Action | Price | FGI | Momentum | Velocity | Volatility | Reason |
|------|--------|-------|-----|----------|----------|------------|--------|
| 10-22 15:50 | NO_ACTION | $762.33 | 18.8 | -6.73 | -2.20 | 0.1022 | SELL incomplete - still holding 66 after 5 attempts |
| 10-21 15:51 | NO_ACTION | $773.23 | 31.49 | 3.76 | 4.56 | 0.0887 | BUY did not fill after 3 attempts |
| 10-20 15:50 | NO_ACTION | $768.18 | 26.29 | 3.12 | -1.99 | 0.0897 | Insufficient momentum/velocity for entry |
| 10-19 15:50 | NO_ACTION | $767.30 | 25.4 | 0.24 | -2.19 | 0.0905 | Insufficient momentum/velocity for entry |
| 10-16 15:50 | NO_ACTION | $762.83 | 17.82 | -9.53 | -9.16 | 0.106 | Insufficient momentum/velocity for entry |

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
