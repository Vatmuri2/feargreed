"""
update_readme.py — Regenerates README.md dashboard from live data.
Run manually or via cron after market close:
    python3 update_readme.py
"""

import os
import datetime
import pandas as pd
import pytz
import fear_and_greed as fg
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import OrderSide
from alpaca.common.exceptions import APIError

BOD_KEY    = os.environ['ALPACA_BOD_API_KEY']
BOD_SECRET = os.environ['ALPACA_BOD_API_SECRET']
EOD_KEY    = os.environ['ALPACA_EOD_API_KEY']
EOD_SECRET = os.environ['ALPACA_EOD_API_SECRET']

BOD_LOG = 'trading_log_BOD.csv'
EOD_LOG = 'trading_log_EOD.csv'
SYMBOL  = 'SPY'


def get_account_info(key, secret):
    c = TradingClient(key, secret, paper=True)
    a = c.get_account()
    try:
        p = c.get_open_position(SYMBOL)
        has_position = True
        position_qty = int(float(p.qty))
    except APIError:
        has_position = False
        position_qty = 0
    return {
        'portfolio_value': float(a.equity),
        'buying_power':    float(a.buying_power),
        'has_position':    has_position,
        'position_qty':    position_qty,
    }


def parse_log(log_file):
    try:
        df = pd.read_csv(log_file)
        if df.empty:
            return pd.DataFrame()
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        return df
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def compute_trade_history(df):
    """Match BOUGHT/SOLD pairs into round trips."""
    if df.empty:
        return [], 0, 0, 0.0

    buys  = df[df['Action'] == 'BOUGHT'].sort_values('Timestamp')
    sells = df[df['Action'] == 'SOLD'].sort_values('Timestamp')

    trades = []
    used_sells = set()

    for _, buy in buys.iterrows():
        match = None
        for sidx, sell in sells.iterrows():
            if sidx not in used_sells and sell['Timestamp'] > buy['Timestamp']:
                match = sell
                used_sells.add(sidx)
                break

        buy_price = float(buy['Price'])
        qty       = int(float(buy['Quantity']))

        if match is not None:
            sell_price = float(match['Price'])
            pnl        = (sell_price - buy_price) * qty
            ret        = (sell_price - buy_price) / buy_price * 100
            trades.append({
                'buy_date':   buy['Timestamp'].strftime('%Y-%m-%d'),
                'sell_date':  match['Timestamp'].strftime('%Y-%m-%d'),
                'buy_price':  buy_price,
                'sell_price': sell_price,
                'qty':        qty,
                'pnl':        pnl,
                'ret':        ret,
                'result':     'WIN' if pnl > 0 else 'LOSS',
            })
        else:
            trades.append({
                'buy_date':   buy['Timestamp'].strftime('%Y-%m-%d'),
                'sell_date':  '—',
                'buy_price':  buy_price,
                'sell_price': None,
                'qty':        qty,
                'pnl':        None,
                'ret':        None,
                'result':     'OPEN',
            })

    closed    = [t for t in trades if t['result'] != 'OPEN']
    wins      = sum(1 for t in closed if t['result'] == 'WIN')
    losses    = sum(1 for t in closed if t['result'] == 'LOSS')
    total_pnl = sum(t['pnl'] for t in closed)
    return trades, wins, losses, total_pnl


def fmt_trade_history(trades):
    if not trades:
        return "_No trades yet._"
    rows = [
        "| Buy Date | Sell Date | Buy Price | Sell Price | Qty | P&L | Return | Result |",
        "|----------|-----------|-----------|------------|-----|-----|--------|--------|",
    ]
    for t in trades:
        sell_price = f"${t['sell_price']:.2f}" if t['sell_price'] else "—"
        pnl        = f"${t['pnl']:+,.0f}"       if t['pnl'] is not None else "—"
        ret        = f"{t['ret']:+.2f}%"         if t['ret'] is not None else "—"
        rows.append(
            f"| {t['buy_date']} | {t['sell_date']} | ${t['buy_price']:.2f} | {sell_price} | {t['qty']} | {pnl} | {ret} | {t['result']} |"
        )
    return "\n".join(rows)


def fmt_recent_activity(df, n=5):
    if df.empty:
        return "_No activity yet._"
    recent = df.sort_values('Timestamp', ascending=False).head(n)
    rows = [
        "| Time | Action | Price | FGI | Momentum | Velocity | Volatility | Reason |",
        "|------|--------|-------|-----|----------|----------|------------|--------|",
    ]
    for _, row in recent.iterrows():
        t = row['Timestamp'].strftime('%m-%d %H:%M')
        rows.append(
            f"| {t} | {row['Action']} | ${float(row['Price']):.2f} | {row['FGI_Value']} "
            f"| {float(row['FGI_Momentum']):.2f} | {float(row['FGI_Velocity']):.2f} "
            f"| {float(row['Volatility']):.4f} | {row['Signal_Reason']} |"
        )
    return "\n".join(rows)


def get_last_signal(df):
    if df.empty:
        return "No activity"
    last = df.sort_values('Timestamp', ascending=False).iloc[0]
    return f"{last['Action']} @ {last['Timestamp'].strftime('%Y-%m-%d %H:%M')}"


def build_readme(bod_acct, eod_acct, bod_log, eod_log, current_fgi):
    bod_trades, bod_wins, bod_losses, bod_pnl = compute_trade_history(bod_log)
    eod_trades, eod_wins, eod_losses, eod_pnl = compute_trade_history(eod_log)

    def winrate(wins, losses):
        total = wins + losses
        return f"{int(wins/total*100)}% ({wins}W / {losses}L)" if total > 0 else "N/A"

    def pnl_str(pnl, wins, losses):
        return f"**${pnl:+,.0f}**" if (wins + losses) > 0 else "**N/A**"

    now_pst = datetime.datetime.now(pytz.timezone('US/Pacific')).strftime('%Y-%m-%d %H:%M PST')
    fgi_str = f"{current_fgi:.2f}" if current_fgi is not None else "N/A"

    return f"""# Fear & Greed Index Trading Bot

> Dashboard auto-updated daily at market close | Last update: **{now_pst}**

![Portfolio Performance](assets/portfolio_chart.png)

---

## BOD (Morning) Strategy

| Metric | Value |
|--------|-------|
| Portfolio Value | **${bod_acct['portfolio_value']:,.2f}** |
| Buying Power | ${bod_acct['buying_power']:,.2f} |
| Current FGI | {fgi_str} |
| Position | {'IN POSITION' if bod_acct['has_position'] else 'NO POSITION'} |
| Total P&L | {pnl_str(bod_pnl, bod_wins, bod_losses)} |
| Win Rate | {winrate(bod_wins, bod_losses)} |
| Total Round Trips | {bod_wins + bod_losses} |
| Last Signal | {get_last_signal(bod_log)} |

<details>
<summary>Trade History ({len(bod_trades)} trades)</summary>

{fmt_trade_history(bod_trades)}

</details>

<details>
<summary>Recent Activity (last 5 entries)</summary>

{fmt_recent_activity(bod_log)}

</details>

---

## EOD (Afternoon) Strategy

| Metric | Value |
|--------|-------|
| Portfolio Value | **${eod_acct['portfolio_value']:,.2f}** |
| Buying Power | ${eod_acct['buying_power']:,.2f} |
| Current FGI | {fgi_str} |
| Position | {'IN POSITION' if eod_acct['has_position'] else 'NO POSITION'} |
| Total P&L | {pnl_str(eod_pnl, eod_wins, eod_losses)} |
| Win Rate | {winrate(eod_wins, eod_losses)} |
| Total Round Trips | {eod_wins + eod_losses} |
| Last Signal | {get_last_signal(eod_log)} |

<details>
<summary>Trade History ({len(eod_trades)} trades)</summary>

{fmt_trade_history(eod_trades)}

</details>

<details>
<summary>Recent Activity (last 5 entries)</summary>

{fmt_recent_activity(eod_log)}

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
"""


if __name__ == '__main__':
    print("Fetching account data...")
    bod_acct = get_account_info(BOD_KEY, BOD_SECRET)
    eod_acct = get_account_info(EOD_KEY, EOD_SECRET)

    print("Reading trade logs...")
    bod_log = parse_log(BOD_LOG)
    eod_log = parse_log(EOD_LOG)

    print("Fetching FGI...")
    try:
        current_fgi = round(fg.get().value, 2)
    except Exception as e:
        print(f"Warning: could not fetch FGI: {e}")
        current_fgi = None

    print("Writing README.md...")
    with open('README.md', 'w') as f:
        f.write(build_readme(bod_acct, eod_acct, bod_log, eod_log, current_fgi))

    print("Done.")
