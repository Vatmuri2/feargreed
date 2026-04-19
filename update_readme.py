"""
update_readme.py — Regenerates README.md dashboard from live data.
Run manually or via cron after market close:
    python3 update_readme.py
"""

import os
import subprocess
import datetime
import pandas as pd
import pytz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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


PORTFOLIO_HISTORY = 'portfolio_history.csv'

def snapshot_portfolio(bod_acct, eod_acct):
    """Append today's portfolio values to portfolio_history.csv regardless of trading activity."""
    today = datetime.date.today().isoformat()
    try:
        hist = pd.read_csv(PORTFOLIO_HISTORY)
        if today in hist['Date'].values:
            print("Portfolio snapshot already recorded today — skipping.")
            return
    except (FileNotFoundError, pd.errors.EmptyDataError):
        hist = pd.DataFrame(columns=['Date', 'BOD_Value', 'EOD_Value'])

    new_row = pd.DataFrame([{
        'Date': today,
        'BOD_Value': bod_acct['portfolio_value'],
        'EOD_Value': eod_acct['portfolio_value'],
    }])
    hist = pd.concat([hist, new_row], ignore_index=True)
    hist.to_csv(PORTFOLIO_HISTORY, index=False)
    print(f"Portfolio snapshot saved: BOD=${bod_acct['portfolio_value']:,.2f}  EOD=${eod_acct['portfolio_value']:,.2f}")


def generate_chart(bod_log, eod_log, bod_acct, eod_acct):
    """Generate portfolio performance chart using daily portfolio_history.csv snapshots."""
    os.makedirs('assets', exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    try:
        hist = pd.read_csv(PORTFOLIO_HISTORY, parse_dates=['Date'])
        hist = hist.sort_values('Date')

        if len(hist) >= 2:
            ax.plot(hist['Date'], hist['BOD_Value'], label='BOD Strategy', color='#2196F3', linewidth=2)
            ax.plot(hist['Date'], hist['EOD_Value'], label='EOD Strategy', color='#4CAF50', linewidth=2)
            ax.scatter(hist['Date'].iloc[-1], hist['BOD_Value'].iloc[-1], color='#2196F3', s=60, zorder=5)
            ax.scatter(hist['Date'].iloc[-1], hist['EOD_Value'].iloc[-1], color='#4CAF50', s=60, zorder=5)
        else:
            # Only one data point so far — show dots
            if not hist.empty:
                ax.scatter(hist['Date'], hist['BOD_Value'], label='BOD Strategy', color='#2196F3', s=80, zorder=5, marker='D')
                ax.scatter(hist['Date'], hist['EOD_Value'], label='EOD Strategy', color='#4CAF50', s=80, zorder=5, marker='D')
            else:
                now = datetime.datetime.now()
                ax.scatter([now], [bod_acct['portfolio_value']], label='BOD Strategy (no data yet)', color='#2196F3', s=80, zorder=5, marker='D')
                ax.scatter([now], [eod_acct['portfolio_value']], label='EOD Strategy (no data yet)', color='#4CAF50', s=80, zorder=5, marker='D')
    except (FileNotFoundError, pd.errors.EmptyDataError):
        now = datetime.datetime.now()
        ax.scatter([now], [bod_acct['portfolio_value']], label='BOD Strategy (no data yet)', color='#2196F3', s=80, zorder=5, marker='D')
        ax.scatter([now], [eod_acct['portfolio_value']], label='EOD Strategy (no data yet)', color='#4CAF50', s=80, zorder=5, marker='D')

    ax.axhline(y=25000, color='#555', linestyle='--', linewidth=1, label='Starting Capital ($25,000)')

    ax.set_title('Portfolio Performance — Live Forward Test', color='white', fontsize=14, pad=12)
    ax.set_xlabel('Date', color='#aaa')
    ax.set_ylabel('Portfolio Value ($)', color='#aaa')
    ax.tick_params(colors='#aaa')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    fig.autofmt_xdate()
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.legend(facecolor='#1a1a2e', edgecolor='#333', labelcolor='white')
    ax.grid(True, color='#222', linewidth=0.5)

    plt.tight_layout()
    plt.savefig('assets/portfolio_chart.png', dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print("Chart saved to assets/portfolio_chart.png")


def git_push():
    """Commit and push README and chart to GitHub."""
    try:
        subprocess.run(['git', 'add', 'README.md', 'assets/portfolio_chart.png', 'portfolio_history.csv'], check=True)
        result = subprocess.run(['git', 'diff', '--cached', '--quiet'])
        if result.returncode == 0:
            print("Nothing changed — skipping commit.")
            return
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
        subprocess.run(['git', 'commit', '-m', f'Dashboard update {now}'], check=True)
        subprocess.run(['git', 'push'], check=True)
        print("Pushed to GitHub.")
    except subprocess.CalledProcessError as e:
        print(f"Git error: {e}")


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

    print("Snapshotting portfolio...")
    snapshot_portfolio(bod_acct, eod_acct)

    print("Generating chart...")
    generate_chart(bod_log, eod_log, bod_acct, eod_acct)

    print("Writing README.md...")
    with open('README.md', 'w') as f:
        f.write(build_readme(bod_acct, eod_acct, bod_log, eod_log, current_fgi))

    print("Pushing to GitHub...")
    git_push()

    print("Done.")
