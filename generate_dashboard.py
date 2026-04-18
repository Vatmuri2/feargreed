import pandas as pd
import datetime
import os

LOG_BOD = 'trading_log_BOD.csv'
LOG_EOD = 'trading_log_EOD.csv'
README = 'README.md'


def load_log(path):
    if not os.path.exists(path) or os.path.getsize(path) < 10:
        return pd.DataFrame()
    df = pd.read_csv(path)
    if len(df) == 0:
        return df
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    return df


def calc_stats(df, label):
    if df.empty:
        return None

    latest = df.iloc[-1]
    portfolio = latest.get('Portfolio_Value', 0)
    buying_power = latest.get('Buying_Power', 0)
    fgi = latest.get('FGI_Value', 'N/A')
    last_action = latest.get('Action', 'N/A')
    last_date = latest['Timestamp'].strftime('%Y-%m-%d %H:%M')

    trades = df[df['Action'].isin(['BOUGHT', 'SOLD'])]
    buys = trades[trades['Action'] == 'BOUGHT']
    sells = trades[trades['Action'] == 'SOLD']

    wins = 0
    losses = 0
    total_pnl = 0
    trade_rows = []

    buy_list = buys.reset_index(drop=True)
    sell_list = sells.reset_index(drop=True)
    n_pairs = min(len(buy_list), len(sell_list))

    for i in range(n_pairs):
        b = buy_list.iloc[i]
        s = sell_list.iloc[i]
        pnl = (s['Price'] - b['Price']) * b['Quantity']
        pct = ((s['Price'] - b['Price']) / b['Price']) * 100
        total_pnl += pnl
        if pnl > 0:
            wins += 1
        else:
            losses += 1
        trade_rows.append({
            'Buy Date': b['Timestamp'].strftime('%Y-%m-%d'),
            'Sell Date': s['Timestamp'].strftime('%Y-%m-%d'),
            'Buy Price': f"${b['Price']:.2f}",
            'Sell Price': f"${s['Price']:.2f}",
            'Qty': int(b['Quantity']),
            'PnL': f"${pnl:+,.0f}",
            'Return': f"{pct:+.2f}%",
            'Result': 'WIN' if pnl > 0 else 'LOSS'
        })

    has_open = len(buy_list) > len(sell_list)
    if has_open:
        ob = buy_list.iloc[-1]
        trade_rows.append({
            'Buy Date': ob['Timestamp'].strftime('%Y-%m-%d'),
            'Sell Date': '—',
            'Buy Price': f"${ob['Price']:.2f}",
            'Sell Price': '—',
            'Qty': int(ob['Quantity']),
            'PnL': '—',
            'Return': '—',
            'Result': 'OPEN'
        })

    win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0

    return {
        'label': label,
        'portfolio': portfolio,
        'buying_power': buying_power,
        'fgi': fgi,
        'last_action': last_action,
        'last_date': last_date,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'total_trades': wins + losses,
        'has_open': has_open,
        'trade_rows': trade_rows,
        'recent': df.tail(5).iloc[::-1]
    }


def render_md(stats_list):
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M PST')
    lines = []
    lines.append('# Fear & Greed Index Trading Bot')
    lines.append('')
    lines.append(f'> Dashboard auto-updated daily at market close | Last update: **{now}**')
    lines.append('')
    lines.append('---')
    lines.append('')

    for s in stats_list:
        if s is None:
            continue

        lines.append(f'## {s["label"]}')
        lines.append('')

        position = 'IN POSITION' if s['has_open'] else 'FLAT'
        pnl_sign = '+' if s['total_pnl'] >= 0 else ''

        lines.append('| Metric | Value |')
        lines.append('|--------|-------|')
        lines.append(f'| Portfolio Value | **${s["portfolio"]:,.2f}** |')
        lines.append(f'| Buying Power | ${s["buying_power"]:,.2f} |')
        lines.append(f'| Current FGI | {s["fgi"]} |')
        lines.append(f'| Position | {position} |')
        lines.append(f'| Total P&L | **${s["total_pnl"]:+,.0f}** |')
        lines.append(f'| Win Rate | {s["win_rate"]:.0f}% ({s["wins"]}W / {s["losses"]}L) |')
        lines.append(f'| Total Round Trips | {s["total_trades"]} |')
        lines.append(f'| Last Signal | {s["last_action"]} @ {s["last_date"]} |')
        lines.append('')

        if s['trade_rows']:
            lines.append('<details>')
            lines.append(f'<summary>Trade History ({len(s["trade_rows"])} trades)</summary>')
            lines.append('')
            lines.append('| Buy Date | Sell Date | Buy Price | Sell Price | Qty | P&L | Return | Result |')
            lines.append('|----------|-----------|-----------|------------|-----|-----|--------|--------|')
            for t in s['trade_rows']:
                lines.append(f'| {t["Buy Date"]} | {t["Sell Date"]} | {t["Buy Price"]} | {t["Sell Price"]} | {t["Qty"]} | {t["PnL"]} | {t["Return"]} | {t["Result"]} |')
            lines.append('')
            lines.append('</details>')
            lines.append('')

        lines.append('<details>')
        lines.append('<summary>Recent Activity (last 5 entries)</summary>')
        lines.append('')
        lines.append('| Time | Action | Price | FGI | Momentum | Velocity | Volatility | Reason |')
        lines.append('|------|--------|-------|-----|----------|----------|------------|--------|')
        for _, r in s['recent'].iterrows():
            vol = r.get('Volatility', 'N/A')
            lines.append(f'| {r["Timestamp"].strftime("%m-%d %H:%M")} | {r["Action"]} | ${r["Price"]:.2f} | {r["FGI_Value"]} | {r["FGI_Momentum"]:.2f} | {r["FGI_Velocity"]:.2f} | {vol} | {r["Signal_Reason"]} |')
        lines.append('')
        lines.append('</details>')
        lines.append('')
        lines.append('---')
        lines.append('')

    lines.append('## Strategy')
    lines.append('')
    lines.append('Momentum-based strategy using CNN Fear & Greed Index to trade SPY.')
    lines.append('')
    lines.append('| Parameter | Value |')
    lines.append('|-----------|-------|')
    lines.append('| Momentum Threshold | 0.2 |')
    lines.append('| Velocity Threshold | 0.15 |')
    lines.append('| Volatility Buy Limit | 0.6 |')
    lines.append('| Volatility Sell Limit | 0.5 |')
    lines.append('| Max Days Held | 8 |')
    lines.append('| Lookback Days | 3 |')
    lines.append('| BOD Execution | 6:40 AM PST |')
    lines.append('| EOD Execution | 1:10 PM PST |')
    lines.append('')

    return '\n'.join(lines)


bod_stats = calc_stats(load_log(LOG_BOD), 'BOD (Morning) Strategy')
eod_stats = calc_stats(load_log(LOG_EOD), 'EOD (Afternoon) Strategy')

md = render_md([bod_stats, eod_stats])

with open(README, 'w') as f:
    f.write(md)

print(f'Dashboard generated: {README}')
