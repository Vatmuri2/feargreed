import pandas as pd
import datetime
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

LOG_BOD = 'trading_log_BOD.csv'
LOG_EOD = 'trading_log_EOD.csv'
README = 'README.md'
CHART_PATH = 'assets/portfolio_chart.png'


def load_log(path):
    if not os.path.exists(path) or os.path.getsize(path) < 10:
        return pd.DataFrame()
    df = pd.read_csv(path)
    if len(df) == 0:
        return df
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    return df


def generate_chart(bod_df, eod_df):
    os.makedirs('assets', exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
    fig.patch.set_facecolor('#0d1117')

    has_data = False

    # --- Portfolio Value Chart ---
    ax1.set_facecolor('#0d1117')
    for df, label, color in [(bod_df, 'BOD', '#58a6ff'), (eod_df, 'EOD', '#3fb950')]:
        if df.empty or 'Portfolio_Value' not in df.columns:
            continue
        has_data = True
        dates = df['Timestamp']
        values = df['Portfolio_Value']
        ax1.plot(dates, values, color=color, linewidth=1.8, label=label)

        # Mark buys and sells
        buys = df[df['Action'] == 'BOUGHT']
        sells = df[df['Action'] == 'SOLD']
        if not buys.empty:
            ax1.scatter(buys['Timestamp'], buys['Portfolio_Value'],
                        color=color, marker='^', s=60, zorder=5, alpha=0.9)
        if not sells.empty:
            ax1.scatter(sells['Timestamp'], sells['Portfolio_Value'],
                        color='#f85149', marker='v', s=60, zorder=5, alpha=0.9)

    if not has_data:
        plt.close(fig)
        return False

    ax1.set_title('Portfolio Value Over Time', color='white', fontsize=14, fontweight='bold', pad=12)
    ax1.set_ylabel('Portfolio Value ($)', color='white', fontsize=11)
    ax1.legend(loc='upper left', facecolor='#161b22', edgecolor='#30363d',
               labelcolor='white', fontsize=10)
    ax1.tick_params(colors='white')
    ax1.spines['bottom'].set_color('#30363d')
    ax1.spines['left'].set_color('#30363d')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(True, alpha=0.15, color='white')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))

    # --- FGI Chart ---
    ax2.set_facecolor('#0d1117')
    for df, label, color in [(bod_df, 'BOD', '#58a6ff'), (eod_df, 'EOD', '#3fb950')]:
        if df.empty or 'FGI_Value' not in df.columns:
            continue
        ax2.plot(df['Timestamp'], df['FGI_Value'], color=color, linewidth=1.2, alpha=0.8)

    # FGI zones
    ax2.axhspan(0, 25, alpha=0.08, color='#f85149')
    ax2.axhspan(75, 100, alpha=0.08, color='#3fb950')
    ax2.axhline(y=25, color='#f85149', linestyle='--', alpha=0.3, linewidth=0.8)
    ax2.axhline(y=50, color='#8b949e', linestyle='--', alpha=0.3, linewidth=0.8)
    ax2.axhline(y=75, color='#3fb950', linestyle='--', alpha=0.3, linewidth=0.8)

    ax2.set_title('Fear & Greed Index', color='white', fontsize=12, fontweight='bold', pad=8)
    ax2.set_ylabel('FGI', color='white', fontsize=11)
    ax2.set_ylim(0, 100)
    ax2.tick_params(colors='white')
    ax2.spines['bottom'].set_color('#30363d')
    ax2.spines['left'].set_color('#30363d')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(True, alpha=0.15, color='white')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))

    plt.tight_layout(pad=2.0)
    plt.savefig(CHART_PATH, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close(fig)
    print(f'Chart saved: {CHART_PATH}')
    return True


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


def render_md(stats_list, has_chart):
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M PST')
    lines = []
    lines.append('# Fear & Greed Index Trading Bot')
    lines.append('')
    lines.append(f'> Dashboard auto-updated daily at market close | Last update: **{now}**')
    lines.append('')

    if has_chart:
        lines.append('![Portfolio Performance](assets/portfolio_chart.png)')
        lines.append('')

    lines.append('---')
    lines.append('')

    for s in stats_list:
        if s is None:
            continue

        lines.append(f'## {s["label"]}')
        lines.append('')

        position = 'IN POSITION' if s['has_open'] else 'FLAT'

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
    lines.append('| BOD Execution | 6:20 AM PST |')
    lines.append('| EOD Execution | 1:10 PM PST |')
    lines.append('')

    return '\n'.join(lines)


bod_df = load_log(LOG_BOD)
eod_df = load_log(LOG_EOD)

bod_stats = calc_stats(bod_df, 'BOD (Morning) Strategy')
eod_stats = calc_stats(eod_df, 'EOD (Afternoon) Strategy')

has_chart = generate_chart(bod_df, eod_df)

md = render_md([bod_stats, eod_stats], has_chart)

with open(README, 'w') as f:
    f.write(md)

print(f'Dashboard generated: {README}')
