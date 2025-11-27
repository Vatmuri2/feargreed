import pandas as pd
import numpy as np
import yfinance as yf

TICKER = 'QQQ'
TOTAL_CAPITAL = 10000

# Load data
fg_file = "datasets/fear_greed_combined_2011_2025.csv"
fg_df = pd.read_csv(fg_file, parse_dates=['Date'])
fg_df.set_index('Date', inplace=True)
fg_df = fg_df[~fg_df.index.duplicated(keep='first')]
fg_df['fear_greed'] = fg_df['fear_greed'].ffill()

start_date = fg_df.index.min()
end_date = fg_df.index.max()
data = yf.download(TICKER, start=start_date, end=end_date, auto_adjust=True)
vix_data = yf.download('^VIX', start=start_date, end=end_date, auto_adjust=True)

price = data['Close'].squeeze()
price = price[~price.index.duplicated(keep='first')]
vix = vix_data['Close'].squeeze()
vix = vix[~vix.index.duplicated(keep='first')]

common_idx = fg_df.index.intersection(price.index).intersection(vix.index)
df = pd.DataFrame({
    'FG': fg_df.loc[common_idx, 'fear_greed'],
    'Price': price.loc[common_idx],
    'VIX': vix.loc[common_idx]
}).sort_index().ffill().dropna()

df['MA200'] = df['Price'].rolling(200).mean()
df['MA50'] = df['Price'].rolling(50).mean()
df['above_MA200'] = df['Price'] > df['MA200']
df['above_MA50'] = df['Price'] > df['MA50']
df = df.dropna()

print("="*70)
print("BUY-THE-DIP STRATEGY BACKTEST")
print("="*70)
print("""
Strategy Rules:
- Initial entry: 35% when FG < 10, above MA200
- After 1 week: If price is LOWER than entry, deploy remaining 65%
- After 1 week: If price is HIGHER, deploy 35% more (total 70%)
- No stop loss — hold through volatility
- Exit: FG > 70 (or hold to 1 year max)
""")

# Find entry points
entry_mask = (df['FG'] < 10) & (df['above_MA200']) & (~df['above_MA50'])
entry_dates = df.index[entry_mask]

# Group consecutive days into single entry events
entry_events = []
prev_date = None
for date in entry_dates:
    if prev_date is None or (date - prev_date).days > 10:
        entry_events.append(date)
    prev_date = date

print(f"\nFound {len(entry_events)} distinct entry events")

results = []

for entry_date in entry_events:
    entry_loc = df.index.get_loc(entry_date)
    
    if entry_loc + 252 >= len(df):
        continue
    
    entry_price = df['Price'].iloc[entry_loc]
    
    # INITIAL ENTRY: 35%
    cash = TOTAL_CAPITAL
    shares = 0
    
    buy_amount = TOTAL_CAPITAL * 0.35
    shares += buy_amount / entry_price
    cash -= buy_amount
    
    trades = [f"Day 0: Buy 35% at ${entry_price:.2f}"]
    
    # AFTER 1 WEEK (5 trading days)
    week_price = df['Price'].iloc[entry_loc + 5]
    price_change_1w = (week_price - entry_price) / entry_price * 100
    
    if week_price < entry_price:
        # Price dropped — deploy ALL remaining (65%)
        add_amount = cash  # All remaining
        shares += add_amount / week_price
        cash = 0
        trades.append(f"Day 5: Price dropped {price_change_1w:.1f}% → BUY remaining 65% at ${week_price:.2f}")
        strategy_type = "full_deploy_on_dip"
    else:
        # Price went up — deploy another 35% (total 70%)
        add_amount = min(cash, TOTAL_CAPITAL * 0.35)
        shares += add_amount / week_price
        cash -= add_amount
        trades.append(f"Day 5: Price up {price_change_1w:.1f}% → BUY 35% more at ${week_price:.2f}")
        strategy_type = "partial_deploy_on_rise"
    
    # Calculate average cost basis
    total_invested = TOTAL_CAPITAL - cash
    avg_cost = total_invested / shares
    
    # Look for FG > 75 exit
    exit_day = None
    exit_price = None
    
    for i in range(6, 253):
        if entry_loc + i >= len(df):
            break
        if df['FG'].iloc[entry_loc + i] > 75:
            exit_day = i
            exit_price = df['Price'].iloc[entry_loc + i]
            trades.append(f"Day {i}: EXIT at FG={df['FG'].iloc[entry_loc + i]:.0f}, price ${exit_price:.2f}")
            break
    
    # If no FG > 70 exit, hold to 252 days
    if exit_day is None:
        exit_day = 252
        exit_price = df['Price'].iloc[entry_loc + 252]
        trades.append(f"Day 252: EXIT (max hold), price ${exit_price:.2f}")
    
    # Calculate return
    final_value = cash + (shares * exit_price)
    total_return = (final_value - TOTAL_CAPITAL) / TOTAL_CAPITAL * 100
    
    # Also track return from avg cost basis
    return_from_avg = (exit_price - avg_cost) / avg_cost * 100
    
    results.append({
        'entry_date': entry_date,
        'entry_price': entry_price,
        'week_price': week_price,
        'price_change_1w': price_change_1w,
        'strategy_type': strategy_type,
        'avg_cost': avg_cost,
        'exit_price': exit_price,
        'exit_day': exit_day,
        'total_return': total_return,
        'return_from_avg': return_from_avg,
        'trades': trades
    })

results_df = pd.DataFrame(results)

print(f"\n{'='*70}")
print("BUY-THE-DIP STRATEGY RESULTS")
print(f"{'='*70}")
print(f"\nTotal trades: {len(results_df)}")
print(f"Win rate: {(results_df['total_return'] > 0).mean()*100:.1f}%")
print(f"Average return: {results_df['total_return'].mean():.2f}%")
print(f"Median return: {results_df['total_return'].median():.2f}%")
print(f"Best trade: {results_df['total_return'].max():.2f}%")
print(f"Worst trade: {results_df['total_return'].min():.2f}%")
print(f"Avg holding period: {results_df['exit_day'].mean():.0f} days")

# Breakdown by strategy type
print(f"\n--- Breakdown by 1-week price action ---")
dip_trades = results_df[results_df['strategy_type'] == 'full_deploy_on_dip']
rise_trades = results_df[results_df['strategy_type'] == 'partial_deploy_on_rise']

if len(dip_trades) > 0:
    print(f"\nPrice DROPPED after 1 week ({len(dip_trades)} trades):")
    print(f"  Avg return: {dip_trades['total_return'].mean():.2f}%")
    print(f"  Win rate: {(dip_trades['total_return'] > 0).mean()*100:.1f}%")

if len(rise_trades) > 0:
    print(f"\nPrice ROSE after 1 week ({len(rise_trades)} trades):")
    print(f"  Avg return: {rise_trades['total_return'].mean():.2f}%")
    print(f"  Win rate: {(rise_trades['total_return'] > 0).mean()*100:.1f}%")

# Compare to buy and hold
print(f"\n{'='*70}")
print("COMPARISON: ALL STRATEGIES")
print(f"{'='*70}")

bh_returns = []
for entry_date in entry_events:
    entry_loc = df.index.get_loc(entry_date)
    if entry_loc + 252 >= len(df):
        continue
    entry_price = df['Price'].iloc[entry_loc]
    exit_price = df['Price'].iloc[entry_loc + 252]
    bh_ret = (exit_price - entry_price) / entry_price * 100
    bh_returns.append(bh_ret)

bh_returns = np.array(bh_returns)

print(f"\n{'Strategy':<30} {'Avg Return':>12} {'Win Rate':>10} {'Worst':>10}")
print("-"*65)
print(f"{'Buy & Hold (all-in day 1)':<30} {bh_returns.mean():>+11.2f}% {(bh_returns > 0).mean()*100:>9.1f}% {bh_returns.min():>+9.2f}%")
print(f"{'Buy-the-Dip (35% + rest if dip)':<30} {results_df['total_return'].mean():>+11.2f}% {(results_df['total_return'] > 0).mean()*100:>9.1f}% {results_df['total_return'].min():>+9.2f}%")

# Sample trades
print(f"\n{'='*70}")
print("SAMPLE TRADES")
print(f"{'='*70}")

for idx, row in results_df.iterrows():
    print(f"\n📅 Entry: {row['entry_date'].strftime('%Y-%m-%d')}")
    print(f"   1-week price change: {row['price_change_1w']:+.1f}%")
    print(f"   Avg cost basis: ${row['avg_cost']:.2f}")
    print(f"   Exit price: ${row['exit_price']:.2f}")
    print(f"   Total return: {row['total_return']:+.2f}%")
    print(f"   Days held: {row['exit_day']}")
    for t in row['trades']:
        print(f"   → {t}")

# Risk/Reward
print(f"\n{'='*70}")
print("RISK/REWARD SUMMARY - BUY-THE-DIP STRATEGY")
print(f"{'='*70}")

winners = results_df[results_df['total_return'] > 0]
losers = results_df[results_df['total_return'] <= 0]

win_rate = len(winners) / len(results_df)
avg_win = winners['total_return'].mean() if len(winners) > 0 else 0
avg_loss = losers['total_return'].mean() if len(losers) > 0 else 0

print(f"\nWin rate: {win_rate*100:.1f}%")
print(f"Average winning trade: +{avg_win:.2f}%")
if len(losers) > 0:
    print(f"Average losing trade: {avg_loss:.2f}%")
    rr_ratio = abs(avg_win / avg_loss)
    print(f"Risk/Reward ratio: {rr_ratio:.2f}:1")
else:
    print(f"Average losing trade: N/A (no losses)")
    rr_ratio = float('inf')
    print(f"Risk/Reward ratio: ∞")

expected_value = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
print(f"Expected value per trade: +{expected_value:.2f}%")

print("\n✅ Buy-the-Dip strategy backtest complete!")