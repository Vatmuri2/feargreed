import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load your CSV (make sure this path is correct)
trade_df = pd.read_csv('trade_analysis_detailed.csv')

print("🔍 ADVANCED TRADE ANALYSIS")
print("=" * 50)

# 1. Holding Period Analysis
print("\n1. Holding Period Analysis")
print("-" * 30)

def analyze_holding_periods(df, periods):
    for min_days, max_days in periods:
        subset = df[(df['days_held'] >= min_days) & (df['days_held'] <= max_days)]
        win_rate = (subset['profit_loss'] == 'PROFIT').mean()
        avg_return = subset['trade_return_pct'].mean()
        print(f"{min_days}-{max_days} days: {len(subset)} trades, {win_rate:.1%} win rate, {avg_return:.2f}% avg return")

analyze_holding_periods(trade_df, [(1, 3), (4, 7), (8, 14), (15, float('inf'))])

# 2. Volatility Impact
print("\n2. Volatility Impact")
print("-" * 30)

def analyze_volatility_impact(df, vol_ranges):
    for min_vol, max_vol in vol_ranges:
        subset = df[(df['entry_volatility'] >= min_vol) & (df['entry_volatility'] < max_vol)]
        win_rate = (subset['profit_loss'] == 'PROFIT').mean()
        avg_return = subset['trade_return_pct'].mean()
        print(f"Volatility {min_vol:.2f}-{max_vol:.2f}: {len(subset)} trades, {win_rate:.1%} win rate, {avg_return:.2f}% avg return")

analyze_volatility_impact(trade_df, [(0, 0.15), (0.15, 0.20), (0.20, 0.25), (0.25, float('inf'))])

# 3. Signal Strength Optimization
print("\n3. Signal Strength Optimization")
print("-" * 30)

def analyze_signal_strength(df, strength_thresholds):
    for threshold in strength_thresholds:
        subset = df[df['entry_signal_strength'] >= threshold]
        win_rate = (subset['profit_loss'] == 'PROFIT').mean()
        avg_return = subset['trade_return_pct'].mean()
        print(f"Signal Strength ≥ {threshold:.2f}: {len(subset)} trades, {win_rate:.1%} win rate, {avg_return:.2f}% avg return")

analyze_signal_strength(trade_df, [1.2, 1.3, 1.4, 1.5, 1.6])

# 4. Entry Day Analysis
print("\n4. Entry Day Analysis")
print("-" * 30)

trade_df['entry_day'] = pd.to_datetime(trade_df['entry_date']).dt.day_name()
day_performance = trade_df.groupby('entry_day').agg({
    'trade_return_pct': 'mean',
    'profit_loss': lambda x: (x == 'PROFIT').mean(),
    'trade_num': 'count'
}).rename(columns={'trade_return_pct': 'avg_return', 'profit_loss': 'win_rate', 'trade_num': 'count'})

print(day_performance.sort_values('avg_return', ascending=False))

# 5. Cumulative Returns by Criteria
print("\n5. Cumulative Returns by Criteria")
print("-" * 30)

def plot_cumulative_returns(df, condition, label):
    subset = df[condition].copy()
    subset['cumulative_return'] = (1 + subset['trade_return_pct'] / 100).cumprod()
    plt.plot(subset.index, subset['cumulative_return'], label=label)

plt.figure(figsize=(12, 6))

# Plot for different holding periods
plot_cumulative_returns(trade_df, trade_df['days_held'] <= 3, '1-3 days')
plot_cumulative_returns(trade_df, (trade_df['days_held'] > 3) & (trade_df['days_held'] <= 7), '4-7 days')
plot_cumulative_returns(trade_df, trade_df['days_held'] > 7, '8+ days')

plt.title('Cumulative Returns by Holding Period')
plt.xlabel('Trade Number')
plt.ylabel('Cumulative Return')
plt.legend()
plt.show()

# Additional visualization for volatility and signal strength
plt.figure(figsize=(12, 6))

# Plot for different volatility ranges
plot_cumulative_returns(trade_df, trade_df['entry_volatility'] < 0.15, 'Vol < 0.15')
plot_cumulative_returns(trade_df, (trade_df['entry_volatility'] >= 0.15) & (trade_df['entry_volatility'] < 0.25), 'Vol 0.15-0.25')
plot_cumulative_returns(trade_df, trade_df['entry_volatility'] >= 0.25, 'Vol >= 0.25')

plt.title('Cumulative Returns by Volatility Range')
plt.xlabel('Trade Number')
plt.ylabel('Cumulative Return')
plt.legend()
plt.show()

print("\nAnalysis complete. Please review the output and generated plots.")