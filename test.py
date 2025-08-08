import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Load Fear & Greed Index Data
fear_greed_file = "datasets/fear-greed.csv"
fear_greed_data = pd.read_csv(fear_greed_file)
fear_greed_data.rename(columns={"Fear Greed": "fear_greed"}, inplace=True)
fear_greed_data['Date'] = pd.to_datetime(fear_greed_data['Date'])

# Plotting data range from 2011-01-03
start_date = '2011-01-03'
end_date = '2020-09-18'

# Fetch SPY data
spy_data = yf.download('SPY', start=start_date, end=end_date)

# Flatten multi-level columns if they exist
if spy_data.columns.nlevels > 1:
    spy_data.columns = spy_data.columns.droplevel(1)

spy_data.reset_index(inplace=True)

# Merge SPY data with Fear & Greed data
merged_data = pd.merge(spy_data, fear_greed_data[['Date', 'fear_greed']], on='Date', how='left')
merged_data['fear_greed'] = merged_data['fear_greed'].ffill()

# Determine price column
if 'Adj Close' in merged_data.columns:
    price_col = 'Adj Close'
elif 'Close' in merged_data.columns:
    price_col = 'Close'
else:
    price_col = [col for col in merged_data.columns if 'close' in col.lower()][0]

# SUPER ENHANCED STRATEGY WITH ADVANCED FEATURES
def super_enhanced_strategy(data, lookback_days=3, momentum_threshold=1.5, 
                          use_dynamic_sizing=True, use_regime_detection=True,
                          use_multi_timeframe=True, use_volatility_adjustment=True,
                          base_position_size=1.0, max_position_size=2.0):
    
    data = data.copy()
    
    # === FEATURE ENGINEERING ===
    
    # 1. Multi-timeframe Fear & Greed analysis
    data['fg_change'] = data['fear_greed'].diff()
    data['fg_sma_short'] = data['fear_greed'].rolling(lookback_days).mean()
    data['fg_sma_medium'] = data['fear_greed'].rolling(lookback_days * 2).mean()
    data['fg_sma_long'] = data['fear_greed'].rolling(lookback_days * 4).mean()
    
    # Momentum indicators
    data['fg_momentum_short'] = data['fear_greed'] - data['fg_sma_short']
    data['fg_momentum_medium'] = data['fear_greed'] - data['fg_sma_medium']
    data['fg_momentum_long'] = data['fear_greed'] - data['fg_sma_long']
    data['fg_velocity'] = data['fg_change'].rolling(3).mean()
    data['fg_acceleration'] = data['fg_velocity'].diff()
    
    # 2. Advanced price analysis
    data['price_returns'] = data[price_col].pct_change()
    data['volatility'] = data['price_returns'].rolling(20).std() * np.sqrt(252)
    data['price_sma_20'] = data[price_col].rolling(20).mean()
    data['price_sma_50'] = data[price_col].rolling(50).mean()
    data['price_momentum'] = data[price_col] / data['price_sma_20'] - 1
    
    # 3. Regime detection (Bull/Bear/Sideways market)
    if use_regime_detection:
        data['regime_trend'] = data['price_sma_20'] / data['price_sma_50'] - 1
        data['regime'] = np.where(data['regime_trend'] > 0.02, 'Bull',
                                 np.where(data['regime_trend'] < -0.02, 'Bear', 'Sideways'))
    
    # 4. Correlation analysis
    data['fg_price_corr'] = data['fear_greed'].rolling(30).corr(data[price_col])
    
    # 5. Z-score for extreme detection
    data['fg_zscore'] = (data['fear_greed'] - data['fear_greed'].rolling(60).mean()) / data['fear_greed'].rolling(60).std()
    
    # === DYNAMIC POSITION SIZING ===
    def calculate_position_size(momentum, velocity, volatility, regime, base_size=1.0):
        if not use_dynamic_sizing:
            return base_size
            
        size = base_size
        
        # Increase size for stronger signals
        if abs(momentum) > momentum_threshold * 1.5:
            size *= 1.3
        if abs(velocity) > 1.0:
            size *= 1.2
            
        # Adjust for market regime
        if regime == 'Bull':
            size *= 1.15
        elif regime == 'Bear':
            size *= 0.8
            
        # Reduce size in high volatility
        if volatility > 0.3:
            size *= 0.7
        elif volatility < 0.15:
            size *= 1.1
            
        return min(size, max_position_size)
    
    # Initialize strategy columns
    data['position'] = 0.0
    data['position_size'] = 0.0
    data['signal'] = 0
    data['signal_strength'] = 0.0
    data['days_in_position'] = 0
    
    current_position = 0.0
    days_held = 0
    buy_signals = []
    sell_signals = []
    
    for i in range(max(lookback_days * 4, 60), len(data)):
        fg_momentum = data.iloc[i]['fg_momentum_short']
        fg_velocity = data.iloc[i]['fg_velocity']
        fg_acceleration = data.iloc[i]['fg_acceleration']
        volatility = data.iloc[i]['volatility']
        regime = data.iloc[i]['regime'] if use_regime_detection else 'Bull'
        price_momentum = data.iloc[i]['price_momentum']
        fg_zscore = data.iloc[i]['fg_zscore']
        
        # Multi-timeframe confirmation
        if use_multi_timeframe:
            mtf_bullish = (data.iloc[i]['fg_momentum_short'] > 0 and 
                          data.iloc[i]['fg_momentum_medium'] > -1 and
                          data.iloc[i]['fg_momentum_long'] > -2)
            mtf_bearish = (data.iloc[i]['fg_momentum_short'] < 0 and 
                          data.iloc[i]['fg_momentum_medium'] < 1 and
                          data.iloc[i]['fg_momentum_long'] < 2)
        else:
            mtf_bullish = mtf_bearish = True
        
        # Increment days held
        if current_position != 0:
            days_held += 1
        
        if pd.notna(fg_momentum) and pd.notna(fg_velocity):
            
            # === ENHANCED BUY CONDITIONS ===
            base_buy_condition = (
                fg_momentum > momentum_threshold and
                fg_velocity > 0.3 and
                mtf_bullish
            )
            
            # Strength multipliers
            strength_multipliers = []
            
            # Strong acceleration
            if pd.notna(fg_acceleration) and fg_acceleration > 0.2:
                strength_multipliers.append(1.2)
            
            # Extreme fear reversal (contrarian element)
            if fg_zscore < -1.5 and fg_velocity > 0:
                strength_multipliers.append(1.3)
            
            # Price momentum alignment
            if price_momentum > 0:
                strength_multipliers.append(1.1)
            
            # Bull market bonus
            if regime == 'Bull':
                strength_multipliers.append(1.1)
            
            signal_strength = np.prod(strength_multipliers) if strength_multipliers else 1.0
            
            # Enhanced buy condition
            buy_condition = (
                current_position <= 0 and
                base_buy_condition and
                signal_strength > 1.0 and
                (not use_volatility_adjustment or volatility < 0.4)
            )
            
            # === ENHANCED SELL CONDITIONS ===
            base_sell_condition = (
                fg_momentum < -momentum_threshold or
                fg_velocity < -0.3 or
                days_held >= 8
            )
            
            # Enhanced sell conditions
            sell_condition = (
                current_position > 0 and (
                    base_sell_condition or
                    (mtf_bearish and fg_velocity < 0) or
                    (regime == 'Bear' and fg_momentum < 0) or
                    (volatility > 0.5)  # Exit in extreme volatility
                )
            )
            
            # === AGGRESSIVE SHORT CONDITIONS ===
            short_condition = (
                current_position >= 0 and
                fg_momentum < -momentum_threshold * 1.8 and
                fg_velocity < -0.8 and
                mtf_bearish and
                (not use_volatility_adjustment or volatility < 0.6)
            )
            
            # Execute trades
            if buy_condition:
                position_size = calculate_position_size(fg_momentum, fg_velocity, volatility, regime, base_position_size)
                data.iloc[i, data.columns.get_loc('signal')] = 1
                data.iloc[i, data.columns.get_loc('signal_strength')] = signal_strength
                current_position = position_size
                days_held = 0
                buy_signals.append(i)
                
            elif sell_condition:
                data.iloc[i, data.columns.get_loc('signal')] = -1
                current_position = 0.0
                days_held = 0
                sell_signals.append(i)
                
            elif short_condition:
                position_size = calculate_position_size(fg_momentum, fg_velocity, volatility, regime, base_position_size)
                data.iloc[i, data.columns.get_loc('signal')] = -2
                current_position = -position_size
                days_held = 0
                sell_signals.append(i)
        
        data.iloc[i, data.columns.get_loc('position')] = current_position
        data.iloc[i, data.columns.get_loc('position_size')] = abs(current_position)
        data.iloc[i, data.columns.get_loc('days_in_position')] = days_held
    
    return data, buy_signals, sell_signals

# Only run the Multi-Timeframe strategy
multi_timeframe_params = {
    'lookback_days': 3, 
    'momentum_threshold': 1.5, 
    'use_dynamic_sizing': True,
    'use_regime_detection': False, 
    'use_multi_timeframe': True, 
    'use_volatility_adjustment': False
}

data_result, buy_signals, sell_signals = super_enhanced_strategy(merged_data, **multi_timeframe_params)

# Calculate returns with position sizing
data_result['returns'] = data_result[price_col].pct_change()
data_result['strategy_returns'] = data_result['returns'] * data_result['position'].shift(1)
data_result['cumulative_strategy_returns'] = (1 + data_result['strategy_returns']).cumprod()

starting_capital = 10000  # $10,000 initial investment
data_result['capital'] = starting_capital * data_result['cumulative_strategy_returns']


print(f"DEBUG - Starting Capital: ${starting_capital:,.2f}")
print(f"DEBUG - Final Cumulative Return Factor: {data_result['cumulative_strategy_returns'].iloc[-1]:,.4f}")
print(f"DEBUG - Final Balance: ${data_result['capital'].iloc[-1]:,.2f}")

strategy_return = data_result['cumulative_strategy_returns'].iloc[-1] - 1

# Enhanced metrics
total_trades = len(buy_signals) + len(sell_signals)

# Sharpe ratio calculation
strategy_returns_series = data_result['strategy_returns'].dropna()
sharpe_ratio = strategy_returns_series.mean() / strategy_returns_series.std() * np.sqrt(252) if strategy_returns_series.std() > 0 else 0

# Maximum drawdown
cumulative = data_result['cumulative_strategy_returns'].fillna(1)
running_max = cumulative.expanding().max()
drawdown = (cumulative / running_max - 1)
max_drawdown = drawdown.min()

# Win rate (improved calculation)
winning_trades = 0
if buy_signals:
    for signal_idx in buy_signals:
        if signal_idx < len(data_result) - 5:
            entry_price = data_result.iloc[signal_idx][price_col]
            future_prices = data_result.iloc[signal_idx+1:signal_idx+6][price_col]
            if len(future_prices) > 0 and future_prices.max() > entry_price:
                winning_trades += 1
    win_rate = winning_trades / len(buy_signals) if buy_signals else 0
else:
    win_rate = 0

# Calculate buy & hold metrics
buy_hold_return = (merged_data[price_col].iloc[-1] / merged_data[price_col].iloc[0]) - 1
bh_returns = merged_data[price_col].pct_change().dropna()
bh_sharpe = bh_returns.mean() / bh_returns.std() * np.sqrt(252)

# Enhanced visualization
fig, axes = plt.subplots(4, 1, figsize=(18, 16))

# Plot 1: Price with position sizing
axes[0].plot(data_result['Date'], data_result[price_col], color='blue', label=f'SPY {price_col}', linewidth=1)

if buy_signals:
    buy_dates = data_result.iloc[buy_signals]['Date']
    buy_prices = data_result.iloc[buy_signals][price_col]
    buy_sizes = data_result.iloc[buy_signals]['position_size']
    
    # Size-based marker scaling
    marker_sizes = 50 + (buy_sizes - 1) * 100
    axes[0].scatter(buy_dates, buy_prices, c=buy_sizes, cmap='Greens', 
                   marker='^', s=marker_sizes, label='Buy Signal (size)', zorder=5, alpha=0.8)

if sell_signals:
    sell_dates = data_result.iloc[sell_signals]['Date']
    sell_prices = data_result.iloc[sell_signals][price_col]
    axes[0].scatter(sell_dates, sell_prices, color='red', marker='v', s=100, label='Sell Signal', zorder=5)

axes[0].set_ylabel(f'SPY {price_col} Price')
axes[0].set_title(f'Multi-Timeframe Strategy (Position Sizing Shown)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Multi-timeframe Fear & Greed
axes[1].plot(data_result['Date'], data_result['fear_greed'], color='orange', label='F&G Index', linewidth=1)
axes[1].plot(data_result['Date'], data_result['fg_sma_short'], color='green', label='Short MA', alpha=0.7)
axes[1].plot(data_result['Date'], data_result['fg_sma_medium'], color='blue', label='Medium MA', alpha=0.7)
axes[1].plot(data_result['Date'], data_result['fg_sma_long'], color='purple', label='Long MA', alpha=0.7)

axes[1].set_ylabel('Fear & Greed Index')
axes[1].set_title('Multi-Timeframe Fear & Greed Analysis')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot 3: Advanced momentum indicators
axes[2].plot(data_result['Date'], data_result['fg_momentum_short'], color='red', label='Short Momentum', linewidth=1)
axes[2].plot(data_result['Date'], data_result['fg_velocity'], color='blue', label='Velocity', linewidth=1)
axes[2].plot(data_result['Date'], data_result['fg_acceleration'], color='green', label='Acceleration', linewidth=1)
axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.5)

axes[2].set_ylabel('Momentum Indicators')
axes[2].set_title('Advanced Fear & Greed Momentum Analysis')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

# Plot 4: Strategy performance
axes[3].plot(data_result['Date'], data_result['cumulative_strategy_returns'], color='green', label='Multi-Timeframe Strategy', linewidth=2)

# Calculate buy & hold performance for comparison
data_result['buy_hold_returns'] = data_result[price_col] / data_result[price_col].iloc[0]
axes[3].plot(data_result['Date'], data_result['buy_hold_returns'], color='blue', label='Buy & Hold', linewidth=1, alpha=0.7)

axes[3].set_ylabel('Cumulative Returns')
axes[3].set_title('Strategy Performance Comparison')
axes[3].legend()
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print results summary
print(f"\n=== MULTI-TIMEFRAME STRATEGY RESULTS ===")
print(f"Strategy Return: {strategy_return:.2%}")
print(f"Buy & Hold Return: {buy_hold_return:.2%}")
print(f"Outperformance: {strategy_return - buy_hold_return:+.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f} (Buy & Hold: {bh_sharpe:.2f})")
print(f"Maximum Drawdown: {max_drawdown:.2%}")
print(f"Total Trades: {total_trades}")
print(f"Win Rate: {win_rate:.1%}")

print(f"\n=== MULTI-TIMEFRAME STRATEGY FEATURES ===")
print("📊 MULTI-TIMEFRAME ANALYSIS: Short, medium, long-term momentum confirmation")
print("🎯 DYNAMIC POSITION SIZING: Larger positions for stronger signals")
print("⚡ SIGNAL STRENGTH: Multiple confirmation factors")