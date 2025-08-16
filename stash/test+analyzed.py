import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load Fear & Greed Index Data
fear_greed_file = "datasets/fear-greed.csv"
fear_greed_data = pd.read_csv(fear_greed_file)
fear_greed_data['Date'] = pd.to_datetime(fear_greed_data['Date'])
fear_greed_data = fear_greed_data[['Date', 'fear_greed']]

# Data range
start_date = '2011-01-03'
end_date = '2020-09-18'
print("Start Date:", start_date, "End Date:", end_date)

# Fetch SPY data
print("Fetching SPY data...")
spy_data = yf.download('SPY', start=start_date, end=end_date, auto_adjust=True)
if spy_data.columns.nlevels > 1:
    spy_data.columns = spy_data.columns.droplevel(1)
spy_data.reset_index(inplace=True)
print("SPY data fetched successfully.")

# Merge data
merged_data = pd.merge(spy_data, fear_greed_data[['Date', 'fear_greed']], on='Date', how='left')
merged_data['fear_greed'] = merged_data['fear_greed'].ffill()
price_col = 'Close'

# Debug print to check merged data
print("Merged data head:\n", merged_data.head())

def strategy(data, lookback_days=3, momentum_threshold=1.5, 
             use_dynamic_sizing=True, use_regime_detection=True,
             use_multi_timeframe=True, use_volatility_adjustment=True,
             base_position_size=1.0, max_position_size=2.0):
    data = data.copy()

    # Fear & Greed analysis
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
    
    # Price analysis
    data['price_returns'] = data[price_col].pct_change()
    data['volatility'] = data['price_returns'].rolling(20).std() * np.sqrt(252)
    data['price_sma_20'] = data[price_col].rolling(20).mean()
    data['price_sma_50'] = data[price_col].rolling(50).mean()
    data['price_momentum'] = data[price_col] / data['price_sma_20'] - 1

    # Regime detection
    if use_regime_detection:
        data['regime_trend'] = data['price_sma_20'] / data['price_sma_50'] - 1
        data['regime'] = np.where(data['regime_trend'] > 0.02, 'Bull',
                                 np.where(data['regime_trend'] < -0.02, 'Bear', 'Sideways'))
    
    data['fg_price_corr'] = data['fear_greed'].rolling(30).corr(data[price_col])
    data['fg_zscore'] = (data['fear_greed'] - data['fear_greed'].rolling(60).mean()) / data['fear_greed'].rolling(60).std()
    
    # Dynamic position sizing
    def calculate_position_size(momentum, velocity, volatility, regime, base_size=1.0):
        if not use_dynamic_sizing:
            return base_size
            
        size = base_size
        if abs(momentum) > momentum_threshold * 1.5:
            size *= 1.3
        if abs(velocity) > 1.0:
            size *= 1.2
        if regime == 'Bull':
            size *= 1.15
        elif regime == 'Bear':
            size *= 0.8
        if volatility > 0.3:
            size *= 0.7
        elif volatility < 0.15:
            size *= 1.1
            
        return min(size, max_position_size)
    
    # Initialize
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
        
        if current_position != 0:
            days_held += 1
        
        if pd.notna(fg_momentum) and pd.notna(fg_velocity):
            # Buy conditions
            base_buy_condition = (
                fg_momentum > momentum_threshold and
                fg_velocity > 0.3 and
                mtf_bullish
            )
            
            # Strength multipliers
            strength_multipliers = []
            if pd.notna(fg_acceleration) and fg_acceleration > 0.2:
                strength_multipliers.append(1.2)
            if fg_zscore < -1.5 and fg_velocity > 0:
                strength_multipliers.append(1.3)
            if price_momentum > 0:
                strength_multipliers.append(1.1)
            if regime == 'Bull':
                strength_multipliers.append(1.1)
            
            signal_strength = np.prod(strength_multipliers) if strength_multipliers else 1.0
            
            buy_condition = (
                current_position <= 0 and
                base_buy_condition and
                signal_strength > 1.0 and
                (not use_volatility_adjustment or volatility < 0.4)
            )
            
            # Sell conditions
            base_sell_condition = (
                fg_momentum < -momentum_threshold or
                fg_velocity < -0.3 or
                days_held >= 8
            )
            
            sell_condition = (
                current_position > 0 and (
                    base_sell_condition or
                    (mtf_bearish and fg_velocity < 0) or
                    (regime == 'Bear' and fg_momentum < 0) or
                    (volatility > 0.5)
                )
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
        
        data.iloc[i, data.columns.get_loc('position')] = current_position
        data.iloc[i, data.columns.get_loc('position_size')] = abs(current_position)
        data.iloc[i, data.columns.get_loc('days_in_position')] = days_held
    
    return data, buy_signals, sell_signals

# Run strategy
multi_timeframe_params = {
    'momentum_threshold': 1.0,
    'lookback_days': 3,
    'use_dynamic_sizing': True,
    'use_multi_timeframe': True,
    'use_volatility_adjustment': True,  # KEY CHANGE!
    'base_position_size': 1.2,
    'max_position_size': 2.5,           # INCREASED
    'use_regime_detection': False
}

data_result, buy_signals, sell_signals = strategy(merged_data, **multi_timeframe_params)

# Debug print to check the first few rows of the result
print("data_result head:\n", data_result[['Date', 'Close', 'position', 'signal', 'signal_strength']].head())

# Calculate returns (without slippage initially)
data_result['returns'] = data_result[price_col].pct_change()
data_result['strategy_returns'] = data_result['returns'] * data_result['position'].shift(1)
data_result['cumulative_strategy_returns'] = (1 + data_result['strategy_returns']).cumprod()

starting_capital = 10000
data_result['capital'] = starting_capital * data_result['cumulative_strategy_returns']

# Debug print to check cumulative strategy returns
print("Cumulative strategy returns tail:\n", data_result[['Date', 'cumulative_strategy_returns']].tail())

# Calculate trades and apply slippage/taxes properly
slippage = 0.001  # 0.1% slippage per trade
tax_rate = 0.25   # 25% tax on gains

trades = []
position = 0
entry_idx = None

for i, row in data_result.iterrows():
    pos = row['position']
    if position == 0 and pos != 0:
        position = pos
        entry_idx = i
    elif position != 0:
        if pos == 0:
            exit_idx = i
            trades.append((entry_idx, exit_idx, position))
            position = 0
            entry_idx = None
        elif (position > 0 and pos < 0) or (position < 0 and pos > 0):
            exit_idx = i
            trades.append((entry_idx, exit_idx, position))
            position = pos
            entry_idx = i

if position != 0 and entry_idx is not None:
    exit_idx = data_result.index[-1]
    trades.append((entry_idx, exit_idx, position))

# Calculate realistic slippage cost
total_slippage_cost = 0
winning_trades = 0
total_profit_before_costs = 0
taxable_gains = 0

for entry_idx, exit_idx, pos_size in trades:
    entry_price = data_result.loc[entry_idx, price_col]
    exit_price = data_result.loc[exit_idx, price_col]
    
    # Calculate trade value for slippage
    trade_value = starting_capital * abs(pos_size)
    trade_slippage = trade_value * slippage * 2  # Entry + Exit slippage
    total_slippage_cost += trade_slippage
    
    # Calculate profit/loss
    if pos_size > 0:  # Long position
        trade_return = (exit_price - entry_price) / entry_price
        if trade_return > 0:
            winning_trades += 1
        trade_profit = trade_return * trade_value - trade_slippage
        total_profit_before_costs += trade_return * trade_value
        if trade_profit > 0:
            taxable_gains += trade_profit
    elif pos_size < 0:  # Short position  
        trade_return = (entry_price - exit_price) / entry_price
        if trade_return > 0:
            winning_trades += 1
        trade_profit = trade_return * trade_value - trade_slippage
        total_profit_before_costs += trade_return * trade_value
        if trade_profit > 0:
            taxable_gains += trade_profit

# Calculate final performance after costs
total_tax_paid = taxable_gains * tax_rate
total_costs = total_slippage_cost + total_tax_paid

# Adjust final capital for realistic costs
strategy_profit_before_costs = data_result['capital'].iloc[-1] - starting_capital
final_capital_after_costs = starting_capital + strategy_profit_before_costs - total_costs

total_trades = len(trades)
win_rate = winning_trades / total_trades if total_trades > 0 else 0
strategy_return_before_costs = data_result['cumulative_strategy_returns'].iloc[-1] - 1
strategy_return_after_costs = (final_capital_after_costs / starting_capital) - 1
buy_hold_return = (merged_data[price_col].iloc[-1] / merged_data[price_col].iloc[0]) - 1

# Basic metrics
strategy_returns_series = data_result['strategy_returns'].dropna()
sharpe_ratio = strategy_returns_series.mean() / strategy_returns_series.std() * np.sqrt(252) if strategy_returns_series.std() > 0 else 0
cumulative = data_result['cumulative_strategy_returns'].fillna(1)
running_max = cumulative.expanding().max()
drawdown = (cumulative / running_max - 1)
max_drawdown = drawdown.min()

# Final debug print for verification
print("Final cumulative strategy returns:\n", data_result['cumulative_strategy_returns'].iloc[-1])
print("Final portfolio value (before costs):", data_result['capital'].iloc[-1])
print("Final portfolio value (after costs):", final_capital_after_costs)
print("Total slippage cost:", total_slippage_cost)
print("Total tax paid:", total_tax_paid)
print("Total trades executed:", total_trades)
print("Number of winning trades:", winning_trades)
print("Winning trade rate:", win_rate)
print("Strategy return (before costs):", strategy_return_before_costs)
print("Strategy return (after costs):", strategy_return_after_costs)
print("Buy and hold return:", buy_hold_return)
print("Sharpe ratio:", sharpe_ratio)
print("Maximum drawdown:", max_drawdown)

# Results
print(f"Strategy Return (Before Costs): {strategy_return_before_costs:.2%}")
print(f"Strategy Return (After Costs): {strategy_return_after_costs:.2%}")
print(f"Buy & Hold Return: {buy_hold_return:.2%}")
print(f"Outperformance (After Costs): {strategy_return_after_costs - buy_hold_return:+.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Maximum Drawdown: {max_drawdown:.2%}")
print(f"Total Trades: {total_trades}")
print(f"Win Rate: {win_rate:.1%}")
print(f"Total Slippage Cost: ${total_slippage_cost:,.2f}")
print(f"Total Tax Paid: ${total_tax_paid:,.2f}")
print(f"Final Portfolio Value (Before Costs): ${data_result['capital'].iloc[-1]:,.2f}")
print(f"Final Portfolio Value (After Costs): ${final_capital_after_costs:,.2f}")

# Calculate Year-over-Year Performance
def calculate_yoy_performance(data_result, price_col='Close'):
    # Create annual snapshots
    data_result['Year'] = data_result['Date'].dt.year
    annual_data = data_result.groupby('Year').last().reset_index()
    
    # Calculate YoY changes
    annual_data['spy_yoy'] = annual_data[price_col].pct_change()
    annual_data['strategy_yoy'] = annual_data['cumulative_strategy_returns'].pct_change()
    annual_data['portfolio_value'] = 10000 * annual_data['cumulative_strategy_returns']
    
    print("Year-over-Year Performance Analysis")
    print("=" * 50)
    
    for i, row in annual_data.iterrows():
        if i == 0:  # Skip first year (no YoY data)
            print(f"{int(row['Year'])}: Starting Year")
            print(f"  Portfolio Value: ${row['portfolio_value']:,.2f}")
            continue
            
        spy_yoy = row['spy_yoy'] * 100
        strategy_yoy = row['strategy_yoy'] * 100
        outperformance = strategy_yoy - spy_yoy
        
        print(f"\n{int(row['Year'])}:")
        print(f"  SPY YoY Return: {spy_yoy:+.1f}%")
        print(f"  Strategy YoY Return: {strategy_yoy:+.1f}%")
        print(f"  Outperformance: {outperformance:+.1f}%")
        print(f"  Portfolio Value: ${row['portfolio_value']:,.2f}")
    
    # Calculate average annual returns
    avg_spy_annual = annual_data['spy_yoy'].mean() * 100
    avg_strategy_annual = annual_data['strategy_yoy'].mean() * 100
    avg_outperformance = avg_strategy_annual - avg_spy_annual
    
    print(f"\n{'='*50}")
    print("AVERAGE ANNUAL PERFORMANCE:")
    print(f"SPY Average Annual Return: {avg_spy_annual:.1f}%")
    print(f"Strategy Average Annual Return: {avg_strategy_annual:.1f}%")
    print(f"Average Annual Outperformance: {avg_outperformance:+.1f}%")
    
    # Calculate compound annual growth rate (CAGR)
    years = len(annual_data) - 1
    spy_cagr = ((annual_data[price_col].iloc[-1] / annual_data[price_col].iloc[0]) ** (1/years) - 1) * 100
    strategy_cagr = ((annual_data['cumulative_strategy_returns'].iloc[-1] / annual_data['cumulative_strategy_returns'].iloc[0]) ** (1/years) - 1) * 100
    
    print(f"\nCOMPOUND ANNUAL GROWTH RATES (CAGR):")
    print(f"SPY CAGR: {spy_cagr:.1f}%")
    print(f"Strategy CAGR: {strategy_cagr:.1f}%")
    print(f"CAGR Outperformance: {strategy_cagr - spy_cagr:+.1f}%")
    
    return annual_data

# Run the analysis
annual_performance = calculate_yoy_performance(data_result)