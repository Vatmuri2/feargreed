import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Load Fear & Greed Index Data
fear_greed_file = "datasets/fear_greed_combined_2011_2025.csv"
fear_greed_data = pd.read_csv(fear_greed_file)
fear_greed_data['Date'] = pd.to_datetime(fear_greed_data['Date'])
fear_greed_data = fear_greed_data[['Date', 'fear_greed']]

# Data range
start_date = '2011-01-03'
end_date = '2025-08-08'
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

def calculate_slippage(trade_size, volatility, base_slippage=0.001):
    """
    Calculate slippage based on trade size and market volatility
    """
    # Slippage increases with position size and volatility
    volatility_factor = max(1.0, volatility / 0.20)  # Normalize to 20% volatility
    size_factor = 1.0 + (abs(trade_size) - 1.0) * 0.0005  # Additional slippage for larger positions
    
    return base_slippage * volatility_factor * size_factor

def calculate_taxes(realized_gains, holding_period_days, short_term_rate=0.37, long_term_rate=0.20):
    """
    Calculate taxes on realized gains
    holding_period_days: number of days the position was held
    short_term_rate: tax rate for positions held < 365 days
    long_term_rate: tax rate for positions held >= 365 days
    """
    if realized_gains <= 0:
        return 0  # No taxes on losses (simplification - in reality losses can offset gains)
    
    if holding_period_days >= 365:
        return realized_gains * long_term_rate
    else:
        return realized_gains * short_term_rate

def strategy(data, lookback_days=3, momentum_threshold=1.5, 
             use_dynamic_sizing=True, use_regime_detection=True,
             use_multi_timeframe=True, use_volatility_adjustment=True,
             base_position_size=1.0, max_position_size=2.0,
             base_slippage=0.001, commission_per_trade=0.0):
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
    data['slippage_cost'] = 0.0
    data['commission_cost'] = 0.0
    data['tax_cost'] = 0.0
    data['net_trade_pnl'] = 0.0
    
    current_position = 0.0
    days_held = 0
    buy_signals = []
    sell_signals = []
    entry_price = None
    entry_date = None
    
    for i in range(max(lookback_days * 4, 60), len(data)):
        fg_momentum = data.iloc[i]['fg_momentum_short']
        fg_velocity = data.iloc[i]['fg_velocity']
        fg_acceleration = data.iloc[i]['fg_acceleration']
        volatility = data.iloc[i]['volatility']
        regime = data.iloc[i]['regime'] if use_regime_detection else 'Bull'
        price_momentum = data.iloc[i]['price_momentum']
        fg_zscore = data.iloc[i]['fg_zscore']
        current_price = data.iloc[i][price_col]
        current_date = data.iloc[i]['Date']
        
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
                
                # Calculate transaction costs
                slippage = calculate_slippage(position_size, volatility, base_slippage)
                effective_buy_price = current_price * (1 + slippage)
                commission = commission_per_trade
                
                data.iloc[i, data.columns.get_loc('signal')] = 1
                data.iloc[i, data.columns.get_loc('signal_strength')] = signal_strength
                data.iloc[i, data.columns.get_loc('slippage_cost')] = slippage * current_price * abs(position_size)
                data.iloc[i, data.columns.get_loc('commission_cost')] = commission
                
                current_position = position_size
                days_held = 0
                entry_price = effective_buy_price
                entry_date = current_date
                buy_signals.append(i)
                
            elif sell_condition and entry_price is not None:
                # Calculate transaction costs for selling
                slippage = calculate_slippage(current_position, volatility, base_slippage)
                effective_sell_price = current_price * (1 - slippage)
                commission = commission_per_trade
                
                # Calculate P&L and taxes
                gross_pnl = (effective_sell_price - entry_price) * abs(current_position)
                holding_period = (current_date - entry_date).days
                tax_cost = calculate_taxes(gross_pnl, holding_period)
                
                # Net P&L after all costs
                net_pnl = gross_pnl - (slippage * current_price * abs(current_position)) - commission - tax_cost
                
                data.iloc[i, data.columns.get_loc('signal')] = -1
                data.iloc[i, data.columns.get_loc('slippage_cost')] = slippage * current_price * abs(current_position)
                data.iloc[i, data.columns.get_loc('commission_cost')] = commission
                data.iloc[i, data.columns.get_loc('tax_cost')] = tax_cost
                data.iloc[i, data.columns.get_loc('net_trade_pnl')] = net_pnl
                
                current_position = 0.0
                days_held = 0
                entry_price = None
                entry_date = None
                sell_signals.append(i)
        
        data.iloc[i, data.columns.get_loc('position')] = current_position
        data.iloc[i, data.columns.get_loc('position_size')] = abs(current_position)
        data.iloc[i, data.columns.get_loc('days_in_position')] = days_held
    
    return data, buy_signals, sell_signals

# Run strategy with slippage and commission parameters
multi_timeframe_params = {
    'momentum_threshold': 1.0,
    'lookback_days': 3,
    'use_dynamic_sizing': True,
    'use_multi_timeframe': True,
    'use_volatility_adjustment': True,
    'base_position_size': 1.2,
    'max_position_size': 2.5,
    'use_regime_detection': False,
    'base_slippage': 0.0015,  # 0.15% base slippage (can be higher for ETFs than individual stocks)
    'commission_per_trade': 0.0  # Most brokers now offer commission-free ETF trading
}

data_result, buy_signals, sell_signals = strategy(merged_data, **multi_timeframe_params)

# Debug print to check the first few rows of the result
print("data_result head:\n", data_result[['Date', 'Close', 'position', 'signal', 'signal_strength']].head())

# Calculate returns with transaction costs
def calculate_adjusted_returns(data, price_col, starting_capital=10000):
    """Calculate strategy returns accounting for all transaction costs"""
    data = data.copy()
    
    # Basic returns
    data['returns'] = data[price_col].pct_change()
    
    # Strategy returns before costs
    data['gross_strategy_returns'] = data['returns'] * data['position'].shift(1)
    
    # Calculate transaction costs as percentage of capital
    data['total_transaction_costs'] = (data['slippage_cost'] + 
                                     data['commission_cost'] + 
                                     data['tax_cost'])
    
    # Convert transaction costs to returns (negative impact)
    data['transaction_cost_returns'] = -data['total_transaction_costs'] / starting_capital
    
    # Net strategy returns after all costs
    data['net_strategy_returns'] = data['gross_strategy_returns'] + data['transaction_cost_returns']
    
    # Cumulative returns
    data['cumulative_gross_returns'] = (1 + data['gross_strategy_returns']).cumprod()
    data['cumulative_net_returns'] = (1 + data['net_strategy_returns']).cumprod()
    
    return data

data_result = calculate_adjusted_returns(data_result, price_col)

# Calculate capital
starting_capital = 10000
data_result['gross_capital'] = starting_capital * data_result['cumulative_gross_returns']
data_result['net_capital'] = starting_capital * data_result['cumulative_net_returns']

# Calculate trades with detailed cost analysis
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

# Calculate detailed trade metrics including costs
winning_trades = 0
total_slippage_costs = 0
total_commission_costs = 0
total_tax_costs = 0
gross_profits = 0
net_profits = 0

trade_details = []

for entry_idx, exit_idx, pos_size in trades:
    entry_price = data_result.loc[entry_idx, price_col]
    exit_price = data_result.loc[exit_idx, price_col]
    entry_date = data_result.loc[entry_idx, 'Date']
    exit_date = data_result.loc[exit_idx, 'Date']
    
    # Calculate gross P&L
    if pos_size > 0:  # Long position
        gross_pnl = (exit_price - entry_price) * abs(pos_size)
    else:  # Short position
        gross_pnl = (entry_price - exit_price) * abs(pos_size)
    
    # Get transaction costs for this trade
    entry_slippage = data_result.loc[entry_idx, 'slippage_cost']
    exit_slippage = data_result.loc[exit_idx, 'slippage_cost']
    entry_commission = data_result.loc[entry_idx, 'commission_cost']
    exit_commission = data_result.loc[exit_idx, 'commission_cost']
    tax_cost = data_result.loc[exit_idx, 'tax_cost']
    
    total_costs = entry_slippage + exit_slippage + entry_commission + exit_commission + tax_cost
    net_pnl = gross_pnl - total_costs
    
    # Accumulate totals
    total_slippage_costs += entry_slippage + exit_slippage
    total_commission_costs += entry_commission + exit_commission
    total_tax_costs += tax_cost
    gross_profits += gross_pnl
    net_profits += net_pnl
    
    # Count winning trades (based on net P&L)
    if net_pnl > 0:
        winning_trades += 1
    
    # Store trade details
    holding_period = (exit_date - entry_date).days
    trade_details.append({
        'entry_date': entry_date,
        'exit_date': exit_date,
        'holding_period': holding_period,
        'entry_price': entry_price,
        'exit_price': exit_price,
        'position_size': pos_size,
        'gross_pnl': gross_pnl,
        'slippage_costs': entry_slippage + exit_slippage,
        'commission_costs': entry_commission + exit_commission,
        'tax_costs': tax_cost,
        'total_costs': total_costs,
        'net_pnl': net_pnl,
        'gross_return': gross_pnl / (entry_price * abs(pos_size)) if entry_price > 0 else 0,
        'net_return': net_pnl / (entry_price * abs(pos_size)) if entry_price > 0 else 0
    })

# Calculate comprehensive metrics
total_trades = len(trades)
win_rate = winning_trades / total_trades if total_trades > 0 else 0

# Strategy returns (gross and net)
gross_strategy_return = data_result['cumulative_gross_returns'].iloc[-1] - 1
net_strategy_return = data_result['cumulative_net_returns'].iloc[-1] - 1
buy_hold_return = (merged_data[price_col].iloc[-1] / merged_data[price_col].iloc[0]) - 1

# Calculate Sharpe ratios
gross_returns_series = data_result['gross_strategy_returns'].dropna()
net_returns_series = data_result['net_strategy_returns'].dropna()

gross_sharpe = gross_returns_series.mean() / gross_returns_series.std() * np.sqrt(252) if gross_returns_series.std() > 0 else 0
net_sharpe = net_returns_series.mean() / net_returns_series.std() * np.sqrt(252) if net_returns_series.std() > 0 else 0

# Calculate drawdowns
gross_cumulative = data_result['cumulative_gross_returns'].fillna(1)
net_cumulative = data_result['cumulative_net_returns'].fillna(1)

gross_running_max = gross_cumulative.expanding().max()
net_running_max = net_cumulative.expanding().max()

gross_drawdown = (gross_cumulative / gross_running_max - 1)
net_drawdown = (net_cumulative / net_running_max - 1)

max_gross_drawdown = gross_drawdown.min()
max_net_drawdown = net_drawdown.min()

# Cost impact analysis
total_costs_impact = total_slippage_costs + total_commission_costs + total_tax_costs
cost_impact_on_return = (gross_strategy_return - net_strategy_return)
annual_cost_rate = cost_impact_on_return / ((pd.to_datetime(end_date) - pd.to_datetime(start_date)).days / 365.25)

# Print comprehensive results
print("=" * 80)
print("STRATEGY PERFORMANCE ANALYSIS (WITH TRANSACTION COSTS)")
print("=" * 80)

print(f"\nRETURNS COMPARISON:")
print(f"Strategy Return (Gross):     {gross_strategy_return:+.2%}")
print(f"Strategy Return (Net):       {net_strategy_return:+.2%}")
print(f"Buy & Hold Return:           {buy_hold_return:+.2%}")
print(f"Gross Outperformance:        {gross_strategy_return - buy_hold_return:+.2%}")
print(f"Net Outperformance:          {net_strategy_return - buy_hold_return:+.2%}")

print(f"\nRISK METRICS:")
print(f"Gross Sharpe Ratio:          {gross_sharpe:.2f}")
print(f"Net Sharpe Ratio:            {net_sharpe:.2f}")
print(f"Max Drawdown (Gross):        {max_gross_drawdown:.2%}")
print(f"Max Drawdown (Net):          {max_net_drawdown:.2%}")

print(f"\nTRADING METRICS:")
print(f"Total Trades:                {total_trades}")
print(f"Winning Trades:              {winning_trades}")
print(f"Win Rate:                    {win_rate:.1%}")
print(f"Average Holding Period:      {np.mean([t['holding_period'] for t in trade_details]):.1f} days")

print(f"\nCOST BREAKDOWN:")
print(f"Total Slippage Costs:        ${total_slippage_costs:,.2f}")
print(f"Total Commission Costs:      ${total_commission_costs:,.2f}")
print(f"Total Tax Costs:             ${total_tax_costs:,.2f}")
print(f"Total Transaction Costs:     ${total_costs_impact:,.2f}")
print(f"Cost Impact on Returns:      {cost_impact_on_return:.2%}")
print(f"Annualized Cost Rate:        {annual_cost_rate:.2%}")

print(f"\nPORTFOLIO VALUES:")
print(f"Starting Capital:            ${starting_capital:,.2f}")
print(f"Final Value (Gross):         ${data_result['gross_capital'].iloc[-1]:,.2f}")
print(f"Final Value (Net):           ${data_result['net_capital'].iloc[-1]:,.2f}")
print(f"Total Profit (Gross):        ${data_result['gross_capital'].iloc[-1] - starting_capital:,.2f}")
print(f"Total Profit (Net):          ${data_result['net_capital'].iloc[-1] - starting_capital:,.2f}")

# Create trade pairs for visualization
trade_pairs = []
for trade in trade_details:
    trade_pairs.append({
        'entry_idx': data_result[data_result['Date'] == trade['entry_date']].index[0],
        'exit_idx': data_result[data_result['Date'] == trade['exit_date']].index[0],
        'entry_price': trade['entry_price'],
        'exit_price': trade['exit_price'],
        'gross_pnl': trade['gross_pnl'],
        'net_pnl': trade['net_pnl'],
        'total_costs': trade['total_costs'],
        'holding_period': trade['holding_period']
    })

# Create enhanced visualization
fig = make_subplots(
    rows=3, cols=1, 
    shared_xaxes=True, 
    vertical_spacing=0.08,
    subplot_titles=('SPY Price with Buy/Sell Signals', 
                   'Portfolio Value (Gross vs Net)', 
                   'Cumulative Transaction Costs'),
    row_heights=[0.5, 0.3, 0.2]
)

# Add SPY closing prices
fig.add_trace(
    go.Scatter(x=data_result['Date'], y=data_result[price_col], 
              mode='lines', name='SPY Price', line=dict(color='blue')),
    row=1, col=1
)

# Add buy and sell markers with enhanced information
for i, trade in enumerate(trade_pairs):
    entry_idx = trade['entry_idx']
    exit_idx = trade['exit_idx']
    
    entry_date = data_result['Date'].iloc[entry_idx]
    exit_date = data_result['Date'].iloc[exit_idx]
    entry_price = trade['entry_price']
    exit_price = trade['exit_price']
    gross_pnl = trade['gross_pnl']
    net_pnl = trade['net_pnl']
    total_costs = trade['total_costs']
    holding_period = trade['holding_period']
    
    # Color based on net profitability
    marker_color = 'green' if net_pnl > 0 else 'red'
    
    # Add buy marker
    fig.add_trace(
        go.Scatter(x=[entry_date], y=[entry_price], mode='markers',
                   marker=dict(color=marker_color, symbol='triangle-up', size=10),
                   name=f'Buy Signal {i+1}' if i < 5 else 'Buy Signal',
                   text=[f'Buy: ${entry_price:.2f}<br>Hold: {holding_period} days<br>Gross P&L: ${gross_pnl:.2f}<br>Net P&L: ${net_pnl:.2f}<br>Costs: ${total_costs:.2f}'],
                   hoverinfo='text',
                   showlegend=(i < 3)),
        row=1, col=1
    )
    
    # Add sell marker
    fig.add_trace(
        go.Scatter(x=[exit_date], y=[exit_price], mode='markers',
                   marker=dict(color=marker_color, symbol='triangle-down', size=10),
                   name=f'Sell Signal {i+1}' if i < 5 else 'Sell Signal',
                   text=[f'Sell: ${exit_price:.2f}<br>Hold: {holding_period} days<br>Gross P&L: ${gross_pnl:.2f}<br>Net P&L: ${net_pnl:.2f}<br>Costs: ${total_costs:.2f}'],
                   hoverinfo='text',
                   showlegend=(i < 3)),
        row=1, col=1
    )

# Add portfolio values (gross and net)
fig.add_trace(
    go.Scatter(x=data_result['Date'], y=data_result['gross_capital'], 
              mode='lines', name='Gross Portfolio Value', 
              line=dict(color='green', dash='solid')),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(x=data_result['Date'], y=data_result['net_capital'], 
              mode='lines', name='Net Portfolio Value', 
              line=dict(color='darkgreen', dash='solid')),
    row=2, col=1
)

# Add buy & hold comparison
buy_hold_values = starting_capital * (data_result[price_col] / data_result[price_col].iloc[0])
fig.add_trace(
    go.Scatter(x=data_result['Date'], y=buy_hold_values, 
              mode='lines', name='Buy & Hold Value', 
              line=dict(color='gray', dash='dash')),
    row=2, col=1
)

# Add cumulative transaction costs
cumulative_costs = data_result['total_transaction_costs'].cumsum()
fig.add_trace(
    go.Scatter(x=data_result['Date'], y=cumulative_costs, 
              mode='lines', name='Cumulative Transaction Costs', 
              line=dict(color='red')),
    row=3, col=1
)

# Update layout
fig.update_layout(
    title='Enhanced Strategy Performance Analysis with Transaction Costs',
    height=800,
    hovermode='x unified'
)

fig.update_xaxes(title_text="Date", row=3, col=1)
fig.update_yaxes(title_text="Price ($)", row=1, col=1)
fig.update_yaxes(title_text="Portfolio Value ($)", row=2, col=1)
fig.update_yaxes(title_text="Cumulative Costs ($)", row=3, col=1)

# Show the enhanced figure
fig.show()

# Create a summary table of individual trades
# Create a summary table of individual trades
trade_summary_df = pd.DataFrame(trade_details)
if len(trade_summary_df) > 0:
    print(f"\nTOP 10 MOST PROFITABLE TRADES (NET):")
    print("=" * 120)
    top_trades = trade_summary_df.nlargest(10, 'net_pnl')[['entry_date', 'exit_date', 'holding_period', 
                                                          'entry_price', 'exit_price', 'gross_pnl', 
                                                          'total_costs', 'net_pnl', 'net_return']]
    top_trades['entry_date'] = top_trades['entry_date'].dt.strftime('%Y-%m-%d')
    top_trades['exit_date'] = top_trades['exit_date'].dt.strftime('%Y-%m-%d')
    
    for col in ['entry_price', 'exit_price', 'gross_pnl', 'total_costs', 'net_pnl']:
        top_trades[col] = top_trades[col].apply(lambda x: f"${x:.2f}")
    top_trades['net_return'] = top_trades['net_return'].apply(lambda x: f"{x:.2%}")
    
    print(top_trades.to_string(index=False))
    
    print(f"\nWORST 5 TRADES (NET):")
    print("=" * 120)
    worst_trades = trade_summary_df.nsmallest(5, 'net_pnl')[['entry_date', 'exit_date', 'holding_period',
                                                            'entry_price', 'exit_price', 'gross_pnl',
                                                            'total_costs', 'net_pnl', 'net_return']]
    worst_trades['entry_date'] = worst_trades['entry_date'].dt.strftime('%Y-%m-%d')
    worst_trades['exit_date'] = worst_trades['exit_date'].dt.strftime('%Y-%m-%d')
    
    for col in ['entry_price', 'exit_price', 'gross_pnl', 'total_costs', 'net_pnl']:
        worst_trades[col] = worst_trades[col].apply(lambda x: f"${x:.2f}")
    worst_trades['net_return'] = worst_trades['net_return'].apply(lambda x: f"{x:.2%}")
    
    print(worst_trades.to_string(index=False))


# First, let's see what columns are available
print("Available columns in data_result:")
print(data_result.columns.tolist())
print("\nSample of data_result:")
print(data_result[['Date', 'Close']].head())

# Fixed function that creates the cumulative returns if missing
def calculate_yoy_performance_extended(data_result, price_col='Close'):
    data_result = data_result.copy()
    
    # Check if we have the necessary columns, if not create them
    if 'returns' not in data_result.columns:
        data_result['returns'] = data_result[price_col].pct_change()
    
    if 'strategy_returns' not in data_result.columns:
        # Assuming position column exists
        if 'position' in data_result.columns:
            data_result['strategy_returns'] = data_result['returns'] * data_result['position'].shift(1)
        else:
            print("ERROR: No 'position' column found. Cannot calculate strategy returns.")
            return None
    
    if 'cumulative_strategy_returns' not in data_result.columns:
        data_result['cumulative_strategy_returns'] = (1 + data_result['strategy_returns'].fillna(0)).cumprod()
    
    # Create annual snapshots
    data_result['Year'] = data_result['Date'].dt.year
    annual_data = data_result.groupby('Year').last().reset_index()
    
    # Calculate YoY changes
    annual_data['spy_yoy'] = annual_data[price_col].pct_change()
    annual_data['strategy_yoy'] = annual_data['cumulative_strategy_returns'].pct_change()
    annual_data['portfolio_value'] = 10000 * annual_data['cumulative_strategy_returns']
    
    print("YEAR-OVER-YEAR PERFORMANCE ANALYSIS (2011-2023)")
    print("=" * 70)
    
    for i, row in annual_data.iterrows():
        if i == 0:  # Skip first year (no YoY data)
            print(f"{int(row['Year'])}: Starting Year")
            print(f"  Portfolio Value: ${row['portfolio_value']:,.2f}")
            continue
            
        spy_yoy = row['spy_yoy'] * 100
        strategy_yoy = row['strategy_yoy'] * 100
        outperformance = strategy_yoy - spy_yoy
        
        # Color coding for performance
        if outperformance > 20:
            status = "🔥 CRUSHING IT"
        elif outperformance > 10:
            status = "🚀 STRONG"
        elif outperformance > 0:
            status = "✅ WINNING"
        elif outperformance > -5:
            status = "⚠️ SLIGHT LAG"
        else:
            status = "❌ UNDERPERFORM"
            
        print(f"\n{int(row['Year'])} {status}:")
        print(f"  SPY YoY Return:      {spy_yoy:+6.1f}%")
        print(f"  Strategy YoY Return: {strategy_yoy:+6.1f}%")
        print(f"  Outperformance:      {outperformance:+6.1f}%")
        print(f"  Portfolio Value:     ${row['portfolio_value']:,.2f}")
    
    # Split performance into periods
    print(f"\n{'='*70}")
    print("PERFORMANCE BY PERIOD:")
    
    # Pre-2020 vs Post-2020
    pre_2020 = annual_data[annual_data['Year'] < 2020]
    post_2020 = annual_data[annual_data['Year'] >= 2020]
    
    if len(pre_2020) > 1:
        pre_2020_outperf = pre_2020['strategy_yoy'] - pre_2020['spy_yoy']
        pre_2020_avg_outperf = pre_2020_outperf.mean() * 100
        print(f"2011-2019 Average Outperformance: {pre_2020_avg_outperf:+.1f}%")
    
    if len(post_2020) > 0:
        post_2020_outperf = post_2020['strategy_yoy'] - post_2020['spy_yoy'] 
        post_2020_avg_outperf = post_2020_outperf.mean() * 100
        print(f"2020-2023 Average Outperformance: {post_2020_avg_outperf:+.1f}%")
    
    # Calculate overall metrics
    strategy_yoy_clean = annual_data['strategy_yoy'].dropna()
    spy_yoy_clean = annual_data['spy_yoy'].dropna()
    
    avg_spy_annual = spy_yoy_clean.mean() * 100
    avg_strategy_annual = strategy_yoy_clean.mean() * 100
    avg_outperformance = avg_strategy_annual - avg_spy_annual
    
    # Win rate
    outperformance_comparison = strategy_yoy_clean > spy_yoy_clean
    outperformance_years = outperformance_comparison.sum()
    total_years = len(outperformance_comparison)
    annual_win_rate = outperformance_years / total_years if total_years > 0 else 0
    
    print(f"\n{'='*70}")
    print("OVERALL ANNUAL PERFORMANCE SUMMARY:")
    print(f"SPY Average Annual Return:        {avg_spy_annual:6.1f}%")
    print(f"Strategy Average Annual Return:   {avg_strategy_annual:6.1f}%")
    print(f"Average Annual Outperformance:    {avg_outperformance:+6.1f}%")
    print(f"Years Beating SPY:               {outperformance_years}/{total_years} ({annual_win_rate:.1%})")
    
    # Calculate CAGR for full period
    years = len(annual_data) - 1
    if years > 0:
        start_price = annual_data[price_col].iloc[0]
        end_price = annual_data[price_col].iloc[-1]
        start_portfolio = annual_data['cumulative_strategy_returns'].iloc[0]
        end_portfolio = annual_data['cumulative_strategy_returns'].iloc[-1]
        
        spy_cagr = ((end_price / start_price) ** (1/years) - 1) * 100
        strategy_cagr = ((end_portfolio / start_portfolio) ** (1/years) - 1) * 100
        
        print(f"\nCOMPOUND ANNUAL GROWTH RATES ({annual_data['Year'].iloc[0]:.0f}-{annual_data['Year'].iloc[-1]:.0f}):")
        print(f"SPY CAGR:                         {spy_cagr:6.1f}%")
        print(f"Strategy CAGR:                    {strategy_cagr:6.1f}%")
        print(f"CAGR Outperformance:              {strategy_cagr - spy_cagr:+6.1f}%")
    
    # Performance consistency analysis
    print(f"\nRISK ANALYSIS:")
    print(f"Strategy Annual Volatility:       {strategy_yoy_clean.std():.1f}%")
    print(f"SPY Annual Volatility:            {spy_yoy_clean.std():.1f}%")
    print(f"Strategy Best Year:               {strategy_yoy_clean.max():.1f}%")
    print(f"Strategy Worst Year:              {strategy_yoy_clean.min():.1f}%")
    print(f"Negative Years:                   {(strategy_yoy_clean < 0).sum()}")
    
    return annual_data

# Run the analysis
print("\n" + "="*80)
annual_performance_extended = calculate_yoy_performance_extended(data_result)
print("="*80)