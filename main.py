import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

# Load Fear & Greed Index Data
fear_greed_file = "datasets/fear_greed_combined_2011_2025.csv"
fear_greed_data = pd.read_csv(fear_greed_file)
fear_greed_data['Date'] = pd.to_datetime(fear_greed_data['Date'])
fear_greed_data = fear_greed_data[['Date', 'fear_greed']]

# Data range
start_date = '2011-01-03'
end_date = '2025-08-09'
print("Start Date:", start_date, "End Date:", end_date)


spy_data = yf.download('SPY', start=start_date, end=end_date, auto_adjust=True)
if spy_data.columns.nlevels > 1:
    spy_data.columns = spy_data.columns.droplevel(1)
spy_data.reset_index(inplace=True)

print(spy_data)

merged_data = pd.merge(spy_data, fear_greed_data[['Date', 'fear_greed']], on='Date', how='left')
merged_data['fear_greed'] = merged_data['fear_greed'].ffill()
price_col = 'Close'


def strategy(data, momentum_threshold=1.0, velocity_threshold=0.3, 
             max_days_held=8, volatility_buy_limit=0.6, volatility_sell_limit=0.5,
             lookback_days=3):
    
    data = data.copy()
    price_col = 'Close'
    
    # Fear & Greed momentum and velocity calculations
    data['fg_momentum'] = data['fear_greed'] - data['fear_greed'].rolling(lookback_days, min_periods=1).mean()
    data['fg_change'] = data['fear_greed'].diff().fillna(0)
    data['fg_velocity'] = data['fg_change'].rolling(lookback_days, min_periods=1).mean()
    
    # Price analysis for volatility
    data['price_returns'] = data[price_col].pct_change().fillna(0)
    data['volatility'] = data['price_returns'].rolling(20, min_periods=1).std() * np.sqrt(252)
    
    # Initialize tracking columns
    data['position'] = 0
    data['signal'] = 0
    data['days_in_position'] = 0
    
    current_position = 0
    days_held = 0
    buy_signals = []
    sell_signals = []
    
    for i in range(len(data)):
        
        fg_momentum = data.iloc[i]['fg_momentum']
        fg_velocity = data.iloc[i]['fg_velocity']
        volatility = data.iloc[i]['volatility']
        
        # Buy conditions
        buy_signal = (
            current_position == 0 and
            fg_momentum > momentum_threshold and
            fg_velocity > velocity_threshold and
            (pd.isna(volatility) or volatility < volatility_buy_limit)
        )
        
        # Sell conditions
        sell_signal = (
            current_position == 1 and (
                fg_momentum < -momentum_threshold or
                fg_velocity < -velocity_threshold or
                days_held >= max_days_held or
                (pd.notna(volatility) and volatility > volatility_sell_limit)
            )
        )
        
        # Execute trades
        if buy_signal:
            data.iloc[i, data.columns.get_loc('signal')] = 1
            current_position = 1
            days_held = 0
            buy_signals.append(i)
            
        elif sell_signal:
            data.iloc[i, data.columns.get_loc('signal')] = -1
            current_position = 0
            days_held = 0
            sell_signals.append(i)
        
        # Update position tracking
        if current_position == 1:
            days_held += 1
            
        data.iloc[i, data.columns.get_loc('position')] = current_position
        data.iloc[i, data.columns.get_loc('days_in_position')] = days_held
    
    # Force close any open position at the end
    if current_position == 1 and len(buy_signals) > len(sell_signals):
        sell_signals.append(len(data) - 1)  # Close at last day
        data.iloc[-1, data.columns.get_loc('signal')] = -1
    
    return data, buy_signals, sell_signals


# Updated parameters to match run_fgi_strategy defaults
strategy_params = {
    'momentum_threshold': 1.0,
    'velocity_threshold': 0.3,
    'max_days_held': 8,
    'volatility_buy_limit': 0.6,
    'volatility_sell_limit': 0.5,
    'lookback_days': 3,
}

# Run strategy
data_result, buy_signals, sell_signals = strategy(merged_data, **strategy_params)


data_result['price_returns'].to_csv('results.csv')

# Fix for unmatched signals
min_length = min(len(buy_signals), len(sell_signals))
buy_signals_matched = buy_signals[:min_length]
sell_signals_matched = sell_signals[:min_length]

trades = pd.DataFrame({
    'buy_signals': buy_signals_matched,
    'sell_signals': sell_signals_matched
})

# Example federal + California brackets (2024) for single filer
federal_brackets = [
    (0, 11000, 0.10),
    (11000, 44725, 0.12),
    (44725, 95375, 0.22),
    (95375, 182100, 0.24),
    (182100, 231250, 0.32),
    (231250, 578125, 0.35),
    (578125, float('inf'), 0.37)
]

california_brackets = [
    (0, 10275, 0.01),
    (10275, 24475, 0.02),
    (24475, 37725, 0.04),
    (37725, 52475, 0.06),
    (52475, 66295, 0.08),
    (66295, 338639, 0.093),
    (338639, 406364, 0.103),
    (406364, 677275, 0.113),
    (677275, float('inf'), 0.123)
]

def calculate_total_tax(income, federal_brackets, state_brackets):
    """
    Calculate total tax (federal + state) for a given income
    """
    federal_tax = 0
    for start, end, rate in federal_brackets:
        if income > start:
            taxable_in_bracket = min(end, income) - start
            if taxable_in_bracket > 0:
                federal_tax += taxable_in_bracket * rate
    
    state_tax = 0
    for start, end, rate in state_brackets:
        if income > start:
            taxable_in_bracket = min(end, income) - start
            if taxable_in_bracket > 0:
                state_tax += taxable_in_bracket * rate
    
    return federal_tax + state_tax

def calculate_tax_on_gain(gain, base_income, federal_brackets, state_brackets):
    """
    Calculate total tax on short-term capital gains (taxed as ordinary income)
    Returns the additional tax caused by the gain
    """
    # Calculate tax at base income level
    base_tax = calculate_total_tax(base_income, federal_brackets, state_brackets)
    
    # Calculate tax at base income + gain level
    total_tax = calculate_total_tax(base_income + gain, federal_brackets, state_brackets)
    
    # The tax attributable to the gain is the difference
    return total_tax - base_tax

initial_capital = 10000
capital = initial_capital
leverage = 2.0
num_wins = 0
base_income = 10000

# Track portfolio values for drawdown calculation
portfolio_values = [initial_capital]
trade_dates = [merged_data['Date'].iloc[0]]

print(f"Initial Capital: ${initial_capital:,.2f}")
print(f"Base Income: ${base_income:,.2f}")
print(f"Leverage: {leverage}x")
print("-" * 100)

for i in range(len(trades)):
    buy_index = trades['buy_signals'][i]
    sell_index = trades['sell_signals'][i]
    buy_price = merged_data['Close'][buy_index]
    sell_price = merged_data['Close'][sell_index]
    buy_date = merged_data['Date'][buy_index]
    sell_date = merged_data['Date'][sell_index]
    shares = (capital * leverage) / buy_price

    raw_pnl = (sell_price - buy_price) * shares
    
    # Calculate tax on the gain (only on positive gains)
    if raw_pnl > 0:
        tax_amount = calculate_tax_on_gain(raw_pnl, base_income, federal_brackets, california_brackets)
    else:
        tax_amount = 0  # No tax on losses
    
    pnl_after_tax = raw_pnl - tax_amount

    if pnl_after_tax > 0:
        num_wins += 1
        win_loss = "WIN"
    else:
        win_loss = "LOSS"
    
    capital += pnl_after_tax
    portfolio_values.append(capital)
    trade_dates.append(sell_date)
    
    # Condensed print statement with dates
    print(f"Trade {i+1:2d}: {buy_date.strftime('%Y-%m-%d')} -> {sell_date.strftime('%Y-%m-%d')} | "
          f"Buy: ${buy_price:.2f}, Sell: ${sell_price:.2f} ({((sell_price-buy_price)/buy_price*100):+5.1f}%) | "
          f"Shares: {shares:.0f} | "
          f"PnL: ${raw_pnl:+,.0f} -> ${pnl_after_tax:+,.0f} after tax | "
          f"Portfolio: ${capital:,.0f} | {win_loss}")

# Calculate maximum drawdown
def calculate_max_drawdown(portfolio_values):
    """
    Calculate maximum drawdown from portfolio values
    """
    portfolio_series = pd.Series(portfolio_values)
    running_max = portfolio_series.expanding().max()
    drawdown = (portfolio_series - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Find the peak and trough values
    max_dd_idx = drawdown.idxmin()
    peak_idx = running_max[:max_dd_idx+1].idxmax()
    
    peak_value = portfolio_values[peak_idx]
    trough_value = portfolio_values[max_dd_idx]
    
    return max_drawdown, peak_value, trough_value, peak_idx, max_dd_idx

max_drawdown, peak_value, trough_value, peak_idx, trough_idx = calculate_max_drawdown(portfolio_values)

print("=" * 100)
print("FINAL RESULTS:")
print("=" * 100)
print(f"Percent Change: {((capital - initial_capital) / initial_capital * 100):+.2f}%")
print(f"Final Capital: ${capital:,.2f}")
print(f"Win Rate: {round((num_wins / len(trades)) * 100, 2)}%")
print(f"Total Trades: {len(trades)}")
print(f"Winning Trades: {num_wins}")
print(f"Losing Trades: {len(trades) - num_wins}")
print(f"Maximum Drawdown: {max_drawdown:.2%}")
print(f"Peak Portfolio Value: ${peak_value:,.2f} (Trade {peak_idx})")
print(f"Trough Portfolio Value: ${trough_value:,.2f} (Trade {trough_idx})")
if len(trade_dates) > peak_idx and len(trade_dates) > trough_idx:
    print(f"Drawdown Period: {trade_dates[peak_idx].strftime('%Y-%m-%d')} to {trade_dates[trough_idx].strftime('%Y-%m-%d')}")