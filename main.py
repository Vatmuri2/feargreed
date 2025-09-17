"""
Fear & Greed Index Trading Strategy - Live Execution Simulation

Backtests a momentum-based strategy using CNN Fear & Greed Index data to trade
a synthetic 2x leveraged S&P 500 ETF. Simulates real-world execution at 12:50 PM PST
(10 minutes before market close) with comprehensive tax calculations.

Strategy: Enter long positions when Fear & Greed momentum and velocity exceed 
thresholds with low volatility. Exit on momentum reversal or risk management triggers.

Author: Vikram Atmuri | Date: 15 September 2025
"""

import pandas as pd
import yfinance as yf
import numpy as np

# =============================================================================
# DATA LOADING
# =============================================================================

fear_greed_file = "datasets/fear_greed_combined_2011_2025.csv"
fear_greed_data = pd.read_csv(fear_greed_file)
fear_greed_data['Date'] = pd.to_datetime(fear_greed_data['Date'])

start_date = '2011-01-03'
end_date = '2025-09-12'
print("Start Date:", start_date, "End Date:", end_date)

# Download SPY data and merge with Fear & Greed Index
spy_data = yf.download('SPY', start=start_date, end=end_date, auto_adjust=True)
if spy_data.columns.nlevels > 1:
    spy_data.columns = spy_data.columns.droplevel(1)
spy_data.reset_index(inplace=True)

print(spy_data)

merged_data = pd.merge(spy_data, fear_greed_data[['Date', 'fear_greed']], on='Date', how='left')
merged_data['fear_greed'] = merged_data['fear_greed'].ffill()  # Fill weekends/holidays

# =============================================================================
# TRADING STRATEGY
# =============================================================================

def strategy(data, momentum_threshold=1.0, velocity_threshold=0.3, 
             max_days_held=8, volatility_buy_limit=0.6, volatility_sell_limit=0.5,
             lookback_days=3):
    """
    Fear & Greed momentum strategy with volatility-based risk management
    
    Entry: F&G momentum > threshold AND velocity > threshold AND volatility < buy_limit
    Exit: momentum < -threshold OR velocity < -threshold OR max_days OR volatility > sell_limit
    """
    
    data = data.copy()
    price_col = 'Close'
    
    # Calculate Fear & Greed indicators
    data['fg_momentum'] = data['fear_greed'] - data['fear_greed'].rolling(lookback_days, min_periods=1).mean()
    data['fg_change'] = data['fear_greed'].diff().fillna(0)
    data['fg_velocity'] = data['fg_change'].rolling(lookback_days, min_periods=1).mean()
    
    # Calculate 20-day annualized volatility for risk management
    data['price_returns'] = data[price_col].pct_change().fillna(0)
    data['volatility'] = data['price_returns'].rolling(20, min_periods=1).std() * np.sqrt(252)
    
    # Initialize position tracking
    data['position'] = 0
    data['signal'] = 0
    data['days_in_position'] = 0
    
    current_position = 0
    days_held = 0
    buy_signals = []
    sell_signals = []
    
    # Strategy execution loop
    for i in range(len(data)):
        fg_momentum = data.iloc[i]['fg_momentum']
        fg_velocity = data.iloc[i]['fg_velocity']
        volatility = data.iloc[i]['volatility']
        
        # Entry conditions
        buy_signal = (
            current_position == 0 and
            fg_momentum > momentum_threshold and
            fg_velocity > velocity_threshold and
            (pd.isna(volatility) or volatility < volatility_buy_limit)
        )
        
        # Exit conditions
        sell_signal = (
            current_position == 1 and (
                fg_momentum < -momentum_threshold or
                fg_velocity < -velocity_threshold or
                days_held >= max_days_held or
                (pd.notna(volatility) and volatility > volatility_sell_limit)
            )
        )
        
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
        
        if current_position == 1:
            days_held += 1
            
        data.iloc[i, data.columns.get_loc('position')] = current_position
        data.iloc[i, data.columns.get_loc('days_in_position')] = days_held
    
    # Close final position if needed
    if current_position == 1 and len(buy_signals) > len(sell_signals):
        sell_signals.append(len(data) - 1)
        data.iloc[-1, data.columns.get_loc('signal')] = -1
    
    return data, buy_signals, sell_signals

# =============================================================================
# STRATEGY EXECUTION
# =============================================================================

strategy_params = {
    'momentum_threshold': 1.0,
    'velocity_threshold': 0.3,
    'max_days_held': 8,
    'volatility_buy_limit': 0.6,
    'volatility_sell_limit': 0.5,
    'lookback_days': 3,
}

data_result, buy_signals, sell_signals = strategy(merged_data, **strategy_params)

# Match buy/sell signals and create trade pairs
min_length = min(len(buy_signals), len(sell_signals))
trades = pd.DataFrame({
    'buy_signals': buy_signals[:min_length],
    'sell_signals': sell_signals[:min_length]
})

# =============================================================================
# TAX CALCULATIONS
# =============================================================================

# 2024 tax brackets for single filer (short-term capital gains = ordinary income)
federal_brackets = [
    (0, 11000, 0.10), (11000, 44725, 0.12), (44725, 95375, 0.22),
    (95375, 182100, 0.24), (182100, 231250, 0.32), 
    (231250, 578125, 0.35), (578125, float('inf'), 0.37)
]

california_brackets = [
    (0, 10275, 0.01), (10275, 24475, 0.02), (24475, 37725, 0.04),
    (37725, 52475, 0.06), (52475, 66295, 0.08), (66295, 338639, 0.093),
    (338639, 406364, 0.103), (406364, 677275, 0.113), (677275, float('inf'), 0.123)
]

def calculate_total_tax(income, federal_brackets, state_brackets):
    """Calculate progressive tax on total income"""
    total_tax = 0
    for brackets in [federal_brackets, state_brackets]:
        for start, end, rate in brackets:
            if income > start:
                taxable_in_bracket = min(end, income) - start
                if taxable_in_bracket > 0:
                    total_tax += taxable_in_bracket * rate
    return total_tax

def calculate_tax_on_gain(gain, base_income, federal_brackets, state_brackets):
    """Calculate marginal tax on capital gain added to base income"""
    base_tax = calculate_total_tax(base_income, federal_brackets, state_brackets)
    total_tax = calculate_total_tax(base_income + gain, federal_brackets, state_brackets)
    return total_tax - base_tax

# =============================================================================
# BACKTESTING EXECUTION
# =============================================================================

initial_capital = 10000
capital = initial_capital
num_wins = 0
base_income = 50000  # Base income for tax calculations

portfolio_values = [initial_capital]
trade_dates = [merged_data['Date'].iloc[0]]

print(f"Initial Capital: ${initial_capital:,.2f}")
print(f"Base Income: ${base_income:,.2f}")
print("-" * 100)

# Process each trade
for i in range(len(trades)):
    buy_index = trades['buy_signals'][i]
    sell_index = trades['sell_signals'][i]
    buy_date = merged_data['Date'][buy_index]
    sell_date = merged_data['Date'][sell_index]
    
    # Calculate synthetic 2x leveraged ETF return
    holding_period_data = merged_data.loc[buy_index:sell_index].copy()
    holding_period_data['spy_daily_return'] = holding_period_data['Close'].pct_change().fillna(0)
    synthetic_2x_return = (1 + 2 * holding_period_data['spy_daily_return']).prod() - 1
    raw_pnl = capital * synthetic_2x_return
    
    # Apply taxes only on gains
    if raw_pnl > 0:
        tax_amount = calculate_tax_on_gain(raw_pnl, base_income, federal_brackets, california_brackets)
    else:
        tax_amount = 0
    
    pnl_after_tax = raw_pnl - tax_amount
    
    if pnl_after_tax > 0:
        num_wins += 1
        win_loss = "WIN"
    else:
        win_loss = "LOSS"
    
    capital += pnl_after_tax
    portfolio_values.append(capital)
    trade_dates.append(sell_date)
    
    print(f"Trade {i+1:2d}: {buy_date.strftime('%Y-%m-%d')} -> {sell_date.strftime('%Y-%m-%d')} | "
          f"Return: {synthetic_2x_return*100:+.1f}% | "
          f"PnL: ${raw_pnl:+,.0f} -> ${pnl_after_tax:+,.0f} after tax | "
          f"Portfolio: ${capital:,.0f} | {win_loss}")

# =============================================================================
# PERFORMANCE ANALYTICS
# =============================================================================

def calculate_max_drawdown(portfolio_values):
    """Calculate maximum peak-to-trough decline"""
    portfolio_series = pd.Series(portfolio_values)
    running_max = portfolio_series.expanding().max()
    drawdown = (portfolio_series - running_max) / running_max
    max_drawdown = drawdown.min()
    
    max_dd_idx = drawdown.idxmin()
    peak_idx = running_max[:max_dd_idx+1].idxmax()
    
    return max_drawdown, portfolio_values[peak_idx], portfolio_values[max_dd_idx], peak_idx, max_dd_idx

def calculate_sharpe_ratio(portfolio_values, trade_dates, risk_free_rate=0.05):
    """Calculate annualized risk-adjusted returns"""
    if len(portfolio_values) < 2:
        return 0, 0, 0
    
    portfolio_series = pd.Series(portfolio_values)
    returns = portfolio_series.pct_change().dropna()
    
    if len(returns) == 0:
        return 0, 0, 0
    
    # Annualize metrics
    total_days = (trade_dates[-1] - trade_dates[0]).days
    years = total_days / 365.25
    
    total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
    annual_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
    
    if len(returns) > 1:
        return_volatility = returns.std()
        trades_per_year = len(returns) / years if years > 0 else len(returns)
        annual_volatility = return_volatility * np.sqrt(trades_per_year)
    else:
        annual_volatility = 0
    
    excess_return = annual_return - risk_free_rate
    sharpe_ratio = excess_return / annual_volatility if annual_volatility > 0 else 0
    
    return sharpe_ratio, annual_return, annual_volatility

# Calculate metrics
max_drawdown, peak_value, trough_value, peak_idx, trough_idx = calculate_max_drawdown(portfolio_values)
sharpe_ratio, annual_return, annual_volatility = calculate_sharpe_ratio(portfolio_values, trade_dates)

# =============================================================================
# RESULTS SUMMARY
# =============================================================================

print("=" * 100)
print("FINAL RESULTS:")
print("=" * 100)
print(f"Percent Change: {((capital - initial_capital) / initial_capital * 100):+.2f}%")
print(f"Final Capital: ${capital:,.2f}")
print(f"Annualized Return: {annual_return * 100:.2f}%")
print(f"Annualized Volatility: {annual_volatility * 100:.2f}%")
print(f"Sharpe Ratio: {sharpe_ratio:.3f}")
print(f"Win Rate: {round((num_wins / len(trades)) * 100, 2)}%")
print(f"Total Trades: {len(trades)}")
print(f"Winning Trades: {num_wins}")
print(f"Losing Trades: {len(trades) - num_wins}")
print(f"Maximum Drawdown: {max_drawdown:.2%}")
print(f"Peak Portfolio Value: ${peak_value:,.2f} (Trade {peak_idx})")
print(f"Trough Portfolio Value: ${trough_value:,.2f} (Trade {trough_idx})")

if len(trade_dates) > peak_idx and len(trade_dates) > trough_idx:
    print(f"Drawdown Period: {trade_dates[peak_idx].strftime('%Y-%m-%d')} to {trade_dates[trough_idx].strftime('%Y-%m-%d')}")