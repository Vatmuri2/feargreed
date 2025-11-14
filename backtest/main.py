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
# DATA LOADING AND PREPROCESSING SECTION
# =============================================================================

"""
This section handles:
1. Loading the Fear & Greed Index historical data from CSV
2. Downloading SPY ETF price data from Yahoo Finance
3. Merging datasets and handling missing values
4. Setting up the date range for backtesting
"""

# Load Fear & Greed Index Data
fear_greed_file = "datasets/fear_greed_combined_2011_2025.csv"
fear_greed_data = pd.read_csv(fear_greed_file)
fear_greed_data['Date'] = pd.to_datetime(fear_greed_data['Date'])
fear_greed_data = fear_greed_data[['Date', 'fear_greed']]

# Define backtesting period
start_date = '2025-09-18'
end_date = '2025-10-24'
print("Start Date:", start_date, "End Date:", end_date)

# Download SPY ETF data from Yahoo Finance
spy_data = yf.download('SPY', start=start_date, end=end_date, auto_adjust=True)
if spy_data.columns.nlevels > 1:
    spy_data.columns = spy_data.columns.droplevel(1)
spy_data.reset_index(inplace=True)

print(spy_data)

# Merge SPY price data with Fear & Greed Index data
# Forward fill missing Fear & Greed values for weekends/holidays
merged_data = pd.merge(spy_data, fear_greed_data[['Date', 'fear_greed']], on='Date', how='left')
merged_data['fear_greed'] = merged_data['fear_greed'].ffill()
price_col = 'Open'

# =============================================================================
# TRADING STRATEGY IMPLEMENTATION SECTION
# =============================================================================

"""
This section contains the core trading strategy logic:
1. Fear & Greed momentum and velocity calculations
2. Volatility-based risk management
3. Position sizing and holding period limits
4. Signal generation (buy/sell conditions)

Strategy Logic:
- BUY when Fear & Greed momentum and velocity exceed thresholds AND volatility is low
- SELL when momentum/velocity reverse, max holding period reached, or volatility spikes
"""

def strategy(data, momentum_threshold=1.0, velocity_threshold=0.3, 
             max_days_held=8, volatility_buy_limit=0.6, volatility_sell_limit=0.5,
             lookback_days=3):
    """
    Implements the Fear & Greed Index trading strategy
    
    Parameters:
    - momentum_threshold: Minimum F&G momentum for buy signal
    - velocity_threshold: Minimum F&G velocity for buy signal  
    - max_days_held: Maximum days to hold a position
    - volatility_buy_limit: Max volatility to allow new positions
    - volatility_sell_limit: Volatility level that triggers sell
    - lookback_days: Rolling window for momentum/velocity calculations
    
    Returns:
    - Dataset with signals and position tracking
    - Lists of buy and sell signal indices
    """
    
    data = data.copy()
    price_col = 'Close'
    
    # Calculate Fear & Greed technical indicators
    # Momentum: Current F&G vs recent average
    data['fg_momentum'] = data['fear_greed'] - data['fear_greed'].rolling(lookback_days, min_periods=1).mean()
    
    # Velocity: Rate of change in F&G index
    data['fg_change'] = data['fear_greed'].diff().fillna(0)
    data['fg_velocity'] = data['fg_change'].rolling(lookback_days, min_periods=1).mean()
    
    # Calculate price volatility for risk management
    data['price_returns'] = data[price_col].pct_change().fillna(0)
    data['volatility'] = data['price_returns'].rolling(20, min_periods=1).std() * np.sqrt(252)
    
    # Initialize tracking columns for positions and signals
    data['position'] = 0
    data['signal'] = 0
    data['days_in_position'] = 0
    
    # State tracking variables
    current_position = 0  # 0 = no position, 1 = long position
    days_held = 0
    buy_signals = []
    sell_signals = []
    
    # Main strategy loop - process each trading day
    for i in range(len(data)):
        
        fg_momentum = data.iloc[i]['fg_momentum']
        fg_velocity = data.iloc[i]['fg_velocity']
        volatility = data.iloc[i]['volatility']
        
        # BUY SIGNAL CONDITIONS
        # Enter long position when:
        # 1. No current position
        # 2. F&G momentum is strongly positive
        # 3. F&G velocity is accelerating upward
        # 4. Market volatility is below risk threshold
        buy_signal = (
            current_position == 0 and
            fg_momentum > momentum_threshold and
            fg_velocity > velocity_threshold and
            (pd.isna(volatility) or volatility < volatility_buy_limit)
        )
        
        # SELL SIGNAL CONDITIONS  
        # Exit position when:
        # 1. Currently holding position AND
        # 2. (F&G momentum turns negative OR
        #     F&G velocity turns negative OR
        #     Maximum holding period reached OR
        #     Volatility exceeds risk threshold)
        sell_signal = (
            current_position == 1 and (
                fg_momentum < momentum_threshold or
                fg_velocity < velocity_threshold or
                days_held >= max_days_held or
                (pd.notna(volatility) and volatility > volatility_sell_limit)
            )
        )
        
        # Execute trading signals
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
    
    # Force close any open position at end of backtest period
    if current_position == 1 and len(buy_signals) > len(sell_signals):
        sell_signals.append(len(data) - 1)
        data.iloc[-1, data.columns.get_loc('signal')] = -1
    
    return data, buy_signals, sell_signals

# =============================================================================
# STRATEGY PARAMETER CONFIGURATION SECTION
# =============================================================================

"""
This section defines the strategy parameters that control:
- Signal sensitivity (momentum and velocity thresholds)
- Risk management (volatility limits and max holding period)
- Technical indicator calculations (lookback periods)
"""

# Strategy parameters - these can be optimized through backtesting
strategy_params = {
    'momentum_threshold': 0.2,      # F&G momentum threshold for entry
    'velocity_threshold': 0.15,       # F&G velocity threshold for entry
    'max_days_held': 8,             # Maximum days to hold any position
    'volatility_buy_limit': 0.6,    # Max volatility to allow new positions
    'volatility_sell_limit': 0.5,   # Volatility level that triggers exit
    'lookback_days': 3,             # Rolling window for calculations
}

# Execute the trading strategy
data_result, buy_signals, sell_signals = strategy(merged_data, **strategy_params)


# =============================================================================
# TRADE MATCHING AND VALIDATION SECTION  
# =============================================================================

"""
This section ensures that buy and sell signals are properly matched
to create valid trade pairs. Handles edge cases where signals might
be unbalanced due to strategy logic or data issues.
"""

# Ensure buy and sell signals are properly matched
# Handle cases where strategy might generate unequal signal counts
min_length = min(len(buy_signals), len(sell_signals))
buy_signals_matched = buy_signals[:min_length]
sell_signals_matched = sell_signals[:min_length]

# Create trade pairs dataframe
trades = pd.DataFrame({
    'buy_signals': buy_signals_matched,
    'sell_signals': sell_signals_matched
})

# =============================================================================
# TAX CALCULATION SETUP SECTION
# =============================================================================

"""
This section implements comprehensive tax calculations for short-term capital gains.
Uses actual 2024 federal and California tax brackets for accurate after-tax returns.

Tax Strategy:
- Short-term capital gains are taxed as ordinary income
- Progressive tax brackets mean marginal rates apply
- Only positive gains are subject to taxation
"""

# 2024 Federal tax brackets for single filer
federal_brackets = [
    (0, 11000, 0.10),           # 10% on first $11,000
    (11000, 44725, 0.12),       # 12% on $11,000 - $44,725
    (44725, 95375, 0.22),       # 22% on $44,725 - $95,375
    (95375, 182100, 0.24),      # 24% on $95,375 - $182,100
    (182100, 231250, 0.32),     # 32% on $182,100 - $231,250
    (231250, 578125, 0.35),     # 35% on $231,250 - $578,125
    (578125, float('inf'), 0.37) # 37% on $578,125+
]

# 2024 California tax brackets for single filer
california_brackets = [
    (0, 10275, 0.01),           # 1% on first $10,275
    (10275, 24475, 0.02),       # 2% on $10,275 - $24,475
    (24475, 37725, 0.04),       # 4% on $24,475 - $37,725
    (37725, 52475, 0.06),       # 6% on $37,725 - $52,475
    (52475, 66295, 0.08),       # 8% on $52,475 - $66,295
    (66295, 338639, 0.093),     # 9.3% on $66,295 - $338,639
    (338639, 406364, 0.103),    # 10.3% on $338,639 - $406,364
    (406364, 677275, 0.113),    # 11.3% on $406,364 - $677,275
    (677275, float('inf'), 0.123) # 12.3% on $677,275+
]

def calculate_total_tax(income, federal_brackets, state_brackets):
    """
    Calculate total tax (federal + state) for a given income level
    using progressive tax bracket system
    
    Parameters:
    - income: Total taxable income
    - federal_brackets: List of (min, max, rate) tuples for federal taxes
    - state_brackets: List of (min, max, rate) tuples for state taxes
    
    Returns:
    - Total tax amount (federal + state)
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
    Calculate marginal tax rate effect of capital gains on total tax burden
    
    Short-term capital gains are taxed as ordinary income, so they're added
    to base income and taxed at the marginal rate for that income level.
    
    Parameters:
    - gain: Capital gain amount
    - base_income: Base income before capital gains
    - federal_brackets: Federal tax bracket structure
    - state_brackets: State tax bracket structure
    
    Returns:
    - Additional tax burden caused by the capital gain
    """
    # Calculate tax at base income level
    base_tax = calculate_total_tax(base_income, federal_brackets, state_brackets)
    
    # Calculate tax at base income + gain level
    total_tax = calculate_total_tax(base_income + gain, federal_brackets, state_brackets)
    
    # The tax attributable to the gain is the difference
    return total_tax - base_tax

# =============================================================================
# BACKTESTING EXECUTION AND PERFORMANCE TRACKING SECTION
# =============================================================================

"""
This section executes the backtest by:
1. Processing each trade pair (buy/sell)
2. Calculating synthetic 2x leveraged ETF returns
3. Applying tax calculations to gains
4. Tracking portfolio value and performance metrics
5. Computing maximum drawdown analytics
"""

# Backtesting configuration
initial_capital = 10000        # Starting portfolio value
capital = initial_capital      # Current portfolio value
num_wins = 0                  # Count of profitable trades
base_income = 30000           # Assumed base income for tax calculations

# Track portfolio values over time for drawdown analysis
portfolio_values = [initial_capital]
trade_dates = [merged_data['Date'].iloc[0]]

print(f"Initial Capital: ${initial_capital:,.2f}")
print(f"Base Income: ${base_income:,.2f}")
print("-" * 100)

# Process each trade pair
for i in range(len(trades)):
    buy_index = trades['buy_signals'][i]
    sell_index = trades['sell_signals'][i]
    buy_date = merged_data['Date'][buy_index]
    sell_date = merged_data['Date'][sell_index]
    
    """
    Calculate synthetic 2x leveraged ETF returns:
    - Takes daily SPY returns during holding period
    - Applies 2x leverage to each daily return
    - Compounds the leveraged returns over the holding period
    
    Formula: (1 + 2 * daily_return1) * (1 + 2 * daily_return2) * ... - 1
    """
    holding_period_data = merged_data.loc[buy_index:sell_index].copy()
    holding_period_data['spy_daily_return'] = holding_period_data['Close'].pct_change().fillna(0)
    synthetic_2x_return = (1 + 2 * holding_period_data['spy_daily_return']).prod() - 1
    raw_pnl = capital * synthetic_2x_return
    
    # Apply tax calculations (only on positive gains)
    if raw_pnl > 0:
        tax_amount = calculate_tax_on_gain(raw_pnl, base_income, federal_brackets, california_brackets)
    else:
        tax_amount = 0  # No tax on losses
    
    pnl_after_tax = raw_pnl - tax_amount

    # Track win/loss statistics
    if pnl_after_tax > 0:
        num_wins += 1
        win_loss = "WIN"
    else:
        win_loss = "LOSS"
    
    # Update portfolio value and tracking arrays
    capital += pnl_after_tax
    portfolio_values.append(capital)
    trade_dates.append(sell_date)
    
    # Display trade results
    print(f"Trade {i+1:2d}: {buy_date.strftime('%Y-%m-%d')} -> {sell_date.strftime('%Y-%m-%d')} | "
          f"Return: {synthetic_2x_return*100:+.1f}% | "
          f"PnL: ${raw_pnl:+,.0f} -> ${pnl_after_tax:+,.0f} after tax | "
          f"Portfolio: ${capital:,.0f} | {win_loss}")

# =============================================================================
# RISK ANALYTICS AND PERFORMANCE METRICS SECTION
# =============================================================================

"""
This section calculates comprehensive performance and risk metrics:
1. Maximum Drawdown: Largest peak-to-trough decline
2. Win Rate: Percentage of profitable trades  
3. Total Return: Overall portfolio performance
4. Trade Statistics: Detailed breakdown of results
"""

def calculate_max_drawdown(portfolio_values):
    """
    Calculate maximum drawdown from portfolio value series
    
    Maximum drawdown measures the largest peak-to-trough decline
    in portfolio value, which is a key risk metric for trading strategies.
    
    Parameters:
    - portfolio_values: List of portfolio values over time
    
    Returns:
    - max_drawdown: Maximum drawdown as a percentage
    - peak_value: Portfolio value at the peak before max drawdown
    - trough_value: Portfolio value at the trough of max drawdown
    - peak_idx: Index of the peak
    - max_dd_idx: Index of the trough
    """
    portfolio_series = pd.Series(portfolio_values)
    running_max = portfolio_series.expanding().max()
    drawdown = (portfolio_series - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Identify peak and trough points for max drawdown
    max_dd_idx = drawdown.idxmin()
    peak_idx = running_max[:max_dd_idx+1].idxmax()
    
    peak_value = portfolio_values[peak_idx]
    trough_value = portfolio_values[max_dd_idx]
    
    return max_drawdown, peak_value, trough_value, peak_idx, max_dd_idx

# Calculate maximum drawdown and related metrics
max_drawdown, peak_value, trough_value, peak_idx, trough_idx = calculate_max_drawdown(portfolio_values)

def calculate_sharpe_ratio(portfolio_values, trade_dates, risk_free_rate=0.05):
    """
    Calculate the Sharpe ratio for the trading strategy
    
    Sharpe ratio = (Average Return - Risk Free Rate) / Standard Deviation of Returns
    
    The Sharpe ratio measures risk-adjusted returns by comparing the strategy's
    excess return over the risk-free rate to the volatility of those returns.
    Higher Sharpe ratios indicate better risk-adjusted performance.
    
    Parameters:
    - portfolio_values: List of portfolio values over time
    - trade_dates: Corresponding dates for each portfolio value
    - risk_free_rate: Annual risk-free rate (default 5% for current environment)
    
    Returns:
    - sharpe_ratio: Annualized Sharpe ratio
    - annual_return: Annualized return of the strategy
    - annual_volatility: Annualized volatility of returns
    """
    if len(portfolio_values) < 2:
        return 0, 0, 0
    
    # Convert portfolio values to returns
    portfolio_series = pd.Series(portfolio_values)
    returns = portfolio_series.pct_change().dropna()
    
    if len(returns) == 0:
        return 0, 0, 0
    
    # Calculate time period for annualization
    total_days = (trade_dates[-1] - trade_dates[0]).days
    years = total_days / 365.25
    
    # Calculate annualized metrics
    total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
    annual_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
    
    # Calculate volatility of returns and annualize
    if len(returns) > 1:
        return_volatility = returns.std()
        # Annualize based on trade frequency
        trades_per_year = len(returns) / years if years > 0 else len(returns)
        annual_volatility = return_volatility * np.sqrt(trades_per_year)
    else:
        annual_volatility = 0
    
    # Calculate Sharpe ratio
    excess_return = annual_return - risk_free_rate
    sharpe_ratio = excess_return / annual_volatility if annual_volatility > 0 else 0
    
    return sharpe_ratio, annual_return, annual_volatility

# Calculate Sharpe ratio and related metrics
sharpe_ratio, annual_return, annual_volatility = calculate_sharpe_ratio(portfolio_values, trade_dates)

# =============================================================================
# RESULTS REPORTING SECTION
# =============================================================================

"""
This section provides a comprehensive summary of backtest results including:
- Overall performance metrics
- Risk-adjusted returns  
- Trade statistics and win rates
- Maximum drawdown analysis with timing
"""

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

# Display drawdown timing if data is available
if len(trade_dates) > peak_idx and len(trade_dates) > trough_idx:
    print(f"Drawdown Period: {trade_dates[peak_idx].strftime('%Y-%m-%d')} to {trade_dates[trough_idx].strftime('%Y-%m-%d')}")

"""
END OF BACKTESTING SCRIPT

This script provides a complete backtesting framework for the Fear & Greed Index
trading strategy, including realistic tax considerations and comprehensive 
performance analytics. The modular structure allows for easy parameter 
optimization and strategy refinement.

Key Features:
- Momentum and velocity-based signal generation
- Risk management through volatility filters
- Realistic tax calculations for short-term gains
- Comprehensive performance and risk metrics
- Maximum drawdown analysis with timing details
"""