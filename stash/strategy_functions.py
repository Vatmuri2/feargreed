import numpy as np
import pandas as pd

# --- FGI STRATEGY ---
def run_fgi_strategy(
    merged_data,
    initial_capital=10000,
    leverage=2.0,
    do_tax=False,
    base_income=25000,
    federal_brackets=None,
    state_brackets=None,
    momentum_threshold=1.0,
    velocity_threshold=0.3,
    max_days_held=8,
    volatility_buy_limit=0.6,
    volatility_sell_limit=0.5
):
    dates = merged_data['Date']
    prices = merged_data['Close']

    fg = merged_data['fear_greed']
    fg_momentum = fg - fg.rolling(3, min_periods=1).mean()
    fg_change = fg.diff().fillna(0)
    fg_velocity = fg_change.rolling(3, min_periods=1).mean()
    price_ret = prices.pct_change().fillna(0)
    volatility = price_ret.rolling(20, min_periods=1).std() * (252 ** 0.5)

    n = len(merged_data)
    portfolio = []
    position = 0
    capital = initial_capital
    days_held = 0
    buy_price = 0
    shares = 0

    for i in range(n):
        buy_signal = (
            position == 0 and
            fg_momentum.iloc[i] > momentum_threshold and
            fg_velocity.iloc[i] > velocity_threshold and
            (pd.isna(volatility.iloc[i]) or volatility.iloc[i] < volatility_buy_limit)
        )
        sell_signal = (
            position == 1 and (
                fg_momentum.iloc[i] < -momentum_threshold or
                fg_velocity.iloc[i] < -velocity_threshold or
                days_held >= max_days_held or
                (pd.notna(volatility.iloc[i]) and volatility.iloc[i] > volatility_sell_limit)
            )
        )
        if buy_signal and position == 0:
            buy_price = prices.iloc[i]
            shares = (capital * leverage) / buy_price
            position = 1
            days_held = 0
        elif sell_signal and position == 1:
            sell_price = prices.iloc[i]
            raw_pnl = (sell_price - buy_price) * shares
            
            # Only apply tax to gains (positive PnL)
            if do_tax and federal_brackets and state_brackets and raw_pnl > 0:
                tax_amount = calculate_tax_on_gain(
                    raw_pnl, base_income, federal_brackets, state_brackets
                )
                pnl_after_tax = raw_pnl - tax_amount
            else:
                pnl_after_tax = raw_pnl  # No tax on losses
                
            capital += pnl_after_tax
            position = 0
            days_held = 0
            buy_price = 0
            shares = 0

        if position == 1:
            days_held += 1
            equity = capital + (prices.iloc[i] - buy_price) * shares
        else:
            equity = capital
        portfolio.append(equity)
    return pd.Series(portfolio, index=dates)

# --- RSI30/70 STRATEGY WITH TAX ---
def run_rsi_strategy(price_data, initial_capital=10000, leverage=2.0, do_tax=False, base_income=25000, federal_brackets=None, state_brackets=None):
    from ta.momentum import RSIIndicator
    rsi = RSIIndicator(close=price_data['Close'], window=14).rsi()
    positions = np.where(rsi < 30, 1, np.where(rsi > 70, 0, np.nan))
    positions = pd.Series(positions).ffill().fillna(0)
    capital = initial_capital
    curve = []
    in_pos = False
    buy_price = None
    shares = 0
    for i in range(len(price_data)):
        if positions.iloc[i] == 1 and not in_pos:  # Buy signal
            buy_price = price_data['Close'].iloc[i]
            shares = (capital * leverage) / buy_price
            in_pos = True
        elif in_pos and positions.iloc[i] == 0:  # Sell signal
            sell_price = price_data['Close'].iloc[i]
            raw_pnl = (sell_price - buy_price) * shares
            
            # Only apply tax to gains (positive PnL)
            if do_tax and federal_brackets and state_brackets and raw_pnl > 0:
                tax_amount = calculate_tax_on_gain(
                    raw_pnl, base_income, federal_brackets, state_brackets
                )
                pnl_after_tax = raw_pnl - tax_amount
            else:
                pnl_after_tax = raw_pnl  # No tax on losses
                
            capital += pnl_after_tax
            in_pos = False
            buy_price = None
            shares = 0
        if in_pos:
            equity = capital + (price_data['Close'].iloc[i] - buy_price) * shares
        else:
            equity = capital
        curve.append(equity)
    return pd.Series(curve, index=price_data['Date'])

# --- BUY AND HOLD SPY STRATEGY ---
def run_buy_and_hold_curve(prices_df, initial_capital=10000, do_tax=False, base_income=25000, federal_brackets=None, state_brackets=None):
    """
    Buys SPY at the first date, never sells until the last.
    Optionally applies tax only at the end.
    """
    prices = prices_df['Close']
    dates = prices_df['Date']
    start_price = prices.iloc[0]
    shares = initial_capital / start_price
    curve = initial_capital * (prices / start_price)
    
    if do_tax and federal_brackets and state_brackets:
        end_price = prices.iloc[-1]
        gross_gain = (end_price - start_price) * shares
        
        # Only apply tax if there's a gain
        if gross_gain > 0:
            tax_amount = calculate_tax_on_gain(
                gross_gain, base_income, federal_brackets, state_brackets
            )
            final_value = initial_capital + gross_gain - tax_amount
        else:
            # No tax on losses
            final_value = initial_capital + gross_gain
            
        # Make curve match the pre-tax value, but the last day is after-tax
        curve.iloc[-1] = final_value
    return pd.Series(curve.values, index=dates)

# --- RISK FREE CURVE ---
def run_risk_free_curve(dates, initial_capital=10000, rate=0.045):
    n_days = len(dates)
    daily_return = (1 + rate) ** (1/252) - 1
    curve = initial_capital * (1 + daily_return) ** np.arange(n_days)
    return pd.Series(curve, index=dates)

# --- DRAWDOWN FUNCTION ---
def calculate_drawdown(series):
    peak = series.cummax()
    drawdown = (series - peak)/peak
    return drawdown

def calculate_total_tax(income, federal_brackets, state_brackets):
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
    base_tax = calculate_total_tax(base_income, federal_brackets, state_brackets)
    total_tax = calculate_total_tax(base_income + gain, federal_brackets, state_brackets)
    return total_tax - base_tax