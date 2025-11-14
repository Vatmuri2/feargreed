import pandas as pd
import datetime
import time
import fear_and_greed as fg
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import APIError
import numpy as np
import pandas_market_calendars as mcal
import logging
import os
import pytz
import yfinance as yf

# =============================================================================
# CONFIGURATION
# =============================================================================
API_KEY = 'PKVVMQPZU4DBWK9DH51G'
API_SECRET = 'Nm1UpSPe7xzyYeQOtVGTSwUG2xbgMRstpfmsfGGb'
BASE_URL = 'https://paper-api.alpaca.markets'

TRADE_SYMBOL = 'SPY'  
LOG_FILE = 'trading_log_BOD.csv'
FG_PATH = 'datasets/fear_greed_forward_test_morning.csv'  # Path to historical F&G data


# Strategy Parameters
MOMENTUM_THRESHOLD = 0.2
VELOCITY_THRESHOLD = 0.15
VOLATILITY_BUY_LIMIT = 0.6
VOLATILITY_SELL_LIMIT = 0.5
LOOKBACK_DAYS = 3
VOLATILITY_WINDOW = 20  # Days for volatility calculation
MAX_DAYS_HELD = 8  # Max days to hold a position

# =============================================================================
# SETUP LOGGING
# =============================================================================
# Create a comprehensive log file if it doesn't exist
if not os.path.exists(LOG_FILE):
    log_df = pd.DataFrame(columns=[
        'Timestamp', 'Action', 'Symbol', 'Quantity', 'Price', 'FGI_Value',
        'FGI_Momentum', 'FGI_Velocity', 'Volatility', 'Portfolio_Value',
        'Buying_Power', 'Signal_Reason'
    ])
    log_df.to_csv(LOG_FILE, index=False)
    print(f"Created new log file: {LOG_FILE}")

# =============================================================================
# ALPACA API INITIALIZATION
# =============================================================================
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

def fetch_and_update_fgi():
    """Fetches the latest Fear & Greed Index value and appends it to the dataset."""
    try:
        # Get current F&G value, rating and date
        fg_index = fg.get()
        curr_fgi = round(fg_index.value, 2)  # ✅ Already floored as requested
        curr_rating = fg_index.description
        curr_date = datetime.date.today()
        
        # Load existing historical data
        fg_data = pd.read_csv(FG_PATH)
        fg_data['Date'] = pd.to_datetime(fg_data['Date']).dt.date
        
        # Check which column contains the F&G values
        if 'fear_greed' in fg_data.columns:
            fgi_column = 'fear_greed'
        elif 'Fear Greed' in fg_data.columns:
            fgi_column = 'Fear Greed'
        else:
            fgi_column = 'Index'
        
        # Check if we already have today's data
        if fg_data['Date'].iloc[-1] != curr_date:
            new_row = pd.DataFrame({
                'Date': [curr_date], 
                fgi_column: [curr_fgi],
                'rating': [curr_rating]
            })
            fg_data = pd.concat([fg_data, new_row], ignore_index=True)
            fg_data.to_csv(FG_PATH, index=False)
            print(f"Updated F&G Index: {curr_date} - {curr_fgi} ({curr_rating})")
        
        return fg_data, curr_fgi, fgi_column
        
    except Exception as e:
        print(f"Error updating F&G Index: {e}")
        try:
            fg_data = pd.read_csv(FG_PATH)
            fg_data['Date'] = pd.to_datetime(fg_data['Date']).dt.date
            
            if 'fear_greed' in fg_data.columns:
                fgi_column = 'fear_greed'
            elif 'Fear Greed' in fg_data.columns:
                fgi_column = 'Fear Greed'
            else:
                fgi_column = 'Index'
            print(fgi_column)
                
            last_fgi = fg_data[fgi_column].iloc[-1]
            return fg_data, last_fgi, fgi_column
        except:
            print("Fatal error: Could not load F&G data file.")
            return None, None, None

def calculate_indicators(fg_data, fgi_column):
    """Calculates trading indicators from Fear & Greed data only."""
    data = fg_data.copy()
    data['fg_momentum'] = data[fgi_column] - data[fgi_column].rolling(LOOKBACK_DAYS, min_periods=1).mean()
    data['fg_change'] = data[fgi_column].diff().fillna(0)
    data['fg_velocity'] = data['fg_change'].rolling(LOOKBACK_DAYS, min_periods=1).mean()
    return data

def get_current_volatility(symbol, window=VOLATILITY_WINDOW):
    """Calculates current volatility directly from the traded symbol's historical data."""
    try:
        # ✅ FIXED: Use correct timeframe format
        spy = yf.download("SPY", start="2025-08-01", end=datetime.date.today())
        spy['price_returns'] = spy['Close'].pct_change()
        spy['volatility'] = spy['price_returns'].rolling(20, min_periods=1).std() * np.sqrt(252)

        curr_volatility = spy['volatility'].iloc[-1]

        return round(curr_volatility, 4)
    except Exception as e:
        print(f"Error calculating volatility for {symbol}: {e}")
        return None

def get_current_position(symbol):
    """Checks if we have a position in the given symbol. Returns (True, qty) or (False, 0)."""
    try:
        position = api.get_position(symbol)
        return True, int(position.qty)
    except APIError as e:  # ✅ FIXED: Use correct exception class
        # 404 error means no position exists
        return False, 0

def get_position_entry_date(symbol):
    """
    Finds the entry date by looking at the most recent 'buy' order for this symbol
    that has been filled.
    """
    try:
        # Get orders for this symbol, sorted by most recent first
        orders = api.list_orders(
            status='closed',
            limit=100,
            direction='desc'
        )
        
        # Find the most recent filled buy order for this symbol
        for order in orders:
            if order.symbol == symbol and order.side == 'buy' and order.filled_qty:
                # Convert the order timestamp to a date
                order_date = pd.to_datetime(order.submitted_at).date()
                return order_date
                
        print(f"No buy orders found for {symbol}")
        return None
        
    except Exception as e:
        print(f"Error getting order history for {symbol}: {e}")
        return None
    
def generate_signal(fg_data, current_volatility):
    """Generates a BUY or SELL signal based on the latest calculated indicators."""
    latest_data = fg_data.iloc[-1]
    has_position, position_qty = get_current_position(TRADE_SYMBOL)
    
    # NEW: Check how long we've been in the position
    days_held = 0
    if has_position:
        entry_date = get_position_entry_date(TRADE_SYMBOL)
        if entry_date:
            days_held = (datetime.date.today() - entry_date).days
        print(f"DEBUG: Position has been held for {days_held} days (Max: {MAX_DAYS_HELD})")
    
    signal = "HOLD"
    reason = ""
    
    if not has_position:
        if (latest_data['fg_momentum'] > MOMENTUM_THRESHOLD and 
            latest_data['fg_velocity'] > VELOCITY_THRESHOLD and
            current_volatility < VOLATILITY_BUY_LIMIT):
            signal = "BUY"
            reason = "Strong momentum/velocity, low volatility"
        elif latest_data['fg_momentum'] > MOMENTUM_THRESHOLD and latest_data['fg_velocity'] > VELOCITY_THRESHOLD:
            reason = "Strong momentum/velocity but high volatility"
        elif current_volatility >= VOLATILITY_BUY_LIMIT:
            reason = "Volatility too high for entry"
        else:
            reason = "Insufficient momentum/velocity for entry"
    else:
        # UPDATED SELL LOGIC: Added 'days_held >= MAX_DAYS_HELD' condition
        if (latest_data['fg_momentum'] < MOMENTUM_THRESHOLD or 
            latest_data['fg_velocity'] < VELOCITY_THRESHOLD or
            current_volatility > VOLATILITY_SELL_LIMIT or
            days_held >= MAX_DAYS_HELD):  # <-- THIS IS THE NEW CONDITION
            
            signal = "SELL"
            # Specify the reason for selling
            if days_held >= MAX_DAYS_HELD:
                reason = f"Maximum holding period reached ({days_held} days >= {MAX_DAYS_HELD})"
            else:
                reason = "Momentum reversal or high volatility"
        else:
            reason = f"Holding position - indicators still favorable ({days_held}/{MAX_DAYS_HELD} days)"
    
    return signal, reason, latest_data['fg_momentum'], latest_data['fg_velocity'], days_held # <-- Return days_held

def execute_trade(signal, current_price):
    """Executes trade based on signal and logs it."""
    account = api.get_account()
    portfolio_value = float(account.portfolio_value)
    buying_power = float(account.buying_power)
    has_position, position_qty = get_current_position(TRADE_SYMBOL)
    
    if signal == "BUY" and not has_position:
        qty = int((buying_power * 0.95) / current_price)
        
        # ✅ FIXED: Prevent qty=0 orders
        if qty <= 0:
            print("Insufficient buying power to place buy order.")
            return "NO_ACTION", 0, portfolio_value, buying_power

        api.submit_order(
            symbol=TRADE_SYMBOL,
            qty=qty,
            side='buy',
            type='market',
            time_in_force='day'
        )
        action = "BOUGHT"
    elif signal == "SELL" and has_position:
        api.submit_order(
            symbol=TRADE_SYMBOL,
            qty=position_qty,
            side='sell',
            type='market',
            time_in_force='day'
        )
        action = "SOLD"
        qty = position_qty
    else:
        action = "NO_ACTION"
        qty = 0
        
    return action, qty, portfolio_value, buying_power

def log_trade(action, qty, price, fgi_value, momentum, velocity, volatility, portfolio_value, buying_power, reason, days_held): # Add parameter
    """Logs all trade details to CSV file."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    new_log = pd.DataFrame({
        'Timestamp': [timestamp],
        'Action': [action],
        'Symbol': [TRADE_SYMBOL],
        'Quantity': [qty],
        'Price': [price],
        'FGI_Value': [fgi_value],
        'FGI_Momentum': [momentum],
        'FGI_Velocity': [velocity],
        'Volatility': [volatility],
        'Portfolio_Value': [portfolio_value],
        'Buying_Power': [buying_power],
        'Signal_Reason': [reason],
        'Days_Held': [days_held]  # Add new column
    })
    
    new_log.to_csv(LOG_FILE, mode='a', header=False, index=False)
    print(f"Logged trade: {timestamp} - {action} {qty} {TRADE_SYMBOL} @ ${price} (Held for {days_held} days)")

def is_market_open():
    """Check if market is currently open and it's a trading day."""
    nyse = mcal.get_calendar('NYSE')
    today = datetime.date.today()
    schedule = nyse.schedule(start_date=today, end_date=today)
    
    if schedule.empty:
        return False
    
    # ✅ FIXED: Handle timezone-aware datetime comparison
    eastern = pytz.timezone('US/Eastern')
    now_utc = datetime.datetime.now(pytz.UTC)
    now_et = now_utc.astimezone(eastern)
    
    market_open_et = schedule.iloc[0]['market_open'].to_pydatetime().astimezone(eastern)
    market_close_et = schedule.iloc[0]['market_close'].to_pydatetime().astimezone(eastern)
    
    return market_open_et <= now_et <= market_close_et

def get_next_market_open():
    nyse = mcal.get_calendar('NYSE')
    eastern = pytz.timezone('US/Eastern')
    now_et = datetime.datetime.now(pytz.UTC).astimezone(eastern)
    
    # Get schedule for today and tomorrow
    today = now_et.date()
    tomorrow = today + datetime.timedelta(days=1)
    schedule = nyse.schedule(start_date=today, end_date=tomorrow)
    
    # Iterate through schedule to find next open that is in the future
    for _, row in schedule.iterrows():
        market_open = row['market_open'].to_pydatetime().astimezone(eastern)
        if market_open > now_et:
            return market_open
    
    # Fallback: if no market open found, return tomorrow 9:30 ET
    return eastern.localize(datetime.datetime.combine(tomorrow, datetime.time(9,30)))
def main():
    """Main trading function - runs continuously"""
    print("=" * 60)
    print("Fear & Greed Strategy Execution - Starting Continuous Mode")
    print("=" * 60)
    
    TARGET_HOUR = 6 # 9:00 AM PST
    TARGET_MINUTE = 40
    target_time = datetime.time(TARGET_HOUR, TARGET_MINUTE)
    
    MARKET_OPEN_HOUR = 6
    MARKET_OPEN_MINUTE = 30
    market_open_time = datetime.time(MARKET_OPEN_HOUR, MARKET_OPEN_MINUTE)
    
    while True:
        current_datetime = datetime.datetime.now()
        current_date = current_datetime.date()
        current_time = current_datetime.time()
        
        print(f"\nWoke up at {current_time.strftime('%H:%M:%S')} PST on {current_date}...")
        
        market_open_today = is_market_open()
        
        if not is_market_open():
            next_market_open_et = get_next_market_open()
            # Convert to local timezone (PST/PDT)
            local_tz = pytz.timezone('US/Pacific')
            next_market_open_local = next_market_open_et.astimezone(local_tz)
            
            sleep_seconds = (next_market_open_local - datetime.datetime.now(local_tz)).total_seconds()
            sleep_seconds = max(sleep_seconds, 60)
            
            print(f"Market is closed today. Sleeping until market open at {next_market_open_local.strftime('%Y-%m-%d %H:%M %Z')} ({sleep_seconds/3600:.1f} hours)...")
            time.sleep(sleep_seconds)
            continue
        
        print("Market is open today.")
        target_datetime_today = datetime.datetime.combine(current_date, target_time)
        
        if current_datetime < target_datetime_today:
            sleep_seconds = (target_datetime_today - current_datetime).total_seconds()
            sleep_seconds = max(sleep_seconds, 60)  # ✅ FIXED: Prevent negative sleep
            print(f"Sleeping {sleep_seconds/60:.1f} minutes until execution time {target_time.strftime('%H:%M')} PST...")
            time.sleep(sleep_seconds)
            current_datetime = datetime.datetime.now()
            print(f"Woke up at execution time {current_datetime.time().strftime('%H:%M:%S')} PST")
            execute_trading_logic(current_date)
        else:
            print(f"Already past execution time. Executing now...")
            execute_trading_logic(current_date)
        
        tomorrow = current_date + datetime.timedelta(days=1)
        next_market_open = datetime.datetime.combine(tomorrow, market_open_time)
        sleep_seconds = (next_market_open - datetime.datetime.now()).total_seconds()
        sleep_seconds = max(sleep_seconds, 60)  # ✅ FIXED: Prevent negative sleep
        print(f"Sleeping until market open tomorrow at {next_market_open.strftime('%Y-%m-%d %H:%M')} PST ({sleep_seconds/3600:.1f} hours)...")
        time.sleep(sleep_seconds)

def execute_trading_logic(current_date):
    """Execute the actual trading logic"""
    print("Executing trading logic...")
    
    fg_data, current_fgi, fgi_column = fetch_and_update_fgi()
    if current_fgi is None or fg_data is None:
        print("Failed to get F&G data. Skipping execution today.")
        return
    
    fg_data = calculate_indicators(fg_data, fgi_column)
    
    try:
        current_volatility = get_current_volatility(TRADE_SYMBOL)
        # ✅ FIXED: Use .price instead of .p
        current_price = api.get_latest_trade(TRADE_SYMBOL).price
    except Exception as e:
        print(f"Error getting market data: {e}. Skipping execution today.")
        return
    
    signal, reason, momentum, velocity, days_held = generate_signal(fg_data, current_volatility)
    action, qty, portfolio_value, buying_power = execute_trade(signal, current_price)
    
    log_trade(action, qty, current_price, current_fgi, momentum, 
          velocity, current_volatility, portfolio_value, buying_power, reason, days_held)
    
    print("\n" + "=" * 60)
    print("EXECUTION SUMMARY:")
    print("=" * 60)
    print(f"Signal Generated: {signal} ({reason})")
    print(f"Action Taken: {action} {qty} shares")
    print(f"F&G Index: {current_fgi}")
    print(f"F&G Momentum: {momentum:.2f} (threshold: {MOMENTUM_THRESHOLD:.2f})")
    print(f"F&G Velocity: {velocity:.2f} (threshold: {VELOCITY_THRESHOLD:.2f})")
    print(f"Volatility: {current_volatility:.4f} (limit: {VOLATILITY_BUY_LIMIT:.4f})")
    print(f"SPY Price: ${current_price:.2f}")
    print(f"Portfolio Value: ${portfolio_value:.2f}")
    print(f"Buying Power: ${buying_power:.2f}")
    print("=" * 60)

if __name__ == "__main__":
    main()