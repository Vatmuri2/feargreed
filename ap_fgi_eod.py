import pandas as pd
import numpy as np
import datetime
import time
import os
import pytz
import fear_and_greed as fg
import pandas_market_calendars as mcal
import requests

# Alpaca Trading imports (still used for orders)
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.common.exceptions import APIError


# =============================================================================
# CONFIGURATION
# =============================================================================
# =============================================================================
# CONFIGURATION
# =============================================================================
API_KEY = 'PKVFXBFXK01KNZWV6JM9'
API_SECRET = 'QaJcd9Dsc9fBeR3JeWtwAgP7nshz9cmcypBG2UTA'
BASE_URL = 'https://paper-api.alpaca.markets'

POLYGON_API_KEY = 'tAj9_5sMUEaQt0Y_m5fYkfF24dzMsSUp'  

TRADE_SYMBOL = 'SPY'  
LOG_FILE = 'trading_log_EOD.csv'
FG_PATH = 'datasets/fear_greed_forward_test_afternoon.csv'  # Path to historical F&G data

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
if not os.path.exists(LOG_FILE):
    log_df = pd.DataFrame(columns=[
        'Timestamp', 'Action', 'Symbol', 'Quantity', 'Price', 'FGI_Value',
        'FGI_Momentum', 'FGI_Velocity', 'Volatility', 'Portfolio_Value',
        'Buying_Power', 'Signal_Reason', 'Days_Held'
    ])
    log_df.to_csv(LOG_FILE, index=False)
    print(f"Created new log file: {LOG_FILE}")

# =============================================================================
# ALPACA API INITIALIZATION
# =============================================================================
trading_client = TradingClient(API_KEY, API_SECRET, paper=True)

def fetch_and_update_fgi():
    """Same as before - unchanged"""
    try:
        fg_index = fg.get()
        curr_fgi = round(fg_index.value, 2)
        curr_rating = fg_index.description
        curr_date = datetime.date.today()
        
        fg_data = pd.read_csv(FG_PATH)
        fg_data['Date'] = pd.to_datetime(fg_data['Date']).dt.date
        
        if 'fear_greed' in fg_data.columns:
            fgi_column = 'fear_greed'
        elif 'Fear Greed' in fg_data.columns:
            fgi_column = 'Fear Greed'
        else:
            fgi_column = 'Index'
        
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
                
            last_fgi = fg_data[fgi_column].iloc[-1]
            return fg_data, last_fgi, fgi_column
        except:
            print("Fatal error: Could not load F&G data file.")
            return None, None, None

def calculate_indicators(fg_data, fgi_column):
    """Same as before"""
    data = fg_data.copy()
    data['fg_momentum'] = data[fgi_column] - data[fgi_column].rolling(LOOKBACK_DAYS, min_periods=1).mean()
    data['fg_change'] = data[fgi_column].diff().fillna(0)
    data['fg_velocity'] = data['fg_change'].rolling(LOOKBACK_DAYS, min_periods=1).mean()
    return data

def get_current_volatility(symbol, window=VOLATILITY_WINDOW):
    """
    Calculate annualized volatility using Polygon.io free API.
    """
    try:
        end = datetime.date.today()
        start = end - datetime.timedelta(days=window*3)  # buffer for holidays

        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start}/{end}?adjusted=true&apiKey={POLYGON_API_KEY}"
        r = requests.get(url).json()

        if "results" not in r or not r["results"]:
            print(f"No data returned from Polygon for {symbol}")
            return None

        df = pd.DataFrame(r["results"])
        df["close"] = df["c"]
        df["returns"] = df["close"].pct_change()
        df["volatility"] = df["returns"].rolling(window, min_periods=1).std() * np.sqrt(252)

        return round(df["volatility"].iloc[-1], 4)
    except Exception as e:
        print(f"Error calculating volatility for {symbol} via Polygon: {e}")
        return None

def get_current_position(symbol):
    """Alpaca-py replacement for checking current position"""
    try:
        position = trading_client.get_open_position(symbol)
        return True, int(float(position.qty))
    except APIError:
        return False, 0

def get_position_entry_date(symbol):
    """Alpaca-py replacement for order history"""
    try:
        orders = trading_client.get_orders(GetOrdersRequest(status="closed", direction="desc"))
        for order in orders:
            if order.symbol == symbol and order.side == OrderSide.BUY and order.filled_qty:
                return order.filled_at.date()
        return None
    except Exception as e:
        print(f"Error getting order history for {symbol}: {e}")
        return None

def generate_signal(fg_data, current_volatility):
    """Same logic as before - unchanged"""
    latest_data = fg_data.iloc[-1]
    has_position, position_qty = get_current_position(TRADE_SYMBOL)
    
    days_held = 0
    if has_position:
        entry_date = get_position_entry_date(TRADE_SYMBOL)
        if entry_date:
            days_held = (datetime.date.today() - entry_date).days
        print(f"DEBUG: Position held for {days_held} days (Max: {MAX_DAYS_HELD})")
    
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
        if (latest_data['fg_momentum'] < MOMENTUM_THRESHOLD or 
            latest_data['fg_velocity'] < VELOCITY_THRESHOLD or
            current_volatility > VOLATILITY_SELL_LIMIT or
            days_held >= MAX_DAYS_HELD):
            
            signal = "SELL"
            if days_held >= MAX_DAYS_HELD:
                reason = f"Maximum holding period reached ({days_held} days >= {MAX_DAYS_HELD})"
            else:
                reason = "Momentum reversal or high volatility"
        else:
            reason = f"Holding position - indicators still favorable ({days_held}/{MAX_DAYS_HELD} days)"
    
    return signal, reason, latest_data['fg_momentum'], latest_data['fg_velocity'], days_held

def execute_trade(signal, current_price):
    """Alpaca-py replacement for trade execution"""
    account = trading_client.get_account()
    portfolio_value = float(account.portfolio_value)
    buying_power = float(account.buying_power)
    has_position, position_qty = get_current_position(TRADE_SYMBOL)
    
    if signal == "BUY" and not has_position:
        qty_to_buy = int((buying_power * 0.95) / current_price)
        if qty_to_buy <= 0:
            return "NO_ACTION", 0, portfolio_value, buying_power
        order = MarketOrderRequest(
            symbol=TRADE_SYMBOL,
            qty=qty_to_buy,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY
        )
        trading_client.submit_order(order)
        return "BOUGHT", qty_to_buy, portfolio_value, buying_power
    elif signal == "SELL" and has_position:
        order = MarketOrderRequest(
            symbol=TRADE_SYMBOL,
            qty=position_qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY
        )
        trading_client.submit_order(order)
        return "SOLD", position_qty, portfolio_value, buying_power
    else:
        return "NO_ACTION", 0, portfolio_value, buying_power

def log_trade(action, qty, price, fgi_value, momentum, velocity, volatility, portfolio_value, buying_power, reason, days_held):
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
        'Days_Held': [days_held]
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
    
    TARGET_HOUR = 12
    TARGET_MINUTE = 50
    target_time = datetime.time(TARGET_HOUR, TARGET_MINUTE)
    
    MARKET_OPEN_HOUR = 6
    MARKET_OPEN_MINUTE = 30
    market_open_time = datetime.time(MARKET_OPEN_HOUR, MARKET_OPEN_MINUTE)
    
    while True:
        current_datetime = datetime.datetime.now()
        current_date = current_datetime.date()
        current_time = current_datetime.time()
        
        print(f"\nWoke up at {current_time.strftime('%H:%M:%S')} PST on {current_date}...")
        
        
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
            sleep_seconds = max(sleep_seconds, 60)
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
        sleep_seconds = max(sleep_seconds, 60)
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
    
    current_volatility = get_current_volatility(TRADE_SYMBOL)

    # Fetch latest price
    try:
        url = f"https://api.polygon.io/v2/last/trade/{TRADE_SYMBOL}?apiKey={POLYGON_API_KEY}"
        trade = requests.get(url).json()
        current_price = trade["results"]["p"]
    except Exception as e:
        print(f"Error fetching latest price for {TRADE_SYMBOL} via Polygon: {e}")
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
    print(f"Days Held: {days_held} (max: {MAX_DAYS_HELD})")
    print("=" * 60)

if __name__ == "__main__":
    main()