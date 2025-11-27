import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta, time as dt_time
import numpy as np
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import APIError
import pandas_market_calendars as mcal
import pytz
import time
import os
import csv

# =============================================================================
# CONFIGURATION
# =============================================================================
API_KEY = 'YOUR_API_KEY_HERE'
API_SECRET = 'YOUR_API_SECRET_HERE'
BASE_URL = 'https://paper-api.alpaca.markets'  # Use paper trading for testing

TRADE_SYMBOL = 'TQQQ'  # 3x leveraged NASDAQ ETF
LOG_FILE = 'eod_contrarian_trades.csv'

# Strategy Parameters
STRATEGY_MODE = 'contrarian_long_only'  # Options: 'contrarian_long_only', 'contrarian_both'
MOVE_THRESHOLD = 0.005  # 0.5% move required
MIN_SIGNALS = 2         # Minimum number of stocks signaling
MAX_SIGNALS = 6         # Avoid extreme consensus
CONSENSUS_THRESHOLD = 0.60  # 60% agreement required

# Watch stocks (illiquid tech)
WATCH_STOCKS = [
    'RGTI', 'QBTS', 'IONQ', 'QUBT',  # Quantum computing
    'SOUN', 'BBAI', 'INOD', 'AEYE'   # Small-cap AI
]

# Risk Management
ACCOUNT_SIZE = 10000  # Will be updated from actual account
MAX_RISK_PER_TRADE = 0.02  # 2% max risk per trade
STOP_LOSS_PCT = 0.03  # 3% stop loss
MAX_POSITION_SIZE = 0.30  # Max 30% of account per trade

# Execution Time (in Pacific Time)
SIGNAL_CHECK_HOUR = 12  # 12:00 PM PST = 3:00 PM EST (near market close)
SIGNAL_CHECK_MINUTE = 55  # Check at 12:55 PM PST (3:55 PM EST)

EXIT_CHECK_HOUR = 6  # 6:30 AM PST = 9:30 AM EST (market open)
EXIT_CHECK_MINUTE = 35  # Check at 6:35 AM PST (9:35 AM EST)

# =============================================================================
# SETUP LOGGING
# =============================================================================
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Timestamp', 'Action', 'Symbol', 'Quantity', 'Entry_Price', 
            'Exit_Price', 'Direction', 'Confidence', 'Signals', 
            'Position_Value', 'Stop_Loss', 'Return_Pct', 'PnL_Dollars',
            'Portfolio_Value', 'Signal_Reason'
        ])
    print(f"Created new log file: {LOG_FILE}")

# =============================================================================
# ALPACA API INITIALIZATION
# =============================================================================
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def is_market_open():
    """Check if market is currently open"""
    nyse = mcal.get_calendar('NYSE')
    today = datetime.now().date()
    schedule = nyse.schedule(start_date=today, end_date=today)
    
    if schedule.empty:
        return False
    
    eastern = pytz.timezone('US/Eastern')
    now_utc = datetime.now(pytz.UTC)
    now_et = now_utc.astimezone(eastern)
    
    market_open_et = schedule.iloc[0]['market_open'].to_pydatetime().astimezone(eastern)
    market_close_et = schedule.iloc[0]['market_close'].to_pydatetime().astimezone(eastern)
    
    return market_open_et <= now_et <= market_close_et

def get_next_market_open():
    """Get next market open time"""
    nyse = mcal.get_calendar('NYSE')
    eastern = pytz.timezone('US/Eastern')
    now_et = datetime.now(pytz.UTC).astimezone(eastern)
    
    # Get schedule for today and tomorrow
    today = now_et.date()
    tomorrow = today + timedelta(days=1)
    schedule = nyse.schedule(start_date=today, end_date=tomorrow)
    
    # Iterate through schedule to find next open that is in the future
    for _, row in schedule.iterrows():
        market_open = row['market_open'].to_pydatetime().astimezone(eastern)
        if market_open > now_et:
            return market_open
    
    # Fallback: if no market open found, return tomorrow 9:30 ET
    return eastern.localize(datetime.combine(tomorrow, dt_time(9, 30)))

def get_current_position(symbol):
    """Check if we have a position in the given symbol"""
    try:
        position = api.get_position(symbol)
        return True, int(position.qty), float(position.avg_entry_price)
    except APIError:
        return False, 0, 0.0

def get_account_value():
    """Get current account portfolio value"""
    try:
        account = api.get_account()
        return float(account.portfolio_value)
    except Exception as e:
        print(f"Error getting account value: {e}")
        return ACCOUNT_SIZE

# =============================================================================
# STRATEGY CORE FUNCTIONS
# =============================================================================

def fetch_intraday_data(ticker, period='1d', interval='5m'):
    """Fetch intraday data from yfinance"""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval=interval)
        return data
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None

def detect_eod_move(df, interval='5m'):
    """Detect directional move in final period before close"""
    if df is None or len(df) < 2:
        return 'NONE', 0.0
    
    try:
        df = df.between_time('09:30', '16:00')
    except:
        # If timezone not set, just use the data as-is
        pass
    
    if len(df) < 2:
        return 'NONE', 0.0
    
    window = 6 if interval == '1m' else 2
    final_bars = df.tail(window)
    
    if len(final_bars) < 2:
        return 'NONE', 0.0
    
    start_price = final_bars['Close'].iloc[0]
    end_price = final_bars['Close'].iloc[-1]
    move_pct = (end_price - start_price) / start_price
    
    if abs(move_pct) >= MOVE_THRESHOLD:
        return ('UP' if move_pct > 0 else 'DOWN'), move_pct
    
    return 'NONE', move_pct

def generate_trade_signal():
    """
    Check EOD moves and generate LONG/SHORT signal
    Returns: (direction, confidence, reason, signals)
    """
    print("\n" + "="*60)
    print("CHECKING END-OF-DAY SIGNALS")
    print("="*60)
    
    signals = []
    
    # Collect signals from all watch stocks
    for ticker in WATCH_STOCKS:
        data = fetch_intraday_data(ticker, period='1d', interval='5m')
        if data is None or len(data) == 0:
            print(f"  ⚠️  No data for {ticker}")
            continue
        
        direction, move_pct = detect_eod_move(data, interval='5m')
        if direction != 'NONE':
            signals.append((ticker, direction, move_pct))
            arrow = "📈" if direction == 'UP' else "📉"
            print(f"  {arrow} {ticker}: {move_pct*100:+.2f}%")
    
    if not signals:
        print("  ❌ No significant moves detected")
        return 'SKIP', 0, "No signals", []
    
    # Check minimum signals
    if len(signals) < MIN_SIGNALS:
        reason = f"Only {len(signals)} signals (need {MIN_SIGNALS}+)"
        print(f"  ⚠️  {reason}")
        return 'SKIP', 0, reason, signals
    
    # Count directional signals
    up_count = sum(1 for s in signals if s[1] == 'UP')
    down_count = len(signals) - up_count
    total = len(signals)
    
    # Check for extreme consensus
    if total >= MAX_SIGNALS:
        reason = f"Extreme consensus ({total} signals)"
        print(f"  ⚠️  {reason}")
        return 'SKIP', 0, reason, signals
    
    up_pct = up_count / total
    down_pct = down_count / total
    
    # Check consensus threshold
    if up_pct < CONSENSUS_THRESHOLD and down_pct < CONSENSUS_THRESHOLD:
        reason = f"No consensus: {up_count} UP vs {down_count} DOWN"
        print(f"  ⚠️  {reason}")
        return 'SKIP', 0, reason, signals
    
    # Generate signal based on mode
    if STRATEGY_MODE == 'contrarian_long_only':
        if down_pct >= CONSENSUS_THRESHOLD:
            reason = f"Contrarian: {down_count} stocks dumped → BUY"
            print(f"  ✅ {reason}")
            return 'LONG', down_pct, reason, signals
        else:
            reason = "No dump signal (stocks pumped)"
            print(f"  ⚠️  {reason}")
            return 'SKIP', 0, reason, signals
    
    elif STRATEGY_MODE == 'contrarian_both':
        if down_pct >= CONSENSUS_THRESHOLD:
            reason = f"Contrarian: {down_count} dumped → BUY"
            print(f"  ✅ {reason}")
            return 'LONG', down_pct, reason, signals
        elif up_pct >= CONSENSUS_THRESHOLD:
            reason = f"Contrarian: {up_count} pumped → SHORT"
            print(f"  ✅ {reason}")
            return 'SHORT', up_pct, reason, signals
    
    return 'SKIP', 0, "No valid setup", signals

def calculate_position_size(confidence, entry_price, account_value):
    """Calculate position size based on confidence and risk management"""
    # Base size on confidence
    if confidence < 0.70:
        size_multiplier = 0.5
    elif confidence < 0.85:
        size_multiplier = 0.75
    else:
        size_multiplier = 1.0
    
    # Calculate position size based on risk
    risk_amount = account_value * MAX_RISK_PER_TRADE
    stop_distance = entry_price * STOP_LOSS_PCT
    shares = risk_amount / stop_distance
    
    # Apply max position size limit
    max_shares = (account_value * MAX_POSITION_SIZE) / entry_price
    shares = min(shares, max_shares)
    
    # Apply confidence multiplier
    shares = shares * size_multiplier
    shares = int(shares)
    
    if shares <= 0:
        return None
    
    position_value = shares * entry_price
    stop_loss = entry_price * (1 - STOP_LOSS_PCT)
    
    return {
        'shares': shares,
        'position_value': position_value,
        'position_pct': position_value / account_value,
        'stop_loss': stop_loss
    }

def execute_entry_trade(direction, confidence, signals, reason):
    """Execute entry trade at EOD"""
    print("\n" + "="*60)
    print("EXECUTING ENTRY TRADE")
    print("="*60)
    
    # Check if we already have a position
    has_position, qty, entry_price = get_current_position(TRADE_SYMBOL)
    if has_position:
        print(f"  ⚠️  Already holding {qty} shares of {TRADE_SYMBOL}")
        return
    
    # Get current price
    try:
        current_price = api.get_latest_trade(TRADE_SYMBOL).price
    except Exception as e:
        print(f"  ❌ Error getting price: {e}")
        return
    
    # Get account value
    account_value = get_account_value()
    
    # Calculate position size
    position = calculate_position_size(confidence, current_price, account_value)
    if position is None:
        print("  ❌ Position size too small to trade")
        return
    
    print(f"  Direction: {direction}")
    print(f"  Confidence: {confidence:.0%}")
    print(f"  Entry Price: ${current_price:.2f}")
    print(f"  Shares: {position['shares']}")
    print(f"  Position Value: ${position['position_value']:.0f} ({position['position_pct']*100:.1f}%)")
    print(f"  Stop Loss: ${position['stop_loss']:.2f}")
    
    # Execute order
    try:
        side = 'buy' if direction == 'LONG' else 'sell'
        
        order = api.submit_order(
            symbol=TRADE_SYMBOL,
            qty=position['shares'],
            side=side,
            type='market',
            time_in_force='day'
        )
        
        print(f"  ✅ Order submitted: {side.upper()} {position['shares']} {TRADE_SYMBOL}")
        
        # Log the trade
        log_entry_trade(direction, confidence, signals, reason, current_price, position, account_value)
        
    except Exception as e:
        print(f"  ❌ Error submitting order: {e}")

def execute_exit_trade():
    """Check and execute exit trade at market open"""
    print("\n" + "="*60)
    print("CHECKING EXIT CONDITIONS")
    print("="*60)
    
    # Check if we have a position
    has_position, qty, entry_price = get_current_position(TRADE_SYMBOL)
    if not has_position:
        print("  ℹ️  No position to exit")
        return
    
    print(f"  Holding: {qty} shares @ ${entry_price:.2f}")
    
    # Get current price
    try:
        current_price = api.get_latest_trade(TRADE_SYMBOL).price
    except Exception as e:
        print(f"  ❌ Error getting price: {e}")
        return
    
    # Calculate return
    return_pct = (current_price - entry_price) / entry_price
    pnl_dollars = (current_price - entry_price) * qty
    
    print(f"  Current Price: ${current_price:.2f}")
    print(f"  Return: {return_pct*100:+.2f}%")
    print(f"  P&L: ${pnl_dollars:+,.0f}")
    
    # Check if stop loss hit
    stop_loss = entry_price * (1 - STOP_LOSS_PCT)
    if current_price <= stop_loss:
        print(f"  🛑 STOP LOSS HIT @ ${current_price:.2f}")
    
    # Exit position
    try:
        order = api.submit_order(
            symbol=TRADE_SYMBOL,
            qty=qty,
            side='sell',
            type='market',
            time_in_force='day'
        )
        
        print(f"  ✅ Exit order submitted: SELL {qty} {TRADE_SYMBOL}")
        
        # Log the exit
        log_exit_trade(entry_price, current_price, qty, return_pct, pnl_dollars)
        
    except Exception as e:
        print(f"  ❌ Error submitting exit order: {e}")

# =============================================================================
# LOGGING FUNCTIONS
# =============================================================================

def log_entry_trade(direction, confidence, signals, reason, entry_price, position, account_value):
    """Log entry trade to CSV"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    signal_str = ' | '.join([f"{s[0]}:{s[2]*100:+.1f}%" for s in signals])
    
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp, 'ENTRY', TRADE_SYMBOL, position['shares'], entry_price,
            '', direction, f"{confidence*100:.1f}%", signal_str,
            position['position_value'], position['stop_loss'], '', '',
            account_value, reason
        ])
    
    print(f"  📝 Logged entry trade")

def log_exit_trade(entry_price, exit_price, qty, return_pct, pnl_dollars):
    """Log exit trade to CSV"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    account_value = get_account_value()
    
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp, 'EXIT', TRADE_SYMBOL, qty, entry_price,
            exit_price, '', '', '',
            '', '', f"{return_pct*100:+.2f}%", f"${pnl_dollars:+,.0f}",
            account_value, 'Overnight exit'
        ])
    
    print(f"  📝 Logged exit trade")

# =============================================================================
# MAIN EXECUTION LOOP
# =============================================================================

def main():
    """Main trading bot - runs continuously"""
    print("="*70)
    print("EOD CONTRARIAN TQQQ TRADING BOT")
    print("="*70)
    print(f"Strategy Mode: {STRATEGY_MODE}")
    print(f"Trade Symbol: {TRADE_SYMBOL}")
    print(f"Watching: {', '.join(WATCH_STOCKS)}")
    print(f"Signal Check: {SIGNAL_CHECK_HOUR:02d}:{SIGNAL_CHECK_MINUTE:02d} PST")
    print(f"Exit Check: {EXIT_CHECK_HOUR:02d}:{EXIT_CHECK_MINUTE:02d} PST")
    print("="*70)
    
    # Target times in PST
    signal_check_time = dt_time(SIGNAL_CHECK_HOUR, SIGNAL_CHECK_MINUTE)
    exit_check_time = dt_time(EXIT_CHECK_HOUR, EXIT_CHECK_MINUTE)
    
    while True:
        try:
            current_datetime = datetime.now()
            current_date = current_datetime.date()
            current_time = current_datetime.time()
            
            print(f"\n⏰ Woke up at {current_time.strftime('%H:%M:%S')} PST on {current_date}...")
            
            # Check if market is open today
            if not is_market_open():
                next_market_open_et = get_next_market_open()
                # Convert to local timezone (PST/PDT)
                local_tz = pytz.timezone('US/Pacific')
                next_market_open_local = next_market_open_et.astimezone(local_tz)
                
                sleep_seconds = (next_market_open_local - datetime.now(local_tz)).total_seconds()
                sleep_seconds = max(sleep_seconds, 60)
                
                print(f"📅 Market is closed. Sleeping until market open at {next_market_open_local.strftime('%Y-%m-%d %H:%M %Z')} ({sleep_seconds/3600:.1f} hours)...")
                time.sleep(sleep_seconds)
                continue
            
            print("✅ Market is open today.")
            
            # Define target times for today
            exit_target = datetime.combine(current_date, exit_check_time)
            signal_target = datetime.combine(current_date, signal_check_time)
            
            # Check if we should execute exit (morning)
            if current_datetime < exit_target:
                sleep_seconds = (exit_target - current_datetime).total_seconds()
                sleep_seconds = max(sleep_seconds, 60)
                print(f"💤 Sleeping {sleep_seconds/60:.1f} minutes until exit check at {exit_check_time.strftime('%H:%M')} PST...")
                time.sleep(sleep_seconds)
                current_datetime = datetime.now()
                print(f"⏰ Woke up at {current_datetime.time().strftime('%H:%M:%S')} PST")
                execute_exit_trade()
                
            # Check if we should look for entry signal (afternoon)
            elif current_datetime < signal_target:
                sleep_seconds = (signal_target - current_datetime).total_seconds()
                sleep_seconds = max(sleep_seconds, 60)
                print(f"💤 Sleeping {sleep_seconds/60:.1f} minutes until signal check at {signal_check_time.strftime('%H:%M')} PST...")
                time.sleep(sleep_seconds)
                current_datetime = datetime.now()
                print(f"⏰ Woke up at {current_datetime.time().strftime('%H:%M:%S')} PST")
                
                direction, confidence, reason, signals = generate_trade_signal()
                if direction != 'SKIP':
                    execute_entry_trade(direction, confidence, signals, reason)
                else:
                    print(f"  ℹ️  No trade today: {reason}")
            else:
                # Already past signal time today, execute now then sleep until tomorrow
                print(f"⚠️  Already past execution time. Executing now...")
                
                direction, confidence, reason, signals = generate_trade_signal()
                if direction != 'SKIP':
                    execute_entry_trade(direction, confidence, signals, reason)
                else:
                    print(f"  ℹ️  No trade today: {reason}")
            
            # Sleep until next market open
            tomorrow = current_date + timedelta(days=1)
            next_market_open = datetime.combine(tomorrow, exit_check_time)
            sleep_seconds = (next_market_open - datetime.now()).total_seconds()
            sleep_seconds = max(sleep_seconds, 60)
            print(f"💤 Sleeping until market open tomorrow at {next_market_open.strftime('%Y-%m-%d %H:%M')} PST ({sleep_seconds/3600:.1f} hours)...")
            time.sleep(sleep_seconds)
            
        except KeyboardInterrupt:
            print("\n\n🛑 Bot stopped by user")
            break
        except Exception as e:
            print(f"❌ Error in main loop: {e}")
            print("💤 Sleeping 5 minutes before retry...")
            time.sleep(300)

if __name__ == "__main__":
    main()