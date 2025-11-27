import pandas as pd
import datetime
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import APIError

# =============================================================================
# CONFIGURATION
# =============================================================================
API_KEY = 'PKVVMQPZU4DBWK9DH51G'
API_SECRET = 'Nm1UpSPe7xzyYeQOtVGTSwUG2xbgMRstpfmsfGGb'
BASE_URL = 'https://paper-api.alpaca.markets'
TRADE_SYMBOL = 'SPY'

# =============================================================================
# ALPACA API INITIALIZATION
# =============================================================================
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

def get_current_position(symbol):
    """Checks if we have a position in the given symbol. Returns (True, qty) or (False, 0)."""
    try:
        position = api.get_position(symbol)
        return True, int(position.qty)
    except APIError as e:
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

# =============================================================================
# MAIN TEST EXECUTION
# =============================================================================
print("=" * 50)
print("POSITION DAYS HELD TEST")
print("=" * 50)

# Check if we have a position
has_position, position_qty = get_current_position(TRADE_SYMBOL)

if not has_position:
    print(f"No current position found for {TRADE_SYMBOL}.")
    print("This script only works when you have an open position.")
else:
    print(f"Found position: {position_qty} shares of {TRADE_SYMBOL}")
    
    # Get the entry date from order history
    entry_date = get_position_entry_date(TRADE_SYMBOL)
    
    if entry_date:
        today = datetime.date.today()
        days_held = (today - entry_date).days
        
        print(f"Entry Date (from buy order): {entry_date}")
        print(f"Today's Date: {today}")
        print(f"Days Held: {days_held} calendar days")
        
        # Show position details
        try:
            position = api.get_position(TRADE_SYMBOL)
            print(f"\nPosition Details:")
            print(f"Current Price: ${float(position.current_price):.2f}")
            print(f"Entry Price: ${float(position.avg_entry_price):.2f}")
            print(f"Unrealized P/L: ${float(position.unrealized_pl):.2f}")
            print(f"Unrealized P/L %: {float(position.unrealized_plpc) * 100:.2f}%")
        except Exception as e:
            print(f"Could not get full position details: {e}")
    else:
        print("Error: Could not retrieve entry date for the position.")

print("=" * 50)