import pandas as pd
import datetime
import fear_and_greed as fg
import alpaca_trade_api as tradeapi
import numpy as np

API_KEY = 'PKVVMQPZU4DBWK9DH51G'
API_SECRET = 'Nm1UpSPe7xzyYeQOtVGTSwUG2xbgMRstpfmsfGGb'
BASE_URL = 'https://paper-api.alpaca.markets'
SYMBOL = 'SPY'
FG_PATH = 'datasets/present.csv'

current_position = 0
days_held = 0
def update_fear_greed_csv(csv_path):
    import fear_and_greed as fg
    today = datetime.datetime.now(datetime.timezone.utc).date()
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    if today not in df['Date'].values:
        idx = fg.get()
        new_row = {
            'Date': today,
            'fear_greed_desc': str(idx.description)
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.sort_values('Date', inplace=True)
        df.to_csv(csv_path, index=False)
        print(f"Added new Fear & Greed value for {today} to CSV.")

def get_latest_close(csv_path):
    """Optionally update today's Close price from Alpaca (or other API) if missing."""
    df = pd.read_csv(csv_path)
    today = datetime.datetime.now(datetime.timezone.utc).date()
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    latest_row = df[df['Date'] == today]
    if latest_row.empty or np.isnan(latest_row['Close'].values[0]):
        api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL)
        bar = api.get_latest_trade(SYMBOL)
        close_val = bar.price
        df.loc[df['Date'] == today, 'Close'] = close_val
        df.to_csv(csv_path, index=False)
        print(f"Updated today's Close price in CSV: {close_val}")
    else:
        print("Close price for today already present.")
def fear_greed_signal(csv_path, 
                      lookback_days=14,
                      momentum_threshold=0.5,
                      base_position_size=1.0):
    global current_position, days_held
    
    # Load CSV
    data = pd.read_csv(csv_path)
    
    # Ensure date is datetime
    data['Date'] = pd.to_datetime(data['Date']).dt.date
    
    # Convert numeric columns
    price_col = 'Close'
    data[price_col] = pd.to_numeric(data[price_col], errors='coerce')
    data['fear_greed'] = pd.to_numeric(data['fear_greed'], errors='coerce')
    
    # Sort data by date
    data = data.sort_values(by='Date').reset_index(drop=True)
    
    # --- Indicators ---
    data['fg_change'] = data['fear_greed'].diff()
    data['fg_sma_short'] = data['fear_greed'].rolling(lookback_days).mean()
    data['fg_sma_medium'] = data['fear_greed'].rolling(lookback_days * 2).mean()
    data['fg_sma_long'] = data['fear_greed'].rolling(lookback_days * 4).mean()
    
    data['fg_momentum_short'] = data['fear_greed'] - data['fg_sma_short']
    data['fg_momentum_medium'] = data['fear_greed'] - data['fg_sma_medium']
    data['fg_momentum_long'] = data['fear_greed'] - data['fg_sma_long']
    data['fg_velocity'] = data['fg_change'].rolling(3).mean()
    data['fg_acceleration'] = data['fg_velocity'].diff()
    
    data['price_returns'] = data[price_col].pct_change()
    data['volatility'] = data['price_returns'].rolling(20).std() * np.sqrt(252)
    data['price_sma_20'] = data[price_col].rolling(20).mean()
    data['price_sma_50'] = data[price_col].rolling(50).mean()
    data['price_momentum'] = data[price_col] / data['price_sma_20'] - 1
    
    regime_trend = data['price_sma_20'] / data['price_sma_50'] - 1
    data['regime'] = np.where(regime_trend > 0.02, 'Bull',
                            np.where(regime_trend < -0.02, 'Bear', 'Sideways'))
    
    data['fg_zscore'] = (data['fear_greed'] - data['fear_greed'].rolling(60).mean()) / \
                        data['fear_greed'].rolling(60).std()
    
    # --- Latest values ---
    latest = data.iloc[-1]
    fg_momentum = latest['fg_momentum_short']
    fg_velocity = latest['fg_velocity']
    fg_acceleration = latest['fg_acceleration']
    volatility = latest['volatility']
    regime = latest['regime']
    price_momentum = latest['price_momentum']
    fg_zscore = latest['fg_zscore']
    
    # Multi-timeframe confirmation
    mtf_bullish = (
        latest['fg_momentum_short'] > 0 and 
        latest['fg_momentum_medium'] > -1 and
        latest['fg_momentum_long'] > -2
    )
    mtf_bearish = (
        latest['fg_momentum_short'] < 0 and 
        latest['fg_momentum_medium'] < 1 and
        latest['fg_momentum_long'] < 2
    )
    
    if current_position != 0:
        days_held += 1
    
    # --- Buy conditions ---
    base_buy_condition = (
        fg_momentum > momentum_threshold and
        fg_velocity > 0.3 and
        mtf_bullish
    )
    
    strength_multipliers = []
    if not np.isnan(fg_acceleration) and fg_acceleration > 0.2:
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
        volatility < 0.6
    )
    
    # --- Sell conditions ---
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
            volatility > 0.5)
    )
    
    # --- Decision ---
    if buy_condition:
        current_position = base_position_size
        days_held = 0
        return "buy"
    
    elif sell_condition:
        current_position = 0
        days_held = 0
        return "sell"
    
    else:
        return "hold"


# ---- Alpaca trade execution functions ----
def buy_full_position(symbol):
    api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL)
    account = api.get_account()
    buying_power = float(account.buying_power)
    bar = api.get_latest_trade(symbol)
    
    price = float(bar.price)
    qty = int(buying_power // price)
    if qty > 0:
        api.submit_order(symbol=symbol, qty=qty, side='buy', type='market', time_in_force='gtc')
        print(f"Bought {qty} shares of {symbol} at ${price}")
    else:
        print("Not enough buying power.")

def sell_full_position(symbol):
    api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL)
    try:
        position = api.get_position(symbol)
        qty = int(position.qty)
    except tradeapi.rest.APIError:
        qty = 0
    if qty > 0:
        api.submit_order(symbol=symbol, qty=qty, side='sell', type='market', time_in_force='gtc')
        print(f"Sold {qty} shares of {symbol}")
    else:
        print("No position to sell.")

# ---- Main bot logic ----
if __name__ == '__main__':
    # Update CSV with new fear & greed value if needed
    update_fear_greed_csv(FG_PATH)
    # Optionally update latest price
    # Generate signal
    signal = fear_greed_signal(FG_PATH)
    print(f"Trading signal: {signal}")

    # Execute trade
    if signal == "buy":
        buy_full_position(SYMBOL)
    elif signal == "sell":
        sell_full_position(SYMBOL)
    else:
        print("Hold: No trade executed.")

