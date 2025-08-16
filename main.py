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


def strategy(data, lookback_days=3, momentum_threshold=1.5, 
             use_dynamic_sizing=True, use_regime_detection=True,
             use_multi_timeframe=True, use_volatility_adjustment=True,
             base_position_size=1.0, max_position_size=2.0):
    
    data = data.copy()
    price_col = 'Close'
    
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
    else:
        data['regime'] = 'Bull'  # Default regime
    
    data['fg_zscore'] = (data['fear_greed'] - data['fear_greed'].rolling(60).mean()) / data['fear_greed'].rolling(60).std()
    
    # Dynamic position sizing function
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
    
    # Initialize columns
    data['position'] = 0.0
    data['position_size'] = 0.0
    data['signal'] = 0
    data['signal_strength'] = 0.0
    data['days_in_position'] = 0
    
    current_position = 0.0
    days_held = 0
    buy_signals = []
    sell_signals = []
    
    for i in range(max(lookback_days * 4, 60), len(data)):
        
        fg_momentum = data.iloc[i]['fg_momentum_short']
        fg_velocity = data.iloc[i]['fg_velocity']
        fg_acceleration = data.iloc[i]['fg_acceleration']
        volatility = data.iloc[i]['volatility']
        regime = data.iloc[i]['regime']
        price_momentum = data.iloc[i]['price_momentum']
        fg_zscore = data.iloc[i]['fg_zscore']
        
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
                (not use_volatility_adjustment or volatility < 0.6)
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
                    (use_volatility_adjustment and volatility > 0.5))
            )
            
            # Execute trades
            if buy_condition:
                position_size = calculate_position_size(fg_momentum, fg_velocity, volatility, regime, base_position_size)
                
                data.iloc[i, data.columns.get_loc('signal')] = 1
                data.iloc[i, data.columns.get_loc('signal_strength')] = signal_strength
                
                current_position = position_size
                days_held = 0
                buy_signals.append(i)
                
            elif sell_condition:
                data.iloc[i, data.columns.get_loc('signal')] = -1
                
                current_position = 0.0
                days_held = 0
                sell_signals.append(i)
        
        data.iloc[i, data.columns.get_loc('position')] = current_position
        data.iloc[i, data.columns.get_loc('position_size')] = abs(current_position)
        data.iloc[i, data.columns.get_loc('days_in_position')] = days_held
    
    return data, buy_signals, sell_signals

# Parameters
multi_timeframe_params = {
    'momentum_threshold': 1.0,
    'lookback_days': 3,
    'use_dynamic_sizing': True,
    'use_multi_timeframe': True,
    'use_volatility_adjustment': True,
    'base_position_size': 1.0,
    'max_position_size': 2.0,
}

# Run strategy
data_result, buy_signals, sell_signals = strategy(merged_data, **multi_timeframe_params)


data_result['price_returns'].to_csv('results.csv')

trades = pd.DataFrame(
    {'buy_signals': buy_signals,
    'sell_signals': sell_signals
    }
)

pct_change = 0
total_pnl = 0
capital = 10000
num_wins = 0


print(max(data_result['signal_strength']))
for i in range(len(trades)):
    buy_index = trades['buy_signals'][i]
    sell_index = trades['sell_signals'][i]

    buy_price = merged_data['Close'][buy_index]
    sell_price = merged_data['Close'][sell_index]
    signal_strength = data_result['signal_strength'][buy_index]   
    shares = capital * 2 / buy_price

    pnl = (sell_price - buy_price) * shares * 0.63
    
    if pnl > 0:
        num_wins += 1

    capital += pnl

    
print(f"Final Capital: ${capital:2,}")
print("Win Rate: ", round((num_wins / len(trades))*100, 2), "%")

