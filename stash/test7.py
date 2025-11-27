import pandas as pd
import numpy as np
import yfinance as yf
import fear_and_greed as fg
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# ------------------------
# 1️⃣ Load historical Fear & Greed data
# ------------------------
try:
    fg_file = "datasets/fear_greed_combined_2011_2025.csv"
    fg_df = pd.read_csv(fg_file, parse_dates=['Date'])
    fg_df.set_index('Date', inplace=True)
    fg_df['fear_greed'] = fg_df['fear_greed'].ffill()
    print(f"✅ Loaded Fear & Greed data with {len(fg_df)} records")
except FileNotFoundError:
    print("❌ Fear & Greed data file not found")
    exit()
except Exception as e:
    print(f"❌ Error loading Fear & Greed data: {e}")
    exit()

# ------------------------
# 2️⃣ Download SPY and VIX historical data
# ------------------------
start_date = fg_df.index.min()
end_date = fg_df.index.max()

try:
    spy_data = yf.download('SPY', start=start_date, end=end_date, auto_adjust=True)
    vix_data = yf.download('^VIX', start=start_date, end=end_date, auto_adjust=True)
    
    spy = spy_data['Close'].squeeze()
    vix = vix_data['Close'].squeeze()
    
    print(f"✅ Downloaded SPY data with {len(spy)} records")
    print(f"✅ Downloaded VIX data with {len(vix)} records")
except Exception as e:
    print(f"❌ Error downloading market data: {e}")
    exit()

# Remove duplicates
fg_df = fg_df[~fg_df.index.duplicated(keep='first')]
spy = spy[~spy.index.duplicated(keep='first')]
vix = vix[~vix.index.duplicated(keep='first')]

# Create a common date index
all_dates = pd.date_range(start=start_date, end=end_date, freq='D')

# Reindex all series
fg_series = fg_df['fear_greed'].reindex(all_dates).ffill()
spy_series = spy.reindex(all_dates).ffill()
vix_series = vix.reindex(all_dates).ffill()

# Combine into one DataFrame
df = pd.DataFrame({
    'FG': fg_series,
    'SPY': spy_series,
    'VIX': vix_series
}).sort_index().dropna()

# ------------------------
# 3️⃣ Compute indicators
# ------------------------
df['SPY_ret'] = df['SPY'].pct_change()
df['SPY_vol_20'] = df['SPY_ret'].rolling(20, min_periods=10).std() * np.sqrt(252)
df['SPY_MA200'] = df['SPY'].rolling(200, min_periods=100).mean()
df['SPY_MA50'] = df['SPY'].rolling(50, min_periods=25).mean()
df['SPY_vs_MA200'] = (df['SPY'] / df['SPY_MA200'] - 1) * 100

df = df.dropna()

# ------------------------
# 4️⃣ Get current market conditions
# ------------------------
try:
    current_fg = fg.get().value
    print(f"✅ Current Fear & Greed index: {current_fg:.2f}")
except Exception as e:
    current_fg = df['FG'].iloc[-1]
    print(f"⚠️ Using historical FG value: {current_fg:.2f}")

current_price = df['SPY'].iloc[-1]
current_ma200 = df['SPY_MA200'].iloc[-1]
current_vol = df['SPY_vol_20'].iloc[-1]
current_vix = df['VIX'].iloc[-1]
current_vs_ma200 = df['SPY_vs_MA200'].iloc[-1]

print("\n📊 Current market snapshot:")
print(f"FG: {current_fg:.2f} (Extreme Fear)")
print(f"SPY Price: {current_price:.2f}")
print(f"MA200: {current_ma200:.2f} ({current_vs_ma200:+.1f}% above)")
print(f"Volatility: {current_vol:.4f}")
print(f"VIX: {current_vix:.2f}")

# ------------------------
# 5️⃣ DEEPER ANALYSIS: Break down by different scenarios
# ------------------------

print("\n" + "="*70)
print("DETAILED SCENARIO ANALYSIS")
print("="*70)

# Scenario 1: Just Extreme Fear (FG < 25) regardless of price position
extreme_fear_mask = df['FG'] < 25
extreme_fear_dates = df.index[extreme_fear_mask]

print(f"\n📉 Scenario 1: Pure Extreme Fear (FG < 25)")
print(f"Found {len(extreme_fear_dates)} historical periods")

if len(extreme_fear_dates) > 0:
    forward_returns = []
    for date in extreme_fear_dates[:1000]:  # Limit for performance
        try:
            date_loc = df.index.get_loc(date)
            if date_loc + 63 < len(df):  # 3 months forward
                entry_price = df.loc[date, 'SPY']
                future_prices = df['SPY'].iloc[date_loc:date_loc+64]
                ret_1m = (future_prices.iloc[21] - entry_price) / entry_price * 100
                ret_3m = (future_prices.iloc[63] - entry_price) / entry_price * 100
                forward_returns.append((ret_1m, ret_3m))
        except:
            continue
    
    if forward_returns:
        returns_1m, returns_3m = zip(*forward_returns)
        print(f"1-month forward: {np.mean(returns_1m):.2f}% (win rate: {(np.array(returns_1m) > 0).mean()*100:.1f}%)")
        print(f"3-month forward: {np.mean(returns_3m):.2f}% (win rate: {(np.array(returns_3m) > 0).mean()*100:.1f}%)")

# Scenario 2: Extreme Fear + Price ABOVE MA200 (current situation)
scenario2_mask = (df['FG'] < 25) & (df['SPY'] > df['SPY_MA200'])
scenario2_dates = df.index[scenario2_mask]

print(f"\n📊 Scenario 2: Extreme Fear + Price ABOVE MA200 (Current Situation)")
print(f"Found {len(scenario2_dates)} historical periods")

if len(scenario2_dates) > 0:
    forward_returns = []
    for date in scenario2_dates:
        try:
            date_loc = df.index.get_loc(date)
            if date_loc + 252 < len(df):  # 1 year forward
                entry_price = df.loc[date, 'SPY']
                future_1m = df['SPY'].iloc[date_loc + 21] if date_loc + 21 < len(df) else np.nan
                future_3m = df['SPY'].iloc[date_loc + 63] if date_loc + 63 < len(df) else np.nan
                future_1y = df['SPY'].iloc[date_loc + 252] if date_loc + 252 < len(df) else np.nan
                
                if not np.isnan(future_1m):
                    ret_1m = (future_1m - entry_price) / entry_price * 100
                    ret_3m = (future_3m - entry_price) / entry_price * 100 if not np.isnan(future_3m) else np.nan
                    ret_1y = (future_1y - entry_price) / entry_price * 100 if not np.isnan(future_1y) else np.nan
                    forward_returns.append((ret_1m, ret_3m, ret_1y))
        except:
            continue
    
    if forward_returns:
        returns_1m, returns_3m, returns_1y = zip(*forward_returns)
        print(f"1-month:  {np.nanmean(returns_1m):.2f}% (win rate: {(np.array(returns_1m) > 0).mean()*100:.1f}%)")
        print(f"3-month: {np.nanmean(returns_3m):.2f}% (win rate: {(np.array(returns_3m) > 0).mean()*100:.1f}%)")
        print(f"1-year:   {np.nanmean(returns_1y):.2f}% (win rate: {(np.array(returns_1y) > 0).mean()*100:.1f}%)")

# Scenario 3: Extreme Fear + Price BELOW MA200 (more typical fear scenario)
scenario3_mask = (df['FG'] < 25) & (df['SPY'] < df['SPY_MA200'])
scenario3_dates = df.index[scenario3_mask]

print(f"\n📈 Scenario 3: Extreme Fear + Price BELOW MA200 (Typical Fear)")
print(f"Found {len(scenario3_dates)} historical periods")

if len(scenario3_dates) > 0:
    forward_returns = []
    for date in scenario3_dates[:1000]:  # Limit for performance
        try:
            date_loc = df.index.get_loc(date)
            if date_loc + 252 < len(df):
                entry_price = df.loc[date, 'SPY']
                future_1m = df['SPY'].iloc[date_loc + 21] if date_loc + 21 < len(df) else np.nan
                future_3m = df['SPY'].iloc[date_loc + 63] if date_loc + 63 < len(df) else np.nan
                future_1y = df['SPY'].iloc[date_loc + 252] if date_loc + 252 < len(df) else np.nan
                
                if not np.isnan(future_1m):
                    ret_1m = (future_1m - entry_price) / entry_price * 100
                    ret_3m = (future_3m - entry_price) / entry_price * 100 if not np.isnan(future_3m) else np.nan
                    ret_1y = (future_1y - entry_price) / entry_price * 100 if not np.isnan(future_1y) else np.nan
                    forward_returns.append((ret_1m, ret_3m, ret_1y))
        except:
            continue
    
    if forward_returns:
        returns_1m, returns_3m, returns_1y = zip(*forward_returns)
        print(f"1-month:  {np.nanmean(returns_1m):.2f}% (win rate: {(np.array(returns_1m) > 0).mean()*100:.1f}%)")
        print(f"3-month: {np.nanmean(returns_3m):.2f}% (win rate: {(np.array(returns_3m) > 0).mean()*100:.1f}%)")
        print(f"1-year:   {np.nanmean(returns_1y):.2f}% (win rate: {(np.array(returns_1y) > 0).mean()*100:.1f}%)")

# ------------------------
# 6️⃣ MARKET REGIME ANALYSIS
# ------------------------

print("\n" + "="*70)
print("MARKET REGIME ANALYSIS")
print("="*70)

# Define market regimes based on FG and price position
conditions = [
    (df['FG'] < 25) & (df['SPY'] > df['SPY_MA200']),
    (df['FG'] < 25) & (df['SPY'] < df['SPY_MA200']),
    (df['FG'] > 75) & (df['SPY'] > df['SPY_MA200']),
    (df['FG'] > 75) & (df['SPY'] < df['SPY_MA200']),
]

regimes = [
    "Extreme Fear + Above MA200",
    "Extreme Fear + Below MA200", 
    "Extreme Greed + Above MA200",
    "Extreme Greed + Below MA200"
]

for condition, regime in zip(conditions, regimes):
    regime_dates = df.index[condition]
    if len(regime_dates) > 0:
        # Calculate 3-month forward returns
        returns = []
        for date in regime_dates[:500]:  # Sample for performance
            try:
                date_loc = df.index.get_loc(date)
                if date_loc + 63 < len(df):
                    entry_price = df.loc[date, 'SPY']
                    future_price = df['SPY'].iloc[date_loc + 63]
                    ret = (future_price - entry_price) / entry_price * 100
                    returns.append(ret)
            except:
                continue
        
        if returns:
            avg_return = np.mean(returns)
            win_rate = (np.array(returns) > 0).mean() * 100
            print(f"{regime:35} -> {len(regime_dates):4d} periods | 3M return: {avg_return:6.2f}% | win rate: {win_rate:5.1f}%")

# ------------------------
# 7️⃣ TRADING IMPLICATIONS
# ------------------------

print("\n" + "="*70)
print("TRADING IMPLICATIONS & INSIGHTS")
print("="*70)

print(f"\n🔍 CURRENT SITUATION: Extreme Fear (FG: {current_fg:.1f}) + Price +{current_vs_ma200:.1f}% above MA200")
print("This is a RARE combination that suggests:")
print("• 📉 Market is in extreme fear DESPITE being in an uptrend (above MA200)")
print("• ⚠️ This often occurs during sharp pullbacks within bull markets")
print("• 🎯 Could indicate panic selling during temporary corrections")

print(f"\n📊 Historical context for FG {current_fg:.1f}:")
fg_similar = df[(df['FG'] >= current_fg-2) & (df['FG'] <= current_fg+2)]
if len(fg_similar) > 0:
    similar_returns_3m = []
    for date in fg_similar.index[:500]:
        try:
            date_loc = df.index.get_loc(date)
            if date_loc + 63 < len(df):
                entry_price = df.loc[date, 'SPY']
                future_price = df['SPY'].iloc[date_loc + 63]
                ret = (future_price - entry_price) / entry_price * 100
                similar_returns_3m.append(ret)
        except:
            continue
    
    if similar_returns_3m:
        avg_3m = np.mean(similar_returns_3m)
        positive_pct = (np.array(similar_returns_3m) > 0).mean() * 100
        print(f"• 3-month forward returns after similar FG levels: {avg_3m:.2f}%")
        print(f"• Positive returns in {positive_pct:.1f}% of cases")

print(f"\n💡 STRATEGY SUGGESTIONS:")
print("1. Consider dollar-cost averaging given extreme fear levels")
print("2. Monitor for stabilization - extreme fear often precedes rebounds") 
print("3. Set stop losses given the negative short-term historical bias")
print("4. Look for confirmation of trend resumption before major commitments")

# Show the actual matching dates for transparency
print(f"\n📅 The {len(scenario2_dates)} historical matches occurred on:")
for date in scenario2_dates:
    fg_val = df.loc[date, 'FG']
    vs_ma = df.loc[date, 'SPY_vs_MA200']
    print(f"   {date.strftime('%Y-%m-%d')}: FG={fg_val:.1f}, +{vs_ma:.1f}% above MA200")