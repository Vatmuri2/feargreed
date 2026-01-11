import pandas as pd
import numpy as np
import yfinance as yf
from itertools import product

# ------------------------
# 1️⃣ Load Fear & Greed Data
# ------------------------
fg_file = "datasets/fear_greed_combined_2011_2025.csv"
fg = pd.read_csv(fg_file, parse_dates=['Date'])
fg.set_index('Date', inplace=True)
fg['fear_greed'] = fg['fear_greed'].ffill()

# ------------------------
# 2️⃣ Download SPY and VIX Data
# ------------------------
spy = yf.download('SPY', start=fg.index.min(), end=fg.index.max(), auto_adjust=True)['Close'].squeeze()
vix = yf.download('^VIX', start=fg.index.min(), end=fg.index.max(), auto_adjust=True)['Close'].squeeze()

# Remove duplicate dates
fg = fg[~fg.index.duplicated(keep='first')]
spy = spy[~spy.index.duplicated(keep='first')]
vix = vix[~vix.index.duplicated(keep='first')]

# Combine into DataFrame
df = pd.DataFrame({
    'FG': fg['fear_greed'],
    'SPY': spy,
    'VIX': vix
}).sort_index().ffill().dropna()

# ------------------------
# 3️⃣ Compute SPY Indicators
# ------------------------
df['SPY_ret'] = df['SPY'].pct_change()
df['SPY_vol_20'] = df['SPY_ret'].rolling(20).std() * np.sqrt(252)
df['SPY_vol_50'] = df['SPY_ret'].rolling(50).std() * np.sqrt(252)
df['SPY_MA20'] = df['SPY'].rolling(20).mean()
df['SPY_MA50'] = df['SPY'].rolling(50).mean()
df['SPY_MA200'] = df['SPY'].rolling(200).mean()

# ------------------------
# 4️⃣ Identify Low F&G Periods & Forward Metrics
# ------------------------
df['is_low_fg'] = df['FG'] <= 5
df['low_group'] = (df['is_low_fg'] != df['is_low_fg'].shift()).cumsum()
low_periods = df[df['is_low_fg']].groupby('low_group')

rebound_table = []
for _, group in low_periods:
    entry_date = group.index[0]
    entry_price = df.loc[entry_date, 'SPY']
    min_fg = group['FG'].min()
    min_fg_date = group['FG'].idxmin()
    days_in_low = len(group)
    fg_trend = group['FG'].iloc[-1] - group['FG'].iloc[0]
    
    forward_periods = [30, 60, 90, 180, 365]
    forward_metrics = {}
    for days in forward_periods:
        future_idx = df.index.get_loc(entry_date) + days
        if future_idx < len(df):
            future_prices = df['SPY'].iloc[df.index.get_loc(entry_date)+1:future_idx+1]
            fwd_ret = (future_prices.iloc[-1] - entry_price) / entry_price * 100
            max_gain = (future_prices.max() - entry_price) / entry_price * 100
            max_draw = (future_prices.min() - entry_price) / entry_price * 100
            rr = max_gain / abs(max_draw) if max_draw != 0 else np.nan
            forward_metrics[f'{days}d_Return'] = fwd_ret
            forward_metrics[f'{days}d_MaxGain'] = max_gain
            forward_metrics[f'{days}d_MaxDrawdown'] = max_draw
            forward_metrics[f'{days}d_RR'] = rr
        else:
            forward_metrics[f'{days}d_Return'] = np.nan
            forward_metrics[f'{days}d_MaxGain'] = np.nan
            forward_metrics[f'{days}d_MaxDrawdown'] = np.nan
            forward_metrics[f'{days}d_RR'] = np.nan
    
    rebound_table.append({
        'Entry Date': entry_date,
        'Entry Price': entry_price,
        'Min F&G': min_fg,
        'Min F&G Date': min_fg_date,
        'Days in Low F&G': days_in_low,
        'F&G Trend': fg_trend,
        **forward_metrics
    })

rebound_df = pd.DataFrame(rebound_table)
rebound_df.to_csv('fear_greed_rebound_metrics_enhanced.csv', index=False)

# ------------------------
# 5️⃣ Test Indicator Combinations
# ------------------------
fg_thresh_range = [2, 3, 4, 5, 6]
days_low_range = [1, 2, 3, 4, 5]
fg_trend_min = [0, 0.25, 0.5]
ma_conditions = ['above', 'below']
vol_thresh = [0.1, 0.15, 0.2, 0.25]
vix_thresh = [20, 25, 30]

results = []
for fg_th, days_low, fg_tr, ma_cond, vol_max, vix_max in product(
    fg_thresh_range, days_low_range, fg_trend_min, ma_conditions, vol_thresh, vix_thresh
):
    filtered = rebound_df[
        (rebound_df['Min F&G'] <= fg_th) &
        (rebound_df['Days in Low F&G'] >= days_low) &
        (rebound_df['F&G Trend'] >= fg_tr)
    ].copy()
    if len(filtered) == 0:
        continue
    
    # MA200 filter
    ma200_vals = df['SPY_MA200'].reindex(filtered['Entry Date']).values
    if ma_cond == 'above':
        filtered = filtered[filtered['Entry Price'].values >= ma200_vals]
    else:
        filtered = filtered[filtered['Entry Price'].values < ma200_vals]
    
    # Volatility filter
    vol_vals = df['SPY_vol_20'].reindex(filtered['Entry Date']).values
    filtered = filtered[vol_vals <= vol_max]
    
    # VIX filter
    vix_vals = df['VIX'].reindex(filtered['Entry Date']).values
    filtered = filtered[vix_vals <= vix_max]
    
    if len(filtered) == 0:
        continue
    
    metrics = {}
    for p in [30, 60, 90, 180, 365]:
        ret_col = f'{p}d_Return'
        rr_col = f'{p}d_RR'
        metrics[f'{p}d_MeanReturn'] = filtered[ret_col].mean()
        metrics[f'{p}d_StdReturn'] = filtered[ret_col].std()
        metrics[f'{p}d_Sharpe'] = (filtered[ret_col].mean() / filtered[ret_col].std() if filtered[ret_col].std() != 0 else np.nan)
        metrics[f'{p}d_TotalRR'] = filtered[rr_col].sum()
    
    metrics.update({
        'FG_Threshold': fg_th,
        'Days_in_Low_Min': days_low,
        'FG_Trend_Min': fg_tr,
        'MA200_condition': ma_cond,
        'Vol_Max': vol_max,
        'VIX_Max': vix_max,
        'Num_Entries': len(filtered)
    })
    results.append(metrics)

comb_df = pd.DataFrame(results)
comb_df.sort_values(by='365d_TotalRR', ascending=False, inplace=True)
comb_df.to_csv('fear_greed_combination_metrics.csv', index=False)
comb_df.head(10).to_csv('fear_greed_top_combinations.csv', index=False)

# ------------------------
# 6️⃣ Analysis Output
# ------------------------
metric_cols = [c for c in comb_df.columns if 'Return' in c or 'Sharpe' in c or 'TotalRR' in c]
comb_df[metric_cols] = comb_df[metric_cols].apply(pd.to_numeric, errors='coerce')

best_sharpe_365 = comb_df.sort_values('365d_Sharpe', ascending=False).head(3)
best_rr_365 = comb_df.sort_values('365d_TotalRR', ascending=False).head(3)

def print_analysis(title, subset):
    print(f"\n{'='*20} {title} {'='*20}")
    for idx, row in subset.iterrows():
        print(f"\nEntry Parameters:")
        print(f" FG Threshold: {row['FG_Threshold']}, Days in Low: {row['Days_in_Low_Min']}, FG Trend: {row['FG_Trend_Min']}, MA200: {row['MA200_condition']}, Vol_Max: {row['Vol_Max']}, VIX_Max: {row['VIX_Max']}")
        print(f"Performance Metrics:")
        for p in [30, 60, 90, 180, 365]:
            print(f" {p}d: Sharpe={row[f'{p}d_Sharpe']:.2f}, Total R/R={row[f'{p}d_TotalRR']:.2f}")
        print(f" Number of entries: {row['Num_Entries']}")

print_analysis("Top 3 by 365d Sharpe", best_sharpe_365)
print_analysis("Top 3 by 365d Total R/R", best_rr_365)

# Summary of averages
avg_best_sharpe = best_sharpe_365[['FG_Threshold','Days_in_Low_Min','Vol_Max','VIX_Max']].mean()
avg_best_rr = best_rr_365[['FG_Threshold','Days_in_Low_Min','Vol_Max','VIX_Max']].mean()

print("\nSummary of Optimal Conditions (Averages of Top 3 by Sharpe):")
print(f" FG Threshold ~ {avg_best_sharpe['FG_Threshold']:.2f}")
print(f" Days in Low ~ {avg_best_sharpe['Days_in_Low_Min']:.2f}")
print(f" Vol_Max ~ {avg_best_sharpe['Vol_Max']:.2f}")
print(f" VIX_Max ~ {avg_best_sharpe['VIX_Max']:.2f}")

print("\n✅ Done. Top combinations saved to 'fear_greed_top_combinations.csv'")
