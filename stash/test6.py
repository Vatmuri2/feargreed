import pandas as pd
import yfinance as yf
import fear_and_greed as fg
from datetime import datetime

# ------------------------
# 1️⃣ Get current Fear & Greed
# ------------------------
current_fg = fg.get().value  # <-- use .value to get a float
print("Current Fear & Greed Index:", current_fg)

# ------------------------
# 2️⃣ Download latest SPY and VIX prices
# ------------------------
end_date = datetime.today()
start_date = end_date - pd.Timedelta(days=365)

spy = yf.download('SPY', start=start_date, end=end_date, auto_adjust=True)['Close']
vix = yf.download('^VIX', start=start_date, end=end_date, auto_adjust=True)['Close']

# ------------------------
# 3️⃣ Compute SPY indicators
# ------------------------
spy_ret = spy.pct_change()
spy_vol_20 = spy_ret.rolling(20).std() * (252**0.5)
spy_ma200 = spy.rolling(200).mean()

current_price = spy.iloc[-1].item()  # convert single-value Series to float
current_vol = spy_vol_20.iloc[-1].item()
current_ma200 = spy_ma200.iloc[-1].item()
current_vix = vix.iloc[-1].item()


# ------------------------
# 4️⃣ Check your entry conditions
# ------------------------
fg_threshold = 5.67
vol_max = 0.25
vix_max = 21.67
ma200_cond = 'above'

print("\nCurrent market snapshot:")
print(f"SPY Price: {current_price:.2f}")
print(f"SPY 200-day MA: {current_ma200:.2f}")
print(f"SPY 20-day Volatility: {current_vol:.2f}")
print(f"VIX: {current_vix:.2f}")
print(f"Fear & Greed: {current_fg:.2f}")

# Check conditions
fg_ok = current_fg <= fg_threshold
vol_ok = current_vol <= vol_max
vix_ok = current_vix <= vix_max
ma_ok = current_price >= current_ma200 if ma200_cond == 'above' else current_price < current_ma200

print("\nEntry conditions check:")
print(f"FG <= {fg_threshold}? {fg_ok}")
print(f"Vol <= {vol_max}? {vol_ok}")
print(f"VIX <= {vix_max}? {vix_ok}")
print(f"MA200 condition ({ma200_cond})? {ma_ok}")

if fg_ok and vol_ok and vix_ok and ma_ok:
    print("\n✅ All conditions met — potential entry signal!")
else:
    print("\n❌ Conditions not met — wait for better setup.")
