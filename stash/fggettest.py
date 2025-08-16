import requests
import pandas as pd
import numpy as np
import time
import datetime as dt

# 1) CNN Fear & Greed (using a historical date that exists)
# Use 2018-01-01 as starting point since we know that data exists
cnn_url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata/2018-01-01"
j = requests.get(cnn_url, headers={"User-Agent":"Mozilla/5.0"}).json()
fgi = pd.DataFrame(j["fear_and_greed_historical"]["data"])
fgi["date"] = pd.to_datetime(fgi["x"], unit="ms").dt.date
fgi = fgi.rename(columns={"y":"fgi"}).loc[:,["date","fgi"]]

# 2) SPY daily (Yahoo Finance download API)
# Use actual historical data instead of future dates
period1 = int(time.mktime(dt.datetime(2018,1,1).timetuple()))
period2 = int(time.mktime(dt.datetime.now().timetuple()))  # Current date
yf = f"https://query1.finance.yahoo.com/v7/finance/download/SPY?period1={period1}&period2={period2}&interval=1d&events=history&includeAdjustedClose=true"
spy = pd.read_csv(yf, parse_dates=["Date"])
spy["date"] = spy["Date"].dt.date
spy = spy[["date","Adj Close"]].rename(columns={"Adj Close":"close"})

# 3) Merge, forward-fill FGI (CNN sometimes misses weekends/holidays)
df = spy.merge(fgi, on="date", how="left").sort_values("date")
df["fgi"] = df["fgi"].ffill()

# 4) Signals: long when FGI <= 25; go to cash when FGI >= 75; otherwise hold last state
df["buy_sig"]  = (df["fgi"] <= 25)
df["sell_sig"] = (df["fgi"] >= 75)

# Trade state: enter next close after buy_sig, exit next close after sell_sig
state = []  # 1 = long, 0 = cash
longing = False
for i in range(len(df)):
    if i>0 and df.iloc[i-1]["buy_sig"] and not longing:
        longing = True
    if i>0 and df.iloc[i-1]["sell_sig"] and longing:
        longing = False
    state.append(1 if longing else 0)
df["position"] = state

# 5) Strategy equity (close-to-close returns when in position)
df["ret"] = df["close"].pct_change().fillna(0)
df["strat_ret"] = df["ret"] * df["position"]
df["strat_equity"] = (1 + df["strat_ret"]).cumprod()
df["bh_equity"] = (1 + df["ret"]).cumprod()

# 6) Stats
def max_drawdown(series):
    roll_max = series.cummax()
    dd = series/roll_max - 1
    return dd.min()

out = {
    "Start": df["date"].iloc[0].isoformat(),
    "End":   df["date"].iloc[-1].isoformat(),
    "Trades": int(((df["position"].diff().fillna(0))==1).sum()),
    "Strategy Total Return": f"{(df['strat_equity'].iloc[-1]-1):.2%}",
    "Buy&Hold Total Return": f"{(df['bh_equity'].iloc[-1]-1):.2%}",
    "Strategy Max DD": f"{max_drawdown(df['strat_equity']):.2%}",
    "Buy&Hold Max DD": f"{max_drawdown(df['bh_equity']):.2%}",
    "Time in Market": f"{df['position'].mean():.1%}",
}
print(out)

# Optional: list trades
trades = []
in_trade = False
for i in range(1, len(df)):
    if not in_trade and df.iloc[i-1]["buy_sig"]:
        entry_idx = i
        entry_date = df.iloc[i]["date"]
        entry_px = df.iloc[i]["close"]
        in_trade = True
    if in_trade and df.iloc[i-1]["sell_sig"]:
        exit_idx = i
        exit_date = df.iloc[i]["date"]
        exit_px = df.iloc[i]["close"]
        ret = (exit_px/entry_px - 1)
        trades.append({"entry":entry_date, "exit":exit_date, "ret_pct": round(100*ret,2)})
        in_trade = False

# If still in trade at the end, mark open P&L
if in_trade:
    exit_date = df.iloc[-1]["date"]
    exit_px = df.iloc[-1]["close"]
    ret = (exit_px/entry_px - 1)
    trades.append({"entry":entry_date, "exit":"OPEN"+exit_date.isoformat(), "ret_pct": round(100*ret,2)})

print(pd.DataFrame(trades))