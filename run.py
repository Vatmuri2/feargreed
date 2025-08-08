import backtrader as bt
import pandas as pd
import numpy as np
import yfinance as yf


# ====================== DATA PREPARATION ======================
def prepare_data():
    # Load Fear & Greed data
    fear_greed = pd.read_csv("datasets/fear-greed.csv")
    fear_greed.columns = ['Date', 'Open', 'High', 'Low', 'fear_greed']
    fear_greed['Date'] = pd.to_datetime(fear_greed['Date'])

    # Get SPY price data from Yahoo Finance
    start_date = fear_greed['Date'].min().strftime('%Y-%m-%d')
    end_date = '2023-12-31'  # Update to current or actual end date
    spy_data = yf.download('SPY', start=start_date, end=end_date)
    spy_data.reset_index(inplace=True)  # Ensure single-level index
    spy_data.rename(columns={'Date': 'Date'}, inplace=True)

    # Get VIX data from Yahoo Finance
    vix_data = yf.download('^VIX', start=start_date, end=end_date)
    vix_data.reset_index(inplace=True)  # Ensure single-level index
    vix_data.rename(columns={'Date': 'Date', 'Close': 'vix'}, inplace=True)
    vix_data = vix_data[['Date', 'vix']]

    # Placeholder for Put/Call ratio data
    put_call_data = spy_data[['Date']].copy()
    put_call_data['put_call'] = 1.0  # Default value of 1.0

    # Merge data frames
    merged_data = pd.merge(spy_data, fear_greed[['Date', 'fear_greed']], on='Date', how='left')
    merged_data = pd.merge(merged_data, vix_data, on='Date', how='left')
    merged_data = pd.merge(merged_data, put_call_data, on='Date', how='left')

    # Handle missing values
    merged_data['fear_greed'].fillna(50, inplace=True)  # Default to neutral when missing
    merged_data['vix'].fillna(merged_data['vix'].mean(), inplace=True)
    merged_data['put_call'].fillna(1.0, inplace=True)

    # Save merged data to CSV
    merged_data.to_csv("datasets/spy-put-call-fear-greed-vix-complete.csv", index=False)
    return merged_data


# ====================== DATA LOADING (same as before) ======================
class SPYPutCallFearGreedVixData(bt.feeds.GenericCSVData):
    lines = ('put_call', 'fear_greed', 'vix')
    params = (
        ('dtformat', '%Y-%m-%d'),
        ('date', 0),
        ('open', 1),
        ('high', 2),
        ('low', 3),
        ('close', 4),
        ('adj', 5),
        ('volume', 6),
        ('put_call', 7),
        ('fear_greed', 8),
        ('vix', 9)
    )


# ====================== STRATEGY VARIANTS ======================
class BaseFearGreedStrategy(bt.Strategy):
    params = (
        ('fear_threshold', 20),
        ('greed_threshold', 80),
        ('use_vix', False),
        ('vix_threshold', 30),
        ('use_putcall', False),
        ('putcall_threshold', 1.0),
    )

    def __init__(self):
        self.fear_greed = self.datas[0].fear_greed
        self.close = self.datas[0].close
        if self.p.use_vix:
            self.vix = self.datas[0].vix
        if self.p.use_putcall:
            self.put_call = self.datas[0].put_call

    def next(self):
        if self.close[0] <= 0:
            return  # Skip if no valid price

        cash = self.broker.getcash()
        size = int(cash * 0.95 / self.close[0])  # Use 95% of cash, leave 5% buffer

        # Buy conditions
        buy_condition = self.fear_greed[0] < self.p.fear_threshold
        if self.p.use_vix:
            buy_condition &= self.vix[0] > self.p.vix_threshold
        if self.p.use_putcall:
            buy_condition &= self.put_call[0] > self.p.putcall_threshold

        if buy_condition and not self.position:
            self.buy(size=size)

        # Sell conditions
        sell_condition = self.fear_greed[0] > self.p.greed_threshold
        if self.p.use_vix:
            sell_condition |= self.vix[0] < self.p.vix_threshold * 0.7  # VIX drops significantly
        if self.p.use_putcall:
            sell_condition |= self.put_call[0] < self.p.putcall_threshold * 0.7

        if sell_condition and self.position.size > 0:
            self.sell(size=self.position.size)


# Optimize strategy
def run_strategy(**params):
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(100000)

    # Load data
    data = SPYPutCallFearGreedVixData(dataname="datasets/spy-put-call-fear-greed-vix-complete.csv")
    cerebro.adddata(data)

    # Check data range
    df = pd.read_csv("datasets/spy-put-call-fear-greed-vix-complete.csv")
    print(f"Data range: {df['Date'].min()} to {df['Date'].max()} ({len(df)} days)")

    # Add strategy
    cerebro.addstrategy(BaseFearGreedStrategy, **params)

    # Run backtest
    results = cerebro.run()
    final_value = cerebro.broker.getvalue()

    return final_value


# Prepare the data
prepare_data()

# Strategy variations to test
strategy_variations = [
    {"name": "Pure Fear/Greed", "params": {"use_vix": False, "use_putcall": False}},
    {"name": "Fear/Greed + VIX", "params": {"use_vix": True, "vix_threshold": 30, "use_putcall": False}},
    {"name": "Fear/Greed + Put/Call", "params": {"use_vix": False, "use_putcall": True, "putcall_threshold": 1.0}},
    {"name": "All Indicators", "params": {"use_vix": True, "vix_threshold": 30, "use_putcall": True, "putcall_threshold": 1.0}},
    {"name": "Aggressive", "params": {"fear_threshold": 25, "greed_threshold": 75, "use_vix": True, "vix_threshold": 25}},
]

# Run optimization
results = []
for strat in strategy_variations:
    final_value = run_strategy(**strat["params"])
    results.append({
        "Strategy": strat["name"],
        "Final Portfolio Value": final_value,
        "Return %": (final_value - 100000) / 100000 * 100,
        "Params": strat["params"]
    })

# Convert to DataFrame for analysis
results_df = pd.DataFrame(results)
print(results_df.sort_values("Final Portfolio Value", ascending=False))