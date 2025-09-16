import pandas as pd
import datetime, time
import fear_and_greed as fg
import alpaca_trade_api as tradeapi
import numpy as np
import pandas_market_calendars as mcal


API_KEY = 'PKVVMQPZU4DBWK9DH51G'
API_SECRET = 'Nm1UpSPe7xzyYeQOtVGTSwUG2xbgMRstpfmsfGGb'
BASE_URL = 'https://paper-api.alpaca.markets'
SYMBOL = 'SPY'
FG_PATH = 'datasets/present.csv'
LOOKBACK_DAYS = 3


def fetch_and_store_fgi(date, fg_path):

    fg_data = pd.read_csv(fg_path)
    fg_data['Date'] = pd.to_datetime(fg_data['Date'], errors='coerce')
    fg_data['Date'] = fg_data['Date'].dt.date


    curr_fgi = round(fg.get().value, 2)
    curr_date = fg.get().last_update.date()

    if (fg_data['Date'][len(fg_data) - 1] != curr_date):
        new_row = pd.DataFrame({'Date': [curr_date], 'Index': [curr_fgi]})
        fg_data = pd.concat([fg_data, new_row], ignore_index=True)
        fg_data.to_csv(fg_path, index=False)

"""
def calculate_indicators(date, fg_data_path,fg_indicators_path ):
    fg_data = pd.read_csv(fg_data_path)
    fg_data['Date'] = pd.to_datetime(fg_data['Date'], errors='coerce')
    fg_data['Date'] = fg_data['Date'].dt.date

    fg_indicators = pd.read_csv(fg_indicators_path)
    fg_indicators['Date'] = pd.to_datetime(fg_data['Date'], errors='coerce')
    fg_indicators['Date'] = fg_data['Date'].dt.date

    # Fear & Greed momentum and velocity calculations
    fg_indicators['fg_momentum'] = fg_data['fear_greed'] - fg_data['fear_greed'].rolling(LOOKBACK_DAYS, min_periods=1).mean()
    fg_indicators['fg_change'] = fg_data['fear_greed'].diff().fillna(0)
    fg_indicators['fg_velocity'] = fg_data['fg_change'].rolling(LOOKBACK_DAYS, min_periods=1).mean()
    
    # Price analysis for volatility
    fg_indicators['price_returns'] = fg_indicators[price_col].pct_change().fillna(0)
    fg_indicators['volatility'] = fg_indicators['price_returns'].rolling(20, min_periods=1).std() * np.sqrt(252)
"""


if __name__ == "__main__":
    nyse = mcal.get_calendar('NYSE')
    today = datetime.date.today()
    schedule = nyse.schedule(start_date=today, end_date=today)
    is_trading_day = not schedule.empty

    print(is_trading_day)

    fg_data = pd.read_csv(FG_PATH)
    fg_data['Date'] = pd.to_datetime(fg_data['Date'], errors='coerce')
    fg_data['Date'] = fg_data['Date'].dt.date

    spy_data = pd.read_csv('datasets/spy_data.csv')
    spy_data['Date'] = pd.to_datetime(spy_data['Date'], errors='coerce')
    spy_data['Date'] = spy_data['Date'].dt.date

    aligned_fg = pd.DataFrame({"Date": spy_data['Date']})

    aligned_fg = aligned_fg.merge(fg_data, on="Date", how="left")

    num_missing = aligned_fg['Index'].isna().sum()
    print(f"Number of rows to be forward-filled: {num_missing}")

    aligned_fg['Index'] = aligned_fg['Index'].ffill()

    aligned_fg.to_csv('fear_greed_aligned.csv', index=False)





    print("HI")