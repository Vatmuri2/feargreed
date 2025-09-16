import yfinance as yf
import pandas as pd


file_path = "datasets/spy_data.csv"
start_date = '2022-01-01'
end_date = '2025-09-06'
print("Start Date:", start_date, "End Date:", end_date)


spy_data = yf.download('SPY', start=start_date, end=end_date, auto_adjust=True)
if spy_data.columns.nlevels > 1:
    spy_data.columns = spy_data.columns.droplevel(1)
spy_data.reset_index(inplace=True)

spy_data.drop(columns=['High', 'Low', 'Open', 'Volume'], inplace=True)
print(spy_data)

spy_data.to_csv(file_path, index=False)