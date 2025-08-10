import pandas as pd
from io import StringIO

data = "trade_log.csv"  # replace with actual CSV text

df = pd.read_csv(StringIO(data), header=None, names=[
    'start_date', 'end_date', 'start_price', 'end_price', 'return', 'duration', 
    'trade_result', 'val1', 'val2', 'val3', 'val4', 'trend', 'metric1', 'metric2'
])

# Check for stop_loss with return >= 0 (including zero and slightly positive)
stop_loss_non_negative = df[(df['trade_result'] == 'stop_loss') & (df['return'] >= 0)]
print(stop_loss_non_negative)
