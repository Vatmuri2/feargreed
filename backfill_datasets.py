"""
backfill_datasets.py — One-time script to fill gaps in BOD/EOD F&G datasets.

The CNN API returns ~252 days of daily history. Run this once on the Pi to
patch both datasets before the bots need accurate 3-day rolling indicators.

    python3 backfill_datasets.py
"""

import datetime
import pandas as pd
from fear_and_greed.cnn import Fetcher

BOD_PATH = 'datasets/fear_greed_forward_test_morning.csv'
EOD_PATH = 'datasets/fear_greed_forward_test_afternoon.csv'


def fetch_historical():
    f = Fetcher()
    data = f()
    hist = data['fear_and_greed_historical']['data']
    rows = []
    for p in hist:
        date = datetime.datetime.fromtimestamp(p['x'] / 1000).date()
        rows.append({'Date': date, 'Fear Greed': round(p['y'], 2), 'rating': p['rating']})
    df = pd.DataFrame(rows).drop_duplicates('Date').sort_values('Date')
    print(f"Fetched {len(df)} historical points: {df['Date'].iloc[0]} → {df['Date'].iloc[-1]}")
    return df


def backfill(path, historical_df):
    existing = pd.read_csv(path)
    existing['Date'] = pd.to_datetime(existing['Date']).dt.date

    # Determine the FGI column name
    if 'fear_greed' in existing.columns:
        fgi_col = 'fear_greed'
    elif 'Fear Greed' in existing.columns:
        fgi_col = 'Fear Greed'
    else:
        fgi_col = existing.columns[1]

    existing_dates = set(existing['Date'])
    historical_df = historical_df.rename(columns={'Fear Greed': fgi_col})

    new_rows = historical_df[~historical_df['Date'].isin(existing_dates)].copy()

    if new_rows.empty:
        print(f"  {path}: already up to date, no gaps found.")
        return

    combined = pd.concat([existing, new_rows], ignore_index=True)
    combined = combined.sort_values('Date').reset_index(drop=True)
    combined.to_csv(path, index=False)

    print(f"  {path}: added {len(new_rows)} missing rows "
          f"({new_rows['Date'].min()} → {new_rows['Date'].max()}). "
          f"Total rows: {len(combined)}")


def main():
    print("Fetching CNN historical F&G data...")
    historical = fetch_historical()

    print("\nBackfilling datasets:")
    backfill(BOD_PATH, historical.copy())
    backfill(EOD_PATH, historical.copy())

    print("\nDone. Re-check datasets:")
    for path in [BOD_PATH, EOD_PATH]:
        df = pd.read_csv(path)
        print(f"  {path}: {len(df)} rows, last date: {df['Date'].iloc[-1]}")


if __name__ == '__main__':
    main()
