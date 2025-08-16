import requests
import pandas as pd
from datetime import datetime

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
}

def fetch_data(start_date):
    url = f"https://production.dataviz.cnn.io/index/fearandgreed/graphdata/{start_date}"
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    data = r.json().get('fear_and_greed_historical', {}).get('data', [])
    return [
        {
            'Date': datetime.fromtimestamp(item['x'] / 1000).date(),
            'Index': round(item['y'], 2),
            'Rating': item['rating']
        }
        for item in data
    ]

# Break into yearly chunks to avoid 500 error
start_dates = [
    "2021-01-01", "2022-01-01",
    "2023-01-01", "2024-01-01"
]

all_data = []
for sd in start_dates:
    print(f"Fetching from {sd}...")
    all_data.extend(fetch_data(sd))

# Convert to DataFrame and deduplicate
df = pd.DataFrame(all_data).drop_duplicates(subset=['Date']).sort_values('Date').reset_index(drop=True)

df.to_csv("fear_greed_2020_present.csv", index=False)
print(df.head())
print(f"\nSaved {len(df)} rows to fear_greed_2020_present.csv")
