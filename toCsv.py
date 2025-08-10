import pandas as pd
from datetime import datetime

# Read the CSV files
df1 = pd.read_csv('datasets/fear_greed_2020_present.csv')
df2 = pd.read_csv('datasets/fear-greed-2011-2023.csv')

# Convert date columns to datetime
df1['Date'] = pd.to_datetime(df1['Date'])
df2['Date'] = pd.to_datetime(df2['Date'])

# Check if df2 has a Rating column, if not, create it based on fear_greed values
if 'Rating' not in df2.columns:
    def get_rating(value):
        if value <= 25:
            return 'extreme fear'
        elif value <= 45:
            return 'fear'
        elif value <= 55:
            return 'neutral'
        elif value <= 75:
            return 'greed'
        else:
            return 'extreme greed'
    
    df2['Rating'] = df2['fear_greed'].apply(get_rating)

# Make sure both dataframes have the same column structure
df1 = df1[['Date', 'fear_greed', 'Rating']]
df2 = df2[['Date', 'fear_greed', 'Rating']]

# Find the overlap period
overlap_start = max(df1['Date'].min(), df2['Date'].min())
overlap_end = min(df1['Date'].max(), df2['Date'].max())

print(f"Overlap period: {overlap_start} to {overlap_end}")

# For the overlap period, prioritize df1 (2020-present) as it might be more recent/accurate
# Keep df2 data before overlap and df1 data from overlap onwards
df2_before_overlap = df2[df2['Date'] < overlap_start]
df1_from_overlap = df1[df1['Date'] >= overlap_start]

# Combine the datasets
combined_df = pd.concat([df2_before_overlap, df1_from_overlap], ignore_index=True)

# Sort by date
combined_df = combined_df.sort_values('Date').reset_index(drop=True)

# Remove any duplicate dates (keep the first occurrence)
combined_df = combined_df.drop_duplicates(subset=['Date'], keep='first')

# Display info about the combined dataset
print(f"\nCombined dataset:")
print(f"Date range: {combined_df['Date'].min()} to {combined_df['Date'].max()}")
print(f"Total records: {len(combined_df)}")
print(f"Date gaps: {len(pd.date_range(combined_df['Date'].min(), combined_df['Date'].max(), freq='D')) - len(combined_df)} days")

print("\nFirst few rows:")
print(combined_df.head())
print("\nLast few rows:")
print(combined_df.tail())

# Save to CSV
combined_df.to_csv('datasets/fear_greed_combined_2011_2025.csv', index=False)
print(f"\nCombined dataset saved to 'datasets/fear_greed_combined_2011_2025.csv'")