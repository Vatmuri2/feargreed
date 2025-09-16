import requests
import pandas as pd
from datetime import datetime
import numpy as np

# Define the base URL for CNN's Fear and Greed Index API
cnn_base_url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata/"

# Specify the start date for the data retrieval
start_date = "2025-08-09"  # Adjust this date as needed

# Set the headers to mimic a browser request
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:20.0) Gecko/20100101 Firefox/20.0'}

# Send a GET request to fetch the data
response = requests.get(cnn_base_url + start_date, headers=headers)

# Check if the request was successful
if response.status_code == 200:
    data = response.json()
    # Extract the Fear and Greed Index data
    fgi_data = data['fear_and_greed_historical']['data']
    
    # Convert the data into a pandas DataFrame
    df_index_new = pd.DataFrame(fgi_data)
    df_index_new['Date'] = pd.to_datetime(df_index_new['x'], unit='ms').dt.strftime('%Y-%m-%d')
    df_index_new = df_index_new.rename(columns={'x': 'Timestamp', 'y': 'Fear Greed'})
    df_index_new = df_index_new.drop(columns=['Timestamp'])

    df_index_new['Fear Greed'] = df_index_new['Fear Greed'].round(2)
    
    # Display the first few rows of the DataFrame

    df_index_new = df_index_new[['Date', 'Fear Greed', 'rating']]

    print(df_index_new.head())
    df_index_new['Fear Greed'] = np.floor(df_index_new['Fear Greed']).astype(int)
    # Save the DataFrame to a CSV file
    df_index_new.to_csv('datasets/present.csv', index=False)
else:
    print(f"Failed to retrieve data: {response.status_code}")
