import yfinance as yf
import pandas as pd

# Symbols for Turkish Lira and US Dollar
symbols = ['TRY=X', 'USDT=X']

# Data retrieval range and interval
start_date = '2023-01-01'
end_date = '2023-02-01'
interval = '1d'  # Daily basis

# Retrieve data
data = yf.download(symbols, start=start_date, end=end_date, interval=interval)

# Save to CSV file
csv_file_path = 'exchange_rates_data.csv'
data.to_csv(csv_file_path)

# Read the saved CSV file
loaded_data = pd.read_csv(csv_file_path)

# Display the loaded data
print(loaded_data)
