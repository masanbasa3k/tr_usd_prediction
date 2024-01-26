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

df = pd.DataFrame(data)

df = df.iloc[:, :2]

# Save to CSV file
df.to_csv('tr_usd_exchance/exchange_rates_data.csv', index=False)


