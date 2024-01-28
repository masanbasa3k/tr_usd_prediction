import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
file_path = "tr_usd_exchance/exchange_rates_data.csv"
data_df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')

# Normalize the data
scaler = MinMaxScaler(feature_range=(-1, 1))
data_np = scaler.fit_transform(data_df['Value'].values.reshape(-1, 1))
data_tensor = torch.FloatTensor(data_np).view(-1)

# Helper function: Create an LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq.view(len(input_seq), 1, -1))
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

# Load the model
loaded_model = LSTMModel()
loaded_model.load_state_dict(torch.load('lstm_model.pth'))
loaded_model.eval()

# Get input from the user
forecast_days = int(input("How many days ahead do you want to forecast? "))

# Get the recent data
recent_data = data_tensor[-loaded_model.hidden_layer_size:]

# Forecast for the specified number of days
future_predictions = []

with torch.no_grad():
    for _ in range(forecast_days):
        input_seq = recent_data[-loaded_model.hidden_layer_size:].view(1, -1)
        with torch.no_grad():
            loaded_model.hidden = (torch.zeros(1, 1, loaded_model.hidden_layer_size),
                                   torch.zeros(1, 1, loaded_model.hidden_layer_size))
            prediction = loaded_model(input_seq.unsqueeze(0)).item()
            future_predictions.append(prediction)
            recent_data = torch.cat((recent_data, torch.FloatTensor([prediction])))

# Convert future predictions to the original scale
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Visualize future predictions and real values
plt.plot(data_df.index, data_df['Value'], label='Real Values')
plt.plot(pd.date_range(start=data_df.index[-1], periods=forecast_days+1, freq='D')[1:], future_predictions, color='orange', label=f'Future {forecast_days} Days Predictions')
plt.legend()
plt.show()
