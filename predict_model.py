import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Load the dataset
file_path = "texchange_rates_data.csv"
data_df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')

# Normalize the data
scaler = MinMaxScaler(feature_range=(-1, 1))
data_np = scaler.fit_transform(data_df['Value'].values.reshape(-1, 1))
data_tensor = torch.FloatTensor(data_np).view(-1)

# Split the data into training and test sets
train_size = int(len(data_tensor) * 0.8)
train_data, test_data = data_tensor[0:train_size], data_tensor[train_size:]

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

# Create the model
model = LSTMModel()

# Loss function and optimizer
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
epochs = 10

for epoch in range(epochs):
    for i in range(len(train_data) - 1):
        input_seq = train_data[i:i+1]
        target = train_data[i+1:i+2]

        optimizer.zero_grad()
        model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))

        prediction = model(input_seq)

        loss = loss_function(prediction, target)
        loss.backward()
        optimizer.step()

    # Print training loss every 10 epochs
    if epoch % 10 == 0:
        print(f'Epoch {epoch}/{epochs}, Loss: {loss.item()}')

# Save the model
torch.save(model.state_dict(), 'lstm_model.pth')

# Testing
model.eval()
test_predictions = []

for i in range(len(test_data)):
    input_seq = test_data[i:i+1]
    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
        test_predictions.append(model(input_seq).item())

# Compare predictions with real data
test_predictions = scaler.inverse_transform(np.array(test_predictions).reshape(-1, 1))
real_data = scaler.inverse_transform(test_data.reshape(-1, 1))

# Calculate error metrics
mse = mean_squared_error(real_data, test_predictions)
print('Mean Squared Error:', mse)

# Visualize predictions and real values with dates on the x-axis
plt.figure(figsize=(10, 6))
plt.plot(data_df.index[:-len(test_data)], scaler.inverse_transform(train_data.reshape(-1, 1)), label='Training Data')
plt.plot(data_df.index[-len(test_data):], real_data, label='Real Values')
plt.plot(data_df.index[-len(test_data):], test_predictions, color='red', label='Predictions')
plt.xlabel('Date')
plt.ylabel('Exchange Rate')
plt.title('Exchange Rate Prediction with LSTM')
plt.legend()
plt.show()
