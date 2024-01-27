import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("tr_usd_exchance/exchange_rates_data.csv")

# Display the column names in the DataFrame
print("Column names:", data.columns)

# Ensure that 'Date' is present in the DataFrame
if 'Date' not in data.columns:
    raise ValueError("Column 'Date' not found in the DataFrame.")

# Create a DataFrame
df = pd.DataFrame(data)

# Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Filter data for the year 2010 onwards
df = df[df['Date'].dt.year >= 2010]

# Sort the DataFrame by 'Date'
df = df.sort_values(by='Date')

# Create features (X) and target (y)
X = df[['Date']]
y = df['Value']

# Standardize the 'Date' feature
scaler = StandardScaler()
X_scaled = scaler.fit_transform(np.array(X).reshape(-1, 1))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Standardize the 'Date' feature using the same scaler
X_train_scaled = scaler.transform(np.array(X_train).reshape(-1, 1))
X_test_scaled = scaler.transform(np.array(X_test).reshape(-1, 1))

# Convert data to PyTorch tensors
X_train_tensor = torch.from_numpy(X_train_scaled).float()
X_test_tensor = torch.from_numpy(X_test_scaled).float()
y_train_tensor = torch.from_numpy(y_train.values).float()

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        x = self.fc(x)
        return x

# Create an instance of the model
model = SimpleNN()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
num_epochs = 2000
for epoch in range(num_epochs):
    # Forward pass
    predictions = model(X_train_tensor)
    loss = criterion(predictions, y_train_tensor)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the trained model (turn on if you want to download model)
torch.save(model.state_dict(), 'simple_nn_model.pth')

# Test the model
with torch.no_grad():
    predicted_values = model(X_test_tensor)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Value'], label='Actual values')
plt.plot(X_test, predicted_values.numpy(), label='Predicted values', linestyle='dashed', color='orange')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Neural Network Prediction')
plt.legend()
plt.show()
