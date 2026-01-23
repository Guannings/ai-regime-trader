import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

# 1. SETUP: Use M4 Pro GPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"🍏 Fetching Real World Data for AI on {device}...")

# 2. DATA: Download Apple Stock Data (Last 5 Years)
ticker = "AAPL"
data = yf.download(ticker, start="2020-01-01", end="2026-01-01")
closing_prices = data['Close'].values

# CRITICAL STEP: Normalization
# Neural Networks hate big numbers (like $230). We MUST squish them between 0 and 1.
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(closing_prices)


# 3. PREPARATION: Create "Sliding Windows"
# X = Past 60 Days, Y = Day 61 (The Target)
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


SEQ_LENGTH = 60  # Look back 60 days to predict tomorrow
x_data, y_data = create_sequences(scaled_data, SEQ_LENGTH)

# Convert to PyTorch Tensors on GPU
x_tensor = torch.from_numpy(x_data).float().to(device)
y_tensor = torch.from_numpy(y_data).float().to(device)


# 4. MODEL: A Slightly Bigger Brain
# Input Size = 60 (The past 60 days)
# Output Size = 1 (Tomorrow's Price)
class StockBrain(nn.Module):
    def __init__(self):
        super(StockBrain, self).__init__()
        self.layer1 = nn.Linear(SEQ_LENGTH, 128)  # Layer 1: Look for patterns in 60 days
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(128, 64)  # Layer 2: Refine the pattern
        self.layer3 = nn.Linear(64, 1)  # Output: The Price

    def forward(self, x):
        # We need to flatten the input because Linear layers expect flat lists
        x = x.view(x.shape[0], -1)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        return x


model = StockBrain().to(device)

# 5. TRAINING
# We use MSE (Variance) because we want to minimize the price error
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Low learning rate for stability

print("🚀 Training on Real Market Data...")
epochs = 500
for i in range(epochs):
    optimizer.zero_grad()
    y_pred = model(x_tensor)
    loss = loss_function(y_pred, y_tensor)
    loss.backward()
    optimizer.step()

    if i % 50 == 0:
        print(f"Epoch {i}: Loss (Variance) = {loss.item():.6f}")

# 6. VISUALIZATION
# Switch to CPU to plot
predicted_stock = model(x_tensor).cpu().detach().numpy()

# REVERSE the Normalization (Turn 0.5 back into $150)
predicted_prices = scaler.inverse_transform(predicted_stock)
real_prices = scaler.inverse_transform(y_tensor.cpu().numpy())

plt.figure(figsize=(10, 6))
plt.plot(real_prices, color='green', label=f'Actual {ticker} Price')
plt.plot(predicted_prices, color='red', label='AI Prediction')
plt.title(f'{ticker} Price Prediction using M4 Pro')
plt.xlabel('Days (Since Jan 2020)')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()