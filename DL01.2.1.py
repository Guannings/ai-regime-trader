import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

# 1. SETUP
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"☁️ MC Dropout Training on {device}...")

# 2. DATA
ticker = "AAPL"
data = yf.download(ticker, start="2021-01-01", end="2026-01-01")
closing_prices = data['Close'].values

split_idx = int(len(closing_prices) * 0.8)
train_raw = closing_prices[:split_idx]
test_raw = closing_prices[split_idx:]

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_raw)
train_scaled = scaler.transform(train_raw)
test_scaled = scaler.transform(test_raw)

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

SEQ_LENGTH = 60
x_train, y_train = create_sequences(train_scaled, SEQ_LENGTH)
x_test, y_test = create_sequences(test_scaled, SEQ_LENGTH)

x_train_tensor = torch.from_numpy(x_train).float().to(device)
y_train_tensor = torch.from_numpy(y_train).float().to(device)
x_test_tensor = torch.from_numpy(x_test).float().to(device)
y_test_tensor = torch.from_numpy(y_test).float().to(device)

# 3. MODEL: Added Dropout Layers
class BayesianStockBrain(nn.Module):
    def __init__(self):
        super(BayesianStockBrain, self).__init__()
        self.layer1 = nn.Linear(SEQ_LENGTH, 128)
        self.dropout1 = nn.Dropout(0.2) # Kill 20% of neurons
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.2) # Kill 20% of neurons
        self.layer3 = nn.Linear(64, 1)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.layer1(x)
        x = self.dropout1(x) # Apply damage
        x = self.relu(x)
        x = self.layer2(x)
        x = self.dropout2(x) # Apply damage
        x = self.relu(x)
        x = self.layer3(x)
        return x

model = BayesianStockBrain().to(device)

# 4. TRAINING
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(f"🚀 Training with Uncertainty...")
epochs = 50000
for i in range(epochs):
    model.train() # Enable Dropout
    optimizer.zero_grad()
    y_pred = model(x_train_tensor)
    loss = loss_function(y_pred, y_train_tensor)
    loss.backward()
    optimizer.step()
    if i % 100 == 0:
        print(f"Epoch {i}: Loss = {loss.item():.6f}")

# 5. MONTE CARLO PREDICTION (The Magic Step)
print("🔮 Running 100 Simulations per day...")
model.train() # CRITICAL: Keep Dropout ON to simulate different opinions

n_simulations = 100
mc_predictions = []

with torch.no_grad():
    for i in range(n_simulations):
        # Predict the ENTIRE test set 100 times, differently each time
        pred = model(x_test_tensor)
        mc_predictions.append(pred.cpu().numpy())

# Stack them into a shape of (100, Days, 1)
mc_predictions = np.array(mc_predictions).squeeze() # Shape: (100, Days)

# Calculate Mean and Standard Deviation (Uncertainty)
mean_pred_scaled = mc_predictions.mean(axis=0)
std_pred_scaled = mc_predictions.std(axis=0)

# Inverse Transform to Real Dollars
# Note: We need to shape it correctly for the scaler
mean_pred_price = scaler.inverse_transform(mean_pred_scaled.reshape(-1, 1))
# Approximate the upper/lower bound in dollars
upper_bound = scaler.inverse_transform((mean_pred_scaled + 2 * std_pred_scaled).reshape(-1, 1))
lower_bound = scaler.inverse_transform((mean_pred_scaled - 2 * std_pred_scaled).reshape(-1, 1))

y_test_plot = scaler.inverse_transform(y_test_tensor.cpu().numpy())

# 6. VISUALIZATION WITH CLOUD
plt.figure(figsize=(12,6))

# The Truth (Blue)
test_start_idx = len(y_train_tensor) + SEQ_LENGTH
plt.plot(range(test_start_idx, test_start_idx + len(y_test_plot)), y_test_plot, color='blue', label='Actual Price', linewidth=2)

# The AI Consensus (Red Line)
plt.plot(range(test_start_idx, test_start_idx + len(mean_pred_price)), mean_pred_price, color='green', linestyle='--', label='AI Mean Prediction')

# The Uncertainty Cloud (Shaded Area)
# We flatten the arrays to make matplotlib happy
plt.fill_between(
    range(test_start_idx, test_start_idx + len(mean_pred_price)),
    lower_bound.flatten(),
    upper_bound.flatten(),
    color='red',
    alpha=0.3, # Transparency
    label='95% Confidence Interval (Uncertainty)'
)

plt.title(f'Monte Carlo Dropout: {ticker} Price Prediction with Risk')
plt.xlabel('Days')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()