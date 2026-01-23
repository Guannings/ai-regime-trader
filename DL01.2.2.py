#will the market crash???
#

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import StandardScaler

# 1. SETUP
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"📉 Crash Predictor initializing on {device}...")

# 2. DATA: Download S&P 500 (The Market)
ticker = "^GSPC"
data = yf.download(ticker, start="2000-01-01", end="2024-01-01")

# Feature Engineering: What does the AI see?
# 1. Returns (Daily % Change)
data['Returns'] = data['Close'].pct_change()
# 2. Volatility (30-Day Standard Deviation) - A classic fear gauge
data['Volatility'] = data['Returns'].rolling(window=30).std()

data = data.dropna()

# 3. CREATE LABELS (The "Truth")
# Definition of a "Crash": Price drops > 10% in the next 3 months (60 trading days)
target_window = 60
crash_threshold = -0.10

labels = []
prices = data['Close'].values
for i in range(len(prices) - target_window):
    # Look 3 months into the future
    future_price = prices[i + target_window]
    current_price = prices[i]

    # Calculate future return
    change = (future_price - current_price) / current_price

    # If the drop is worse than -10%, it's a CRASH (Label = 1)
    if change < crash_threshold:
        labels.append(1.0)  # YES
    else:
        labels.append(0.0)  # NO

# Trim data to match labels
data = data.iloc[:len(labels)]
# Input: Volatility and Returns
features = data[['Returns', 'Volatility']].values
labels = np.array(labels)

# 4. SPLIT & SCALE
split_idx = int(len(features) * 0.8)

scaler = StandardScaler()  # Standardize data (Mean=0, Std=1)
features_scaled = scaler.fit_transform(features)

x_train = torch.from_numpy(features_scaled[:split_idx]).float().to(device)
y_train = torch.from_numpy(labels[:split_idx]).float().to(device).unsqueeze(1)  # Shape needs to be (N, 1)

x_test = torch.from_numpy(features_scaled[split_idx:]).float().to(device)
y_test = torch.from_numpy(labels[split_idx:]).float().to(device).unsqueeze(1)


# 5. MODEL: The Classifier
class CrashBrain(nn.Module):
    def __init__(self):
        super(CrashBrain, self).__init__()
        self.layer1 = nn.Linear(2, 16)  # Input: 2 features (Return, Volatility)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(16, 1)  # Output: 1 Probability
        self.sigmoid = nn.Sigmoid()  # <--- THE MAGIC SWITCH (0 to 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.sigmoid(x)  # Squish output to probability
        return x


model = CrashBrain().to(device)

# 6. TRAINING (BCE Loss)
loss_function = nn.BCELoss()  # Binary Cross Entropy
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print("🚨 Training Crash Detector...")
epochs = 10000
for i in range(epochs):
    optimizer.zero_grad()
    y_pred = model(x_train)
    loss = loss_function(y_pred, y_train)
    loss.backward()
    optimizer.step()

    if i % 200 == 0:
        print(f"Epoch {i}: Loss = {loss.item():.4f}")

# 7. TESTING & VISUALIZATION
with torch.no_grad():
    test_probs = model(x_test).cpu().numpy()

# Plot
plt.figure(figsize=(12, 6))

# Plot the "Risk Score" (Probability of Crash)
plt.subplot(2, 1, 1)
plt.plot(test_probs, color='red', label='Crash Probability Guessed By AI')
plt.axhline(y=0.5, color='gray', linestyle='--', label='50% Threshold')
plt.title('AI Crash Warning System (Test Data)')
plt.legend()

# Plot the Actual Market (to see if it was right)
plt.subplot(2, 1, 2)
# We plot the actual Close prices corresponding to the test set
test_prices = data['Close'].values[split_idx:]
plt.plot(test_prices, color='green', label='S&P 500 Price')
plt.title('Actual Market Performance')
plt.legend()

plt.tight_layout()
plt.show()