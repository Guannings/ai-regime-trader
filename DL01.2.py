import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

# 1. SETUP
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"🍏 Honest AI Training on {device}...")

# 2. DATA: Fetch & Split
ticker = "AAPL"
data = yf.download(ticker, start="2021-01-01", end="2026-01-01")
closing_prices = data['Close'].values

# Calculate the "Cut Point" (80% for training, 20% for testing)
split_idx = int(len(closing_prices) * 0.8)
train_raw = closing_prices[:split_idx]
test_raw = closing_prices[split_idx:]

# NORMALIZATION (Crucial: Fit ONLY on training data to avoid cheating)
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_raw)  # Learn the range from history only

# Apply scale to both sets
train_scaled = scaler.transform(train_raw)
test_scaled = scaler.transform(test_raw)


# 3. HELPER: Create Sliding Windows
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


SEQ_LENGTH = 60

# Create separate datasets
x_train, y_train = create_sequences(train_scaled, SEQ_LENGTH)
x_test, y_test = create_sequences(test_scaled, SEQ_LENGTH)

# Convert to Tensors
x_train_tensor = torch.from_numpy(x_train).float().to(device)
y_train_tensor = torch.from_numpy(y_train).float().to(device)
x_test_tensor = torch.from_numpy(x_test).float().to(device)
y_test_tensor = torch.from_numpy(y_test).float().to(device)


# 4. MODEL (Same as before)
class StockBrain(nn.Module):
    def __init__(self):
        super(StockBrain, self).__init__()
        self.layer1 = nn.Linear(SEQ_LENGTH, 1028)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(1028, 64)
        self.layer3 = nn.Linear(64, 1)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        return x


model = StockBrain().to(device)

# 5. TRAINING (Only on x_train!)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(f"🚀 Training on first {split_idx} days only...")
epochs = 100000
for i in range(epochs):
    optimizer.zero_grad()
    y_pred = model(x_train_tensor)  # Look at study guide
    loss = loss_function(y_pred, y_train_tensor)
    loss.backward()
    optimizer.step()

    if i % 50 == 0:
        print(f"Epoch {i}: Loss = {loss.item():.6f}")

# 6. TESTING (The Final Exam)
model.eval()  # Switch to "Test Mode" (turns off learning features like Dropout)
with torch.no_grad():  # Don't calculate gradients, just predict
    test_predictions = model(x_test_tensor)

# 7. VISUALIZATION
# Move data back to CPU
train_predict = model(x_train_tensor).cpu().detach().numpy()
test_predict = test_predictions.cpu().detach().numpy()

# Inverse Transform (Convert back to Dollars)
train_predict_plot = scaler.inverse_transform(train_predict)
test_predict_plot = scaler.inverse_transform(test_predict)
y_train_plot = scaler.inverse_transform(y_train_tensor.cpu().numpy())
y_test_plot = scaler.inverse_transform(y_test_tensor.cpu().numpy())

# Plotting
plt.figure(figsize=(12, 6))

# Plot Training Data (The Past)
plt.plot(range(len(y_train_plot)), y_train_plot, color='green', label='Training Actual')
plt.plot(range(len(train_predict_plot)), train_predict_plot, color='orange', linestyle='--', label='AI Training Fit')

# Plot Testing Data (The Future / The Test)
test_start_idx = len(y_train_plot) + SEQ_LENGTH  # Offset for the graph
plt.plot(range(test_start_idx, test_start_idx + len(y_test_plot)), y_test_plot, color='blue',
         label='Test Actual (The Truth)')
plt.plot(range(test_start_idx, test_start_idx + len(test_predict_plot)), test_predict_plot, color='red',
         label='AI Test Prediction')

plt.title(f'The "Honest" Test: Training vs Testing ({ticker})')
plt.xlabel('Days')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()