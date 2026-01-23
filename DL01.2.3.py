#swapping the simple "Feed-Forward" brain for an LSTM.
#This allows the AI to analyze the sequence of events leading up to a crash, not just the snapshot of a single day.


import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import StandardScaler

# 1. SETUP
device = torch.device("cpu")
print(f"🧠 LSTM Crash Detector initializing on {device}...")

# 2. DATA
ticker = "^GSPC"  # S&P 500
data = yf.download(ticker, start="2000-01-01", end="2026-01-01")

# Feature Engineering
data['Returns'] = data['Close'].pct_change()
data['Volatility'] = data['Returns'].rolling(window=30).std()
data = data.dropna()

# 3. CREATE LABELS (Same logic as before)
target_window = 60  # 3 Months lookahead
crash_threshold = -0.10  # -10% drop

labels = []
prices = data['Close'].values
for i in range(len(prices) - target_window):
    future_price = prices[i + target_window]
    current_price = prices[i]
    change = (future_price - current_price) / current_price
    labels.append(1.0 if change < crash_threshold else 0.0)

# Align data length
data = data.iloc[:len(labels)]
features = data[['Returns', 'Volatility']].values
labels = np.array(labels)

# 4. SEQUENCE CREATION (Critical for LSTM)
# An LSTM needs a "history" window for every single prediction.
# We will give it the past 30 days to predict the crash probability.
SEQ_LENGTH = 30


def create_sequences(feats, labs, seq_len):
    xs, ys = [], []
    for i in range(len(feats) - seq_len):
        x = feats[i:(i + seq_len)]
        y = labs[i + seq_len]  # The label for the day AFTER the sequence ends
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


x_seq, y_seq = create_sequences(features, labels, SEQ_LENGTH)

# 5. SPLIT & SCALE
split_idx = int(len(x_seq) * 0.8)

# Scale features (We have to be careful with 3D shapes)
# Flatten -> Scale -> Reshape
scaler = StandardScaler()
N, L, F = x_seq.shape  # Number of samples, Sequence Length, Features
x_seq_flat = x_seq.reshape(N * L, F)
x_seq_scaled = scaler.fit_transform(x_seq_flat).reshape(N, L, F)

x_train = torch.from_numpy(x_seq_scaled[:split_idx]).float().to(device)
y_train = torch.from_numpy(y_seq[:split_idx]).float().to(device).unsqueeze(1)
x_test = torch.from_numpy(x_seq_scaled[split_idx:]).float().to(device)
y_test = torch.from_numpy(y_seq[split_idx:]).float().to(device).unsqueeze(1)


# 6. MODEL: The LSTM Architecture (Upgraded to 64 Neurons + Dropout)
class CrashLSTM(nn.Module):
    def __init__(self):
        super(CrashLSTM, self).__init__()
        # CHANGED: Memory increased from 32 to 64
        self.lstm = nn.LSTM(input_size=2, hidden_size=64, batch_first=True)

        # ADDED: Dropout to prevent "Conspiracy Theories" (Overfitting)
        self.dropout = nn.Dropout(0.2)

        # CHANGED: Input to this layer must match hidden_size (64)
        self.layer_out = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        last_step_out = out[:, -1, :]

        # ADDED: Apply the forgetting drug before deciding
        last_step_out = self.dropout(last_step_out)

        prediction = self.layer_out(last_step_out)
        return self.sigmoid(prediction)

model = CrashLSTM().to(device)

# 7. TRAINING
loss_function = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)  # Lower LR for LSTMs

print("🧠 Training LSTM Memory...")
epochs = 5000
for i in range(epochs):
    optimizer.zero_grad()
    y_pred = model(x_train)
    loss = loss_function(y_pred, y_train)
    loss.backward()
    optimizer.step()

    if i % 100 == 0:
        print(f"Epoch {i}: Loss = {loss.item():.4f}")

# 8. VISUALIZATION
with torch.no_grad():
    test_probs = model(x_test).cpu().numpy()

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(test_probs, color='purple', label='LSTM Crash Probability')  # Purple for "Royal" Intelligence
plt.axhline(y=0.5, color='gray', linestyle='--', label='50% Threshold')
plt.title('LSTM "Smart" Crash Detector (Test Data)')
plt.legend()

plt.subplot(2, 1, 2)
# Align prices with test data (accounting for seq_length offset)
test_price_start = split_idx + SEQ_LENGTH
test_prices = data['Close'].values[test_price_start:]

plt.plot(test_prices, color='green', label='S&P 500 Price')
plt.title('Actual Market Performance')
plt.legend()

plt.tight_layout()
plt.show()