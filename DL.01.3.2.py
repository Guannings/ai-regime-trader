import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# --- CONFIGURATION ---
PERIOD = "59d"
INTERVAL = "1h"
SEQ_LENGTH = 24
EMBED_DIM = 64
NUM_HEADS = 4
EPOCHS = 150  # REDUCED: Stop before it memorizes
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.3  # NEW: High dropout to prevent memorization


# 1. THE TRANSFORMER (With Dropout)
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.att = nn.MultiHeadAttention(dims=embed_dim, num_heads=num_heads)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),  # <--- Brain fog added here
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)  # And here

    def __call__(self, x, mask=None):
        att_out = self.att(x, x, x, mask=mask)
        x = self.ln1(x + self.dropout(att_out))
        mlp_out = self.mlp(x)
        return self.ln2(x + self.dropout(mlp_out))


class HourlyGPT(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, seq_len, dropout):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.pos_embedding = nn.Embedding(seq_len, embed_dim)
        self.transformer = TransformerBlock(embed_dim, num_heads, dropout)
        self.head = nn.Linear(embed_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def __call__(self, x):
        batch_size, seq_len, _ = x.shape
        x = self.embedding(x)
        positions = mx.arange(seq_len)
        x = x + self.pos_embedding(positions)
        x = self.transformer(x)
        x = mx.mean(x, axis=1)
        return self.sigmoid(self.head(x))


# 2. GET DATA
print("📡 Downloading Hourly Data (SPY)...")
try:
    spy = yf.download("SPY", period=PERIOD, interval=INTERVAL, progress=False)
    if len(spy) == 0: raise ValueError("No data.")

    data = pd.DataFrame()
    if isinstance(spy.columns, pd.MultiIndex):
        data['Close'] = spy['Close']['SPY']
        data['High'] = spy['High']['SPY']
        data['Low'] = spy['Low']['SPY']
    else:
        data['Close'] = spy['Close']
        data['High'] = spy['High']
        data['Low'] = spy['Low']

    data = data.ffill().dropna()

except Exception as e:
    print(f"❌ DATA FAILURE: {e}")
    exit()

# 3. FEATURE ENGINEERING (Simpler & More Robust)
data['Returns'] = data['Close'].pct_change()
data['Range'] = (data['High'] - data['Low']) / data['Close']
# MOMENTUM (The Fix): 4-hour trend.
# If this is negative, the AI will learn "Don't Buy" even if price is low.
data['Mom_4h'] = data['Close'].pct_change(4)
# Volatility Acceleration
data['Vol_Change'] = data['Range'].diff()

data = data.dropna()

# 4. TARGET
data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
data = data.dropna()

# 5. PREPARE SEQUENCES
feature_cols = ['Returns', 'Range', 'Mom_4h', 'Vol_Change']
features = data[feature_cols].values
labels = data['Target'].values

mean = np.mean(features, axis=0)
std = np.std(features, axis=0)
features_scaled = (features - mean) / std


def create_sequences(feats, labs, seq_len):
    xs, ys = [], []
    for i in range(len(feats) - seq_len):
        xs.append(feats[i:i + seq_len])
        ys.append(labs[i + seq_len])
    return np.array(xs), np.array(ys)


x_seq, y_seq = create_sequences(features_scaled, labels, SEQ_LENGTH)

split_idx = int(len(x_seq) * 0.80)
x_train = mx.array(x_seq[:split_idx])
y_train = mx.array(y_seq[:split_idx]).reshape(-1, 1)
x_test = mx.array(x_seq[split_idx:])
y_test = mx.array(y_seq[split_idx:]).reshape(-1, 1)

# 6. TRAIN
model = HourlyGPT(input_dim=len(feature_cols), embed_dim=EMBED_DIM, num_heads=NUM_HEADS, seq_len=SEQ_LENGTH,
                  dropout=DROPOUT_RATE)
mx.eval(model.parameters())
optimizer = optim.Adam(learning_rate=LEARNING_RATE)


def loss_fn(model, x, y):
    pred = model(x)
    return -mx.mean(y * mx.log(pred + 1e-7) + (1 - y) * mx.log(1 - pred + 1e-7))


loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

print(f"🧠 Training (With Dropout, {EPOCHS} Epochs)...")
for i in range(EPOCHS):
    loss, grads = loss_and_grad_fn(model, x_train, y_train)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
    if i % 50 == 0:
        print(f"Epoch {i}: Loss = {loss.item():.4f}")

# 7. BACKTEST (SMOOTHED)
print("\n🎨 Calculating Positions...")
probs = np.array(model(x_test)).flatten()
test_prices = data['Close'].values[SEQ_LENGTH:][split_idx:]
test_dates = data.index[SEQ_LENGTH:][split_idx:]

portfolio = []
buy_hold = []
cash = 10000.0
stock_value = 0.0
exposures = []

# Smoothing Factor (0.0 to 1.0)
# 0.0 = No smoothing, 0.9 = Very slow turns
ALPHA = 0.7
current_exposure = 0.0

for i in range(len(test_prices)):
    price = float(test_prices[i])
    raw_prob = float(probs[i])

    # Target from AI
    raw_exposure = (raw_prob - 0.5) * 2.0

    # SMOOTHING: Don't flip instantly. Turn slowly.
    # New Exposure = 30% New Signal + 70% Old Signal
    current_exposure = (1 - ALPHA) * raw_exposure + (ALPHA) * current_exposure

    # Deadzone (Noise Filter)
    if abs(current_exposure) < 0.25:
        trade_exposure = 0.0
    else:
        trade_exposure = current_exposure

    exposures.append(trade_exposure)

    # Rebalance
    total_value = cash + stock_value
    target_stock_value = total_value * trade_exposure
    stock_value = target_stock_value
    cash = total_value - stock_value

    # Apply Return
    if i < len(test_prices) - 1:
        next_price = float(test_prices[i + 1])
        pct_change = (next_price - price) / price
        stock_value = stock_value * (1 + pct_change)

    portfolio.append(total_value)
    buy_hold.append(10000.0 / test_prices[0] * price)

# PLOTTING
plt.style.use('dark_background')
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

ax1.plot(test_dates, portfolio, label='Smoothed AI', color='#00ccff', linewidth=2)
ax1.plot(test_dates, buy_hold, label='Buy & Hold', color='gray', linestyle='--', alpha=0.7)
ax1.set_title(f"Portfolio Value (Dropout={DROPOUT_RATE})")
ax1.legend()

ax2.plot(test_dates, exposures, color='white', linewidth=1)
ax2.fill_between(test_dates, exposures, 0, where=(np.array(exposures) > 0), color='green', alpha=0.5)
ax2.fill_between(test_dates, exposures, 0, where=(np.array(exposures) < 0), color='red', alpha=0.5)
ax2.set_title("Smoothed Exposure (No Flickering)")
ax2.set_ylim(-1.1, 1.1)

ax3.plot(test_dates, test_prices, color='white', alpha=0.5)
ax3.set_title("Market Price")

plt.tight_layout()
plt.show()

print("\n" + "=" * 50)
print(f"📉 Buy & Hold:    ${buy_hold[-1]:,.2f}")
print(f"🤖 Smoothed AI:   ${portfolio[-1]:,.2f}")
print("=" * 50)