import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# --- CONFIGURATION ---
PERIOD = "59d"  # Safe limit for hourly data
INTERVAL = "1h"  # Resolution
SEQ_LENGTH = 24  # Look back 24 hours
EMBED_DIM = 64
NUM_HEADS = 4
EPOCHS = 300
LEARNING_RATE = 0.001


# 1. THE TRANSFORMER (MLX)
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.att = nn.MultiHeadAttention(dims=embed_dim, num_heads=num_heads)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def __call__(self, x, mask=None):
        att_out = self.att(x, x, x, mask=mask)
        x = self.ln1(x + att_out)
        mlp_out = self.mlp(x)
        return self.ln2(x + mlp_out)


class HourlyGPT(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, seq_len):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.pos_embedding = nn.Embedding(seq_len, embed_dim)
        self.transformer = TransformerBlock(embed_dim, num_heads)
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


# 2. GET HOURLY DATA (Robust)
print("📡 Downloading Hourly Data (SPY)...")
try:
    # We ONLY download SPY. We do not touch VIX to avoid errors.
    spy = yf.download("SPY", period=PERIOD, interval=INTERVAL, progress=False)

    if len(spy) == 0:
        raise ValueError("Yahoo Finance returned 0 rows.")

    # Clean Data
    data = pd.DataFrame()
    if isinstance(spy.columns, pd.MultiIndex):
        data['Close'] = spy['Close']['SPY']
        data['High'] = spy['High']['SPY']
        data['Low'] = spy['Low']['SPY']
    else:
        data['Close'] = spy['Close']
        data['High'] = spy['High']
        data['Low'] = spy['Low']

    # CREATE THE VIX PROXY (Calculated, not Downloaded)
    # This prevents the KeyError 'VIX'
    data['VIX_Proxy'] = (data['High'] - data['Low']) / data['Close'] * 100.0

    data = data.ffill().dropna()
    print(f"✅ Downloaded {len(data)} hours of data.")

except Exception as e:
    print(f"❌ DATA FAILURE: {e}")
    exit()

# 3. FEATURE ENGINEERING
# We use 'VIX_Proxy' instead of 'VIX'
data['Returns'] = data['Close'].pct_change()
data['Range'] = (data['High'] - data['Low']) / data['Close']
data['VIX_Change'] = data['VIX_Proxy'].diff()  # <--- FIXED LINE
data['Hour'] = data.index.hour
data['Hour_Norm'] = (data['Hour'] - 9) / 7.0
# RSI
data['RSI'] = 100 - (100 / (1 + data['Returns'].rolling(14).apply(
    lambda x: x[x > 0].sum() / abs(x[x < 0].sum()) if abs(x[x < 0].sum()) > 0 else 1)))
data = data.dropna()

# 4. TARGET
data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
data = data.dropna()

# 5. PREPARE SEQUENCES
feature_cols = ['Returns', 'Range', 'VIX_Change', 'Hour_Norm', 'RSI']
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

# Split
split_idx = int(len(x_seq) * 0.80)
x_train = mx.array(x_seq[:split_idx])
y_train = mx.array(y_seq[:split_idx]).reshape(-1, 1)
x_test = mx.array(x_seq[split_idx:])
y_test = mx.array(y_seq[split_idx:]).reshape(-1, 1)

# 6. TRAIN
model = HourlyGPT(input_dim=len(feature_cols), embed_dim=EMBED_DIM, num_heads=NUM_HEADS, seq_len=SEQ_LENGTH)
mx.eval(model.parameters())
optimizer = optim.Adam(learning_rate=LEARNING_RATE)


def loss_fn(model, x, y):
    pred = model(x)
    return -mx.mean(y * mx.log(pred + 1e-7) + (1 - y) * mx.log(1 - pred + 1e-7))


loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

print(f"🧠 Training Hourly Transformer ({EPOCHS} Epochs)...")
for i in range(EPOCHS):
    loss, grads = loss_and_grad_fn(model, x_train, y_train)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
    if i % 50 == 0:
        print(f"Epoch {i}: Loss = {loss.item():.4f}")

# 7. BACKTEST
print("\n🎨 Generating Graphs...")
probs = np.array(model(x_test)).flatten()
test_prices = data['Close'].values[SEQ_LENGTH:][split_idx:]
test_dates = data.index[SEQ_LENGTH:][split_idx:]

portfolio = []
buy_hold = []
cash = 10000.0
shares = 0
in_market = False
initial_shares = 10000.0 / test_prices[0]

correct_preds = 0
total_preds = 0

for i in range(len(test_prices)):
    price = float(test_prices[i])
    prob = float(probs[i])

    # STRATEGY
    if prob > 0.60:
        if not in_market:
            shares = cash / price
            cash = 0
            in_market = True
    elif prob < 0.40:
        if in_market:
            cash = shares * price
            shares = 0
            in_market = False

    # Stats
    actual_move = 1 if (i < len(test_prices) - 1 and test_prices[i + 1] > price) else 0
    pred_move = 1 if prob > 0.5 else 0
    if actual_move == pred_move:
        correct_preds += 1
    total_preds += 1

    val = (shares * price) if in_market else cash
    portfolio.append(val)
    buy_hold.append(initial_shares * price)

# PLOT
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
ax1.plot(test_dates, portfolio, label='Hourly AI', color='blue')
ax1.plot(test_dates, buy_hold, label='Buy & Hold', color='gray', linestyle='--')
ax1.set_title("Hourly Strategy")
ax1.legend()
sns.histplot(probs, bins=50, kde=True, ax=ax2, color='purple')
ax2.set_title("Confidence Distribution")
plt.tight_layout()
plt.savefig('hourly_results.png')
print("📸 Saved 'hourly_results.png'")

final_val = portfolio[-1]
final_bh = buy_hold[-1]
acc = (correct_preds / total_preds) * 100

print("\n" + "=" * 50)
print(f"🎯 AI Accuracy:   {acc:.2f}%")
print("-" * 30)
print(f"📉 Buy & Hold:    ${final_bh:,.2f}")
print(f"🤖 Hourly AI:     ${final_val:,.2f}")
print("=" * 50)