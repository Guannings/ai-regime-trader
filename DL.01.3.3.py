import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# --- CONFIGURATION (SNIPER MODE) ---
PERIOD = "59d"
INTERVAL = "1h"
SEQ_LENGTH = 24
EMBED_DIM = 64
NUM_HEADS = 4
EPOCHS = 300
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.2
# PRECISION SETTINGS
CONFIDENCE_THRESHOLD = 0.70  # Only trade if > 70% sure (Sniper)
LEVERAGE = 1.0  # Set to 1.5 or 2.0 for higher returns (Riskier)


# 1. THE TRANSFORMER
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.att = nn.MultiHeadAttention(dims=embed_dim, num_heads=num_heads)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

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
    # Handle MultiIndex
    if isinstance(spy.columns, pd.MultiIndex):
        data['Open'] = spy['Open']['SPY']
        data['Close'] = spy['Close']['SPY']
        data['High'] = spy['High']['SPY']
        data['Low'] = spy['Low']['SPY']
    else:
        data['Open'] = spy['Open']
        data['Close'] = spy['Close']
        data['High'] = spy['High']
        data['Low'] = spy['Low']

    data = data.ffill().dropna()

except Exception as e:
    print(f"❌ DATA FAILURE: {e}")
    exit()

# 3. FEATURE ENGINEERING (CANDLESTICK PHYSICS)
data['Returns'] = data['Close'].pct_change()

# CANDLESTICK MATH
# Body: The real move (Close - Open)
data['Body'] = (data['Close'] - data['Open']) / data['Open']
# Upper Wick: Rejection from highs (High - Max(Open, Close))
data['Upper_Wick'] = (data['High'] - data[['Open', 'Close']].max(axis=1)) / data['Open']
# Lower Wick: Rejection from lows (Min(Open, Close) - Low)
data['Lower_Wick'] = (data[['Open', 'Close']].min(axis=1) - data['Low']) / data['Open']

# Volatility & RSI
data['Range'] = (data['High'] - data['Low']) / data['Close']
data['RSI'] = 100 - (100 / (1 + data['Returns'].rolling(14).apply(
    lambda x: x[x > 0].sum() / abs(x[x < 0].sum()) if abs(x[x < 0].sum()) > 0 else 1)))

data = data.dropna()

# 4. TARGET
data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
data = data.dropna()

# 5. PREPARE SEQUENCES
# We feed the AI the shape of the candles now
feature_cols = ['Returns', 'Body', 'Upper_Wick', 'Lower_Wick', 'Range', 'RSI']
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

print(f"🧠 Training Precision Model ({EPOCHS} Epochs)...")
for i in range(EPOCHS):
    loss, grads = loss_and_grad_fn(model, x_train, y_train)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
    if i % 50 == 0:
        print(f"Epoch {i}: Loss = {loss.item():.4f}")

# 7. BACKTEST (RESTORED VISUALS)
print("\n🎨 Calculating Snipe Points...")
probs = np.array(model(x_test)).flatten()
test_prices = data['Close'].values[SEQ_LENGTH:][split_idx:]
test_dates = data.index[SEQ_LENGTH:][split_idx:]

portfolio = []
buy_hold = []
cash = 10000.0
stock_value = 0.0
exposures = []

# Visual Markers
buy_dates, buy_prices = [], []
short_dates, short_prices = [], []
exit_dates, exit_prices = [], []

# Smoothing Logic (Reduced for Snappiness)
ALPHA = 0.5
current_exposure = 0.0

for i in range(len(test_prices)):
    price = float(test_prices[i])
    prob = float(probs[i])

    # SNIPER LOGIC
    # Only move if confidence is HIGH
    if prob > CONFIDENCE_THRESHOLD:
        target = 1.0 * LEVERAGE  # Long
    elif prob < (1.0 - CONFIDENCE_THRESHOLD):
        target = -1.0 * LEVERAGE  # Short
    else:
        target = 0.0  # Cash (Unsure)

    # Smooth the transition
    current_exposure = (1 - ALPHA) * target + (ALPHA) * current_exposure

    # Record Visual Markers (Only when crossing thresholds)
    if len(exposures) > 0:
        prev = exposures[-1]
        # Green Triangle: Crossing into Long
        if prev <= 0.2 and current_exposure > 0.5:
            buy_dates.append(test_dates[i])
            buy_prices.append(price)
        # Red Triangle: Crossing into Short
        elif prev >= -0.2 and current_exposure < -0.5:
            short_dates.append(test_dates[i])
            short_prices.append(price)
        # Gray X: Exiting to Cash
        elif abs(prev) > 0.5 and abs(current_exposure) < 0.2:
            exit_dates.append(test_dates[i])
            exit_prices.append(price)

    exposures.append(current_exposure)

    # Rebalance
    total_value = cash + stock_value
    target_stock_value = total_value * current_exposure
    stock_value = target_stock_value
    cash = total_value - stock_value

    # Apply Return
    if i < len(test_prices) - 1:
        next_price = float(test_prices[i + 1])
        pct_change = (next_price - price) / price
        stock_value = stock_value * (1 + pct_change)

    portfolio.append(total_value)
    buy_hold.append(10000.0 / test_prices[0] * price)

# --- PLOTTING ---
plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Panel 1: Price + Markers (THEY ARE BACK!)
ax1.plot(test_dates, test_prices, label='Price', color='white', alpha=0.5)
ax1.scatter(buy_dates, buy_prices, marker='^', color='#00ff00', s=120, label='Long Entry', zorder=5)
ax1.scatter(short_dates, short_prices, marker='v', color='#ff0000', s=120, label='Short Entry', zorder=5)
ax1.scatter(exit_dates, exit_prices, marker='x', color='yellow', s=60, label='Exit', zorder=4)
ax1.set_title("Precision Entries (Sniper Mode)")
ax1.legend()

# Panel 2: Portfolio vs Buy & Hold
ax2.plot(test_dates, portfolio, label='AI Sniper', color='#00ccff', linewidth=2)
ax2.plot(test_dates, buy_hold, label='Buy & Hold', color='gray', linestyle='--', alpha=0.7)
ax2.set_title(f"Portfolio Value (Threshold={CONFIDENCE_THRESHOLD * 100}%)")
ax2.legend()
ax2.grid(True, alpha=0.2)

plt.tight_layout()
plt.show()

print("\n" + "=" * 50)
print(f"📉 Buy & Hold:    ${buy_hold[-1]:,.2f}")
print(f"🤖 AI Sniper:     ${portfolio[-1]:,.2f}")
print("=" * 50)