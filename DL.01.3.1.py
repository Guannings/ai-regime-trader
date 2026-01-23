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
PERIOD = "59d"
INTERVAL = "1h"
SEQ_LENGTH = 24
EMBED_DIM = 64
NUM_HEADS = 4
EPOCHS = 500
LEARNING_RATE = 0.001


# 1. THE TRANSFORMER
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


# 2. GET DATA
print("📡 Downloading Hourly Data (SPY)...")
try:
    spy = yf.download("SPY", period=PERIOD, interval=INTERVAL, progress=False)
    if len(spy) == 0: raise ValueError("No data found.")

    data = pd.DataFrame()
    if isinstance(spy.columns, pd.MultiIndex):
        data['Close'] = spy['Close']['SPY']
        data['High'] = spy['High']['SPY']
        data['Low'] = spy['Low']['SPY']
    else:
        data['Close'] = spy['Close']
        data['High'] = spy['High']
        data['Low'] = spy['Low']

    data['VIX_Proxy'] = (data['High'] - data['Low']) / data['Close'] * 100.0
    data = data.ffill().dropna()

except Exception as e:
    print(f"❌ DATA FAILURE: {e}")
    exit()

# 3. FEATURE ENGINEERING
data['Returns'] = data['Close'].pct_change()
data['Range'] = (data['High'] - data['Low']) / data['Close']
data['VIX_Change'] = data['VIX_Proxy'].diff()
data['Hour'] = data.index.hour
data['Hour_Norm'] = (data['Hour'] - 9) / 7.0
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
    if i % 100 == 0:
        print(f"Epoch {i}: Loss = {loss.item():.4f}")

# 7. BACKTEST (DYNAMIC SIZING)
print("\n🎨 Calculating Dynamic Positions...")
probs = np.array(model(x_test)).flatten()
test_prices = data['Close'].values[SEQ_LENGTH:][split_idx:]
test_dates = data.index[SEQ_LENGTH:][split_idx:]

portfolio = []
buy_hold = []
cash = 10000.0
# Current Value of Stock Position (Can be negative for shorts)
stock_value = 0.0
total_value = 10000.0

initial_shares = 10000.0 / test_prices[0]

# Metrics for plotting
exposures = []  # Records "How much did we bet?" (-1.0 to 1.0)

for i in range(len(test_prices)):
    price = float(test_prices[i])
    prob = float(probs[i])

    # --- STRATEGY: CONFIDENCE SCALING ---
    # Convert Probability (0.0 to 1.0) into Exposure (-1.0 to 1.0)
    # Prob 1.0 (Super Bull) -> Exposure +1.0 (100% Long)
    # Prob 0.0 (Super Bear) -> Exposure -1.0 (100% Short)
    # Prob 0.5 (Confused)   -> Exposure  0.0 (100% Cash)

    target_exposure = (prob - 0.5) * 2.0

    # Filter: If the AI is "meh" (between -0.2 and 0.2), just stay in Cash to save fees
    if abs(target_exposure) < 0.2:
        target_exposure = 0.0

    exposures.append(target_exposure)

    # REBALANCE LOOP
    # 1. Calculate current Total Equity
    total_value = cash + stock_value

    # 2. Calculate ideal Stock Value
    # (e.g., $10,000 * -0.5 exposure = -$5,000 target in stocks)
    target_stock_value = total_value * target_exposure

    # 3. Calculate difference
    # Note: For simplicity in backtest, we assume we can adjust perfectly instantly
    # In reality, you would buy/sell shares to match this.

    # We update our "virtual" allocation for the next step returns
    stock_value = target_stock_value
    cash = total_value - stock_value

    # 4. Apply Market Move for NEXT step
    # If we are Long (+), price up = profit.
    # If we are Short (-), price up = loss.
    if i < len(test_prices) - 1:
        next_price = float(test_prices[i + 1])
        pct_change = (next_price - price) / price

        # Apply change to the stock portion
        stock_value = stock_value * (1 + pct_change)

    portfolio.append(total_value)
    buy_hold.append(initial_shares * price)

# --- PLOTTING ---
plt.style.use('dark_background')
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

# Panel 1: Portfolio Value
ax1.plot(test_dates, portfolio, label='AI Strategy', color='#00ccff', linewidth=2)
ax1.plot(test_dates, buy_hold, label='Buy & Hold', color='gray', linestyle='--', alpha=0.7)
ax1.set_title("Portfolio Value ($10k Start)")
ax1.legend()
ax1.grid(True, alpha=0.2)

# Panel 2: The "Bet Size" (Exposure) - THIS ANSWERS YOUR QUESTION
ax2.plot(test_dates, exposures, color='white', linewidth=1)
ax2.fill_between(test_dates, exposures, 0, where=(np.array(exposures) > 0), color='green', alpha=0.5, label='Long %')
ax2.fill_between(test_dates, exposures, 0, where=(np.array(exposures) < 0), color='red', alpha=0.5, label='Short %')
ax2.set_title("AI Bet Size (+100% Long to -100% Short)")
ax2.set_ylabel("Exposure")
ax2.set_ylim(-1.1, 1.1)
ax2.legend()
ax2.grid(True, alpha=0.2)

# Panel 3: Price Action
ax3.plot(test_dates, test_prices, label='SPY Price', color='white', alpha=0.5)
ax3.set_title("Market Price")
ax3.grid(True, alpha=0.2)

plt.tight_layout()
print("✨ Graphs generated...")
plt.show()

print("\n" + "=" * 50)
print(f"📉 Buy & Hold:    ${buy_hold[-1]:,.2f}")
print(f"🤖 AI L/S:        ${portfolio[-1]:,.2f}")
print("=" * 50)