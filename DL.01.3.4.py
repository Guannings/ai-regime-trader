import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# --- CONFIGURATION ---
SEQ_LENGTH = 30  # Look back 30 Days (1 Month)
EMBED_DIM = 64
NUM_HEADS = 4
EPOCHS = 10000  # We have 25 years of data, so we train longer
LEARNING_RATE = 0.0005  # Slower, more careful learning
CONFIDENCE_THRESHOLD = 0.60  # Only trade if 60% sure


# 1. THE TRANSFORMER (MLX)
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


class DailyGPT(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, seq_len, dropout=0.2):
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


# 2. GET DAILY DATA (2000 - Present)
print("📡 Downloading 25 Years of Daily Data...")
current_date = datetime.now().strftime('%Y-%m-%d')
sp500 = yf.download("^GSPC", start="2000-01-01", end=current_date)
vix = yf.download("^VIX", start="2000-01-01", end=current_date)

data = pd.DataFrame()
data['Close'] = sp500['Close']
data['VIX'] = vix['Close']
data = data.ffill().dropna()

# 3. FEATURE ENGINEERING
# Returns & Volatility
data['Returns'] = data['Close'].pct_change()
data['Vol_20'] = data['Returns'].rolling(20).std()
# The VIX (Fear Gauge) - Real data this time!
data['VIX_Norm'] = data['VIX'] / 100.0
# The Boss: 200-Day Moving Average
data['SMA_200'] = data['Close'].rolling(200).mean()
data['Dist_SMA'] = (data['Close'] - data['SMA_200']) / data['SMA_200']

data = data.dropna()

# 4. TARGET
# Predict: Will price be higher tomorrow?
data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
data = data.dropna()

# 5. PREPARE SEQUENCES
feature_cols = ['Returns', 'Vol_20', 'VIX_Norm', 'Dist_SMA']
features = data[feature_cols].values
labels = data['Target'].values

# Scale
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

# TIME TRAVEL SPLIT
# Train on 2000-2020 (Historical Crashes)
# Test on 2020-Present (Modern Era)
# Find the index for Jan 1, 2020
dates = data.index[SEQ_LENGTH:]
split_date = pd.Timestamp("2020-01-01")
split_idx = len(dates[dates < split_date])

x_train = mx.array(x_seq[:split_idx])
y_train = mx.array(y_seq[:split_idx]).reshape(-1, 1)
x_test = mx.array(x_seq[split_idx:])
y_test = mx.array(y_seq[split_idx:]).reshape(-1, 1)

print(f"📉 Training Days: {len(x_train)}")
print(f"🧪 Testing Days: {len(x_test)}")

# 6. TRAIN
model = DailyGPT(input_dim=len(feature_cols), embed_dim=EMBED_DIM, num_heads=NUM_HEADS, seq_len=SEQ_LENGTH)
mx.eval(model.parameters())
optimizer = optim.Adam(learning_rate=LEARNING_RATE)


def loss_fn(model, x, y):
    pred = model(x)
    return -mx.mean(y * mx.log(pred + 1e-7) + (1 - y) * mx.log(1 - pred + 1e-7))


loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

print(f"🧠 Training Daily Model ({EPOCHS} Epochs)...")
for i in range(EPOCHS):
    loss, grads = loss_and_grad_fn(model, x_train, y_train)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
    if i % 100 == 0:
        print(f"Epoch {i}: Loss = {loss.item():.4f}")

# 7. BACKTEST (HYBRID STRATEGY)
print("\n🎨 Running Hybrid Daily Strategy...")
probs = np.array(model(x_test)).flatten()
test_prices = data['Close'].values[SEQ_LENGTH:][split_idx:]
test_sma = data['SMA_200'].values[SEQ_LENGTH:][split_idx:]
test_dates = data.index[SEQ_LENGTH:][split_idx:]

portfolio = []
buy_hold = []
cash = 10000.0
shares = 0
position = 'flat'
entry_price = 0.0
initial_shares = 10000.0 / test_prices[0]

buy_dates, buy_prices = [], []
short_dates, short_prices = [], []
exit_dates, exit_prices = [], []

for i in range(len(test_prices)):
    price = float(test_prices[i])
    sma = float(test_sma[i])
    prob = float(probs[i])

    # --- THE HYBRID LOGIC ---
    # The Boss: 200-Day Moving Average
    is_bull_regime = (price > sma)
    is_bear_regime = (price < sma)

    signal = 'hold'

    if is_bull_regime:
        # Bull Market: Buy dips, Ignore short signals
        if prob > CONFIDENCE_THRESHOLD:
            signal = 'buy'
        elif prob < 0.40:
            signal = 'exit'  # Cash is safe, Shorting is suicide

    elif is_bear_regime:
        # Bear Market: Short rallies, Ignore buy signals
        if prob < (1.0 - CONFIDENCE_THRESHOLD):
            signal = 'short'
        elif prob > 0.60:
            signal = 'exit'

    # EXECUTION
    if signal == 'buy':
        if position == 'short':  # Cover
            profit = (entry_price - price) * shares
            cash += profit + (shares * entry_price)
            shares = 0
            position = 'flat'
            exit_dates.append(test_dates[i])
            exit_prices.append(price)
        if position == 'flat':  # Long
            shares = cash / price
            cash = 0
            position = 'long'
            buy_dates.append(test_dates[i])
            buy_prices.append(price)

    elif signal == 'short':
        if position == 'long':  # Sell
            cash = shares * price
            shares = 0
            position = 'flat'
            exit_dates.append(test_dates[i])
            exit_prices.append(price)
        if position == 'flat':  # Short
            shares = cash / price
            entry_price = price
            cash -= (shares * price)
            position = 'short'
            short_dates.append(test_dates[i])
            short_prices.append(price)

    elif signal == 'exit':
        if position == 'long':
            cash = shares * price
            shares = 0
            position = 'flat'
            exit_dates.append(test_dates[i])
            exit_prices.append(price)
        elif position == 'short':
            profit = (entry_price - price) * shares
            cash += profit + (shares * entry_price)
            shares = 0
            position = 'flat'
            exit_dates.append(test_dates[i])
            exit_prices.append(price)

    # VALUATION
    if position == 'long':
        val = shares * price
    elif position == 'short':
        current_profit = (entry_price - price) * shares
        val = (shares * entry_price) + current_profit
    else:
        val = cash

    portfolio.append(val)
    buy_hold.append(initial_shares * price)

# PLOTTING
plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

ax1.plot(test_dates, test_prices, label='Price', color='white', alpha=0.5)
ax1.plot(test_dates, test_sma, label='200-Day SMA', color='yellow', linestyle='--', alpha=0.8)
ax1.scatter(buy_dates, buy_prices, marker='^', color='#00ff00', s=100, label='Long', zorder=5)
ax1.scatter(short_dates, short_prices, marker='v', color='#ff0000', s=100, label='Short', zorder=5)
ax1.set_title("Daily Strategy: 200-SMA + AI Timing")
ax1.legend()

ax2.plot(test_dates, portfolio, label='Hybrid AI', color='#00ccff', linewidth=2)
ax2.plot(test_dates, buy_hold, label='Buy & Hold', color='gray', linestyle='--', alpha=0.7)
ax2.set_title("Portfolio Value")
ax2.legend()
ax2.grid(True, alpha=0.2)

plt.tight_layout()
plt.show()

print("\n" + "=" * 50)
print(f"📉 Buy & Hold:    ${buy_hold[-1]:,.2f}")
print(f"🤖 Hybrid AI:     ${portfolio[-1]:,.2f}")
print("=" * 50)