import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import yfinance as yf
import pandas as pd
from datetime import datetime

# --- CONFIGURATION ---
EPOCHS = 800
LEARNING_RATE = 0.001


# We focus on the "Reflex" (Feed Forward), not LSTM memory, to avoid overthinking
# Target: Predict if next 5 days are POSITIVE.

# 1. SETUP MODEL
class BullSurfer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, 32)
        self.layer2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def __call__(self, x):
        x = self.relu(self.layer1(x))
        return self.sigmoid(self.layer2(x))


# 2. GET DATA
print("📡 Downloading Data...")
current_date = datetime.now().strftime('%Y-%m-%d')
sp500 = yf.download("^GSPC", start="2000-01-01", end=current_date)

data = pd.DataFrame()
data['SP500'] = sp500['Close']
# We drop VIX. It just scares the AI. We focus on Price Action.
data = data.ffill().dropna()

# 3. FEATURE ENGINEERING
data['Returns'] = data['SP500'].pct_change()
# Momentum: Is the market moving up recently?
data['Mom_5d'] = data['SP500'].pct_change(5)
data['Mom_20d'] = data['SP500'].pct_change(20)
# Volatility: Is it shaky?
data['Vol_20'] = data['Returns'].rolling(20).std()
# Trend: Are we above the average?
data['Dist_SMA50'] = (data['SP500'] - data['SP500'].rolling(50).mean()) / data['SP500'].rolling(50).mean()
data = data.dropna()

# 4. CREATE TARGET (The Pivot)
# OLD: Will it crash? -> Result: Paranoia.
# NEW: Will it go UP in the next 5 days? -> Result: Greed.
future_returns = data['SP500'].shift(-5) / data['SP500'] - 1
data['Target'] = (future_returns > 0.0).astype(int)  # 1 if Up, 0 if Down
data = data.dropna()

# 5. PREPARE FOR MLX
feature_cols = ['Mom_5d', 'Mom_20d', 'Vol_20', 'Dist_SMA50']
features = data[feature_cols].values
labels = data['Target'].values

# Scale
mean = np.mean(features, axis=0)
std = np.std(features, axis=0)
features_scaled = (features - mean) / std

# Split 2020
split_date = "2020-01-01"
split_idx = len(data[data.index < split_date])

x_train = mx.array(features_scaled[:split_idx])
y_train = mx.array(labels[:split_idx]).reshape(-1, 1)
x_test = mx.array(features_scaled[split_idx:])
y_test = mx.array(labels[split_idx:]).reshape(-1, 1)

# 6. TRAIN
model = BullSurfer(input_dim=len(feature_cols))
mx.eval(model.parameters())
optimizer = optim.Adam(learning_rate=LEARNING_RATE)


def loss_fn(model, x, y):
    pred = model(x)
    # Binary Cross Entropy (Standard, no weighted penalty)
    return -mx.mean(y * mx.log(pred + 1e-7) + (1 - y) * mx.log(1 - pred + 1e-7))


loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

print(f"🧠 Training BullSurfer ({EPOCHS} Epochs)...")
for i in range(EPOCHS):
    loss, grads = loss_and_grad_fn(model, x_train, y_train)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
    if i % 200 == 0:
        print(f"Epoch {i}: Loss = {loss.item():.4f}")

# 7. BACKTEST (Dynamic Allocation)
print("\n💰 Simulating Dynamic Trading...")
test_data = data.iloc[split_idx:].copy()
probs = np.array(model(x_test)).flatten()

# Smooth the confidence slightly
probs = pd.Series(probs).rolling(3).mean().fillna(0.5).values

portfolio = []
buy_hold = []
cash = 10000.0
initial_buy_hold_shares = 10000.0 / test_data['SP500'].iloc[0]

# Start with 100% Cash
current_cash = 10000.0
current_shares = 0.0

prices = test_data['SP500'].values

for i in range(len(prices)):
    price = float(prices[i])
    confidence = float(probs[i])

    # DYNAMIC ALLOCATION STRATEGY
    # If AI is 80% sure it's going up, we put 80% of our portfolio in stocks.
    # We never go 100% Cash unless Confidence is 0%.

    # 1. Calculate Total Portfolio Value
    total_value = current_cash + (current_shares * price)

    # 2. Determine Target Amount in Stocks
    # We floor it at 0.0 (No Shorts) and cap at 1.0 (No Margin)
    target_stock_value = total_value * confidence

    # 3. Rebalance
    current_stock_value = current_shares * price
    diff = target_stock_value - current_stock_value

    if diff > 0:
        # Buy more
        shares_to_buy = diff / price
        current_shares += shares_to_buy
        current_cash -= diff
    elif diff < 0:
        # Sell some
        shares_to_sell = abs(diff) / price
        current_shares -= shares_to_sell
        current_cash += abs(diff)

    portfolio.append(total_value)
    buy_hold.append(initial_buy_hold_shares * price)

# 8. RESULTS
final_val = portfolio[-1]
final_bh = buy_hold[-1]

print("\n" + "=" * 50)
print(f"📉 Buy & Hold:    ${final_bh:,.2f}")
print(f"🏄 Bull Surfer:    ${final_val:,.2f}")
print("-" * 30)
diff = final_val - final_bh
if diff > 0:
    print(f"🏆 VICTORY: +${diff:,.2f}")
else:
    print(f"💀 DEFEAT: -${abs(diff):,.2f}")
    print("   (But did we lose LESS money than before?)")
print("=" * 50)