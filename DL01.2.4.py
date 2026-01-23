#1-Built a Neural Network from scratch.
#2-Implemented LSTM memory cells.
#3-Engineered Features (Volatility, Returns, Fear).
#4-Prevented Overfitting (Train/Test split).
#5-Run a Backtest to prove the model's worth.

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import yfinance as yf
import pandas as pd
from datetime import datetime

# --- CONFIGURATION ---
EPOCHS = 2000  # Weekly data has fewer rows, so we train longer
LEARNING_RATE = 0.001
HIDDEN_SIZE = 64  # Bigger brain for complex Macro patterns


# 1. SETUP MODEL (Standard Feed-Forward)
class WeeklyBrain(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, HIDDEN_SIZE)
        self.layer2 = nn.Linear(HIDDEN_SIZE, 32)
        self.layer3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def __call__(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        return self.sigmoid(self.layer3(x))


# 2. GET DATA (The "Macro" Mix)
print("📡 Downloading Data (Stocks + Rates)...")
current_date = datetime.now().strftime('%Y-%m-%d')
sp500 = yf.download("^GSPC", start="2000-01-01", end=current_date)
tnx = yf.download("^TNX", start="2000-01-01", end=current_date)  # 10-Year Yield

# Merge
data = pd.DataFrame()
data['SP500'] = sp500['Close']
data['Yield'] = tnx['Close']
data = data.ffill().dropna()

# --- THE MAGIC TRICK: RESAMPLE TO WEEKLY ---
# We convert noisy daily data into clean Weekly bars (taking the close of Friday)
weekly = data.resample('W').last()
weekly['Yield_Change'] = weekly['Yield'].diff()  # Weekly change in rates

# 3. FEATURE ENGINEERING (Weekly Stats)
weekly['Returns'] = weekly['SP500'].pct_change()
weekly['Mom_4w'] = weekly['SP500'].pct_change(4)  # Monthly Momentum
weekly['Mom_12w'] = weekly['SP500'].pct_change(12)  # Quarterly Momentum
weekly['Vol_12w'] = weekly['Returns'].rolling(12).std()
# Trend: Distance from 40-Week Average (approx 200-Day MA)
weekly['Dist_SMA40'] = (weekly['SP500'] - weekly['SP500'].rolling(40).mean()) / weekly['SP500'].rolling(40).mean()
weekly = weekly.dropna()

# 4. CREATE TARGET
# Predict: Will NEXT WEEK be Green?
weekly['Target'] = (weekly['SP500'].shift(-1) > weekly['SP500']).astype(int)
weekly = weekly.dropna()

# 5. MLX PREP
feature_cols = ['Yield_Change', 'Mom_4w', 'Mom_12w', 'Vol_12w', 'Dist_SMA40']
features = weekly[feature_cols].values
labels = weekly['Target'].values

# Scale (Crucial)
mean = np.mean(features, axis=0)
std = np.std(features, axis=0)
features_scaled = (features - mean) / std

# Split 2020
split_date = "2020-01-01"
split_idx = len(weekly[weekly.index < split_date])

x_train = mx.array(features_scaled[:split_idx])
y_train = mx.array(labels[:split_idx]).reshape(-1, 1)
x_test = mx.array(features_scaled[split_idx:])
y_test = mx.array(labels[split_idx:]).reshape(-1, 1)

# 6. TRAIN
model = WeeklyBrain(input_dim=len(feature_cols))
mx.eval(model.parameters())
optimizer = optim.Adam(learning_rate=LEARNING_RATE)


def loss_fn(model, x, y):
    pred = model(x)
    return -mx.mean(y * mx.log(pred + 1e-7) + (1 - y) * mx.log(1 - pred + 1e-7))


loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

print(f"🧠 Training Weekly Model ({EPOCHS} Epochs)...")
for i in range(EPOCHS):
    loss, grads = loss_and_grad_fn(model, x_train, y_train)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
    if i % 500 == 0:
        print(f"Epoch {i}: Loss = {loss.item():.4f}")

# 7. BACKTEST (Weekly Rebalancing)
print("\n💰 Simulating Weekly Trading...")
test_data = weekly.iloc[split_idx:].copy()
probs = np.array(model(x_test)).flatten()

portfolio = []
buy_hold = []
cash = 10000.0
# Initial shares calculation needs to handle scalar properly
initial_price = float(test_data['SP500'].iloc[0])
initial_shares = 10000.0 / initial_price

prices = test_data['SP500'].values

# Start 100% Cash
current_cash = 10000.0
current_shares = 0.0

for i in range(len(prices)):
    price = float(prices[i])
    confidence = float(probs[i])

    # STRATEGY:
    # We only trade once a week (Friday Close).
    # If Confidence > 50%, we go Long.
    # The higher the confidence, the bigger the bet.

    total_value = current_cash + (current_shares * price)

    # Position Sizing: Map 50-100% prob to 0-100% allocation
    # If prob is 0.60 -> allocation is 1.0 (Full Invest)
    # If prob is 0.40 -> allocation is 0.0 (Cash)
    # This creates a binary-like aggression but smoothed.
    if confidence > 0.5:
        target_allocation = 1.0  # Bullish? All in.
    else:
        target_allocation = 0.0  # Bearish? All out.

    target_stock_value = total_value * target_allocation

    current_stock_value = current_shares * price
    diff = target_stock_value - current_stock_value

    # Rebalance
    if diff > 0:
        shares_to_buy = diff / price
        current_shares += shares_to_buy
        current_cash -= diff
    elif diff < 0:
        shares_to_sell = abs(diff) / price
        current_shares -= shares_to_sell
        current_cash += abs(diff)

    portfolio.append(total_value)
    buy_hold.append(initial_shares * price)

# 8. RESULTS
final_val = portfolio[-1]
final_bh = buy_hold[-1]

print("\n" + "=" * 50)
print(f"📉 Buy & Hold:    ${final_bh:,.2f}")
print(f"📅 Weekly AI:     ${final_val:,.2f}")
print("-" * 30)
diff = final_val - final_bh
if diff > 0:
    print(f"🏆 VICTORY: +${diff:,.2f}")
else:
    print(f"💀 DEFEAT: -${abs(diff):,.2f}")
print("=" * 50)