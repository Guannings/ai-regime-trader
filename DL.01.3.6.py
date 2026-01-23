#The "High-Vis" 2x Leveraged Script

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# --- CONFIGURATION ---
TRAIN_START = "2000-01-01"
TEST_START = "2020-01-01"
LEVERAGE = 2.0

# 1. GET DATA
print("📡 Downloading Data...")
current_date = datetime.now().strftime('%Y-%m-%d')
spy = yf.download("SPY", start=TRAIN_START, end=current_date, progress=False)
vix = yf.download("^VIX", start=TRAIN_START, end=current_date, progress=False)

data = pd.DataFrame()
if isinstance(spy.columns, pd.MultiIndex):
    data['Close'] = spy['Close']['SPY']
else:
    data['Close'] = spy['Close']

if isinstance(vix.columns, pd.MultiIndex):
    data['VIX'] = vix['Close']['^VIX']
else:
    data['VIX'] = vix['Close']

data = data.ffill().dropna()

# 2. FEATURE ENGINEERING
data['SMA_200'] = data['Close'].rolling(200).mean()
data['Dist_SMA200'] = (data['Close'] - data['SMA_200']) / data['SMA_200']
data['VIX_Norm'] = data['VIX'] / 100.0
data['Vol_20'] = data['Close'].pct_change().rolling(20).std()
data['RSI'] = 100 - (100 / (1 + data['Close'].pct_change().rolling(14).apply(
    lambda x: x[x > 0].sum() / abs(x[x < 0].sum()) if abs(x[x < 0].sum()) > 0 else 1)))

data = data.dropna()

# 3. TARGET
future_returns = data['Close'].shift(-10) / data['Close'] - 1
data['Target'] = (future_returns > 0.0).astype(int)
data = data.dropna()

# 4. SPLIT
train = data[data.index < TEST_START]
test = data[data.index >= TEST_START]

feature_cols = ['Dist_SMA200', 'VIX_Norm', 'Vol_20', 'RSI']
X_train = train[feature_cols]
y_train = train['Target']
X_test = test[feature_cols]
y_test = test['Target']

# 5. TRAIN
print("🌲 Training Regime Detector...")
model = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)

# 6. BACKTEST
print(f"💰 Running {LEVERAGE}x Smart Leverage Strategy...")
probs = model.predict_proba(X_test)[:, 1]
prices = test['Close'].values
dates = test.index
daily_returns = test['Close'].pct_change().fillna(0).values

portfolio = [10000.0]
buy_hold = [10000.0]
regime_signals = []

for i in range(1, len(prices)):
    prob = float(probs[i - 1])
    ret = float(daily_returns[i])

    leverage_mult = 0.0
    if prob > 0.60:
        leverage_mult = LEVERAGE
    elif prob > 0.50:
        leverage_mult = 1.0
    else:
        leverage_mult = 0.0

        # Cost of borrowing (approx 5% annual)
    borrow_cost_daily = (0.05 / 252) * (leverage_mult - 1) if leverage_mult > 1 else 0
    strat_ret = (ret * leverage_mult) - borrow_cost_daily

    new_val = portfolio[-1] * (1 + strat_ret)
    portfolio.append(new_val)

    bh_val = buy_hold[-1] * (1 + ret)
    buy_hold.append(bh_val)

    regime_signals.append(leverage_mult)

regime_signals.append(0)

# 7. HIGH-VISIBILITY VISUALIZATION
# Use a clean, light style for better contrast
plt.style.use('seaborn-v0_8-darkgrid')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Panel 1: Portfolio (Thick Blue vs Dotted Black)
ax1.plot(dates, portfolio, label=f'{LEVERAGE}x Smart AI', color='royalblue', linewidth=2.5)
ax1.plot(dates, buy_hold, label='Buy & Hold (1x)', color='black', linestyle='--', linewidth=1.5, alpha=0.7)
ax1.set_title(f"Portfolio Value (Smart Leverage)", fontsize=14, fontweight='bold')
ax1.set_ylabel("Account Value ($)", fontsize=12)
ax1.legend(loc='upper left', frameon=True, facecolor='white', framealpha=1)
ax1.grid(True, alpha=0.5)

# Panel 2: Leverage (Solid Green Fill)
ax2.plot(dates, regime_signals, color='darkgreen', linewidth=1.5, label='Leverage Used')
ax2.fill_between(dates, regime_signals, 0, color='limegreen', alpha=0.4)
# Add a red line at 0 so you see when we are in Cash
ax2.axhline(0, color='red', linewidth=2, linestyle='-', label='Cash (Safety)')
ax2.set_title("Leverage Level (2.0 = Aggressive, 0.0 = Cash)", fontsize=14, fontweight='bold')
ax2.set_ylabel("Leverage Multiplier", fontsize=12)
ax2.legend(loc='upper left', frameon=True, facecolor='white', framealpha=1)
ax2.grid(True, alpha=0.5)

plt.tight_layout()
plt.show()

# RESULTS
final_val = portfolio[-1]
final_bh = buy_hold[-1]
print("\n" + "=" * 50)
print(f"📉 Buy & Hold (1x):   ${final_bh:,.2f}")
print(f"🚀 {LEVERAGE}x Smart AI:    ${final_val:,.2f}")
print("-" * 30)
diff = final_val - final_bh
if diff > 0:
    print(f"🏆 VICTORY: +${diff:,.2f}")
else:
    print(f"💀 DEFEAT: -${abs(diff):,.2f}")
print("=" * 50)