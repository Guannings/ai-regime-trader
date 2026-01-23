import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
from datetime import datetime

# --- CONFIGURATION ---
# We use 20 years of data to learn what "Safety" looks like
TRAIN_START = "2000-01-01"
TEST_START = "2020-01-01"

# 1. GET DATA
print("📡 Downloading Data...")
current_date = datetime.now().strftime('%Y-%m-%d')
spy = yf.download("SPY", start=TRAIN_START, end=current_date, progress=False)
vix = yf.download("^VIX", start=TRAIN_START, end=current_date, progress=False)

data = pd.DataFrame()
# Fix for MultiIndex columns in newer yfinance versions
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
# The Boss: 200-Day Moving Average (Trend)
data['SMA_200'] = data['Close'].rolling(200).mean()
data['Above_Trend'] = (data['Close'] > data['SMA_200']).astype(int)
data['Dist_SMA200'] = (data['Close'] - data['SMA_200']) / data['SMA_200']

# Fear: VIX and Volatility
data['VIX_Norm'] = data['VIX'] / 100.0
data['Vol_20'] = data['Close'].pct_change().rolling(20).std()

# Momentum: RSI
data['RSI'] = 100 - (100 / (1 + data['Close'].pct_change().rolling(14).apply(
    lambda x: x[x > 0].sum() / abs(x[x < 0].sum()) if abs(x[x < 0].sum()) > 0 else 1)))

data = data.dropna()

# 3. DEFINE TARGET
# We are NOT predicting "Up or Down tomorrow."
# We are predicting: "Is the market SAFE for the next 10 days?"
# Target = 1 if the next 10 days have a positive return.
future_returns = data['Close'].shift(-10) / data['Close'] - 1
data['Target'] = (future_returns > 0.0).astype(int)
data = data.dropna()

# 4. SPLIT DATA
train = data[data.index < TEST_START]
test = data[data.index >= TEST_START]

feature_cols = ['Above_Trend', 'Dist_SMA200', 'VIX_Norm', 'Vol_20', 'RSI']
X_train = train[feature_cols]
y_train = train['Target']
X_test = test[feature_cols]
y_test = test['Target']

# 5. TRAIN (Using Standard Scikit-Learn)
print("🌲 Training Regime Detector...")
model = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)

# 6. BACKTEST
print("💰 Running Simulation...")
probs = model.predict_proba(X_test)[:, 1]  # Probability of "Safe Regime"
prices = test['Close'].values
dates = test.index

portfolio = []
buy_hold = []
cash = 10000.0
shares = 0
in_market = False
initial_shares = 10000.0 / prices[0]

regime_signals = []

for i in range(len(prices)):
    price = float(prices[i])
    prob = float(probs[i])

    # STRATEGY:
    # If Probability > 52% -> BULL REGIME (Invest)
    # If Probability < 48% -> BEAR REGIME (Cash)
    # The "Buffer Zone" (48-52) prevents over-trading.

    if prob > 0.52:
        if not in_market:
            shares = cash / price
            cash = 0
            in_market = True
    elif prob < 0.48:
        if in_market:
            cash = shares * price
            shares = 0
            in_market = False

    regime_signals.append(1 if in_market else 0)

    val = (shares * price) if in_market else cash
    portfolio.append(val)
    buy_hold.append(initial_shares * price)

# 7. VISUALIZATION
plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Panel 1: Portfolio
ax1.plot(dates, portfolio, label='Regime Strategy', color='#00ccff', linewidth=2)
ax1.plot(dates, buy_hold, label='Buy & Hold', color='gray', linestyle='--', alpha=0.7)
ax1.set_title("Portfolio Value (Safety First)")
ax1.legend()
ax1.grid(True, alpha=0.2)

# Panel 2: Regime (Green = Invested, Black = Cash)
ax2.plot(dates, prices, color='white', alpha=0.5, label='SPY Price')
ax2.fill_between(dates, prices.min(), prices.max(), where=(np.array(regime_signals) == 1), color='green', alpha=0.2,
                 label='Bull Regime (Invested)')
ax2.set_title("Market Regime (Green = Safe, Black = Cash)")
ax2.legend()

plt.tight_layout()
plt.show()

# RESULTS
final_val = portfolio[-1]
final_bh = buy_hold[-1]
print("\n" + "=" * 50)
print(f"📉 Buy & Hold:    ${final_bh:,.2f}")
print(f"🌲 AI Strategy:   ${final_val:,.2f}")
print("-" * 30)
diff = final_val - final_bh
if diff > 0:
    print(f"🏆 VICTORY: +${diff:,.2f}")
else:
    print(f"💀 DEFEAT: -${abs(diff):,.2f}")
print("=" * 50)
