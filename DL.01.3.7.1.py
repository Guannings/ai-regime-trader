import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
from datetime import datetime

# --- CONFIGURATION ---
TRAIN_START = "2000-01-01"
# We want to train on as much history as possible to be smart
TEST_START = "2020-01-01"
RISK_FREE_RATE = 0.04

# 1. GET DATA
print("📡 Downloading Latest Data...")
current_date = datetime.now().strftime('%Y-%m-%d')

# Brain (SPY) & Wallet (SSO)
spy = yf.download("SPY", start=TRAIN_START, end=current_date, progress=False)
vix = yf.download("^VIX", start=TRAIN_START, end=current_date, progress=False)
sso = yf.download("SSO", start=TRAIN_START, end=current_date, progress=False)

data = pd.DataFrame()
def get_col(df, name):
    if isinstance(df.columns, pd.MultiIndex):
        return df['Close'][name]
    else:
        return df['Close']

data['SPY'] = get_col(spy, 'SPY')
data['VIX'] = get_col(vix, '^VIX')
data['SSO'] = get_col(sso, 'SSO')

data = data.ffill().dropna()

# 2. FEATURE ENGINEERING
data['SMA_200'] = data['SPY'].rolling(200).mean()
data['Dist_SMA200'] = (data['SPY'] - data['SMA_200']) / data['SMA_200']
data['VIX_Norm'] = data['VIX'] / 100.0
data['Vol_20'] = data['SPY'].pct_change().rolling(20).std()
data['RSI'] = 100 - (100 / (1 + data['SPY'].pct_change().rolling(14).apply(lambda x: x[x>0].sum() / abs(x[x<0].sum()) if abs(x[x<0].sum()) > 0 else 1)))

data = data.dropna()

# 3. TARGET
future_returns = data['SPY'].shift(-10) / data['SPY'] - 1
data['Target'] = (future_returns > 0.0).astype(int)
# Note: We drop NaN targets for training, but we keep the latest row for prediction!
data_for_prediction = data.copy()
data = data.dropna()

# 4. SPLIT
train = data[data.index < TEST_START]
test = data[data.index >= TEST_START]

feature_cols = ['Dist_SMA200', 'VIX_Norm', 'Vol_20', 'RSI']
X_train = train[feature_cols]
y_train = train['Target']
X_test = test[feature_cols]

# 5. TRAIN
print("🌲 Training AI Model...")
model = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)

# 6. GET TODAY'S SIGNAL
print("🔮 Generating Forecast...")
# We use the VERY LAST row of data (Today/Yesterday's Close)
latest_features = data_for_prediction[feature_cols].iloc[[-1]]
latest_prob = model.predict_proba(latest_features)[0][1] # Probability of Class 1 (Safe)
latest_date = latest_features.index[-1].strftime('%Y-%m-%d')
latest_price = data_for_prediction['SSO'].iloc[-1]

# DECISION LOGIC
signal = "NEUTRAL / HOLD"
color = "⚪"
exposure = "0x or 2x (Unchanged)"

if latest_prob > 0.52:
    signal = "BUY / LONG SSO"
    color = "🟢"
    exposure = "200% (2x Leverage)"
elif latest_prob < 0.48:
    signal = "SELL / CASH"
    color = "🔴"
    exposure = "0% (Cash Yield)"

# 7. DASHBOARD OUTPUT
print("\n" + "="*40)
print(f"      🤖 AI TRADING DASHBOARD")
print("="*40)
print(f"📅 Date:          {latest_date}")
print(f"💲 SSO Price:     ${latest_price:.2f}")
print("-" * 40)
print(f"🧠 AI Confidence: {latest_prob*100:.2f}%")
print(f"🚦 SIGNAL:        {color} {signal}")
print(f"⚖️ Target Exp:    {exposure}")
print("="*40)
print("\nInstructions:")
print("1. If Signal is GREEN: Buy SSO (or Hold if you have it).")
print("2. If Signal is RED: Sell SSO (or Stay in Cash).")
print("3. If Signal is NEUTRAL: Do nothing (Wait).")