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
RISK_FREE_RATE = 0.04  # You earn 4% per year when sitting in Cash (Money Market)

# 1. GET DATA (The Brain & The Wallet)
print("📡 Downloading Real Tickers (SPY + SSO + VIX)...")
current_date = datetime.now().strftime('%Y-%m-%d')

# Brain Data
spy = yf.download("SPY", start=TRAIN_START, end=current_date, progress=False)
vix = yf.download("^VIX", start=TRAIN_START, end=current_date, progress=False)

# Wallet Data (The Actual 2x ETF)
# SSO started in 2006, so we'll have to handle the alignment
sso = yf.download("SSO", start=TRAIN_START, end=current_date, progress=False)

data = pd.DataFrame()


# Handle MultiIndex
def get_col(df, name):
    if isinstance(df.columns, pd.MultiIndex):
        return df['Close'][name]
    else:
        return df['Close']


data['SPY'] = get_col(spy, 'SPY')
data['VIX'] = get_col(vix, '^VIX')
data['SSO'] = get_col(sso, 'SSO')

data = data.ffill().dropna()

# 2. FEATURE ENGINEERING (Using SPY for Signals)
# We calculate indicators on SPY because it's the "Truth"
data['SMA_200'] = data['SPY'].rolling(200).mean()
data['Dist_SMA200'] = (data['SPY'] - data['SMA_200']) / data['SMA_200']
data['VIX_Norm'] = data['VIX'] / 100.0
data['Vol_20'] = data['SPY'].pct_change().rolling(20).std()
data['RSI'] = 100 - (100 / (1 + data['SPY'].pct_change().rolling(14).apply(
    lambda x: x[x > 0].sum() / abs(x[x < 0].sum()) if abs(x[x < 0].sum()) > 0 else 1)))

data = data.dropna()

# 3. TARGET
# Predict Safety based on SPY behavior
future_returns = data['SPY'].shift(-10) / data['SPY'] - 1
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

# 6. REAL-WORLD BACKTEST
print(f"💰 Trading Real Tickers (SSO vs SPY)...")
probs = model.predict_proba(X_test)[:, 1]

# Returns of the assets
ret_spy = test['SPY'].pct_change().fillna(0).values
ret_sso = test['SSO'].pct_change().fillna(0).values
dates = test.index

# Daily Cash Yield (Risk Free Rate / Trading Days)
daily_cash_yield = RISK_FREE_RATE / 252

portfolio = [10000.0]
buy_hold = [10000.0]
signals = []

for i in range(1, len(dates)):
    prob = float(probs[i - 1])

    # STRATEGY:
    # Safe (> 52%) -> Buy SSO (Real 2x ETF)
    # Danger (< 48%) -> Cash (Earn Interest)

    if prob > 0.52:
        # Invested in SSO
        # SSO already includes the fee decay in its price, so we just take the return
        strat_ret = ret_sso[i]
        signal = 2  # Visualization code for "Leveraged"
    elif prob < 0.48:
        # Invested in Cash
        strat_ret = daily_cash_yield
        signal = 0  # Visualization code for "Cash"
    else:
        # Buffer Zone (Hold previous)
        # We simplify: if we were Long, stay Long. If Cash, stay Cash.
        prev_sig = signals[-1] if len(signals) > 0 else 0
        if prev_sig == 2:
            strat_ret = ret_sso[i]
            signal = 2
        else:
            strat_ret = daily_cash_yield
            signal = 0

    new_val = portfolio[-1] * (1 + strat_ret)
    portfolio.append(new_val)

    # Benchmark: Buy & Hold SPY (Standard Market)
    bh_val = buy_hold[-1] * (1 + ret_spy[i])
    buy_hold.append(bh_val)

    signals.append(signal)

signals.append(0)  # Match length

# 7. HIGH-VISUALIZATION
plt.style.use('seaborn-v0_8-darkgrid')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Panel 1: Value
ax1.plot(dates, portfolio, label='Active SSO Strategy (AI)', color='royalblue', linewidth=2.5)
ax1.plot(dates, buy_hold, label='Passive SPY Hold', color='black', linestyle='--', linewidth=1.5, alpha=0.7)
ax1.set_title(f"Real-World Deployment: SSO (2x) vs SPY", fontsize=14, fontweight='bold')
ax1.set_ylabel("Account Value ($)", fontsize=12)
ax1.legend(loc='upper left', frameon=True, facecolor='white', framealpha=1)
ax1.grid(True, alpha=0.5)

# Panel 2: What are we holding?
# We construct a fill for "Held SSO"
is_sso = np.array(signals) == 2
ax2.fill_between(dates, 1, 0, where=is_sso, color='limegreen', alpha=0.5, label='Holding SSO (2x)')
ax2.fill_between(dates, 1, 0, where=~is_sso, color='lightgray', alpha=0.5, label='Holding Cash (Yield)')
ax2.set_title("Asset Allocation", fontsize=14, fontweight='bold')
ax2.set_yticks([])
ax2.legend(loc='upper left', frameon=True, facecolor='white', framealpha=1)

plt.tight_layout()
plt.show()

# RESULTS
final_val = portfolio[-1]
final_bh = buy_hold[-1]
print("\n" + "=" * 50)
print(f"📉 Benchmark (SPY):    ${final_bh:,.2f}")
print(f"🚀 Deployment (SSO):   ${final_val:,.2f}")
print("-" * 30)
diff = final_val - final_bh
if diff > 0:
    print(f"🏆 ALPHA GENERATED: +${diff:,.2f}")
    print("✅ VERDICT: Model is Robust. Ready for Paper Trading.")
else:
    print(f"💀 DEFEAT: -${abs(diff):,.2f}")
    print("❌ VERDICT: Decay ate the profits. Do not deploy.")
print("=" * 50)
# ... (after the plotting code)

print("\n🧠 AI BRAIN SCAN: What did it actually learn?")
# Get feature importance
importances = model.feature_importances_
feature_names = X_test.columns

# Create a dataframe for visualization
fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
fi_df = fi_df.sort_values(by='Importance', ascending=False)

print(fi_df)

# Visualize it
plt.figure(figsize=(10, 6))
plt.barh(fi_df['Feature'], fi_df['Importance'], color='teal')
plt.xlabel("Importance (0.0 to 1.0)")
plt.title("What Matters to the AI?")
plt.gca().invert_yaxis() # Put the most important at the top
plt.show()