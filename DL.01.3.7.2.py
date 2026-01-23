import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(page_title="AI Regime Detector", layout="wide")
TRAIN_START = "2000-01-01"
TEST_START = "2020-01-01"
RISK_FREE_RATE = 0.04


# --- HELPER FUNCTION FOR LOGGING ---
# This function prints to BOTH the console and the web app logs
def log(message):
    print(message)  # To Console


# 1. GET DATA
log("📡 Downloading Latest Data...")
current_date = datetime.now().strftime('%Y-%m-%d')


# Brain (SPY) & Wallet (SSO)
# We use caching so the web app doesn't re-download data every time you click a button
@st.cache_data
def load_data():
    spy = yf.download("SPY", start=TRAIN_START, end=current_date, progress=False)
    vix = yf.download("^VIX", start=TRAIN_START, end=current_date, progress=False)
    sso = yf.download("SSO", start=TRAIN_START, end=current_date, progress=False)
    return spy, vix, sso


spy, vix, sso = load_data()

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
data['RSI'] = 100 - (100 / (1 + data['SPY'].pct_change().rolling(14).apply(
    lambda x: x[x > 0].sum() / abs(x[x < 0].sum()) if abs(x[x < 0].sum()) > 0 else 1)))

data = data.dropna()

# 3. TARGET
future_returns = data['SPY'].shift(-10) / data['SPY'] - 1
data['Target'] = (future_returns > 0.0).astype(int)
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
log("🌲 Training AI Model...")
model = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)

# 6. GET TODAY'S SIGNAL
log("🔮 Generating Forecast...")
latest_features = data_for_prediction[feature_cols].iloc[[-1]]
latest_prob = model.predict_proba(latest_features)[0][1]
latest_date = latest_features.index[-1].strftime('%Y-%m-%d')
latest_price = data_for_prediction['SSO'].iloc[-1]

# DECISION LOGIC
signal = "NEUTRAL / HOLD"
color_code = "gray"  # For web
console_icon = "⚪"  # For console
exposure = "0x or 2x (Unchanged)"

if latest_prob > 0.52:
    signal = "BUY / LONG SSO"
    color_code = "green"
    console_icon = "🟢"
    exposure = "200% (2x Leverage)"
elif latest_prob < 0.48:
    signal = "SELL / CASH"
    color_code = "red"
    console_icon = "🔴"
    exposure = "0% (Cash Yield)"

# --- OUTPUT 1: THE CONSOLE (Black Box) ---
print("\n" + "=" * 40)
print(f"      🤖 AI TRADING DASHBOARD")
print("=" * 40)
print(f"📅 Date:          {latest_date}")
print(f"💲 SSO Price:     ${latest_price:.2f}")
print("-" * 40)
print(f"🧠 AI Confidence: {latest_prob * 100:.2f}%")
print(f"🚦 SIGNAL:        {console_icon} {signal}")
print(f"⚖️ Target Exp:    {exposure}")
print("=" * 40)

# --- OUTPUT 2: THE WEB APP (Browser) ---
st.title("🤖 AI Regime Detector")
st.markdown("### Institutional-Grade Market Safety System")

# Top Metrics Row
col1, col2, col3 = st.columns(3)
col1.metric("Latest Date", latest_date)
col1.metric("SSO Price", f"${latest_price:.2f}")
col2.metric("AI Confidence", f"{latest_prob * 100:.1f}%")

# The Big Signal Box
if color_code == "green":
    st.success(f"### SIGNAL: {signal}")
    st.write(f"**Target Exposure:** {exposure}")
    st.write("✅ The AI detects a **Safe Bull Market**. Leverage is authorized.")
elif color_code == "red":
    st.error(f"### SIGNAL: {signal}")
    st.write(f"**Target Exposure:** {exposure}")
    st.write("⚠️ The AI detects **High Risk**. Move to Cash immediately.")
else:
    st.info(f"### SIGNAL: {signal}")
    st.write(f"**Target Exposure:** {exposure}")
    st.write("The market is indecisive. Hold your current position.")

st.divider()

# Backtest Chart for the Web
st.subheader("📊 Performance Verification (SSO vs SPY)")
with st.spinner("Running Backtest Simulation..."):
    # (Simplified Backtest Logic for Charting)
    probs = model.predict_proba(X_test)[:, 1]
    ret_sso = test['SSO'].pct_change().fillna(0).values
    ret_spy = test['SPY'].pct_change().fillna(0).values
    dates = test.index

    daily_cash_yield = RISK_FREE_RATE / 252
    portfolio = [10000.0]
    buy_hold = [10000.0]
    signals = []

    for i in range(1, len(dates)):
        prob = float(probs[i - 1])
        if prob > 0.52:
            strat_ret = ret_sso[i]
            sig = 2
        elif prob < 0.48:
            strat_ret = daily_cash_yield
            sig = 0
        else:
            prev_sig = signals[-1] if len(signals) > 0 else 0
            if prev_sig == 2:
                strat_ret = ret_sso[i]
                sig = 2
            else:
                strat_ret = daily_cash_yield
                sig = 0
        portfolio.append(portfolio[-1] * (1 + strat_ret))
        buy_hold.append(buy_hold[-1] * (1 + ret_spy[i]))
        signals.append(sig)
    signals.append(0)

    # Plot using Matplotlib but render in Streamlit
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1.plot(dates, portfolio, label='Active AI (SSO)', color='royalblue', linewidth=2)
    ax1.plot(dates, buy_hold, label='Passive SPY', color='black', linestyle='--', alpha=0.6)
    ax1.set_title("Strategy Performance")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    is_sso = np.array(signals) == 2
    ax2.fill_between(dates, 1, 0, where=is_sso, color='green', alpha=0.3, label='Held SSO')
    ax2.fill_between(dates, 1, 0, where=~is_sso, color='gray', alpha=0.3, label='Held Cash')
    ax2.set_title("Asset Allocation")
    ax2.legend()

    st.pyplot(fig)

# ... (Previous code for charts and signals) ...

st.divider()

# LEGAL DISCLAIMER SECTION
with st.expander("⚖️ LEGAL DISCLAIMER & RISK DISCLOSURE (READ CAREFULLY)"):
    st.markdown("""
    **1. GENERAL DISCLAIMER**
    The content provided herein is for **informational and educational purposes only**. The Developer is not a registered financial advisor. Nothing in this application constitutes a recommendation to buy or sell any security.

    **2. RISK OF LOSS**
    Trading financial instruments, especially leveraged ETFs like SSO, involves **significant risk**. You could lose all of your invested capital. Past performance of this AI model is **not indicative of future results**.

    **3. SOFTWARE WARRANTY**
    This software is provided "AS IS", without warranty of any kind. The AI model may make incorrect predictions due to data errors, market anomalies, or model overfitting. Use at your own risk.

    **4. FULL TERMS**
    By using this application, you acknowledge that you are solely responsible for your investment decisions and agree to hold the Developer harmless from any liability.
    """)

    st.caption("Last Updated: January 2026")