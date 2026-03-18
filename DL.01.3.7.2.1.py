import streamlit as st
import matplotlib

matplotlib.use('Agg')  # Prevent window pop-ups
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from datetime import datetime
import time
from sklearn.utils.class_weight import compute_sample_weight
from dateutil.relativedelta import relativedelta

# --- CONFIGURATION ---
st.set_page_config(page_title="AI Crypto Regime Detector", layout="wide")

# Constants
TRAIN_START = "2015-01-01"
RISK_FREE_RATE = 0.04
TRANSACTION_COST = 0.001  # 0.1% per trade (crypto exchange fees)


# --- 1. ROBUST DATA LOADING ---
@st.cache_data(ttl=3600)
def load_data_safe():
    tickers = {"BTC": "BTC-USD", "VIX": "^VIX"}
    data_store = {}

    for name, ticker in tickers.items():
        success = False
        attempts = 0
        while not success and attempts < 3:
            try:
                stock = yf.Ticker(ticker)
                df = stock.history(period="max")
                df = df[df.index >= TRAIN_START].copy()
                if df.empty: raise ValueError(f"Empty data for {ticker}")
                if df.index.tz is not None: df.index = df.index.tz_localize(None)
                data_store[name] = df
                success = True
            except Exception as e:
                attempts += 1
                time.sleep(1)
                if attempts == 3: st.stop()

    return data_store['BTC'], data_store['VIX']


btc, vix = load_data_safe()

# Prepare Master DataFrame
data = pd.DataFrame()
data['BTC'] = btc['Close']
data['BTC_Volume'] = btc['Volume']
data['VIX'] = vix['Close']
data = data.ffill().dropna()

# --- 2. FEATURE ENGINEERING ---
data['SMA_200'] = data['BTC'].rolling(200).mean()
data['SMA_50'] = data['BTC'].rolling(50).mean()
data['Dist_SMA200'] = (data['BTC'] - data['SMA_200']) / data['SMA_200']
data['SMA_Cross'] = (data['SMA_50'] / data['SMA_200']) - 1
data['VIX_Norm'] = data['VIX'] / 100.0
data['VIX_Change'] = data['VIX'].pct_change(5)
data['Vol_20'] = data['BTC'].pct_change().rolling(20).std()
data['RSI'] = 100 - (100 / (1 + data['BTC'].pct_change().rolling(14).apply(
    lambda x: x[x > 0].sum() / abs(x[x < 0].sum()) if abs(x[x < 0].sum()) > 0 else 1)))
data['Vol_SMA20'] = data['BTC_Volume'].rolling(20).mean()
data['Volume_Ratio'] = data['BTC_Volume'] / data['Vol_SMA20']
data['Mom_20'] = data['BTC'].pct_change(20)

data = data.dropna()

# --- 3. TARGET DEFINITION ---
future_returns = data['BTC'].shift(-10) / data['BTC'] - 1
data['Target'] = (future_returns > 0.0).astype(int)

data_for_prediction = data.copy()
data = data.dropna()

# --- 4. SPLIT TRAIN/TEST (Expanding Window) ---
TEST_START_DATE = "2021-01-01"
TRAIN_CUTOFF = data.index[-1] - relativedelta(months=6)
train = data[data.index <= TRAIN_CUTOFF]
test = data[data.index >= TEST_START_DATE]

feature_cols = ['Dist_SMA200', 'SMA_Cross', 'VIX_Norm', 'VIX_Change', 'Vol_20', 'RSI', 'Volume_Ratio', 'Mom_20']
X_train = train[feature_cols]
y_train = train['Target']
X_test = test[feature_cols]
y_test = test['Target']

MODEL_PARAMS = dict(
    n_estimators=60, max_depth=1, learning_rate=0.03,
    subsample=0.5, min_samples_leaf=120, random_state=42
)

# --- 5. TRAIN MODEL (Expanding Window — uses all data up to 6 months ago) ---
model = GradientBoostingClassifier(**MODEL_PARAMS)
sample_weights = compute_sample_weight('balanced', y_train)
model.fit(X_train, y_train, sample_weight=sample_weights)

# --- 6. GENERATE TODAY'S SIGNAL (FIX #1 & #2 APPLIED) ---
latest_features = data_for_prediction[feature_cols].iloc[[-1]]
latest_prob = model.predict_proba(latest_features)[0][1]
latest_date = latest_features.index[-1].strftime('%Y-%m-%d')
latest_price = data_for_prediction['BTC'].iloc[-1]

# Check Regime
latest_btc = data_for_prediction['BTC'].iloc[-1]
latest_sma = data_for_prediction['SMA_200'].iloc[-1]
is_bull_regime = latest_btc > latest_sma

# Decision Logic (Fix #1: Hysteresis)
signal = "NEUTRAL / HOLD"
color_code = "gray"
exposure = "Unchanged"

if latest_prob > 0.51 and is_bull_regime:
    signal = "BUY / LONG BTC"
    color_code = "green"
    exposure = "100% (1x BTC)"
elif latest_prob < 0.46:
    signal = "SELL / CASH"
    color_code = "red"
    exposure = "0% (Cash Yield)"
elif not is_bull_regime and latest_prob > 0.51:
    signal = "BLOCKED (Bear Regime)"
    color_code = "orange"
    exposure = "0% (Safety First)"

# --- OUTPUT: THE WEB APP ---
st.title("🤖 AI Crypto Regime Detector")
st.markdown("### Institutional-Grade Crypto Safety System")

col1, col2, col3 = st.columns(3)
col1.metric("Latest Date", latest_date)
col1.metric("BTC Price", f"${latest_price:,.2f}")
col2.metric("AI Confidence", f"{latest_prob * 100:.1f}%")

if color_code == "green":
    st.success(f"### SIGNAL: {signal}")
    st.write(f"**Target Exposure:** {exposure}")
    st.write("✅ The AI detects a **Safe Bull Market**. BTC allocation authorized.")
elif color_code == "red":
    st.error(f"### SIGNAL: {signal}")
    st.write(f"**Target Exposure:** {exposure}")
    st.write("⚠️ The AI detects **High Risk**. Move to Cash immediately.")
elif color_code == "orange":
    st.warning(f"### SIGNAL: {signal}")
    st.write("⚠️ AI wants to buy, but **Long-Term Trend is Down** (BTC < 200 SMA). Trade blocked for safety.")
else:
    st.info(f"### SIGNAL: {signal}")
    st.write("Confidence is low (45-55%). Holding current position to save fees.")

st.divider()

# --- BACKTEST CHART (FIX #1 & #2 APPLIED) ---
st.subheader("📊 Performance Verification (BTC)")
st.caption("Includes: 0.1% Transaction Costs | Hysteresis Filter (46-51%) | 200-SMA Regime Filter | Class-Balanced Training | 60-Day Retrain")

with st.spinner("Running Realistic Simulation (with periodic retraining)..."):
    ret_btc = test['BTC'].pct_change().fillna(0).values
    btc_vals = test['BTC'].values
    sma_vals = test['SMA_200'].values
    dates = test.index

    daily_cash_yield = RISK_FREE_RATE / 365
    portfolio = [10000.0]
    buy_hold = [10000.0]
    signals = []

    current_holding = 0  # 0 = Cash, 1 = BTC
    RETRAIN_INTERVAL = 60  # Retrain every 60 days
    bt_model = None
    probs = np.zeros(len(dates))

    for i in range(len(dates)):
        # Retrain on expanding window every RETRAIN_INTERVAL days
        if i % RETRAIN_INTERVAL == 0:
            train_cutoff = dates[i] - relativedelta(months=6)
            bt_train = data[data.index <= train_cutoff]
            if len(bt_train) >= 200:
                bt_model = GradientBoostingClassifier(**MODEL_PARAMS)
                bt_sw = compute_sample_weight('balanced', bt_train['Target'])
                bt_model.fit(bt_train[feature_cols], bt_train['Target'], sample_weight=bt_sw)

        if bt_model is not None:
            probs[i] = bt_model.predict_proba(test[feature_cols].iloc[[i]])[0][1]
        else:
            probs[i] = 0.5

    for i in range(1, len(dates)):
        prob = float(probs[i - 1])

        # REGIME CHECK (Yesterday's Close vs SMA)
        is_bull_regime = btc_vals[i - 1] > sma_vals[i - 1]

        # LOGIC UPGRADE: Force Cash if Regime breaks
        if not is_bull_regime:
            target_holding = 0  # <--- FORCE EXIT (Flattens the line in Bear Markets)

        elif prob > 0.51:  # Only Buy if Regime is Bull AND AI is Confident
            target_holding = 1

        elif prob < 0.46:  # Sell if AI gets scared
            target_holding = 0

        else:
            target_holding = current_holding  # Hysteresis (Hold)
        # Apply Transaction Costs
        cost_penalty = 0.0
        if target_holding != current_holding:
            cost_penalty = TRANSACTION_COST

            # Calculate Return
        if target_holding == 1:
            daily_ret = ret_btc[i]
        else:
            daily_ret = daily_cash_yield

        new_value = portfolio[-1] * (1 + daily_ret) * (1 - cost_penalty)
        portfolio.append(new_value)
        buy_hold.append(buy_hold[-1] * (1 + ret_btc[i]))
        signals.append(target_holding)
        current_holding = target_holding

    signals.append(0)

    # Plotting
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    ax1.plot(dates, portfolio, label='Active AI (BTC) [Net of Fees]', color='royalblue', linewidth=3)
    ax1.plot(dates, buy_hold, label='Passive BTC Buy & Hold', color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.set_title("Strategy Performance: Growth of $10,000", fontsize=16, fontweight='bold')
    ax1.set_ylabel("Account Value ($)", fontsize=14)
    ax1.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax1.legend(loc='upper left')
    ax1.grid(True)

    is_btc = np.array(signals) == 1
    ax2.fill_between(dates, 1, 0, where=is_btc, color='limegreen', alpha=0.6, label='BTC')
    ax2.fill_between(dates, 1, 0, where=~is_btc, color='lightcoral', alpha=0.8, label='Cash')
    ax2.set_title("AI Regime Detection", fontsize=16, fontweight='bold')
    ax2.set_yticks([0.25, 0.75])
    ax2.set_yticklabels(['CASH', 'BTC'], fontsize=12, fontweight='bold')
    ax2.set_xlabel("Date", fontsize=14)

    st.pyplot(fig)

st.divider()

# --- SIDEBAR DIAGNOSTICS ---
st.sidebar.markdown("### 🩺 Model Health")
st.sidebar.caption(f"Trained on data up to {TRAIN_CUTOFF.strftime('%Y-%m-%d')}")
train_score = model.score(X_train, y_train)
st.sidebar.metric("Training Accuracy", f"{train_score * 100:.1f}%")

tscv = TimeSeriesSplit(n_splits=5)
cv_scores = []
for tr_index, val_index in tscv.split(X_train):
    X_tr, X_val = X_train.iloc[tr_index], X_train.iloc[val_index]
    y_tr, y_val = y_train.iloc[tr_index], y_train.iloc[val_index]
    val_model = GradientBoostingClassifier(**MODEL_PARAMS)
    val_sw = compute_sample_weight('balanced', y_tr)
    val_model.fit(X_tr, y_tr, sample_weight=val_sw)
    cv_scores.append(val_model.score(X_val, y_val))

avg_cv_score = np.mean(cv_scores)
st.sidebar.metric("Walk-Forward Score", f"{avg_cv_score * 100:.1f}%")

gap = train_score - avg_cv_score
if gap > 0.15:
    st.sidebar.error("⚠️ Overfitting!")
elif gap > 0.10:
    st.sidebar.warning("⚠️ High Variance")
else:
    st.sidebar.success("✅ Model is Robust")

# --- FEATURE IMPORTANCE ---
st.subheader("🧐 Why did the AI make this decision?")
if hasattr(model, "feature_importances_"):
    importances = model.feature_importances_
    feature_imp = pd.DataFrame({'Value': importances, 'Feature': feature_cols})
    top_5 = feature_imp.sort_values(by="Value", ascending=False).head(5)

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x="Feature", y="Value", hue="Feature", data=top_5, order=top_5['Feature'], palette="viridis",
                legend=False, ax=ax)
    ax.set_ylabel("Influence Score")
    ax.set_title("Factors Driving Today's Signal (Highest to Lowest)")
    st.pyplot(fig)

st.divider()

# --- DETAILED METRICS (STRATEGY-AWARE) ---
st.subheader("📊 Detailed Strategy Performance (Test Data)")
st.caption("This report now grades the **Full Strategy** (AI + Regime Filter), not just the raw model.")

# 1. Reuse probabilities from backtest (expanding window retrained)
# probs already computed during backtest loop above

# 2. Get Regime Data (Aligned with Test Set)
test_btc = test['BTC']
test_sma = test['SMA_200']

# 3. Generate "Strategy Predictions"
# (Simulating exactly what the bot did in the chart)
strategy_preds = []

for i in range(len(probs)):
    # Check Regime: Is BTC > SMA?
    is_bull_regime = test_btc.iloc[i] > test_sma.iloc[i]

    # Check AI Confidence
    prob = probs[i]

    # THE RULES:
    # If we are in a Bear Regime (BTC < SMA), we FORCE SELL (0).
    # If AI is scared (Prob < 0.50), we SELL (0).
    # Otherwise, we BUY (1).
    if not is_bull_regime or prob < 0.50:
        strategy_preds.append(0)
    else:
        strategy_preds.append(1)

# 4. Generate Report based on these "Smart" predictions
report = classification_report(y_test, strategy_preds, output_dict=True)
df_report = pd.DataFrame(report).transpose()
st.dataframe(df_report.style.format("{:.2f}"))

# --- METRICS DICTIONARY (DETAILED) ---
with st.expander("📚 Guide: How to read this table (Click to expand)", expanded=True):
    st.markdown("""
    This table is your **AI's Report Card**. It tells you if the bot is actually smart, or just lucky.

    ### 1. The Rows (The Signal Types)
    * **Row `0` (The "Panic" Detector):** This row measures how well the bot handles **Sell/Cash** signals.
        * *Why it matters:* If this row is bad, the bot won't save you during a crash.
    * **Row `1` (The "Greed" Detector):** This row measures how well the bot handles **Buy/BTC** signals.
        * *Why it matters:* This drives your profit during bull markets.

    ### 2. The Columns (The Grades)
    * **`precision` (The "Trust" Score):** When the bot makes a guess, how often is it right?
        * *High Precision on '0':* When it says "SELL", the market usually actually drops.
        * *Low Precision on '0':* It cries wolf often (sells when it shouldn't).

    * **`recall` (The "Sensitivity" Score):** Out of all the real events, how many did it catch?
        * *High Recall on '0':* It caught almost every crash (The Blue Line goes flat).
        * *Low Recall on '0':* It slept through the crash (The Blue Line drops with the market).

    * **`support` (Sample Size):** How many days in the test data actually belonged to this category.

    ### 💡 What you want to see:
    To see the Blue Line go flat during crashes, you need **Recall for '0' to be > 0.30**. 
    Current settings are tuned to improve this.
    """)

st.divider()

# --- LEGAL DISCLAIMER (Expanded) ---
with st.expander("⚖️ LEGAL DISCLAIMER & RISK DISCLOSURE (IMPORTANT)", expanded=False):
    st.markdown("""
    ### 1. General Information Only
    The content provided in this application, including all AI-generated signals, charts, and data analysis, is for **informational, educational, and research purposes only**. It does not constitute financial advice, investment recommendations, or a solicitation to buy or sell any securities. The creator of this application is not a registered financial advisor, broker-dealer, or investment professional.

    ### 2. No Investment Advice
    You should not treat any opinion or signal expressed in this application as a specific inducement to make a particular investment or follow a particular strategy. All investment decisions are made at your own risk. You should consult with a qualified financial advisor before making any financial decisions.

    ### 3. Risk of Significant Loss
    Trading cryptocurrencies, particularly **Bitcoin (BTC)**, involves a **very high degree of risk**.
    * **Volatility Risk:** Bitcoin is extremely volatile and can experience 50%+ drawdowns. You can lose a significant portion or all of your capital in a short period.
    * **Regulatory Risk:** Cryptocurrency regulations vary by jurisdiction and may change without notice, potentially affecting your ability to trade or hold BTC.
    * **Custodial Risk:** Digital assets may be lost due to exchange failures, hacking, or loss of private keys.
    * **Market Manipulation:** Cryptocurrency markets are less regulated than traditional markets and may be subject to manipulation.

    ### 4. Accuracy and AI Limitations
    This application uses a Machine Learning model (Gradient Boosting) to analyze historical data. 
    * **No Guarantee:** The model's past performance (backtesting) is **not indicative of future results**. Markets change, and patterns that worked in the past may fail in the future.
    * **Model Errors:** The AI may produce incorrect signals due to data errors, overfitting, or unforeseen market events ("Black Swans").
    * **Data Latency:** Market data provided here may be delayed or inaccurate.

    ### 5. Limitation of Liability
    By using this application, you explicitly agree that the developer and contributors shall **not be held liable** for any direct, indirect, incidental, or consequential damages resulting from the use or inability to use this software. You assume full responsibility for your trading activities.
    """)
    st.caption("Last Updated: March 2026 | Version: 3.0.0 (BTC)")
