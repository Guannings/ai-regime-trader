import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import time

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="AI Regime Trader", layout="wide")
st.title("🤖 AI Regime Trader: Market Dashboard")
st.markdown("---")

# --- CONSTANTS ---
TICKER = "SSO"
RF_ESTIMATORS = 200
RF_MAX_DEPTH = 5  # Matched to Email Bot
RF_MIN_SAMPLES_LEAF = 5  # Matched to Email Bot
TEST_SIZE = 0.2


# --- 2. HELPER FUNCTIONS ---
@st.cache_data(ttl=3600)  # Cache data for 1 hour to speed up app
def get_data(ticker):
    """Downloads data with error handling"""
    try:
        data = yf.download(ticker, period="2y")
        if data.empty:
            st.error("Downloaded data is empty.")
            return None
        return data
    except Exception as e:
        st.error(f"Failed to download data: {e}")
        return None


def train_model(data):
    # Feature Engineering (Exact copy of Email Bot)
    data['Returns'] = data['Close'].pct_change()
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['RSI'] = 100 - (100 / (1 + data['Close'].pct_change().apply(lambda x: max(x, 0)).rolling(14).mean() /
                                data['Close'].pct_change().apply(lambda x: abs(min(x, 0))).rolling(14).mean()))
    data['Volatility'] = data['Returns'].rolling(window=20).std()
    data['Momentum'] = data['Close'] / data['Close'].shift(10) - 1

    data.dropna(inplace=True)

    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)

    predictors = ['Returns', 'SMA_10', 'SMA_50', 'RSI', 'Volatility', 'Momentum']
    X = data[predictors]
    y = data['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, shuffle=False)

    # Model (Matched to Email Bot)
    model = RandomForestClassifier(
        n_estimators=RF_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        random_state=42
    )
    model.fit(X_train, y_train)

    return model, X, y, X_test, y_test, predictors


# --- 3. MAIN APP LOGIC ---
data = get_data(TICKER)

if data is not None:
    # Train the brain
    model, X, y, X_test, y_test, predictors = train_model(data)

    # Get Today's Prediction
    latest_data = X.iloc[[-1]]
    prediction = model.predict(latest_data)
    probability = model.predict_proba(latest_data)

    signal = "BUY" if prediction[0] == 1 else "SELL"
    confidence = round(probability[0][prediction[0]] * 100, 2)

    # --- DISPLAY METRICS ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Ticker", TICKER)
    col1.metric("Latest Price", f"${round(data['Close'].iloc[-1], 2)}")

    # Color-coded signal
    signal_color = "normal"
    if signal == "BUY":
        signal_color = "inverse"  # Makes it stand out
        st.success(f"### 🟢 Signal: BUY ({confidence}%)")
    else:
        st.error(f"### 🔴 Signal: SELL ({confidence}%)")

    # --- FEATURE IMPORTANCE CHART (The "Why") ---
    st.subheader("🧐 Why did the AI make this decision?")
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feature_imp = pd.DataFrame({'Value': importances, 'Feature': predictors})
        top_5 = feature_imp.sort_values(by="Value", ascending=False).head(5)

        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(x="Value", y="Feature", data=top_5, palette="viridis", ax=ax)
        ax.set_title("Top 5 Factors Driving Today's Signal")
        st.pyplot(fig)

    # --- ADVANCED METRICS (Tier 1 Feedback) ---
    st.subheader("📊 Model Reliability (Test Data)")
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    st.markdown("*Note: 'Precision' for class 1 (Buy) is your most important metric.*")

else:
    st.warning("Waiting for data...")
