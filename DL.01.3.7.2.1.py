import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="AI Regime Trader", layout="wide")

# --- 2. HEADER & DATE ---
st.title("🤖 AI Regime Trader: Market Dashboard")
current_date = datetime.datetime.now().strftime("%Y-%m-%d")
st.markdown(f"**Analysis Date:** {current_date}")
st.markdown("---")

# --- 3. SIDEBAR: DISCLAIMER & INFO ---
with st.sidebar:
    st.header("⚠️ Disclaimer")
    st.caption(
        """
        **IMPORTANT:** This project is for **educational and research purposes only**. 

        Nothing on this page constitutes financial advice, investment recommendations, or a solicitation to buy/sell any assets. 

        Trading stocks and ETFs involves a high degree of risk. The creator of this bot accepts no responsibility for any financial losses incurred by using this model.

        *Always do your own due diligence.*
        """
    )
    st.markdown("---")
    st.info("Created by: Guannings")

# --- 4. CONSTANTS ---
TICKER = "SSO"
RF_ESTIMATORS = 200
RF_MAX_DEPTH = 5
RF_MIN_SAMPLES_LEAF = 5
TEST_SIZE = 0.2


# --- 5. HELPER FUNCTIONS ---
@st.cache_data(ttl=3600)
def get_data(ticker):
    """Downloads data using the Ticker object (More stable for Streamlit Cloud)"""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="2y")

        if data.empty:
            st.error(f"⚠️ Yahoo Finance returned empty data for {ticker}. (Try refreshing the page in 1 minute).")
            return None

        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)

        return data
    except Exception as e:
        st.error(f"Failed to download data: {e}")
        return None


def train_model(data):
    # Feature Engineering
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

    model = RandomForestClassifier(
        n_estimators=RF_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        random_state=42
    )
    model.fit(X_train, y_train)

    return model, X, y, X_test, y_test, predictors


# --- 6. USER GUIDE (EXPLAINER) ---
with st.expander("📖 How to Read This Dashboard (Click to Expand)", expanded=True):
    st.markdown("""
    ### 1. The Signal (Buy vs. Sell)
    * **BUY (Green):** The AI predicts the market (SSO) will close **higher** tomorrow.
    * **SELL (Red):** The AI predicts the market will close **lower** tomorrow.
    * **Confidence:** The percentage indicates how "sure" the AI is. (e.g., 55% is weak, 75% is strong).

    ### 2. The "Why" Chart (Feature Importance)
    * This bar chart reveals the **logic** behind the decision.
    * *Example:* If **RSI** is the tallest bar, the AI is looking heavily at whether the stock is "Overbought" or "Oversold" to make its choice today.

    ### 3. Model Reliability Report
    * Look at the **Precision** for Class 1. This answers: *"When the bot says Buy, how often is it actually right?"*
    """)

# --- 7. MAIN APP LOGIC ---
data = get_data(TICKER)

if data is not None:
    # Train Model
    model, X, y, X_test, y_test, predictors = train_model(data)

    # Get Prediction
    latest_data = X.iloc[[-1]]
    prediction = model.predict(latest_data)
    probability = model.predict_proba(latest_data)

    signal = "BUY" if prediction[0] == 1 else "SELL"
    confidence = round(probability[0][prediction[0]] * 100, 2)

    # --- DISPLAY METRICS ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Ticker", TICKER)
    col1.metric("Latest Price", f"${round(data['Close'].iloc[-1], 2)}")

    if signal == "BUY":
        st.success(f"### 🟢 Signal: BUY ({confidence}%)")
    else:
        st.error(f"### 🔴 Signal: SELL ({confidence}%)")

    # --- FEATURE IMPORTANCE CHART ---
    st.subheader("🧐 Why did the AI make this decision?")
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feature_imp = pd.DataFrame({'Value': importances, 'Feature': predictors})
        top_5 = feature_imp.sort_values(by="Value", ascending=False).head(5)

        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(x="Value", y="Feature", data=top_5, palette="viridis", ax=ax)
        ax.set_title("Top 5 Factors Driving Today's Signal")
        st.pyplot(fig)

    # --- MODEL PERFORMANCE ---
    st.subheader("📊 Model Reliability (Test Data)")
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().style.format("{:.2f}"))

else:
    st.warning("Waiting for data...")