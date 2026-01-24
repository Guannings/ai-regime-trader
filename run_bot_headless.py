import yfinance as yf
import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_score
from sklearn.model_selection import train_test_split
from send_email import send_daily_alert

# --- CONFIGURATION (Magic Numbers moved here) ---
TICKER = "SSO"  # The ETF we are trading
vix_ticker = "^VIX"  # Market Volatility Index
START_DATE = "2020-01-01"
TEST_SIZE = 0.2  # 20% of data used for testing
RF_TREES = 200  # Number of trees in the forest
RF_DEPTH = 5  # Limit depth to prevent overfitting (Crucial Fix)
MIN_SAMPLES = 10  # Minimum samples per leaf (Crucial Fix)


# --- HELPER: ROBUST DATA DOWNLOAD ---
def get_data_with_retry(ticker, retries=3):
    for i in range(retries):
        try:
            df = yf.download(ticker, start=START_DATE, progress=False)
            if not df.empty:
                return df
        except Exception as e:
            print(f"⚠️ Download failed for {ticker}: {e}. Retrying in 10s...")
            time.sleep(10)
    raise ValueError(f"❌ Failed to download {ticker} after {retries} attempts.")


# --- HELPER: SAVE REASONING CHART ---
def save_importance_plot(model, feature_names):
    """Generates a bar chart of the top 5 factors driving the model."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_imp = pd.DataFrame({'Value': importances, 'Feature': feature_names})

        # Sort and take Top 5
        top_5 = feature_imp.sort_values(by="Value", ascending=False).head(5)

        # Plot
        plt.figure(figsize=(8, 4))
        sns.barplot(x="Value", y="Feature", data=top_5, palette="viridis")
        plt.title("Top 5 Factors Driving Today's Signal")
        plt.xlabel("Importance Score")
        plt.tight_layout()

        filename = "reasoning.png"
        plt.savefig(filename)
        plt.close()
        return filename
    return None


if __name__ == "__main__":
    print(f"🚀 Starting AI Bot for {TICKER}...")

    # 1. LOAD DATA
    try:
        df = get_data_with_retry(TICKER)
        print("✅ Data loaded successfully.")
    except Exception as e:
        print(e)
        exit(1)

    # 2. FEATURE ENGINEERING (The "Inputs")
    # Simple Moving Averages
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()

    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Volatility (Standard Deviation)
    df['Volatility'] = df['Close'].rolling(window=10).std()

    # Create Target (1 if price goes UP tomorrow, 0 if down)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    # Drop NaN values created by indicators
    df = df.dropna()

    # Define Predictors (X) and Target (y)
    predictors = ['SMA_10', 'SMA_50', 'RSI', 'Volatility']
    X = df[predictors]
    y = df['Target']

    # 3. TRAIN/TEST SPLIT
    # Use the past to predict the "future" (test set)
    split_idx = int(len(df) * (1 - TEST_SIZE))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # 4. TRAIN MODEL (With Overfitting Protection)
    print("🧠 Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=RF_TREES,
        max_depth=RF_DEPTH,  # Fix: Prevents overfitting
        min_samples_leaf=MIN_SAMPLES,  # Fix: Forces generalization
        random_state=42
    )
    model.fit(X_train, y_train)

    # 5. EVALUATE PERFORMANCE (Better Metrics)
    print("\n📊 Model Evaluation (Test Set):")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # 6. GENERATE TODAY'S SIGNAL
    latest_data = X.iloc[-1].to_frame().T
    today_prob = model.predict_proba(latest_data)[0][1]  # Probability of "1" (UP)
    today_pred = model.predict(latest_data)[0]

    latest_price = df['Close'].iloc[-1]

    if today_pred == 1:
        signal = "BUY"
        confidence = round(today_prob * 100, 1)
    else:
        signal = "SELL"
        confidence = round((1 - today_prob) * 100, 1)

    print(f"\n🔮 Prediction for Tomorrow: {signal} ({confidence}% Confidence)")

    # 7. EXPLAIN THE DECISION (Feature Importance)
    print("🎨 Generating Reasoning Chart...")
    chart_file = save_importance_plot(model, predictors)

    # 8. SEND ALERT
    print("📧 Sending Email...")
    send_daily_alert(signal, confidence, round(latest_price, 2), chart_file)
    print("✅ Done!")
