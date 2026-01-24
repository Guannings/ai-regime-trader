import matplotlib

matplotlib.use('Agg')  # Prevent window pop-ups
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import time
import os

# --- CONFIGURATION ---
TRAIN_START = "2000-01-01"


# --- 1. ROBUST DATA LOADING ---
def load_data_safe():
    tickers = {"SPY": "SPY", "VIX": "^VIX", "SSO": "SSO"}
    data_store = {}

    print("📡 Downloading Data...")
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
                print(f"⚠️ Retry {attempts + 1} for {ticker}: {e}")
                attempts += 1
                time.sleep(1)
                if attempts == 3:
                    print(f"❌ Failed to download {ticker}")
                    return None, None, None

    return data_store['SPY'], data_store['VIX'], data_store['SSO']


def save_importance_plot(model, feature_names):
    """Generates the 'Why' chart and saves it as an image."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feature_imp = pd.DataFrame({'Value': importances, 'Feature': feature_names})
        top_5 = feature_imp.sort_values(by="Value", ascending=False).head(5)

        plt.figure(figsize=(10, 4))
        # Fixed the 'palette' warning by adding hue and legend=False
        sns.barplot(x="Feature", y="Value", hue="Feature", data=top_5,
                    order=top_5['Feature'], palette="viridis", legend=False)
        plt.title("Factors Driving Today's Signal")
        plt.xlabel("Influence Score")
        plt.tight_layout()

        filename = "reasoning_chart.png"
        plt.savefig(filename)
        plt.close()
        return filename
    return None


if __name__ == "__main__":
    # 1. Load Data
    spy, vix, sso = load_data_safe()
    if spy is None: exit()

    # 2. Prepare Master DataFrame
    data = pd.DataFrame()
    data['SPY'] = spy['Close']
    data['VIX'] = vix['Close']
    data['SSO'] = sso['Close']
    data = data.ffill().dropna()

    # 3. Feature Engineering
    data['SMA_200'] = data['SPY'].rolling(200).mean()
    data['Dist_SMA200'] = (data['SPY'] - data['SMA_200']) / data['SMA_200']
    data['VIX_Norm'] = data['VIX'] / 100.0
    data['Vol_20'] = data['SPY'].pct_change().rolling(20).std()
    data['RSI'] = 100 - (100 / (1 + data['SPY'].pct_change().rolling(14).apply(
        lambda x: x[x > 0].sum() / abs(x[x < 0].sum()) if abs(x[x < 0].sum()) > 0 else 1)))

    data = data.dropna()

    # 4. Target Definition
    future_returns = data['SPY'].shift(-10) / data['SPY'] - 1
    data['Target'] = (future_returns > 0.0).astype(int)

    data_for_prediction = data.copy()
    data = data.dropna()

    # 5. Split Train/Test
    TEST_START_DATE = "2020-01-01"
    train = data[data.index < TEST_START_DATE]

    feature_cols = ['Dist_SMA200', 'VIX_Norm', 'Vol_20', 'RSI']
    X_train = train[feature_cols]
    y_train = train['Target']

    # 6. Train Model (MATCHING YOUR APP SETTINGS)
    print("🌲 Training Model...")
    model = GradientBoostingClassifier(
        n_estimators=90,
        max_depth=3,
        learning_rate=0.025,
        subsample=0.7,
        min_samples_leaf=60,
        random_state=42
    )
    model.fit(X_train, y_train)

    # 7. Generate Signal
    latest_features = data_for_prediction[feature_cols].iloc[[-1]]
    latest_prob = model.predict_proba(latest_features)[0][1]
    latest_price = data_for_prediction['SSO'].iloc[-1]

    # Regime Check
    latest_spy = data_for_prediction['SPY'].iloc[-1]
    latest_sma = data_for_prediction['SMA_200'].iloc[-1]
    is_bull_regime = latest_spy > latest_sma

    # Decision Logic (Matching App)
    signal = "NEUTRAL / HOLD"

    if latest_prob > 0.55 and is_bull_regime:
        signal = "BUY / LONG SSO"
    elif latest_prob < 0.45:
        signal = "SELL / CASH"
    elif not is_bull_regime and latest_prob > 0.55:
        signal = "BLOCKED (Bear Regime)"
    else:
        signal = "NEUTRAL (Hysteresis)"

    print(f"🔮 Prediction: {signal} ({latest_prob * 100:.1f}%)")

    # 8. Generate Chart
    plot_filename = save_importance_plot(model, feature_cols)

    # 9. Send Email
    try:
        from send_email import send_daily_alert

        # IMPORTANT: Replace with your actual Streamlit URL
        APP_LINK = "https://ai-regime-trader-dl-01-3-7-2-1.streamlit.app"

        send_daily_alert(
            signal=signal,
            confidence=round(latest_prob * 100, 1),
            price=round(latest_price, 2),
            image_path=plot_filename,
            app_link=APP_LINK
        )
        print("✅ Email Sent successfully.")
    except ImportError:
        print("⚠️ 'send_email.py' not found. Skipping email.")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")