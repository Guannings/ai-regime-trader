import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib.pyplot as plt
import seaborn as sns
import smtplib
import ssl
from email.message import EmailMessage
import os
import time

# --- CONSTANTS ---
TICKER = "SSO"  # The Asset to Trade
BENCHMARK = "SPY"  # The Regime Filter
TRAIN_START = "2000-01-01"
# Streamlit Link (Paste your actual link here)
STREAMLIT_URL = "https://ai-regime-trader-dl-01-3-7-2-1.streamlit.app"


# --- 1. ROBUST DATA LOADING ---
def get_data_safe():
    tickers = {"SPY": "SPY", "VIX": "^VIX", "SSO": "SSO"}
    data_store = {}

    for name, ticker in tickers.items():
        success = False
        attempts = 0
        while not success and attempts < 3:
            try:
                print(f"Downloading {ticker}...")
                stock = yf.Ticker(ticker)
                df = stock.history(period="max")
                df = df[df.index >= TRAIN_START].copy()

                if df.empty: raise ValueError(f"Empty data for {ticker}")
                if df.index.tz is not None: df.index = df.index.tz_localize(None)

                data_store[name] = df
                success = True
            except Exception as e:
                print(f"Error: {e}. Retrying...")
                attempts += 1
                time.sleep(2)

    # Ensure we got everything
    if len(data_store) < 3:
        raise RuntimeError("Failed to download all required tickers.")

    return data_store['SPY'], data_store['VIX'], data_store['SSO']


def save_importance_plot(model, feature_names):
    """Generates the 'Why' chart for the email"""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feature_imp = pd.DataFrame({'Value': importances, 'Feature': feature_names})
        top_5 = feature_imp.sort_values(by="Value", ascending=False).head(5)

        plt.figure(figsize=(10, 5))
        # Fix for Seaborn warning: set hue=Feature and legend=False
        sns.barplot(x="Feature", y="Value", hue="Feature", data=top_5,
                    order=top_5['Feature'], palette="viridis", legend=False)
        plt.title("Top Factors Driving Today's Signal")
        plt.xlabel("Influence Score")
        plt.tight_layout()

        filename = "email_chart.png"
        plt.savefig(filename)
        plt.close()
        return filename
    return None


def send_daily_alert(signal, confidence, price, regime_msg, plot_filename=None):
    """Sends the formatted email"""
    EMAIL_SENDER = os.environ.get('EMAIL_USER')
    EMAIL_PASSWORD = os.environ.get('EMAIL_PASS')
    # Add your personal email to this list
    SUBSCRIBERS = ["cheeperholy@gmail.com"]

    if not EMAIL_SENDER or not EMAIL_PASSWORD:
        print("Error: Email credentials missing in environment variables.")
        return

    context = ssl.create_default_context()

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
            smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)

            for email in SUBSCRIBERS:
                msg = EmailMessage()
                msg['From'] = "AI Regime Bot"
                msg['To'] = email
                msg['Subject'] = f"🔔 {signal} SSO ({confidence}%) - ${price}"

                # HTML Body
                body = f"""
                <html>
                  <body>
                    <h2>Daily AI Market Scan</h2>
                    <p><strong>Signal:</strong> {signal}</p>
                    <p><strong>Confidence:</strong> {confidence}%</p>
                    <p><strong>Price:</strong> ${price}</p>
                    <p><strong>Regime Status:</strong> {regime_msg}</p>
                    <hr>
                    <p>📈 <a href="{STREAMLIT_URL}">Check Full Dashboard & Charts</a></p>
                  </body>
                </html>
                """
                msg.add_alternative(body, subtype='html')

                # Attach Chart
                if plot_filename and os.path.exists(plot_filename):
                    with open(plot_filename, 'rb') as f:
                        img_data = f.read()
                        msg.add_attachment(img_data, maintype='image', subtype='png', filename='reasoning.png')

                smtp.send_message(msg)
                print(f"✅ Email sent to {email}")

    except Exception as e:
        print(f"❌ Failed to send email: {e}")


if __name__ == "__main__":
    try:
        # 1. Load Data
        spy, vix, sso = get_data_safe()

        # 2. Prep Master Data
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

        data.dropna(inplace=True)

        # 4. Define Target & Split
        future_returns = data['SPY'].shift(-10) / data['SPY'] - 1
        data['Target'] = (future_returns > 0.0).astype(int)

        data_for_prediction = data.copy()
        data.dropna(inplace=True)  # Drop NaN targets for training

        TEST_START_DATE = "2020-01-01"
        train = data[data.index < TEST_START_DATE]

        feature_cols = ['Dist_SMA200', 'VIX_Norm', 'Vol_20', 'RSI']
        X_train = train[feature_cols]
        y_train = train['Target']

        # 5. Train Model (SNIPER SETTINGS - Matches your best result)
        # Using Sample Weights + Aggressive Depth for better Recall
        sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

        model = GradientBoostingClassifier(
            n_estimators=90,
            max_depth=3,  # Aggressive Depth
            learning_rate=0.025,  # Aggressive Learning
            subsample=0.7,
            min_samples_leaf=60,  # Sensitivity
            random_state=42
        )
        model.fit(X_train, y_train, sample_weight=sample_weights)

        # 6. Generate Signal
        latest_features = data_for_prediction[feature_cols].iloc[[-1]]
        latest_prob = model.predict_proba(latest_features)[0][1]
        latest_price = round(data_for_prediction['SSO'].iloc[-1], 2)

        # Regime Check
        latest_spy = data_for_prediction['SPY'].iloc[-1]
        latest_sma = data_for_prediction['SMA_200'].iloc[-1]
        is_bull_regime = latest_spy > latest_sma

        # Decision Logic (Hysteresis + Regime)
        confidence = round(latest_prob * 100, 1)
        regime_msg = "Bullish (SPY > 200 SMA)" if is_bull_regime else "BEARISH (SPY < 200 SMA)"

        signal = "HOLD"

        if latest_prob > 0.55 and is_bull_regime:
            signal = "BUY"
        elif latest_prob < 0.45:
            signal = "SELL"
        elif not is_bull_regime:
            signal = "SELL (Regime Block)"  # Force exit if bear market
        else:
            signal = "HOLD (Uncertain)"

        # 7. Generate Chart & Send
        print(f"Signal: {signal} | Conf: {confidence}% | Regime: {regime_msg}")
        plot_file = save_importance_plot(model, feature_cols)
        send_daily_alert(signal, confidence, latest_price, regime_msg, plot_file)

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
