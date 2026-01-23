import smtplib
import ssl
from email.message import EmailMessage
import os
import time


def send_daily_alert(signal, confidence, price):
    # 1. YOUR SUBSCRIBER LIST (Edit this manually)
    # You just add strings to this list to add new users.
    SUBSCRIBERS = [
        "cheeperholy@gmail.com"
    ]

    EMAIL_SENDER = os.environ.get('EMAIL_USER')
    EMAIL_PASSWORD = os.environ.get('EMAIL_PASS')

    # 2. LOGIN ONCE
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
        smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)

        # 3. SEND TO EACH PERSON SEPARATELY
        for email in SUBSCRIBERS:
            msg = EmailMessage()
            msg['From'] = "AI Trading Bot"
            msg['To'] = email  # Only this person sees their name here
            msg['Subject'] = f"📊 AI Trade Alert: {signal} SSO"

            body = f"""
            Hi! Here is today's market scan:

            Signal: {signal}
            Confidence: {confidence}%
            Price: ${price}

            View chart: https://ai-regime-trader.streamlit.app
            """
            msg.set_content(body)

            smtp.send_message(msg)
            print(f"Sent email to {email}")
            time.sleep(1)  # Pause for 1 second to be polite to Gmail servers


if __name__ == "__main__":
    print("Email script loaded.")