import smtplib
import ssl
import os
import time
from email.message import EmailMessage


def send_daily_alert(signal, confidence, price, image_path=None):
    # 👇 STEP 1: Paste your actual Streamlit link here
    DASHBOARD_LINK = "https://your-app-name.streamlit.app"

    SUBSCRIBERS = ["cheeperholy@gmail.com"]
    EMAIL_SENDER = os.environ.get('EMAIL_USER')
    EMAIL_PASSWORD = os.environ.get('EMAIL_PASS')

    context = ssl.create_default_context()

    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
        smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)

        for email in SUBSCRIBERS:
            msg = EmailMessage()
            msg['From'] = "AI Trading Bot"
            msg['To'] = email
            msg['Subject'] = f"📊 AI Trade Alert: {signal} SSO"

            # 👇 STEP 2: Add the link to the message body
            body = f"""
            Hi! Here is today's market scan:

            Signal: {signal}
            Confidence: {confidence}%
            Price: ${price}

            📱 View Live Dashboard: {DASHBOARD_LINK}

            See the attached chart for the 'Why'.
            """
            msg.set_content(body)

            if image_path and os.path.exists(image_path):
                with open(image_path, 'rb') as f:
                    img_data = f.read()
                    msg.add_attachment(img_data, maintype='image', subtype='png', filename='reasoning.png')

            smtp.send_message(msg)
            print(f"Sent email to {email}")
            time.sleep(1)