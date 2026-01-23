import smtplib
import ssl
import os
import time
from email.message import EmailMessage


# Note: We added 'image_path' as a new argument here
def send_daily_alert(signal, confidence, price, image_path=None):
    SUBSCRIBERS = ["your_email@gmail.com"]  # Add friends here
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

            body = f"""
            Hi! Here is today's market scan:

            Signal: {signal}
            Confidence: {confidence}%
            Price: ${price}

            See the attached chart for the 'Why'.
            """
            msg.set_content(body)

            # --- NEW CODE TO ATTACH IMAGE ---
            if image_path and os.path.exists(image_path):
                with open(image_path, 'rb') as f:
                    img_data = f.read()
                    msg.add_attachment(img_data, maintype='image', subtype='png', filename='reasoning.png')
            # -------------------------------

            smtp.send_message(msg)
            print(f"Sent email with chart to {email}")
            time.sleep(1)