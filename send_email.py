import smtplib
import ssl
import os
import time
from email.message import EmailMessage


def send_daily_alert(signal, confidence, price, image_path=None, app_link=""):
    # --- CONFIG ---
    # Add your friends' emails to this list if you want
    SUBSCRIBERS = ["cheeperholy@gmail.com"]

    EMAIL_SENDER = os.environ.get('EMAIL_USER')
    EMAIL_PASSWORD = os.environ.get('EMAIL_PASS')

    if not EMAIL_SENDER or not EMAIL_PASSWORD:
        print("Error: Email credentials not found in environment variables.")
        return

    context = ssl.create_default_context()

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
            smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)

            for email in SUBSCRIBERS:
                msg = EmailMessage()
                msg['From'] = "AI Trading Bot"
                msg['To'] = email
                msg['Subject'] = f"📊 AI Trade Alert: {signal} SSO ({confidence}%)"

                # HTML Body to make the link clickable
                body = f"""
                <html>
                  <body>
                    <h2>Daily Market Scan Complete</h2>
                    <p><strong>Signal:</strong> {signal}</p>
                    <p><strong>Confidence:</strong> {confidence}%</p>
                    <p><strong>Latest Price:</strong> ${price}</p>
                    <hr>
                    <p>📈 <strong><a href="{app_link}">Click here to view the full Dashboard & Live Charts</a></strong></p>
                    <hr>
                    <p><em>See attached chart for the factors driving this decision.</em></p>
                  </body>
                </html>
                """
                msg.add_alternative(body, subtype='html')

                # Attach the Reasoning Chart
                if image_path and os.path.exists(image_path):
                    with open(image_path, 'rb') as f:
                        img_data = f.read()
                        msg.add_attachment(img_data, maintype='image', subtype='png', filename='reasoning.png')

                smtp.send_message(msg)
                print(f"✅ Sent email to {email}")
                time.sleep(1)  # Be nice to Gmail server

    except Exception as e:
        print(f"❌ Failed to send email: {e}")
