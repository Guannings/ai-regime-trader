import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import os
from dotenv import load_dotenv  # <--- NEW IMPORT

# --- 1. CONFIGURATION ---
# Load secrets from the .env file
load_dotenv()

# Get the secrets securely
SENDER_EMAIL = os.getenv("EMAIL_USER")
SENDER_PASSWORD = os.getenv("EMAIL_PASS")
RECEIVER_EMAIL = SENDER_EMAIL  # Send to yourself


def send_daily_alert(signal, confidence, price, image_path, app_link):
    # SAFETY CHECK: Did the .env file load?
    if not SENDER_EMAIL or not SENDER_PASSWORD:
        print("❌ Error: Missing credentials. Check your .env file.")
        print("  Make sure you have a file named '.env' with EMAIL_USER and EMAIL_PASS.")
        return

    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECEIVER_EMAIL
    msg['Subject'] = f"🤖 AI Trade Alert: {signal} ({confidence}%)"

    body = f"""
    <html>
      <body>
        <h2>AI Regime Detector Update</h2>
        <p><b>Signal:</b> {signal}</p>
        <p><b>Confidence:</b> {confidence}%</p>
        <p><b>Current SSO Price:</b> ${price}</p>
        <br>
        <p>See full details here: <a href="{app_link}">Open Dashboard</a></p>
        <br>
        <p><i>This is an automated message from your Python Bot.</i></p>
      </body>
    </html>
    """
    msg.attach(MIMEText(body, 'html'))

    if image_path and os.path.exists(image_path):
        try:
            with open(image_path, 'rb') as f:
                img_data = f.read()
            image = MIMEImage(img_data, name=os.path.basename(image_path))
            msg.attach(image)
        except Exception as e:
            print(f"⚠️ Could not attach image: {e}")

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
        server.quit()
        print(f"📧 Email sent successfully to {RECEIVER_EMAIL}")
    except Exception as e:
        print(f"❌ Email Failed: {e}")
