# notifier.py

import requests
import config
import logger

TELEGRAM_URL = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"


def send_telegram_message(message: str):
    """Send a message to the configured Telegram chat."""
    try:
        payload = {
            "chat_id": config.TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML"
        }
        response = requests.post(TELEGRAM_URL, data=payload, timeout=10)

        if response.status_code != 200:
            logger.log_error(f"Telegram API error: {response.status_code} {response.text}")
    except Exception as e:
        logger.log_error(f"Error sending Telegram message: {e}")