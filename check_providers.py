from llm_tg_bot.config import load_settings
import os

# Mock required env vars if missing
if not os.getenv("TELEGRAM_BOT_TOKENS"):
    os.environ["TELEGRAM_BOT_TOKENS"] = "fake_token"
if not os.getenv("TELEGRAM_ALLOWED_USER_IDS"):
    os.environ["TELEGRAM_ALLOWED_USER_IDS"] = "*"

try:
    settings = load_settings()
    print(f"Detected providers: {list(settings.providers.keys())}")
    print(f"Default provider: {settings.default_provider}")
except Exception as e:
    print(f"Error loading settings: {e}")
