"""Central configuration. Loads settings from environment / .env.

No secrets are hardcoded here. Values come from the .env file (gitignored).
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# Project root = two levels up from this file (src/config.py -> project root).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env from the project root if present.
load_dotenv(PROJECT_ROOT / ".env")

# --- Paths ---
SYSTEM_PROMPT_PATH = PROJECT_ROOT / "prompts" / "system_prompt.md"

# --- Google Sheets ---
# Local dev: path to the service-account JSON file (from_service_account_file).
GOOGLE_APPLICATION_CREDENTIALS = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
# Production (Vercel): the full service-account JSON as a single env var
# (from_service_account_info). Takes precedence over the file path when set.
GOOGLE_CREDENTIALS_JSON = os.environ.get("GOOGLE_CREDENTIALS_JSON")
GOOGLE_SHEETS_SPREADSHEET_ID = os.environ.get("GOOGLE_SHEETS_SPREADSHEET_ID")

# --- OpenRouter ---
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "anthropic/claude-sonnet-5")
OPENROUTER_BASE_URL = os.environ.get(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
)

# Tab names in the workbook (kept here so they are defined in one place).
TAB_USERS = "users"
TAB_CONVERSATION_HISTORY = "conversation_history"
TAB_FOOD_DATA = "food_data"
TAB_DIET_PLANS = "diet_plans"
TAB_USER_MEMORY = "User Memory"

# How many recent messages to include as history.
HISTORY_LIMIT = 20

# --- Web / demo ---
# Default 5050 avoids macOS's AirPlay Receiver, which occupies port 5000.
WEB_PORT = int(os.environ.get("WEB_PORT", "5050"))
DEFAULT_USER_ID = os.environ.get("DEFAULT_USER_ID", "U001")

# --- Firebase (client-side auth config) ---
# These are the Firebase *web app* config values. They are NOT secrets — the
# web SDK is designed to expose them in the browser. We read them from env so
# nothing is hardcoded and they can be set per environment (local / Vercel).
FIREBASE_CONFIG = {
    "apiKey": os.environ.get("FIREBASE_API_KEY"),
    "authDomain": os.environ.get("FIREBASE_AUTH_DOMAIN"),
    "projectId": os.environ.get("FIREBASE_PROJECT_ID"),
    "appId": os.environ.get("FIREBASE_APP_ID"),
    "messagingSenderId": os.environ.get("FIREBASE_MESSAGING_SENDER_ID"),
    "storageBucket": os.environ.get("FIREBASE_STORAGE_BUCKET"),
}


def _resolve_credentials_path() -> str:
    """Return an absolute path to the service-account JSON.

    A relative path in .env is resolved against the project root so the app
    works regardless of the current working directory.
    """
    if not GOOGLE_APPLICATION_CREDENTIALS:
        raise RuntimeError(
            "GOOGLE_APPLICATION_CREDENTIALS is not set. Add it to your .env file."
        )
    p = Path(GOOGLE_APPLICATION_CREDENTIALS)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return str(p)


CREDENTIALS_PATH = None
if GOOGLE_APPLICATION_CREDENTIALS:
    CREDENTIALS_PATH = _resolve_credentials_path()


def validate_sheets_config() -> None:
    """Raise a clear error if required Sheets config is missing.

    Credentials may come from EITHER GOOGLE_CREDENTIALS_JSON (production, e.g.
    Vercel) OR GOOGLE_APPLICATION_CREDENTIALS (a local file path). At least one
    must be present.
    """
    if not GOOGLE_CREDENTIALS_JSON and not GOOGLE_APPLICATION_CREDENTIALS:
        raise RuntimeError(
            "No Google credentials configured. Set GOOGLE_CREDENTIALS_JSON "
            "(production) or GOOGLE_APPLICATION_CREDENTIALS (local file path)."
        )
    if not GOOGLE_SHEETS_SPREADSHEET_ID:
        raise RuntimeError("GOOGLE_SHEETS_SPREADSHEET_ID is not set")


def validate_openrouter_config() -> None:
    """Raise a clear error if the OpenRouter API key is missing."""
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Add it to your .env file "
            "(see .env.example)."
        )
