"""Low-level Google Sheets connection.

This is the ONLY module that authenticates with Google. It exposes a thin
wrapper to fetch worksheets by tab name. It contains no business logic and
knows nothing about profiles, history, or food data.
"""
import json
from functools import lru_cache
from pathlib import Path

import gspread
from google.oauth2.service_account import Credentials

from src import config

# Read + write: the service account has Editor access to the workbook.
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]


def _parse_json_creds(raw: str) -> Credentials:
    """Build credentials from a service-account JSON string."""
    raw = raw.strip()
    # Tolerate a value accidentally wrapped in surrounding quotes.
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "'\"":
        raw = raw[1:-1].strip()
    try:
        info = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Google credentials JSON is set but is not valid JSON. Paste the ENTIRE "
            "service-account JSON as the value (not a file path, no surrounding "
            f"quotes). JSON parse error: {exc}"
        ) from exc
    return Credentials.from_service_account_info(info, scopes=SCOPES)


def _load_credentials() -> Credentials:
    """Load Google service-account credentials — env first, file only as last resort.

    Order (robust across Railway / Vercel / local, and forgiving of which var
    holds the JSON):
      1. GOOGLE_CREDENTIALS_JSON = full JSON  (production / Railway)
      2. GOOGLE_APPLICATION_CREDENTIALS = full JSON  (accepted if someone pasted the
         JSON into this var by mistake — detected by a leading '{')
      3. GOOGLE_APPLICATION_CREDENTIALS = a path to an EXISTING file  (local dev)
      4. otherwise -> clear, actionable error (never a cryptic FileNotFoundError)
    """
    # 1. Preferred: JSON in GOOGLE_CREDENTIALS_JSON.
    raw = (config.GOOGLE_CREDENTIALS_JSON or "").strip()
    if raw:
        return _parse_json_creds(raw)

    # 2. JSON pasted into GOOGLE_APPLICATION_CREDENTIALS by mistake.
    gac = (config.GOOGLE_APPLICATION_CREDENTIALS or "").strip()
    if gac.startswith("{"):
        return _parse_json_creds(gac)

    # 3. A real local file path (dev only).
    if gac and Path(gac).expanduser().exists():
        return Credentials.from_service_account_file(gac, scopes=SCOPES)
    if config.CREDENTIALS_PATH and Path(config.CREDENTIALS_PATH).exists():
        return Credentials.from_service_account_file(config.CREDENTIALS_PATH, scopes=SCOPES)

    # 4. Nothing usable — say exactly what to do (this is what Railway needs).
    raise RuntimeError(
        "No Google credentials found. On Railway/production, set "
        "GOOGLE_CREDENTIALS_JSON to the FULL service-account JSON (paste the file "
        "contents as the value — NOT a file path). Locally, point "
        "GOOGLE_APPLICATION_CREDENTIALS at the JSON file."
    )


@lru_cache(maxsize=1)
def _get_client() -> gspread.Client:
    """Authenticate once and cache the authorized client."""
    config.validate_sheets_config()
    creds = _load_credentials()
    return gspread.authorize(creds)


@lru_cache(maxsize=1)
def _get_spreadsheet() -> gspread.Spreadsheet:
    """Open the existing workbook by ID (Sheets API only, no Drive API)."""
    return _get_client().open_by_key(config.GOOGLE_SHEETS_SPREADSHEET_ID)


def get_worksheet(tab_name: str) -> gspread.Worksheet:
    """Return the worksheet for the given tab name."""
    return _get_spreadsheet().worksheet(tab_name)


def get_all_records(tab_name: str) -> list[dict]:
    """Return all rows of a tab as a list of dicts keyed by header names."""
    return get_worksheet(tab_name).get_all_records()


def get_or_create_worksheet(tab_name: str, header: list[str]):
    """Return the worksheet for tab_name, creating it (with header) if missing."""
    import gspread  # local import to keep module top clean

    ss = _get_spreadsheet()
    try:
        return ss.worksheet(tab_name)
    except gspread.WorksheetNotFound:
        ws = ss.add_worksheet(title=tab_name, rows=200, cols=max(len(header), 8))
        ws.append_row(header, value_input_option="RAW")
        return ws
