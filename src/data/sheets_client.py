"""Low-level Google Sheets connection.

This is the ONLY module that authenticates with Google. It exposes a thin
wrapper to fetch worksheets by tab name. It contains no business logic and
knows nothing about profiles, history, or food data.
"""
import json
from functools import lru_cache

import gspread
from google.oauth2.service_account import Credentials

from src import config

# Read + write: the service account has Editor access to the workbook.
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]


def _load_credentials() -> Credentials:
    """Build service-account credentials from env JSON or a local file.

    Priority:
      1. GOOGLE_CREDENTIALS_JSON  -> from_service_account_info (production/Vercel)
      2. GOOGLE_APPLICATION_CREDENTIALS file path -> from_service_account_file (local)
    """
    raw = config.GOOGLE_CREDENTIALS_JSON
    if raw:
        raw = raw.strip()
        # Tolerate a value accidentally wrapped in surrounding quotes.
        if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "'\"":
            raw = raw[1:-1].strip()
        try:
            info = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "GOOGLE_CREDENTIALS_JSON is set but is not valid JSON. Paste the "
                "ENTIRE contents of the service-account JSON file as the value "
                "(not a file path, no surrounding quotes). JSON parse error: "
                f"{exc}"
            ) from exc
        return Credentials.from_service_account_info(info, scopes=SCOPES)
    return Credentials.from_service_account_file(
        config.CREDENTIALS_PATH, scopes=SCOPES
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
