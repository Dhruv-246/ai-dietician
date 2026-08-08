"""Read a user's manual-onboarding profile from the shared Google Sheet.

The voice agent is a separate service from the web app, so this is a small,
self-contained reader that uses the SAME service-account credentials and
spreadsheet the web app writes to. Given a Firebase uid, it returns the profile
fields the call prompt needs so Mira can greet the user by name and never
re-ask what manual onboarding already collected.

Env (already set on the Railway agent service):
  GOOGLE_CREDENTIALS_JSON       full service-account JSON
  GOOGLE_SHEETS_SPREADSHEET_ID  the workbook id
"""
import json
import os
from functools import lru_cache

import gspread
from google.oauth2.service_account import Credentials

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
TAB_USERS = "users"  # matches the web app's config.TAB_USERS


def _load_credentials() -> Credentials:
    """Build service-account creds from the JSON env var (env-first, like the web app)."""
    raw = (os.getenv("GOOGLE_CREDENTIALS_JSON") or "").strip()
    if not raw:
        # Tolerate the JSON accidentally placed in the file-path var.
        gac = (os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or "").strip()
        if gac.startswith("{"):
            raw = gac
    if not raw:
        raise RuntimeError(
            "No Google credentials. Set GOOGLE_CREDENTIALS_JSON to the full "
            "service-account JSON on the agent service."
        )
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "'\"":
        raw = raw[1:-1].strip()
    return Credentials.from_service_account_info(json.loads(raw), scopes=SCOPES)


@lru_cache(maxsize=1)
def get_spreadsheet():
    """Authorize once and return the workbook handle (cached, shared)."""
    client = gspread.authorize(_load_credentials())
    spreadsheet_id = os.getenv("GOOGLE_SHEETS_SPREADSHEET_ID")
    if not spreadsheet_id:
        raise RuntimeError("GOOGLE_SHEETS_SPREADSHEET_ID is not set")
    return client.open_by_key(spreadsheet_id)


@lru_cache(maxsize=1)
def _worksheet():
    """Return the Users worksheet handle (cached)."""
    return get_spreadsheet().worksheet(TAB_USERS)


def _clean(value) -> str:
    return str(value if value is not None else "").strip()


def _list_field(raw) -> str:
    """allergies/conditions are stored as JSON list strings; render them readable."""
    s = _clean(raw)
    if not s:
        return ""
    try:
        parsed = json.loads(s)
    except (ValueError, TypeError):
        return s  # plain text, use as-is
    if isinstance(parsed, list):
        return ", ".join(_clean(x) for x in parsed if _clean(x))
    return _clean(parsed)


def load_profile_for_uid(firebase_uid: str) -> dict:
    """Return the call-prompt profile for a Firebase uid, or {} if not found.

    Keys match call_prompt.md's {{...}} variables: name, age, gender, height,
    weight, diet, allergies, conditions. Reads fresh rows each call so the
    profile reflects the latest onboarding save.
    """
    firebase_uid = _clean(firebase_uid)
    if not firebase_uid:
        return {}

    for row in _worksheet().get_all_records():
        if _clean(row.get("firebase_uid")) == firebase_uid:
            height = _clean(row.get("height_cm"))
            weight = _clean(row.get("weight_kg"))
            return {
                "name": _clean(row.get("name")),
                "age": _clean(row.get("age")),
                "gender": _clean(row.get("sex")),
                "height": f"{height} cm" if height else "",
                "weight": f"{weight} kg" if weight else "",
                "diet": _clean(row.get("diet")),
                "allergies": _list_field(row.get("allergies")),
                "conditions": _list_field(row.get("conditions")),
            }
    return {}
