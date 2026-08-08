"""Data-access layer.

Business-level accessors over the Google Sheet tabs. Returns plain Python
dicts/lists so the rest of the app never touches gspread directly. Each
accessor filters by user_id where relevant so no cross-user data is exposed.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone

from gspread.utils import rowcol_to_a1

from src import config
from src.data import sheets_client

# Profile columns the onboarding flow is allowed to write (whitelist).
ALLOWED_PROFILE_FIELDS = {
    "name", "age", "sex", "height_cm", "weight_kg", "unit_pref",
    "diet", "allergies", "conditions", "onboarding_completed",
}

# Long-term memory sheet header (created on first use).
MEMORY_HEADER = [
    "memory_id", "user_id", "category", "key", "value",
    "source_message", "created_at", "updated_at",
]


def get_user(user_id: str) -> dict | None:
    """Return the profile row for user_id as a dict, or None if not found.

    Columns: user_id, name, age, gender, height_cm, weight_kg, goal,
             diet_type, activity_level, allergies, medical_conditions
    """
    records = sheets_client.get_all_records(config.TAB_USERS)
    for row in records:
        if str(row.get("user_id", "")).strip() == str(user_id).strip():
            return row
    return None


def get_user_by_firebase_uid(firebase_uid: str) -> dict | None:
    """Return the Users row whose firebase_uid matches, or None."""
    firebase_uid = str(firebase_uid).strip()
    if not firebase_uid:
        return None
    for row in sheets_client.get_all_records(config.TAB_USERS):
        if str(row.get("firebase_uid", "")).strip() == firebase_uid:
            return row
    return None


def _next_user_id(records: list[dict]) -> str:
    """Compute the next sequential user_id (U001, U002, ...)."""
    max_n = 0
    for row in records:
        m = re.match(r"^U(\d+)$", str(row.get("user_id", "")).strip())
        if m:
            max_n = max(max_n, int(m.group(1)))
    return f"U{max_n + 1:03d}"


def ensure_user(firebase_uid: str, email: str | None = None) -> dict:
    """Get-or-create the Users row for a Firebase account.

    Idempotent: if a row with this firebase_uid already exists, it is returned
    unchanged (no duplicate row). Otherwise a new row is created with a
    generated user_id, the email, created_at (UTC), and onboarding_completed=FALSE.

    Returns a dict: {user_id, firebase_uid, onboarding_completed, created}.
    """
    firebase_uid = str(firebase_uid).strip()
    if not firebase_uid:
        raise ValueError("firebase_uid is required")

    ws = sheets_client.get_worksheet(config.TAB_USERS)
    header = ws.row_values(1)
    records = ws.get_all_records()

    # 1. Existing user -> never create a duplicate.
    for row in records:
        if str(row.get("firebase_uid", "")).strip() == firebase_uid:
            return {
                "user_id": str(row.get("user_id", "")).strip(),
                "firebase_uid": firebase_uid,
                "onboarding_completed": str(row.get("onboarding_completed", "")).strip(),
                # Set by the voice agent when the onboarding call (Step 2) finishes.
                # Absent column / new user -> empty -> treated as not done.
                "onboarding_call_done": str(row.get("onboarding_call_done", "")).strip(),
                "created": False,
            }

    # 2. New user -> create a row aligned to the sheet's header order.
    new_id = _next_user_id(records)
    now = datetime.now(timezone.utc).isoformat()
    values = {
        "user_id": new_id,
        "firebase_uid": firebase_uid,
        "email": email or "",
        "created_at": now,
        "onboarding_completed": "FALSE",
    }
    new_row = [values.get(col, "") for col in header]
    ws.append_row(new_row, value_input_option="RAW")

    return {
        "user_id": new_id,
        "firebase_uid": firebase_uid,
        "onboarding_completed": "FALSE",
        "onboarding_call_done": "FALSE",
        "created": True,
    }


def update_user(firebase_uid: str, fields: dict) -> bool:
    """Update whitelisted profile columns on the user's row (by firebase_uid).

    Used by the onboarding flow to save each screen immediately. Lists/dicts are
    stored as JSON strings (e.g. allergies -> '["peanut","sesame"]'). Always
    bumps updated_at. One batched Sheets write per call.
    """
    firebase_uid = str(firebase_uid).strip()
    if not firebase_uid:
        raise ValueError("firebase_uid is required")

    ws = sheets_client.get_worksheet(config.TAB_USERS)
    all_values = ws.get_all_values()
    if not all_values:
        raise ValueError("Users sheet is empty")
    header = all_values[0]
    if "firebase_uid" not in header:
        raise ValueError("firebase_uid column missing from Users sheet")

    uid_col = header.index("firebase_uid")
    target_row = None
    for i in range(1, len(all_values)):
        row = all_values[i]
        if uid_col < len(row) and row[uid_col].strip() == firebase_uid:
            target_row = i + 1  # 1-based sheet row
            break
    if target_row is None:
        raise ValueError("no Users row for this firebase_uid")

    updates = []
    for key, value in fields.items():
        if key not in ALLOWED_PROFILE_FIELDS or key not in header:
            continue
        if isinstance(value, (list, dict)):
            value = json.dumps(value, ensure_ascii=False)
        col = header.index(key) + 1
        updates.append({"range": rowcol_to_a1(target_row, col), "values": [[value]]})

    if "updated_at" in header:
        updates.append({
            "range": rowcol_to_a1(target_row, header.index("updated_at") + 1),
            "values": [[datetime.now(timezone.utc).isoformat()]],
        })

    if updates:
        ws.batch_update(updates, value_input_option="RAW")
    return True


def _memory_ws():
    """Return the User Memory worksheet, creating it (with header) if missing."""
    return sheets_client.get_or_create_worksheet(config.TAB_USER_MEMORY, MEMORY_HEADER)


def get_memories(user_id: str) -> list[dict]:
    """Return all long-term memory rows for a single user (never cross-user)."""
    user_id = str(user_id).strip()
    if not user_id:
        return []
    return [
        r for r in _memory_ws().get_all_records()
        if str(r.get("user_id", "")).strip() == user_id
    ]


def _next_memory_id(all_values: list[list], key_index: int) -> str:
    """Compute the next sequential memory_id (M0001, M0002, ...)."""
    max_n = 0
    for i in range(1, len(all_values)):
        row = all_values[i]
        if key_index < len(row):
            m = re.match(r"^M(\d+)$", row[key_index].strip())
            if m:
                max_n = max(max_n, int(m.group(1)))
    return f"M{max_n + 1:04d}"


def upsert_memory(user_id: str, category: str, key: str, value: str,
                  source_message: str = "") -> str:
    """Insert or update one long-term memory for a user.

    Dedup key is (user_id, key) case-insensitive: re-learning the same fact
    (e.g. target_weight 70 -> 65) UPDATES the existing row instead of adding a
    duplicate. Returns "created" or "updated".
    """
    user_id = str(user_id).strip()
    key = str(key).strip()
    if not user_id or not key or not str(value).strip():
        raise ValueError("user_id, key and value are required")

    ws = _memory_ws()
    all_values = ws.get_all_values()
    header = all_values[0] if all_values else MEMORY_HEADER
    idx = {h: i for i, h in enumerate(header)}
    now = datetime.now(timezone.utc).isoformat()

    # Look for an existing memory with the same user_id + key.
    target_row = None
    for i in range(1, len(all_values)):
        row = all_values[i]
        same_user = idx.get("user_id", 1) < len(row) and row[idx["user_id"]].strip() == user_id
        same_key = idx.get("key", 3) < len(row) and row[idx["key"]].strip().lower() == key.lower()
        if same_user and same_key:
            target_row = i + 1  # 1-based sheet row
            break

    if target_row:
        changes = {"category": category, "value": value,
                   "source_message": source_message, "updated_at": now}
        updates = [
            {"range": rowcol_to_a1(target_row, idx[col] + 1), "values": [[val]]}
            for col, val in changes.items() if col in idx
        ]
        ws.batch_update(updates, value_input_option="RAW")
        return "updated"

    mem_id = _next_memory_id(all_values, idx.get("memory_id", 0))
    values = {
        "memory_id": mem_id, "user_id": user_id, "category": category,
        "key": key, "value": value, "source_message": source_message,
        "created_at": now, "updated_at": now,
    }
    ws.append_row([values.get(h, "") for h in header], value_input_option="RAW")
    return "created"


def get_recent_history(user_id: str, limit: int = config.HISTORY_LIMIT) -> list[dict]:
    """Return the most recent `limit` messages for user_id, chronological order.

    Columns: user_id, timestamp, role, message
    Rows in the sheet are assumed to be appended in time order; we filter by
    user, keep the last `limit`, and preserve chronological order.
    """
    records = sheets_client.get_all_records(config.TAB_CONVERSATION_HISTORY)
    user_rows = [
        r for r in records
        if str(r.get("user_id", "")).strip() == str(user_id).strip()
    ]
    # Keep only the most recent `limit`, preserving chronological order.
    recent = user_rows[-limit:] if limit else user_rows
    return recent


def append_message(user_id: str, role: str, message: str) -> None:
    """Append a single message to conversation_history.

    Writes a row in the exact column order: user_id, timestamp, role, message.
    Timestamp is UTC ISO-8601.
    """
    ws = sheets_client.get_worksheet(config.TAB_CONVERSATION_HISTORY)
    timestamp = datetime.now(timezone.utc).isoformat()
    ws.append_row(
        [str(user_id), timestamp, role, message],
        value_input_option="RAW",
    )


def get_food_data() -> list[dict]:
    """Return all food_data rows.

    Columns: food, calories_per_100g, protein_g, carbs_g, fat_g, category
    The full table is small; relevant-food matching happens in memory
    (see context.food_matcher). Only matched rows are ever sent to the LLM.
    """
    return sheets_client.get_all_records(config.TAB_FOOD_DATA)


def get_diet_plans(user_id: str) -> list[dict]:
    """Return diet_plans rows for user_id.

    Columns: user_id, date, meal_type, food, quantity, calories

    NOTE: Provided for future use only. Diet plans are NOT sent to the LLM
    automatically in this prototype.
    """
    records = sheets_client.get_all_records(config.TAB_DIET_PLANS)
    return [
        r for r in records
        if str(r.get("user_id", "")).strip() == str(user_id).strip()
    ]
