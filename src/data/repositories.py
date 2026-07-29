"""Data-access layer.

Business-level accessors over the Google Sheet tabs. Returns plain Python
dicts/lists so the rest of the app never touches gspread directly. Each
accessor filters by user_id where relevant so no cross-user data is exposed.
"""
from __future__ import annotations

from datetime import datetime, timezone

from src import config
from src.data import sheets_client


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
