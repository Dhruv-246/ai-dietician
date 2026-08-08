"""Long-term memory + session records for the ongoing (Step-3) product.

Stores per-user long-term memory and per-session records in the SAME Google
Sheet the web app uses. Memory is CUMULATIVE: consolidation always merges new
info into the existing memory doc — nothing from past sessions is lost unless
it's explicitly contradicted. Self-contained; reuses the authenticated
spreadsheet handle from profile_store.

Sheet layout it manages (created automatically if missing):
  Users tab  → extra columns: long_term_memory, open_loops,
               last_session_summary, last_session_at, session_count,
               onboarding_call_done
  Sessions tab (new) → one row per call: session_id(run_id), user_id, type,
               started_at, ended_at, session_summary, open_loops
"""
import json

import profile_store

TAB_USERS = "users"
TAB_SESSIONS = "Sessions"

_MEM_COLS = [
    "long_term_memory", "open_loops", "last_session_summary",
    "last_session_at", "session_count", "onboarding_call_done",
]
_SESSIONS_HEADER = [
    "session_id", "user_id", "type", "started_at", "ended_at",
    "session_summary", "open_loops",
]

_schema_ready = False


def _users_ws():
    return profile_store.get_spreadsheet().worksheet(TAB_USERS)


def _sessions_ws():
    ss = profile_store.get_spreadsheet()
    try:
        return ss.worksheet(TAB_SESSIONS)
    except Exception:
        ws = ss.add_worksheet(title=TAB_SESSIONS, rows=1000, cols=len(_SESSIONS_HEADER))
        ws.append_row(_SESSIONS_HEADER, value_input_option="RAW")
        return ws


def ensure_schema():
    """Add memory columns to Users (if missing) and create the Sessions tab. Idempotent."""
    global _schema_ready
    if _schema_ready:
        return
    ws = _users_ws()
    header = ws.row_values(1)
    for col in _MEM_COLS:
        if col not in header:
            header.append(col)
            idx = len(header)
            if ws.col_count < idx:
                ws.add_cols(idx - ws.col_count)
            ws.update_cell(1, idx, col)
    _sessions_ws()
    _schema_ready = True


def _find_row(ws, firebase_uid):
    """Return (row_index_1based, header, row_values) for firebase_uid, else (None, header, None)."""
    values = ws.get_all_values()
    if not values:
        return None, [], None
    header = values[0]
    if "firebase_uid" not in header:
        return None, header, None
    uid_col = header.index("firebase_uid")
    target = str(firebase_uid).strip()
    for i in range(1, len(values)):
        row = values[i]
        if uid_col < len(row) and row[uid_col].strip() == target:
            return i + 1, header, row
    return None, header, None


def _cell(header, row, name):
    if name in header:
        idx = header.index(name)
        if idx < len(row):
            return row[idx].strip()
    return ""


def _parse_json(raw, default):
    raw = (raw or "").strip()
    if not raw:
        return default
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return default


def load_memory(firebase_uid):
    """Return this user's memory + continuity signals (safe defaults if new/none)."""
    default = {
        "user_id": "", "long_term_memory": {}, "open_loops": [],
        "last_session_summary": "", "last_session_at": "",
        "session_count": 0, "onboarding_call_done": False,
    }
    if not firebase_uid:
        return default
    try:
        ensure_schema()
        row_idx, header, row = _find_row(_users_ws(), firebase_uid)
        if not row_idx:
            return default
        return {
            "user_id": _cell(header, row, "user_id"),
            "long_term_memory": _parse_json(_cell(header, row, "long_term_memory"), {}),
            "open_loops": _parse_json(_cell(header, row, "open_loops"), []),
            "last_session_summary": _cell(header, row, "last_session_summary"),
            "last_session_at": _cell(header, row, "last_session_at"),
            "session_count": int(_cell(header, row, "session_count") or 0),
            "onboarding_call_done": _cell(header, row, "onboarding_call_done").upper() == "TRUE",
        }
    except Exception as exc:
        print(f"[memory] load failed: {exc}", flush=True)
        return default


def save_consolidation(firebase_uid, user_id, run_id, session_type,
                       started_at, ended_at, merged_memory, session_summary, open_loops):
    """Merge-write updated memory to the Users row and append a Sessions row.

    Cumulative: `merged_memory` is already the full merged doc produced by the
    consolidation step (existing + new), so this just persists it.
    """
    ensure_schema()
    ws = _users_ws()
    row_idx, header, row = _find_row(ws, firebase_uid)
    if row_idx:
        prev_count = int(_cell(header, row, "session_count") or 0)
        updates = {
            "long_term_memory": json.dumps(merged_memory, ensure_ascii=False),
            "open_loops": json.dumps(open_loops, ensure_ascii=False),
            "last_session_summary": session_summary,
            "last_session_at": ended_at,
            "session_count": prev_count + 1,
        }
        if session_type == "onboarding":
            updates["onboarding_call_done"] = "TRUE"
        for name, val in updates.items():
            if name in header:
                ws.update_cell(row_idx, header.index(name) + 1, val)
    _sessions_ws().append_row(
        [run_id, user_id, session_type, started_at, ended_at,
         session_summary, json.dumps(open_loops, ensure_ascii=False)],
        value_input_option="RAW",
    )
