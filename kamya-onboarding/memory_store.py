"""Long-term memory + session records for the ongoing (Step-3) product.

Stores per-user long-term memory and per-session records in the SAME Google
Sheet the web app uses. Memory is CUMULATIVE: consolidation always merges new
info into the existing memory doc — nothing from past sessions is lost unless
it's explicitly contradicted. Self-contained; reuses the authenticated
spreadsheet handle from profile_store.

Durability: the raw transcript is written to the Sessions tab BEFORE any
consolidation is attempted (`save_session_raw`), and the row is updated
afterwards (`finalize_session`). A consolidation failure therefore costs the
memory *update*, never the conversation itself — the row keeps the full
transcript with status=failed and can be replayed later.

Sheet layout it manages (created automatically, columns added if missing):
  Users tab  → extra columns: long_term_memory, open_loops,
               last_session_summary, last_session_at, session_count,
               onboarding_call_done
  Sessions tab → one row per call. Original columns: session_id(run_id),
               user_id, type, started_at, ended_at, session_summary,
               open_loops. Appended for durability: firebase_uid, turns,
               status, transcript, error, attempts, consolidated_at.
"""
import json

from gspread.utils import rowcol_to_a1

import profile_store

TAB_USERS = "users"
TAB_SESSIONS = "Sessions"

# Session lifecycle. A row is written as PENDING before consolidation runs;
# it then becomes DONE, or FAILED with the error kept for diagnosis. Both
# PENDING and FAILED rows still hold the transcript, so both are replayable.
STATUS_PENDING = "pending"
STATUS_DONE = "consolidated"
STATUS_FAILED = "failed"

# A single Google Sheets cell holds at most 50k characters. Leave headroom
# rather than losing the whole write to a hard API rejection.
MAX_TRANSCRIPT_CHARS = 45000

_MEM_COLS = [
    "long_term_memory", "open_loops", "last_session_summary",
    "last_session_at", "session_count", "onboarding_call_done",
]
# NOTE: new columns are APPENDED, never reordered — gspread maps rows by header
# name, so appending keeps every pre-existing Sessions row readable.
_SESSIONS_HEADER = [
    "session_id", "user_id", "type", "started_at", "ended_at",
    "session_summary", "open_loops",
    "firebase_uid", "turns", "status", "transcript", "error",
    "attempts", "consolidated_at",
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


def _add_missing_columns(ws, wanted):
    """Append any missing columns to a worksheet's header row. Idempotent."""
    header = ws.row_values(1)
    for col in wanted:
        if col not in header:
            header.append(col)
            idx = len(header)
            if ws.col_count < idx:
                ws.add_cols(idx - ws.col_count)
            ws.update_cell(1, idx, col)


def ensure_schema():
    """Add memory columns to Users and durability columns to Sessions. Idempotent.

    Safe to run against a Sessions tab created before the transcript columns
    existed — the new columns are appended, so old rows keep their values and
    simply read back empty for the new fields.
    """
    global _schema_ready
    if _schema_ready:
        return
    _add_missing_columns(_users_ws(), _MEM_COLS)
    _add_missing_columns(_sessions_ws(), _SESSIONS_HEADER)
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


def get_sessions(user_id, limit=15):
    """Return this user's recent session records (newest first)."""
    user_id = str(user_id or "").strip()
    if not user_id:
        return []
    try:
        ensure_schema()
        rows = _sessions_ws().get_all_records()
    except Exception as exc:
        print(f"[memory] sessions read failed: {exc}", flush=True)
        return []
    mine = [r for r in rows if str(r.get("user_id", "")).strip() == user_id]
    return list(reversed(mine))[:limit]


def parse_loops(raw):
    """Parse an open_loops cell (JSON list string) into a Python list."""
    return _parse_json(raw, [])


def _find_session_row(ws, session_id):
    """Return (row_index_1based, header) for a session_id, else (None, header).

    Reads only the session_id column rather than the whole sheet — the
    transcript column makes a full get_all_values() needlessly expensive.
    """
    header = ws.row_values(1)
    if "session_id" not in header:
        return None, header
    ids = ws.col_values(header.index("session_id") + 1)
    target = str(session_id).strip()
    for i in range(1, len(ids)):
        if ids[i].strip() == target:
            return i + 1, header
    return None, header


def save_session_raw(session_id, firebase_uid, user_id, session_type,
                     started_at, ended_at, transcript, turns):
    """Persist the raw transcript as a PENDING session row. Call BEFORE consolidating.

    THE DURABILITY BOUNDARY. Once this returns, the conversation is on disk and
    survives any later failure — a malformed LLM response, a crashed process, a
    revoked API key. Everything downstream (consolidation, memory merge) is a
    derived artefact that can be recomputed from this row.

    Raises on failure: the caller must know that the transcript was NOT saved.
    """
    ensure_schema()
    ws = _sessions_ws()
    header = ws.row_values(1)
    text = (transcript or "")
    if len(text) > MAX_TRANSCRIPT_CHARS:
        # Keep the tail: the end of a call carries the commitments and follow-ups.
        text = "[TRUNCATED]\n" + text[-MAX_TRANSCRIPT_CHARS:]
    values = {
        "session_id": session_id,
        "user_id": user_id,
        "firebase_uid": firebase_uid,
        "type": session_type,
        "started_at": started_at,
        "ended_at": ended_at,
        "turns": turns,
        "status": STATUS_PENDING,
        "transcript": text,
        "attempts": 0,
    }
    ws.append_row([values.get(col, "") for col in header], value_input_option="RAW")


def finalize_session(session_id, status, session_summary="", open_loops=None,
                     error="", consolidated_at=""):
    """Update an already-persisted session row after a consolidation attempt.

    Never touches the transcript column, so a FAILED finalize still leaves the
    conversation replayable. One batched write.
    """
    ensure_schema()
    ws = _sessions_ws()
    row_idx, header = _find_session_row(ws, session_id)
    if not row_idx:
        print(f"[memory] finalize: no session row for {session_id}", flush=True)
        return False

    prev_attempts = 0
    if "attempts" in header:
        raw = ws.cell(row_idx, header.index("attempts") + 1).value
        prev_attempts = int(str(raw or "0").strip() or 0)

    updates = {
        "status": status,
        "attempts": prev_attempts + 1,
        "error": (error or "")[:900],
    }
    if session_summary:
        updates["session_summary"] = session_summary
    if open_loops is not None:
        updates["open_loops"] = json.dumps(open_loops, ensure_ascii=False)
    if consolidated_at:
        updates["consolidated_at"] = consolidated_at

    batch = [
        {"range": rowcol_to_a1(row_idx, header.index(name) + 1), "values": [[val]]}
        for name, val in updates.items() if name in header
    ]
    if batch:
        ws.batch_update(batch, value_input_option="RAW")
    return True


def get_replayable_sessions(limit=25):
    """Return sessions that still need consolidating, oldest first.

    A row qualifies when it holds a transcript but never reached DONE. This is
    the recovery queue: every conversation whose memory write failed.
    """
    ensure_schema()
    out = []
    for r in _sessions_ws().get_all_records():
        status = str(r.get("status", "")).strip().lower()
        transcript = str(r.get("transcript", "")).strip()
        if not transcript or status == STATUS_DONE:
            continue
        out.append({
            "session_id": str(r.get("session_id", "")).strip(),
            "firebase_uid": str(r.get("firebase_uid", "")).strip(),
            "user_id": str(r.get("user_id", "")).strip(),
            "type": str(r.get("type", "")).strip() or "ongoing",
            "started_at": str(r.get("started_at", "")).strip(),
            "ended_at": str(r.get("ended_at", "")).strip(),
            "turns": str(r.get("turns", "")).strip(),
            "status": status or STATUS_PENDING,
            "attempts": str(r.get("attempts", "")).strip() or "0",
            "error": str(r.get("error", "")).strip(),
            "transcript": transcript,
        })
        if len(out) >= limit:
            break
    return out


def save_user_memory(firebase_uid, session_type, ended_at,
                     merged_memory, session_summary, open_loops):
    """Merge-write consolidated memory onto the user's Users row.

    Only the Users row — the Sessions row is written separately by
    save_session_raw/finalize_session so that transcript durability never
    depends on consolidation succeeding. One batched write.

    Cumulative: `merged_memory` is already the full merged doc.
    """
    ensure_schema()
    ws = _users_ws()
    row_idx, header, row = _find_row(ws, firebase_uid)
    if not row_idx:
        return False
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
    batch = [
        {"range": rowcol_to_a1(row_idx, header.index(name) + 1), "values": [[val]]}
        for name, val in updates.items() if name in header
    ]
    if batch:
        ws.batch_update(batch, value_input_option="RAW")
    return True
