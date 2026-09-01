"""Durable chat sessions, so a restart does not eat the conversation.

WHY. Sessions lived only in memory. A Railway deploy — or any crash — took
the live thread with it: the user came back to an empty screen, and every fact
extracted since the last consolidation was gone. Consolidated facts were safe
in the ledger, but the pending buffer was not, and on a chat surface that
window can be 45 minutes wide.

Worse, an in-memory session that dies is never CLOSED, so consolidation never
runs for it. The conversation is not merely interrupted — it is never written
down at all.

WHERE. Supabase, which the RAG path already talks to over REST with httpx.
No new dependency, no new credential, and it is the only durable store here
that is fast enough to write on every turn (Sheets takes about a second).

BEST EFFORT BY DESIGN. Every function swallows its errors. A persistence
outage must degrade to today's behaviour — a live conversation that is lost on
restart — never break the conversation actually in progress. Losing history is
bad; failing the user's next message because a database was slow is worse.

    create table if not exists chat_sessions (
      firebase_uid    text primary key,
      session_id      text not null,
      started_at      double precision,
      last_activity   double precision,
      messages        jsonb default '[]'::jsonb,
      rolling_summary text default '',
      pending_facts   jsonb default '{}'::jsonb,
      threads         jsonb default '[]'::jsonb,
      closed          boolean default false,
      updated_at      timestamptz default now()
    );
    create index if not exists chat_sessions_open_idx
      on chat_sessions (closed, last_activity);
"""
from __future__ import annotations

import dataclasses
import os

TABLE = os.getenv("CHAT_SESSIONS_TABLE", "chat_sessions")
TIMEOUT = float(os.getenv("CHAT_STORE_TIMEOUT", "6"))


def enabled() -> bool:
    return bool(os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_KEY"))


def _url(path: str = "") -> str:
    return os.getenv("SUPABASE_URL", "").rstrip("/") + f"/rest/v1/{TABLE}{path}"


def _headers(extra=None) -> dict:
    key = os.getenv("SUPABASE_KEY", "")
    h = {"apikey": key, "Authorization": f"Bearer {key}",
         "Content-Type": "application/json"}
    h.update(extra or {})
    return h


def _threads_to_json(threads):
    out = []
    for t in threads or []:
        try:
            out.append(dataclasses.asdict(t))
        except Exception:
            pass          # a thread we cannot serialise is dropped, not fatal
    return out


def _threads_from_json(rows):
    import thread_machine as tm
    out = []
    for r in rows or []:
        try:
            out.append(tm.Thread(**r))
        except Exception:
            pass          # schema drift must not break loading the messages
    return out


async def save(session, log=None) -> bool:
    """Upsert the live session. Called after every turn."""
    log = log or (lambda m: None)
    if not enabled():
        return False
    try:
        import httpx
        row = {
            "firebase_uid": session.firebase_uid,
            "session_id": session.session_id,
            "started_at": session.started_at,
            "last_activity": session.last_activity,
            "messages": session.messages,
            "rolling_summary": session.rolling_summary,
            "pending_facts": session.pending_facts,
            "threads": _threads_to_json(session.threads),
            "closed": session.closed,
        }
        async with httpx.AsyncClient(timeout=TIMEOUT) as cx:
            r = await cx.post(
                _url(), headers=_headers({"Prefer": "resolution=merge-duplicates"}),
                json=row)
            r.raise_for_status()
        return True
    except Exception as exc:
        log(f"chat store save failed: {type(exc).__name__}: {exc}")
        return False


async def load(uid: str, log=None):
    """The user's open session, or None. Restores a thread across a restart."""
    log = log or (lambda m: None)
    if not enabled() or not uid:
        return None
    try:
        import httpx
        async with httpx.AsyncClient(timeout=TIMEOUT) as cx:
            r = await cx.get(_url(), headers=_headers(),
                             params={"firebase_uid": f"eq.{uid}",
                                     "closed": "is.false", "limit": "1"})
            r.raise_for_status()
            rows = r.json() or []
        return rows[0] if rows else None
    except Exception as exc:
        log(f"chat store load failed: {type(exc).__name__}: {exc}")
        return None


async def open_sessions(log=None):
    """Every unclosed session, for the reaper to adopt after a restart.

    Without this an interrupted session is never closed and therefore never
    consolidated -- its facts are not delayed, they are lost.
    """
    log = log or (lambda m: None)
    if not enabled():
        return []
    try:
        import httpx
        async with httpx.AsyncClient(timeout=TIMEOUT) as cx:
            r = await cx.get(_url(), headers=_headers(),
                             params={"closed": "is.false",
                                     "order": "last_activity.asc",
                                     "limit": "200"})
            r.raise_for_status()
            return r.json() or []
    except Exception as exc:
        log(f"chat store list failed: {type(exc).__name__}: {exc}")
        return []


async def mark_closed(uid: str, log=None) -> bool:
    log = log or (lambda m: None)
    if not enabled() or not uid:
        return False
    try:
        import httpx
        async with httpx.AsyncClient(timeout=TIMEOUT) as cx:
            r = await cx.patch(_url(), headers=_headers(),
                               params={"firebase_uid": f"eq.{uid}"},
                               json={"closed": True})
            r.raise_for_status()
        return True
    except Exception as exc:
        log(f"chat store close failed: {type(exc).__name__}: {exc}")
        return False


def restore(session, row):
    """Copy a stored row back onto a fresh ChatSession, in place."""
    if not row:
        return session
    session.session_id = row.get("session_id") or session.session_id
    session.started_at = row.get("started_at") or session.started_at
    session.last_activity = row.get("last_activity") or session.last_activity
    session.messages = list(row.get("messages") or [])
    session.rolling_summary = row.get("rolling_summary") or ""
    session.pending_facts = dict(row.get("pending_facts") or {})
    session.threads = _threads_from_json(row.get("threads"))
    session.turn_index = sum(1 for m in session.messages
                             if m.get("role") == "user")
    return session
