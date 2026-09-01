"""Chat session lifecycle — the thing a voice call gets for free.

WHY THIS MODULE EXISTS.
    Every durable memory guarantee in this system fires on hangup. The model
    proposes a patch, `memory_facts` validates it against the schema and the
    transcript, facts land in the ledger. Hangup is the trigger.

    Chat has no hangup. A user messages at 9am, again at 11pm, again next
    Tuesday. Port the call design directly and consolidation never runs, so
    nothing is ever remembered.

    So a session here is a DERIVED thing: it opens on the first message and
    closes when the user goes quiet, or when it gets long enough that leaving
    it unwritten is a risk. Closing runs the SAME consolidation the call uses
    — that is deliberate. Every hard-won property (evidence grounding,
    supersession instead of overwrite, no invented paths) lives in there, and
    a second write path would reproduce all of them badly.

TWO-STAGE MEMORY.
    Per message  a cheap extraction into a pending buffer, so Mira knows what
                 you said four messages ago without waiting for anything.
    On close     the real consolidation: validate, write the ledger, refresh
                 the projection, merge open loops.

    Exactly the shape P-2 already uses (`_extracted` during, `consolidate_patch`
    after), which is why it is worth copying rather than inventing.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import uuid

# A session closes after this much silence. Long enough that stepping away to
# eat does not split one conversation in two; short enough that memory is
# written while the day is still relevant.
IDLE_CLOSE_SECONDS = int(os.getenv("CHAT_IDLE_CLOSE_SECONDS", "2700"))    # 45 min

# ...and after this many messages regardless, so a marathon thread cannot sit
# unwritten. A long chat is exactly the one whose facts you least want to lose.
MAX_MESSAGES_BEFORE_CLOSE = int(os.getenv("CHAT_MAX_MESSAGES", "60"))

# Verbatim window handed to the model. Older turns survive as a rolling
# summary; anything older than THAT should have become a fact or an open loop,
# and if it did not, that is a consolidation bug rather than a reason to carry
# more raw history.
HISTORY_WINDOW = int(os.getenv("CHAT_HISTORY_WINDOW", "20"))

# People send WhatsApp in fragments -- "hey" / "so I have a question" /
# "about my dinner". Replying to each separately is the most bot-like thing
# there is, so wait briefly for the rest of the thought.
COALESCE_SECONDS = float(os.getenv("CHAT_COALESCE_SECONDS", "2.5"))


class ChatSession:
    """One user's live conversation. Held in memory; durable state is the
    ledger, which is written on close."""

    def __init__(self, firebase_uid: str, profile=None, memory=None):
        self.firebase_uid = firebase_uid
        self.session_id = f"chat-{uuid.uuid4().hex[:12]}"
        self.profile = profile or {}
        self.memory = memory or {}
        self.started_at = time.time()
        self.last_activity = time.time()

        self.messages: list = []          # {role, text, ts}
        self.rolling_summary = ""         # compressed history behind the window
        self.pending_facts: dict = {}     # extracted, not yet consolidated
        self.threads: list = []           # thread-machine state
        self.turn_index = 0
        self.closed = False

    # ------------------------------------------------------------ messages --
    def add(self, role: str, text: str):
        self.messages.append({"role": role, "text": text, "ts": time.time()})
        self.last_activity = time.time()

    def window(self):
        """The verbatim tail the model sees, as chat-completion messages."""
        return [{"role": m["role"], "content": m["text"]}
                for m in self.messages[-HISTORY_WINDOW:]]

    def behind_window(self):
        """Messages that have fallen out of the verbatim window and now need
        to live in the rolling summary instead."""
        return self.messages[:-HISTORY_WINDOW] if len(self.messages) > HISTORY_WINDOW else []

    def transcript(self) -> str:
        """Full text, in the shape consolidation expects.

        `memory_facts` checks every proposed fact against the USER's own words
        in this transcript, so the role labels are load-bearing -- not
        decoration. Mislabel them and Mira's own words could ground a fact
        about the user.
        """
        out = []
        for m in self.messages:
            who = "User" if m["role"] == "user" else "Mira"
            out.append(f"{who}: {m['text']}")
        return "\n".join(out)

    # ---------------------------------------------------------- lifecycle --
    def idle_seconds(self) -> float:
        return time.time() - self.last_activity

    def should_close(self) -> str:
        """Returns a reason string, or "" to stay open."""
        if self.closed:
            return ""
        if self.idle_seconds() >= IDLE_CLOSE_SECONDS:
            return f"idle {int(self.idle_seconds())}s"
        if len(self.messages) >= MAX_MESSAGES_BEFORE_CLOSE:
            return f"{len(self.messages)} messages"
        return ""

    def to_dict(self):
        return {
            "session_id": self.session_id,
            "firebase_uid": self.firebase_uid,
            "started_at": self.started_at,
            "last_activity": self.last_activity,
            "messages": len(self.messages),
            "pending_facts": len(self.pending_facts),
            "threads": len(self.threads),
            "closed": self.closed,
        }


class SessionStore:
    """Live sessions, keyed by user.

    In memory on purpose: a restart loses the live window, not the memory.
    Facts already consolidated are in the ledger, and an interrupted session
    simply reopens -- the user repeats at most their last few messages, which
    is a far better failure than a half-written ledger.
    """

    def __init__(self):
        self._by_uid: dict = {}
        self._locks: dict = {}

    def lock(self, uid: str) -> asyncio.Lock:
        """One in-flight turn per user. Two messages arriving together must not
        both read the same history and both append to it."""
        if uid not in self._locks:
            self._locks[uid] = asyncio.Lock()
        return self._locks[uid]

    def get(self, uid: str):
        return self._by_uid.get(uid)

    def open(self, uid: str, profile=None, memory=None) -> ChatSession:
        s = ChatSession(uid, profile, memory)
        self._by_uid[uid] = s
        return s

    def get_or_open(self, uid: str, profile=None, memory=None):
        """Returns (session, is_new). A session that has gone quiet past the
        idle limit is retired rather than reused, so its facts get written."""
        s = self._by_uid.get(uid)
        if s and not s.closed and not s.should_close():
            return s, False
        return self.open(uid, profile, memory), True

    def drop(self, uid: str):
        self._by_uid.pop(uid, None)

    def due_for_close(self):
        return [(uid, s, r) for uid, s in list(self._by_uid.items())
                if (r := s.should_close())]

    def all(self):
        return list(self._by_uid.items())


STORE = SessionStore()
