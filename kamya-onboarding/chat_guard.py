"""Output guardrails for chat — what the prompt asks for, code guarantees.

WHY THIS EXISTS, in one observation. On 2026-09-02 the OFF_TOPIC directive
spelled out the wanted shape, with worked examples, and three live turns gave:

    "capital of france"  -> "Paris hai." + the scope offer      followed it
    "IPL ka final"        -> answer only, no offer               ignored it
    "mera phone slow"     -> five troubleshooting steps          ignored it

One in three. That is the whole case for this module: a prompt rule is a
request, and the model answers requests when it feels like it. Nine separate
rules were written into Mira's prompts this week and the ones that held were
the ones enforced in code.

DESIGN. Every check is:
  - deterministic where possible (no model call, no latency, no new failure)
  - a REWRITE or an APPEND, never a block, unless something unsafe is at stake
  - reversible in one env var, because a guardrail that over-fires is worse
    than none -- the medical deferral over-firing made Mira useless for a day

Checks that need a model, or that judge tone, do not belong here. "Is this
warm?" is the prompt's job forever.
"""
from __future__ import annotations

import os
import random
import re

GUARD_ENABLED = os.getenv("CHAT_GUARD", "1") != "0"

# ------------------------------------------------------- off-topic offer ---
# She must ANSWER, then say warmly what she is for. Naming your purpose is a
# person setting a boundary; handing over a list of topics is an IVR. These
# name the purpose and invite, without asking them to pick anything.
SCOPE_OFFERS = [
    "Waise main yahan aapki diet aur health ke liye hoon — us bare mein kuch "
    "bhi poochh sakte ho 🙂",
    "Main yahan khaane-peene aur health ke liye hoon — kuch bhi poochhna ho "
    "toh bataiye.",
    "Baaki main aapki diet aur sehat ke liye hoon — jab kuch poochna ho, "
    "bata dena.",
]

# Does the reply ALREADY name her purpose? Then leave it alone -- appending a
# second offer is the robotic repetition we are trying to avoid.
_HAS_OFFER = re.compile(
    r"(diet|health|khaane|khane|खाने|सेहत|स्वास्थ्य|nutrition)"
    r"[^.!?\n]{0,60}"
    r"(ke liye hoon|के लिए हूँ|help|madad|मदद|poochh|पूछ|bata|बता)",
    re.I)


def needs_scope_offer(reply: str) -> bool:
    return not bool(_HAS_OFFER.search(reply or ""))


def add_scope_offer(reply: str, rng=None) -> str:
    """Append the offer to an off-topic reply that lacks one."""
    reply = (reply or "").rstrip()
    if not reply:
        return reply
    pick = (rng or random).choice(SCOPE_OFFERS)
    return f"{reply}\n\n{pick}"


# ------------------------------------------------------------- markdown ----
# Chat bubbles render literally, so "**Diet tips**" reaches the user with the
# asterisks attached. Seen live when three problems arrived in one message.
_BOLD = re.compile(r"\*\*(.+?)\*\*", re.S)
_ITALIC = re.compile(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", re.S)
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+", re.M)
_BULLET = re.compile(r"^\s{0,3}[-*+]\s+", re.M)


def strip_markdown(text: str) -> str:
    """Remove markup a chat bubble cannot render.

    Bullets become "• " rather than vanishing -- a list the model chose to
    write is usually a real list, and collapsing it into prose loses the
    structure. The marker just has to be one that renders.
    """
    if not text:
        return text
    out = _HEADING.sub("", text)
    out = _BOLD.sub(r"\1", out)
    out = _ITALIC.sub(r"\1", out)
    out = _BULLET.sub("• ", out)
    return out


# ------------------------------------------------------------------ apply --
def apply(reply: str, *, situation: str = "", log=None) -> str:
    """Run every guardrail over one reply. Returns the reply to send.

    Never raises: a guardrail that breaks a conversation has failed worse than
    the thing it was checking for.
    """
    log = log or (lambda m: None)
    if not GUARD_ENABLED or not reply:
        return reply
    try:
        before = reply
        reply = strip_markdown(reply)
        if reply != before:
            log("chat guard: stripped markdown")

        if (situation or "").upper() == "OFF_TOPIC" and needs_scope_offer(reply):
            reply = add_scope_offer(reply)
            log("chat guard: added scope offer to an off-topic reply")
        return reply
    except Exception as exc:
        log(f"chat guard failed, passing reply through: "
            f"{type(exc).__name__}: {exc}")
        return reply
