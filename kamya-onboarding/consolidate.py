"""End-of-session memory consolidation (Step-3 architecture).

Runs ONCE when a call ends, off the audio path. Takes the user's EXISTING
long-term memory plus the transcript of the call that just happened, and asks a
Groq model to return the UPDATED (cumulative) memory + a session summary +
open loops to follow up on next time.

Cumulative rule: the model is told to keep everything still true, add new
durable facts, update changed ones, and drop only what was contradicted.

Open loops are MERGED (not regenerated): the existing open loops are passed in,
and the model removes only the ones this conversation resolved, keeps the rest
(even if not discussed this session), and adds any new follow-ups.

Uses a small/cheap Groq model by default (GROQ_CONSOLIDATION_MODEL) so this
background step doesn't eat the conversation model's daily token budget.
"""
import json
import os

from groq import Groq

_MODEL = os.getenv("GROQ_CONSOLIDATION_MODEL", "llama-3.1-8b-instant")

_SYSTEM = """You maintain the long-term memory of a user for an AI dietician assistant named Mira.

You are given: the user's EXISTING long-term memory (JSON), the EXISTING open loops (things Mira was planning to follow up on), and the TRANSCRIPT of the conversation that just happened.

Produce the UPDATED memory.

long_term_memory rules:
- CUMULATIVE: keep everything in the existing memory that is still true.
- ADD new durable facts learned in this conversation.
- UPDATE facts that changed, and DROP only facts the user explicitly contradicted (e.g. "I used to dislike oats but I'm fine with them now" -> remove the dislike).
- Ignore small talk, greetings, and anything not durably useful.
- Keep entries as short phrases, not sentences.

open_loops rules (VERY IMPORTANT — start from the EXISTING open loops, do not regenerate from scratch):
- REMOVE an existing open loop ONLY if this conversation clearly RESOLVED or answered it. Example: existing loop "Ask whether the poha worked" and in the transcript the user says they tried the poha -> it is resolved, remove it.
- KEEP every existing open loop that is still unresolved — even if it was NOT mentioned in this conversation. Never drop a loop just because it did not come up this time.
- ADD new follow-ups discovered in this conversation.
- Do NOT duplicate loops that mean the same thing.
- Return the FULL updated open_loops list = (unresolved existing loops that were kept) + (new loops added), with resolved ones removed.

Return STRICT JSON only (no prose, no markdown), exactly this shape:
{
  "long_term_memory": {"facts": [], "goals": [], "preferences": [], "dislikes": []},
  "session_summary": "one or two sentence recap of THIS conversation",
  "open_loops": ["concrete things Mira should follow up on next time"]
}"""


def consolidate(existing_memory: dict, existing_open_loops: list, transcript: str) -> dict:
    """Return {long_term_memory, session_summary, open_loops}. Raises on failure.

    `existing_open_loops` is the user's current open loops; the model merges them
    with this conversation (removes resolved, keeps unresolved, adds new).
    """
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    user_msg = (
        "EXISTING MEMORY:\n"
        + json.dumps(existing_memory or {}, ensure_ascii=False)
        + "\n\nEXISTING OPEN LOOPS:\n"
        + json.dumps(existing_open_loops or [], ensure_ascii=False)
        + "\n\nTRANSCRIPT:\n"
        + (transcript or "").strip()
    )
    resp = client.chat.completions.create(
        model=_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    data = json.loads(resp.choices[0].message.content)
    return {
        "long_term_memory": data.get("long_term_memory", {}) or {},
        "session_summary": (data.get("session_summary", "") or "").strip(),
        "open_loops": data.get("open_loops", []) or [],
    }
