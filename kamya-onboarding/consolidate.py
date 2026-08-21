"""End-of-session memory consolidation — PATCH based.

Runs ONCE when a call ends, off the audio path.

The model does NOT write memory. It proposes a PATCH: a list of set / append /
invalidate operations, each carrying the user's own words as evidence.
`memory_facts` then validates every operation against a fixed schema and
applies it in code.

Why it changed: the model used to return the whole 12-section document and we
stored whatever came back. A section it forgot to copy through was silently
lost, and a hallucinated fact became next call's ground truth with no way to
trace it. Emitting a patch makes both impossible — an op naming an unknown
path, or citing a quote absent from the transcript, is rejected and recorded
rather than absorbed.

Open loops are still returned whole (a short list of strings, cheap to
regenerate and not part of the fact ledger).

Uses GROQ_CONSOLIDATION_MODEL so this background step doesn't eat the
conversation model's token budget.
"""
import json
import os

from groq import Groq

import memory_facts

# llama-3.1-8b-instant was decommissioned 2026-08-16, and its drop-in
# replacement (gpt-oss-20b) fails this task with json_validate_failed and an
# empty generation. The 120b follows a strict schema reliably. This runs once,
# after hangup — latency doesn't matter here, correctness does.
_MODEL = os.getenv("GROQ_CONSOLIDATION_MODEL", "openai/gpt-oss-120b")

_PATHS_BLOCK = "\n".join(
    f"  {p:44s} {t}" for p, t in sorted(memory_facts.SCHEMA.items())
)

_SYSTEM = f"""\
You maintain the long-term memory of a user for Mira, an AI dietician for Indian users.

You are given the user's CURRENT MEMORY, their EXISTING OPEN LOOPS, and the
TRANSCRIPT of the call that just ended.

You do NOT rewrite memory. You propose a PATCH describing only what CHANGED.

OUTPUT — valid JSON, exactly this shape:
{{
  "ops": [
    {{"op": "set",        "path": "<path>", "value": <value>, "evidence": "<user's words>", "confidence": "high|medium|low"}},
    {{"op": "append",     "path": "<path>", "value": <value>, "evidence": "<user's words>", "confidence": "high|medium|low"}},
    {{"op": "invalidate", "path": "<path>", "reason": "<why it is no longer true>", "evidence": "<user's words>"}}
  ],
  "session_summary": "1-2 sentences on what happened this call",
  "open_loops": ["follow-up for next call", "..."]
}}

OPERATIONS
  set         Replace a single-valued field. Use when a value CHANGED or is new.
  append      Add one item to a list field. Use for a NEW item only.
  invalidate  The user contradicted something already in memory. The old value
              is preserved and closed off — never removed.

HARD RULES
  1. Emit an op ONLY for something this transcript establishes. If a fact is
     already in CURRENT MEMORY and unchanged, emit NOTHING for it. Silence
     means "still true" — memory is cumulative and nothing is lost by omission.
  2. `evidence` MUST quote or closely paraphrase what the USER actually said.
     It is checked against the transcript. If you cannot ground it, omit the op.
  3. `path` MUST be one of the paths listed below. Never invent a path.
  4. Durable facts only. "I ate poha today" is an event, not memory. "I eat
     poha most mornings" is memory.
  5. Never infer medical facts. A condition, medication or allergy needs the
     user to have stated it.
  6. If the user contradicts a stored fact, use `invalidate` (or `set` for a
     single-valued field), never a silent overwrite.
  7. An empty "ops" list is a perfectly good answer for a short call.

CONFIDENCE
  high    the user stated it plainly
  medium  clearly implied
  low     uncertain — still emit it, we track the weakness

ALLOWED PATHS (name, then kind)
{_PATHS_BLOCK}

  scalar   one value       -> use `set`
  list     list of strings -> use `append` (one op per new item)
  objlist  list of objects -> use `append`
           health.medications items: {{"name","dosage","timing","frequency"}}
           entities items:           {{"type","what","status","given_on"}}
           entities.status is one of: suggested | trying | following | absorbed

MEAL SLOTS under current_pattern: morning, mid_morning, lunch, evening,
dinner, late_night. `.time` and `.note` are scalars; `.frequent` and `.gaps`
are lists.

OPEN LOOPS — return the FULL list, merged: drop the ones this call resolved,
keep the ones still outstanding (even if not discussed), add anything new.
Never regenerate from scratch."""


class ConsolidationError(RuntimeError):
    """The model's response was unusable. Carries the reason for a repair retry."""


def _validate_patch(data) -> dict:
    """Shape-check the patch envelope. Per-op validation lives in memory_facts."""
    if not isinstance(data, dict):
        raise ConsolidationError("top level is not a JSON object")

    ops = data.get("ops", [])
    if ops is None:
        ops = []
    if not isinstance(ops, list):
        raise ConsolidationError("`ops` must be a list")
    if not all(isinstance(o, dict) for o in ops):
        raise ConsolidationError("every entry in `ops` must be an object")

    loops = data.get("open_loops", [])
    if loops is None:
        loops = []
    if not isinstance(loops, list):
        raise ConsolidationError("`open_loops` must be a list of strings")

    summary = data.get("session_summary", "") or ""
    if not isinstance(summary, str):
        raise ConsolidationError("`session_summary` must be a string")

    return {
        "ops": ops,
        "session_summary": summary.strip(),
        "open_loops": [str(x) for x in loops if str(x).strip()],
    }


def consolidate_patch(current_view: dict, existing_open_loops: list,
                      transcript: str, attempts: int = 3) -> dict:
    """Ask the model for a PATCH. Returns {ops, session_summary, open_loops}.

    Retries with the rejection reason fed back, because the observed failures
    are recoverable (empty generation, malformed JSON). Raises
    ConsolidationError once attempts are exhausted — the caller keeps the
    stored transcript, so the session stays replayable.
    """
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    base_msg = (
        "CURRENT MEMORY:\n"
        + json.dumps(current_view or {}, ensure_ascii=False)
        + "\n\nEXISTING OPEN LOOPS:\n"
        + json.dumps(existing_open_loops or [], ensure_ascii=False)
        + "\n\nTRANSCRIPT:\n"
        + (transcript or "").strip()
    )

    last_error = ""
    for attempt in range(1, max(1, attempts) + 1):
        user_msg = base_msg
        if last_error:
            user_msg += (
                f"\n\nYOUR PREVIOUS ATTEMPT WAS REJECTED: {last_error}\n"
                "Return ONLY valid JSON matching the schema exactly."
            )
        try:
            resp = client.chat.completions.create(
                model=_MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                # Nudge off a deterministic bad path on retries.
                temperature=0.2 if attempt == 1 else 0.4,
                response_format={"type": "json_object"},
            )
            content = (resp.choices[0].message.content or "").strip()
            if not content:
                raise ConsolidationError("model returned an empty response")
            return _validate_patch(json.loads(content))
        except Exception as exc:  # API errors, JSON errors, validation errors
            last_error = f"{type(exc).__name__}: {exc}"[:400]
            print(f"[consolidate] attempt {attempt}/{attempts} failed: {last_error}",
                  flush=True)

    raise ConsolidationError(f"all {attempts} attempts failed. last error: {last_error}")
