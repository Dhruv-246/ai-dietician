"""Long-term memory extractor.

Given the most recent user/assistant exchange, ask the LLM to pull out STABLE
facts about the user worth remembering across future conversations. Returns a
list of {category, key, value}. Uses the same OpenRouter client as chat.
"""
from __future__ import annotations

import json
import re

from src.llm import openrouter_client

_SYSTEM = """You extract STABLE, long-term facts about the USER from a chat with an AI dietician.

Store ONLY durable facts that stay useful in future conversations, such as:
- food preferences and dislikes
- long-term goals, target weight, motivation
- routines: wake-up time, meal timings, exercise routine, work schedule (e.g. night shifts)
- lifestyle, chronic habits, dietary pattern (e.g. intermittent fasting)
- family context

Do NOT store:
- temporary/one-off events ("I ate pizza today", "I'm hungry now")
- the assistant's suggestions or advice (only facts the USER stated about themselves)
- greetings, small talk, or anything not durable

Return a JSON array. Each item: {"category": <short>, "key": <stable snake_case id>, "value": <concise third-person fact>}.
- "key" must be a canonical snake_case identifier so re-learning the SAME fact reuses it
  (e.g. target_weight, wake_up_time, work_schedule, exercise_routine, dislikes_mushrooms,
  diet_pattern, meal_timing, motivation, family_context).
- "value" is a short natural sentence, e.g. "User's target weight is 65 kg",
  "User dislikes mushrooms", "User works night shifts".
If there is nothing durable to store, return exactly [].
Return ONLY the JSON array — no prose, no code fences."""


def _parse(raw: str) -> list[dict]:
    """Extract a JSON array of memory items from the model output, robustly."""
    if not raw:
        return []
    match = re.search(r"\[.*\]", raw.strip(), re.S)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except (ValueError, TypeError):
        return []
    out = []
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            key = str(item.get("key", "")).strip()
            value = str(item.get("value", "")).strip()
            category = str(item.get("category", "")).strip() or "general"
            if key and value:
                out.append({"category": category, "key": key, "value": value})
    return out


def extract(exchange_text: str, existing_keys: list[str] | None = None) -> list[dict]:
    """Return stable memory items from the latest exchange (may be empty)."""
    if not exchange_text.strip():
        return []
    known = ", ".join(existing_keys or []) or "(none yet)"
    user_msg = (
        f"Keys already stored for this user: {known}\n\n"
        f"LATEST EXCHANGE:\n{exchange_text}\n\n"
        "Return the JSON array of durable facts (or [])."
    )
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": user_msg},
    ]
    raw = openrouter_client.chat(messages)
    return _parse(raw)
