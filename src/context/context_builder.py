"""Context builder — the assembly hub.

This is the ONLY place where information from different sources (system prompt,
user profile, conversation history, food data) is combined into the final
message list for the LLM. It does not talk to OpenRouter; it only prepares the
context. It uses the data-access layer for all sheet reads.

Final context shape (OpenAI/OpenRouter chat format):

    [ system: <system_prompt.md> ]
    [ system: USER PROFILE ... ]
    [ system: RELEVANT FOOD DATA ...  (omitted if no match) ]
    [ ...recent history as user/assistant turns, chronological... ]
    [ user: <current message> ]
"""
from __future__ import annotations

import json

from src import config
from src.context import food_matcher, prompt_loader
from src.data import repositories

# Language steering for the voice demo. Kept here (not in system_prompt.md) so
# the prompt file stays language-neutral. Tells the model to mirror the user's
# language, including Hindi / Hinglish, and to stay speech-friendly.
_LANGUAGE_GUIDANCE = (
    "LANGUAGE: Reply in the SAME language the user speaks. If they use Hindi, "
    "reply in natural Hindi; if they mix Hindi and English (Hinglish), reply in "
    "the same casual mix. Keep replies short and natural for speaking aloud."
)

# Profile fields surfaced to the model, in a sensible order. Matches the Users
# sheet schema. Safety-critical fields (allergies, conditions) are always
# included. Labels are the human-friendly names shown to the model.
_PROFILE_FIELDS = [
    ("name", "name"),
    ("age", "age"),
    ("sex", "sex"),
    ("height_cm", "height (cm)"),
    ("weight_kg", "weight (kg)"),
    ("diet", "diet"),
    ("allergies", "allergies"),
    ("conditions", "health conditions"),
]

# Fields stored as JSON arrays in the sheet.
_LIST_FIELDS = {"allergies", "conditions"}


def _readable_value(field: str, raw: str) -> str:
    """Turn a stored cell into readable text; JSON arrays -> comma list."""
    raw = str(raw).strip()
    if field in _LIST_FIELDS:
        try:
            items = json.loads(raw)
            if isinstance(items, list):
                return ", ".join(str(x) for x in items) if items else "none"
        except (ValueError, TypeError):
            pass
    return raw


def _format_profile(user: dict | None) -> str:
    """Render the authenticated user's profile as a compact key: value block."""
    if not user:
        return "USER PROFILE:\n(No profile found for this user.)"
    lines = ["USER PROFILE:"]
    for field, label in _PROFILE_FIELDS:
        value = _readable_value(field, user.get(field, ""))
        if value:
            lines.append(f"- {label}: {value}")
    return "\n".join(lines)


def _format_food_data(rows: list[dict]) -> str | None:
    """Render matched food rows as a readable block, or None if no matches."""
    if not rows:
        return None
    lines = [
        "RELEVANT FOOD DATA (per 100g unless noted). Prefer these values; "
        "do not invent numbers for foods not listed here:"
    ]
    for r in rows:
        lines.append(
            f"- {r.get('food')}: {r.get('calories_per_100g')} kcal, "
            f"protein {r.get('protein_g')}g, carbs {r.get('carbs_g')}g, "
            f"fat {r.get('fat_g')}g"
            + (f" (category: {r.get('category')})" if r.get("category") else "")
        )
    return "\n".join(lines)


def _map_history(history: list[dict]) -> list[dict]:
    """Convert conversation_history rows into chat messages.

    Sheet roles are normalized to 'user'/'assistant'. Unknown roles default to
    'user' so nothing is silently dropped.
    """
    messages = []
    for row in history:
        role = str(row.get("role", "")).strip().lower()
        if role not in ("user", "assistant"):
            role = "user"
        content = str(row.get("message", "")).strip()
        if content:
            messages.append({"role": role, "content": content})
    return messages


def build_context(user_id: str, user_message: str) -> list[dict]:
    """Assemble the full chat message list for one user turn."""
    # 1. System prompt (loaded fresh from disk each turn).
    system_prompt = prompt_loader.load_system_prompt()

    # 2. User profile.
    user = repositories.get_user(user_id)

    # 3. Recent conversation history (chronological).
    history = repositories.get_recent_history(user_id, config.HISTORY_LIMIT)

    # 4. Relevant food data from the current message only.
    food_rows = repositories.get_food_data()
    relevant_foods = food_matcher.find_relevant_foods(user_message, food_rows)

    # 5. Assemble.
    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    messages.append({"role": "system", "content": _LANGUAGE_GUIDANCE})
    messages.append({"role": "system", "content": _format_profile(user)})

    food_block = _format_food_data(relevant_foods)
    if food_block:
        messages.append({"role": "system", "content": food_block})

    messages.extend(_map_history(history))
    messages.append({"role": "user", "content": user_message})

    return messages
