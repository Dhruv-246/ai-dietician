"""Simple relevant-food lookup.

Prototype-level keyword matching: find food_data rows whose `food` name
appears in the user's message. No embeddings, no vector search. Only matched
rows are surfaced to the LLM so we never dump the whole table.
"""
import re


def _tokenize(text: str) -> set[str]:
    """Lowercase word tokens from a string."""
    return set(re.findall(r"[a-z]+", text.lower()))


def _singularize(word: str) -> str:
    """Very small plural -> singular helper (rotis -> roti, eggs -> egg)."""
    if len(word) > 3 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("es"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s"):
        return word[:-1]
    return word


def find_relevant_foods(message: str, food_rows: list[dict]) -> list[dict]:
    """Return food_data rows relevant to the message.

    Matching strategy (case-insensitive):
      1. Whole food name appears as a substring of the message, OR
      2. Any word of the food name matches a message token
         (with light singular/plural normalization).
    Returns an empty list when nothing matches (the LLM is then instructed by
    the system prompt not to invent nutrition values).
    """
    if not message or not food_rows:
        return []

    msg_lower = message.lower()
    msg_tokens = {_singularize(t) for t in _tokenize(message)}

    matches = []
    for row in food_rows:
        food_name = str(row.get("food", "")).strip()
        if not food_name:
            continue
        name_lower = food_name.lower()

        # 1. Direct substring match (handles multi-word foods).
        if name_lower in msg_lower:
            matches.append(row)
            continue

        # 2. Token overlap with singular/plural normalization.
        name_tokens = {_singularize(t) for t in _tokenize(food_name)}
        if name_tokens & msg_tokens:
            matches.append(row)

    return matches
