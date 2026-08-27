"""Decide WHETHER to retrieve, and WHAT to retrieve for.

Two problems this fixes, both observed in real call logs:

  rag matched 3 q='Yeah.'         <- retrieval on a backchannel
  rag matched 3 q='Slow slow.'    <- retrieval on an instruction to Mira
  rag matched 3 q='अच्छा और कुछ?'  <- meaningless without the previous turn

The first two are noise: three irrelevant dietician Q&As get injected into the
prompt because the user grunted. The third is a real question whose meaning
lives in the previous turn, so searching the raw words finds nothing useful.

Deliberately LLM-free. This runs on the audio path, once per user turn, so an
extra model call would add hundreds of milliseconds to every reply. Last-turn
concatenation gets most of the benefit of a rewriter at zero latency.
"""
import re

# Pure backchannel / acknowledgement. An utterance made only of these carries
# no retrievable intent. Romanised Hinglish and Devanagari both appear in
# Deepgram output, so both are listed.
FILLER = {
    # acknowledgement
    "haan", "han", "haa", "ha", "hn", "hmm", "hm", "hmmm", "mmm", "mm",
    "ok", "okay", "okey", "k", "achha", "accha", "acha", "achchha",
    "thik", "theek", "teek", "ji", "yes", "yeah", "yep", "ya", "yup",
    "right", "sure", "sahi", "correct", "exactly", "bilkul", "bas",
    "अच्छा", "ठीक", "जी", "हाँ", "हां", "सही", "बिलकुल", "बस",
    "हम्म", "हम्म्म", "हम", "हँ", "म्म",
    # negation-only
    "no", "nahi", "nahin", "na", "नहीं", "ना",
    # hesitation / discourse
    "arre", "are", "oh", "ohh", "ah", "uh", "um", "umm", "err", "hain",
    "matlab", "waise", "अरे", "मतलब", "वैसे",
    # greetings & politeness
    "hello", "hi", "hey", "please", "thanks", "thank", "dhanyavad",
    "sorry", "suno", "सुनो", "नमस्ते", "namaste",
    # instructions to Mira, not dietician questions
    "slow", "slowly", "dheere", "धीरे", "wait", "ruko", "रुको",
    "repeat", "phir", "again", "dubara", "दुबारा",
}

# Function words. They do not count as "content", but an utterance containing
# them is not automatically filler either.
_STOP = FILLER | {
    "main", "mai", "mein", "me", "hum", "tum", "aap", "tu", "toh", "to",
    "hai", "hain", "hoon", "hu", "tha", "thi", "the", "ho", "hota", "hoti",
    "ka", "ki", "ke", "ko", "se", "par", "pe", "aur", "ya", "bhi", "hi",
    "kya", "kaise", "kaisa", "kitna", "kab", "kahan", "kaun", "kyun", "kyu",
    "a", "an", "the", "is", "am", "are", "was", "were", "be", "do", "does",
    "i", "you", "it", "and", "or", "of", "in", "on", "for", "with", "my",
    "मैं", "मुझे", "आप", "है", "हैं", "हूँ", "हूं", "का", "की", "के", "को",
    "से", "पर", "और", "भी", "क्या", "कैसे", "कितना", "कब", "कहाँ", "क्यों",
}

# Referring expressions. Their presence means the utterance points at
# something said earlier, so the query needs the previous turn to make sense.
_REFERRING = {
    "aur", "और", "kuch", "कुछ", "ye", "yeh", "ये", "isse", "iska", "isme",
    "isko", "wo", "woh", "वो", "usse", "uska", "usme", "usko", "waisa",
    "aisa", "ऐसा", "वैसा", "same", "that", "this", "it", "those", "these",
    "else", "more", "other", "another", "instead", "iski", "uski",
}

MIN_CHARS = 3          # anything shorter cannot be a question
MAX_QUERY_CHARS = 400  # long queries dilute the embedding
_CONTEXT_WORD_FLOOR = 3  # <= this many content words -> pull in the last turn


def _tokens(text):
    return [t for t in re.split(r"[^\wऀ-ॿ]+", (text or "").lower()) if t]


def content_words(text):
    """Tokens that actually carry retrievable meaning."""
    return [t for t in _tokens(text) if t not in _STOP]


# Devanagari hums get written every which way -- हम्म, हम्म्म, हँ, म्म -- and
# enumerating the spellings is a losing game. Any token built only from
# ह / म / ँ / ं / ् / ा is a hum, never a question.
_HUM_RE = re.compile(r"^[\u0939\u092e\u0901\u0902\u094d\u093e]+$")


def _is_filler_token(tok):
    return tok in FILLER or bool(_HUM_RE.match(tok))


def is_retrievable(text):
    """Return (ok, reason). Cheap gate — no model call, no network.

    Reason is logged so a skip is always explainable.
    """
    raw = (text or "").strip()
    if len(raw) < MIN_CHARS:
        return False, "too short"

    toks = _tokens(raw)
    if not toks:
        return False, "no words"
    if all(_is_filler_token(t) for t in toks):
        return False, "filler only"
    if not content_words(raw):
        return False, "no content words"
    return True, ""


def needs_context(text):
    """True when the utterance leans on the previous turn to mean anything."""
    toks = _tokens(text)
    if any(t in _REFERRING for t in toks):
        return True
    return len(content_words(text)) <= _CONTEXT_WORD_FLOOR


def _is_filler_turn(text):
    toks = _tokens(text)
    return not toks or all(_is_filler_token(t) for t in toks)


def recent_turns(messages, limit=2, scan=12):
    """Last few SUBSTANTIVE turns as (role, text), oldest first.

    Skips the system prompt, injected reference blocks, and filler turns.
    Reaching past "Yeah." / "Achha." matters: those carry no topic, and
    padding the query with them is how a context rewrite ends up worse than
    no rewrite at all. `scan` bounds how far back we look.
    """
    out = []
    for m in reversed((messages or [])[-scan:]):
        role = m.get("role")
        if role not in ("user", "assistant"):
            continue
        text = m.get("content")
        if not isinstance(text, str) or not text.strip():
            continue
        text = text.strip()
        if _is_filler_turn(text):
            continue
        out.append((role, text))
        if len(out) >= limit:
            break
    return list(reversed(out))


def build_query(text, messages=None, limit=2):
    """Return (query, strategy) for retrieval.

    strategy is "direct" when the utterance stands on its own, or
    "contextual" when the previous turn was folded in to resolve it. Mira's
    own last turn is the useful half: it names the topic the user is replying
    to, which is exactly what "और कुछ?" is missing.
    """
    raw = (text or "").strip()
    if not needs_context(raw):
        return raw[:MAX_QUERY_CHARS], "direct"

    turns = recent_turns(messages or [], limit=limit)
    if not turns:
        return raw[:MAX_QUERY_CHARS], "direct"

    parts = []
    for role, txt in turns:
        tag = "Mira asked" if role == "assistant" else "User said"
        parts.append(f"{tag}: {txt[:200]}")
    parts.append(f"User now: {raw[:200]}")
    return " | ".join(parts)[:MAX_QUERY_CHARS], "contextual"
