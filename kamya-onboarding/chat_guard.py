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

# Used when a reply consisted ENTIRELY of something she must not say. Says
# nothing false, hands the specifics to the people who actually know, and
# leaves the conversation open.
SAFE_SUBSTITUTE = os.getenv(
    "CHAT_SAFE_SUBSTITUTE",
    "Us bare mein Kamya team aapko sahi bata payegi. "
    "Khaane ya health ka kuch poochna ho toh main yahan hoon 🙂")

# When her whole reply was a claim about an edit that has not happened yet,
# the generic substitute above is wrong -- she IS doing it, and handing the
# user to the team reads as a refusal. Say the true thing instead. The amend
# posts the actual change a minute later.
PLAN_EDIT_SUBSTITUTE = os.getenv(
    "CHAT_PLAN_EDIT_SUBSTITUTE",
    "Haan, plan update kar rahi hoon — ek minute mein bhejti hoon 👍")

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


# ---------------------------------------------------------------- menus ----
# "Kya dikkat hai — weakness, digestion, ya neend?" The single most persistent
# failure in this product: banned in the prompt three separate times, and each
# time it reappeared on whichever code path lacked the ban.
#
# Rewritten rather than blocked, and conservatively: the option list is cut and
# the question stem kept. "Kya dikkat hai?" is a real question and a strictly
# better one -- it lets them answer in their own words, which is what the
# stage machine wanted in the first place.
_MENU_TAIL = re.compile(
    r"\s*[—–\-:(]\s*"                       # the dash/colon/paren that opens it
    r"[^—–\-:()?]{0,40}?,"                   # first option, then a comma
    r"[^?()]{0,60}?"                          # more options
    r"\b(ya|या|or)\b"                        # the giveaway
    r"[^?()]{0,40}\)?"                        # last option
    r"(?=\s*\?)",                            # immediately before the "?"
    re.I)

# "(low, normal, high)" -- same failure without the dash.
_MENU_PARENS = re.compile(r"\s*\([^()?]{0,60}?,[^()?]{0,60}?\)", re.I)


def has_menu(text: str) -> bool:
    return bool(_MENU_TAIL.search(text or "") or _MENU_PARENS.search(text or ""))


def strip_menu(text: str) -> str:
    """Cut the option list, keep the question."""
    out = _MENU_TAIL.sub("", text or "")
    out = _MENU_PARENS.sub("", out)
    return re.sub(r"\s{2,}", " ", out).strip()


# -------------------------------------------------------------- promises ---
# "hum aapka personalized plan banayenge. kal ya parso message aayega."
# She cannot commit the team to anything, and a written promise is screenshot
# and held against you weeks later.
_PROMISE = re.compile(
    r"[^.!?\n]*\b("
    r"kal\s+(tak|se)?\s*(bhej|mil|aa)|parso|kal\s+message|"
    r"bhej\s*(dungi|doongi|denge|dunga)|"
    r"plan\s+(bana\s*denge|ready\s+ho|bhej)|"
    r"promise\s+karti|guarantee|"
    # "Team aapko call karegi" slipped through for want of a gap here: the
    # pattern demanded team and call be adjacent.
    r"team\s+[^.!?\n]{0,24}?(call|contact|message|reply)\s*kar"
    r")[^.!?\n]*[.!?]?", re.I)


# Sending the diet plan stopped being a promise the day she could actually do
# it. "Plan bhej rahi hoon" matched `plan\s+bhej` and was deleted outright,
# leaving her confirmation missing while the PDF arrived anyway.
_IS_PLAN = re.compile(r"\b(diet\s*plan|plan|pdf|chart)\b", re.I)
# SENDING an existing plan is the exempt act. MAKING one is still a promise:
# "hum plan bana denge" commits the team to work, and she cannot do that.
_IS_SENDING = re.compile(
    r"\b(bhej|send|share|attach|de\s+rahi|bhejti|bheja)\b", re.I)
# ...but only without a date on it. "Kal plan bhej dungi" is still a promise,
# because tomorrow is not something she controls.
_HAS_DATE = re.compile(
    r"\b(kal|parso|aaj\s+raat|baad\s+mein|thodi\s+der|kuch\s+din|"
    r"hafte|week|monday|somvar|shaam\s+tak|subah\s+tak)\b", re.I)


def strip_promises(text: str):
    """Remove any sentence that commits to a deliverable or a date.

    Exempts the one deliverable she genuinely controls: the plan PDF, which
    the server attaches to the conversation for her.
    """
    def _keep_or_cut(m):
        sent = m.group(0)
        if (_IS_PLAN.search(sent) and _IS_SENDING.search(sent)
                and not _HAS_DATE.search(sent)):
            return sent
        return ""

    out, n = _PROMISE.subn(_keep_or_cut, text or "")
    # subn counts matches, not removals -- an exempted sentence is not a cut.
    n = sum(1 for m in _PROMISE.finditer(text or "") if not _keep_or_cut(m))
    return re.sub(r"\s{2,}", " ", out).strip(), n


# ------------------------------------------------------ plan edit claims ---
# Her reply is written BEFORE the plan is amended, so anything specific she
# says about the edit is a guess. A live run: the user asked to remove lauki
# from Monday, she answered "Haan, hata di. Aloo rakh diya hai Monday dinner
# mein" -- and the actual amend put zucchini in Monday LUNCH. Two models
# narrating the same change, disagreeing on the food, the meal and the tense.
#
# Only one of them knows: the amend, which posts its own message afterwards.
# So her completion CLAIMS are removed here and the rest of her reply stands.
# Past tense and first person only -- "roti hata dijiye" is advice, not a
# claim, and must survive.
_EDIT_CLAIM = re.compile(
    r"[^.!?\n]*\b("
    r"hata\s*(di|diya|diye)|nikal\s*(di|diya)|"
    r"(rakh|daal|dal|add\s*kar|replace\s*kar|swap\s*kar)\s*(di|diya|diye)|"
    r"(change|update|badal)\s*(kar\s*)?(di|diya|diye)"
    r")\b[^.!?\n]*[.!?]?", re.I)


def strip_plan_edit_claims(text: str):
    """Drop sentences claiming a plan edit is already done.

    The edit takes a minute and happens after this reply. Saying it is done,
    and naming a food she has not chosen, is wrong twice over.
    """
    out, n = _EDIT_CLAIM.subn("", text or "")
    return re.sub(r"\s{2,}", " ", out).strip(), n


# ---------------------------------------------------------------- gender ---
# Mira is a woman; the user may not be. Both directions have failed live:
# "samajh sakta hoon" (her, masculine) and "skip kar rahi ho" (to a man).
_HER_MASC = [
    (re.compile(r"\b(sakta)\s+(hoon|hun)\b", re.I), r"sakti \2"),
    (re.compile(r"\b(samajhta|karta|dekhta|bolta|deta|leta|rehta)\s+(hoon|hun)\b", re.I),
     lambda m: m.group(1)[:-1] + "i " + m.group(2)),
    (re.compile(r"\b(karunga|poochhunga|bataunga|dekhunga|hoonga)\b", re.I),
     lambda m: m.group(1).replace("unga", "ungi").replace("oonga", "oongi")),
    (re.compile(r"(सकता|समझता|करता|देखता|रहता)(\s+हूँ)"),
     lambda m: m.group(1)[:-1] + "ी" + m.group(2)),
]
# Feminine verbs aimed at the USER. Only corrected when we know they are male.
_YOU_FEM = re.compile(r"\b(rahi|karti|khati|leti|soti|hoti)\s+ho\b", re.I)
_FEM_TO_MASC = {"rahi": "rahe", "karti": "karte", "khati": "khate",
                "leti": "lete", "soti": "sote", "hoti": "hote"}


def fix_gender(text: str, user_gender: str = "") -> tuple:
    """Her verbs feminine; the user's matching THEIR gender."""
    out, n = text or "", 0
    for pat, rep in _HER_MASC:
        out, k = pat.subn(rep, out)
        n += k
    if (user_gender or "").strip().lower() in ("male", "m", "man", "पुरुष"):
        def _m(mo):
            return _FEM_TO_MASC.get(mo.group(1).lower(), mo.group(1)) + " ho"
        out, k = _YOU_FEM.subn(_m, out)
        n += k
    return out, n


# ------------------------------------------------------------ banned words --
# "समझी"/"samjhi" is the most repetitive-sounding habit on a call and slipped
# past the prompt twice in one conversation.
_SAMJHI = re.compile(r"^\s*(जी\s*)?(समझ\s*गई|समझी|समझा|samajh\s*gayi|samjhi)\s*[.!,]?\s*",
                     re.I)
# The prompt allows at most one emoji; more reads as trying too hard.
_EMOJI = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF]")


def trim_openers(text: str) -> str:
    return _SAMJHI.sub("", text or "", count=1).strip() or (text or "")


def cap_emoji(text: str, limit: int = 1) -> str:
    found = _EMOJI.findall(text or "")
    if len(found) <= limit:
        return text
    out, kept = [], 0
    for ch in text:
        if _EMOJI.match(ch):
            kept += 1
            if kept > limit:
                continue
        out.append(ch)
    return "".join(out)


# ------------------------------------------------------- Kamya invention ---
# "Kamya team ke behind mein actual nutritionists hain" -- a claim about the
# business she has no way of knowing and which may simply be false.
_KAMYA_CLAIM = re.compile(
    r"[^.!?\n]*\bkamya\b[^.!?\n]*\b("
    r"nutritionists?|doctors?|dietici|experts?|team\s+mein\s+\d|"
    r"certified|qualified|years?\s+of\s+experience"
    r")[^.!?\n]*[.!?]?", re.I)


def strip_kamya_claims(text: str):
    out, n = _KAMYA_CLAIM.subn("", text or "")
    return re.sub(r"\s{2,}", " ", out).strip(), n


# --------------------------------------------------------------- counters --
# COUNT, never block. These are quality signals: a rising rate means the
# PROMPT is slipping and the fix belongs there, not in a harder rewrite.
_HEDGE = re.compile(
    r"(kya agar|agar aap.{0,40}toh.{0,20}(sakta|sakti|karega|hoga)|"
    r"madad karega\?|kaam karega\?|theek rahega\?|would that help)", re.I)


def quality_flags(text: str, budget: int = 0) -> dict:
    words = len((text or "").split())
    return {
        "words": words,
        "over_budget": bool(budget) and words > budget * 1.6,
        "hedged": bool(_HEDGE.search(text or "")),
        "questions": (text or "").count("?"),
    }


# ------------------------------------------------------------------ apply --
def apply(reply: str, *, situation: str = "", user_gender: str = "",
          budget: int = 0, log=None) -> str:
    """Run every guardrail over one reply. Returns the reply to send.

    Never raises: a guardrail that breaks a conversation has failed worse than
    the thing it was checking for.
    """
    log = log or (lambda m: None)
    if not GUARD_ENABLED or not reply:
        return reply
    original = reply
    try:
        sit = (situation or "").upper()
        hits = []

        # ---- REWRITE: wrong, but not unsafe -------------------------------
        step = strip_markdown(reply)
        if step != reply:
            hits.append("markdown")
        reply = step

        if has_menu(reply):
            stripped = strip_menu(reply)
            # Only accept the rewrite if a real question survives it.
            if len(stripped) > 12 and "?" in stripped:
                reply, _ = stripped, hits.append("menu")

        # These two remove things she must not say. If removing them empties
        # the reply, the reply was ENTIRELY the thing we are removing -- so
        # restoring the original would ship exactly what the check exists to
        # stop. Substitute instead.
        unsafe_removed = False
        plan_edit_removed = False

        reply, n = strip_promises(reply)
        if n:
            hits.append(f"promise x{n}")
            unsafe_removed = True

        reply, n = strip_plan_edit_claims(reply)
        plan_edit_removed = bool(n)
        if n:
            hits.append(f"plan edit claim x{n}")
            unsafe_removed = True

        reply, n = fix_gender(reply, user_gender)
        if n:
            hits.append(f"gender x{n}")

        reply, n = strip_kamya_claims(reply)
        if n:
            hits.append("kamya-claim")
            unsafe_removed = True

        step = trim_openers(reply)
        if step != reply:
            hits.append("samjhi")
        reply = step

        step = cap_emoji(reply)
        if step != reply:
            hits.append("emoji")
        reply = step

        # ---- APPEND: off-topic must say what she is for --------------------
        if sit == "OFF_TOPIC" and needs_scope_offer(reply):
            reply = add_scope_offer(reply)
            hits.append("scope-offer")

        # ---- BLOCK / SUBSTITUTE: a rewrite that ate the whole reply -------
        if not reply.strip():
            if plan_edit_removed:
                log("chat guard: reply was ENTIRELY a claim about an edit "
                    "that has not run yet -- substituting")
                return PLAN_EDIT_SUBSTITUTE
            if unsafe_removed:
                log("chat guard: reply was ENTIRELY a promise or an invented "
                    "claim -- substituting")
                return SAFE_SUBSTITUTE
            log("chat guard: rewrites emptied the reply -- keeping the original")
            return original

        # ---- COUNT: quality signals, logged only ---------------------------
        flags = quality_flags(reply, budget)
        noted = [k for k in ("over_budget", "hedged") if flags.get(k)]
        if hits:
            log(f"chat guard: {', '.join(hits)}")
        if noted:
            log(f"chat quality: {', '.join(noted)} ({flags['words']}w)")
        return reply
    except Exception as exc:
        # A guardrail that breaks the conversation has failed worse than the
        # thing it was checking for.
        log(f"chat guard failed, passing reply through: "
            f"{type(exc).__name__}: {exc}")
        return original
