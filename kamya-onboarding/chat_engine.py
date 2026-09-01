"""P-3 chat turn handler.

Reuses the voice brain deliberately. `p3_graph` imports no pipecat -- safety
check, router, five lanes, and the UNDERSTAND -> GATHER -> REFLECT -> ADVISE
-> CONFIRM -> CLOSE stage machine are all medium-independent, because they
encode how a dietician should think rather than anything about audio. Forking
them for chat would mean maintaining two copies of the one part that is
genuinely hard to get right.

What IS medium-specific lives here:
  - the prompt (chat_prompt.md, not the voice one -- see WHY below)
  - length budgets per message type, replacing the voice word cap
  - splitting a reply into WhatsApp-style bubbles
  - per-message fact extraction, since chat has no hangup to consolidate on

WHY NOT REUSE ask_prompt.md.
    Its second line is "You are on a live VOICE call", and a large block of it
    exists to control a text-to-speech engine: no digits, no symbols, no
    lists, Devanagari for pronunciation. Those rules are not neutral in chat,
    they are actively wrong -- "10 pm" beats "दस बजे" on a screen, and a diet
    plan without structure is unreadable. A single prompt with "if voice... if
    chat..." branches would be wrong in both.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import chat_session
import llm_client
import memory_facts

# Per-message-type budgets. A single global word cap cannot survive into chat:
# an acknowledgement and a seven-day diet plan are both legitimate replies.
BUDGETS = {
    "ack": 10,
    "question": 25,
    "explain": 60,
    "advice": 90,
    "plan": 0,          # 0 = unbounded; a plan is a document, not a message
}

MAX_BUBBLES = int(os.getenv("CHAT_MAX_BUBBLES", "3"))

# Openers and one-word acknowledgements. A greeting handed a 30-word budget is
# a greeting with room to pad, and the model fills it -- "hi" came back as
# "Hey Ansh! Kaise ho? Kuch naya hua is week mein — sleep, digestion, ya
# khana — jo discuss karein?" A prompt rule alone loses to an available
# budget, so bound it mechanically too.
_SMALL_TALK = re.compile(
    r"^\s*(?:hi+|hey+|hello+|helo|yo|namaste|namaskar|good\s*(?:morning|"
    r"afternoon|evening|night)|gm|gn|salaam|assalam\w*|kaise\s*ho|kya\s*"
    r"haal|sup|thanks?|thank\s*you|thx|shukriya|ok|okay|okk+|hmm+|haan|ha|"
    r"ji|yes|no|nahi|bye|cya|tata|good\s*night)\W*$", re.I)


def is_small_talk(text: str) -> bool:
    """True for an opener or a bare acknowledgement -- something that wants a
    reply in kind, not an agenda."""
    return bool(_SMALL_TALK.match((text or "").strip()))
CHAT_MAX_TOKENS = int(os.getenv("CHAT_MAX_TOKENS", "700"))


def _prompt_template() -> str:
    return Path(__file__).with_name("chat_prompt.md").read_text(encoding="utf-8")


# --------------------------------------------------------------- extraction --
_EXTRACT_SYSTEM = """\
You read ONE chat message from a dietician's client. You never speak to the
user. Return JSON only:

{"extracted": {"<schema path>": "<value>"},
 "trigger": null|"EMERGENCY"|"MEDICAL"|"PRICING"|"MIRA_IDENTITY"|"SENSITIVE",
 "answered": true|false,
 "stage_done": true|false}

TRIGGER — set it whenever the message belongs to that category, however it is
phrased. A pattern list only catches wordings someone thought of in advance;
you are the layer that catches the rest.
  EMERGENCY      happening RIGHT NOW and needs help immediately: chest pain,
                 cannot breathe, fainting, heavy bleeding, self-harm.
  MEDICAL        things a DOCTOR owns, not a dietician: a NAMED diagnosis or
                 condition, a medication, a test result, a procedure.
                 YES: "mujhe thyroid hai", "BP ki problem hai", "depression ki
                      medicine leta hoon", "doctor ne operation bola",
                      "sugar 180 rehta hai"

                 NOT MEDICAL -- these are a dietician's ordinary work and
                 Mira must ENGAGE with them, never hand them to a doctor:
                      poor sleep, low energy, tiredness, bloating, gas,
                      constipation, acidity, cravings, appetite, weight,
                      "kuch acha nahi lag raha", feeling off.
                 A vague complaint is NOT a diagnosis. "sleep acchi nahi ho
                 rahi" is the single most normal thing a client says, and
                 answering it with "apne doctor se poochhiye" is a failure --
                 it is exactly what she is FOR.
  PRICING        cost, fees, plans, what is free or paid.
  MIRA_IDENTITY  whether you are human, AI, a bot.
  SENSITIVE      shame, body image, what people will think.

answered   — did this message give Mira any of the INFORMATION she asked for?
             Judge the content, not the effort.
             FALSE: "pata nahi", "kuch bhi", "hmm", "acha", silence, a garbled
                    message, or an answer to a different question. Polite, but
                    they carry nothing.
             TRUE:  any real detail, even partial or vague -- "raat mein",
                    "roz", "thoda sa". Also true when they genuinely cannot
                    know a fact and say so specifically ("apna weight nahi
                    pata"), because re-asking will never produce it.
stage_done — has the CURRENT stage finished its job? You are told the stage.
             UNDERSTAND  do you know WHAT the problem is, in their words?
             GATHER      do you know enough about it to say something useful --
                         roughly when, how often, what it affects?
             REFLECT     has the pattern been named back to them?
             ADVISE      has ONE concrete recommendation actually been given?
             CONFIRM     have they responded to that recommendation?
             Judge the CONVERSATION, not the number of turns. If Mira has been
             asking and getting nowhere, the stage is NOT done.

Default to null. Only set MEDICAL when a real diagnosis, medication, test
result or procedure is NAMED, or when it is a genuine EMERGENCY. Over-firing
makes Mira useless; the deferral is for what she genuinely must not touch.

EXTRACTED rules:
  - Use ONLY paths from the list below. Never invent one.
  - Only what the user ACTUALLY said in this message. Never infer, never
    carry over from earlier, never guess.
  - Durable facts only. "I ate poha today" is an event, not memory.
    "I eat poha most mornings" is memory.
  - Never extract a medical fact unless the user stated it plainly about
    themselves.
  - An empty object is the correct answer for most messages. Say nothing
    rather than something weak.

ALLOWED PATHS
"""


def merge_facts(pending: dict, found: dict) -> dict:
    """Fold a turn's extraction into the pending buffer.

    NOT dict.update(). Roughly a third of the schema is LIST-typed --
    health.conditions, preferences.dislikes, health.medications -- and a plain
    update replaces the list with whatever came last.

    Live chat test, 2026-09-01: the user said "mujhe pichle mahine heart
    attack aaya tha" and, four messages later, "mera sugar 180 rehta hai".
    Both landed on health.conditions. The buffer ended the session holding
    only the sugar reading; the heart attack was gone. Silently, on a health
    product, for the single most important fact in the conversation.

    Scalars still replace -- a corrected sleep time should not accumulate into
    a list of every answer the user has ever given.
    """
    out = dict(pending)
    for path, value in (found or {}).items():
        kind = memory_facts.SCHEMA.get(path)
        if kind != "list":
            out[path] = value
            continue
        have = out.get(path)
        items = list(have) if isinstance(have, list) else ([have] if have else [])
        for v in (value if isinstance(value, list) else [value]):
            v = str(v).strip()
            # Case-insensitive dedupe: "PCOS" and "pcos" are one condition.
            if v and v.lower() not in {str(x).lower() for x in items}:
                items.append(v)
        out[path] = items
    return out


async def read_message(user_text: str, mira_last: str = "",
                       stage: str = "") -> dict:
    """One cheap call per message: durable facts AND a safety category.

    The category matters as much as the facts. Chat reached production with
    REGEX-ONLY safety -- p3_graph matches patterns and stops there. On a live
    thread "mujhe pichle saal se thyroid hai" produced no trigger at all and
    Mira simply asked which medicine he takes, with no deferral to a doctor.
    The same list misses "BP ki problem hai", "depression ki medicine leta
    hoon" and "doctor ne operation bola hai", because MEDICAL requires a
    NUMBER after the condition.

    P-2 gained this backstop after "heart attack" slipped through; P-3 never
    got it. Folding it into the extraction call means no extra round trip.

    The extraction half is what gives Mira memory WITHIN a conversation --
    without it she would know nothing said since the last consolidation, which
    in chat could be hours ago. Anything it returns is provisional; the real
    validation (evidence grounding, supersession) happens on session close.

    Returns {"extracted": {...}, "trigger": name-or-None}.
    """
    if not (user_text or "").strip():
        return {"extracted": {}, "trigger": None,
                "answered": False, "stage_done": False}
    paths = "\n".join(f"  {p}" for p in sorted(memory_facts.SCHEMA))
    data = await llm_client.complete_json(
        _EXTRACT_SYSTEM + "\nALLOWED PATHS\n" + paths,
        json.dumps({"mira_just_asked": (mira_last or "")[:300],
                    "current_stage": stage or "UNDERSTAND",
                    "user_said": user_text}, ensure_ascii=False),
        kind="fast", max_tokens=400, temperature=0.0, timeout=8.0)
    out = {}
    for k, v in (data.get("extracted") or {}).items():
        if k in memory_facts.SCHEMA and str(v).strip():
            out[k] = v

    import p3_graph
    trig = str(data.get("trigger") or "").strip().upper() or None
    # Only categories P-3 actually honours. DEFLECT and WHAT_NEXT are
    # onboarding artefacts -- refusing to advise is the whole point of P-2 and
    # exactly wrong here.
    if trig not in (p3_graph.P3_HARD_TRIGGERS | p3_graph.P3_SOFT_TRIGGERS):
        trig = None
    return {"extracted": out, "trigger": trig,
            # Default answered=True: treating a real answer as a non-answer
            # makes Mira re-ask something she was just told, which is the
            # failure users notice most. Missing/uncertain is the safer no.
            "answered": bool(data.get("answered", True)),
            "stage_done": bool(data.get("stage_done", False))}


# ----------------------------------------------------------- summarisation --
_SUMMARY_SYSTEM = """\
You keep a running summary of a chat between a dietician (Mira) and her client.
Given the previous summary and the messages that have just scrolled out of
view, return an updated summary.

Return JSON only: {"summary": "<text>"}

  - Under 120 words. This is a memory aid, not a transcript.
  - Keep: what the user asked about, what was advised, anything unresolved.
  - Drop: pleasantries, and anything already captured as a durable fact.
  - Write it as continuous prose, in the third person.
"""


async def update_rolling_summary(previous: str, dropped_messages) -> str:
    """Compress what has fallen out of the verbatim window.

    A call's history is bounded at ~30 turns. A chat's is not, so something
    has to give: either the window grows without limit, or the tail is
    compressed. This is the compression.
    """
    if not dropped_messages:
        return previous
    body = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Mira'}: {m['text']}"
        for m in dropped_messages)
    data = await llm_client.complete_json(
        _SUMMARY_SYSTEM,
        json.dumps({"previous_summary": previous, "new_messages": body},
                   ensure_ascii=False),
        kind="fast", max_tokens=400, temperature=0.2, timeout=10.0)
    return str(data.get("summary") or previous).strip()


# ------------------------------------------------------- context assembly --
def build_system_prompt(session, user_context: str, directive: str,
                        retrieved_block: str = "") -> str:
    """Assemble the system message: stable content first, volatile last.

    The ordering is not cosmetic. A cache matches a stable PREFIX, so anything
    that changes per turn has to sit at the end or it invalidates everything
    after it -- the exact mistake that made the onboarding prompt uncacheable.
    """
    parts = [_prompt_template().replace("{{user_context}}", user_context)]

    if session.rolling_summary:
        parts.append("## Earlier in this conversation\n" + session.rolling_summary)

    loops = (session.memory or {}).get("open_loops") or []
    if loops:
        parts.append("## Still open from before\n"
                     + "\n".join(f"- {l}" for l in loops[:5]))

    last = (session.memory or {}).get("last_session_summary") or ""
    if last and not session.rolling_summary:
        parts.append("## Last time you spoke\n" + last)

    if session.pending_facts:
        parts.append("## Learned in this conversation\n"
                     + "\n".join(f"- {k}: {v}"
                                 for k, v in session.pending_facts.items()))

    if retrieved_block:
        parts.append(retrieved_block)

    if directive:
        parts.append("## For this reply\n" + directive)

    return "\n\n".join(parts)


# --------------------------------------------------------- reply shaping --
_SENT_SPLIT = re.compile(r"(?<=[।.!?])\s+")

# Real structure: bullets, numbered lines, headings, or many short lines. This
# is what "keep it whole" was always meant to catch -- not a paragraph break.
_LIST_LINE = re.compile(r"^\s*(?:[-*•]|\d+[.)]|#{1,6}\s|Day\s*\d)", re.I | re.M)


def _is_structured(text: str) -> bool:
    if _LIST_LINE.search(text):
        return True
    lines = [l for l in text.split("\n") if l.strip()]
    return len(lines) >= 4


def split_bubbles(text: str, max_bubbles: int = MAX_BUBBLES):
    """One 60-word paragraph reads as a form letter; the same words in two or
    three bubbles read as someone thinking. Split on sentence boundaries only
    -- never mid-thought, which reads as a glitch rather than a pause.

    A plan is left whole: chopping a structured document into bubbles destroys
    the structure that makes it readable.
    """
    text = (text or "").strip()
    if not text:
        return []

    # A DOCUMENT stays whole -- chopping a plan destroys the structure that
    # makes it readable. But "has a newline" is the wrong test for that: the
    # model puts a blank line between ordinary paragraphs, so the old guard
    # matched nearly every reply and bubbles never split at all. Look for
    # actual structure instead.
    if _is_structured(text) or len(text) > 900:
        return [text]

    # A blank line IS a natural bubble boundary -- the model has already told
    # us where the thought breaks. Prefer it to sentence splitting.
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paras) > 1:
        return paras[:max_bubbles] if len(paras) <= max_bubbles else \
            paras[:max_bubbles - 1] + [" ".join(paras[max_bubbles - 1:])]

    text = " ".join(text.split())

    # Splitting a very short reply is worse than not splitting it. "Of course.
    # No problem." across two bubbles reads as a stutter, not as thinking.
    if len(text.split()) < 12:
        return [text]

    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    if len(sents) <= 1:
        return [text]

    # Aim for roughly even bubbles rather than one long and one short.
    n = min(max_bubbles, len(sents))
    per = max(1, len(sents) // n)
    out, cur = [], []
    for s in sents:
        cur.append(s)
        if len(cur) >= per and len(out) < n - 1:
            out.append(" ".join(cur))
            cur = []
    if cur:
        out.append(" ".join(cur))
    return [b for b in out if b]


def typing_delay(text: str) -> float:
    """Seconds to show 'typing…' before a bubble.

    An instant reply to a hard question reads as machine even when the answer
    is right; a short pause reads as care. Floored so it never flickers,
    capped so it never feels broken.
    """
    return max(0.9, min(6.0, len(text or "") / 25.0))


# ------------------------------------------------------------ the turn --- #
async def _chat_completion(system: str, history, max_tokens: int) -> str:
    """One prose reply. Uses the same provider selection as everything else."""
    import httpx
    msgs = [{"role": m["role"], "content": m["content"]} for m in history]
    if llm_client.provider() == "bedrock":
        model = llm_client.bedrock_model("chat")
        body = llm_client.anthropic_body(system, msgs, max_tokens, 0.7)
        async with httpx.AsyncClient(timeout=45.0) as c:
            r = await c.post(llm_client._bedrock_url(model),
                             headers=llm_client.bedrock_headers(), json=body)
            r.raise_for_status()
            data = r.json()
        return "".join(b.get("text", "") for b in data.get("content", [])).strip()

    from groq import AsyncGroq
    client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
    resp = await client.chat.completions.create(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        messages=[{"role": "system", "content": system}] + msgs,
        temperature=0.7, max_tokens=max_tokens,
        reasoning_effort=os.getenv("LLM_REASONING_EFFORT", "low"))
    return (resp.choices[0].message.content or "").strip()


async def handle_turn(session, user_text: str, user_context: str,
                      log=None) -> dict:
    """One user message in, one reply out.

    Order matters. Safety is checked before anything else runs, and a safety
    hit short-circuits the LLM entirely rather than asking it nicely to
    include a fixed line -- a compliance answer that depends on the model
    choosing to comply is not a compliance answer.
    """
    log = log or (lambda m: None)
    import p3_graph
    import rag

    session.add("user", user_text)
    session.turn_index += 1

    # 1. Extract before routing, so a fact stated this turn is available to
    #    the reply that answers it -- not one turn late.
    read = {"extracted": {}, "trigger": None}
    try:
        mira_last = next((m["text"] for m in reversed(session.messages[:-1])
                          if m["role"] == "assistant"), "")
        cur_stage = next((t.stage for t in session.threads if not t.parked), "")
        read = await read_message(user_text, mira_last, cur_stage)
        found = read.get("extracted") or {}
        if found:
            # merge_facts, not update: a list path must accumulate. See above.
            session.pending_facts = merge_facts(session.pending_facts, found)
            log(f"chat extracted {list(found)} (pending {len(session.pending_facts)})")
    except Exception as exc:
        log(f"chat read failed: {type(exc).__name__}: {exc}")

    # 2. The shared brain: safety, router, lane, stage, retrieval.
    view = dict((session.memory or {}).get("long_term_memory") or {})
    graph = p3_graph.get_graph()
    out = await graph.ainvoke({
        "turn_text": user_text,
        "turn_index": session.turn_index,
        "threads": session.threads,
        "ledger_view": view,
        "fact_ages": {},
        "history": session.window()[:-1],
        "answered": read.get("answered", True),
        "stage_done": read.get("stage_done", False),
        "trace": [],
    })
    for line in out.get("trace", []):
        log(f"  chat graph: {line}")

    # 3. Safety. Two different behaviours, because they are different risks.
    #
    #    EMERGENCY stops everything: fixed line, no model call, no follow-up
    #    question. Someone describing chest pain must not be asked what they
    #    had for lunch.
    #
    #    Everything else says its fixed line and then CARRIES ON. A live
    #    thread showed why: three replies in a row were "apne doctor se
    #    poochhiye" and the user wrote "aap batao kuch". A dead stop on every
    #    health-adjacent word makes her useless -- the deferral is meant to
    #    protect what she must not touch, not to end the conversation.
    import onboarding_nodes
    hard = None
    if out.get("safety_hit"):
        hard = out["safety_hit"]
    elif read.get("trigger") in p3_graph.P3_HARD_TRIGGERS:
        hard = onboarding_nodes.trigger_response(read["trigger"])
        if hard:
            log(f"chat SEMANTIC trigger {read['trigger']} (regex missed it)")

    is_emergency = bool(hard) and "112" in hard
    if hard and is_emergency:
        session.add("assistant", hard)
        log("chat EMERGENCY -- fixed reply, no LLM, conversation stops")
        return {"bubbles": [hard], "safety": True,
                "stage": None, "lane": "SAFETY"}

    session.threads = out.get("threads", session.threads)

    retrieved = out.get("retrieved") or []
    ref = rag.format_reference(retrieved) if retrieved else ""

    directive = out.get("directive") or ""
    if hard:
        # Say the fixed wording, then keep going. The wording is fixed for
        # compliance reasons and must not be paraphrased; the turn after it
        # is ordinary conversation.
        directive = (
            f'Begin your reply with EXACTLY this sentence, word for word: '
            f'"{hard}"\n'
            "Then continue naturally with whatever you CAN help with — food, "
            "timing, habits, routine. Do not add medical guidance of your own "
            "and do not interpret any test result, but do not end the "
            "conversation either. Ask one useful thing, or offer what you can "
            "actually do.\n\n" + directive)
        log(f"chat trigger {read.get('trigger') or 'regex'} -- scripted line + continue")

    if not read.get("answered", True) and not hard:
        session.unclear = getattr(session, "unclear", 0) + 1
        n = session.unclear
        if n == 1:
            directive = (
                "That did not answer what you asked — they may not have "
                "understood it. Ask the SAME thing again in DIFFERENT, simpler "
                "words. Never repeat your previous sentence.\n\n" + directive)
        elif n == 2:
            directive = (
                "They still have not answered. Stop asking openly — make it "
                "trivially easy with a concrete either/or they can pick from, "
                "in one short line.\n\n" + directive)
        else:
            directive = (
                "They have not answered this three times. STOP asking it. Say "
                "warmly that it is fine, and either move to something else you "
                "can help with or offer what you can without it. Do NOT ask "
                "this again.\n\n" + directive)
        log(f"chat unanswered #{n} -- re-asking differently")
    elif read.get("answered", True):
        session.unclear = 0

    budget = out.get("budget") or BUDGETS["explain"]
    if is_small_talk(user_text) and not hard:
        budget = BUDGETS["ack"]
        directive = (
            "They said hello, or acknowledged you. Reply in kind and STOP.\n"
            "Do NOT offer topics, do NOT list what you could help with, do "
            "NOT bring up anything you remember about them, do NOT ask what "
            "they want to discuss. A greeting is not an opening for work.\n"
            'GOOD: "Hey Ansh! Kaise ho?"   GOOD: "Haan bilkul 🙂"\n'
            'BAD:  "Kaise ho? Kuch naya hua is week — sleep, digestion, ya '
            'khana?"\n\n' + directive)
        log("chat small talk -- ack budget, no agenda")
    if not out.get("may_advise"):
        directive += ("\n\nYou do NOT have enough about this yet to advise. "
                      "Ask the ONE thing you most need, and do not prescribe.")
    directive += f"\n\nKeep this reply to about {budget} words."

    system = build_system_prompt(session, user_context, directive, ref)
    try:
        reply = await _chat_completion(system, session.window(), CHAT_MAX_TOKENS)
    except Exception as exc:
        log(f"chat completion failed: {type(exc).__name__}: {exc}")
        reply = "Ek second, kuch issue aa gaya. Dobara bhejiye?"

    if not reply.strip():
        reply = "Sorry, samajh nahi aaya. Thoda aur bata sakte ho?"

    session.add("assistant", reply)

    # 4. Slide the window: compress whatever fell out of view.
    dropped = session.behind_window()
    if dropped and len(dropped) >= 4:
        try:
            session.rolling_summary = await update_rolling_summary(
                session.rolling_summary, dropped)
            session.messages = session.messages[-HISTORY_WINDOW_KEEP:]
            log(f"chat summary updated ({len(session.rolling_summary)} chars)")
        except Exception as exc:
            log(f"chat summary failed: {type(exc).__name__}: {exc}")

    words = len(reply.split())
    if budget and words > budget * 1.6:
        log(f"chat reply over budget ({words}w, target {budget}w)")

    # 5. Persist. AFTER the reply is composed, so a slow or failing store can
    #    never delay the user's message -- worst case the turn is lost on a
    #    restart, which is exactly today's behaviour rather than a regression.
    try:
        import chat_store
        await chat_store.save(session, log=log)
    except Exception as exc:
        log(f"chat persist failed: {type(exc).__name__}: {exc}")

    return {"bubbles": split_bubbles(reply), "safety": False,
            "stage": out.get("stage"), "lane": out.get("lane"),
            "may_advise": out.get("may_advise"), "words": words}


# Keep a little more than the verbatim window in memory, so the summariser
# always has overlap to work from rather than a hard cut.
HISTORY_WINDOW_KEEP = chat_session.HISTORY_WINDOW + 10


async def close_session(session, log=None) -> dict:
    """Run the SAME consolidation the voice call uses.

    Deliberately not a chat-specific write path. Evidence grounding,
    supersession instead of overwrite, and rejection of invented paths all
    live inside `consolidate_patch` + `memory_facts.apply_patch`. A second
    path would have to reproduce every one of them, and would get at least one
    wrong.
    """
    log = log or (lambda m: None)
    import consolidate
    import memory_store

    if session.closed:
        return {"ok": False, "reason": "already closed"}
    session.closed = True
    try:
        import chat_store
        await chat_store.mark_closed(session.firebase_uid, log=log)
    except Exception:
        pass

    transcript = session.transcript()
    user_turns = sum(1 for m in session.messages if m["role"] == "user")
    if user_turns == 0:
        log(f"chat session {session.session_id} closed empty -- nothing to write")
        return {"ok": True, "skipped": "no user turns"}

    # Persist the raw transcript BEFORE consolidating. If the model call
    # fails, the session is still replayable -- this is the Tier-0 rule the
    # voice path already follows.
    try:
        import datetime as _dt

        def _iso(ts):
            return _dt.datetime.utcfromtimestamp(ts).replace(
                tzinfo=_dt.timezone.utc).isoformat()

        memory_store.save_session_raw(
            session.session_id, session.firebase_uid,
            (session.profile or {}).get("user_id", ""), "chat",
            _iso(session.started_at), _iso(session.last_activity),
            transcript, len(session.messages))
    except Exception as exc:
        log(f"chat transcript save failed: {type(exc).__name__}: {exc}")

    try:
        view = dict((session.memory or {}).get("long_term_memory") or {})
        loops = (session.memory or {}).get("open_loops") or []
        patch = await consolidate.consolidate_patch(view, loops, transcript)
    except Exception as exc:
        log(f"chat consolidation failed: {type(exc).__name__}: {exc}")
        return {"ok": False, "reason": str(exc)[:200]}

    try:
        facts = memory_store.load_facts(session.firebase_uid)
        new_rows, invalidated, applied = memory_facts.apply_patch(
            facts, patch.get("ops", []), session.session_id, transcript)
        if new_rows:
            memory_store.append_facts(new_rows)
        if invalidated:
            memory_store.stamp_invalidations(invalidated, memory_facts._now())
        merged = memory_facts.build_current_view(facts + new_rows)
        memory_store.cache_current_view(
            session.firebase_uid, merged, patch.get("open_loops", []),
            patch.get("session_summary", ""), memory_facts._now(), "chat")
        rejected = [a for a in applied if not a.get("applied")]
        log(f"chat consolidated {session.session_id}: "
            f"{len(new_rows)} facts, {len(invalidated)} superseded, "
            f"{len(rejected)} rejected")
        return {"ok": True, "facts": len(new_rows),
                "superseded": len(invalidated), "rejected": len(rejected)}
    except Exception as exc:
        log(f"chat memory write failed: {type(exc).__name__}: {exc}")
        return {"ok": False, "reason": str(exc)[:200]}
