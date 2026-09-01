"""P-3 turn graph (LangGraph).

WHERE THIS STARTS AND ENDS
    It starts on a final transcript and ends by producing a DIRECTIVE. It does
    NOT generate Mira's reply — Pipecat's existing LLM service does that, so
    streaming into TTS and context aggregation are untouched. The graph is
    everything between "the user finished speaking" and "we know how Mira
    should behave this turn".

ONE MODEL CALL PER TURN, PLUS ONE SMALL ROUTER CALL
    The router is a single cheap call that does classification, thread
    planning and slot extraction together, and it runs CONCURRENTLY with
    retrieval — so its latency hides inside a round-trip we were already
    paying for. Adding a serial model call here would undo the latency work
    that made this product usable.

Nodes
    safety     deterministic keyword match, can end the turn outright
    sense      router + retrieval, run together (the only model call here)
    lane       apply QUICK / ADVANCE / SWITCH / RESUME to the thread stack
    plan       pre-fill from long-term memory, work out what is still missing
    directive  build the behavioural block appended to the base prompt
"""
import asyncio
import json
import os

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict

from langgraph.graph import END, StateGraph

import llm_client
import memory_facts
import rag
import rag_query
import thread_machine as tm

_ROUTER_MODEL = os.getenv("P3_ROUTER_MODEL", "openai/gpt-oss-20b")
# Measured live: router p50 ~700-1700ms but with a long tail. At 3.0s roughly
# one turn in six timed out, and every timeout silently became QUICK — which
# is how an explicit "let's go back to the hunger thing" failed to RESUME.
# 5s captures the tail. The real fix is a faster router; this stops the
# fallback from firing on turns that would have classified correctly.
_ROUTER_TIMEOUT = float(os.getenv("P3_ROUTER_TIMEOUT", "5.0"))

def _paths_block():
    """Group the 49 schema paths by family.

    The flat listing was the largest single block in the router prompt, and
    prompt size is router latency, which is turn latency. Grouping keeps every
    path visible while roughly halving the characters.
    """
    fams = {}
    for path in sorted(memory_facts.SCHEMA):
        fam, _, leaf = path.rpartition(".")
        fams.setdefault(fam or path, []).append(leaf if fam else "")
    out = []
    for fam, leaves in fams.items():
        leaves = [l for l in leaves if l]
        out.append(f"  {fam}.{{{', '.join(leaves)}}}" if leaves else f"  {fam}")
    return "\n".join(out)


_PATHS = _paths_block()

# What each stage means, sent to the router so it can read a short reply in
# context. "हाँ" after a reflection is a confirmation that should advance the
# thread; the same word after a factual answer is a backchannel.
_STAGE_MEANING = {
    tm.S_UNDERSTAND: "Mira just asked what the problem is. A reply describing "
                     "the problem ADVANCES.",
    tm.S_GATHER: "Mira just asked for one specific detail. Any answer to it, "
                 "even partial or vague, ADVANCES.",
    tm.S_REFLECT: "Mira just stated the problem back and is waiting for the "
                  "user to confirm it. A confirmation ADVANCES toward advice.",
    tm.S_ADVISE: "Mira just gave advice. A reaction to that advice ADVANCES.",
    tm.S_CONFIRM: "Mira just asked whether the advice is doable. Any yes/no "
                  "ADVANCES.",
    tm.S_CLOSE: "This thread is finished.",
}

_ROUTER_SYSTEM = f"""\
You are the router for Mira, an AI dietician on a live Hinglish voice call.
You never speak to the user. You read ONE user turn IN CONTEXT and return JSON.

Return exactly this shape, nothing else:
{{
  "lane": "QUICK" | "ADVANCE" | "SWITCH" | "RESUME",
  "situation": "FACTUAL"|"PERSONAL"|"PROBLEM"|"PLAN"|"UPDATE"|"CORRECTION"|"MEMORY_QUERY"|"AMBIGUOUS"|"OFF_TOPIC"|"SOCIAL"|"AFFIRMATION",
  "topic": "<short topic name — only when opening a new thread>",
  "template": "PROBLEM" | "PLAN" | "HABIT",
  "needed_paths": ["<memory path>", ...],
  "adhoc": ["<something the schema cannot express>"],
  "extracted": {{"<memory path>": "<value the user just gave>"}},
  "sufficient": true | false,
  "explicit_advice_request": true | false,
  "resume_hint": "<topic they are returning to — only for RESUME>"
}}

=====================  LANE  =====================
Decide by intent IN CONTEXT, never by sentence length. Work through these in
order and stop at the first that matches.

1. RESUME — they are going back to a parked topic.
   "वो भूख वाली बात पे वापस आते हैं" / "पहले वाली बात" / "जो हम discuss कर रहे थे"
   Only if parked_threads is non-empty. Set resume_hint.

2. SWITCH — a genuinely NEW problem, or they explicitly drop the current one.
   "अरे वो छोड़िए, शादी है weight कम करना है"
   Opening a thread when none is active is also SWITCH.

3. ADVANCE — the turn belongs to the active thread. This is the DEFAULT
   whenever a thread is active and the turn relates to it in any way.
   Includes ALL of these:
     - answering what Mira just asked, even partially: "सात बजे", "पता नहीं"
     - CONFIRMING what Mira just said: "हाँ", "हाँ बिल्कुल", "सही", "बिल्कुल सही",
       "हाँ यही", "आपने सही समझा", "correct", "exactly", "that's right", "yes"
     - disagreeing: "नहीं, ऐसा नहीं है"
     - reacting to advice: "ठीक है", "try करूँगी", "मुश्किल है"
     - adding detail unprompted: "और मैं देर से सोती भी हूँ"
     - a vague or sideways reply that is still a reply: "बस ऐसे ही", "हम्म"

4. QUICK — ONLY when the turn genuinely does not move the active thread:
     - a self-contained factual question on a DIFFERENT subject
       ("green tea रात को चलेगी?" while discussing night hunger)
     - greetings and small talk with NO thread open
     - off-topic ("आप शादीशुदा हैं?")
     - correcting a stored fact ("नहीं, मैं egg खाती हूँ")
     - asking what you remember ("पिछली बार क्या बोला था?")

CRITICAL — SHORT REPLIES ARE NOT AUTOMATICALLY QUICK.
If a thread is active and the reply plausibly responds to Mira's last message,
it is ADVANCE. Use stage_means to decide. Examples with a thread at REFLECT,
where Mira has just stated the problem back:
   "हाँ बिल्कुल सही कहा"  -> ADVANCE, situation AFFIRMATION   (NOT QUICK)
   "हाँ"                  -> ADVANCE, situation AFFIRMATION   (NOT QUICK)
   "सही"                  -> ADVANCE, situation AFFIRMATION   (NOT QUICK)
   "हम्म"                 -> ADVANCE, situation AMBIGUOUS     (NOT QUICK)
   "green tea चलेगी?"      -> QUICK  (different subject, does not answer Mira)
When genuinely unsure between ADVANCE and QUICK with a thread open, choose
ADVANCE. Losing the thread is worse than advancing it a turn early.

=====================  needed_paths  =====================
ONLY when opening a thread (SWITCH). AT MOST 3, most useful first.
Choose ONLY from the list below. NEVER invent a path.

Choose by asking: "which facts, if I knew their VALUES, would let me explain
WHY this is happening and what to change?" — NOT "which path names share words
with what the user said."

A symptom is almost always caused by something EARLIER, not by itself. Look
UPSTREAM of the complaint:
   night hunger      -> what and when DINNER was (the cause), not what they
                        snack on at night (that is the symptom restated);
                        lifestyle.sleep_time matters too, because dinner time
                        plus sleep time is what defines how long they are
                        awake and hungry
   trouble sleeping  -> when they eat and when they sleep
   afternoon slump   -> breakfast and lunch, not the slump
   bloating at night -> what the evening meal contains
   no energy in gym  -> the meal before the workout

Prefer what and when they EAT, and when the complaint involves nights, sleep
or energy, WHEN THEY SLEEP (lifestyle.sleep_time) is usually one of the three. Do NOT choose identity.basics.* or
identity.body.* unless the problem is literally about age or body
measurements. Do not request a fact you could not act on.

ALLOWED PATHS
{_PATHS}

adhoc — at most ONE, only for something the schema genuinely cannot express
(e.g. "stress at work"). Sleep timing IS in the schema now
(lifestyle.sleep_time) — use the path, never adhoc. Never restate the topic
as adhoc.

=====================  extracted  =====================
Anything the user told you IN THIS TURN, keyed by the EXACT schema path.
   "सात बजे dinner करती हूँ"  -> {{"current_pattern.dinner.time": "7pm"}}
   "दो roti और sabzi"        -> {{"current_pattern.dinner.frequent": "roti, sabzi"}}
   "मुझे PCOS है"             -> {{"health.conditions": "PCOS"}}
   "ग्यारह बजे सोती हूँ"        -> {{"lifestyle.sleep_time": "11pm"}}
Never ask again for anything you put here, and never list it in needed_paths.
Empty object if they gave no new fact.

=====================  other fields  =====================
sufficient — do you UNDERSTAND THE COMPLAINT well enough to advise on it?
This is about the problem, NOT about how many facts are on file.
  "रात को भूख लगती है"           -> true  (a specific, actionable complaint)
  "खाने के बाद पेट फूलता है"      -> true
  "मुझे और भी दिक्कतें हो रही हैं"  -> FALSE (which problems? nothing to act on)
  "तबीयत ठीक नहीं लग रही"         -> FALSE (too vague)
  "कुछ अजीब लग रहा है"            -> FALSE
If you could not tell another dietician what is actually wrong in one
sentence, it is false.

explicit_advice_request — true when they directly ask what to eat or do
("बस बता दीजिए क्या खाऊँ", "diet plan दे दीजिए").

Output JSON only."""


class ConvState(TypedDict, total=False):
    # inputs
    turn_text: str
    turn_index: int
    threads: List[Any]
    ledger_view: Dict[str, Any]
    fact_ages: Dict[str, str]
    history: List[Dict[str, str]]
    # produced
    safety_hit: Optional[str]
    soft_safety: Optional[str]
    lane: str
    situation: str
    router_raw: Dict[str, Any]
    retrieved: List[Dict[str, Any]]
    gather: Dict[str, Any]
    directive: str
    budget: int
    may_advise: bool
    stage: str
    trace: List[str]


# ------------------------------------------------------------------ nodes --
# P-2's global triggers are reused, but NOT all of them. Two are artefacts of
# the onboarding call and are actively wrong here:
#   DEFLECT    "I'm not giving advice on this call" — P-3 IS the advice product.
#              Left in, it fires on "क्या खाऊँ" and blocks the core use case.
#   WHAT_NEXT  "this call was to understand you" — false in P-3.
# SENSITIVE is kept but demoted: answering body-image distress with a canned
# line is worse than a warm, personal reply, so it steers the LLM instead of
# replacing it.
# EMERGENCY is hard on EVERY surface. It is the one case where stopping
# the conversation dead is the correct product behaviour.
P3_HARD_TRIGGERS = {"EMERGENCY", "MEDICAL", "PRICING", "MIRA_IDENTITY"}
P3_SOFT_TRIGGERS = {"SENSITIVE"}

_SENSITIVE_DIRECTIVE = (
    "The user has said something vulnerable about their body or how they feel "
    "about themselves. Acknowledge it warmly and normalise it. Do NOT give diet "
    "advice in this turn and do NOT be clinical."
)


def _p3_trigger(text):
    """Return (name, response) for the first matching P-3-relevant trigger."""
    import re
    import onboarding_nodes
    body = (text or "").strip()
    if not body:
        return None, None
    for trig in onboarding_nodes.GLOBAL_TRIGGERS:
        name = trig.get("name")
        if name not in P3_HARD_TRIGGERS and name not in P3_SOFT_TRIGGERS:
            continue
        for pat in trig.get("patterns", []):
            try:
                if re.search(pat, body, re.IGNORECASE):
                    return name, trig.get("response", "")
            except re.error:
                continue
    return None, None


def _node_safety(state: ConvState) -> ConvState:
    """Deterministic. Sits above the router so no routing decision can bypass it."""
    trace = list(state.get("trace", []))
    name, response = _p3_trigger(state.get("turn_text", ""))
    if name in P3_HARD_TRIGGERS:
        trace.append(f"safety HARD {name} -> fixed reply, no LLM")
        return {"safety_hit": response, "lane": tm.LANE_SAFETY, "trace": trace}
    if name in P3_SOFT_TRIGGERS:
        # Also forced to QUICK downstream: someone saying "I feel fat" needs a
        # human response, not a consultation that opens by asking their height
        # and weight — which is exactly what the router did in live testing.
        trace.append(f"safety SOFT {name} -> directive + no new thread")
        return {"safety_hit": None, "soft_safety": _SENSITIVE_DIRECTIVE, "trace": trace}
    return {"safety_hit": None, "soft_safety": None, "trace": trace}


async def _call_router(text, threads, history):
    """The single model call. Small model, JSON out, hard timeout.

    Provider-agnostic via llm_client: Bedrock when AWS credentials are present,
    Groq otherwise. Bedrock has no response_format, so llm_client prefills the
    assistant turn with "{" to force a JSON object.
    """
    active = next((t for t in threads if not t.parked), None)
    parked = [t.topic for t in threads if t.parked]

    # Mira's last line is what a short reply is REPLYING TO. Without it the
    # router cannot tell a confirmation ("हाँ" after a reflection) from a
    # backchannel ("हाँ" after a factual answer), and it was defaulting both
    # to QUICK — which silently abandoned the thread.
    mira_last = next((m.get("content") for m in reversed(history or [])
                      if m.get("role") == "assistant"), "")

    ctx = {
        "active_thread": None if not active else {
            "topic": active.topic,
            "stage": active.stage,
            "stage_means": _STAGE_MEANING.get(active.stage, ""),
            "turns_in_stage": active.stage_turns,
            "already_known": {k: str(v)[:60] for k, v in
                              list(active.slots.items())[:6]},
            "still_missing": active.gaps()[:5],
        },
        "parked_threads": parked,
        "mira_last_said": str(mira_last)[:300],
        "user_said": text,
        "recent_turns": history[-4:],
    }
    return await llm_client.complete_json(
        _ROUTER_SYSTEM, json.dumps(ctx, ensure_ascii=False),
        kind="fast", max_tokens=600, temperature=0.1,
        timeout=_ROUTER_TIMEOUT, groq_model=_ROUTER_MODEL)


async def _node_sense(state: ConvState) -> ConvState:
    """Router and retrieval CONCURRENTLY — the router hides inside RAG latency."""
    text = state.get("turn_text", "")
    threads = state.get("threads", [])
    history = state.get("history", [])
    trace = list(state.get("trace", []))

    ok, why = rag_query.is_retrievable(text)

    async def _route():
        try:
            return await asyncio.wait_for(
                _call_router(text, threads, history), timeout=_ROUTER_TIMEOUT)
        except Exception as exc:
            # Loud on purpose: this rate is the single most important health
            # metric for the graph. Every one of these is a lost SWITCH/RESUME.
            kind = "TIMEOUT" if isinstance(exc, asyncio.TimeoutError) else type(exc).__name__
            trace.append(f"ROUTER {kind} -> degraded to QUICK (thread not advanced)")
            return {}

    async def _retrieve():
        if not ok or not rag.enabled():
            return []
        try:
            query, _ = rag_query.build_query(text, history)
            return await rag.retrieve(query, k=int(os.getenv("RAG_TOP_K", "3")),
                                      min_similarity=float(os.getenv("RAG_MIN_SIMILARITY", "0.5")))
        except Exception:
            return []

    routed, docs = await asyncio.gather(_route(), _retrieve())

    lane = str(routed.get("lane", "")).upper()
    if lane not in tm.LANES:
        lane = tm.LANE_QUICK               # unclassifiable degrades to today's P-3
    if state.get("soft_safety") and lane == tm.LANE_SWITCH:
        lane = tm.LANE_QUICK               # acknowledge, do not start an intake
    # No thread open and the router did not open one -> nothing to advance.
    if lane in (tm.LANE_ADVANCE, tm.LANE_RESUME) and not state.get("threads"):
        lane = tm.LANE_QUICK
    if not ok:
        trace.append(f"rag skipped ({why})")
    trace.append(f"lane={lane} situation={routed.get('situation', '?')}")
    return {"lane": lane, "situation": str(routed.get("situation", "")),
            "router_raw": routed, "retrieved": docs, "trace": trace}


def _node_lane(state: ConvState) -> ConvState:
    """Apply the lane to the thread stack. Pure state manipulation, no model."""
    lane = state.get("lane")
    routed = state.get("router_raw", {}) or {}
    threads = list(state.get("threads", []))
    trace = list(state.get("trace", []))

    if lane == tm.LANE_SWITCH:
        # Restating a problem is not a new problem. Reuse the existing thread —
        # unparking it if needed — so collected slots and stage survive.
        existing = tm.find_thread(threads, routed.get("topic") or "")
        if existing is not None:
            tm.park_active(threads)
            existing.parked = False
            threads.remove(existing)
            threads.insert(0, existing)
            trace.append(f"SWITCH -> existing thread '{existing.topic}' @{existing.stage}")
            lane = tm.LANE_ADVANCE          # continue it, do not restart it
        else:
            tm.park_active(threads)
        # Invented paths are dropped, and the plan is capped: a longer plan
        # cannot be satisfied inside the dwell limit, so it only guarantees an
        # interrogation that then gets force-advanced anyway.
    if lane == tm.LANE_SWITCH:
        # F2: dedupe while preserving the router's relevance ordering —
        # duplicates inflate the gap count and make GATHER think there is
        # more outstanding than there is.
        seen = set()
        needed = []
        for p in (routed.get("needed_paths") or []):
            if p in memory_facts.SCHEMA and p not in seen:
                seen.add(p)
                needed.append(p)
        needed = needed[:tm.MAX_NEEDED_PATHS]
        topic = str(routed.get("topic") or "this")[:60]
        adhoc = []
        for a in (routed.get("adhoc") or [])[:2]:
            a = str(a)[:60].strip()
            # Reject adhoc that is really the topic restated, or a schema path
            # wearing a different hat.
            if a and a not in memory_facts.SCHEMA and \
                    not tm._tokens_of(a) <= tm._tokens_of(topic):
                adhoc.append(a)
        th = tm.Thread(
            topic=topic,
            template=str(routed.get("template") or "PROBLEM").upper(),
            needed_paths=needed,
            adhoc=adhoc[:1],
            opened_at=int(state.get("turn_index", 0)),
        )
        threads.insert(0, th)
        threads, spill = tm.spill_oldest(threads)
        if spill:
            trace.append(f"spilled {len(spill)} thread(s) to open loops")
        trace.append(f"thread OPEN '{th.topic}' paths={len(th.needed_paths)}")

    elif lane == tm.LANE_RESUME:
        target = tm.find_parked(threads, routed.get("resume_hint") or routed.get("topic"))
        if target:
            tm.park_active(threads)
            target.parked = False
            threads.remove(target)
            threads.insert(0, target)
            trace.append(f"thread RESUME '{target.topic}' @{target.stage}")
        else:
            trace.append("resume: no parked match -> QUICK")
            return {"threads": threads, "lane": tm.LANE_QUICK, "trace": trace}

    # Slots the user just filled land on the active thread regardless of lane.
    active = next((t for t in threads if not t.parked), None)
    if active:
        # Map whatever keys the router used onto real schema paths, so the slot
        # lands on the path GATHER is actually waiting for. Without this the gap
        # never closes and Mira re-asks what she was just told.
        mapped, extra = tm.map_extracted(routed.get("extracted") or {}, active)
        for path, val in mapped.items():
            active.slots[path] = val
        for key, val in extra.items():
            active.slots[key] = val
        if mapped or extra:
            trace.append(f"extracted -> {list(mapped) or list(extra)}")
        if lane in (tm.LANE_ADVANCE, tm.LANE_RESUME):
            if active.stage == tm.S_CLOSE:
                # A finished thread has nothing left to advance. Without this
                # it sits at CLOSE incrementing forever and Mira keeps being
                # told to "wrap up warmly" on every subsequent turn.
                trace.append("thread already CLOSE -> nothing to advance")
                lane = tm.LANE_QUICK
            else:
                active.stage_turns = min(active.stage_turns + 1,
                                         tm.MAX_STAGE_TURNS)

    return {"threads": threads, "lane": lane, "trace": trace}


def _node_plan(state: ConvState) -> ConvState:
    """Pre-fill from long-term memory; work out what is genuinely still missing."""
    threads = list(state.get("threads", []))
    active = next((t for t in threads if not t.parked), None)
    trace = list(state.get("trace", []))
    if not active:
        return {"gather": {"known": {}, "stale": [], "missing": []}, "trace": trace}

    gather = tm.plan_gather(active, state.get("ledger_view", {}),
                            state.get("fact_ages", {}))
    tm.prefill(active, gather)
    trace.append(f"gather known={len(gather['known'])} "
                 f"stale={len(gather['stale'])} missing={len(gather['missing'])}")

    # Decide the stage this turn will actually be spoken at.
    if state.get("lane") == tm.LANE_SWITCH:
        active.stage = tm.S_UNDERSTAND
        active.stage_turns = 1
        # Skipping ahead needs BOTH: the facts to advise with (missing empty)
        # AND an actual understanding of the complaint (router's `sufficient`).
        # Live failure: the user said only "मुझे और भी दिक्कतें होने लग रही हैं"
        # — nothing about WHAT — but their ledger already held the three paths
        # the router happened to pick, so missing was 0 and the graph jumped
        # straight to REFLECT and advised. Having the facts is not the same as
        # knowing the problem. When the complaint is vague, UNDERSTAND must
        # keep its turn and Mira must ask.
        sufficient = bool((state.get("router_raw") or {}).get("sufficient"))
        if not gather["missing"] and sufficient:
            active.stage = tm.S_REFLECT
            trace.append("GATHER skipped — memory had everything")
        elif not gather["missing"]:
            trace.append("holding at UNDERSTAND — problem statement is vague")
    else:
        sufficient = bool((state.get("router_raw") or {}).get("sufficient"))
        nxt = tm.next_stage(active, gather, sufficient)
        if nxt != active.stage:
            trace.append(f"stage {active.stage} -> {nxt}")
            active.stage = nxt
            active.stage_turns = 1
    return {"threads": threads, "gather": gather, "trace": trace}


def _node_directive(state: ConvState) -> ConvState:
    """Build the block appended to the base system prompt. No model call."""
    lane = state.get("lane")
    trace = list(state.get("trace", []))

    threads = state.get("threads", [])
    active = next((t for t in threads if not t.parked), None)
    routed = state.get("router_raw", {}) or {}
    explicit = str(routed.get("situation", "")).upper() in ("PLAN",) or \
        bool(routed.get("explicit_advice_request"))

    if lane == tm.LANE_QUICK:
        pol = tm.quick_directive(state.get("situation", ""), active, explicit)
        stage = "-"
    else:
        if not active:
            pol = tm.quick_directive(state.get("situation", ""), None, explicit)
            stage = "-"
        else:
            pol = tm.stage_directive(active, state.get("gather", {}))
            stage = active.stage
            if active.stage == tm.S_REFLECT:
                active.reflected = True
    directive = pol["directive"]
    may_advise = pol["may_advise"]
    soft = state.get("soft_safety")
    if soft:
        directive = soft + " " + directive
        may_advise = False
    trace.append(f"stage={stage} budget={pol['budget']} may_advise={may_advise}")
    return {"directive": directive, "budget": pol["budget"],
            "may_advise": may_advise, "stage": stage, "trace": trace}


# ------------------------------------------------------------------ graph --
def _after_safety(state: ConvState):
    return END if state.get("safety_hit") else "sense"


def _after_lane(state: ConvState):
    return "directive" if state.get("lane") == tm.LANE_QUICK else "plan"


def build_graph():
    g = StateGraph(ConvState)
    g.add_node("safety", _node_safety)
    g.add_node("sense", _node_sense)
    g.add_node("lane", _node_lane)
    g.add_node("plan", _node_plan)
    g.add_node("directive", _node_directive)

    g.set_entry_point("safety")
    g.add_conditional_edges("safety", _after_safety, {END: END, "sense": "sense"})
    g.add_edge("sense", "lane")
    g.add_conditional_edges("lane", _after_lane, {"plan": "plan", "directive": "directive"})
    g.add_edge("plan", "directive")
    g.add_edge("directive", END)
    return g.compile()


GRAPH = None


def get_graph():
    global GRAPH
    if GRAPH is None:
        GRAPH = build_graph()
    return GRAPH
