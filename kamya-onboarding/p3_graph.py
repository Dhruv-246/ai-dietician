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

_PATHS = "\n".join(f"  {p}" for p in sorted(memory_facts.SCHEMA))

_ROUTER_SYSTEM = f"""\
You are the router for Mira, an AI dietician on a live voice call. You do NOT
talk to the user. You classify one user turn and return JSON.

Return exactly:
{{
  "lane": "QUICK" | "ADVANCE" | "SWITCH" | "RESUME",
  "situation": "FACTUAL"|"PERSONAL"|"PROBLEM"|"PLAN"|"UPDATE"|"CORRECTION"|"MEMORY_QUERY"|"AMBIGUOUS"|"OFF_TOPIC"|"SOCIAL",
  "topic": "<short topic name, only when opening a new thread>",
  "template": "PROBLEM" | "PLAN" | "HABIT",
  "needed_paths": ["<memory path>", ...],
  "adhoc": ["<something not covered by the paths>", ...],
  "extracted": {{"<path or adhoc>": "<value the user just gave>"}},
  "sufficient": true | false,
  "resume_hint": "<topic they are returning to, only for RESUME>"
}}

LANE — the most important field. Bias toward QUICK when unsure.
  QUICK    A question or remark that does NOT need a consultation: a factual
           question, an aside, a correction, a greeting, something off-topic,
           a memory lookup, or a short answer that needs no follow-up.
           Also use QUICK for any turn while no thread is open that is not
           itself a new problem.
  ADVANCE  The user is responding to the thread that is currently open —
           answering the question Mira just asked, or continuing that topic.
  SWITCH   The user raises a NEW problem, or explicitly drops the current one.
  RESUME   The user returns to a topic that was parked earlier.

needed_paths — ONLY when opening a thread (SWITCH). AT MOST 3, ordered most
useful first. Choose ONLY from this list; never invent a path. Pick what you
would genuinely need to advise on THIS problem — not everything plausible.
Do NOT ask for age, gender, height or weight unless the problem is literally
about body measurements. Prefer what and when they eat.
{_PATHS}

adhoc — only for something genuinely not covered above (e.g. "stress at work").
At most ONE. Never restate the topic itself as adhoc.

extracted — anything the user JUST told you in this turn. Key it by the exact
schema path from the list above wherever one fits — not by a made-up name.
"सात बजे dinner करती हूँ" -> {{"current_pattern.dinner.time": "7pm"}}.

sufficient — true if there is now enough to give useful advice, even if some
details are still unknown. A good dietician acts on partial information.

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
P3_HARD_TRIGGERS = {"MEDICAL", "PRICING", "MIRA_IDENTITY"}
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
    """The single model call. Small model, JSON out, hard timeout."""
    from groq import AsyncGroq
    active = next((t for t in threads if not t.parked), None)
    parked = [t.topic for t in threads if t.parked]
    ctx = {
        "active_thread": None if not active else {
            "topic": active.topic, "stage": active.stage,
            "still_missing": active.gaps()[:5],
        },
        "parked_threads": parked,
        "recent_turns": history[-4:],
    }
    client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
    resp = await client.chat.completions.create(
        model=_ROUTER_MODEL,
        messages=[
            {"role": "system", "content": _ROUTER_SYSTEM},
            {"role": "user", "content":
                "CONTEXT:\n" + json.dumps(ctx, ensure_ascii=False)
                + "\n\nUSER TURN:\n" + text},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content or "{}")


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
        tm.park_active(threads)
        # Invented paths are dropped, and the plan is capped: a longer plan
        # cannot be satisfied inside the dwell limit, so it only guarantees an
        # interrogation that then gets force-advanced anyway.
        needed = [p for p in (routed.get("needed_paths") or [])
                  if p in memory_facts.SCHEMA][:tm.MAX_NEEDED_PATHS]
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
            active.stage_turns += 1

    return {"threads": threads, "trace": trace}


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
        if not gather["missing"]:
            active.stage = tm.S_REFLECT      # ledger already answered everything
            trace.append("GATHER skipped — memory had everything")
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
