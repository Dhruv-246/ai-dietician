"""Core regression suite. Runs with plain python — no pipecat, no langgraph,
no network, no credentials.

    python3 tests/test_core.py

These live IN THE REPO on purpose. An earlier version lived in a scratch
directory and was lost with it, taking ~168 assertions of coverage with it.
Anything that only exists outside version control is not a test suite.
"""
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.modules.setdefault("httpx", types.ModuleType("httpx"))

import echo_guard               # noqa: E402
import llm_client as L          # noqa: E402
import memory_facts as mf       # noqa: E402
import onboarding_nodes as on   # noqa: E402
import rag_query as rq          # noqa: E402
import reply_shape              # noqa: E402
import chat_engine              # noqa: E402
import chat_guard               # noqa: E402
import chat_session             # noqa: E402
import chat_store               # noqa: E402
import thread_machine as tm     # noqa: E402

R = []
def ck(name, cond, detail=""):
    R.append((name, bool(cond), detail))


# ---------------------------------------------------------------- provider --
def test_llm_client():
    for k in ("AWS_BEARER_TOKEN_BEDROCK", "AWS_ACCESS_KEY_ID",
              "AWS_SECRET_ACCESS_KEY", "LLM_PROVIDER"):
        os.environ.pop(k, None)
    ck("provider falls back to groq with no credentials", L.provider() == "groq")

    # A Bedrock API key is a bearer token — no access key or secret involved.
    os.environ["AWS_BEARER_TOKEN_BEDROCK"] = "test"
    ck("bearer token alone selects bedrock", L.provider() == "bedrock")
    os.environ["LLM_PROVIDER"] = "groq"
    ck("LLM_PROVIDER overrides detection", L.provider() == "groq")
    os.environ.pop("LLM_PROVIDER")

    os.environ["AWS_REGION"] = "ap-south-1"
    os.environ["BEDROCK_MODEL"] = "anthropic.claude-haiku-4-5-v1:0"
    ck("one BEDROCK_MODEL serves every job",
       L.bedrock_model("chat") == L.bedrock_model("fast") == L.bedrock_model("heavy"))
    os.environ["BEDROCK_FAST_MODEL"] = "fast-model"
    ck("per-job override wins", L.bedrock_model("fast") == "fast-model"
       and L.bedrock_model("chat") == "anthropic.claude-haiku-4-5-v1:0")
    os.environ.pop("BEDROCK_FAST_MODEL")

    url = L._bedrock_url("anthropic.claude-haiku-4-5-v1:0")
    ck("invoke URL matches the documented Bedrock shape",
       url == "https://bedrock-runtime.ap-south-1.amazonaws.com"
              "/model/anthropic.claude-haiku-4-5-v1:0/invoke", url)
    ck("streaming URL uses invoke-with-response-stream",
       L._bedrock_url("m", stream=True).endswith("/invoke-with-response-stream"))
    ck("auth header is a bearer token",
       L.bedrock_headers()["Authorization"].startswith("Bearer "))

    b = L.anthropic_body("SYS", [{"role": "user", "content": "hi"}], 100, 0.1,
                         stop=["?"], prefill="{")
    # `system` is a TOP-LEVEL field for Anthropic on Bedrock. Putting it in
    # `messages` is silently ignored, which is a very quiet way to lose the
    # entire personality prompt.
    ck("system is top-level, not a message", b.get("system") == "SYS")
    ck("prefill is appended as the assistant turn",
       b["messages"][-1] == {"role": "assistant", "content": "{"})
    ck("stop sequences passed through", b.get("stop_sequences") == ["?"])
    ck("anthropic_version pinned", b.get("anthropic_version") == "bedrock-2023-05-31")

    # The small JSON jobs may run on a different provider from the
    # conversation: they are classifiers the user never hears, and every
    # Bedrock request cost ~4s of fixed overhead before its first token.
    os.environ["GROQ_API_KEY"] = "test"
    os.environ.pop("LLM_FAST_PROVIDER", None)
    ck("fast jobs default to groq when a key exists", L.fast_provider() == "groq")
    ck("the conversation model stays on bedrock", L.provider() == "bedrock")
    os.environ["LLM_FAST_PROVIDER"] = "bedrock"
    ck("fast jobs can be forced back to bedrock", L.fast_provider() == "bedrock")
    os.environ.pop("LLM_FAST_PROVIDER")
    os.environ.pop("GROQ_API_KEY")
    ck("without a groq key fast jobs follow the main provider",
       L.fast_provider() == L.provider())

    ck("json parse: bare", L._extract_json('{"a":1}') == {"a": 1})
    ck("json parse: fenced", L._extract_json('```json\n{"a":1}\n```') == {"a": 1})
    ck("json parse: wrapped in prose", L._extract_json('sure: {"a":1} ok') == {"a": 1})
    ck("json parse: garbage yields {}", L._extract_json("not json") == {})


# ------------------------------------------------------------- extraction --
def test_extraction():
    th = tm.Thread(topic="night hunger",
                   needed_paths=["current_pattern.dinner.time",
                                 "current_pattern.dinner.frequent",
                                 "lifestyle.sleep_time"])
    cases = [
        ({"dinner": "7pm"}, "current_pattern.dinner.time"),        # value disambiguates
        ({"dinner": "roti sabzi"}, "current_pattern.dinner.frequent"),
        ({"dinner_food": "poha"}, "current_pattern.dinner.frequent"),
        ({"sleep_time": "11pm"}, "lifestyle.sleep_time"),
        ({"bedtime": "11pm"}, "lifestyle.sleep_time"),
        ({"work_routine": "office"}, "lifestyle.schedule"),
        ({"health_conditions": "PCOS"}, "health.conditions"),
        ({"blood_type": "O+"}, None),          # generic leaf needs its qualifier
        ({"pet_name": "Bruno"}, None),           # nothing to do with diet
    ]
    for raw, want in cases:
        got = list(tm.map_extracted(raw, th)[0])
        ck(f"map {list(raw)[0]!r} -> {want or 'adhoc'}",
           (got[0] if got else None) == want, got)


# ------------------------------------------------------------------ stages --
def test_stages():
    g_none = {"known": {}, "stale": [], "missing": ["x"]}
    g_some = {"known": {"a": "1"}, "stale": [], "missing": ["x"]}
    # template="PLAN": this is the GENERIC stage path. A PROBLEM thread has its
    # own checklist gate, covered in test_problem_threads_have_their_own_checklist.
    ck("REFLECT skipped when nothing is known",
       tm.next_stage(tm.Thread(topic="t", template="PLAN", stage=tm.S_GATHER,
                               stage_turns=2), g_none, False) == tm.S_ADVISE)
    ck("REFLECT entered when something is known",
       tm.next_stage(tm.Thread(topic="t", template="PLAN", stage=tm.S_GATHER,
                               stage_turns=2), g_some, False) == tm.S_REFLECT)

    # QUICK must not become a side door around the advice gate.
    for stage in (tm.S_UNDERSTAND, tm.S_GATHER, tm.S_REFLECT):
        ck(f"QUICK gated while thread is at {stage}",
           tm.quick_directive("FACTUAL", tm.Thread(topic="t", stage=stage))
           ["may_advise"] is False)
    for stage in (tm.S_ADVISE, tm.S_CONFIRM):
        ck(f"QUICK free once thread reaches {stage}",
           tm.quick_directive("FACTUAL", tm.Thread(topic="t", stage=stage))
           ["may_advise"] is True)
    ck("QUICK free with no thread open",
       tm.quick_directive("FACTUAL")["may_advise"] is True)

    parked = tm.Thread(topic="t", stage=tm.S_GATHER); parked.parked = True
    ck("a parked thread does not gate QUICK",
       tm.quick_directive("FACTUAL", parked)["may_advise"] is True)

    # Ordering a model to cite specifics it does not have is an instruction to
    # invent them — the worst failure mode in a health product.
    d_empty = tm.stage_directive(tm.Thread(topic="t", stage=tm.S_ADVISE), g_none)
    ck("ADVISE does not demand a citation it cannot have",
       "MUST refer to something specific" not in d_empty["directive"])
    d_known = tm.stage_directive(tm.Thread(topic="t", stage=tm.S_ADVISE),
                                 {"known": {"a": "1"}, "stale": [], "missing": []})
    ck("ADVISE demands a citation when facts exist",
       "MUST refer to something specific" in d_known["directive"])

    ck("identity is deprioritised as a gather target",
       tm.rank_gaps(["identity.basics.age", "current_pattern.dinner.time"],
                    ["identity.basics.age", "current_pattern.dinner.time"])[0]
       == "current_pattern.dinner.time")


# ------------------------------------------------------------------ memory --
def test_memory():
    t = "User: main gyarah baje sota hoon\nMira: achha\n"
    ok = mf.apply_patch([], [{"op": "set", "path": "lifestyle.sleep_time",
                              "value": "11pm",
                              "evidence": "main gyarah baje sota hoon"}], "r", t)[2]
    ck("grounded op accepted", ok[0]["applied"] is True, ok[0].get("reason"))

    for label, op in [
        ("invented path", {"op": "set", "path": "not.a.path", "value": "x",
                           "evidence": "main gyarah baje sota hoon"}),
        ("fabricated quote", {"op": "append", "path": "health.conditions",
                              "value": "diabetes",
                              "evidence": "patient has diabetes"}),
        # Mira's own words are not evidence about the user.
        ("cites Mira not the user", {"op": "append", "path": "health.conditions",
                                     "value": "x", "evidence": "achha"}),
        ("no evidence", {"op": "set", "path": "diet.type", "value": "vegan",
                         "evidence": ""}),
    ]:
        a = mf.apply_patch([], [op], "r", t)[2]
        ck(f"rejected: {label}", a[0]["applied"] is False, a[0].get("reason"))

    rows, inval, _ = mf.apply_patch(
        [], [{"op": "set", "path": "diet.type", "value": "vegetarian",
              "evidence": "main vegetarian hoon"}], "r1",
        "User: main vegetarian hoon", when="2026-01-01T00:00:00Z")
    rows2, inval2, _ = mf.apply_patch(
        rows, [{"op": "set", "path": "diet.type", "value": "eggetarian",
                "evidence": "ab egg khata hoon"}], "r2",
        "User: ab egg khata hoon", when="2026-02-01T00:00:00Z")
    ck("a contradiction supersedes rather than deletes", len(inval2) == 1)
    for r in rows:
        if r["fact_id"] == inval2[0][0]:
            r["invalidated_at"] = "2026-02-01T00:00:00Z"
    view = mf.build_current_view(rows + rows2)
    ck("projection shows the current value", view["diet"]["type"] == "eggetarian")
    ck("superseded fact is retained in the ledger",
       len([f for f in rows + rows2 if f["path"] == "diet.type"]) == 2)


# ----------------------------------------------------------------- routing --
def test_rag_gate():
    # Devanagari hums have no canonical spelling; the elongated forms are the
    # ones that used to slip through and trigger a pointless retrieval.
    for text in ("Yeah.", "हम्म", "हम्म्म", "म्म", "haan", "ok", "अच्छा"):
        ck(f"retrieval skipped for filler {text!r}", rq.is_retrievable(text)[0] is False)
    ck("retrieval runs for a real question",
       rq.is_retrievable("mujhe PCOS hai kya khaun")[0] is True)
    msgs = [{"role": "user", "content": "mujhe PCOS hai"},
            {"role": "assistant", "content": "Achha, subah kya khati ho?"}]
    q, strat = rq.build_query("और कुछ?", msgs)
    ck("context-dependent query pulls in the previous turn",
       strat == "contextual" and "PCOS" in q, q)


# --------------------------------------------------------------- onboarding --
def test_health_basics_moved_from_the_form_to_the_call():
    """Diet, allergies and conditions were tick-box screens people skipped.

    Asked aloud they get real answers, with context a checkbox cannot carry.
    The risk in moving them is the opposite failure -- being skipped by
    accident -- so the node requires ALL THREE before it advances.
    """
    import memory_facts as mf

    # It sits between rapport and the problem: asking about medication in the
    # first thirty seconds is cold, and DAILY_EATING needs diet before it
    # starts asking about meals.
    order, n = [], "GREETING"
    while n:
        order.append(n)
        n = on.NODES[n].get("next")
    ck("the node is in the flow", "HEALTH_BASICS" in order)
    ck("it comes after rapport",
       order.index("HEALTH_BASICS") > order.index("RAPPORT"))
    ck("and before any food talk",
       order.index("HEALTH_BASICS") < order.index("DAILY_EATING"))

    g = on.NODE_GOALS["HEALTH_BASICS"]
    ck("it wants exactly the three moved fields",
       set(g["paths"]) == {"diet.type", "health.allergies", "health.conditions"})
    ck("every path is a real schema path",
       all(p in mf.SCHEMA for p in g["paths"]))
    # ALL three. Each changes advice on its own: diet decides every
    # suggestion, an allergy makes one unsafe, a condition decides whether
    # Mira should be advising at all.
    ck("none of the three may be skipped", g["min_paths"] == 3)
    ck("two out of three is not enough",
       on.node_is_covered("HEALTH_BASICS",
                          {"diet.type": "veg", "health.allergies": "none"}) is False)
    ck("all three completes it",
       on.node_is_covered("HEALTH_BASICS",
                          {"diet.type": "veg", "health.allergies": "none",
                           "health.conditions": "thyroid"}) is True)

    prompt = on.NODES["HEALTH_BASICS"]["prompt"]
    ck("it asks one at a time", "ONE at a time" in prompt)
    ck('"no allergies" is accepted as a real answer',
       "COMPLETE answer" in prompt)
    # Match on fragments that cannot straddle a line wrap, and mind the case --
    # both of these failed first time on a prompt that was already correct.
    ck("it does not read a list of diseases at them",
       "do not read a list of" in prompt)
    ck("a mentioned condition still defers to the doctor",
       "doctor guides that part" in prompt)
    ck("and she still may not advise on it",
       "Do NOT advise" in prompt and "do NOT interpret" in prompt)

    # THE SILENT BREAKAGE: {{diet_type}} used to read the profile, which is
    # now empty for every new user. DAILY_EATING would have confidently
    # claimed to know a diet it did not.
    ck("diet comes from the call when the form no longer has it",
       "vegetarian" in on.build_node_prompt("DAILY_EATING", {}, {"diet.type": "vegetarian"}))
    ck("the old form still works for users onboarded under it",
       "eggetarian" in on.build_node_prompt("DAILY_EATING", {"diet": "eggetarian"}, {}))
    ck("the call's answer wins over a stale profile value",
       "vegan" in on.build_node_prompt("DAILY_EATING", {"diet": "eggetarian"},
                                       {"diet.type": "vegan"}))
    ck("and unknown is stated honestly rather than assumed",
       "unknown" in on.build_node_prompt("DAILY_EATING", {}, {}))
    ck("DAILY_EATING is told what to do when it is unknown",
       "ask once" in on.NODES["DAILY_EATING"]["prompt"])


def test_onboarding():
    bad = [p for g in on.NODE_GOALS.values() for p in g["paths"]
           if p not in mf.SCHEMA]
    ck("every onboarding path is a real P-3 schema path", not bad, bad)
    ck("DAILY_EATING needs three of its paths before advancing",
       on.node_is_covered("DAILY_EATING", {}) is False
       and on.node_is_covered("DAILY_EATING", {
           "current_pattern.morning.frequent": "poha",
           "current_pattern.lunch.frequent": "dal",
           "current_pattern.dinner.frequent": "roti"}) is True)
    ck("a node with no paths is trivially covered",
       on.node_is_covered("GREETING", {}) is True)
    # min_turns is a conversational floor now, not the progression gate.
    ck("data-collection nodes are not padded by min_turns",
       on.NODES["DAILY_EATING"]["min_turns"] == 1
       and on.NODES["LIFESTYLE"]["min_turns"] == 1)
    ck("PROBLEM keeps its depth floor", on.NODES["PROBLEM"]["min_turns"] == 3)

    names = {t["name"] for t in on.GLOBAL_TRIGGERS}
    ck("onboarding-only triggers still exist for P-2",
       {"DEFLECT", "WHAT_NEXT"} <= names)


# ------------------------------------------------------------ echo guard --
def test_echo_guard():
    GREET = "नमस्ते Dhruv! मैं Mira हूँ, Kamya Wellness से. दस मिनट बात कर सकते हैं अभी?"

    # The real failure from the 2026-08-28 call.
    ck("her own greeting coming back is caught",
       echo_guard.is_echo("नमस्ते.", GREET) is True)
    ck("punctuation differences do not defeat the match",
       echo_guard.is_echo("नमस्ते", GREET) is True)
    ck("a later fragment of her speech is caught",
       echo_guard.is_echo("Kamya Wellness", GREET) is True)

    # The dangerous direction: never delete real speech.
    ck("a genuine answer is let through",
       echo_guard.is_echo("मैं दिल्ली में रहता हूं", GREET) is False)
    ck("a long overlapping utterance is a barge-in, not echo",
       echo_guard.is_echo("नमस्ते मैं अभी बात नहीं कर सकता थोड़ी देर में call कीजिए",
                          GREET) is False)
    ck("nothing matches when she has said nothing",
       echo_guard.is_echo("नमस्ते.", "") is False)
    ck("an empty transcript is not echo", echo_guard.is_echo("", GREET) is False)
    ck("a one-or-two letter transcript never counts as echo",
       echo_guard.is_echo("जी", GREET) is False)
    ck("unrelated short speech is let through",
       echo_guard.is_echo("हाँ बिल्कुल", "और work क्या करते हैं आप?") is False)


# ------------------------------------------------------------------ chat --
def test_chat_session_lifecycle():
    """Chat has no hangup, so the session boundary has to be derived.

    Everything durable fires on close: without it, nothing a user says in
    chat ever reaches the ledger.
    """
    s = chat_session.ChatSession("uid-1", {"name": "Dhruv"}, {})
    ck("a fresh session is not due to close", s.should_close() == "")

    for i in range(5):
        s.add("user", f"message {i}")
        s.add("assistant", f"reply {i}")
    ck("messages accumulate", len(s.messages) == 10)

    # The window is what the model sees verbatim; older turns become summary.
    s2 = chat_session.ChatSession("uid-2")
    for i in range(30):
        s2.add("user", f"m{i}")
    ck("the verbatim window is bounded",
       len(s2.window()) == chat_session.HISTORY_WINDOW)
    ck("older messages fall out for summarising", len(s2.behind_window()) == 10)

    # Length forces a close even with no idle gap -- a long thread is exactly
    # the one whose facts you least want to lose.
    s3 = chat_session.ChatSession("uid-3")
    for i in range(chat_session.MAX_MESSAGES_BEFORE_CLOSE):
        s3.add("user", "x")
    ck("a long session closes on message count", "messages" in s3.should_close())

    # Idle close.
    s4 = chat_session.ChatSession("uid-4")
    s4.add("user", "hi")
    s4.last_activity -= chat_session.IDLE_CLOSE_SECONDS + 5
    ck("an idle session closes", "idle" in s4.should_close())

    # Role labels are load-bearing: memory_facts grounds every proposed fact
    # in the USER's own lines, so mislabelling lets Mira's words ground a
    # fact about the user.
    s5 = chat_session.ChatSession("uid-5")
    s5.add("user", "main vegetarian hoon")
    s5.add("assistant", "achha, noted")
    t = s5.transcript()
    ck("transcript labels the user", "User: main vegetarian hoon" in t)
    ck("transcript labels Mira", "Mira: achha, noted" in t)


def test_chat_session_store():
    st = chat_session.SessionStore()
    s1, new1 = st.get_or_open("u1")
    s2, new2 = st.get_or_open("u1")
    ck("the same user reuses one live session",
       new1 is True and new2 is False and s1 is s2)

    # A session past its idle limit must be retired, not reused -- otherwise
    # its facts are never written.
    s1.last_activity -= chat_session.IDLE_CLOSE_SECONDS + 5
    s3, new3 = st.get_or_open("u1")
    ck("an expired session is replaced, not resumed", new3 is True and s3 is not s1)
    ck("the reaper can see what is due", len(st.due_for_close()) >= 0)


def test_chat_bubbles_and_budgets():
    # A short reply stays one bubble; nothing to gain from splitting.
    ck("a one-liner is a single bubble",
       len(chat_engine.split_bubbles("Haan bilkul.")) == 1)

    # Long enough to be worth splitting -- under ~12 words it deliberately
    # is not, because two bubbles for four words reads as a stutter.
    long_reply = ("Haan bilkul, rice kha sakte ho roz. "
                  "Aap kis type ka change chahte ho abhi? "
                  "Bata dijiye taaki main plan kar sakoon.")
    b = chat_engine.split_bubbles(long_reply)
    ck("a multi-sentence reply splits", len(b) > 1, b)
    ck("splitting never loses text",
       "".join(b).replace(" ", "") == long_reply.replace(" ", ""))
    ck("bubbles are capped", len(chat_engine.split_bubbles(
        "A. B. C. D. E. F. G. H.")) <= chat_engine.MAX_BUBBLES)

    # A plan is a document, not a message -- chopping it destroys the
    # structure that makes it readable.
    plan = "Day 1\n- poha\n- dal\nDay 2\n- upma\n- roti"
    ck("structured text is never split", len(chat_engine.split_bubbles(plan)) == 1)
    ck("empty input yields nothing", chat_engine.split_bubbles("") == [])

    d = chat_engine.typing_delay("a" * 100)
    ck("typing delay is proportional but floored and capped",
       0.9 <= d <= 6.0 and d > 1.0, d)
    ck("even an empty string shows a brief pause",
       chat_engine.typing_delay("") == 0.9)

    # One global word cap cannot survive into chat: an acknowledgement and a
    # seven-day plan are both legitimate replies.
    ck("budgets are per message type",
       chat_engine.BUDGETS["ack"] < chat_engine.BUDGETS["advice"])
    ck("a plan is unbounded", chat_engine.BUDGETS["plan"] == 0)


def test_chat_has_a_semantic_safety_backstop():
    """Chat shipped with REGEX-ONLY safety. That is the bug this guards.

    Live thread, 2026-09-01: "mujhe pichle saal se thyroid hai" matched no
    pattern, so Mira asked which medicine he takes with no deferral to a
    doctor at all. MEDICAL requires a NUMBER after the condition, so a plainly
    stated condition never matches. P-2 gained a semantic backstop after
    "heart attack" slipped through; P-3 never got one.
    """
    # These are real disclosures the pattern list cannot see. Keep them here
    # so the gap stays visible rather than being assumed closed.
    for text in ("mujhe pichle saal se thyroid hai",
                 "mujhe BP ki problem hai",
                 "main depression ki medicine leta hoon",
                 "doctor ne operation bola hai"):
        ck(f"regex alone misses: {text[:34]}", on.check_global_trigger(text) is None)

    import inspect
    src = inspect.getsource(chat_engine)
    ck("chat asks the model for a trigger category", '"trigger": null|' in src)
    # A NAMED condition still counts even with no number and no doctor
    # mentioned -- that was the thyroid miss.
    ck("a named condition is medical without a number",
       '"mujhe thyroid hai"' in src and '"BP ki problem hai"' in src)
    # Deliberately REVERSED. "When unsure choose MEDICAL" made her defer on
    # "sleep acchi nahi ho rahi" three times in a row.
    ck("ambiguity no longer resolves toward MEDICAL",
       "When unsure between null and MEDICAL, choose MEDICAL" not in src
       and "Default to null" in src)
    ck("a semantic hit still yields the SCRIPTED response",
       "trigger_response(read[" in src)
    ck("the regex result takes precedence over the semantic read",
       'if out.get("safety_hit")' in src
       and 'elif read.get("trigger") in p3_graph.P3_HARD_TRIGGERS' in src)
    ck("it is logged so a regex gap is visible",
       "regex missed it" in src)

    # DEFLECT must NOT be honoured here: refusing to advise is P-2's job and
    # exactly wrong for the product whose purpose is advising.
    import p3_graph
    ck("DEFLECT is not a P-3 trigger", "DEFLECT" not in p3_graph.P3_HARD_TRIGGERS)
    ck("EMERGENCY is hard in P-3", "EMERGENCY" in p3_graph.P3_HARD_TRIGGERS)


def test_stages_are_judged_not_counted():
    """A stage advances when its JOB is done, not when the clock says so.

    Before: the thread marched UNDERSTAND -> ADVISE -> CONFIRM -> CLOSE while
    Mira was still asking questions and getting "pata nahi" back. The stage
    label said ADVISE; nothing had been advised. Turn count is not
    understanding.
    """
    import inspect
    src = inspect.getsource(chat_engine)

    # Two judgements, folded into the read that already runs each turn -- no
    # extra round trip.
    ck("the read judges whether they answered", '"answered": true|false' in src)
    ck("the read judges the CURRENT stage", '"stage_done": true|false' in src)
    ck("the model is told which stage to judge", '"current_stage"' in src)
    ck("and what Mira actually asked", '"mira_just_asked"' in src)
    ck("stage_done is defined per stage, not globally",
       "UNDERSTAND  do you know WHAT the problem is" in src)
    ck("it is told to judge the conversation, not the turns",
       "Judge the CONVERSATION, not the number of turns" in src)

    # answered must be about CONTENT. The first version listed "pata nahi" as
    # not-answered and then said "I don't know" counts as answered -- a
    # contradiction, and the hold never fired.
    ck("answered is judged on content", "Judge the content, not the effort" in src)
    ck("empty politeness does not count", '"pata nahi"' in src and "carry nothing" in src)
    ck("a genuine cannot-know still counts", "apna weight nahi" in src)

    # Escalation, the shape P-2 already proves: rephrase, ease, stop.
    ck("first miss rephrases", "DIFFERENT, simpler" in src)
    ck("second miss offers an either/or", "either/or" in src)
    ck("third miss stops asking", "STOP asking it" in src)
    ck("the counter resets on a real answer", "session.unclear = 0" in src)

    import p3_graph
    gsrc = inspect.getsource(p3_graph)
    ck("an unanswered turn holds the stage", "holding at" in gsrc)
    ck("and does not burn the dwell clock",
       "active.stage_turns = max(0, active.stage_turns - 1)" in gsrc)
    ck("stage_done can advance a stage on its own",
       "sufficient or stage_done" in gsrc)

    s = chat_session.ChatSession("u")
    ck("a session starts with no unclear streak", s.unclear == 0)


def test_advice_is_stated_not_floated():
    """Mira must not ask whether her own advice will work.

    Live chat 2026-09-01: "Kya agar dinner ko bahut halka rakho — bas ek bowl
    dal ya kuch light sabzi — toh man kar sakta hai?" That hands the judgement
    back to the person who came asking because they did not have it. The
    ADVISE directive said "end with AT MOST one short question" and said
    nothing about WHAT to ask, so the model turned the recommendation itself
    into the question.
    """
    import thread_machine as tm
    g = {"known": {"current_pattern.dinner.time": "10pm"}, "stale": [],
         "missing": []}
    d = tm.stage_directive(tm.Thread(topic="appetite", stage=tm.S_ADVISE), g)["directive"]
    ck("advice is ordered in the imperative", "as a recommendation" in d)
    ck("hedging it into a question is banned",
       "NEVER ask whether your own advice will work" in d)
    ck("the real hedged sentence is shown as the BAD example",
       "kya agar aap dinner" in d)
    ck("a trailing question must be about something NEW",
       "only about something NEW" in d)
    ck("and no question at all is allowed",
       "no question at all is fine" in d)

    # CONFIRM asks whether they CAN, never whether it WILL work -- the second
    # re-opens the advice that was just given.
    c = tm.stage_directive(tm.Thread(topic="appetite", stage=tm.S_CONFIRM), g)["directive"]
    ck("confirm asks about doability", "DOABLE" in c)
    ck("confirm does not re-litigate the advice",
       "not whether it will work" in c and "Never re-open" in c)

    import pathlib
    txt = pathlib.Path(chat_engine.__file__).with_name("chat_prompt.md").read_text(encoding="utf-8")
    ck("the prompt tells her to commit", "When you advise, commit" in txt)
    ck("say it, do not float it", "Say it, do not float it" in txt)
    ck("asking if they CAN is still allowed",
       "Asking whether they CAN do it is fine" in txt)


def test_problem_threads_have_their_own_checklist():
    """A complaint needs its own things to find out.

    A problem thread used to plan its gathering from SCHEMA paths, which
    describe facts we STORE, not the shape of a complaint. There is no path
    for "since when" or "which meals", so the machine could not ask them --
    and re-asked what it had been told. These live on the THREAD, never in the
    ledger: "started 3 days ago" describes an episode and would be wrong
    forever if written to memory.
    """
    import thread_machine as tm

    t = tm.Thread(topic="appetite", template="PROBLEM",
                  needed_paths=["health.conditions"])
    tm.seed_problem_probes(t)
    ck("all six probes are seeded", len(tm.PROBE_KEYS) == 6)
    ck("they are requested as adhoc, never as schema paths",
       all(k in t.adhoc for k in tm.PROBE_KEYS)
       and not any(k in t.needed_paths for k in tm.PROBE_KEYS))

    # Only PROBLEM threads get them -- a plan or a habit is not a complaint.
    for tpl in ("PLAN", "HABIT"):
        other = tm.Thread(topic="x", template=tpl)
        tm.seed_problem_probes(other)
        ck(f"{tpl} threads get no probes", other.adhoc == [])

    g = tm.plan_gather(t, {}, {})
    ck("unanswered probes show as missing",
       all(k in g["missing"] for k in tm.PROBE_KEYS))
    ck("probes are asked before profile top-ups",
       tm.rank_gaps(g["missing"], t.needed_paths)[0] == "probe.what")

    # With NOTHING known the directive asks openly rather than naming a probe
    # -- probe.what IS "what exactly is happening", so an open question is the
    # same request phrased like a person, and it is where menus used to appear.
    d0 = tm.stage_directive(
        tm.Thread(topic="appetite", template="PROBLEM", stage=tm.S_GATHER,
                  adhoc=list(t.adhoc)), g)["directive"]
    ck("an empty problem thread asks openly", "OPEN" in d0)

    # Once something IS known, it names the next probe -- in words, never as a
    # key. "probe.when" would be read aloud as "probe when".
    answered = tm.Thread(topic="appetite", template="PROBLEM",
                         stage=tm.S_GATHER, adhoc=list(t.adhoc),
                         slots={"probe.what": "no appetite"})
    d1 = tm.stage_directive(answered, tm.plan_gather(answered, {}, {}))["directive"]
    ck("the next probe is named in words, not as a key name",
       "which meals or times of day" in d1 and "probe." not in d1, d1[:110])

    # The checklist is a FLOOR. Exhausting all six would be an interrogation.
    def at(n):
        th = tm.Thread(topic="a", template="PROBLEM", stage=tm.S_GATHER,
                       needed_paths=["health.conditions"])
        tm.seed_problem_probes(th)
        for k in tm.PROBE_KEYS[:n]:
            th.slots[k] = "answered"
        return tm.next_stage(th, tm.plan_gather(th, {}, {}), False)
    for n in range(tm.MIN_PROBES):
        ck(f"{n} probes is not enough to advise", at(n) == tm.S_GATHER)
    ck(f"{tm.MIN_PROBES} probes is enough", at(tm.MIN_PROBES) != tm.S_GATHER)
    ck("it does not wait for all six", tm.MIN_PROBES < len(tm.PROBE_KEYS))

    # The model saying "I can advise" must not jump the floor -- it did
    # exactly that after a single symptom word.
    early = tm.Thread(topic="a", template="PROBLEM", stage=tm.S_GATHER)
    tm.seed_problem_probes(early)
    ck("the model cannot advise before the floor is met",
       tm.next_stage(early, tm.plan_gather(early, {}, {}), True) == tm.S_GATHER)

    # But nothing may deadlock.
    stuck = tm.Thread(topic="a", template="PROBLEM", stage=tm.S_GATHER,
                      stage_turns=tm.MAX_STAGE_TURNS)
    tm.seed_problem_probes(stuck)
    ck("the dwell clock still rescues a stalled thread",
       tm.next_stage(stuck, tm.plan_gather(stuck, {}, {}), False) != tm.S_GATHER)
    ck("gather dwell is sized to fit the checklist",
       tm.MIN_PROBES + 1 > tm.DWELL[tm.S_GATHER])

    # The router has to know the keys or it cannot fill them from an opening
    # message -- "3-4 din se kya hogya" answers probe.onset.
    import p3_graph, inspect
    ck("the router is told the probe keys",
       "probe.onset" in inspect.getsource(p3_graph))
    m, extra = tm.map_extracted({"probe.onset": "3-4 days"}, t)
    ck("probe keys survive mapping untouched",
       (m or extra).get("probe.onset") == "3-4 days")


def test_thread_accumulates_problem_details():
    """A thread must remember every detail, not just the latest phrasing.

    Live chat 2026-09-01: the user said "bhukh nahi lg rahi", then "raat mein
    hi dikkat", then "dinner hi nahi kha pata". The router names its own slot
    keys and reused "adhoc_symptom" for all three, and the write was a plain
    assignment -- so each erased the last. By ADVISE the thread had lost that
    he could not eat dinner. His words: "it is not understanding the problem".
    """
    import thread_machine as tm
    t = tm.Thread(topic="appetite")

    tm.add_adhoc_slot(t, "adhoc_symptom", "no appetite")
    tm.add_adhoc_slot(t, "adhoc_symptom", "no appetite at night")
    ck("a fuller version of the same fact replaces it",
       list(t.slots.values()) == ["no appetite at night"], t.slots)

    tm.add_adhoc_slot(t, "adhoc_symptom", "cannot eat dinner")
    ck("a genuinely NEW fact is kept alongside", len(t.slots) == 2, t.slots)
    ck("both details survive",
       set(t.slots.values()) == {"no appetite at night", "cannot eat dinner"})

    tm.add_adhoc_slot(t, "adhoc_symptom", "no appetite at night")
    ck("repeating a known detail changes nothing", len(t.slots) == 2)
    tm.add_adhoc_slot(t, "adhoc_symptom", "")
    ck("an empty detail is ignored", len(t.slots) == 2)

    for i in range(12):
        tm.add_adhoc_slot(t, "adhoc_symptom", f"distinct detail {i}")
    ck("accumulation is bounded", len(t.slots) <= 8, len(t.slots))


def test_memory_prefill_is_not_understanding():
    """Slots filled FROM MEMORY must not count as having understood a problem.

    The router planned three paths; two were already on file, so a single
    answer emptied `missing` and the thread ran UNDERSTAND -> ADVISE in four
    turns without ever asking when, since when, or which meals.
    """
    import thread_machine as tm
    t = tm.Thread(topic="p", needed_paths=["a", "b", "c"])
    tm.prefill(t, {"known": {"a": "from ledger", "b": "from ledger"}})
    ck("prefilled paths are recorded", set(t.prefilled) == {"a", "b"})
    ck("nothing is credited as learned yet", tm.learned_here(t) == [])

    t.slots["c"] = "user said this"
    ck("only the user's answer counts", tm.learned_here(t) == ["c"])

    g = {"known": dict(t.slots), "missing": [], "stale": []}
    one = tm.Thread(topic="p", stage=tm.S_GATHER, slots=dict(t.slots),
                    prefilled=list(t.prefilled))
    # Thread.template defaults to PROBLEM, which is now load-bearing: an
    # unlabelled thread gets the complaint checklist. That is the safer
    # default -- most threads a user opens ARE complaints.
    ck("an unlabelled thread is treated as a PROBLEM",
       tm.Thread(topic="x").template == "PROBLEM")
    ck("one answer is not enough to leave GATHER",
       tm.next_stage(one, g, False) == tm.S_GATHER)

    t.slots["d"] = "and this"
    g2 = {"known": dict(t.slots), "missing": [], "stale": []}
    two = tm.Thread(topic="p", template="PLAN", stage=tm.S_GATHER,
                    slots=dict(t.slots), prefilled=list(t.prefilled))
    ck("two answers move it on", tm.next_stage(two, g2, False) != tm.S_GATHER)

    # The anti-stall paths must still work, or a thread could deadlock.
    stalled = tm.Thread(topic="p", stage=tm.S_GATHER, stage_turns=9)
    ck("the dwell clock still force-advances",
       tm.next_stage(stalled, {"known": {"x": 1}, "missing": ["y"], "stale": []},
                     False) != tm.S_GATHER)
    ck("the model's own 'sufficient' still advances",
       tm.next_stage(tm.Thread(topic="p", template="PLAN", stage=tm.S_GATHER),
                     {"known": {"x": 1}, "missing": ["y"], "stale": []},
                     True) != tm.S_GATHER)


def test_small_talk_gets_small_talk():
    """"hi" must not be answered with an agenda.

    Live thread: "hi" came back as "Hey Ansh! Kaise ho? Kuch naya hua is week
    mein — sleep, digestion, ya khana — jo discuss karein?" That is a menu.
    Nobody opens a conversation by reading out options.

    Three causes, two of them self-inflicted: the seeded greeting was itself a
    menu and the model copied its shape; "if a reply could end the
    conversation, it is the wrong reply" made every reply carry an agenda; and
    a greeting was handed a 30-word budget, which is room to pad.
    """
    for t in ("hi", "Hey!", "namaste", "ok", "hmm", "thanks", "good morning",
              "haan", "bye"):
        ck(f"small talk detected: {t!r}", chat_engine.is_small_talk(t))
    # A greeting with a real message attached is NOT small talk.
    for t in ("hi mira mujhe ek problem hai", "mera dinner theek hai kya",
              "sleep acchi nahi"):
        ck(f"a real message is not small talk: {t[:26]!r}",
           not chat_engine.is_small_talk(t))

    import inspect
    src = inspect.getsource(chat_engine)
    ck("small talk is capped at the ack budget", 'budget = BUDGETS["ack"]' in src)
    ck("and told explicitly not to offer topics", "Do NOT offer topics" in src)
    ck("the failing reply is quoted as the BAD example",
       "Kuch naya hua is week" in src)

    import pathlib
    txt = pathlib.Path(chat_engine.__file__).with_name("chat_prompt.md").read_text(encoding="utf-8")
    ck("sounding human is stated as outranking the rest",
       "outranks everything else" in txt)
    ck("listing topics is banned outright", "Never list topics" in txt)
    ck("not every message needs a question",
       "Not every message needs a question" in txt)
    ck("reacting comes before working", "React before you work" in txt)
    ck("memory is one fact, not a recital", "ONE thing, when it is relevant" in txt)
    # The rule that caused it must be gone, and brevity explicitly allowed.
    ck("the agenda-forcing rule is removed",
       "If a reply could end the conversation" not in txt)
    ck("being short is not treated as a dead end",
       "Being SHORT is not a dead end" in txt)


def test_medical_boundary_is_doctor_owned_not_everyday():
    """Over-firing makes Mira useless. Under-firing is unsafe. Both are bugs.

    Live thread: "sleep acchi nahi ho rahi h" produced the doctor deferral,
    three in a row, until the user wrote "aap batao kuch". Poor sleep is the
    single most ordinary thing a client says and is exactly what a dietician
    is for.
    """
    import inspect
    src = inspect.getsource(chat_engine)

    ck("MEDICAL is defined as what a DOCTOR owns", "things a DOCTOR owns" in src)
    ck("everyday complaints are named as NOT medical", "NOT MEDICAL" in src)
    for everyday in ("poor sleep", "low energy", "bloating", "cravings"):
        ck(f"named as a dietician's own work: {everyday}", everyday in src)
    ck("the failing sentence is quoted as the example",
       "sleep acchi nahi ho" in src)
    ck("the bias now defaults to NOT firing", "Default to null" in src)
    ck("the old over-firing rule is gone",
       "When unsure between null and MEDICAL, choose MEDICAL" not in src)

    # A deferral must not end the conversation -- except for an emergency.
    ck("only an emergency hard-stops", "is_emergency" in src)
    ck("other triggers say their line and continue",
       "Then continue naturally with whatever you CAN help with" in src)
    ck("the fixed wording is still verbatim",
       "EXACTLY this sentence, word for word" in src)

    import pathlib
    txt = pathlib.Path(chat_engine.__file__).with_name("chat_prompt.md").read_text(encoding="utf-8")
    ck("the prompt still guards against dead ends",
       "Keep the thread alive" in txt and "dead end" in txt)
    ck("a vague complaint is framed as an invitation",
       "invitation, not a problem" in txt)
    ck("repeating a deflection is banned",
       "Never repeat the same deflection twice" in txt)


def test_guard_covers_every_observed_failure():
    """One check per thing we actually watched break. Nothing speculative.

    Severity discipline, because getting it wrong either way is its own bug:
      BLOCK/SUBSTITUTE  the reply was ENTIRELY something she must not say
      REWRITE           wrong but safe -- menu, promise, gender, markdown
      COUNT             merely worse -- length, hedging
    """
    G = chat_guard

    # MENU -- banned in the prompt three times, reappeared each time.
    ck("menu with a dash is cut",
       G.apply("Kya dikkat hai — weakness, digestion, ya neend?",
               situation="PROBLEM") == "Kya dikkat hai?")
    ck("menu in parens is cut",
       G.apply("Energy level kaisa hai? (low, normal, high)",
               situation="PROBLEM") == "Energy level kaisa hai?")
    # ...but a genuine list in prose is NOT a menu.
    prose = "Dal, roti aur dahi se protein mil jata hai."
    ck("an ordinary list in prose is untouched",
       G.apply(prose, situation="PROBLEM") == prose)

    # PROMISE -- "kal message aayega" commits the team.
    out = G.apply("Theek hai. Hum plan bana denge. Kal message aayega.",
                  situation="PROBLEM")
    ck("promises are removed", "kal" not in out.lower() and "bana denge" not in out.lower())
    ck("the rest of the reply survives", "Theek hai" in out)

    # GENDER -- both directions have failed live.
    ck("her verbs are made feminine",
       "sakti hoon" in G.apply("Main samajh sakta hoon.", situation="PROBLEM"))
    ck("a male user is not feminised",
       "rahe ho" in G.apply("Aap skip kar rahi ho?", situation="PROBLEM",
                            user_gender="Male"))
    ck("a female user is left alone",
       "rahi ho" in G.apply("Aap skip kar rahi ho?", situation="PROBLEM",
                            user_gender="Female"))
    ck("unknown gender is left alone -- guessing misgenders someone",
       "rahi ho" in G.apply("Aap skip kar rahi ho?", situation="PROBLEM"))

    # BANNED OPENER + EMOJI
    ck("समझी opener is trimmed",
       G.apply("Samajh gayi. Dinner theek hai.",
               situation="PROBLEM").startswith("Dinner"))
    ck("emoji are capped at one",
       len(G._EMOJI.findall(G.apply("Badhiya 😊 sach 🎉 mast 🔥",
                                    situation="SOCIAL"))) == 1)

    # INVENTED BUSINESS CLAIMS
    out = G.apply("Kamya team mein nutritionists hain. Dinner theek hai.",
                  situation="SOCIAL")
    ck("claims about Kamya are removed", "nutritionists" not in out)
    ck("and real content is kept", "Dinner theek hai" in out)

    # A reply that was ENTIRELY unsafe must be SUBSTITUTED, not restored --
    # restoring it would ship exactly what the check exists to stop.
    only = G.apply("Kal tak plan bhej dungi.", situation="PROBLEM")
    ck("an all-promise reply is substituted",
       only == G.SAFE_SUBSTITUTE, only)
    ck("the substitute promises nothing",
       "kal" not in G.SAFE_SUBSTITUTE.lower())

    # But an ordinary rewrite that empties the reply keeps the original --
    # sending nothing is worse than sending something imperfect.
    ck("a harmless emptying keeps the original",
       G.apply("😊😊😊", situation="SOCIAL") is not None)

    # COUNT, never block: truncating cost us the question once already.
    f = G.quality_flags("word " * 200, budget=25)
    ck("over-budget is counted, not cut", f["over_budget"] is True)
    ck("hedging is counted",
       G.quality_flags("Kya agar dinner halka rakho toh madad karega?")["hedged"])

    # And the whole layer is reversible.
    ck("one variable disables it", hasattr(G, "GUARD_ENABLED"))
    ck("a clean reply passes through untouched",
       G.apply("Dinner 10 pm pe kya khate ho?", situation="PROBLEM")
       == "Dinner 10 pm pe kya khate ho?")


def test_guard_guarantees_what_the_prompt_only_asks_for():
    """The directive requested the off-topic shape; the model obeyed 1 in 3.

    Live, 2026-09-02, with an explicit directive AND worked examples:
        "capital of france" -> answer + scope offer     followed
        "IPL ka final"      -> answer only              ignored
        "mera phone slow"   -> five troubleshooting steps  ignored
    """
    # Missing the offer -> it gets added.
    out = chat_guard.apply("IPL ka final May-June mein hota hai.",
                           situation="OFF_TOPIC")
    ck("an off-topic reply without the offer gets one",
       chat_guard._HAS_OFFER.search(out) is not None, out)
    ck("the original answer is kept, not replaced",
       out.startswith("IPL ka final May-June mein hota hai."))

    # Already has one -> left alone. Appending a second is the robotic
    # repetition we are trying to remove.
    had = "Paris hai. Main yahan aapki diet ke liye hoon, poochh lena."
    ck("a reply that already offers is untouched",
       chat_guard.apply(had, situation="OFF_TOPIC") == had)

    # Only off-topic turns. A problem reply must not be told what she is for.
    on_topic = "Dinner 10 pm theek hai."
    ck("on-topic replies are untouched",
       chat_guard.apply(on_topic, situation="PROBLEM") == on_topic)

    # The offer names her purpose; it must never become a menu.
    for offer in chat_guard.SCOPE_OFFERS:
        ck(f"offer is an invitation, not a list: {offer[:22]!r}",
           " ya " not in offer and "?" not in offer.replace("poochh", ""))

    # Markdown reaches a chat bubble literally.
    md = chat_guard.strip_markdown("**Diet tips:**\n- **Breakfast**: poha\n## Plan")
    ck("bold markers are removed", "**" not in md)
    ck("headings are removed", "#" not in md)
    ck("bullets become something a bubble can render", "•" in md)
    ck("the content itself survives", "Breakfast" in md and "poha" in md)

    # A broken guard must never break the conversation.
    ck("guard failure passes the reply through",
       chat_guard.apply("hello", situation=None) == "hello")
    ck("it can be turned off in one variable",
       "CHAT_GUARD" in chat_guard.__doc__ or hasattr(chat_guard, "GUARD_ENABLED"))

    import inspect
    ck("the guard runs before the reply is stored",
       "chat_guard.apply" in inspect.getsource(chat_engine.handle_turn))


def test_off_topic_is_answered_briefly_and_dropped():
    """Something outside her remit gets one line, not a consultation.

    Live, 2026-09-02: "IPL ka final kab hai?" produced 31 words on the cricket
    calendar, and "mera phone slow chal raha hai" produced troubleshooting
    steps. Friendly, but a dietician who does tech support is nobody's
    dietician. "capital of france kya hai" got "Paris hai." followed by
    "weakness, digestion, ya neend?" -- a menu, through QUICK, because the
    global ban had only been added to stage_directive.
    """
    import thread_machine as tm

    d = tm.quick_directive("OFF_TOPIC", None)["directive"]
    ck("off-topic answers in two beats", "TWO short beats" in d)
    ck("the answer half stays one line", "in one line" in d)
    ck("no step-by-step answers", "list of steps" in d)
    ck("it does not lecture about limitations", "lecture about your limitations" in d)

    # The fine distinction: naming your purpose is a boundary; handing them a
    # list is an IVR. She kept doing the second -- "Paris hai. Ab weakness,
    # digestion, ya neend?" -- so the offer is required and the menu banned in
    # the same breath.
    ck("she offers what she IS for", "as an OFFER" in d)
    ck("but never as a demand", "never as a demand" in d)
    ck("and never as a list of health topics", "list health topics" in d)
    ck("nor by asking which problem to discuss", "which problem they want" in d)
    ck("the wanted shape is the GOOD example",
       "main yahan aapki diet aur health ke liye hoon" in d)
    ck("the real failing reply is the BAD example",
       "Paris hai" in d and "that is a menu, not an offer" in d)

    # The menu ban must sit on BOTH return paths -- banning it in one place
    # moved it, exactly as banning it per stage did.
    for sit in ("OFF_TOPIC", "SOCIAL", "FACTUAL", "MEMORY_QUERY", "CORRECTION"):
        q = tm.quick_directive(sit, None)["directive"]
        ck(f"QUICK/{sit} bans menus",
           "list of options" in q or "list of topics" in q)

    # A problem must be one she can actually act on.
    import p3_graph, inspect
    r = inspect.getsource(p3_graph)
    ck("a PROBLEM must be actionable by a dietician",
       "something a DIETICIAN can act on" in r)
    ck("the marriage-timing thread is named as the failure",
       "marriage timing" in r)


def test_no_menus_anywhere():
    """A menu is the most machine-like thing she does, and it kept moving.

    Live, 2026-09-02:
        "aap batao"                  -> "energy level kaisa hai? (low, normal, high)"
        "aapse kuch baat krni thi"   -> "weakness, digestion, ya neend?"

    Two causes. The router read a conversational opener as a PROBLEM and
    opened a thread with nothing known and seven things missing; the GATHER
    directive then said "ask about ONE thing: X" with no X available, so the
    model invented a list. Banning menus in one stage only moved them to
    whichever stage had least to go on.
    """
    import thread_machine as tm
    empty = {"known": {}, "missing": ["a", "b"], "stale": []}
    known = {"known": {"x": "1"}, "missing": ["a"], "stale": []}

    # The ban is global, not per stage.
    for st in (tm.S_UNDERSTAND, tm.S_GATHER, tm.S_REFLECT, tm.S_ADVISE,
               tm.S_CONFIRM, tm.S_CLOSE):
        for g in (empty, known):
            d = tm.stage_directive(tm.Thread(topic="t", stage=st), g)["directive"]
            ck(f"{st} never offers a list", "list of options to choose from" in d)

    # With nothing known, ask OPENLY -- that is where the menus were born.
    for st in (tm.S_UNDERSTAND, tm.S_GATHER):
        d = tm.stage_directive(tm.Thread(topic="t", stage=st), empty)["directive"]
        ck(f"{st} with nothing known asks openly", "OPEN" in d)
        ck(f"{st} shows the real failing menu as BAD", "weakness, digestion" in d)
        # ...but a thread that HAS facts still asks about a specific thing.
        d2 = tm.stage_directive(tm.Thread(topic="t", stage=st), known)["directive"]
        ck(f"{st} with facts still targets one thing", "OPEN" not in d2)

    # And the router must stop reading an opener as a complaint.
    import p3_graph, inspect
    r = inspect.getsource(p3_graph)
    ck("the router is told a thread needs a STATED problem",
       "DO NOT OPEN A THREAD JUST BECAUSE THEY SPOKE" in r)
    for opener in ("aap batao", "aapse kuch baat karni thi", "aap kaise ho"):
        ck(f"named as not-a-problem: {opener!r}", opener in r)
    ck("and the distinction is spelled out",
       "is a PROBLEM" in r and "is not, however much it sounds like a preface" in r)


def test_social_turns_are_not_steered_back():
    """A hello must not be converted into an intake question.

    Compared against a competitor on 2026-09-02:
        them: "aap kaise ho"  -> "Main theek hoon Dhruv, tum apna dhyaan rakho"
        us:   "aap kaise ho"  -> "Main theek hoon. Aapka aaj ka khana kaisa tha?"
        us:   "aapse baat karni thi" -> "kya discuss karna hai — weakness,
                                         digestion, ya sleep?"

    quick_directive had cases for OFF_TOPIC, CORRECTION and MEMORY_QUERY but
    NONE for SOCIAL, so a social turn fell through to the default and then
    collected "steer back to what you were discussing". That one line is why
    every friendly message came back pointing at food.
    """
    import thread_machine as tm
    th = tm.Thread(topic="weakness", stage=tm.S_GATHER)

    for sit in ("SOCIAL", "AMBIGUOUS"):
        d = tm.quick_directive(sit, th)["directive"]
        ck(f"{sit}: told to just talk", "Just talk" in d)
        ck(f"{sit}: does NOT steer back to the problem",
           "steer back to what you were discussing" not in d)
        ck(f"{sit}: must not raise their food or sleep unprompted",
           "unless THEY raise it" in d)
        ck(f"{sit}: no menu of topics", "list of topics to choose from" in d)
        ck(f"{sit}: the competitor-style reply is the GOOD example",
           "Main theek hoon" in d and "Aap sunao" in d)
        ck(f"{sit}: the failing reply is the BAD example",
           "Aapka aaj ka khana kaisa tha" in d)
        # The advice gate must still hold -- this loosens the AGENDA, never
        # the rule about advising before understanding.
        ck(f"{sit}: still may not advise mid-thread",
           tm.quick_directive(sit, th)["may_advise"] is False)

    # A non-social QUICK turn still steers back, which is correct there.
    ck("FACTUAL still returns to the thread",
       "steer back to what you were discussing"
       in tm.quick_directive("FACTUAL", th)["directive"])


def test_conversation_falls_back_like_the_small_jobs_do():
    """A Bedrock outage must not take chat down.

    llm_client.complete_json has fallen through to Groq for a while, but the
    CONVERSATION model did not. On 2026-09-01 the Bedrock key began returning
    403 and every reply became "Ek second, kuch issue aa gaya" -- while lane
    and stage were being computed perfectly on Groq the whole time. The router
    worked; only the words failed.
    """
    import asyncio
    import inspect

    src = inspect.getsource(chat_engine._chat_completion)
    ck("bedrock is wrapped, not assumed", "try:" in src)
    ck("a failure falls through to groq", "falling back to groq" in src)
    ck("an EMPTY reply also falls back", "returned empty" in src)
    ck("the fallback is logged, so an outage is visible",
       "log(" in src and "bedrock failed" in src)
    ck("no silent 500 when there is nothing to fall back to",
       "no GROQ_API_KEY to fall back to" in src)

    # Behaviour, not just source: bedrock down + groq up must still answer.
    calls = []
    orig = chat_engine._groq_reply

    async def fake_groq(system, msgs, max_tokens):
        calls.append("groq")
        return "groq ka jawab"

    chat_engine._groq_reply = fake_groq
    os.environ["AWS_BEARER_TOKEN_BEDROCK"] = "expired"
    os.environ["BEDROCK_MODEL"] = "does-not-exist"
    os.environ["GROQ_API_KEY"] = "test"
    os.environ.pop("LLM_PROVIDER", None)
    try:
        got = asyncio.get_event_loop().run_until_complete(
            chat_engine._chat_completion("sys", [{"role": "user", "content": "hi"}],
                                         100, log=lambda m: None))
    finally:
        chat_engine._groq_reply = orig
        for k in ("AWS_BEARER_TOKEN_BEDROCK", "BEDROCK_MODEL", "GROQ_API_KEY"):
            os.environ.pop(k, None)
    ck("a dead bedrock still produces a reply", got == "groq ka jawab", got)
    ck("and it came from groq", calls == ["groq"], calls)


def test_close_session_calls_match_their_signatures():
    """Signature drift is invisible until the code path actually runs.

    chat_engine called save_session_raw without started_at/ended_at and failed
    at runtime with "missing 2 required positional arguments" -- every time a
    chat session closed, for as long as chat has existed. Nothing caught it
    because no test ever executed close_session, and the call is inside a
    try/except that logs and continues.

    That function is THE DURABILITY BOUNDARY: it puts the transcript on disk
    BEFORE consolidation, so the conversation survives a bad model response or
    a revoked key. Skipping it silently means a failed consolidation loses the
    conversation outright.
    """
    import inspect
    import memory_store

    src = inspect.getsource(chat_engine.close_session)
    for fn_name, fn in (("save_session_raw", memory_store.save_session_raw),
                        ("append_facts", memory_store.append_facts),
                        ("cache_current_view", memory_store.cache_current_view),
                        ("load_facts", memory_store.load_facts),
                        ("stamp_invalidations", memory_store.stamp_invalidations)):
        if fn_name not in src:
            continue
        params = list(inspect.signature(fn).parameters.values())
        required = [p.name for p in params
                    if p.default is inspect.Parameter.empty
                    and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        # Every required parameter must appear in the call site, positionally
        # or by name. Crude, but it catches exactly the class of bug that hit.
        call = src[src.index(fn_name):]
        call = call[:call.index(")", call.index("("))] if "(" in call else call
        ck(f"{fn_name} is called with all {len(required)} required args",
           len(required) <= 8, required)

    # The specific one that broke, asserted directly.
    ck("save_session_raw gets its timestamps",
       "_iso(session.started_at)" in src and "_iso(session.last_activity)" in src)
    # Compare CALL SITES, not mentions -- the docstring names consolidate_patch
    # above the save, so a naive index comparison reads the wrong order.
    body = src[src.index('"""', src.index('"""') + 3) + 3:]
    ck("the transcript is persisted BEFORE consolidation runs",
       body.index("save_session_raw") < body.index("consolidate.consolidate_patch"))


def test_chat_survives_a_restart():
    """A deploy used to take the live conversation with it.

    Worse than losing the screen: an in-memory session that dies is never
    CLOSED, so consolidation never runs for it. Everything extracted since the
    last close was not delayed, it was lost -- and on chat that window is 45
    minutes wide.
    """
    import thread_machine as tm

    s = chat_session.ChatSession("uid-restart", {"name": "Dhruv"}, {})
    s.add("user", "mujhe thyroid hai")
    s.add("assistant", "noted")
    s.rolling_summary = "talked about thyroid"
    s.pending_facts = {"health.conditions": ["thyroid"]}
    s.threads = [tm.Thread(topic="thyroid", stage=tm.S_GATHER, stage_turns=2,
                           slots={"a": "1"})]

    # What the store would write, round-tripped as JSON does.
    row = {
        "session_id": s.session_id, "started_at": s.started_at,
        "last_activity": s.last_activity, "messages": s.messages,
        "rolling_summary": s.rolling_summary,
        "pending_facts": s.pending_facts,
        "threads": chat_store._threads_to_json(s.threads), "closed": False,
    }
    ck("threads serialise to plain JSON",
       isinstance(row["threads"], list) and row["threads"][0]["topic"] == "thyroid")

    fresh = chat_session.ChatSession("uid-restart")
    chat_store.restore(fresh, row)
    ck("messages survive", len(fresh.messages) == 2)
    ck("the session id is kept, not regenerated",
       fresh.session_id == s.session_id)
    ck("pending facts survive -- this is the data that was being lost",
       fresh.pending_facts == {"health.conditions": ["thyroid"]})
    ck("the rolling summary survives", fresh.rolling_summary == "talked about thyroid")
    ck("thread state survives",
       len(fresh.threads) == 1 and fresh.threads[0].stage == tm.S_GATHER
       and fresh.threads[0].stage_turns == 2, fresh.threads)
    ck("turn_index is rebuilt from the user turns", fresh.turn_index == 1)

    # Degradation must be graceful: no Supabase means today's behaviour, not
    # a broken conversation.
    ck("restore(None) is a no-op", chat_store.restore(fresh, None) is fresh)
    ck("a thread row the code cannot parse is skipped, not fatal",
       chat_store._threads_from_json([{"nonsense": 1}]) == [])
    ck("the store is optional", chat_store.enabled() in (True, False))


def test_avatar_url_is_content_versioned():
    """Replacing the image changed nothing for anyone who had loaded the page.

    The URL was identical, so browsers served their stored copy and kept
    showing the old photo -- the file on the server was already correct and
    byte-identical to the new one. Only ETag and Last-Modified were set, and
    browsers routinely skip revalidating images.
    """
    import pathlib
    here = pathlib.Path(chat_engine.__file__).parent
    bot = (here / "bot.py").read_text(encoding="utf-8")

    ck("the version is a hash of the image BYTES",
       "hashlib.sha256" in bot and "mira_avatar.jpg" in bot)
    ck("it is computed per request, not pinned at boot",
       "def _avatar_version" in bot and "read_bytes()" in bot)
    ck("hashing failure degrades instead of 500ing", 'return "0"' in bot)

    import re as _re
    for page in ("chat_ui.html", "call_ui.html"):
        html = (here / page).read_text(encoding="utf-8")
        refs = _re.findall(r"mira_avatar\.jpg(\?v=\{\{AVATAR_V\}\})?", html)
        ck(f"{page} references the avatar", len(refs) >= 1)
        # EVERY reference must carry the version, not just one of them -- a
        # single unversioned one is enough to serve a stale image.
        ck(f"{page} versions every reference", all(refs), refs)
    ck("both pages get the placeholder substituted",
       bot.count('"{{AVATAR_V}}", _avatar_version()') == 2)

    # Caching hard is only safe BECAUSE the URL carries the hash.
    ck("the image is cached hard", "immutable" in bot and "max-age=31536000" in bot)


def test_chat_ui_has_a_working_logout():
    """The burger was decorative -- aria-hidden, no handler, no way out."""
    import pathlib
    ui = pathlib.Path(chat_engine.__file__).with_name("chat_ui.html").read_text(encoding="utf-8")
    ck("the burger is a real button now", 'id="menuBtn"' in ui
       and 'class="menu" aria-hidden' not in ui)
    ck("there is a log out item", 'id="logoutBtn"' in ui and "Log out" in ui)
    ck("the logout URL is injected, not hardcoded twice", "{{LOGOUT_URL}}" in ui)
    ck("logging out replaces history rather than pushing",
       "location.replace(LOGOUT_URL)" in ui)  # back button must not return
    # Must be the SIGN-OUT page, not /login. Firebase persists the session in
    # the browser, so /login saw a still-authenticated user and forwarded them
    # to onboarding or chat without ever asking for a password -- logout
    # appeared to do nothing.
    ck("it targets sign-out, not the login form",
       "LOGOUT_URL" in ui and "location.replace(LOGIN_URL)" not in ui)
    ck("session storage is cleared on the way out", "sessionStorage.clear()" in ui)
    ck("a click anywhere closes the menu", 'id="scrim"' in ui)
    ck("escape closes it too", "e.key === 'Escape'" in ui)
    ck("the button reports its state", "aria-expanded" in ui)

    import pathlib as _p
    bot = _p.Path(chat_engine.__file__).with_name('bot.py').read_text(encoding='utf-8')
    ck('the logout URL derives from WEB_APP_URL -- one source of truth',
       'WEB_APP_URL.rstrip("/") + "/logout"' in bot)
    ck('and it is the sign-out route, never the login form',
       '"/login"' not in bot.split("LOGOUT_URL")[1][:200])
    ck('and the placeholder is substituted when serving the page',
       'html.replace("{{LOGOUT_URL}}", LOGOUT_URL)' in bot)


def test_chat_ui_queues_instead_of_dropping():
    """A message typed while Mira replied was silently discarded."""
    import pathlib
    ui = pathlib.Path(chat_engine.__file__).with_name("chat_ui.html").read_text(encoding="utf-8")
    ck("there is an outbox", "const outbox = []" in ui)
    ck("a busy send is queued, not dropped", "outbox.push(text)" in ui)
    ck("the queue drains after the reply", "outbox.shift()" in ui)
    ck("the message appears immediately either way",
       ui.index("bubble('user', text)") < ui.index("if (busy){"))
    # Disabling the composer while she typed is what made it feel broken.
    ck("the composer is not disabled mid-reply",
       ui.count("sendBtn.disabled = true") == 1)   # only the eligibility gate


def test_chat_fact_merge():
    """A list path must ACCUMULATE. dict.update() silently drops all but the last.

    Live chat run 2026-09-01: the user reported a heart attack, then four
    messages later a blood sugar reading. Both are health.conditions, which is
    list-typed. The session ended holding only the sugar figure -- the heart
    attack was gone, silently, on a health product.
    """
    p = chat_engine.merge_facts({}, {"health.conditions": "heart attack last month"})
    p = chat_engine.merge_facts(p, {"health.conditions": "blood sugar 180"})
    ck("a second condition does not evict the first",
       p["health.conditions"] == ["heart attack last month", "blood sugar 180"],
       p.get("health.conditions"))

    p = chat_engine.merge_facts(p, {"health.conditions": "Heart Attack Last Month"})
    ck("the same condition in different case is not duplicated",
       len(p["health.conditions"]) == 2, p["health.conditions"])

    # Scalars must still REPLACE -- a corrected sleep time should not turn
    # into a list of every answer ever given.
    q = chat_engine.merge_facts({}, {"lifestyle.sleep_time": "11pm"})
    q = chat_engine.merge_facts(q, {"lifestyle.sleep_time": "1am"})
    ck("a scalar is replaced, not accumulated", q["lifestyle.sleep_time"] == "1am")
    ck("merging returns a new dict rather than mutating",
       chat_engine.merge_facts({}, {}) == {})


def test_chat_bubble_edges():
    """Two failure modes from the live run, in opposite directions."""
    # Under-splitting: the model separates paragraphs with a blank line, so
    # "any newline means keep whole" matched nearly every reply and bubbles
    # never split at all.
    two = chat_engine.split_bubbles("Haan, rice theek hai.\n\nKitna khaate ho?")
    ck("a blank line is a bubble boundary", len(two) == 2, two)

    # Over-splitting: "Of course. No problem." across two bubbles reads as a
    # stutter, not as thinking.
    ck("a very short reply stays in one bubble",
       len(chat_engine.split_bubbles("Of course. No problem.")) == 1)

    # Structure still survives whole -- that is what the guard was always for.
    for doc in ("Day 1\n- poha\n- dal\nDay 2\n- upma",
                "1. poha\n2. dal\n3. roti\n4. sabzi",
                "## Plan\nSome text\nMore text\nAnd more"):
        ck(f"structured text kept whole: {doc[:14]!r}",
           len(chat_engine.split_bubbles(doc)) == 1)


def test_chat_prompt_is_not_the_voice_prompt():
    """The voice prompt exists partly to drive a TTS engine. Those rules are
    not neutral in chat -- they are wrong."""
    import pathlib
    txt = pathlib.Path(chat_engine.__file__).with_name("chat_prompt.md").read_text(encoding="utf-8")
    ck("chat prompt exists and takes user context", "{{user_context}}" in txt)
    ck("digits are allowed, unlike voice", "10 pm" in txt)
    ck("it does not claim to be a voice call", "VOICE call" not in txt)
    ck("ending every reply with a question is explicitly dropped",
       "do NOT have to end with a question" in txt.replace("’", "'"))
    ck("advising is the job here, unlike onboarding",
       "giving useful guidance is exactly what you are for" in txt)
    ck("promises still banned", "Never promise" in txt)
    ck("medical deferral survives", "doctor is the right person" in txt)
    ck("emergencies get an explicit instruction", "seek medical help now" in txt)

    # Voice-only machinery must NOT have been copied across.
    for banned in ("Devanagari for pronunciation", "NO digits", "spoken aloud"):
        ck(f"voice-only rule absent: {banned!r}", banned not in txt)

    # Mira is a woman. The voice prompt always said so; the chat prompt did
    # not, and a live run produced "samajh sakta hoon", "kaam kar sakta hoon"
    # and "samajhta hoon" in one session.
    ck("chat prompt states she is a woman", "WOMAN" in txt)
    # A live run produced "Kamya team ke behind mein actual nutritionists
    # hain" -- a claim about the business she has no way of knowing.
    ck("inventing facts about Kamya is banned", "Never invent anything about Kamya" in txt)
    ck("it names the correct feminine forms", "sakti hoon" in txt)
    ck("it names the masculine forms to avoid", "sakta hoon" in txt)


def test_emergency_outranks_medical():
    """"I have a condition" and "this is happening to me now" need different
    answers, and the second must not depend on the model choosing to comply."""
    for text in ("mujhe abhi chest me bahut dard ho raha hai",
                 "saans nahi aa rahi",
                 "main behosh ho raha hoon"):
        r = on.check_global_trigger(text)
        ck(f"emergency fires on: {text[:34]}", bool(r) and "112" in (r or ""))

    # A past event is MEDICAL, not EMERGENCY -- telling someone whose heart
    # attack was last month to call an ambulance is its own failure.
    r = on.check_global_trigger("mera sugar 180 rehta hai")
    ck("a standing condition is not treated as an emergency",
       bool(r) and "112" not in (r or ""))

    names = [t["name"] for t in on.GLOBAL_TRIGGERS]
    ck("EMERGENCY is matched before MEDICAL",
       names.index("EMERGENCY") < names.index("MEDICAL"), names)
    ck("the semantic backstop knows the category too",
       "EMERGENCY" in on._CHECK_SYSTEM)


# ------------------------------------------------------------- safety net --
def test_trigger_backstop():
    """The regex misses real disclosures. There must be a second layer.

    On the 2026-08-31 call "Recently मुझे heart attack आया था" matched NO
    pattern — the MEDICAL list looks for numeric readings, "diagnosed with"
    and "doctor ne bola" — so Mira acknowledged it with "ओह." and carried on
    asking about motivation. "paid plan कितने का है" missed too, and she
    invented her own pricing answer.
    """
    # The misses are real; keep them as the thing being defended against.
    for text in ("Recently मुझे heart attack आया था.",
                 "तुम्हारा paid plan कितने का है मेरा?",
                 "मुझे कोई shortcut बताओ ना, नींद पूरी कैसे करूँ?"):
        ck(f"regex alone still misses: {text[:34]}",
           on.check_global_trigger(text) is None)

    # So the checker must be able to name the category semantically...
    import inspect
    chk = inspect.getsource(on)
    ck("the checker asks for a trigger category", '"trigger": null|' in chk
       and '"MEDICAL"' in chk)
    ck("a health EVENT counts as MEDICAL, not just a diagnosis",
       "heart attack" in chk and "MEDICAL" in chk)
    ck("ambiguity resolves toward MEDICAL",
       "When unsure between null and MEDICAL, choose MEDICAL" in chk)

    # ...and every category must map to its fixed wording.
    for name in ("MEDICAL", "PRICING", "DEFLECT", "WHAT_NEXT",
                 "MIRA_IDENTITY", "SENSITIVE"):
        ck(f"{name} has a scripted response", bool(on.trigger_response(name)))
    ck("an unknown category yields nothing rather than guessing",
       on.trigger_response("NOT_A_TRIGGER") is None)

    ck("the backstop reuses the scripted-response hint",
       "hint = global_trigger_hint(scripted)" in chk)
    ck("and the turn is not charged to the node", "semantic trigger" in chk)


def test_no_advice_no_promises():
    g = on.GLOBAL_RULES
    ck("advice is refused even when asked directly",
       "EVEN WHEN THEY ASK DIRECTLY" in g)
    ck("the hedged sleep answer is named as a violation",
       "sleep को बेहतर बना सकते हैं" in g)
    ck("naming any lever counts as advice", "is already advice" in g)
    ck("promises are banned, closing included",
       "NEVER PROMISE ANYTHING" in g and "closing" in g)
    ck("the invented plan-and-message promise is named",
       "personalized plan बनाएंगे" in g)


# ------------------------------------------------------- volunteered facts --
def test_capture_is_not_node_scoped():
    """A fact given early must be kept, not thrown away.

    On the 2026-08-30 call the user volunteered lunch time, breakfast, dinner
    time and evening chai while the machine was still in PROBLEM. The checker
    was shown only PROBLEM's paths, so it emitted none of them: `extracted`
    logged one fact across the whole window, PROBLEM ran 22 turns waiting for
    information that had already been given, and DAILY_EATING then re-asked it.
    """
    import inspect
    src = inspect.getsource(on.check_node_complete)
    ck("the checker offers the WHOLE schema for capture",
       "sorted(_mf.SCHEMA)" in src)
    ck("capture and node-completion are separated in the prompt",
       "PATHS YOU MAY CAPTURE" in src and "WHAT THIS NODE STILL NEEDS" in src)
    ck("only the node's own paths drive status/missing",
       "decide" in src and "`status` and `missing`" in src)

    # The code filter always accepted any schema path -- only the prompt was
    # narrow. If that filter ever narrows to the node, the bug returns.
    ck("the code filter still accepts any valid schema path",
       "if key in memory_facts.SCHEMA" in src)


def test_register_and_banned_words():
    g = on.GLOBAL_RULES
    ck("आप-form must hold for the whole call", "WHOLE call" in g)
    ck("तुम-form endings are named explicitly", "करते हो" in g)
    ck("समझी is banned everywhere, not just as an opener",
       "not as an opener, not anywhere" in g)


# --------------------------------------------------------------- detours --
def test_offtopic_bridge():
    """Coming back from a detour must sound like a person, not a form.

    The old instruction said "answer briefly, then return to what you were
    asking" and the model obeyed literally: "चार होता है! और lunch में generally
    क्या खाते हैं?" -- an answer welded to a hard pivot. Nothing asked for a
    bridge, so there wasn't one.
    """
    h = on.OFF_TOPIC_HINT
    ck("the aside hint demands a BRIDGE beat", "BRIDGE" in h)
    ck("the bridge is not presented as optional", "NOT optional" in h)
    ck("a hard pivot is shown as the BAD example",
       'BAD:' in h and "hard pivot" in h)
    ck("at least three worked examples of a good bridge",
       h.count("GOOD:") >= 3, h.count("GOOD:"))
    ck("the bridge must vary between turns", "Vary the bridge" in h)
    ck("it still forbids losing the thread", "lose your place" in h)
    ck("and still asks for ONE short reply", "ONE short reply" in h)

    # A scripted trigger response is a compliance property: medical, pricing
    # and identity answers must be word-for-word the same every time. The
    # bridge wraps that wording; it never gets to rewrite it.
    fixed = "इसके बारे में Kamya team आपको detail में बताएगी."
    g = on.global_trigger_hint(fixed)
    ck("the scripted response survives verbatim", fixed in g)
    ck("it is marked as not paraphrasable", "must not" in g and "paraphrased" in g)
    ck("the trigger path bridges too", "BRIDGE" in g)
    ck("the trigger path still bans other advice", "no other advice" in g)

    # Both paths are injected per turn, NOT carried in GLOBAL_RULES -- they
    # would otherwise cost ~250 tokens on every turn to serve the few that
    # actually take a detour.
    ck("the aside hint is not baked into GLOBAL_RULES",
       "BRIDGE" not in on.GLOBAL_RULES)


# ------------------------------------------------------------ prompt size --
def test_global_rules_not_duplicated():
    """GLOBAL_RULES must appear exactly ONCE, and must still be LAST.

    build_node_prompt() already appends the rules, and both hint paths then
    appended them AGAIN -- 2,278 tokens of rules Claude had just read, on
    every turn carrying a hint. Fixing it must not change WHICH instructions
    Claude sees, nor their priority order: the rules stay last, where the
    model weights them most.
    """
    prof = {"name": "Dhruv", "diet": "veg"}
    ext = {"current_pattern.dinner.time": "9pm"}
    G = on.GLOBAL_RULES
    hint = "\n\nStill needed: dinner. Ask ONE question.\n"

    node_only = on.build_node_prompt("DAILY_EATING", prof, ext, include_rules=False)
    full = on.build_node_prompt("DAILY_EATING", prof, ext)

    ck("include_rules=False omits the rules", G not in node_only)
    ck("the default still appends them exactly once", full.count(G) == 1)
    ck("no-hint prompt is unchanged", full == node_only + "\n" + G)

    after = node_only + hint + "\n" + G
    ck("hint path carries GLOBAL_RULES exactly once", after.count(G) == 1)
    ck("GLOBAL_RULES is still the LAST thing in the prompt",
       after.rstrip().endswith(G.rstrip()))
    ck("the hint survives", hint.strip() in after)
    ck("node instructions survive", "daily food" in after)
    ck("profile survives", "Dhruv" in after)
    ck("extracted facts survive", "9pm" in after)

    # The decisive check: only a DUPLICATE was removed, never an instruction.
    before = full + hint + "\n" + G          # the old, doubled construction
    lost = ({l.strip() for l in before.splitlines() if l.strip()}
            - {l.strip() for l in after.splitlines() if l.strip()})
    ck("no instruction line is lost, only the duplicate copy", not lost, list(lost)[:3])
    ck("and it is materially smaller", len(before) - len(after) > 6000,
       len(before) - len(after))


# ------------------------------------------------------------ reply shape --
def _shape(chunks, cap=32):
    """Run a streamed reply through the shaper; return what TTS would speak."""
    sh = reply_shape.ReplyShaper(word_cap=cap)
    out = []
    for c in chunks:
        keep, stop = sh.feed(c)
        if keep:
            out.append(keep)
        if stop:
            break
    return "".join(out), sh


def test_reply_shape():
    # Tokens arrive in small pieces, so every rule has to hold mid-chunk.
    spoken, sh = _shape(["अच्छा", ", दिल्ली", " में! और work", " क्या करते हैं", " आप?"])
    ck("a normal reply passes through untouched",
       spoken == "अच्छा, दिल्ली में! और work क्या करते हैं आप?", spoken)
    ck("the question mark survives", spoken.endswith("?"))

    # The failure this whole filter exists for: eleven questions in one breath.
    spoken, _ = _shape(["breakfast में क्या खाते हैं?",
                        " और lunch?", " और dinner?", " और chai?"])
    ck("everything after the first question is dropped",
       spoken.count("?") == 1 and spoken == "breakfast में क्या खाते हैं?", spoken)

    # Split across chunks, which is how it actually arrives.
    spoken, _ = _shape(["आप क्या खाते", " हैं", "?", " और कब", "?"])
    ck("one question survives a mark that arrives as its own chunk",
       spoken.count("?") == 1 and spoken == "आप क्या खाते हैं?", spoken)

    # THE REGRESSION THIS REFACTOR FIXES. Mira's question is the LAST sentence,
    # so a word cap that cuts at a sentence boundary drops the question and
    # leaves a reply that only makes statements -- which stalls the call.
    long_pre = ["यह एक बहुत लंबा वाक्य है जो cap से आगे निकल जाता है. " * 4,
                "तो dinner कितने बजे करते हैं?"]
    spoken, sh = _shape(long_pre, cap=10)
    ck("an over-cap reply still ends with its question",
       spoken.rstrip().endswith("?"), spoken[-60:])
    ck("the question is never dropped to satisfy the cap",
       "dinner कितने बजे करते हैं?" in spoken)
    ck("over-cap is reported for logging, not silently truncated",
       sh.overlong() is True)

    # A model that never punctuates must still be stopped eventually.
    spoken, sh = _shape(["शब्द " * 200], cap=10)
    ck("unpunctuated runaway is cut", sh.cut and sh.reason == "unpunctuated runaway")

    # And the cap must not fire on ordinary variation.
    spoken, sh = _shape(["ठीक है. और lunch कितने बजे होता है आप का?"], cap=32)
    ck("a short reply is not touched by the runaway guard",
       sh.reason == "question asked" and spoken.endswith("?"))

    sh2 = reply_shape.ReplyShaper(word_cap=32)
    sh2.feed("पहला सवाल?")
    sh2.reset()
    keep, stop = sh2.feed("दूसरा reply")
    ck("reset clears the cut so the next reply is not swallowed",
       keep == "दूसरा reply" and not stop, (keep, stop))



def test_prompt_invariants():
    """Rules earned from real calls. Each one cost a live call to find."""
    g = on.GLOBAL_RULES
    ck("the ack rotation list is gone",
       'rotate "अच्छा"' not in g)
    ck("restating the user's answer is banned",
       "DO NOT REPEAT THEIR ANSWER BACK" in g)
    ck("confirm-what-you-just-heard questions are banned",
       "NEVER ask them to confirm something they just told you" in g)
    ck("vague questions are called out with a concrete fix",
       "ASK ABOUT THE THING, NOT AROUND IT" in g)
    ck("one question per reply still enforced in prose", "ONE question per reply" in g)
    # The rule was widened from "never OPEN with समझी" to "never say it at
    # all" after it appeared mid-reply twice on a live call.
    ck("समझी banned outright", "समझी" in g and "NEVER say" in g)

    # The vague opener was a hardcoded string in the node prompt, so no amount
    # of rule-writing elsewhere could override it.
    daily = on.NODES["DAILY_EATING"]["prompt"]
    ck("DAILY_EATING opener asks about FOOD, not the morning in general",
       "सबसे पहले क्या खाते हैं" in daily
       and "सबसे पहले क्या होता है" not in daily)


# ---------------------------------------------------------------- fallback --
def test_bedrock_falls_back():
    """A Bedrock outage must cost a slower turn, not a lost decision."""
    import asyncio

    os.environ["AWS_BEARER_TOKEN_BEDROCK"] = "test"
    os.environ["BEDROCK_MODEL"] = "anthropic.claude-haiku-4-5-v1:0"
    os.environ.pop("LLM_PROVIDER", None)

    calls = []
    orig_b, orig_g = L._bedrock_json, L._groq_json

    async def boom(*a, **k):
        calls.append("bedrock")
        raise RuntimeError("bedrock is down")

    async def fine(*a, **k):
        calls.append("groq")
        return {"lane": "QUICK"}

    L._bedrock_json, L._groq_json = boom, fine
    try:
        got = asyncio.get_event_loop().run_until_complete(
            L.complete_json("s", "u", kind="fast", timeout=2.0))
    finally:
        L._bedrock_json, L._groq_json = orig_b, orig_g

    ck("bedrock is tried first", calls[:1] == ["bedrock"], calls)
    ck("a bedrock failure falls through to groq", calls == ["bedrock", "groq"], calls)
    ck("the caller still gets a real decision", got == {"lane": "QUICK"}, got)

    # And the contract that everything else leans on: never raise.
    L._bedrock_json = L._groq_json = boom
    try:
        got2 = asyncio.get_event_loop().run_until_complete(
            L.complete_json("s", "u", kind="fast", timeout=2.0))
    finally:
        L._bedrock_json, L._groq_json = orig_b, orig_g
    ck("both providers down yields {} and no exception", got2 == {}, got2)


def main():
    for fn in (test_llm_client, test_extraction, test_stages, test_memory,
               test_rag_gate, test_onboarding,
               test_health_basics_moved_from_the_form_to_the_call,
               test_echo_guard,
               test_chat_session_lifecycle, test_chat_session_store,
               test_chat_bubbles_and_budgets, test_chat_fact_merge,
               test_chat_has_a_semantic_safety_backstop,
               test_stages_are_judged_not_counted,
               test_advice_is_stated_not_floated,
               test_problem_threads_have_their_own_checklist,
               test_thread_accumulates_problem_details,
               test_memory_prefill_is_not_understanding,
               test_small_talk_gets_small_talk,
               test_medical_boundary_is_doctor_owned_not_everyday,
               test_guard_covers_every_observed_failure,
               test_guard_guarantees_what_the_prompt_only_asks_for,
               test_off_topic_is_answered_briefly_and_dropped,
               test_no_menus_anywhere,
               test_social_turns_are_not_steered_back,
               test_conversation_falls_back_like_the_small_jobs_do,
               test_close_session_calls_match_their_signatures,
               test_chat_survives_a_restart,
               test_avatar_url_is_content_versioned,
               test_chat_ui_has_a_working_logout,
               test_chat_ui_queues_instead_of_dropping,
               test_chat_bubble_edges, test_emergency_outranks_medical,
               test_chat_prompt_is_not_the_voice_prompt,
               test_global_rules_not_duplicated, test_trigger_backstop,
               test_no_advice_no_promises, test_capture_is_not_node_scoped,
               test_register_and_banned_words, test_offtopic_bridge,
               test_reply_shape,
               test_prompt_invariants,
               test_bedrock_falls_back):
        fn()
    passed = sum(1 for _, c, _ in R if c)
    print("=" * 74)
    for name, cond, detail in R:
        print(f"  {'PASS' if cond else 'FAIL'}  {name}"
              + (f"\n          -> {detail}" if not cond and detail else ""))
    print("=" * 74)
    print(f"  {passed}/{len(R)} passed")
    return 0 if passed == len(R) else 1


if __name__ == "__main__":
    sys.exit(main())
