"""Weekly diet plan: generate, validate, store.

WHY THIS IS A TOOL AND NOT JUST A PROMPT.
    A diet plan is the highest-stakes thing Mira produces. It is downloaded,
    screenshotted, and followed for a week without her in the room. Everything
    else she says is a message; this is a document that outlives the
    conversation.

    So the model PROPOSES and code DISPOSES, the same shape as
    memory_facts.apply_patch. The model returns structured days and meals; this
    module then checks every line against what we actually know -- diet type,
    allergies, conditions -- and rejects the plan if it contradicts them. A
    plan containing a food the user told us they cannot eat is not a style
    problem, and no prompt rule reliably prevents it.

STORED AS DATA, RENDERED ON DEMAND.
    Supabase holds the plan as JSON; the PDF is built at download time by
    plan_pdf. Nothing binary is stored, a download is always current, and
    changing one day is a JSON edit rather than a file rewrite.

TABLES (run once in the Supabase SQL editor):

    create table if not exists diet_plans (
      firebase_uid text primary key,
      week_start   text not null,
      plan         jsonb not null,
      updated_at   timestamptz default now()
    );
    alter table diet_plans enable row level security;

    -- Preferences stated in chat, banked until the next build consumes them.
    create table if not exists plan_preferences (
      id           bigserial primary key,
      firebase_uid text not null,
      note         text not null,
      created_at   timestamptz default now(),
      applied_at   timestamptz
    );
    create index if not exists plan_preferences_pending
      on plan_preferences (firebase_uid, applied_at);
    alter table plan_preferences enable row level security;

    -- The service key bypasses RLS. Leaving RLS on with no policy means
    -- nothing else can read these rows, which is what we want: a diet plan
    -- is health data and only the server should touch it.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re

import llm_client
import memory_facts

TABLE = os.getenv("DIET_PLANS_TABLE", "diet_plans")
TIMEOUT = float(os.getenv("PLAN_STORE_TIMEOUT", "8"))
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
            "Saturday", "Sunday"]

# Foods that contradict a stated diet. Deliberately small and concrete: this
# is a REJECTION check, so a false positive costs a regeneration and a false
# negative costs the user's trust.
_NON_VEG = ["chicken", "mutton", "fish", "prawn", "shrimp", "egg", "anda",
            "keema", "beef", "pork", "bacon", "ham", "tuna", "salmon",
            "murgh", "machhli", "seekh", "omelette", "omelet"]
_NON_VEGAN = _NON_VEG + ["milk", "curd", "dahi", "paneer", "cheese", "ghee",
                         "butter", "cream", "yoghurt", "yogurt", "lassi",
                         "buttermilk", "chaas", "khoya", "malai", "honey"]
_EGG = ["egg", "anda", "omelette", "omelet", "bhurji"]


def _norm(s) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()


def week_start(today=None) -> str:
    """Monday of the current week, ISO. Plans always run Monday to Sunday so
    'next week' is unambiguous for both the user and the Sunday refresh."""
    d = today or dt.date.today()
    return (d - dt.timedelta(days=d.weekday())).isoformat()


def next_week_start(today=None) -> str:
    d = dt.date.fromisoformat(week_start(today))
    return (d + dt.timedelta(days=7)).isoformat()


# ------------------------------------------------------------- validation --
def _stem(w: str) -> str:
    """Crude singular. 'peanuts' and 'peanut' must be the same word here."""
    w = _norm(w)
    for suf in ("es", "s"):
        if len(w) > 4 and w.endswith(suf):
            return w[: -len(suf)]
    return w


# Allergies get stated as categories far more often than as ingredients --
# "dairy", "nuts", "gluten" -- and the plan will name the member, not the
# category. Expanding here is the difference between catching "Milkshake"
# under a milk allergy and shipping it.
_ALLERGY_FAMILY = {
    "dairy": ["milk", "curd", "dahi", "paneer", "cheese", "ghee", "butter",
              "cream", "yoghurt", "yogurt", "lassi", "buttermilk", "chaas",
              "khoya", "malai", "raita", "kheer"],
    "lactose": ["milk", "curd", "dahi", "paneer", "cheese", "cream", "lassi",
                "buttermilk", "chaas", "kheer"],
    "nut": ["almond", "badam", "cashew", "kaju", "walnut", "akhrot", "peanut",
            "moongphali", "pista", "pistachio", "hazelnut"],
    "tree nut": ["almond", "badam", "cashew", "kaju", "walnut", "akhrot",
                 "pista", "pistachio", "hazelnut"],
    "gluten": ["wheat", "atta", "roti", "chapati", "paratha", "bread", "maida",
               "suji", "rava", "daliya", "pasta", "noodle", "barley"],
    "wheat": ["atta", "roti", "chapati", "paratha", "bread", "maida", "suji",
              "rava", "daliya"],
    "seafood": ["fish", "prawn", "shrimp", "crab", "machhli", "tuna", "salmon"],
    "shellfish": ["prawn", "shrimp", "crab", "lobster"],
    "soy": ["soya", "tofu", "edamame"],
}


# A qualifier in front of a word changes what the word IS. "Coconut milk" is
# not dairy, "jowar roti" is not wheat, "peanut butter" is not butter. Without
# this table the validator punishes the model for being careful -- it rejected
# a correct vegan gluten-free plan three times over exactly these phrases.
_QUALIFIERS = {
    "milk": ["almond", "badam", "soy", "soya", "coconut", "nariyal", "oat",
             "rice", "cashew", "kaju", "hemp", "peanut", "plant", "vegan"],
    "butter": ["peanut", "almond", "cashew", "nut", "moongphali", "plant",
               "vegan", "apple"],
    "cream": ["coconut", "cashew", "oat", "soy", "non-dairy", "vegan"],
    "cheese": ["vegan", "cashew", "tofu", "plant"],
    "curd": ["soy", "coconut", "almond", "plant", "vegan"],
    "dahi": ["soy", "coconut", "almond", "plant", "vegan"],
    "yoghurt": ["soy", "coconut", "almond", "plant", "vegan"],
    "yogurt": ["soy", "coconut", "almond", "plant", "vegan"],
}
# Flatbreads and porridges are named by their FLOUR, and most of those flours
# are not wheat.
_GF_FLOURS = ["jowar", "bajra", "ragi", "nachni", "besan", "makki", "corn",
              "rice", "buckwheat", "kuttu", "singhara", "amaranth", "rajgira",
              "almond", "coconut", "millet", "quinoa", "sama", "gluten-free",
              "gluten free"]
for _w in ("roti", "chapati", "paratha", "bread", "pasta", "noodle", "daliya",
           "atta", "flour"):
    _QUALIFIERS[_w] = list(_GF_FLOURS)


def _qualified(text: str, start: int, word: str) -> bool:
    """Is the match at `start` preceded by a qualifier that neutralises it?

    Looks only at the two words immediately before, so "milk" in
    "coconut milk" is exempt but "milk" in "coconut chutney, milk" is not.
    """
    # "gluten-free oats" and "dairy free" are the OPPOSITE of a violation,
    # and they contain the very word being searched for.
    if re.match(r"[a-z]*[\s-]*free\b", text[start + len(word):]):
        return True
    quals = _QUALIFIERS.get(_stem(word)) or _QUALIFIERS.get(word) or []
    if not quals:
        return False
    before = text[max(0, start - 30):start]
    # A qualifier only counts inside the same food. Punctuation ends a food,
    # so "coconut chutney, milk" must NOT read as coconut milk.
    before = re.split(r"[,;()/+&]|\band\b|\bwith\b", before)[-1]
    tail = " ".join(re.findall(r"[a-z][a-z-]*", before)[-2:])
    return any(q in tail for q in quals)


def _mentions(text: str, word: str) -> bool:
    """Diet-rule match: bounded, so 'egg' does not fire on 'eggplant'.

    Over-matching here costs a wasted regeneration on a food the user could
    actually have eaten, so the boundary is worth keeping.
    """
    w = re.escape(_stem(word))
    for probe in (text, _stem(text)):
        for m in re.finditer(rf"\b{w}(?:e?s)?\b", probe):
            if not _qualified(probe, m.start(), word):
                return True
    return False


def _allergy_hit(text: str, word: str) -> str:
    """Allergy match: deliberately looser than _mentions.

    Expands categories to their members, and matches a PREFIX rather than a
    whole word, so 'milk' catches 'Milkshake'. A false positive costs one
    regeneration; a false negative puts an allergen in a document the user
    follows for a week without us. The asymmetry decides the design.
    """
    stem = _stem(word)
    # Look the family up under both forms: _stem leaves short words like
    # "nuts" alone, and the family key is "nut".
    family = (_ALLERGY_FAMILY.get(stem) or _ALLERGY_FAMILY.get(_norm(word))
              or _ALLERGY_FAMILY.get(_norm(word).rstrip("s")) or [])
    for probe in [stem] + family:
        for m in re.finditer(rf"\b{re.escape(probe)}", text):
            if not _qualified(text, m.start(), probe):
                return probe
    return ""


def _p(day_i, meal_i, msg) -> dict:
    return {"day": day_i, "meal": meal_i, "msg": msg}


def check(plan: dict, diet: str, allergies) -> list:
    """Return a list of problems. Empty means the plan is safe to hand over.

    Each problem carries its day index and slot so `_repair` can rewrite just
    that meal instead of re-rolling the whole week.
    """
    problems = []
    diet_l = _norm(diet)
    banned = []
    if "vegan" in diet_l:
        banned = _NON_VEGAN
    elif "egg" in diet_l:                      # eggetarian: eggs fine, meat not
        banned = [w for w in _NON_VEG if w not in _EGG]
    elif "veg" in diet_l and "non" not in diet_l:
        banned = _NON_VEG
    if "jain" in diet_l:
        banned = banned + ["onion", "garlic", "pyaz", "lehsun", "potato", "aloo"]

    allergy_words = []
    for a in (allergies if isinstance(allergies, list) else [allergies]):
        a = _norm(a)
        if a and a not in ("none", "no", "nahi", "koi nahi", "-"):
            allergy_words += [w.strip() for w in re.split(r"[,/;]| and ", a)
                              if len(w.strip()) > 2]

    for di, day in enumerate(plan.get("days") or []):
        where = day.get("weekday") or f"day {day.get('day')}"
        for mi, meal in enumerate(day.get("meals") or []):
            text = _norm(meal.get("items"))
            slot = meal.get("slot")
            for w in banned:
                if _mentions(text, w):
                    problems.append(_p(di, mi, f"{where}/{slot}: '{w}' "
                                       f"contradicts the diet '{diet}'"))
            for w in allergy_words:
                hit = _allergy_hit(text, w)
                if hit:
                    detail = f"'{hit}'" if hit == _stem(w) else f"'{hit}' (from stated allergy '{w}')"
                    problems.append(_p(di, mi, f"{where}/{slot}: {detail} "
                                       f"is an allergen for this user"))
    return problems


def shape_ok(plan: dict) -> list:
    """Structural check, before the content check.

    Shape failures are NOT repairable meal-by-meal -- a missing day means the
    whole plan has to come again -- so they are kept separate.
    """
    problems = []
    days = plan.get("days") or []
    if len(days) != 7:
        problems.append(f"expected 7 days, got {len(days)}")
    for i, d in enumerate(days):
        if not (d.get("meals") or []):
            problems.append(f"day {i + 1} has no meals")
        for m in (d.get("meals") or []):
            if not _norm(m.get("slot")) or not _norm(m.get("items")):
                problems.append(f"day {i + 1}: a meal is missing slot or items")
    return problems


# ------------------------------------------------------------- generation --
_SYSTEM = """\
You are a careful Indian dietician building a SEVEN DAY meal plan.

Return JSON only, exactly this shape:
{
  "summary": "3-4 sentences on the thinking behind the week",
  "days": [
    {"day": 1, "weekday": "Monday", "date": "YYYY-MM-DD",
     "meals": [
       {"slot": "Early Morning", "time": "07:00",
        "items": "Warm water (1 glass), Soaked almonds (5-6)",
        "note": "one short line on why - Total: ~50 kcal"}
     ]}
  ]
}

HARD RULES -- a plan breaking any of these is rejected by code, not by me:
  1. Respect their DIET exactly. Vegetarian means no meat, fish or egg.
     Vegan additionally means no milk, curd, paneer, ghee, butter or honey.
     Eggetarian allows egg but no meat or fish. Jain additionally excludes
     onion, garlic and potato.
  2. Never include ANY food listed under their allergies, in any form.
  3. Seven days. Every day needs the full set of meals.
  4. Use THEIR meal timings where you know them. A plan built around 8am
     breakfast is useless to someone who eats at 3pm.
  5. Ordinary Indian home food they can actually get. No supplements, no
     exotic imports, nothing that needs a special shop.
  6. Vary the days. Seven identical days is not a plan.

MEDICAL BOUNDARY
  If they have a condition, keep the plan conservative and ordinary, and put
  nothing in `note` that reads as treatment. Do not name medicines, do not
  give clinical targets, do not claim a food treats anything. Their doctor
  owns that.

WRITING
  English, with Indian food names in roman script (dal, roti, sabzi, poha).
  NEVER use Devanagari -- the PDF font cannot render it and it becomes boxes.
  Notes are one short line, ending with an approximate calorie total.
"""


def _context_block(profile, memory, extra_notes="") -> str:
    ltm = (memory or {}).get("long_term_memory") or {}
    view = json.dumps(ltm, ensure_ascii=False)[:2500]
    prof = {k: v for k, v in (profile or {}).items() if str(v).strip()}
    return json.dumps({
        "profile": prof,
        "what_we_know": view,
        "open_loops": (memory or {}).get("open_loops") or [],
        "special_instructions": extra_notes or "",
    }, ensure_ascii=False)


_REPAIR_SYSTEM = """\
You are fixing SPECIFIC meals in an otherwise-good diet plan.

You get a list of broken meals. Each has a reason it was rejected. Replace
ONLY the items in those meals. Do not touch anything else, do not renumber,
do not comment on the rest of the plan.

Return JSON only:
{"fixes": [{"day": 0, "meal": 2,
            "items": "replacement items with quantities",
            "note": "one short line - Total: ~NNN kcal"}]}

Use the SAME `day` and `meal` numbers you were given. Keep the replacement in
the same spirit as the meal it replaces -- same slot, similar size, ordinary
Indian home food, roman script only, never Devanagari. And make sure the
replacement does not repeat the mistake you are being asked to fix.

The `note` is read by the USER, who never saw the rejected version. Write it
as if this had always been the meal. Never write "Replaced X with Y",
"instead of", "swapped", or anything else referring to the earlier draft.
"""


# The repair prompt asks for this; code enforces it. A note reading "Replaced
# ghee with coconut oil" is incoherent to someone who only ever sees the final
# plan.
_EDIT_LEAK = re.compile(
    r"^\s*(replaced|swapped|substituted|changed|switched)\b[^.;]*[.;]\s*|"
    r"\s*\(?\b(instead of|in place of|rather than)\b[^.;)]*\)?",
    re.I)


def _clean_note(note: str) -> str:
    out = _EDIT_LEAK.sub(" ", str(note or "")).strip()
    out = re.sub(r"\s{2,}", " ", out)
    out = re.sub(r"\s+([.,;])", r"\1", out).strip(" ,;-")
    return (out[:1].upper() + out[1:]) if out else ""


async def _repair(plan: dict, problems: list, diet: str, allergies,
                  log=None) -> dict:
    """Rewrite only the offending meals.

    Re-rolling the whole week to fix two meals re-rolls the eighteen that were
    already correct, which is how the first version of this kept converging to
    2-3 violations and never to zero. Repair touches only what failed.
    """
    log = log or (lambda m: None)
    broken, seen = [], set()
    for pr in problems:
        key = (pr["day"], pr["meal"])
        if key in seen:
            continue
        seen.add(key)
        try:
            meal = plan["days"][pr["day"]]["meals"][pr["meal"]]
        except (IndexError, KeyError, TypeError):
            continue
        broken.append({
            "day": pr["day"], "meal": pr["meal"],
            "weekday": plan["days"][pr["day"]].get("weekday", ""),
            "slot": meal.get("slot", ""), "time": meal.get("time", ""),
            "current_items": meal.get("items", ""),
            "why_rejected": [q["msg"] for q in problems
                             if (q["day"], q["meal"]) == key],
        })
    if not broken:
        return plan

    ask = json.dumps({
        "diet": diet or "vegetarian", "allergies": allergies or [],
        "broken_meals": broken,
    }, ensure_ascii=False)
    data = await llm_client.complete_json(
        _REPAIR_SYSTEM, ask, kind="heavy", max_tokens=2500, temperature=0.3,
        timeout=120.0)

    applied = 0
    for fix in (data.get("fixes") or []):
        try:
            meal = plan["days"][int(fix["day"])]["meals"][int(fix["meal"])]
        except (IndexError, KeyError, TypeError, ValueError):
            continue
        if _norm(fix.get("items")):
            meal["items"] = str(fix["items"])
            if _norm(fix.get("note")):
                meal["note"] = _clean_note(fix["note"]) or meal.get("note", "")
            applied += 1
    log(f"repair: {len(broken)} meal(s) sent, {applied} replaced")
    return plan


async def generate(profile: dict, memory: dict, *, start: str = "",
                   extra_notes: str = "", attempts: int = 2,
                   repairs: int = 3, log=None) -> dict:
    """Build a validated 7-day plan. Raises if it cannot produce a safe one.

    Two loops, because the two failure kinds are different. A malformed plan
    (wrong day count, empty meals) needs a fresh generation. A plan that is
    structurally fine but names a forbidden food needs a REPAIR of those meals
    -- see _repair for why regenerating instead makes it worse.
    """
    log = log or (lambda m: None)
    start = start or week_start()
    d0 = dt.date.fromisoformat(start)
    ltm = (memory or {}).get("long_term_memory") or {}
    diet = (((ltm.get("diet") or {}).get("type")) or profile.get("diet") or "")
    allergies = ((ltm.get("health") or {}).get("allergies")
                 or profile.get("allergies") or [])

    dates = [(d0 + dt.timedelta(days=i)).isoformat() for i in range(7)]
    ask = (_context_block(profile, memory, extra_notes)
           + f"\n\nWeek starts {start}. The seven dates are {dates}."
           + f"\nTheir diet is: {diet or 'unknown -- assume vegetarian'}."
           + f"\nAllergies: {allergies or 'none stated'}.")

    last = ""
    for attempt in range(1, max(1, attempts) + 1):
        data = await llm_client.complete_json(
            _SYSTEM, ask, kind="heavy", max_tokens=8000, temperature=0.4,
            timeout=180.0)
        if not data:
            last = "empty response from the model"
            log(f"plan attempt {attempt}: empty response")
            continue

        # Stamp the calendar ourselves. Models drift on dates and weekday
        # names, and it is not worth a retry to get them right.
        for n, day in enumerate(data.get("days") or []):
            if n < 7:
                day["day"] = n + 1
                day["weekday"] = WEEKDAYS[n]
                day["date"] = dates[n]

        bad_shape = shape_ok(data)
        if bad_shape:
            last = "; ".join(bad_shape[:4])
            log(f"plan attempt {attempt}: bad shape -- {last}")
            continue

        problems = check(data, diet, allergies)
        for r in range(1, max(0, repairs) + 1):
            if not problems:
                break
            log(f"attempt {attempt}, repair {r}: {len(problems)} problem(s); "
                f"first: {problems[0]['msg']}")
            data = await _repair(data, problems, diet, allergies, log=log)
            problems = check(data, diet, allergies)

        if not problems:
            for day in data["days"]:
                for meal in day.get("meals") or []:
                    meal["note"] = _clean_note(meal.get("note"))
            data["week_start"] = start
            data["generated_for"] = {"diet": diet, "allergies": allergies}
            data["basics"] = {
                "name": profile.get("name", ""), "age": profile.get("age", ""),
                "gender": profile.get("gender", ""),
                "height": profile.get("height", ""),
                "weight": profile.get("weight", ""),
            }
            data["diet"] = diet
            log(f"plan ready for week {start} (attempt {attempt})")
            return data

        last = "\n".join(f"- {q['msg']}" for q in problems[:10])
        log(f"attempt {attempt} still unsafe after {repairs} repairs")

    raise RuntimeError(f"could not produce a safe plan after {attempts} "
                       f"attempt(s). Last problems:\n{last}")


# ------------------------------------------------- chat-driven plan edits --
# Cheap prefilter. Regenerating a week costs a minute and several model calls,
# so a keyword gate runs first and the LLM only judges messages that could
# plausibly be about the plan. Everything here is a HINT -- `classify` makes
# the actual call.
_PLAN_WORDS = ["plan", "chart", "pdf", "diet", "khana", "khaana", "meal",
               "breakfast", "lunch", "dinner", "nashta", "snack"]
_CHANGE_WORDS = ["change", "badal", "badl", "replace", "hata", "nahi mil",
                 "available nahi", "nahi hai", "nahi h", "pasand nahi",
                 "bore", "alag", "kuch aur", "swap", "update", "skip",
                 "nahi kha", "allergy", "band kar", "chhod", "nahi khani",
                 "nahi khaunga", "nahi khaungi", "shuru kar", "start kar",
                 "heavy", "halka", "bhaari", "zyada ho", "kam kar",
                 "add kar", "daal do", "de dijiye", "chahiye", "mat do",
                 "nahi chahiye", "ho jata", "problem"]
_DAY_WORDS = [d.lower() for d in WEEKDAYS] + ["somvar", "mangal", "budh",
              "guru", "shukra", "shani", "ravi", "kal", "aaj", "tomorrow",
              "today", "weekend"]


def maybe_plan_change(text: str) -> bool:
    """Could this message be asking to change the plan? Deliberately loose --
    a false positive costs one cheap classifier call, a false negative means
    the user's request is silently ignored."""
    t = _norm(text)
    if not t or len(t) < 6:
        return False
    # ONE group is enough.
    #
    # This asked for two, and silently dropped real requests: "mujhe roti
    # nahi khani ab" and "breakfast heavy lag raha hai, kuch halka do" both
    # failed the gate and never reached the classifier. The classifier itself
    # turns out to be accurate -- it rejects feedback, questions and thanks
    # cleanly -- so the gate should only be filtering out messages with no
    # food or change content at all, not adjudicating.
    #
    # The cost of being loose is one small background call on a message that
    # was never about the plan. The cost of being strict is the user asking
    # for a change and nothing happening.
    return any(any(w in t for w in group)
               for group in (_PLAN_WORDS, _CHANGE_WORDS, _DAY_WORDS))


_CHANGE_SYSTEM = """\
Read one message from a diet-plan user and decide which of three things it is.

Return JSON only:
{"kind": "update_request" | "preference" | "none", "note": "..."}

"update_request" -- they are ASKING for the written plan/PDF to be rebuilt or
changed NOW. In any phrasing: "plan update kar do", "naya diet chart bhejo",
"pdf update karo", "monday ka change kar do", "plan dobara banao", "diet plan
badal do", "new plan chahiye". The giveaway is a request aimed at the PLAN or
the PDF, not just a statement about food.

"preference" -- they told us something that should shape their NEXT plan, but
did not ask for it to be rebuilt. "Mujhe roti nahi khani ab", "main ab egg
khana shuru kar raha hoon", "mujhe peanuts se allergy hai", "breakfast bahut
heavy lagta hai", "sunday ko main bahar rehta hoon". Real and worth keeping --
just not a request to act right now.

"none" -- everything else. Asking what is in the plan, general food or
nutrition questions, saying a meal WAS good or bad, reporting a meal they
already missed, worrying about progress, small talk.

TENSE matters for the skip cases. "Kal ka dinner miss ho gaya" is a report
about a day gone: none. "Sunday ko main bahar hoon" is a standing fact about
their week: preference.

`note` is one plain English sentence a dietician could act on, naming the day
and meal if the user did. Empty for "none".
"""


async def classify(text: str, recent: str = "") -> dict:
    """What is this message: a request to rebuild, a preference, or neither?

    Split deliberately. Rebuilding on every stated preference means the plan
    changes under the user without them asking; ignoring preferences means
    they have to repeat themselves when they finally do ask. So preferences
    are BANKED and applied at the next build.
    """
    if not maybe_plan_change(text):
        return {"kind": "none", "note": ""}
    data = await llm_client.complete_json(
        _CHANGE_SYSTEM,
        json.dumps({"message": text, "recent_context": recent[-800:]},
                   ensure_ascii=False),
        kind="fast", max_tokens=250, temperature=0.0, timeout=8.0)
    if not isinstance(data, dict):
        return {"kind": "none", "note": ""}
    kind = str(data.get("kind") or "none").strip().lower()
    if kind not in ("update_request", "preference", "none"):
        kind = "none"
    return {"kind": kind, "note": str(data.get("note") or "").strip()[:400]}


_AMEND_SYSTEM = """\
You are making a SMALL, TARGETED change to an existing weekly diet plan.

The user asked for one thing to change. Change only what they asked for.
Everything else in their week stays exactly as it is -- they have already
seen this plan and the rest of it is working for them.

You are given the plan as a numbered list of meals. Return JSON only:
{"edits": [{"day": 0, "meal": 3,
            "items": "new items with quantities",
            "note": "one short line - Total: ~NNN kcal"}],
 "reply": "one short Hinglish line telling the user what changed"}

Use the day and meal numbers exactly as given. Return ONLY the meals that
change -- usually one to three. If the request touches every day (say, they
have stopped eating something entirely), you may return more.

Keep replacements in the same slot, similar size, ordinary Indian home food,
roman script only, never Devanagari. Respect their diet and allergies -- code
rejects a plan that breaks them.

The `note` is read by the user. Write it as if the meal had always been this.
Never write "Replaced X with Y" or "instead of".
"""


def _index(plan: dict) -> list:
    """Flatten the plan into the numbered form the amend prompt expects."""
    out = []
    for di, day in enumerate(plan.get("days") or []):
        for mi, meal in enumerate(day.get("meals") or []):
            out.append({"day": di, "meal": mi,
                        "weekday": day.get("weekday", ""),
                        "slot": meal.get("slot", ""),
                        "time": meal.get("time", ""),
                        "items": meal.get("items", "")})
    return out


async def amend(plan: dict, instruction: str, *, diet: str = "",
                allergies=None, log=None) -> tuple:
    """Apply one targeted change to an existing plan. Returns (plan, reply).

    Deliberately NOT a regeneration. Rebuilding the week because Monday's dal
    is unavailable would also replace the six days the user was happy with,
    which is a worse outcome than not changing anything.

    Raises if the edit cannot be made safely, so the caller keeps the old plan
    rather than saving a broken one.
    """
    log = log or (lambda m: None)
    diet = diet or plan.get("diet", "")
    if allergies is None:
        allergies = (plan.get("generated_for") or {}).get("allergies") or []

    ask = json.dumps({
        "request": instruction, "diet": diet, "allergies": allergies,
        "meals": _index(plan),
    }, ensure_ascii=False)
    data = await llm_client.complete_json(
        _AMEND_SYSTEM, ask, kind="heavy", max_tokens=3000, temperature=0.3,
        timeout=120.0)

    edits = (data or {}).get("edits") or []
    if not edits:
        raise RuntimeError("model returned no edits")

    # Work on a copy: an edit that fails validation must not leave the user's
    # saved plan half-modified.
    draft = json.loads(json.dumps(plan))
    applied = []
    for e in edits:
        try:
            meal = draft["days"][int(e["day"])]["meals"][int(e["meal"])]
        except (IndexError, KeyError, TypeError, ValueError):
            continue
        if not _norm(e.get("items")):
            continue
        meal["items"] = str(e["items"])
        if _norm(e.get("note")):
            meal["note"] = _clean_note(e["note"])
        applied.append((int(e["day"]), int(e["meal"])))
    if not applied:
        raise RuntimeError("no edit could be applied")

    problems = check(draft, diet, allergies)
    for r in range(1, 3):
        if not problems:
            break
        log(f"amend repair {r}: {problems[0]['msg']}")
        draft = await _repair(draft, problems, diet, allergies, log=log)
        problems = check(draft, diet, allergies)
    if problems:
        raise RuntimeError(f"amended plan still unsafe: {problems[0]['msg']}")

    for day in draft.get("days") or []:
        for meal in day.get("meals") or []:
            meal["note"] = _clean_note(meal.get("note"))
    log(f"amend applied to {len(applied)} meal(s): {applied}")
    reply = str((data or {}).get("reply") or "").strip()
    return draft, reply


# ------------------------------------------------------ preference ledger --
# Append-only, like the fact ledger. A preference is banked when the user
# states it and CONSUMED at the next build -- Sunday's, or one they ask for.
#
# The alternative, rebuilding the moment anyone mentions a food, changes the
# plan under the user without them asking for it. Banking means "mujhe roti
# nahi khani" is remembered and honoured next Sunday without a surprise PDF
# appearing today.
PREFS_TABLE = os.getenv("PLAN_PREFS_TABLE", "plan_preferences")


def _prefs_url():
    return os.getenv("SUPABASE_URL", "").rstrip("/") + f"/rest/v1/{PREFS_TABLE}"


async def add_preference(uid: str, note: str, log=None) -> bool:
    """Bank one preference for the next build."""
    log = log or (lambda m: None)
    if not enabled() or not uid or not _norm(note):
        return False
    try:
        import httpx
        async with httpx.AsyncClient(timeout=TIMEOUT) as c:
            r = await c.post(_prefs_url(), headers=_headers(),
                             json={"firebase_uid": uid, "note": str(note)[:400]})
            r.raise_for_status()
        log(f"plan preference banked: {str(note)[:80]}")
        return True
    except Exception as exc:
        log(f"plan preference save failed: {type(exc).__name__}: {exc}")
        return False


async def pending_preferences(uid: str, log=None) -> list:
    """Everything banked and not yet applied, oldest first."""
    log = log or (lambda m: None)
    if not enabled() or not uid:
        return []
    try:
        import httpx
        async with httpx.AsyncClient(timeout=TIMEOUT) as c:
            r = await c.get(_prefs_url(), headers=_headers(),
                            params={"firebase_uid": f"eq.{uid}",
                                    "applied_at": "is.null",
                                    "order": "created_at.asc",
                                    "select": "id,note", "limit": "40"})
            r.raise_for_status()
            return r.json() or []
    except Exception as exc:
        log(f"plan preference load failed: {type(exc).__name__}: {exc}")
        return []


async def mark_applied(uid: str, ids: list, log=None) -> bool:
    """Consume preferences once a plan has actually been built with them.

    Marked only AFTER a successful build, so a failed generation leaves them
    pending rather than silently dropping what the user asked for.
    """
    log = log or (lambda m: None)
    if not enabled() or not uid or not ids:
        return False
    try:
        import httpx
        idlist = ",".join(str(int(i)) for i in ids)
        async with httpx.AsyncClient(timeout=TIMEOUT) as c:
            r = await c.patch(
                _prefs_url(), headers=_headers({"Prefer": "return=minimal"}),
                params={"firebase_uid": f"eq.{uid}", "id": f"in.({idlist})"},
                json={"applied_at": dt.datetime.now(dt.timezone.utc).isoformat()})
            r.raise_for_status()
        log(f"plan preferences applied: {len(ids)}")
        return True
    except Exception as exc:
        log(f"plan preference mark failed: {type(exc).__name__}: {exc}")
        return False


def prefs_text(rows) -> str:
    """Render banked preferences for a plan prompt."""
    notes = [str((r or {}).get("note") or "").strip() for r in (rows or [])]
    notes = [n for n in notes if n]
    if not notes:
        return ""
    return ("Things this user has told us since their last plan. Honour ALL "
            "of them:\n" + "\n".join(f"- {n}" for n in notes))


# ---------------------------------------------------------------- storage --
def enabled() -> bool:
    return bool(os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_KEY"))


def _url(): return os.getenv("SUPABASE_URL", "").rstrip("/") + f"/rest/v1/{TABLE}"


def _headers(extra=None):
    k = os.getenv("SUPABASE_KEY", "")
    h = {"apikey": k, "Authorization": f"Bearer {k}", "Content-Type": "application/json"}
    h.update(extra or {})
    return h


async def save(uid: str, plan: dict, log=None) -> bool:
    log = log or (lambda m: None)
    if not enabled():
        return False
    try:
        import httpx
        row = {"firebase_uid": uid, "week_start": plan.get("week_start", ""),
               "plan": plan, "updated_at": dt.datetime.now(dt.timezone.utc).isoformat()}
        async with httpx.AsyncClient(timeout=TIMEOUT) as c:
            r = await c.post(_url(), headers=_headers(
                {"Prefer": "resolution=merge-duplicates"}), json=row)
            r.raise_for_status()
        return True
    except Exception as exc:
        log(f"plan save failed: {type(exc).__name__}: {exc}")
        return False


async def uids_with_plans(log=None) -> list:
    """Everyone who already has a plan. Used by the Sunday sweep."""
    log = log or (lambda m: None)
    if not enabled():
        return []
    try:
        import httpx
        async with httpx.AsyncClient(timeout=TIMEOUT) as c:
            r = await c.get(_url(), headers=_headers(),
                            params={"select": "firebase_uid", "limit": "500"})
            r.raise_for_status()
            rows = r.json() or []
        seen, out = set(), []
        for row in rows:
            u = (row or {}).get("firebase_uid")
            if u and u not in seen:
                seen.add(u)
                out.append(u)
        return out
    except Exception as exc:
        log(f"plan uid listing failed: {type(exc).__name__}: {exc}")
        return []


async def load(uid: str, log=None):
    log = log or (lambda m: None)
    if not enabled() or not uid:
        return None
    try:
        import httpx
        async with httpx.AsyncClient(timeout=TIMEOUT) as c:
            r = await c.get(_url(), headers=_headers(),
                            params={"firebase_uid": f"eq.{uid}", "limit": "1"})
            r.raise_for_status()
            rows = r.json() or []
        return (rows[0] or {}).get("plan") if rows else None
    except Exception as exc:
        log(f"plan load failed: {type(exc).__name__}: {exc}")
        return None
