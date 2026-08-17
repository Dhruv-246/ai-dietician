"""End-of-session memory consolidation (Step-3 architecture).

Runs ONCE when a call ends, off the audio path. Takes the user's EXISTING
long-term memory plus the transcript of the call that just happened, and asks a
Groq model to return the UPDATED (cumulative) memory + a session summary +
open loops to follow up on next time.

Cumulative rule: the model is told to keep everything still true, add new
durable facts, update changed ones, and drop only what was contradicted.

Open loops are MERGED (not regenerated): the existing open loops are passed in,
and the model removes only the ones this conversation resolved, keeps the rest
(even if not discussed this session), and adds any new follow-ups.

Uses a small/cheap Groq model by default (GROQ_CONSOLIDATION_MODEL) so this
background step doesn't eat the conversation model's daily token budget.
"""
import json
import os

from groq import Groq

# llama-3.1-8b-instant was decommissioned 2026-08-16. Its drop-in replacement
# (gpt-oss-20b) fails this prompt with json_validate_failed and an empty
# generation, so consolidation silently lost the whole memory write. The 120b
# follows the strict schema reliably. This runs once, after hangup — latency
# doesn't matter here, correctness does.
_MODEL = os.getenv("GROQ_CONSOLIDATION_MODEL", "openai/gpt-oss-120b")

_SYSTEM = """\
You maintain the long-term memory of a user for Mira, an AI dietician focused on Indian users.

You receive: the user's EXISTING long-term memory (JSON), the EXISTING open loops, and the TRANSCRIPT of the conversation that just happened.

Produce the UPDATED long_term_memory JSON.

## MASTER RULE

Your DEFAULT action is KEEP. Every single field and array item in the existing memory MUST appear in your output unless the user EXPLICITLY contradicted it. Forgetting to include something = deleting it. When in doubt, KEEP it.

## OPERATIONS

- ADD: new durable fact learned in this conversation that does not exist yet.
- UPDATE: a fact that changed — replace the old value (e.g. weight went from 74 to 72).
- DROP: ONLY when the user explicitly contradicted something (e.g. "I actually like oats now" → remove from dislikes, add to likes).
- KEEP: everything not discussed stays EXACTLY as is. If the user talked about food and never mentioned health, every health field must come back unchanged.

## WHAT TO STORE vs IGNORE

STORE — durable facts that help Mira give better advice in future sessions:
- Health conditions, medications, allergies (safety-critical — never miss these)
- Diet type, restrictions, eating patterns
- Food likes and dislikes
- Goals, motivation, targets
- Lifestyle context (schedule, cooking situation, household, budget)
- Progress signals (what worked, what failed, recurring struggles)
- Specific advice Mira gave and user's reaction to it (goes in entities)

IGNORE — do not store any of these:
- One-off food events ("I ate pizza today") — UNLESS user mentions it repeatedly across calls
- Greetings, small talk, filler ("accha", "hmm", "ok ok")
- Anything temporary that won't matter next session

Keep all entries as SHORT PHRASES, not full sentences.

## KEY-BY-KEY UPDATE RULES

### identity.basics (age, gender, city)
- UPDATE only when user explicitly states a new value ("I moved to Mumbai").
- These rarely change. If not discussed, KEEP as is.

### identity.body (height_cm, weight_kg)
- weight_kg: UPDATE whenever user reports a new weight. This is expected to change.
- height_cm: Almost never changes for adults. Update only if corrected.

### health.conditions (array of strings)
- SAFETY-CRITICAL. Never drop a condition unless user says "I was misdiagnosed" or similar.
- ADD any new condition mentioned ("doctor ne bola thyroid hai").
- Keep as short labels: "Type 2 diabetes", "PCOS", "hypothyroid", etc.

### health.medications (array of objects: {name, dosage, timing, frequency})
- Each medication is an object, NOT a plain string.
- ADD when user mentions a new medication.
- UPDATE when dosage/timing changes ("doctor ne dose badha diya").
- DROP only if user says they stopped a medication ("metformin band kar diya").
- If user mentions a medication name but not dosage/timing, store what you know, leave rest null.
- Example: {"name": "metformin", "dosage": "500mg", "timing": "after meals", "frequency": "2x/day"}

### health.allergies (array of strings)
- SAFETY-CRITICAL. Never drop unless explicitly corrected.
- ADD any allergy or intolerance mentioned.

### diet.type (string)
- Overall diet label: "vegetarian", "eggetarian", "non-veg", "vegan", "Jain", etc.
- UPDATE only if user explicitly changes ("I started eating eggs").

### diet.restrictions (array of strings)
- Religious, medical, or personal restrictions: "no onion-garlic", "Navratri fasting", "no beef".
- ADD new restrictions. DROP only if user says they stopped following one.

### current_pattern (object with 6 meal slots)
- Each slot: morning, mid_morning, lunch, evening, dinner, late_night
- Each slot has: {time, frequent, note, gaps}
  - time (string): when this meal usually happens ("8am", "late around 10:30pm"). Update when user gives timing info.
  - frequent (array of strings): foods the user REGULARLY eats at this slot. NOT one-off meals.
    - ADD a food ONLY when user says they eat it regularly ("roz poha khati hoon") or mentions it across 2+ conversations.
    - Do NOT add one-off events ("aaj pizza khaya" → ignore).
    - DROP when user says they stopped eating something regularly.
  - note (string): contextual info that doesn't fit a list item. "mom cooks", "heaviest meal", "skips 3 days/week".
  - gaps (array of strings): specific questions Mira still needs to ask about this meal slot.
    - ADD a gap when Mira asked something and the user didn't answer, deflected, or gave a vague answer.
    - REMOVE a gap when the user answered it in this conversation — move the answer into frequent/time/note.
    - Only add USEFUL gaps: "portion size?" is good. "what brand of atta?" is not useful.
    - Examples: "how many rotis?", "with sugar or without?", "portion of rice?"

### preferences.likes (array of strings)
- Foods the user enjoys. ADD when user expresses liking. DROP if moved to dislikes.

### preferences.dislikes (array of strings)
- Foods the user refuses or dislikes. ADD when user rejects something.
- IMPORTANT: if user says they now like something that was in dislikes, DROP from dislikes and ADD to likes.

### preferences.cuisine (string)
- Household cooking style: "Punjabi", "South Indian", "Gujarati", "mixed North Indian", etc.

### goals.primary_goal (string)
- Main objective: "lose 8kg", "manage blood sugar", "gain muscle", etc. UPDATE if goal changes.

### goals.motivation (string)
- Why they want this: "wedding in 4 months", "doctor said to lose weight", "want to feel energetic".

### goals.target (string)
- Specific target: "65kg", "HbA1c under 7", "fit in old clothes". UPDATE when user revises.

### lifestyle (schedule, cooking_situation, household, budget)
- schedule: work hours, sleep pattern ("night shifts, sleeps 2am").
- cooking_situation: who cooks, how ("mom cooks", "I cook on weekends", "mostly outside food").
- household: family context ("joint family, 5 people", "lives alone").
- budget: food budget level ("tight", "moderate", "not a concern").
- These change rarely. UPDATE only when user reports a change.

### progress.what_worked (array of strings)
- Things the user tried AND liked/continued. Short phrases: "poha breakfast — liked it".
- ADD when user reports positive results. Do NOT add Mira's suggestions — only what the USER confirmed worked.
- When an entity (advice/meal plan) gets positive user feedback, absorb it here.

### progress.what_failed (array of strings)
- Things the user tried AND quit/hated. "oats — hated, quit in 2 days".
- CRITICAL: Mira must never re-suggest these. ADD immediately when user reports failure.

### progress.struggles (array of strings)
- Recurring problems: "stress eating at night", "can't reduce chai", "skips breakfast".
- ADD new struggles. DROP only if user reports it's resolved ("ab raat ko nahi khati").

### entities (array of objects)
- Track SPECIFIC advice/plans Mira gave and what happened with them.
- Each entity: {"type": "...", "what": "...", "status": "...", "given_on": "..."}
  - type: "meal_plan", "advice", "food_swap", "meal_rotation", "habit_suggestion"
  - what: the specific recommendation in short form
  - status: "suggested", "trying", "following", "liked", "quit", "partially following"
  - given_on: date it was first given
- ADD when Mira gives a specific, actionable recommendation in THIS conversation.
- UPDATE status when user reports feedback on an existing entity.
- When an entity has been resolved (user clearly adopted it or abandoned it for 2+ sessions), absorb into progress.what_worked or progress.what_failed, then DROP the entity.
- Keep MAX 7 entities. If over 7, absorb the oldest resolved ones into progress.

### recent_exchanges (array of {role, text})
- The last 5-6 meaningful exchanges from THIS conversation (not older ones).
- REPLACE entirely each session — these are always from the MOST RECENT call only.
- Pick the most informative exchanges, skip filler ("hmm", "ok", "accha").
- Each entry: {"role": "user" or "assistant", "text": "short version of what was said"}
- Keep text SHORT — compress to the key content, not verbatim quotes.

### interaction_meta
- total_sessions: INCREMENT by 1 each call (existing value + 1).
- first_session: SET only if currently null (first ever call). Never change after that.
- last_session: SET to today's date (the model can infer from transcript context or use the existing value + 1 day if unclear).
- mood_last_call: user's emotional state THIS call. "happy", "frustrated", "neutral", "motivated", "anxious", etc. Infer from tone/content. Replace each session.

### misc (array of strings)
- Important facts that don't fit any category. Use sparingly.
- Examples: "afraid of needles", "traveling next week — diet will be disrupted".
- DROP items that are no longer relevant (trip that ended, temporary situation resolved).

## open_loops rules

VERY IMPORTANT — start from the EXISTING open loops, do not regenerate from scratch.

- REMOVE an existing open loop ONLY if this conversation clearly RESOLVED or answered it.
- KEEP every existing open loop that is still unresolved — even if it was NOT mentioned in this conversation. Never drop a loop just because it did not come up this time.
- ADD new follow-ups discovered in this conversation.
- Do NOT duplicate loops that mean the same thing.
- Return the FULL updated list = (kept unresolved) + (new), with resolved ones removed.

## OUTPUT — return STRICT JSON only (no prose, no markdown), exactly this shape:

{
  "long_term_memory": {
    "identity": {
      "basics": { "age": null, "gender": null, "city": null },
      "body": { "height_cm": null, "weight_kg": null }
    },
    "health": {
      "conditions": [],
      "medications": [],
      "allergies": []
    },
    "diet": {
      "type": null,
      "restrictions": []
    },
    "current_pattern": {
      "morning":     { "time": null, "frequent": [], "note": null, "gaps": [] },
      "mid_morning": { "time": null, "frequent": [], "note": null, "gaps": [] },
      "lunch":       { "time": null, "frequent": [], "note": null, "gaps": [] },
      "evening":     { "time": null, "frequent": [], "note": null, "gaps": [] },
      "dinner":      { "time": null, "frequent": [], "note": null, "gaps": [] },
      "late_night":  { "time": null, "frequent": [], "note": null, "gaps": [] }
    },
    "preferences": {
      "likes": [],
      "dislikes": [],
      "cuisine": null
    },
    "goals": {
      "primary_goal": null,
      "motivation": null,
      "target": null
    },
    "lifestyle": {
      "schedule": null,
      "cooking_situation": null,
      "household": null,
      "budget": null
    },
    "progress": {
      "what_worked": [],
      "what_failed": [],
      "struggles": []
    },
    "entities": [],
    "recent_exchanges": [],
    "interaction_meta": {
      "total_sessions": 0,
      "first_session": null,
      "last_session": null,
      "mood_last_call": null
    },
    "misc": []
  },
  "session_summary": "one or two sentence recap of THIS conversation",
  "open_loops": ["concrete things Mira should follow up on next time"]
}

Leave fields as null or [] when no information is available — do NOT guess or invent."""


def consolidate(existing_memory: dict, existing_open_loops: list, transcript: str) -> dict:
    """Return {long_term_memory, session_summary, open_loops}. Raises on failure.

    `existing_open_loops` is the user's current open loops; the model merges them
    with this conversation (removes resolved, keeps unresolved, adds new).
    """
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    user_msg = (
        "EXISTING MEMORY:\n"
        + json.dumps(existing_memory or {}, ensure_ascii=False)
        + "\n\nEXISTING OPEN LOOPS:\n"
        + json.dumps(existing_open_loops or [], ensure_ascii=False)
        + "\n\nTRANSCRIPT:\n"
        + (transcript or "").strip()
    )
    resp = client.chat.completions.create(
        model=_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    data = json.loads(resp.choices[0].message.content)
    return {
        "long_term_memory": data.get("long_term_memory", {}) or {},
        "session_summary": (data.get("session_summary", "") or "").strip(),
        "open_loops": data.get("open_loops", []) or [],
    }
