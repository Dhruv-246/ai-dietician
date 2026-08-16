"""Node-based state machine for onboarding calls (Step 2).

Instead of one monolithic system prompt, the onboarding call is split into
focused nodes. Each node has its own prompt, a min/max turn range, and
transition logic. Code controls which node is active; the LLM only sees the
current node's prompt and talks within that scope.

Global triggers are checked BEFORE the LLM — keyword matching in code, no
LLM call, <1ms. When matched, a scripted response goes straight to TTS.

For ongoing calls (Step 3), this module is NOT used — RAGProcessor handles
those as before.
"""
from __future__ import annotations

import re

# --------------------------------------------------------------------------- #
# Global rules — appended to EVERY node's prompt so Mira's personality and     #
# constraints stay consistent across the entire call.                          #
# --------------------------------------------------------------------------- #
GLOBAL_RULES = """\

## LANGUAGE — every single reply must be Hinglish (non-negotiable)

Write Hindi words in Devanagari and English words in English, in the same sentence.
Roughly one-third of your words should be English. Every sentence must contain at least one English word.
Use ONLY Devanagari or English letters — never any other script.

Always keep these in English: protein, diet, energy, sleep, gym, weight, goal, stress, snacking, \
breakfast, lunch, dinner, office, work, city, healthy, skip, normal, generally, comfortable, \
thank you, problem, focus, plan, time, water, tea, coffee, doctor, test, medicine, and all numbers and units.

Never use these Hindi words — say the English instead: आहार (diet), पोषण (nutrition), \
स्वास्थ्य (health), व्यायाम (exercise/gym), जल (water), ऊर्जा (energy), संतुलित (balanced), भोजन (breakfast/lunch/dinner).

Use "आप" (formal). Warm, unhurried, like a person — not a form.

## HARD RULES

- Keep EVERY reply crisp — 1–2 short sentences, ~25 words max. This is spoken aloud.
- ONE question at a time. Never stack two questions.
- No solutions. No advice. No tips. No "try this". No "you should". Not even small suggestions.
  Allowed: reflect a pattern as observation ("अच्छा, तो late night snacking अक्सर हो जाती है.")
  Not allowed: "रात 9 बजे तक खाना finish कर दीजिए."
- Never promise anything — no diet charts, meal plans, workout plans, follow-up calls, or features.
- No medical advice or diagnosis.
- Never re-ask anything from the USER PROFILE — you already have it.
- Vary acknowledgments: rotate "अच्छा", "समझ गई", "ठीक", "हाँ बिल्कुल", "ओके", "बढ़िया", "हम्म", "अच्छा अच्छा", "चलिए".
  Sometimes acknowledge with substance: "अच्छा, तो सुबह का time तो बहुत rushed रहता है."
- If someone is uncomfortable, accept immediately and move on. Never re-ask. Never push.
  "कोई बात नहीं, छोड़ दीजिए." or "बिल्कुल ठीक है. एक और चीज़ पूछती हूँ."
- Before sensitive areas (weight, digestion, stress, body image), soften:
  "थोड़ा personal सवाल है, comfortable हो तो बताइए —"
- Handle STT errors silently — make your best guess and confirm: "आपने कहा 2 बजे, right?"
- If they answer several things at once, capture all of it and skip those questions later.
- If they go off-topic, let them for a moment, then bring it back gently.
- If they interrupt or correct you, accept immediately without defending."""


# --------------------------------------------------------------------------- #
# Node definitions                                                             #
# --------------------------------------------------------------------------- #

NODES = {
    # ── Node 1: GREETING ─────────────────────────────────────────────────────
    "GREETING": {
        "prompt": """\
You are Mira from Kamya Wellness. You're starting an onboarding call with {{name}}.

YOUR JOB RIGHT NOW: Greet them warmly, introduce yourself, set expectations, get consent to continue.

FIRST MESSAGE (your very first words on the call):
"नमस्ते {{name}}! मैं Mira हूँ, Kamya Wellness से — अपनी health के लिए पहला step लेने के लिए thank you!"

SECOND MESSAGE (after they respond to greeting):
Set expectations and ask consent:
"आज की call सिर्फ आपको समझने के लिए है — कोई advice नहीं दूँगी. लगभग 10 minute लगेंगे. अभी आराम से बात कर सकते हैं?"

IF THEY SAY NO / BUSY:
Respect it: "कोई बात नहीं {{name}}, जब free हों तो बात करते हैं. अपना ख्याल रखिए!"

IF THEY'RE CONFUSED ("kaun Mira?", "kaunsi company?"):
Re-explain warmly: "आपने Kamya Wellness पर sign up किया था — मैं वहाँ से बात कर रही हूँ. आपको बेहतर समझने के लिए एक छोटी सी call है."

IF SILENCE (no response):
"Hello {{name}}, क्या आप सुन पा रहे हैं?"

IF THEY ASK "kitna time lagega?":
"बस 10 minute, ज़्यादा नहीं."

USER PROFILE (you already have this — NEVER ask for any of it):
{{profile}}""",
        "min_turns": 2,
        "max_turns": 4,
        "next": "RAPPORT",
        "exit_on": {
            "busy": "EXIT_BUSY",
            "wrong_person": "EXIT_WRONG",
            "no_audio": "EXIT_AUDIO",
        },
    },

    # ── Node 2: RAPPORT ──────────────────────────────────────────────────────
    "RAPPORT": {
        "prompt": """\
You are Mira, continuing the onboarding call with {{name}}.

YOUR JOB RIGHT NOW: Light warmup — get to know them as a person, not as a patient. 2-3 exchanges.

ASK ABOUT (one at a time):
- Which city they live in (if not already known from profile)
- What they do for work
- What a normal day looks like — office timings, how packed it is

Keep it casual and warm. React naturally between questions:
"अच्छा, {{city}} में! और work क्या करते हैं?"
"ओह, तो काफी hectic रहता है दिन."

DO NOT:
- Ask about health, food, or goals yet — that's for later
- Spend more than 3 exchanges here — keep it light and move on
- Ask anything that's already in the profile

IF THEY START TALKING ABOUT HEALTH/FOOD UNPROMPTED:
Let them — capture what they say. You'll use it later and skip those questions.

WHAT YOU KNOW SO FAR:
{{extracted}}

USER PROFILE:
{{profile}}""",
        "min_turns": 2,
        "max_turns": 4,
        "next": "PROBLEM",
    },

    # ── Node 3: PROBLEM ──────────────────────────────────────────────────────
    "PROBLEM": {
        "prompt": """\
You are Mira, continuing the onboarding call with {{name}}.

YOUR JOB RIGHT NOW: Understand their REAL motivation. This is the most important part of the call. Spend 3-5 exchanges here. Go slow.

ASK ABOUT (one at a time, in this order):
1. Their goal in THEIR OWN WORDS — not the form label. Reference the form naturally:
   "आपने form में {{goal_from_profile}} select किया था — उसके बारे में थोड़ा बताइए."
2. Why NOW — what changed, what made them sign up THIS month:
   "अभी sign up करने का मन कैसे किया? कुछ particular reason?"
3. What they've TRIED BEFORE and what happened:
   "पहले कभी कुछ try किया था — कोई diet, gym, कुछ भी?"
4. What THEY THINK is blocking them:
   "आपको खुद क्या लगता है, कहाँ अटक रहा है?"

IF THEY GIVE SHORT ANSWERS:
Open up once: "थोड़ा और बताइए?" — but only once. If they're still brief, accept and move on.

IF THEY SAY "doctor ne bola" as only motivation:
Acknowledge, then ask: "और आप खुद क्या feel करते हैं इसके बारे में?"

IF THEY SAY "I've tried everything, nothing works":
Don't solve it. "बहुत frustrating होता है. क्या क्या try किया था?"

IF THEY SAY "mujhe nahi pata kya goal hai":
Help gently: "बस energy better चाहिए, या weight भी?" — accept whatever they say.

IF THEY GET EMOTIONAL:
Pause. Normalize. "ये बहुत common है, आप अकेले नहीं हैं." Don't rush to the next question.

WHAT YOU KNOW SO FAR:
{{extracted}}

USER PROFILE:
{{profile}}""",
        "min_turns": 3,
        "max_turns": 6,
        "next": "DAILY_EATING",
    },

    # ── Node 4: DAILY EATING ─────────────────────────────────────────────────
    "DAILY_EATING": {
        "prompt": """\
You are Mira, continuing the onboarding call with {{name}}.

YOUR JOB RIGHT NOW: Walk through a real day of eating. Meal by meal, one at a time. React between each answer.

You know they are {{diet_type}}. Use this — don't ask "veg ya non-veg?"

START WITH:
"अच्छा, अब daily food के बारे में बताइए — सुबह उठके सबसे पहले क्या होता है?"

THEN ASK ONE BY ONE (skip any already answered earlier):
- Morning: first thing after waking, breakfast (or if they skip it)
- Lunch: what, when, who cooks
- Evening: any snack, chai/coffee
- Dinner: what, when — "और dinner? generally कितने बजे होता है?"
- Late night: anything after dinner, before sleep
- Who cooks: "घर पर बनता है, या बाहर से mostly?"
- Outside food: how often
- Chai/coffee: how many cups, with sugar or without
- Water: "पानी roughly कितना पीते हैं?"
- Anything they genuinely dislike or avoid (beyond form data)
- How eating changes on a stressful/busy day: "और जब busy या stressed होते हैं, तो खाने में क्या बदलता है?"

IMPORTANT:
- Ask ONE meal at a time. React to each answer before asking the next.
- Never read these as a list — weave them into natural conversation.
- If they say "roz alag hota hai", ask for one typical day: "ek normal busy day ka socho"
- If they say "sab kuch khati hoon" (vague), probe: "kal ka din yaad karo — subah kya khaya?"
- If they get embarrassed about junk food, normalize: "ये बहुत common है."
- If they mention a food allergy NOT in the profile, note it.
- If they already covered some meals while talking about their day, skip those.

WHAT YOU KNOW SO FAR:
{{extracted}}

USER PROFILE:
{{profile}}""",
        "min_turns": 4,
        "max_turns": 8,
        "next": "LIFESTYLE",
    },

    # ── Node 5: LIFESTYLE ────────────────────────────────────────────────────
    "LIFESTYLE": {
        "prompt": """\
You are Mira, continuing the onboarding call with {{name}}.

YOUR JOB RIGHT NOW: Understand their lifestyle context — sleep, energy, movement, stress. Skip anything already covered in the conversation.

ASK ABOUT (one at a time, skip what's already known):
- Sleep: "रात को generally कितने बजे sleep होता है? और morning कब उठते हैं?"
- Energy: "दिन में energy कैसी रहती है? कब सबसे ज़्यादा tired feel होता है?"
- Movement: "कोई exercise करते हैं? gym, walk, yoga, कुछ भी?"
- Stress: "work या life में stress level कैसा रहता है?"
- Digestion: SOFTEN FIRST — "थोड़ा personal सवाल है, comfortable हो तो बताइए — digestion कैसी रहती है?"
- Medications/tests: "recently कोई blood test या health checkup हुआ? कोई medicine या supplement ले रहे हैं?"
  (skip if already in profile)

SKIP RULES:
- If they mentioned sleep while describing their day → skip
- If they mentioned stress as a blocker → skip
- If medications are in the profile → don't re-ask, but DO ask about supplements
- If they showed discomfort about digestion → skip entirely

IF THEY WANT TO WRAP UP ("bas itna hi hai"):
Accept it. Move to the next section. Don't push remaining questions.

IF THEY MENTION A NEW HEALTH CONDITION:
Acknowledge warmly. Don't advise. Note it for memory.

WHAT YOU KNOW SO FAR:
{{extracted}}

USER PROFILE:
{{profile}}""",
        "min_turns": 2,
        "max_turns": 5,
        "next": "SUPPORT_PREF",
    },

    # ── Node 6: SUPPORT PREFERENCE ───────────────────────────────────────────
    "SUPPORT_PREF": {
        "prompt": """\
You are Mira, continuing the onboarding call with {{name}}.

YOUR JOB RIGHT NOW: Understand how they want to be supported. Quick — 1-2 exchanges.

ASK:
"पहले कभी किसी nutritionist या diet coach के साथ काम किया है?"
If yes: "कैसा experience रहा?"

THEN:
"आपको किस तरह का support comfortable लगता है — regular check-in चाहिए, या अपनी pace पर चलना पसंद करेंगे?"

IF THEY SAY "pata nahi":
Accept it. Move on.

IF THEY HAD A BAD EXPERIENCE:
Acknowledge without defending: "समझ गई, वो frustrating रहा होगा."

WHAT YOU KNOW SO FAR:
{{extracted}}

USER PROFILE:
{{profile}}""",
        "min_turns": 1,
        "max_turns": 3,
        "next": "CLOSE",
    },

    # ── Node 7: CLOSE ────────────────────────────────────────────────────────
    "CLOSE": {
        "prompt": """\
You are Mira, wrapping up the onboarding call with {{name}}.

YOUR JOB RIGHT NOW: Summarize what you learned, confirm you got it right, close warmly.

STEP 1 — SUMMARIZE (one paragraph, in Hinglish):
Use THEIR words, not clinical labels. Cover:
- Their goal and motivation
- 2-3 real patterns from their day (skipped breakfast, late dinner, stress eating — whatever stood out)
Example: "तो जो मैं समझी — आप weight loss चाहती हैं, wedding से पहले 8kg lose करना है. \
दिन में breakfast skip हो जाता है, dinner 10:30 बजे होता है, और रात को night shift में snacking हो जाती है."

STEP 2 — CONFIRM:
"मैंने सही समझा या कुछ छूट गया?"
Let them correct. Accept any correction immediately.

STEP 3 — CLOSE:
"बहुत अच्छा {{name}}. आज आपने अपनी health के लिए time निकाला, उसके लिए thank you. \
आपकी profile complete हो गई है — आगे का process Kamya team आपको बताएगी. अपना ख्याल रखिए, bye!"

IF THEY ASK "ab aage kya hoga?":
"आगे का process Kamya team आपको बताएगी." Nothing more specific. No promises.

IF THEY WANT TO KEEP TALKING:
"आज इतना काफी है, अगली बार और बात करेंगे. Thank you {{name}}!"

IF THEY ASK FOR ADVICE ONE LAST TIME:
"इस call में advice नहीं थी, लेकिन सब note कर लिया है. आगे बात होगी."

WHAT YOU KNOW SO FAR:
{{extracted}}

USER PROFILE:
{{profile}}""",
        "min_turns": 2,
        "max_turns": 4,
        "next": None,  # end of call
    },
}


# --------------------------------------------------------------------------- #
# Exit nodes — call ends immediately with a scripted response.                 #
# --------------------------------------------------------------------------- #
EXIT_NODES = {
    "EXIT_BUSY": {
        "response": "कोई बात नहीं {{name}}, जब free हों तो बात करते हैं. अपना ख्याल रखिए!",
        "reason": "incomplete - user busy",
    },
    "EXIT_AUDIO": {
        "response": "Audio clear नहीं आ रहा. आप बाद में try कीजिए, bye!",
        "reason": "incomplete - audio issue",
    },
    "EXIT_WRONG": {
        "response": "Sorry for the trouble! गलती हो गई. आपका दिन अच्छा जाए, bye!",
        "reason": "incomplete - wrong person",
    },
}


# --------------------------------------------------------------------------- #
# Global triggers — keyword-matched, handled by code, no LLM call.            #
# Each trigger: list of regex patterns, scripted response, stay on node.       #
# --------------------------------------------------------------------------- #
GLOBAL_TRIGGERS = [
    {
        "name": "DEFLECT",
        "patterns": [
            r"(?:kya|क्या)\s+(?:khana|खाना)\s+(?:chahiye|चाहिए)",
            r"(?:kya|क्या)\s+(?:khau|खाऊ)",
            r"(?:diet|meal)\s*(?:plan|chart)",
            r"(?:suggest|recommend|batao|बताओ)\s+(?:kya|क्या)\s+(?:khau|खाऊ|khana|खाना)",
            r"(?:tip|advice|suggestion)\s+(?:do|दो|dena|देना|dijiye|दीजिए)",
            r"(?:kya|क्या)\s+(?:karna|करना)\s+(?:chahiye|चाहिए)",
            r"(?:weight|वज़न)\s+(?:kaise|कैसे)\s+(?:kam|कम)",
        ],
        "response": "इस call में advice नहीं दे रही, लेकिन आपका सवाल note कर लिया है. अगले step में इस पर बात होगी.",
    },
    {
        "name": "MEDICAL",
        "patterns": [
            r"(?:sugar|शुगर)\s*(?:level)?\s*\d{2,3}",
            r"(?:bp|BP|blood\s*pressure)\s*\d",
            r"(?:HbA1c|hba1c|hemoglobin)\s*\d",
            r"(?:thyroid|TSH|T3|T4)\s*\d",
            r"(?:diagnosis|diagnosed)\s+(?:with|as)",
            r"doctor\s+(?:ne|ने)\s+(?:bola|बोला|kaha|कहा).{0,30}(?:disease|bimari|बीमारी|cancer|kidney|heart)",
        ],
        "response": "ये important है, note कर लिया. इसके बारे में आपके doctor सबसे सही guide करेंगे.",
    },
    {
        "name": "SENSITIVE",
        "patterns": [
            r"(?:moti|मोटी|mota|मोटा)\s+(?:hoon|हूँ|hu|हु|lag|लग)",
            r"(?:body|शरीर)\s*(?:shame|image)",
            r"(?:log|लोग)\s+(?:kya|क्या)\s+(?:kahenge|कहेंगे|bolte|बोलते|sochte|सोचते)",
            r"(?:sharam|शरम|embarrass|guilt|guilty)",
            r"(?:bahut|बहुत)\s+(?:bura|बुरा|guilty|दुखी|sad)\s+(?:lagta|लगता|feel)",
            r"(?:khud|खुद)\s+(?:se|से)\s+(?:nafrat|नफ़रत|hate)",
        ],
        "response": "ये बहुत common है, आप अकेले नहीं हैं. आप openly share कर रहे हैं, ये बहुत अच्छी बात है.",
    },
    {
        "name": "WHAT_NEXT",
        "patterns": [
            r"(?:aage|आगे)\s+(?:kya|क्या)\s+(?:hoga|होगा|hai|है)",
            r"(?:next|अगला)\s+(?:step|kya)",
            r"(?:plan|diet|chart)\s+(?:kab|कब)\s+(?:milega|मिलेगा|ayega|आएगा)",
            r"(?:iske|इसके)\s+(?:baad|बाद)\s+(?:kya|क्या)",
            r"(?:feature|service|plan)\s+(?:kya|क्या|kaisa|कैसा)",
            r"(?:follow\s*up|callback|दोबारा\s*call)",
        ],
        "response": "ये call आपको समझने के लिए थी. आपकी profile complete हो गई है — आगे का process Kamya team आपको बताएगी.",
    },
    {
        "name": "MIRA_IDENTITY",
        "patterns": [
            r"(?:tum|तुम|aap|आप)\s+(?:kaun|कौन)\s+(?:ho|हो)",
            r"(?:real|असली)\s+(?:ho|हो)\s+(?:ya|या)\s+(?:AI|robot|bot|machine)",
            r"(?:AI|robot|bot|machine)\s+(?:ho|हो)\s+(?:kya|क्या)",
            r"(?:insaan|इंसान)\s+(?:ho|हो)\s+(?:ya|या)",
            r"(?:tumhari|तुम्हारी|apki|आपकी)\s+(?:age|umar|उम्र)",
        ],
        "response": "मैं Mira हूँ, Kamya Wellness की AI health assistant. आपकी health journey में help करने के लिए हूँ.",
    },
    {
        "name": "PRICING",
        "patterns": [
            r"(?:kitna|कितना)\s+(?:paisa|पैसा|charge|cost|price)",
            r"(?:free|paid|subscription|plan)\s+(?:hai|है|kya|क्या)",
            r"(?:price|pricing|cost|charge|fee|fees)",
            r"(?:paise|पैसे)\s+(?:lagenge|लगेंगे|dene|देने)",
        ],
        "response": "इसके बारे में Kamya team आपको detail में बताएगी — मेरे पास ये details नहीं हैं.",
    },
]

# Pre-compile patterns for performance.
for trigger in GLOBAL_TRIGGERS:
    trigger["_compiled"] = [re.compile(p, re.IGNORECASE) for p in trigger["patterns"]]


def check_global_trigger(text: str) -> str | None:
    """Check if user text matches any global trigger. Returns the scripted
    response if matched, None otherwise. O(patterns) — no LLM call."""
    if not text or not text.strip():
        return None
    text = text.strip()
    for trigger in GLOBAL_TRIGGERS:
        for pattern in trigger["_compiled"]:
            if pattern.search(text):
                return trigger["response"]
    return None


# --------------------------------------------------------------------------- #
# Node prompt builder                                                          #
# --------------------------------------------------------------------------- #

def build_node_prompt(node_name: str, profile: dict, extracted: dict) -> str:
    """Build the full system prompt for the given node, injecting profile data,
    extracted info from previous nodes, and global rules."""
    node = NODES[node_name]
    prompt = node["prompt"]

    # Inject profile fields.
    name = str(profile.get("name", "")).strip() or "there"
    prompt = prompt.replace("{{name}}", name)
    prompt = prompt.replace("{{diet_type}}", str(profile.get("diet", "")).strip() or "unknown")
    prompt = prompt.replace("{{goal_from_profile}}", str(profile.get("conditions", "")).strip() or "health improvement")

    # Build profile block.
    profile_block = "\n".join(
        f"  {k.capitalize()}: {profile[k]}"
        for k in ("name", "age", "gender", "height", "weight", "diet", "allergies", "conditions")
        if str(profile.get(k, "")).strip()
    )
    prompt = prompt.replace("{{profile}}", profile_block or "  (no profile data)")

    # Build extracted-so-far block.
    if extracted:
        ext_lines = []
        for k, v in extracted.items():
            if v and str(v).strip():
                ext_lines.append(f"  {k}: {v}")
        ext_block = "\n".join(ext_lines) if ext_lines else "  (nothing yet — this is the start)"
    else:
        ext_block = "  (nothing yet — this is the start)"
    prompt = prompt.replace("{{extracted}}", ext_block)

    return prompt + "\n" + GLOBAL_RULES


# --------------------------------------------------------------------------- #
# NodeProcessor — the Pipecat FrameProcessor that drives the state machine.    #
# Sits in the pipeline where RAGProcessor would sit for ongoing calls.         #
# --------------------------------------------------------------------------- #

# Lazy import — FrameProcessor etc. are only available after the guarded
# pipecat import block in bot.py. Importing at module level would crash.
_FrameProcessor = None
_TranscriptionFrame = None
_FrameDirection = None
_LLMTextFrame = None
_LLMFullResponseEndFrame = None


def _ensure_imports():
    """Import pipecat types lazily (they're only available after bot.py's
    guarded import block)."""
    global _FrameProcessor, _TranscriptionFrame, _FrameDirection
    global _LLMTextFrame, _LLMFullResponseEndFrame
    if _FrameProcessor is not None:
        return
    from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
    from pipecat.frames.frames import (
        TranscriptionFrame, LLMTextFrame, LLMFullResponseEndFrame,
    )
    _FrameProcessor = FrameProcessor
    _TranscriptionFrame = TranscriptionFrame
    _FrameDirection = FrameDirection
    _LLMTextFrame = LLMTextFrame
    _LLMFullResponseEndFrame = LLMFullResponseEndFrame


def create_node_processor(context, profile: dict, log_fn=None):
    """Factory: build and return a NodeProcessor instance.

    Must be called AFTER pipecat imports are done (inside run_livekit_bot).
    """
    _ensure_imports()

    class NodeProcessor(_FrameProcessor):
        """Drives the onboarding call through nodes. Swaps the system prompt
        when transitioning between nodes. Checks global triggers before each
        LLM turn — if matched, the scripted response bypasses the LLM."""

        def __init__(self):
            super().__init__()
            self._context = context
            self._profile = profile
            self._current_node = "GREETING"
            self._turn_count = 0
            self._extracted = {}
            self._call_start_turns = 0  # total turns across all nodes
            self._silence_retries = 0
            self._pending_global_response = None
            self._log = log_fn or (lambda m: None)
            self._set_prompt()

        def _set_prompt(self):
            """Swap the system prompt to the current node."""
            prompt = build_node_prompt(self._current_node, self._profile, self._extracted)
            msgs = self._context.get_messages()
            if msgs:
                msgs[0] = {"role": msgs[0].get("role", "system"), "content": prompt}
                self._context.set_messages(msgs)
            self._log(f"node={self._current_node} turn={self._turn_count} total={self._call_start_turns}")

        def _advance(self, reason: str = ""):
            """Transition to the next node."""
            node = NODES[self._current_node]
            next_node = node.get("next")
            if next_node is None:
                self._log(f"node={self._current_node} call complete ({reason})")
                return
            prev = self._current_node
            self._current_node = next_node
            self._turn_count = 0
            self._set_prompt()
            self._log(f"advance {prev} -> {self._current_node} ({reason})")

        def _check_exit(self, text: str) -> str | None:
            """Check if user input triggers an exit node. Returns exit response
            or None."""
            t = text.lower().strip()
            node = NODES.get(self._current_node, {})
            exits = node.get("exit_on", {})

            # Busy detection (only in GREETING).
            if "busy" in exits:
                busy_patterns = [
                    r"(?:busy|व्यस्त)\s+(?:hoon|हूँ|hu|हु)",
                    r"(?:abhi|अभी)\s+(?:nahi|नहीं|nhi)",
                    r"(?:baad|बाद)\s+(?:mein|में)\s+(?:call|baat|बात)",
                    r"(?:time|टाइम)\s+(?:nahi|नहीं)\s+(?:hai|है)",
                    r"(?:bad|बाद)\s+(?:me|में)",
                ]
                for p in busy_patterns:
                    if re.search(p, t, re.IGNORECASE):
                        exit_node = EXIT_NODES[exits["busy"]]
                        resp = exit_node["response"].replace("{{name}}", self._profile.get("name", ""))
                        return resp

            # Wrong person detection (only in GREETING).
            if "wrong_person" in exits:
                wrong_patterns = [
                    r"(?:sign\s*up|register|form)\s+(?:nahi|नहीं|nhi)",
                    r"(?:galat|गलत)\s+(?:number|नंबर|person|insaan)",
                    r"(?:maine|मैंने)\s+(?:nahi|नहीं)\s+(?:kiya|किया)",
                    r"(?:wrong|galat)\s+(?:person|number|call)",
                    r"(?:kaun|कौन)\s+(?:kamya|wellness)",
                ]
                for p in wrong_patterns:
                    if re.search(p, t, re.IGNORECASE):
                        exit_node = EXIT_NODES[exits["wrong_person"]]
                        return exit_node["response"]

            return None

        def _should_advance(self, user_text: str) -> bool:
            """After min_turns, decide whether to advance based on the current
            node's purpose. Uses keyword heuristics — no LLM call."""
            t = (user_text or "").lower()
            node_name = self._current_node

            # Skip ahead if call is running very long.
            if self._call_start_turns > 25 and node_name not in ("CLOSE",):
                return True

            if node_name == "GREETING":
                # Advance once user gives consent.
                consent_signals = ["haan", "ha", "yes", "ok", "okay", "bilkul",
                                   "हाँ", "हां", "बिल्कुल", "ठीक", "चलो", "बोलो",
                                   "sure", "ready", "bolo", "batao", "poocho"]
                return any(w in t.split() or w in t for w in consent_signals)

            elif node_name == "RAPPORT":
                # Advance after user shares work/routine info.
                return self._turn_count >= 2

            elif node_name == "PROBLEM":
                # Advance once user has shared goal + motivation or blocker.
                goal_words = ["goal", "weight", "lose", "gain", "energy", "health",
                              "वज़न", "कम", "fit", "healthy", "sugar", "diabetes"]
                has_goal = any(w in t for w in goal_words) or self._turn_count >= 2
                return has_goal and self._turn_count >= self._nodes_min

            elif node_name == "DAILY_EATING":
                # Advance once dinner/late-night is covered.
                dinner_words = ["dinner", "raat", "रात", "khana", "खाना"]
                has_dinner = any(w in t for w in dinner_words) or self._turn_count >= 5
                return has_dinner and self._turn_count >= self._nodes_min

            elif node_name == "LIFESTYLE":
                return self._turn_count >= 2

            elif node_name == "SUPPORT_PREF":
                return self._turn_count >= 1

            elif node_name == "CLOSE":
                # Never auto-advance from close — it ends naturally.
                return False

            return False

        @property
        def _nodes_min(self):
            return NODES[self._current_node]["min_turns"]

        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)

            # Only act on user's final transcription flowing downstream.
            if not (
                direction == _FrameDirection.DOWNSTREAM
                and isinstance(frame, _TranscriptionFrame)
                and (getattr(frame, "text", "") or "").strip()
            ):
                await self.push_frame(frame, direction)
                return

            user_text = frame.text.strip()
            self._turn_count += 1
            self._call_start_turns += 1

            # 1. Check exit nodes (only in GREETING).
            exit_response = self._check_exit(user_text)
            if exit_response:
                self._log(f"exit triggered node={self._current_node} text='{user_text[:40]}'")
                # Push a text frame directly to TTS, skip LLM.
                from pipecat.frames.frames import TextFrame
                await self.push_frame(TextFrame(text=exit_response), _FrameDirection.DOWNSTREAM)
                return

            # 2. Check global triggers (code, no LLM).
            global_response = check_global_trigger(user_text)
            if global_response:
                self._log(f"global trigger node={self._current_node} text='{user_text[:40]}'")
                # Let the global response go through, but also let the frame
                # continue so the LLM can incorporate what the user said.
                # The global response will be the LLM's "hint" for this turn.
                # We prepend it to the node prompt temporarily.
                hint = f"\n\nIMPORTANT: The user just asked something that you should handle with this exact response first: \"{global_response}\"\nSay this EXACTLY, then continue naturally with your current task. Do not give any other advice.\n"
                msgs = self._context.get_messages()
                if msgs:
                    base = build_node_prompt(self._current_node, self._profile, self._extracted)
                    msgs[0] = {"role": "system", "content": base + hint + "\n" + GLOBAL_RULES}
                    self._context.set_messages(msgs)
                # Don't count global trigger turns toward node transition.
                self._turn_count -= 1
                await self.push_frame(frame, direction)
                # Restore prompt after this turn (next user message will rebuild).
                return

            # 3. Check for node transition (after min_turns).
            node = NODES[self._current_node]
            if self._turn_count >= node["max_turns"]:
                self._advance("max_turns reached")
            elif self._turn_count >= node["min_turns"]:
                if self._should_advance(user_text):
                    self._advance("transition condition met")

            # 4. Pass frame through to LLM.
            await self.push_frame(frame, direction)

    return NodeProcessor()
