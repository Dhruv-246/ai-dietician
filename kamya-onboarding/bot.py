"""Kamya Wellness — voice onboarding agent (Pipecat + LiveKit).

A real-time voice bot that "calls" the user and runs a friendly onboarding
interview in Hinglish. Media is carried by LiveKit Cloud (a managed WebRTC
media server), so the call connects reliably from any host — including
Railway, which cannot do peer-to-peer WebRTC.

Pipeline:   mic → Sarvam STT → Groq (Llama-70B) → ElevenLabs TTS → speaker
Features:   barge-in (interruptions), Silero VAD turn-taking, streaming.

How it works:
  • The browser opens "/" (the green-start / red-hangup call screen).
  • Clicking the green button POSTs /connect. The server creates a unique
    LiveKit room, launches Mira (this bot) into it, and returns a join token.
  • The browser joins the same room with the LiveKit JS SDK. LiveKit relays
    the audio both ways, so no direct UDP path to the server is needed.

Run locally:  python bot.py  → open the printed URL.
Requires: GROQ_API_KEY, SARVAM_API_KEY (STT), ELEVENLABS_API_KEY (TTS) and
          LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET.
"""
import argparse
import asyncio
import json
import os
import re
import sys
import tempfile
import time
import traceback
import uuid
from collections import deque
from pathlib import Path

from dotenv import load_dotenv

# --- NLTK CWD-guard workaround (must wrap ALL pipecat imports) ------------- #
# Pipecat imports NLTK, whose 2026 security hook (nltk/inisec.py) blocks
# importing any package that resolves to a path INSIDE the current working
# directory. When the virtualenv lives inside the project folder (the usual
# case), that wrongly blocks NLTK's own deps and the import crashes. We dodge
# it by importing Pipecat while the cwd is a temp dir, then restore the cwd.
_ORIG_CWD = os.getcwd()
os.chdir(tempfile.gettempdir())
try:
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.audio.vad.vad_analyzer import VADParams
    from pipecat.frames.frames import (
        BotStartedSpeakingFrame,
        ErrorFrame,
        LLMFullResponseEndFrame,
        LLMFullResponseStartFrame,
        LLMRunFrame,
        LLMTextFrame,
        TranscriptionFrame,
        TTSStartedFrame,
        TTSAudioRawFrame,
        UserStartedSpeakingFrame,
        BotStoppedSpeakingFrame,
    )
    from pipecat.observers.base_observer import BaseObserver, FramePushed
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.runner import PipelineRunner
    from pipecat.pipeline.task import PipelineParams, PipelineTask
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.processors.aggregators.llm_response_universal import (
        LLMContextAggregatorPair,
        LLMUserAggregatorParams,
    )
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
    from pipecat.turns.user_start import MinWordsUserTurnStartStrategy
    from pipecat.turns.user_turn_strategies import UserTurnStrategies
    from pipecat.utils.text.base_text_filter import BaseTextFilter
    from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
    from pipecat.services.google.tts import GeminiTTSService
    from pipecat.services.cartesia.tts import CartesiaTTSService
    from pipecat.services.sarvam.tts import SarvamTTSService
    from pipecat.services.sarvam.stt import SarvamSTTService
    from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService
    from pipecat.services.google.llm import GoogleLLMService
    from pipecat.services.deepseek.llm import DeepSeekLLMService
    from pipecat.services.groq.llm import GroqLLMService
    from pipecat.transcriptions.language import Language
    from pipecat.transports.livekit.transport import LiveKitParams, LiveKitTransport
    from pipecat.runner.livekit import generate_token, generate_token_with_agent
finally:
    os.chdir(_ORIG_CWD)
# --------------------------------------------------------------------------- #

from datetime import datetime, timezone

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse

import consolidate
import memory_store
import onboarding_nodes
import rag

# Local dev loads a .env next to this file. On a host (Railway/Render/etc.) there
# is no .env — the platform injects the variables into the environment directly.
_ENV_PATH = Path(__file__).with_name(".env")
if _ENV_PATH.exists():
    load_dotenv(_ENV_PATH)

# Fail early with a clear message if any required key is missing/blank.
_REQUIRED = [
    # GOOGLE_API_KEY drives Mira's live responses (Gemini). GROQ_API_KEY is used
    # only for the background memory consolidation. ELEVENLABS_API_KEY is optional
    # — if missing/out of credits, TTS falls back to Sarvam bulbul (SARVAM_API_KEY).
    "GOOGLE_API_KEY", "GROQ_API_KEY", "SARVAM_API_KEY",
    "LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET",
]
_missing = [k for k in _REQUIRED if not os.getenv(k)]
if _missing:
    sys.exit(
        f"[config] Missing keys: {', '.join(_missing)}. "
        f"Set them in Railway → Variables (or a local .env for dev)."
    )

# --------------------------------------------------------------------------- #
# Onboarding call prompt — loaded from call_prompt.md and filled with the      #
# user's MANUAL-ONBOARDING data (profile.json) so Mira already knows them and  #
# never re-asks form fields.                                                   #
#                                                                              #
# In PRODUCTION: replace load_profile() to fetch the row for the user this     #
# call is for (e.g. from the Google Sheet by user_id) instead of profile.json. #
# --------------------------------------------------------------------------- #
_PROFILE_FIELDS = ["name", "age", "gender", "height", "weight",
                   "diet", "allergies", "conditions"]


def load_profile() -> dict:
    """Fallback profile (local dev / no uid): read the sample profile.json."""
    try:
        return json.loads(Path(__file__).with_name("profile.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def load_profile_for_call(firebase_uid: str | None) -> dict:
    """Resolve the profile for THIS call.

    If a Firebase uid was handed off from manual onboarding, read that user's
    row from the shared Google Sheet so Mira greets them by name. Falls back to
    the local sample profile if there's no uid or the lookup fails.
    """
    if firebase_uid:
        try:
            import profile_store  # local, imported lazily so the server boots even without sheets deps
            profile = profile_store.load_profile_for_uid(firebase_uid)
            if profile:
                return profile
        except Exception as exc:  # never let a sheet hiccup block the call
            print(f"[profile] lookup failed for uid={firebase_uid!r}: {exc}")
    return load_profile()


def _build_user_context(profile: dict, memory: dict) -> str:
    """Assemble the 'what you already know' block for the ongoing (Step-3) prompt
    from the user's profile + cumulative long-term memory + continuity signals."""
    profile = profile or {}
    memory = memory or {}
    ltm = memory.get("long_term_memory") or {}
    lines = []

    # --- Onboarding profile (manual form data) ---
    prof_bits = [f"{k.capitalize()}: {profile[k]}" for k in _PROFILE_FIELDS
                 if str(profile.get(k, "")).strip()]
    if prof_bits:
        lines.append("Profile — " + " | ".join(prof_bits))

    # --- Backward compat: detect old flat format ---
    _has_new = any(isinstance(ltm.get(k), dict) for k in
                   ("identity", "health", "diet", "preferences", "goals",
                    "lifestyle", "progress", "current_pattern"))
    if not _has_new:
        for label, key in (("Goals", "goals"), ("Preferences", "preferences"),
                           ("Dislikes", "dislikes"), ("Known facts", "facts")):
            vals = ltm.get(key) or []
            if vals:
                lines.append(f"{label}: " + "; ".join(str(v) for v in vals))
        open_loops = memory.get("open_loops") or []
        if open_loops:
            lines.append("Open loops: " + "; ".join(str(v) for v in open_loops))
        return "\n".join(f"- {ln}" for ln in lines) if lines else "- (No prior context yet.)"

    # --- Identity ---
    identity = ltm.get("identity") or {}
    basics = identity.get("basics") or {}
    body = identity.get("body") or {}
    id_bits = [f"{k}: {v}" for k, v in basics.items() if v]
    id_bits += [f"{k}: {v}" for k, v in body.items() if v]
    if id_bits:
        lines.append("Identity — " + " | ".join(id_bits))

    # --- Health (safety-critical, marked with ⚠) ---
    health = ltm.get("health") or {}
    for label, key in (("Conditions", "conditions"),
                       ("Allergies", "allergies")):
        vals = health.get(key) or []
        if vals:
            lines.append(f"⚠ {label}: " + "; ".join(str(v) for v in vals))
    meds = health.get("medications") or []
    if meds:
        med_strs = []
        for m in meds:
            if isinstance(m, dict):
                parts = [m.get("name", "")]
                if m.get("dosage"):
                    parts.append(m["dosage"])
                if m.get("timing"):
                    parts.append(m["timing"])
                if m.get("frequency"):
                    parts.append(m["frequency"])
                med_strs.append(" ".join(p for p in parts if p))
            else:
                med_strs.append(str(m))
        lines.append("⚠ Medications: " + "; ".join(med_strs))

    # --- Diet type & restrictions ---
    diet = ltm.get("diet") or {}
    if diet.get("type"):
        lines.append(f"Diet type: {diet['type']}")
    if diet.get("restrictions"):
        lines.append("Restrictions: " + "; ".join(str(v) for v in diet["restrictions"]))

    # --- Current eating pattern (per meal slot) ---
    pattern = ltm.get("current_pattern") or {}
    _SLOT_LABELS = {
        "morning": "Morning", "mid_morning": "Mid-morning", "lunch": "Lunch",
        "evening": "Evening", "dinner": "Dinner", "late_night": "Late night",
    }
    pattern_lines = []
    gap_lines = []
    for slot_key, slot_label in _SLOT_LABELS.items():
        slot = pattern.get(slot_key) or {}
        if not isinstance(slot, dict):
            continue
        parts = []
        if slot.get("time"):
            parts.append(f"~{slot['time']}")
        freq = slot.get("frequent") or []
        if freq:
            parts.append(", ".join(str(f) for f in freq))
        if slot.get("note"):
            parts.append(f"({slot['note']})")
        if parts:
            pattern_lines.append(f"  {slot_label}: " + " — ".join(parts))
        gaps = slot.get("gaps") or []
        for g in gaps:
            gap_lines.append(f"  {slot_label}: {g}")
    if pattern_lines:
        lines.append("Current eating pattern:")
        lines.extend(pattern_lines)
    if gap_lines:
        lines.append("Gaps to fill (ask naturally when relevant, don't interrogate):")
        lines.extend(gap_lines)

    # --- Preferences ---
    prefs = ltm.get("preferences") or {}
    if prefs.get("likes"):
        lines.append("Likes: " + "; ".join(str(v) for v in prefs["likes"]))
    if prefs.get("dislikes"):
        lines.append("Dislikes (NEVER suggest these): " + "; ".join(str(v) for v in prefs["dislikes"]))
    if prefs.get("cuisine"):
        lines.append(f"Cuisine: {prefs['cuisine']}")

    # --- Goals ---
    goals = ltm.get("goals") or {}
    goal_bits = []
    if goals.get("primary_goal"):
        goal_bits.append(goals["primary_goal"])
    if goals.get("target"):
        goal_bits.append(f"target: {goals['target']}")
    if goals.get("motivation"):
        goal_bits.append(f"why: {goals['motivation']}")
    if goal_bits:
        lines.append("Goals: " + " | ".join(goal_bits))

    # --- Lifestyle ---
    lifestyle = ltm.get("lifestyle") or {}
    life_bits = [f"{k}: {v}" for k, v in lifestyle.items() if v]
    if life_bits:
        lines.append("Lifestyle — " + " | ".join(life_bits))

    # --- Progress ---
    progress = ltm.get("progress") or {}
    if progress.get("what_worked"):
        lines.append("What worked: " + "; ".join(str(v) for v in progress["what_worked"]))
    if progress.get("what_failed"):
        lines.append("What failed (don't repeat these): " + "; ".join(str(v) for v in progress["what_failed"]))
    if progress.get("struggles"):
        lines.append("Ongoing struggles: " + "; ".join(str(v) for v in progress["struggles"]))

    # --- Entities (active advice/plans Mira has given) ---
    entities = ltm.get("entities") or []
    if entities:
        lines.append("Active advice/plans you gave this user:")
        for e in entities:
            if isinstance(e, dict):
                status = e.get("status", "")
                what = e.get("what", "")
                given = e.get("given_on", "")
                lines.append(f"  - {what} (status: {status}, given: {given})")
            else:
                lines.append(f"  - {e}")

    # --- Recent exchanges (conversational continuity) ---
    recent = ltm.get("recent_exchanges") or []
    if recent:
        lines.append("Last conversation highlights:")
        for ex in recent[-6:]:
            if isinstance(ex, dict):
                role = "User" if ex.get("role") == "user" else "You"
                lines.append(f'  {role}: "{ex.get("text", "")}"')

    # --- Misc ---
    misc = ltm.get("misc") or []
    if misc:
        lines.append("Other notes: " + "; ".join(str(v) for v in misc))

    # --- Open loops ---
    open_loops = memory.get("open_loops") or []
    if open_loops:
        lines.append("Open loops (gently follow up on ONE if it fits — do not interrogate): "
                     + "; ".join(str(v) for v in open_loops))

    # --- Continuity signals ---
    meta = ltm.get("interaction_meta") or {}
    count = int(meta.get("total_sessions") or memory.get("session_count") or 0)
    if count > 0:
        cont = f"This is check-in #{count + 1}."
        last_at = str(meta.get("last_session") or memory.get("last_session_at") or "").strip()
        if last_at:
            try:
                delta = datetime.now(timezone.utc) - datetime.fromisoformat(last_at)
                days = max(0, delta.days)
                cont += f" Last call was {'today' if days == 0 else f'{days} day(s) ago'}."
            except Exception:
                pass
        mood = meta.get("mood_last_call")
        if mood:
            cont += f" Mood last time: {mood}."
        last_sum = str(memory.get("last_session_summary") or "").strip()
        if last_sum:
            cont += f' Last time: "{last_sum}"'
        lines.append(cont)

    return "\n".join(f"- {ln}" for ln in lines) if lines else "- (No prior context yet — this is your first talk with them.)"


def build_system_prompt(mode: str, profile: dict | None = None, memory: dict | None = None) -> str:
    """Build the system prompt for the call.

    mode="onboarding" → Mira LEADS the scripted onboarding interview (call_prompt.md).
    mode="ongoing"    → Mira answers the user's questions (ask_prompt.md), primed
                        with the user's profile + cumulative long-term memory.
    """
    profile = profile or {}
    if mode == "ongoing":
        template = Path(__file__).with_name("ask_prompt.md").read_text(encoding="utf-8")
        template = template.replace("{{name}}", str(profile.get("name", "")).strip() or "there")
        return template.replace("{{user_context}}", _build_user_context(profile, memory))

    # onboarding (Step 2) — unchanged behaviour.
    template = Path(__file__).with_name("call_prompt.md").read_text(encoding="utf-8")
    for key in _PROFILE_FIELDS:
        value = str(profile.get(key, "")).strip() or "—"
        template = template.replace("{{" + key + "}}", value)
    return template


# --- Lightweight diagnostics -------------------------------------------------
# A small in-memory ring buffer of call lifecycle events, exposed at /debug so
# call problems can be diagnosed over HTTP without shell access. Records NO user
# data (no names, no uids) — only whether they were present.
_EVENTS: deque = deque(maxlen=250)


def _log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    _EVENTS.append(line)
    print("[mira]", line, flush=True)


# Mira must only ever say/show Hindi (Devanagari) or English. LLMs occasionally
# leak a stray character from another script (e.g. a CJK token). This keeps only
# Devanagari + printable ASCII + whitespace + a few common punctuation marks and
# strips everything else — applied to both the spoken audio (TTS) and captions.
_FOREIGN_CHARS = re.compile(r"[^ऀ-ॿ -~\s–—‘’“”…]")


def _strip_foreign(text: str) -> str:
    return _FOREIGN_CHARS.sub("", text or "")


def _elevenlabs_has_credits() -> bool:
    """True if ElevenLabs is usable right now: key present, valid, and enough
    monthly credits left for a call. Any failure (no key, 401/402, network, low
    balance) returns False so TTS falls back to Sarvam bulbul — Mira never goes
    silent. Threshold is a full-call buffer so we don't switch to ElevenLabs
    only to run dry mid-call. (Blocking call — run via asyncio.to_thread.)"""
    key = os.getenv("ELEVENLABS_API_KEY")
    if not key:
        return False
    import urllib.request
    min_left = int(os.getenv("ELEVENLABS_MIN_CREDITS", "1500"))
    req = urllib.request.Request(
        "https://api.elevenlabs.io/v1/user/subscription",
        headers={"xi-api-key": key},
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as r:
            data = json.loads(r.read())
        remaining = data.get("character_limit", 0) - data.get("character_count", 0)
        return remaining >= min_left
    except Exception:
        return False


class ScriptTextFilter(BaseTextFilter):
    """TTS filter: remove non-Hindi/English characters before speech synthesis."""

    async def filter(self, text: str) -> str:
        return _strip_foreign(text)


class StylePrefixTextFilter(BaseTextFilter):
    """TTS filter (Gemini ONLY): prepend a natural-language style directive so
    Gemini TTS speaks in that style. Gemini treats a leading instruction as
    delivery steering and does NOT speak it. Never use with ElevenLabs/Sarvam —
    they would read the directive aloud."""

    def __init__(self, style: str):
        super().__init__()
        self._style = (style or "").strip()

    async def filter(self, text: str) -> str:
        if not self._style or not (text or "").strip():
            return text
        return f"{self._style}: {text}"


class _DiagObserver(BaseObserver):
    """Non-intrusive observer that records the greeting pipeline's milestones —
    LLM responding, TTS producing audio, bot speaking — and any ErrorFrame, so
    /debug can show whether the LLM/TTS actually produced audio or errored."""

    def __init__(self, room_name: str):
        super().__init__()
        self._room = room_name
        self._seen: set = set()

    async def on_push_frame(self, data: "FramePushed"):
        frame = data.frame
        if isinstance(frame, ErrorFrame):
            _log(f"ERROR FRAME: {getattr(frame, 'error', frame)}")
            return
        # Count audio frames quietly; log only the first + a running summary.
        if isinstance(frame, TTSAudioRawFrame):
            self._audio = getattr(self, "_audio", 0) + 1
            if self._audio == 1:
                _log(f"tts audio flowing (sr={getattr(frame,'sample_rate','?')})")
            return
        for cls, label in (
            (LLMFullResponseStartFrame, "llm responding"),
            (TTSStartedFrame, "tts started"),
            (BotStartedSpeakingFrame, "BOT speaking START"),
            (BotStoppedSpeakingFrame, f"BOT speaking STOP audio_frames={getattr(self,'_audio',0)}"),
            (UserStartedSpeakingFrame, "USER started speaking (may interrupt)"),
        ):
            if isinstance(frame, cls):
                _log(label)
                break


class _CaptionObserver(BaseObserver):
    """Streams live captions to the browser: the user's final transcript and
    Mira's spoken text, sent as JSON data messages over LiveKit so the call
    screen can show both sides of the conversation as text."""

    def __init__(self, transport):
        super().__init__()
        self._transport = transport
        self._seen_ids: set = set()     # dedupe: observer sees each frame once per hop
        self._bot_text: list = []       # accumulate Mira's streamed tokens

    async def _send(self, role: str, text: str):
        text = _strip_foreign(text).strip()  # captions: Hindi/English only
        if not text:
            return
        try:
            await self._transport.send_message(json.dumps({"role": role, "text": text}))
            _log(f"caption sent role={role} len={len(text)}")
        except Exception as exc:
            _log(f"caption send failed: {exc}")

    async def on_push_frame(self, data: "FramePushed"):
        frame = data.frame
        fid = getattr(frame, "id", None)
        if fid is not None:
            if fid in self._seen_ids:
                return
            self._seen_ids.add(fid)

        if isinstance(frame, TranscriptionFrame):        # user's final speech-to-text
            await self._send("user", frame.text)
        elif isinstance(frame, LLMTextFrame):            # Mira's streamed tokens
            self._bot_text.append(frame.text or "")
        elif isinstance(frame, LLMFullResponseEndFrame):  # Mira finished a reply
            full = "".join(self._bot_text)
            self._bot_text = []
            await self._send("assistant", full)


class RAGProcessor(FrameProcessor):
    """RAG injection. Sits between STT and the user aggregator. On each final
    user transcription it retrieves the closest dietician Q&A from Supabase and
    refreshes the base system prompt with a REFERENCE block (or clears it when
    nothing relevant is found), so Mira answers in style — not by copying.

    Why refresh the FIRST system message (not append a new one): Gemini only
    honours a single system instruction, so the reference must live inside the
    base prompt. We keep the original prompt in `self._base` and rebuild
    messages[0] each turn = base [+ references]. It never accumulates, and any
    retrieval error leaves the base prompt untouched (call proceeds normally)."""

    def __init__(self, context, *, top_k: int = 3, min_similarity: float = 0.5):
        super().__init__()
        self._context = context
        self._top_k = top_k
        self._min_sim = min_similarity
        msgs = context.get_messages()
        self._base = (msgs[0].get("content") if msgs else "") or ""

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        # Only act on the user's FINAL transcription, flowing downstream.
        if (
            direction == FrameDirection.DOWNSTREAM
            and isinstance(frame, TranscriptionFrame)
            and (getattr(frame, "text", "") or "").strip()
        ):
            try:
                await self._refresh(frame.text.strip())
            except Exception as exc:  # never break the call over retrieval
                _log(f"rag refresh failed room=? : {exc}")
        await self.push_frame(frame, direction)

    async def _refresh(self, question: str):
        matches = await rag.retrieve(question, k=self._top_k, min_similarity=self._min_sim)
        msgs = self._context.get_messages()
        if not msgs:
            return
        if matches:
            content = self._base + "\n\n" + rag.format_reference(matches)
            _log(f"rag matched {len(matches)} q='{question[:40]}'")
        else:
            content = self._base
        msgs[0] = {**msgs[0], "role": msgs[0].get("role", "system"), "content": content}
        self._context.set_messages(msgs)


# --------------------------------------------------------------------------- #
# The voice pipeline for a single call, joined to one LiveKit room.            #
# --------------------------------------------------------------------------- #
async def run_livekit_bot(room_name: str, system_prompt: str, *,
                          firebase_uid=None, user_id="", run_id="",
                          mode="onboarding", existing_memory=None, existing_open_loops=None,
                          profile=None):
    """Join `room_name` as Mira, run the conversation, then consolidate memory."""
    _log(f"bot starting room={room_name} mode={mode} prompt_chars={len(system_prompt)}")
    _onboarding_profile = profile or {}
    url = os.getenv("LIVEKIT_URL")
    key = os.getenv("LIVEKIT_API_KEY")
    secret = os.getenv("LIVEKIT_API_SECRET")

    # Bot joins with an agent-flagged token so the client knows the agent is in.
    bot_token = generate_token_with_agent(room_name, "Mira", key, secret)

    transport = LiveKitTransport(
        url=url,
        token=bot_token,
        room_name=room_name,
        params=LiveKitParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
    )

    # STT engine. STT_ENGINE env:
    #  - "deepgram" (DEFAULT): Deepgram Flux multilingual (flux-general-multi) with
    #    Hindi+English hints, and built-in end-of-turn detection.
    #  - "sarvam": Sarvam Saarika (auto-detect = code-mixed Hinglish; best Hindi).
    # Falls back to Sarvam if DEEPGRAM_API_KEY is missing, so the bot never fails
    # to start. (Sarvam auto-detect keeps English in Latin, Hindi in Devanagari;
    # STT_LANGUAGE=hi-IN forces full Devanagari.)
    _stt_engine = os.getenv("STT_ENGINE", "deepgram").strip().lower()
    if _stt_engine == "deepgram" and os.getenv("DEEPGRAM_API_KEY"):
        dg_model = os.getenv("DEEPGRAM_STT_MODEL", "flux-general-multi")
        _log(f"stt=deepgram-flux model={dg_model} hints=hi,en room={room_name}")
        # Flux has BUILT-IN turn-taking:
        #  - End-of-turn: detects when the user is done (semantic + acoustic, incl.
        #    trailing "hmm"/pauses) -> emits final transcript so Mira replies.
        #  - Barge-in: on the user speaking again it interrupts Mira's speech.
        # Tunables (env, optional): DEEPGRAM_EOT_THRESHOLD (confidence, def 0.7),
        # DEEPGRAM_EOT_TIMEOUT_MS (hard cap, def 5000), DEEPGRAM_EAGER_EOT (enable
        # early-response prediction; lower = snappier but more re-tries).
        flux_kwargs = dict(model=dg_model, language_hints=[Language.HI, Language.EN])
        if os.getenv("DEEPGRAM_EOT_THRESHOLD"):
            flux_kwargs["eot_threshold"] = float(os.getenv("DEEPGRAM_EOT_THRESHOLD"))
        if os.getenv("DEEPGRAM_EOT_TIMEOUT_MS"):
            flux_kwargs["eot_timeout_ms"] = int(os.getenv("DEEPGRAM_EOT_TIMEOUT_MS"))
        if os.getenv("DEEPGRAM_EAGER_EOT"):
            flux_kwargs["eager_eot_threshold"] = float(os.getenv("DEEPGRAM_EAGER_EOT"))
        stt = DeepgramFluxSTTService(
            api_key=os.getenv("DEEPGRAM_API_KEY"),
            settings=DeepgramFluxSTTService.Settings(**flux_kwargs),
        )
    else:
        if _stt_engine == "deepgram":
            _log(f"stt=sarvam (DEEPGRAM_API_KEY missing) room={room_name}")
        stt_kwargs = dict(model=os.getenv("STT_MODEL", "saarika:v2.5"))
        _stt_lang = os.getenv("STT_LANGUAGE", "auto").strip().lower()
        if _stt_lang in ("hi", "hi-in", "hindi"):
            stt_kwargs["language"] = Language.HI_IN
            _log(f"stt=sarvam language=hi-IN room={room_name}")
        else:
            _log(f"stt=sarvam language=auto room={room_name}")
        stt = SarvamSTTService(
            api_key=os.getenv("SARVAM_API_KEY"),
            settings=SarvamSTTService.Settings(**stt_kwargs),
        )

    # LLM engine. LLM_ENGINE env:
    #  - "groq" (DEFAULT): Groq llama-3.3-70b-versatile — fast, free, good Hinglish.
    #    Not a reasoning model (answers directly), which is best for voice latency.
    #  - "deepseek": DeepSeek deepseek-reasoner (thinking model; needs a PAID key).
    #  - "gemini": Gemini (GEMINI_MODEL, default gemini-3.5-flash).
    # Model overridable via GROQ_MODEL / DEEPSEEK_MODEL / GEMINI_MODEL. Groq is the
    # safe default because GROQ_API_KEY is always set (also used for consolidation).
    _llm_engine = os.getenv("LLM_ENGINE", "groq").strip().lower()
    if _llm_engine == "deepseek" and os.getenv("DEEPSEEK_API_KEY"):
        deepseek_model = os.getenv("DEEPSEEK_MODEL", "deepseek-reasoner")
        _log(f"llm=deepseek model={deepseek_model} room={room_name}")
        llm = DeepSeekLLMService(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            settings=DeepSeekLLMService.Settings(model=deepseek_model),
        )
    elif _llm_engine == "gemini":
        gemini_model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
        _log(f"llm=gemini model={gemini_model} room={room_name}")
        llm = GoogleLLMService(
            api_key=os.getenv("GOOGLE_API_KEY"),
            settings=GoogleLLMService.Settings(model=gemini_model),
        )
    else:  # groq (default; also the fallback when deepseek is requested w/o a key)
        if _llm_engine == "deepseek":
            _log(f"llm=groq (DEEPSEEK_API_KEY missing) room={room_name}")
        groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        _log(f"llm=groq model={groq_model} room={room_name}")
        llm = GroqLLMService(
            api_key=os.getenv("GROQ_API_KEY"),
            settings=GroqLLMService.Settings(model=groq_model),
        )

    # TTS engine selection. TTS_ENGINE env:
    #  - "cartesia" (DEFAULT): Cartesia Sonic 3.5 — 42 languages incl. Hindi,
    #    native Hinglish voice, WebSocket streaming. Needs CARTESIA_API_KEY.
    #  - "gemini": Gemini TTS — quota-limited preview model.
    #  - "sarvam": Sarvam bulbul — native Indian accent, free, very reliable.
    #  - "elevenlabs": ElevenLabs (Indian "Simran" voice needs a PAID plan/owned voice).
    #  - "auto": ElevenLabs when it has credits, else Sarvam.
    _tts_engine = os.getenv("TTS_ENGINE", "cartesia").strip().lower()
    if _tts_engine == "auto":
        _tts_engine = "elevenlabs" if await asyncio.to_thread(_elevenlabs_has_credits) else "sarvam"

    if _tts_engine == "cartesia":
        cartesia_voice = os.getenv("CARTESIA_VOICE_ID", "95d51f79-c397-46f9-b49a-23763d3eaa2d")
        cartesia_model = os.getenv("CARTESIA_MODEL", "sonic-3.5")
        _log(f"tts=cartesia model={cartesia_model} voice={cartesia_voice} room={room_name}")
        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id=cartesia_voice,
            model=cartesia_model,
            sample_rate=16000,
            text_filters=[ScriptTextFilter()],
            params=CartesiaTTSService.InputParams(
                language=Language.HI,
            ),
        )
    elif _tts_engine == "gemini":
        gemini_tts_model = os.getenv("GEMINI_TTS_MODEL", "gemini-2.5-flash-preview-tts")
        gemini_style = os.getenv("GEMINI_TTS_STYLE", "")
        tts_filters = [ScriptTextFilter()]
        if gemini_style.strip():
            tts_filters.append(StylePrefixTextFilter(gemini_style))
        gemini_tts_key = os.getenv("GEMINI_TTS_API_KEY") or os.getenv("GOOGLE_API_KEY")
        _log(f"tts=gemini model={gemini_tts_model} voice={os.getenv('GEMINI_TTS_VOICE', 'Kore')} "
             f"style={'on' if gemini_style.strip() else 'off'} "
             f"key={'separate' if os.getenv('GEMINI_TTS_API_KEY') else 'shared'} room={room_name}")
        tts = GeminiTTSService(
            api_key=gemini_tts_key,
            use_genai=True,
            sample_rate=24000,
            text_filters=tts_filters,
            settings=GeminiTTSService.Settings(
                model=gemini_tts_model,
                voice=os.getenv("GEMINI_TTS_VOICE", "Kore"),
            ),
        )
    elif _tts_engine == "elevenlabs":
        _log(f"tts=elevenlabs room={room_name}")
        tts = ElevenLabsTTSService(
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            text_filters=[ScriptTextFilter()],
            settings=ElevenLabsTTSService.Settings(
                voice=os.getenv("ELEVENLABS_VOICE_ID", "TRnaQb7q41oL7sV0w6Bu"),  # "Simran" — Indian-accented female (natural Hinglish)
                model=os.getenv("ELEVENLABS_MODEL", "eleven_flash_v2_5"),
                language=Language.HI,
                # Consistent, calm delivery. Without these, flash_v2_5 runs on its
                # expressive defaults and randomly spikes volume/pitch on some
                # sentences ("suddenly rises its voice"). High stability + no
                # speaker-boost + zero style = steady loudness. Env-tunable.
                stability=float(os.getenv("ELEVENLABS_STABILITY", "0.8")),
                similarity_boost=float(os.getenv("ELEVENLABS_SIMILARITY", "0.75")),
                style=float(os.getenv("ELEVENLABS_STYLE", "0.0")),
                use_speaker_boost=os.getenv("ELEVENLABS_SPEAKER_BOOST", "0") == "1",
            ),
        )
    else:  # sarvam
        _log(f"tts=sarvam-bulbul room={room_name}")
        tts = SarvamTTSService(
            api_key=os.getenv("SARVAM_API_KEY"),
            text_filters=[ScriptTextFilter()],
            settings=SarvamTTSService.Settings(
                voice=os.getenv("SARVAM_VOICE", "anushka"),
                model=os.getenv("SARVAM_TTS_MODEL", "bulbul:v2"),
                language=Language.HI,
            ),
        )

    # Turn-taking + barge-in. CRITICAL: there must be exactly ONE turn authority.
    #  - Deepgram Flux does its OWN end-of-turn + interruption. Adding a second VAD
    #    here makes Mira get interrupted the instant she starts speaking (audio is
    #    cancelled -> "text appears but no sound"). So with Flux, use a plain
    #    aggregator and let Flux drive turns/barge-in.
    #  - Sarvam has NO turn detection, so it needs the aggregator's SileroVAD:
    #    confidence/min_volume tuned so noise/echo isn't treated as speech, and
    #    interruptions require >= 2 transcribed words.
    context = LLMContext([{"role": "system", "content": system_prompt}])
    _flux_active = _stt_engine == "deepgram" and bool(os.getenv("DEEPGRAM_API_KEY"))
    if _flux_active:
        _log(f"turn-taking: Deepgram Flux (single authority) room={room_name}")
        aggregator = LLMContextAggregatorPair(context)
    else:
        _log(f"turn-taking: aggregator SileroVAD room={room_name}")
        aggregator = LLMContextAggregatorPair(
            context,
            user_params=LLMUserAggregatorParams(
                vad_analyzer=SileroVADAnalyzer(params=VADParams(
                    confidence=0.8, start_secs=0.2, stop_secs=0.5, min_volume=0.75,
                )),
                user_turn_strategies=UserTurnStrategies(
                    start=[MinWordsUserTurnStartStrategy(min_words=2)],
                ),
            ),
        )

    # Pipeline mid-section: onboarding uses NodeProcessor (state machine);
    # ongoing uses RAGProcessor (knowledge base retrieval).
    stages = [
        transport.input(),            # mic in (from LiveKit)
        stt,                          # speech -> text
    ]
    if mode == "onboarding":
        # Onboarding: node-based state machine controls the call flow.
        # Each node has its own focused prompt; code drives transitions.
        # No RAG needed during onboarding — Mira is collecting info, not advising.
        node_proc = onboarding_nodes.create_node_processor(context, _onboarding_profile, log_fn=_log)
        _log(f"onboarding node system enabled room={room_name}")
        stages.append(node_proc)
    elif rag.enabled():
        _log(f"rag enabled room={room_name} top_k={os.getenv('RAG_TOP_K', '3')}")
        stages.append(RAGProcessor(
            context,
            top_k=int(os.getenv("RAG_TOP_K", "3")),
            min_similarity=float(os.getenv("RAG_MIN_SIMILARITY", "0.5")),
        ))
    else:
        _log(f"rag disabled room={room_name} (no supabase/google key)")
    stages += [
        aggregator.user(),            # add user turn to context
        llm,                          # reasoning (sees any injected references)
        tts,                          # text -> speech
        transport.output(),           # speaker out (to LiveKit)
        aggregator.assistant(),       # add bot turn to context
    ]
    pipeline = Pipeline(stages)

    # Note: `allow_interruptions` no longer exists in Pipecat 1.7.0 — barge-in is
    # configured on the user aggregator's VAD above, not here.
    task = PipelineTask(
        pipeline,
        params=PipelineParams(enable_metrics=True),
        observers=[_DiagObserver(room_name), _CaptionObserver(transport)],
    )

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant_id):
        # Make Mira speak first: run the LLM once so she greets and begins.
        _log(f"participant joined room={room_name} -> queue greeting")
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_participant_disconnected")
    async def on_participant_disconnected(transport, participant_id):
        # User hung up (red button) → end the call and free resources.
        _log(f"participant left room={room_name} -> cancel")
        await task.cancel()

    # handle_sigint=False: this runs as a background task inside the web
    # server's event loop, so it must NOT try to install process signal handlers.
    _log(f"pipeline built room={room_name} -> running")
    started_at = datetime.now(timezone.utc).isoformat()
    await PipelineRunner(handle_sigint=False).run(task)
    _log(f"pipeline finished room={room_name}")

    # --- End-of-session memory consolidation (off the audio path) ------------
    # The call has ended. Read the transcript from the context, extract durable
    # memory + summary + open loops (one LLM call), and MERGE into the user's
    # cumulative long-term memory. Runs for both modes (onboarding seeds v1).
    if firebase_uid:
        try:
            transcript = _transcript_from_context(context)
            user_turns = sum(1 for m in context.get_messages() if m.get("role") == "user")
            if user_turns >= 1:
                _log(f"consolidating room={room_name} turns={user_turns}")
                result = consolidate.consolidate(existing_memory or {}, existing_open_loops or [], transcript)
                memory_store.save_consolidation(
                    firebase_uid=firebase_uid, user_id=user_id, run_id=run_id,
                    session_type=mode, started_at=started_at,
                    ended_at=datetime.now(timezone.utc).isoformat(),
                    merged_memory=result["long_term_memory"],
                    session_summary=result["session_summary"],
                    open_loops=result["open_loops"],
                )
                _log(f"memory saved room={room_name} open_loops={len(result['open_loops'])}")
            else:
                _log(f"no user turns room={room_name} -> skip consolidation")
        except Exception as exc:
            _log(f"consolidation failed room={room_name}: {exc}")


def _transcript_from_context(context) -> str:
    """Flatten the LLM context's user/assistant turns into a plain transcript."""
    lines = []
    for msg in context.get_messages():
        role = msg.get("role")
        if role not in ("user", "assistant"):
            continue
        content = msg.get("content")
        if isinstance(content, list):
            content = " ".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        text = (content or "").strip()
        if text:
            lines.append(f"{'User' if role == 'user' else 'Mira'}: {text}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Web server: serves the call UI and mints LiveKit join tokens.                #
# --------------------------------------------------------------------------- #
app = FastAPI(title="Kamya Wellness — Mira onboarding call")

_HERE = Path(__file__).parent

# The product entry point is the WEB app (login / signup / onboarding). The call
# is step 2 and is only reached AFTER onboarding, via a handoff that carries the
# user's ?uid=. Opening the bare call URL (no uid) must NOT drop you on the call
# screen — send you to the web app to sign in / onboard first.
WEB_APP_URL = os.getenv("WEB_APP_URL", "https://web-production-45f0d.up.railway.app/")


@app.get("/")
@app.get("/call")
async def call_ui(request: Request):
    """The call screen — only shown when arriving from onboarding (has ?uid=).

    Without a uid, redirect to the web app so the user signs in / onboards first.
    """
    if not request.query_params.get("uid"):
        return RedirectResponse(WEB_APP_URL)
    return HTMLResponse((_HERE / "call_ui.html").read_text(encoding="utf-8"))


@app.get("/mira_avatar.jpg")
async def avatar():
    """Mira's avatar used by the call screen (falls back to an emoji if absent)."""
    path = _HERE / "mira_avatar.jpg"
    if path.exists():
        return FileResponse(path, media_type="image/jpeg")
    return HTMLResponse(status_code=404)


@app.get("/healthz")
async def healthz():
    return {"ok": True}


@app.get("/events")
async def events():
    """Temporary: last pipeline events (to see where audio dies)."""
    return {"events": list(_EVENTS)}




@app.get("/memory", response_class=HTMLResponse)
async def memory_view(request: Request):
    """Readable view of a user's long-term memory, open loops, and session
    history. Prototype view — anyone with the uid can see it, so keep it
    auth-gated before production (it's personal health data)."""
    import html as _html
    uid = request.query_params.get("uid")
    if not uid:
        return HTMLResponse("<p style='font-family:sans-serif;padding:40px'>Add <code>?uid=…</code> to view a user's memory.</p>")

    mem = memory_store.load_memory(uid)
    user_id = mem.get("user_id", "")
    try:
        profile = profile_store.load_profile_for_uid(uid)
    except Exception:
        profile = {}
    name = profile.get("name") or "This user"
    ltm = mem.get("long_term_memory") or {}
    sessions = memory_store.get_sessions(user_id) if user_id else []

    def esc(x):
        return _html.escape(str(x))

    def ul(items):
        items = [i for i in (items or []) if str(i).strip()]
        if not items:
            return "<p class='empty'>— nothing yet —</p>"
        return "<ul>" + "".join(f"<li>{esc(i)}</li>" for i in items) + "</ul>"

    def kv_table(obj, highlight_keys=None):
        """Render a dict as a compact key-value list."""
        if not obj or not isinstance(obj, dict):
            return "<p class='empty'>— nothing yet —</p>"
        highlight_keys = highlight_keys or set()
        rows = []
        for k, v in obj.items():
            if v is None or v == [] or v == "":
                continue
            label = k.replace("_", " ").capitalize()
            cls = " class='warn'" if k in highlight_keys else ""
            if isinstance(v, list):
                val = esc("; ".join(str(i) for i in v))
            else:
                val = esc(str(v))
            rows.append(f"<li{cls}><strong>{esc(label)}:</strong> {val}</li>")
        joined = "".join(rows)
        return f"<ul>{joined}</ul>" if rows else "<p class='empty'>— nothing yet —</p>"

    def med_list(meds):
        if not meds:
            return "<p class='empty'>— none —</p>"
        items = []
        for m in meds:
            if isinstance(m, dict):
                parts = [m.get("name", "?")]
                if m.get("dosage"):
                    parts.append(m["dosage"])
                if m.get("timing"):
                    parts.append(m["timing"])
                if m.get("frequency"):
                    parts.append(f"({m['frequency']})")
                items.append(" ".join(p for p in parts if p))
            else:
                items.append(str(m))
        return "<ul>" + "".join(f"<li>{esc(i)}</li>" for i in items) + "</ul>"

    def pattern_html(pattern):
        if not pattern or not isinstance(pattern, dict):
            return "<p class='empty'>— not tracked yet —</p>"
        slot_labels = {"morning": "Morning", "mid_morning": "Mid-morning",
                       "lunch": "Lunch", "evening": "Evening",
                       "dinner": "Dinner", "late_night": "Late night"}
        rows = []
        for slot_key, slot_label in slot_labels.items():
            slot = pattern.get(slot_key) or {}
            if not isinstance(slot, dict):
                continue
            time_str = esc(slot.get("time") or "—")
            freq = slot.get("frequent") or []
            freq_str = esc(", ".join(str(f) for f in freq)) if freq else "<span class='empty'>?</span>"
            note = esc(slot.get("note") or "")
            gaps = slot.get("gaps") or []
            gaps_str = "; ".join(esc(str(g)) for g in gaps) if gaps else ""
            rows.append(
                f"<tr><td class='slot-label'>{esc(slot_label)}</td>"
                f"<td>{time_str}</td>"
                f"<td>{freq_str}</td>"
                f"<td class='note'>{note}</td>"
                f"<td class='gaps'>{gaps_str}</td></tr>"
            )
        if not rows:
            return "<p class='empty'>— not tracked yet —</p>"
        return (
            "<div class='table-scroll'><table class='pattern'>"
            "<thead><tr><th>Meal</th><th>Time</th><th>Frequent foods</th><th>Note</th><th>Gaps</th></tr></thead>"
            "<tbody>" + "".join(rows) + "</tbody></table></div>"
        )

    def entities_html(entities):
        if not entities:
            return "<p class='empty'>— no active advice tracked —</p>"
        items = []
        for e in entities:
            if isinstance(e, dict):
                status_cls = "pill-green" if e.get("status") in ("following", "liked") else \
                             "pill-red" if e.get("status") in ("quit",) else "pill"
                items.append(
                    f"<div class='entity'>"
                    f"<span class='pill {status_cls}'>{esc(e.get('type', ''))}</span> "
                    f"<strong>{esc(e.get('what', ''))}</strong>"
                    f"<span class='entity-meta'>status: {esc(e.get('status', '?'))} · "
                    f"given: {esc(e.get('given_on', '?'))}</span></div>"
                )
            else:
                items.append(f"<div class='entity'>{esc(str(e))}</div>")
        return "".join(items)

    def exchanges_html(exchanges):
        if not exchanges:
            return "<p class='empty'>— no recent exchanges —</p>"
        items = []
        for ex in exchanges[-6:]:
            if isinstance(ex, dict):
                role = "user" if ex.get("role") == "user" else "mira"
                items.append(f"<div class='exchange {role}'>"
                             f"<span class='role'>{esc(role.capitalize())}</span> "
                             f"{esc(ex.get('text', ''))}</div>")
        return "".join(items) if items else "<p class='empty'>— none —</p>"

    # --- Build LTM HTML sections ---
    identity = ltm.get("identity") or {}
    health = ltm.get("health") or {}
    diet = ltm.get("diet") or {}
    prefs = ltm.get("preferences") or {}
    goals_d = ltm.get("goals") or {}
    lifestyle_d = ltm.get("lifestyle") or {}
    progress_d = ltm.get("progress") or {}
    meta_d = ltm.get("interaction_meta") or {}

    ltm_html = ""

    # Identity
    basics = identity.get("basics") or {}
    body = identity.get("body") or {}
    id_merged = {**basics, **body}
    if any(v for v in id_merged.values()):
        ltm_html += f"<h3>Identity</h3>{kv_table(id_merged)}"

    # Health
    health_items = []
    for label, key in (("Conditions", "conditions"), ("Allergies", "allergies")):
        vals = health.get(key) or []
        if vals:
            health_items.append(f"<li class='warn'><strong>{label}:</strong> {esc('; '.join(str(v) for v in vals))}</li>")
    meds = health.get("medications") or []
    if health_items or meds:
        ltm_html += "<h3>Health</h3>"
        if health_items:
            ltm_html += "<ul>" + "".join(health_items) + "</ul>"
        if meds:
            ltm_html += "<h4>Medications</h4>" + med_list(meds)

    # Diet
    diet_items = {}
    if diet.get("type"):
        diet_items["type"] = diet["type"]
    if diet.get("restrictions"):
        diet_items["restrictions"] = diet["restrictions"]
    if diet_items:
        ltm_html += f"<h3>Diet</h3>{kv_table(diet_items)}"

    # Current pattern
    cp = ltm.get("current_pattern") or {}
    if cp:
        ltm_html += f"<h3>Eating Pattern</h3>{pattern_html(cp)}"

    # Preferences
    pref_items = {}
    if prefs.get("likes"):
        pref_items["likes"] = prefs["likes"]
    if prefs.get("dislikes"):
        pref_items["dislikes"] = prefs["dislikes"]
    if prefs.get("cuisine"):
        pref_items["cuisine"] = prefs["cuisine"]
    if pref_items:
        ltm_html += f"<h3>Preferences</h3>{kv_table(pref_items, highlight_keys={{'dislikes'}})}"

    # Goals
    if any(goals_d.get(k) for k in ("primary_goal", "target", "motivation")):
        ltm_html += f"<h3>Goals</h3>{kv_table(goals_d)}"

    # Lifestyle
    if any(v for v in lifestyle_d.values() if v):
        ltm_html += f"<h3>Lifestyle</h3>{kv_table(lifestyle_d)}"

    # Progress
    prog_items = {}
    if progress_d.get("what_worked"):
        prog_items["what_worked"] = progress_d["what_worked"]
    if progress_d.get("what_failed"):
        prog_items["what_failed"] = progress_d["what_failed"]
    if progress_d.get("struggles"):
        prog_items["struggles"] = progress_d["struggles"]
    if prog_items:
        ltm_html += f"<h3>Progress</h3>{kv_table(prog_items, highlight_keys={{'what_failed', 'struggles'}})}"

    # Entities
    ents = ltm.get("entities") or []
    if ents:
        ltm_html += f"<h3>Active Advice & Plans</h3>{entities_html(ents)}"

    # Recent exchanges
    recent = ltm.get("recent_exchanges") or []
    if recent:
        ltm_html += f"<h3>Last Conversation</h3>{exchanges_html(recent)}"

    # Interaction meta
    if any(v for v in meta_d.values() if v):
        ltm_html += f"<h3>Interaction History</h3>{kv_table(meta_d)}"

    # Misc
    misc_items = ltm.get("misc") or []
    if misc_items:
        ltm_html += f"<h3>Other Notes</h3>{ul(misc_items)}"

    if not ltm_html:
        # Fallback for old flat format
        ltm_html = "".join(
            f"<h3>{lbl}</h3>{ul(ltm.get(k))}"
            for lbl, k in (("Facts", "facts"), ("Goals", "goals"),
                           ("Preferences", "preferences"), ("Dislikes", "dislikes"))
        )

    sess_html = ""
    for s in sessions:
        sess_html += (
            "<div class='card'><div class='meta'>"
            f"<span class='pill'>{esc(s.get('type', '') or 'session')}</span>"
            f"<span class='when'>{esc(s.get('ended_at', '') or s.get('started_at', ''))}</span>"
            f"<span class='rid'>{esc(s.get('session_id', ''))}</span></div>"
            f"<p class='sum'>{esc(s.get('session_summary') or '—')}</p>"
            + ul(memory_store.parse_loops(s.get('open_loops'))) + "</div>"
        )
    if not sess_html:
        sess_html = "<p class='empty'>No sessions recorded yet.</p>"

    try:
        prompt_text = esc((_HERE / "ask_prompt.md").read_text(encoding="utf-8"))
    except Exception:
        prompt_text = "(prompt file not found)"

    page = f"""<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mira's memory — {esc(name)}</title>
<style>
  :root{{ --accent:#2f6fed; --accent-bg:rgba(47,111,237,.1); --accent-fg:#1e4fb0;
    --ink:#101a2e; --muted:#5a6b86; --line:#e2e9f6; --bg:#eef4fc; --card:#fff; }}
  *{{box-sizing:border-box;margin:0;padding:0;}}
  body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
    background:var(--bg);color:var(--ink);line-height:1.5;padding:28px 18px 60px;}}
  .wrap{{max-width:680px;margin:0 auto;}}
  .eyebrow{{font-size:11.5px;font-weight:800;letter-spacing:.14em;text-transform:uppercase;
    color:var(--accent-fg);background:var(--accent-bg);padding:5px 12px;border-radius:999px;display:inline-block;}}
  h1{{font-size:30px;margin:12px 0 4px;letter-spacing:-.02em;}}
  .sub{{color:var(--muted);font-size:14px;margin-bottom:24px;}}
  section{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px 20px;margin-bottom:16px;}}
  h2{{font-size:16px;color:var(--accent-fg);margin-bottom:10px;letter-spacing:.01em;}}
  h3{{font-size:12.5px;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;margin:12px 0 4px;}}
  h3:first-of-type{{margin-top:0;}}
  ul{{padding-left:18px;}} li{{margin:3px 0;}}
  .empty{{color:#9aa7bd;font-style:italic;font-size:14px;}}
  .card{{border:1px solid var(--line);border-radius:12px;padding:13px 15px;margin-top:10px;background:#fbfdff;}}
  .meta{{display:flex;gap:10px;align-items:center;flex-wrap:wrap;font-size:12px;color:var(--muted);margin-bottom:6px;}}
  .pill{{background:var(--accent-bg);color:var(--accent-fg);font-weight:700;padding:2px 9px;border-radius:999px;text-transform:capitalize;}}
  .rid{{font-family:ui-monospace,Menlo,monospace;font-size:11px;color:#9aa7bd;margin-left:auto;}}
  .sum{{font-size:14.5px;margin-bottom:6px;}}
  .card ul{{font-size:13.5px;color:var(--muted);}}
  details.promptbox>summary{{cursor:pointer;list-style:none;font-size:16px;font-weight:700;color:var(--accent-fg);}}
  details.promptbox>summary::-webkit-details-marker{{display:none;}}
  details.promptbox>summary::after{{content:" ▸ tap to expand";font-size:12px;font-weight:600;color:var(--muted);}}
  details.promptbox[open]>summary::after{{content:" ▾ tap to collapse";font-size:12px;font-weight:600;color:var(--muted);}}
  details.promptbox pre{{margin:12px 0 0;white-space:pre-wrap;word-wrap:break-word;font-family:ui-monospace,Menlo,monospace;
    font-size:12.5px;line-height:1.55;color:var(--ink);max-height:55vh;overflow:auto;background:#f7faff;border:1px solid var(--line);border-radius:10px;padding:14px;}}
  h4{{font-size:13px;color:var(--muted);margin:8px 0 4px;}}
  .warn{{color:#b44;font-weight:600;}}
  .table-scroll{{overflow-x:auto;margin:8px 0;}}
  table.pattern{{width:100%;border-collapse:collapse;font-size:13px;}}
  table.pattern th{{text-align:left;font-size:11px;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.04em;padding:6px 10px;border-bottom:2px solid var(--line);}}
  table.pattern td{{padding:6px 10px;border-bottom:1px solid var(--line);vertical-align:top;}}
  table.pattern .slot-label{{font-weight:600;white-space:nowrap;}}
  table.pattern .note{{color:var(--muted);font-style:italic;font-size:12px;}}
  table.pattern .gaps{{color:#b44;font-size:12px;}}
  .entity{{padding:8px 0;border-bottom:1px solid var(--line);}}
  .entity:last-child{{border-bottom:none;}}
  .entity-meta{{display:block;font-size:11.5px;color:var(--muted);margin-top:2px;}}
  .pill-green{{background:rgba(34,139,34,.12);color:#1a6b1a;}}
  .pill-red{{background:rgba(180,40,40,.1);color:#a22;}}
  .exchange{{padding:5px 0;font-size:13.5px;}}
  .exchange .role{{font-weight:700;font-size:11.5px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);margin-right:6px;}}
  .exchange.user .role{{color:var(--accent-fg);}}
  .exchange.mira .role{{color:#1a6b1a;}}
</style></head><body>
<div class="wrap">
  <span class="eyebrow">Mira's memory</span>
  <h1>{esc(name)}</h1>
  <p class="sub">{esc(user_id or 'unknown user')} · check-in #{esc(mem.get('session_count') or 0)} · last call: {esc(mem.get('last_session_at') or '—')}</p>

  <section>
    <h2>🧠 Long-term memory (latest)</h2>
    {ltm_html}
  </section>

  <section>
    <h2>🔁 Open loops — to follow up next time</h2>
    {ul(mem.get('open_loops'))}
  </section>

  <section>
    <h2>📝 Last session summary</h2>
    <p>{esc(mem.get('last_session_summary') or '—')}</p>
  </section>

  <section>
    <h2>📚 Session history (newest first)</h2>
    {sess_html}
  </section>

  <section>
    <details class="promptbox">
      <summary>⚙️ System prompt — how Mira is told to behave (Step 3)</summary>
      <pre>{prompt_text}</pre>
    </details>
  </section>
</div></body></html>"""
    return HTMLResponse(page)


@app.post("/connect")
async def connect(request: Request):
    """Create a room, launch Mira into it, and return the browser's join token.

    The body may include `{"uid": "<firebase_uid>"}` handed off from manual
    onboarding — used to load that user's profile so Mira greets them by name.
    """
    url = os.getenv("LIVEKIT_URL")
    key = os.getenv("LIVEKIT_API_KEY")
    secret = os.getenv("LIVEKIT_API_SECRET")

    try:
        body = await request.json()
    except Exception:
        body = {}
    firebase_uid = (body or {}).get("uid") or None
    mode = str((body or {}).get("mode") or "onboarding").strip().lower()
    if mode not in ("onboarding", "ongoing"):
        mode = "onboarding"

    # Profile (manual-onboarding data) + cumulative long-term memory for this user.
    profile = load_profile_for_call(firebase_uid)
    memory = memory_store.load_memory(firebase_uid) if firebase_uid else {}
    user_id = memory.get("user_id", "") if memory else ""

    system_prompt = build_system_prompt(mode, profile, memory)
    run_id = f"run-{uuid.uuid4().hex[:8]}"
    _log(
        f"connect mode={mode} uid_present={bool(firebase_uid)} "
        f"name_present={bool(profile.get('name'))} "
        f"has_memory={bool((memory or {}).get('long_term_memory'))} run_id={run_id}"
    )

    room_name = f"mira-{uuid.uuid4().hex[:10]}"
    user_token = generate_token(room_name, "user", key, secret)

    # Launch Mira into the room; she waits for the user, then greets.
    asyncio.create_task(_run_bot_safe(
        room_name, system_prompt,
        firebase_uid=firebase_uid, user_id=user_id, run_id=run_id,
        mode=mode, existing_memory=(memory or {}).get("long_term_memory", {}),
        existing_open_loops=(memory or {}).get("open_loops", []),
        profile=profile,
    ))

    return {"url": url, "token": user_token, "room": room_name,
            "name": profile.get("name", ""), "mode": mode}


async def _run_bot_safe(room_name: str, system_prompt: str, **kwargs):
    """Run the bot as a background task, logging any crash (which is otherwise
    swallowed by the event loop) so failures are visible in the server logs."""
    try:
        await run_livekit_bot(room_name, system_prompt, **kwargs)
    except Exception as exc:
        _log(f"BOT ERROR room={room_name}: {type(exc).__name__}: {exc}")
        print("[mira] traceback:\n" + traceback.format_exc(), flush=True)






def main():
    parser = argparse.ArgumentParser(description="Kamya Wellness onboarding call")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "7860")))
    args = parser.parse_args()
    print(f"🚀 Mira onboarding call ready → http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
