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
import copy
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
        TTSSpeakFrame,
        TTSStartedFrame,
        TTSAudioRawFrame,
        UserStartedSpeakingFrame,
        UserStoppedSpeakingFrame,
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
    # Optional. The `aws` extra was added to requirements.txt after this
    # service was deployed, so an image built before that rebuild will not
    # have it. A hard import would then take the WHOLE bot down -- including
    # Groq, which needs nothing from AWS. Degrade to Groq instead; the
    # provider check below sees None and falls back.
    try:
        from pipecat.services.aws.llm import AWSBedrockLLMService
    except ImportError:
        AWSBedrockLLMService = None
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
import llm_client
import memory_facts
import memory_store
import onboarding_nodes
import rag
import rag_query
import stt_vocab
import thread_machine

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
        # The observer sees every frame once PER PIPELINE HOP. Without this the
        # log showed one barge-in as 12 identical "USER started speaking" lines
        # and one utterance as 8-12 "BOT speaking START" lines, which reads like
        # a turn-detection problem that isn't there. _CaptionObserver below has
        # always deduped; this one declared the set and never used it.
        fid = getattr(frame, "id", None)
        if fid is not None:
            if fid in self._seen:
                return
            self._seen.add(fid)
            if len(self._seen) > 4000:      # bound memory on long calls
                self._seen.clear()
                self._seen.add(fid)
        if isinstance(frame, ErrorFrame):
            _log(f"ERROR FRAME: {getattr(frame, 'error', frame)}")
            return
        # Count audio frames quietly; log only the first + a running summary.
        if isinstance(frame, TTSAudioRawFrame):
            self._audio = getattr(self, "_audio", 0) + 1
            if self._audio == 1:
                _log(f"tts audio flowing (sr={getattr(frame,'sample_rate','?')})")
            return
        # THE GAP NOBODY WAS MEASURING. Every latency number so far started at
        # "caption sent role=user" — i.e. AFTER Deepgram decided the user had
        # finished. Flux's eot_timeout_ms defaults to 5000ms, so an utterance it
        # is not confident about sits silent for up to five seconds before a
        # transcript exists at all. That is invisible in the pipeline log and
        # is felt by the user as Mira being slow.
        if isinstance(frame, UserStoppedSpeakingFrame):
            self._spoke_end = time.perf_counter()
            return
        if isinstance(frame, TranscriptionFrame) and getattr(self, "_spoke_end", None):
            gap = (time.perf_counter() - self._spoke_end) * 1000
            # Report the value we actually SENT to Flux. This used to print
            # the raw env var, so an unset var logged "default 5000" while the
            # code was in fact sending 2000 -- the log contradicted the
            # configuration and sent us hunting a timeout that was not there.
            _log(f"STT end-of-turn gap {gap:.0f}ms "
                 f"(eot_timeout_ms={os.getenv('DEEPGRAM_EOT_TIMEOUT_MS', '2000')})")
            self._spoke_end = None
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


# Deepgram Flux sends keyterms in the WEBSOCKET QUERY STRING. The full
# stt_vocab list (140 terms, many multi-word) made a URL long enough that
# Deepgram rejected the connection outright:
#   ERROR FRAME: server rejected WebSocket connection: HTTP 400
# STT then never connected, so nothing the user said was transcribed while
# Mira still spoke her greeting normally. Default is now OFF: boosting a few
# words is not worth risking the microphone.
KEYTERM_LIMIT = int(os.getenv("DEEPGRAM_KEYTERM_LIMIT", "20"))


def _stt_keyterms():
    """Keyterms for Flux. OFF unless DEEPGRAM_KEYTERMS is set explicitly.

    Set DEEPGRAM_KEYTERMS to a short comma-separated list and raise it
    gradually — verify STT still connects after each increase. Curated
    candidates live in stt_vocab.py; start with stt_vocab.CONDITIONS, which
    is where a mis-hear costs the most.
    """
    raw = (os.getenv("DEEPGRAM_KEYTERMS") or "").strip()
    if not raw:
        return []
    terms = [t.strip() for t in raw.split(",") if t.strip()]
    if len(terms) > KEYTERM_LIMIT:
        _log(f"keyterms truncated {len(terms)} -> {KEYTERM_LIMIT} "
             f"(query-string length limit)")
    return terms[:KEYTERM_LIMIT]


def _transcript_confidence(frame):
    """Best-effort confidence for a TranscriptionFrame, or None.

    TranscriptionFrame has no confidence field in pipecat 1.7.0 — only
    `result`, the raw STT payload, whose shape is provider-specific and not
    guaranteed. So this probes the common shapes and returns None rather than
    guessing. None means "unknown", which is treated as acceptable: refusing
    real speech is worse than passing a bad transcript to the next gate.
    """
    res = getattr(frame, "result", None)
    if res is None:
        return None
    for probe in (
        lambda r: r.get("confidence"),
        lambda r: r["channel"]["alternatives"][0]["confidence"],
        lambda r: r["alternatives"][0]["confidence"],
        lambda r: getattr(r, "confidence"),
    ):
        try:
            val = probe(res)
            if val is not None:
                return float(val)
        except Exception:
            continue
    return None


class TranscriptGate(FrameProcessor):
    """Drop unusable transcriptions before they reach RAG or the LLM.

    The log showed Mira answering this:

      caption sent role=user len=38
      rag matched 3 q='तूं तक चला से देना और मैं सोचूंगा ना चलो'

    Nobody said that. STT produced word salad, and the whole pipeline —
    retrieval, LLM, TTS — ran on it, so Mira replied to something the user
    never said. That is worse than silence: it makes her look like she is not
    listening.

    Two gates, cheapest first:
      1. confidence, when the STT service exposes it (Flux `min_confidence`
         handles most of this server-side; this catches the rest)
      2. a repetition heuristic — the same token three or more times in a row
         is a recogniser artefact, not speech

    On a drop the transcription is swallowed (no LLM turn) and Mira asks the
    user to repeat, so they get feedback instead of dead air. Consecutive drops
    are capped: after MAX_STRIKES the transcript is let through rather than
    looping "say that again" forever.
    """

    MAX_STRIKES = 2

    def __init__(self, min_confidence=None):
        super().__init__()
        env = (os.getenv("STT_MIN_CONFIDENCE") or "").strip()
        self._min_conf = min_confidence if min_confidence is not None else (
            float(env) if env else 0.0)
        self._strikes = 0
        self._reprompts = [
            "Sorry, wo theek se sunai nahi diya. Phir se boliye?",
            "Aawaz kat rahi hai — ek baar phir bataiye?",
        ]
        self._i = 0

    def _reason(self, frame, text):
        conf = _transcript_confidence(frame)
        if self._min_conf > 0 and conf is not None and conf < self._min_conf:
            return f"low confidence {conf:.2f} < {self._min_conf:.2f}"
        toks = [t for t in re.split(r"[^\w\u0900-\u097f]+", text.lower()) if t]
        run = 1
        for a, b in zip(toks, toks[1:]):
            run = run + 1 if a == b else 1
            if run >= 3:
                return f"repeated token {a!r} x{run}"
        return None

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if (
            direction == FrameDirection.DOWNSTREAM
            and isinstance(frame, TranscriptionFrame)
            and (getattr(frame, "text", "") or "").strip()
        ):
            text = frame.text.strip()
            reason = self._reason(frame, text)
            if reason:
                if self._strikes < self.MAX_STRIKES:
                    self._strikes += 1
                    _log(f"transcript DROPPED ({reason}) strike={self._strikes} "
                         f"text={text[:50]!r}")
                    prompt = self._reprompts[self._i % len(self._reprompts)]
                    self._i += 1
                    await self.push_frame(TTSSpeakFrame(prompt), direction)
                    return                  # never reaches RAG or the LLM
                # Cap reached — let it through and STAY capped. Resetting here
                # would re-drop the next bad transcript, giving an endless
                # drop/drop/pass cycle. Only a clean transcript clears it, so a
                # persistently bad mic degrades to "pass everything" rather
                # than "ask them to repeat forever".
                _log(f"transcript kept despite {reason} (strike cap reached)")
            else:
                self._strikes = 0
        await self.push_frame(frame, direction)


class ThreadMachineProcessor(FrameProcessor):
    """P-3 thread machine. Replaces RAGProcessor for ongoing calls.

    WHERE LANGGRAPH STARTS AND ENDS
        Starts on a final TranscriptionFrame, ends by producing a DIRECTIVE.
        It does NOT generate Mira's reply — the existing Pipecat LLM service
        still does that, so streaming into TTS and context aggregation are
        untouched, and there is still exactly ONE generation call per turn.

    HOW THE DIRECTIVE REACHES THE MODEL
        Same mechanism RAGProcessor uses: a separate system message appended
        at the end of context, immediately before the user's utterance (the
        aggregator adds that after us). messages[0] — the personality, the
        Hinglish rules, the safety framing — is never touched. Exactly one
        block ever exists; the previous one is stripped first.

    INTERRUPTION
        Nothing is committed during generation. Stage advances and memory ops
        are staged in `self._pending` and applied only when
        BotStoppedSpeakingFrame arrives, i.e. once the user has actually heard
        the reply. Barge-in therefore rolls the turn back rather than
        recording advice nobody heard.
    """

    MARKER = "[[MIRA-TURN-POLICY]]"

    def __init__(self, context, *, firebase_uid="", ledger_view=None,
                 fact_ages=None):
        super().__init__()
        self._context = context
        self._uid = firebase_uid
        self._ledger_view = ledger_view or {}
        self._fact_ages = fact_ages or {}
        self._threads = []
        self._turn = 0
        self._pending = None
        self._graph = None
        # Last turn's policy, read by ResponseValidator after generation.
        self.last_policy = {}

    # -- context plumbing --------------------------------------------------
    def _strip_policy(self, msgs):
        return [m for m in msgs
                if not (isinstance(m, dict)
                        and isinstance(m.get("content"), str)
                        and m["content"].startswith(self.MARKER))]

    def _history(self, msgs):
        return [{"role": m.get("role"), "content": m.get("content")}
                for m in msgs
                if m.get("role") in ("user", "assistant")
                and isinstance(m.get("content"), str)][-6:]

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        # Commit point. Fires once the reply has actually been spoken.
        if isinstance(frame, BotStoppedSpeakingFrame):
            self._commit()

        if (
            direction == FrameDirection.DOWNSTREAM
            and isinstance(frame, TranscriptionFrame)
            and (getattr(frame, "text", "") or "").strip()
        ):
            try:
                await self._run(frame.text.strip())
            except Exception as exc:
                # Never break a call over the graph. Falling through leaves the
                # base prompt alone, which is exactly the old P-3 behaviour.
                _log(f"thread machine failed ({type(exc).__name__}): {exc}")
        await self.push_frame(frame, direction)

    # -- the turn ----------------------------------------------------------
    async def _run(self, text):
        if self._graph is None:
            import p3_graph
            self._graph = p3_graph.get_graph()

        msgs = self._strip_policy(self._context.get_messages())
        self._turn += 1

        # DEEP copy, not list(). The graph mutates Thread objects in place —
        # stage, slots, stage_turns — so a shallow copy would let an
        # interrupted turn change committed state even though _pending was
        # never applied. Found by test: a barge-in advanced the thread to
        # REFLECT and filled two slots from a reply the user never heard.
        out = await self._graph.ainvoke({
            "turn_text": text,
            "turn_index": self._turn,
            "threads": copy.deepcopy(self._threads),
            "ledger_view": self._ledger_view,
            "fact_ages": self._fact_ages,
            "history": self._history(msgs),
            "trace": [],
        })

        for line in out.get("trace", []):
            _log(f"  graph: {line}")

        # Safety short-circuit: speak the fixed line, run no LLM at all.
        if out.get("safety_hit"):
            self._context.set_messages(msgs)
            await self.push_frame(TTSSpeakFrame(out["safety_hit"]),
                                  FrameDirection.DOWNSTREAM)
            self._pending = None
            return

        block = self.MARKER + "\n" + (out.get("directive") or "")
        docs = out.get("retrieved") or []
        if docs:
            block += "\n\n" + rag.format_reference(docs)
        msgs.append({"role": "system", "content": block})
        self._context.set_messages(msgs)

        self.last_policy = {
            "stage": out.get("stage"), "lane": out.get("lane"),
            "budget": out.get("budget"), "may_advise": out.get("may_advise"),
        }
        # Staged, not applied. See INTERRUPTION above.
        self._pending = {
            "threads": out.get("threads", self._threads),
            "stage": out.get("stage"),
            "lane": out.get("lane"),
        }

    def _commit(self):
        if not self._pending:
            return
        self._threads = self._pending["threads"]
        active = next((t for t in self._threads if not t.parked), None)
        if active and self._pending.get("stage") == thread_machine.S_ADVISE:
            active.advice.append(active.topic)
        _log(f"  graph: COMMIT lane={self._pending.get('lane')} "
             f"stage={self._pending.get('stage')} threads={len(self._threads)}")
        self._pending = None

    # -- end of call -------------------------------------------------------
    def open_loops(self):
        """Unfinished threads become open loops for the next call."""
        return [thread_machine.open_loop_text(t) for t in self._threads
                if t.stage != thread_machine.S_CLOSE]


class ReplyShapeFilter(FrameProcessor):
    """Physically bound what Mira says: ONE question, and one or two lines.

    Two separate failures, one mechanism, because both were asked for in prose
    and ignored.

    QUESTION STACKING. A live onboarding call produced this as ONE utterance:

      "...आप weight loss से क्या achieve करना चाहते हैं?थोड़ा और बताइए?अभी sign
       up करने का मन कैसे किया?...पहले कभी कुछ try किया था?...कहाँ अटक रहा है?"

    and eleven questions in a row at DAILY_EATING — verbatim the example
    questions from the node prompts, recited as a script. The DAILY_EATING
    prompt already said "Ask ONE meal at a time" AND "Never read these as a
    list", and GLOBAL_RULES already said "ONE question at a time". Three
    instructions, all ignored.

    LENGTH. Also prose only: GLOBAL_RULES said "1-2 short sentences, ~25 words
    max", the P-3 stage budgets said "keep it under N words", and
    ResponseValidator logged "too long" without ever withholding anything. The
    same call still produced questions running several lines.

    So both are enforced here instead. Everything after the first question
    mark is dropped, and length is cut at a SENTENCE BOUNDARY past the cap —
    never mid-word, because a reply chopped inside a word sounds broken, which
    is worse than a slightly long one. A hard stop at twice the cap catches a
    model that never punctuates at all.

    Streaming-safe: frames pass straight through until a cut, so nothing is
    buffered and no latency is added.
    """

    _SENT_END = ("।", ".", "?", "!")

    def __init__(self, label="", word_cap=40, restore_question_mark=False):
        super().__init__()
        self._label = label
        self._cap = word_cap
        # Only true when the LLM runs with stop=["?"], which eats the mark.
        # Appending one unconditionally would put a "?" on the CLOSE node's
        # farewell, which is a statement and must stay one.
        self._restore_q = restore_question_mark
        self._cut = False
        self._dropped = 0
        self._words = 0
        self._reason = ""
        self._sent = False              # did we emit any text this response?
        self._sent_end_punct = False    # did it already end in punctuation?

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, LLMFullResponseStartFrame):
                self._cut, self._dropped, self._words = False, 0, 0
                self._reason = ""
                self._sent, self._sent_end_punct = False, False
            elif isinstance(frame, LLMFullResponseEndFrame):
                # The model is run with stop=["?"], which ends generation at the
                # first question mark AND strips it. Put it back so TTS speaks
                # the sentence with question intonation rather than flatly.
                # Pushed BEFORE the End frame so it joins the pending sentence.
                if self._restore_q and self._sent and not self._sent_end_punct:
                    await self.push_frame(LLMTextFrame("?"), direction)
                # Log whenever the filter BOUND the reply, even if zero chars
                # were dropped — a cut that lands exactly on the last sentence
                # still means the cap was doing the work, and that rate is
                # what tells you whether the prompts are pulling their weight.
                if self._cut:
                    _log(f"  shape filter{self._label}: cut on {self._reason}, "
                         f"dropped {self._dropped} chars "
                         f"({self._words} words kept, cap {self._cap})")
                self._cut, self._dropped, self._words = False, 0, 0
                self._reason = ""
            elif isinstance(frame, LLMTextFrame):
                text = getattr(frame, "text", "") or ""
                if self._cut:
                    self._dropped += len(text)
                    return                      # suppress the rest of the reply

                # 1. one question only
                if "?" in text:
                    keep = text[: text.index("?") + 1]
                    self._dropped += len(text) - len(keep)
                    self._cut, self._reason = True, "second question"
                    self._words += len(keep.split())
                    if keep:
                        self._sent, self._sent_end_punct = True, True
                        await self.push_frame(LLMTextFrame(keep), direction)
                    return

                # 2. length — finish the sentence in progress, then stop
                self._words += len(text.split())
                if text.strip():
                    self._sent = True
                    self._sent_end_punct = text.rstrip()[-1] in self._SENT_END
                if self._words > self._cap:
                    idx = max(text.rfind(c) for c in self._SENT_END)
                    if idx >= 0:
                        keep = text[: idx + 1]
                        self._dropped += len(text) - len(keep)
                        self._cut, self._reason = True, f"length>{self._cap}w"
                        if keep:
                            await self.push_frame(LLMTextFrame(keep), direction)
                        return
                    if self._words > self._cap * 2:
                        # never punctuated — stop rather than run on forever
                        self._dropped += len(text)
                        self._cut, self._reason = True, "unpunctuated runaway"
                        return
        await self.push_frame(frame, direction)


class ResponseValidator(FrameProcessor):
    """Check what Mira actually said against the policy for that turn.

    Sits between the LLM and TTS, accumulating streamed tokens and checking at
    LLMFullResponseEndFrame. It is a PASS-THROUGH: frames are never held, so
    it costs no latency and cannot introduce dead air.

    It LOGS, it does not block. On a phone call a silence is worse than a
    slightly long sentence, and regenerating would double the turn. The point
    is that stage-rule violations become visible and countable instead of
    being invisible — "did she ask two questions?", "did she advise before we
    let her?" were previously unanswerable.
    """

    # Spoken when generation produces nothing at all. Deliberately vague about
    # the cause — the user does not need to hear about token limits — and
    # deliberately hands the turn back rather than dead-ending.
    FALLBACK = "माफ़ कीजिए, एक second — फिर से बताइएगा?"
    FALLBACK_REPEAT = "लगता है connection में कुछ दिक्कत है. थोड़ी देर में बात करते हैं?"

    def __init__(self, thread_proc):
        super().__init__()
        self._thread = thread_proc
        self._buf = []
        self._empties = 0

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, LLMFullResponseStartFrame):
                self._buf = []
            elif isinstance(frame, LLMTextFrame):
                self._buf.append(getattr(frame, "text", "") or "")
            elif isinstance(frame, LLMFullResponseEndFrame):
                text = "".join(self._buf)
                self._buf = []
                problems = self._check(text)
                # DEAD AIR IS THE WORST VOICE FAILURE. When the model returns
                # nothing — Groq 429, a timeout, a safety refusal — the user
                # hears silence and assumes the call dropped. Observed live on
                # a daily-token-limit 429: Mira simply stopped answering and
                # the user had to say "मेरा बोलिए". Speak instead.
                if "empty" in problems:
                    self._empties += 1
                    line = (self.FALLBACK if self._empties == 1
                            else self.FALLBACK_REPEAT)
                    _log(f"  empty response #{self._empties} -> speaking fallback")
                    await self.push_frame(TTSSpeakFrame(line), direction)
                else:
                    self._empties = 0
        await self.push_frame(frame, direction)

    def _check(self, text):
        policy = dict(getattr(self._thread, "last_policy", {}) or {})
        try:
            problems = thread_machine.validate(text, policy)
        except Exception as exc:
            _log(f"validate failed: {exc}")
            return []
        words = len((text or "").split())
        if problems:
            _log(f"  policy VIOLATION stage={policy.get('stage')} "
                 f"lane={policy.get('lane')}: {'; '.join(problems)}")
        else:
            _log(f"  policy ok stage={policy.get('stage')} words={words} "
                 f"budget={policy.get('budget')}")
        return problems


class RAGProcessor(FrameProcessor):
    """RAG injection. Sits between STT and the user aggregator.

    Three things it is careful about, each fixing an observed problem:

    1. IT DOES NOT RETRIEVE ON EVERYTHING. Logs used to show
       `rag matched 3 q='Yeah.'` — three irrelevant dietician Q&As injected
       because the user grunted. rag_query.is_retrievable() gates that out
       with no model call.

    2. IT RESOLVES CONTEXT-DEPENDENT QUERIES. "और कुछ?" means nothing on its
       own; the topic lives in Mira's previous turn. rag_query.build_query()
       folds the last turns in when (and only when) the utterance needs it.

    3. IT NEVER TOUCHES messages[0]. The base system prompt — 4.7k chars on
       ongoing calls, 9k on onboarding — stays byte-identical for the whole
       call, so the prompt prefix is stable and re-processing it every turn is
       avoidable. The reference block is a SEPARATE message appended at the end
       of the context, which places it immediately before the user's new
       utterance (the aggregator adds that after us). Exactly one reference
       block ever exists: the previous one is removed before the new one lands.

    Any retrieval error leaves the context untouched and the call proceeds.

    NOTE: the block is injected as a second `system` message. Groq (the default
    LLM_ENGINE) handles multiple system messages fine. If you switch
    LLM_ENGINE=gemini, which honours only one system instruction, revisit this.
    """

    def __init__(self, context, *, top_k: int = 3, min_similarity: float = 0.5):
        super().__init__()
        self._context = context
        self._top_k = top_k
        self._min_sim = min_similarity

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
                _log(f"rag refresh failed: {exc}")
        await self.push_frame(frame, direction)

    def _strip_reference(self, msgs):
        """Drop any previous reference block so it can never accumulate."""
        return [m for m in msgs if not rag.is_reference_message(m)]

    async def _refresh(self, question: str):
        msgs = self._context.get_messages()
        if not msgs:
            return
        cleaned = self._strip_reference(msgs)
        had_reference = len(cleaned) != len(msgs)

        ok, why = rag_query.is_retrievable(question)
        if not ok:
            # Clear last turn's stale reference, then stop. No embedding call,
            # no Supabase round-trip, no noise in the prompt.
            if had_reference:
                self._context.set_messages(cleaned)
            _log(f"rag skipped ({why}) q={question[:40]!r}")
            return

        query, strategy = rag_query.build_query(question, cleaned)
        matches = await rag.retrieve(query, k=self._top_k, min_similarity=self._min_sim)

        if matches:
            cleaned.append({"role": "system", "content": rag.format_reference(matches)})
            _log(f"rag matched {len(matches)} strategy={strategy} q={query[:70]!r}")
        else:
            _log(f"rag no match strategy={strategy} q={query[:70]!r}")
        self._context.set_messages(cleaned)


# --------------------------------------------------------------------------- #
# The voice pipeline for a single call, joined to one LiveKit room.            #
# --------------------------------------------------------------------------- #
async def run_livekit_bot(room_name: str, system_prompt: str, *,
                          firebase_uid=None, user_id="", run_id="",
                          mode="onboarding", existing_memory=None, existing_open_loops=None,
                          profile=None, fact_ages=None):
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
        _log(f"stt=deepgram-flux model={dg_model} hints=hi,en "
             f"keyterms={len(_stt_keyterms()) or 'off'} room={room_name}")
        # Flux has BUILT-IN turn-taking:
        #  - End-of-turn: detects when the user is done (semantic + acoustic, incl.
        #    trailing "hmm"/pauses) -> emits final transcript so Mira replies.
        #  - Barge-in: on the user speaking again it interrupts Mira's speech.
        # Tunables (env, optional): DEEPGRAM_EOT_THRESHOLD (confidence, def 0.7),
        # DEEPGRAM_EOT_TIMEOUT_MS (hard cap, def 5000), DEEPGRAM_EAGER_EOT (enable
        # early-response prediction; lower = snappier but more re-tries).
        #
        # ON FALSE BARGE-INS: the 2026-08-17 log appeared to show a dozen
        # "USER started speaking" events per turn. That was _DiagObserver
        # logging one frame once per pipeline hop, not a dozen interruptions —
        # fixed there, not here. Deepgram defaults are therefore left ALONE
        # until a deduped log shows a real problem. If it does, raise
        # DEEPGRAM_EOT_THRESHOLD (0.8-0.9) so Flux needs more certainty before
        # ending a turn, and/or raise DEEPGRAM_EOT_TIMEOUT_MS to let the user
        # pause longer mid-thought. Both trade latency for patience — measure
        # before changing.
        flux_kwargs = dict(model=dg_model, language_hints=[Language.HI, Language.EN])
        # Keyterm boosting: Flux weights these toward being recognised. Domain
        # vocabulary is exactly what a general model gets wrong, and a wrong
        # condition name is worse than a wrong food name. Override wholesale
        # with DEEPGRAM_KEYTERMS (comma-separated).
        _kt = _stt_keyterms()
        if _kt:
            flux_kwargs["keyterm"] = _kt
        # min_confidence makes Deepgram itself withhold transcripts it does not
        # believe, so garbage never enters the pipeline at all. Kept low by
        # default because dropping real speech is worse than a bad transcript;
        # TranscriptGate below is the second line of defence.
        if os.getenv("DEEPGRAM_MIN_CONFIDENCE"):
            flux_kwargs["min_confidence"] = float(os.getenv("DEEPGRAM_MIN_CONFIDENCE"))
        if os.getenv("DEEPGRAM_EOT_THRESHOLD"):
            flux_kwargs["eot_threshold"] = float(os.getenv("DEEPGRAM_EOT_THRESHOLD"))
        # Deepgram's own default is 5000ms — five seconds of silence after the
        # user stops, whenever Flux is not confident the turn ended. On a call
        # that reads as Mira being broken. 2000ms still leaves room for the
        # mid-sentence pauses Hindi speakers actually make, and only affects
        # the UNCERTAIN case: a confident end-of-turn still fires immediately
        # via eot_threshold, which is deliberately left at its default so we
        # do not start cutting people off.
        flux_kwargs["eot_timeout_ms"] = int(os.getenv("DEEPGRAM_EOT_TIMEOUT_MS", "2000"))
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
    #  - "groq" (DEFAULT): Groq openai/gpt-oss-120b — fast, free, good Hinglish.
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
        # llama-3.3-70b-versatile was decommissioned 2026-08-16; this is Groq's
        # recommended replacement. Alternative: qwen/qwen3.6-27b.
        groq_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        # gpt-oss is a REASONING model: it spends completion tokens thinking
        # before it writes anything. Measured 2026-08-22 via /p3/dryrun —
        #   max_tokens=120, default effort -> 118 of 120 tokens were reasoning,
        #   finish_reason=length, content EMPTY. Mira said nothing at all.
        # reasoning_effort=low drops that to ~10 tokens and the reply arrives
        # in ~40 total. So: cap generously, and buy brevity with effort rather
        # than with a token ceiling. Length is controlled by the prompt and by
        # the thread machine's per-stage word budgets, which the model follows.
        _max_tok = int(os.getenv("LLM_MAX_TOKENS", "400"))
        _effort = (os.getenv("LLM_REASONING_EFFORT", "low") or "").strip()
        # OFF BY DEFAULT, because it cost ~4s on every single turn.
        #
        # Stopping at "?" also STRIPS it, and a Mira reply is usually one
        # question -- so the text handed to TTS ended with no terminal
        # punctuation at all. Pipecat's TTS aggregates by sentence, so it had
        # nothing to flush and sat waiting until generation finished and
        # ReplyShapeFilter appended the "?" at LLMFullResponseEndFrame. That
        # serialised what should have been streaming: measured 2026-08-28,
        # `llm responding` -> `tts started` was 4s on EVERY turn, suspiciously
        # constant, because it was not generation time -- it was waiting.
        #
        # The same End-frame append also put the "?" one reply late in the
        # captions, so every line of the transcript began with a stray "?".
        #
        # None of this bought anything: ReplyShapeFilter already truncates at
        # the first "?" IN STREAM. Letting the model emit the "?" itself keeps
        # the one-question guarantee, lets TTS start on the first sentence, and
        # puts the "?" where it belongs.
        _stops = ["?", "？"] if os.getenv("LLM_STOP_AT_QUESTION", "0") != "0" else []

        if llm_client.provider() == "bedrock" and AWSBedrockLLMService is not None:
            # Claude on Bedrock: no reasoning tokens before the first spoken
            # word, and the best instruction-following available at
            # conversational speed — which matters because almost all of
            # Mira's behaviour lives in prompt rules.
            _model = llm_client.bedrock_model("chat")
            _log(f"llm=bedrock model={_model} region={llm_client._region()} "
                 f"max_tokens={_max_tok} room={room_name}")
            llm = AWSBedrockLLMService(
                model=_model,
                aws_access_key=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                aws_session_token=os.getenv("AWS_SESSION_TOKEN") or None,
                aws_region=llm_client._region(),
                settings=AWSBedrockLLMService.Settings(
                    model=_model,
                    max_tokens=_max_tok,
                    temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
                    stop_sequences=_stops,
                ),
            )
        else:
            if llm_client.provider() == "bedrock":
                _log("llm=groq WARNING bedrock requested but pipecat's aws "
                     "extra is not installed in this image -- redeploy to pick "
                     f"up requirements.txt room={room_name}")
            _extra = {"reasoning_effort": _effort} if _effort else {}
            if _stops:
                _extra["stop"] = _stops
            _log(f"llm=groq model={groq_model} max_tokens={_max_tok} "
                 f"reasoning_effort={_effort or 'default'} room={room_name}")
            llm = GroqLLMService(
                api_key=os.getenv("GROQ_API_KEY"),
                settings=GroqLLMService.Settings(model=groq_model,
                                                 max_tokens=_max_tok,
                                                 extra=_extra),
            )

    # TTS engine selection. TTS_ENGINE env:
    #  - "cartesia" (DEFAULT): Cartesia Sonic 3.5 — Hindi + native Hinglish voice,
    #    WebSocket streaming, ~0.3s to first audio. Paid. Needs CARTESIA_API_KEY.
    #  - "gemini": Gemini TTS — free, but measured 5-7s from `tts started` to first
    #    audio on live calls (2026-08-17), vs Cartesia's ~0.3s. The preview TTS
    #    models are built for offline rendering, not conversation. Usable for
    #    development; too slow to put in front of users. Steer Hinglish with
    #    GEMINI_TTS_STYLE — the genai backend ignores `language`, so the style
    #    prefix is the only lever, and it is prepended to EVERY utterance (keep
    #    it short; a long prefix is re-processed on every turn).
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
        # Sits immediately after STT so an unusable transcript never reaches
        # the node machine, RAG, or the LLM. Applies to BOTH modes.
        TranscriptGate(),
    ]
    thread_proc = None
    if mode == "onboarding":
        # Onboarding: node-based state machine controls the call flow.
        # Each node has its own focused prompt; code drives transitions.
        # No RAG needed during onboarding — Mira is collecting info, not advising.
        node_proc = onboarding_nodes.create_node_processor(context, _onboarding_profile, log_fn=_log)
        _log(f"onboarding node system enabled room={room_name}")
        stages.append(node_proc)
    else:
        # P-3. P3_ENGINE=graph (default) runs the thread machine; P3_ENGINE=rag
        # falls back to the previous single-prompt + retrieval behaviour, so
        # the two can be compared without a redeploy.
        _p3 = os.getenv("P3_ENGINE", "graph").strip().lower()
        if _p3 == "graph":
            thread_proc = ThreadMachineProcessor(
                context, firebase_uid=firebase_uid or "",
                ledger_view=existing_memory or {}, fact_ages=fact_ages or {},
            )
            _log(f"p3 thread machine enabled room={room_name} "
                 f"ledger_paths={len(fact_ages or {})} rag={rag.enabled()}")
            stages.append(thread_proc)
        elif rag.enabled():
            _log(f"p3 rag-only (P3_ENGINE=rag) room={room_name} top_k={os.getenv('RAG_TOP_K', '3')}")
            stages.append(RAGProcessor(
                context,
                top_k=int(os.getenv("RAG_TOP_K", "3")),
                min_similarity=float(os.getenv("RAG_MIN_SIMILARITY", "0.5")),
            ))
        else:
            _log(f"p3 plain prompt room={room_name} (no supabase/google key)")
    stages += [
        aggregator.user(),            # add user turn to context
        llm,                          # reasoning (sees any injected references)
    ]
    if thread_proc is not None:
        # Observes what Mira actually said vs the stage policy. Pass-through,
        # so it adds no latency and never withholds audio.
        stages.append(ResponseValidator(thread_proc))
    # Both products stack questions when the prompt lists examples. Enforced
    # here rather than requested in prose, because the prose already failed.
    # Caps are per product: onboarding asks for ~25 words, P-3's largest stage
    # budget (ADVISE) is 45. Both are set above the request so the filter only
    # catches genuine runaway, not ordinary variation.
    stages.append(ReplyShapeFilter(
        " (onboarding)" if mode == "onboarding" else "",
        word_cap=int(os.getenv("P2_WORD_CAP", "32")) if mode == "onboarding"
        else int(os.getenv("P3_WORD_CAP", "60")),
        # Read the env var rather than the _stops local: that local is only
        # bound inside the groq/bedrock branch, so referencing it here would
        # NameError on any other engine.
        restore_question_mark=os.getenv("LLM_STOP_AT_QUESTION", "0") != "0"))
    stages += [
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

    # --- End of session: PERSIST FIRST, then consolidate --------------------
    # Ordering here is the whole point. The transcript is written to the
    # Sessions tab before any LLM is involved, so a malformed consolidation
    # response costs the memory *update* and never the conversation. Anything
    # left un-consolidated keeps its transcript and is replayable via
    # POST /reprocess. Consolidated memory is a derived artefact; the
    # transcript is the source of truth.
    if firebase_uid:
        ended_at = datetime.now(timezone.utc).isoformat()
        transcript = _transcript_from_context(context)
        user_turns = sum(1 for m in context.get_messages() if m.get("role") == "user")

        stored = False
        if user_turns >= 1:
            try:
                memory_store.save_session_raw(
                    session_id=run_id, firebase_uid=firebase_uid, user_id=user_id,
                    session_type=mode, started_at=started_at, ended_at=ended_at,
                    transcript=transcript, turns=user_turns,
                )
                stored = True
                _log(f"transcript saved room={room_name} session={run_id} "
                     f"turns={user_turns} chars={len(transcript)}")
            except Exception as exc:
                # The only genuinely unrecoverable failure left. Dump the
                # transcript to the log so it is at least retrievable by hand.
                _log(f"CRITICAL transcript save FAILED room={room_name} "
                     f"session={run_id}: {exc}")
                _log(f"CRITICAL unsaved transcript session={run_id}: {transcript}")
        else:
            _log(f"no user turns room={room_name} -> nothing to store")

        if stored:
            try:
                _log(f"consolidating room={room_name} session={run_id} turns={user_turns}")

                # The ledger is the source of truth. On first use for a user who
                # already has a pre-ledger memory document, seed it so nobody
                # starts from empty.
                facts = memory_store.load_facts(firebase_uid)
                if not facts and (existing_memory or {}):
                    seed_rows, _, _ = memory_facts.apply_patch(
                        [], memory_facts.seed_ops_from_document(existing_memory),
                        "migration", "", when=ended_at,
                        firebase_uid=firebase_uid, user_id=user_id)
                    memory_store.append_facts(seed_rows)
                    facts = seed_rows
                    _log(f"ledger seeded from legacy memory session={run_id} "
                         f"facts={len(seed_rows)}")

                current_view = memory_facts.build_current_view(facts)

                # The model proposes; the code decides.
                patch = await consolidate.consolidate_patch(
                    current_view, existing_open_loops or [], transcript)
                new_rows, invalidations, audit = memory_facts.apply_patch(
                    facts, patch["ops"], run_id, transcript, when=ended_at,
                    firebase_uid=firebase_uid, user_id=user_id)

                memory_store.append_facts(new_rows)
                memory_store.stamp_invalidations(invalidations, ended_at)

                # Reflect the invalidations locally so the projection is correct
                # without re-reading the sheet.
                closed_by = dict(invalidations)
                for f in facts:
                    if f.get("fact_id") in closed_by:
                        f["invalidated_at"] = ended_at
                        f["invalidated_by"] = closed_by[f["fact_id"]]
                view = memory_facts.build_current_view(facts + new_rows)

                # Threads left unfinished carry forward, so a topic the user
                # switched away from is picked up next call instead of vanishing.
                loops = list(patch["open_loops"])
                if thread_proc is not None:
                    for extra in thread_proc.open_loops():
                        if extra not in loops:
                            loops.append(extra)
                    if len(loops) != len(patch["open_loops"]):
                        _log(f"carried {len(loops) - len(patch['open_loops'])} "
                             f"unfinished thread(s) into open loops")

                memory_store.cache_current_view(
                    firebase_uid=firebase_uid, view=view,
                    open_loops=loops,
                    session_summary=patch["session_summary"],
                    ended_at=ended_at, session_type=mode,
                )
                memory_store.finalize_session(
                    run_id, status=memory_store.STATUS_DONE,
                    session_summary=patch["session_summary"],
                    open_loops=loops,
                    consolidated_at=datetime.now(timezone.utc).isoformat(),
                )
                applied = sum(1 for a in audit if a.get("applied"))
                rejected = len(audit) - applied
                for a in audit:
                    if not a.get("applied") and a.get("reason") not in ("unchanged", "duplicate"):
                        _log(f"  op REJECTED {a.get('op')} {a.get('path')}: {a.get('reason')}")
                _log(f"memory saved room={room_name} session={run_id} "
                     f"ops={len(patch['ops'])} applied={applied} rejected={rejected} "
                     f"closed={len(invalidations)} open_loops={len(patch['open_loops'])}")
            except Exception as exc:
                _log(f"consolidation failed room={room_name} session={run_id} "
                     f"(transcript kept, replayable): {exc}")
                try:
                    memory_store.finalize_session(
                        run_id, status=memory_store.STATUS_FAILED, error=str(exc))
                except Exception as mark_exc:
                    _log(f"could not mark session failed {run_id}: {mark_exc}")


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
    """Liveness, plus WHICH BUILD is answering.

    Railway injects RAILWAY_GIT_COMMIT_SHA at build time. Without it there is
    no way to tell a finished deploy from a still-running old one -- the
    endpoint returns an identical {"ok": true} either way, so a push that
    never landed looks exactly like a push that did.

    `llm` reports the provider this process actually resolved, which is not
    always the one the env vars imply: a bearer token that is not picked up,
    or a missing pipecat aws extra, both silently leave it on groq.
    """
    sha = (os.getenv("RAILWAY_GIT_COMMIT_SHA")
           or os.getenv("GIT_COMMIT_SHA") or "")
    return {
        "ok": True,
        "commit": sha[:7] or "unknown",
        "llm": llm_client.provider(),
        "bedrock_model": llm_client.bedrock_model("chat") or None,
        "aws_extra": AWSBedrockLLMService is not None,
    }


@app.get("/events")
async def events():
    """Temporary: last pipeline events (to see where audio dies)."""
    return {"events": list(_EVENTS)}


@app.get("/pending")
async def pending_sessions():
    """Sessions whose transcript is stored but whose consolidation never landed.

    The recovery queue. Every row here still holds the full conversation, so
    nothing is lost — the memory update just needs re-running.
    """
    try:
        rows = await asyncio.to_thread(memory_store.get_replayable_sessions, 50)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)
    return {
        "count": len(rows),
        "sessions": [
            {k: v for k, v in r.items() if k != "transcript"} | {
                "transcript_chars": len(r["transcript"])
            }
            for r in rows
        ],
    }


def _reprocess_sync(session_id=None, limit=10):
    """Re-run consolidation for stored-but-unconsolidated sessions. Blocking."""
    rows = memory_store.get_replayable_sessions(limit=100)
    if session_id:
        rows = [r for r in rows if r["session_id"] == session_id]
    rows = rows[:limit]

    results = []
    for r in rows:
        sid = r["session_id"]
        uid = r["firebase_uid"]
        if not uid:
            # Pre-durability rows have no firebase_uid, so memory cannot be
            # attributed to a user. The transcript is still kept.
            results.append({"session_id": sid, "ok": False,
                            "error": "no firebase_uid on row"})
            continue
        try:
            when = r["ended_at"] or datetime.now(timezone.utc).isoformat()
            mem = memory_store.load_memory(uid)
            facts = memory_store.load_facts(uid)
            if not facts and (mem.get("long_term_memory") or {}):
                seed_rows, _, _ = memory_facts.apply_patch(
                    [], memory_facts.seed_ops_from_document(mem["long_term_memory"]),
                    "migration", "", when=when, firebase_uid=uid,
                    user_id=r["user_id"])
                memory_store.append_facts(seed_rows)
                facts = seed_rows

            # _reprocess_sync is blocking (run via asyncio.to_thread), so the
            # now-async consolidate_patch needs its own loop here.
            patch = asyncio.run(consolidate.consolidate_patch(
                memory_facts.build_current_view(facts),
                mem.get("open_loops") or [],
                r["transcript"],
            ))
            new_rows, invalidations, audit = memory_facts.apply_patch(
                facts, patch["ops"], sid, r["transcript"], when=when,
                firebase_uid=uid, user_id=r["user_id"])
            memory_store.append_facts(new_rows)
            memory_store.stamp_invalidations(invalidations, when)

            closed_by = dict(invalidations)
            for f in facts:
                if f.get("fact_id") in closed_by:
                    f["invalidated_at"] = when
                    f["invalidated_by"] = closed_by[f["fact_id"]]

            memory_store.cache_current_view(
                firebase_uid=uid,
                view=memory_facts.build_current_view(facts + new_rows),
                open_loops=patch["open_loops"],
                session_summary=patch["session_summary"],
                ended_at=when, session_type=r["type"],
            )
            memory_store.finalize_session(
                sid, status=memory_store.STATUS_DONE,
                session_summary=patch["session_summary"],
                open_loops=patch["open_loops"],
                consolidated_at=datetime.now(timezone.utc).isoformat(),
            )
            applied = sum(1 for a in audit if a.get("applied"))
            _log(f"reprocessed session={sid} applied={applied}/{len(audit)} "
                 f"closed={len(invalidations)}")
            results.append({"session_id": sid, "ok": True, "applied": applied,
                            "rejected": len(audit) - applied,
                            "open_loops": len(patch["open_loops"])})
        except Exception as exc:
            try:
                memory_store.finalize_session(
                    sid, status=memory_store.STATUS_FAILED, error=str(exc))
            except Exception:
                pass
            _log(f"reprocess failed session={sid}: {exc}")
            results.append({"session_id": sid, "ok": False, "error": str(exc)[:300]})
    return results


@app.post("/p3/dryrun")
async def p3_dryrun(request: Request):
    """Run the P-3 thread machine over scripted turns — REAL router, REAL LLM,
    no audio.

    TEST TOOLING, NOT A PRODUCT SURFACE. It exists because router quality and
    real latency cannot be observed any other way without making live calls,
    and live calls burn TTS credit.

    STRICTLY READ-ONLY: nothing is written to the ledger, the users row or the
    Sessions tab. STT and TTS are never invoked.

    Body: {"uid": "<firebase_uid>", "turns": ["...", "..."], "verbose": true}
    Guard with ADMIN_TOKEN (?token=...) — it spends Groq tokens.
    """
    # FAIL CLOSED. This endpoint spends Groq tokens on request, so unlike
    # /reprocess it refuses outright when ADMIN_TOKEN is unset rather than
    # falling open. It is test tooling that happens to live in production.
    admin = (os.getenv("ADMIN_TOKEN") or "").strip()
    if not admin:
        return JSONResponse(
            {"error": "disabled: set ADMIN_TOKEN to enable /p3/dryrun"},
            status_code=403)
    if request.query_params.get("token") != admin:
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    try:
        body = await request.json()
    except Exception:
        body = {}
    turns = [str(t) for t in (body or {}).get("turns", []) if str(t).strip()]
    overrides = {
        "max_tokens": (body or {}).get("max_tokens"),
        "reasoning_effort": (body or {}).get("reasoning_effort"),
        "model": (body or {}).get("model"),
    }
    if not turns:
        return JSONResponse({"error": "pass {\"turns\": [\"...\"]}"}, status_code=400)
    uid = str((body or {}).get("uid") or "").strip()

    try:
        return await asyncio.to_thread(_dryrun_sync, uid, turns[:20], overrides)
    except Exception as exc:
        import traceback
        return JSONResponse({"error": str(exc),
                             "trace": traceback.format_exc()[-1500:]}, status_code=500)


def _dryrun_sync(uid, turns, overrides=None):
    """Blocking body of /p3/dryrun. Mirrors the real turn path exactly."""
    import time
    import p3_graph
    from groq import Groq

    profile = load_profile_for_call(uid) if uid else {}
    memory = memory_store.load_memory(uid) if uid else {}
    ledger_view = (memory or {}).get("long_term_memory") or {}

    fact_ages = {}
    if uid:
        for f in memory_store.load_facts(uid):
            if f.get("status") == memory_facts.STATUS_APPLIED and not \
                    str(f.get("invalidated_at", "")).strip():
                fact_ages[f.get("path")] = f.get("valid_from") or f.get("created_at")

    base_prompt = build_system_prompt("ongoing", profile, memory)
    graph = p3_graph.get_graph()
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    overrides = overrides or {}
    model = overrides.get("model") or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    max_tok = int(overrides.get("max_tokens") or os.getenv("LLM_MAX_TOKENS", "400"))
    effort = overrides.get("reasoning_effort") or os.getenv("LLM_REASONING_EFFORT", "low")

    threads, history, out_rows = [], [], []

    for i, text in enumerate(turns, 1):
        t0 = time.perf_counter()
        state = asyncio.run(graph.ainvoke({
            "turn_text": text, "turn_index": i, "threads": threads,
            "ledger_view": ledger_view, "fact_ages": fact_ages,
            "history": list(history), "trace": [],
        }))
        graph_ms = (time.perf_counter() - t0) * 1000
        threads = state.get("threads", threads)

        row = {
            "turn": i, "user": text,
            "lane": state.get("lane"), "situation": state.get("situation"),
            "stage": state.get("stage"),
            "graph_ms": round(graph_ms, 1),
            "trace": state.get("trace", []),
            "retrieved": len(state.get("retrieved") or []),
        }
        active = next((t for t in threads if not t.parked), None)
        row["thread"] = None if not active else {
            "topic": active.topic, "stage": active.stage,
            "slots": dict(active.slots), "gaps": active.gaps(),
        }
        row["parked"] = [t.topic for t in threads if t.parked]
        g = state.get("gather") or {}
        row["needed"] = list(active.needed_paths) + list(active.adhoc) if active else []
        row["known_from_memory"] = list((g.get("known") or {}).keys())
        row["stale"] = g.get("stale") or []
        row["missing"] = g.get("missing") or []
        row["directive"] = state.get("directive")
        row["budget_words"] = state.get("budget")
        row["may_advise"] = state.get("may_advise")

        if state.get("safety_hit"):
            row["mira"] = state["safety_hit"]
            row["llm_ms"] = 0
            row["safety"] = True
        else:
            msgs = [{"role": "system", "content": base_prompt}]
            msgs += history
            block = (state.get("directive") or "")
            docs = state.get("retrieved") or []
            if docs:
                block += "\n\n" + rag.format_reference(docs)
            msgs.append({"role": "system", "content": block})
            msgs.append({"role": "user", "content": text})
            t1 = time.perf_counter()
            try:
                kw = dict(model=model, messages=msgs,
                          max_tokens=max_tok, temperature=0.7)
                if effort:
                    kw["reasoning_effort"] = effort
                resp = client.chat.completions.create(**kw)
                choice = resp.choices[0]
                reply = (choice.message.content or "").strip()
                row["finish_reason"] = getattr(choice, "finish_reason", None)
                usage = getattr(resp, "usage", None)
                row["completion_tokens"] = getattr(usage, "completion_tokens", None)
                row["reasoning_tokens"] = getattr(
                    getattr(usage, "completion_tokens_details", None),
                    "reasoning_tokens", None)
            except Exception as exc:
                reply = f"[LLM ERROR: {exc}]"
            row["llm_ms"] = round((time.perf_counter() - t1) * 1000, 1)
            row["mira"] = reply
            row["safety"] = False

        row["words"] = len(str(row["mira"]).split())
        row["total_ms"] = round(row["graph_ms"] + row["llm_ms"], 1)
        history += [{"role": "user", "content": text},
                    {"role": "assistant", "content": row["mira"]}]
        out_rows.append(row)

    return {
        "uid": uid or None,
        "ledger_paths_known": len(fact_ages),
        "base_prompt_chars": len(base_prompt),
        "router_model": p3_graph._ROUTER_MODEL,
        "compose_model": model,
        "max_tokens": max_tok,
        "reasoning_effort": effort,
        "turns": out_rows,
        "note": "READ-ONLY — no memory, ledger or session writes were performed.",
    }


@app.get("/audit")
async def audit(request: Request):
    """Why does Mira believe this? The full fact ledger for one user.

    ?uid=<firebase_uid>          every fact-version, newest first
    &path=diet.type              narrow to one field's timeline
    &include_rejected=0          hide refused claims (default shows them)

    Each row carries the evidence the model cited, the session it came from,
    when it became true, and when/what superseded it.
    """
    uid = (request.query_params.get("uid") or "").strip()
    if not uid:
        return JSONResponse({"error": "pass ?uid=<firebase_uid>"}, status_code=400)
    path = (request.query_params.get("path") or "").strip() or None
    show_rejected = (request.query_params.get("include_rejected") or "1") != "0"
    try:
        facts = await asyncio.to_thread(memory_store.load_facts, uid)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)

    rows = memory_facts.history_for(facts, path)
    if not show_rejected:
        rows = [r for r in rows if r.get("status") == memory_facts.STATUS_APPLIED]

    live = memory_facts.live_facts(facts)
    return {
        "uid": uid,
        "path": path,
        "totals": {
            "ledger_rows": len(facts),
            "live_facts": len(live),
            "superseded": sum(1 for f in facts
                              if str(f.get("invalidated_at", "")).strip()),
            "rejected": sum(1 for f in facts
                            if f.get("status") == memory_facts.STATUS_REJECTED),
        },
        "current_view": memory_facts.build_current_view(facts),
        "history": [
            {
                "fact_id": r.get("fact_id"), "path": r.get("path"),
                "op": r.get("op"), "value": r.get("value"),
                "status": r.get("status"), "confidence": r.get("confidence"),
                "evidence": r.get("evidence"), "session_id": r.get("session_id"),
                "valid_from": r.get("valid_from"),
                "invalidated_at": r.get("invalidated_at"),
                "invalidated_by": r.get("invalidated_by"),
                "reason": r.get("reason"),
            }
            for r in rows[:400]
        ],
    }


@app.post("/reprocess")
async def reprocess(request: Request):
    """Replay consolidation for sessions that stored a transcript but failed.

    Body (all optional): {"session_id": "run-abc123", "limit": 10}

    Set ADMIN_TOKEN on the service to require ?token=... — this endpoint writes
    to user memory, so lock it down in anything user-facing.
    """
    admin = (os.getenv("ADMIN_TOKEN") or "").strip()
    if admin and request.query_params.get("token") != admin:
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    try:
        body = await request.json()
    except Exception:
        body = {}
    session_id = (body or {}).get("session_id")
    limit = int((body or {}).get("limit") or 10)
    try:
        results = await asyncio.to_thread(_reprocess_sync, session_id, limit)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)
    ok = sum(1 for r in results if r["ok"])
    return {"attempted": len(results), "succeeded": ok, "results": results}




@app.get("/memory", response_class=HTMLResponse)
async def memory_view(request: Request):
    """Readable view of a user's long-term memory, open loops, and session
    history. Prototype view — anyone with the uid can see it, so keep it
    auth-gated before production (it's personal health data)."""
    import html as _html
    import traceback as _tb
    uid = request.query_params.get("uid")
    if not uid:
        return HTMLResponse("<p style='font-family:sans-serif;padding:40px'>Add <code>?uid=…</code> to view a user's memory.</p>")
    try:
        return await _memory_view_inner(request, uid)
    except Exception:
        err = _html.escape(_tb.format_exc())
        return HTMLResponse(f"<pre style='font-family:monospace;padding:20px;white-space:pre-wrap'>{err}</pre>", status_code=500)

async def _memory_view_inner(request, uid):
    import html as _html
    import json as _json

    mem = memory_store.load_memory(uid)
    user_id = mem.get("user_id", "")
    try:
        profile = profile_store.load_profile_for_uid(uid)
    except Exception:
        profile = {}
    name = profile.get("name") or "This user"
    ltm = mem.get("long_term_memory") or {}
    sessions = memory_store.get_sessions(user_id) if user_id else []
    try:
        facts = memory_store.load_facts(uid)
    except Exception:
        facts = []

    def esc(x):
        return _html.escape(str(x))

    def ul(items):
        items = [i for i in (items or []) if str(i).strip()]
        if not items:
            return "<p class='empty'>— nothing yet —</p>"
        return "<ul>" + "".join(f"<li>{esc(i)}</li>" for i in items) + "</ul>"

    def fval(raw):
        """Ledger values are JSON-encoded; render them readably."""
        if raw in (None, ""):
            return "—"
        try:
            v = _json.loads(raw) if isinstance(raw, str) else raw
        except (ValueError, TypeError):
            return str(raw)
        if isinstance(v, (dict, list)):
            return _json.dumps(v, ensure_ascii=False)
        return str(v)

    def conf_pill(conf):
        c = (conf or "").strip().lower()
        cls = {"high": "conf-high", "medium": "conf-medium",
               "low": "conf-low", "migrated": "conf-mig"}.get(c, "conf-low")
        return f"<span class='pill {cls}'>{esc(c or 'unknown')}</span>" if c else ""

    def when(ts):
        return esc((ts or "")[:16].replace("T", " ")) or "—"

    def prov_line(f):
        """session + timestamp + confidence — the 'where did this come from' line."""
        sid = f.get("session_id") or "—"
        bits = [f"<span class='rid-inline'>{esc(sid)}</span>",
                f"<span>{when(f.get('valid_from') or f.get('created_at'))}</span>"]
        cp = conf_pill(f.get("confidence"))
        if cp:
            bits.append(cp)
        return "<div class='prov'>" + " · ".join(bits) + "</div>"

    def quote(f):
        ev = (f.get("evidence") or "").strip()
        if not ev:
            return ""
        return f"<div class='quote'>“{esc(ev)}”</div>"

    def provenance_html(all_facts):
        """Every fact Mira currently believes, with its source. Grouped by section."""
        live = [f for f in memory_facts.live_facts(all_facts)
                if f.get("op") != memory_facts.OP_INVALIDATE]
        if not live:
            return "<p class='empty'>— no facts recorded yet (populates after the next call) —</p>"
        groups = {}
        for f in live:
            groups.setdefault((f.get("path") or "").split(".")[0], []).append(f)
        out = []
        for section in sorted(groups):
            out.append(f"<h3>{esc(section.replace('_', ' '))}</h3>")
            for f in groups[section]:
                out.append(
                    "<div class='fact'>"
                    f"<div class='fact-head'><code class='fact-path'>{esc(f.get('path'))}</code>"
                    f"<span class='fact-val'>{esc(fval(f.get('value')))}</span></div>"
                    + quote(f) + prov_line(f) +
                    "</div>"
                )
        return "".join(out)

    def changed_html(all_facts):
        """Facts that were true and no longer are — kept, never deleted."""
        closed = [f for f in (all_facts or [])
                  if str(f.get("invalidated_at") or "").strip()
                  and f.get("status") == memory_facts.STATUS_APPLIED]
        if not closed:
            return "<p class='empty'>— nothing has changed yet —</p>"
        closed.sort(key=lambda f: str(f.get("invalidated_at")), reverse=True)
        by_id = {f.get("fact_id"): f for f in (all_facts or [])}
        out = []
        for f in closed:
            repl = by_id.get(f.get("invalidated_by")) or {}
            new_val = fval(repl.get("value")) if repl.get("value") else None
            if repl.get("op") == memory_facts.OP_INVALIDATE or not new_val:
                right = "<span class='fact-val removed'>no longer tracked</span>"
            else:
                right = f"<span class='fact-val'>{esc(new_val)}</span>"
            out.append(
                "<div class='fact'>"
                f"<div class='fact-head'><code class='fact-path'>{esc(f.get('path'))}</code></div>"
                "<div class='change'>"
                f"<span class='fact-val old'>{esc(fval(f.get('value')))}</span>"
                f"<span class='arrow'>→</span>{right}</div>"
                + quote(repl)
                + (f"<div class='reason'>{esc(repl.get('reason'))}</div>"
                   if (repl.get('reason') or '').strip() else "")
                + "<div class='prov'>was true "
                f"<span>{when(f.get('valid_from') or f.get('created_at'))}</span>"
                f" → <span>{when(f.get('invalidated_at'))}</span> · changed in "
                f"<span class='rid-inline'>{esc(repl.get('session_id') or '—')}</span></div>"
                "</div>"
            )
        return "".join(out)

    def rejected_html(all_facts):
        """Claims the model proposed that validation refused. Kept as an audit trail."""
        rej = [f for f in (all_facts or [])
               if f.get("status") == memory_facts.STATUS_REJECTED]
        if not rej:
            return "<p class='empty'>— nothing has been rejected —</p>"
        rej.sort(key=lambda f: str(f.get("created_at")), reverse=True)
        out = []
        for f in rej:
            out.append(
                "<div class='fact rej'>"
                f"<div class='fact-head'><code class='fact-path'>{esc(f.get('path') or '?')}</code>"
                f"<span class='fact-val'>{esc(fval(f.get('value')))}</span></div>"
                f"<div class='reason warn'>blocked: {esc(f.get('reason') or 'unknown')}</div>"
                + quote(f) +
                "<div class='prov'>"
                f"<span class='rid-inline'>{esc(f.get('session_id') or '—')}</span> · "
                f"<span>{when(f.get('created_at'))}</span> · "
                f"<span>op: {esc(f.get('op') or '?')}</span></div>"
                "</div>"
            )
        return "".join(out)

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
        _hl_dislike = {"dislikes"}
        ltm_html += f"<h3>Preferences</h3>{kv_table(pref_items, highlight_keys=_hl_dislike)}"

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
        _hl_prog = {"what_failed", "struggles"}
        ltm_html += f"<h3>Progress</h3>{kv_table(prog_items, highlight_keys=_hl_prog)}"

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

    _live_facts = [f for f in memory_facts.live_facts(facts)
                   if f.get("op") != memory_facts.OP_INVALIDATE]
    _closed_facts = [f for f in facts if str(f.get("invalidated_at") or "").strip()
                     and f.get("status") == memory_facts.STATUS_APPLIED]
    _rej_facts = [f for f in facts if f.get("status") == memory_facts.STATUS_REJECTED]
    _sessions_seen = len({f.get("session_id") for f in facts if f.get("session_id")})
    prov_block = provenance_html(facts)
    changed_block = changed_html(facts)
    rejected_block = rejected_html(facts)

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
  /* --- fact ledger: provenance / history / rejected --- */
  .fact{{padding:9px 0;border-bottom:1px solid var(--line);}}
  .fact:last-child{{border-bottom:none;}}
  .fact.rej{{opacity:.92;}}
  .fact-head{{display:flex;gap:9px;align-items:baseline;flex-wrap:wrap;}}
  .fact-path{{font-family:ui-monospace,Menlo,monospace;font-size:11px;color:var(--muted);
    background:#f2f6fd;border:1px solid var(--line);border-radius:5px;padding:1px 6px;white-space:nowrap;}}
  .fact-val{{font-size:14px;font-weight:600;color:var(--ink);}}
  .fact-val.old{{text-decoration:line-through;color:#9aa7bd;font-weight:500;}}
  .fact-val.removed{{color:#b44;font-weight:500;font-style:italic;}}
  .change{{display:flex;gap:8px;align-items:baseline;flex-wrap:wrap;margin-top:4px;}}
  .arrow{{color:var(--accent-fg);font-weight:700;}}
  .quote{{margin:5px 0 0;padding:3px 0 3px 10px;border-left:3px solid var(--accent-bg);
    font-size:13px;color:var(--muted);font-style:italic;}}
  .reason{{font-size:12.5px;color:var(--muted);margin-top:4px;}}
  .prov{{display:flex;gap:7px;align-items:center;flex-wrap:wrap;font-size:11px;
    color:#9aa7bd;margin-top:5px;}}
  .rid-inline{{font-family:ui-monospace,Menlo,monospace;font-size:10.5px;color:#8494ad;
    background:#f7faff;border:1px solid var(--line);border-radius:4px;padding:1px 5px;}}
  .prov .pill{{font-size:10px;padding:1px 7px;}}
  .conf-high{{background:rgba(34,139,34,.12);color:#1a6b1a;}}
  .conf-medium{{background:rgba(200,140,20,.14);color:#8a5a00;}}
  .conf-low{{background:rgba(180,40,40,.1);color:#a22;}}
  .conf-mig{{background:#eef1f6;color:#6b7a92;}}
  .stats{{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:2px;}}
  .stat{{flex:1;min-width:88px;background:#fbfdff;border:1px solid var(--line);
    border-radius:10px;padding:9px 11px;}}
  .stat b{{display:block;font-size:19px;letter-spacing:-.02em;}}
  .stat span{{font-size:10.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;}}
  .hint{{font-size:12px;color:var(--muted);margin-bottom:8px;}}
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
    <h2>📊 Memory ledger</h2>
    <p class="hint">Every fact is stored with its source and its time bounds. Nothing is ever
      overwritten — a change closes the old fact and opens a new one.</p>
    <div class="stats">
      <div class="stat"><b>{len(_live_facts)}</b><span>facts held</span></div>
      <div class="stat"><b>{len(_closed_facts)}</b><span>changed</span></div>
      <div class="stat"><b>{len(_rej_facts)}</b><span>blocked</span></div>
      <div class="stat"><b>{_sessions_seen}</b><span>calls</span></div>
      <div class="stat"><b>{len(facts)}</b><span>ledger rows</span></div>
    </div>
  </section>

  <section>
    <h2>🔎 Where each fact came from</h2>
    <p class="hint">The exact words Mira heard, which call it was said in, and how confident
      the extraction was.</p>
    {prov_block}
  </section>

  <section>
    <h2>🕓 What changed over time</h2>
    <p class="hint">Previous values are kept, not deleted — so you can see how this user's
      situation moved, and what evidence caused each change.</p>
    {changed_block}
  </section>

  <section>
    <h2>🚫 Claims Mira was not allowed to remember</h2>
    <p class="hint">Proposed facts that failed validation — an unknown field, a wrong type, or
      evidence that did not appear in the transcript. Kept so hallucinations are visible
      rather than silent.</p>
    {rejected_block}
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

    # Age of each known fact, for the thread machine's staleness check. Read
    # once here, never on the audio path — the live turn only ever touches the
    # cached projection, so this costs nothing per turn.
    fact_ages = {}
    if firebase_uid and mode != "onboarding":
        try:
            for f in await asyncio.to_thread(memory_store.load_facts, firebase_uid):
                if f.get("status") == memory_facts.STATUS_APPLIED and not \
                        str(f.get("invalidated_at", "")).strip():
                    fact_ages[f.get("path")] = f.get("valid_from") or f.get("created_at")
        except Exception as exc:
            _log(f"fact ages unavailable: {exc}")

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
        fact_ages=fact_ages,
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
