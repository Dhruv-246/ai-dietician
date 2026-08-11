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
    from pipecat.turns.user_start import MinWordsUserTurnStartStrategy
    from pipecat.turns.user_turn_strategies import UserTurnStrategies
    from pipecat.utils.text.base_text_filter import BaseTextFilter
    from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
    from pipecat.services.sarvam.tts import SarvamTTSService
    from pipecat.services.sarvam.stt import SarvamSTTService
    from pipecat.services.google.llm import GoogleLLMService
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

    prof_bits = [f"{k.capitalize()}: {profile[k]}" for k in _PROFILE_FIELDS
                 if str(profile.get(k, "")).strip()]
    if prof_bits:
        lines.append("Profile — " + " | ".join(prof_bits))

    for label, key in (("Goals", "goals"), ("Preferences", "preferences"),
                       ("Dislikes", "dislikes"), ("Known facts", "facts")):
        vals = ltm.get(key) or []
        if vals:
            lines.append(f"{label}: " + "; ".join(str(v) for v in vals))

    open_loops = memory.get("open_loops") or []
    if open_loops:
        lines.append("Open loops (gently follow up on ONE if it fits — do not interrogate): "
                     + "; ".join(str(v) for v in open_loops))

    # Continuity signals
    count = int(memory.get("session_count") or 0)
    if count > 0:
        cont = f"This is check-in #{count + 1}."
        last_at = str(memory.get("last_session_at") or "").strip()
        if last_at:
            try:
                delta = datetime.now(timezone.utc) - datetime.fromisoformat(last_at)
                days = max(0, delta.days)
                cont += f" Last call was {'today' if days == 0 else f'{days} day(s) ago'}."
            except Exception:
                pass
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
_EVENTS: deque = deque(maxlen=80)


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
            _log(f"ERROR FRAME room={self._room}: {getattr(frame, 'error', frame)}")
            return
        for cls, label in (
            (LLMFullResponseStartFrame, "llm responding"),
            (TTSStartedFrame, "tts producing audio"),
            (BotStartedSpeakingFrame, "bot speaking"),
        ):
            if isinstance(frame, cls) and label not in self._seen:
                self._seen.add(label)
                _log(f"{label} room={self._room}")
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


# --------------------------------------------------------------------------- #
# The voice pipeline for a single call, joined to one LiveKit room.            #
# --------------------------------------------------------------------------- #
async def run_livekit_bot(room_name: str, system_prompt: str, *,
                          firebase_uid=None, user_id="", run_id="",
                          mode="onboarding", existing_memory=None, existing_open_loops=None):
    """Join `room_name` as Mira, run the conversation, then consolidate memory."""
    _log(f"bot starting room={room_name} mode={mode} prompt_chars={len(system_prompt)}")
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

    # STT: Sarvam Saarika. Force Hindi (hi-IN) instead of auto-detect — auto-detect
    # was mis-reading Hindi speech as Punjabi/Gurmukhi. saarika:v2.5 still handles
    # English words (Hinglish) spoken within Hindi.
    stt = SarvamSTTService(
        api_key=os.getenv("SARVAM_API_KEY"),
        settings=SarvamSTTService.Settings(
            model=os.getenv("STT_MODEL", "saarika:v2.5"),
            language=Language.HI_IN,
        ),
    )

    # LLM for Mira's responses: Gemini 2.5 Flash — strong, natural Hinglish and
    # good instruction-following (a quality upgrade over Llama-70B). Override
    # with GEMINI_MODEL. (Groq is still used, but ONLY for the cheap background
    # memory consolidation in consolidate.py — not for live responses.)
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    _log(f"llm gemini model={gemini_model}")
    llm = GoogleLLMService(
        api_key=os.getenv("GOOGLE_API_KEY"),
        settings=GoogleLLMService.Settings(model=gemini_model),
    )

    # TTS with automatic fallback (both get the sanitizer so only Hindi/English
    # is ever spoken):
    #  - PREFER ElevenLabs (eleven_flash_v2_5) — best natural English + names —
    #    whenever the account has credits.
    #  - FALL BACK to Sarvam bulbul (free, Hindi-first, English slightly accented)
    #    when ElevenLabs is out of monthly credits / key missing, so Mira never
    #    goes silent. Checked once at call start, off the audio path.
    if await asyncio.to_thread(_elevenlabs_has_credits):
        _log(f"tts=elevenlabs room={room_name}")
        tts = ElevenLabsTTSService(
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            text_filters=[ScriptTextFilter()],
            settings=ElevenLabsTTSService.Settings(
                voice=os.getenv("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL"),  # "Sarah" (premade)
                model=os.getenv("ELEVENLABS_MODEL", "eleven_flash_v2_5"),
                language=Language.HI,
            ),
        )
    else:
        _log(f"tts=sarvam-bulbul (fallback) room={room_name}")
        tts = SarvamTTSService(
            api_key=os.getenv("SARVAM_API_KEY"),
            text_filters=[ScriptTextFilter()],
            settings=SarvamTTSService.Settings(
                voice=os.getenv("SARVAM_VOICE", "anushka"),
                model=os.getenv("SARVAM_TTS_MODEL", "bulbul:v2"),
                language=Language.HI,
            ),
        )

    # Turn-taking + barge-in, tuned to NOT trip on background noise / speaker echo:
    #  - VAD: higher confidence + min_volume so quiet background isn't treated as
    #    speech; stop_secs=0.5 keeps replies snappy without cutting the user off.
    #  - Interruptions require >= 2 actual transcribed WORDS (MinWords strategy).
    #    A cough, background TV, or residual speaker echo won't produce two clean
    #    words, so it can't barge in on Mira — but a deliberate couple of words
    #    still interrupts her immediately. (When Mira isn't speaking, one word
    #    starts a normal turn, so responsiveness is unchanged.)
    context = LLMContext([{"role": "system", "content": system_prompt}])
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

    pipeline = Pipeline(
        [
            transport.input(),        # mic in (from LiveKit)
            stt,                      # speech -> text
            aggregator.user(),        # add user turn to context
            llm,                      # reasoning
            tts,                      # text -> speech
            transport.output(),       # speaker out (to LiveKit)
            aggregator.assistant(),   # add bot turn to context
        ]
    )

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


@app.get("/gemini-models")
async def gemini_models():
    """Temporary: list the Gemini models this key can use (to see 2.5 vs 3)."""
    import urllib.request
    key = os.getenv("GOOGLE_API_KEY") or ""
    req = urllib.request.Request(
        "https://generativelanguage.googleapis.com/v1beta/models?pageSize=200",
        headers={"x-goog-api-key": key},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
        names = [m.get("name", "").replace("models/", "") for m in data.get("models", [])
                 if "generateContent" in (m.get("supportedGenerationMethods") or [])]
        return {
            "in_use": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            "flash_models": sorted(n for n in names if "flash" in n),
            "has_gemini_3": any(n.startswith("gemini-3") for n in names),
        }
    except Exception as exc:
        return {"error": str(exc)[:300]}


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
