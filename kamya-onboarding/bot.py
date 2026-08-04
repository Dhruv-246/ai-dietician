"""Kamya Wellness — voice onboarding agent (Pipecat + LiveKit).

A real-time voice bot that "calls" the user and runs a friendly onboarding
interview in Hinglish. Media is carried by LiveKit Cloud (a managed WebRTC
media server), so the call connects reliably from any host — including
Railway, which cannot do peer-to-peer WebRTC.

Pipeline:   mic → Sarvam STT → Groq (Llama-70B) → Cartesia TTS → speaker
Features:   barge-in (interruptions), Silero VAD turn-taking, streaming.

How it works:
  • The browser opens "/" (the green-start / red-hangup call screen).
  • Clicking the green button POSTs /connect. The server creates a unique
    LiveKit room, launches Mira (this bot) into it, and returns a join token.
  • The browser joins the same room with the LiveKit JS SDK. LiveKit relays
    the audio both ways, so no direct UDP path to the server is needed.

Run locally:  python bot.py  → open the printed URL.
Requires: GROQ_API_KEY, SARVAM_API_KEY, CARTESIA_API_KEY and
          LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET.
"""
import argparse
import asyncio
import json
import os
import sys
import tempfile
import uuid
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
    from pipecat.frames.frames import LLMRunFrame
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.runner import PipelineRunner
    from pipecat.pipeline.task import PipelineParams, PipelineTask
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.processors.aggregators.llm_response_universal import (
        LLMContextAggregatorPair,
        LLMUserAggregatorParams,
    )
    from pipecat.services.cartesia.tts import CartesiaTTSService
    from pipecat.services.sarvam.stt import SarvamSTTService
    from pipecat.services.groq.llm import GroqLLMService
    from pipecat.transcriptions.language import Language
    from pipecat.transports.livekit.transport import LiveKitParams, LiveKitTransport
    from pipecat.runner.livekit import generate_token, generate_token_with_agent
finally:
    os.chdir(_ORIG_CWD)
# --------------------------------------------------------------------------- #

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse

# Local dev loads a .env next to this file. On a host (Railway/Render/etc.) there
# is no .env — the platform injects the variables into the environment directly.
_ENV_PATH = Path(__file__).with_name(".env")
if _ENV_PATH.exists():
    load_dotenv(_ENV_PATH)

# Fail early with a clear message if any required key is missing/blank.
_REQUIRED = [
    "GROQ_API_KEY", "SARVAM_API_KEY", "CARTESIA_API_KEY",
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


def build_system_prompt(profile: dict | None = None) -> str:
    """Read call_prompt.md and fill its {{...}} variables from the profile."""
    template = Path(__file__).with_name("call_prompt.md").read_text(encoding="utf-8")
    profile = profile or {}
    for key in _PROFILE_FIELDS:
        value = str(profile.get(key, "")).strip() or "—"
        template = template.replace("{{" + key + "}}", value)
    return template


# --------------------------------------------------------------------------- #
# The voice pipeline for a single call, joined to one LiveKit room.            #
# --------------------------------------------------------------------------- #
async def run_livekit_bot(room_name: str, system_prompt: str):
    """Join `room_name` as Mira and run the onboarding conversation."""
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

    # STT: Sarvam Saarika — built for Indian languages + Hinglish code-mixing.
    stt = SarvamSTTService(
        api_key=os.getenv("SARVAM_API_KEY"),
        model=os.getenv("STT_MODEL", "saarika:v2.5"),
    )

    # Groq's "versatile" 70B model for reasoning (fast, cheap, good enough here).
    llm = GroqLLMService(
        api_key=os.getenv("GROQ_API_KEY"),
        settings=GroqLLMService.Settings(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        ),
    )

    # TTS: multilingual + Hindi so Devanagari in replies is pronounced right.
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        settings=CartesiaTTSService.Settings(
            voice=os.getenv("CARTESIA_VOICE_ID", "71a7ad14-091c-4e8e-a314-022ece01c121"),
            model=os.getenv("CARTESIA_MODEL", "sonic-3.5"),
            language=Language.HI,
        ),
    )

    # Barge-in (interruptions) in Pipecat 1.7.0 is driven by the user aggregator's
    # VAD controller — the VAD analyzer MUST be passed here (not to the transport).
    # When the VAD detects the user starting to speak while Mira is talking, the
    # turn controller fires an interruption that stops her TTS immediately.
    context = LLMContext([{"role": "system", "content": system_prompt}])
    aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
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
    )

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant_id):
        # Make Mira speak first: run the LLM once so she greets and begins.
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_participant_disconnected")
    async def on_participant_disconnected(transport, participant_id):
        # User hung up (red button) → end the call and free resources.
        await task.cancel()

    # handle_sigint=False: this runs as a background task inside the web
    # server's event loop, so it must NOT try to install process signal handlers.
    await PipelineRunner(handle_sigint=False).run(task)


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

    # Build the prompt for THIS user (Sheet lookup by uid; falls back to sample).
    profile = load_profile_for_call(firebase_uid)
    system_prompt = build_system_prompt(profile)

    room_name = f"mira-{uuid.uuid4().hex[:10]}"
    user_token = generate_token(room_name, "user", key, secret)

    # Launch Mira into the room; she waits for the user, then greets by name.
    asyncio.create_task(run_livekit_bot(room_name, system_prompt))

    return {"url": url, "token": user_token, "room": room_name, "name": profile.get("name", "")}


def main():
    parser = argparse.ArgumentParser(description="Kamya Wellness onboarding call")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "7860")))
    args = parser.parse_args()
    print(f"🚀 Mira onboarding call ready → http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
