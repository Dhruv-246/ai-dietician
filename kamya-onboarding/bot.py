"""Kamya Wellness — voice onboarding agent (Pipecat prototype).

A real-time voice bot that "calls" the user and runs a friendly onboarding
interview: basic info + physical specs, food preferences, current health, and
their goals for using Kamya Wellness. Then it reads back a short summary.

Pipeline:   mic → Deepgram STT → Groq (Llama-70B) → Cartesia TTS → speaker
Features:   barge-in (interruptions), Silero VAD turn-taking, streaming.

Run it with `python bot.py`, then open the local URL it prints and click
Connect. See README.md for setup. Targets the current Pipecat quickstart API
(Python 3.10+).
"""
import json
import os
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

# --- NLTK CWD-guard workaround (must wrap ALL pipecat imports) ------------- #
# Pipecat imports NLTK, whose 2026 security hook (nltk/inisec.py) blocks
# importing any package that resolves to a path INSIDE the current working
# directory. When your virtualenv lives inside the project folder (the usual
# case), that wrongly blocks NLTK's own deps (regex, defusedxml, ...) and the
# import crashes. We dodge it by importing Pipecat while the cwd is a temp dir
# (so the venv is no longer "under cwd"), then restore the real cwd.
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
    )
    from pipecat.services.cartesia.tts import CartesiaTTSService
    from pipecat.services.sarvam.stt import SarvamSTTService
    from pipecat.services.groq.llm import GroqLLMService
    from pipecat.transcriptions.language import Language
    from pipecat.transports.base_transport import BaseTransport, TransportParams

    # Dev runner: serves a browser client over SmallWebRTC
    from pipecat.runner.run import app as runner_app, main
    from pipecat.runner.types import RunnerArguments
    from pipecat.runner.utils import create_transport
    from fastapi.responses import HTMLResponse
finally:
    os.chdir(_ORIG_CWD)
# --------------------------------------------------------------------------- #


# Custom call UI (green start / red hang-up) served at /call. It connects to the
# agent's /api/offer signaling endpoint via plain browser WebRTC.
@runner_app.get("/call")
async def _call_ui():
    return HTMLResponse(Path(__file__).with_name("call_ui.html").read_text(encoding="utf-8"))

# Local dev loads a .env next to this file. On a host (Railway/Render/etc.) there
# is no .env — the platform injects the variables into the environment directly.
_ENV_PATH = Path(__file__).with_name(".env")
if _ENV_PATH.exists():
    load_dotenv(_ENV_PATH)

# Fail early with a clear message if any required key is missing/blank.
_REQUIRED = ["GROQ_API_KEY", "SARVAM_API_KEY", "CARTESIA_API_KEY"]
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
# call is for (e.g. from your DB / Google Sheet by user_id) instead of the     #
# local profile.json.                                                          #
# --------------------------------------------------------------------------- #
_PROFILE_FIELDS = ["name", "age", "gender", "height", "weight",
                   "diet", "allergies", "conditions"]


def load_profile() -> dict:
    """Load the manual-onboarding data for the user this call is for."""
    try:
        return json.loads(Path(__file__).with_name("profile.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def build_system_prompt() -> str:
    """Read call_prompt.md and fill its {{...}} variables from the profile."""
    template = Path(__file__).with_name("call_prompt.md").read_text(encoding="utf-8")
    profile = load_profile()
    for key in _PROFILE_FIELDS:
        value = str(profile.get(key, "")).strip() or "—"
        template = template.replace("{{" + key + "}}", value)
    return template


SYSTEM_PROMPT = build_system_prompt()


async def run_bot(transport: BaseTransport):
    # STT: Sarvam Saarika — built for Indian languages + Hinglish code-mixing.
    # model saarika:v2.5 auto-detects the language ("unknown" mode), so it handles
    # mixed Hindi + English natively (unlike English-first engines).
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

    # TTS: multilingual model + Hindi so the Devanagari in replies is pronounced right.
    # Use a Hindi / multilingual Cartesia voice (set CARTESIA_VOICE_ID in .env).
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        settings=CartesiaTTSService.Settings(
            voice=os.getenv("CARTESIA_VOICE_ID", "71a7ad14-091c-4e8e-a314-022ece01c121"),
            model=os.getenv("CARTESIA_MODEL", "sonic-3.5"),
            language=Language.HI,
        ),
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    context = LLMContext(messages)
    aggregator = LLMContextAggregatorPair(context)

    pipeline = Pipeline(
        [
            transport.input(),        # mic in
            stt,                      # speech -> text
            aggregator.user(),        # add user turn to context
            llm,                      # reasoning
            tts,                      # text -> speech
            transport.output(),       # speaker out
            aggregator.assistant(),   # add bot turn to context
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,   # <-- barge-in: user can cut the bot off
            enable_metrics=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        # Make the bot speak first: run the LLM once against the system prompt,
        # which instructs it to greet and begin onboarding.
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        await task.cancel()

    await PipelineRunner().run(task)


# --------------------------------------------------------------------------- #
# Entry point — the dev runner creates the transport and serves a web client. #
# --------------------------------------------------------------------------- #
async def bot(runner_args: RunnerArguments):
    transport = await create_transport(
        runner_args,
        {
            "webrtc": lambda: TransportParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),  # detects speech / silence for turns
            ),
            "daily": lambda: TransportParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
            ),
        },
    )
    await run_bot(transport)


if __name__ == "__main__":
    main()
