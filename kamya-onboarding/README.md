# Kamya Wellness — Voice Onboarding Agent (Pipecat prototype)

A real-time **voice** bot that runs a friendly onboarding interview and collects:
basics + physical specs, food preferences, current health, and the user's goals for
Kamya Wellness — then reads back a short summary.

**Stack:** Deepgram (STT) → Groq Llama-70B (reasoning) → Cartesia (TTS), over
WebRTC, with **barge-in** (interruptions) and Silero VAD turn-taking.

---

## Requirements
- **Python 3.10+** — your default `python3` is 3.9, which pulls an ancient Pipecat.
  Use 3.11 explicitly (Homebrew: `/opt/homebrew/bin/python3.11`).
- API keys: **Groq**, **Deepgram**, **Cartesia** (all have free tiers)

## Setup
```bash
cd kamya-onboarding
python3.11 -m venv .venv && source .venv/bin/activate    # MUST be 3.10+, not 3.9
pip install -r requirements.txt

cp .env.example .env        # then paste your API keys into .env
```
(The venv in this folder is already built with 3.11 and these deps installed, so you
can skip straight to `cp .env.example .env` and `python bot.py` if you like.)

Get keys: Groq → console.groq.com · Deepgram → console.deepgram.com · Cartesia → play.cartesia.ai

## Run
```bash
python bot.py
```
It starts a local server and prints a URL (the SmallWebRTC dev client, usually
`http://localhost:7860`). Open it in Chrome, click **Connect**, allow the mic, and
Mira from Kamya Wellness will greet you and start the onboarding. Talk naturally —
you can **interrupt** her any time.

## What it does
- Speaks first, then asks one question at a time across five areas (basics, physical,
  food, health, goals), acknowledging each answer.
- Voice-tuned: short spoken replies, natural turn-taking, barge-in enabled.
- Ends with a spoken summary of what it captured.
- Does **not** give medical advice — onboarding only.

## Language: Hinglish (default)
The bot speaks **Hinglish** — Hindi words in Devanagari + English words, e.g.
*"अच्छा! और आप किस city में रहते हैं?"* This needs:
- `STT_LANGUAGE=multi` (Deepgram transcribes Hindi + English together) — already set.
- A **Hindi / multilingual Cartesia voice**: set `CARTESIA_VOICE_ID` to one from the
  Cartesia dashboard (the default is English and will sound off). `CARTESIA_MODEL=sonic-2`.
- The `SYSTEM_PROMPT` in `bot.py` instructs Hinglish output in Devanagari.

## Tuning
- **Model:** `GROQ_MODEL` (default `openai/gpt-oss-120b`).
- **More Hindi / less English:** set `STT_LANGUAGE=hi`.
- **The script/persona:** edit `SYSTEM_PROMPT` in `bot.py`.

## Note: venv-inside-project + NLTK
Pipecat imports NLTK, whose new security hook blocks imports that resolve *inside* the
current directory — which wrongly catches your own venv when it lives in the project
folder. `bot.py` handles this automatically (it imports Pipecat with the cwd temporarily
set to a temp dir). No action needed.

## Next steps (not in this prototype)
- Persist the collected profile (e.g. write to a DB/Sheet) via an LLM function tool
  called at the end of the call.
- Phone calls instead of browser: swap the transport for Daily + Twilio.

---

### Notes
- This scaffold targets Pipecat's **current** quickstart API
  (https://docs.pipecat.ai/getting-started/quickstart). If your installed version's
  imports differ, the **pipeline logic in `bot.py` is version-stable** — only the
  transport bootstrap (`create_transport` / `main` in `pipecat.runner`) may need to be
  aligned with that page.
- It was scaffolded on a Python 3.9 machine, so it was **not executed here** — run the
  steps above in a 3.10+ venv with your keys.
