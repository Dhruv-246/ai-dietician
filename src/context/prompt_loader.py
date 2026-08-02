"""Loads the system prompt from disk at runtime.

The system prompt lives in prompts/system_prompt.md and is NEVER hardcoded.
This lets the prompt be edited without touching application code.
"""
from src import config


def load_system_prompt() -> str:
    """Read and return the full (base) system prompt text."""
    path = config.SYSTEM_PROMPT_PATH
    if not path.exists():
        raise FileNotFoundError(f"System prompt not found at {path}")
    return path.read_text(encoding="utf-8").strip()


def load_voice_overlay() -> str:
    """Read and return the voice-only overlay (Devanagari mixed-script rule)."""
    path = config.VOICE_OVERLAY_PATH
    if not path.exists():
        raise FileNotFoundError(f"Voice overlay not found at {path}")
    return path.read_text(encoding="utf-8").strip()
