"""OpenRouter client.

Sole responsibility: send an already-built message list to OpenRouter and
return the assistant's reply text. It does NOT touch Google Sheets or build
context. The API key comes from the environment (never hardcoded).
"""
import requests

from src import config

_TIMEOUT_SECONDS = 60


def chat(messages: list[dict]) -> str:
    """Send messages to OpenRouter and return the assistant's reply text."""
    config.validate_openrouter_config()

    headers = {
        "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        # Optional attribution headers recommended by OpenRouter.
        "HTTP-Referer": "https://localhost/ai-dietician",
        "X-Title": "AI Dietician Prototype",
    }
    payload = {
        "model": config.OPENROUTER_MODEL,
        "messages": messages,
    }

    resp = requests.post(
        f"{config.OPENROUTER_BASE_URL}/chat/completions",
        headers=headers,
        json=payload,
        timeout=_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()
