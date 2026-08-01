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

    # Surface OpenRouter HTTP errors with their body (e.g. invalid key, 402, 429).
    if resp.status_code >= 400:
        raise RuntimeError(
            f"OpenRouter API error {resp.status_code}: {resp.text[:300]}"
        )

    try:
        data = resp.json()
    except ValueError as exc:  # non-JSON body
        raise RuntimeError(
            f"OpenRouter returned a non-JSON response (status {resp.status_code}): "
            f"{resp.text[:200]}"
        ) from exc

    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            f"Unexpected OpenRouter response shape: {str(data)[:300]}"
        ) from exc
