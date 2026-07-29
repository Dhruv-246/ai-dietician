"""Shared conversation logic.

One place that runs a full turn so the CLI and the web server use the exact
same pipeline (no duplication). This does NOT change any existing
context-building or LLM behaviour — it just wraps the existing components.

Flow:
    build context (profile + recent history + relevant food data + system prompt)
      -> OpenRouter
      -> save user message + assistant response to conversation_history
      -> return the assistant reply
"""
from src.context import context_builder
from src.data import repositories
from src.llm import openrouter_client


def run_turn(user_id: str, message: str) -> str:
    """Run one full turn and return the assistant's reply."""
    # 1. Build the context (existing, unchanged logic).
    messages = context_builder.build_context(user_id, message)

    # 2. Ask the LLM.
    reply = openrouter_client.chat(messages)

    # 3. Persist both sides of the exchange.
    repositories.append_message(user_id, "user", message)
    repositories.append_message(user_id, "assistant", reply)

    return reply
