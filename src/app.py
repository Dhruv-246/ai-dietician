"""CLI entry point for the AI Dietician prototype.

Flow per turn:
    user message
      -> build context (profile + recent history + relevant food data)
      -> send to OpenRouter
      -> receive AI response
      -> save user message + assistant response to conversation_history
      -> display the response

Text-only. No voice, no web frontend.

Usage:
    python -m src.app --user-id U001
    python -m src.app --user-id U001 --message "How much protein is in 2 rotis?"
"""
import argparse
import sys

from src.conversation import run_turn


def handle_turn(user_id: str, message: str) -> str:
    """Run one full turn and return the assistant's reply.

    Thin wrapper over the shared conversation logic so CLI and web share it.
    """
    return run_turn(user_id, message)


def _run_single(user_id: str, message: str) -> None:
    reply = handle_turn(user_id, message)
    print(f"\nDietician: {reply}\n")


def _run_interactive(user_id: str) -> None:
    print("AI Dietician (prototype). Type 'exit' or 'quit' to leave.\n")
    while True:
        try:
            message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if message.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        if not message:
            continue
        try:
            reply = handle_turn(user_id, message)
        except Exception as exc:  # keep the loop alive on transient errors
            print(f"[error] {exc}\n")
            continue
        print(f"\nDietician: {reply}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Dietician CLI prototype")
    parser.add_argument("--user-id", required=True, help="user_id in the users tab")
    parser.add_argument(
        "--message",
        help="Send a single message and exit. Omit for interactive chat.",
    )
    args = parser.parse_args()

    if args.message:
        _run_single(args.user_id, args.message)
    else:
        _run_interactive(args.user_id)


if __name__ == "__main__":
    sys.exit(main())
