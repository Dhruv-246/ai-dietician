"""Flask web server for the voice-to-voice AI Dietician demo.

Responsibilities (kept thin on purpose):
  - GET  /            -> serve the single-page talking-agent UI
  - GET  /api/config  -> expose the demo user_id + AI name to the frontend
  - POST /api/chat    -> hand transcribed text to the EXISTING backend pipeline
                         and return the assistant reply text

All context-building, LLM, and history logic lives in the existing backend
(src.conversation.run_turn). The frontend only does audio I/O (STT/TTS) and one
HTTP call — no backend logic is duplicated in the browser.
"""
from flask import Flask, jsonify, request, send_from_directory

from src import config
from src.conversation import run_turn

app = Flask(__name__, static_folder="static", static_url_path="")

AI_NAME = "AI Dietician"


@app.get("/")
def index():
    """Serve the talking-agent UI."""
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/config")
def get_config():
    """Frontend bootstrap data. No secrets."""
    return jsonify({"user_id": config.DEFAULT_USER_ID, "ai_name": AI_NAME})


@app.post("/api/chat")
def chat():
    """Run one conversation turn through the existing backend."""
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    user_id = (data.get("user_id") or config.DEFAULT_USER_ID).strip()

    if not message:
        return jsonify({"error": "empty message"}), 400

    try:
        reply = run_turn(user_id, message)
    except Exception as exc:  # surface a clean error to the UI, no secrets
        return jsonify({"error": str(exc)}), 502

    return jsonify({"reply": reply, "user_id": user_id})


def main() -> None:
    # host=127.0.0.1 keeps it local; localhost is a secure context so the
    # browser will grant microphone access without HTTPS.
    app.run(host="127.0.0.1", port=config.WEB_PORT, debug=False)


if __name__ == "__main__":
    main()
