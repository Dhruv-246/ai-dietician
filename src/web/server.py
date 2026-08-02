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
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token as google_id_token

from src import config
from src.context import memory_extractor
from src.conversation import run_turn
from src.data import repositories

app = Flask(__name__, static_folder="static", static_url_path="")

AI_NAME = "Mira"


@app.get("/")
def index():
    """Serve the talking-agent UI (protected client-side by the auth guard)."""
    return send_from_directory(app.static_folder, "index.html")


@app.get("/login")
def login_page():
    """Serve the login page."""
    return send_from_directory(app.static_folder, "login.html")


@app.get("/signup")
def signup_page():
    """Serve the signup page."""
    return send_from_directory(app.static_folder, "signup.html")


@app.get("/onboarding")
def onboarding_page():
    """Serve the onboarding wizard (protected client-side by the auth guard)."""
    return send_from_directory(app.static_folder, "onboarding.html")


def _verify_firebase_request():
    """Verify the Firebase ID token on the request.

    Returns (claims, None) on success or (None, (response, status)) on failure.
    """
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return None, (jsonify({"error": "missing bearer token"}), 401)
    token = auth_header.split(" ", 1)[1].strip()

    project_id = config.FIREBASE_CONFIG.get("projectId")
    if not project_id:
        return None, (jsonify({"error": "Firebase is not configured on the server"}), 500)

    try:
        claims = google_id_token.verify_firebase_token(
            token, google_requests.Request(), audience=project_id,
            clock_skew_in_seconds=10,
        )
    except Exception as exc:
        return None, (jsonify({"error": f"invalid token: {exc}"}), 401)
    if not claims:
        return None, (jsonify({"error": "invalid token"}), 401)
    return claims, None


@app.get("/api/config")
def get_config():
    """Frontend bootstrap data. No secrets."""
    return jsonify({"user_id": config.DEFAULT_USER_ID, "ai_name": AI_NAME})


@app.get("/api/firebase-config")
def firebase_config():
    """Firebase web config for the client SDK.

    These values are public by design (the Firebase web SDK exposes them in the
    browser). Only non-empty fields are returned; if unset, the frontend shows a
    "not configured" message.
    """
    cfg = {k: v for k, v in config.FIREBASE_CONFIG.items() if v}
    return jsonify(cfg)


@app.post("/api/user/sync")
def user_sync():
    """Ensure the signed-in Firebase user has a row in the Users sheet.

    The client sends its Firebase ID token as `Authorization: Bearer <token>`.
    We verify the token with google-auth (so firebase_uid/email are trusted,
    not client-supplied), then get-or-create the Users row. Idempotent: existing
    users never get a duplicate row.
    """
    claims, err = _verify_firebase_request()
    if err:
        return err

    firebase_uid = claims.get("sub") or claims.get("user_id")
    email = claims.get("email")
    if not firebase_uid:
        return jsonify({"error": "token missing uid"}), 401

    try:
        result = repositories.ensure_user(firebase_uid, email)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 502

    return jsonify(result)


@app.post("/api/user/profile")
def user_profile():
    """Save onboarding profile fields for the signed-in user (one screen at a time).

    Body: {"fields": {...}}. The token identifies the user; only whitelisted
    profile columns are written (see repositories.ALLOWED_PROFILE_FIELDS).
    """
    claims, err = _verify_firebase_request()
    if err:
        return err

    firebase_uid = claims.get("sub") or claims.get("user_id")
    if not firebase_uid:
        return jsonify({"error": "token missing uid"}), 401

    data = request.get_json(silent=True) or {}
    fields = data.get("fields")
    if not isinstance(fields, dict):
        fields = {k: v for k, v in data.items() if k != "fields"}
    if not fields:
        return jsonify({"error": "no fields provided"}), 400

    try:
        repositories.update_user(firebase_uid, fields)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 502

    return jsonify({"ok": True})


@app.post("/api/chat")
def chat():
    """Run one conversation turn for the AUTHENTICATED user.

    The user is identified from the verified Firebase token (via firebase_uid),
    NOT from any client-supplied user_id — so each user only ever reads and
    writes their own row and their own conversation history.
    """
    claims, err = _verify_firebase_request()
    if err:
        return err

    firebase_uid = claims.get("sub") or claims.get("user_id")
    if not firebase_uid:
        return jsonify({"error": "token missing uid"}), 401

    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"error": "empty message"}), 400

    # Resolve the user's own row by firebase_uid; derive their user_id.
    user = repositories.get_user_by_firebase_uid(firebase_uid)
    if not user:
        return jsonify({"error": "no user row; complete signup first"}), 404
    user_id = str(user.get("user_id", "")).strip()
    if not user_id:
        return jsonify({"error": "user row missing user_id"}), 500

    try:
        reply = run_turn(user_id, message)
    except Exception as exc:  # surface a clean error to the UI, no secrets
        return jsonify({"error": str(exc)}), 502

    return jsonify({"reply": reply, "user_id": user_id})


@app.post("/api/memory/extract")
def memory_extract():
    """Extract durable long-term memories from the user's latest exchange.

    Called by the frontend AFTER a chat reply (background, non-blocking) so chat
    latency is unchanged. Reads the last exchange from the user's own history,
    runs LLM extraction for STABLE facts, and upserts (dedup by user_id + key).
    """
    claims, err = _verify_firebase_request()
    if err:
        return err
    firebase_uid = claims.get("sub") or claims.get("user_id")
    if not firebase_uid:
        return jsonify({"error": "token missing uid"}), 401

    user = repositories.get_user_by_firebase_uid(firebase_uid)
    if not user:
        return jsonify({"error": "no user row"}), 404
    user_id = str(user.get("user_id", "")).strip()

    # Build the latest exchange text from this user's own recent history.
    recent = repositories.get_recent_history(user_id, 4)
    if not recent:
        return jsonify({"stored": 0})
    exchange = "\n".join(
        f"{str(r.get('role','')).capitalize()}: {str(r.get('message','')).strip()}"
        for r in recent if str(r.get("message", "")).strip()
    )
    last_user_msg = next(
        (str(r.get("message", "")).strip() for r in reversed(recent)
         if str(r.get("role", "")).lower() == "user"), ""
    )
    existing = [str(m.get("key", "")).strip() for m in repositories.get_memories(user_id)]

    try:
        facts = memory_extractor.extract(exchange, existing)
        stored = 0
        for f in facts:
            repositories.upsert_memory(
                user_id, f["category"], f["key"], f["value"], last_user_msg
            )
            stored += 1
    except Exception as exc:
        return jsonify({"error": str(exc)}), 502

    return jsonify({"stored": stored})


def main() -> None:
    # host=127.0.0.1 keeps it local; localhost is a secure context so the
    # browser will grant microphone access without HTTPS.
    app.run(host="127.0.0.1", port=config.WEB_PORT, debug=False)


if __name__ == "__main__":
    main()
