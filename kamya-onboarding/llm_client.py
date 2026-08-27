"""One place that decides WHICH LLM provider runs, and how JSON is asked for.

Three jobs in this codebase need a small JSON answer — the P-3 router, the P-2
node-completion check, and post-call consolidation. They used to each build
their own Groq client. Now they share this, so switching provider is one env
var rather than four edits.

WHY THIS ISN'T A ONE-LINE SWAP.
    Groq is OpenAI-compatible and supports response_format={"type":"json_object"},
    which guarantees parseable output. Bedrock's Converse API has no equivalent.
    The reliable equivalent for Claude is to PREFILL the assistant turn with "{"
    — the model then has no way to open with prose, and we prepend the brace
    back before parsing. That trick lives here so no caller has to know it.

PROVIDER SELECTION.
    LLM_PROVIDER=bedrock|groq, defaulting to bedrock when AWS credentials are
    present and groq otherwise. That means a missing/incorrect AWS setup falls
    back to the working path instead of taking the product down.
"""
import asyncio
import json
import os
import re


def provider() -> str:
    explicit = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    if explicit in ("bedrock", "groq"):
        return explicit
    if os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"):
        return "bedrock"
    return "groq"


def bedrock_model(kind: str = "chat") -> str:
    """Model id for a job. One variable covers everything; override per job only
    if you actually need to."""
    base = os.getenv("BEDROCK_MODEL", "").strip()
    override = {
        "chat": os.getenv("BEDROCK_CHAT_MODEL", ""),
        "fast": os.getenv("BEDROCK_FAST_MODEL", ""),      # router, node check
        "heavy": os.getenv("BEDROCK_CONSOLIDATION_MODEL", ""),  # consolidation
    }.get(kind, "").strip()
    return override or base


def _region() -> str:
    return (os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
            or "us-east-1").strip()


def _extract_json(text: str) -> dict:
    """Parse a model reply that should be one JSON object.

    Tolerates the two things models do anyway: fencing it in ```json, and
    adding a sentence before or after.
    """
    s = (text or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    try:
        return json.loads(s)
    except (ValueError, TypeError):
        pass
    start, depth = s.find("{"), 0
    if start < 0:
        return {}
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(s[start:i + 1])
                except (ValueError, TypeError):
                    return {}
    return {}


def _bedrock_json_sync(system, user, model, max_tokens, temperature):
    """Blocking Bedrock Converse call that returns a dict. Runs off the loop."""
    import boto3
    client = boto3.client("bedrock-runtime", region_name=_region())
    resp = client.converse(
        modelId=model,
        system=[{"text": system}],
        messages=[
            {"role": "user", "content": [{"text": user}]},
            # Prefill: the reply can only continue an object that has already
            # been opened, so it cannot start with prose. Bedrock's Converse
            # API has no response_format, and this is the reliable substitute.
            {"role": "assistant", "content": [{"text": "{"}]},
        ],
        inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
    )
    parts = resp.get("output", {}).get("message", {}).get("content", [])
    body = "".join(p.get("text", "") for p in parts)
    return _extract_json("{" + body)


async def _groq_json(system, user, model, max_tokens, temperature):
    from groq import AsyncGroq
    client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        reasoning_effort=os.getenv("LLM_REASONING_EFFORT", "low"),
    )
    return _extract_json(resp.choices[0].message.content or "{}")


async def complete_json(system: str, user: str, *, kind: str = "fast",
                        max_tokens: int = 700, temperature: float = 0.1,
                        timeout: float = 6.0, groq_model: str = "") -> dict:
    """Ask for one JSON object. Returns {} on any failure — never raises.

    Callers already treat {} as "no decision" and fall back to safe defaults,
    so a provider outage degrades behaviour instead of breaking a live call.
    """
    if provider() == "bedrock":
        model = bedrock_model(kind)
        if model:
            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(_bedrock_json_sync, system, user, model,
                                      max_tokens, temperature),
                    timeout=timeout)
            except Exception as exc:
                print(f"[llm_client] bedrock {kind} failed: "
                      f"{type(exc).__name__}: {exc}", flush=True)
                return {}
    try:
        return await asyncio.wait_for(
            _groq_json(system, user,
                       groq_model or os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
                       max_tokens, temperature),
            timeout=timeout)
    except Exception as exc:
        print(f"[llm_client] groq {kind} failed: {type(exc).__name__}: {exc}",
              flush=True)
        return {}
