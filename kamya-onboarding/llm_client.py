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
from __future__ import annotations

import asyncio
import json
import os
import re


def bedrock_configured() -> bool:
    """True if Bedrock can authenticate.

    Two shapes are accepted. A Bedrock API KEY is a single bearer token in
    AWS_BEARER_TOKEN_BEDROCK, which boto3 picks up on its own for
    bedrock-runtime — no access key or secret involved. The older shape is the
    usual access-key/secret pair.
    """
    return bool(os.getenv("AWS_BEARER_TOKEN_BEDROCK")
                or (os.getenv("AWS_ACCESS_KEY_ID")
                    and os.getenv("AWS_SECRET_ACCESS_KEY")))


def fast_provider() -> str:
    """Which provider serves the SMALL JSON jobs -- the P-3 router and the P-2
    node check. Deliberately allowed to differ from the conversation model.

    Measured on the 2026-08-29 call: every Bedrock request carried ~4s of
    fixed overhead before its first token (34 samples, 3847-4217ms -- a spread
    of ±5%, which is routing cost, not generation). The node check is a second
    such request, and it runs BEFORE generation starts, so the user waited
    ~3s for it and then ~4s for the reply.

    These jobs emit a small JSON object that the user never sees or hears.
    Nothing about them needs Claude's Hinglish prose quality -- they are
    classifiers. Groq answers them in a fraction of the time, so the default
    is Groq whenever a key is present, and the conversation model stays on
    Claude where the quality actually matters.

    Force with LLM_FAST_PROVIDER=bedrock|groq.
    """
    explicit = (os.getenv("LLM_FAST_PROVIDER") or "").strip().lower()
    if explicit in ("bedrock", "groq"):
        return explicit
    if os.getenv("GROQ_API_KEY"):
        return "groq"
    return provider()


def provider() -> str:
    explicit = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    if explicit in ("bedrock", "groq"):
        return explicit
    return "bedrock" if bedrock_configured() else "groq"


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


BEDROCK_TIMEOUT = float(os.getenv("BEDROCK_HTTP_TIMEOUT", "30"))


def _bedrock_url(model: str, stream: bool = False) -> str:
    verb = "invoke-with-response-stream" if stream else "invoke"
    return (f"https://bedrock-runtime.{_region()}.amazonaws.com"
            f"/model/{model}/{verb}")


def bedrock_headers() -> dict:
    """Bearer auth. A Bedrock API key is a plain token — no SigV4 signing, no
    boto3 credential chain, which is why this uses httpx directly."""
    return {
        "Authorization": f"Bearer {os.getenv('AWS_BEARER_TOKEN_BEDROCK', '')}",
        "Content-Type": "application/json",
    }


def anthropic_body(system: str, messages: list, max_tokens: int,
                   temperature: float, stop: list | None = None,
                   prefill: str | None = None) -> dict:
    """Anthropic's native Bedrock body. `system` is a TOP-LEVEL field here, not
    a message — putting it in `messages` is silently ignored."""
    msgs = list(messages)
    if prefill:
        msgs = msgs + [{"role": "assistant", "content": prefill}]
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": msgs,
    }
    if system:
        body["system"] = system
    if stop:
        body["stop_sequences"] = stop
    return body


async def _bedrock_json(system, user, model, max_tokens, temperature):
    import httpx
    body = anthropic_body(
        system, [{"role": "user", "content": user}], max_tokens, temperature,
        # Prefill: the reply can only continue an object that is already open,
        # so it cannot begin with prose. Bedrock has no response_format, and
        # this is the reliable substitute for Claude.
        prefill="{")
    async with httpx.AsyncClient(timeout=BEDROCK_TIMEOUT) as client:
        r = await client.post(_bedrock_url(model), headers=bedrock_headers(),
                              json=body)
        r.raise_for_status()
        data = r.json()
    text = "".join(b.get("text", "") for b in data.get("content", []))
    return _extract_json("{" + text)


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
    # "fast" jobs may run on a different provider from the conversation --
    # see fast_provider(). "heavy" (consolidation) runs after hangup, where
    # latency is irrelevant and schema adherence is everything, so it follows
    # the main provider.
    chosen = fast_provider() if kind == "fast" else provider()
    if chosen == "bedrock":
        model = bedrock_model(kind)
        if model:
            try:
                return await asyncio.wait_for(
                    _bedrock_json(system, user, model, max_tokens, temperature),
                    timeout=timeout)
            except Exception as exc:
                # Fall THROUGH to Groq rather than returning {}. A dropped
                # decision costs a live turn -- the router loses its lane, the
                # node check loses its evidence. Groq is already configured and
                # is a better answer than no answer. Consolidation retries
                # anyway, so the extra call there is harmless.
                print(f"[llm_client] bedrock {kind} failed: "
                      f"{type(exc).__name__}: {exc} -- falling back to groq",
                      flush=True)
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
