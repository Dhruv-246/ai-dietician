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

async def probe(rounds: int = 3) -> dict:
    """Time the SMALLEST possible Bedrock call, a few times.

    WHY THIS EXISTS. A live call showed 4s to first token, held to within ±5%
    across 34 samples while the prompt grew steadily through the call. Flat
    latency under a growing prompt rules out prefill, and 44ms from first
    token to first audio rules out the pipeline. What is left is fixed
    per-request overhead -- but "overhead" is a symptom, not a cause, and it
    could be cross-region routing, auth, or cold start, which have different
    fixes.

    So: send a request with a trivial prompt and a 5-token ceiling. Generation
    and prefill are then negligible by construction, and whatever time remains
    is the floor cost of reaching this model from this region.

      floor ~= 4s   -> the overhead is the round trip itself. Prompt work will
                       not help; the inference profile or region is the lever.
      floor << 4s   -> the round trip is fine and the cost is in handling our
                       actual prompt after all.

    First call in a process also pays TLS and client setup, so rounds are
    reported separately -- a large first value that then drops is warm-up,
    not steady-state.
    """
    import time
    if provider() != "bedrock":
        return {"skipped": "provider is not bedrock"}
    model = bedrock_model("chat")
    if not model:
        return {"skipped": "no bedrock model configured"}

    out = []
    for _ in range(max(1, rounds)):
        t0 = time.perf_counter()
        ok, err = True, ""
        try:
            import httpx
            body = anthropic_body("Reply with the single word ok.",
                                  [{"role": "user", "content": "ping"}],
                                  max_tokens=5, temperature=0.0)
            async with httpx.AsyncClient(timeout=30.0) as c:
                r = await c.post(_bedrock_url(model), headers=bedrock_headers(),
                                 json=body)
                r.raise_for_status()
                r.json()
        except Exception as exc:
            ok, err = False, f"{type(exc).__name__}: {exc}"[:200]
        out.append({"ms": round((time.perf_counter() - t0) * 1000), "ok": ok,
                    **({"error": err} if err else {})})
    return {"model": model, "region": _region(), "rounds": out}

async def count_tokens(system: str, user: str = "ping") -> dict:
    """Ask Bedrock how many INPUT TOKENS a given prompt actually costs.

    Decision-critical for prompt caching: Claude Haiku 4.5 will not create a
    cache checkpoint for a prefix under 4,096 tokens. Below that the request
    still succeeds and simply is not cached, silently. Character counts cannot
    settle this -- Devanagari tokenises far more expensively than Latin, and
    this prompt is a mix -- so ask the model rather than estimate.

    Sends max_tokens=1, so generation cost is nil and only the prefill is
    measured.
    """
    if provider() != "bedrock":
        return {"skipped": "provider is not bedrock"}
    model = bedrock_model("chat")
    if not model:
        return {"skipped": "no bedrock model configured"}
    try:
        import httpx
        body = anthropic_body(system, [{"role": "user", "content": user}],
                              max_tokens=1, temperature=0.0)
        async with httpx.AsyncClient(timeout=30.0) as c:
            r = await c.post(_bedrock_url(model), headers=bedrock_headers(),
                             json=body)
            r.raise_for_status()
            usage = (r.json() or {}).get("usage", {}) or {}
        return {"chars": len(system), "input_tokens": usage.get("input_tokens"),
                "chars_per_token": (round(len(system) / usage["input_tokens"], 2)
                                    if usage.get("input_tokens") else None)}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"[:200]}

async def _ttft_stream(model: str, system: str, cache: bool, max_tokens: int):
    """Time to FIRST BYTE from the streaming endpoint.

    The non-streaming probe measures a whole request; a live call measures
    time to first token, and generates ~40-60 tokens rather than 4. Those are
    different quantities, so a conclusion drawn from one does not transfer to
    the other. This uses invoke-with-response-stream and stops at the first
    byte of the body, which is the closest analogue to what the pipeline logs
    as `llm first token`.
    """
    import time
    import httpx
    sysblock = [{"type": "text", "text": system}]
    if cache:
        sysblock[0]["cache_control"] = {"type": "ephemeral"}
    body = {"anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens, "temperature": 0.0,
            "system": sysblock,
            "messages": [{"role": "user", "content":
                          "Ek chhota sawaal poochho."}]}
    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=60.0) as c:
        async with c.stream("POST", _bedrock_url(model, stream=True),
                            headers=bedrock_headers(), json=body) as r:
            r.raise_for_status()
            async for _chunk in r.aiter_bytes():
                return round((time.perf_counter() - t0) * 1000)
    return None


async def compare_models(models, system: str, rounds: int = 2) -> list:
    """Time the SAME real prompt on several models, with caching enabled.

    Answers three questions at once, per model:
      - time to first byte with a realistic prefill (not a toy ping)
      - whether a cache checkpoint is actually created (cache_write > 0)
      - whether the SECOND identical call reads it back (cache_read > 0)

    That last one is the whole question for Sonnet: Haiku 4.5 needs a
    4,096-token prefix to cache and this prompt measures 2,748, so it can
    never cache. Sonnet 4.5/4.6 need only 1,024, so the same prompt should
    cache -- but "should" has been wrong repeatedly here, so measure it.

    max_tokens=4: generation is negligible, so what is timed is round trip
    plus prefill, which is exactly the cost caching targets.
    """
    import time
    out = []
    for model in models:
        rows = []
        for i in range(rounds):
            t0 = time.perf_counter()
            try:
                import httpx
                body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 4,
                    "temperature": 0.0,
                    # cache_control on the system block is what creates the
                    # checkpoint. Under the minimum it is silently ignored.
                    "system": [{"type": "text", "text": system,
                                "cache_control": {"type": "ephemeral"}}],
                    "messages": [{"role": "user", "content": "ping"}],
                }
                async with httpx.AsyncClient(timeout=60.0) as c:
                    r = await c.post(_bedrock_url(model), headers=bedrock_headers(),
                                     json=body)
                    r.raise_for_status()
                    u = (r.json() or {}).get("usage", {}) or {}
                rows.append({
                    "call": i + 1,
                    "ms": round((time.perf_counter() - t0) * 1000),
                    "in": u.get("input_tokens"),
                    "cache_write": u.get("cache_creation_input_tokens"),
                    "cache_read": u.get("cache_read_input_tokens"),
                })
            except Exception as exc:
                rows.append({"call": i + 1,
                             "error": f"{type(exc).__name__}: {exc}"[:160]})
                break

        # Streaming time-to-first-byte, with and without caching, at a
        # realistic output length. Several rounds because a two-sample
        # comparison cannot survive this endpoint's variance -- an earlier
        # run of it spread 1671-3178ms on one model.
        ttft = {}
        for label, cache in (("nocache", False), ("cached", True)):
            vals = []
            for _ in range(5):
                try:
                    v = await _ttft_stream(model, system, cache, 64)
                    if v:
                        vals.append(v)
                except Exception as exc:
                    vals.append(f"{type(exc).__name__}")
                    break
            nums = sorted(v for v in vals if isinstance(v, int))
            ttft[label] = {"all": vals,
                           "median": nums[len(nums) // 2] if nums else None}
        out.append({"model": model, "rounds": rows, "ttft_stream_ms": ttft})
    return out
