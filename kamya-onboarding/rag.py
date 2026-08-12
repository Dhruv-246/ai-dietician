"""Retrieval-Augmented Generation for Mira.

A small, self-contained knowledge base of real dietician Q&A lives in Supabase
(pgvector). On each user question we embed it (Gemini), fetch the closest few
Q&A, and hand them to the LLM as REFERENCE examples — for tone, direction, and
facts — NOT as answers to copy. Mira still answers for THIS user in her own
crisp Hinglish, using the person's own profile + memory.

Design goals:
  - Never breaks a call. Any misconfig / network error -> retrieve() returns []
    and Mira just answers normally (graceful degradation).
  - No new dependencies: uses google-genai (already installed for the LLM) for
    embeddings and httpx (already installed) for Supabase's REST API.
  - One embedding model used at BOTH index and query time (dims must match the
    `vector(768)` column) — text-embedding-004 = 768 dims.

Env:
  GOOGLE_API_KEY     - embeddings (same key as the LLM)
  SUPABASE_URL       - e.g. https://xxxx.supabase.co
  SUPABASE_KEY       - the SECRET (service_role / sb_secret_...) key. The public
                       "publishable" key will NOT work while RLS is enabled.
  EMBED_MODEL        - default "text-embedding-004"
  RAG_ENABLED        - set "0" to hard-disable retrieval
  RAG_TOP_K          - default 3
  RAG_MIN_SIMILARITY - default 0.5 (cosine; drop weak matches)
"""
from __future__ import annotations

import asyncio
import os

import httpx

EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
# gemini-embedding-001 defaults to 3072 dims but supports Matryoshka truncation;
# pin to 768 to match the vector(768) column. Cosine ranking is scale-invariant,
# so truncated vectors still rank correctly.
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))
TABLE = os.getenv("RAG_TABLE", "dietician_qa")
RPC = os.getenv("RAG_RPC", "match_dietician_qa")

_client = None


def _genai_client():
    """Lazily build a google-genai client (import kept local so importing this
    module never fails even if google-genai isn't present)."""
    global _client
    if _client is None:
        import google.genai as genai
        _client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    return _client


def embed(text: str, task_type: str = "RETRIEVAL_QUERY") -> list[float]:
    """Embed one string. task_type must be RETRIEVAL_DOCUMENT when indexing the
    stored questions and RETRIEVAL_QUERY when embedding a live user question —
    this asymmetry is what makes semantic retrieval accurate."""
    from google.genai import types

    resp = _genai_client().models.embed_content(
        model=EMBED_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            task_type=task_type, output_dimensionality=EMBED_DIM
        ),
    )
    return list(resp.embeddings[0].values)


def vec_literal(emb: list[float]) -> str:
    """pgvector text literal: '[f1,f2,...]'. Used for both inserts and the RPC
    arg so Postgres casts text -> vector reliably (safer than a JSON array)."""
    return "[" + ",".join(repr(float(x)) for x in emb) + "]"


def enabled() -> bool:
    if os.getenv("RAG_ENABLED", "1") == "0":
        return False
    return bool(
        os.getenv("GOOGLE_API_KEY")
        and os.getenv("SUPABASE_URL")
        and os.getenv("SUPABASE_KEY")
    )


def _headers() -> dict:
    key = os.getenv("SUPABASE_KEY", "")
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }


async def retrieve(question: str, k: int = 3, min_similarity: float = 0.5) -> list[dict]:
    """Embed `question` and return up to k reference Q&A above min_similarity.
    Returns [] on any problem so the caller can proceed without references."""
    if not enabled() or not (question or "").strip():
        return []
    try:
        emb = await asyncio.to_thread(embed, question, "RETRIEVAL_QUERY")
        url = os.getenv("SUPABASE_URL", "").rstrip("/") + f"/rest/v1/rpc/{RPC}"
        payload = {"query_embedding": vec_literal(emb), "match_count": k}
        async with httpx.AsyncClient(timeout=8.0) as cx:
            resp = await cx.post(url, json=payload, headers=_headers())
            resp.raise_for_status()
            rows = resp.json()
        if not isinstance(rows, list):
            return []
        return [r for r in rows if float(r.get("similarity", 0) or 0) >= min_similarity]
    except Exception:
        return []


def format_reference(matches: list[dict]) -> str:
    """Render matches into a REFERENCE block for the system prompt. The framing
    is deliberate: use for style/direction, do not parrot."""
    lines = [
        "REFERENCE — how real dieticians answered questions like this. Use them "
        "ONLY for tone, direction, and correct facts. Do NOT read them back or "
        "copy their wording. Answer for THIS user, in your own short casual "
        "Hinglish, using what you already know about them:",
        "",
    ]
    for i, m in enumerate(matches, 1):
        q = (m.get("question") or "").strip()
        a = (m.get("answer") or "").strip()
        lines.append(f"{i}. Q: {q}")
        lines.append(f"   A: {a}")
    return "\n".join(lines)
