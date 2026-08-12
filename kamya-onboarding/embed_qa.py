"""One-time (re-runnable) loader: embed the dietician Q&A CSV and upload it to
Supabase (pgvector).

Usage:
    # needs GOOGLE_API_KEY, SUPABASE_URL, SUPABASE_KEY (the SECRET key) in env
    # or in kamya-onboarding/.env
    python embed_qa.py /path/to/mira_nutrition_qa_dataset.csv
    python embed_qa.py file.csv --replace     # wipe the table first (full reload)

CSV columns: question, answer, category (source_page etc. are ignored).

We embed the QUESTION with task_type=RETRIEVAL_DOCUMENT so live user questions
(embedded as RETRIEVAL_QUERY) match against them accurately. The row keeps the
full answer so retrieval can hand both Q and A to the LLM as reference.
"""
from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

import re

import httpx

# load kamya-onboarding/.env if present (so secrets don't need to be exported)
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).with_name(".env"))
except Exception:
    pass

import rag  # noqa: E402  (reuses embed() + vec_literal())

BATCH = 25
SLEEP = float(os.getenv("RAG_EMBED_SLEEP", "0.7"))  # spacing to stay under free-tier RPM


def _fatal(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def _embed_retry(text: str, tries: int = 6) -> list[float]:
    """Embed with backoff on 429 (free tier is 100 req/min)."""
    delay = 8.0
    for attempt in range(tries):
        try:
            return rag.embed(text, task_type="RETRIEVAL_DOCUMENT")
        except Exception as ex:
            s = str(ex)
            if "429" in s or "RESOURCE_EXHAUSTED" in s:
                m = re.search(r"retry in ([0-9.]+)s", s) or re.search(r"([0-9.]+)s", s)
                wait = (float(m.group(1)) + 1) if m else delay
                print(f"    rate-limited; waiting {wait:.0f}s (attempt {attempt + 1}/{tries})")
                time.sleep(wait)
                delay = min(delay * 1.5, 45)
                continue
            raise
    _fatal("embedding kept hitting the rate limit — try again in a minute")


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    replace = "--replace" in sys.argv[1:]
    if not args:
        _fatal("give the CSV path: python embed_qa.py <file.csv> [--replace]")
    csv_path = Path(args[0]).expanduser()
    if not csv_path.exists():
        _fatal(f"CSV not found: {csv_path}")

    url = os.getenv("SUPABASE_URL", "").rstrip("/")
    key = os.getenv("SUPABASE_KEY", "")
    if not url or not key:
        _fatal("SUPABASE_URL / SUPABASE_KEY not set (SECRET key needed)")
    if not os.getenv("GOOGLE_API_KEY"):
        _fatal("GOOGLE_API_KEY not set (needed to embed)")
    if key.startswith("sb_publishable_") or key.startswith("eyJ") and "anon" in key:
        print("WARNING: this looks like a PUBLIC key; with RLS on, writes will be "
              "rejected. Use the SECRET (service_role / sb_secret_...) key.")

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f)]
    rows = [r for r in rows if (r.get("question") or "").strip() and (r.get("answer") or "").strip()]
    if not rows:
        _fatal("no usable rows (need non-empty question & answer)")
    print(f"loaded {len(rows)} Q&A from {csv_path.name}")

    endpoint = f"{url}/rest/v1/{rag.TABLE}"
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    with httpx.Client(timeout=30.0) as cx:
        if replace:
            print("--replace: deleting existing rows ...")
            d = cx.delete(f"{endpoint}?id=gt.0", headers={**headers, "Prefer": "return=minimal"})
            if d.status_code >= 300:
                _fatal(f"delete failed [{d.status_code}]: {d.text[:300]}")
            existing = set()
        else:
            # Idempotent: skip questions already stored, so re-runs only fill gaps
            # (and don't waste the embedding quota re-embedding what's there).
            g = cx.get(f"{endpoint}?select=question", headers=headers)
            existing = {(row.get("question") or "").strip() for row in g.json()} if g.status_code < 300 else set()
            print(f"skip mode: {len(existing)} already present")

        todo = [r for r in rows if r["question"].strip() not in existing]
        print(f"embedding {len(todo)} new rows (spacing {SLEEP}s to respect free-tier RPM) ...")

        batch, done = [], 0
        for i, r in enumerate(todo, 1):
            emb = _embed_retry(r["question"].strip())
            batch.append({
                "question": r["question"].strip(),
                "answer": r["answer"].strip(),
                "category": (r.get("category") or "").strip() or None,
                "embedding": rag.vec_literal(emb),
            })
            time.sleep(SLEEP)
            if len(batch) >= BATCH or i == len(todo):
                resp = cx.post(endpoint, json=batch, headers={**headers, "Prefer": "return=minimal"})
                if resp.status_code >= 300:
                    _fatal(f"insert failed [{resp.status_code}]: {resp.text[:400]}")
                done += len(batch)
                print(f"  uploaded {done}/{len(todo)}")
                batch = []

    print(f"done — {done} new Q&A embedded ({rag.EMBED_MODEL}) and stored in '{rag.TABLE}'.")


if __name__ == "__main__":
    main()
