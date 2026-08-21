"""Append-only fact ledger: the source of truth for what Mira knows.

WHY THIS EXISTS
    Consolidation used to hand back the entire 12-section memory document, and
    we stored whatever came back. Two failure modes followed from that:

      * silent loss   — a section the model forgot to copy through vanished
      * drift         — a hallucinated fact became next call's "truth" and
                        compounded, with no way to see where it came from

    So the model no longer writes memory. It proposes PATCHES (set / append /
    invalidate). This module validates each one, records it as an immutable
    ledger row with provenance and time bounds, and PROJECTS the current view
    in code.

SHAPE
    ledger  (append-only)   one row per fact-version, never updated in place
                            except to stamp invalidated_at/invalidated_by
    view    (derived)       the same 12-section dict the prompt builder and
                            /memory page already consume — rebuilt from the
                            ledger, cached on the users row so the live call
                            path never reads the ledger at all

Nothing is ever deleted. A contradiction closes the old fact and opens a new
one, so "she used to skip breakfast, stopped in August" stays answerable.
"""
import json
import re
from datetime import datetime, timezone

OP_SET = "set"
OP_APPEND = "append"
OP_INVALIDATE = "invalidate"
OPS = (OP_SET, OP_APPEND, OP_INVALIDATE)

STATUS_APPLIED = "applied"
STATUS_REJECTED = "rejected"

MEAL_SLOTS = ("morning", "mid_morning", "lunch", "evening", "dinner", "late_night")

# Every writable path, with its kind. An op targeting anything not in here is
# REJECTED — the model cannot invent structure, which is the main lever against
# contamination. "scalar" holds one value; "list" holds many; "objlist" holds
# dicts (medications, entities).
SCHEMA = {
    "identity.basics.age": "scalar",
    "identity.basics.gender": "scalar",
    "identity.basics.city": "scalar",
    "identity.body.height_cm": "scalar",
    "identity.body.weight_kg": "scalar",
    "health.conditions": "list",
    "health.allergies": "list",
    "health.medications": "objlist",
    "diet.type": "scalar",
    "diet.restrictions": "list",
    "preferences.likes": "list",
    "preferences.dislikes": "list",
    "preferences.cuisine": "scalar",
    "goals.primary_goal": "scalar",
    "goals.motivation": "scalar",
    "goals.target": "scalar",
    "lifestyle.schedule": "scalar",
    "lifestyle.cooking_situation": "scalar",
    "lifestyle.household": "scalar",
    "lifestyle.budget": "scalar",
    "progress.what_worked": "list",
    "progress.what_failed": "list",
    "progress.struggles": "list",
    "entities": "objlist",
    "misc": "list",
}
for _slot in MEAL_SLOTS:
    SCHEMA[f"current_pattern.{_slot}.time"] = "scalar"
    SCHEMA[f"current_pattern.{_slot}.note"] = "scalar"
    SCHEMA[f"current_pattern.{_slot}.frequent"] = "list"
    SCHEMA[f"current_pattern.{_slot}.gaps"] = "list"

# Evidence must actually resemble something in the transcript. Paraphrase is
# normal, so this is deliberately loose — it catches invention, not rewording.
_MIN_EVIDENCE_OVERLAP = 0.34
_MIN_EVIDENCE_CHARS = 4


def _now():
    return datetime.now(timezone.utc).isoformat()


def _norm_tokens(text):
    return [t for t in re.split(r"\W+", (text or "").lower()) if len(t) > 1]


def _evidence_grounded(evidence, transcript):
    """True if enough of the evidence's tokens appear in the transcript.

    Guards against the model citing a quote the user never said. Returns True
    when there is no transcript to check against (replay of a truncated row).
    """
    if not transcript:
        return True
    ev = _norm_tokens(evidence)
    if not ev:
        return False
    hay = set(_norm_tokens(transcript))
    hits = sum(1 for t in ev if t in hay)
    return (hits / len(ev)) >= _MIN_EVIDENCE_OVERLAP


def validate_op(op, transcript=""):
    """Return (ok, reason). Reason is recorded in the ledger on rejection."""
    if not isinstance(op, dict):
        return False, "op is not an object"

    kind = str(op.get("op", "")).strip().lower()
    if kind not in OPS:
        return False, f"unknown op '{kind}' (expected set/append/invalidate)"

    path = str(op.get("path", "")).strip()
    if path not in SCHEMA:
        return False, f"path '{path}' is not in the schema"

    ptype = SCHEMA[path]
    value = op.get("value")

    if kind == OP_INVALIDATE:
        if not str(op.get("reason", "")).strip():
            return False, "invalidate needs a reason"
        return True, ""

    if value is None or value == "" or value == [] or value == {}:
        return False, "empty value"

    if ptype == "scalar":
        if isinstance(value, (list, dict)):
            return False, f"'{path}' is scalar but got {type(value).__name__}"
    elif ptype == "list":
        if isinstance(value, dict):
            return False, f"'{path}' is a list of strings but got an object"
    elif ptype == "objlist":
        items = value if isinstance(value, list) else [value]
        if not all(isinstance(i, dict) for i in items):
            return False, f"'{path}' expects objects"

    evidence = str(op.get("evidence", "")).strip()
    if len(evidence) < _MIN_EVIDENCE_CHARS:
        return False, "missing evidence — quote what the user said"
    if not _evidence_grounded(evidence, transcript):
        return False, f"evidence not found in transcript: {evidence[:60]!r}"

    return True, ""


def _same_value(a, b):
    try:
        return json.dumps(a, sort_keys=True, ensure_ascii=False) == \
               json.dumps(b, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(a) == str(b)


def live_facts(facts):
    """Applied, not-yet-invalidated facts, in ledger order."""
    return [f for f in facts
            if f.get("status", STATUS_APPLIED) == STATUS_APPLIED
            and not str(f.get("invalidated_at", "")).strip()]


def apply_patch(existing_facts, ops, session_id, transcript="", when=None,
                firebase_uid="", user_id=""):
    """Validate and apply a patch. Pure — returns rows for the caller to persist.

    Returns (new_rows, invalidations, audit) where:
      new_rows      ledger rows to append (both applied and rejected; a
                    rejected row is the audit record of a refused claim)
      invalidations [(fact_id, invalidated_by)] to stamp on existing rows
      audit         per-op summary for logging / the audit view
    """
    when = when or _now()
    existing_facts = existing_facts or []
    new_rows, invalidations, audit = [], [], []
    seq = 0

    def _mk(op_kind, path, value, evidence, confidence, status, reason):
        nonlocal seq
        seq += 1
        return {
            "fact_id": f"{session_id}#{seq:03d}",
            "firebase_uid": firebase_uid,
            "user_id": user_id,
            "path": path,
            "op": op_kind,
            "value": json.dumps(value, ensure_ascii=False) if value is not None else "",
            "valid_from": when if status == STATUS_APPLIED else "",
            "invalidated_at": "",
            "invalidated_by": "",
            "session_id": session_id,
            "evidence": (evidence or "")[:500],
            "confidence": confidence or "",
            "status": status,
            "reason": (reason or "")[:300],
            "created_at": when,
        }

    current = live_facts(existing_facts)

    for raw in (ops or []):
        ok, reason = validate_op(raw, transcript)
        kind = str(raw.get("op", "")).strip().lower() if isinstance(raw, dict) else "?"
        path = str(raw.get("path", "")).strip() if isinstance(raw, dict) else "?"
        value = raw.get("value") if isinstance(raw, dict) else None
        evidence = str(raw.get("evidence", "")).strip() if isinstance(raw, dict) else ""
        conf = str(raw.get("confidence", "")).strip() if isinstance(raw, dict) else ""

        if not ok:
            # Rejections are recorded, not discarded — that is the audit trail.
            new_rows.append(_mk(kind, path, value, evidence, conf,
                                STATUS_REJECTED, reason))
            audit.append({"op": kind, "path": path, "applied": False, "reason": reason})
            continue

        at_path = [f for f in current if f["path"] == path]

        if kind == OP_INVALIDATE:
            if not at_path:
                new_rows.append(_mk(kind, path, None, evidence, conf,
                                    STATUS_REJECTED, "nothing live at this path"))
                audit.append({"op": kind, "path": path, "applied": False,
                              "reason": "nothing live at this path"})
                continue
            marker = _mk(kind, path, None, evidence, conf, STATUS_APPLIED,
                         str(raw.get("reason", "")).strip())
            new_rows.append(marker)
            for f in at_path:
                invalidations.append((f["fact_id"], marker["fact_id"]))
            audit.append({"op": kind, "path": path, "applied": True,
                          "closed": len(at_path)})
            continue

        if kind == OP_SET:
            # No-op if unchanged: keeps the ledger free of churn and preserves
            # the original valid_from, so "since when" stays accurate.
            if any(_same_value(_load(f["value"]), value) for f in at_path):
                audit.append({"op": kind, "path": path, "applied": False,
                              "reason": "unchanged"})
                continue
            row = _mk(kind, path, value, evidence, conf, STATUS_APPLIED, "")
            new_rows.append(row)
            for f in at_path:            # 2.5 — supersede, never delete
                invalidations.append((f["fact_id"], row["fact_id"]))
            audit.append({"op": kind, "path": path, "applied": True,
                          "superseded": len(at_path)})
            continue

        if kind == OP_APPEND:
            if any(_same_value(_load(f["value"]), value) for f in at_path):
                audit.append({"op": kind, "path": path, "applied": False,
                              "reason": "duplicate"})
                continue
            new_rows.append(_mk(kind, path, value, evidence, conf,
                                STATUS_APPLIED, ""))
            audit.append({"op": kind, "path": path, "applied": True})

    return new_rows, invalidations, audit


def _load(raw):
    if raw in (None, ""):
        return None
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return raw


def _assign(tree, path, value, as_list):
    parts = path.split(".")
    node = tree
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    leaf = parts[-1]
    if as_list:
        bucket = node.setdefault(leaf, [])
        if isinstance(value, list):
            for v in value:
                if v not in bucket:
                    bucket.append(v)
        elif value not in bucket:
            bucket.append(value)
    else:
        node[leaf] = value


def build_current_view(facts):
    """Project the ledger into the 12-section dict the rest of the app expects.

    This is what makes the change non-breaking: `_build_user_context()` and the
    /memory page keep consuming exactly the shape they always did.
    """
    view = {}
    for f in live_facts(facts):
        path, kind = f.get("path", ""), f.get("op", "")
        if path not in SCHEMA or kind == OP_INVALIDATE:
            continue
        ptype = SCHEMA[path]
        value = _load(f.get("value"))
        if value is None:
            continue
        _assign(view, path, value, as_list=(ptype in ("list", "objlist")))
    return view


def history_for(facts, path=None):
    """Full timeline, newest first — the answer to 'why does Mira believe this?'"""
    rows = [f for f in (facts or []) if not path or f.get("path") == path]
    rows.sort(key=lambda f: str(f.get("created_at", "")), reverse=True)
    return rows


def seed_ops_from_document(doc):
    """Convert a pre-ledger long_term_memory doc into `set`/`append` ops.

    One-time migration so users who already have memory do not start empty.
    Marked with session_id 'migration' and evidence 'pre-existing memory', and
    exempt from the grounding check because there is no transcript to cite.
    """
    ops = []

    def walk(node, prefix):
        for key, val in (node or {}).items():
            path = f"{prefix}.{key}" if prefix else key
            if path in SCHEMA:
                if val in (None, "", [], {}):
                    continue
                ptype = SCHEMA[path]
                if ptype == "scalar":
                    ops.append({"op": OP_SET, "path": path, "value": val})
                else:
                    for item in (val if isinstance(val, list) else [val]):
                        if item not in (None, "", [], {}):
                            ops.append({"op": OP_APPEND, "path": path, "value": item})
            elif isinstance(val, dict):
                walk(val, path)

    walk(doc or {}, "")
    for o in ops:
        o["evidence"] = "pre-existing memory"
        o["confidence"] = "migrated"
    return ops
