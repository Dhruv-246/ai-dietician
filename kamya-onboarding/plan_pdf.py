"""Render a weekly diet plan (structured dict) into a PDF.

RENDERS, NEVER DECIDES. Everything here is layout: what goes in the plan is
decided in diet_plan.py and validated against the user's diet, allergies and
conditions before it ever reaches this file. Keeping generation and rendering
apart means the PDF can be re-rendered any time from stored data, so nothing
binary has to be kept and a download is always current.

reportlab on purpose: pure Python, no cairo/pango, so it cannot break a
Railway build the way weasyprint can.

SCRIPT NOTE. The plan is written in English and roman Hinglish, not
Devanagari. reportlab's built-in fonts have no Devanagari glyphs, and a
missing glyph renders as a black box rather than failing loudly -- a PDF full
of boxes is worse than one in roman script. `_ascii` enforces that at the
boundary instead of trusting the model to remember.
"""
from __future__ import annotations

import datetime as dt
import io
import re
import unicodedata

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Frame, NextPageTemplate,
                                PageBreak, PageTemplate, Paragraph, Spacer,
                                Table, TableStyle)

DARK = colors.HexColor("#14532D")     # deep green, the cover and day headers
GOLD = colors.HexColor("#F2C14E")     # the cover title
LEAF = colors.HexColor("#B7CE8F")     # the basics card
CREAM = colors.HexColor("#FAF8EC")    # day-table body
INK = colors.HexColor("#1B2A21")
MUTED = colors.HexColor("#5C7268")
LINE = colors.HexColor("#D7E0D6")

PAGE_W, PAGE_H = A4
MARGIN = 16 * mm


def _ascii(s) -> str:
    """Fold to characters the built-in fonts can actually draw.

    Devanagari would render as black boxes. Accented Latin is folded to its
    base letter, which is legible; anything else is dropped rather than shown
    as a box.
    """
    s = str(s or "")
    # BEFORE the NFKD fold. "1/2 cup" written as a vulgar fraction decomposes
    # to "1<U+2044>2", the fraction slash is not ASCII, and the strip below
    # turned it into "12 cup" -- a twelvefold portion error in a document
    # people cook from.
    s = s.replace("\u2044", "/").replace("\u2215", "/")
    # "1½ cups" must become "1 1/2 cups", not "11/2 cups".
    s = re.sub(r"(?<=\d)(?=[\u00BC-\u00BE\u2150-\u215E])", " ", s)
    for _frac, _txt in (("½", "1/2"), ("¼", "1/4"), ("¾", "3/4"),
                        ("⅓", "1/3"), ("⅔", "2/3"), ("⅛", "1/8"),
                        ("⅜", "3/8"), ("⅝", "5/8"), ("⅞", "7/8"),
                        ("⅕", "1/5"), ("⅖", "2/5"), ("⅙", "1/6")):
        s = s.replace(_frac, _txt)
    s = unicodedata.normalize("NFKD", s)
    s = s.replace("\u2044", "/").replace("\u2215", "/")
    s = "".join(c for c in s if not unicodedata.combining(c))
    # Spaces around the folded dash: an em dash between two words folds to
    # "bloating-common", which reads as a hyphenated compound that is not a
    # word.
    s = (s.replace("—", " - ").replace("–", " - ")
           .replace("‘", "'").replace("’", "'")
           .replace("“", '"').replace("”", '"'))
    s = re.sub(r"[^\x20-\x7E\n]", "", s)
    return re.sub(r" {2,}", " ", s).strip()


def _pretty(iso: str, fmt: str = "%d %b %Y") -> str:
    """2026-09-07 -> '07 Sep 2026'. An ISO date in a document meant for a
    person on a phone is a needless second of decoding."""
    try:
        d = dt.date.fromisoformat(str(iso).strip())
    except (ValueError, TypeError):
        return _ascii(iso)
    return d.strftime(fmt)


def _styles():
    base = dict(fontName="Helvetica", textColor=INK, leading=13)
    return {
        "title": ParagraphStyle("t", fontName="Helvetica-Bold", fontSize=30,
                                leading=34, textColor=GOLD, alignment=TA_CENTER),
        "sub": ParagraphStyle("s", fontName="Helvetica", fontSize=12.5,
                              leading=18, textColor=colors.white,
                              alignment=TA_CENTER),
        "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=15,
                             leading=19, textColor=DARK, spaceAfter=8),
        "body": ParagraphStyle("b", fontSize=10.5, **base),
        "slot": ParagraphStyle("sl", fontName="Helvetica-Bold", fontSize=10.5,
                               leading=13, textColor=INK),
        "time": ParagraphStyle("tm", fontName="Helvetica", fontSize=9,
                               leading=11, textColor=MUTED),
        "items": ParagraphStyle("it", fontSize=10, leading=13.5,
                                fontName="Helvetica", textColor=INK),
        "note": ParagraphStyle("nt", fontName="Helvetica-Oblique", fontSize=8.6,
                               leading=11, textColor=MUTED),
        "kv": ParagraphStyle("kv", fontName="Helvetica-Bold", fontSize=10.5,
                             leading=15, textColor=DARK),
        "kvv": ParagraphStyle("kvv", fontName="Helvetica", fontSize=10.5,
                              leading=15, textColor=INK),
        "foot": ParagraphStyle("f", fontName="Helvetica", fontSize=8,
                               textColor=MUTED, alignment=TA_CENTER),
    }


def _cover_bg(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(DARK)
    canvas.rect(0, 0, PAGE_W, PAGE_H, stroke=0, fill=1)
    canvas.setFillColor(LEAF)
    canvas.circle(PAGE_W - 18 * mm, PAGE_H - 24 * mm, 34 * mm, stroke=0, fill=1)
    canvas.restoreState()


def _plain_bg(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(colors.white)
    canvas.rect(0, 0, PAGE_W, PAGE_H, stroke=0, fill=1)
    # Page number, from page 2 on -- the cover should stay clean.
    if doc.page > 1:
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(MUTED)
        canvas.drawCentredString(PAGE_W / 2, 10 * mm, str(doc.page - 1))
    canvas.restoreState()


def _basics_card(plan, st):
    b = plan.get("basics") or {}
    rows = [[Paragraph("BASIC DETAILS", ParagraphStyle(
        "bd", fontName="Helvetica-Bold", fontSize=13, textColor=DARK,
        alignment=TA_CENTER)), ""]]
    for label in ("Name", "Height", "Weight", "Gender", "Age"):
        val = _ascii(b.get(label.lower(), ""))
        if val:
            rows.append([Paragraph(label, st["kv"]), Paragraph(val, st["kvv"])])
    t = Table(rows, colWidths=[42 * mm, 62 * mm], hAlign="CENTER")
    t.setStyle(TableStyle([
        ("SPAN", (0, 0), (1, 0)),
        ("BACKGROUND", (0, 0), (-1, -1), LEAF),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING", (0, 0), (-1, -1), 14),
        ("RIGHTPADDING", (0, 0), (-1, -1), 14),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, DARK),
    ]))
    return t


def _day_table(day, st):
    head = [[Paragraph(f'<font color="#FFFFFF"><b>DAY {day.get("day","")}</b></font>',
                       st["body"]),
             Paragraph(f'<font color="#FFFFFF"><b>{_ascii(day.get("weekday",""))}</b></font>',
                       ParagraphStyle("c", parent=st["body"], alignment=TA_CENTER)),
             Paragraph(f'<font color="#FFFFFF">{_pretty(day.get("date",""), "%d %b")}</font>',
                       ParagraphStyle("r", parent=st["body"], alignment=2))]]
    ht = Table(head, colWidths=[30 * mm, 78 * mm, 70 * mm])
    ht.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), DARK),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))

    rows = []
    for m in (day.get("meals") or []):
        left = [Paragraph(_ascii(m.get("slot", "")), st["slot"])]
        if m.get("time"):
            left.append(Paragraph(_ascii(m["time"]), st["time"]))
        right = [Paragraph(_ascii(m.get("items", "")), st["items"])]
        if m.get("note"):
            right.append(Spacer(1, 2))
            right.append(Paragraph(_ascii(m["note"]), st["note"]))
        rows.append([left, right])
    if not rows:
        rows = [[Paragraph("-", st["slot"]), Paragraph("No meals listed.", st["items"])]]

    bt = Table(rows, colWidths=[38 * mm, 140 * mm])
    bt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), CREAM),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("LINEBELOW", (0, 0), (-1, -2), 0.5, LINE),
    ]))
    return ht, bt


def build(plan: dict) -> bytes:
    """Structured plan in, PDF bytes out. Never raises on odd content."""
    st = _styles()
    buf = io.BytesIO()
    doc = BaseDocTemplate(buf, pagesize=A4,
                          leftMargin=MARGIN, rightMargin=MARGIN,
                          topMargin=MARGIN, bottomMargin=MARGIN,
                          title="Diet Chart", author="Kamya Wellness")
    frame = Frame(MARGIN, MARGIN, PAGE_W - 2 * MARGIN, PAGE_H - 2 * MARGIN,
                  id="f", showBoundary=0)
    doc.addPageTemplates([
        PageTemplate(id="cover", frames=[frame], onPage=_cover_bg),
        PageTemplate(id="page", frames=[frame], onPage=_plain_bg),
    ])

    flow = [Spacer(1, 62 * mm),
            Paragraph("DIETARY ROADMAP TO<br/>HEALTHY LIFESTYLE", st["title"]),
            Spacer(1, 10 * mm),
            Paragraph("Starting is the hardest part<br/>"
                      "Congratulations on taking the leap!", st["sub"]),
            Spacer(1, 26 * mm),
            Paragraph(f'<font color="#B7CE8F">Week of '
                      f'{_pretty(plan.get("week_start", ""))}</font>', st["sub"])]
    who = _ascii((plan.get("basics") or {}).get("name", ""))
    if who:
        flow += [Spacer(1, 4 * mm),
                 Paragraph(f'<font color="#F2C14E">Prepared for {who}</font>',
                           st["sub"])]

    # NextPageTemplate BEFORE the break, or reportlab keeps using the cover
    # template for the whole document and every page comes out dark green with
    # the leaf circle on it.
    flow += [NextPageTemplate("page"), PageBreak(),
             Spacer(1, 8 * mm), _basics_card(plan, st)]
    if plan.get("summary"):
        flow += [Spacer(1, 12 * mm),
                 Paragraph("Your week at a glance", st["h2"]),
                 Paragraph(_ascii(plan["summary"]), st["body"])]

    for day in (plan.get("days") or []):
        ht, bt = _day_table(day, st)
        flow += [PageBreak(), ht, bt]
        if day.get("note"):
            flow += [Spacer(1, 5 * mm), Paragraph(_ascii(day["note"]), st["note"])]

    flow += [Spacer(1, 10 * mm),
             Paragraph("Kamya Wellness - guidance only, not medical advice. "
                       "Talk to your doctor about any condition or medication.",
                       st["foot"])]

    doc.build(flow)
    return buf.getvalue()
