"""Bill and voice input → catalogue items.

The LLM is called once per bill: Claude reads the photo into a fixed JSON
schema (item ids constrained to the catalogue), the result is re-validated
with pydantic, and any line the model could not place is matched against the
catalogue's English/Hindi/Marathi aliases.
"""
from __future__ import annotations

import base64
import difflib
import json
import os
import re
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from .data import catalogue

MODEL = "claude-opus-5-5"


class BillLine(BaseModel):
    raw: str
    item_id: str
    name: str
    qty: float
    unit: Literal["kg", "L", "pc"]
    price: float | None = None
    printed_expiry: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)


class Bill(BaseModel):
    shop: str
    date: str | None = None
    lines: list[BillLine]


def bill_schema() -> dict:
    ids = sorted(catalogue()) + ["unknown"]
    return {
        "type": "object",
        "properties": {
            "shop": {"type": "string"},
            "date": {"type": ["string", "null"], "description": "Bill date as YYYY-MM-DD if printed"},
            "lines": {"type": "array", "items": {
                "type": "object",
                "properties": {
                    "raw": {"type": "string", "description": "The line exactly as printed"},
                    "item_id": {"type": "string", "enum": ids},
                    "name": {"type": "string"},
                    "qty": {"type": "number", "description": "Total quantity in `unit` (e.g. 10 x 500ml = 5 L)"},
                    "unit": {"type": "string", "enum": ["kg", "L", "pc"]},
                    "price": {"type": ["number", "null"]},
                    "printed_expiry": {"type": ["string", "null"], "description": "YYYY-MM-DD if a use-by date is printed"},
                    "confidence": {"type": "number", "description": "0-1: how sure you are of the item AND quantity "
                                   "(below 0.7 if the line is smudged, handwritten, cut off or ambiguous)"},
                },
                "required": ["raw", "item_id", "name", "qty", "unit", "price", "printed_expiry", "confidence"],
                "additionalProperties": False,
            }},
        },
        "required": ["shop", "date", "lines"],
        "additionalProperties": False,
    }


def _prompt() -> str:
    cat = catalogue()
    rows = "\n".join(f"{i.id}: {i.name_en} / {i.name_hi} / {i.name_mr} — unit {i.unit}"
                     + (f" (1 pc = one {'bunch' if i.category == 'leafy' else 'piece'})" if i.unit == 'pc' else "")
                     for i in cat.values())
    return (
        "This is an Indian grocery bill (kirana paper bill or a quick-commerce order screenshot). "
        "Read every purchased line. Lines may mix English, Marathi and Hindi, or use shop shorthand "
        "(e.g. KANDA = onion, BATATA = potato, JUDI = bunch, PKT = packet, DZN = dozen).\n\n"
        "Map each line to one catalogue id below and convert the quantity to that id's unit as a total "
        "(10 packets of 500 ml milk = 5 L; 2 dozen bananas = 24 pc; 3 bunches palak = 3 pc). "
        "Use item_id \"unknown\" for anything not in the catalogue (soap, snacks, etc.). "
        "Price is the line total in rupees. Ignore totals, taxes and the shop's details except its name.\n\n"
        f"Catalogue:\n{rows}"
    )


class ExtractionUnavailable(RuntimeError):
    pass


def extract_bill_image(image_bytes: bytes, media_type: str) -> Bill:
    """Claude vision → schema-checked bill. Raises ExtractionUnavailable when
    there are no credentials or the API fails, so the caller can fall back to
    a preloaded scan (the demo must never depend on a slow API)."""
    try:
        import anthropic
    except ImportError as e:
        raise ExtractionUnavailable("anthropic SDK not installed") from e
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        raise ExtractionUnavailable("no Anthropic credentials configured")
    client = anthropic.Anthropic()
    try:
        resp = client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={"effort": "medium", "format": {"type": "json_schema", "schema": bill_schema()}},
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type,
                                             "data": base64.standard_b64encode(image_bytes).decode()}},
                {"type": "text", "text": _prompt()},
            ]}],
        )
    except anthropic.APIConnectionError as e:
        raise ExtractionUnavailable("network error reaching Claude") from e
    except anthropic.RateLimitError as e:
        raise ExtractionUnavailable("rate limited") from e
    except anthropic.APIStatusError as e:
        raise ExtractionUnavailable(f"Claude API error {e.status_code}") from e
    if resp.stop_reason in ("refusal", "max_tokens"):
        raise ExtractionUnavailable(f"extraction stopped: {resp.stop_reason}")
    text = next((b.text for b in resp.content if b.type == "text"), "")
    try:
        bill = Bill.model_validate(json.loads(text))
    except (json.JSONDecodeError, ValidationError) as e:
        raise ExtractionUnavailable("model output failed schema check") from e
    for ln in bill.lines:
        if ln.item_id not in catalogue():
            guess, score = match_item(f"{ln.name} {ln.raw}")
            ln.item_id = guess or "unknown"
            ln.confidence = min(ln.confidence or 1, match_confidence(score) if guess else 0.2)
        elif match_item(ln.raw)[0] not in (None, ln.item_id):
            ln.confidence = min(ln.confidence or 1, 0.6)  # the printed words point at a different item
    return bill


# ---------------------------------------------------------------- catalogue matching

def _norm(s: str) -> str:
    return re.sub(r"[^a-zऀ-ॿ ]+", " ", s.lower()).strip()


@lru_cache
def _alias_index() -> list[tuple[str, str]]:
    """(item id, normalised name) for every alias and every language's name."""
    out = []
    for item in catalogue().values():
        for a in list(item.aliases) + [item.name_en.lower(), item.name_hi, item.name_mr]:
            a = _norm(a)
            if a:
                out.append((item.id, a))
    return out


def match_item(text: str) -> tuple[str | None, float]:
    """Best catalogue id for free text in English, Hindi or Marathi."""
    t = f" {_norm(text)} "
    best, best_score = None, 0.0
    for iid, a in _alias_index():  # exact word matches first; longer alias wins ("moong dal" over "dal")
        if f" {a} " in t and 1.0 + len(a) / 100 > best_score:
            best, best_score = iid, 1.0 + len(a) / 100
    if best:
        return best, round(best_score, 2)
    words = t.split()
    for iid, a in _alias_index():  # then typos and spelling variants ("shengdane" for "shengdana")
        for w in words:
            sm = difflib.SequenceMatcher(None, a, w)
            if sm.real_quick_ratio() * 0.9 <= best_score or sm.quick_ratio() * 0.9 <= best_score:
                continue
            score = sm.ratio() * 0.9
            if score > best_score:
                best, best_score = iid, score
    return (best, round(best_score, 2)) if best_score >= 0.75 else (None, round(best_score, 2))


def match_confidence(score: float) -> float:
    """Matcher score → 0–1 confidence (exact alias ≈ 0.95, fuzzy 0.75 → 0.6)."""
    return 0.95 if score >= 1 else round(max(0.0, min(0.9, 0.6 + (score - 0.75) * 2)), 2)


# ---------------------------------------------------------------- printed / pasted bill text

SKIP_LINE = re.compile(r"\b(sub\s*total|total|gst|cgst|sgst|tax|discount|savings|amount|cash|change|round(ed)?\s*off|"
                       r"bill\s*no|invoice|date|time|thank|phone|mob|delivery|handling|fee|tip|qty|items?)\b", re.I)
SIZE = re.compile(r"(\d+(?:\.\d+)?)\s*(kgs?|g|gms?|grams?|l|ltrs?|litres?|liters?|ml)\b(?:\s*[x×*]\s*(\d+))?", re.I)
COUNT = re.compile(r"(\d+(?:\.\d+)?)\s*(nos?|pkts?|packets?|pcs?|bags?|judi|bunch(?:es)?|dzn|dozen)\b", re.I)
TIMES = re.compile(r"[x×*]\s*(\d+)\b", re.I)
PRICE = re.compile(r"(?:₹|rs\.?|inr)?\s*(\d+\.\d{2}|(?<=₹)\s*\d+|(?<=rs)\s*\d+)\s*$", re.I)
TO_BASE = {"kg": 1, "kgs": 1, "g": 0.001, "gm": 0.001, "gms": 0.001, "gram": 0.001, "grams": 0.001,
           "l": 1, "ltr": 1, "ltrs": 1, "litre": 1, "litres": 1, "liter": 1, "liters": 1, "ml": 0.001}


def parse_bill_line(raw: str) -> dict | None:
    """One printed or pasted bill line → {item_id, qty (in the item's unit), price, confidence}.

    Deterministic, no network: 'TOMATO HYBRID 2.000 KG 80.00', 'GOKUL MILK 500ML x10 10 PKT 280.00',
    'KELI (BANANA) 2 DZN 120.00', 'Fresh Bhindi (Okra) 250 g x 2'. Returns None for totals,
    taxes and blank lines; an unknown item comes back with item_id "unknown".
    """
    line = raw.strip().replace("~", " ")
    if not line or SKIP_LINE.search(line) or not re.search(r"[A-Za-zऀ-ॿ]", line):
        return None
    price = None
    m = PRICE.search(line)
    if m and (re.search(r"\d", line[:m.start()]) or "." in m.group(1)):
        price = float(m.group(1))
        line = line[:m.start()].strip()
    first_digit = re.search(r"\d", line)
    name = line[:first_digit.start()] if first_digit else line
    item_id, score = match_item(name)
    if not item_id:
        item_id, score = match_item(line)
    if not item_id and not re.search(r"\d", raw):
        return None  # a header such as the shop's name: no quantity, no price, no item
    if not item_id:
        return {"raw": raw.strip(), "item_id": "unknown", "name": name.strip(), "qty": 0, "unit": "pc",
                "price": price, "printed_expiry": None, "confidence": 0.2}
    item = catalogue()[item_id]
    sizes = [(float(a) * TO_BASE[u.lower()], int(x) if x else None) for a, u, x in SIZE.findall(line)]
    counts = [(float(a) * (12 if u.lower() in ("dzn", "dozen") else 1)) for a, u in COUNT.findall(line)]
    times = [int(x) for x in TIMES.findall(line)]
    sure = True
    if item.unit in ("kg", "L"):
        if sizes:
            size, mult = sizes[0]
            qty = size * (mult or (counts[0] if counts else (times[0] if times else 1)))
        elif counts:
            qty, sure = counts[0] * item.pack_size, False  # "PANEER 2 PKT": packs of the usual size
        else:
            qty, sure = item.pack_size, False
    else:
        if counts:
            qty = counts[0]
        elif times:
            qty = times[0]
        elif sizes and item.pc_kg:
            qty, sure = max(1.0, round(sizes[0][0] / item.pc_kg)), False
        else:
            qty, sure = 1.0, False
    conf = match_confidence(score) * (1 if sure else 0.7)
    return {"raw": raw.strip(), "item_id": item_id, "name": item.name_en, "qty": round(qty, 3), "unit": item.unit,
            "price": price, "printed_expiry": None, "confidence": round(conf, 2)}


def parse_bill_text(text: str) -> list[dict]:
    """A whole pasted order or OCR'd bill → lines (totals, headers and blanks dropped)."""
    return [ln for ln in (parse_bill_line(r) for r in text.splitlines()) if ln]


NUM_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "half": 0.5, "dozen": 12,
    "ek": 1, "do": 2, "teen": 3, "char": 4, "paanch": 5, "aadha": 0.5, "dhai": 2.5, "dedh": 1.5,
    "don": 2, "tin": 3, "chaar": 4, "paach": 5, "ardha": 0.5, "pav": 0.25,
    "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पांच": 5, "आधा": 0.5, "डेढ़": 1.5,
    "दोन": 2, "पाच": 5, "अर्धा": 0.5, "दीड": 1.5, "पाव": 0.25,
}
UNIT_WORDS = {
    "kg": 1, "kilo": 1, "किलो": 1, "g": 0.001, "gram": 0.001, "grams": 0.001, "ग्राम": 0.001,
    "litre": 1, "liter": 1, "l": 1, "लीटर": 1, "लिटर": 1, "ml": 0.001,
}


def parse_voice(text: str) -> list[dict]:
    """'2 kilo tomato aur ek palak' / 'दोन किलो कांदा' → [{item_id, qty}].
    Cheap rule-based parse; the LLM is not needed for short purchase notes."""
    cat = catalogue()
    chunks = re.split(r",|\band\b|\baur\b|\bani\b|और|आणि|\n", text, flags=re.I)
    out = []
    for c in chunks:
        if not c.strip():
            continue
        item_id, conf = match_item(c)
        if not item_id:
            continue
        item = cat[item_id]
        tokens = c.lower().split()
        qty, mult = None, None
        for i, tok in enumerate(tokens):
            m = re.match(r"^(\d+(?:\.\d+)?)([a-z]*)$", tok)
            if m and qty is None:
                qty = float(m.group(1))
                if m.group(2) in UNIT_WORDS:
                    mult = UNIT_WORDS[m.group(2)]
            elif tok in NUM_WORDS and qty is None:
                qty = NUM_WORDS[tok]
            elif tok in UNIT_WORDS and mult is None:
                mult = UNIT_WORDS[tok]
        guessed = qty is None
        qty = qty if qty is not None else (item.pack_size if item.unit != "pc" else 1)
        if mult is not None and item.unit in ("kg", "L"):
            qty *= mult
        out.append({"raw": c.strip(), "item_id": item_id, "name": item.name_en, "qty": round(qty, 3),
                    "unit": item.unit, "price": round(qty * item.price), "printed_expiry": None,
                    "confidence": round(match_confidence(conf) * (0.7 if guessed else 1), 2)})
    return out
