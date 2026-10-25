"""Bill photos read by OCR.space, then parsed by the deterministic bill-line parser.

OCR.space only turns the photo into text. Which items, how much and what price come from
extract.parse_bill_text, the same rules used for pasted order text, so every line can be
explained and checked. The free OCR.space tier accepts files up to 1 MB; larger photos are
shrunk here first.
"""
from __future__ import annotations

import io
import os
import re

import httpx

from .extract import ExtractionUnavailable, match_item, parse_bill_line

OCR_URL = "https://api.ocr.space/parse/image"
MAX_BYTES = 1_000_000  # free-tier upload limit
OCR_CONFIDENCE_CAP = 0.9  # OCR can misread a digit; never present a line as certain


def configured() -> bool:
    return bool(os.environ.get("OCR_SPACE_API_KEY"))


def shrink(data: bytes, media_type: str) -> tuple[bytes, str]:
    """Fit a photo under the upload limit: downscale and re-encode as JPEG (text stays readable)."""
    if len(data) <= MAX_BYTES:
        return data, media_type
    from PIL import Image, ImageOps

    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    side = 2000
    while True:
        im = img.copy()
        im.thumbnail((side, side))
        for quality in (85, 75, 65, 55):
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=quality, optimize=True)
            if buf.tell() <= MAX_BYTES:
                return buf.getvalue(), "image/jpeg"
        side = int(side * 0.8)
        if side < 600:
            raise ExtractionUnavailable("photo is too large to read")


def ocr_text(data: bytes, media_type: str) -> str:
    """Photo → raw text via OCR.space. Raises ExtractionUnavailable on any failure."""
    key = os.environ.get("OCR_SPACE_API_KEY")
    if not key:
        raise ExtractionUnavailable("no OCR.space key configured")
    data, media_type = shrink(data, media_type)
    ext = "png" if media_type == "image/png" else "jpg"
    try:
        r = httpx.post(
            OCR_URL,
            headers={"apikey": key},
            data={"language": "eng", "isTable": "true", "OCREngine": "2", "scale": "true",
                  "detectOrientation": "true", "isOverlayRequired": "false"},
            files={"file": (f"bill.{ext}", data, media_type)},
            timeout=60,
        )
    except httpx.HTTPError as e:
        raise ExtractionUnavailable("couldn't reach OCR.space") from e
    if r.status_code == 403:
        raise ExtractionUnavailable("OCR.space refused the key or the daily limit is reached")
    if r.status_code >= 400:
        raise ExtractionUnavailable(f"OCR.space error {r.status_code}")
    try:
        body = r.json()
    except ValueError as e:
        raise ExtractionUnavailable("OCR.space sent an unreadable reply") from e
    if isinstance(body, str):  # OCR.space reports some errors as a bare string
        raise ExtractionUnavailable(body[:120])
    if body.get("IsErroredOnProcessing"):
        msg = body.get("ErrorMessage") or body.get("ErrorDetails") or "OCR failed"
        raise ExtractionUnavailable(str(msg[0] if isinstance(msg, list) else msg)[:160])
    text = "\n".join(p.get("ParsedText", "") for p in body.get("ParsedResults") or [])
    if not text.strip():
        raise ExtractionUnavailable("no text found in the photo")
    return text


_NUMLIKE = re.compile(r"^[\dOoIl|.,]+(kg|kgs|g|gm|gms|ml|l|ltr|nos|pkt|pc|pcs|dzn)?$", re.I)


def fix_ocr_digits(line: str) -> str:
    """Undo common OCR confusions inside numbers: 2.OOO → 2.000, l50 → 150, 80,00 → 80.00."""
    out = []
    for tok in line.split():
        if _NUMLIKE.match(tok) and re.search(r"\d", tok):
            m = re.match(r"^([\dOoIl|.,]+)(.*)$", tok)
            num, unit = m.group(1), m.group(2)
            num = num.translate(str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "|": "1"}))
            num = re.sub(r",(\d{2})$", r".\1", num).replace(",", "")
            tok = num + unit
        out.append(tok)
    return " ".join(out)


def bill_from_text(text: str) -> tuple[str, list[dict]]:
    """OCR text → (shop name guess, bill lines)."""
    rows = [re.sub(r"\s+", " ", r.replace("\t", " ")).strip() for r in text.splitlines()]
    rows = [r for r in rows if r]
    shop = "Scanned bill"
    for r in rows[:4]:  # the shop's name is usually the first wordy line without numbers
        if not re.search(r"\d", r) and len(r) >= 4 and not match_item(r)[0]:
            shop = r.title()[:60]
            break
    lines: list[dict] = []
    pending: float | None = None  # a price the OCR put on a row of its own
    for r in rows:
        fixed = fix_ocr_digits(r)
        lone = LONE_PRICE.match(fixed)
        if lone:
            # table OCR sometimes splits a row: "68.00" then "BATATA 2.000 KG". Attach it to the
            # previous item if that one has no price and nothing is waiting; else hold it for the next.
            if pending is None and lines and lines[-1]["price"] is None and lines[-1].get("_last"):
                _attach(lines[-1], float(lone.group(1)))
            else:
                pending = float(lone.group(1))
            continue
        ln = parse_bill_line(fixed)
        for prev in lines:
            prev.pop("_last", None)
        if not ln:
            pending = None  # "1869.00" followed by "TOTAL": the total, not an item's price
            continue
        if ln["item_id"] == "unknown" and ln["price"] is None:
            pending = None
            continue  # addresses, phone numbers, footers: not purchases
        if ln["price"] is None and pending is not None:
            _attach(ln, pending)
        pending = None
        ln["raw"] = r
        ln["confidence"] = min(ln["confidence"], OCR_CONFIDENCE_CAP)
        ln["_last"] = True
        lines.append(ln)
    for ln in lines:
        ln.pop("_last", None)
    return shop, lines


LONE_PRICE = re.compile(r"^(?:₹|rs\.?)?\s*(\d+\.\d{2})$", re.I)


def _attach(ln: dict, price: float) -> None:
    ln["price"] = price
    ln["confidence"] = min(ln["confidence"], 0.75)  # matched by position: flagged for a glance
