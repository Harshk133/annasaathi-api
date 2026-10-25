"""Bill-scan accuracy on the fixed set of real bills (PRD target: 8 of 10 correct).

Put each bill photo in data/bills/ with a ground-truth file next to it:
    data/bills/bill01.jpg
    data/bills/bill01.json   -> {"lines": [{"item_id": "tomato", "qty": 2}, ...]}

A bill counts as correct when every ground-truth item is found with a quantity
within 10%, and nothing extra (non-"unknown") is invented. Run after every
change to the prompt or catalogue:

    python eval_bills.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from anna import env  # noqa: F401  (loads api/.env)
from anna import ocr
from anna.extract import ExtractionUnavailable, extract_bill_image
from anna.main import photo_reader

BILLS = Path(__file__).parent / "data" / "bills"
MEDIA = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}


def score(truth: list[dict], got: list[dict]) -> tuple[int, int, list[str]]:
    want = {l["item_id"]: float(l["qty"]) for l in truth}
    have: dict[str, float] = {}
    for l in got:
        if l["item_id"] != "unknown":
            have[l["item_id"]] = have.get(l["item_id"], 0) + float(l["qty"])
    issues, ok = [], 0
    for iid, q in want.items():
        if iid not in have:
            issues.append(f"missed {iid}")
        elif abs(have[iid] - q) > 0.1 * q:
            issues.append(f"{iid}: qty {have[iid]:g} vs {q:g}")
        else:
            ok += 1
    issues += [f"invented {iid}" for iid in have if iid not in want]
    return ok, len(want), issues


def main() -> int:
    photos = sorted(p for p in BILLS.glob("*") if p.suffix.lower() in MEDIA) if BILLS.exists() else []
    if not photos:
        print(f"No bills in {BILLS}. Add photos + ground-truth JSON (see docstring).")
        return 1
    correct = 0
    for p in photos:
        truth = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))["lines"]
        try:
            reader = photo_reader()
            if reader == "claude":
                got = [l.model_dump() for l in extract_bill_image(p.read_bytes(), MEDIA[p.suffix.lower()]).lines]
            elif reader == "ocr_space":
                got = ocr.bill_from_text(ocr.ocr_text(p.read_bytes(), MEDIA[p.suffix.lower()]))[1]
            else:
                raise ExtractionUnavailable("no photo reader configured (set OCR_SPACE_API_KEY or ANTHROPIC_API_KEY)")
        except ExtractionUnavailable as e:
            print(f"{p.name}: extraction unavailable ({e})")
            return 2
        ok, n, issues = score(truth, got)
        good = not issues
        correct += good
        print(f"{'PASS' if good else 'FAIL'} {p.name}: {ok}/{n} lines" + (f" — {'; '.join(issues)}" if issues else ""))
    print(f"\n{correct} of {len(photos)} bills fully correct (target: 8 of 10)")
    return 0 if correct >= 0.8 * len(photos) else 1


if __name__ == "__main__":
    sys.exit(main())
