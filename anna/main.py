"""AnnaSaathi API. Stateless: every request reads the household from SQLite,
runs the explainable models and returns plain JSON."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import date, datetime

from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

from . import env  # noqa: F401  (loads api/.env before anything reads the environment)
from . import db, engine, extract, guests, household, llm, occasions, ocr, planner, simulator, videos, weather
from .data import catalogue, cooked_tiers, fmt_qty, fmt_rate, recipes, sample_bills, tiers

app = FastAPI(title="AnnaSaathi API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def today() -> date:
    return date.fromisoformat(os.environ["ANNA_TODAY"]) if os.environ.get("ANNA_TODAY") else date.today()


def current_weather(h: dict) -> dict:
    o = h.get("weather_override")
    if o:  # a what-if set from the pantry screen, for showing how heat changes shelf life
        return {"temp_c": float(o["temp_c"]), "humidity": float(o.get("humidity", 60)), "source": "what-if",
                "city": h.get("city", "")}
    return weather.current(h["lat"], h["lon"], h.get("city", ""))


def ctx(con) -> engine.Ctx:
    h = db.get_household(con)
    return engine.Ctx(household=h, today=today(), weather=current_weather(h),
                      use_rates=db.kv(con, "use_rates"), shelf_corrections=db.kv(con, "shelf_corrections"))


def item_card(item_id: str, c: engine.Ctx) -> dict:
    it = catalogue()[item_id]
    lo, hi, notes = engine.shelf_range(it, engine.default_storage(it, c.household), c)
    return {"item_id": it.id, "emoji": it.emoji, "name_en": it.name_en, "name_hi": it.name_hi, "name_mr": it.name_mr,
            "unit": it.unit, "tier": it.tier, "tier_label": tiers()[it.tier].label, "category": it.category,
            "days_left": [round(lo, 1), round(hi, 1)], "notes": notes}


@app.get("/api/health")
def health():
    """For the host's health check: cheap, touches no external service."""
    return {"ok": True, "today": today().isoformat(), "photo_reader": photo_reader(), "ai_recipes": llm.configured()}


# ---------------------------------------------------------------- household

@app.get("/api/household")
def get_household():
    with db.connect() as con:
        c = ctx(con)
        return {**c.household, "adult_equivalents": round(engine.adult_equivalents(c.household), 2),
                "weather": c.weather, "today": c.today.isoformat()}


@app.get("/api/household/options")
def household_options():
    return household.options()


PREVIEW_ITEMS = ["rice", "wheat_atta", "toor_dal", "milk", "tomato", "onion", "palak", "banana"]


def _preview(h: dict) -> dict:
    """What this profile means in quantities: the proof that it drives the plan."""
    from .data import factors

    c = engine.Ctx(household=h, today=today(), weather={"temp_c": 28, "humidity": 60})
    f = factors()
    rows = []
    for iid in PREVIEW_ITEMS:
        it = catalogue()[iid]
        use, src = engine.daily_use(it, c)
        if use <= 0:
            rows.append({"item_id": iid, "emoji": it.emoji, "name_en": it.name_en, "per_day": None,
                         "per_week": None, "note": src})
            continue
        rows.append({"item_id": iid, "emoji": it.emoji, "name_en": it.name_en, "per_day": fmt_rate(it, use),
                     "per_week": fmt_qty(it, use * 7), "note": src})
    return {
        "adult_equivalents": round(engine.adult_equivalents(h), 2),
        "members": [{"name": m["name"], "group": m["group"], "factor": f.get(f"member:{m['group']}", 1.0)}
                    for m in h["members"]],
        "portion_factor": f.get(f"portion:{h['portion']}", 1.0),
        "home_meal_share": round((h["cooked_meals_per_day"] * 7 - h["eat_out_meals_per_week"])
                                 / (h["cooked_meals_per_day"] * 7), 3),
        "examples": rows,
    }


@app.post("/api/household/preview")
def preview_household(body: household.Household):
    return _preview(body.model_dump())


def _save(con, body: household.Household) -> dict:
    old = db.get_household(con)
    h = body.model_dump()
    if old.get("weather_override"):
        h["weather_override"] = old["weather_override"]
    if h.get("setup_seconds") is None and old.get("setup_seconds") is not None:
        h["setup_seconds"] = old["setup_seconds"]
    db.save_household(con, h)
    return h


@app.put("/api/household")
def put_household(body: household.Household):
    with db.connect() as con:
        _save(con, body)
    return get_household()


class SetupIn(BaseModel):
    household: household.Household
    fresh: bool = True  # a new family starts with an empty pantry


@app.post("/api/household/setup")
def setup_household(body: SetupIn):
    with db.connect() as con:
        if body.fresh:
            db.clear_family_data(con)
        _save(con, body.household)
    return get_household()


@app.post("/api/demo/reset")
def demo_reset():
    with db.connect() as con:
        db.reset_demo(con, today())
    return {"ok": True}


@app.get("/api/festivals")
def upcoming_festivals():
    """The family's festivals that are on now or within the next 60 days (dates are approximate for lunar festivals)."""
    from . import occasions

    with db.connect() as con:
        c = ctx(con)
    return {"festivals": occasions.upcoming(c.household, c.today), "today": c.today.isoformat()}


@app.get("/api/catalogue")
def get_catalogue():
    return [{"id": i.id, "emoji": i.emoji, "name_en": i.name_en, "name_hi": i.name_hi, "name_mr": i.name_mr,
             "unit": i.unit, "category": i.category, "tier": i.tier, "pack_size": i.pack_size, "price": i.price}
            for i in catalogue().values()]


# ---------------------------------------------------------------- bills

@app.get("/api/bills/samples")
def bill_samples():
    with db.connect() as con:
        scanned = {r["id"] for r in con.execute("SELECT id FROM bills")}
    return [{"id": b["id"], "shop": b["shop"], "address": b["address"], "bill_no": b["bill_no"], "source": b["source"],
             "raw_lines": [l["raw"] for l in b["lines"]], "total": sum(l["price"] for l in b["lines"]),
             "scanned": b["id"] in scanned}
            for b in sample_bills().values()]


REVIEW_BELOW = 0.8  # lines under this confidence are flagged for the family to check


def _enrich_lines(lines: list[dict], c: engine.Ctx) -> list[dict]:
    out = []
    for ln in lines:
        iid = ln.get("item_id")
        conf = ln.get("confidence")
        if iid and iid in catalogue():
            it = catalogue()[iid]
            row = {**ln, **item_card(iid, c), "qty_label": fmt_qty(it, float(ln["qty"] or 0)), "matched": True}
        else:
            guess, score = extract.match_item(ln.get("raw", "") + " " + ln.get("name", ""))
            if guess:
                it = catalogue()[guess]
                conf = min(conf or 1, extract.match_confidence(score))
                row = {**ln, **item_card(guess, c), "qty_label": fmt_qty(it, float(ln.get("qty") or 0)),
                       "matched": True, "fuzzy": score}
            else:
                conf = 0.0
                row = {**ln, "item_id": "unknown", "matched": False}
        conf = 0.95 if conf is None else conf  # preloaded lines were checked by hand
        if row["matched"] and not ln.get("qty"):
            conf = min(conf, 0.3)  # no quantity read
        issues = []
        if not row["matched"]:
            issues.append("Not in our item list — pick the item or leave it out")
        elif conf < REVIEW_BELOW:
            issues.append("Please check the item and quantity")
        if row["matched"] and ln.get("price") in (None, 0):
            issues.append("No price read")
        out.append({**row, "confidence": round(conf, 2), "needs_review": (not row["matched"]) or conf < REVIEW_BELOW,
                    "issues": issues})
    # the same item on two lines (e.g. two tomato entries) is merged on confirm; flag it here
    seen: dict[str, int] = {}
    for r in out:
        if r["matched"]:
            seen[r["item_id"]] = seen.get(r["item_id"], 0) + 1
    for r in out:
        if r["matched"] and seen[r["item_id"]] > 1:
            r["issues"].append("Appears more than once on this bill — quantities will be added together")
    return out


def _scan_result(bill_id: str, shop: str, source: str, mode: str, lines: list[dict], c: engine.Ctx, note=None):
    with db.connect() as con:
        imported = con.execute("SELECT scanned_on FROM bills WHERE id = ?", (bill_id,)).fetchone()
    enriched = _enrich_lines(lines, c)
    return {"bill_id": bill_id, "shop": shop, "source": source, "mode": mode, "note": note, "lines": enriched,
            "already_imported": imported["scanned_on"] if imported else None,
            "review_count": sum(l["needs_review"] for l in enriched),
            "total": round(sum(l.get("price") or 0 for l in enriched))}


def sniff_image(data: bytes) -> str | None:
    """The image type from the file's first bytes: phones don't always label uploads."""
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def photo_reader() -> str | None:
    """Which service reads bill photos: Claude vision if configured, else OCR.space, else none."""
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return "claude"
    if ocr.configured():
        return "ocr_space"
    return None


@app.get("/api/bills/status")
def scan_status():
    """Whether photo reading is configured, and by whom, so the app can say so honestly before a scan."""
    reader = photo_reader()
    return {"live_vision": reader is not None, "reader": reader,
            "model": extract.MODEL if reader == "claude" else None,
            "fallback": None if reader else "preloaded sample bill"}


@app.post("/api/bills/scan")
async def scan_bill(file: UploadFile | None = File(None), sample_id: str | None = Form(None)):
    """Photo → Claude vision (schema-checked JSON). Without credentials or on API
    failure, fall back to a preloaded scan so the demo never stalls; the response
    says which one happened (`mode`)."""
    with db.connect() as con:
        c = ctx(con)
    note = None
    if file is not None:
        data = await file.read()
        if len(data) > 12 * 1024 * 1024:
            raise HTTPException(413, "photo is too large (over 12 MB)")
        # the same photo always gets the same id, so importing it twice replaces instead of doubling
        bill_id = f"photo_{hashlib.sha1(data).hexdigest()[:12]}"
        media = sniff_image(data) or file.content_type or "image/jpeg"
        reader = photo_reader()
        try:
            if reader == "claude":
                bill = extract.extract_bill_image(data, media)
                return _scan_result(bill_id, bill.shop, "photo", "claude", [l.model_dump() for l in bill.lines], c)
            if reader == "ocr_space":
                shop, lines = ocr.bill_from_text(ocr.ocr_text(data, media))
                if not any(l["item_id"] != "unknown" for l in lines):
                    raise HTTPException(422, {"code": "no_items", "message": "No grocery lines found in this photo. "
                                              "Try again closer, flat and in good light, or paste the order text."})
                return _scan_result(bill_id, shop, "photo", "ocr", lines, c)
        except extract.ExtractionUnavailable as e:
            # a real photo was taken: say it couldn't be read rather than showing someone else's bill
            raise HTTPException(502, {"code": "read_failed", "message": f"Couldn't read the photo: {e}. "
                                      "Try again, or paste the order text."}) from e
        note = "Photo reading isn't set up on this server. Showing the preloaded sample bill instead."
    b = sample_bills().get(sample_id or "ganesh_kirana_oct24")
    if not b:
        raise HTTPException(404, "unknown sample bill")
    return _scan_result(b["id"], b["shop"], b["source"], "preloaded", [dict(l) for l in b["lines"]], c, note)


class BillText(BaseModel):
    text: str = Field(min_length=1, max_length=20000)
    shop: str = "Pasted order"


@app.post("/api/bills/parse_text")
def parse_bill_text(body: BillText):
    """A pasted quick-commerce order or bill text → the same review screen, with no AI and no network."""
    with db.connect() as con:
        c = ctx(con)
    lines = extract.parse_bill_text(body.text)
    if not lines:
        raise HTTPException(422, "no grocery lines found in that text")
    bill_id = f"text_{hashlib.sha1(body.text.strip().encode()).hexdigest()[:12]}"
    return _scan_result(bill_id, body.shop, "text", "parsed", lines, c)


@app.get("/api/bills")
def bills_history():
    with db.connect() as con:
        return db.list_bills(con)


class ConfirmLine(BaseModel):
    item_id: str
    qty: float
    price: float | None = None
    raw: str = ""
    printed_expiry: str | None = None
    storage: str | None = None


class ConfirmBill(BaseModel):
    bill_id: str
    shop: str
    source: str = "paper"
    lines: list[ConfirmLine]
    allow_duplicate: bool = False  # the family confirmed it really is a second, identical bill


def merge_lines(lines: list[dict]) -> list[dict]:
    """One entry per item: two lines of the same item on a bill are added together."""
    out: dict[str, dict] = {}
    for l in lines:
        if l["item_id"] in out:
            m = out[l["item_id"]]
            m["qty"] += l["qty"]
            m["price"] = (m["price"] or 0) + (l["price"] or 0) if (m["price"] or l["price"]) else None
            m["raw"] = f"{m['raw']} + {l['raw']}"
        else:
            out[l["item_id"]] = dict(l)
    return list(out.values())


@app.post("/api/bills/confirm")
def confirm_bill(body: ConfirmBill):
    """Add a reviewed bill to the pantry. Re-confirming the same bill replaces it (never doubles);
    a different bill with the same shop and items is refused unless the family says it's a new one."""
    lines = merge_lines([l.model_dump() for l in body.lines if l.item_id in catalogue() and l.qty > 0])
    if not lines:
        raise HTTPException(422, "no items to add")
    storage = {l["item_id"]: l["storage"] for l in lines if l["storage"]}
    with db.connect() as con:
        h = db.get_household(con)
        replaced = con.execute("SELECT 1 FROM bills WHERE id = ?", (body.bill_id,)).fetchone() is not None
        dup = None if body.allow_duplicate else db.find_duplicate_bill(con, body.bill_id, body.shop, lines)
        if dup:
            raise HTTPException(409, {"code": "duplicate_bill", "scanned_on": dup["scanned_on"], "shop": dup["shop"],
                                      "message": f"This looks like the {dup['shop']} bill already added on "
                                                 f"{dup['scanned_on']}."})
        db.add_bill(con, body.bill_id, body.shop, today(), body.source, lines,
                    lambda iid: storage.get(iid) or engine.default_storage(catalogue()[iid], h),
                    lot_source="voice" if body.source == "voice" else "bill")
    return {"ok": True, "added": len(lines), "replaced": replaced}


class VoiceIn(BaseModel):
    text: str


@app.post("/api/voice/parse")
def voice_parse(body: VoiceIn):
    with db.connect() as con:
        c = ctx(con)
    return {"lines": _enrich_lines(extract.parse_voice(body.text), c)}


# ---------------------------------------------------------------- inventory and plans

@app.get("/api/inventory")
def get_inventory():
    with db.connect() as con:
        c = ctx(con)
        inv = engine.inventory(db.get_lots(con), c)
    active = [r for r in inv if r["status"] != "finished"]
    return {
        "items": inv, "weather": c.weather, "today": c.today.isoformat(),
        "counts": {s: sum(1 for r in inv if r["status"] == s)
                   for s in ("spoiled", "use_today", "use_soon", "fresh", "finished")},
        "at_risk_rupees": sum(r["at_risk_rupees"] for r in active),
    }


class InventoryEdit(BaseModel):
    storage: Literal["fridge", "pantry"] | None = None
    printed_expiry: date | None = None
    clear_printed_expiry: bool = False
    qty: float | None = Field(default=None, ge=0, le=1000)  # what's actually left now, in the item's unit
    purchased_on: date | None = None


@app.patch("/api/inventory/{item_id}")
def edit_inventory(item_id: str, body: InventoryEdit):
    """Correct an item: how much is left, where it's kept, the printed date or the purchase date."""
    it = catalogue().get(item_id)
    if not it:
        raise HTTPException(404, "unknown item")
    with db.connect() as con:
        c = ctx(con)
        if body.storage == "fridge" and not c.household.get("has_fridge", True):
            raise HTTPException(400, "this home has no fridge (change it in the family profile)")
        if body.purchased_on and body.purchased_on > c.today:
            raise HTTPException(400, "the purchase date can't be in the future")
        n = db.update_active_lots(con, item_id, storage=body.storage,
                                  printed_expiry=body.printed_expiry, clear_expiry=body.clear_printed_expiry)
        if not n:
            raise HTTPException(404, "nothing of this item in the pantry")
        if body.purchased_on:  # the newest lot is the purchase the family is correcting
            lot = max((l for l in db.get_lots(con) if l["item_id"] == item_id and l["status"] == "active"),
                      key=lambda l: (l["purchased_on"], l["id"]))
            con.execute("UPDATE lots SET purchased_on = ? WHERE id = ?", (body.purchased_on.isoformat(), lot["id"]))
        if body.qty is not None:
            db.reanchor(con, item_id, c.today, engine.daily_use(it, c)[0], total=body.qty)
        con.commit()
    return get_inventory()


@app.delete("/api/inventory/{item_id}")
def remove_inventory(item_id: str):
    """Take an item out of My Kitchen. Returns the lot ids so the app can offer Undo."""
    with db.connect() as con:
        ids = db.remove_item(con, item_id)
    if not ids:
        raise HTTPException(404, "nothing of this item in the pantry")
    return {"ok": True, "lot_ids": ids}


class RestoreIn(BaseModel):
    lot_ids: list[int] = Field(min_length=1)


@app.post("/api/inventory/restore")
def restore_inventory(body: RestoreIn):
    with db.connect() as con:
        n = db.restore_lots(con, body.lot_ids)
    return {"ok": True, "restored": n}


class InventoryAdd(BaseModel):
    item_id: str
    qty: float = Field(gt=0)
    storage: Literal["fridge", "pantry"] | None = None
    printed_expiry: date | None = None


@app.post("/api/inventory/add")
def add_inventory(body: InventoryAdd):
    """Add something bought without a bill (a one-tap fix)."""
    it = catalogue().get(body.item_id)
    if not it:
        raise HTTPException(404, "unknown item")
    with db.connect() as con:
        h = db.get_household(con)
        storage = body.storage or engine.default_storage(it, h)
        if storage == "fridge" and not h.get("has_fridge", True):
            raise HTTPException(400, "this home has no fridge")
        db.add_lot(con, it.id, body.qty, today(), storage, body.printed_expiry, None, round(body.qty * it.price))
        con.commit()
    return get_inventory()


class WeatherIn(BaseModel):
    temp_c: float | None = Field(default=None, ge=0, le=50)
    humidity: float = Field(default=60, ge=0, le=100)


@app.put("/api/weather")
def set_weather(body: WeatherIn):
    """Pin a temperature for a what-if (temp_c null goes back to live weather)."""
    with db.connect() as con:
        h = db.get_household(con)
        if body.temp_c is None:
            h.pop("weather_override", None)
        else:
            h["weather_override"] = {"temp_c": body.temp_c, "humidity": body.humidity}
        db.save_household(con, h)
        return current_weather(h)


@app.get("/api/plan/today")
def plan_today():
    with db.connect() as con:
        c = ctx(con)
        inv = engine.inventory(db.get_lots(con), c)
    return {"dishes": planner.cook_first(inv, c), "today": c.today.isoformat(), "weather": c.weather}


def _shopping(con) -> dict:
    """The buy-right plan plus this trip's edits.

    While a trip is open the plan is for today and ignores what was already ticked off on
    this trip, so ticking an item doesn't make the rest of the list jump around. Those
    purchases are in the pantry straight away (My Kitchen and Cook Today see them).
    """
    c = ctx(con)
    trip = db.get_trip(con)
    overrides = db.shop_overrides(con)
    lots = db.get_lots(con)
    bought = {}
    if trip:
        bought = db.trip_purchases(con, trip["trip_id"])
        lots = [l for l in lots if l["bill_id"] != trip["trip_id"]]
    manual = [i for i, o in overrides.items() if o["manual"]]
    plan = planner.buy_right(lots, c, shop_day=c.today if trip else None, extra_items=manual)
    return {**plan, **planner.shopping_view(plan, overrides, bought), "trip": trip}


@app.get("/api/shopping")
def shopping():
    with db.connect() as con:
        return _shopping(con)


class ShopItemIn(BaseModel):
    qty: float | None = Field(default=None, gt=0, le=1000)
    excluded: bool | None = None


@app.put("/api/shopping/items/{item_id}")
def edit_shop_item(item_id: str, body: ShopItemIn):
    """Change the quantity, or leave the item out of this trip (excluded) and back in."""
    if item_id not in catalogue():
        raise HTTPException(404, "unknown item")
    with db.connect() as con:
        cur = next((r for r in _shopping(con)["rows"] if r["item_id"] == item_id), None)
        fields = {k: v for k, v in body.model_dump().items() if v is not None}
        if body.qty is not None and (cur is None or cur["status"] == "have"):
            fields["manual"] = True  # buying something the plan said to skip: the family's call
        db.set_shop_override(con, item_id, **fields)
        return _shopping(con)


class ShopAddIn(BaseModel):
    item_id: str
    qty: float | None = Field(default=None, gt=0, le=1000)


@app.post("/api/shopping/items")
def add_shop_item(body: ShopAddIn):
    """Add an item by hand. Without a quantity, the buy-right formula suggests one."""
    if body.item_id not in catalogue():
        raise HTTPException(404, "unknown item")
    with db.connect() as con:
        db.set_shop_override(con, body.item_id, manual=True, excluded=False, qty=body.qty)
        return _shopping(con)


@app.delete("/api/shopping/items/{item_id}")
def reset_shop_item(item_id: str):
    """Undo the family's edits to this item (and remove it if they had added it)."""
    with db.connect() as con:
        db.clear_shop_override(con, item_id)
        return _shopping(con)


class PurchaseIn(BaseModel):
    qty: float | None = Field(default=None, gt=0, le=1000)


@app.post("/api/shopping/items/{item_id}/purchase")
def purchase_item(item_id: str, body: PurchaseIn):
    """Tick an item off: it goes into the pantry now. Ticking again updates, never doubles."""
    it = catalogue().get(item_id)
    if not it:
        raise HTTPException(404, "unknown item")
    with db.connect() as con:
        c = ctx(con)
        row = next((r for r in _shopping(con)["rows"] if r["item_id"] == item_id), None)
        qty = body.qty or (row["qty"] if row and row["qty"] > 0 else it.pack_size)
        trip = db.start_trip(con, c.today)
        if not row:
            db.set_shop_override(con, item_id, manual=True)
        db.purchase(con, trip["trip_id"], item_id, qty, c.today, engine.default_storage(it, c.household),
                    round(qty * it.price))
        return _shopping(con)


@app.post("/api/shopping/items/{item_id}/unpurchase")
def unpurchase_item(item_id: str):
    with db.connect() as con:
        trip = db.get_trip(con)
        if trip:
            db.unpurchase(con, trip["trip_id"], item_id)
        return _shopping(con)


@app.post("/api/shopping/start")
def start_shopping():
    """'I'm at the shop': plan for today instead of the usual shop day."""
    with db.connect() as con:
        db.start_trip(con, today())
        return _shopping(con)


@app.post("/api/shopping/finish")
def finish_shopping():
    with db.connect() as con:
        summary = db.finish_trip(con, today())
        return {**summary, "shopping": _shopping(con)}


# ---------------------------------------------------------------- what needs attention

@app.get("/api/alerts")
def alerts():
    """Calm, actionable reminders from the stored data. One per item or dish, with a stable
    id so the app can show each only once and let the family dismiss it."""
    with db.connect() as con:
        c = ctx(con)
        inv = engine.inventory(db.get_lots(con), c)
        cooked = [dict(r) for r in con.execute("SELECT * FROM cooked WHERE status = 'active' ORDER BY finish_by")]
    day = c.today.isoformat()
    now = datetime.now()
    out = []
    for r in cooked:
        fb = datetime.fromisoformat(r["finish_by"])
        hours = (fb - now).total_seconds() / 3600
        if hours < 0:
            out.append({"id": f"cooked:{r['id']}:past", "kind": "cooked", "level": "urgent", "item_id": None,
                        "cooked_id": r["id"],
                        "title": f"Your {r['dish'].lower()} is past its finish-by time",
                        "body": "Official guidance says not to eat it now. When in doubt, throw it out."})
        elif hours <= 3:
            out.append({"id": f"cooked:{r['id']}:soon", "kind": "cooked", "level": "soon", "item_id": None,
                        "cooked_id": r["id"],
                        "title": f"Finish the {r['dish'].lower()} by {fb.strftime('%I:%M %p').lstrip('0')}",
                        "body": "Or move it to the fridge now to keep it longer."})
    for r in inv:
        name = r["name_en"].split(" (")[0]
        if r["status"] == "spoiled":
            out.append({"id": f"check:{r['item_id']}:{day}", "kind": "check", "level": "urgent", "item_id": r["item_id"],
                        "title": f"Check your {name.lower()}",
                        "body": "It's past its usual freshness window. Look and smell first — when in doubt, throw it out."})
        elif r["status"] == "use_today":
            out.append({"id": f"use:{r['item_id']}:{day}", "kind": "use", "level": "soon", "item_id": r["item_id"],
                        "title": f"Use your {name.lower()} today",
                        "body": r["reason"]})
        elif r["status"] == "use_soon" and r["at_risk"] > 0:
            out.append({"id": f"soon:{r['item_id']}:{day}", "kind": "use", "level": "info", "item_id": r["item_id"],
                        "title": f"Your {name.lower()} may need attention soon",
                        "body": r["reason"]})
    order = {"urgent": 0, "soon": 1, "info": 2}
    out.sort(key=lambda a: order[a["level"]])
    return {"alerts": out, "today": day}


@app.get("/api/impact")
def impact():
    return simulator.results()


# ---------------------------------------------------------------- weekly 10-second check

@app.get("/api/check")
def check_items():
    with db.connect() as con:
        c = ctx(con)
        lots = db.get_lots(con)
        inv = engine.inventory(lots, c)
    cards = []
    for r in inv:
        if catalogue()[r["item_id"]].perishable or r["status"] == "finished":
            cards.append({k: r[k] for k in ("item_id", "emoji", "name_en", "name_mr", "name_hi", "category", "remaining_label",
                                            "status", "days_left", "lot_ids")})
    return {"cards": cards}


class CheckIn(BaseModel):
    item_id: str
    action: str  # still_have | finished | thrown


@app.post("/api/check")
def check(body: CheckIn):
    """Swipe result → update this family's use rate and shelf-life correction."""
    it = catalogue().get(body.item_id)
    if not it or body.action not in ("still_have", "finished", "thrown"):
        raise HTTPException(400, "bad check")
    with db.connect() as con:
        c = ctx(con)
        lots = [l for l in db.get_lots(con) if l["item_id"] == it.id and l["status"] == "active"]
        if not lots:
            return {"ok": True, "message": "Nothing to update"}
        use, _ = engine.daily_use(it, c)
        rem = engine.project_remaining(lots, use, c.today)
        start = min(l["anchor_date"] for l in lots)
        days = max(1, (c.today - start).days)
        total = sum(l["anchor_qty"] for l in lots)
        left = sum(rem.values())
        msg = ""
        if body.action == "finished":
            observed = total / days
            new = 0.5 * use + 0.5 * max(observed, use)  # finished early means we under-estimated
            for l in lots:
                con.execute("UPDATE lots SET status = 'finished' WHERE id = ?", (l["id"],))
            msg = f"Got it — your family uses ~{fmt_qty(it, new)}/day. Next list adjusts."
        elif body.action == "thrown":
            wasted = max(left, it.pack_size * 0.25)
            db.log_waste(con, it.id, wasted, it.kg(wasted), wasted * it.price, c.today)
            new = max(0.0, use * 0.85)
            first_age = min((c.today - l["purchased_on"]).days for l in lots)
            lo, _, _ = engine.shelf_range(it, lots[0].get("storage") or it.storage, c)
            if first_age < lo:  # spoiled sooner than our range: learn this home's conditions
                f = c.shelf_corrections.get(it.id, 1.0) * 0.9
                con.execute("INSERT OR REPLACE INTO shelf_corrections (item_id, factor) VALUES (?, ?)", (it.id, f))
            for l in lots:
                con.execute("UPDATE lots SET status = 'thrown' WHERE id = ?", (l["id"],))
            msg = f"Logged ~{fmt_qty(it, wasted)} thrown. We'll suggest less {it.name_en.lower()} next time."
        else:
            have = max(left, total * 0.25)
            observed = (total - have) / days
            new = 0.5 * use + 0.5 * observed
            for l in lots:
                share = rem[l["id"]] / left if left > 0 else 1 / len(lots)
                con.execute("UPDATE lots SET anchor_qty = ?, anchor_date = ? WHERE id = ?",
                            (have * share, c.today.isoformat(), l["id"]))
            msg = f"Noted. You use ~{fmt_qty(it, new)}/day."
        con.execute("INSERT OR REPLACE INTO use_rates (item_id, rate) VALUES (?, ?)", (it.id, new))
        con.commit()
    return {"ok": True, "message": msg}


@app.get("/api/waste")
def waste_log():
    with db.connect() as con:
        rows = [dict(r) for r in con.execute("SELECT * FROM waste_log ORDER BY id DESC")]
    return {"rows": rows, "kg": round(sum(r["kg"] for r in rows), 2), "rupees": round(sum(r["rupees"] for r in rows))}


# ---------------------------------------------------------------- guests coming

class GuestsIn(BaseModel):
    adults: int = Field(default=0, ge=0, le=50)
    minors: int = Field(default=0, ge=0, le=50)
    conditions: list[str] = []  # health needs any guest has (planning only)
    meal: Literal["breakfast", "lunch", "dinner", "snack"] = "dinner"
    include_family: bool = True
    selected: list[str] | None = None  # recipe ids the family picked; default: the best-covered two

    @model_validator(mode="after")
    def _someone(self):
        from .occasions import conditions

        if self.adults + self.minors == 0:
            raise ValueError("add at least one guest")
        unknown = [c for c in self.conditions if c not in conditions()]
        if unknown:
            raise ValueError(f"unknown health conditions: {', '.join(unknown)}")
        return self


def _guest_context(body: GuestsIn):
    with db.connect() as con:
        c = ctx(con)
        inv = engine.inventory(db.get_lots(con), c)
    ae, heads = guests.people(c.household, body.adults, body.minors, body.include_family)
    conds = guests.everyone_conditions(c.household, body.conditions, body.include_family)
    return c, inv, ae, heads, conds


@app.post("/api/guests/plan")
def guests_plan(body: GuestsIn):
    """A meal for the family and guests from what's in the kitchen, with recipe videos,
    and only the missing items to buy."""
    c, inv, ae, heads, conds = _guest_context(body)
    menu = guests.suggest_menu(inv, c, body.meal, ae, conds)
    by_id = {d["recipe_id"]: d for d in menu}
    if body.selected:
        stock = {r["item_id"]: r["remaining"] for r in inv if r["status"] not in ("finished", "spoiled")}
        at_risk = {r["item_id"]: r["at_risk"] for r in inv}
        for rid in body.selected:  # a dish picked earlier may have dropped out of the top list
            if rid not in by_id:
                r = next((r for r in recipes() if r["id"] == rid), None)
                if r:
                    by_id[rid] = guests.dish_card(r, ae, stock, at_risk, conds)
                    menu.append(by_id[rid])
        selected = [rid for rid in body.selected if rid in by_id]
    else:
        selected = [d["recipe_id"] for d in menu[:2]]
    vids = videos.recipe_videos([d["name_en"] for d in menu])
    for d in menu:
        d["video"] = vids.get(d["name_en"])
        d["to_buy"] = guests.shortfall([d])
    to_buy = guests.shortfall([by_id[i] for i in selected])
    return {
        "people": {"adults": body.adults, "minors": body.minors, "heads": heads, "adult_equivalents": round(ae, 1),
                   "family": len(c.household["members"]) if body.include_family else 0},
        "health": [occasions.conditions()[x]["label_en"] for x in sorted(conds)],
        "meal": body.meal, "dishes": menu, "selected": selected,
        "to_buy": to_buy, "to_buy_rupees": sum(x["price"] for x in to_buy),
        "ai_available": llm.configured(),
    }


@app.post("/api/guests/ideas")
def guests_ideas(body: GuestsIn):
    """More dish ideas from an LLM (OpenRouter), checked against the kitchen by our own code."""
    if not llm.configured():
        raise HTTPException(503, {"code": "no_ai", "message": "AI recipe ideas need an OpenRouter key on the server."})
    c, inv, ae, heads, conds = _guest_context(body)
    kitchen = [f"{r['name_en']}: {r['remaining_label']}" for r in inv if r["status"] not in ("finished", "spoiled")]
    avoid = sorted({p for x in conds for rule in occasions.health_rules().get(x, [])
                    if rule["effect"] == "avoid_prep" for p in rule["items"]})
    care = [f"{occasions.conditions()[x]['label_en']} (avoid foods {', '.join(sorted({r['note'] for r in occasions.health_rules()[x] if r['effect'] == 'flag'}))})"
            for x in sorted(conds)]
    try:
        raw = llm.ideas(kitchen, heads, body.meal, c.household.get("diet", "veg"), avoid, care)
    except extract.ExtractionUnavailable as e:
        raise HTTPException(502, {"code": "ai_failed", "message": f"Couldn't get AI ideas: {e}."}) from e
    stock = {r["item_id"]: r["remaining"] for r in inv if r["status"] not in ("finished", "spoiled")}
    at_risk = {r["item_id"]: r["at_risk"] for r in inv}
    cards = []
    for i, d in enumerate(raw):
        known = {g["item_id"]: g["per_person"] for g in d["ingredients"] if g["item_id"]}
        recipe = {"id": f"ai_{i}", "name_en": d["name"], "name_hi": d["name"], "name_mr": d["name"], "prep": "ai",
                  "meal": body.meal, "time_min": d["time_min"] or 30, "steps": d["steps"], "ingredients": known}
        card = guests.dish_card(recipe, ae, stock, at_risk, conds)
        card.update({"source": "ai", "why": d["why"], "other_ingredients": [g["name"] for g in d["ingredients"] if not g["item_id"]],
                     "to_buy": guests.shortfall([card])})
        cards.append(card)
    vids = videos.recipe_videos([d["name_en"] for d in cards])
    for d in cards:
        d["video"] = vids.get(d["name_en"])
    cards.sort(key=lambda d: (-d["cover"], len(d["missing"])))
    return {"dishes": cards, "model": llm.model()}


# ---------------------------------------------------------------- cooked food

def _recipe(rid: str) -> dict:
    r = next((r for r in recipes() if r["id"] == rid), None)
    if not r:
        raise HTTPException(404, "unknown recipe")
    return r


@app.get("/api/recipes/{rid}/consumption")
def recipe_consumption(rid: str):
    """Prefilled 'what this used' for the Mark-as-cooked sheet, scaled to this family."""
    r = _recipe(rid)
    with db.connect() as con:
        c = ctx(con)
        inv = engine.inventory(db.get_lots(con), c)
    return {"recipe_id": rid, "serves": round(engine.adult_equivalents(c.household), 1),
            "items": planner.recipe_consumption(r, inv, c)}


class UsedIn(BaseModel):
    item_id: str
    qty: float = Field(ge=0, le=100)


class CookedIn(BaseModel):
    recipe_id: str
    storage: Literal["room", "fridge"] = "room"
    consumed: list[UsedIn] | None = None  # confirmed amounts to take out of the pantry


@app.post("/api/cooked")
def log_cooked(body: CookedIn):
    """Log a cooked dish: start its finish-by clock and take the confirmed ingredients out of the pantry."""
    r = _recipe(body.recipe_id)
    prep = r["prep"] if r["prep"] in cooked_tiers() else "wet"
    used = []
    with db.connect() as con:
        c = ctx(con)
        for u in body.consumed or []:
            it = catalogue().get(u.item_id)
            if not it or u.qty <= 0:
                continue
            left = db.reanchor(con, it.id, c.today, engine.daily_use(it, c)[0], eaten=u.qty)
            used.append({"item_id": it.id, "qty": u.qty, "qty_label": fmt_qty(it, u.qty),
                         "left_label": fmt_qty(it, left) if left > 1e-6 else "finished"})
        now = datetime.now()
        fb, ct = planner.finish_by(prep, body.storage, now, c)
        cur = con.execute(
            "INSERT INTO cooked (dish, emoji, prep, storage, cooked_at, finish_by, recipe_id, consumed)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (r["name_en"], r["emoji"], prep, body.storage, now.isoformat(timespec="seconds"),
             fb.isoformat(timespec="seconds"), r["id"], json.dumps(used)))
        con.commit()
    return {"ok": True, "id": cur.lastrowid, "dish": r["name_en"], "finish_by": fb.isoformat(timespec="seconds"),
            "rule": ct["habit_note"], "official": ct["official_note"], "used": used}


@app.get("/api/cooked")
def cooked_list():
    with db.connect() as con:
        rows = [{**dict(r), "consumed": json.loads(r["consumed"] or "[]")}
                for r in con.execute("SELECT * FROM cooked WHERE status = 'active' ORDER BY finish_by")]
    for r in rows:
        ct = cooked_tiers().get(r["prep"])
        r["official"] = ct["official_note"] if ct else None
    return {"rows": rows, "now": datetime.now().isoformat(timespec="seconds")}


class CookedEdit(BaseModel):
    storage: Literal["room", "fridge"]


@app.patch("/api/cooked/{cid}")
def cooked_move(cid: int, body: CookedEdit):
    """Moved to the fridge (or taken out): the finish-by time follows, counted from when it was cooked."""
    with db.connect() as con:
        c = ctx(con)
        row = con.execute("SELECT * FROM cooked WHERE id = ?", (cid,)).fetchone()
        if not row:
            raise HTTPException(404, "unknown dish")
        fb, _ = planner.finish_by(row["prep"], body.storage, datetime.fromisoformat(row["cooked_at"]), c)
        con.execute("UPDATE cooked SET storage = ?, finish_by = ? WHERE id = ?",
                    (body.storage, fb.isoformat(timespec="seconds"), cid))
        con.commit()
    return cooked_list()


@app.post("/api/cooked/{cid}/done")
def cooked_done(cid: int, outcome: Literal["eaten", "thrown"] = "eaten"):
    with db.connect() as con:
        con.execute("UPDATE cooked SET status = ? WHERE id = ?", ("done" if outcome == "eaten" else "thrown", cid))
        con.commit()
    return {"ok": True}


@app.get("/api/recipes")
def recipe_list():
    return recipes()
