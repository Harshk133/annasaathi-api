"""The explainable planning models: consumption, shelf life, urgency,
cook-first and buy-right. Pure functions over plain dicts, so they are easy to
unit-test with hand-checked numbers and the API stays stateless.

Lot dict shape: {id, item_id, qty, purchased_on (date), storage, printed_expiry
(date|None), status, anchor_qty, anchor_date (date)}. The anchor is the last
moment we knew the quantity for sure (purchase or weekly check).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta

from .data import Item, catalogue, factors, fmt_days, fmt_qty, fmt_rate, tiers
from .household import needs

EPS = 1e-6
FRIDGE_HELPS = ("D3", "D5", "D8")  # counter produce the fridge keeps longer (not onion, potato)


@dataclass
class Ctx:
    household: dict
    today: date
    weather: dict  # {"temp_c": float, "humidity": float}
    use_rates: dict[str, float] = field(default_factory=dict)  # learned per household
    shelf_corrections: dict[str, float] = field(default_factory=dict)  # learned per household


# ---------------------------------------------------------------- consumption

def adult_equivalents(h: dict) -> float:
    f = factors()
    ae = sum(f.get(f'member:{m["group"]}', 1.0) for m in h["members"])
    ae *= f.get(f'portion:{h.get("portion", "normal")}', 1.0)
    weekly_meals = h.get("cooked_meals_per_day", 3) * 7
    ae *= (weekly_meals - h.get("eat_out_meals_per_week", 0)) / weekly_meals
    return ae


def allowed(item_diet: str, household_diet: str) -> bool:
    if item_diet == "nojain":
        return household_diet != "jain"
    if item_diet == "egg":
        return household_diet in ("egg", "nonveg")
    if item_diet == "meat":
        return household_diet == "nonveg"
    return True


def daily_use(item: Item, ctx: Ctx) -> tuple[float, str]:
    """Daily use in the item's unit, and where the number came from."""
    if item.id in ctx.use_rates:
        return ctx.use_rates[item.id], "learned from your weekly checks"
    h = ctx.household
    if not allowed(item.diet, h.get("diet", "veg")):
        return 0.0, "not in your diet"
    from .occasions import skipped_share

    skip_ae, skipped = skipped_share(h, item.id)
    use = item.use_per_ae * max(0.0, adult_equivalents(h) - skip_ae)
    fam = f"family of {len(h['members'])}"
    if skipped:
        fam += f", not counting {' and '.join(skipped)}"
    extra, needers = 0.0, []
    for m in h["members"]:
        for n in m.get("needs", []):
            need = needs().get(n)
            if need and need["kind"] == "item" and need["target"] == item.id:
                extra += item.use_per_ae * need["boost"]
                needers.append(m["name"])
    if needers:
        return use + extra, f"{fam} (extra for {', '.join(needers)})"
    return use, fam


def default_storage(item: Item, h: dict) -> str:
    """Where a fresh purchase goes: its usual place, or the counter in a home without a fridge."""
    return item.storage if h.get("has_fridge", True) else "pantry"


def storage_sensitive(item: Item) -> bool:
    return item.form in ("fresh", "wet") or (item.form == "liquid" and item.perishable)


# ---------------------------------------------------------------- shelf life

def shelf_factors(item: Item, storage: str, ctx: Ctx) -> list[dict]:
    """Every adjustment applied to the item's tier, each with its number."""
    f = factors()
    out = []
    if storage_sensitive(item):
        if item.storage == "fridge" and storage == "pantry":
            out.append({"kind": "storage", "label": "kept outside the fridge",
                        "factor": f["storage:fresh:pantry_when_fridge_default"]})
        elif item.storage == "pantry" and storage == "fridge" and item.tier in FRIDGE_HELPS:
            out.append({"kind": "storage", "label": "kept in the fridge",
                        "factor": f["storage:fresh:fridge_when_pantry_default"]})
        if storage == "pantry":
            temp = ctx.weather.get("temp_c", 28)
            wk = (f["weather:hot_35"] if temp >= 35 else f["weather:hot_30"] if temp >= 30
                  else f["weather:warm_25"] if temp >= 25 else 1.0)
            if wk < 1:
                out.append({"kind": "weather", "label": f"{temp:.0f}°C on the counter", "factor": wk})
    elif item.form == "dry" and ctx.weather.get("humidity", 60) >= 75:
        out.append({"kind": "humidity", "label": f"{ctx.weather['humidity']:.0f}% humidity",
                    "factor": f["weather:humid_75"]})
    corr = ctx.shelf_corrections.get(item.id)
    if corr and abs(corr - 1) > 0.01:
        out.append({"kind": "home", "label": "how it lasts in your home", "factor": round(corr, 3)})
    return out


def shelf_range(item: Item, storage: str, ctx: Ctx) -> tuple[float, float, list[str]]:
    """(min_days, max_days, notes) for a freshly bought item: tier × every factor."""
    t = tiers()[item.tier]
    fs = shelf_factors(item, storage, ctx)
    k = math.prod(x["factor"] for x in fs)
    return t.min_days * k, t.max_days * k, [x["label"] for x in fs]


def lot_days_left(lot: dict, item: Item, ctx: Ctx) -> tuple[float, float]:
    if lot.get("printed_expiry"):
        d = (lot["printed_expiry"] - ctx.today).days
        return d, d
    lo, hi, _ = shelf_range(item, lot.get("storage") or default_storage(item, ctx.household), ctx)
    age = (ctx.today - lot["purchased_on"]).days
    return lo - age, hi - age


# ---------------------------------------------------------------- inventory projection

def project_remaining(lots: list[dict], use: float, until: date) -> dict[int, float]:
    """Estimate what is left of each lot on `until`, eating oldest first.

    No daily logging: consumption is the household's daily use rate, applied
    from each lot's anchor (purchase or last weekly check).
    """
    lots = sorted(lots, key=lambda l: (l["purchased_on"], l["id"]))
    rem = {l["id"]: 0.0 for l in lots}
    if not lots:
        return rem
    start = min(l["anchor_date"] for l in lots)
    day = start
    while day < until:
        for l in lots:
            if l["anchor_date"] == day:
                rem[l["id"]] = l["anchor_qty"]
        need = use
        for l in lots:
            if need <= EPS:
                break
            take = min(rem[l["id"]], need)
            rem[l["id"]] -= take
            need -= take
        day += timedelta(days=1)
    for l in lots:  # lots anchored today or later are untouched
        if l["anchor_date"] >= until:
            rem[l["id"]] = l["anchor_qty"]
    return rem


def forecast_waste(stock: list[tuple[float, float]], use: float) -> list[float]:
    """Given [(qty, days_left_mid)] eaten freshest-deadline first at `use`/day,
    return how much of each portion would spoil before it is eaten."""
    order = sorted(range(len(stock)), key=lambda i: stock[i][1])
    left = [q for q, _ in stock]
    waste = [0.0] * len(stock)
    t = 0.0
    for i in order:
        deadline = max(0.0, stock[i][1])
        can_eat = max(0.0, (deadline - t)) * use
        eaten = min(left[i], can_eat)
        waste[i] = left[i] - eaten
        t += eaten / use if use > EPS else 0
    return waste


def inventory(lots: list[dict], ctx: Ctx) -> list[dict]:
    """Every active item with tier, days-left range, urgency and a reason line."""
    cat = catalogue()
    by_item: dict[str, list[dict]] = {}
    for l in lots:
        if l["status"] == "active":
            by_item.setdefault(l["item_id"], []).append(l)
    out = []
    for item_id, ls in by_item.items():
        item = cat[item_id]
        use, use_src = daily_use(item, ctx)
        rem = project_remaining(ls, use, ctx.today)
        live = [l for l in ls if rem[l["id"]] > EPS]
        if not live:
            out.append(_finished_row(item, ls, use, ctx))
            continue
        ranges = [lot_days_left(l, item, ctx) for l in live]
        waste = forecast_waste([(rem[l["id"]], (a + b) / 2) for l, (a, b) in zip(live, ranges)], use)
        remaining = sum(rem[l["id"]] for l in live)
        at_risk = sum(waste)
        first = min(range(len(live)), key=lambda i: ranges[i][0])
        dl_min, dl_max = ranges[first]
        status = ("spoiled" if dl_max <= 0 else "use_today" if dl_min <= 1
                  else "use_soon" if dl_min <= 3 or at_risk > item.pack_size * 0.2 else "fresh")
        time_u = 1 / (1 + max(0.0, dl_min))
        waste_u = at_risk / remaining if remaining > EPS else 0
        urgency = round(min(1.0, max(time_u, waste_u) + (0.15 if status == "spoiled" else 0)), 3)
        storage = live[first].get("storage") or default_storage(item, ctx.household)
        lo, hi, notes = shelf_range(item, storage, ctx)
        age = (ctx.today - live[first]["purchased_on"]).days
        out.append({
            "item_id": item.id, "emoji": item.emoji, "name_en": item.name_en, "name_hi": item.name_hi,
            "name_mr": item.name_mr, "category": item.category, "form": item.form, "tier": item.tier,
            "tier_label": tiers()[item.tier].label, "storage": storage,
            "tier_days": [tiers()[item.tier].min_days, tiers()[item.tier].max_days],
            "factors": shelf_factors(item, storage, ctx), "age_days": age,
            "estimate": "printed date" if live[first].get("printed_expiry") else "estimated",
            "alt_storage": None if live[first].get("printed_expiry") else _alt_storage(item, storage, age, ctx),
            "remaining": round(remaining, 3), "remaining_label": fmt_qty(item, remaining),
            "bought": round(sum(l["qty"] for l in live), 3),
            "unit": item.unit, "daily_use": round(use, 4), "daily_use_label": fmt_rate(item, use),
            "days_left": [round(max(dl_min, 0), 1), round(max(dl_max, 0), 1)],
            "shelf_life": [round(lo, 1), round(hi, 1)],
            "at_risk": round(at_risk, 3), "at_risk_label": fmt_qty(item, at_risk) if at_risk > EPS else None,
            "at_risk_rupees": round(at_risk * item.price),
            "status": status, "urgency": urgency,
            # under two days at the family's pace, and not about to spoil: worth a top-up
            "low_stock": status in ("fresh", "use_soon") and use > EPS and remaining < use * 2,
            "days_of_use": round(remaining / use, 1) if use > EPS else None,
            "purchased_on": live[first]["purchased_on"].isoformat(),
            "source": live[first].get("source") or "bill",
            "printed_expiry": live[first]["printed_expiry"].isoformat() if live[first].get("printed_expiry") else None,
            "lot_ids": [l["id"] for l in live],
            "reason": _inventory_reason(item, lo, hi, notes, live[first], use, use_src, remaining, at_risk, status, ctx),
        })
    out.sort(key=lambda r: (-r["urgency"], r["days_left"][0]))
    return out


def _alt_storage(item: Item, storage: str, age: int, ctx: Ctx) -> dict | None:
    """What the other place would give, when moving the item would change anything."""
    if not storage_sensitive(item):
        return None
    other = "pantry" if storage == "fridge" else "fridge"
    if other == "fridge" and not ctx.household.get("has_fridge", True):
        return None
    lo, hi, _ = shelf_range(item, other, ctx)
    cur_lo, cur_hi, _ = shelf_range(item, storage, ctx)
    if abs(lo - cur_lo) < 0.05 and abs(hi - cur_hi) < 0.05:
        return None
    return {"storage": other, "shelf_life": [round(lo, 1), round(hi, 1)],
            "days_left": [round(max(lo - age, 0), 1), round(max(hi - age, 0), 1)]}


def _finished_row(item: Item, ls: list[dict], use: float, ctx: Ctx) -> dict:
    storage = ls[-1].get("storage") or default_storage(item, ctx.household)
    lo, hi, _ = shelf_range(item, storage, ctx)
    return {
        "item_id": item.id, "emoji": item.emoji, "name_en": item.name_en, "name_hi": item.name_hi,
        "name_mr": item.name_mr, "category": item.category, "form": item.form, "tier": item.tier,
        "tier_label": tiers()[item.tier].label, "storage": storage,
        "tier_days": [tiers()[item.tier].min_days, tiers()[item.tier].max_days],
        "factors": shelf_factors(item, storage, ctx), "age_days": (ctx.today - ls[-1]["purchased_on"]).days,
        "estimate": "estimated", "alt_storage": None, "remaining": 0,
        "remaining_label": "probably finished", "bought": round(sum(l["qty"] for l in ls), 3),
        "unit": item.unit, "daily_use": round(use, 4), "daily_use_label": fmt_rate(item, use),
        "days_left": [0, 0], "shelf_life": [round(lo, 1), round(hi, 1)], "at_risk": 0, "at_risk_label": None,
        "at_risk_rupees": 0,
        "status": "finished", "urgency": 0, "low_stock": True, "days_of_use": 0,
        "purchased_on": ls[-1]["purchased_on"].isoformat(), "source": ls[-1].get("source") or "bill",
        "printed_expiry": None, "lot_ids": [l["id"] for l in ls],
        "reason": "At your usual pace this should be used up. Confirm in the weekly check.",
    }


def _inventory_reason(item, lo, hi, notes, lot, use, use_src, remaining, at_risk, status, ctx) -> str:
    if lot.get("printed_expiry"):
        life = f"printed date {lot['printed_expiry'].strftime('%d %b')}"
    else:
        t = tiers()[item.tier]
        life = f"{item.name_en} keeps {t.min_days:g}–{t.max_days:g} days"
        if notes:
            life += f"; {', '.join(notes)} → {fmt_days(lo, hi)}"
    age = (ctx.today - lot["purchased_on"]).days
    bought = "bought today" if age == 0 else f"bought {age} day{'s' if age != 1 else ''} ago"
    pace = f"~{fmt_rate(item, use)}"
    if status == "spoiled":
        return f"{life}; {bought}. Check look and smell — when in doubt, throw it out."
    if at_risk > EPS:
        return f"{life}; {bought}. At {pace}, ~{fmt_qty(item, at_risk)} of {fmt_qty(item, remaining)} may spoil — cook it first."
    return f"{life}; {bought}. At {pace} you'll finish it in time."


