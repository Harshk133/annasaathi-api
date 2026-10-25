"""Unexpected guests: plan a meal for everyone from what's already in the kitchen, and list only
what's missing (to order, e.g. on Blinkit, or add to the shopping list).

Portions use the same factors as the family: an adult guest is 1 adult-equivalent, a minor 0.6.
Health needs (the family's and the guests') rule out a cooking style where they must (heart:
no fried dishes) and label dishes that are high in sugar or salt; nothing is prescribed.
"""
from __future__ import annotations

import math

from . import occasions
from .data import catalogue, factors, fmt_qty, recipes
from .engine import EPS, Ctx, allowed

MINOR_KG = 0.013  # same as the planner: chilli, coriander and spices never decide a dish
SLOTS = {"breakfast": ("breakfast", "snack"), "lunch": ("lunch", "dinner"), "dinner": ("dinner", "lunch"),
         "snack": ("snack", "breakfast")}


def people(h: dict, adults: int, minors: int, include_family: bool) -> tuple[float, int]:
    """(adult-equivalents, head count) eating this one meal."""
    f = factors()
    portion = f.get(f'portion:{h.get("portion", "normal")}', 1.0)
    ae = (adults * f.get("member:adult", 1.0) + minors * f.get("member:child", 0.6)) * portion
    heads = adults + minors
    if include_family:
        ae += sum(f.get(f'member:{m["group"]}', 1.0) for m in h["members"]) * portion
        heads += len(h["members"])
    return ae, heads


def everyone_conditions(h: dict, guest_conditions: list[str], include_family: bool) -> set[str]:
    out = set(guest_conditions)
    if include_family:
        for m in h["members"]:
            out.update(m.get("conditions") or [])
    return out


def _stock(inv: list[dict]) -> dict[str, float]:
    return {r["item_id"]: r["remaining"] for r in inv if r["status"] not in ("finished", "spoiled")}


def _health_notes(item_ids, conditions: set[str]) -> list[str]:
    notes = []
    for c in sorted(conditions):
        for rule in occasions.health_rules().get(c, []):
            if rule["effect"] == "flag" and any(i in rule["items"] for i in item_ids):
                notes.append(f"{rule['note'].capitalize()}: take care for anyone with {occasions.conditions()[c]['label_en'].lower()}")
    return notes


def dish_card(r: dict, ae: float, stock: dict[str, float], at_risk: dict[str, float], conditions: set[str]) -> dict:
    """How well the kitchen covers one recipe for this many people."""
    cat = catalogue()
    lines, have_main, main, rescued = [], 0, 0, 0.0
    for iid, per_ae in r["ingredients"].items():
        it = cat[iid]
        need = per_ae * ae
        have = stock.get(iid, 0.0)
        is_main = it.kg(per_ae) >= MINOR_KG
        main += is_main
        enough = have >= need - EPS
        have_main += is_main and enough
        rescued += min(need, have, at_risk.get(iid, 0.0)) * it.price
        lines.append({"item_id": iid, "name_en": it.name_en, "name_hi": it.name_hi, "name_mr": it.name_mr,
                      "need": round(need, 3), "need_label": fmt_qty(it, need), "have": round(have, 3),
                      "have_label": fmt_qty(it, have) if have > EPS else "none", "enough": enough, "minor": not is_main})
    cover = have_main / main if main else 1.0
    missing = [l for l in lines if not l["enough"] and not l["minor"]]
    return {
        "recipe_id": r["id"], "name_en": r["name_en"], "name_hi": r["name_hi"], "name_mr": r["name_mr"],
        "prep": r["prep"], "meal": r["meal"], "time_min": r["time_min"], "steps": r.get("steps", []),
        "ingredients": lines, "in_kitchen": have_main, "main_ingredients": main, "cover": round(cover, 2),
        "missing": [l["name_en"] for l in missing], "uses_at_risk_rupees": round(rescued),
        "health_notes": _health_notes(r["ingredients"], conditions), "source": "catalogue",
    }


def suggest_menu(inv: list[dict], ctx: Ctx, meal: str, ae: float, conditions: set[str], limit: int = 6) -> list[dict]:
    """Dishes for this meal, best covered by the kitchen first, then by food that needs using up."""
    h = ctx.household
    stock = _stock(inv)
    at_risk = {r["item_id"]: r["at_risk"] for r in inv}
    avoid = {p for c in conditions for rule in occasions.health_rules().get(c, [])
             if rule["effect"] == "avoid_prep" for p in rule["items"]}
    slots = SLOTS.get(meal, (meal,))
    cards = []
    for r in recipes():
        if r["meal"] not in slots or r["prep"] in avoid or not allowed(r["diet"], h.get("diet", "veg")):
            continue
        cards.append(dish_card(r, ae, stock, at_risk, conditions))
    # dishes that suit everyone eating come first; a flagged dish (e.g. a sweet with a diabetic guest)
    # stays available but is never picked by default
    cards.sort(key=lambda d: (bool(d["health_notes"]), -d["cover"], len(d["missing"]), -d["uses_at_risk_rupees"], d["time_min"]))
    return cards[:limit]


def shortfall(dishes: list[dict]) -> list[dict]:
    """What to buy for the chosen dishes: total need across dishes minus what's at home, in packs."""
    cat = catalogue()
    need: dict[str, float] = {}
    have: dict[str, float] = {}
    for d in dishes:
        for l in d["ingredients"]:
            need[l["item_id"]] = need.get(l["item_id"], 0.0) + l["need"]
            have[l["item_id"]] = l["have"]
    out = []
    for iid, n in need.items():
        it = cat[iid]
        gap = n - have.get(iid, 0.0)
        if gap <= EPS:
            continue
        packs = max(1, math.ceil(gap / it.pack_size - 0.05))
        qty = round(packs * it.pack_size, 3)
        out.append({"item_id": iid, "name_en": it.name_en, "name_hi": it.name_hi, "name_mr": it.name_mr,
                    "category": it.category, "unit": it.unit, "need_label": fmt_qty(it, n),
                    "have_label": fmt_qty(it, have.get(iid, 0.0)) if have.get(iid, 0.0) > EPS else "none",
                    "short_label": fmt_qty(it, gap), "qty": qty, "qty_label": fmt_qty(it, qty),
                    "price": round(qty * it.price)})
    out.sort(key=lambda x: (x["category"], x["name_en"]))
    return out
