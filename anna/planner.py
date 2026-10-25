"""Cook-first and buy-right: the two decisions AnnaSaathi makes for a family,
built on the stock and shelf-life models in engine.py."""
from __future__ import annotations

import math
from datetime import date, timedelta

from .data import catalogue, cooked_tiers, fmt_days, fmt_qty, fmt_rate, recipes
from .engine import (EPS, Ctx, adult_equivalents, allowed, daily_use, default_storage, lot_days_left,
                     project_remaining, shelf_range)
from . import occasions
from .household import needs

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MINOR_KG = 0.013  # per-person amounts below this (garlic, chilli, coriander) never block a dish


# ---------------------------------------------------------------- cook-first

def cook_first(inv: list[dict], ctx: Ctx, slots=("breakfast", "lunch", "dinner")) -> list[dict]:
    """Pick one dish per meal that rescues the most food (in ₹) that would
    otherwise spoil, matching the family's diet and preparation style."""
    h = ctx.household
    ae = adult_equivalents(h)
    cat = catalogue()
    stock = {r["item_id"]: r for r in inv if r["status"] not in ("finished", "spoiled")}
    at_risk = {k: r["at_risk"] for k, r in stock.items()}  # shrinks as earlier picks use it up
    likes, avoids = _prep_prefs(h)
    styles = set(h.get("prep_styles") or ["wet", "roasted", "fried", "sprouts"]) | {"raw"}
    picks, used = [], set()
    health_avoid = set(occasions.avoided_preps(h))  # e.g. a heart condition: no fried dishes
    for slot in slots:
        best, best_score, best_detail = None, -1e9, None
        fits = [r for r in recipes() if r["id"] not in used and allowed(r["diet"], h.get("diet", "veg"))
                and (r["meal"] == slot or (slot == "breakfast" and r["meal"] == "snack"))
                and r["prep"] not in health_avoid]  # a member's health condition rules a style out entirely
        # the family's cooking styles first; any style only if nothing in style can be cooked from what's home
        in_style = [r for r in fits if r["prep"] in styles]
        for pool in ([in_style, fits] if in_style else [fits]):
            for r in pool:
                rescued, urgent, missing, uses, have_main = 0.0, 0, [], [], 0
                for iid, per_ae in r["ingredients"].items():
                    need = per_ae * ae
                    row = stock.get(iid)
                    have = row["remaining"] if row else 0
                    main = cat[iid].kg(per_ae) >= MINOR_KG  # chilli, coriander, spices never decide a dish
                    if have < 0.3 * need:
                        if main:
                            missing.append(iid)
                        continue
                    have_main += main
                    take = min(need, have)
                    saves = min(take, at_risk.get(iid, 0))
                    rescued += saves * cat[iid].price
                    urgent += row["status"] == "use_today"
                    uses.append((iid, take, saves))
                # cook from what is at home: more main ingredients in the kitchen than missing
                if have_main == 0 or (missing and len(missing) >= have_main):
                    continue
                score = (rescued + 12 * urgent - 30 * len(missing)
                         + 15 * bool(likes.get(r["prep"])) - 25 * bool(avoids.get(r["prep"])))
                if score > best_score:
                    best, best_score, best_detail = r, score, (rescued, uses, missing)
            if best is not None:
                break
        if best is None:
            continue
        used.add(best["id"])
        rescued, uses, missing = best_detail
        use_rows = []
        for iid, take, saves in uses:
            at_risk[iid] = max(0.0, at_risk.get(iid, 0) - saves)
            if not cat[iid].perishable:
                continue
            row = stock[iid]
            use_rows.append({"item_id": iid, "emoji": row["emoji"], "name_en": row["name_en"], "name_mr": row["name_mr"],
                             "name_hi": row["name_hi"], "qty_label": fmt_qty(cat[iid], take),
                             "days_left": row["days_left"], "status": row["status"], "rescues": saves > EPS})
        use_rows.sort(key=lambda u: (not u["rescues"], u["days_left"][0]))
        ct = cooked_tiers().get(best["prep"])
        ingredients = []
        for iid in best["ingredients"]:
            row, it = stock.get(iid), cat[iid]
            ingredients.append({"item_id": iid, "name_en": it.name_en, "name_hi": it.name_hi, "name_mr": it.name_mr,
                                "have": bool(row and row["remaining"] > EPS),
                                "minor": it.kg(best["ingredients"][iid]) < MINOR_KG})
        picks.append({
            "steps": best.get("steps", []), "ingredients": ingredients,
            "slot": slot, "recipe_id": best["id"], "emoji": best["emoji"], "name_en": best["name_en"],
            "name_hi": best["name_hi"], "name_mr": best["name_mr"], "prep": best["prep"],
            "time_min": best["time_min"], "serves": round(ae, 1), "uses": use_rows,
            "missing": [cat[m].name_en for m in missing], "rescued_rupees": round(rescued),
            "reason": _cook_reason([u for u in use_rows if u["rescues"]], best, likes.get(best["prep"], [])),
            "keeps": _keeps_line(best["prep"], ct, ctx) if ct else None,
        })
    return picks


def _prep_prefs(h: dict) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Which members like or avoid each way of cooking, from their stated needs."""
    likes: dict[str, list[str]] = {}
    avoids: dict[str, list[str]] = {}
    for m in h["members"]:
        for n in m.get("needs", []):
            need = needs().get(n)
            if need and need["kind"] == "prep_like":
                likes.setdefault(need["target"], []).append(m["name"])
            elif need and need["kind"] == "prep_avoid":
                avoids.setdefault(need["target"], []).append(m["name"])
    return likes, avoids


def _cook_reason(rescues, recipe, fans: list[str]) -> str:
    if rescues:
        parts = [f"{u['name_en'].split(' (')[0].lower()} {u['qty_label']} ({_days(u['days_left'][0])})"
                 for u in rescues[:3]]
        s = "Uses up " + ", ".join(parts)
    else:
        s = "Nothing urgent — a balanced pick from what you have"
    if fans:
        s += f"; {recipe['prep']}, the way {' and '.join(fans)} like{'s' if len(fans) == 1 else ''} it"
    return s + "."


def _days(d: float) -> str:
    if d <= 0.5:
        return "today"
    if d <= 1.5:
        return "1 day left"
    return f"{d:.0f} days left"


def _keeps_line(prep: str, ct: dict, ctx: Ctx) -> str:
    room = ct["room_hours"]
    if prep == "wet" and ctx.weather.get("temp_c", 28) > 32:
        room = 1
    room_s = f"{room:g} h" if room < 24 else f"{room / 24:g} days"
    fridge_s = f"{ct['fridge_hours']:g} h" if ct["fridge_hours"] < 48 else f"{ct['fridge_hours'] / 24:g} days"
    return f"Keeps {room_s} out, {fridge_s} in the fridge"


def finish_by(prep: str, storage: str, cooked_at, ctx: Ctx):
    """Cooked-food deadline: the stricter of home habit and official guidance."""
    ct = cooked_tiers()[prep]
    hours = ct["fridge_hours"] if storage == "fridge" else ct["room_hours"]
    if prep == "wet" and storage != "fridge" and ctx.weather.get("temp_c", 28) > 32:
        hours = 1
    return cooked_at + timedelta(hours=hours), ct


# ---------------------------------------------------------------- buy-right

def round_to_pack(qty: float, pack: float) -> float:
    if qty < 0.3 * pack:
        return 0.0
    return round(max(1, math.ceil(qty / pack - 0.2)) * pack, 3)


def next_shop_date(lots: list[dict], ctx: Ctx) -> date:
    interval = ctx.household.get("shop_interval_days", 7)
    last = max((l["purchased_on"] for l in lots), default=ctx.today)
    nxt = last + timedelta(days=interval)
    return nxt if nxt > ctx.today else ctx.today + timedelta(days=1)


def spoil_qty(stock: float, use: float, shelf: float) -> float:
    """What can't be eaten before it spoils, eating steadily from today."""
    return max(0.0, stock - use * shelf)


def buy_right(lots: list[dict], ctx: Ctx, shop_day: date | None = None, extra_items=()) -> dict:
    """Shopping list for the next shop.

    quantity = daily use × min(shelf life, days until the following shop)
               − stock still fresh at home on shop day, rounded to pack size.
    Long-life staples keep the family's usual pack; they don't spoil in a week.

    `shop_day`: plan for this day (today, while the family is in the shop) instead of
    the usual next shop. `extra_items`: items the family added by hand; they are planned
    with the same formula even if they were never bought before.
    """
    cat = catalogue()
    interval = ctx.household.get("shop_interval_days", 7)
    shop_day = shop_day or next_shop_date(lots, ctx)
    days_to_shop = (shop_day - ctx.today).days
    by_item: dict[str, list[dict]] = {i: [] for i in extra_items if i in cat}
    for l in lots:
        by_item.setdefault(l["item_id"], []).append(l)
    h = ctx.household
    fests = occasions.festivals_for_shop(h, shop_day, interval)
    fest_extra = _festival_extras(fests, ctx)

    buy, skip = [], []
    for item_id, ls in by_item.items():
        item = cat[item_id]
        use, use_src = daily_use(item, ctx)
        if use <= EPS:
            continue
        active = [l for l in ls if l["status"] == "active"]
        now = sum(project_remaining(active, use, ctx.today).values())
        rem = project_remaining(active, use, shop_day)
        fresh = 0.0
        for l in active:
            if rem[l["id"]] <= EPS:
                continue
            a, b = lot_days_left(l, item, ctx)
            if (a + b) / 2 - days_to_shop > 0.5:
                fresh += rem[l["id"]]
        lo, hi, notes = shelf_range(item, default_storage(item, ctx.household), ctx)
        shelf_mid = (lo + hi) / 2
        cover = min(shelf_mid, interval)
        extra = sum(e["qty"] for e in fest_extra.get(item_id, []))
        need = use * cover - fresh + extra
        habit = max(ls, key=lambda l: (l["purchased_on"], l["id"]))["qty"] if ls else item.pack_size
        base = {
            "item_id": item.id, "emoji": item.emoji, "name_en": item.name_en, "name_hi": item.name_hi,
            "name_mr": item.name_mr, "category": item.category, "unit": item.unit, "tier": item.tier,
            "pack_size": item.pack_size, "unit_price": item.price,
            "daily_use_label": fmt_rate(item, use), "shelf_life": [round(lo, 1), round(hi, 1)],
            "stock_now": round(now, 3), "stock_now_label": fmt_qty(item, now) if now > EPS else "none",
            "stock_at_shop": round(fresh, 3),
            "stock_at_shop_label": fmt_qty(item, fresh) if fresh > EPS else "none",
            "need_qty": round(use * cover, 3), "need_label": fmt_qty(item, use * cover), "cover_days": round(cover, 1),
            "perishable": item.perishable,
            "festival": [{**e["festival"], "qty": round(e["qty"], 3), "qty_label": fmt_qty(item, e["qty"]),
                          "purpose": e["purpose"]} for e in fest_extra.get(item_id, [])],
            "health": occasions.health_flags(h, item_id),
        }
        # everyday use is covered but a festival needs more: still buy, and say it's for the festival
        festival_shortfall = extra > EPS and need > EPS and need - extra < item.pack_size * 0.3
        if need < item.pack_size * 0.3 and not festival_shortfall:
            if item.perishable or fresh < use * interval * 3 or item_id in extra_items or extra > EPS:
                when = "at home now" if days_to_shop <= 0 else f"still at home on {WEEKDAYS[shop_day.weekday()]}"
                incl = f", including ~{fmt_qty(item, extra)} for {fest_extra[item_id][0]['festival']['name_en']}" if extra > EPS else ""
                skip.append({**base, "reason": f"You have ~{fmt_qty(item, fresh)} {when} — "
                                               f"enough for ~{max(fresh - extra, 0) / use:.0f} days{incl}."})
            continue

        shelf_s = f"keeps {fmt_days(lo, hi)} days" + (f" ({', '.join(notes)})" if notes else "")
        at_home = f", minus ~{fmt_qty(item, fresh)} still at home" if fresh > EPS else ""
        topup, daily = None, False
        if not item.perishable:
            # long-life: the usual pack is fine unless it can't be finished before it turns
            ok = spoil_qty(habit + fresh, use, shelf_mid) <= EPS
            qty = habit if ok else round_to_pack(use * shelf_mid * 0.8 - fresh, item.pack_size)
            reason = f"~{fmt_rate(item, use)}; {shelf_s}. This lasts you ~{(qty + fresh) / use:.0f} days."
        elif item.tier == "D1":
            daily = True
            qty = round_to_pack(use, item.pack_size)
            reason = f"~{fmt_rate(item, use)}; {shelf_s}. Get it daily instead of {fmt_qty(item, habit)} at once."
        else:
            qty = round_to_pack(need, item.pack_size)
            reason = f"{_cap(use_src)} uses ~{fmt_rate(item, use)}; {shelf_s}, so buy for {cover:.0f} days{at_home}."
            if shelf_mid < interval - 0.5:
                step = max(1, int(shelf_mid))
                top_day = shop_day + timedelta(days=step)
                top_qty = round_to_pack(use * min(step, interval - cover), item.pack_size)
                if top_qty > EPS:
                    topup = {"day": WEEKDAYS[top_day.weekday()], "date": top_day.isoformat(), "qty": top_qty,
                             "qty_label": fmt_qty(item, top_qty), "every_days": step}
        if extra > EPS:
            names = " and ".join(f"{e['festival']['name_en']} ({e['purpose']})" for e in fest_extra[item_id])
            if festival_shortfall:
                qty = max(round_to_pack(need, item.pack_size), item.pack_size)
                reason = (f"Enough at home for everyday use, but {names} needs ~{fmt_qty(item, extra)} more "
                          f"(~{fmt_qty(item, need)} short).")
            else:
                if not daily:  # festival cooking needs more than the usual pack or the everyday formula
                    qty = max(qty, round_to_pack(need, item.pack_size))
                reason += f" Includes ~{fmt_qty(item, extra)} extra for {names}."
        if "not counting" in use_src and _cap(use_src) not in reason:
            reason = f"{_cap(use_src)}. {reason}"
        # money saved = food the usual purchase would let spoil, minus what this plan lets spoil
        plan_qty = qty * min(interval, shelf_mid) if daily else qty
        saved_qty = max(0.0, spoil_qty(habit + fresh, use, shelf_mid) - spoil_qty(plan_qty + fresh, use, shelf_mid))
        less = item.perishable and habit - qty > item.pack_size * 0.49
        per = " a day" if daily else ""
        buy.append({
            **base, "qty": qty, "qty_label": fmt_qty(item, qty) + per, "price": round(qty * item.price),
            "habit_qty": habit, "habit_label": fmt_qty(item, habit), "less_than_habit": less, "daily": daily,
            "saved_rupees": round(saved_qty * item.price), "saved_kg": round(item.kg(saved_qty), 2),
            "topup": topup, "reason": reason,
            "headline": (f"Buy {fmt_qty(item, qty)}{per}, not {fmt_qty(item, habit)}" if less
                         else f"Buy {fmt_qty(item, qty)}{per}"),
        })
    buy.sort(key=lambda r: (-r["saved_rupees"], not r["perishable"], r["category"]))
    planned = {r["item_id"] for r in buy} | {r["item_id"] for r in skip}
    suggestions = _suggestions(fest_extra, planned, ctx)

    def cost(r):
        return r["price"] + (round(r["topup"]["qty"] * cat[r["item_id"]].price) if r["topup"] else 0)

    return {
        "shop_date": shop_day.isoformat(), "shop_day": WEEKDAYS[shop_day.weekday()],
        "days_to_shop": days_to_shop, "interval": interval,
        "buy": buy, "skip": skip, "total_rupees": sum(cost(r) for r in buy),
        "habit_rupees": sum(round(r["habit_qty"] * cat[r["item_id"]].price) for r in buy),
        "saved_rupees": sum(r["saved_rupees"] for r in buy),
        "saved_kg": round(sum(r["saved_kg"] for r in buy), 2),
        "rule": "quantity = daily use × min(shelf life, days until next shop) − stock at home, rounded to pack size",
        "festivals": [occasions.festival_card(f, ctx.today) for f in fests],
        "suggestions": suggestions,
    }


def _festival_extras(fests, ctx: Ctx) -> dict[str, list[dict]]:
    """Extra quantity per item for the festivals this shop covers: extra per adult-equivalent
    for the whole festival × the family's adult-equivalents, leaving out anyone whose health
    condition excludes the item, and skipping items outside the family's diet."""
    cat = catalogue()
    h = ctx.household
    ae = adult_equivalents(h)
    out: dict[str, list[dict]] = {}
    for f in fests:
        for row in occasions.festival_items().get(f.id, []):
            item = cat.get(row["item_id"])
            if not item or not allowed(item.diet, h.get("diet", "veg")):
                continue
            skip_ae, _ = occasions.skipped_share(h, item.id)
            qty = row["extra_per_ae"] * max(0.0, ae - skip_ae)
            if qty <= EPS:
                continue
            out.setdefault(item.id, []).append({
                "qty": qty, "purpose": row["purpose"],
                "festival": {"id": f.id, "name_en": f.name_en, "name_hi": f.name_hi, "name_mr": f.name_mr,
                             "start": f.start.isoformat(), "end": f.end.isoformat()},
            })
    return out


def _suggestions(fest_extra: dict[str, list[dict]], planned: set[str], ctx: Ctx) -> list[dict]:
    """Things worth adding that the everyday plan doesn't cover: festival items the family
    doesn't usually buy, and alternatives for a family member's health condition."""
    cat = catalogue()
    out = []
    for iid, extras in fest_extra.items():
        if iid in planned:
            continue
        item = cat[iid]
        total = sum(e["qty"] for e in extras)
        qty = round_to_pack(total, item.pack_size) or item.pack_size
        names = " and ".join(f"{e['festival']['name_en']} ({e['purpose']})" for e in extras)
        out.append({**_item_fields(item), "kind": "festival", "qty": qty, "qty_label": fmt_qty(item, qty),
                    "price": round(qty * item.price), "festival": extras[0]["festival"],
                    "reason": f"For {names}: about {fmt_qty(item, total)} for your family."})
    for s in occasions.health_suggestions(ctx.household):
        item = cat.get(s["item_id"])
        if not item or s["item_id"] in planned:
            continue
        who = " and ".join(s["for"])
        out.append({**_item_fields(item), "kind": "health", "qty": item.pack_size,
                    "qty_label": fmt_qty(item, item.pack_size), "price": round(item.pack_size * item.price),
                    "condition": s["condition"], "for": s["for"],
                    "reason": f"{s['note'].capitalize()} for {who}. Sugar and sweets are already planned without "
                              f"{who}'s share. Check with their doctor before changing their diet."
                    if s["condition"] == "diabetes" else f"{s['note'].capitalize()} for {who}."})
    out.sort(key=lambda r: (r["kind"] != "health", r.get("festival", {}).get("start", ""), r["category"]))
    return out


def _cap(s: str) -> str:
    """Capitalise the first letter only (str.capitalize would lowercase names like "Aaji")."""
    return s[:1].upper() + s[1:]


def _item_fields(item) -> dict:
    return {"item_id": item.id, "emoji": item.emoji, "name_en": item.name_en, "name_hi": item.name_hi,
            "name_mr": item.name_mr, "category": item.category, "unit": item.unit, "pack_size": item.pack_size}


def recipe_consumption(recipe: dict, inv: list[dict], ctx: Ctx) -> list[dict]:
    """What cooking this dish for the family takes out of the pantry: the recipe amount
    per adult-equivalent × the family's adult-equivalents, capped at what is at home.
    Prefilled for the family to confirm; nothing is deducted until they do."""
    cat = catalogue()
    ae = adult_equivalents(ctx.household)
    stock = {r["item_id"]: r for r in inv if r["status"] not in ("finished",)}
    out = []
    for iid, per_ae in recipe["ingredients"].items():
        it = cat[iid]
        need = per_ae * ae
        have = stock[iid]["remaining"] if iid in stock else 0.0
        take = min(need, have)
        out.append({"item_id": iid, "name_en": it.name_en, "name_hi": it.name_hi, "name_mr": it.name_mr,
                    "unit": it.unit, "pack_size": it.pack_size, "qty": round(take, 3),
                    "qty_label": fmt_qty(it, take) if take > EPS else "none",
                    "recipe_qty": round(need, 3), "recipe_label": fmt_qty(it, need),
                    "have": round(have, 3), "have_label": fmt_qty(it, have) if have > EPS else "none at home"})
    return out


SHOP_ORDER = {"buy": 0, "buy_less": 0, "manual": 1, "have": 2}


def shopping_view(plan: dict, overrides: dict[str, dict], bought: dict[str, dict]) -> dict:
    """The buy-right plan merged with the family's edits for this trip.

    status: buy (the plan says buy) · buy_less (buy, and less than last time) ·
            have (enough at home: skip) · manual (added by the family).
    `excluded` (not this trip) and `purchased` are flags on top of the status.
    """
    cat = catalogue()
    rows, seen = [], set()

    def row(r: dict, status: str) -> dict:
        iid = r["item_id"]
        it = cat[iid]
        o = overrides.get(iid, {})
        planned = r.get("qty", 0.0)
        qty = o["qty"] if o.get("qty") is not None else (planned or it.pack_size)
        per = " a day" if r.get("daily") else ""
        b = bought.get(iid)
        edited = o.get("qty") is not None and abs(o["qty"] - planned) > 1e-6
        reason = r["reason"]
        if status == "manual":
            reason = "Added by you. " + reason
        return {
            **r, "status": status, "planned_qty": planned, "qty": round(qty, 3),
            "qty_label": fmt_qty(it, qty) + per, "price": round(qty * it.price), "edited": edited,
            "headline": (f"Buy {fmt_qty(it, qty)}{per}" if edited or status == "manual"
                         else r.get("headline") or "Skip this trip"),
            "excluded": bool(o.get("excluded")), "purchased": b is not None,
            "purchased_qty": round(b["qty"], 3) if b else None,
            "purchased_label": fmt_qty(it, b["qty"]) if b else None,
        }

    for r in plan["buy"]:
        rows.append(row(r, "buy_less" if r["less_than_habit"] else "buy"))
        seen.add(r["item_id"])
    for r in plan["skip"]:
        manual = overrides.get(r["item_id"], {}).get("manual")
        rows.append(row({**r, "qty": 0.0, "daily": False}, "manual" if manual else "have"))
        seen.add(r["item_id"])
    for iid in list(overrides) + list(bought):  # added or bought, but outside the plan (e.g. not in this diet)
        if iid in seen or iid not in cat or not (overrides.get(iid, {}).get("manual") or iid in bought):
            continue
        it = cat[iid]
        seen.add(iid)
        rows.append(row({"item_id": iid, "emoji": it.emoji, "name_en": it.name_en, "name_hi": it.name_hi,
                         "name_mr": it.name_mr, "category": it.category, "unit": it.unit, "tier": it.tier,
                         "pack_size": it.pack_size, "unit_price": it.price, "perishable": it.perishable,
                         "stock_now_label": "—", "need_label": "—", "qty": 0.0, "daily": False,
                         "reason": "Not part of your family's usual plan."}, "manual"))
    rows.sort(key=lambda r: (SHOP_ORDER[r["status"]], r["excluded"]))
    to_buy = [r for r in rows if r["status"] != "have" and not r["excluded"]]
    on_list = {r["item_id"] for r in rows}
    return {
        "suggestions": [x for x in plan.get("suggestions", []) if x["item_id"] not in on_list],
        "rows": rows,
        "counts": {"to_buy": len(to_buy), "bought": sum(r["purchased"] for r in rows),
                   "have": sum(r["status"] == "have" for r in rows), "excluded": sum(r["excluded"] for r in rows)},
        "list_rupees": sum(r["price"] for r in to_buy),
    }
