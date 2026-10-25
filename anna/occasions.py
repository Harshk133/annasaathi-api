"""Festivals and family health: the two things that change *what* a family needs this week.

Both are data, not code: festivals.csv (dated rows, approximate for lunar festivals; add next
year's rows to keep it current), festival_items.csv (extra per adult-equivalent for the whole
festival), health.csv (what a condition does to the plan) and conditions.csv (labels).

Health conditions are only family-entered planning preferences. They leave a person's share out
of an item, label items "high in sugar / salt / fat", avoid a cooking style, or suggest an
alternative. They never prescribe anything; the app tells families to follow their doctor.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, timedelta
from functools import lru_cache

from .data import DATA_DIR, factors

LEAD_DAYS = 3  # festival cooking starts a few days before the day itself


def _rows(name: str) -> list[dict]:
    with open(DATA_DIR / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


@dataclass(frozen=True)
class Festival:
    id: str
    name_en: str
    name_hi: str
    name_mr: str
    start: date
    end: date
    note: str


@lru_cache
def festivals() -> list[Festival]:
    return sorted((Festival(r["id"], r["name_en"], r["name_hi"], r["name_mr"], date.fromisoformat(r["start"]),
                            date.fromisoformat(r["end"]), r["note"]) for r in _rows("festivals.csv")),
                  key=lambda f: f.start)


@lru_cache
def festival_items() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in _rows("festival_items.csv"):
        out.setdefault(r["festival_id"], []).append({"item_id": r["item_id"], "extra_per_ae": float(r["extra_per_ae"]),
                                                     "purpose": r["purpose"]})
    return out


def festival_options() -> list[dict]:
    seen = {}
    for f in festivals():
        seen.setdefault(f.id, {"id": f.id, "label": f.name_en, "label_hi": f.name_hi, "label_mr": f.name_mr})
    return list(seen.values())


def upcoming(h: dict, today: date, days: int = 60) -> list[dict]:
    """The family's festivals that are on now or start within `days`."""
    chosen = set(h.get("festivals") or [])
    out = []
    for f in festivals():
        if f.id in chosen and f.end >= today and f.start <= today + timedelta(days=days):
            out.append(festival_card(f, today))
    return out


def festival_card(f: Festival, today: date) -> dict:
    return {"id": f.id, "name_en": f.name_en, "name_hi": f.name_hi, "name_mr": f.name_mr, "start": f.start.isoformat(),
            "end": f.end.isoformat(), "days_away": max(0, (f.start - today).days), "on_now": f.start <= today <= f.end,
            "note": f.note}


def festivals_for_shop(h: dict, shop_day: date, interval: int) -> list[Festival]:
    """Festivals this shop should buy for: on now, or starting before the following shop (plus cooking lead time)."""
    chosen = set(h.get("festivals") or [])
    horizon = shop_day + timedelta(days=interval + LEAD_DAYS)
    return [f for f in festivals() if f.id in chosen and f.end >= shop_day and f.start <= horizon]


# ---------------------------------------------------------------- health

@lru_cache
def conditions() -> dict[str, dict]:
    return {r["id"]: r for r in _rows("conditions.csv")}


@lru_cache
def health_rules() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in _rows("health.csv"):
        out.setdefault(r["condition"], []).append({"effect": r["effect"], "items": [i for i in r["items"].split("|") if i],
                                                   "note": r["note"]})
    return out


def member_ae(h: dict, m: dict) -> float:
    """One member's share of the family's adult-equivalents (same factors as engine.adult_equivalents)."""
    f = factors()
    ae = f.get(f'member:{m["group"]}', 1.0) * f.get(f'portion:{h.get("portion", "normal")}', 1.0)
    weekly = h.get("cooked_meals_per_day", 3) * 7
    return ae * (weekly - h.get("eat_out_meals_per_week", 0)) / weekly


def _members_with(h: dict, item_id: str, effect: str) -> list[tuple[dict, str, str]]:
    """(member, condition, note) for members whose condition applies `effect` to this item."""
    out = []
    for m in h["members"]:
        for c in m.get("conditions") or []:
            for rule in health_rules().get(c, []):
                if rule["effect"] == effect and item_id in rule["items"]:
                    out.append((m, c, rule["note"]))
    return out


def skipped_share(h: dict, item_id: str) -> tuple[float, list[str]]:
    """Adult-equivalents to leave out of an item, and who they are ("Aaji (diabetes)")."""
    hits = {m["name"]: (m, c) for m, c, _ in _members_with(h, item_id, "skip_share")}
    ae = sum(member_ae(h, m) for m, _ in hits.values())
    return ae, [f"{name} ({conditions()[c]['label_en'].split(' (')[0].lower()})" for name, (_, c) in hits.items()]


def health_flags(h: dict, item_id: str) -> list[dict]:
    flags = []
    for m, c, note in _members_with(h, item_id, "flag"):
        flags.append({"member": m["name"], "condition": c, "condition_label": conditions()[c]["label_en"], "note": note})
    return flags


def avoided_preps(h: dict) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for m in h["members"]:
        for c in m.get("conditions") or []:
            for rule in health_rules().get(c, []):
                if rule["effect"] == "avoid_prep":
                    for prep in rule["items"]:
                        out.setdefault(prep, []).append(m["name"])
    return out


def health_suggestions(h: dict) -> list[dict]:
    """Alternatives worth offering, once per item, naming who they're for."""
    out: dict[str, dict] = {}
    for m in h["members"]:
        for c in m.get("conditions") or []:
            for rule in health_rules().get(c, []):
                if rule["effect"] == "suggest":
                    for iid in rule["items"]:
                        s = out.setdefault(iid, {"item_id": iid, "for": [], "condition": c, "note": rule["note"]})
                        s["for"].append(m["name"])
    return list(out.values())
