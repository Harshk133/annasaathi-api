"""Loads the item catalogue, shelf-life tiers, factors and recipes.

Everything the planner knows about food lives in api/data as CSV/JSON, so a new
item, region or language is a data edit, not a code change.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass(frozen=True)
class Item:
    id: str
    emoji: str
    name_en: str
    name_hi: str
    name_mr: str
    category: str
    form: str  # dry | fresh | wet | liquid
    tier: str
    storage: str  # pantry | fridge
    unit: str  # kg | L | pc
    pack_size: float
    price: float  # ₹ per unit
    use_per_ae: float  # unit per adult-equivalent per day
    diet: str  # all | nojain | egg | meat
    aliases: tuple[str, ...]
    pc_kg: float | None
    pc_word: tuple[str, str] | None = None  # (singular, plural) for piece items, e.g. ("pack", "packs")
    sim: bool = False  # part of the simulator's typical basket

    def kg(self, qty: float) -> float:
        """Weight in kg, used for waste totals."""
        return qty * self.pc_kg if self.unit == "pc" and self.pc_kg else qty

    @property
    def perishable(self) -> bool:
        return self.tier.startswith("D")


@dataclass(frozen=True)
class Tier:
    tier: str
    label: str
    min_days: float
    max_days: float
    examples: str
    source: str


def _rows(name: str) -> list[dict]:
    with open(DATA_DIR / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


@lru_cache
def catalogue() -> dict[str, Item]:
    items = {}
    for r in _rows("catalogue.csv"):
        items[r["id"]] = Item(
            id=r["id"], emoji=r["emoji"], name_en=r["name_en"], name_hi=r["name_hi"],
            name_mr=r["name_mr"], category=r["category"], form=r["form"], tier=r["tier"],
            storage=r["storage"], unit=r["unit"], pack_size=float(r["pack_size"]),
            price=float(r["price"]), use_per_ae=float(r["use_per_ae"]), diet=r["diet"],
            aliases=tuple(a.strip().lower() for a in r["aliases"].split("|") if a.strip()),
            pc_kg=float(r["pc_kg"]) if r.get("pc_kg") else None,
            pc_word=tuple(r["pc_word"].split("|")) if r.get("pc_word") else None,
            sim=r.get("sim") == "1",
        )
    return items


@lru_cache
def tiers() -> dict[str, Tier]:
    return {
        r["tier"]: Tier(r["tier"], r["label"], float(r["min_days"]), float(r["max_days"]), r["examples"], r["source"])
        for r in _rows("shelf_life_tiers.csv")
    }


@lru_cache
def cooked_tiers() -> dict[str, dict]:
    out = {}
    for r in _rows("cooked_tiers.csv"):
        out[r["prep"]] = {**r, "room_hours": float(r["room_hours"]), "fridge_hours": float(r["fridge_hours"])}
    return out


@lru_cache
def factors() -> dict[str, float]:
    return {f'{r["kind"]}:{r["key"]}': float(r["factor"]) for r in _rows("factors.csv")}


@lru_cache
def recipes() -> list[dict]:
    return json.loads((DATA_DIR / "recipes.json").read_text(encoding="utf-8"))


@lru_cache
def sample_bills() -> dict[str, dict]:
    return {b["id"]: b for b in json.loads((DATA_DIR / "sample_bills.json").read_text(encoding="utf-8"))}


def demo_household() -> dict:
    return json.loads((DATA_DIR / "demo_household.json").read_text(encoding="utf-8"))


UNIT_WORDS = {"leafy": ("bunch", "bunches"), "bakery": ("loaf", "loaves")}


def pc_words(item: Item) -> tuple[str, str]:
    return item.pc_word or UNIT_WORDS.get(item.category, ("pc", "pc"))


def fmt_qty(item: Item, qty: float) -> str:
    """Human quantity: 0.25 kg -> '250 g', 3 pc of palak -> '3 bunches'."""
    if item.unit in ("kg", "L"):
        small = "g" if item.unit == "kg" else "ml"
        if qty < 1:
            v = round(qty * 1000 / 50) * 50 if qty >= 0.1 else round(qty * 1000 / 5) * 5
            return f"{max(v, 5):g} {small}"
        return f"{round(qty * 4) / 4:g} {item.unit}"
    n = round(qty * 2) / 2 if qty < 3 else round(qty)
    one, many = pc_words(item)
    return f"{n:g} {one if n == 1 else many}"


def fmt_rate(item: Item, per_day: float) -> str:
    """Daily use: '200 g/day', or '1 loaf every 4 days' for slow piece items."""
    if item.unit == "pc" and 0 < per_day < 0.75:
        return f"1 {pc_words(item)[0]} every {1 / per_day:.0f} days"
    return f"{fmt_qty(item, per_day)}/day"


def fmt_days(lo: float, hi: float) -> str:
    a, b = round(lo), round(hi)
    if a == b:
        return f"{lo:.1f}".rstrip("0").rstrip(".") if lo < 3 else f"{a}"
    return f"{a}–{b}"
