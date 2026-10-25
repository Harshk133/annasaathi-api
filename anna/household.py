"""The family profile: validated input, age groups and the options the setup
screens offer. Everything AnnaSaathi plans starts from this record."""
from __future__ import annotations

import csv
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .data import DATA_DIR

Group = Literal["toddler", "child", "teen", "adult", "elder"]
Diet = Literal["veg", "jain", "egg", "nonveg"]
Portion = Literal["light", "normal", "hearty"]
Prep = Literal["roasted", "fried", "wet", "sprouts", "raw"]

PREP_LABELS = {
    "roasted": "Roasted / dry (bhakri, thalipeeth)",
    "fried": "Fried (bhaji, puri)",
    "wet": "Dal, bhaji, rice, curry",
    "sprouts": "Sprouts (usal)",
    "raw": "Salads, fruit",
}


@lru_cache
def needs() -> dict[str, dict]:
    """Per-member preferences that change quantities or dishes (data/needs.csv)."""
    with open(DATA_DIR / "needs.csv", encoding="utf-8") as f:
        return {r["id"]: {**r, "boost": float(r["boost"]) if r["boost"] else 0.0} for r in csv.DictReader(f)}


def group_for_age(age: int) -> Group:
    if age < 5:
        return "toddler"
    if age < 13:
        return "child"
    if age < 20:
        return "teen"
    if age < 60:
        return "adult"
    return "elder"


class Member(BaseModel):
    name: str = Field(min_length=1, max_length=30)
    age: int = Field(ge=0, le=110)
    group: Group | None = None
    needs: list[str] = []
    conditions: list[str] = []  # family-entered health conditions (planning only, not medical advice)
    note: str = Field(default="", max_length=80)

    @field_validator("name")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("name can't be empty")
        return v

    @field_validator("needs")
    @classmethod
    def _known_needs(cls, v: list[str]) -> list[str]:
        unknown = [n for n in v if n not in needs()]
        if unknown:
            raise ValueError(f"unknown needs: {', '.join(unknown)}")
        return list(dict.fromkeys(v))

    @field_validator("conditions")
    @classmethod
    def _known_conditions(cls, v: list[str]) -> list[str]:
        from .occasions import conditions

        unknown = [c for c in v if c not in conditions()]
        if unknown:
            raise ValueError(f"unknown health conditions: {', '.join(unknown)}")
        return list(dict.fromkeys(v))

    @model_validator(mode="after")
    def _group_from_age(self):
        # the age decides the portion group; a stated group is only kept if it agrees
        self.group = group_for_age(self.age)
        return self


class Household(BaseModel):
    family_name: str = Field(min_length=1, max_length=40)
    city: str = "Pune"
    lat: float = 18.5204
    lon: float = 73.8567
    language: Literal["en", "hi", "mr"] = "en"
    diet: Diet = "veg"
    portion: Portion = "normal"
    cooked_meals_per_day: int = Field(default=3, ge=1, le=4)
    eat_out_meals_per_week: int = Field(default=0, ge=0)
    shop_interval_days: int = Field(default=7, ge=2, le=14)
    has_fridge: bool = True
    prep_styles: list[Prep] = Field(default_factory=lambda: ["wet", "roasted"], min_length=1)
    members: list[Member] = Field(min_length=1, max_length=12)
    festivals: list[str] = []  # festivals the family celebrates; the list buys extra for them
    setup_seconds: float | None = None

    @field_validator("festivals")
    @classmethod
    def _known_festivals(cls, v: list[str]) -> list[str]:
        from .occasions import festival_options

        known = {f["id"] for f in festival_options()}
        unknown = [f for f in v if f not in known]
        if unknown:
            raise ValueError(f"unknown festivals: {', '.join(unknown)}")
        return list(dict.fromkeys(v))

    @model_validator(mode="after")
    def _eat_out_fits(self):
        if self.eat_out_meals_per_week > self.cooked_meals_per_day * 7:
            raise ValueError("more meals eaten out than meals in the week")
        self.prep_styles = list(dict.fromkeys(self.prep_styles))
        return self


def options() -> dict:
    from .occasions import conditions as _conditions, festival_options

    """What the setup screens can offer, straight from the data tables."""
    return {
        "groups": [
            {"id": "toddler", "label": "Toddler", "ages": "0–4", "default_age": 3},
            {"id": "child", "label": "Child", "ages": "5–12", "default_age": 9},
            {"id": "teen", "label": "Teen", "ages": "13–19", "default_age": 15},
            {"id": "adult", "label": "Adult", "ages": "20–59", "default_age": 35},
            {"id": "elder", "label": "Elder", "ages": "60+", "default_age": 68},
        ],
        "diets": [
            {"id": "veg", "label": "Vegetarian"}, {"id": "jain", "label": "Jain (no onion, garlic, potato)"},
            {"id": "egg", "label": "Veg + eggs"}, {"id": "nonveg", "label": "Non-veg"},
        ],
        "portions": [{"id": "light", "label": "Light"}, {"id": "normal", "label": "Normal"}, {"id": "hearty", "label": "Hearty"}],
        "prep_styles": [{"id": k, "label": v} for k, v in PREP_LABELS.items()],
        "needs": [{"id": n["id"], "label": n["label"], "kind": n["kind"]} for n in needs().values()],
        "conditions": [{"id": c["id"], "label": c["label_en"], "label_hi": c["label_hi"], "label_mr": c["label_mr"]}
                       for c in _conditions().values()],
        "festivals": festival_options(),
    }
