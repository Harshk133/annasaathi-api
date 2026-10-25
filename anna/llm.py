"""Extra dish ideas from an LLM through OpenRouter (OPENROUTER_API_KEY; model OPENROUTER_MODEL).

The model only proposes dishes and per-person amounts. Every ingredient is matched to the
catalogue and checked against the kitchen by our own code, so "in your kitchen" and "to buy"
are never the model's word. Without a key this module reports itself unavailable.
"""
from __future__ import annotations

import json
import os
import re

import httpx
from pydantic import BaseModel, Field, ValidationError

from .extract import ExtractionUnavailable, match_item

URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemini-2.5-flash"


def configured() -> bool:
    return bool(os.environ.get("OPENROUTER_API_KEY"))


def model() -> str:
    return os.environ.get("OPENROUTER_MODEL") or DEFAULT_MODEL


class IdeaIngredient(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    amount_per_person: float = Field(ge=0, le=2000)
    unit: str = Field(pattern=r"^(g|ml|pc)$")


class Idea(BaseModel):
    name: str = Field(min_length=2, max_length=60)
    why: str = Field(default="", max_length=240)
    time_min: int | None = Field(default=None, ge=5, le=240)
    ingredients: list[IdeaIngredient] = Field(min_length=1, max_length=14)
    steps: list[str] = Field(default_factory=list, max_length=8)


class Ideas(BaseModel):
    dishes: list[Idea] = Field(min_length=1, max_length=5)


def _prompt(kitchen: list[str], heads: int, meal: str, diet: str, avoid: list[str], care: list[str]) -> str:
    return (
        "You plan home-cooked Indian meals. Suggest 3 dishes for "
        f"{meal} for {heads} people (diet: {diet}). Use mainly what is already in this kitchen:\n"
        + "\n".join(f"- {k}" for k in kitchen)
        + ("\nDo not suggest these cooking styles: " + ", ".join(avoid) + "." if avoid else "")
        + ("\nSome people eating need care with: " + "; ".join(care) + ". Prefer dishes that suit them." if care else "")
        + "\nGive ingredient amounts per person (unit g, ml or pc). Keep steps short and practical."
        "\nReply with JSON only, exactly this shape:\n"
        '{"dishes":[{"name":"...","why":"one sentence: what it uses from the kitchen","time_min":30,'
        '"ingredients":[{"name":"paneer","amount_per_person":60,"unit":"g"}],"steps":["..."]}]}'
    )


def _json_from(text: str) -> dict:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1)
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start:end + 1])


def ideas(kitchen: list[str], heads: int, meal: str, diet: str, avoid: list[str], care: list[str]) -> list[dict]:
    """Dish ideas with ingredients mapped to catalogue ids: [{name, why, time_min, steps, ingredients:
    [{item_id|None, name, per_person (in the item's unit)}]}]. Raises ExtractionUnavailable."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise ExtractionUnavailable("no OpenRouter key configured")
    try:
        r = httpx.post(URL, timeout=45, headers={"Authorization": f"Bearer {key}", "X-Title": "AnnaSaathi",
                                                  "HTTP-Referer": "https://annasaathi-api.onrender.com"},
                       json={"model": model(), "temperature": 0.4, "response_format": {"type": "json_object"},
                             "messages": [{"role": "user", "content": _prompt(kitchen, heads, meal, diet, avoid, care)}]})
    except httpx.HTTPError as e:
        raise ExtractionUnavailable("couldn't reach OpenRouter") from e
    if r.status_code in (401, 403):
        raise ExtractionUnavailable("OpenRouter refused the key")
    if r.status_code == 402:
        raise ExtractionUnavailable("the OpenRouter account has no credit left")
    if r.status_code >= 400:
        raise ExtractionUnavailable(f"OpenRouter error {r.status_code}")
    try:
        content = r.json()["choices"][0]["message"]["content"]
        parsed = Ideas.model_validate(_json_from(content))
    except (KeyError, IndexError, ValueError, ValidationError) as e:
        raise ExtractionUnavailable("the AI answer wasn't in the expected format") from e

    from .data import catalogue

    cat = catalogue()
    out = []
    for d in parsed.dishes:
        ings = []
        for ing in d.ingredients:
            iid, _ = match_item(ing.name)
            per = ing.amount_per_person
            if iid:
                unit = cat[iid].unit
                per = per / 1000 if unit in ("kg", "L") and ing.unit in ("g", "ml") else per
                if unit == "pc" and ing.unit in ("g", "ml") and cat[iid].pc_kg:
                    per = per / 1000 / cat[iid].pc_kg
            ings.append({"item_id": iid, "name": ing.name, "per_person": per})
        out.append({"name": d.name, "why": d.why, "time_min": d.time_min, "steps": d.steps, "ingredients": ings})
    return out
