"""Hand-checked examples for the planning formulas (numbers worked out on paper)."""
from datetime import date, timedelta

import pytest

from anna import engine, extract, planner, simulator
from anna.data import catalogue, demo_household, sample_bills

TODAY = date(2026, 10, 24)
COOL = {"temp_c": 20, "humidity": 50}


def one_adult(**kw):
    return {"members": [{"name": "A", "group": "adult", "needs": []}], "portion": "normal", "diet": "veg",
            "cooked_meals_per_day": 3, "eat_out_meals_per_week": 0, "shop_interval_days": 7, **kw}


def lot(id, item_id, qty, days_ago=0, storage=None):
    d = TODAY - timedelta(days=days_ago)
    return {"id": id, "item_id": item_id, "qty": qty, "purchased_on": d, "storage": storage, "printed_expiry": None,
            "status": "active", "anchor_qty": qty, "anchor_date": d}


def test_adult_equivalents_patil_family():
    # 1 + 1 + 0.85 (elder) + 1.1 (teen) + 0.6 (child) = 4.55, × (21 − 2 eaten out) / 21
    assert engine.adult_equivalents(demo_household()) == pytest.approx(4.55 * 19 / 21)


def test_shelf_life_hot_counter():
    # tomato tier D5 = 4–6 days; 33 °C on the counter × 0.8 → 3.2–4.8
    c = engine.Ctx(one_adult(), TODAY, {"temp_c": 33, "humidity": 50})
    lo, hi, notes = engine.shelf_range(catalogue()["tomato"], "pantry", c)
    assert (lo, hi) == pytest.approx((3.2, 4.8))
    assert "33°C on the counter" in notes[0]


def test_fridge_does_not_change_with_weather():
    c = engine.Ctx(one_adult(), TODAY, {"temp_c": 38, "humidity": 50})
    lo, hi, _ = engine.shelf_range(catalogue()["palak"], "fridge", c)
    assert (lo, hi) == (2, 3)


def test_project_remaining_eats_oldest_first():
    # 1 kg bought 4 days ago, then 1 kg 2 days ago, 0.1 kg/day → the old lot loses 0.4, the new one nothing
    lots = [lot(1, "tomato", 1, 4), lot(2, "tomato", 1, 2)]
    rem = engine.project_remaining(lots, 0.1, TODAY)
    assert rem[1] == pytest.approx(0.6) and rem[2] == pytest.approx(1.0)


def test_forecast_waste():
    # 1 kg spoils on day 2, 1 kg on day 10; eating 0.25/day → 0.5 of the first spoils, the second is safe
    assert engine.forecast_waste([(1, 2), (1, 10)], 0.25) == pytest.approx([0.5, 0])


def test_urgency_palak_overbought():
    # 3 bunches, one adult eats 0.1/day, keeps 2–3 days (mid 2.5) → 0.25 eaten, 2.75 at risk
    c = engine.Ctx(one_adult(), TODAY, COOL)
    row = engine.inventory([lot(1, "palak", 3)], c)[0]
    assert row["at_risk"] == pytest.approx(2.75)
    assert row["status"] == "use_soon"
    assert row["urgency"] == pytest.approx(round(2.75 / 3, 3))  # waste share beats 1/(1+2 days)


def test_spoiled_item_is_flagged():
    c = engine.Ctx(one_adult(), TODAY, COOL)
    row = engine.inventory([lot(1, "bhindi", 0.5, days_ago=4)], c)[0]
    assert row["status"] == "spoiled" and "when in doubt, throw it out" in row["reason"]


def test_round_to_pack():
    assert planner.round_to_pack(0.74, 0.5) == 1.0  # ceil(1.48 − 0.2) = 2 packs
    assert planner.round_to_pack(0.1, 0.5) == 0.0  # under 30% of a pack: skip
    assert planner.round_to_pack(0.55, 0.5) == 0.5  # 1.1 packs rounds down, no over-buying


def test_buy_right_tomatoes():
    # one adult: 0.045 kg/day; keeps 4–6 days (mid 5) at 20 °C; next shop in 7 days
    # → buy for min(5, 7) = 5 days = 0.225 kg → one 500 g pack, not the usual 2 kg
    c = engine.Ctx(one_adult(), TODAY, COOL)
    out = planner.buy_right([lot(1, "tomato", 2)], c)
    row = out["buy"][0]
    assert out["days_to_shop"] == 7
    assert row["qty"] == 0.5
    assert row["headline"] == "Buy 500 g, not 2 kg"
    assert row["topup"] is None  # 5-day shelf life covers most of the week: (7 − 5) × 0.045 < 30% of a pack
    # usual 2 kg would let 2 − 0.045 × 5 = 1.775 kg spoil; the plan lets 0.275 kg spoil → 1.5 kg × ₹40
    assert row["saved_rupees"] == 60


def test_buy_right_skips_items_in_stock():
    # 2 kg onions bought today, 0.055 kg/day, keeps 12–18 days → ~1.6 kg still fresh on shop day
    c = engine.Ctx(one_adult(), TODAY, COOL)
    out = planner.buy_right([lot(1, "onion", 2)], c)
    assert out["buy"] == [] and out["skip"][0]["item_id"] == "onion"


def test_long_life_keeps_usual_pack():
    c = engine.Ctx(one_adult(), TODAY, COOL)
    out = planner.buy_right([lot(1, "rice", 5, days_ago=60)], c)
    row = out["buy"][0]
    assert row["qty"] == 5 and "not" not in row["headline"]


def test_cook_first_rescues_at_risk_and_respects_diet():
    c = engine.Ctx(demo_household(), TODAY, COOL)
    inv = engine.inventory([lot(1, "palak", 3), lot(2, "paneer", 0.4), lot(3, "wheat_atta", 5), lot(4, "eggs", 12)], c)
    dishes = planner.cook_first(inv, c)
    assert any(u["item_id"] == "palak" for d in dishes for u in d["uses"])
    assert all(d["recipe_id"] != "tomato_omelette" for d in dishes)  # Patils are vegetarian


def test_bill_lines_match_catalogue():
    lines = [l for b in sample_bills().values() for l in b["lines"]]
    hits = sum(extract.match_item(l["raw"])[0] == l["item_id"] for l in lines)
    assert hits / len(lines) >= 0.9, hits


def test_voice_marathi_hindi_english():
    got = {r["item_id"]: r["qty"] for r in extract.parse_voice("दोन किलो कांदा आणि 2 judi palak and half kg tomato")}
    assert got == {"onion": 2, "palak": 2, "tomato": 0.5}


def test_simulator_is_reproducible_and_helps():
    a = simulator.simulate(simulator.make_households()[0], 0.7, 1)
    b = simulator.simulate(simulator.make_households()[0], 0.7, 1)
    assert a["waste_kg"] == b["waste_kg"]
    r = simulator.results()
    base, lv = r["baseline"], r["levels"]
    assert lv["100"]["kg"] < lv["70"]["kg"] < lv["50"]["kg"] < base["kg"]
    assert all(lv[p]["shortages"] <= base["shortages"] for p in ("50", "70", "100"))
