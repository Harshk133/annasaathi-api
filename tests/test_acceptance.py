"""PRD acceptance criteria, checked through the HTTP API on a throwaway database.

Family profile — "Set up in under 2 minutes; drives all quantities"
Inventory      — "Every item has a tier and a range; sorted by urgency",
                 with the range adjusted for storage and weather
"""
import copy

import pytest
from fastapi.testclient import TestClient

from anna import db
from anna.data import demo_household


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setenv("ANNA_DEMO", "1")  # fixed 31 °C, no network
    monkeypatch.setenv("ANNA_TODAY", "2026-10-24")
    from anna.main import app

    c = TestClient(app)
    c.post("/api/demo/reset")
    return c


def profile(**changes):
    h = {k: v for k, v in demo_household().items() if k != "seed_bills"}
    h = copy.deepcopy(h)
    h.update(changes)
    return h


def scan_todays_bill(api):
    s = api.post("/api/bills/scan", data={"sample_id": "ganesh_kirana_oct24"}).json()
    lines = [{"item_id": l["item_id"], "qty": l["qty"], "price": l["price"], "raw": l["raw"]} for l in s["lines"]]
    api.post("/api/bills/confirm", json={"bill_id": s["bill_id"], "shop": s["shop"], "lines": lines})


def buy(api, item_id):
    return next((r for r in api.get("/api/shopping").json()["buy"] if r["item_id"] == item_id), None)


# ---------------------------------------------------------------- family profile


def test_minimal_setup_is_a_name_and_the_people(api):
    # the fewest answers a family must give; everything else has a sensible default
    r = api.post("/api/household/setup", json={"household": {
        "family_name": "Shah", "members": [{"name": "Asha", "age": 34}, {"name": "Dev", "age": 6}]}})
    assert r.status_code == 200
    h = r.json()
    assert [m["group"] for m in h["members"]] == ["adult", "child"]
    assert h["adult_equivalents"] == pytest.approx(1.6)  # 1.0 + 0.6, normal portions, every meal at home


@pytest.mark.parametrize("bad", [
    {"members": []},
    {"members": [{"name": "A", "age": 130}]},
    {"members": [{"name": "  ", "age": 30}]},
    {"diet": "keto"},
    {"cooked_meals_per_day": 2, "eat_out_meals_per_week": 15},
    {"prep_styles": []},
    {"members": [{"name": "A", "age": 30, "needs": ["chocolate"]}]},
])
def test_profile_rejects_bad_input(api, bad):
    assert api.put("/api/household", json=profile(**bad)).status_code == 422


def test_age_decides_the_portion_group(api):
    h = profile(members=[{"name": "Nani", "age": 72, "group": "teen"}, {"name": "Baby", "age": 2}])
    out = api.post("/api/household/preview", json=h).json()
    assert [m["group"] for m in out["members"]] == ["elder", "toddler"]
    assert out["adult_equivalents"] == pytest.approx((0.85 + 0.3) * 19 / 21, abs=0.01)


def test_more_people_means_more_to_buy(api):
    scan_todays_bill(api)
    before = buy(api, "tomato")["qty"]
    h = profile()
    h["members"] += [{"name": f"Guest {i}", "age": 30} for i in range(4)]
    api.put("/api/household", json=h)
    assert buy(api, "tomato")["qty"] > before


def test_meals_eaten_out_reduce_quantities(api):
    a = api.post("/api/household/preview", json=profile(eat_out_meals_per_week=0)).json()["adult_equivalents"]
    b = api.post("/api/household/preview", json=profile(eat_out_meals_per_week=7)).json()["adult_equivalents"]
    assert b == pytest.approx(a * 14 / 21, abs=0.01)


def test_jain_diet_drops_onion_and_potato(api):
    scan_todays_bill(api)
    api.put("/api/household", json=profile(diet="jain"))
    shop = api.get("/api/shopping").json()
    listed = {r["item_id"] for r in shop["buy"] + shop["skip"]}
    assert not {"onion", "potato", "garlic"} & listed
    from anna.data import recipes
    nojain = {r["id"] for r in recipes() if r["diet"] != "all"}
    assert not {d["recipe_id"] for d in api.get("/api/plan/today").json()["dishes"]} & nojain


def test_cooking_style_shapes_the_plan(api):
    scan_todays_bill(api)
    api.put("/api/household", json=profile(prep_styles=["wet"], members=[
        {"name": m["name"], "age": m["age"], "needs": []} for m in demo_household()["members"]]))
    preps = {d["prep"] for d in api.get("/api/plan/today").json()["dishes"]}
    assert preps <= {"wet", "raw"}


def test_member_needs_change_quantities_and_dishes(api):
    scan_todays_bill(api)
    plain = [{"name": m["name"], "age": m["age"], "needs": []} for m in demo_household()["members"]]
    milky = copy.deepcopy(plain)
    milky[0]["needs"] = ["extra_milk"]
    a = api.post("/api/household/preview", json=profile(members=plain)).json()["examples"]
    b = api.post("/api/household/preview", json=profile(members=milky)).json()["examples"]
    milk = lambda rows: next(r for r in rows if r["item_id"] == "milk")["per_day"]
    assert milk(a) != milk(b)
    # Aaji's "prefers roasted" puts a roasted dish on the plan and says why
    dishes = api.get("/api/plan/today").json()["dishes"]
    assert any(d["prep"] == "roasted" and "Aaji" in d["reason"] for d in dishes)


def test_no_fridge_means_everything_on_the_counter(api):
    api.post("/api/household/setup", json={"household": profile(has_fridge=False)})
    api.post("/api/inventory/add", json={"item_id": "palak", "qty": 2})
    palak = next(i for i in api.get("/api/inventory").json()["items"] if i["item_id"] == "palak")
    assert palak["storage"] == "pantry"
    assert palak["shelf_life"][1] < 3  # 2–3 days in a fridge, less out of it
    assert api.post("/api/inventory/add", json={"item_id": "milk", "qty": 1, "storage": "fridge"}).status_code == 400


def test_fresh_setup_starts_an_empty_pantry(api):
    api.post("/api/household/setup", json={"household": profile(family_name="Kulkarni"), "fresh": True})
    assert api.get("/api/inventory").json()["items"] == []
    assert api.get("/api/household").json()["family_name"] == "Kulkarni"


# ---------------------------------------------------------------- inventory


def test_every_item_has_a_tier_and_a_range_sorted_by_urgency(api):
    scan_todays_bill(api)
    items = api.get("/api/inventory").json()["items"]
    assert len(items) >= 20
    for i in items:
        assert i["tier"] and i["tier_label"]
        lo, hi = i["days_left"]
        assert 0 <= lo <= hi
        assert i["shelf_life"][0] <= i["shelf_life"][1] and i["shelf_life"][1] > 0
    urg = [i["urgency"] for i in items]
    assert urg == sorted(urg, reverse=True)


def tomato(api):
    return next(i for i in api.get("/api/inventory").json()["items"] if i["item_id"] == "tomato")


def test_range_follows_the_weather_on_the_counter(api):
    scan_todays_bill(api)
    t = tomato(api)  # D5 = 4–6 days; 31 °C on the counter × 0.8
    assert t["shelf_life"] == [3.2, 4.8]
    assert {"kind": "weather", "label": "31°C on the counter", "factor": 0.8} in t["factors"]
    api.put("/api/weather", json={"temp_c": 38})
    assert tomato(api)["shelf_life"] == [2.8, 4.2]  # × 0.7
    api.put("/api/weather", json={"temp_c": 22})
    assert tomato(api)["shelf_life"] == [4.0, 6.0]
    palak = next(i for i in api.get("/api/inventory").json()["items"] if i["item_id"] == "palak")
    assert palak["shelf_life"] == [2.0, 3.0]  # the fridge doesn't care about the weather


def test_moving_to_the_fridge_changes_the_range(api):
    scan_todays_bill(api)
    t = tomato(api)
    assert t["alt_storage"]["storage"] == "fridge"
    assert t["alt_storage"]["shelf_life"] == [5.6, 8.4]  # × 1.4, and no heat factor
    api.patch("/api/inventory/tomato", json={"storage": "fridge"})
    moved = tomato(api)
    assert moved["storage"] == "fridge" and moved["shelf_life"] == [5.6, 8.4]


def test_milk_out_of_the_fridge_spoils_faster(api):
    scan_todays_bill(api)
    api.patch("/api/inventory/milk", json={"storage": "pantry"})
    milk = next(i for i in api.get("/api/inventory").json()["items"] if i["item_id"] == "milk")
    assert milk["shelf_life"][1] < 2


def test_printed_date_overrides_the_estimate(api):
    scan_todays_bill(api)
    api.patch("/api/inventory/paneer", json={"printed_expiry": "2026-10-26"})
    p = next(i for i in api.get("/api/inventory").json()["items"] if i["item_id"] == "paneer")
    assert p["days_left"] == [2, 2] and p["estimate"] == "printed date"
    api.patch("/api/inventory/paneer", json={"clear_printed_expiry": True})
    p = next(i for i in api.get("/api/inventory").json()["items"] if i["item_id"] == "paneer")
    assert p["estimate"] == "estimated"


def test_humid_weather_shortens_dry_goods(api):
    api.put("/api/weather", json={"temp_c": 30, "humidity": 82})
    api.post("/api/inventory/add", json={"item_id": "poha", "qty": 1})
    poha = next(i for i in api.get("/api/inventory").json()["items"] if i["item_id"] == "poha")
    assert poha["shelf_life"] == [38.2, 51.0]  # M2 = 45–60 days × 0.85


def test_hand_added_item_gets_a_tier_and_a_range(api):
    r = api.post("/api/inventory/add", json={"item_id": "cucumber", "qty": 0.5})
    c = next(i for i in r.json()["items"] if i["item_id"] == "cucumber")
    assert c["tier"] == "D5" and c["days_left"][1] > 0
    assert api.post("/api/inventory/add", json={"item_id": "cucumber", "qty": 0}).status_code == 422
