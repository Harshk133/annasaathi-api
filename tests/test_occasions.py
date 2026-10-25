"""Festival-aware lists and health-aware planning, through the API on the demo family
(the Patils celebrate Diwali; Aaji has diabetes; demo date 24 Oct 2026, Diwali 6-11 Nov 2026)."""
import copy

import pytest
from fastapi.testclient import TestClient

from anna import db, occasions
from anna.data import catalogue, demo_household


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setenv("ANNA_DEMO", "1")
    monkeypatch.setenv("ANNA_TODAY", "2026-10-24")
    from anna.main import app

    c = TestClient(app)
    c.post("/api/demo/reset")
    return c


def profile(**changes):
    h = copy.deepcopy({k: v for k, v in demo_household().items() if k != "seed_bills"})
    h.update(changes)
    return h


def row(shop, item_id):
    return next((r for r in shop["rows"] if r["item_id"] == item_id), None)


def test_festival_data_is_consistent():
    cat = catalogue()
    ids = {f.id for f in occasions.festivals()}
    for fid, items in occasions.festival_items().items():
        assert fid in ids, fid
        for it in items:
            assert it["item_id"] in cat and it["extra_per_ae"] > 0, (fid, it)
    for f in occasions.festivals():
        assert f.start <= f.end


def test_diwali_shows_up_before_the_shop(api):
    shop = api.get("/api/shopping").json()
    assert [f["id"] for f in shop["festivals"]] == ["diwali"]
    assert api.get("/api/festivals").json()["festivals"][0]["days_away"] == 13


def test_festival_adds_to_items_the_family_buys(api):
    besan = row(api.get("/api/shopping").json(), "besan")
    assert besan["festival"] and "Diwali" in besan["reason"]
    without = api.put("/api/household", json=profile(festivals=[])).json()
    besan_plain = row(api.get("/api/shopping").json(), "besan")
    assert not besan_plain or besan_plain["qty"] <= besan["qty"]
    assert not besan_plain or not besan_plain["festival"]


def test_festival_shortfall_is_bought_even_when_everyday_stock_is_enough(api):
    sugar = row(api.get("/api/shopping").json(), "sugar")
    assert sugar["status"] in ("buy", "buy_less") and sugar["qty"] > 0 and "Diwali" in sugar["reason"]


def test_festival_only_items_are_suggestions_not_forced_onto_the_list(api):
    shop = api.get("/api/shopping").json()
    sugg = {s["item_id"]: s for s in shop["suggestions"]}
    assert "mithai" in sugg and sugg["mithai"]["kind"] == "festival"
    assert row(shop, "mithai") is None
    s = api.post("/api/shopping/items", json={"item_id": "mithai", "qty": sugg["mithai"]["qty"]}).json()
    assert row(s, "mithai") and "mithai" not in {x["item_id"] for x in s["suggestions"]}


def test_no_festival_extras_out_of_season(api, monkeypatch):
    monkeypatch.setenv("ANNA_TODAY", "2026-06-15")
    api.post("/api/demo/reset")  # the demo family's bills, dated around mid-June
    shop = api.get("/api/shopping").json()
    assert shop["festivals"] == [] and not any(r.get("festival") for r in shop["rows"])
    assert all(s["kind"] != "festival" for s in shop["suggestions"])


def test_family_that_does_not_celebrate_gets_no_extras(api):
    api.put("/api/household", json=profile(festivals=["christmas"]))
    assert api.get("/api/shopping").json()["festivals"] == []


def test_diabetic_members_share_of_sugar_is_left_out(api):
    with_aaji = row(api.get("/api/shopping").json(), "sugar")
    assert "not counting Aaji (diabetes)" in with_aaji["reason"]
    assert with_aaji["health"][0]["member"] == "Aaji" and with_aaji["health"][0]["note"] == "high in sugar"
    members = profile()["members"]
    for m in members:
        m["conditions"] = []
    api.put("/api/household", json=profile(members=members, festivals=[]))
    healthy = api.get("/api/inventory").json()
    sugar_use = next(i for i in healthy["items"] if i["item_id"] == "sugar")["daily_use"]
    api.put("/api/household", json=profile(festivals=[]))
    diabetic_use = next(i for i in api.get("/api/inventory").json()["items"] if i["item_id"] == "sugar")["daily_use"]
    assert diabetic_use < sugar_use


def test_sugar_free_option_is_suggested_with_care(api):
    sugg = next(s for s in api.get("/api/shopping").json()["suggestions"] if s["item_id"] == "sugar_free")
    assert sugg["kind"] == "health" and sugg["for"] == ["Aaji"] and "doctor" in sugg["reason"]


def test_heart_condition_flags_fatty_items_and_avoids_fried_dishes(api):
    members = profile()["members"]
    for m in members:
        if m["name"] == "Rahul":
            m["conditions"] = ["heart"]
    api.put("/api/household", json=profile(members=members, prep_styles=["wet", "roasted", "fried"]))
    assert all(d["prep"] != "fried" for d in api.get("/api/plan/today").json()["dishes"])
    assert occasions.health_flags({"members": members}, "ghee")[0]["member"] == "Rahul"


def test_unknown_condition_or_festival_is_refused(api):
    members = profile()["members"]
    members[0]["conditions"] = ["flu"]
    assert api.put("/api/household", json=profile(members=members)).status_code == 422
    assert api.put("/api/household", json=profile(festivals=["not_a_festival"])).status_code == 422
