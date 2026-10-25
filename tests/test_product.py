"""The working product loop through the HTTP API: bill text, duplicate bills, the shopping
trip, inventory corrections, cooking deductions, alerts and the catalogue data."""
import copy

import pytest
from fastapi.testclient import TestClient

from anna import db, extract
from anna.data import catalogue, demo_household, fmt_qty, recipes, sample_bills, tiers


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setenv("ANNA_DEMO", "1")  # fixed 31 °C, no network
    monkeypatch.setenv("ANNA_TODAY", "2026-10-24")
    from anna.main import app

    c = TestClient(app)
    c.post("/api/demo/reset")
    return c


def inv_item(api, item_id):
    return next((r for r in api.get("/api/inventory").json()["items"] if r["item_id"] == item_id), None)


def shop_row(shop, item_id):
    return next((r for r in shop["rows"] if r["item_id"] == item_id), None)


def scan_todays_bill(api):
    s = api.post("/api/bills/scan", data={"sample_id": "ganesh_kirana_oct24"}).json()
    lines = [{"item_id": l["item_id"], "qty": l["qty"], "price": l["price"], "raw": l["raw"]} for l in s["lines"]]
    assert confirm(api, s["bill_id"], lines, shop=s["shop"]).status_code == 200


def confirm(api, bill_id, lines, shop="Test Kirana", **kw):
    return api.post("/api/bills/confirm", json={"bill_id": bill_id, "shop": shop, "lines": lines, **kw})


# ---------------------------------------------------------------- bill text and matching


def test_text_parser_reads_every_sample_bill_line():
    for bill in sample_bills().values():
        for line in bill["lines"]:
            got = extract.parse_bill_line(line["raw"])
            assert got["item_id"] == line["item_id"], line["raw"]
            assert got["qty"] == pytest.approx(line["qty"]), line["raw"]


def test_text_parser_drops_totals_and_taxes():
    text = "SHREE GANESH KIRANA\nTOMATO 1.000 KG 40.00\nCGST 2.5% 1.00\nTOTAL 41.00\nThank you"
    lines = extract.parse_bill_text(text)
    assert [(l["item_id"], l["qty"], l["price"]) for l in lines] == [("tomato", 1.0, 40.0)]


def test_unknown_line_is_flagged_not_guessed(api):
    r = api.post("/api/bills/parse_text", json={"text": "DETTOL SOAP 2 NOS 90.00\nTOMATO 1.000 KG 40.00"}).json()
    soap = r["lines"][0]
    assert soap["item_id"] == "unknown" and soap["needs_review"] and r["review_count"] >= 1
    assert r["lines"][1]["item_id"] == "tomato" and not r["lines"][1]["needs_review"]


def test_guessed_quantity_lowers_confidence():
    sure = extract.parse_bill_line("PANEER 0.200 KG 84.00")
    guess = extract.parse_bill_line("PANEER 84.00")  # no quantity printed
    assert guess["confidence"] < sure["confidence"] and guess["confidence"] < 0.8


def test_voice_marathi_purchase():
    (line,) = extract.parse_voice("आज अर्धा किलो टोमॅटो घेतले")
    assert (line["item_id"], line["qty"]) == ("tomato", 0.5)


# ---------------------------------------------------------------- duplicate bills


def test_same_bill_twice_replaces_instead_of_doubling(api):
    before = inv_item(api, "paneer")
    lines = [{"item_id": "paneer", "qty": 0.4, "price": 168, "raw": "PANEER 0.4 KG"}]
    assert confirm(api, "b1", lines).json()["replaced"] is False
    once = inv_item(api, "paneer")["remaining"]
    assert confirm(api, "b1", lines).json()["replaced"] is True
    assert inv_item(api, "paneer")["remaining"] == pytest.approx(once)
    assert once == pytest.approx((before["remaining"] if before else 0) + 0.4)


def test_same_bill_from_another_photo_is_refused_until_confirmed(api):
    lines = [{"item_id": "tomato", "qty": 1, "price": 40, "raw": "TOMATO 1 KG"}]
    assert confirm(api, "photo_a", lines).status_code == 200
    r = confirm(api, "photo_b", lines)
    assert r.status_code == 409 and r.json()["detail"]["code"] == "duplicate_bill"
    assert confirm(api, "photo_b", lines, allow_duplicate=True).status_code == 200


def test_two_lines_of_one_item_are_added_together(api):
    lines = [{"item_id": "methi", "qty": 1, "price": 20, "raw": "METHI"},
             {"item_id": "methi", "qty": 2, "price": 40, "raw": "METHI JUDI"}]
    r = confirm(api, "b2", lines).json()
    assert r["added"] == 1
    bill = next(b for b in api.get("/api/bills").json() if b["id"] == "b2")
    assert bill["lines"] == 1


# ---------------------------------------------------------------- buy-right list and the shopping trip


def test_every_row_has_quantity_reason_and_never_negative(api):
    shop = api.get("/api/shopping").json()
    assert shop["rows"]
    for r in shop["rows"]:
        assert r["reason"] and r["qty"] >= 0
        assert r["status"] in ("buy", "buy_less", "have", "manual")
        if r["status"] in ("buy", "buy_less"):
            assert r["planned_qty"] > 0


def test_stock_at_home_is_deducted(api):
    # the buy-right quantity covers need minus what is still fresh at home (to within pack rounding)
    for r in api.get("/api/shopping").json()["buy"]:
        if r["perishable"] and not r["daily"]:
            assert r["qty"] + r["stock_at_shop"] >= 0.7 * r["need_qty"] - 1e-6, r["item_id"]
            assert r["qty"] <= max(r["need_qty"] - r["stock_at_shop"], 0) + r["pack_size"] + 1e-6, r["item_id"]


def test_running_out_puts_it_back_on_the_list(api):
    # shortage prevention: if the family says palak is finished, the list must not skip it
    scan_todays_bill(api)
    assert api.patch("/api/inventory/palak", json={"qty": 0}).status_code == 200
    row = shop_row(api.get("/api/shopping").json(), "palak")
    assert row and row["status"] in ("buy", "buy_less") and row["qty"] > 0


def test_purchase_adds_to_kitchen_once(api):
    scan_todays_bill(api)
    api.post("/api/shopping/start")
    before = inv_item(api, "tomato")["remaining"]
    s = api.post("/api/shopping/items/tomato/purchase", json={"qty": 1}).json()
    assert shop_row(s, "tomato")["purchased"] and s["trip"]
    assert inv_item(api, "tomato")["remaining"] == pytest.approx(before + 1)
    api.post("/api/shopping/items/tomato/purchase", json={"qty": 1})  # double tap
    assert inv_item(api, "tomato")["remaining"] == pytest.approx(before + 1)
    api.post("/api/shopping/items/tomato/unpurchase")
    assert inv_item(api, "tomato")["remaining"] == pytest.approx(before)


def test_ticking_an_item_keeps_the_rest_of_the_list_steady(api):
    shop = api.post("/api/shopping/start").json()
    first = {r["item_id"]: r["planned_qty"] for r in shop["rows"]}
    tick = next(r["item_id"] for r in shop["rows"] if r["status"] in ("buy", "buy_less"))
    s = api.post(f"/api/shopping/items/{tick}/purchase", json={}).json()
    after = {r["item_id"]: r["planned_qty"] for r in s["rows"]}
    assert after == first


def test_exclude_and_edit_quantity(api):
    shop = api.get("/api/shopping").json()
    row = next(r for r in shop["rows"] if r["status"] in ("buy", "buy_less"))
    s = api.put(f"/api/shopping/items/{row['item_id']}", json={"excluded": True}).json()
    assert shop_row(s, row["item_id"])["excluded"] and s["list_rupees"] < shop["list_rupees"]
    s = api.put(f"/api/shopping/items/{row['item_id']}", json={"excluded": False, "qty": row["qty"] * 2}).json()
    r2 = shop_row(s, row["item_id"])
    assert not r2["excluded"] and r2["edited"] and r2["qty"] == pytest.approx(row["qty"] * 2)
    s = api.delete(f"/api/shopping/items/{row['item_id']}").json()
    assert not shop_row(s, row["item_id"])["edited"]


def test_manual_item_gets_a_formula_quantity(api):
    s = api.post("/api/shopping/items", json={"item_id": "cabbage"}).json()
    row = shop_row(s, "cabbage")
    assert row["status"] in ("manual", "buy", "buy_less") and row["qty"] > 0 and row["reason"]


def test_finish_trip_records_purchases_and_clears_edits(api):
    api.post("/api/shopping/items", json={"item_id": "cabbage", "qty": 0.5})
    api.post("/api/shopping/items/cabbage/purchase", json={})
    out = api.post("/api/shopping/finish").json()
    assert out["items"] == 1 and out["shopping"]["trip"] is None
    assert any(b["source"] == "shopping" for b in api.get("/api/bills").json())
    assert inv_item(api, "cabbage")["remaining"] >= 0.5 - 1e-6
    assert shop_row(out["shopping"], "cabbage") is None or not shop_row(out["shopping"], "cabbage")["purchased"]


# ---------------------------------------------------------------- kitchen corrections


def test_edit_quantity_remove_and_undo(api):
    api.patch("/api/inventory/rice", json={"qty": 2.5})
    assert inv_item(api, "rice")["remaining"] == pytest.approx(2.5)
    lot_ids = api.delete("/api/inventory/rice").json()["lot_ids"]
    assert inv_item(api, "rice") is None
    api.post("/api/inventory/restore", json={"lot_ids": lot_ids})
    assert inv_item(api, "rice")["remaining"] == pytest.approx(2.5)


def test_purchase_date_cannot_be_in_the_future(api):
    assert api.patch("/api/inventory/rice", json={"purchased_on": "2026-11-30"}).status_code == 400


def test_every_item_says_where_it_came_from(api):
    for r in api.get("/api/inventory").json()["items"]:
        assert r["source"] in ("bill", "manual", "shopping", "voice") and r["purchased_on"]


# ---------------------------------------------------------------- cooking


def test_marking_cooked_takes_ingredients_out(api):
    dish = api.get("/api/plan/today").json()["dishes"][0]
    assert dish["steps"] and dish["ingredients"]
    est = api.get(f"/api/recipes/{dish['recipe_id']}/consumption").json()["items"]
    for e in est:
        assert 0 <= e["qty"] <= e["have"] + 1e-9
    used = [e for e in est if e["qty"] > 0]
    before = {e["item_id"]: inv_item(api, e["item_id"])["remaining"] for e in used}
    r = api.post("/api/cooked", json={"recipe_id": dish["recipe_id"], "storage": "room",
                                      "consumed": [{"item_id": e["item_id"], "qty": e["qty"]} for e in used]}).json()
    assert r["finish_by"] and len(r["used"]) == len(used)
    for e in used:
        now = inv_item(api, e["item_id"])
        assert (now["remaining"] if now else 0) == pytest.approx(max(0, before[e["item_id"]] - e["qty"]), abs=1e-6)


def test_cooked_food_moves_to_fridge(api):
    r = api.post("/api/cooked", json={"recipe_id": "dal_palak", "storage": "room"}).json()
    rows = api.patch(f"/api/cooked/{r['id']}", json={"storage": "fridge"}).json()["rows"]
    moved = next(x for x in rows if x["id"] == r["id"])
    assert moved["storage"] == "fridge" and moved["finish_by"] > r["finish_by"]


# ---------------------------------------------------------------- alerts


def test_alerts_are_actionable_and_unique(api):
    a = api.get("/api/alerts").json()["alerts"]
    assert a, "the demo household has palak to use first"
    assert len({x["id"] for x in a}) == len(a)
    assert len({x["item_id"] for x in a if x["item_id"]}) == len([x for x in a if x["item_id"]])
    for x in a:
        assert x["title"] and x["body"] and x["level"] in ("urgent", "soon", "info")


# ---------------------------------------------------------------- data


def test_catalogue_has_about_150_items_with_local_names():
    cat = catalogue()
    assert len(cat) >= 150
    for it in cat.values():
        assert it.name_hi and it.name_mr and it.tier in tiers() and it.pack_size > 0 and it.price > 0


def test_aliases_point_at_one_item():
    owner = {}
    for it in catalogue().values():
        for a in it.aliases:
            assert owner.setdefault(a, it.id) == it.id, f"alias {a!r} used by {owner[a]} and {it.id}"


def test_recipes_use_catalogue_items_and_have_steps():
    for r in recipes():
        assert r["steps"] and all(i in catalogue() for i in r["ingredients"]), r["id"]


def test_pack_words():
    cat = catalogue()
    assert fmt_qty(cat["biscuits"], 2) == "2 packs"
    assert fmt_qty(cat["palak"], 1) == "1 bunch"
    assert fmt_qty(cat["tomato"], 0.5) == "500 g"


def test_dishes_are_cooked_from_what_is_at_home(api):
    for d in api.get("/api/plan/today").json()["dishes"]:
        main = [i for i in d["ingredients"] if not i["minor"]]
        have = sum(i["have"] for i in main)
        assert have > 0 and (not d["missing"] or len(d["missing"]) < have), d["name_en"]


def test_empty_kitchen_suggests_nothing(api):
    h = {k: v for k, v in demo_household().items() if k != "seed_bills"}
    api.post("/api/household/setup", json={"household": h, "fresh": True})
    assert api.get("/api/plan/today").json()["dishes"] == []


def test_health_check(api):
    r = api.get("/api/health").json()
    assert r["ok"] and r["today"] == "2026-10-24"
