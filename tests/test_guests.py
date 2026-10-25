"""Unexpected guests: menu from the kitchen, missing items only, health-aware, with recipe videos.
YouTube and OpenRouter are replaced by fakes so these run offline."""
import pytest
from fastapi.testclient import TestClient

from anna import db, guests, llm, videos


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setenv("ANNA_DEMO", "1")
    monkeypatch.setenv("ANNA_TODAY", "2026-10-24")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(videos, "recipe_video", lambda dish: {"query": dish, "url": "https://www.youtube.com/watch?v=abcdefghijk",
                                                              "video_id": "abcdefghijk", "search_url": videos.search_url(dish)})
    from anna.main import app

    c = TestClient(app)
    c.post("/api/demo/reset")
    s = c.post("/api/bills/scan", data={"sample_id": "ganesh_kirana_oct24"}).json()
    c.post("/api/bills/confirm", json={"bill_id": s["bill_id"], "shop": s["shop"], "lines": [
        {"item_id": l["item_id"], "qty": l["qty"], "price": l["price"], "raw": l["raw"]} for l in s["lines"]]})
    return c


def plan(api, **kw):
    body = {"adults": 8, "minors": 2, "meal": "dinner", **kw}
    return api.post("/api/guests/plan", json=body)


def test_guests_count_with_the_family():
    h = {"portion": "normal", "members": [{"group": "adult"}, {"group": "adult"}]}
    ae, heads = guests.people(h, adults=2, minors=1, include_family=True)
    assert heads == 5 and ae == pytest.approx(2 + 0.6 + 2)
    ae_only, _ = guests.people(h, adults=2, minors=1, include_family=False)
    assert ae_only == pytest.approx(2.6)


def test_menu_prefers_what_is_in_the_kitchen(api):
    r = plan(api).json()
    covers = [d["cover"] for d in r["dishes"] if not d["health_notes"]]
    assert covers == sorted(covers, reverse=True)
    assert r["selected"] == [d["recipe_id"] for d in r["dishes"][:2]]
    assert all(d["video"]["url"].startswith("https://www.youtube.com/") for d in r["dishes"])


def test_only_the_missing_amount_is_bought(api):
    r = plan(api).json()
    picked = [d for d in r["dishes"] if d["recipe_id"] in r["selected"]]
    for x in r["to_buy"]:
        lines = [l for d in picked for l in d["ingredients"] if l["item_id"] == x["item_id"]]
        need = sum(l["need"] for l in lines)
        assert lines and need > lines[0]["have"] and x["qty"] >= need - lines[0]["have"] - 1e-6


def test_more_guests_need_more(api):
    small = plan(api, adults=1, minors=0, selected=["dal_palak"]).json()
    big = plan(api, adults=12, minors=4, selected=["dal_palak"]).json()
    need = lambda r: sum(l["need"] for d in r["dishes"] if d["recipe_id"] == "dal_palak" for l in d["ingredients"])
    assert need(big) > need(small)


def test_diabetic_guest_gets_no_sweet_dish_by_default(api):
    r = plan(api, conditions=["diabetes"]).json()
    picked = [d for d in r["dishes"] if d["recipe_id"] in r["selected"]]
    assert all(not d["health_notes"] for d in picked)


def test_heart_condition_rules_out_fried_dishes(api):
    r = plan(api, conditions=["heart"], meal="lunch").json()
    assert all(d["prep"] != "fried" for d in r["dishes"])


def test_family_can_pick_their_own_dishes(api):
    r = plan(api, selected=["palak_paneer", "varan_bhaat"]).json()
    assert r["selected"] == ["palak_paneer", "varan_bhaat"]


def test_family_only_meal_needs_no_guests(api):
    r = plan(api, adults=0, minors=0)
    assert r.status_code == 200 and r.json()["people"]["heads"] == 5  # just the Patils
    assert plan(api, adults=0, minors=0, include_family=False).status_code == 422


def test_cook_today_dishes_have_videos(api):
    for d in api.get("/api/plan/today").json()["dishes"]:
        assert d["video"]["url"].startswith("https://www.youtube.com/")


def test_missing_video_is_reported(monkeypatch):
    from anna import videos as v

    v._cache.clear()
    monkeypatch.setattr(v, "_fetch", lambda query: "<html>nothing here</html>")
    out = v.recipe_video("an unheard-of dish")
    assert out["video_id"] is None and out["url"] == out["search_url"]


def test_ai_ideas_need_a_key(api):
    r = api.post("/api/guests/ideas", json={"adults": 2, "meal": "dinner"})
    assert r.status_code == 503 and r.json()["detail"]["code"] == "no_ai"


def test_ai_ideas_are_checked_against_the_kitchen(api, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setattr(llm, "ideas", lambda *a, **k: [{
        "name": "Paneer tikka masala", "why": "Uses the paneer and tomatoes at home", "time_min": 40, "steps": ["Cook"],
        "ingredients": [{"item_id": "paneer", "name": "paneer", "per_person": 0.08},
                        {"item_id": "tomato", "name": "tomato", "per_person": 0.06},
                        {"item_id": None, "name": "kasuri methi leaves", "per_person": 1}]}])
    d = api.post("/api/guests/ideas", json={"adults": 8, "minors": 2, "meal": "dinner"}).json()["dishes"][0]
    assert d["source"] == "ai" and d["other_ingredients"] == ["kasuri methi leaves"]
    paneer = next(l for l in d["ingredients"] if l["item_id"] == "paneer")
    assert paneer["need"] == pytest.approx(0.08 * 13.8, rel=0.05)  # our maths, not the model's
    assert any(x["item_id"] == "paneer" for x in d["to_buy"]) == (not paneer["enough"])


def test_llm_answer_is_validated(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")

    class R:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": "```json\n{\"dishes\":[{\"name\":\"Matar paneer\",\"ingredients\":"
                                                        "[{\"name\":\"paneer\",\"amount_per_person\":70,\"unit\":\"g\"}]}]}\n```"}}]}

    monkeypatch.setattr(llm.httpx, "post", lambda *a, **k: R())
    out = llm.ideas(["paneer: 400 g"], 4, "dinner", "veg", [], [])
    assert out[0]["ingredients"][0]["item_id"] == "paneer" and out[0]["ingredients"][0]["per_person"] == pytest.approx(0.07)
