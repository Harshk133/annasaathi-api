"""Bill photos through OCR.space: the API call is replaced by the text OCR.space really
returned for a photo of the sample bill (tests/fixtures/ocr_space_ganesh.txt), so these run
offline and never use the key."""
import io
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from anna import db, ocr
from anna.data import sample_bills
from anna.extract import ExtractionUnavailable

FIXTURE = (Path(__file__).parent / "fixtures" / "ocr_space_ganesh.txt").read_text(encoding="utf-8")


class FakeResponse:
    def __init__(self, body, status=200):
        self._body, self.status_code = body, status

    def json(self):
        return self._body


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setenv("ANNA_DEMO", "1")
    monkeypatch.setenv("ANNA_TODAY", "2026-10-24")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("OCR_SPACE_API_KEY", "test-key")
    from anna.main import app

    c = TestClient(app)
    c.post("/api/demo/reset")
    return c


def ocr_replies(monkeypatch, body, status=200):
    calls = []

    def fake_post(url, **kw):
        calls.append(kw)
        return FakeResponse(body, status)

    monkeypatch.setattr(ocr.httpx, "post", fake_post)
    return calls


def ok_body(text=FIXTURE):
    return {"ParsedResults": [{"ParsedText": text, "FileParseExitCode": 1}], "OCRExitCode": 1, "IsErroredOnProcessing": False}


def photo(api, data=b"\xff\xd8fake-jpeg"):
    return api.post("/api/bills/scan", files={"file": ("bill.jpg", data, "image/jpeg")})


def test_ocr_text_reads_every_item_of_the_sample_bill():
    shop, lines = ocr.bill_from_text(FIXTURE)
    assert shop == "Shree Ganesh Kirana Stores"
    want = {l["item_id"]: (l["qty"], l["price"]) for l in sample_bills()["ganesh_kirana_oct24"]["lines"]}
    got = {l["item_id"]: (l["qty"], l["price"]) for l in lines}
    assert got == want  # items, quantities and prices, including the two rows OCR split apart


def test_bill_total_is_not_mistaken_for_an_item_price():
    _, lines = ocr.bill_from_text("1869.00\nTOTAL\nTOMATO 1.000 KG 40.00")
    assert [(l["item_id"], l["price"]) for l in lines] == [("tomato", 40.0)]


def test_ocr_lines_are_never_presented_as_certain():
    _, lines = ocr.bill_from_text(FIXTURE)
    assert max(l["confidence"] for l in lines) <= ocr.OCR_CONFIDENCE_CAP


@pytest.mark.parametrize("raw,fixed", [("2.OOO KG", "2.000 KG"), ("l50.00", "150.00"), ("80,00", "80.00"), ("PALAK", "PALAK")])
def test_common_ocr_digit_mistakes_are_fixed(raw, fixed):
    assert ocr.fix_ocr_digits(raw) == fixed


def test_large_photo_is_shrunk_under_the_upload_limit():
    from PIL import Image

    noise = Image.effect_noise((3000, 4000), 80).convert("RGB")  # incompressible, like a detailed photo
    buf = io.BytesIO()
    noise.save(buf, "PNG")
    assert len(buf.getvalue()) > ocr.MAX_BYTES
    data, media = ocr.shrink(buf.getvalue(), "image/png")
    assert len(data) <= ocr.MAX_BYTES and media == "image/jpeg"


def test_photo_scan_goes_through_ocr_space(api, monkeypatch):
    calls = ocr_replies(monkeypatch, ok_body())
    r = photo(api).json()
    assert r["mode"] == "ocr" and r["source"] == "photo" and r["shop"] == "Shree Ganesh Kirana Stores"
    assert len([l for l in r["lines"] if l["matched"]]) == 18
    assert calls[0]["headers"]["apikey"] == "test-key"  # the key goes in the header, from the server's env
    assert api.get("/api/bills/status").json()["reader"] == "ocr_space"


def test_ocr_bill_can_be_added_to_the_kitchen(api, monkeypatch):
    ocr_replies(monkeypatch, ok_body())
    s = photo(api).json()
    lines = [{"item_id": l["item_id"], "qty": l["qty"], "price": l["price"], "raw": l["raw"]} for l in s["lines"] if l["matched"]]
    r = api.post("/api/bills/confirm", json={"bill_id": s["bill_id"], "shop": s["shop"], "source": "photo", "lines": lines})
    assert r.json()["added"] == 18
    assert any(i["item_id"] == "palak" for i in api.get("/api/inventory").json()["items"])


def test_ocr_failure_is_reported_not_replaced_by_the_sample(api, monkeypatch):
    ocr_replies(monkeypatch, {"IsErroredOnProcessing": True, "ErrorMessage": ["Unable to recognize the file type"]})
    r = photo(api)
    assert r.status_code == 502 and r.json()["detail"]["code"] == "read_failed"


def test_network_failure_is_reported(api, monkeypatch):
    def boom(*a, **k):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(ocr.httpx, "post", boom)
    assert photo(api).status_code == 502


def test_photo_without_groceries_says_so(api, monkeypatch):
    ocr_replies(monkeypatch, ok_body("Happy Diwali\nfrom all of us"))
    r = photo(api)
    assert r.status_code == 422 and r.json()["detail"]["code"] == "no_items"


def test_daily_limit_or_bad_key(monkeypatch):
    monkeypatch.setenv("OCR_SPACE_API_KEY", "k")
    ocr_replies(monkeypatch, "Invalid API key", status=403)
    with pytest.raises(ExtractionUnavailable, match="refused"):
        ocr.ocr_text(b"x", "image/jpeg")


def test_without_a_reader_the_sample_is_shown_and_labelled(api, monkeypatch):
    monkeypatch.delenv("OCR_SPACE_API_KEY")
    r = photo(api).json()
    assert r["mode"] == "preloaded" and "isn't set up" in r["note"]
    assert api.get("/api/bills/status").json()["reader"] is None


def test_prices_matched_by_position_are_flagged_for_review(api, monkeypatch):
    ocr_replies(monkeypatch, ok_body())
    r = photo(api).json()
    flagged = {l["item_id"] for l in r["lines"] if l["needs_review"]}
    assert flagged == {"potato", "cauliflower"}  # the two rows OCR split; everything else read cleanly
