"""SQLite storage. The API holds no session state; everything lives here, so
moving to PostgreSQL is a connection change, not a logic change."""
from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path

from .data import demo_household, sample_bills

DB_PATH = Path(os.environ.get("ANNA_DB", Path(__file__).resolve().parent.parent / "anna.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS household (id INTEGER PRIMARY KEY CHECK (id = 1), json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS bills (id TEXT PRIMARY KEY, shop TEXT, scanned_on TEXT, source TEXT, json TEXT);
CREATE TABLE IF NOT EXISTS lots (
  id INTEGER PRIMARY KEY AUTOINCREMENT, item_id TEXT NOT NULL, qty REAL NOT NULL,
  purchased_on TEXT NOT NULL, storage TEXT, printed_expiry TEXT, status TEXT NOT NULL DEFAULT 'active',
  anchor_qty REAL NOT NULL, anchor_date TEXT NOT NULL, bill_id TEXT, price REAL
);
CREATE TABLE IF NOT EXISTS use_rates (item_id TEXT PRIMARY KEY, rate REAL NOT NULL);
CREATE TABLE IF NOT EXISTS shelf_corrections (item_id TEXT PRIMARY KEY, factor REAL NOT NULL);
CREATE TABLE IF NOT EXISTS waste_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT, item_id TEXT, qty REAL, kg REAL, rupees REAL, logged_on TEXT
);
CREATE TABLE IF NOT EXISTS cooked (
  id INTEGER PRIMARY KEY AUTOINCREMENT, dish TEXT, emoji TEXT, prep TEXT, storage TEXT,
  cooked_at TEXT, finish_by TEXT, status TEXT DEFAULT 'active'
);
CREATE TABLE IF NOT EXISTS shop_trip (id INTEGER PRIMARY KEY CHECK (id = 1), trip_id TEXT NOT NULL, started_on TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS shop_items (
  item_id TEXT PRIMARY KEY, qty REAL, excluded INTEGER NOT NULL DEFAULT 0, manual INTEGER NOT NULL DEFAULT 0
);
"""

# Columns added after the first release: (table, column, type). Added in place, so an
# existing anna.db keeps its data.
MIGRATIONS = [
    ("lots", "source", "TEXT"),  # bill | manual | shopping | voice: where this stock came from
    ("cooked", "recipe_id", "TEXT"),
    ("cooked", "consumed", "TEXT"),  # JSON [{item_id, qty}] taken out of the pantry
]
FAMILY_TABLES = ("bills", "lots", "use_rates", "shelf_corrections", "waste_log", "cooked", "shop_trip", "shop_items")


def connect() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    for table, col, typ in MIGRATIONS:
        if col not in {r["name"] for r in con.execute(f"PRAGMA table_info({table})")}:
            con.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")
    return con


def _d(s):
    return date.fromisoformat(s) if s else None


def get_household(con) -> dict:
    row = con.execute("SELECT json FROM household WHERE id = 1").fetchone()
    if not row:
        reset_demo(con)
        row = con.execute("SELECT json FROM household WHERE id = 1").fetchone()
    return json.loads(row["json"])


def save_household(con, h: dict) -> None:
    con.execute("INSERT INTO household (id, json) VALUES (1, ?) ON CONFLICT(id) DO UPDATE SET json = excluded.json",
                (json.dumps(h, ensure_ascii=False),))
    con.commit()


def get_lots(con) -> list[dict]:
    out = []
    for r in con.execute("SELECT * FROM lots ORDER BY purchased_on, id"):
        out.append({**dict(r), "purchased_on": _d(r["purchased_on"]), "printed_expiry": _d(r["printed_expiry"]),
                    "anchor_date": _d(r["anchor_date"])})
    return out


def add_lot(con, item_id, qty, purchased_on: date, storage=None, printed_expiry=None, bill_id=None, price=None,
            source="manual"):
    con.execute(
        "INSERT INTO lots (item_id, qty, purchased_on, storage, printed_expiry, anchor_qty, anchor_date, bill_id, price,"
        " source) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (item_id, qty, purchased_on.isoformat(), storage, printed_expiry.isoformat() if printed_expiry else None,
         qty, purchased_on.isoformat(), bill_id, price, source),
    )


def add_bill(con, bill_id, shop, scanned_on: date, source, lines, storage_for, lot_source="bill"):
    con.execute("INSERT OR REPLACE INTO bills (id, shop, scanned_on, source, json) VALUES (?, ?, ?, ?, ?)",
                (bill_id, shop, scanned_on.isoformat(), source, json.dumps(lines, ensure_ascii=False)))
    con.execute("DELETE FROM lots WHERE bill_id = ?", (bill_id,))
    for ln in lines:
        if not ln.get("item_id") or ln["item_id"] == "unknown" or not ln.get("qty"):
            continue
        add_lot(con, ln["item_id"], float(ln["qty"]), scanned_on, storage_for(ln["item_id"]),
                _d(ln.get("printed_expiry")), bill_id, ln.get("price"), source=lot_source)
    con.commit()


def bill_fingerprint(shop: str, lines: list[dict]) -> str:
    """Same shop and same items/quantities = the same bill, even from a different photo."""
    items = sorted((l["item_id"], round(float(l["qty"]), 3)) for l in lines
                   if l.get("item_id") and l["item_id"] != "unknown" and l.get("qty"))
    return json.dumps([shop.strip().lower(), items])


def find_duplicate_bill(con, bill_id: str, shop: str, lines: list[dict]) -> dict | None:
    fp = bill_fingerprint(shop, lines)
    for r in con.execute("SELECT * FROM bills WHERE id != ?", (bill_id,)):
        if bill_fingerprint(r["shop"] or "", json.loads(r["json"] or "[]")) == fp:
            return dict(r)
    return None


def list_bills(con) -> list[dict]:
    return [{"id": r["id"], "shop": r["shop"], "scanned_on": r["scanned_on"], "source": r["source"],
             "lines": len(json.loads(r["json"] or "[]"))}
            for r in con.execute("SELECT * FROM bills ORDER BY scanned_on DESC, rowid DESC")]


def reanchor(con, item_id: str, today: date, use: float, *, eaten: float | None = None,
             total: float | None = None) -> float:
    """Pin an item's stock to a known amount as of today, so the projection restarts from it.

    `eaten`: take this much off today's estimate, oldest lot first (a cooked meal).
    `total`: the family says this is what's left (a quantity edit).
    Lots that reach zero are marked finished. Returns the new total.
    """
    from .engine import EPS, project_remaining

    lots = sorted((l for l in get_lots(con) if l["item_id"] == item_id and l["status"] == "active"),
                  key=lambda l: (l["purchased_on"], l["id"]))
    if not lots:
        return 0.0
    rem = project_remaining(lots, use, today)
    now = sum(rem.values())
    if total is not None:
        if now > EPS:
            new = {i: v * total / now for i, v in rem.items()}
        else:  # we estimated it was finished; the family says otherwise
            new = {l["id"]: 0.0 for l in lots}
            new[lots[-1]["id"]] = total
    else:
        new, need = dict(rem), max(0.0, eaten or 0.0)
        for l in lots:
            take = min(new[l["id"]], need)
            new[l["id"]] -= take
            need -= take
    for l in lots:
        q = max(0.0, new[l["id"]])
        con.execute("UPDATE lots SET anchor_qty = ?, anchor_date = ?, status = ? WHERE id = ?",
                    (q, today.isoformat(), "active" if q > EPS else "finished", l["id"]))
    return sum(new.values())


def remove_item(con, item_id: str) -> list[int]:
    """Take an item out of the pantry (kept as 'removed' so it can be undone)."""
    ids = [r["id"] for r in con.execute("SELECT id FROM lots WHERE item_id = ? AND status = 'active'", (item_id,))]
    con.executemany("UPDATE lots SET status = 'removed' WHERE id = ?", [(i,) for i in ids])
    con.commit()
    return ids


def restore_lots(con, lot_ids: list[int]) -> int:
    n = 0
    for i in lot_ids:
        n += con.execute("UPDATE lots SET status = 'active' WHERE id = ? AND status = 'removed'", (i,)).rowcount
    con.commit()
    return n


# ---------------------------------------------------------------- shopping trip

def get_trip(con) -> dict | None:
    r = con.execute("SELECT * FROM shop_trip WHERE id = 1").fetchone()
    return {"trip_id": r["trip_id"], "started_on": r["started_on"]} if r else None


def start_trip(con, today: date) -> dict:
    trip = get_trip(con)
    if not trip:
        trip = {"trip_id": f"trip_{today:%Y%m%d}_{uuid.uuid4().hex[:6]}", "started_on": today.isoformat()}
        con.execute("INSERT INTO shop_trip (id, trip_id, started_on) VALUES (1, ?, ?)",
                    (trip["trip_id"], trip["started_on"]))
        con.commit()
    return trip


def shop_overrides(con) -> dict[str, dict]:
    return {r["item_id"]: dict(r) for r in con.execute("SELECT * FROM shop_items")}


def set_shop_override(con, item_id: str, **fields) -> None:
    cur = shop_overrides(con).get(item_id, {"qty": None, "excluded": 0, "manual": 0})
    cur.update({k: v for k, v in fields.items() if k in ("qty", "excluded", "manual")})
    con.execute("INSERT OR REPLACE INTO shop_items (item_id, qty, excluded, manual) VALUES (?, ?, ?, ?)",
                (item_id, cur["qty"], int(bool(cur["excluded"])), int(bool(cur["manual"]))))
    con.commit()


def clear_shop_override(con, item_id: str) -> None:
    con.execute("DELETE FROM shop_items WHERE item_id = ?", (item_id,))
    con.commit()


def trip_purchases(con, trip_id: str) -> dict[str, dict]:
    return {r["item_id"]: dict(r) for r in con.execute("SELECT * FROM lots WHERE bill_id = ?", (trip_id,))}


def purchase(con, trip_id: str, item_id: str, qty: float, today: date, storage: str, price: float) -> None:
    """Mark bought on this trip: one lot per item per trip, so tapping twice never doubles the stock."""
    existing = trip_purchases(con, trip_id).get(item_id)
    if existing:
        con.execute("UPDATE lots SET qty = ?, anchor_qty = ?, anchor_date = ?, price = ?, status = 'active' WHERE id = ?",
                    (qty, qty, today.isoformat(), price, existing["id"]))
    else:
        add_lot(con, item_id, qty, today, storage, None, trip_id, price, source="shopping")
    con.commit()


def unpurchase(con, trip_id: str, item_id: str) -> None:
    con.execute("DELETE FROM lots WHERE bill_id = ? AND item_id = ?", (trip_id, item_id))
    con.commit()


def finish_trip(con, today: date) -> dict:
    """Close the trip: its purchases become a bill in the history and this trip's list edits are cleared."""
    trip = get_trip(con)
    if not trip:
        return {"items": 0, "rupees": 0}
    bought = list(trip_purchases(con, trip["trip_id"]).values())
    if bought:
        lines = [{"item_id": l["item_id"], "qty": l["qty"], "price": l["price"], "raw": "shopping list"} for l in bought]
        con.execute("INSERT OR REPLACE INTO bills (id, shop, scanned_on, source, json) VALUES (?, ?, ?, ?, ?)",
                    (trip["trip_id"], "Shopping list", today.isoformat(), "shopping", json.dumps(lines)))
    con.execute("DELETE FROM shop_items")
    con.execute("DELETE FROM shop_trip")
    con.commit()
    return {"items": len(bought), "rupees": round(sum(l["price"] or 0 for l in bought))}


def update_active_lots(con, item_id, storage=None, printed_expiry=None, clear_expiry=False) -> int:
    lots = con.execute("SELECT id FROM lots WHERE item_id = ? AND status = 'active'", (item_id,)).fetchall()
    for r in lots:
        if storage:
            con.execute("UPDATE lots SET storage = ? WHERE id = ?", (storage, r["id"]))
        if clear_expiry:
            con.execute("UPDATE lots SET printed_expiry = NULL WHERE id = ?", (r["id"],))
        elif printed_expiry:
            con.execute("UPDATE lots SET printed_expiry = ? WHERE id = ?", (printed_expiry.isoformat(), r["id"]))
    con.commit()
    return len(lots)


def clear_family_data(con) -> None:
    """A new family: empty pantry, nothing learned yet."""
    for t in FAMILY_TABLES:
        con.execute(f"DELETE FROM {t}")
    con.commit()


def kv(con, table) -> dict[str, float]:
    col = "rate" if table == "use_rates" else "factor"
    return {r["item_id"]: r[col] for r in con.execute(f"SELECT * FROM {table}")}


def reset_demo(con, today: date | None = None) -> None:
    """Wipe and preload the Patil household: older bills already scanned,
    today's kirana bill left for the live demo."""
    from .data import catalogue

    today = today or date.today()
    for t in ("household", *FAMILY_TABLES):
        con.execute(f"DELETE FROM {t}")
    h = demo_household()
    save_household(con, h)
    cat = catalogue()
    bills = sample_bills()
    for bid in h.get("seed_bills", []):
        b = bills[bid]
        add_bill(con, bid, b["shop"], today + timedelta(days=b["date_offset_days"]), b["source"], b["lines"],
                 lambda iid: cat[iid].storage)
    con.commit()


def log_waste(con, item_id, qty, kg, rupees, on: date):
    con.execute("INSERT INTO waste_log (item_id, qty, kg, rupees, logged_on) VALUES (?, ?, ?, ?, ?)",
                (item_id, qty, kg, rupees, on.isoformat()))


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")
