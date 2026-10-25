"""Household simulator: 50 synthetic households × 90 days, baseline habits vs
AnnaSaathi advice followed some of the time. Fixed seed, so every run gives the
same numbers.

Both arms see the same households and the same day-by-day appetite (common
random numbers), so the difference comes only from how they buy and cook.
Waste is calibrated once so that the baseline matches UNEP's 55 kg per person
per year for Indian households; the same factor scales both arms.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from .data import DATA_DIR, catalogue, factors, tiers

SEED = 42
N_HOUSEHOLDS = 50
DAYS = 90
SHOP_EVERY = 7
UNEP_KG_PER_PERSON_YEAR = 55.0
COOKED_RUPEES_PER_KG = 60.0
GROUPS = ["adult", "elder", "teen", "child", "toddler"]


@dataclass
class Lot:
    qty: float
    bought: int
    life: float  # true days until it spoils
    est_life: float  # what the planner thinks


@dataclass
class HH:
    persons: int
    ae: float
    temp: float
    overbuy: float
    items: dict[str, float]  # item -> true daily use
    est_err: dict[str, float]
    demand: dict[str, list[float]] = field(default_factory=dict)
    cooked: list[float] = field(default_factory=list)


def _weather_factor(temp: float) -> float:
    f = factors()
    return f["weather:hot_35"] if temp >= 35 else f["weather:hot_30"] if temp >= 30 else f["weather:warm_25"] if temp >= 25 else 1.0


def _sim_items() -> list[str]:
    # the typical weekly basket (catalogue column `sim`), not every item a family could buy
    return [i.id for i in catalogue().values() if i.sim and (i.tier.startswith("D") or i.tier in ("M1", "M2"))]


def make_households(seed: int = SEED) -> list[HH]:
    rng = random.Random(seed)
    cat = catalogue()
    f = factors()
    out = []
    for _ in range(N_HOUSEHOLDS):
        size = rng.choices([2, 3, 4, 5, 6], weights=[10, 20, 35, 25, 10])[0]
        groups = ["adult", "adult"][: min(2, size)]
        while len(groups) < size:
            groups.append(rng.choices(["elder", "teen", "child", "toddler"], weights=[30, 25, 35, 10])[0])
        ae = sum(f[f"member:{g}"] for g in groups) * rng.choice([0.85, 1.0, 1.0, 1.15]) * rng.uniform(0.85, 0.95)
        egg = rng.random() < 0.35
        items, est = {}, {}
        for iid in _sim_items():
            it = cat[iid]
            if it.diet == "egg" and not egg:
                continue
            if rng.random() > (0.95 if it.category in ("dairy", "flour") else 0.7):
                continue
            items[iid] = it.use_per_ae * ae * math.exp(rng.gauss(0, 0.3))
            est[iid] = math.exp(rng.gauss(0, 0.2))  # formula error before any learning
        hh = HH(persons=size, ae=ae, temp=rng.gauss(31, 3), overbuy=rng.uniform(1.15, 1.6), items=items, est_err=est)
        for iid, u in items.items():
            hh.demand[iid] = [0.0 if rng.random() < 0.08 else u * rng.uniform(0.55, 1.45) for _ in range(DAYS)]
        hh.cooked = [ae * 0.45 * rng.uniform(0.9, 1.1) for _ in range(DAYS)]
        out.append(hh)
    return out


def _round_pack(q: float, pack: float) -> float:
    return 0.0 if q < 0.3 * pack else max(1, math.ceil(q / pack - 0.2)) * pack


def _life(rng: random.Random, iid: str, temp: float) -> tuple[float, float]:
    it = catalogue()[iid]
    t = tiers()[it.tier]
    k = _weather_factor(temp) if it.storage == "pantry" and it.form in ("fresh", "wet") else 1.0
    true = rng.uniform(t.min_days, t.max_days) * k
    return true, (t.min_days + t.max_days) / 2 * k


def simulate(hh: HH, follow: float, seed: int) -> dict:
    """Run one household. follow=None means pure baseline habits."""
    cat = catalogue()
    rng = random.Random(seed)  # purchase/spoilage draws
    adv = random.Random(seed * 7 + 1)  # whether advice is followed
    stock: dict[str, list[Lot]] = {i: [] for i in hh.items}
    u_est = {i: hh.items[i] * hh.est_err[i] for i in hh.items}
    eaten_week = {i: 0.0 for i in hh.items}
    ran_out = {i: False for i in hh.items}
    topups: dict[str, list[tuple[int, float]]] = {i: [] for i in hh.items}
    waste_kg = waste_rs = 0.0
    shortages = 0
    curve = []
    by_cat: dict[str, float] = {}
    baseline = follow is None

    def buy(iid, q, day):
        if q > 1e-9:
            life, est = _life(rng, iid, hh.temp)
            stock[iid].append(Lot(q, day, life, est))

    for day in range(DAYS):
        if day % SHOP_EVERY == 0:
            if not baseline and day > 0 and adv.random() < follow:  # weekly 10-second check
                for i in hh.items:
                    seen = eaten_week[i] / SHOP_EVERY
                    # "finished" before the shop means demand was higher than what we saw eaten
                    u_est[i] = max(u_est[i] * 1.15, seen) if ran_out[i] else 0.5 * u_est[i] + 0.5 * seen
            eaten_week = {i: 0.0 for i in hh.items}
            ran_out = {i: False for i in hh.items}
            for i in hh.items:
                it = cat[i]
                if it.tier == "D1":
                    continue
                have = sum(l.qty for l in stock[i])
                habit = _round_pack(hh.items[i] * SHOP_EVERY * hh.overbuy, it.pack_size)
                if not baseline and adv.random() < follow:
                    t = tiers()[it.tier]
                    k = _weather_factor(hh.temp) if it.storage == "pantry" and it.form in ("fresh", "wet") else 1.0
                    shelf = (t.min_days + t.max_days) / 2 * k
                    cover = min(shelf, SHOP_EVERY)
                    fresh = sum(l.qty for l in stock[i] if l.est_life - (day - l.bought) > 0.5)
                    buy(i, _round_pack(u_est[i] * cover * 1.1 - fresh, it.pack_size), day)
                    step = max(1, int(shelf))
                    for t in range(day + step, day + SHOP_EVERY, step):  # short-life items: small top-ups
                        topups[i].append((t, u_est[i] * min(step, day + SHOP_EVERY - t) * 1.1))
                else:
                    buy(i, habit if have < 0.5 * habit else _round_pack(habit / 2, it.pack_size), day)
        for i in hh.items:  # 1-2 day items (milk) come daily, as in most Indian homes
            it = cat[i]
            if it.tier != "D1":
                continue
            have = sum(l.qty for l in stock[i])
            if not baseline and adv.random() < follow:
                buy(i, _round_pack(u_est[i] * 1.05 - have, it.pack_size), day)
            else:
                buy(i, _round_pack(hh.items[i] * (1 + (hh.overbuy - 1) / 2) - have * 0.5, it.pack_size), day)
        if not baseline:
            for i in hh.items:
                due = [t for t in topups[i] if t[0] == day]
                for _, q in due:
                    fresh = sum(l.qty for l in stock[i] if l.est_life - (day - l.bought) > 0.5)
                    buy(i, _round_pack(q - fresh, cat[i].pack_size), day)
                topups[i] = [t for t in topups[i] if t[0] != day]
        cook_first = not baseline and adv.random() < follow
        for i in hh.items:
            it = cat[i]
            need = hh.demand[i][day]
            lots = stock[i]
            if cook_first:
                lots.sort(key=lambda l: l.est_life - (day - l.bought))
                # rescue: an item about to turn gets cooked a little extra in place of something else
                if lots and lots[0].est_life - (day - lots[0].bought) <= 1.5:
                    need *= 1.25
            elif rng.random() < 0.5:
                lots.sort(key=lambda l: -l.bought)  # habit: grab the newest
            else:
                lots.sort(key=lambda l: l.bought)
            want = hh.demand[i][day]
            for l in lots:
                take = min(l.qty, need)
                l.qty -= take
                need -= take
                eaten_week[i] += take
                if need <= 1e-9:
                    break
            if want > 1e-9 and need > want * 0.5:  # really ran out, not just a lighter portion
                shortages += 1
                ran_out[i] = True
                if (SHOP_EVERY - day % SHOP_EVERY) > 2:  # quick top-up from the corner shop
                    buy(i, _round_pack(hh.items[i] * 3, it.pack_size), day)
            keep = []
            for l in lots:
                if l.qty <= 1e-9:
                    continue
                if day - l.bought >= l.life:
                    kg = it.kg(l.qty)
                    waste_kg += kg
                    waste_rs += l.qty * it.price
                    by_cat[it.category] = by_cat.get(it.category, 0) + kg
                else:
                    keep.append(l)
            stock[i] = keep
        # cooked food: leftovers of wet dishes that nobody finishes in time
        follow_today = cook_first
        left = hh.cooked[day] * (rng.uniform(0.05, 0.1) * (0.55 if follow_today else 1.0))
        spoiled = left * (0.35 if follow_today else 0.6)
        waste_kg += spoiled
        waste_rs += spoiled * COOKED_RUPEES_PER_KG
        by_cat["cooked"] = by_cat.get("cooked", 0) + spoiled
        curve.append(waste_kg)
    return {"waste_kg": waste_kg, "waste_rs": waste_rs, "shortages": shortages, "curve": curve, "by_cat": by_cat}


def _run_arm(hhs, follow):
    res = [simulate(h, follow, SEED + n) for n, h in enumerate(hhs)]
    months = DAYS / 30
    n = len(hhs)
    return {
        "kg": sum(r["waste_kg"] for r in res) / n / months,
        "rs": sum(r["waste_rs"] for r in res) / n / months,
        "shortages": sum(r["shortages"] for r in res) / n / months,
        "curve": [sum(r["curve"][d] for r in res) / n for d in range(DAYS)],
        "by_cat": {k: sum(r["by_cat"].get(k, 0) for r in res) / n / months
                   for k in sorted({k for r in res for k in r["by_cat"]})},
    }


@lru_cache
def results() -> dict:
    import hashlib

    src = [DATA_DIR / n for n in ("catalogue.csv", "shelf_life_tiers.csv", "factors.csv")] + [Path(__file__)]
    key = hashlib.sha1(b"".join(f.read_bytes() for f in src)).hexdigest()[:12]
    cache = DATA_DIR / "sim_cache.json"
    if cache.exists():
        cached = json.loads(cache.read_text(encoding="utf-8"))
        if cached.get("key") == key:
            return cached
    hhs = make_households()
    persons = sum(h.persons for h in hhs) / len(hhs)
    base = _run_arm(hhs, None)
    per_person_year = base["kg"] * 12 / persons
    k = UNEP_KG_PER_PERSON_YEAR / per_person_year

    def scale(arm):
        return {
            "kg": round(arm["kg"] * k, 2), "rs": round(arm["rs"] * k), "shortages": round(arm["shortages"], 2),
            "curve": [round(v * k, 2) for v in arm["curve"][::3]] + [round(arm["curve"][-1] * k, 2)],
            "by_cat": {c: round(v * k, 2) for c, v in arm["by_cat"].items()},
        }

    levels = {}
    for pct in range(0, 101, 10):
        levels[str(pct)] = scale(_run_arm(hhs, pct / 100))
    out = {
        "key": key, "households": len(hhs), "days": DAYS, "seed": SEED, "avg_persons": round(persons, 2),
        "calibration": {"raw_kg_per_person_year": round(per_person_year, 1), "factor": round(k, 3),
                        "target": UNEP_KG_PER_PERSON_YEAR,
                        "note": "Raw simulated waste is scaled by one factor so the baseline equals UNEP's 55 kg per "
                                "person per year; AnnaSaathi is scaled by the same factor, so the % cut is unchanged."},
        "baseline": scale(base), "levels": levels,
    }
    cache.write_text(json.dumps(out), encoding="utf-8")
    return out


if __name__ == "__main__":
    import time

    t0 = time.time()
    (DATA_DIR / "sim_cache.json").unlink(missing_ok=True)
    r = results()
    b = r["baseline"]
    print(f"ran in {time.time() - t0:.1f}s; calibration {r['calibration']}")
    print(f"baseline   {b['kg']:6.2f} kg/hh/mo  ₹{b['rs']:5}  shortages {b['shortages']}")
    for pct in ("50", "70", "100"):
        a = r["levels"][pct]
        print(f"follow {pct:>3}% {a['kg']:6.2f} kg/hh/mo  ₹{a['rs']:5}  shortages {a['shortages']}"
              f"  (−{(1 - a['kg'] / b['kg']) * 100:.0f}%)")
