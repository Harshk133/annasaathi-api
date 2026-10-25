"""Current temperature and humidity from Open-Meteo (free, no key).
Demo mode and network failures fall back to a typical Pune afternoon."""
from __future__ import annotations

import os
import time

import httpx

FALLBACK = {"temp_c": 31.0, "humidity": 62.0, "source": "typical Pune afternoon (offline)"}
_cache: dict[tuple, tuple[float, dict]] = {}


def current(lat: float, lon: float, city: str = "") -> dict:
    if os.environ.get("ANNA_DEMO") == "1":
        return {**FALLBACK, "source": "demo mode", "city": city}
    key = (round(lat, 2), round(lon, 2))
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < 1800:
        return hit[1]
    try:
        # Shelf life follows the day's high, not the reading at the moment of the request.
        r = httpx.get("https://api.open-meteo.com/v1/forecast", timeout=2.5, params={
            "latitude": lat, "longitude": lon, "current": "temperature_2m,relative_humidity_2m",
            "daily": "temperature_2m_max", "forecast_days": 1, "timezone": "auto"})
        r.raise_for_status()
        body = r.json()
        cur = body["current"]
        w = {"temp_c": float(body["daily"]["temperature_2m_max"][0]), "now_c": float(cur["temperature_2m"]),
             "humidity": float(cur["relative_humidity_2m"]), "source": "Open-Meteo live", "city": city}
    except (httpx.HTTPError, KeyError, ValueError):
        w = {**FALLBACK, "city": city}
    _cache[key] = (time.time(), w)
    return w
