"""A recipe video for a dish, from YouTube's public search page (no API key needed).

Returns the top result's watch link; if YouTube can't be reached or changes its page, the
YouTube search link for the same words is returned instead, so the button always works.
"""
from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote_plus

import httpx

_cache: dict[str, tuple[float, dict]] = {}
TTL = 24 * 3600
VIDEO_ID = re.compile(r'"videoId":"([A-Za-z0-9_-]{11})"')
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
           "Accept-Language": "en-IN,en;q=0.9"}


def search_url(query: str) -> str:
    return f"https://www.youtube.com/results?search_query={quote_plus(query)}"


def recipe_video(dish: str) -> dict:
    query = f"{dish} recipe"
    hit = _cache.get(query)
    if hit and time.time() - hit[0] < TTL:
        return hit[1]
    out = {"query": query, "search_url": search_url(query), "video_id": None, "url": search_url(query)}
    try:
        r = httpx.get("https://www.youtube.com/results", params={"search_query": query, "hl": "en", "gl": "IN"},
                      headers=HEADERS, timeout=5, follow_redirects=True)
        m = VIDEO_ID.search(r.text) if r.status_code == 200 else None
        if m:
            out = {**out, "video_id": m.group(1), "url": f"https://www.youtube.com/watch?v={m.group(1)}",
                   "thumbnail": f"https://i.ytimg.com/vi/{m.group(1)}/hqdefault.jpg"}
    except httpx.HTTPError:
        pass
    _cache[query] = (time.time(), out)
    return out


def recipe_videos(dishes: list[str]) -> dict[str, dict]:
    """Look several dishes up at once (each lookup is a web request)."""
    with ThreadPoolExecutor(max_workers=6) as pool:
        return dict(zip(dishes, pool.map(recipe_video, dishes)))
