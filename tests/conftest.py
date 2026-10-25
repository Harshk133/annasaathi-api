import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def offline_youtube(monkeypatch):
    """Tests never call YouTube: every lookup finds the same fake video, and the cache starts empty."""
    from anna import videos

    videos._cache.clear()
    monkeypatch.setattr(videos, "_fetch", lambda query: '"videoId":"abcdefghijk"')
