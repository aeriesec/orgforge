"""
Common fetcher infrastructure: HTTP session, on-disk cache, throttle, robots.txt
respect, identified User-Agent.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import urllib.robotparser
from pathlib import Path
from typing import Any, Iterable, Optional, Protocol
from urllib.parse import urlparse

import requests

from .seed import Seed

logger = logging.getLogger("orgforge.grounding.fetcher")

CACHE_ROOT = Path(__file__).parent.parent / "_cache"
USER_AGENT = (
    "OrgForge-Grounding-POC/0.1 "
    "(research; structural pattern extraction; "
    "contact: research@velomind.example)"
)
DEFAULT_THROTTLE_SECONDS = 2.0


class BaseFetcher:
    """All fetchers inherit this. Subclasses implement :py:meth:`_query`.

    The base class provides:
      - a requests.Session with the identified User-Agent
      - on-disk caching keyed by query hash (raw responses)
      - a per-instance throttle (sleep between requests)
      - robots.txt enforcement (configurable; off for govt/API sources)
    """

    name: str = "base"
    tier: int = 0
    license_tag: str = ""
    respect_robots: bool = False
    throttle_seconds: float = DEFAULT_THROTTLE_SECONDS

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self._last_request_at: float = 0.0
        self._robots_cache: dict[str, urllib.robotparser.RobotFileParser] = {}
        self._cache_dir = CACHE_ROOT / self.name
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ── public API ─────────────────────────────────────────────────────────
    def fetch_for_genre(self, genre: str, n: int) -> list[Seed]:
        """Stage 1 — return up to n seeds representative of the genre."""
        return self._query(genre=genre, n=n, query_hint="")

    def fetch_topical(self, genre: str, query_hint: str, n: int) -> list[Seed]:
        """Stage 2 — return up to n seeds matching this specific event's
        topical context (component / symptom / actor mix)."""
        return self._query(genre=genre, n=n, query_hint=query_hint)

    # ── subclass override ──────────────────────────────────────────────────
    def _query(self, genre: str, n: int, query_hint: str) -> list[Seed]:
        raise NotImplementedError

    # ── shared helpers ─────────────────────────────────────────────────────
    def _throttle(self) -> None:
        now = time.monotonic()
        wait = self.throttle_seconds - (now - self._last_request_at)
        if wait > 0:
            time.sleep(wait)
        self._last_request_at = time.monotonic()

    def _robots_allows(self, url: str) -> bool:
        if not self.respect_robots:
            return True
        parsed = urlparse(url)
        host = f"{parsed.scheme}://{parsed.netloc}"
        rp = self._robots_cache.get(host)
        if rp is None:
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(f"{host}/robots.txt")
            try:
                rp.read()
            except Exception as exc:
                logger.warning(
                    "[%s] robots.txt read failed for %s: %s — defaulting to deny",
                    self.name, host, exc,
                )
                return False
            self._robots_cache[host] = rp
        return rp.can_fetch(USER_AGENT, url)

    def _cache_key(self, url: str, params: Optional[dict] = None) -> str:
        m = hashlib.sha256()
        m.update(url.encode())
        if params:
            m.update(json.dumps(params, sort_keys=True).encode())
        return m.hexdigest()[:16]

    def _cached_get(
        self,
        url: str,
        *,
        params: Optional[dict] = None,
        json_response: bool = True,
        ttl_days: int = 30,
    ) -> Any:
        key = self._cache_key(url, params)
        ext = "json" if json_response else "html"
        cache_file = self._cache_dir / f"{key}.{ext}"
        if cache_file.exists():
            age_days = (time.time() - cache_file.stat().st_mtime) / 86400
            if age_days < ttl_days:
                logger.debug("[%s] cache hit %s", self.name, url)
                if json_response:
                    return json.loads(cache_file.read_text())
                return cache_file.read_text()
        if not self._robots_allows(url):
            logger.warning("[%s] robots.txt disallows %s", self.name, url)
            return None
        self._throttle()
        try:
            r = self.session.get(url, params=params, timeout=30)
            r.raise_for_status()
        except Exception as exc:
            logger.warning("[%s] GET failed %s: %s", self.name, url, exc)
            return None
        if json_response:
            try:
                payload = r.json()
            except Exception:
                logger.warning("[%s] non-JSON response from %s", self.name, url)
                return None
            cache_file.write_text(json.dumps(payload))
            return payload
        cache_file.write_text(r.text)
        return r.text

    # ── helpful for subclasses building Seeds ──────────────────────────────
    def _make_seed(self, **kwargs) -> Seed:
        kwargs.setdefault("source", self.name)
        kwargs.setdefault("tier", self.tier)
        kwargs.setdefault("license", self.license_tag)
        kwargs.setdefault("fetched_at", Seed.now_iso())
        return Seed(**kwargs)


class FetcherProtocol(Protocol):
    """Structural type for the dispatcher."""
    name: str
    tier: int
    def fetch_for_genre(self, genre: str, n: int) -> list[Seed]: ...
    def fetch_topical(self, genre: str, query_hint: str, n: int) -> list[Seed]: ...
