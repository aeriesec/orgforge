"""
Tier-3 disciplined forum scraper.

A single class that handles vBulletin/phpBB/Discourse-style forum threads
listed by URL pattern. robots.txt is respected, requests are throttled,
identified User-Agent.

We use it for:
  - Endless Sphere   (e-bike DIY/technical discussion)
  - Adafruit Forum   (hardware engineering project discussion)
  - BikeForums.net   (specialty bicycle retail Industry & Mechanic forum)

Each instance is parameterized by a list of seed thread URLs. Discovery is
intentionally manual — we don't crawl an entire forum, only fetch a curated
list of threads from the configuration.

When robots.txt disallows or the host is unreachable, we degrade gracefully
to an empty list.
"""
from __future__ import annotations

import logging
import re
from typing import Iterable, Optional

from .base import BaseFetcher
from .seed import Seed, LICENSE_RESEARCH_USE

logger = logging.getLogger("orgforge.grounding.fetcher.forum_scraper")


class ForumScraperFetcher(BaseFetcher):
    """Generic forum HTML scraper. Subclasses configure curated thread URLs
    per genre and an HTML extractor."""

    name = "forum_scraper"
    tier = 3
    license_tag = LICENSE_RESEARCH_USE
    respect_robots = True
    throttle_seconds = 2.0

    # Subclasses set these.
    SEED_THREADS: dict[str, list[str]] = {}  # genre -> [thread_url, ...]
    DISPLAY_NAME: str = "forum_scraper"

    # Crude HTML→text extractor; subclasses can override for forum-specific markup.
    POST_REGEX = re.compile(
        r"<(?:div|article)[^>]*class=\"[^\"]*post[^\"]*\"[^>]*>(.*?)</(?:div|article)>",
        re.I | re.S,
    )
    TAG_STRIP = re.compile(r"<[^>]+>")

    def _query(self, genre: str, n: int, query_hint: str) -> list[Seed]:
        urls = self.SEED_THREADS.get(genre) or []
        if not urls:
            return []
        seeds: list[Seed] = []
        for url in urls[: n * 2]:
            if len(seeds) >= n:
                break
            html = self._cached_get(url, json_response=False)
            if not html:
                continue
            seed = self._seed_from_html(url, html, genre)
            if seed:
                seeds.append(seed)
        logger.info(
            "[%s] fetched %d seeds for genre=%s",
            self.name, len(seeds), genre,
        )
        return seeds

    def _seed_from_html(self, url: str, html: str, genre: str) -> Optional[Seed]:
        title_match = re.search(r"<title>(.*?)</title>", html, re.I | re.S)
        title = (title_match.group(1).strip() if title_match else url)
        # Extract post bodies; if structured extraction fails, fall back to a
        # plain-text strip of the whole page.
        posts = self.POST_REGEX.findall(html)
        if not posts:
            body = self.TAG_STRIP.sub(" ", html)
            body = re.sub(r"\s+", " ", body).strip()[:8000]
        else:
            body = "\n\n---\n\n".join(
                self.TAG_STRIP.sub(" ", p).strip()[:1500] for p in posts[:20]
            )
        return self._make_seed(
            seed_id=f"{self.name}/{abs(hash(url)) % (10**12)}",
            url=url,
            title=title,
            body=body,
            actors=[],  # username extraction is forum-specific; subclass override
            timestamps=[],
            topic_tags=[self.DISPLAY_NAME],
            target_genre=genre,
            raw={"source_url": url},
        )


class EndlessSphereFetcher(ForumScraperFetcher):
    """Endless Sphere e-bike/EV community."""

    name = "endless_sphere"
    DISPLAY_NAME = "endless_sphere"
    SEED_THREADS = {
        "hardware_firmware_incident_thread": [
            "https://endless-sphere.com/forums/viewtopic.php?t=121011",
            "https://endless-sphere.com/forums/viewtopic.php?t=120442",
        ],
        "battery_lab_discussion": [
            "https://endless-sphere.com/forums/viewtopic.php?t=119876",
            "https://endless-sphere.com/forums/viewtopic.php?t=118994",
        ],
    }


class AdafruitForumFetcher(ForumScraperFetcher):
    """Adafruit Forum — hardware engineering project discussion."""

    name = "adafruit_forum"
    DISPLAY_NAME = "adafruit_forum"
    SEED_THREADS = {
        "knowledge_gap_escalation": [
            "https://forums.adafruit.com/viewtopic.php?f=19&t=200001",
        ],
        "hardware_firmware_incident_thread": [
            "https://forums.adafruit.com/viewtopic.php?f=19&t=200042",
        ],
    }


class BikeForumsIndustryFetcher(ForumScraperFetcher):
    """BikeForums.net Industry & Mechanic forum (dealer-shop discussion)."""

    name = "bikeforums_industry"
    DISPLAY_NAME = "bikeforums_industry"
    SEED_THREADS = {
        "dealer_escalation": [
            "https://www.bikeforums.net/professional-cycling-racing-industry/",
        ],
    }
