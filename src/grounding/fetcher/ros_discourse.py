"""
ROS Discourse — Robot Operating System community forum.

Closest open match for VeloMind's small-team hardware+software+firmware+launch
coordination. Discourse exposes JSON at most pages by appending `.json`.

CC-BY for posts (research/non-commercial use OK).

Used for genres: hardware_firmware_incident_thread, battery_lab_discussion,
internal_sprint_planning, knowledge_gap_escalation.
"""
from __future__ import annotations

import logging
from typing import Optional

from .base import BaseFetcher
from .seed import Seed, LICENSE_CC_BY_NC

logger = logging.getLogger("orgforge.grounding.fetcher.ros_discourse")


class ROSDiscourseFetcher(BaseFetcher):
    name = "ros_discourse"
    tier = 2
    license_tag = LICENSE_CC_BY_NC

    SEARCH_URL = "https://discourse.ros.org/search.json"
    TOPIC_URL_FMT = "https://discourse.ros.org/t/{topic_id}.json"

    GENRE_HINTS: dict[str, str] = {
        "hardware_firmware_incident_thread": "firmware bug deadlock fix",
        "battery_lab_discussion": "battery cold weather",
        "internal_sprint_planning": "release planning sprint",
        "knowledge_gap_escalation": "deprecated maintainer left",
        "new_hire_onboarding": "newcomer first time getting started",
        "marketing_launch_copy_review": "documentation wording clarity",
    }

    def _query(self, genre: str, n: int, query_hint: str) -> list[Seed]:
        q = query_hint.strip() or self.GENRE_HINTS.get(genre, "")
        if not q:
            return []
        payload = self._cached_get(self.SEARCH_URL, params={"q": q})
        if not payload:
            logger.warning("[ros_discourse] empty search response (CDN gating?)")
            return []
        topics = payload.get("topics") or []
        seeds: list[Seed] = []
        for t in topics[: n * 2]:
            if len(seeds) >= n:
                break
            seed = self._seed_from_topic(t, genre)
            if seed:
                seeds.append(seed)
        logger.info(
            "[ros_discourse] fetched %d seeds for genre=%s q=%r",
            len(seeds), genre, q,
        )
        return seeds

    def _seed_from_topic(self, t: dict, genre: str) -> Optional[Seed]:
        topic_id = t.get("id")
        if not topic_id:
            return None
        full_url = f"https://discourse.ros.org/t/{topic_id}"
        topic_payload = self._cached_get(self.TOPIC_URL_FMT.format(topic_id=topic_id))
        body = ""
        actors: list[str] = []
        timestamps: list[str] = []
        if topic_payload:
            posts = (topic_payload.get("post_stream") or {}).get("posts") or []
            body = "\n\n".join(
                (p.get("cooked") or p.get("raw") or "") for p in posts[:8]
            )
            actors = list({p.get("username", "") for p in posts if p.get("username")})
            timestamps = [p.get("created_at", "") for p in posts]
        return self._make_seed(
            seed_id=f"ros_discourse/{topic_id}",
            url=full_url,
            title=t.get("title") or f"ROS Discourse #{topic_id}",
            body=body,
            actors=actors,
            timestamps=timestamps,
            topic_tags=t.get("tags") or [],
            target_genre=genre,
            raw=t,
        )
