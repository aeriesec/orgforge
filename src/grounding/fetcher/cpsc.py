"""
CPSC SaferProducts.gov — Consumer Product Safety Commission.

Public-domain. Only the `Recall` endpoint is currently exposed; consumer
complaints (`Complaint`) are not in the public REST API as of mid-2026.
We use Recall data as the texture donor for product-safety language —
recalls describe defects in the same surface form a manufacturer would use
internally when documenting an issue.

For VeloMind, this maps to:
  - beta_rider_safety_report (manufacturer-side language about a defect)
  - hardware_firmware_incident_thread (defect descriptions)
  - dealer_escalation (recall communications)
"""
from __future__ import annotations

import logging
from typing import Optional

from .base import BaseFetcher
from .seed import Seed, LICENSE_PUBLIC_US_GOV

logger = logging.getLogger("orgforge.grounding.fetcher.cpsc")


class CPSCRecallFetcher(BaseFetcher):
    name = "cpsc_saferproducts"
    tier = 1
    license_tag = LICENSE_PUBLIC_US_GOV

    BASE_URL = "https://www.saferproducts.gov/RestWebServices/Recall"

    # Heuristic keyword filters — we're seeding for an e-bike company so we
    # rank candidates that mention these terms higher.
    EBIKE_HINT_KEYWORDS = (
        "bicycle", "bike", "battery", "lithium", "charger", "motor",
        "wheel", "scooter", "rider", "fall", "fire", "burn", "shock",
    )

    def _query(self, genre: str, n: int, query_hint: str) -> list[Seed]:
        payload = self._cached_get(self.BASE_URL, params={"format": "json"})
        if not payload:
            return []
        # The bare endpoint returns recent recalls; we filter & rank.
        ranked = self._rank(payload, query_hint)
        seeds: list[Seed] = []
        for item in ranked:
            if len(seeds) >= n:
                break
            seed = self._seed_from_recall(item, genre)
            if seed:
                seeds.append(seed)
        logger.info("[cpsc] fetched %d seeds for genre=%s", len(seeds), genre)
        return seeds

    def _rank(self, items: list[dict], query_hint: str) -> list[dict]:
        hint = (query_hint or "").lower()

        def score(item: dict) -> int:
            text = (
                (item.get("Description") or "")
                + " "
                + (item.get("Title") or "")
            ).lower()
            s = 0
            for kw in self.EBIKE_HINT_KEYWORDS:
                if kw in text:
                    s += 1
            if hint:
                if hint in text:
                    s += 5
            return s

        return sorted(items, key=score, reverse=True)

    def _seed_from_recall(self, item: dict, genre: str) -> Optional[Seed]:
        recall_id = item.get("RecallID") or item.get("RecallNumber")
        if not recall_id:
            return None
        title = item.get("Title") or f"CPSC Recall {recall_id}"
        body = item.get("Description") or ""
        url = item.get("URL") or f"https://www.cpsc.gov/Recalls/?id={recall_id}"
        return self._make_seed(
            seed_id=f"cpsc/{recall_id}",
            url=url,
            title=title,
            body=body,
            actors=[
                m.get("Name", "")
                for m in (item.get("Manufacturers") or [])
                if isinstance(m, dict)
            ],
            timestamps=[str(item.get("RecallDate") or "")],
            topic_tags=[
                p.get("Name", "")
                for p in (item.get("Products") or [])
                if isinstance(p, dict)
            ],
            target_genre=genre,
            raw=item,
        )
