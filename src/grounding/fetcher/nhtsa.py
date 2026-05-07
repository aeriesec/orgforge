"""
NHTSA fetchers — National Highway Traffic Safety Administration.

Two endpoints, both public-domain:
  - VOQ Complaints  : real consumer safety complaints (rider-near-miss texture)
  - Recalls / TSBs  : real manufacturer-to-public technical bulletins

For VeloMind these map to:
  - beta_rider_safety_report          → VOQ
  - dealer_escalation                  → VOQ (filtered to dealer-keyword)
  - hardware_firmware_incident_thread → Recalls (manufacturer-tone)
  - pilot_line_manufacturing_bulletin → Recalls
  - supplier_eco_vendor_email         → Recalls (filtered to component issues)
"""
from __future__ import annotations

import logging
from typing import Iterable

from .base import BaseFetcher
from .seed import Seed, LICENSE_PUBLIC_US_GOV

logger = logging.getLogger("orgforge.grounding.fetcher.nhtsa")

# We hand-pick a small set of vehicle make/model combos rich in
# electrical/firmware/battery complaints to seed the corpus.
EBIKE_PROXY_QUERIES: list[tuple[str, str, int]] = [
    # (make, model, modelYear) — vehicles with rich battery/firmware/OTA
    # complaint texture that resembles VeloMind's incident genre.
    ("Tesla", "Model 3", 2024),
    ("Tesla", "Model Y", 2024),
    ("Rivian", "R1T", 2024),
    ("Ford", "Mustang Mach-E", 2024),
    ("Chevrolet", "Bolt EV", 2023),
    ("Hyundai", "Ioniq 5", 2024),
]


class NHTSAVOQFetcher(BaseFetcher):
    """NHTSA VOQ — consumer safety complaints (rider/driver voice)."""

    name = "nhtsa_voq"
    tier = 1
    license_tag = LICENSE_PUBLIC_US_GOV

    BASE_URL = "https://api.nhtsa.gov/complaints/complaintsByVehicle"

    def _query(self, genre: str, n: int, query_hint: str) -> list[Seed]:
        seeds: list[Seed] = []
        for make, model, year in EBIKE_PROXY_QUERIES:
            if len(seeds) >= n:
                break
            payload = self._cached_get(
                self.BASE_URL,
                params={"make": make, "model": model, "modelYear": str(year)},
            )
            if not payload or not payload.get("results"):
                continue
            results = payload["results"]
            # Optional topical filter — keep complaints whose summary contains
            # the hint terms.
            if query_hint:
                hint_lower = query_hint.lower()
                ranked = sorted(
                    results,
                    key=lambda r: int(hint_lower in (r.get("summary") or "").lower()),
                    reverse=True,
                )
            else:
                ranked = results
            for item in ranked:
                if len(seeds) >= n:
                    break
                seeds.append(self._seed_from_voq(item, genre, make, model, year))
        logger.info("[nhtsa_voq] fetched %d seeds for genre=%s", len(seeds), genre)
        return seeds

    def _seed_from_voq(
        self, item: dict, genre: str, make: str, model: str, year: int
    ) -> Seed:
        summary = item.get("summary") or ""
        components = item.get("components") or ""
        return self._make_seed(
            seed_id=f"nhtsa_voq/{item.get('odiNumber')}",
            url=(
                "https://www.nhtsa.gov/recalls?nhtsaId=" f"{item.get('odiNumber')}"
            ),
            title=f"VOQ {item.get('odiNumber')}: {components}",
            body=summary,
            actors=[item.get("manufacturer") or "Unknown manufacturer"],
            timestamps=[
                str(item.get("dateOfIncident") or ""),
                str(item.get("dateComplaintFiled") or ""),
            ],
            topic_tags=[
                t.strip() for t in components.split(",") if t.strip()
            ] + [f"{make} {model} {year}"],
            target_genre=genre,
            raw=item,
        )


class NHTSARecallFetcher(BaseFetcher):
    """NHTSA Recalls — manufacturer-tone technical bulletins."""

    name = "nhtsa_tsb"
    tier = 1
    license_tag = LICENSE_PUBLIC_US_GOV

    BASE_URL = "https://api.nhtsa.gov/recalls/recallsByVehicle"

    def _query(self, genre: str, n: int, query_hint: str) -> list[Seed]:
        seeds: list[Seed] = []
        for make, model, year in EBIKE_PROXY_QUERIES:
            if len(seeds) >= n:
                break
            payload = self._cached_get(
                self.BASE_URL,
                params={"make": make, "model": model, "modelYear": str(year)},
            )
            if not payload or not payload.get("results"):
                continue
            results = payload["results"]
            if query_hint:
                hint_lower = query_hint.lower()
                results = sorted(
                    results,
                    key=lambda r: int(
                        hint_lower in (r.get("Summary") or "").lower()
                        or hint_lower in (r.get("Component") or "").lower()
                    ),
                    reverse=True,
                )
            for item in results:
                if len(seeds) >= n:
                    break
                seeds.append(self._seed_from_recall(item, genre, make, model, year))
        logger.info("[nhtsa_tsb] fetched %d seeds for genre=%s", len(seeds), genre)
        return seeds

    def _seed_from_recall(
        self, item: dict, genre: str, make: str, model: str, year: int
    ) -> Seed:
        body_parts = [
            item.get("Summary") or "",
            "",
            "Consequence: " + (item.get("Consequence") or ""),
            "",
            "Remedy: " + (item.get("Remedy") or ""),
        ]
        body = "\n".join(p for p in body_parts if p).strip()
        return self._make_seed(
            seed_id=f"nhtsa_recall/{item.get('NHTSACampaignNumber')}",
            url=(
                "https://www.nhtsa.gov/recalls?nhtsaId="
                f"{item.get('NHTSACampaignNumber')}"
            ),
            title=(
                f"Recall {item.get('NHTSACampaignNumber')}: "
                f"{item.get('Component') or 'unspecified'}"
            ),
            body=body,
            actors=[item.get("Manufacturer") or "Unknown manufacturer"],
            timestamps=[str(item.get("ReportReceivedDate") or "")],
            topic_tags=[item.get("Component") or "", f"{make} {model} {year}"],
            target_genre=genre,
            raw=item,
        )
