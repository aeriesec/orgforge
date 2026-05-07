"""
SEC EDGAR — full-text search over public-company filings.

Two endpoints used:
  - efts.sec.gov/LATEST/search-index : Elasticsearch full-text search
  - www.sec.gov/Archives/edgar/data/... : individual filing text

Public domain (US gov filings). For VeloMind we use this for two genres:
  - supplier_eco_vendor_email   → EX-10 contracts (real corporate vendor MSAs)
  - investor_diligence_qa        → S-1 risk-factor and management-discussion
                                    sections (closest analog to investor Q&A)
  - exec_launch_decision         → S-1 risk factors for launch-readiness tone
"""
from __future__ import annotations

import logging
from typing import Optional

from .base import BaseFetcher
from .seed import Seed, LICENSE_PUBLIC_US_GOV

logger = logging.getLogger("orgforge.grounding.fetcher.sec_edgar")


class SECEdgarFetcher(BaseFetcher):
    name = "sec_edgar"
    tier = 1
    license_tag = LICENSE_PUBLIC_US_GOV

    SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"

    # SEC full-text search filters by *parent* form (10-K, S-1, etc.).
    # EX-10 contracts appear as exhibits inside those parent filings, so we
    # query parent forms with phrases likely to appear in EX-10 content.
    GENRE_TO_FORMS: dict[str, list[str]] = {
        "supplier_eco_vendor_email": ["10-K", "10-Q", "S-1"],
        "investor_diligence_qa": ["S-1", "S-1/A"],
        "exec_launch_decision": ["S-1", "10-K"],
    }

    DEFAULT_SUBSOURCES: dict[str, str] = {
        "supplier_eco_vendor_email": "supply agreement manufacturing",
        "investor_diligence_qa": "risk factors supply concentration",
        "exec_launch_decision": "launch readiness manufacturing",
    }

    def _query(self, genre: str, n: int, query_hint: str) -> list[Seed]:
        forms = self.GENRE_TO_FORMS.get(genre)
        if not forms:
            return []
        q = query_hint.strip() or self.DEFAULT_SUBSOURCES.get(genre, "")
        if not q:
            return []
        # SEC EDGAR full-text search treats unquoted queries as OR-of-terms,
        # which gives broader recall. Wrapping in quotes gave 0 hits for many
        # phrases. We use the unquoted form intentionally.
        payload = self._cached_get(
            self.SEARCH_URL,
            params={"q": q, "forms": ",".join(forms)},
        )
        if not payload:
            return []
        hits = (payload.get("hits") or {}).get("hits") or []
        seeds: list[Seed] = []
        for hit in hits[: n * 2]:
            if len(seeds) >= n:
                break
            seed = self._seed_from_hit(hit, genre)
            if seed:
                seeds.append(seed)
        logger.info("[sec_edgar] fetched %d seeds for genre=%s", len(seeds), genre)
        return seeds

    def _seed_from_hit(self, hit: dict, genre: str) -> Optional[Seed]:
        src = hit.get("_source") or {}
        adsh = src.get("adsh") or ""
        if not adsh:
            return None
        # adsh is e.g. "0000950170-24-012345" -> filing index page on edgar
        clean = adsh.replace("-", "")
        cik = (src.get("ciks") or [""])[0]
        url = (
            f"https://www.sec.gov/Archives/edgar/data/{cik}/{clean}/"
            if cik else f"https://efts.sec.gov/LATEST/search-index?q={adsh}"
        )
        display = src.get("display_names") or []
        actors = display if isinstance(display, list) else [str(display)]
        title = " — ".join(
            x for x in [
                src.get("forms", [""])[0] if src.get("forms") else "",
                src.get("file_type") or "",
                actors[0] if actors else "",
            ] if x
        )
        body = (
            src.get("description") or src.get("displayNames") or ""
        )
        return self._make_seed(
            seed_id=f"sec_edgar/{adsh}",
            url=url,
            title=title or f"SEC filing {adsh}",
            body=body if isinstance(body, str) else str(body),
            actors=actors,
            timestamps=[src.get("file_date") or ""],
            topic_tags=src.get("forms") or [],
            target_genre=genre,
            raw=src,
        )
