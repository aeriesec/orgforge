"""
Genre → fetcher routing dispatcher.

Reads `genre_taxonomy.yaml`. For a given (genre, event_context) pair:

  1. Looks up the ordered source list for the genre (with industry override
     applied if the company config sets industry: <name>).
  2. For each source in order, instantiates the fetcher and calls either
     fetch_for_genre (Stage 1 — bulk profile corpus) or fetch_topical
     (Stage 2 — per-event topical match), accumulating seeds.
  3. Returns when `n` seeds are collected or all sources exhausted.

If a source is reachable for genre A but unreachable for the current run,
the dispatcher silently moves to the next source — partial coverage is OK
for the POC.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import yaml

from .base import BaseFetcher
from .seed import Seed
from .nhtsa import NHTSAVOQFetcher, NHTSARecallFetcher
from .cpsc import CPSCRecallFetcher
from .sec_edgar import SECEdgarFetcher
from .ros_discourse import ROSDiscourseFetcher
from .convokit_local import ConvokitFetcher
from .irc_disentanglement import IRCDisentanglementFetcher
from .apache_lists import ApacheListsFetcher
from .forum_scraper import (
    EndlessSphereFetcher,
    AdafruitForumFetcher,
    BikeForumsIndustryFetcher,
)

logger = logging.getLogger("orgforge.grounding.fetcher.dispatcher")

TAXONOMY_PATH = Path(__file__).parent.parent / "genre_taxonomy.yaml"

# Map source-name (as used in genre_taxonomy.yaml) → fetcher class.
SOURCE_REGISTRY: dict[str, type[BaseFetcher]] = {
    "nhtsa_voq": NHTSAVOQFetcher,
    "nhtsa_complaints": NHTSAVOQFetcher,  # alias used in taxonomy
    "nhtsa_tsb": NHTSARecallFetcher,
    "nhtsa_investigations": NHTSARecallFetcher,  # alias
    "cpsc_saferproducts": CPSCRecallFetcher,
    "sec_edgar_ex10": SECEdgarFetcher,
    "sec_edgar_s1": SECEdgarFetcher,
    "sec_edgar_s1_risk_factors": SECEdgarFetcher,
    "ros_discourse": ROSDiscourseFetcher,
    "irc_disentanglement": IRCDisentanglementFetcher,
    "apache_lists": ApacheListsFetcher,
    "convokit_reddit_managers": ConvokitFetcher,
    "convokit_reddit_ebikes": ConvokitFetcher,
    "convokit_wikipedia_afd": ConvokitFetcher,
    "convokit_fomc": ConvokitFetcher,
    "endless_sphere": EndlessSphereFetcher,
    "adafruit_forum": AdafruitForumFetcher,
    "bikeforums_industry": BikeForumsIndustryFetcher,
    # the taxonomy file uses several other names that don't yet have fetchers;
    # they're silently skipped by the dispatcher.
}


class FetcherDispatcher:
    def __init__(
        self,
        taxonomy_path: Path = TAXONOMY_PATH,
        industry: Optional[str] = None,
    ):
        with taxonomy_path.open() as f:
            self.taxonomy = yaml.safe_load(f)
        self.industry = industry
        self._instances: dict[str, BaseFetcher] = {}

    def _instance(self, source_name: str) -> Optional[BaseFetcher]:
        if source_name in self._instances:
            return self._instances[source_name]
        cls = SOURCE_REGISTRY.get(source_name)
        if cls is None:
            return None
        try:
            inst = cls()
        except Exception as exc:
            logger.warning("[dispatcher] failed to construct %s: %s", source_name, exc)
            return None
        self._instances[source_name] = inst
        return inst

    def _sources_for_genre(self, genre: str) -> list[dict]:
        # Industry override takes precedence
        if self.industry:
            industries = self.taxonomy.get("industries") or {}
            override = (industries.get(self.industry) or {}).get(genre)
            if override and "sources" in override:
                return override["sources"]
        gconf = (self.taxonomy.get("genres") or {}).get(genre) or {}
        return gconf.get("sources") or []

    def stage1_corpus(self, genre: str, n: Optional[int] = None) -> list[Seed]:
        """One-time per-genre corpus pull for profile extraction."""
        gconf = (self.taxonomy.get("genres") or {}).get(genre) or {}
        if n is None:
            n = int(gconf.get("sample_n", 10))
        sources = self._sources_for_genre(genre)
        seeds: list[Seed] = []
        for src in sources:
            if len(seeds) >= n:
                break
            inst = self._instance(src.get("name") or "")
            if not inst:
                continue
            try:
                got = inst.fetch_for_genre(genre, n=n - len(seeds))
            except Exception as exc:
                logger.warning(
                    "[dispatcher] %s.fetch_for_genre failed: %s",
                    src.get("name"), exc,
                )
                continue
            seeds.extend(got)
        logger.info(
            "[dispatcher] stage1 genre=%s collected %d/%d seeds across %d sources",
            genre, len(seeds), n, len(sources),
        )
        return seeds

    def stage2_topical(
        self,
        genre: str,
        query_hint: str,
        n: int = 3,
    ) -> list[Seed]:
        """Per-event topical pull. Caller composes query_hint from event
        context (component, symptom, actor mix, etc.)."""
        sources = self._sources_for_genre(genre)
        seeds: list[Seed] = []
        for src in sources:
            if len(seeds) >= n:
                break
            inst = self._instance(src.get("name") or "")
            if not inst:
                continue
            try:
                got = inst.fetch_topical(genre, query_hint=query_hint, n=n - len(seeds))
            except Exception as exc:
                logger.warning(
                    "[dispatcher] %s.fetch_topical failed: %s",
                    src.get("name"), exc,
                )
                continue
            seeds.extend(got)
        return seeds

    def stats(self) -> dict[str, int]:
        return {name: 1 for name in self._instances.keys()}
