"""
Tiered fetcher dispatcher.

Tier 1 — public-domain government APIs (Task #5)
Tier 2 — open-license community archives (Task #6)
Tier 3 — disciplined scrapers (Task #7)

Public API (to be implemented):
    fetch_topical_examples(event_ctx, sources, n=3) -> list[dict]
        For Stage 2: per-event topical fetch.
    fetch_genre_corpus(genre_name, sources, n) -> list[dict]
        For Stage 1: one-time per-genre profile-extraction corpus.
"""
from __future__ import annotations
