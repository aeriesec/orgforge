"""
Normalized seed schema. Every fetcher (Tier 1/2/3) returns objects of this
shape so downstream profile extraction and topical retrieval are uniform.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class Seed:
    """A single real-world artifact, fetched from a specific source.

    Bodies are stored *inline* during the fetch session (small enough for
    profile extraction). After Stage 1, these are summarized into the
    cached genre profile and the raw bodies are deleted from disk.
    """

    seed_id: str                     # globally unique, e.g. "nhtsa_voq/11733743"
    source: str                      # short canonical name, e.g. "nhtsa_voq"
    tier: int                        # 1, 2, or 3
    license: str                     # short tag, e.g. "public_domain_us_gov"
    url: str                         # canonical URL of the artifact
    fetched_at: str                  # ISO 8601 UTC

    # Structured fields (fetcher does its best to fill these)
    title: str = ""
    body: str = ""
    actors: list[str] = field(default_factory=list)
    timestamps: list[str] = field(default_factory=list)
    topic_tags: list[str] = field(default_factory=list)

    # Genre tag this seed was fetched FOR (Stage 1 corpus or Stage 2 topical match)
    target_genre: str = ""
    # Free-form raw payload for source-specific fields
    raw: dict[str, Any] = field(default_factory=dict)
    pii_scrubbed: bool = False

    @classmethod
    def now_iso(cls) -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# License tags we use across the corpus.
LICENSE_PUBLIC_US_GOV = "public_domain_us_gov"
LICENSE_CC_BY_4 = "cc_by_4_0"
LICENSE_CC_BY_SA_4 = "cc_by_sa_4_0"
LICENSE_CC_BY_NC = "cc_by_nc_4_0"
LICENSE_RESEARCH_USE = "research_use_only"
LICENSE_PUBLIC_RECORDS = "public_records"
