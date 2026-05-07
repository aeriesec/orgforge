"""
ConvoKit local-corpus loader.

ConvoKit (Cornell) ships several downloadable corpora that are perfect for
multi-actor decision deliberation and chat-shaped texture:

  - reddit-corpus-small / wiki-articles-for-deletion-corpus / fomc-corpus

This fetcher does NOT live-download (the corpora are several GB). It assumes
they have been downloaded once into:
    ~/.convokit/saved-corpora/<corpus_name>/
or
    src/grounding/_cache/convokit/<corpus_name>/

If the corpus is missing, fetch_for_genre returns an empty list and logs a
hint about how to download.

Used for genres:
  - investor_diligence_qa             → wikipedia_afd (decision deliberation)
  - exec_launch_decision               → fomc / wikipedia_afd
  - new_hire_onboarding                → reddit (r/managers slice)
  - marketing_launch_copy_review       → reddit (r/ebikes slice)
  - beta_rider_safety_report           → reddit (r/ebikes slice)
"""
from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import Optional

from .base import BaseFetcher
from .seed import Seed, LICENSE_CC_BY_SA_4

logger = logging.getLogger("orgforge.grounding.fetcher.convokit")

DEFAULT_LOOKUP_DIRS = [
    Path.home() / ".convokit" / "saved-corpora",
    Path(__file__).parent.parent / "_cache" / "convokit",
]


class ConvokitFetcher(BaseFetcher):
    """Reads pre-downloaded ConvoKit corpora from disk. No network."""

    name = "convokit_local"
    tier = 2
    license_tag = LICENSE_CC_BY_SA_4

    GENRE_TO_CORPUS: dict[str, str] = {
        "investor_diligence_qa": "wiki-articles-for-deletion-corpus",
        "exec_launch_decision": "wiki-articles-for-deletion-corpus",
        "new_hire_onboarding": "reddit-corpus-small",
        "marketing_launch_copy_review": "reddit-corpus-small",
        "beta_rider_safety_report": "reddit-corpus-small",
    }

    def _resolve_corpus_path(self, corpus_name: str) -> Optional[Path]:
        for root in DEFAULT_LOOKUP_DIRS:
            cand = root / corpus_name
            if cand.is_dir():
                return cand
        return None

    def _query(self, genre: str, n: int, query_hint: str) -> list[Seed]:
        corpus_name = self.GENRE_TO_CORPUS.get(genre)
        if not corpus_name:
            return []
        corpus_path = self._resolve_corpus_path(corpus_name)
        if not corpus_path:
            logger.warning(
                "[convokit_local] corpus %s not present; skipping. "
                "Download via: python -c \"from convokit import download; "
                "download('%s')\"",
                corpus_name, corpus_name,
            )
            return []
        # ConvoKit corpora are zipped JSONLs:
        #   conversations.json, utterances.jsonl, speakers.json
        return self._read_conversations(corpus_path, genre, n, query_hint)

    def _read_conversations(
        self, corpus_path: Path, genre: str, n: int, query_hint: str
    ) -> list[Seed]:
        utt_file = corpus_path / "utterances.jsonl"
        conv_file = corpus_path / "conversations.json"
        if not utt_file.exists() or not conv_file.exists():
            logger.warning(
                "[convokit_local] %s missing utterances/conversations files",
                corpus_path,
            )
            return []

        # Index utterances by conversation_id
        by_conv: dict[str, list[dict]] = {}
        with utt_file.open() as f:
            for line in f:
                if not line.strip():
                    continue
                u = json.loads(line)
                cid = u.get("conversation_id") or u.get("root")
                if cid:
                    by_conv.setdefault(cid, []).append(u)

        with conv_file.open() as f:
            convs = json.load(f)
        # convs is dict id -> meta
        items = list(convs.items())
        hint = (query_hint or "").lower()
        if hint:
            def score(it):
                _, meta = it
                blob = json.dumps(meta).lower()
                return blob.count(hint)
            items = sorted(items, key=score, reverse=True)
        seeds: list[Seed] = []
        for cid, meta in items[: n * 4]:
            if len(seeds) >= n:
                break
            utts = by_conv.get(cid) or []
            if len(utts) < 2:
                continue
            utts.sort(key=lambda u: u.get("timestamp") or 0)
            body = "\n\n".join(f"{u.get('speaker','')}: {u.get('text','')}" for u in utts[:30])
            actors = list({u.get("speaker", "") for u in utts if u.get("speaker")})
            timestamps = [str(u.get("timestamp", "")) for u in utts]
            seeds.append(self._make_seed(
                seed_id=f"convokit/{cid}",
                url=str(corpus_path / cid),
                title=meta.get("meta", {}).get("title", "") or cid,
                body=body,
                actors=actors,
                timestamps=timestamps,
                topic_tags=[corpus_path.name],
                target_genre=genre,
                raw={"meta": meta, "utterance_count": len(utts)},
            ))
        logger.info(
            "[convokit_local] fetched %d seeds for genre=%s corpus=%s",
            len(seeds), genre, corpus_path.name,
        )
        return seeds
