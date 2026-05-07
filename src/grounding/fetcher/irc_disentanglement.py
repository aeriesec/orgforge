"""
IRC-Disentanglement (Kummerfeld et al., ACL 2019 + ALTA 2023) — annotated
multi-actor IRC threads with parent-link graphs.

This is the canonical source for chat-shaped multi-actor texture: real
workplace-IM-style technical Q&A, disagreement, blocker disclosure, and
multi-actor decision deliberation. Far closer to a Slack/Talk channel than
NHTSA recall language.

Layout (after `git clone` into _cache/irc_disentanglement/):

    data/train/<date>.train-c.ascii.txt        # raw IRC messages, one per line
    data/train/<date>.train-c.annotation.txt   # per-line parent-link graph
    data/gold.train.clusters.txt               # pre-computed clusters
    data/list.ubuntu.train.txt                 # train file list

We use the pre-computed gold clusters (one cluster = one thread) so we don't
have to re-derive the connected components from the annotation graph.

Each cluster of size >= MIN_CLUSTER_SIZE becomes one Seed.

Genre routing — we map every chat-shaped genre to this source:
    async_question, blocker_thread, one_on_one (size=2 filter),
    design_discussion, collision_event, knowledge_gap_escalation
"""
from __future__ import annotations

import logging
import random
import re
from pathlib import Path
from typing import Iterable, Optional

from .base import BaseFetcher
from .seed import Seed, LICENSE_RESEARCH_USE

logger = logging.getLogger("orgforge.grounding.fetcher.irc_disentanglement")

# Where the corpus lives once `git clone`d
CORPUS_ROOT = Path(__file__).parent.parent / "_cache" / "irc_disentanglement"
CLUSTER_FILE = CORPUS_ROOT / "data" / "gold.train.clusters.txt"
TRAIN_DIR = CORPUS_ROOT / "data" / "train"

# `[HH:MM] <username> message text` or `[HH:MM] * username action text`
MESSAGE_RE = re.compile(r"^\[(\d{2}:\d{2})\]\s+[<*]([^>\s]+)>?\s*(.*)$")

# Per-genre cluster-size targets. one_on_one wants 2-actor short threads;
# design_discussion / collision wants 3+; blocker is short focused exchange.
GENRE_CLUSTER_FILTER: dict[str, dict] = {
    "async_question":              {"min_msgs": 4, "max_msgs": 25, "min_actors": 2, "max_actors": 6},
    "blocker_thread":              {"min_msgs": 3, "max_msgs": 15, "min_actors": 2, "max_actors": 4},
    "one_on_one":                  {"min_msgs": 4, "max_msgs": 30, "min_actors": 2, "max_actors": 2},
    "design_discussion":           {"min_msgs": 6, "max_msgs": 40, "min_actors": 3, "max_actors": 8},
    "design_discussion_zoom":      {"min_msgs": 6, "max_msgs": 40, "min_actors": 3, "max_actors": 8},
    "collision_event":             {"min_msgs": 5, "max_msgs": 30, "min_actors": 3, "max_actors": 6},
    "hardware_firmware_incident_thread": {"min_msgs": 5, "max_msgs": 35, "min_actors": 2, "max_actors": 6},
    "knowledge_gap_escalation":    {"min_msgs": 4, "max_msgs": 25, "min_actors": 2, "max_actors": 5},
}


class IRCDisentanglementFetcher(BaseFetcher):
    name = "irc_disentanglement"
    tier = 2
    license_tag = LICENSE_RESEARCH_USE

    def __init__(self):
        super().__init__()
        self._messages_by_date: dict[str, dict[int, tuple[str, str, str]]] = {}
        self._clusters: list[tuple[str, list[int]]] = []
        self._loaded = False

    def _load_corpus(self) -> None:
        if self._loaded:
            return
        if not CLUSTER_FILE.exists() or not TRAIN_DIR.is_dir():
            logger.warning(
                "[%s] corpus missing at %s — clone with: "
                "git clone github.com/jkkummerfeld/irc-disentanglement.git %s",
                self.name, CORPUS_ROOT, CORPUS_ROOT,
            )
            self._loaded = True
            return

        # Parse cluster file. Lines like:
        #   2004-12-25:1000 1009
        #   2004-12-25:1001
        # The line indices are absolute within the dataset (0-indexed across
        # the whole corpus), so we resolve them against per-date raw files.
        with CLUSTER_FILE.open() as f:
            for line in f:
                line = line.strip()
                if not line or ":" not in line:
                    continue
                date, idxs = line.split(":", 1)
                indices = [int(x) for x in idxs.split() if x.isdigit()]
                if indices:
                    self._clusters.append((date, indices))

        # Parse all per-date ascii.txt files. The line indices in clusters
        # correspond to LINE NUMBERS within each date's ascii.txt, NOT global.
        # Verify: peek at min/max indices in clusters and the line count of
        # one ascii.txt — if min~=900 and ascii has 900+ lines, indices are
        # absolute file-line offsets, not 0-indexed.
        for ascii_file in TRAIN_DIR.glob("*.ascii.txt"):
            stem = ascii_file.name.split(".")[0]  # 2004-12-25
            messages: dict[int, tuple[str, str, str]] = {}
            with ascii_file.open(encoding="utf-8", errors="replace") as f:
                for i, raw_line in enumerate(f):
                    raw_line = raw_line.rstrip("\n")
                    m = MESSAGE_RE.match(raw_line)
                    if m:
                        ts, user, text = m.group(1), m.group(2), m.group(3)
                        messages[i] = (ts, user, text)
            self._messages_by_date[stem] = messages
        logger.info(
            "[%s] loaded %d clusters across %d dates",
            self.name, len(self._clusters), len(self._messages_by_date),
        )
        self._loaded = True

    def _query(self, genre: str, n: int, query_hint: str) -> list[Seed]:
        self._load_corpus()
        if not self._clusters:
            return []

        flt = GENRE_CLUSTER_FILTER.get(genre, {
            "min_msgs": 4, "max_msgs": 30, "min_actors": 2, "max_actors": 8,
        })
        # Walk clusters with shuffle for variety per run, but stable order
        # within the same genre call.
        rng = random.Random(hash(genre) & 0xFFFFFFFF)
        ordered = list(range(len(self._clusters)))
        rng.shuffle(ordered)

        hint_lower = (query_hint or "").lower()
        seeds: list[Seed] = []
        for ci in ordered:
            if len(seeds) >= n:
                break
            date, indices = self._clusters[ci]
            day_msgs = self._messages_by_date.get(date)
            if not day_msgs:
                continue
            # Resolve indices: corpus uses 0-indexed line offsets within each
            # date's ascii.txt for that day. Some indices might not parse as
            # `<user> message` (e.g. system messages); skip those.
            resolved: list[tuple[str, str, str]] = []
            for idx in indices:
                if idx in day_msgs:
                    resolved.append(day_msgs[idx])
            if not resolved:
                continue
            actors = sorted({u for _, u, _ in resolved})
            if not (
                flt["min_msgs"] <= len(resolved) <= flt["max_msgs"]
                and flt["min_actors"] <= len(actors) <= flt["max_actors"]
            ):
                continue
            # Topical filter — keep clusters whose body contains hint terms.
            body_text = " ".join(t for _, _, t in resolved).lower()
            if hint_lower and hint_lower not in body_text:
                # Loose miss — only filter half the candidates so we still
                # return seeds when the hint is rare.
                if rng.random() < 0.6:
                    continue

            body_lines = [f"[{ts}] <{u}> {t}" for ts, u, t in resolved]
            timestamps = [ts for ts, _, _ in resolved]
            title = (
                resolved[0][2][:80]
                if resolved and resolved[0][2]
                else f"IRC thread {date}#{indices[0]}"
            )
            seeds.append(self._make_seed(
                seed_id=f"irc/{date}/{indices[0]}",
                url=(
                    "https://github.com/jkkummerfeld/irc-disentanglement"
                    f"/blob/master/data/train/{date}.train-c.ascii.txt"
                ),
                title=title,
                body="\n".join(body_lines),
                actors=actors,
                timestamps=timestamps,
                topic_tags=["irc", "ubuntu_channel"],
                target_genre=genre,
                raw={
                    "date": date,
                    "cluster_size": len(resolved),
                    "indices": indices,
                },
            ))
        logger.info(
            "[%s] genre=%s returned %d/%d seeds",
            self.name, genre, len(seeds), n,
        )
        return seeds
