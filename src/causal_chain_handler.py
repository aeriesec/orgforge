"""
causal_chain_handler.py
=======================
Causal chain tracking, recurrence detection, and fine-tuning data collection
for OrgForge.

Responsibilities:
  - CausalChainHandler  : builds and appends causal chains on live incidents
  - RecurrenceDetector  : hybrid MongoDB text + vector search to find prior
                          incidents matching the current root cause
  - RecurrenceMatchStore: persists every match decision to MongoDB so the
                          recurrence_matches collection can be used for
                          threshold calibration and fine-tuning without
                          reading the simulation log

Architecture:
  flow.py / _handle_incident()
      → RecurrenceDetector.find_prior_incident()   (called once at open)
      → CausalChainHandler.start()                 (initialises chain)

  flow.py / normal_day.py — as artifacts are created:
      → CausalChainHandler.append()                (grows the chain)
      → CausalChainHandler.snapshot()              (returns chain at this moment)

  Memory (injected):
      → RecurrenceMatchStore.log()                 (called inside detector)
      → Memory.search_events()                     (new vector search on events)

Usage:
    from causal_chain_handler import CausalChainHandler, RecurrenceDetector

    # At incident open
    detector = RecurrenceDetector(mem)
    prior    = detector.find_prior_incident(root_cause, current_day, ticket_id)

    chain_handler = CausalChainHandler(ticket_id)
    chain_handler.append(slack_thread_id)
    chain_handler.append(conf_id)

    # Snapshot at any point — safe to pass into facts={}
    facts["causal_chain"] = chain_handler.snapshot()
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from memory import Memory, SimEvent

logger = logging.getLogger("orgforge.causal_chain")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Reciprocal Rank Fusion damping constant — 60 is the standard value from
# the original RRF paper. Higher = less aggressive rank boosting.
_RRF_K = 60

# Default fusion weights — vector wins for semantic similarity since our
# Ollama embedding model is local and essentially free to call
_TEXT_WEIGHT = 0.35
_VECTOR_WEIGHT = 0.65

# Minimum scores to accept a match — tune these after reviewing
# the recurrence_matches collection after a sim run
_MIN_VECTOR_SCORE = 0.72
_MIN_TEXT_SCORE = 0.40

# How many candidates to retrieve from each source before fusion
_RETRIEVAL_LIMIT = 10

ARTIFACT_KEY_JIRA = "jira"
ARTIFACT_KEY_CONFLUENCE = "confluence"
ARTIFACT_KEY_SLACK = "slack"
ARTIFACT_KEY_SLACK_THREAD = "slack_thread"

# ─────────────────────────────────────────────────────────────────────────────
# CAUSAL CHAIN HANDLER
# Tracks the growing set of artifact IDs that causally produced an incident.
# Lives on an ActiveIncident — one instance per open incident.
# ─────────────────────────────────────────────────────────────────────────────


class CausalChainHandler:
    """
    Append-only causal chain for a single incident or feature thread.

    The chain starts with the root artifact (usually a Jira ticket ID) and
    grows as the incident progresses — Slack threads, PRs, postmortems are
    appended in order. Snapshots are taken at each SimEvent so the historical
    record shows the chain as it existed at that exact moment, not retroactively.

    Usage:
        handler = CausalChainHandler(root_id="ORG-042")
        handler.append("slack_incidents_2024-01-15T10:30")
        handler.append("PR-117")
        handler.append("CONF-ENG-012")

        # In SimEvent facts:
        facts["causal_chain"] = handler.snapshot()

        # In downstream artifact SimEvents:
        facts["causal_chain"] = handler.snapshot()   # grows with each append
    """

    def __init__(self, root_id: str):
        self._chain: List[str] = [root_id]

    def append(self, artifact_id: str) -> None:
        """Add an artifact to the chain. Silently ignores duplicates."""
        if artifact_id and artifact_id not in self._chain:
            self._chain.append(artifact_id)

    def snapshot(self) -> List[str]:
        """Return an immutable copy of the chain at this moment."""
        return list(self._chain)

    @property
    def root(self) -> str:
        return self._chain[0]

    def __len__(self) -> int:
        return len(self._chain)

    def __repr__(self) -> str:
        return f"CausalChainHandler(root={self.root}, length={len(self)})"


# ─────────────────────────────────────────────────────────────────────────────
# RECURRENCE MATCH STORE
# Persists every detection decision to MongoDB regardless of outcome.
# Negative examples (rejected matches) are as valuable as positives for
# threshold calibration.
# ─────────────────────────────────────────────────────────────────────────────


class RecurrenceMatchStore:
    """
    Writes one document per recurrence detection attempt to the
    recurrence_matches collection.

    Schema:
        query_root_cause      str   — the root cause being matched
        current_ticket_id     str   — the incident being opened
        current_day           int   — sim day
        matched               bool  — whether a prior incident was found
        matched_ticket_id     str?  — jira ID of the match (None if rejected)
        matched_root_cause    str?  — root cause text of the match
        matched_day           int?  — day the prior incident occurred
        recurrence_gap_days   int?  — current_day - matched_day
        text_score            float — normalised MongoDB textScore (0-1)
        vector_score          float — cosine similarity from vector search (0-1)
        fused_score           float — weighted fusion score
        rrf_score             float — reciprocal rank fusion score
        fusion_strategy       str   — "rrf" | "weighted" | "text_only" | "vector_only"
        confidence            str   — "high" | "low" | "rejected"
        candidates_evaluated  int   — total candidates before fusion
        threshold_gate        dict  — thresholds used at decision time
        timestamp             str   — UTC ISO when the decision was made
        sim_day               int   — duplicate of current_day for easy grouping
    """

    COLLECTION = "recurrence_matches"

    def __init__(self, mem: Memory):
        self._coll = mem._db[self.COLLECTION]
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        self._coll.create_index([("current_ticket_id", 1)])
        self._coll.create_index([("matched_ticket_id", 1)])
        self._coll.create_index([("matched", 1), ("sim_day", 1)])
        self._coll.create_index([("confidence", 1)])
        self._coll.create_index([("vector_score", -1)])
        self._coll.create_index([("text_score", -1)])

    def log(
        self,
        *,
        query_root_cause: str,
        current_ticket_id: str,
        current_day: int,
        matched_event: Optional[SimEvent],
        text_score: float,
        vector_score: float,
        fused_score: float,
        rrf_score: float,
        fusion_strategy: str,
        confidence: str,
        candidates_evaluated: int,
        threshold_gate: Dict[str, float],
    ) -> None:
        doc: Dict[str, Any] = {
            # Query
            "query_root_cause": query_root_cause,
            "current_ticket_id": current_ticket_id,
            "current_day": current_day,
            # Match result
            "matched": matched_event is not None,
            "matched_ticket_id": matched_event.artifact_ids.get("jira")
            if matched_event
            else None,
            "matched_root_cause": matched_event.facts.get("root_cause")
            if matched_event
            else None,
            "matched_day": matched_event.day if matched_event else None,
            "recurrence_gap_days": (
                current_day - matched_event.day if matched_event else None
            ),
            # Scores
            "text_score": round(text_score, 4),
            "vector_score": round(vector_score, 4),
            "fused_score": round(fused_score, 4),
            "rrf_score": round(rrf_score, 4),
            "fusion_strategy": fusion_strategy,
            "confidence": confidence,
            "candidates_evaluated": candidates_evaluated,
            "threshold_gate": threshold_gate,
            # Metadata
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sim_day": current_day,
        }
        try:
            self._coll.insert_one(doc)
        except Exception as e:
            logger.warning(f"[causal_chain] recurrence_match insert failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# RECURRENCE DETECTOR
# Hybrid retrieval: fuses MongoDB text search + vector search to identify
# whether a new incident root cause has occurred before.
# ─────────────────────────────────────────────────────────────────────────────


class RecurrenceDetector:
    """
    Finds the most relevant prior incident for a given root cause string
    using a two-stage hybrid retrieval pipeline:

      Stage 1 — MongoDB $text search:
          Fast, lexical, handles stemming and stop-word removal natively.
          Best for exact-ish recurrences ("connection pool exhausted" → same words).

      Stage 2 — Vector search (via Memory.search_events):
          Semantic similarity via Ollama embeddings.
          Best for paraphrased recurrences ("DB timeout" ↔ "connection pool saturated").

      Fusion — Reciprocal Rank Fusion (when both return results):
          Rank-based combination that is robust to score distribution differences
          between the two retrieval systems. Falls back to weighted score fusion
          when only one system returns results.

    Every detection attempt — matched or rejected — is persisted to the
    recurrence_matches collection via RecurrenceMatchStore for post-run analysis.
    """

    def __init__(
        self,
        mem: Memory,
        text_weight: float = _TEXT_WEIGHT,
        vector_weight: float = _VECTOR_WEIGHT,
        min_vector: float = _MIN_VECTOR_SCORE,
        min_text: float = _MIN_TEXT_SCORE,
    ):
        self._mem = mem
        self._text_w = text_weight
        self._vector_w = vector_weight
        self._min_vector = min_vector
        self._min_text = min_text
        self._store = RecurrenceMatchStore(mem)

        # Ensure text index exists on the events collection
        self._ensure_text_index()

    # ── Public ────────────────────────────────────────────────────────────────

    def find_prior_incident(
        self,
        root_cause: str,
        current_day: int,
        current_ticket_id: str,
    ) -> Optional[SimEvent]:
        """
        Main entry point. Returns the best-matching prior SimEvent, or None
        if no confident match is found.

        Change from original: collects ALL candidates above threshold and
        returns the EARLIEST one (lowest day) to prevent daisy-chaining.

        Always writes one document to recurrence_matches regardless of outcome.
        """

        candidates: Dict[str, Dict[str, Any]] = {}

        # ── Stage 1: MongoDB text search ──────────────────────────────────────
        text_results = self._text_search(root_cause, current_day)

        # Fixed normalisation: use a corpus-calibrated ceiling rather than
        # result-set max, so rank-1 doesn't always get 1.0
        _TEXT_CEILING = 8.0
        for rank, result in enumerate(text_results):
            event = SimEvent.from_dict(result)
            key = event.artifact_ids.get("jira", event.timestamp)
            raw = result.get("score", 0)
            normalised = round(min(raw / _TEXT_CEILING, 1.0), 4)
            candidates.setdefault(key, self._empty_candidate(event))
            candidates[key]["text_score"] = normalised
            candidates[key]["text_rrf"] = 1 / (rank + 1 + _RRF_K)

        # ── Stage 2: Vector search ─────────────────────────────────────────────
        vector_results = self._vector_search(root_cause, current_day)

        for rank, (event, vscore) in enumerate(vector_results):
            key = event.artifact_ids.get("jira", event.timestamp)
            candidates.setdefault(key, self._empty_candidate(event))
            candidates[key]["vector_score"] = vscore
            candidates[key]["vector_rrf"] = 1 / (rank + 1 + _RRF_K)

        if not candidates:
            self._store.log(
                query_root_cause=root_cause,
                current_ticket_id=current_ticket_id,
                current_day=current_day,
                matched_event=None,
                text_score=0.0,
                vector_score=0.0,
                fused_score=0.0,
                rrf_score=0.0,
                fusion_strategy="none",
                confidence="rejected",
                candidates_evaluated=0,
                threshold_gate=self._threshold_gate(),
            )
            return None

        # ── Fusion ────────────────────────────────────────────────────────────
        both_returned = bool(text_results) and bool(vector_results)
        fusion_strategy = (
            "rrf" if both_returned else ("text_only" if text_results else "vector_only")
        )

        for c in candidates.values():
            c["fused_score"] = (
                self._text_w * c["text_score"] + self._vector_w * c["vector_score"]
            )
            c["rrf_score"] = c.get("text_rrf", 0.0) + c.get("vector_rrf", 0.0)

        sort_key = "rrf_score" if both_returned else "fused_score"
        ranked = sorted(candidates.values(), key=lambda c: c[sort_key], reverse=True)

        # ── Threshold gate — collect ALL passing candidates ───────────────────
        accepted = [
            c
            for c in ranked
            if c["vector_score"] >= self._min_vector
            or c["text_score"] >= self._min_text
        ]

        if not accepted:
            best = ranked[0]
            self._store.log(
                query_root_cause=root_cause,
                current_ticket_id=current_ticket_id,
                current_day=current_day,
                matched_event=None,
                text_score=best["text_score"],
                vector_score=best["vector_score"],
                fused_score=best["fused_score"],
                rrf_score=best["rrf_score"],
                fusion_strategy=fusion_strategy,
                confidence="rejected",
                candidates_evaluated=len(candidates),
                threshold_gate=self._threshold_gate(),
            )
            logger.debug(
                f"[causal_chain] No prior incident — "
                f"best vector={best['vector_score']:.3f}, "
                f"text={best['text_score']:.3f}"
            )
            return None

        # Anti-daisy-chain: among all passing candidates, pick the EARLIEST
        # so ORG-164 links to ORG-140, not ORG-161
        best = min(accepted, key=lambda c: c["event"].day)

        confidence = (
            "high"
            if (
                best["vector_score"] >= self._min_vector
                and best["text_score"] >= self._min_text
            )
            else "low"
        )

        self._store.log(
            query_root_cause=root_cause,
            current_ticket_id=current_ticket_id,
            current_day=current_day,
            matched_event=best["event"],
            text_score=best["text_score"],
            vector_score=best["vector_score"],
            fused_score=best["fused_score"],
            rrf_score=best["rrf_score"],
            fusion_strategy=fusion_strategy,
            confidence=confidence,
            candidates_evaluated=len(candidates),
            threshold_gate=self._threshold_gate(),
        )

        logger.info(
            f"[causal_chain] Recurrence matched ({confidence}): "
            f"{best['event'].artifact_ids.get('jira', '?')} "
            f"(vector={best['vector_score']:.3f}, "
            f"text={best['text_score']:.3f}, "
            f"gap={current_day - best['event'].day}d)"
        )

        return best["event"]

    def find_postmortem_for_ticket(self, ticket_id: str) -> Optional[str]:
        """
        Returns the Confluence ID of a postmortem written for a given ticket,
        or None if no postmortem exists.
        """
        event = next(
            (
                e
                for e in self._mem.get_event_log()
                if e.type == "postmortem_created"
                and e.artifact_ids.get("jira") == ticket_id
            ),
            None,
        )
        return event.artifact_ids.get("confluence") if event else None

    def get_causal_chain(self, artifact_id: str) -> List[SimEvent]:
        """
        Walk backwards through causal parents and recurrence links to
        reconstruct the full SimEvent history for any artifact.

        Returns events sorted chronologically (earliest first).
        """
        chain: List[SimEvent] = []
        visited: set = set()
        queue: List[str] = [artifact_id]

        while queue:
            aid = queue.pop(0)
            if aid in visited:
                continue
            visited.add(aid)

            events = [
                e
                for e in self._mem.get_event_log()
                if aid in e.artifact_ids.values()
                or aid in e.facts.get("causal_chain", [])
                or aid == e.facts.get("recurrence_of")
            ]
            chain.extend(events)

            for event in events:
                # Queue causal parents
                for parent in event.facts.get("causal_chain", []):
                    if parent not in visited:
                        queue.append(parent)
                # Queue prior incident in recurrence chain
                prior = event.facts.get("recurrence_of")
                if prior and prior not in visited:
                    queue.append(prior)

        # Deduplicate (same event may appear via multiple paths)
        seen_ids: set = set()
        unique: List[SimEvent] = []
        for e in chain:
            eid = (e.day, e.type, e.timestamp)
            if eid not in seen_ids:
                seen_ids.add(eid)
                unique.append(e)

        return sorted(unique, key=lambda e: e.day)

    def get_recurrence_history(self, ticket_id: str) -> List[SimEvent]:
        """
        Returns all SimEvents that reference ticket_id as a recurrence_of
        parent — i.e. every time this class of problem has recurred.
        """
        return [
            e
            for e in self._mem.get_event_log()
            if e.facts.get("recurrence_of") == ticket_id
            or ticket_id in e.facts.get("causal_chain", [])
        ]

    # ── Private ───────────────────────────────────────────────────────────────

    def _text_search(self, root_cause: str, current_day: int) -> List[Dict[str, Any]]:
        """MongoDB $text search — returns results with normalised scores."""
        try:
            results = list(
                self._mem._events.find(
                    {
                        "$text": {"$search": root_cause},
                        "type": {"$in": ["incident_opened", "incident_resolved"]},
                        "day": {"$lt": current_day},
                    },
                    {"score": {"$meta": "textScore"}, "_id": 0},
                )
                .sort([("score", {"$meta": "textScore"})])
                .limit(_RETRIEVAL_LIMIT)
            )
        except Exception as e:
            logger.warning(f"[causal_chain] Text search failed: {e}")
            return []

        if not results:
            return []

        # Normalise against the MAX score in this result set, not rank-1.
        # This means rank-1 still gets 1.0, but rank-2 gets a real relative score
        # rather than everyone below rank-1 getting arbitrary lower values.
        # More importantly, when there's only ONE candidate it still gets 1.0 —
        # the threshold check (min_text=0.4) then does actual filtering work
        # because a weak single match will have a low raw score.
        raw_scores = [r.get("score", 0.0) for r in results]
        max_score = max(raw_scores) if raw_scores else 1.0

        # Use log-normalisation so weak matches don't cluster near 1.0
        import math

        normalised = []
        for r, raw in zip(results, raw_scores):
            # log1p normalisation preserves ordering, spreads out the range
            norm_score = (
                math.log1p(raw) / math.log1p(max_score) if max_score > 0 else 0.0
            )
            r["text_score"] = round(norm_score, 4)
            normalised.append(r)

        return normalised

    def _vector_search(
        self, root_cause: str, current_day: int
    ) -> List[Tuple[SimEvent, float]]:
        """
        Vector search over the events collection scoped to incident types.
        Uses Memory.search_events() which runs $vectorSearch on _events.
        """
        try:
            return self._mem.search_events(
                query=root_cause,
                event_types=["incident_opened", "incident_resolved"],
                n=_RETRIEVAL_LIMIT,
                as_of_day=current_day - 1,
            )
        except Exception as e:
            logger.warning(f"[causal_chain] Vector search failed: {e}")
            return []

    def _ensure_text_index(self) -> None:
        """Create the text index on events if it doesn't already exist."""
        try:
            existing = self._mem._events.index_information()
            if not any("text" in str(v.get("key")) for v in existing.values()):
                self._mem._events.create_index(
                    [("facts.root_cause", "text"), ("summary", "text")],
                    name="event_text_search",
                )
                logger.info("[causal_chain] Created text index on events collection")
        except Exception as e:
            logger.warning(f"[causal_chain] Could not create text index: {e}")

    @staticmethod
    def _empty_candidate(event: SimEvent) -> Dict[str, Any]:
        return {
            "event": event,
            "text_score": 0.0,
            "vector_score": 0.0,
            "text_rrf": 0.0,
            "vector_rrf": 0.0,
            "fused_score": 0.0,
            "rrf_score": 0.0,
        }

    def _threshold_gate(self) -> Dict[str, float]:
        return {
            "min_vector": self._min_vector,
            "min_text": self._min_text,
        }


# ─────────────────────────────────────────────────────────────────────────────
# MEMORY EXTENSION — search_events()
# Add this method to memory.py Memory class.
# Reproduced here as a standalone function so it can be monkey-patched in
# or copy-pasted into Memory directly.
# ─────────────────────────────────────────────────────────────────────────────


def search_events(
    mem: Memory,
    query: str,
    event_types: Optional[List[str]] = None,
    n: int = 10,
    as_of_day: Optional[int] = None,
) -> List[Tuple[SimEvent, float]]:
    """
    Vector search over the events collection.
    Returns (SimEvent, cosine_score) pairs sorted by descending relevance.

    Add this as a method on Memory:
        Memory.search_events = lambda self, **kw: search_events(self, **kw)
    Or copy the body directly into the Memory class.
    """
    query_vector = mem._embedder.embed(query)
    if not query_vector:
        return []

    pipeline_filter: Dict[str, Any] = {}
    if event_types:
        pipeline_filter["type"] = {"$in": event_types}
    if as_of_day is not None:
        pipeline_filter["day"] = {"$lte": as_of_day}

    pipeline: List[Dict] = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": query_vector,
                "numCandidates": n * 10,
                "limit": n,
            }
        },
        {"$addFields": {"vector_score": {"$meta": "vectorSearchScore"}}},
        {"$project": {"_id": 0, "embedding": 0}},
    ]

    if pipeline_filter:
        pipeline[0]["$vectorSearch"]["filter"] = pipeline_filter

    try:
        results = list(mem._events.aggregate(pipeline))
        return [
            (SimEvent.from_dict(r), round(r.get("vector_score", 0.0), 4))
            for r in results
        ]
    except Exception as e:
        logger.warning(f"[causal_chain] Event vector search failed: {e}")
        return []
