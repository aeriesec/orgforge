"""
causal_chain_handler.py
=======================
Causal chain tracking and recurrence detection for OrgForge.

Architecture:
  flow.py / _handle_incident()
      → RecurrenceDetector.find_prior_incident()   (called once at open)
      → CausalChainHandler.start()                 (initialises chain)

  flow.py / normal_day.py -- as artifacts are created:
      → CausalChainHandler.append()                (grows the chain)
      → CausalChainHandler.snapshot()              (returns chain at this moment)

Usage:
    from causal_chain_handler import CausalChainHandler, RecurrenceDetector

    detector = RecurrenceDetector(mem)
    prior    = detector.find_prior_incident(root_cause, current_day, ticket_id)

    chain_handler = CausalChainHandler(ticket_id)
    chain_handler.append(slack_thread_id)
    chain_handler.append(conf_id)

    facts["causal_chain"] = chain_handler.snapshot()
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from memory import Memory, SimEvent

logger = logging.getLogger("orgforge.causal_chain")

_MIN_TEXT_SCORE = 0.40
_RETRIEVAL_LIMIT = 10

ARTIFACT_KEY_JIRA = "jira"
ARTIFACT_KEY_CONFLUENCE = "confluence"
ARTIFACT_KEY_SLACK = "slack"
ARTIFACT_KEY_SLACK_THREAD = "slack_thread"


class CausalChainHandler:
    """
    Append-only causal chain for a single incident or feature thread.

    The chain starts with the root artifact (usually a JIRA ticket ID) and
    grows as the incident progresses. Snapshots are taken at each SimEvent
    so the historical record shows the chain as it existed at that exact
    moment, not retroactively.

    Usage:
        handler = CausalChainHandler(root_id="ENG-042")
        handler.append("slack_incidents_2024-01-15T10:30")
        handler.append("PR-117")
        handler.append("CONF-ENG-012")

        facts["causal_chain"] = handler.snapshot()
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


class RecurrenceMatchStore:
    """
    Writes one document per recurrence detection attempt to the
    recurrence_matches collection.

    Schema (text-only pipeline):
        query_root_cause      str   -- the root cause being matched
        current_ticket_id     str   -- the incident being opened
        current_day           int   -- sim day
        matched               bool  -- whether a prior incident was found
        matched_ticket_id     str?  -- jira ID of the match (None if rejected)
        matched_root_cause    str?  -- root cause text of the match
        matched_day           int?  -- day the prior incident occurred
        recurrence_gap_days   int?  -- current_day - matched_day
        text_score            float -- normalised MongoDB $text score (0-1)
        confidence            str   -- "high" | "low" | "rejected"
        timestamp             str   -- UTC ISO when the decision was made
        sim_day               int   -- duplicate of current_day for easy grouping
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
        self._coll.create_index([("text_score", -1)])

    def log(
        self,
        *,
        query_root_cause: str,
        current_ticket_id: str,
        current_day: int,
        matched_event: Optional[SimEvent],
        text_score: float,
        confidence: str,
    ) -> None:
        doc: Dict[str, Any] = {
            "query_root_cause": query_root_cause,
            "current_ticket_id": current_ticket_id,
            "current_day": current_day,
            "matched": matched_event is not None,
            "matched_ticket_id": (
                matched_event.artifact_ids.get("jira") if matched_event else None
            ),
            "matched_root_cause": (
                matched_event.facts.get("root_cause") if matched_event else None
            ),
            "matched_day": matched_event.day if matched_event else None,
            "recurrence_gap_days": (
                current_day - matched_event.day if matched_event else None
            ),
            "text_score": round(text_score, 4),
            "confidence": confidence,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sim_day": current_day,
        }
        try:
            self._coll.insert_one(doc)
        except Exception as e:
            logger.warning(f"[causal_chain] recurrence_match insert failed: {e}")


class RecurrenceDetector:
    """
    Finds the most relevant prior incident for a given root cause string
    using MongoDB $text search over the events collection.

    The text index on events.facts.root_cause and events.summary is created
    by memory._init_text_indexes() at startup. _ensure_text_index() provides
    a belt-and-suspenders fallback for environments where Atlas Search index
    creation is delayed.

    Every detection attempt -- matched or rejected -- is persisted to the
    recurrence_matches collection via RecurrenceMatchStore.

    Confidence tiers:
        "high"     -- text_score >= _MIN_TEXT_SCORE * 1.5
        "low"      -- text_score >= _MIN_TEXT_SCORE but below the high tier
        "rejected" -- no result exceeded _MIN_TEXT_SCORE
    """

    _HIGH_CONFIDENCE_MULTIPLIER = 1.5

    def __init__(
        self,
        mem: Memory,
        min_text: float = _MIN_TEXT_SCORE,
    ):
        self._mem = mem
        self._min_text = min_text
        self._store = RecurrenceMatchStore(mem)
        self._ensure_text_index()

    def find_prior_incident(
        self,
        root_cause: str,
        current_day: int,
        current_ticket_id: str,
    ) -> Optional[SimEvent]:
        """
        Main entry point. Returns the best-matching prior SimEvent, or None
        if no confident match is found.

        Accepts ALL candidates above _MIN_TEXT_SCORE and returns the EARLIEST
        one (lowest day) to prevent daisy-chaining where a recurrence is
        matched to its immediate predecessor rather than the true origin.

        Always writes one document to recurrence_matches regardless of outcome.
        """
        text_results = self._text_search(root_cause, current_day)

        if not text_results:
            self._store.log(
                query_root_cause=root_cause,
                current_ticket_id=current_ticket_id,
                current_day=current_day,
                matched_event=None,
                text_score=0.0,
                confidence="rejected",
            )
            return None

        accepted = [
            r for r in text_results if r.get("text_score", 0.0) >= self._min_text
        ]

        if not accepted:
            best_score = max(r.get("text_score", 0.0) for r in text_results)
            self._store.log(
                query_root_cause=root_cause,
                current_ticket_id=current_ticket_id,
                current_day=current_day,
                matched_event=None,
                text_score=best_score,
                confidence="rejected",
            )
            logger.debug(
                f"[causal_chain] No prior incident -- best text={best_score:.3f} "
                f"(threshold={self._min_text})"
            )
            return None

        best_result = min(accepted, key=lambda r: SimEvent.from_dict(r).day)
        best_event = SimEvent.from_dict(best_result)
        best_score = best_result.get("text_score", 0.0)

        high_threshold = self._min_text * self._HIGH_CONFIDENCE_MULTIPLIER
        confidence = "high" if best_score >= high_threshold else "low"

        self._store.log(
            query_root_cause=root_cause,
            current_ticket_id=current_ticket_id,
            current_day=current_day,
            matched_event=best_event,
            text_score=best_score,
            confidence=confidence,
        )

        logger.info(
            f"[causal_chain] Recurrence matched ({confidence}): "
            f"{best_event.artifact_ids.get('jira', '?')} "
            f"(text={best_score:.3f}, "
            f"gap={current_day - best_event.day}d)"
        )

        return best_event

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
                for parent in event.facts.get("causal_chain", []):
                    if parent not in visited:
                        queue.append(parent)
                prior = event.facts.get("recurrence_of")
                if prior and prior not in visited:
                    queue.append(prior)

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
        parent -- every time this class of problem has recurred.
        """
        return [
            e
            for e in self._mem.get_event_log()
            if e.facts.get("recurrence_of") == ticket_id
            or ticket_id in e.facts.get("causal_chain", [])
        ]

    def _text_search(self, root_cause: str, current_day: int) -> List[Dict[str, Any]]:
        """
        MongoDB $text search over the events collection.

        Scores are normalised against the maximum raw score in the result set
        using log1p scaling so the distribution is comparable across queries
        of different lengths. Results are filtered to incident event types only
        and scoped to days prior to the current incident.
        """
        import math

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

        raw_scores = [r.get("score", 0.0) for r in results]
        max_score = max(raw_scores) if raw_scores else 1.0

        normalised = []
        for r, raw in zip(results, raw_scores):
            norm_score = (
                math.log1p(raw) / math.log1p(max_score) if max_score > 0 else 0.0
            )
            r["text_score"] = round(norm_score, 4)
            normalised.append(r)

        return normalised

    def _ensure_text_index(self) -> None:
        """
        Belt-and-suspenders fallback. memory._init_text_indexes() creates the
        $text index at startup. This guard catches cases where the Atlas Search
        index creation is delayed or the local MongoDB instance is running
        without Atlas Search support.
        """
        try:
            existing = self._mem._events.index_information()
            if not any("text" in str(v.get("key")) for v in existing.values()):
                self._mem._events.create_index(
                    [("facts.root_cause", "text"), ("summary", "text")],
                    name="event_text_search_legacy",
                )
                logger.info("[causal_chain] Created $text index on events collection.")
        except Exception as e:
            logger.warning(
                f"[causal_chain] Could not create $text index on events: {e}"
            )
