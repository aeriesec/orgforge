"""
scorer.py
=========
Per-question-type scoring for the OrgForge eval dataset.

Design principles
-----------------
  1. Scores are always in [0.0, 1.0] — comparable across question types.
  2. Partial credit is supported everywhere via evidence_chain overlap.
  3. Each scorer returns a ScorerResult so callers can aggregate, filter,
     or report by type independently.
  4. No LLM involvement — scoring is deterministic and reproducible.

Question types handled
----------------------
  RETRIEVAL     Exact artifact_id match + optional timestamp proximity bonus.
  CAUSAL        Artifact match AND event_type match required for full credit.
  TEMPORAL      Boolean + optional departure_day agreement.
  GAP_DETECTION Boolean was_actioned + downstream artifact overlap.
  ROUTING       first_recipient exact match.
  PLAN          dept + theme match (theme uses substring matching for LLM prose).
  ESCALATION    escalation_actors set match, partial credit for overlap.
  KNOWLEDGE_GAP gap_areas set match, partial credit for overlap.
  ZD_RESOLUTION resolved boolean + duration_days exact match; escalated bonus.
  SF_RISK       incident_id match + at_risk_accounts set overlap.
  NPS_SCORE     nps_score exact + classification match; escalated_tickets bonus.
  INVOICE_SLA   breach_duration_days exact + sla_credit_per_org within 5%.

Partial credit via evidence_chain
----------------------------------
All question types award partial credit if the agent retrieved relevant
artifacts from evidence_chain even when the final answer is wrong.
This separates retrieval quality from reasoning quality, which is useful
for diagnosing whether failures come from the retriever or the reader.

Usage
-----
    from scorer import OrgForgeScorer, ScorerResult

    scorer = OrgForgeScorer()
    result = scorer.score(question, agent_answer)

    # Batch
    results = scorer.score_all(questions, agent_answers)
    report  = scorer.report(results)

Agent answer format (per question type)
----------------------------------------
RETRIEVAL:
    {
        "artifact_id": "ORG-42",                  # required
        "artifact_type": "jira",                   # optional
        "timestamp": "2024-01-15T10:32:00",        # optional — used for proximity
        "retrieved_artifact_ids": ["ORG-42", ...]  # optional — evidence credit
    }

CAUSAL:
    {
        "artifact_id": "CONF-ENG-007",
        "event_type": "confluence_created",
        "actors": ["Alice", "Bob"],
        "retrieved_artifact_ids": [...]
    }

TEMPORAL:
    {
        "had_knowledge": true,
        "person": "Alice",
        "domain": "auth-service",
        "departure_day": null,                     # null if agent thinks no departure
        "reasoning": "..."                         # free text — not scored
    }

GAP_DETECTION:
    {
        "was_actioned": false,
        "artifact_id": "EMAIL-003",
        "downstream_artifacts": [],
        "retrieved_artifact_ids": [...]
    }

ROUTING:
    {
        "first_recipient": "Alice",
        "retrieved_artifact_ids": [...]
    }

PLAN:
    {
        "dept": "Engineering_Backend",
        "theme": "Stabilize sensor ingest and Kafka reliability",
        "retrieved_artifact_ids": [...]            # optional — evidence credit
    }

ESCALATION:
    {
        "escalation_actors": ["Jax", "Chloe"],    # order-insensitive
        "retrieved_artifact_ids": [...]
    }

KNOWLEDGE_GAP:
    {
        "gap_areas": ["auth-service", "redis-cache"],  # order-insensitive
        "retrieved_artifact_ids": [...]
    }
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("orgforge.scorer")

# ── Weights ───────────────────────────────────────────────────────────────────
# Final score = primary_score * PRIMARY_WEIGHT + evidence_score * EVIDENCE_WEIGHT
# Evidence credit is always secondary so a lucky retrieval can't mask a bad answer.
PRIMARY_WEIGHT = 0.80
EVIDENCE_WEIGHT = 0.20

# Temporal proximity bonus: awarded when predicted timestamp is within this
# many minutes of the ground truth. Adds up to PROXIMITY_BONUS to primary score
# before weighting (capped at 1.0).
PROXIMITY_BONUS = 0.10
PROXIMITY_WINDOW_MIN = 30  # minutes


# ─────────────────────────────────────────────────────────────────────────────
# RESULT DATA CLASS
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ScorerResult:
    question_id: str
    question_type: str
    difficulty: str
    score: float  # [0.0, 1.0]
    primary_score: float  # [0.0, 1.0] — main answer correctness
    evidence_score: float  # [0.0, 1.0] — retrieved right artifacts?
    correct: bool  # True if score >= 0.9
    partial: bool  # True if 0.2 <= score < 0.9
    failure_reason: Optional[str]  # populated when score < 0.9
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# INDIVIDUAL SCORERS
# ─────────────────────────────────────────────────────────────────────────────


class _BaseScorer:
    """Shared helpers for all question-type scorers."""

    def _evidence_overlap(
        self,
        ground_truth_chain: List[str],
        agent_retrieved: List[str],
    ) -> float:
        """Recall of ground-truth evidence chain in agent-retrieved IDs.

        Using recall (hits / |chain|) rather than Jaccard because retrievers
        always return top-K docs — penalising for retrieving non-chain documents
        structurally caps scores below 0.9 for short evidence chains, making
        accuracy always 0 regardless of answer quality.
        """
        if not ground_truth_chain:
            return 1.0
        gt_set = set(ground_truth_chain)
        agent_set = set(agent_retrieved or [])
        if not agent_set:
            return 0.0
        hits = len(gt_set & agent_set)
        return hits / len(gt_set)

    def _timestamp_proximity(
        self,
        gt_ts: Optional[str],
        agent_ts: Optional[str],
    ) -> float:
        """
        Returns PROXIMITY_BONUS if the predicted timestamp is within
        PROXIMITY_WINDOW_MIN of ground truth, else 0.0.
        """
        if not gt_ts or not agent_ts:
            return 0.0
        try:
            gt = datetime.fromisoformat(str(gt_ts))
            agent = datetime.fromisoformat(str(agent_ts))
            delta = abs((gt - agent).total_seconds()) / 60
            return PROXIMITY_BONUS if delta <= PROXIMITY_WINDOW_MIN else 0.0
        except (ValueError, TypeError):
            return 0.0

    def _combine(self, primary: float, evidence: float) -> float:
        return min(1.0, primary * PRIMARY_WEIGHT + evidence * EVIDENCE_WEIGHT)


class RetrievalScorer(_BaseScorer):
    """
    RETRIEVAL — "Which artifact first documented X?"

    Full credit  (1.0): artifact_id matches ground truth exactly.
    Partial credit: evidence_chain overlap when artifact_id is wrong.
    Timestamp proximity bonus if agent also provides a timestamp.
    """

    def score(
        self, question: dict, agent_answer: dict
    ) -> Tuple[float, float, Optional[str]]:
        gt = question["ground_truth"]
        gt_id = gt.get("artifact_id", "")
        agent_id = agent_answer.get("artifact_id", "")

        primary = 1.0 if agent_id == gt_id else 0.0
        primary += self._timestamp_proximity(
            gt.get("timestamp"), agent_answer.get("timestamp")
        )
        primary = min(1.0, primary)

        evidence = self._evidence_overlap(
            question.get("evidence_chain", []),
            agent_answer.get("retrieved_artifact_ids", []),
        )

        failure = (
            None
            if primary >= 1.0
            else (f"Expected artifact_id={gt_id!r}, got {agent_id!r}")
        )
        return primary, evidence, failure


class CausalScorer(_BaseScorer):
    """
    CAUSAL — "What happened immediately after X?"

    Full credit requires matching artifact_id AND event_type.
    Partial credit for artifact match without event_type match (0.5 primary).
    """

    def score(
        self, question: dict, agent_answer: dict
    ) -> Tuple[float, float, Optional[str]]:
        gt = question["ground_truth"]
        # POSTMORTEM questions are typed CAUSAL but use postmortem_confluence_id
        gt_id = gt.get("artifact_id") or gt.get("postmortem_confluence_id") or ""
        gt_etype = gt.get("event_type", "")
        agent_id = agent_answer.get("artifact_id", "")
        agent_et = agent_answer.get("event_type", "")

        id_correct = agent_id == gt_id
        et_correct = agent_et == gt_etype

        if id_correct and et_correct:
            primary = 1.0
            failure = None
        elif id_correct:
            primary = 0.6
            failure = f"Correct artifact but wrong event_type: expected {gt_etype!r}, got {agent_et!r}"
        else:
            primary = 0.0
            failure = f"Expected artifact_id={gt_id!r}, got {agent_id!r}"

        evidence = self._evidence_overlap(
            question.get("evidence_chain", []),
            agent_answer.get("retrieved_artifact_ids", []),
        )
        return primary, evidence, failure


class TemporalScorer(_BaseScorer):
    """
    TEMPORAL — "Did person P know about domain D when incident I was opened?"

    Full credit: had_knowledge boolean matches AND departure_day matches
                 (within ±1 day tolerance for off-by-one).
    Partial:     boolean correct but departure_day wrong or missing (0.6).
    """

    def score(
        self, question: dict, agent_answer: dict
    ) -> Tuple[float, float, Optional[str]]:
        gt = question["ground_truth"]
        gt_bool = gt.get("had_knowledge")
        gt_dep_day = gt.get("departure_day")  # int or None
        agent_bool = agent_answer.get("had_knowledge")
        agent_dep_day = agent_answer.get("departure_day")

        bool_match = agent_bool == gt_bool

        if not bool_match:
            primary = 0.0
            failure = f"had_knowledge expected {gt_bool}, got {agent_bool}"
        else:
            # Departure day agreement
            if gt_dep_day is None and agent_dep_day is None:
                # Both agree no departure occurred — full credit.
                # Note: when the dataset has had_knowledge=True for all questions
                # (e.g. short sim runs where no incident touches a departed
                # employee's domains), an agent that always returns
                # {"had_knowledge": true, "departure_day": null} will score 1.0
                # here. This is a known dataset limitation, not a scorer bug —
                # disclosed in the accompanying paper.
                primary = 1.0
                failure = None
            elif gt_dep_day is not None and agent_dep_day is not None:
                day_delta = abs(int(gt_dep_day) - int(agent_dep_day))
                if day_delta <= 1:
                    primary = 1.0
                    failure = None
                else:
                    primary = 0.6
                    failure = f"Departure day off by {day_delta} days (gt={gt_dep_day}, agent={agent_dep_day})"
            else:
                primary = 0.6
                failure = (
                    f"Agent missed departure day (expected {gt_dep_day})"
                    if gt_dep_day is not None
                    else "Agent reported a departure day that doesn't exist"
                )

        evidence = self._evidence_overlap(
            question.get("evidence_chain", []),
            agent_answer.get("retrieved_artifact_ids", []),
        )
        return primary, evidence, failure


class GapDetectionScorer(_BaseScorer):
    """
    GAP_DETECTION — "Was this email ever actioned?"

    Full credit: was_actioned boolean matches AND (if actioned=True)
                 downstream artifact overlap is ≥ 0.5.
    Partial:     boolean correct but poor downstream recall.
    """

    def score(
        self, question: dict, agent_answer: dict
    ) -> Tuple[float, float, Optional[str]]:
        gt = question["ground_truth"]
        gt_bool = gt.get("was_actioned")
        gt_downstream = gt.get("downstream_artifacts", [])
        agent_bool = agent_answer.get("was_actioned")
        agent_downstream = agent_answer.get("downstream_artifacts", [])

        if agent_bool != gt_bool:
            primary = 0.0
            failure = f"was_actioned expected {gt_bool}, got {agent_bool}"
        elif not gt_bool:
            primary = 1.0
            failure = None
        else:
            ds_overlap = self._evidence_overlap(gt_downstream, agent_downstream)
            primary = 0.6 + 0.4 * ds_overlap
            failure = (
                None
                if ds_overlap >= 0.5
                else (
                    f"Correct boolean but poor downstream artifact recall ({ds_overlap:.2f})"
                )
            )

        evidence = self._evidence_overlap(
            question.get("evidence_chain", []),
            agent_answer.get("retrieved_artifact_ids", []),
        )
        return primary, evidence, failure


class RoutingScorer(_BaseScorer):
    """
    ROUTING — "Who was the first internal person to receive this email?"
    Full credit: first_recipient matches.
    """

    def score(
        self, question: dict, agent_answer: dict
    ) -> Tuple[float, float, Optional[str]]:
        gt = question["ground_truth"]
        gt_recipient = gt.get("first_recipient", "")
        agent_recipient = agent_answer.get("first_recipient", "")

        recipient_match = (
            agent_recipient.strip().lower() == gt_recipient.strip().lower()
        )

        if recipient_match:
            primary = 1.0
            failure = None
        else:
            primary = 0.0
            failure = (
                f"Expected first_recipient={gt_recipient!r}, got {agent_recipient!r}"
            )

        evidence = self._evidence_overlap(
            question.get("evidence_chain", []),
            agent_answer.get("retrieved_artifact_ids", []),
        )
        return primary, evidence, failure


class PlanScorer(_BaseScorer):
    """
    PLAN — "What was department X's focus on Day N?"

    Full credit: dept AND theme both match.
    Partial:     dept correct but theme wrong (0.5).
    """

    def score(
        self, question: dict, agent_answer: dict
    ) -> Tuple[float, float, Optional[str]]:
        gt = question["ground_truth"]
        gt_dept = gt.get("dept", "").lower()
        gt_theme = gt.get("theme", "").lower()
        agent_dept = agent_answer.get("dept", "").lower()
        agent_theme = agent_answer.get("theme", "").lower()

        dept_match = agent_dept == gt_dept
        # Theme is LLM-generated prose — use substring match rather than exact.
        # Minimum length guard prevents trivially short strings (e.g. "stable")
        # from matching any ground truth that happens to contain them.
        _MIN_THEME_LEN = 5
        if len(gt_theme) < _MIN_THEME_LEN or len(agent_theme) < _MIN_THEME_LEN:
            theme_match = gt_theme == agent_theme
        else:
            theme_match = gt_theme in agent_theme or agent_theme in gt_theme

        if dept_match and theme_match:
            primary = 1.0
            failure = None
        elif dept_match:
            primary = 0.5
            failure = f"Correct dept but theme mismatch: expected {gt_theme!r}, got {agent_theme!r}"
        else:
            primary = 0.0
            failure = f"Expected dept={gt_dept!r}, got {agent_dept!r}"

        evidence = self._evidence_overlap(
            question.get("evidence_chain", []),
            agent_answer.get("retrieved_artifact_ids", []),
        )
        return primary, evidence, failure


class EscalationScorer(_BaseScorer):
    """
    ESCALATION — "Who was in the escalation chain for ticket X?"

    Full credit: all escalation_actors match (order-insensitive).
    Partial:     at least one actor correct (scaled by overlap).
    """

    def score(
        self, question: dict, agent_answer: dict
    ) -> Tuple[float, float, Optional[str]]:
        gt = question["ground_truth"]
        gt_actors = [a.lower() for a in gt.get("escalation_actors", [])]
        agent_actors = [a.lower() for a in agent_answer.get("escalation_actors", [])]

        if not gt_actors:
            primary = 1.0
            failure = None
        else:
            gt_set = set(gt_actors)
            agent_set = set(agent_actors)
            overlap = len(gt_set & agent_set) / len(gt_set)

            if overlap == 1.0 and len(agent_set) == len(gt_set):
                primary = 1.0
                failure = None
            elif overlap > 0:
                primary = round(0.4 + 0.6 * overlap, 4)  # 0.4 floor for partial
                failure = (
                    f"Partial actor match ({len(gt_set & agent_set)}/{len(gt_set)}): "
                    f"missing {gt_set - agent_set}"
                )
            else:
                primary = 0.0
                failure = f"No escalation actors matched. Expected {gt_actors}"

        evidence = self._evidence_overlap(
            question.get("evidence_chain", []),
            agent_answer.get("retrieved_artifact_ids", []),
        )
        return primary, evidence, failure


class PostmortemScorer(_BaseScorer):
    """
    POSTMORTEM — "Which Confluence doc contains the postmortem for incident X?"

    Ground truth uses postmortem_confluence_id (not artifact_id).
    Full credit: artifact_id matches postmortem_confluence_id.
    Partial credit: evidence_chain overlap (incident + confluence doc present).
    """

    def score(
        self, question: dict, agent_answer: dict
    ) -> Tuple[float, float, Optional[str]]:
        gt = question["ground_truth"]
        gt_id = gt.get("postmortem_confluence_id", "")
        agent_id = agent_answer.get("artifact_id", "")

        primary = 1.0 if agent_id == gt_id else 0.0
        failure = (
            None
            if primary == 1.0
            else f"Expected postmortem_confluence_id={gt_id!r}, got {agent_id!r}"
        )

        evidence = self._evidence_overlap(
            question.get("evidence_chain", []),
            agent_answer.get("retrieved_artifact_ids", []),
        )
        return primary, evidence, failure


class KnowledgeGapScorer(_BaseScorer):
    """
    KNOWLEDGE_GAP — "Which domain was undocumented during incident X?"

    Full credit: all gap_areas matched (order-insensitive).
    Partial:     at least one gap area correct (scaled by overlap).
    """

    def score(
        self, question: dict, agent_answer: dict
    ) -> Tuple[float, float, Optional[str]]:
        gt = question["ground_truth"]
        gt_gaps = [g.lower() for g in gt.get("gap_areas", [])]
        agent_gaps = [g.lower() for g in agent_answer.get("gap_areas", [])]

        if not gt_gaps:
            primary = 1.0
            failure = None
        else:
            gt_set = set(gt_gaps)
            agent_set = set(agent_gaps)
            overlap = len(gt_set & agent_set) / len(gt_set)

            if overlap == 1.0 and len(agent_set) == len(gt_set):
                primary = 1.0
                failure = None
            elif overlap > 0:
                primary = round(0.4 + 0.6 * overlap, 4)
                failure = (
                    f"Partial gap match ({len(gt_set & agent_set)}/{len(gt_set)}): "
                    f"missing {gt_set - agent_set}"
                )
            else:
                primary = 0.0
                failure = f"No gap areas matched. Expected {gt_gaps}"

        evidence = self._evidence_overlap(
            question.get("evidence_chain", []),
            agent_answer.get("retrieved_artifact_ids", []),
        )
        return primary, evidence, failure


class PRReviewScorer(_BaseScorer):
    """
    PR_REVIEW — "Who reviewed PR-X and what was the verdict?"

    Full credit:  pr_id matches AND verdict matches.
    Partial:      pr_id correct but verdict wrong (0.5).
                  reviewer correct adds 0.15 bonus on top, capped at 1.0.

    Verdict is case-insensitive and normalised so "LGTM" / "approve" /
    "approved" all resolve to "approved", and "changes" / "request changes"
    resolve to "changes_requested" before comparison.
    """

    _APPROVE_ALIASES = {"approved", "approve", "lgtm", "merged", "merge"}
    _CHANGES_ALIASES = {
        "changes_requested",
        "changes requested",
        "request changes",
        "needs changes",
        "needs work",
    }

    @staticmethod
    def _normalise_verdict(raw: str) -> str:
        v = raw.strip().lower()
        if v in PRReviewScorer._APPROVE_ALIASES:
            return "approved"
        if v in PRReviewScorer._CHANGES_ALIASES:
            return "changes_requested"
        return v  # return as-is — will fail comparison cleanly

    def score(
        self, question: dict, agent_answer: dict
    ) -> Tuple[float, float, Optional[str]]:
        gt = question["ground_truth"]
        gt_pr = gt.get("pr_id", "")
        gt_verdict = self._normalise_verdict(gt.get("verdict", ""))
        gt_reviewer = gt.get("reviewer", "").strip().lower()

        agent_pr = agent_answer.get("pr_id", "")
        agent_verdict = self._normalise_verdict(agent_answer.get("verdict", ""))
        agent_reviewer = agent_answer.get("reviewer", "").strip().lower()

        pr_match = agent_pr == gt_pr

        if not pr_match:
            primary = 0.0
            failure = f"Expected pr_id={gt_pr!r}, got {agent_pr!r}"
        elif agent_verdict == gt_verdict:
            primary = 1.0
            failure = None
        else:
            primary = 0.5
            failure = (
                f"Correct PR but wrong verdict: expected {gt_verdict!r}, "
                f"got {agent_verdict!r}"
            )

        # Reviewer identification bonus
        if gt_reviewer and agent_reviewer == gt_reviewer:
            primary = min(1.0, primary + 0.15)

        evidence = self._evidence_overlap(
            question.get("evidence_chain", []),
            agent_answer.get("retrieved_artifact_ids", []),
        )
        return primary, evidence, failure


class ZDResolutionScorer(_BaseScorer):
    """
    ZD_RESOLUTION — "Was Zendesk ticket X resolved and how long did it take?"

    Full credit:    resolved boolean matches AND duration_days matches exactly.
    Partial credit: boolean correct but duration wrong or missing (0.6).
                    escalated flag correct adds 0.1 bonus on top of partial.
    """

    def score(
        self, question: dict, agent_answer: dict
    ) -> Tuple[float, float, Optional[str]]:
        gt = question["ground_truth"]
        gt_resolved = gt.get("resolved")
        gt_duration = gt.get("duration_days")
        gt_escalated = gt.get("escalated", False)

        agent_resolved = agent_answer.get("resolved")
        agent_duration = agent_answer.get("duration_days")
        agent_escalated = agent_answer.get("escalated")

        if agent_resolved != gt_resolved:
            primary = 0.0
            failure = f"resolved expected {gt_resolved}, got {agent_resolved}"
        else:
            # Resolution boolean correct — check duration
            if gt_duration is None:
                # Ticket unresolved: correct if agent also has no duration
                primary = 1.0 if agent_duration is None else 0.7
                failure = (
                    None
                    if agent_duration is None
                    else "Correctly identified unresolved but reported a duration"
                )
            elif agent_duration is not None and int(agent_duration) == int(gt_duration):
                primary = 1.0
                failure = None
            elif agent_duration is not None:
                primary = 0.6
                failure = (
                    f"Duration off: expected {gt_duration}d, got {agent_duration}d"
                )
            else:
                primary = 0.6
                failure = f"Correct resolution status but missing duration (expected {gt_duration}d)"

            # Escalation awareness bonus (+0.1, capped at 1.0)
            if agent_escalated is not None and agent_escalated == gt_escalated:
                primary = min(1.0, primary + 0.1)

        evidence = self._evidence_overlap(
            question.get("evidence_chain", []),
            agent_answer.get("retrieved_artifact_ids", []),
        )
        return primary, evidence, failure


class SFRiskScorer(_BaseScorer):
    """
    SF_RISK — "Which Salesforce accounts were flagged at-risk after incident X?"

    Full credit:  all at_risk_accounts matched (order-insensitive) AND
                  incident_id correct.
    Partial:      incident_id correct but incomplete account list (scaled by overlap).
                  0.3 floor for incident match with zero account overlap.
    """

    def score(
        self, question: dict, agent_answer: dict
    ) -> Tuple[float, float, Optional[str]]:
        gt = question["ground_truth"]
        gt_incident = gt.get("incident_id", "")
        gt_accounts = [a.lower() for a in gt.get("at_risk_accounts", [])]

        agent_incident = agent_answer.get("incident_id", "")
        agent_accounts = [a.lower() for a in agent_answer.get("at_risk_accounts", [])]

        incident_match = agent_incident == gt_incident

        if not gt_accounts:
            primary = 1.0 if incident_match else 0.0
            failure = (
                None
                if incident_match
                else f"Expected incident_id={gt_incident!r}, got {agent_incident!r}"
            )
        elif not incident_match:
            primary = 0.0
            failure = f"Expected incident_id={gt_incident!r}, got {agent_incident!r}"
        else:
            gt_set = set(gt_accounts)
            agent_set = set(agent_accounts)
            overlap = len(gt_set & agent_set) / len(gt_set) if gt_set else 1.0

            if overlap == 1.0 and len(agent_set) == len(gt_set):
                primary = 1.0
                failure = None
            elif overlap > 0:
                primary = round(0.3 + 0.7 * overlap, 4)
                failure = (
                    f"Partial account match ({len(gt_set & agent_set)}/{len(gt_set)}): "
                    f"missing {gt_set - agent_set}"
                )
            else:
                primary = 0.3  # floor for correct incident identification
                failure = (
                    f"Correct incident but no accounts matched. Expected {gt_accounts}"
                )

        evidence = self._evidence_overlap(
            question.get("evidence_chain", []),
            agent_answer.get("retrieved_artifact_ids", []),
        )
        return primary, evidence, failure


class NPSScoreScorer(_BaseScorer):
    """
    NPS_SCORE — "What NPS score did customer X give and what drove it?"

    Full credit:  nps_score exact match AND classification correct.
    Partial:      classification correct but score wrong (0.6).
                  escalated_tickets count correct adds 0.1 bonus.
    """

    def score(
        self, question: dict, agent_answer: dict
    ) -> Tuple[float, float, Optional[str]]:
        gt = question["ground_truth"]
        gt_score = gt.get("nps_score")
        gt_class = gt.get("classification", "").lower()
        gt_escalated = gt.get("escalated_tickets", 0)

        agent_score = agent_answer.get("nps_score")
        agent_class = agent_answer.get("classification", "").lower()
        agent_escalated = agent_answer.get("escalated_tickets")

        class_match = agent_class == gt_class
        score_match = agent_score is not None and int(agent_score) == int(gt_score)

        if class_match and score_match:
            primary = 1.0
            failure = None
        elif class_match:
            primary = 0.6
            failure = f"Correct classification but wrong score: expected {gt_score}, got {agent_score}"
        else:
            primary = 0.0
            failure = (
                f"Classification wrong: expected {gt_class!r}, got {agent_class!r}"
            )

        # Escalation count awareness bonus
        if agent_escalated is not None and int(agent_escalated) == int(gt_escalated):
            primary = min(1.0, primary + 0.1)

        evidence = self._evidence_overlap(
            question.get("evidence_chain", []),
            agent_answer.get("retrieved_artifact_ids", []),
        )
        return primary, evidence, failure


class InvoiceSLAScorer(_BaseScorer):
    """
    INVOICE_SLA — "What SLA credit appeared on the invoice for customers
                   affected by incident X?"

    Full credit:  breach_duration_days exact AND sla_credit_per_org within 5%.
    Partial:      incident identified correctly but wrong duration/credit (0.5).
    Evidence:     affected_orgs overlap used as secondary evidence score.
    """

    def score(
        self, question: dict, agent_answer: dict
    ) -> Tuple[float, float, Optional[str]]:
        gt = question["ground_truth"]
        gt_incident = gt.get("incident_id", "")
        gt_duration = gt.get("breach_duration_days")
        gt_credit = gt.get("sla_credit_per_org")
        gt_orgs = [o.lower() for o in gt.get("affected_orgs", [])]

        agent_incident = agent_answer.get("incident_id", "")
        agent_duration = agent_answer.get("breach_duration_days")
        agent_credit = agent_answer.get("sla_credit_per_org")
        agent_orgs = [o.lower() for o in agent_answer.get("affected_orgs", [])]

        if agent_incident != gt_incident:
            primary = 0.0
            failure = f"Expected incident_id={gt_incident!r}, got {agent_incident!r}"
        else:
            duration_ok = agent_duration is not None and int(agent_duration) == int(
                gt_duration
            )
            credit_ok = False
            if agent_credit is not None and gt_credit:
                try:
                    ratio = abs(float(agent_credit) - float(gt_credit)) / float(
                        gt_credit
                    )
                    credit_ok = ratio <= 0.05
                except (TypeError, ZeroDivisionError):
                    pass

            if duration_ok and credit_ok:
                primary = 1.0
                failure = None
            elif duration_ok:
                primary = 0.7
                failure = f"Correct duration but credit off: expected {gt_credit}, got {agent_credit}"
            elif credit_ok:
                primary = 0.7
                failure = f"Correct credit but duration off: expected {gt_duration}d, got {agent_duration}d"
            else:
                primary = 0.4  # floor for correct incident identification
                failure = (
                    f"Correct incident but duration ({agent_duration} vs {gt_duration}) "
                    f"and credit ({agent_credit} vs {gt_credit}) both wrong"
                )

        # Affected orgs as evidence (secondary signal for partial credit)
        evidence = (
            self._evidence_overlap(
                gt_orgs,
                agent_orgs,
            )
            if gt_orgs
            else self._evidence_overlap(
                question.get("evidence_chain", []),
                agent_answer.get("retrieved_artifact_ids", []),
            )
        )
        return primary, evidence, failure


_SCORERS: Dict[str, _BaseScorer] = {
    "RETRIEVAL": RetrievalScorer(),
    "CAUSAL": CausalScorer(),
    "TEMPORAL": TemporalScorer(),
    "GAP_DETECTION": GapDetectionScorer(),
    "ROUTING": RoutingScorer(),
    "PLAN": PlanScorer(),
    "ESCALATION": EscalationScorer(),
    "KNOWLEDGE_GAP": KnowledgeGapScorer(),
    "POSTMORTEM": PostmortemScorer(),
    "STANDUP": RetrievalScorer(),
    "CUSTOMER_ESC": CausalScorer(),
    "ZD_RESOLUTION": ZDResolutionScorer(),
    "ZD_ESCALATION": CausalScorer(),
    "SF_RISK": SFRiskScorer(),
    "NPS_SCORE": NPSScoreScorer(),
    "INVOICE_SLA": InvoiceSLAScorer(),
    "DATADOG_ALERT": RetrievalScorer(),
    "PR_REVIEW": PRReviewScorer(),
}


class OrgForgeScorer:
    """
    Entry point for scoring OrgForge eval questions.

    scorer = OrgForgeScorer()
    result = scorer.score(question_dict, agent_answer_dict)
    """

    def score(self, question: dict, agent_answer: dict) -> ScorerResult:
        qtype = question.get("question_type", "UNKNOWN")
        qid = question.get("question_id", "")

        # POSTMORTEM questions are stored with question_type=CAUSAL but use a
        # different ground_truth schema (postmortem_confluence_id, not artifact_id).
        # Route by question_id prefix so they get the right scorer.
        if qid.startswith("postmortem_"):
            qtype = "POSTMORTEM"

        scorer_impl = _SCORERS.get(qtype)

        if scorer_impl is None:
            logger.warning(f"No scorer for question type: {qtype!r}. Returning 0.")
            return ScorerResult(
                question_id=question.get("question_id", "?"),
                question_type=qtype,
                difficulty=question.get("difficulty", "unknown"),
                score=0.0,
                primary_score=0.0,
                evidence_score=0.0,
                correct=False,
                partial=False,
                failure_reason=f"No scorer registered for type {qtype!r}",
            )

        primary, evidence, failure = scorer_impl.score(question, agent_answer)
        combined = scorer_impl._combine(primary, evidence)

        return ScorerResult(
            question_id=question.get("question_id", "?"),
            question_type=qtype,
            difficulty=question.get("difficulty", "unknown"),
            score=round(combined, 4),
            primary_score=round(primary, 4),
            evidence_score=round(evidence, 4),
            correct=combined >= 0.90,
            partial=0.20 <= combined < 0.90,
            failure_reason=failure,
            meta={
                "requires_reasoning": question.get("requires_reasoning", False),
                "chain_id": question.get("chain_id"),
            },
        )

    def score_all(
        self,
        questions: List[dict],
        agent_answers: Dict[str, dict],  # {question_id: answer_dict}
    ) -> List[ScorerResult]:
        """
        Score every question. Questions without a matching answer receive 0.
        """
        results = []
        for q in questions:
            qid = q.get("question_id", "")
            answer = agent_answers.get(qid, {})
            if not answer:
                logger.debug(f"No answer provided for {qid!r} — scoring as 0.")
            results.append(self.score(q, answer))
        return results

    def report(self, results: List[ScorerResult]) -> Dict[str, Any]:
        """
        Aggregate statistics over a scored result set.
        Returns a dict suitable for JSON serialisation or dataset-card reporting.
        """
        if not results:
            return {}

        def _mean(vals):
            return round(sum(vals) / len(vals), 4) if vals else 0.0

        by_type: Dict[str, List[float]] = {}
        by_difficulty: Dict[str, List[float]] = {}

        for r in results:
            by_type.setdefault(r.question_type, []).append(r.score)
            by_difficulty.setdefault(r.difficulty, []).append(r.score)

        total = len(results)
        all_scores = [r.score for r in results]

        return {
            "total_questions": total,
            "overall_score": _mean(all_scores),
            "accuracy": round(sum(r.correct for r in results) / total, 4),
            "partial_rate": round(sum(r.partial for r in results) / total, 4),
            "by_type": {
                qtype: {
                    "n": len(scores),
                    "mean_score": _mean(scores),
                    "accuracy": round(
                        sum(
                            r.score >= 0.90 for r in results if r.question_type == qtype
                        )
                        / len(scores),
                        4,
                    ),
                }
                for qtype, scores in by_type.items()
            },
            "by_difficulty": {
                diff: {
                    "n": len(scores),
                    "mean_score": _mean(scores),
                }
                for diff, scores in by_difficulty.items()
            },
            "reasoning_vs_direct": {
                "requires_reasoning": _mean(
                    [r.score for r in results if r.meta.get("requires_reasoning")]
                ),
                "direct": _mean(
                    [r.score for r in results if not r.meta.get("requires_reasoning")]
                ),
            },
        }


if __name__ == "__main__":
    import json
    import pathlib
    import sys

    eval_path = pathlib.Path("export/eval/eval_questions.json")
    if not eval_path.exists():
        print("No eval_questions.json found. Run eval_harness.py first.")
        sys.exit(1)

    data = json.loads(eval_path.read_text())
    questions = data.get("questions", [])

    mock_answers = {}
    for q in questions:
        gt = q["ground_truth"]
        qid = q["question_id"]
        qtype = q["question_type"]
        if qtype == "RETRIEVAL":
            mock_answers[qid] = {"artifact_id": gt.get("artifact_id", "")}
        elif qtype == "CAUSAL":
            mock_answers[qid] = {
                "artifact_id": gt.get("artifact_id", ""),
                "event_type": "wrong_type",  # intentionally wrong to demonstrate partial credit scoring
            }
        elif qtype == "TEMPORAL":
            mock_answers[qid] = {"had_knowledge": gt.get("had_knowledge")}
        elif qtype == "GAP_DETECTION":
            mock_answers[qid] = {"was_actioned": gt.get("was_actioned")}
        elif qtype == "ROUTING":
            mock_answers[qid] = {"first_recipient": gt.get("first_recipient", "")}
        elif qtype == "PLAN":
            mock_answers[qid] = {
                "dept": gt.get("dept", ""),
                "theme": gt.get("theme", ""),
            }
        elif qtype == "ESCALATION":
            mock_answers[qid] = {
                "escalation_actors": gt.get("escalation_actors", []),
            }
        elif qtype == "KNOWLEDGE_GAP":
            mock_answers[qid] = {
                "gap_areas": gt.get("gap_areas", []),
            }
        elif qtype == "ZD_RESOLUTION":
            mock_answers[qid] = {
                "resolved": gt.get("resolved"),
                "duration_days": gt.get("duration_days"),
                "escalated": gt.get("escalated"),
            }
        elif qtype == "ZD_ESCALATION":
            mock_answers[qid] = {
                "artifact_id": gt.get("artifact_id", ""),
                "event_type": gt.get("event_type", ""),
            }
        elif qtype == "SF_RISK":
            mock_answers[qid] = {
                "incident_id": gt.get("incident_id", ""),
                "at_risk_accounts": gt.get("at_risk_accounts", []),
            }
        elif qtype == "NPS_SCORE":
            mock_answers[qid] = {
                "nps_score": gt.get("nps_score"),
                "classification": gt.get("classification", ""),
                "escalated_tickets": gt.get("escalated_tickets", 0),
            }
        elif qtype == "INVOICE_SLA":
            mock_answers[qid] = {
                "incident_id": gt.get("incident_id", ""),
                "breach_duration_days": gt.get("breach_duration_days"),
                "sla_credit_per_org": gt.get("sla_credit_per_org"),
                "affected_orgs": gt.get("affected_orgs", []),
            }
        elif qtype == "DATADOG_ALERT":
            mock_answers[qid] = {"artifact_id": gt.get("artifact_id", "")}
        elif qtype == "PR_REVIEW":
            mock_answers[qid] = {
                "pr_id": gt.get("pr_id", ""),
                "verdict": gt.get("verdict", ""),
                "reviewer": gt.get("reviewer", ""),
            }

    scorer = OrgForgeScorer()
    results = scorer.score_all(questions, mock_answers)
    report = scorer.report(results)
    print(json.dumps(report, indent=2))
