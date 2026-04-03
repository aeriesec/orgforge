"""
agentic_eval_harness.py
=======================
OrgForge Agentic Evaluation Harness — v2

Evaluates AI agents on three novel tracks that require the deterministic
state machine to exist. No retrieval scoring. Each track has its own
trajectory model and scorer because the reasoning structure is fundamentally
different for each.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TRACK 1 — PERSPECTIVE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Temporal gate: as_of_time from question (actor's knowledge horizon)
Actor gate: tool calls filtered to actor_visible_artifacts
Trajectory: did the agent stay within the actor's visibility cone?
            did it correctly identify what was and wasn't accessible?
Score penalty: using artifacts outside the actor's cone, even to reach
               the correct answer. The point is epistemic discipline.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TRACK 2 — COUNTERFACTUAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Temporal gate: as_of_time of the effect event (agent can see everything up to
               and including the effect to understand what happened)
No actor gate: agent has read access to all subsystems
Trajectory: did the agent identify the correct causal mechanism?
            did it trace cause → effect correctly?
Answer scoring: structured extraction of (outcome_changed, mechanism, actors)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TRACK 3 — SILENCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Temporal gate: end of simulation (agent can see the full corpus)
No actor gate: agent has read access to all subsystems
Trajectory: CRITICAL — did the agent search expected_search_space before
            concluding absence? A correct "no" without checking the right
            places is scored as a trajectory failure even if the boolean is right.
Answer scoring: boolean only — did the agent correctly conclude absence?

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Score weights (per track)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Track           Answer  Trajectory  Notes
─────────────── ─────── ──────────  ─────────────────────────
PERSPECTIVE      0.40    0.60       Trajectory is primary — epistemic discipline matters
COUNTERFACTUAL   0.50    0.50       Both matter — wrong mechanism, wrong answer
SILENCE          0.30    0.70       Can't score a "no" without proof of search
"""

from __future__ import annotations

import json
import logging
import re
from statistics import mean
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import argparse
from config_loader import CONFIG
import yaml

from eval_harness import _ARTIFACT_SUBSYSTEM

logger = logging.getLogger("orgforge.agentic_eval")

with open(Path(__file__).resolve().parent.parent / "config" / "config.yaml") as f:
    _CFG = yaml.safe_load(f)

_SIM_CFG = _CFG.get("simulation", {})
BASE = Path(_SIM_CFG.get("output_dir", "./export"))
EVAL_DIR = BASE / "eval"
_SIM_START = datetime.strptime(CONFIG["simulation"]["start_date"], "%Y-%m-%d")

# Per-track answer/trajectory weights
_TRACK_WEIGHTS = {
    "PERSPECTIVE": {"answer": 0.40, "trajectory": 0.60},
    "COUNTERFACTUAL": {"answer": 0.50, "trajectory": 0.50},
    "SILENCE": {"answer": 0.30, "trajectory": 0.70},
}

# Doc type → tool name mapping
_DOCTYPE_TO_TOOL = {
    "jira": "get_ticket",
    "confluence": "get_confluence_page",
    "slack": "get_slack_thread",
    "email": "get_email",
    "pr": "get_pr",
    "zd_ticket": "get_zd_ticket",
    "sf_opp": "get_sf_opportunity",
    "sf_account": "get_sf_account",
    "zoom": "get_zoom_transcript",
    "datadog": "get_datadog_alert",
    "invoice": "get_invoice",
    "nps": "get_nps_response",
}

# Tool names that imply a subsystem — used to detect actor gate violations
_TOOL_SUBSYSTEM = {
    "get_ticket": "jira",
    "get_confluence_page": "confluence",
    "get_slack_thread": "slack",
    "get_email": "email",
    "get_pr": "git",
    "get_zd_ticket": "zendesk",
    "get_sf_opportunity": "salesforce",
    "get_sf_account": "salesforce",
    "get_zoom_transcript": "zoom",
    "get_datadog_alert": "datadog",
    "get_invoice": "email",
    "get_nps_response": "salesforce",
    "get_events_for_day": None,  # cross-subsystem — handled separately
    "search_artifacts": None,
}

_SUBSYSTEM_EVENT_TYPES: Dict[str, Set[str]] = {
    "jira": {
        "incident_opened",
        "incident_resolved",
        "ticket_progress",
        "pr_review",
        "sprint_planned",
        "sprint_goal_updated",
        "postmortem_created",
    },
    "slack": {
        "standup",
        "normal_day_slack",
        "watercooler_chat",
        "farewell_message",
        "onboarding_session",
        "warmup_1on1",
        "morale_intervention",
        "1on1_scheduled",
    },
    "confluence": {
        "confluence_created",
        "design_discussion",
        "retrospective",
        "leadership_sync",
    },
    "git": {
        "pr_review",
        "code_review_comment",
    },
    "email": {
        "inbound_external_email",
        "customer_email_routed",
        "vendor_email_routed",
        "hr_outbound_email",
        "sales_outbound_email",
        "email_dropped",
        "hr_checkin",
    },
    "zoom": {
        "zoom_meeting",
        "design_discussion",
        "vendor_meeting",
        "async_question",
        "deep_work_session",
    },
    "salesforce": {
        "crm_touchpoint",
        "crm_account_at_risk",
        "customer_health_briefing",
        "feature_request_from_sales",
        "stability_update_to_sales",
        "proactive_outreach_initiated",
        "sf_deals_risk_flagged",
    },
    "zendesk": {
        "zd_ticket_opened",
        "zd_tickets_escalated",
        "zd_tickets_resolved",
        "customer_escalation",
    },
    "datadog": {
        "dlp_alert",
        "secret_detected",
    },
}

# Sim-internal types never exposed to any actor
_INTERNAL_EVENT_TYPES = {
    "knowledge_gap_detected",
    "escalation_chain",
    "assignment_domain_mismatch",
    "sf_ownership_lapsed",
    "fix_in_progress",
    "day_summary",
    "employee_departed",
    "employee_hired",
    "external_contact_summarized",
    "vendor_email_routed",
    "secret_detected",
}

KNOWN_EVENT_TYPES = {
    "incident_opened",
    "incident_resolved",
    "escalation_chain",
    "fix_in_progress",
    "postmortem_created",
    "knowledge_gap_detected",
    "standup",
    "pr_review",
    "ticket_progress",
    "design_discussion",
    "async_question",
    "code_review_comment",
    "deep_work_session",
    "sprint_planned",
    "retrospective",
    "sprint_goal_updated",
    "leadership_sync",
    "feature_request_from_sales",
    "stability_update_to_sales",
    "hr_checkin",
    "morale_intervention",
    "1on1_scheduled",
    "external_contact_summarized",
    "vendor_meeting",
    "customer_escalation",
    "normal_day_slack",
    "confluence_created",
    "day_summary",
    "employee_departed",
    "employee_hired",
    "onboarding_session",
    "farewell_message",
    "warmup_1on1",
    "watercooler_chat",
    "inbound_external_email",
    "customer_email_routed",
    "customer_escalation",
    "vendor_email_routed",
    "hr_outbound_email",
    "email_dropped",
    "dlp_alert",
    "secret_detected",
    "zoom_meeting",
    "sales_outbound_email",
    "proactive_outreach_initiated",
    "zd_ticket_opened",
    "zd_tickets_escalated",
    "zd_tickets_resolved",
    "sf_deals_risk_flagged",
    "sf_ownership_lapsed",
    "crm_touchpoint",
    "crm_account_at_risk",
    "customer_health_briefing",
    "assignment_domain_mismatch",
}

_TEMPORAL_DRIFT_THRESHOLD_DAYS = 5


def _business_day_to_date(start: datetime, n: int) -> datetime:
    """Convert a 1-based business day counter to a calendar date."""
    current = start
    days_counted = 0
    while days_counted < n:
        current += timedelta(days=1)
        if current.weekday() < 5:
            days_counted += 1
    return current


def _date_to_business_day(start: datetime, target: datetime) -> int:
    count = 0
    current = start
    while current < target:
        current += timedelta(days=1)
        if current.weekday() < 5:
            count += 1
    return count


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ToolCall:
    tool_name: str
    arguments: Dict[str, Any]
    result_ids: List[str]
    result_types: List[str]
    timestamp_requested: Optional[str]
    timestamp_applied: Optional[str]
    temporal_drift_days: Optional[float]
    temporal_drift_violation: bool
    horizon_violation: bool  # artifact timestamp > as_of_time
    actor_gate_violation: (
        bool  # artifact outside actor's visibility cone (PERSPECTIVE only)
    )
    subsystem_violation: (
        bool  # tool subsystem not in actor's access set (PERSPECTIVE only)
    )
    returned_empty: bool
    latency_ms: float


@dataclass
class AgentTrajectory:
    question_id: str
    question_type: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    final_answer: Dict[str, Any] = field(default_factory=dict)
    total_latency_ms: float = 0.0
    horizon_violations: int = 0
    actor_gate_violations: int = 0  # PERSPECTIVE track
    subsystem_violations: int = 0  # PERSPECTIVE track
    search_space_coverage: float = 0.0  # SILENCE track
    causal_mechanism_found: bool = False  # COUNTERFACTUAL track
    dead_ends_hit: int = 0
    dead_ends_recovered: int = 0


@dataclass
class PerspectiveTrajectoryScore:
    epistemic_discipline: float  # 1.0 - (cone violations / total calls)
    subsystem_discipline: float  # 1.0 - (subsystem violations / total calls)
    horizon_discipline: float  # 1.0 - (horizon violations / total calls)
    temporal_precision: float
    temporal_drift_discipline: float
    conclusion_grounding: float  # did final answer cite in-cone artifacts?
    dead_end_recovery: float
    composite: float


@dataclass
class CounterfactualTrajectoryScore:
    cause_identified: float  # did agent retrieve the cause event?
    effect_identified: float  # did agent retrieve the effect event?
    mechanism_correct: float  # did agent name the correct link_type?
    causal_chain_complete: float  # did agent traverse cause → effect in order?
    horizon_discipline: float
    composite: float


@dataclass
class SilenceTrajectoryScore:
    search_space_coverage: float  # fraction of expected_search_space the agent checked
    correct_absence_conclusion: float  # did agent explicitly conclude "does not exist"?
    premature_conclusion: float  # did agent conclude before searching? (penalty)
    horizon_discipline: float
    composite: float


@dataclass
class EvalResult:
    question_id: str
    question_type: str
    difficulty: str
    answer_score: float
    answer_correct: bool
    trajectory_score: float
    combined_score: float
    failure_reason: Optional[str]
    tool_call_count: int
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# GATED TOOL LAYER
# ─────────────────────────────────────────────────────────────────────────────


class GatedTools:
    """
    Wraps the document corpus and enforces gates per question type.

    PERSPECTIVE: temporal gate (as_of_time) + actor gate (visibility cone)
    COUNTERFACTUAL: temporal gate only (as_of_time = effect event timestamp)
    SILENCE: no gate (agent sees full corpus — the absence must be real)

    Violations are logged but results are still returned (the agent should
    observe them and self-correct). Violations penalize trajectory score.
    """

    def __init__(
        self,
        mem,
        question: dict,
        as_of_time: str,
        actor_visible_artifacts: Optional[Set[str]] = None,
        actor_subsystem_access: Optional[Set[str]] = None,
    ):
        self._mem = mem
        self._question = question
        self._as_of_time = as_of_time
        self._actor_visible = actor_visible_artifacts or set()
        self._actor_subsystems = actor_subsystem_access or set()
        self._question_type = question.get("question_type", "")
        self._call_log: List[ToolCall] = []

    def _gate_ts(self) -> str:
        if self._question_type == "SILENCE":
            trigger_day = self._question.get("trigger_day", 30)
            return _business_day_to_date(_SIM_START, trigger_day).isoformat()
        return self._as_of_time

    @property
    def call_log(self) -> List[ToolCall]:
        return self._call_log

    def _temporal_gate(self, doc: dict) -> bool:
        ts = doc.get("timestamp") or doc.get("created") or doc.get("date")
        if not ts:
            return True
        try:
            return datetime.fromisoformat(str(ts)) <= datetime.fromisoformat(
                self._gate_ts()
            )
        except (ValueError, TypeError):
            return True

    def _check_actor_gate(self, doc_id: str, doc_type: str) -> Tuple[bool, bool]:
        """
        Returns (actor_gate_violation, subsystem_violation).
        Only meaningful for PERSPECTIVE questions.
        """
        if self._question_type != "PERSPECTIVE":
            return False, False

        subsystem = _ARTIFACT_SUBSYSTEM.get(doc_type, "default")

        subsystem_violation = (
            bool(self._actor_subsystems)
            and subsystem not in self._actor_subsystems
            and subsystem != "default"
        )

        actor_gate_violation = (
            bool(self._actor_visible) and doc_id not in self._actor_visible
        )

        return actor_gate_violation, subsystem_violation

    def _record(
        self,
        tool_name: str,
        arguments: Dict,
        results: List[dict],
        t0: float,
        horizon_violation: bool = False,
        timestamp_applied: Optional[str] = None,
    ) -> List[dict]:
        latency = (time.time() - t0) * 1000
        filtered = [r for r in results if self._temporal_gate(r)]
        horizon_violation = horizon_violation or len(filtered) < len(results)

        result_ids = [str(r.get("id", r.get("_id", ""))) for r in filtered]
        result_types = [str(r.get("doc_type", r.get("type", ""))) for r in filtered]

        actor_gate_violation = False
        subsystem_violation = False
        for rid, rtype in zip(result_ids, result_types):
            agv, sv = self._check_actor_gate(rid, rtype)
            if agv:
                actor_gate_violation = True
            if sv:
                subsystem_violation = True

        # Check subsystem from tool name too
        tool_subsystem = _TOOL_SUBSYSTEM.get(tool_name)
        if (
            self._question_type == "PERSPECTIVE"
            and tool_subsystem
            and self._actor_subsystems
            and tool_subsystem not in self._actor_subsystems
        ):
            subsystem_violation = True

        requested = arguments.get("as_of_time")
        drift = None
        if requested and timestamp_applied:
            try:
                drift = (
                    datetime.fromisoformat(timestamp_applied)
                    - datetime.fromisoformat(requested)
                ).days
            except (ValueError, TypeError):
                pass

        temporal_drift_violation = (
            drift is not None and drift < -_TEMPORAL_DRIFT_THRESHOLD_DAYS
        )

        self._call_log.append(
            ToolCall(
                tool_name=tool_name,
                arguments=arguments,
                result_ids=result_ids,
                result_types=result_types,
                timestamp_requested=requested,
                timestamp_applied=timestamp_applied,
                temporal_drift_days=drift,
                temporal_drift_violation=temporal_drift_violation,
                horizon_violation=horizon_violation,
                actor_gate_violation=actor_gate_violation,
                subsystem_violation=subsystem_violation,
                returned_empty=len(filtered) == 0,
                latency_ms=latency,
            )
        )
        return filtered

    # ── Tool implementations ──────────────────────────────────────────────────
    # Each mirrors a real MongoDB query. The agent is given these as tools.

    _COLLECTION_TS_FIELD = {
        "jira": "created_at",
        "jira_tickets": "created_at",
        "confluence": "timestamp",
        "slack": "timestamp",
        "email": "timestamp",
        "pr": "created_at",
        "zd_ticket": "timestamp",
        "sf_opp": "timestamp",
        "sf_account": "timestamp",
        "zoom": "timestamp",
        "datadog": "timestamp",
        "invoice": "timestamp",
        "nps": "timestamp",
    }

    def _build_query(
        self,
        base: dict,
        doc_type: str = "",
        id_field: str = "id",
        agent_as_of_time: Optional[str] = None,
    ) -> Tuple[Optional[dict], str]:
        """
        Constructs a MongoDB filter with temporal and actor gates applied.
        base: the caller's own filter fields e.g. {"id": ticket_id}
        doc_type: the artifact type for subsystem gate checking
        """
        ceiling = self._gate_ts()
        if agent_as_of_time:
            effective_ts = min(agent_as_of_time, ceiling)
        else:
            effective_ts = ceiling

        query = {**base}

        ts_field = self._COLLECTION_TS_FIELD.get(doc_type, "timestamp")
        query[ts_field] = {"$lte": effective_ts}

        if self._question_type == "PERSPECTIVE" and doc_type:
            subsystem = _ARTIFACT_SUBSYSTEM.get(doc_type, "default")

            if (
                self._actor_subsystems
                and subsystem not in self._actor_subsystems
                and subsystem != "default"
            ):
                return None, effective_ts

            if self._actor_visible:
                query[id_field] = {"$in": list(self._actor_visible)}

                if "id" in base:
                    query[id_field] = (
                        base["id"]
                        if base["id"] in self._actor_visible
                        else "__blocked__"
                    )

        return query, effective_ts

    def get_ticket(self, ticket_id: str) -> dict:
        t0 = time.time()
        gate = self._gate_ts()
        query = self._build_query({"id": ticket_id}, doc_type="jira")
        if query is None:
            self._record("get_ticket", {"ticket_id": ticket_id}, [], t0)
            return {}

        doc = self._mem._db["jira_tickets"].find_one(query) or {}

        if doc:
            comments = doc.get("comments", [])
            doc["comments"] = [c for c in comments if c.get("created", "9999") <= gate]

            created = doc.get("created_at", "9999")
            in_progress_day = doc.get("in_progress_since")
            in_review_day = doc.get("in_review_since")

            def day_to_iso(day):
                return (
                    (_SIM_START + timedelta(days=day - 1)).isoformat()
                    if day
                    else "9999"
                )

            in_progress_dt = day_to_iso(in_progress_day)
            in_review_dt = day_to_iso(in_review_day)
            completed = (
                doc.get("updated_at", "9999") if doc.get("status") == "Done" else "9999"
            )

            if completed <= gate:
                derived_status = "Done"
            elif in_review_dt <= gate:
                derived_status = "In Review"
            elif in_progress_dt <= gate:
                derived_status = "In Progress"
            else:
                derived_status = "To Do"

            doc["status"] = derived_status
            if derived_status != "Done":
                doc.pop("completion_artifact", None)

            doc.pop("causal_chain", None)
            doc.pop("updated_at", None)

            if doc.get("linked_prs"):
                visible_prs = []
                for pr_id in doc["linked_prs"]:
                    pr = self._mem._db["prs"].find_one(
                        {"id": pr_id, "created_at": {"$lte": gate}}, {"id": 1}
                    )
                    if pr:
                        visible_prs.append(pr_id)
                doc["linked_prs"] = visible_prs

            if in_progress_day:
                if in_progress_dt > gate:
                    doc.pop("in_progress_since", None)
            if in_review_day:
                if in_review_dt > gate:
                    doc.pop("in_review_since", None)
                    doc.pop("last_review_requested_day", None)

        results = self._record(
            "get_ticket", {"ticket_id": ticket_id}, [doc] if doc else [], t0
        )
        return results[0] if results else {}

    def get_confluence_page(self, page_id: str) -> dict:
        t0 = time.time()
        query, effective_ts = self._build_query({"id": page_id}, doc_type="confluence")
        if query is None:
            self._record("get_confluence_page", {"page_id": page_id}, [], t0)
            return {}
        doc = self._mem._db["confluence"].find_one(query, {"_id": 0}) or {}
        results = self._record(
            "get_confluence_page",
            {"page_id": page_id},
            [doc] if doc else [],
            t0,
            timestamp_applied=effective_ts,
        )
        return results[0] if results else {}

    def get_slack_thread(self, thread_id: str) -> List[dict]:
        t0 = time.time()
        query, effective_ts = self._build_query(
            {"thread_id": thread_id},
            doc_type="slack",
            id_field="thread_id",
        )
        if query is None:
            return self._record("get_slack_thread", {"thread_id": thread_id}, [], t0)
        docs = list(self._mem._db["slack"].find(query, {"_id": 0}))
        return self._record(
            "get_slack_thread",
            {"thread_id": thread_id},
            docs,
            t0,
            timestamp_applied=effective_ts,
        )

    def get_email(self, email_id: str) -> dict:
        t0 = time.time()
        query, effective_ts = self._build_query({"id": email_id}, doc_type="email")
        if query is None:
            self._record("get_email", {"email_id": email_id}, [], t0)
            return {}
        doc = self._mem._db["emails"].find_one(query, {"_id": 0}) or {}
        results = self._record(
            "get_email",
            {"email_id": email_id},
            [doc] if doc else [],
            t0,
            timestamp_applied=effective_ts,
        )
        return results[0] if results else {}

    def get_pr(self, pr_id: str) -> dict:
        t0 = time.time()
        query, effective_ts = self._build_query({"id": pr_id}, doc_type="pr")
        if query is None:
            self._record("get_pr", {"pr_id": pr_id}, [], t0)
            return {}
        doc = self._mem._db["prs"].find_one(query, {"_id": 0}) or {}
        results = self._record(
            "get_pr",
            {"pr_id": pr_id},
            [doc] if doc else [],
            t0,
            timestamp_applied=effective_ts,
        )
        return results[0] if results else {}

    def get_zd_ticket(self, ticket_id: str) -> dict:
        t0 = time.time()
        query, effective_ts = self._build_query({"id": ticket_id}, doc_type="zd_ticket")
        if query is None:
            self._record("get_zd_ticket", {"ticket_id": ticket_id}, [], t0)
            return {}
        doc = self._mem._db["zd_tickets"].find_one(query, {"_id": 0}) or {}
        results = self._record(
            "get_zd_ticket",
            {"ticket_id": ticket_id},
            [doc] if doc else [],
            t0,
            timestamp_applied=effective_ts,
        )
        return results[0] if results else {}

    def get_sf_opportunity(self, opp_id: str) -> dict:
        t0 = time.time()
        query, effective_ts = self._build_query(
            {"id": opp_id}, doc_type="sf_opportunity"
        )
        if query is None:
            self._record("get_sf_opportunity", {"opp_id": opp_id}, [], t0)
            return {}
        doc = self._mem._db["salesforce_opps"].find_one(query, {"_id": 0}) or {}
        results = self._record(
            "get_sf_opportunity",
            {"opp_id": opp_id},
            [doc] if doc else [],
            t0,
            timestamp_applied=effective_ts,
        )
        return results[0] if results else {}

    def get_sf_account(self, account_id: str) -> dict:
        t0 = time.time()
        query, effective_ts = self._build_query(
            {"id": account_id}, doc_type="sf_account"
        )
        if query is None:
            self._record("get_sf_account", {"account_id": account_id}, [], t0)
            return {}
        doc = self._mem._db["salesforce_accounts"].find_one(query, {"_id": 0}) or {}
        results = self._record(
            "get_sf_account",
            {"account_id": account_id},
            [doc] if doc else [],
            t0,
            timestamp_applied=effective_ts,
        )
        return results[0] if results else {}

    def get_zoom_transcript(self, transcript_id: str) -> dict:
        t0 = time.time()
        query, effective_ts = self._build_query({"id": transcript_id}, doc_type="zoom")
        if query is None:
            self._record(
                "get_zoom_transcript", {"transcript_id": transcript_id}, [], t0
            )
            return {}

        doc = (
            self._mem._db["artifacts"].find_one(query, {"_id": 0, "embedding": 0}) or {}
        )
        if not doc:
            self._record(
                "get_zoom_transcript",
                {"transcript_id": transcript_id},
                [],
                t0,
                timestamp_applied=effective_ts,
            )
            return {}

        date_str = doc.get("date", "")
        md_path = BASE / "zoom" / date_str / f"{transcript_id}.md"
        try:
            doc["transcript"] = md_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            logger.warning(
                f"[get_zoom_transcript] Transcript file not found: {md_path}"
            )
            doc["transcript"] = doc.get("content", "")

        results = self._record(
            "get_zoom_transcript",
            {"transcript_id": transcript_id},
            [doc],
            t0,
            timestamp_applied=effective_ts,
        )
        return results[0] if results else {}

    def get_datadog_alert(self, alert_id: str) -> dict:
        t0 = time.time()
        effective_ts = self._gate_ts()

        path = BASE / "datadog" / "alerts.jsonl"
        if not path.exists():
            self._record(
                "get_datadog_alert",
                {"alert_id": alert_id},
                [],
                t0,
                timestamp_applied=effective_ts,
            )
            return {}

        doc = None
        with open(path) as f:
            for line in f:
                try:
                    alert = json.loads(line)
                    if alert.get("id") == alert_id:
                        doc = alert
                        break
                except json.JSONDecodeError:
                    continue

        if not doc:
            self._record(
                "get_datadog_alert",
                {"alert_id": alert_id},
                [],
                t0,
                timestamp_applied=effective_ts,
            )
            return {}

        if self._question_type != "SILENCE":
            date_happened = doc.get("date_happened", 0)
            if date_happened:
                doc_ts = datetime.fromtimestamp(date_happened).isoformat()
                if doc_ts > self._gate_ts():
                    self._record(
                        "get_datadog_alert",
                        {"alert_id": alert_id},
                        [],
                        t0,
                        timestamp_applied=effective_ts,
                    )
                    return {}

        results = self._record("get_datadog_alert", {"alert_id": alert_id}, [doc], t0)
        return results[0] if results else {}

    def get_invoice(self, invoice_id: str) -> dict:
        t0 = time.time()
        effective_ts = self._gate_ts()

        path = BASE / "invoices" / f"{invoice_id}.json"
        if not path.exists():
            self._record(
                "get_invoice",
                {"invoice_id": invoice_id},
                [],
                t0,
                timestamp_applied=effective_ts,
            )
            return {}
        doc = json.loads(path.read_text())

        ts = doc.get("timestamp") or doc.get("date") or doc.get("created_at", "")
        if self._question_type != "SILENCE" and ts and ts > effective_ts:
            self._record(
                "get_invoice",
                {"invoice_id": invoice_id},
                [],
                t0,
                timestamp_applied=effective_ts,
            )
            return {}

        results = self._record(
            "get_invoice",
            {"invoice_id": invoice_id},
            [doc] if doc else [],
            t0,
            timestamp_applied=effective_ts,
        )
        return results[0] if results else {}

    def get_nps_response(self, account_name: str) -> dict:
        t0 = time.time()
        effective_ts = self._gate_ts()

        fname = (
            account_name.lower().replace(" ", "_").replace(".", "").replace(",", "")
            + ".json"
        )
        path = BASE / "nps" / "responses" / fname
        if not path.exists():
            self._record(
                "get_nps_response",
                {"account_name": account_name},
                [],
                t0,
                timestamp_applied=effective_ts,
            )
            return {}

        doc = json.loads(path.read_text())

        ts = doc.get("timestamp") or doc.get("date") or doc.get("created_at", "")
        if self._question_type != "SILENCE" and ts and ts > effective_ts:
            self._record(
                "get_nps_response",
                {"account_name": account_name},
                [],
                t0,
                timestamp_applied=effective_ts,
            )
            return {}

        results = self._record(
            "get_nps_response",
            {"account_name": account_name},
            [doc] if doc else [],
            t0,
            timestamp_applied=effective_ts,
        )
        return results[0] if results else {}

    def get_events_for_day(
        self, day: int, event_type: Optional[str] = None
    ) -> List[dict]:
        t0 = time.time()

        if self._question_type == "SILENCE":
            gate_day = self._question.get("trigger_day", 30)
            gate_ts = (_SIM_START + timedelta(days=gate_day)).isoformat()
        else:
            gate_day = (datetime.fromisoformat(self._as_of_time) - _SIM_START).days + 1
            gate_ts = self._as_of_time

        if day > gate_day:
            logger.warning(
                f"[get_events_for_day] Day {day} requested but gate is Day {gate_day} — blocked"
            )
            return self._record(
                "get_events_for_day", {"day": day, "event_type": event_type}, [], t0
            )

        query: Dict = {
            "day": day,
        }

        print("timestamp")
        print(query)

        allowed_types: Set[str] = set()
        for subsystem in self._actor_subsystems:
            allowed_types.update(_SUBSYSTEM_EVENT_TYPES.get(subsystem, set()))

        if not self._actor_subsystems:
            allowed_types = set(KNOWN_EVENT_TYPES) - _INTERNAL_EVENT_TYPES

        if event_type:
            if event_type in _INTERNAL_EVENT_TYPES:
                logger.warning(
                    f"[get_events_for_day] Internal event type requested: {event_type}"
                )
                return self._record(
                    "get_events_for_day", {"day": day, "event_type": event_type}, [], t0
                )
            query["type"] = event_type
        else:
            query["type"] = {"$in": list(allowed_types)}

        docs = list(
            self._mem._db["events"].find(
                query,
                {
                    "_id": 0,
                    "event_id": 1,
                    "type": 1,
                    "day": 1,
                    "date": 1,
                    "timestamp": 1,
                    "actors": 1,
                    "summary": 1,
                    "artifact_ids": 1,
                    "tags": 1,
                },
            )
        )

        return self._record(
            "get_events_for_day", {"day": day, "event_type": event_type}, docs, t0
        )

    def search_artifacts(
        self,
        query: str,
        doc_type: str,
        actor: Optional[str] = None,
        after_day: Optional[int] = None,
        limit: int = 6,
    ) -> List[dict]:
        t0 = time.time()
        effective_ts = self._gate_ts()
        MAX_SEARCH_LIMIT = 15
        limit = min(limit, MAX_SEARCH_LIMIT)

        exact_doc = self._mem._db["artifacts"].find_one(
            {"_id": query},
            {
                "embedding": 0,
            },
        )
        if exact_doc:
            ts_filter = {"timestamp": {"$lte": effective_ts}}
            if after_day is not None:
                floor_ts = _business_day_to_date(_SIM_START, after_day).isoformat()
                ts_filter["timestamp"]["$gte"] = floor_ts
            exact_doc_ts = exact_doc.get("timestamp", "")
            if exact_doc_ts <= effective_ts and (
                after_day is None or exact_doc_ts >= floor_ts
            ):
                return self._record(
                    "search_artifacts",
                    {
                        "query": query,
                        "doc_type": doc_type,
                        "actor": actor,
                        "after_day": after_day,
                    },
                    [exact_doc],
                    t0,
                    timestamp_applied=effective_ts,
                )
            return self._record(
                "search_artifacts",
                {
                    "query": query,
                    "doc_type": doc_type,
                    "actor": actor,
                    "after_day": after_day,
                },
                [],
                t0,
                timestamp_applied=effective_ts,
            )

        text_filter: dict = {
            "$text": {"$search": query},
            "timestamp": {"$lte": effective_ts},
        }
        if after_day is not None:
            floor_ts = _business_day_to_date(_SIM_START, after_day).isoformat()
            text_filter["timestamp"] = {
                "$gte": floor_ts,
                "$lte": effective_ts,
            }
        if doc_type:
            text_filter["type"] = doc_type
        if actor:
            text_filter["metadata.author"] = actor

        results = list(
            self._mem._db["artifacts"]
            .find(
                text_filter,
                {
                    "content": 0,
                    "embedding": 0,
                    "score": {"$meta": "textScore"},
                },
            )
            .sort([("score", {"$meta": "textScore"})])
            .limit(limit)
        )

        return self._record(
            "search_artifacts",
            {
                "query": query,
                "doc_type": doc_type,
                "actor": actor,
                "after_day": after_day,
            },
            results,
            t0,
            timestamp_applied=effective_ts,
        )


# ─────────────────────────────────────────────────────────────────────────────
# SCORERS
# ─────────────────────────────────────────────────────────────────────────────


class PerspectiveScorer:
    """
    Scores a PERSPECTIVE trajectory.

    Answer scoring:
      - Exact match on ground_truth.could_actor_have_known (boolean)
      - Partial credit for correctly identifying blocked_subsystems
      - Partial credit for citing in-cone evidence

    Trajectory scoring:
      - Epistemic discipline: fraction of tool calls that stayed within cone
      - Subsystem discipline: fraction of tool calls to accessible subsystems
      - Conclusion grounding: did the final answer cite in-cone artifacts?
    """

    def score_answer(
        self, final_answer: Dict, ground_truth: Dict
    ) -> Tuple[float, bool]:
        if not final_answer:
            return 0.0, False

        gt_bool = ground_truth.get("could_actor_have_known", False)

        # Extract boolean from agent answer
        agent_bool = self._extract_boolean(final_answer)
        if agent_bool is None:
            return 0.1, False

        correct = agent_bool == gt_bool
        if not correct:
            return 0.0, False

        # Partial credit for explaining the mechanism correctly
        score = 0.6  # base for correct boolean

        gt_blocked = set(ground_truth.get("blocked_subsystems", []))
        agent_blocked = set(final_answer.get("blocked_subsystems", []))
        if gt_blocked and agent_blocked:
            overlap = len(gt_blocked & agent_blocked) / len(gt_blocked)
            score += 0.2 * overlap

        gt_evidence = set(ground_truth.get("evidence_artifacts", []))
        agent_evidence = set(final_answer.get("evidence_artifacts", []))
        if gt_evidence and agent_evidence:
            overlap = len(gt_evidence & agent_evidence) / len(gt_evidence)
            score += 0.2 * overlap
        elif not gt_evidence:
            score += 0.2  # no evidence required — agent doesn't need to cite any

        return min(score, 1.0), True

    def score_trajectory(
        self,
        trajectory: AgentTrajectory,
        question: dict,
    ) -> PerspectiveTrajectoryScore:
        calls = trajectory.tool_calls
        if not calls:
            return PerspectiveTrajectoryScore(0, 0, 0, 0, 1.0, 0.0)

        n = len(calls)
        actor_cone_violations = sum(1 for c in calls if c.actor_gate_violation)
        subsystem_violations = sum(1 for c in calls if c.subsystem_violation)
        horizon_violations = sum(1 for c in calls if c.horizon_violation)
        dead_ends = sum(1 for c in calls if c.returned_empty)
        dead_ends_recovered = trajectory.dead_ends_recovered

        epistemic_discipline = 1.0 - (actor_cone_violations / n)
        subsystem_discipline = 1.0 - (subsystem_violations / n)
        horizon_discipline = 1.0 - (horizon_violations / n)

        # Conclusion grounding: did agent cite any in-cone artifact in final answer?
        actor_visible = set(question.get("actor_visible_artifacts", []))
        cited = set(trajectory.final_answer.get("evidence_artifacts", []))
        conclusion_grounding = 1.0 if (cited & actor_visible) else 0.5 if cited else 0.0

        dead_end_recovery = dead_ends_recovered / dead_ends if dead_ends > 0 else 1.0

        sim_days = CONFIG["simulation"].get("num_days", 60)
        drifts = [
            c.temporal_drift_days for c in calls if c.temporal_drift_days is not None
        ]
        temporal_precision = (
            1.0 - mean(abs(d) / sim_days for d in drifts) if drifts else 1.0
        )

        drift_violations = sum(1 for c in calls if c.temporal_drift_violation)
        temporal_drift_discipline = 1.0 - (drift_violations / n)

        composite = (
            0.30 * epistemic_discipline
            + 0.25 * subsystem_discipline
            + 0.20 * conclusion_grounding
            + 0.10 * horizon_discipline
            + 0.05 * temporal_precision
            + 0.05 * temporal_drift_discipline
            + 0.05 * dead_end_recovery
        )

        return PerspectiveTrajectoryScore(
            epistemic_discipline=round(epistemic_discipline, 4),
            subsystem_discipline=round(subsystem_discipline, 4),
            horizon_discipline=round(horizon_discipline, 4),
            temporal_precision=round(temporal_precision, 4),
            temporal_drift_discipline=round(temporal_drift_discipline, 4),  # new
            conclusion_grounding=round(conclusion_grounding, 4),
            dead_end_recovery=round(dead_end_recovery, 4),
            composite=round(composite, 4),
        )

    def _extract_boolean(self, answer: Dict) -> Optional[bool]:
        for key in ("could_actor_have_known", "answer", "result", "known", "visible"):
            val = answer.get(key)
            if isinstance(val, bool):
                return val
            if isinstance(val, str):
                if val.lower() in ("true", "yes", "1"):
                    return True
                if val.lower() in ("false", "no", "0"):
                    return False

        # Try to find a boolean in free-text reasoning.
        # IMPORTANT: check negative phrases FIRST and use full-phrase matching so
        # that "did not have access" cannot shadow the later "had access" check —
        # both would match under simple substring logic since "had access" is a
        # substring of "did not have access". We resolve this by checking the
        # negative patterns against the exact negated forms only, not as substrings
        # of longer phrases.
        reasoning = str(answer.get("reasoning", answer.get("explanation", ""))).lower()

        # Negative indicators — listed as complete phrases, no substring ambiguity
        _NEGATIVE_PHRASES = (
            "could not have known",
            "did not have access",
            "does not have access",
            "had no access",
            "was not visible",
            "not visible to",
            "outside their visibility",
            "outside their access",
            "outside their cone",
            "not in their subsystem",
            "blocked from",
            "no access to",
        )

        _POSITIVE_PHRASES = (
            "could have known",
            "would have known",
            "did have access",
            "has access to",
            "was visible to",
            "visible to this actor",
            "within their visibility",
            "within their access",
            "within their cone",
            "in their subsystem",
            "had direct access",
            "had full access",
        )

        for phrase in _NEGATIVE_PHRASES:
            if phrase in reasoning:
                return False

        for phrase in _POSITIVE_PHRASES:
            if phrase in reasoning:
                return True

        return None


class CounterfactualScorer:
    """
    Scores a COUNTERFACTUAL trajectory.

    Answer scoring:
      - outcome_changed: boolean match (0.4)
      - mechanism: correct link_type identified (0.35)
      - actors: at least one correct actor identified (0.25)

    Trajectory scoring:
      - cause_identified: agent retrieved the cause event
      - effect_identified: agent retrieved the effect event
      - mechanism_correct: agent named the right link_type
      - causal_chain_complete: agent traversed cause → effect in order
    """

    _MECHANISM_ALIASES = {
        "involves_gap": {
            "knowledge gap",
            "gap",
            "undocumented",
            "missing documentation",
            "knowledge_gap",
        },
        "recurrence_of": {
            "recurrence",
            "repeat incident",
            "recurred",
            "same issue",
            "recurring",
        },
        "spawned_doc": {
            "spawned",
            "documentation",
            "confluence",
            "design discussion",
            "produced doc",
        },
        "email_dropped": {
            "dropped",
            "unactioned",
            "routing failure",
            "missed email",
            "no response",
        },
        "sf_ownership_lapsed": {
            "ownership lapsed",
            "crm gap",
            "salesforce",
            "account owner",
            "orphaned",
        },
        "zd_escalation_source": {
            "zendesk",
            "support ticket",
            "escalated from",
            "zd escalation",
        },
        "blocker_flagged": {
            "blocker",
            "blocked",
            "delay",
            "progress",
            "technical blocker",
            "blocker_flagged",
        },
        "incident_coordination": {
            "coordination",
            "external contact",
            "external party",
            "incident_coordination",
            "coordinated with",
        },
        "departure_reassignment": {
            "reassignment",
            "departed",
            "departure",
            "reassigned",
            "departure_reassignment",
            "not reassigned",
        },
    }

    def score_answer(
        self, final_answer: Dict, ground_truth: Dict
    ) -> Tuple[float, bool]:
        if not final_answer:
            return 0.0, False

        score = 0.0
        gt_outcome = ground_truth.get("outcome_changed", True)
        agent_outcome = self._extract_boolean(final_answer, "outcome_changed")

        if agent_outcome is None:
            return 0.0, False
        if agent_outcome == gt_outcome:
            score += 0.4

        # Mechanism match
        gt_mechanism = ground_truth.get("causal_mechanism", "")
        agent_mechanism = str(
            final_answer.get("mechanism", final_answer.get("causal_mechanism", ""))
        ).lower()
        aliases = self._MECHANISM_ALIASES.get(gt_mechanism, {gt_mechanism})
        if any(alias in agent_mechanism for alias in aliases):
            score += 0.35

        # Actor match
        gt_actors = set(ground_truth.get("actors", []))
        agent_actors_raw = final_answer.get(
            "actors", final_answer.get("involved_actors", [])
        )
        agent_actors = (
            set(agent_actors_raw) if isinstance(agent_actors_raw, list) else set()
        )
        if gt_actors and agent_actors and (gt_actors & agent_actors):
            score += 0.25
        elif not gt_actors:
            score += 0.25

        is_correct = score >= 0.75
        return round(min(score, 1.0), 4), is_correct

    def score_trajectory(
        self,
        trajectory: AgentTrajectory,
        question: dict,
        ground_truth: Dict,
    ) -> CounterfactualTrajectoryScore:
        calls = trajectory.tool_calls
        if not calls:
            return CounterfactualTrajectoryScore(0, 0, 0, 0, 1.0, 0.0)

        n = len(calls)
        retrieved_ids = set()
        for call in calls:
            retrieved_ids.update(call.result_ids)

        # Use artifact IDs from evidence_chain_artifacts, not synthetic event IDs.
        # Synthetic event IDs (e.g. "evt_incident_opened_5_IT-108_alex") are internal
        # keys that never appear in MongoDB documents. Agents retrieve documents by
        # their actual artifact IDs (e.g. "IT-108"), so we must match on those instead.
        evidence_artifacts = ground_truth.get("evidence_chain_artifacts", {})
        cause_artifacts = set(evidence_artifacts.get("cause", []))
        effect_artifacts = set(evidence_artifacts.get("effect", []))

        cause_identified = (
            1.0 if (cause_artifacts and cause_artifacts & retrieved_ids) else 0.0
        )
        effect_identified = (
            1.0 if (effect_artifacts and effect_artifacts & retrieved_ids) else 0.0
        )

        # Mechanism: did agent use keyword in its tool calls or final answer?
        gt_mechanism = ground_truth.get("causal_mechanism", "")
        aliases = self._MECHANISM_ALIASES.get(gt_mechanism, {gt_mechanism})
        agent_text = " ".join(
            [str(c.arguments) for c in calls] + [str(trajectory.final_answer)]
        ).lower()
        mechanism_correct = (
            1.0 if any(alias in agent_text for alias in aliases) else 0.0
        )

        # Causal chain: did agent retrieve a cause artifact before an effect artifact?
        # Since we no longer have single cause_id/effect_id to index into call.result_ids,
        # we find the FIRST call that returned any cause artifact and the FIRST that
        # returned any effect artifact, then check ordering.
        cause_call_idx = next(
            (i for i, c in enumerate(calls) if cause_artifacts & set(c.result_ids)),
            None,
        )
        effect_call_idx = next(
            (i for i, c in enumerate(calls) if effect_artifacts & set(c.result_ids)),
            None,
        )
        causal_chain_complete = (
            1.0
            if (
                cause_call_idx is not None
                and effect_call_idx is not None
                and cause_call_idx <= effect_call_idx
            )
            else 0.5
            if (cause_call_idx is not None or effect_call_idx is not None)
            else 0.0
        )

        horizon_violations = sum(1 for c in calls if c.horizon_violation)
        horizon_discipline = 1.0 - (horizon_violations / n)

        composite = (
            0.25 * cause_identified
            + 0.25 * effect_identified
            + 0.25 * mechanism_correct
            + 0.15 * causal_chain_complete
            + 0.10 * horizon_discipline
        )

        return CounterfactualTrajectoryScore(
            cause_identified=cause_identified,
            effect_identified=effect_identified,
            mechanism_correct=mechanism_correct,
            causal_chain_complete=causal_chain_complete,
            horizon_discipline=round(horizon_discipline, 4),
            composite=round(composite, 4),
        )

    def _extract_boolean(self, answer: Dict, key: str) -> Optional[bool]:
        val = answer.get(key)
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            if val.lower() in ("true", "yes"):
                return True
            if val.lower() in ("false", "no"):
                return False

        # Inspect free-text reasoning for outcome_changed signal.
        #
        # The original had two bugs:
        #
        # 1. "would not" was mapped to True (outcome DID change — the thing would
        #    NOT have happened). This is semantically correct for counterfactuals
        #    ("the incident would not have occurred") but the bare phrase is too
        #    short — "this would not be my first choice" would also match.
        #    Replaced with longer, unambiguous anchors.
        #
        # 2. "would have prevented" → True is correct but collides with
        #    "nothing would have prevented" → should be False.
        #    Fixed by checking the negated form first.
        #
        # 3. "no change" → False is a two-word phrase that can appear in
        #    unrelated contexts ("no change in personnel"). Replaced with
        #    longer anchors.
        #
        # Strategy: check negated/False-indicating phrases FIRST (longer, more
        # specific), then check True-indicating phrases that are phrased to not
        # overlap with any negated form above.

        reasoning = str(answer.get("reasoning", "")).lower()

        # False indicators — outcome did NOT change (removing cause = no difference)
        _OUTCOME_UNCHANGED = (
            "would still have occurred",
            "would have happened regardless",
            "outcome would not have changed",
            "outcome would be the same",
            "would not have been prevented",
            "nothing would have prevented",
            "no change in outcome",
            "would have proceeded regardless",
            "result would be unchanged",
            "would still have taken place",
        )

        # True indicators — outcome WOULD change (removing cause = different result)
        # Phrased to not be substrings of any _OUTCOME_UNCHANGED phrase above.
        _OUTCOME_CHANGED = (
            "would not have occurred",
            "would have been prevented",
            "would have been avoided",
            "outcome would have changed",
            "would have changed the outcome",
            "would have been diagnosed faster",
            "would not have escalated",
            "would have been resolved",
            "would not have happened",
            "would have been different",
            "causal chain would have been broken",
        )

        # Check False indicators first — they are more specific and longer
        for phrase in _OUTCOME_UNCHANGED:
            if phrase in reasoning:
                return False

        # Check True indicators second
        for phrase in _OUTCOME_CHANGED:
            if phrase in reasoning:
                return True

        return None


class SilenceScorer:
    """
    Scores a SILENCE trajectory.

    The key insight: absence is only meaningful if the agent searched the
    right places. A lucky "no" without evidence of search is scored as a
    trajectory failure.

    Answer scoring:
      - Boolean correct (did agent conclude the artifact does not exist): 1.0
      - If agent concluded "yes" (artifact exists): 0.0

    Trajectory scoring:
      - search_space_coverage: fraction of expected_search_space the agent checked
      - correct_absence_conclusion: did agent say "does not exist" explicitly?
      - premature_conclusion: penalty if agent concluded before searching
    """

    def score_answer(
        self, final_answer: Dict, ground_truth: Dict
    ) -> Tuple[float, bool]:
        if not final_answer:
            return 0.0, False

        gt_answer = ground_truth.get("answer", False)  # Always False for SILENCE
        agent_answer = self._extract_absence_conclusion(final_answer)

        if agent_answer is None:
            return 0.1, False

        correct = agent_answer == gt_answer
        return (1.0, True) if correct else (0.0, False)

    def score_trajectory(
        self,
        trajectory: AgentTrajectory,
        question: dict,
    ) -> SilenceTrajectoryScore:
        calls = trajectory.tool_calls
        expected_space = set(question.get("expected_search_space", []))
        if not calls:
            return SilenceTrajectoryScore(0.0, 0.0, 0.0, 1.0, 0.0)

        n = len(calls)

        # What did the agent search?
        searched_ids: Set[str] = set()
        searched_tool_args: List[str] = []
        for call in calls:
            searched_ids.update(call.result_ids)
            searched_tool_args.append(str(call.arguments).lower())

        def _normalize_search_term(s: str) -> str:
            """
            Extract the terminal component of a path-style search space entry.
            e.g. "confluence/postmortems/IT-108" → "it-108"
                 "slack/channels/incidents"       → "incidents"
                 "IT-108"                         → "it-108"
            """
            return s.strip("/").split("/")[-1].lower()

        normalized_expected: Dict[str, str] = {
            _normalize_search_term(e): e for e in expected_space
        }

        # Also normalize all tool arg strings and result IDs for matching.
        normalized_tool_args: List[str] = [arg.lower() for arg in searched_tool_args]
        normalized_result_ids: Set[str] = {
            _normalize_search_term(rid) for rid in searched_ids
        }

        covered = set()
        for norm_term, original in normalized_expected.items():
            # Primary match: terminal component appears anywhere in a tool arg string.
            # This catches {"page_id": "IT-108"} matching "confluence/postmortems/IT-108"
            # and {"query": "incidents"} matching "slack/channels/incidents".
            if any(norm_term in arg for arg in normalized_tool_args):
                covered.add(original)

            # Secondary match: terminal component matches a normalized result ID.
            # This catches cases where the agent retrieved the document directly
            # and its ID is the terminal path component.
            elif norm_term in normalized_result_ids:
                covered.add(original)

            # Tertiary match: the full original path appears verbatim in a tool arg.
            # Preserves the original behaviour for agents that do pass full paths.
            elif any(original.lower() in arg for arg in normalized_tool_args):
                covered.add(original)

        search_space_coverage = (
            len(covered) / len(expected_space) if expected_space else 1.0
        )

        conclusion_text = str(trajectory.final_answer.get("reasoning", "")).lower()
        conclusion_text += str(trajectory.final_answer.get("answer", "")).lower()
        explicit_negative = any(
            phrase in conclusion_text
            for phrase in (
                "does not exist",
                "was not created",
                "no postmortem",
                "not found",
                "could not find",
                "no record",
                "never created",
                "not in the corpus",
                "no evidence",
                "not present",
                "absent",
            )
        )
        correct_absence_conclusion = 1.0 if explicit_negative else 0.3

        # Premature conclusion: did agent conclude before searching?
        # Heuristic: if the final answer came after fewer than 2 tool calls, penalize
        premature_conclusion = 1.0 - max(0.0, min(1.0, (n - 1) / 3))

        horizon_violations = sum(1 for c in calls if c.horizon_violation)
        horizon_discipline = (
            1.0  # SILENCE has no temporal gate, so no violations possible
        )

        composite = (
            0.50 * search_space_coverage
            + 0.30 * correct_absence_conclusion
            - 0.10 * (1.0 - premature_conclusion)  # penalty for rushing
            + 0.10 * horizon_discipline
        )

        return SilenceTrajectoryScore(
            search_space_coverage=round(search_space_coverage, 4),
            correct_absence_conclusion=round(correct_absence_conclusion, 4),
            premature_conclusion=round(premature_conclusion, 4),
            horizon_discipline=round(horizon_discipline, 4),
            composite=round(max(0.0, composite), 4),
        )

    def _extract_absence_conclusion(self, answer: Dict) -> Optional[bool]:
        """True = artifact exists, False = artifact does not exist."""
        val = answer.get("exists", answer.get("found", answer.get("answer")))
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            if val.lower() in ("true", "yes", "exists", "found"):
                return True
            if val.lower() in ("false", "no", "not found", "does not exist", "absent"):
                return False

        reasoning = str(answer.get("reasoning", "")).lower()

        # False indicators — artifact does NOT exist
        # These are checked first and are long enough to be unambiguous.
        _ABSENCE_PHRASES = (
            "does not exist",
            "did not exist",
            "was not created",
            "has not been created",
            "no record exists",
            "no record was found",
            "could not be found",
            "could not find",
            "was not found",
            "is not present",
            "was not present",
            "no evidence of",
            "never created",
            "not in the corpus",
            "absent from",
            "no postmortem",
            "no ticket was",
            "no confluence page",
        )

        # True indicators — artifact DOES exist
        # Reworded so none are substrings of any _ABSENCE_PHRASES entry above.
        _PRESENCE_PHRASES = (
            "artifact exists",
            "document exists",
            "ticket exists",
            "page exists",
            "record exists",
            "was successfully created",
            "has been created",
            "is present in",
            "appears in the corpus",
            "was located",
            "has been found",
            "confirmed to exist",
            "did find",
        )

        # Check absence first — these are longer and more specific
        for phrase in _ABSENCE_PHRASES:
            if phrase in reasoning:
                return False

        # Check presence second — phrased to not overlap with any absence phrase
        for phrase in _PRESENCE_PHRASES:
            if phrase in reasoning:
                return True

        return None


# ─────────────────────────────────────────────────────────────────────────────
# AGENT RUNNER
# ─────────────────────────────────────────────────────────────────────────────


class AgenticEvalRunner:
    """
    Runs the agent on each question and scores the result.

    For each question:
    1. Sets up the gated tool layer appropriate for the track
    2. Runs the agent with the typed tool surface
    3. Collects the trajectory
    4. Scores answer + trajectory with the track-specific scorer
    5. Combines scores with track-specific weights
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        max_steps: int = 5,
        ungated: bool = False,
        zero_shot: bool = False,
    ):
        self._model = model
        self._max_steps = max_steps

        # --ungated: all actor/subsystem gates disabled regardless of question type.
        # Establishes the "god-mode" information ceiling for the Epistemic Tax.
        self._ungated = ungated

        # --zero-shot: agent receives no tools at all (no corpus access).
        # Establishes the hallucination / prior-knowledge floor.
        # Mutually exclusive with --ungated; zero_shot takes precedence if both set.
        self._zero_shot = zero_shot

        from flow import build_llm
        from memory import Memory

        self._mem = Memory()
        self._llm = build_llm("worker")

        self._perspective_scorer = PerspectiveScorer()
        self._counterfactual_scorer = CounterfactualScorer()
        self._silence_scorer = SilenceScorer()

    def run(
        self,
        questions_path: Path,
        out_path: Path,
        question_types: Optional[List[str]] = None,
        max_questions: Optional[int] = None,
    ) -> None:
        with open(questions_path) as f:
            data = json.load(f)

        questions = data["questions"]
        if question_types:
            questions = [q for q in questions if q["question_type"] in question_types]
        if max_questions:
            questions = questions[:max_questions]

        logger.info(
            f"Running agentic eval on {len(questions)} questions "
            f"(model={self._model}, max_steps={self._max_steps})"
        )

        results: List[EvalResult] = []
        per_question: List[dict] = []

        for i, question in enumerate(questions):
            qtype = question["question_type"]
            logger.info(
                f"[{i + 1}/{len(questions)}] {qtype} — {question['question_id']}"
            )

            try:
                result = self._run_question(question)
            except Exception as exc:
                logger.error(f"  Failed: {exc}")
                result = EvalResult(
                    question_id=question["question_id"],
                    question_type=qtype,
                    difficulty=question.get("difficulty", "unknown"),
                    answer_score=0.0,
                    answer_correct=False,
                    trajectory_score=0.0,
                    combined_score=0.0,
                    failure_reason=str(exc),
                    tool_call_count=0,
                    meta={"error": str(exc)},
                )

            results.append(result)
            per_question.append(result.to_dict())
            logger.info(
                f"  answer={result.answer_score:.3f} "
                f"trajectory={result.trajectory_score:.3f} "
                f"combined={result.combined_score:.3f} "
                f"tools={result.tool_call_count}"
            )

        summary = self._aggregate(results)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        output = {
            "meta": {
                "model": self._model,
                "max_steps": self._max_steps,
                "n_questions": len(results),
                "track_weights": _TRACK_WEIGHTS,
            },
            "summary": summary,
            "per_question": per_question,
        }
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2, default=str)

        logger.info(f"Results written to {out_path}")
        logger.info(
            f"Overall — answer: {summary['overall']['answer_score']:.3f} "
            f"trajectory: {summary['overall']['trajectory_score']:.3f} "
            f"combined: {summary['overall']['combined_score']:.3f}"
            + (
                f" | violation_adjusted: "
                f"{summary['overall'].get('violation_adjusted_combined_score', 'n/a')}"
            )
        )

    def _run_question(self, question: dict) -> EvalResult:
        qtype = question["question_type"]
        ground_truth = question["ground_truth"]

        # Set up gated tools.
        # --ungated: strip all actor/subsystem gates by passing None for both,
        # regardless of question type. Temporal gate still applies.
        # --zero-shot: GatedTools is still constructed (for consistent call
        # logging infrastructure) but _tool_list() returns [] so the agent
        # never actually invokes any tool.
        as_of_time = self._infer_as_of_time(question)

        if self._ungated:
            actor_visible = None
            actor_subsystems = None
        else:
            actor_visible = (
                set(question.get("actor_visible_artifacts", []))
                if qtype == "PERSPECTIVE"
                else None
            )
            actor_subsystems = (
                set(question.get("subsystem_access", []))
                if qtype == "PERSPECTIVE"
                else None
            )

        tools = GatedTools(
            mem=self._mem,
            question=question,
            as_of_time=as_of_time,
            actor_visible_artifacts=actor_visible,
            actor_subsystem_access=actor_subsystems,
        )

        # Run agent
        trajectory = self._run_agent(question, tools)

        # Score
        if qtype == "PERSPECTIVE":
            answer_score, answer_correct = self._perspective_scorer.score_answer(
                trajectory.final_answer, ground_truth
            )
            traj = self._perspective_scorer.score_trajectory(trajectory, question)
            traj_score = traj.composite
            traj_detail = asdict(traj)

        elif qtype == "COUNTERFACTUAL":
            answer_score, answer_correct = self._counterfactual_scorer.score_answer(
                trajectory.final_answer, ground_truth
            )
            traj = self._counterfactual_scorer.score_trajectory(
                trajectory, question, ground_truth
            )
            traj_score = traj.composite
            traj_detail = asdict(traj)

        elif qtype == "SILENCE":
            answer_score, answer_correct = self._silence_scorer.score_answer(
                trajectory.final_answer, ground_truth
            )
            traj = self._silence_scorer.score_trajectory(trajectory, question)
            traj_score = traj.composite
            traj_detail = asdict(traj)

        else:
            raise ValueError(f"Unknown question type: {qtype}")

        weights = _TRACK_WEIGHTS[qtype]
        combined = weights["answer"] * answer_score + weights["trajectory"] * traj_score

        return EvalResult(
            question_id=question["question_id"],
            question_type=qtype,
            difficulty=question.get("difficulty", "unknown"),
            answer_score=round(answer_score, 4),
            answer_correct=answer_correct,
            trajectory_score=round(traj_score, 4),
            combined_score=round(combined, 4),
            failure_reason=None,
            tool_call_count=len(trajectory.tool_calls),
            meta={
                "model": self._model,
                "eval_mode": (
                    "zero_shot"
                    if self._zero_shot
                    else "ungated"
                    if self._ungated
                    else "gated"
                ),
                "as_of_time": as_of_time,
                "trajectory_detail": traj_detail,
                "horizon_violations": trajectory.horizon_violations,
                "actor_gate_violations": trajectory.actor_gate_violations,
                "subsystem_violations": trajectory.subsystem_violations,
                "dead_ends_hit": trajectory.dead_ends_hit,
                "dead_ends_recovered": trajectory.dead_ends_recovered,
                "total_latency_ms": round(trajectory.total_latency_ms, 1),
                "tool_calls": [asdict(tc) for tc in trajectory.tool_calls],
                "final_answer": trajectory.final_answer,
            },
        )

    def _run_agent(self, question: dict, tools: GatedTools) -> AgentTrajectory:
        """
        Runs the agent against the question using the gated tool surface.
        Returns a populated AgentTrajectory.

        The agent is given a structured output format so answer extraction
        is reliable across all three tracks.
        """
        from agent_factory import make_agent
        from crewai import Crew, Task

        qtype = question["question_type"]
        trajectory = AgentTrajectory(
            question_id=question["question_id"],
            question_type=qtype,
        )

        CAUSAL_LINK_TAXONOMY = {
            "involves_gap": "incident ← knowledge gap (information was missing/undocumented)",
            "recurrence_of": "incident ← prior unresolved incident (root cause was known but not fixed)",
            "spawned_doc": "confluence ← design discussion (documentation resulted from a specific meeting)",
            "email_dropped": "communication failure ← routing gap",
            "sf_ownership_lapsed": "CRM gap ← employee departure",
            "zd_escalation_source": "incident ← support ticket escalation",
            "blocker_flagged": "blocker → delayed progress",
            "incident_coordination": "incident → external contact",
            "departure_reassignment": "departure → ticket/escalation shift",
            "assignment_domain_mismatch": "planning mismatch → knowledge gap → incident",
        }

        taxonomy_str = "\n".join(
            [f"- {k}: {v}" for k, v in CAUSAL_LINK_TAXONOMY.items()]
        )
        allowed_links = ", ".join(CAUSAL_LINK_TAXONOMY.keys())

        # Build output schema based on track
        output_schema = {
            "PERSPECTIVE": """{
            "could_actor_have_known": <true|false>,
            "reasoning": "<explanation>",
            "evidence_artifacts": ["<artifact_id>", ...],
            "blocked_subsystems": ["<subsystem>", ...]
            }""",
            "COUNTERFACTUAL": f"""{{
            "outcome_changed": <true|false>,
            "mechanism": "<detailed causal mechanism description>",
            "causal_mechanism": "<MUST be one of: {allowed_links}>",
            "actors": ["<actor_name>", ...],
            "reasoning": "<explanation>"
            }}""",
            "SILENCE": """{
            "exists": <true|false>,
            "answer": "<yes|no>",
            "reasoning": "<explanation of what you searched and what you found>"
            }""",
        }[qtype]

        _SEARCH_SPACE_HINTS = {
            "confluence/general": "search Confluence general pages",
            "confluence/retros": "search Confluence retrospectives",
            "zendesk/queue": "search Zendesk tickets",
            "zendesk/tickets": "search Zendesk tickets",
            "zendesk/escalations": "check Zendesk escalations",
            "slack/channels/engineering": "check the engineering Slack channel",
            "slack/channels/digital-hq": "check the digital-hq Slack channel",
            "slack/channels/incidents": "check the incidents Slack channel",
            "slack/channels/general": "check the general Slack channel",
            "slack/channels/support": "check the support Slack channel",
            "zoom/transcripts": "search Zoom transcripts",
            "git/merged-prs": "check merged pull requests",
            "salesforce/opportunities": "search Salesforce opportunities",
            "salesforce/accounts": "search Salesforce accounts",
            "jira/incidents": "search Jira incident tickets",
            "jira/reassignments": "check Jira for ticket reassignments",
        }

        space = question.get("expected_search_space", [])[:5]
        hints = []
        for entry in space:
            hint = next(
                (v for k, v in _SEARCH_SPACE_HINTS.items() if entry.startswith(k)), None
            )
            if hint:
                hints.append(hint)
            elif entry.startswith("export/emails/"):
                continue
            elif entry.startswith("export/"):
                continue
            else:
                if entry.startswith("ext_email_") or entry.startswith("EMAIL-"):
                    hints.append(f"call get_email with email_id='{entry}'")
                elif entry.startswith("CONF-"):
                    hints.append(f"call get_confluence_page with page_id='{entry}'")
                elif entry.startswith("slack_"):
                    hints.append(f"call get_slack_thread with thread_id='{entry}'")
                elif entry.startswith("ENG-") or entry.startswith("IT-"):
                    hints.append(f"call get_ticket with ticket_id='{entry}'")
                elif entry.startswith("ZD-"):
                    hints.append(f"call get_zd_ticket with ticket_id='{entry}'")
                elif entry.startswith("PR-"):
                    hints.append(f"call get_pr with pr_id='{entry}'")
                else:
                    hints.append(f"search for artifact '{entry}'")

        constraint_note = {
            "PERSPECTIVE": (
                f"\n\nIMPORTANT: You are answering from the perspective of {question.get('actor', 'the actor')} "
                f"as of Day {question.get('as_of_day', '?')}. "
                f"This actor only has access to: {', '.join(question.get('subsystem_access', []))}. "
                f"You must not use information from systems outside this list. "
                f"Accessing artifacts outside the actor's visibility cone is a violation."
            ),
            "COUNTERFACTUAL": (
                "\n\nIMPORTANT: This is a counterfactual question. You must identify the explicit "
                "causal link in the data — do not speculate. \n\n"
                "You MUST categorize the link using one of the following labels:\n"
                f"{taxonomy_str}\n\n"
                "Find the cause event and the effect event, then determine whether "
                "removing the cause would have changed the effect."
            ),
            "SILENCE": (
                "\n\nIMPORTANT: This is an absence question. You must search the corpus "
                "thoroughly before concluding absence. Do not guess. "
                "Show your work in the reasoning field — explain what you searched and what you found."
            ),
        }[qtype]

        agent = make_agent(
            role="Enterprise Knowledge Analyst",
            goal="Reason carefully over corporate documents to answer complex questions.",
            backstory=(
                "You are an expert analyst evaluating enterprise AI systems. You reason "
                "carefully, cite evidence, stay within stated constraints, and never guess."
            ),
            llm=self._llm,
            tools=self._tool_list(tools),
            max_iter=self._max_steps,
        )

        task = Task(
            description=(
                f"{question['question_text']}"
                f"{constraint_note}"
                f"\n\nRespond ONLY with a JSON object matching this schema:\n{output_schema}"
            ),
            expected_output="A JSON object matching the schema above. No preamble.",
            agent=agent,
        )

        t_start = time.time()
        try:
            raw_output = Crew(
                agents=[agent],
                tasks=[task],
                verbose=True,
                output_log_file="simulation.log",
            ).kickoff()
            if hasattr(raw_output, "raw"):
                raw = str(raw_output.raw).strip()
            elif isinstance(raw_output, list):
                text_blocks = [
                    b.get("text", "")
                    for b in raw_output
                    if isinstance(b, dict) and b.get("type") == "text"
                ]
                raw = " ".join(text_blocks).strip()
            else:
                raw = str(raw_output).strip()
            final_answer = self._parse_structured_answer(raw)
        except Exception as exc:
            logger.warning(f"  Agent error: {exc}")
            final_answer = {}

        trajectory.total_latency_ms = (time.time() - t_start) * 1000
        trajectory.tool_calls = list(tools.call_log)
        trajectory.final_answer = final_answer
        trajectory.horizon_violations = sum(
            1 for c in trajectory.tool_calls if c.horizon_violation
        )
        trajectory.actor_gate_violations = sum(
            1 for c in trajectory.tool_calls if c.actor_gate_violation
        )
        trajectory.subsystem_violations = sum(
            1 for c in trajectory.tool_calls if c.subsystem_violation
        )
        trajectory.dead_ends_hit = sum(
            1 for c in trajectory.tool_calls if c.returned_empty
        )

        # Dead end recovery: count cases where agent made a successful call after a dead end
        for i, call in enumerate(trajectory.tool_calls):
            if call.returned_empty and i + 1 < len(trajectory.tool_calls):
                if not trajectory.tool_calls[i + 1].returned_empty:
                    trajectory.dead_ends_recovered += 1

        return trajectory

    def _tool_list(self, tools: GatedTools) -> List:
        if self._zero_shot:
            return []

        from crewai.tools import BaseTool
        from pydantic import BaseModel, Field

        class TicketInput(BaseModel):
            ticket_id: str = Field(
                ..., description="Jira ticket ID, e.g. 'ENG-42' or 'ORG-108'"
            )

        class GetTicket(BaseTool):
            name: str = "get_ticket"
            description: str = "Retrieve a Jira ticket by ID."
            args_schema: type[BaseModel] = TicketInput
            _tools: GatedTools

            def _run(self, ticket_id: str) -> dict:
                return tools.get_ticket(ticket_id)

        class ConfluenceInput(BaseModel):
            page_id: str = Field(
                ..., description="Confluence page ID, e.g. 'CONF-ENG-007'"
            )

        class GetConfluencePage(BaseTool):
            name: str = "get_confluence_page"
            description: str = "Retrieve a Confluence page by ID."
            args_schema: type[BaseModel] = ConfluenceInput

            def _run(self, page_id: str) -> dict:
                return tools.get_confluence_page(page_id)

        class SlackInput(BaseModel):
            thread_id: str = Field(
                ...,
                description="Slack thread ID, e.g. 'slack_dm_liam_sanjay_2026-03-02T13:37:00'",
            )

        class GetSlackThread(BaseTool):
            name: str = "get_slack_thread"
            description: str = "Retrieve a Slack thread by ID."
            args_schema: type[BaseModel] = SlackInput

            def _run(self, thread_id: str) -> list:
                return tools.get_slack_thread(thread_id)

        class EmailInput(BaseModel):
            email_id: str = Field(
                ..., description="Email artifact ID, e.g. 'ext_email_name_1_1'"
            )

        class GetEmail(BaseTool):
            name: str = "get_email"
            description: str = "Retrieve an email by ID."
            args_schema: type[BaseModel] = EmailInput

            def _run(self, email_id: str) -> dict:
                return tools.get_email(email_id)

        class PRInput(BaseModel):
            pr_id: str = Field(..., description="Pull request ID, e.g. 'PR-88'")

        class GetPR(BaseTool):
            name: str = "get_pr"
            description: str = "Retrieve a pull request by ID."
            args_schema: type[BaseModel] = PRInput

            def _run(self, pr_id: str) -> dict:
                return tools.get_pr(pr_id)

        class ZDInput(BaseModel):
            ticket_id: str = Field(..., description="Zendesk ticket ID, e.g. 'ZD-55'")

        class GetZDTicket(BaseTool):
            name: str = "get_zd_ticket"
            description: str = "Retrieve a Zendesk support ticket by ID."
            args_schema: type[BaseModel] = ZDInput

            def _run(self, ticket_id: str) -> dict:
                return tools.get_zd_ticket(ticket_id)

        class SFOppInput(BaseModel):
            opp_id: str = Field(
                ..., description="Salesforce opportunity ID, e.g. 'SF-OPP-12'"
            )

        class GetSFOpportunity(BaseTool):
            name: str = "get_sf_opportunity"
            description: str = "Retrieve a Salesforce opportunity by ID."
            args_schema: type[BaseModel] = SFOppInput

            def _run(self, opp_id: str) -> dict:
                return tools.get_sf_opportunity(opp_id)

        class SFAccountInput(BaseModel):
            account_id: str = Field(
                ..., description="Salesforce account ID, e.g. 'SF-ACC-7'"
            )

        class GetSFAccount(BaseTool):
            name: str = "get_sf_account"
            description: str = "Retrieve a Salesforce account by ID."
            args_schema: type[BaseModel] = SFAccountInput

            def _run(self, account_id: str) -> dict:
                return tools.get_sf_account(account_id)

        class ZoomInput(BaseModel):
            transcript_id: str = Field(
                ..., description="Zoom transcript ID, e.g. 'ZOOM-2026-03-15'"
            )

        class GetZoomTranscript(BaseTool):
            name: str = "get_zoom_transcript"
            description: str = "Retrieve a Zoom meeting transcript by ID."
            args_schema: type[BaseModel] = ZoomInput

            def _run(self, transcript_id: str) -> dict:
                return tools.get_zoom_transcript(transcript_id)

        class DatadogInput(BaseModel):
            alert_id: str = Field(
                ..., description="Datadog alert ID, e.g. 'DD-ALERT-3'"
            )

        class GetDatadogAlert(BaseTool):
            name: str = "get_datadog_alert"
            description: str = "Retrieve a Datadog alert by ID."
            args_schema: type[BaseModel] = DatadogInput

            def _run(self, alert_id: str) -> dict:
                return tools.get_datadog_alert(alert_id)

        class InvoiceInput(BaseModel):
            invoice_id: str = Field(..., description="Invoice ID, e.g. 'INV-2026-001'")

        class GetInvoice(BaseTool):
            name: str = "get_invoice"
            description: str = "Retrieve an invoice by ID."
            args_schema: type[BaseModel] = InvoiceInput

            def _run(self, invoice_id: str) -> dict:
                return tools.get_invoice(invoice_id)

        class NPSInput(BaseModel):
            account_name: str = Field(..., description="Account name, e.g. 'Acme Corp'")

        class GetNPSResponse(BaseTool):
            name: str = "get_nps_response"
            description: str = "Retrieve an NPS survey response by account name."
            args_schema: type[BaseModel] = NPSInput

            def _run(self, account_name: str) -> dict:
                return tools.get_nps_response(account_name)

        class EventsInput(BaseModel):
            day: int = Field(..., description="Simulation day number, e.g. 1-30")
            event_type: str = Field(
                None, description="Optional event type filter, e.g. 'incident_opened'"
            )

        class GetEventsForDay(BaseTool):
            name: str = "get_events_for_day"
            description: str = "Retrieve all simulation events for a given day, optionally filtered by type."
            args_schema: type[BaseModel] = EventsInput

            def _run(self, day: int, event_type: str = None) -> list:
                return tools.get_events_for_day(day, event_type)

        class SearchInput(BaseModel):
            query: str = Field(..., description="Artifact ID or keyword to search for.")
            doc_type: str = Field(
                None,
                description="Filter by type, e.g. 'jira', 'confluence', 'slack', 'email', 'pr', 'zd_ticket', 'zoom'.",
            )
            actor: str = Field(None, description="Optional actor name to filter by")
            after_day: int = Field(
                None,
                description=(
                    "Only return artifacts created on or after this simulation day. "
                    "Use this when checking whether something was created in response "
                    "to a specific event."
                ),
            )

        class SearchArtifacts(BaseTool):
            name: str = "search_artifacts"
            description: str = "Search for information when you do not have a specific ID. You MUST provide a specific search string in the 'query' argument."
            args_schema: type[BaseModel] = SearchInput

            def _run(
                self,
                query: str,
                doc_type: str = "",
                actor: str = "",
                after_day: int = None,
            ) -> list:
                return tools.search_artifacts(
                    query, doc_type, actor=actor, after_day=after_day
                )

        return [
            GetTicket(),
            GetConfluencePage(),
            GetSlackThread(),
            GetEmail(),
            GetPR(),
            GetZDTicket(),
            GetSFOpportunity(),
            GetSFAccount(),
            GetZoomTranscript(),
            GetDatadogAlert(),
            GetInvoice(),
            GetNPSResponse(),
            GetEventsForDay(),
            SearchArtifacts(),
        ]

    def _parse_structured_answer(self, raw: str) -> Dict:
        """Extract JSON from agent response. Strips markdown fences."""
        text = raw.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON object in response
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        return {"raw_response": raw}

    def _infer_as_of_time(self, question: dict) -> str:
        qtype = question.get("question_type", "")
        if qtype == "SILENCE":
            events = self._mem.get_event_log(from_db=True)
            max_day = max((e.day for e in events), default=1)
            return _business_day_to_date(_SIM_START, max_day).isoformat()
        if qtype == "PERSPECTIVE":
            return question.get("as_of_time", datetime.now().isoformat())
        if qtype == "COUNTERFACTUAL":
            # Use effect event timestamp
            effect_id = question.get("ground_truth", {}).get("effect_event_id")
            if effect_id:
                try:
                    ev = self._mem._db["events"].find_one({"event_id": effect_id})
                    if ev and ev.get("timestamp"):
                        return str(ev["timestamp"])
                except Exception:
                    pass
        day = question.get("day", question.get("event_day", 1))
        return _business_day_to_date(_SIM_START, day).isoformat()

    def _aggregate(self, results: List[EvalResult]) -> dict:
        def mean(vals):
            return round(sum(vals) / len(vals), 4) if vals else 0.0

        # ── Violation-adjusted scoring ────────────────────────────────────────
        # violation_rate    = total_actor_gate_violations / total_tool_calls
        # compliance_factor = max(0, 1 − violation_rate) ** _VIOLATION_EXPONENT
        # adjusted_score    = combined_score × compliance_factor
        #
        # Quadratic exponent (2) means violations compound non-linearly:
        #   0%  violations → 1.00× multiplier  (no penalty)
        #   25% violations → 0.56× multiplier
        #   50% violations → 0.25× multiplier  (score quartered)
        #   75% violations → 0.06× multiplier  (effectively disqualified)
        #
        # This decouples compliance from trajectory scoring and makes it a
        # multiplicative gate at the aggregate level — a cheating agent cannot
        # overcome the penalty through high answer accuracy alone.
        _VIOLATION_EXPONENT = 2

        def _compliance_tier(rate: float) -> str:
            if rate < 0.05:
                return "compliant"
            if rate < 0.20:
                return "borderline"
            return "non_compliant"

        def _violation_adjusted(combined: float, violation_rate: float) -> float:
            factor = max(0.0, 1.0 - violation_rate) ** _VIOLATION_EXPONENT
            return round(combined * factor, 4)

        by_type: Dict[str, List[EvalResult]] = {}
        by_difficulty: Dict[str, List[EvalResult]] = {}
        for r in results:
            by_type.setdefault(r.question_type, []).append(r)
            by_difficulty.setdefault(r.difficulty, []).append(r)

        by_type_summary = {}
        for qtype, rs in by_type.items():
            total_calls = sum(r.tool_call_count for r in rs)
            total_violations = sum(r.meta.get("actor_gate_violations", 0) for r in rs)
            violation_rate = (
                round(total_violations / total_calls, 4) if total_calls else 0.0
            )
            base_combined = mean([r.combined_score for r in rs])

            summary: Dict[str, Any] = {
                "n": len(rs),
                "answer_score": mean([r.answer_score for r in rs]),
                "trajectory_score": mean([r.trajectory_score for r in rs]),
                "combined_score": base_combined,
                "accuracy": round(sum(r.answer_correct for r in rs) / len(rs), 4),
                "avg_tool_calls": mean([r.tool_call_count for r in rs]),
            }

            if qtype == "PERSPECTIVE":
                compliance_factor = round(
                    max(0.0, 1.0 - violation_rate) ** _VIOLATION_EXPONENT, 4
                )
                summary.update(
                    {
                        "violation_rate": violation_rate,
                        "compliance_factor": compliance_factor,
                        "compliance_tier": _compliance_tier(violation_rate),
                        # Primary leaderboard axis — combined_score alone allows a
                        # cheating agent to rank above a disciplined one. This number
                        # prevents that by applying the compliance penalty independently
                        # of answer quality.
                        "violation_adjusted_combined_score": _violation_adjusted(
                            base_combined, violation_rate
                        ),
                        "avg_actor_gate_violations": mean(
                            [r.meta.get("actor_gate_violations", 0) for r in rs]
                        ),
                        "avg_subsystem_violations": mean(
                            [r.meta.get("subsystem_violations", 0) for r in rs]
                        ),
                    }
                )
            elif qtype == "SILENCE":
                summary["search_space_coverage"] = mean(
                    [
                        r.meta.get("trajectory_detail", {}).get(
                            "search_space_coverage", 0
                        )
                        for r in rs
                    ]
                )

            by_type_summary[qtype] = summary

        # ── Global violation_adjusted_combined_score ──────────────────────────
        # A single number for cross-model ranking. Agents without PERSPECTIVE
        # questions are not penalised (violation_rate = 0, factor = 1.0).
        all_calls = sum(r.tool_call_count for r in results)
        all_violations = sum(r.meta.get("actor_gate_violations", 0) for r in results)
        global_violation_rate = (
            round(all_violations / all_calls, 4) if all_calls else 0.0
        )
        overall_combined = mean([r.combined_score for r in results])

        return {
            "overall": {
                "n": len(results),
                "answer_score": mean([r.answer_score for r in results]),
                "trajectory_score": mean([r.trajectory_score for r in results]),
                "combined_score": overall_combined,
                "accuracy": round(
                    sum(r.answer_correct for r in results) / len(results), 4
                ),
                "avg_tool_calls": mean([r.tool_call_count for r in results]),
                "global_violation_rate": global_violation_rate,
                "global_compliance_factor": round(
                    max(0.0, 1.0 - global_violation_rate) ** _VIOLATION_EXPONENT, 4
                ),
                "global_compliance_tier": _compliance_tier(global_violation_rate),
                # Primary cross-track ranking number
                "violation_adjusted_combined_score": _violation_adjusted(
                    overall_combined, global_violation_rate
                ),
            },
            "by_type": by_type_summary,
            "by_difficulty": {
                diff: {
                    "n": len(rs),
                    "answer_score": mean([r.answer_score for r in rs]),
                    "trajectory_score": mean([r.trajectory_score for r in rs]),
                    "combined_score": mean([r.combined_score for r in rs]),
                }
                for diff, rs in by_difficulty.items()
            },
        }


# ─────────────────────────────────────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="OrgForge Agentic Eval Harness - PERSPECTIVE, COUNTERFACTUAL, SILENCE"
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=EVAL_DIR / "eval_questions.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=EVAL_DIR / "agentic_results.json",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-6",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=5,
        help="Max tool-use steps per question (SILENCE questions may need more)",
    )
    parser.add_argument(
        "--types",
        nargs="+",
        choices=["PERSPECTIVE", "COUNTERFACTUAL", "SILENCE"],
        help="Run only specific tracks",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--ungated",
        action="store_true",
        default=False,
        help=(
            "Disable all actor/subsystem gates — god-mode corpus access. "
            "Establishes the Epistemic Tax ceiling. "
            "Default output: export/eval/ungated_results.json"
        ),
    )
    parser.add_argument(
        "--zero-shot",
        action="store_true",
        default=False,
        help=(
            "Provide no tools to the agent (no corpus access). "
            "Establishes the hallucination / prior-knowledge floor. "
            "Default output: export/eval/zero_shot_results.json"
        ),
    )
    args = parser.parse_args()

    # Default output paths differ by mode so runs don't clobber each other
    if args.out == EVAL_DIR / "agentic_results.json":
        if args.zero_shot:
            args.out = EVAL_DIR / "zero_shot_results.json"
        elif args.ungated:
            args.out = EVAL_DIR / "ungated_results.json"

    runner = AgenticEvalRunner(
        model=args.model,
        max_steps=args.max_steps,
        ungated=args.ungated,
        zero_shot=args.zero_shot,
    )
    runner.run(
        questions_path=args.questions,
        out_path=args.out,
        question_types=args.types,
        max_questions=args.max_questions,
    )
