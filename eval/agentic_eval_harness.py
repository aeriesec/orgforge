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

from collections import Counter
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
from eval_divergence import NLIScorer

from graph_eval_track import (
    register_graph_tool,
    score_graph_trajectory,
    score_graph_answer,
)

logger = logging.getLogger("orgforge.agentic_eval")

with open(Path(__file__).resolve().parent.parent / "config" / "config.yaml") as f:
    _CFG = yaml.safe_load(f)

_SIM_CFG = _CFG.get("simulation", {})
BASE = Path(_SIM_CFG.get("output_dir", "./export"))
EVAL_DIR = BASE / "eval"
_SIM_START = datetime.strptime(CONFIG["simulation"]["start_date"], "%Y-%m-%d")

_TRACK_WEIGHTS = {
    "PERSPECTIVE": {"answer": 0.40, "trajectory": 0.60},
    "COUNTERFACTUAL": {"answer": 0.50, "trajectory": 0.50},
    "SILENCE": {"answer": 0.30, "trajectory": 0.70},
    "GRAPH": {"answer": 0.50, "trajectory": 0.50},
}

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
    "get_invoice": "invoice",
    "get_nps_response": "salesforce",
    "get_events_for_day": None,
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
        "jira_ticket_created",
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
    "deep_work_session",
    "watercooler_chat",
    "ticket_progress",
    "standup",
    "dept_plan_created",
    "mentoring",
    "inbound_external_email",
    "customer_email_routed",
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

_JIRA_PROJECT_ACCESS: Dict[str, Set[str]] = {
    "ENG": {"engineering_backend", "engineering_mobile", "ceo"},
    "HR": {"hr_ops", "ceo"},
    "SALES": {"sales_marketing", "ceo"},
    "PROD": {"product", "ceo"},
    "DES": {"design", "product", "ceo"},
    "QA": {"qa_support", "ceo"},
    "ORG": {
        "engineering_backend",
        "engineering_mobile",
        "product",
        "ceo",
        "hr_ops",
        "design",
        "sales_marketing",
        "qa_support",
    },
}

_TEMPORAL_DRIFT_THRESHOLD_DAYS = 5


def _jira_project_visible(ticket_id: str, role: str) -> bool:
    prefix = ticket_id.split("-")[0] if "-" in ticket_id else ""
    allowed_roles = _JIRA_PROJECT_ACCESS.get(prefix)
    if allowed_roles is None:
        return True  # unknown prefix — don't restrict
    return role in allowed_roles


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
    horizon_violation: bool
    actor_gate_violation: bool
    subsystem_violation: bool
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
    graph_tool_called: bool = False  # GRAPH track
    graph_correct_day_queried: bool = False
    dead_ends_hit: int = 0
    dead_ends_recovered: int = 0
    budget_exceeded: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


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
        actor_visible_by_subsystem: Optional[Dict[str, Set[str]]] = None,
        actor_subsystem_access: Optional[Set[str]] = None,
    ):
        self._mem = mem
        self._question = question
        self._as_of_time = as_of_time
        self._actor_visible = actor_visible_artifacts or set()
        self._actor_visible_by_subsystem = actor_visible_by_subsystem or {}
        self._actor_subsystems = actor_subsystem_access
        self._question_type = question.get("question_type", "")
        self._call_log: List[ToolCall] = []
        self._confluence_seen: set = set()

    def _gate_ts(self) -> str:
        if self._question_type == "SILENCE":
            if not hasattr(self, "_silence_gate_ts"):
                events = self._mem.get_event_log(from_db=True)
                max_day = max((e.day for e in events), default=1) if events else 1
                self._silence_gate_ts = _business_day_to_date(
                    _SIM_START, max_day
                ).isoformat()
            return self._silence_gate_ts
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

        if doc_type in ("jira", "jira_tickets"):
            if not _jira_project_visible(doc_id, self._question.get("actor_role", "")):
                return True, False

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
                if id_field in base:
                    requested_id = base[id_field]
                    query[id_field] = (
                        requested_id
                        if requested_id in self._actor_visible
                        else "__blocked__"
                    )

        return query, effective_ts

    def get_ticket(self, ticket_id: str) -> dict:
        t0 = time.time()
        gate = self._gate_ts()
        query, effective_ts = self._build_query({"id": ticket_id}, doc_type="jira")
        if query is None:
            self._record(
                "get_ticket",
                {"ticket_id": ticket_id},
                [],
                t0,
                timestamp_applied=effective_ts,
            )
            return {}

        doc = (
            self._mem._db["jira_tickets"].find_one(
                query,
                {
                    "_id": 0,
                    "causal_chain": 0,
                    "dept_type": 0,
                    "story_points": 0,
                    "bot_threads": 0,
                    "escalation_narrative": 0,
                    "gap_areas": 0,
                    "recurrence_chain_depth": 0,
                    "recurrence_chain_root": 0,
                    "recurrence_gap_days": 0,
                    "recurrence_of": 0,
                    "prior_postmortem": 0,
                    "sprint": 0,
                },
            )
            or {}
        )

        if doc:
            comments = doc.get("comments", [])
            doc["comments"] = [c for c in comments if c.get("created", "9999") <= gate]

            created = doc.get("created_at", "9999")
            in_progress_day = doc.get("in_progress_since")
            in_review_day = doc.get("in_review_since")

            def day_to_iso(day):
                return (
                    _business_day_to_date(_SIM_START, day).isoformat()
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

            doc["comments"] = [
                {k: v for k, v in c.items() if k not in ("day", "updated")}
                for c in comments
                if c.get("created", "9999") <= gate
            ]

        results = self._record(
            "get_ticket",
            {"ticket_id": ticket_id},
            [doc] if doc else [],
            t0,
            timestamp_applied=effective_ts,
        )

        return results[0] if results else {}

    def get_confluence_page(self, page_id: str) -> dict:
        t0 = time.time()
        query, effective_ts = self._build_query({"_id": page_id}, doc_type="confluence")
        if query is None:
            self._record("get_confluence_page", {"page_id": page_id}, [], t0)
            return {}
        query["type"] = "confluence"

        doc = (
            self._mem._db["artifacts"].find_one(
                query,
                {
                    "_id": 0,
                    "embedding": 0,
                    "type": 0,
                    "metadata": 0,
                    "date": 0,
                    "timestamp": 0,
                },
            )
            or {}
        )

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
            {"_id": thread_id},
            doc_type="slack",
            id_field="_id",
        )
        if query is None:
            return self._record("get_slack_thread", {"thread_id": thread_id}, [], t0)
        query["type"] = "slack_thread"
        doc = (
            self._mem._db["artifacts"].find_one(query, {"_id": 0, "embedding": 0}) or {}
        )
        docs = [doc] if doc else []
        return self._record(
            "get_slack_thread",
            {"thread_id": thread_id},
            docs,
            t0,
            timestamp_applied=effective_ts,
        )

    def get_email(self, email_id: str) -> dict:
        t0 = time.time()
        query, effective_ts = self._build_query({"_id": email_id}, doc_type="email")
        if query is None:
            self._record("get_email", {"email_id": email_id}, [], t0)
            return {}
        query["type"] = "email"
        doc = (
            self._mem._db["artifacts"].find_one(query, {"_id": 0, "embedding": 0}) or {}
        )
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
        query, effective_ts = self._build_query({"pr_id": pr_id}, doc_type="pr")
        if query is None:
            self._record("get_pr", {"pr_id": pr_id}, [], t0)
            return {}
        doc = (
            self._mem._db["pull_requests"].find_one(
                query, {"_id": 0, "author_email": 0}
            )
            or {}
        )

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

    _ARTIFACT_ID_BLOCKLIST = {
        "eml_path",
        "artifact_path",
        "slack_path",
        "jira",
        "slack",
    }

    def _project_event(self, event: dict, actor_visible: Set[str]) -> dict:
        drop_keys = {"tags", "date", "artifact_path", "eml_path", "slack_path"}
        doc = {k: v for k, v in event.items() if k not in drop_keys}

        if "artifact_ids" in doc:
            doc["artifact_ids"] = {
                k: v
                for k, v in doc["artifact_ids"].items()
                if v
                and v != "[]"
                and k not in self._ARTIFACT_ID_BLOCKLIST
                and (
                    (isinstance(v, list) and any(item in actor_visible for item in v))
                    or (isinstance(v, str) and v in actor_visible)
                )
            }

        return doc

    def get_events_for_day(
        self, day: int, event_type: Optional[str] = None
    ) -> List[dict]:
        t0 = time.time()

        if not event_type:
            logger.warning("[get_events_for_day] Called without event_type — blocked")
            return self._record(
                "get_events_for_day", {"day": day, "event_type": None}, [], t0
            )

        if self._question_type == "SILENCE":
            trigger_day = self._question.get("trigger_day", 1)
            gate_ts = self._gate_ts()
            gate_day = _date_to_business_day(
                _SIM_START, datetime.fromisoformat(gate_ts)
            )
            if day < trigger_day or day > gate_day:
                logger.warning(
                    f"[get_events_for_day] Day {day} requested but window is Day {trigger_day}–{gate_day} — blocked"
                )
                return self._record(
                    "get_events_for_day", {"day": day, "event_type": event_type}, [], t0
                )
        else:
            gate_day = _date_to_business_day(
                _SIM_START, datetime.fromisoformat(self._as_of_time)
            )
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

        allowed_types: Set[str] = set()
        if self._actor_subsystems is None:
            allowed_types = set(KNOWN_EVENT_TYPES)
        else:
            for subsystem in self._actor_subsystems:
                allowed_types.update(_SUBSYSTEM_EVENT_TYPES.get(subsystem, set()))

        if self._question_type != "COUNTERFACTUAL":
            allowed_types -= _INTERNAL_EVENT_TYPES

        if event_type:
            if (
                event_type in _INTERNAL_EVENT_TYPES
                and self._question_type != "COUNTERFACTUAL"
            ):
                logger.warning(
                    f"[get_events_for_day] Internal event type requested: {event_type}"
                )
                return self._record(
                    "get_events_for_day", {"day": day, "event_type": event_type}, [], t0
                )
            query["type"] = event_type
        else:
            query["type"] = {"$in": list(allowed_types)}

        if self._question_type == "PERSPECTIVE":
            actor = self._question.get("actor", "")
            visible = self._actor_visible_by_subsystem
            query["$or"] = [
                {"actors": actor},
                {"artifact_ids.email": {"$in": list(visible.get("email", set()))}},
                {"artifact_ids.jira": {"$in": list(visible.get("jira", set()))}},
                {
                    "artifact_ids.confluence": {
                        "$in": list(visible.get("confluence", set()))
                    }
                },
                {
                    "artifact_ids.slack_thread": {
                        "$in": list(visible.get("slack", set()))
                    }
                },
                {"artifact_ids.pr": {"$in": list(visible.get("git", set()))}},
                {
                    "artifact_ids.zoom_transcript": {
                        "$in": list(visible.get("zoom", set()))
                    }
                },
            ]

        # logger.info(f"[get_events_for_day] - query: {query}")

        docs = list(
            self._mem._db["events"].find(
                query,
                {
                    "_id": 0,
                    "event_id": 1,
                    "type": 1,
                    "day": 1,
                    "actors": 1,
                    "summary": 1,
                    "artifact_ids": 1,
                },
            )
        )

        # logger.info(f"[get_events_for_day] - results before project: {docs}")
        if self._actor_visible is not None:
            docs = [self._project_event(d, self._actor_visible) for d in docs]

        # logger.info(f"[get_events_for_day] - results: {docs}")

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

        if after_day is not None:
            try:
                after_day = int(after_day)
            except (ValueError, TypeError):
                after_day = None

        effective_ts = self._gate_ts()
        MAX_SEARCH_LIMIT = 15
        limit = min(limit, MAX_SEARCH_LIMIT)

        actor_id_filter = None
        if self._question_type == "PERSPECTIVE" and self._actor_visible:
            actor_id_filter = list(self._actor_visible)

        exact_doc = self._mem._db["artifacts"].find_one(
            {"_id": query},
            {"embedding": 0, "timestamp": 0, "created_at": 0, "date": 0},
        )
        if exact_doc:
            # logger.info(f"[search_artifacts] - exact_doc: {exact_doc}")

            ts_filter = {"timestamp": {"$lte": effective_ts}}
            if after_day is not None:
                floor_ts = _business_day_to_date(_SIM_START, after_day).isoformat()
                ts_filter["timestamp"]["$gte"] = floor_ts
            if actor_id_filter is not None and query not in self._actor_visible:
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
        if actor_id_filter is not None:
            text_filter["_id"] = {"$in": actor_id_filter}
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

        # logger.info(f"[search_artifacts] - query: {text_filter}")
        # logger.info(f"[search_artifacts] - limit: {limit}")

        results = list(
            self._mem._db["artifacts"]
            .find(
                text_filter,
                {
                    "content": 0,
                    "embedding": 0,
                    "timestamp": 0,
                    "created_at": 0,
                    "date": 0,
                    "metadata.tags": 0,
                    "metadata.parent_id": 0,
                    "metadata.is_chunk": 0,
                    "score": {"$meta": "textScore"},
                },
            )
            .sort([("score", {"$meta": "textScore"})])
            .limit(limit)
        )

        # logger.info(f"[search_artifacts] - results length: {len(results)}")

        # logger.info(f"[search_artifacts] - results: {results}")

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

    def get_graph_snapshot(self, day: int) -> dict:
        """
        Returns the social graph snapshot for the given day:
        nodes (actor names), edges (source, target, weight), and org state.
        Registered dynamically for GRAPH-track questions via register_graph_tool().
        Calling this on a non-GRAPH question logs a tool call but returns {}.
        """

        t0 = time.time()
        if self._question_type != "GRAPH":
            self._record("get_graph_snapshot", {"day": day}, [], t0)
            return {}

        logger.warning(
            "[get_graph_snapshot] stub reached for GRAPH question — register_graph_tool() may not have run"
        )
        self._record("get_graph_snapshot", {"day": day}, [], t0)
        return {}

    def get_stress_snapshot(self, day: int) -> dict:
        """
        Returns {day, stress: {name: int}} for the given day.
        Registered dynamically for GRAPH-track questions.
        """

        t0 = time.time()
        if self._question_type != "GRAPH":
            self._record("get_stress_snapshot", {"day": day}, [], t0)
            return {}
        self._record("get_stress_snapshot", {"day": day}, [], t0)
        return {}


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

        # Extract boolean from agent answer — schema violation scores 0, not 0.1
        agent_bool = self._extract_boolean(final_answer)
        if agent_bool is None:
            return 0.0, False

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
            score += 0.2

        return min(score, 1.0), True

    def score_trajectory(
        self,
        trajectory: AgentTrajectory,
        question: dict,
    ) -> PerspectiveTrajectoryScore:
        calls = trajectory.tool_calls
        if not calls:
            return PerspectiveTrajectoryScore(
                epistemic_discipline=0.0,
                subsystem_discipline=0.0,
                horizon_discipline=0.0,
                temporal_precision=0.0,
                temporal_drift_discipline=1.0,
                conclusion_grounding=0.0,
                dead_end_recovery=0.0,
                composite=0.0,
            )

        n = len(calls)
        actor_cone_violations = sum(1 for c in calls if c.actor_gate_violation)
        subsystem_violations = sum(1 for c in calls if c.subsystem_violation)
        horizon_violations = sum(1 for c in calls if c.horizon_violation)
        dead_ends = sum(1 for c in calls if c.returned_empty)
        dead_ends_recovered = trajectory.dead_ends_recovered

        epistemic_discipline = 1.0 - (actor_cone_violations / n)
        subsystem_discipline = 1.0 - (subsystem_violations / n)
        horizon_discipline = 1.0 - (horizon_violations / n)

        # Conclusion grounding: continuous coverage fraction of required evidence
        # cited within the actor's cone. Binary hit/miss loses resolution at the
        # top of the leaderboard where frontier models will be differentiating.
        actor_visible = set(question.get("actor_visible_artifacts", []))
        gt_evidence = set(
            question.get("ground_truth", {}).get("evidence_artifacts", [])
        )
        cited = set(trajectory.final_answer.get("evidence_artifacts", []))

        if gt_evidence:
            in_cone_required = gt_evidence & actor_visible
            if in_cone_required:
                conclusion_grounding = len(cited & in_cone_required) / len(
                    in_cone_required
                )
            else:
                conclusion_grounding = 1.0  # nothing in-cone required
        elif cited:
            conclusion_grounding = 1.0 if (cited & actor_visible) else 0.3
        else:
            conclusion_grounding = 0.0

        dead_end_recovery = dead_ends_recovered / dead_ends if dead_ends > 0 else 1.0

        sim_days = CONFIG["simulation"].get("num_days", 60)
        drifts = [
            c.temporal_drift_days for c in calls if c.temporal_drift_days is not None
        ]
        temporal_precision = (
            max(0.0, 1.0 - mean(abs(d) / sim_days for d in drifts)) if drifts else 1.0
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
            temporal_drift_discipline=round(temporal_drift_discipline, 4),
            conclusion_grounding=round(conclusion_grounding, 4),
            dead_end_recovery=round(dead_end_recovery, 4),
            composite=round(composite, 4),
        )

    def _extract_boolean(self, answer: Dict) -> Optional[bool]:
        val = answer.get("could_actor_have_known")
        if isinstance(val, bool):
            return val
        # Schema violation — model did not return the required boolean field.
        # Log and return None so the caller scores this as 0.
        logger.warning(
            "  PERSPECTIVE schema violation: 'could_actor_have_known' missing or "
            f"non-boolean (got {val!r})"
        )
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

    @staticmethod
    def _mechanism_matches(aliases: set, text: str) -> bool:
        return any(re.search(rf"\b{re.escape(alias)}\b", text) for alias in aliases)

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
        "zd_escalation_source": {
            "zendesk escalation",
            "support ticket escalation",
            "zd escalation",
            "escalated from zendesk",
            "zd_escalation_source",
        },
        "assignment_domain_mismatch": {
            "domain mismatch",
            "planning mismatch",
            "wrong domain",
            "misassigned",
            "assignment_domain_mismatch",
        },
        "jira_from_vendor_email": {
            "vendor email",
            "jira from vendor",
            "vendor ticket",
            "inbound vendor",
            "jira_from_vendor_email",
        },
        "jira_from_customer_email": {
            "customer email",
            "jira from customer",
            "customer ticket",
            "inbound customer",
            "jira_from_customer_email",
        },
        "customer_escalation_relayed": {
            "escalation relayed",
            "customer escalation",
            "support email routed",
            "customer_escalation_relayed",
        },
        "incident_handoff": {
            "handoff",
            "forced handoff",
            "escalation chain",
            "incident_handoff",
            "handoff on departure",
        },
        "pr_gap_detected": {
            "pr review gap",
            "undocumented domain in pr",
            "pr gap",
            "code review gap",
            "pr_gap_detected",
        },
        "async_gap_detected": {
            "async gap",
            "undocumented domain in async",
            "async thread gap",
            "async_gap_detected",
        },
        "doc_gap_detected": {
            "doc gap",
            "documentation gap",
            "undocumented domain in doc",
            "documentation review gap",
            "doc_gap_detected",
        },
        "centrality_vacuum": {
            "centrality vacuum",
            "key person departure",
            "single point of failure",
            "bus factor",
            "centrality_vacuum",
        },
        "sf_stage_advanced_by_customer": {
            "stage advanced",
            "crm stage",
            "customer advanced stage",
            "deal stage",
            "sf_stage_advanced_by_customer",
        },
        "feature_request_fyi": {
            "feature request",
            "product fyi",
            "inbound feature",
            "feature_request_fyi",
        },
        "proactive_outreach_from_crm_signal": {
            "proactive outreach",
            "at-risk",
            "crm signal",
            "outreach from crm",
            "proactive_outreach_from_crm_signal",
        },
        "ticket_completion_notifies_lead": {
            "ticket completion",
            "lead notified",
            "dependent ticket done",
            "ticket_completion_notifies_lead",
        },
        "org_collision_tension": {
            "org tension",
            "collision",
            "overlapping responsibilities",
            "org conflict",
            "org_collision_tension",
        },
        "postmortem_from_incident": {
            "postmortem",
            "post-mortem",
            "incident postmortem",
            "postmortem created",
            "postmortem_from_incident",
        },
        "incident_triggers_risk_flag": {
            "risk flag",
            "sf risk",
            "incident risk",
            "salesforce risk flag",
            "incident_triggers_risk_flag",
        },
        "review_triggers_revision": {
            "pr revision",
            "code review revision",
            "review revision",
            "revision triggered",
            "review_triggers_revision",
        },
        "hire_fills_knowledge_gap": {
            "hire fills gap",
            "new hire",
            "gap closed",
            "expertise hire",
            "hire_fills_knowledge_gap",
        },
        "escalation_from_zendesk": {
            "zendesk escalation",
            "escalation from zendesk",
            "zd escalation",
            "escalation_from_zendesk",
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

        gt_mechanism = ground_truth.get("causal_mechanism", "")
        agent_mechanism_prose = str(final_answer.get("mechanism", "")).lower()
        agent_mechanism_structured = str(
            final_answer.get("causal_mechanism", "")
        ).lower()
        agent_mechanism = agent_mechanism_prose + " " + agent_mechanism_structured
        aliases = self._MECHANISM_ALIASES.get(gt_mechanism, {gt_mechanism})
        if self._mechanism_matches(aliases, agent_mechanism):
            score += 0.35

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

        evidence_artifacts = ground_truth.get("evidence_chain_artifacts", {})
        cause_artifacts = set(evidence_artifacts.get("cause", []))
        effect_artifacts = set(evidence_artifacts.get("effect", []))

        cause_identified = (
            1.0 if (cause_artifacts and cause_artifacts & retrieved_ids) else 0.0
        )
        effect_identified = (
            1.0 if (effect_artifacts and effect_artifacts & retrieved_ids) else 0.0
        )

        gt_mechanism = ground_truth.get("causal_mechanism", "")
        aliases = self._MECHANISM_ALIASES.get(gt_mechanism, {gt_mechanism})
        agent_text = " ".join(
            [str(c.arguments) for c in calls] + [str(trajectory.final_answer)]
        ).lower()
        mechanism_correct = 1.0 if self._mechanism_matches(aliases, agent_text) else 0.0

        cause_only_artifacts = cause_artifacts - effect_artifacts
        effect_only_artifacts = effect_artifacts - cause_artifacts
        shared_artifacts = cause_artifacts & effect_artifacts

        cause_call_idx = next(
            (
                i
                for i, c in enumerate(calls)
                if cause_only_artifacts & set(c.result_ids)
            ),
            next(
                (
                    i
                    for i, c in enumerate(calls)
                    if shared_artifacts & set(c.result_ids)
                ),
                None,
            ),
        )
        effect_call_idx = next(
            (
                i
                for i, c in enumerate(calls)
                if effect_only_artifacts & set(c.result_ids)
            ),
            next(
                (
                    i
                    for i, c in enumerate(calls)
                    if shared_artifacts & set(c.result_ids)
                ),
                None,
            ),
        )
        if cause_call_idx is not None and effect_call_idx is not None:
            if cause_call_idx == effect_call_idx and not (
                cause_only_artifacts or effect_only_artifacts
            ):
                causal_chain_complete = (
                    0.5  # single shared artifact, no ordered traversal
                )
            elif cause_call_idx <= effect_call_idx:
                causal_chain_complete = 1.0
            else:
                causal_chain_complete = 0.5
        elif cause_call_idx is not None or effect_call_idx is not None:
            causal_chain_complete = 0.5
        else:
            causal_chain_complete = 0.0

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
        # Schema violation — model did not return the required boolean field.
        # Log and return None so the caller scores this as 0.
        logger.warning(
            f"  COUNTERFACTUAL schema violation: '{key}' missing or "
            f"non-boolean (got {val!r})"
        )
        return None


class SilenceScorer:
    """
    Scores a SILENCE trajectory.

    The key insight: absence is only meaningful if the agent searched somewhere
    relevant. A correct "no" is disqualified as a guess if the agent made zero
    tool calls or every call was outside the expected search space.

    Combined score is binary (see _run_question):
      - Wrong answer → 0.0
      - Correct answer + no relevant search → 0.0 (ruled a guess)
      - Correct answer + at least one relevant call → 1.0

    Trajectory scoring (diagnostic only — does not contribute to combined_score):
      - made_relevant_search: did at least one tool call overlap expected search space?
      - correct_absence_conclusion: did agent say "does not exist" explicitly?
      - search_space_coverage: fraction of expected_search_space checked (for analysis)
    """

    def score_answer(
        self, final_answer: Dict, ground_truth: Dict
    ) -> Tuple[float, bool]:
        if not final_answer:
            return 0.0, False

        gt_answer = ground_truth.get("answer", False)  # Always False for SILENCE
        agent_answer = self._extract_absence_conclusion(final_answer)

        if agent_answer is None:
            return 0.0, False

        correct = agent_answer == gt_answer
        return (1.0, True) if correct else (0.0, False)

    _PATH_SUBSYSTEM_MAP: List[Tuple[str, str]] = [
        ("CONF-", "confluence"),
        ("ENG-", "jira"),
        ("IT-", "jira"),
        ("ORG-", "jira"),
        ("QA-", "jira"),
        ("HR-", "jira"),
        ("PROD-", "jira"),
        ("DES-", "jira"),
        ("SALES-", "jira"),
        ("PR-", "git"),
        ("ZD-", "zendesk"),
        ("SF-ACC", "salesforce"),
        ("SF-OPP", "salesforce"),
        ("ZOOM-", "zoom"),
        ("DD-", "datadog"),
        ("INV-", "invoice"),
        ("NPS-", "salesforce"),
        ("slack_", "slack"),
        ("ext_email_", "email"),
        ("customer_reply_", "email"),
        ("reply_customer_", "email"),
        ("ack_vendor_", "email"),
        ("hr_outbound_", "email"),
    ]

    _TOOL_SUBSYSTEM_MAP: Dict[str, str] = {
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
    }

    _SEARCH_DOCTYPE_MAP: Dict[str, str] = {
        "jira": "jira",
        "confluence": "confluence",
        "slack": "slack",
        "email": "email",
        "pr": "git",
        "zd_ticket": "zendesk",
        "zoom": "zoom",
    }

    _QUERY_STOPWORDS: frozenset = frozenset(
        {
            "the",
            "a",
            "an",
            "of",
            "in",
            "for",
            "on",
            "to",
            "at",
            "by",
            "is",
            "was",
            "and",
            "or",
            "not",
            "be",
            "it",
            "its",
        }
    )

    def _infer_artifact_subsystem(self, entry: str) -> Optional[str]:
        for prefix, sub in self._PATH_SUBSYSTEM_MAP:
            if entry.startswith(prefix):
                return sub
        return None

    def _semantic_coverage_check(
        self,
        expected_entry: str,
        calls: List[ToolCall],
        question: dict,
        normalized_tool_args: List[str],
    ) -> bool:
        """
        Returns True when the agent's search behavior would plausibly surface
        the expected artifact without exactly naming its path segment.

        Requires BOTH to pass:
        1. Subsystem coverage — at least one call reached the artifact's subsystem.
            search_artifacts with no doc_type counts as covering all subsystems.
        2. Query relevance   — at least one query contains trigger actor names,
            the link_value domain, or event-type keywords from the question.
        """
        target_sub = self._infer_artifact_subsystem(expected_entry)

        subsystem_covered = False
        for call in calls:
            direct_sub = self._TOOL_SUBSYSTEM_MAP.get(call.tool_name)
            if direct_sub is not None:
                if target_sub is None or direct_sub == target_sub:
                    subsystem_covered = True
                    break
            elif call.tool_name == "search_artifacts":
                doc_type = call.arguments.get("doc_type", "")
                mapped_sub = self._SEARCH_DOCTYPE_MAP.get(doc_type)
                if not doc_type or mapped_sub == target_sub or target_sub is None:
                    subsystem_covered = True
                    break

        if not subsystem_covered:
            return False

        gt = question.get("ground_truth", {})
        semantic_keywords: Set[str] = set()

        for actor in question.get("trigger_actors", gt.get("trigger_actors", [])):
            semantic_keywords.update(
                w.lower() for w in str(actor).split() if len(w) > 2
            )
        for term in str(gt.get("link_value", "")).split():
            if len(term) > 2:
                semantic_keywords.add(term.lower())
        for term in question.get("trigger_event_type", "").replace("_", " ").split():
            if len(term) > 2:
                semantic_keywords.add(term.lower())

        semantic_keywords -= self._QUERY_STOPWORDS

        if not semantic_keywords:
            return subsystem_covered

        return any(
            kw in arg for kw in semantic_keywords for arg in normalized_tool_args
        )

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
            if any(norm_term in arg for arg in normalized_tool_args):
                covered.add(original)
            elif norm_term in normalized_result_ids:
                covered.add(original)
            elif any(original.lower() in arg for arg in normalized_tool_args):
                covered.add(original)
            elif self._semantic_coverage_check(
                original, calls, question, normalized_tool_args
            ):
                covered.add(original)

        search_space_coverage = (
            len(covered) / len(expected_space) if expected_space else 1.0
        )

        # Binary soundness gate: did the agent make at least one tool call that
        # overlaps with the expected search space? A single relevant call is
        # sufficient — we are ruling out guesses, not rewarding exhaustiveness.
        # A call to the wrong subsystem does not count, even if the answer is correct.
        made_relevant_search = len(covered) > 0 if expected_space else len(calls) > 0

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

        # premature_conclusion kept for diagnostics but no longer penalises composite
        premature_conclusion = 1.0 - max(0.0, min(1.0, (n - 1) / 3))

        horizon_discipline = 1.0  # SILENCE has no temporal gate

        # Composite is now the binary relevant-search gate.
        # search_space_coverage and correct_absence_conclusion are preserved
        # in the dataclass for diagnostic use in meta — they do not contribute
        # to combined_score.
        composite = 1.0 if made_relevant_search else 0.0

        return SilenceTrajectoryScore(
            search_space_coverage=round(search_space_coverage, 4),
            correct_absence_conclusion=round(correct_absence_conclusion, 4),
            premature_conclusion=round(premature_conclusion, 4),
            horizon_discipline=round(horizon_discipline, 4),
            composite=round(composite, 4),
        )

    def _extract_absence_conclusion(self, answer: Dict) -> Optional[bool]:
        """True = artifact exists, False = artifact does not exist."""
        val = answer.get("exists", answer.get("found", answer.get("answer")))
        if isinstance(val, bool):
            return val
        # Schema violation — model did not return the required boolean field.
        # Log and return None so the caller scores this as 0.
        logger.warning(
            f"  SILENCE schema violation: 'exists' missing or non-boolean (got {val!r})"
        )
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
        max_steps: int = 6,
        ungated: bool = False,
        zero_shot: bool = False,
        call_delay: float = 1.0,
        service_tier: str = "default",
        skip_nli: bool = False,
    ):
        self._model = model
        self._max_steps = max_steps
        self._call_delay = call_delay
        self._service_tier = service_tier

        from memory import Memory
        import boto3

        self._mem = Memory()
        self._bedrock = boto3.client("bedrock-runtime")

        self._perspective_scorer = PerspectiveScorer()
        self._counterfactual_scorer = CounterfactualScorer()
        self._silence_scorer = SilenceScorer()

        # --ungated: all actor/subsystem gates disabled regardless of question type.
        # Establishes the "god-mode" information ceiling for the Epistemic Tax.
        self._ungated = ungated

        # --zero-shot: agent receives no tools at all (no corpus access).
        # Establishes the hallucination / prior-knowledge floor.
        # Mutually exclusive with --ungated; zero_shot takes precedence if both set.
        self._zero_shot = zero_shot

        # NLI scorer for zero-shot reasoning validation.
        # Loaded only when running zero-shot — no overhead for gated/ungated runs.
        # If skip_nli=True, zero-shot falls back to answer_score with a warning.
        self._nli: Optional[NLIScorer] = None
        if zero_shot and not skip_nli:
            try:
                logger.info("Loading NLI scorer for zero-shot reasoning validation...")
                self._nli = NLIScorer()
                logger.info("NLI scorer loaded.")
            except Exception as exc:
                logger.warning(
                    f"NLI scorer failed to load — zero-shot will use answer_score only: {exc}"
                )

        vis_path = EVAL_DIR / "actor_visibility.json"
        if vis_path.exists():
            with open(vis_path) as f:
                self._actor_visibility = json.load(f)
        else:
            logger.warning(
                "actor_visibility.json not found — PERSPECTIVE gates will not be enforced"
            )
            self._actor_visibility = {}
        self._artifact_timestamps = {
            doc["_id"]: doc.get("timestamp", "")
            for doc in self._mem._db["artifacts"].find({}, {"_id": 1, "timestamp": 1})
        }

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
        token_counter = Counter()

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

            token_counter["prompt"] += result.meta.get("prompt_tokens", 0)
            token_counter["completion"] += result.meta.get("completion_tokens", 0)
            token_counter["total"] += result.meta.get("total_tokens", 0)

            logger.info(
                f"  answer={result.answer_score:.3f} "
                f"trajectory={result.trajectory_score:.3f} "
                f"combined={result.combined_score:.3f} "
                f"tools={result.tool_call_count} "
                f"total tokens={result.meta.get('total_tokens', 0):,} "
                f"input tokens={result.meta.get('prompt_tokens', 0):,} "
                f"[total input tokens: {token_counter['prompt']:,}]"
                f"[running: {token_counter['total']:,}]"
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

    def _build_events_tool_def(self, question: dict) -> dict:
        qtype = question.get("question_type", "")
        actor_subsystems = (
            set(question.get("subsystem_access", []))
            if qtype == "PERSPECTIVE"
            else None
        )

        allowed_types: Set[str] = set()
        if actor_subsystems is None:
            allowed_types = set(KNOWN_EVENT_TYPES)
        else:
            for subsystem in actor_subsystems:
                allowed_types.update(_SUBSYSTEM_EVENT_TYPES.get(subsystem, set()))

        if qtype != "COUNTERFACTUAL":
            allowed_types -= _INTERNAL_EVENT_TYPES

        logger.info(f"[allowed_types] - {allowed_types}")

        return {
            "toolSpec": {
                "name": "get_events_for_day",
                "description": "Retrieve simulation events for a given day, optionally filtered by type.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "day": {
                                "type": "integer",
                                "description": "Simulation day number",
                            },
                            "event_type": {
                                "type": "string",
                                "description": "Required. You must always filter by event type. Never call this tool without specifying an event_type.",
                                "enum": sorted(allowed_types),
                            },
                        },
                        "required": ["day", "event_type"],
                    }
                },
            }
        }

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
        actor_visible_by_subsystem = {}

        if self._ungated:
            actor_visible = None
            actor_subsystems = None
        else:
            if qtype == "PERSPECTIVE" and not self._ungated:
                actor = question.get("actor", "")
                as_of_time = question.get("as_of_time", "")
                cone_data = self._actor_visibility.get(actor, {})
                if cone_data:
                    actor_visible = set(
                        aid
                        for ids in cone_data.get("visible_artifacts", {}).values()
                        for aid in ids
                        if self._artifact_timestamps.get(aid, "9999") <= as_of_time
                    )
                    for subsystem, ids in cone_data.get(
                        "visible_artifacts", {}
                    ).items():
                        filtered = {
                            aid
                            for aid in ids
                            if self._artifact_timestamps.get(aid, "9999") <= as_of_time
                        }
                        actor_visible_by_subsystem[subsystem] = filtered
                        actor_visible.update(filtered)
                else:
                    logger.warning(f"No visibility cone found for {actor}")
                    actor_visible = set()
                actor_subsystems = set(question.get("subsystem_access", []))
            else:
                actor_visible = None
                actor_subsystems = None

        tools = GatedTools(
            mem=self._mem,
            question=question,
            as_of_time=as_of_time,
            actor_visible_artifacts=actor_visible,
            actor_visible_by_subsystem=actor_visible_by_subsystem,
            actor_subsystem_access=actor_subsystems,
        )

        if question.get("question_type") == "GRAPH":
            register_graph_tool(tools, self._mem, question)

        # Run agent
        trajectory = self._run_agent(question, tools)

        # Score
        if qtype == "PERSPECTIVE":
            answer_score, answer_correct = self._perspective_scorer.score_answer(
                trajectory.final_answer, ground_truth
            )
            question_with_cone = {
                **question,
                "actor_visible_artifacts": list(actor_visible or []),
            }
            traj = self._perspective_scorer.score_trajectory(
                trajectory, question_with_cone
            )
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

        elif qtype == "GRAPH":
            graph_subtype = question.get("graph_subtype", "")
            traj_score = score_graph_trajectory(
                trajectory=trajectory,
                question=question,
                ground_truth=question.get("ground_truth", {}),
            )
            answer_score, answer_correct = score_graph_answer(
                agent_answer=trajectory.final_answer,
                ground_truth=ground_truth,
                graph_subtype=graph_subtype,
            )
            traj_detail = {"traj_score": traj_score}  # ← add this line

        else:
            raise ValueError(f"Unknown question type: {qtype}")

        weights = _TRACK_WEIGHTS[qtype]

        # ── Combined score ────────────────────────────────────────────────────
        # Zero-shot: no tools, so trajectory is meaningless. combined = answer_score
        # directly. This establishes the hallucination/prior-knowledge floor without
        # penalising absence of tool use or inflating correct answers to 1.0.
        #
        # Gated (and ungated): binary — "did the agent get it right the right way?"
        #   A wrong answer is always 0 — trajectory cannot rescue it.
        #   A correct answer is disqualified if trajectory was unsound — the agent
        #   got lucky, not good. Otherwise combined = 1.0.
        #
        #   Track-specific soundness conditions:
        #     PERSPECTIVE    — answer correct
        #     COUNTERFACTUAL — answer correct
        #     SILENCE        — answer correct + made at least one relevant tool call
        #                      (traj_score == 0.0 flags no relevant search was made)
        #     GRAPH          — answer correct
        #
        # answer_score and trajectory_score are preserved in meta for diagnostic
        # use and model-improvement analysis. _TRACK_WEIGHTS are retained for the
        # same purpose but no longer drive combined_score in gated mode.

        if self._zero_shot:
            # Zero-shot: no tools, so trajectory is meaningless.
            # Combined is binary on correctness — wrong answers score 0.0 regardless
            # of partial credit, making zero-shot directly comparable to gated mode.
            # NLI contradiction gate still applies on top: if the agent's reasoning
            # contradicts the question premise it hallucinated a justification —
            # combined=0.0 regardless of the boolean answer.
            # If NLI is unavailable, fall back to answer_correct gate with a warning.
            if not answer_correct:
                combined = 0.0
                failure_reason_gate = "wrong_answer_gated"
            else:
                reasoning = str(trajectory.final_answer.get("reasoning", ""))
                question_text = question.get("question_text") or question.get(
                    "question_prose", ""
                )
                if self._nli and reasoning and question_text:
                    nli_result = self._nli.score(
                        premise=question_text,
                        hypothesis=reasoning,
                    )
                    contradiction = nli_result.get("contradiction", 0.0)
                    if contradiction >= 0.70:
                        combined = 0.0
                        failure_reason_gate = "reasoning_contradicts_question"
                    else:
                        combined = answer_score
                        failure_reason_gate = None
                else:
                    if self._nli and not reasoning:
                        logger.warning(
                            "  [zero-shot] no reasoning field — skipping NLI, using answer_score"
                        )
                    combined = answer_score
                    failure_reason_gate = None
        elif not answer_correct:
            combined = 0.0
            failure_reason_gate = "wrong_answer_gated"
        elif qtype == "SILENCE" and traj_score == 0.0:
            combined = 0.0
            failure_reason_gate = "no_relevant_search_gated"
        else:
            combined = 1.0
            failure_reason_gate = None

        failure_reason = None
        if trajectory.budget_exceeded:
            failure_reason = "step_budget_exceeded"
        if failure_reason_gate:
            failure_reason = failure_reason or failure_reason_gate

        return EvalResult(
            question_id=question["question_id"],
            question_type=qtype,
            difficulty=question.get("difficulty", "unknown"),
            answer_score=round(answer_score, 4),
            answer_correct=answer_correct,
            trajectory_score=round(traj_score, 4),
            combined_score=round(combined, 4),
            failure_reason=failure_reason,
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
                "budget_exceeded": trajectory.budget_exceeded,
                "trajectory_detail": traj_detail,
                "horizon_violations": trajectory.horizon_violations,
                "actor_gate_violations": trajectory.actor_gate_violations,
                "subsystem_violations": trajectory.subsystem_violations,
                "dead_ends_hit": trajectory.dead_ends_hit,
                "dead_ends_recovered": trajectory.dead_ends_recovered,
                "total_latency_ms": round(trajectory.total_latency_ms, 1),
                "tool_calls": [asdict(tc) for tc in trajectory.tool_calls],
                "final_answer": trajectory.final_answer,
                "prompt_tokens": trajectory.prompt_tokens,
                "completion_tokens": trajectory.completion_tokens,
                "total_tokens": trajectory.total_tokens,
            },
        )

    # ── Converse API tool definitions ────────────────────────────────────────

    _CONVERSE_TOOL_DEFS: Dict[str, dict] = {
        "get_ticket": {
            "toolSpec": {
                "name": "get_ticket",
                "description": "Retrieve a Jira ticket by ID.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "ticket_id": {
                                "type": "string",
                                "description": "Jira ticket ID, e.g. 'ENG-42'",
                            }
                        },
                        "required": ["ticket_id"],
                    }
                },
            }
        },
        "get_confluence_page": {
            "toolSpec": {
                "name": "get_confluence_page",
                "description": "Retrieve a Confluence page by ID.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "page_id": {
                                "type": "string",
                                "description": "Confluence page ID",
                            }
                        },
                        "required": ["page_id"],
                    }
                },
            }
        },
        "get_slack_thread": {
            "toolSpec": {
                "name": "get_slack_thread",
                "description": "Retrieve a Slack thread by ID.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "thread_id": {
                                "type": "string",
                                "description": "Slack thread ID",
                            }
                        },
                        "required": ["thread_id"],
                    }
                },
            }
        },
        "get_email": {
            "toolSpec": {
                "name": "get_email",
                "description": "Retrieve an email by ID.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "email_id": {
                                "type": "string",
                                "description": "Email artifact ID",
                            }
                        },
                        "required": ["email_id"],
                    }
                },
            }
        },
        "get_pr": {
            "toolSpec": {
                "name": "get_pr",
                "description": "Retrieve a pull request by ID.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "pr_id": {
                                "type": "string",
                                "description": "Pull request ID",
                            }
                        },
                        "required": ["pr_id"],
                    }
                },
            }
        },
        "get_zd_ticket": {
            "toolSpec": {
                "name": "get_zd_ticket",
                "description": "Retrieve a Zendesk support ticket by ID.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "ticket_id": {
                                "type": "string",
                                "description": "Zendesk ticket ID",
                            }
                        },
                        "required": ["ticket_id"],
                    }
                },
            }
        },
        "get_sf_opportunity": {
            "toolSpec": {
                "name": "get_sf_opportunity",
                "description": "Retrieve a Salesforce opportunity by ID.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "opp_id": {
                                "type": "string",
                                "description": "Salesforce opportunity ID",
                            }
                        },
                        "required": ["opp_id"],
                    }
                },
            }
        },
        "get_sf_account": {
            "toolSpec": {
                "name": "get_sf_account",
                "description": "Retrieve a Salesforce account by ID.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "account_id": {
                                "type": "string",
                                "description": "Salesforce account ID",
                            }
                        },
                        "required": ["account_id"],
                    }
                },
            }
        },
        "get_zoom_transcript": {
            "toolSpec": {
                "name": "get_zoom_transcript",
                "description": "Retrieve a Zoom meeting transcript by ID.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "transcript_id": {
                                "type": "string",
                                "description": "Zoom transcript ID",
                            }
                        },
                        "required": ["transcript_id"],
                    }
                },
            }
        },
        "get_datadog_alert": {
            "toolSpec": {
                "name": "get_datadog_alert",
                "description": "Retrieve a Datadog alert by ID.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "alert_id": {
                                "type": "string",
                                "description": "Datadog alert ID",
                            }
                        },
                        "required": ["alert_id"],
                    }
                },
            }
        },
        "get_invoice": {
            "toolSpec": {
                "name": "get_invoice",
                "description": "Retrieve an invoice by ID.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "invoice_id": {
                                "type": "string",
                                "description": "Invoice ID",
                            }
                        },
                        "required": ["invoice_id"],
                    }
                },
            }
        },
        "get_nps_response": {
            "toolSpec": {
                "name": "get_nps_response",
                "description": "Retrieve an NPS survey response by account name.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "account_name": {
                                "type": "string",
                                "description": "Account name",
                            }
                        },
                        "required": ["account_name"],
                    }
                },
            }
        },
        "search_artifacts": {
            "toolSpec": {
                "name": "search_artifacts",
                "description": "Search for artifacts by keyword when you don't have a specific ID.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Keyword or artifact ID to search for",
                            },
                            "doc_type": {
                                "type": "string",
                                "description": "Filter by type: jira, confluence, slack, email, pr, zd_ticket, zoom",
                            },
                            "actor": {
                                "type": "string",
                                "description": "Optional actor name filter",
                            },
                            "after_day": {
                                "type": "integer",
                                "description": "Only return artifacts from this day onward",
                            },
                        },
                        "required": ["query"],
                    }
                },
            }
        },
        "get_graph_snapshot": {
            "toolSpec": {
                "name": "get_graph_snapshot",
                "description": "Returns the social graph for a given simulation day: nodes (actor names), edges (source, target, weight).",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "day": {
                                "type": "integer",
                                "description": "Simulation day number",
                            }
                        },
                        "required": ["day"],
                    }
                },
            }
        },
        "get_stress_snapshot": {
            "toolSpec": {
                "name": "get_stress_snapshot",
                "description": "Returns stress scores {name: int} for all actors on a given day. Stress >= 72 indicates burnout risk.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "day": {
                                "type": "integer",
                                "description": "Simulation day number",
                            }
                        },
                        "required": ["day"],
                    }
                },
            }
        },
    }

    def _build_tool_dispatch(self, gated: GatedTools) -> Dict[str, Any]:
        return {
            "get_ticket": lambda **kw: gated.get_ticket(kw["ticket_id"]),
            "get_confluence_page": lambda **kw: gated.get_confluence_page(
                kw["page_id"]
            ),
            "get_slack_thread": lambda **kw: gated.get_slack_thread(kw["thread_id"]),
            "get_email": lambda **kw: gated.get_email(kw["email_id"]),
            "get_pr": lambda **kw: gated.get_pr(kw["pr_id"]),
            "get_zd_ticket": lambda **kw: gated.get_zd_ticket(kw["ticket_id"]),
            "get_sf_opportunity": lambda **kw: gated.get_sf_opportunity(kw["opp_id"]),
            "get_sf_account": lambda **kw: gated.get_sf_account(kw["account_id"]),
            "get_zoom_transcript": lambda **kw: gated.get_zoom_transcript(
                kw["transcript_id"]
            ),
            "get_datadog_alert": lambda **kw: gated.get_datadog_alert(kw["alert_id"]),
            "get_invoice": lambda **kw: gated.get_invoice(kw["invoice_id"]),
            "get_nps_response": lambda **kw: gated.get_nps_response(kw["account_name"]),
            "get_events_for_day": lambda **kw: gated.get_events_for_day(
                kw["day"], kw.get("event_type")
            ),
            "search_artifacts": lambda **kw: gated.search_artifacts(
                kw["query"],
                kw.get("doc_type", ""),
                actor=kw.get("actor", ""),
                after_day=kw.get("after_day"),
            ),
            "get_graph_snapshot": lambda **kw: gated.get_graph_snapshot(kw["day"]),
            "get_stress_snapshot": lambda **kw: gated.get_stress_snapshot(kw["day"]),
        }

    _SUBSYSTEM_TOOLS: Dict[str, List[str]] = {
        "jira": ["get_ticket"],
        "confluence": ["get_confluence_page"],
        "slack": ["get_slack_thread"],
        "email": ["get_email"],
        "invoice": ["get_invoice"],
        "git": ["get_pr"],
        "zendesk": ["get_zd_ticket"],
        "salesforce": ["get_sf_opportunity", "get_sf_account", "get_nps_response"],
        "zoom": ["get_zoom_transcript"],
        "datadog": ["get_datadog_alert"],
    }

    _SEARCH_SPACE_SUBSYSTEM: List[Tuple[str, str]] = [
        ("CONF-", "confluence"),
        ("ENG-", "jira"),
        ("IT-", "jira"),
        ("ORG-", "jira"),
        ("QA-", "jira"),
        ("HR-", "jira"),
        ("PROD-", "jira"),
        ("DES-", "jira"),
        ("SALES-", "jira"),
        ("PR-", "git"),
        ("ZD-", "zendesk"),
        ("SF-ACC", "salesforce"),
        ("SF-OPP", "salesforce"),
        ("ZOOM-", "zoom"),
        ("DD-", "datadog"),
        ("INV-", "invoice"),
        ("NPS-", "salesforce"),
        ("slack_", "slack"),
        ("ext_email_", "email"),
        ("customer_reply_", "email"),
        ("reply_customer_", "email"),
        ("ack_vendor_", "email"),
        ("hr_outbound_", "email"),
    ]

    def _subsystems_for_question(self, question: dict) -> Optional[Set[str]]:
        qtype = question.get("question_type", "")

        if qtype == "PERSPECTIVE":
            visible = question.get("actor_visible_artifacts", [])
            if visible:
                needed: Set[str] = set()
                for entry in visible:
                    for prefix, sub in self._SEARCH_SPACE_SUBSYSTEM:
                        if entry.startswith(prefix):
                            needed.add(sub)
                            break
                if needed:
                    if not question.get("cross_subsystem") or len(needed) > 1:
                        return needed

            event_type = question.get("event_type", "")
            for sub, types in _SUBSYSTEM_EVENT_TYPES.items():
                if event_type in types:
                    return {sub}

            subs = question.get("subsystem_access")
            return set(subs) if subs else None

        if qtype == "COUNTERFACTUAL":
            subs = question.get("subsystems_involved")
            if subs:
                return set(subs)

            gt = question.get("ground_truth", {})
            chain = gt.get("evidence_chain_artifacts", {})
            all_ids = chain.get("cause", []) + chain.get("effect", [])
            needed: Set[str] = set()
            for entry in all_ids:
                for prefix, sub in self._SEARCH_SPACE_SUBSYSTEM:
                    if entry.startswith(prefix):
                        needed.add(sub)
                        break
            if needed:
                return needed

            cause_type = gt.get("cause_event_type", "")
            effect_type = gt.get("effect_event_type", "")
            for event_type in [cause_type, effect_type]:
                for sub, types in _SUBSYSTEM_EVENT_TYPES.items():
                    if event_type in types:
                        needed.add(sub)
            if needed:
                return needed

            return {"__events_only__"}

        if qtype == "GRAPH":
            return None

        space = question.get("expected_search_space", [])
        if not space:
            return None
        needed: Set[str] = set()
        for entry in space:
            for prefix, sub in self._SEARCH_SPACE_SUBSYSTEM:
                if entry.startswith(prefix) or entry.lower().startswith(prefix.lower()):
                    needed.add(sub)
                    break
        return needed if needed else None

    def _selected_tool_defs(self, question: dict) -> List[dict]:
        if self._zero_shot:
            return []

        needed = self._subsystems_for_question(question)

        if needed is None:
            selected = [
                v
                for k, v in self._CONVERSE_TOOL_DEFS.items()
                if k != "get_events_for_day"
            ]
        else:
            allowed: Set[str] = {"search_artifacts"}
            for sub in needed:
                for tool_name in self._SUBSYSTEM_TOOLS.get(sub, []):
                    allowed.add(tool_name)

            selected = [
                v
                for k, v in self._CONVERSE_TOOL_DEFS.items()
                if k in allowed and k != "get_events_for_day"
            ]

        selected.append(self._build_events_tool_def(question))

        return selected

    def _build_system_prompt(self, question: dict) -> str:
        qtype = question["question_type"]

        base = (
            "You are an Enterprise Knowledge Analyst evaluating enterprise AI systems. "
            "You reason carefully over corporate documents to answer complex questions. "
            "You cite evidence by artifact ID, stay within stated constraints, and never guess. "
            "Respond ONLY with the requested JSON object — no preamble. "
            "You have access to the full conversation history. Never call a tool to retrieve "
            "an artifact you have already retrieved in a previous step. Re-read the earlier "
            "result instead. IMPORTANT: Only call one tool at a time. Calling multiple tools "
            "is a violation."
        )

        if self._zero_shot:
            base += (
                "\nYou have no tools available. "
                "Answer based on your knowledge alone and respond directly with the JSON object."
            )
            if qtype == "COUNTERFACTUAL":
                full_taxonomy = "\n".join(
                    f"- {k}: {v}" for k, v in self._CAUSAL_LINK_TAXONOMY.items()
                )
                return (
                    f"{base}\n\n"
                    f"IMPORTANT: This is a counterfactual question. You must identify the explicit "
                    f"causal link — do not speculate.\n\n"
                    f"You MUST categorize the link using one of the following labels:\n{full_taxonomy}\n\n"
                    f"Determine whether removing the cause would have changed the effect. "
                    f"You must return a boolean for outcome_changed. Use false if you cannot "
                    f"confirm the outcome changed. Never return null."
                )
            if qtype == "PERSPECTIVE":
                actor = question.get("actor", "the actor")
                day = question.get("as_of_day", "?")
                return (
                    f"{base}\n\n"
                    f"IMPORTANT: You are answering from the perspective of {actor} as of Day {day}. "
                    f"Using information the actor would not have had is a scoring violation."
                )
            return base

        if qtype == "PERSPECTIVE":
            actor = question.get("actor", "the actor")
            day = question.get("as_of_day", "?")
            subs = ", ".join(question.get("subsystem_access", []))
            return (
                f"{base}\n\n"
                f"IMPORTANT: You are answering from the perspective of {actor} as of Day {day}. "
                f"This actor has credentials for: {subs}. "
                f"Having credentials for a system does not mean the actor can see all content in it — "
                f"access depends on the actor's role and direct involvement. "
                f"Using information the actor would not have had is a scoring violation."
            )

        if qtype == "COUNTERFACTUAL":
            full_taxonomy = "\n".join(
                f"- {k}: {v}" for k, v in self._CAUSAL_LINK_TAXONOMY.items()
            )
            return (
                f"{base}\n\n"
                f"IMPORTANT: This is a counterfactual question. You must identify the explicit "
                f"causal link in the data — do not speculate.\n\n"
                f"You MUST categorize the link using one of the following labels:\n{full_taxonomy}\n\n"
                f"Find the cause event and the effect event, then determine whether "
                f"removing the cause would have changed the effect. "
                f"Once you have retrieved both the cause artifact and the effect artifact, "
                f"stop using tools immediately and provide your answer. "
                "If you cannot find sufficient evidence to determine whether the outcome changed, "
                "you must still return a boolean for outcome_changed. Use false if you cannot "
                "confirm the outcome changed. Never return null."
            )

        if qtype == "GRAPH":
            return (
                f"{base}\n\n"
                "You have access to two additional tools for GRAPH questions:\n"
                "  get_graph_snapshot(day: int)  — returns the collaboration graph for a given\n"
                "      simulation day: nodes (actor names), edges (source, target, weight).\n"
                "      Edge weights reflect interaction frequency; higher = stronger relationship.\n"
                "  get_stress_snapshot(day: int) — returns stress scores {name: int} for all\n"
                "      actors on a given day. Stress >= 72 indicates burnout risk.\n\n"
                "To answer GRAPH questions you MUST call at least one of these tools.\n"
                "Guessing from memory without calling a graph tool will score 0 on trajectory\n"
                "even if the answer is correct."
            )

        return (
            f"{base}\n\n"
            f"IMPORTANT: This is an absence question. Search the corpus before concluding absence. "
            f"If you find clear evidence, stop immediately and answer. "
            f"Only conclude absence after you have checked the expected locations."
        )

    def _build_user_prompt(self, question: dict) -> str:
        qtype = question["question_type"]
        allowed_links = self._allowed_links_str(question)

        schema = {
            "PERSPECTIVE": (
                '{"could_actor_have_known": bool, "reasoning": "str", '
                '"evidence_artifacts": ["id", ...], "blocked_subsystems": ["str", ...]}'
            ),
            "COUNTERFACTUAL": (
                '{"outcome_changed": bool, '
                f'"causal_mechanism": "<one of: {allowed_links}>", '
                '"mechanism": "str", "actors": ["name", ...], "reasoning": "str"}'
            ),
            "SILENCE": (
                '{"exists": bool, "answer": "yes|no", '
                '"reasoning": "what you searched and found"}'
            ),
            "GRAPH": (
                '{"answer": "str", "reasoning": "str", '
                '"evidence_nodes": ["name", ...], "evidence_days": [int, ...]}'
            ),
        }[qtype]

        return f"{question.get('question_text') or question.get('question_prose', '')}\n\nRespond with JSON:\n{schema}"

    _CAUSAL_LINK_TAXONOMY: Dict[str, str] = {
        "involves_gap": "incident ← knowledge gap",
        "recurrence_of": "incident ← prior unresolved incident",
        "spawned_doc": "confluence ← design discussion",
        "email_dropped": "communication failure ← routing gap",
        "sf_ownership_lapsed": "CRM gap ← employee departure",
        "zd_escalation_source": "incident ← support ticket escalation",
        "blocker_flagged": "blocker → delayed progress",
        "incident_coordination": "incident → external contact",
        "departure_reassignment": "departure → ticket/escalation shift",
        "assignment_domain_mismatch": "planning mismatch → knowledge gap → incident",
        "jira_from_vendor_email": "Jira ticket ← inbound vendor email",
        "jira_from_customer_email": "Jira ticket ← inbound customer email",
        "customer_escalation_relayed": "customer escalation ← support email routed",
        "incident_handoff": "escalation chain ← forced handoff on departure",
        "pr_gap_detected": "knowledge gap ← undocumented domain in PR review",
        "async_gap_detected": "knowledge gap ← undocumented domain in async thread",
        "centrality_vacuum": "knowledge gap ← key person departure",
        "sf_stage_advanced_by_customer": "CRM stage advanced ← inbound customer email",
        "feature_request_fyi": "product FYI ← inbound feature request",
        "proactive_outreach_from_crm_signal": "outreach ← at-risk CRM signal",
        "ticket_completion_notifies_lead": "lead notified ← dependent ticket done",
        "org_collision_tension": "org tension ← overlapping responsibilities",
        "postmortem_from_incident": "postmortem ← incident resolved",
        "incident_triggers_risk_flag": "SF risk flag ← active incident",
        "review_triggers_revision": "PR revision ← code review",
        "hire_fills_knowledge_gap": "gap closed ← new hire expertise",
        "escalation_from_zendesk": "incident ← Zendesk escalation",
        "doc_gap_detected": "knowledge gap ← undocumented domain in documentation review",
    }

    _LINK_DISTRACTORS: Dict[str, List[str]] = {
        "involves_gap": ["recurrence_of", "assignment_domain_mismatch"],
        "recurrence_of": ["involves_gap", "blocker_flagged"],
        "spawned_doc": ["involves_gap", "incident_coordination"],
        "email_dropped": ["sf_ownership_lapsed", "incident_coordination"],
        "sf_ownership_lapsed": ["departure_reassignment", "email_dropped"],
        "zd_escalation_source": ["involves_gap", "incident_coordination"],
        "doc_gap_detected": ["pr_gap_detected", "async_gap_detected"],
        "blocker_flagged": ["recurrence_of", "involves_gap"],
        "incident_coordination": ["zd_escalation_source", "blocker_flagged"],
        "departure_reassignment": ["sf_ownership_lapsed", "incident_handoff"],
        "assignment_domain_mismatch": ["involves_gap", "recurrence_of"],
        "jira_from_vendor_email": ["jira_from_customer_email", "incident_coordination"],
        "jira_from_customer_email": [
            "jira_from_vendor_email",
            "customer_escalation_relayed",
        ],
        "customer_escalation_relayed": [
            "zd_escalation_source",
            "jira_from_customer_email",
        ],
        "incident_handoff": ["departure_reassignment", "centrality_vacuum"],
        "pr_gap_detected": ["async_gap_detected", "involves_gap"],
        "async_gap_detected": ["pr_gap_detected", "involves_gap"],
        "centrality_vacuum": ["involves_gap", "departure_reassignment"],
        "sf_stage_advanced_by_customer": [
            "proactive_outreach_from_crm_signal",
            "feature_request_fyi",
        ],
        "feature_request_fyi": [
            "sf_stage_advanced_by_customer",
            "jira_from_customer_email",
        ],
        "proactive_outreach_from_crm_signal": [
            "sf_stage_advanced_by_customer",
            "incident_triggers_risk_flag",
        ],
        "ticket_completion_notifies_lead": [
            "blocker_flagged",
            "departure_reassignment",
        ],
        "org_collision_tension": ["centrality_vacuum", "assignment_domain_mismatch"],
        "postmortem_from_incident": ["spawned_doc", "involves_gap"],
        "incident_triggers_risk_flag": [
            "proactive_outreach_from_crm_signal",
            "escalation_from_zendesk",
        ],
        "review_triggers_revision": ["pr_gap_detected", "blocker_flagged"],
        "hire_fills_knowledge_gap": ["centrality_vacuum", "involves_gap"],
        "escalation_from_zendesk": ["zd_escalation_source", "incident_coordination"],
    }

    def _trimmed_taxonomy(self, link_type: str) -> str:
        if link_type and link_type in self._CAUSAL_LINK_TAXONOMY:
            distractors = self._LINK_DISTRACTORS.get(link_type, [])
            relevant = {link_type} | set(distractors)
            return "\n".join(
                f"- {k}: {v}"
                for k, v in self._CAUSAL_LINK_TAXONOMY.items()
                if k in relevant
            )
        return "\n".join(f"- {k}: {v}" for k, v in self._CAUSAL_LINK_TAXONOMY.items())

    def _allowed_links_str(self, question: dict) -> str:
        # Always return the full set — narrowing leaks the answer.
        return ", ".join(self._CAUSAL_LINK_TAXONOMY.keys())

    _RETRYABLE_ERRORS = (
        "InternalServerException",
        "ThrottlingException",
        "ServiceUnavailableException",
        "ModelTimeoutException",
    )
    _MAX_RETRIES = 5
    _RETRY_BASE_DELAY = 2.0  # seconds; doubles each attempt

    def _converse_with_retry(self, kwargs: Dict[str, Any]) -> dict:
        """
        Calls bedrock.converse with exponential backoff for transient errors.
        Raises the final exception if all retries are exhausted.
        """
        import random

        delay = self._RETRY_BASE_DELAY
        last_exc: Exception = RuntimeError("No attempts made")

        for attempt in range(self._MAX_RETRIES):
            try:
                if self._call_delay > 0:
                    time.sleep(self._call_delay)
                return self._bedrock.converse(**kwargs)
            except Exception as exc:
                last_exc = exc
                exc_str = str(exc)
                retryable = any(tag in exc_str for tag in self._RETRYABLE_ERRORS)
                if not retryable:
                    raise
                jitter = random.uniform(0, delay * 0.25)
                wait = delay + jitter
                logger.warning(
                    f"  Bedrock transient error (attempt {attempt + 1}/{self._MAX_RETRIES}), "
                    f"retrying in {wait:.1f}s: {exc}"
                )
                time.sleep(wait)
                delay = min(delay * 2, 60.0)

        raise last_exc

    def _strip_confluence_content(self, messages: list, seen: set) -> list:
        """Return messages with confluence page content stripped from all but the first occurrence.

        Bug fix: the original implementation immediately reassigned `seen` to a new
        empty set, so deduplication never occurred and the context window grew
        unboundedly on repeated confluence retrievals.
        """
        # Do NOT reassign `seen` — it is passed in by reference so the caller's
        # set accumulates across multiple calls within the same agent loop.
        stripped = []
        for msg in messages:
            if msg["role"] != "user":
                stripped.append(msg)
                continue
            new_content = []
            for block in msg.get("content", []):
                if "toolResult" not in block:
                    new_content.append(block)
                    continue
                tool_result = block["toolResult"]
                result_text = tool_result.get("content", [{}])[0].get("text", "")
                try:
                    doc = json.loads(result_text)
                    page_id = doc.get("id", "")
                    if page_id and page_id.startswith("CONF-") and "content" in doc:
                        if page_id in seen:
                            doc.pop("content")
                            doc["_note"] = "content already provided in earlier step"
                            block = {
                                "toolResult": {
                                    **tool_result,
                                    "content": [{"text": json.dumps(doc)}],
                                }
                            }
                        else:
                            seen.add(page_id)
                except (json.JSONDecodeError, AttributeError):
                    pass
                new_content.append(block)
            stripped.append({**msg, "content": new_content})
        return stripped

    def _run_agent(self, question: dict, tools: GatedTools) -> AgentTrajectory:
        """
        Runs the agent against the question using the Bedrock Converse API.
        Returns a populated AgentTrajectory.
        """

        qtype = question["question_type"]
        trajectory = AgentTrajectory(
            question_id=question["question_id"],
            question_type=qtype,
        )

        system_prompt = self._build_system_prompt(question)
        user_prompt = self._build_user_prompt(question)
        tool_defs = self._selected_tool_defs(question)
        dispatch = self._build_tool_dispatch(tools)

        messages = [
            {"role": "user", "content": [{"text": user_prompt}]},
        ]

        converse_kwargs: Dict[str, Any] = {
            "modelId": self._model,
            "system": [{"text": system_prompt}],
            "messages": messages,
            "inferenceConfig": {"temperature": 0.0, "maxTokens": 4096},
            "serviceTier": {"type": self._service_tier},
        }

        if tool_defs:
            converse_kwargs["toolConfig"] = {
                "tools": tool_defs,
            }
            if not self._model.startswith("us.anthropic."):
                converse_kwargs["toolConfig"]["toolChoice"] = {"auto": {}}
            elif self._model.startswith("us.anthropic."):
                converse_kwargs["additionalModelRequestFields"] = {
                    "tool_choice": {"type": "auto", "disable_parallel_tool_use": True}
                }
            if self._model.startswith(("qwen.qwen3", "mistral.mistral")):
                converse_kwargs["additionalModelRequestFields"] = {
                    "parallel_tool_calls": False
                }

        _FINAL_ANSWER_TOOL_NAME = "final_answer"
        _FINAL_ANSWER_SCHEMAS: Dict[str, dict] = {
            "PERSPECTIVE": {
                "type": "object",
                "properties": {
                    "could_actor_have_known": {"type": "boolean"},
                    "reasoning": {"type": "string"},
                    "evidence_artifacts": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "blocked_subsystems": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "could_actor_have_known",
                    "reasoning",
                    "evidence_artifacts",
                    "blocked_subsystems",
                ],
            },
            "COUNTERFACTUAL": {
                "type": "object",
                "properties": {
                    "outcome_changed": {"type": "boolean"},
                    "causal_mechanism": {"type": "string"},
                    "mechanism": {"type": "string"},
                    "actors": {"type": "array", "items": {"type": "string"}},
                    "reasoning": {"type": "string"},
                },
                "required": [
                    "outcome_changed",
                    "causal_mechanism",
                    "mechanism",
                    "actors",
                    "reasoning",
                ],
            },
            "SILENCE": {
                "type": "object",
                "properties": {
                    "exists": {"type": "boolean"},
                    "answer": {"type": "string"},
                    "reasoning": {"type": "string"},
                },
                "required": ["exists", "answer", "reasoning"],
            },
            "GRAPH": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                    "reasoning": {"type": "string"},
                    "evidence_nodes": {"type": "array", "items": {"type": "string"}},
                    "evidence_days": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["answer", "reasoning", "evidence_nodes", "evidence_days"],
            },
        }

        t_start = time.time()
        budget_exceeded = False
        total_input = 0
        total_output = 0
        confluence_seen: set = set()
        stop_reason = "end_turn"
        output_message: dict = {"content": []}

        max_steps = self._max_steps
        if qtype == "SILENCE":
            search_space = question.get("expected_search_space", [])
            max_steps = max(self._max_steps, len(search_space) + 5)
        elif qtype == "GRAPH":
            max_steps = max(self._max_steps, 4)

        for step in range(max_steps):
            converse_kwargs["messages"] = self._strip_confluence_content(
                messages, confluence_seen
            )

            if step == max_steps - 1:
                logger.info(f"  [step {step}] forcing final_answer tool for {qtype}")
                final_answer_schema = _FINAL_ANSWER_SCHEMAS.get(
                    qtype,
                    {
                        "type": "object",
                        "properties": {
                            "answer": {"type": "string"},
                            "reasoning": {"type": "string"},
                        },
                        "required": ["answer", "reasoning"],
                    },
                )
                final_answer_tool = {
                    "toolSpec": {
                        "name": "final_answer",
                        "description": (
                            "You have reached your maximum number of steps. "
                            "Call this tool to provide your final answer based on what you have found so far."
                        ),
                        "inputSchema": {"json": final_answer_schema},
                    }
                }

                if self._model.startswith(
                    (
                        "mistral.",
                        "moonshot.",
                        "moonshotai.",
                        "qwen.",
                        "deepseek.",
                        "zai.",
                        "minimax.",
                    )
                ):
                    converse_kwargs["toolConfig"] = {
                        "tools": [final_answer_tool],
                        "toolChoice": {"any": {}},
                    }
                    converse_kwargs.pop("additionalModelRequestFields", None)
                else:
                    converse_kwargs["toolConfig"] = {
                        "tools": [final_answer_tool],
                        "toolChoice": {"tool": {"name": _FINAL_ANSWER_TOOL_NAME}},
                    }
                    converse_kwargs.pop("additionalModelRequestFields", None)

            try:
                response = self._converse_with_retry(converse_kwargs)
            except Exception as exc:
                logger.error(f"  Bedrock Converse error (gave up after retries): {exc}")
                break

            usage = response.get("usage", {})
            total_input += usage.get("inputTokens", 0)
            total_output += usage.get("outputTokens", 0)

            output_message = response["output"]["message"]
            stop_reason = response.get("stopReason", "end_turn")

            messages.append(output_message)
            logger.info(f"  [step {step}] stop_reason: {stop_reason}")
            for block in output_message["content"]:
                if "text" in block:
                    logger.info(f"  [step {step}] agent: {block['text'][:500]}")
                elif "toolUse" in block:
                    logger.info(
                        f"  [step {step}] tool_use: {block['toolUse']['name']}({block['toolUse']['input']})"
                    )

            converse_kwargs["messages"] = messages

            if stop_reason in ("end_turn", "max_tokens"):
                text_parts = [
                    block["text"]
                    for block in output_message["content"]
                    if "text" in block
                ]
                raw = "\n".join(text_parts)
                trajectory.final_answer = self._parse_structured_answer(raw)
                break

            if stop_reason == "tool_use":
                for block in output_message["content"]:
                    if (
                        "toolUse" in block
                        and block["toolUse"]["name"] == _FINAL_ANSWER_TOOL_NAME
                    ):
                        trajectory.final_answer = block["toolUse"].get("input", {})
                        logger.info(
                            f"  [step {step}] final_answer extracted from forced tool call"
                        )
                        break

                if trajectory.final_answer:
                    break

                tool_results = []
                for block in output_message["content"]:
                    if "toolUse" not in block:
                        continue

                    tc = block["toolUse"]
                    fn_name = tc["name"]
                    fn_input = tc["input"]
                    tool_use_id = tc["toolUseId"]

                    handler = dispatch.get(fn_name)
                    if handler:
                        t0 = time.time()
                        try:
                            result = handler(**fn_input)
                        except Exception as exc:
                            result = {"error": str(exc)}
                            tools._call_log.append(
                                ToolCall(
                                    tool_name=fn_name,
                                    arguments=fn_input,
                                    result_ids=[],
                                    result_types=[],
                                    timestamp_requested=None,
                                    timestamp_applied=None,
                                    temporal_drift_days=None,
                                    temporal_drift_violation=False,
                                    horizon_violation=False,
                                    actor_gate_violation=False,
                                    subsystem_violation=False,
                                    returned_empty=True,
                                    latency_ms=(time.time() - t0) * 1000,
                                )
                            )
                    else:
                        result = {"error": f"Unknown tool: {fn_name}"}

                    result_str = (
                        result
                        if isinstance(result, str)
                        else json.dumps(result, default=str)
                    )

                    tool_results.append(
                        {
                            "toolResult": {
                                "toolUseId": tool_use_id,
                                "content": [{"text": result_str}],
                            }
                        }
                    )

                messages.append({"role": "user", "content": tool_results})
                converse_kwargs["messages"] = messages
            else:
                break
        else:
            budget_exceeded = True
            for msg in reversed(messages):
                if msg.get("role") != "assistant":
                    continue
                for block in msg.get("content", []):
                    if "text" in block:
                        trajectory.final_answer = self._parse_structured_answer(
                            block["text"]
                        )
                        break
                if trajectory.final_answer:
                    break
            if not trajectory.final_answer:
                logger.warning(
                    f"  [budget_exceeded] No assistant text found in message history for {question['question_id']}"
                )

        trajectory.total_latency_ms = (time.time() - t_start) * 1000
        trajectory.budget_exceeded = budget_exceeded
        trajectory.tool_calls = list(tools.call_log)
        trajectory.final_answer = trajectory.final_answer or {}
        trajectory.prompt_tokens = total_input
        trajectory.completion_tokens = total_output
        trajectory.total_tokens = total_input + total_output
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
        for i, call in enumerate(trajectory.tool_calls):
            if call.returned_empty and i + 1 < len(trajectory.tool_calls):
                if not trajectory.tool_calls[i + 1].returned_empty:
                    trajectory.dead_ends_recovered += 1

        return trajectory

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
            effect_id = question.get("ground_truth", {}).get("effect_event_id")
            if effect_id:
                try:
                    ev = self._mem._db["events"].find_one({"_id": effect_id})
                    if ev and ev.get("timestamp"):
                        return str(ev["timestamp"])
                except Exception:
                    pass
        if qtype == "GRAPH":
            day = question.get("as_of_day", 1)
            return _business_day_to_date(_SIM_START, day).isoformat()
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
                "budget_exceeded_count": sum(
                    1 for r in rs if r.meta.get("budget_exceeded")
                ),
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
                summary["relevant_search_rate"] = round(
                    sum(
                        1
                        for r in rs
                        if r.meta.get("trajectory_detail", {}).get("composite", 0) > 0
                    )
                    / len(rs),
                    4,
                )
                summary["avg_search_space_coverage"] = mean(
                    [
                        r.meta.get("trajectory_detail", {}).get(
                            "search_space_coverage", 0
                        )
                        for r in rs
                    ]
                )
            elif qtype == "COUNTERFACTUAL":
                total_calls = sum(r.tool_call_count for r in rs)
                total_horizon_violations = sum(
                    r.meta.get("horizon_violations", 0) for r in rs
                )
                cf_violation_rate = (
                    round(total_horizon_violations / total_calls, 4)
                    if total_calls
                    else 0.0
                )
                summary.update(
                    {
                        "horizon_violation_rate": cf_violation_rate,
                        "avg_horizon_violations": mean(
                            [r.meta.get("horizon_violations", 0) for r in rs]
                        ),
                    }
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
                "budget_exceeded_count": sum(
                    1 for r in results if r.meta.get("budget_exceeded")
                ),
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
        default="us.anthropic.claude-sonnet-4-6",
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
        choices=["PERSPECTIVE", "COUNTERFACTUAL", "SILENCE", "GRAPH"],
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
    parser.add_argument(
        "--call-delay",
        type=float,
        default=1.0,
        metavar="SECONDS",
        help="Sleep between LLM calls to avoid throttling (default: 1.0s). "
        "Increase to 2-3 for Opus or if you keep hitting ThrottlingException.",
    )
    parser.add_argument(
        "--service-tier",
        type=str,
        default="default",
        choices=["default", "flex"],
    )
    parser.add_argument(
        "--skip-nli",
        action="store_true",
        default=False,
        help=(
            "Disable NLI-based reasoning validation for zero-shot runs. "
            "Falls back to answer_score only. Useful if transformers/torch unavailable."
        ),
    )

    args = parser.parse_args()

    stem = f"{args.model.replace('/', '_').replace(':', '_')}"
    if args.zero_shot:
        args.out = EVAL_DIR / f"zero_shot_{stem}.json"
    elif args.ungated:
        args.out = EVAL_DIR / f"ungated_{stem}.json"
    else:
        args.out = EVAL_DIR / f"gated_{stem}.json"

    runner = AgenticEvalRunner(
        model=args.model,
        max_steps=args.max_steps,
        ungated=args.ungated,
        zero_shot=args.zero_shot,
        call_delay=args.call_delay,
        service_tier=args.service_tier,
        skip_nli=args.skip_nli,
    )
    runner.run(
        questions_path=args.questions,
        out_path=args.out,
        question_types=args.types,
        max_questions=args.max_questions,
    )
