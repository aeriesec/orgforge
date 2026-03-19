"""
planner_models.py
=================
Data models for the OrgForge DepartmentPlanner system.

These are pure dataclasses — no LLM or engine dependencies.
Import freely from flow.py, day_planner.py, or anywhere else.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


# ─────────────────────────────────────────────────────────────────────────────
# ENGINEER-LEVEL AGENDA
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class AgendaItem:
    """
    A single planned activity for one engineer on one day.
    These are intentions, not guarantees — incidents can compress or defer them.
    """

    activity_type: str  # "ticket_progress", "pr_review", "design_doc",
    # "1on1", "async_question", "mentoring", "deep_work"
    description: str  # human-readable, e.g. "Continue ORG-101 retry logic"
    related_id: Optional[str] = None  # ticket ID, PR ID, or Confluence ID if applicable
    collaborator: List[str] = field(
        default_factory=list
    )  # engineers this touches (drives Slack)
    estimated_hrs: float = 2.0  # rough time weight — used to detect overload
    deferred: bool = False  # set True by incident pressure at runtime
    defer_reason: Optional[str] = None


@dataclass
class EngineerDayPlan:
    """
    One engineer's agenda for the day before incidents land.
    The engine executes items in order, skipping/deferring under incident pressure.
    """

    name: str
    dept: str
    agenda: List[AgendaItem]
    stress_level: int  # from graph_dynamics._stress at plan time
    is_on_call: bool = False
    focus_note: str = (
        ""  # one-sentence LLM colour — e.g. "Jax is in heads-down mode today"
    )

    @property
    def capacity_hrs(self) -> float:
        """Available hours — shrinks under stress and on-call duty."""
        base = 6.0
        if self.is_on_call:
            base -= 1.5
        if self.stress_level > 80:
            base -= 2.0
        elif self.stress_level > 60:
            base -= 1.0
        return max(base, 1.5)

    @property
    def planned_hrs(self) -> float:
        return sum(i.estimated_hrs for i in self.agenda if not i.deferred)

    def apply_incident_pressure(self, incident_title: str, hrs_lost: float = 3.0):
        """
        Called when an incident fires mid-day.
        Defers low-priority items until capacity is restored.
        """
        freed = 0.0
        # Defer in reverse-priority order: deep_work and design_doc first
        defer_order = [
            "deep_work",
            "design_doc",
            "mentoring",
            "1on1",
            "async_question",
            "pr_review",
            "ticket_progress",
        ]
        for activity in defer_order:
            if freed >= hrs_lost:
                break
            for item in self.agenda:
                if item.activity_type == activity and not item.deferred:
                    item.deferred = True
                    item.defer_reason = f"Deferred: {incident_title}"
                    freed += item.estimated_hrs
                    if freed >= hrs_lost:
                        break


# ─────────────────────────────────────────────────────────────────────────────
# DEPARTMENT-LEVEL PLAN
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class CrossDeptSignal:
    """
    A fact from another department's recent activity that should
    influence this department's day. Fed to the DepartmentPlanner prompt.
    """

    source_dept: str
    event_type: str
    summary: str
    day: int
    relevance: str  # "direct" | "indirect" — how strongly it shapes today


@dataclass
class LifecycleContext:
    """
    Recent hire/departure activity surfaced to the DepartmentPlanner.
    Included in the roster section of the planning prompt so the LLM
    can propose onboarding_session or warmup_1on1 events naturally.
    """

    recent_departures: List[Dict[str, Any]]  # [{name, dept, day, domains}]
    recent_hires: List[Dict[str, Any]]  # [{name, dept, day, role}]
    active_gaps: List[str]  # knowledge domain strings still unresolved


@dataclass
class SprintContext:
    """
    Deterministic ticket ownership snapshot built by TicketAssigner before
    any LLM planning occurs.  Injected into every DepartmentPlanner prompt so
    the LLM sees only its legal menu — it never needs to reason about ownership.

    owned_tickets      — tickets already assigned to a specific engineer.
                         LLM must NOT reassign these.
    available_tickets  — unowned tickets the planner may assign freely within
                         this department's capacity.
    in_progress_ids    — subset of owned tickets already being actively worked
                         (status == "In Progress").  Surfaced so the LLM knows
                         what is mid-flight vs. newly started.
    capacity_by_member — {name: available_hrs} pre-computed by TicketAssigner
                         from GraphDynamics stress + on-call status.
    """

    owned_tickets: Dict[str, str]  # {ticket_id: owner_name}
    available_tickets: List[str]  # ticket_ids with no owner yet
    in_progress_ids: List[str]  # owned + status == "In Progress"
    capacity_by_member: Dict[str, float]  # {name: hrs}
    sprint_theme: str = ""


@dataclass
class ProposedEvent:
    """
    An event the LLM DayPlanner wants to fire today.
    Must pass PlanValidator before execution.
    """

    event_type: str  # known type OR novel proposal
    actors: List[str]  # must match real names in org_chart/external_contacts
    rationale: str  # one sentence — why now
    facts_hint: Dict[str, Any]  # seed facts for the executor — NOT invented by LLM
    priority: int  # 1=must fire, 2=fire if capacity, 3=opportunistic
    is_novel: bool = (
        False  # True if LLM proposed an event type not in KNOWN_EVENT_TYPES
    )
    artifact_hint: Optional[str] = None  # e.g. "slack", "jira", "confluence", "email"


@dataclass
class DepartmentDayPlan:
    """
    Full plan for one department on one day.
    Produced by DepartmentPlanner, consumed by the day loop executor.
    """

    dept: str
    theme: str  # dept-specific theme (not org theme)
    engineer_plans: List[EngineerDayPlan]  # one per dept member
    proposed_events: List[ProposedEvent]  # ordered by priority
    cross_dept_signals: List[CrossDeptSignal]  # what influenced this plan
    planner_reasoning: str  # LLM chain-of-thought — for researchers
    day: int
    date: str
    sprint_context: Optional["SprintContext"] = None  # populated by TicketAssigner


# ─────────────────────────────────────────────────────────────────────────────
# ORG-LEVEL PLAN
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class OrgDayPlan:
    """
    Assembled from all DepartmentDayPlans after OrgCoordinator runs.
    This is what the day loop actually executes against.
    """

    org_theme: str
    dept_plans: Dict[str, DepartmentDayPlan]  # keyed by dept name
    collision_events: List[ProposedEvent]  # cross-dept interactions
    coordinator_reasoning: str
    day: int
    date: str
    sprint_contexts: Dict[str, SprintContext] = field(default_factory=dict)

    def all_events_by_priority(self) -> List[ProposedEvent]:
        """Flat list of all events across departments, sorted priority → dept (Eng first)."""
        all_events = list(self.collision_events)
        # Engineering fires first — it's the primary driver
        for dept in sorted(
            self.dept_plans.keys(), key=lambda d: 0 if "eng" in d.lower() else 1
        ):
            all_events.extend(self.dept_plans[dept].proposed_events)
        return sorted(all_events, key=lambda e: e.priority)


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION RESULT
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ValidationResult:
    approved: bool
    event: ProposedEvent
    rejection_reason: Optional[str] = None
    was_novel: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# KNOWN EVENT TYPES
# The validator uses this to flag novel proposals vs known ones.
# Add to this list as the engine gains new capabilities.
# ─────────────────────────────────────────────────────────────────────────────

KNOWN_EVENT_TYPES = {
    # Engineering — operational
    "incident_opened",
    "incident_resolved",
    "escalation_chain",
    "fix_in_progress",
    "postmortem_created",
    "knowledge_gap_detected",
    # Engineering — routine
    "standup",
    "pr_review",
    "ticket_progress",
    "design_discussion",
    "async_question",
    "code_review_comment",
    "deep_work_session",
    # Sprint
    "sprint_planned",
    "retrospective",
    "sprint_goal_updated",
    # Cross-dept
    "leadership_sync",
    "feature_request_from_sales",
    "stability_update_to_sales",
    "hr_checkin",
    "morale_intervention",
    "1on1_scheduled",
    # External
    "external_contact_summarized",
    "vendor_meeting",
    "customer_escalation",
    # Org
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
}
