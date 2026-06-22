"""
org_dynamics_question_builder.py
=================================
OrgForge Organizational Dynamics Question Builder

Generates questions that require multi-step reasoning over the full sim corpus —
the kind of questions a C-suite executive, new manager, or board analyst would
actually ask. These cannot be answered from question text alone or by defaulting
to a safe answer. Corpus access is mandatory.

Five question categories:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CATEGORY 1 — ATTENTION COST
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
How much productive time was lost to distractions, context switching,
and non-primary activities?

All ground truth is derived from the dept_plans collection. The dept_plan
agenda is the authoritativerecord of how each engineer's day was actually
structured — it captures ticket_progress, deep_work, async_question,
design_discussion, pr_review, 1on1, and deferred items with estimated_hrs for each.

Three question sub-types:
  Q1a — activity_type breakdown by dept per week (ticket vs non-ticket split)
  Q1b — sprint alignment: does the team's activity mix match the sprint theme?
  Q1c — collaborator demand: which engineer appeared most as a collaborator
         in others' agendas, and how did that affect their own ticket output?

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CATEGORY 2 — RESOURCE PRESSURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Who was over-capacity, under-utilized, or misaligned with their
assigned work? Requires dept_plans capacity_by_member vs actual load.

Also uses is_on_call + agenda load to flag engineers carrying both
on-call responsibility and heavy scheduled work on the same day.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CATEGORY 3 — CAUSAL PRESSURE PROPAGATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
How did external pressure (customer complaints, vendor alerts) ripple
through the org into planning decisions and engineering priorities?
Requires tracing inbound_external_email → cross_dept_signals → dept themes.

Linkage uses explicit causal flags (jira_from_customer_email,
customer_escalation_relayed) and artifact-ID cross-reference rather than
substring matching against the full facts blob, to avoid false chains.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CATEGORY 4 — ASSIGNMENT QUALITY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Were tickets assigned to the right engineers given stress, expertise,
and capacity? Requires assignment_scores collection (must be persisted
in sim) + stress history + ticket metadata.

Now includes opportunity-cost questions: given all candidate scores for
a ticket (was_assigned=false rows), how much better was the optimal
assignment vs. what actually happened?

Questions do NOT reveal the stress/capacity finding in the question text;
the agent must discover it from the corpus.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CATEGORY 5 — ORGANIZATIONAL FRICTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Where did cross-dept tension surface, who was involved, and what
drove it? Requires collision events + stress snapshots + dept themes.

Now stores event_summary in ground_truth so the embedding scorer in
org_dynamics_scorer.py can compare against SimEvent.summary rather than
the LLM-generated rationale field.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Scoring model (all categories)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Each question has a structured ground_truth with multiple components.
Answer scoring: weighted sum across components (partial credit).
Trajectory scoring: did the agent retrieve the right artifacts before answering?
Combined: 0.50 answer + 0.50 trajectory (both matter equally).

"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from memory import Memory, SimEvent

logger = logging.getLogger("orgforge.org_dynamics_eval")


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

MAX_ATTENTION_COST = 15
MAX_RESOURCE_PRESSURE = 15
MAX_CAUSAL_PRESSURE = 10
MAX_ASSIGNMENT_QUALITY = 10
MAX_ORG_FRICTION = 10

_TRACK_WEIGHTS = {
    "ATTENTION_COST": {"answer": 0.50, "trajectory": 0.50},
    "RESOURCE_PRESSURE": {"answer": 0.50, "trajectory": 0.50},
    "CAUSAL_PRESSURE": {"answer": 0.50, "trajectory": 0.50},
    "ASSIGNMENT_QUALITY": {"answer": 0.50, "trajectory": 0.50},
    "ORG_FRICTION": {"answer": 0.50, "trajectory": 0.50},
}

# activity_type values from the dept_plan agenda schema
_TICKET_TYPES = {"ticket_progress"}
_NON_TICKET_TYPES = {
    "deep_work",
    "async_question",
    "design_discussion",
    "pr_review",
    "1on1",
    "mentoring",
    "watercooler_chat",
}

# Delivery-oriented keywords used to detect sprint theme misalignment
_DELIVERY_KEYWORDS = {
    "fix",
    "ship",
    "deliver",
    "complete",
    "close",
    "resolve",
    "deploy",
    "release",
    "ticket",
    "sprint",
}

# Explicit causal flags that reliably link an inbound email to a downstream event
_EMAIL_CAUSAL_FLAGS = {
    "jira_from_customer_email",
    "customer_escalation_relayed",
    "customer_email_routed",
    "zd_escalation_source",
    "incident_triggers_risk_flag",
}


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class OrgDynamicsQuestion:
    question_id: str
    category: str
    difficulty: str  # "medium" | "hard"
    day_range: Tuple[int, int]
    question_text: str
    ground_truth: Dict[str, Any]
    evidence_search_space: List[str]
    evidence_plan_ids: List[str] = field(default_factory=list)
    # ^^^ PLAN-{day}-{dept} IDs separated out so the harness can route them to
    # get_dept_plan rather than treating them as generic artifact IDs.
    requires_reasoning: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# BUILDER
# ─────────────────────────────────────────────────────────────────────────────


class OrgDynamicsQuestionBuilder:
    """
    Builds OrgDynamicsQuestion objects from live MongoDB sim data.
    Called from eval_harness.py after the sim has completed.
    """

    def __init__(self, mem: Memory, sim_start: datetime, config: dict):
        self._mem = mem
        self._sim_start = sim_start
        self._config = config
        self._org_chart: Dict[str, List[str]] = config.get("org_chart", {})
        self._leads: Dict[str, str] = config.get("leads", {})
        self._events = mem.get_event_log(from_db=True)
        self._max_day = max((e.day for e in self._events), default=1)

        # Pre-build email_id → downstream event index for CAUSAL_PRESSURE.
        self._email_to_downstream: Dict[str, List[SimEvent]] = (
            self._build_email_downstream_index()
        )

    # ── Public entry point ────────────────────────────────────────────────────

    def build_all(self) -> List[Dict]:
        questions: List[OrgDynamicsQuestion] = []
        questions += self._attention_cost_questions()
        questions += self._resource_pressure_questions()
        questions += self._causal_pressure_questions()
        questions += self._assignment_quality_questions()
        questions += self._org_friction_questions()
        logger.info(
            f"[org_dynamics] {len(questions)} questions built — "
            f"Types: { {q.category for q in questions} }"
        )
        return [self._to_dict(q) for q in questions]

    # ─────────────────────────────────────────────────────────────────────────
    # CAUSAL INDEX — built once, used by CAUSAL_PRESSURE
    # ─────────────────────────────────────────────────────────────────────────

    def _build_email_downstream_index(self) -> Dict[str, List[SimEvent]]:
        """
        Maps each email artifact ID to downstream events that explicitly
        reference it via a causal flag or shared artifact_ids entry.
        Uses explicit criteria instead of substring-matching the facts blob.
        """
        index: Dict[str, List[SimEvent]] = {}

        for email_ev in (e for e in self._events if e.type == "inbound_external_email"):
            email_id = (email_ev.artifact_ids or {}).get("email", "")
            if not email_id:
                continue

            downstream: List[SimEvent] = []
            for ev in self._events:
                if ev.day < email_ev.day or ev.day > email_ev.day + 5:
                    continue

                if ev.type in _EMAIL_CAUSAL_FLAGS:
                    artifact_vals: Set[str] = set()
                    for v in (ev.artifact_ids or {}).values():
                        if isinstance(v, list):
                            artifact_vals.update(str(x) for x in v)
                        elif v:
                            artifact_vals.add(str(v))
                    if email_id in artifact_vals or email_id in str(
                        ev.facts.get("source_email_id", "")
                    ):
                        downstream.append(ev)
                        continue

                for v in (ev.artifact_ids or {}).values():
                    if email_id in str(v):
                        downstream.append(ev)
                        break

            if downstream:
                index[email_id] = downstream

        return index

    # ─────────────────────────────────────────────────────────────────────────
    # SHARED PLAN HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _agenda_hours(
        self, ep: dict, activity_types: Optional[Set[str]] = None
    ) -> float:
        """Sum estimated_hrs for non-deferred agenda items, optionally filtered by type."""
        return sum(
            float(item.get("estimated_hrs", 0))
            for item in ep.get("agenda", [])
            if not item.get("deferred")
            and (activity_types is None or item.get("activity_type") in activity_types)
        )

    # ─────────────────────────────────────────────────────────────────────────
    # CATEGORY 1 — ATTENTION COST
    # ─────────────────────────────────────────────────────────────────────────

    def _attention_cost_questions(self) -> List[OrgDynamicsQuestion]:
        questions: List[OrgDynamicsQuestion] = []
        questions += self._attention_ticket_split_questions()
        questions += self._attention_sprint_alignment_questions()
        questions += self._attention_collaborator_demand_questions()
        random.shuffle(questions)
        return questions[:MAX_ATTENTION_COST]

    # Q1a — ticket vs non-ticket split by dept/week

    def _attention_ticket_split_questions(self) -> List[OrgDynamicsQuestion]:
        questions: List[OrgDynamicsQuestion] = []
        cap = MAX_ATTENTION_COST // 3

        for dept in self._org_chart:
            for day_start in range(1, self._max_day - 4, 7):
                day_end = min(day_start + 6, self._max_day)
                result = self._compute_activity_breakdown(dept, day_start, day_end)
                if not result:
                    continue

                by_type, plan_ids = result
                total_hrs = sum(by_type.values())
                ticket_hrs = by_type.get("ticket_progress", 0.0)
                non_ticket = total_hrs - ticket_hrs

                if total_hrs < 5:
                    continue
                pct_non_ticket = round(non_ticket / total_hrs * 100, 1)
                if pct_non_ticket < 20:
                    continue

                questions.append(
                    OrgDynamicsQuestion(
                        question_id=f"attention_split_{dept}_D{day_start}_D{day_end}",
                        category="ATTENTION_COST",
                        difficulty="hard",
                        day_range=(day_start, day_end),
                        question_text=(
                            f"During the week of Day {day_start} to Day {day_end}, "
                            f"what fraction of the {dept} team's planned working hours "
                            f"were allocated to activities other than direct ticket "
                            f"progress, and what does this suggest about the team's "
                            f"capacity that week?"
                        ),
                        ground_truth={
                            "dept": dept,
                            "ticket_hours": round(ticket_hrs, 2),
                            "non_ticket_hours": round(non_ticket, 2),
                            "pct_non_ticket": pct_non_ticket,
                            "activity_breakdown": {
                                k: round(v, 2) for k, v in by_type.items()
                            },
                            "day_range": [day_start, day_end],
                        },
                        evidence_search_space=plan_ids,
                        evidence_plan_ids=plan_ids,
                    )
                )

                if len(questions) >= cap:
                    break
            if len(questions) >= cap:
                break

        return questions

    def _compute_activity_breakdown(
        self, dept: str, day_start: int, day_end: int
    ) -> Optional[Tuple[Dict[str, float], List[str]]]:
        """
        Returns ({activity_type: total_hrs}, plan_ids) for a dept/window.
        Reads directly from dept_plans engineer_plans[].agenda[].
        Deferred items are excluded.
        """
        plans = list(
            self._mem._db["dept_plans"].find(
                {"dept": dept, "day": {"$gte": day_start, "$lte": day_end}},
                {"_id": 0, "engineer_plans": 1, "day": 1, "dept": 1},
            )
        )
        if not plans:
            return None

        by_type: Dict[str, float] = {}
        plan_ids = [f"PLAN-{p['day']}-{p['dept']}" for p in plans]

        for plan in plans:
            for ep in plan.get("engineer_plans", []):
                for item in ep.get("agenda", []):
                    if item.get("deferred"):
                        continue
                    atype = item.get("activity_type", "other")
                    hrs = float(item.get("estimated_hrs", 0))
                    by_type[atype] = by_type.get(atype, 0.0) + hrs

        return by_type, plan_ids

    # Q1b — sprint theme alignment

    def _attention_sprint_alignment_questions(self) -> List[OrgDynamicsQuestion]:
        """
        Flags weeks where the sprint theme implies delivery work but the actual
        activity breakdown shows little or no ticket_progress.

        The agent must read both the dept_plan theme and the agenda breakdown
        to assess alignment — neither alone is sufficient.
        """
        questions: List[OrgDynamicsQuestion] = []
        cap = MAX_ATTENTION_COST // 3

        for dept in self._org_chart:
            for day_start in range(1, self._max_day - 4, 7):
                day_end = min(day_start + 6, self._max_day)
                result = self._compute_activity_breakdown(dept, day_start, day_end)
                if not result:
                    continue

                by_type, plan_ids = result
                total_hrs = sum(by_type.values())
                ticket_hrs = by_type.get("ticket_progress", 0.0)
                if total_hrs < 5:
                    continue

                plan_doc = self._mem._db["dept_plans"].find_one(
                    {"dept": dept, "day": {"$gte": day_start, "$lte": day_end}},
                    {"_id": 0, "theme": 1},
                )
                theme = (plan_doc or {}).get("theme", "")
                if not theme:
                    continue

                theme_implies_delivery = any(
                    kw in theme.lower() for kw in _DELIVERY_KEYWORDS
                )
                pct_ticket = (
                    round(ticket_hrs / total_hrs * 100, 1) if total_hrs else 0.0
                )

                if not (theme_implies_delivery and pct_ticket < 15.0):
                    continue

                no_ticket_count = self._count_engineers_without_ticket_work(
                    dept, day_start, day_end
                )

                questions.append(
                    OrgDynamicsQuestion(
                        question_id=f"attention_alignment_{dept}_D{day_start}_D{day_end}",
                        category="ATTENTION_COST",
                        difficulty="hard",
                        day_range=(day_start, day_end),
                        question_text=(
                            f"The {dept} team's sprint theme for the week of "
                            f"Day {day_start} to Day {day_end} implied delivery work. "
                            f"Looking at how the team actually spent their time that week, "
                            f"how well did their planned activities align with that goal, "
                            f"and how many engineers had no ticket work scheduled at all?"
                        ),
                        ground_truth={
                            "dept": dept,
                            "sprint_theme": theme,
                            "pct_ticket_hours": pct_ticket,
                            "ticket_hours": round(ticket_hrs, 2),
                            "total_hours": round(total_hrs, 2),
                            "engineers_with_no_ticket_work": no_ticket_count,
                            "aligned": False,
                            "day_range": [day_start, day_end],
                        },
                        evidence_search_space=plan_ids,
                        evidence_plan_ids=plan_ids,
                    )
                )

                if len(questions) >= cap:
                    break
            if len(questions) >= cap:
                break

        return questions

    def _count_engineers_without_ticket_work(
        self, dept: str, day_start: int, day_end: int
    ) -> int:
        plans = list(
            self._mem._db["dept_plans"].find(
                {"dept": dept, "day": {"$gte": day_start, "$lte": day_end}},
                {"_id": 0, "engineer_plans": 1},
            )
        )
        all_engineers: Set[str] = set()
        with_ticket_work: Set[str] = set()

        for plan in plans:
            for ep in plan.get("engineer_plans", []):
                name = ep.get("name", "")
                if not name:
                    continue
                all_engineers.add(name)
                if any(
                    item.get("activity_type") == "ticket_progress"
                    and not item.get("deferred")
                    for item in ep.get("agenda", [])
                ):
                    with_ticket_work.add(name)

        return len(all_engineers - with_ticket_work)

    # Q1c — collaborator demand

    def _attention_collaborator_demand_questions(self) -> List[OrgDynamicsQuestion]:
        """
        Reads the collaborator[] field on each agenda item.
        Finds who was most requested as a collaborator by colleagues,
        then checks how many ticket_progress hours they had themselves.
        """
        questions: List[OrgDynamicsQuestion] = []
        cap = MAX_ATTENTION_COST // 3

        for dept in self._org_chart:
            for day_start in range(1, self._max_day - 4, 7):
                day_end = min(day_start + 6, self._max_day)
                result = self._compute_collaborator_demand(dept, day_start, day_end)
                if not result:
                    continue

                top_collab, demand_count, own_ticket_hrs, plan_ids = result
                if demand_count < 2:
                    continue

                questions.append(
                    OrgDynamicsQuestion(
                        question_id=f"attention_collab_{top_collab}_D{day_start}_D{day_end}",
                        category="ATTENTION_COST",
                        difficulty="hard",
                        day_range=(day_start, day_end),
                        question_text=(
                            f"Between Day {day_start} and Day {day_end} in the {dept} team, "
                            f"which engineer was most frequently listed as a collaborator "
                            f"in their colleagues' planned work, and how many hours did "
                            f"that engineer themselves spend on direct ticket progress?"
                        ),
                        ground_truth={
                            "engineer": top_collab,
                            "dept": dept,
                            "collaborator_demand_count": demand_count,
                            "own_ticket_hours": round(own_ticket_hrs, 2),
                            "day_range": [day_start, day_end],
                        },
                        evidence_search_space=plan_ids,
                        evidence_plan_ids=plan_ids,
                    )
                )

                if len(questions) >= cap:
                    break
            if len(questions) >= cap:
                break

        return questions

    def _compute_collaborator_demand(
        self, dept: str, day_start: int, day_end: int
    ) -> Optional[Tuple[str, int, float, List[str]]]:
        """
        Returns (top_collaborator, demand_count, own_ticket_hrs, plan_ids).
        demand_count = number of times an engineer appears in others' collaborator lists.
        """
        plans = list(
            self._mem._db["dept_plans"].find(
                {"dept": dept, "day": {"$gte": day_start, "$lte": day_end}},
                {"_id": 0, "engineer_plans": 1, "day": 1, "dept": 1},
            )
        )
        if not plans:
            return None

        plan_ids: List[str] = [f"PLAN-{p['day']}-{p['dept']}" for p in plans]
        collab_demand: Dict[str, int] = {}
        own_ticket_hrs: Dict[str, float] = {}

        for plan in plans:
            for ep in plan.get("engineer_plans", []):
                owner = ep.get("name", "")
                if not owner:
                    continue
                for item in ep.get("agenda", []):
                    if item.get("deferred"):
                        continue
                    for collab in item.get("collaborator", []):
                        if collab and collab != owner:
                            collab_demand[collab] = collab_demand.get(collab, 0) + 1
                    if item.get("activity_type") == "ticket_progress":
                        own_ticket_hrs[owner] = own_ticket_hrs.get(owner, 0.0) + float(
                            item.get("estimated_hrs", 0)
                        )

        if not collab_demand:
            return None

        top = max(collab_demand, key=lambda e: collab_demand[e])
        return top, collab_demand[top], own_ticket_hrs.get(top, 0.0), plan_ids

    # ─────────────────────────────────────────────────────────────────────────
    # CATEGORY 2 — RESOURCE PRESSURE
    # ─────────────────────────────────────────────────────────────────────────

    def _resource_pressure_questions(self) -> List[OrgDynamicsQuestion]:
        questions: List[OrgDynamicsQuestion] = []
        questions += self._resource_overcapacity_questions()
        questions += self._resource_crossdept_questions()
        questions += self._resource_oncall_overload_questions()
        random.shuffle(questions)
        return questions[:MAX_RESOURCE_PRESSURE]

    def _resource_overcapacity_questions(self) -> List[OrgDynamicsQuestion]:
        questions: List[OrgDynamicsQuestion] = []
        cap = MAX_RESOURCE_PRESSURE // 3

        for day in range(1, self._max_day + 1):
            result = self._find_most_over_capacity_engineer(day)
            if not result:
                continue

            engineer, capacity, actual_load, dept, plan_id = result
            overflow = round(actual_load - capacity, 2)
            if overflow < 0.5:
                continue

            questions.append(
                OrgDynamicsQuestion(
                    question_id=f"resource_overcapacity_{engineer}_D{day}",
                    category="RESOURCE_PRESSURE",
                    difficulty="medium",
                    day_range=(day, day),
                    question_text=(
                        f"On Day {day}, which engineer was most over their available "
                        f"capacity based on their planned workload, and by how many hours?"
                    ),
                    ground_truth={
                        "engineer": engineer,
                        "dept": dept,
                        "day": day,
                        "capacity_hours": round(capacity, 2),
                        "planned_hours": round(actual_load, 2),
                        "overflow_hours": overflow,
                        "plan_artifact_id": plan_id,
                    },
                    evidence_search_space=[plan_id],
                    evidence_plan_ids=[plan_id],
                )
            )

            if len(questions) >= cap:
                break

        return questions

    def _resource_crossdept_questions(self) -> List[OrgDynamicsQuestion]:
        questions: List[OrgDynamicsQuestion] = []
        cap = MAX_RESOURCE_PRESSURE // 3

        for day_start in range(1, self._max_day - 4, 7):
            day_end = min(day_start + 6, self._max_day)
            result = self._find_most_cross_dept_actor(day_start, day_end)
            if not result:
                continue

            actor, cross_events, ticket_hrs, event_ids = result
            if not cross_events:
                continue

            questions.append(
                OrgDynamicsQuestion(
                    question_id=f"resource_crossdept_{actor}_D{day_start}_D{day_end}",
                    category="RESOURCE_PRESSURE",
                    difficulty="hard",
                    day_range=(day_start, day_end),
                    question_text=(
                        f"Between Day {day_start} and Day {day_end}, which team member "
                        f"appeared in the most cross-department discussions relative to "
                        f"their ticket workload, and what departments were involved?"
                    ),
                    ground_truth={
                        "actor": actor,
                        "cross_dept_event_count": len(cross_events),
                        "ticket_hours": round(ticket_hrs, 2),
                        "cross_dept_events": event_ids,
                        "day_range": [day_start, day_end],
                    },
                    evidence_search_space=event_ids + ["dept_plans"],
                )
            )

            if len(questions) >= cap:
                break

        return questions

    def _resource_oncall_overload_questions(self) -> List[OrgDynamicsQuestion]:
        """
        Find days where an on-call engineer also had >= 5h of scheduled work.
        Uses is_on_call from the dept_plan engineer_plans entry.
        An engineer carrying on-call duty alongside heavy scheduled work is a
        capacity risk the assignment scorer may not fully capture.
        """
        questions: List[OrgDynamicsQuestion] = []
        cap = MAX_RESOURCE_PRESSURE // 3

        for day in range(1, self._max_day + 1):
            plans = list(
                self._mem._db["dept_plans"].find(
                    {"day": day}, {"_id": 0, "engineer_plans": 1, "dept": 1}
                )
            )
            for plan in plans:
                dept = plan.get("dept", "")
                plan_id = f"PLAN-{day}-{dept}"
                for ep in plan.get("engineer_plans", []):
                    if not ep.get("is_on_call"):
                        continue
                    name = ep.get("name", "")
                    total_load = self._agenda_hours(ep)
                    if total_load < 5.0:
                        continue

                    questions.append(
                        OrgDynamicsQuestion(
                            question_id=f"resource_oncall_{name}_D{day}",
                            category="RESOURCE_PRESSURE",
                            difficulty="medium",
                            day_range=(day, day),
                            question_text=(
                                f"On Day {day}, which engineer in the {dept} team was "
                                f"carrying on-call responsibility, and how did that "
                                f"interact with their scheduled workload that day?"
                            ),
                            ground_truth={
                                "engineer": name,
                                "dept": dept,
                                "day": day,
                                "is_on_call": True,
                                "scheduled_hours": round(total_load, 2),
                                "plan_artifact_id": plan_id,
                            },
                            evidence_search_space=[plan_id],
                            evidence_plan_ids=[plan_id],
                        )
                    )

            if len(questions) >= cap:
                break

        return questions

    def _find_most_over_capacity_engineer(
        self, day: int
    ) -> Optional[Tuple[str, float, float, str, str]]:
        plans = list(
            self._mem._db["dept_plans"].find(
                {"day": day},
                {"_id": 0, "engineer_plans": 1, "dept": 1, "capacity_by_member": 1},
            )
        )
        if not plans:
            return None

        worst = None
        worst_overflow = 0.0

        for plan in plans:
            dept = plan.get("dept", "")
            capacity_map = plan.get("capacity_by_member", {})
            plan_id = f"PLAN-{day}-{dept}"

            for ep in plan.get("engineer_plans", []):
                name = ep.get("name", "")
                capacity = float(capacity_map.get(name, 6.0))
                # Exclude deferred items — they don't consume real capacity
                actual = self._agenda_hours(ep)
                overflow = actual - capacity
                if overflow > worst_overflow:
                    worst_overflow = overflow
                    worst = (name, capacity, actual, dept, plan_id)

        return worst

    def _find_most_cross_dept_actor(
        self, day_start: int, day_end: int
    ) -> Optional[Tuple[str, List, float, List[str]]]:
        cross_events = [
            e
            for e in self._events
            if day_start <= e.day <= day_end
            and e.type
            in (
                "async_question",
                "design_discussion",
                "org_collision_tension",
                "leadership_sync",
                "feature_request_from_sales",
                "stability_update_to_sales",
            )
            and len(
                set(
                    dept
                    for dept, members in self._org_chart.items()
                    for actor in (e.actors or [])
                    if actor in members
                )
            )
            > 1
        ]
        if not cross_events:
            return None

        actor_cross: Dict[str, List] = {}
        for ev in cross_events:
            for actor in ev.actors or []:
                actor_cross.setdefault(actor, []).append(ev)

        if not actor_cross:
            return None

        top_actor = max(actor_cross, key=lambda a: len(actor_cross[a]))
        events = actor_cross[top_actor]

        plans = list(
            self._mem._db["dept_plans"].find(
                {
                    "day": {"$gte": day_start, "$lte": day_end},
                    "engineer_plans.name": top_actor,
                },
                {"_id": 0, "engineer_plans": 1},
            )
        )
        ticket_hrs = sum(
            self._agenda_hours(ep, activity_types=_TICKET_TYPES)
            for plan in plans
            for ep in plan.get("engineer_plans", [])
            if ep.get("name") == top_actor
        )

        event_ids = [
            str((ev.artifact_ids or {}).get("slack_thread", ev.mongo_id or ""))
            for ev in events
        ]
        return top_actor, events, ticket_hrs, event_ids

    # ─────────────────────────────────────────────────────────────────────────
    # CATEGORY 3 — CAUSAL PRESSURE PROPAGATION
    # ─────────────────────────────────────────────────────────────────────────

    def _causal_pressure_questions(self) -> List[OrgDynamicsQuestion]:
        questions: List[OrgDynamicsQuestion] = []

        complaint_emails = [
            e
            for e in self._events
            if e.type == "inbound_external_email"
            and e.facts.get("email_type") in ("complaint", "escalation")
            and e.facts.get("high_priority", False)
        ]

        for email_event in complaint_emails[:MAX_CAUSAL_PRESSURE]:
            day = email_event.day
            source = email_event.facts.get("source", "an external contact")
            org = email_event.facts.get("org", "")
            subject = email_event.facts.get("subject", "")
            email_id = (email_event.artifact_ids or {}).get("email", "")

            downstream_signals = self._email_to_downstream.get(email_id, [])
            if not downstream_signals:
                continue

            theme_shifts = self._find_theme_shifts_after(day, day + 3)
            if not theme_shifts:
                continue

            signal_ids = [
                str(
                    (ev.artifact_ids or {}).get("slack_thread", "")
                    or (ev.artifact_ids or {}).get("jira", "")
                )
                for ev in downstream_signals
            ]
            signal_ids = [s for s in signal_ids if s]

            plan_ids = [f"PLAN-{t['day']}-{t['dept']}" for t in theme_shifts]

            questions.append(
                OrgDynamicsQuestion(
                    question_id=f"causal_pressure_{email_id or 'email'}_D{day}",
                    category="CAUSAL_PRESSURE",
                    difficulty="hard",
                    day_range=(day, day + 3),
                    question_text=(
                        f"A complaint email arrived from {source}"
                        + (f" ({org})" if org else "")
                        + f" on Day {day}"
                        + (f" regarding '{subject[:60]}'" if subject else "")
                        + ". How did this external pressure propagate internally "
                        "over the following days, and which departments adjusted "
                        "their priorities as a result?"
                    ),
                    ground_truth={
                        "trigger_email": email_id,
                        "trigger_day": day,
                        "source": source,
                        "org": org,
                        "downstream_events": signal_ids,
                        "theme_shifts": theme_shifts,
                        "propagation_chain": ([email_id] + signal_ids)
                        if email_id
                        else signal_ids,
                    },
                    evidence_search_space=([email_id] if email_id else [])
                    + signal_ids
                    + plan_ids,
                    evidence_plan_ids=plan_ids,
                )
            )

        random.shuffle(questions)
        return questions[:MAX_CAUSAL_PRESSURE]

    def _find_theme_shifts_after(self, day_start: int, day_end: int) -> List[Dict]:
        plans = list(
            self._mem._db["dept_plans"].find(
                {"day": {"$gte": day_start, "$lte": day_end}},
                {"_id": 0, "day": 1, "dept": 1, "theme": 1, "raw.planner_reasoning": 1},
            )
        )
        return [
            {
                "day": p["day"],
                "dept": p.get("dept", ""),
                "theme": p.get("theme", ""),
                "reasoning": p.get("raw", {}).get("planner_reasoning", ""),
            }
            for p in plans
        ]

    # ─────────────────────────────────────────────────────────────────────────
    # CATEGORY 4 — ASSIGNMENT QUALITY
    # ─────────────────────────────────────────────────────────────────────────

    def _assignment_quality_questions(self) -> List[OrgDynamicsQuestion]:
        questions: List[OrgDynamicsQuestion] = []

        has_scores = self._mem._db["assignment_scores"].count_documents({}) > 0

        if has_scores:
            questions += self._assignment_skill_mismatch_questions()
            questions += self._assignment_opportunity_cost_questions()
            questions += self._assignment_stress_questions()
        else:
            logger.warning(
                "[org_dynamics] assignment_scores collection is empty — "
                "falling back to stress-based assignment questions only. "
                "Persist scores in ticket_assigner._hungarian_assign() for full coverage."
            )
            questions += self._assignment_stress_questions()

        random.shuffle(questions)
        return questions[:MAX_ASSIGNMENT_QUALITY]

    def _assignment_skill_mismatch_questions(self) -> List[OrgDynamicsQuestion]:
        """Find tickets where skill_score was low but assignment happened anyway."""
        questions: List[OrgDynamicsQuestion] = []

        poor_matches = list(
            self._mem._db["assignment_scores"]
            .find(
                {"was_assigned": True, "skill_score": {"$lt": 0.6}},
                {
                    "_id": 0,
                    "engineer": 1,
                    "ticket_id": 1,
                    "skill_score": 1,
                    "stress_score": 1,
                    "composite_score": 1,
                    "day": 1,
                },
            )
            .sort("skill_score", 1)
            .limit(MAX_ASSIGNMENT_QUALITY)
        )

        for match in poor_matches:
            plan_ids = [
                f"PLAN-{match['day']}-Engineering_Backend",
                f"PLAN-{match['day']}-Engineering_Mobile",
            ]
            questions.append(
                OrgDynamicsQuestion(
                    question_id=f"assignment_mismatch_{match['ticket_id']}_D{match['day']}",
                    category="ASSIGNMENT_QUALITY",
                    difficulty="hard",
                    day_range=(match["day"], match["day"]),
                    question_text=(
                        f"On Day {match['day']}, was {match['engineer']} the best "
                        f"available engineer for ticket {match['ticket_id']}? "
                        f"What does the assignment data suggest about the quality "
                        f"of this match given the team's current state?"
                    ),
                    ground_truth={
                        "ticket_id": match["ticket_id"],
                        "assigned_engineer": match["engineer"],
                        "skill_score": round(match["skill_score"], 3),
                        "stress_score": round(match["stress_score"], 3),
                        "composite_score": round(match["composite_score"], 3),
                        "assessment": "poor_skill_match"
                        if match["skill_score"] < 0.5
                        else "suboptimal_skill_match",
                        "day": match["day"],
                    },
                    evidence_search_space=[match["ticket_id"]] + plan_ids,
                    evidence_plan_ids=plan_ids,
                )
            )

        return questions

    def _assignment_opportunity_cost_questions(self) -> List[OrgDynamicsQuestion]:
        """
        Compares assigned engineer's composite_score against the best available
        candidate (was_assigned=false rows for the same ticket).

        Only generates a question when the gap is meaningful (> 0.10).
        The agent must retrieve all candidate rows, compare scores, and quantify
        the cost of the actual decision.
        """
        questions: List[OrgDynamicsQuestion] = []

        assigned_rows = list(
            self._mem._db["assignment_scores"].find(
                {"was_assigned": True},
                {
                    "_id": 0,
                    "engineer": 1,
                    "ticket_id": 1,
                    "composite_score": 1,
                    "skill_score": 1,
                    "stress_score": 1,
                    "centrality_factor": 1,
                    "day": 1,
                },
            )
        )

        for assigned in assigned_rows:
            ticket_id = assigned["ticket_id"]
            assigned_score = assigned["composite_score"]
            day = assigned["day"]

            candidates = list(
                self._mem._db["assignment_scores"].find(
                    {"ticket_id": ticket_id, "was_assigned": False},
                    {
                        "_id": 0,
                        "engineer": 1,
                        "composite_score": 1,
                        "skill_score": 1,
                        "stress_score": 1,
                        "centrality_factor": 1,
                    },
                )
            )
            if not candidates:
                continue

            best = max(candidates, key=lambda c: c["composite_score"])
            opp_cost = round(best["composite_score"] - assigned_score, 4)

            if opp_cost <= 0.10:
                continue

            plan_ids = [
                f"PLAN-{day}-Engineering_Backend",
                f"PLAN-{day}-Engineering_Mobile",
            ]

            questions.append(
                OrgDynamicsQuestion(
                    question_id=f"assignment_opcost_{ticket_id}_D{day}",
                    category="ASSIGNMENT_QUALITY",
                    difficulty="hard",
                    day_range=(day, day),
                    question_text=(
                        f"On Day {day}, ticket {ticket_id} was assigned to "
                        f"{assigned['engineer']}. Looking at all the engineers "
                        f"who were evaluated for this ticket, was this the optimal "
                        f"choice, and if not, what was the best available alternative "
                        f"and how large was the gap?"
                    ),
                    ground_truth={
                        "ticket_id": ticket_id,
                        "assigned_engineer": assigned["engineer"],
                        "assigned_composite": round(assigned_score, 4),
                        "best_candidate": best["engineer"],
                        "best_composite": round(best["composite_score"], 4),
                        "opportunity_cost": opp_cost,
                        "assessment": "suboptimal",
                        "day": day,
                    },
                    evidence_search_space=[ticket_id] + plan_ids,
                    evidence_plan_ids=plan_ids,
                )
            )

            if len(questions) >= MAX_ASSIGNMENT_QUALITY // 2:
                break

        return questions

    def _assignment_stress_questions(self) -> List[OrgDynamicsQuestion]:
        """
        Find cases where a high-stress engineer was assigned a critical ticket.
        Falls back to dept_plans capacity data when assignment_scores unavailable.
        Question text does NOT reveal the stress finding.
        """
        questions: List[OrgDynamicsQuestion] = []

        incidents = [e for e in self._events if e.type == "incident_opened"]

        for incident in incidents[:MAX_ASSIGNMENT_QUALITY]:
            day = incident.day
            ticket_id = (incident.artifact_ids or {}).get("jira", "")
            if not ticket_id:
                continue

            ticket = (
                self._mem._db["jira_tickets"].find_one(
                    {"id": ticket_id}, {"_id": 0, "assignee": 1, "title": 1}
                )
                or {}
            )
            assignee = ticket.get("assignee", "")
            if not assignee:
                continue

            plan = self._mem._db["dept_plans"].find_one(
                {"day": day, "engineer_plans.name": assignee},
                {"_id": 0, "engineer_plans": 1, "capacity_by_member": 1, "dept": 1},
            )
            if not plan:
                continue

            capacity = plan.get("capacity_by_member", {}).get(assignee, 6.0)
            if float(capacity) >= 5.0:
                continue

            plan_id = f"PLAN-{day}-{plan.get('dept', '')}"

            questions.append(
                OrgDynamicsQuestion(
                    question_id=f"assignment_stress_{ticket_id}_D{day}",
                    category="ASSIGNMENT_QUALITY",
                    difficulty="medium",
                    day_range=(day, day),
                    question_text=(
                        f"On Day {day}, ticket {ticket_id} was assigned to {assignee}. "
                        f"Based on the team's state that day, was this the right "
                        f"assignment, and who — if anyone — might have been a better choice?"
                    ),
                    ground_truth={
                        "ticket_id": ticket_id,
                        "assigned_engineer": assignee,
                        "capacity_that_day": round(float(capacity), 2),
                        "stress_indicator": "high"
                        if float(capacity) < 4.0
                        else "elevated",
                        "day": day,
                        "plan_artifact_id": plan_id,
                    },
                    evidence_search_space=[
                        ticket_id,
                        plan_id,
                        (incident.artifact_ids or {}).get("slack_thread", ""),
                    ],
                    evidence_plan_ids=[plan_id],
                )
            )

        return questions

    # ─────────────────────────────────────────────────────────────────────────
    # CATEGORY 5 — ORGANIZATIONAL FRICTION
    # ─────────────────────────────────────────────────────────────────────────

    def _org_friction_questions(self) -> List[OrgDynamicsQuestion]:
        questions: List[OrgDynamicsQuestion] = []

        tension_events = [
            e
            for e in self._events
            if e.type
            in (
                "org_collision_tension",
                "leadership_sync",
                "assignment_domain_mismatch",
            )
            and e.facts.get("tension_level") in ("high", "medium")
        ]

        for ev in tension_events[:MAX_ORG_FRICTION]:
            day = ev.day
            actors = ev.actors or []
            tension = ev.facts.get("tension_level", "medium")
            rationale = ev.facts.get("rationale", "")
            slack_id = (ev.artifact_ids or {}).get("slack_thread", "")

            actor_depts = {
                actor: next(
                    (
                        dept
                        for dept, members in self._org_chart.items()
                        if actor in members
                    ),
                    "Unknown",
                )
                for actor in actors
            }
            depts_involved = list(set(actor_depts.values()))
            plan_ids = [f"PLAN-{day}-{dept}" for dept in depts_involved]

            questions.append(
                OrgDynamicsQuestion(
                    question_id=f"friction_{ev.mongo_id or 'evt'}_D{day}",
                    category="ORG_FRICTION",
                    difficulty="hard" if tension == "high" else "medium",
                    day_range=(day, day),
                    question_text=(
                        f"On Day {day}, there was a {tension}-tension cross-department "
                        f"interaction involving {' and '.join(depts_involved)}. "
                        f"What caused it, who was involved, and what were each "
                        f"department's competing priorities at the time?"
                    ),
                    ground_truth={
                        "day": day,
                        "tension_level": tension,
                        "actors": actors,
                        "depts_involved": depts_involved,
                        "rationale": rationale,
                        # event_summary is the state-machine-grounded string used by
                        # the embedding scorer in org_dynamics_scorer.py as ground truth.
                        # Falls back to rationale if SimEvent.summary is not present.
                        "event_summary": getattr(ev, "summary", None) or rationale,
                        "event_id": ev.mongo_id or "",
                        "slack_artifact": slack_id,
                        "plan_artifacts": plan_ids,
                    },
                    evidence_search_space=(
                        [slack_id] + plan_ids if slack_id else plan_ids
                    ),
                    evidence_plan_ids=plan_ids,
                )
            )

        # Week-level friction pattern
        week_tension = self._compute_weekly_tension()
        if week_tension:
            worst_week_start, tension_count, week_event_ids = week_tension
            worst_week_end = min(worst_week_start + 6, self._max_day)

            questions.append(
                OrgDynamicsQuestion(
                    question_id=f"friction_worst_week_D{worst_week_start}",
                    category="ORG_FRICTION",
                    difficulty="hard",
                    day_range=(worst_week_start, worst_week_end),
                    question_text=(
                        "Which week of the simulation had the highest cross-department "
                        "tension, what drove it, and which departments were most involved?"
                    ),
                    ground_truth={
                        "week_start_day": worst_week_start,
                        "week_end_day": worst_week_end,
                        "tension_event_count": tension_count,
                        "evidence_events": week_event_ids,
                    },
                    evidence_search_space=week_event_ids,
                )
            )

        random.shuffle(questions)
        return questions[:MAX_ORG_FRICTION]

    def _compute_weekly_tension(self) -> Optional[Tuple[int, int, List[str]]]:
        tension_by_week: Dict[int, List] = {}

        for ev in self._events:
            if ev.type not in (
                "org_collision_tension",
                "assignment_domain_mismatch",
                "morale_intervention",
            ):
                continue
            week_start = ((ev.day - 1) // 5) * 5 + 1
            tension_by_week.setdefault(week_start, []).append(ev)

        if not tension_by_week:
            return None

        worst = max(tension_by_week, key=lambda w: len(tension_by_week[w]))
        events = tension_by_week[worst]
        event_ids = [
            str((ev.artifact_ids or {}).get("slack_thread", ev.mongo_id or ""))
            for ev in events
        ]
        return worst, len(events), event_ids

    # ─────────────────────────────────────────────────────────────────────────
    # SERIALISATION
    # ─────────────────────────────────────────────────────────────────────────

    def _to_dict(self, q: OrgDynamicsQuestion) -> Dict:
        return {
            "question_id": q.question_id,
            "question_type": "ORG_DYNAMICS",
            "category": q.category,
            "difficulty": q.difficulty,
            "day_range": list(q.day_range),
            "question_text": q.question_text,
            "ground_truth": q.ground_truth,
            "evidence_search_space": [s for s in q.evidence_search_space if s],
            # Separated so the harness can route these to get_dept_plan tool
            "evidence_plan_ids": [s for s in q.evidence_plan_ids if s],
            "requires_reasoning": q.requires_reasoning,
            "track_weights": _TRACK_WEIGHTS[q.category],
        }
