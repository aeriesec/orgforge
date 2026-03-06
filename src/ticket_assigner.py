"""
ticket_assigner.py
==================
Deterministic ticket assignment for OrgForge.

Implements Options B + C from the architecture discussion:

  Option C — Graph-weighted assignment
      Scores every (engineer, ticket) pair using:
        • skill match against ticket title keywords
        • inverse stress  (burnt-out engineers get lighter loads)
        • betweenness centrality penalty (key players shouldn't hoard tickets)
        • recency bonus   (engineer already touched this ticket in a prior sprint)
      Uses scipy linear_sum_assignment (Hungarian algorithm) for globally
      optimal matching. Falls back to greedy round-robin if scipy is absent.

  Option B — Two-pass planning
      Pass 1 (this module): builds a fully valid SprintContext with locked
      assignments before any LLM call.
      Pass 2 (DepartmentPlanner): receives SprintContext and only writes
      narrative — it cannot affect who owns what.

The result is a SprintContext injected into every DepartmentPlanner prompt.
The LLM sees only its legal menu; ownership conflicts become structurally
impossible rather than validated-away after the fact.

Public API
----------
    assigner = TicketAssigner(config, graph_dynamics)
    sprint_ctx = assigner.build(state, dept_members)
    # → SprintContext with owned_tickets, available_tickets, capacity_by_member
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from graph_dynamics import GraphDynamics
from planner_models import SprintContext

logger = logging.getLogger("orgforge.ticket_assigner")

# Keyword → skill tags used for matching tickets to engineer expertise.
# Extend this list as new ticket themes are added to config.yaml.
_SKILL_KEYWORDS: Dict[str, List[str]] = {
    "retry":       ["resilience", "backend", "infrastructure"],
    "auth":        ["security", "backend", "api"],
    "migration":   ["database", "backend", "infrastructure"],
    "api":         ["api", "backend", "integration"],
    "ui":          ["frontend", "react", "design"],
    "dashboard":   ["frontend", "data", "analytics"],
    "cache":       ["performance", "backend", "infrastructure"],
    "alert":       ["monitoring", "infrastructure", "devops"],
    "test":        ["qa", "testing", "automation"],
    "deploy":      ["devops", "infrastructure", "ci"],
    "refactor":    ["backend", "architecture"],
    "fix":         [],   # neutral — any engineer can take it
    "error":       [],
    "bug":         [],
}


class TicketAssigner:
    """
    Builds a SprintContext for one department before any LLM planning runs.

    Parameters
    ----------
    config         : the full OrgForge config dict
    graph_dynamics : live GraphDynamics instance (owns stress + betweenness)
    """

    def __init__(self, config: dict, graph_dynamics: GraphDynamics):
        self._config = config
        self._gd = graph_dynamics

    # ── Public ────────────────────────────────────────────────────────────────

    def build(self, state, dept_members: List[str]) -> SprintContext:
        """
        Main entry point.  Call once per department, before DepartmentPlanner.plan().

        Returns a SprintContext with:
          • owned_tickets      — final {ticket_id: engineer} mapping
          • available_tickets  — unowned ticket IDs (for the LLM to reference)
          • in_progress_ids    — tickets already "In Progress"
          • capacity_by_member — {name: available_hrs} for every dept member
        """
        capacity = self._compute_capacity(dept_members, state)

        # Tickets assigned to this department but not yet done
        open_tickets = [
            t for t in state.jira_tickets
            if t.get("assignee") in dept_members
            and t.get("status") != "Done"
        ]

        # Tickets in the sprint with no assignee yet (newly created this sprint)
        unassigned = [
            t for t in state.jira_tickets
            if t.get("assignee") is None
            and t.get("sprint") == state.sprint.sprint_number
            and t.get("status") != "Done"
        ]

        # Already-owned tickets stay owned — we only re-assign unassigned ones
        owned: Dict[str, str] = {
            t["id"]: t["assignee"]
            for t in open_tickets
            if t.get("assignee") in dept_members
        }

        if unassigned and dept_members:
            new_assignments = self._assign(unassigned, dept_members, capacity, state)
            owned.update(new_assignments)
            # Write assignments back onto the state tickets so GitSimulator
            # and the JIRA export see the correct owner immediately.
            tid_map = {t["id"]: t for t in state.jira_tickets}
            for tid, owner in new_assignments.items():
                if tid in tid_map:
                    tid_map[tid]["assignee"] = owner

        in_progress = [
            t["id"] for t in open_tickets if t.get("status") == "In Progress"
        ]

        available = [
            t["id"] for t in open_tickets
            if t["id"] not in owned
        ]

        logger.debug(
            f"[assigner] dept members={dept_members} "
            f"owned={len(owned)} available={len(available)} "
            f"capacity={capacity}"
        )

        return SprintContext(
            owned_tickets=owned,
            available_tickets=available,
            in_progress_ids=in_progress,
            capacity_by_member=capacity,
        )

    # ── Capacity ──────────────────────────────────────────────────────────────

    def _compute_capacity(
        self, members: List[str], state
    ) -> Dict[str, float]:
        """
        Available hours per engineer, mirroring EngineerDayPlan.capacity_hrs
        so the two systems stay in sync.
        """
        on_call_name = self._config.get("on_call_engineer")
        capacity: Dict[str, float] = {}
        for name in members:
            stress = self._gd._stress.get(name, 30)
            base = 6.0
            if name == on_call_name:
                base -= 1.5
            if stress > 80:
                base -= 2.0
            elif stress > 60:
                base -= 1.0
            capacity[name] = max(base, 1.5)
        return capacity

    # ── Assignment ────────────────────────────────────────────────────────────

    def _assign(
        self,
        tickets:     List[dict],
        members:     List[str],
        capacity:    Dict[str, float],
        state,
    ) -> Dict[str, str]:
        """
        Assign unowned tickets to engineers using optimal or greedy matching.
        Returns {ticket_id: engineer_name}.
        """
        try:
            return self._hungarian_assign(tickets, members, capacity, state)
        except Exception as exc:
            logger.warning(
                f"[assigner] scipy unavailable or failed ({exc}), "
                f"falling back to greedy round-robin."
            )
            return self._greedy_assign(tickets, members, capacity)

    def _hungarian_assign(
        self,
        tickets:  List[dict],
        members:  List[str],
        capacity: Dict[str, float],
        state,
    ) -> Dict[str, str]:
        """
        Globally optimal assignment via scipy's Hungarian algorithm.

        Cost matrix  [engineers × tickets]
        Each cell = -(skill_score × stress_score × centrality_factor)
        Negative because linear_sum_assignment minimises cost.
        """
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        centrality = self._gd._get_centrality()
        ticket_history = self._ticket_history(state)

        n_eng = len(members)
        n_tkt = len(tickets)
        cost = np.zeros((n_eng, n_tkt))

        for i, eng in enumerate(members):
            stress = self._gd._stress.get(eng, 30)
            stress_score = 1.0 - (stress / 100)
            cent = centrality.get(eng, 0.0)
            cent_factor = 1.0 - (cent * 0.3)   # key players penalised 0–30%

            for j, ticket in enumerate(tickets):
                skill = self._skill_score(eng, ticket)
                recency = 1.2 if ticket["id"] in ticket_history.get(eng, set()) else 1.0
                score = skill * stress_score * cent_factor * recency
                cost[i][j] = -score   # negate → minimise

        row_ind, col_ind = linear_sum_assignment(cost)

        result: Dict[str, str] = {}
        assigned_eng_load: Dict[str, float] = {m: 0.0 for m in members}

        for i, j in zip(row_ind, col_ind):
            eng = members[i]
            tkt = tickets[j]
            pts = tkt.get("story_points", 2)
            est_hrs = pts * 0.75   # rough 0.75 hr/point heuristic

            if assigned_eng_load[eng] + est_hrs <= capacity[eng]:
                result[tkt["id"]] = eng
                assigned_eng_load[eng] += est_hrs
            else:
                # Over capacity — leave ticket unassigned this day
                logger.debug(
                    f"[assigner] {eng} over capacity, skipping {tkt['id']}"
                )

        return result

    def _greedy_assign(
        self,
        tickets:  List[dict],
        members:  List[str],
        capacity: Dict[str, float],
    ) -> Dict[str, str]:
        """
        Fallback: assign in round-robin, skipping over-capacity engineers.
        """
        load: Dict[str, float] = {m: 0.0 for m in members}
        result: Dict[str, str] = {}
        idx = 0
        for ticket in tickets:
            pts = ticket.get("story_points", 2)
            est_hrs = pts * 0.75
            # Try each member starting from current round-robin position
            for offset in range(len(members)):
                eng = members[(idx + offset) % len(members)]
                if load[eng] + est_hrs <= capacity[eng]:
                    result[ticket["id"]] = eng
                    load[eng] += est_hrs
                    idx = (idx + 1) % len(members)
                    break
        return result

    # ── Skill scoring ─────────────────────────────────────────────────────────

    def _skill_score(self, engineer: str, ticket: dict) -> float:
        """
        0.5–1.5 score based on how well the engineer's expertise matches
        the ticket's keywords.  Neutral tickets score 1.0.
        """
        from flow import PERSONAS, DEFAULT_PERSONA  # late import — avoids circular
        persona = PERSONAS.get(engineer, DEFAULT_PERSONA)
        expertise: List[str] = [e.lower() for e in persona.get("expertise", [])]
        if not expertise:
            return 1.0

        title = ticket.get("title", "").lower()
        matched_tags: List[str] = []
        for kw, tags in _SKILL_KEYWORDS.items():
            if kw in title:
                matched_tags.extend(tags)

        if not matched_tags:
            return 1.0   # neutral ticket — no skill preference

        hits = sum(1 for tag in matched_tags if tag in expertise)
        # Scale: 0 hits → 0.5, all hits → 1.5
        ratio = hits / len(matched_tags)
        return 0.5 + ratio

    # ── History ───────────────────────────────────────────────────────────────

    def _ticket_history(self, state) -> Dict[str, set]:
        """
        Returns {engineer: {ticket_ids they've touched in prior days}}.
        Derived from ticket_actors_today which flow.py accumulates over the sim.
        Also checks jira_tickets assignee history for continuity.
        """
        history: Dict[str, set] = {}
        for ticket in state.jira_tickets:
            assignee = ticket.get("assignee")
            if assignee:
                history.setdefault(assignee, set()).add(ticket["id"])
        return history
