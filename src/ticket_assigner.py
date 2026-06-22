"""
ticket_assigner.py
==================
Deterministic ticket assignment for OrgForge.

  Graph-weighted assignment
      Scores every (engineer, ticket) pair using:
        - skill match via BM25 term overlap (ticket title tokens vs engineer
          expertise tags from domain_registry)
        - inverse stress  (burnt-out engineers get lighter loads)
        - betweenness centrality penalty (key players shouldn't hoard tickets)
        - recency bonus   (engineer already touched this ticket in a prior sprint)
      Uses scipy linear_sum_assignment (Hungarian algorithm) for globally
      optimal matching. Falls back to greedy round-robin if scipy is absent.

  Two-pass planning
      Pass 1 (this module): builds a fully valid SprintContext with locked
      assignments before any LLM call.
      Pass 2 (DepartmentPlanner): receives SprintContext and only writes
      narrative -- it cannot affect who owns what.

Skill scoring
-------------
  Engineer expertise tags (from domain_registry) are matched against
  ticket title tokens via a normalized term overlap score. The overlap is
  rescaled to [0.5, 1.5] -- the same range used by the old cosine path --
  so the Hungarian matrix is numerically identical.

Public API
----------
    assigner = TicketAssigner(config, graph_dynamics, mem)
    sprint_ctx = assigner.build(state, dept_members)
"""

from __future__ import annotations

import logging
import os
import re
from typing import Dict, List, Set

import numpy as np
import json as _json

from graph_dynamics import GraphDynamics
from memory import Memory
from planner_models import SprintContext

logger = logging.getLogger("orgforge.ticket_assigner")


_STOP_WORDS: Set[str] = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "in",
    "of",
    "to",
    "for",
    "with",
    "on",
    "at",
    "by",
    "up",
    "is",
    "are",
    "was",
    "be",
    "as",
    "from",
    "into",
    "that",
    "this",
    "it",
    "its",
    "our",
    "we",
    "i",
    "my",
    "their",
    "not",
    "no",
    "new",
    "add",
    "fix",
    "update",
    "improve",
    "support",
    "using",
    "via",
}


def _tokenize(text: str) -> Set[str]:
    """
    Lowercase, split on non-alphanumeric boundaries, strip stop words.
    Returns a set of meaningful tokens.
    """
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {t for t in tokens if len(t) > 1 and t not in _STOP_WORDS}


class TicketAssigner:
    """
    Builds a SprintContext for one department before any LLM planning runs.

    Parameters
    ----------
    config         : the full OrgForge config dict
    graph_dynamics : live GraphDynamics instance (owns stress + betweenness)
    mem            : shared Memory instance (MongoDB collections)
    """

    def __init__(self, config: dict, graph_dynamics: GraphDynamics, mem: Memory):
        self._config = config
        self._gd = graph_dynamics
        self._mem = mem
        self._base = config["simulation"].get("output_dir", "./export")

    def build(
        self, state, dept_members: List[str], dept_name: str = "", on_call: str = ""
    ) -> SprintContext:
        """
        Main entry point. Call once per department, before DepartmentPlanner.plan().

        Returns a SprintContext with:
          - owned_tickets      -- final {ticket_id: engineer} mapping
          - available_tickets  -- unowned ticket IDs (for the LLM to reference)
          - in_progress_ids    -- tickets already "In Progress"
          - capacity_by_member -- {name: available_hrs} for every dept member
        """
        capacity = self._compute_capacity(dept_members, state, on_call=on_call)

        open_tickets = self._mem.get_open_tickets_for_dept(
            dept_members, dept_name=dept_name
        )

        unassigned = list(
            self._mem._jira.find(
                {"assignee": None, "dept": dept_name, "status": {"$ne": "Done"}},
                {"_id": 0},
            )
        )

        owned: Dict[str, str] = {
            t["id"]: t["assignee"]
            for t in open_tickets
            if t.get("assignee") in dept_members
        }

        if unassigned and dept_members:
            new_assignments = self._assign(unassigned, dept_members, capacity, state)
            owned.update(new_assignments)
            for tid, owner in new_assignments.items():
                ticket = self._mem.get_ticket(tid)
                if ticket:
                    ticket["assignee"] = owner
                    if self._base:
                        path = f"{self._base}/jira/{ticket['id']}.json"
                        os.makedirs(os.path.dirname(path), exist_ok=True)
                        with open(path, "w") as f:
                            _json.dump(ticket, f, indent=2)
                    self._mem.upsert_ticket(ticket)

        in_progress = [
            t["id"] for t in open_tickets if t.get("status") == "In Progress"
        ]

        in_review = [t["id"] for t in open_tickets if t.get("status") == "In Review"]

        available = [t["id"] for t in open_tickets if t["id"] not in owned]

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
            in_review=in_review,
        )

    def _compute_capacity(
        self, members: List[str], state, on_call: str = ""
    ) -> Dict[str, float]:
        """
        Available hours per engineer, mirroring EngineerDayPlan.capacity_hrs
        so the two systems stay in sync.
        """
        capacity: Dict[str, float] = {}
        for name in members:
            stress = self._gd._stress.get(name, 30)
            base = 6.0
            if name == on_call:
                base -= 1.5
            if stress > 80:
                base -= 2.0
            elif stress > 60:
                base -= 1.0
            capacity[name] = max(base, 1.5)
        return capacity

    def _assign(
        self,
        tickets: List[dict],
        members: List[str],
        capacity: Dict[str, float],
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
        tickets: List[dict],
        members: List[str],
        capacity: Dict[str, float],
        state,
    ) -> Dict[str, str]:
        """
        Globally optimal assignment via scipy's Hungarian algorithm.

        Cost matrix  [engineers x tickets]
        Each cell = -(skill_score x stress_score x centrality_factor x recency)
        Negative because linear_sum_assignment minimises cost.
        """
        from scipy.optimize import linear_sum_assignment

        centrality = self._gd._get_centrality()
        ticket_history = self._ticket_history(state)

        n_eng = len(members)
        n_tkt = len(tickets)
        cost = np.zeros((n_eng, n_tkt))

        assignment_scores = []
        for i, eng in enumerate(members):
            stress = self._gd._stress.get(eng, 30)
            stress_score = 1.0 - (stress / 100)
            cent = centrality.get(eng, 0.0)
            cent_factor = 1.0 - (cent * 0.3)

            for j, ticket in enumerate(tickets):
                skill = self._skill_score(eng, ticket)
                recency = 1.2 if ticket["id"] in ticket_history.get(eng, set()) else 1.0
                score = skill * stress_score * cent_factor * recency
                cost[i][j] = -score

                assignment_scores.append(
                    {
                        "day": state.day,
                        "engineer": eng,
                        "ticket_id": ticket["id"],
                        "skill_score": skill,
                        "stress_score": stress_score,
                        "centrality_factor": cent_factor,
                        "composite_score": score,
                        "was_assigned": False,
                    }
                )

        row_ind, col_ind = linear_sum_assignment(cost)

        result: Dict[str, str] = {}
        assigned_eng_load: Dict[str, float] = {m: 0.0 for m in members}

        for i, j in zip(row_ind, col_ind):
            eng = members[i]
            tkt = tickets[j]
            pts = tkt.get("story_points", 2)
            est_hrs = pts * 0.75

            if assigned_eng_load[eng] + est_hrs <= capacity[eng]:
                result[tkt["id"]] = eng
                assigned_eng_load[eng] += est_hrs
            else:
                logger.debug(f"[assigner] {eng} over capacity, skipping {tkt['id']}")

        try:
            self._mem._db["assignment_scores"].insert_many(assignment_scores)
        except Exception as exc:
            logger.warning(f"[assigner] assignment_scores insert failed: {exc}")

        return result

    def _greedy_assign(
        self,
        tickets: List[dict],
        members: List[str],
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

            for offset in range(len(members)):
                eng = members[(idx + offset) % len(members)]
                if load[eng] + est_hrs <= capacity[eng]:
                    result[ticket["id"]] = eng
                    load[eng] += est_hrs
                    idx = (idx + 1) % len(members)
                    break
        return result

    def _skill_score(self, engineer: str, ticket: dict) -> float:
        title_tokens = _tokenize(f"{ticket.get('title', '')} {ticket.get('description', '')}")
        if not title_tokens:
            return 1.0

        engineer_tokens = self._mem.get_author_domain_tokens(engineer)
        if not engineer_tokens:
            return 1.0

        overlap_count = len(engineer_tokens & title_tokens)
        raw_overlap = min(1.0, overlap_count / len(title_tokens))
        return 0.5 + raw_overlap

    def evict_engineer(self, name: str) -> None:
        """
        Called by OrgLifecycleManager after _execute_departure completes.
        No-op in the BM25 pipeline since there are no cached vectors to evict.
        Retained for interface compatibility with OrgLifecycleManager.
        """
        logger.debug(f"[assigner] Eviction noted for departed engineer: {name}")

    def register_hire(self, name: str) -> None:
        """
        Called by OrgLifecycleManager after _execute_hire completes.
        No pre-warming needed -- expertise comes from the live persona config
        which is already populated before this is called.
        Retained for interface compatibility with OrgLifecycleManager.
        """
        logger.debug(f"[assigner] New hire registered: {name} (no pre-warm needed)")

    def _ticket_history(self, state) -> Dict[str, set]:
        """
        Returns {engineer: {ticket_ids they've touched in prior days}}.
        Derived from jira_tickets assignee history for continuity across sprints.
        """
        history: Dict[str, set] = {}
        for ticket in self._mem._jira.find(
            {"assignee": {"$exists": True}}, {"_id": 0, "id": 1, "assignee": 1}
        ):
            assignee = ticket.get("assignee")
            if assignee:
                history.setdefault(assignee, set()).add(ticket["id"])
        return history
