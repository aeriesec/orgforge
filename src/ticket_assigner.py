"""
ticket_assigner.py
==================
Deterministic ticket assignment for OrgForge.

  Graph-weighted assignment
      Scores every (engineer, ticket) pair using:
        • skill match via embedding cosine similarity (ticket title ↔ engineer expertise)
        • inverse stress  (burnt-out engineers get lighter loads)
        • betweenness centrality penalty (key players shouldn't hoard tickets)
        • recency bonus   (engineer already touched this ticket in a prior sprint)
      Uses scipy linear_sum_assignment (Hungarian algorithm) for globally
      optimal matching. Falls back to greedy round-robin if scipy is absent.

  Two-pass planning
      Pass 1 (this module): builds a fully valid SprintContext with locked
      assignments before any LLM call.
      Pass 2 (DepartmentPlanner): receives SprintContext and only writes
      narrative — it cannot affect who owns what.

The result is a SprintContext injected into every DepartmentPlanner prompt.
The LLM sees only its legal menu; ownership conflicts become structurally
impossible rather than validated-away after the fact.

Skill scoring
-------------
  Engineer expertise strings (joined from persona["expertise"]) and ticket
  titles are both embedded at runtime using the same embedder already wired
  into Memory.  Cosine similarity replaces the old hardcoded _SKILL_KEYWORDS
  dict, so the scorer generalises to any domain or industry defined in
  config.yaml without code changes.

  Engineer vectors are computed once in __init__ and cached in memory.
  Ticket title vectors are cached in a dedicated MongoDB collection
  ("ticket_skill_embeddings") keyed by ticket_id, so each title is only
  embedded once across the full simulation.

Public API
----------
    assigner = TicketAssigner(config, graph_dynamics, mem)
    sprint_ctx = assigner.build(state, dept_members)
    # → SprintContext with owned_tickets, available_tickets, capacity_by_member
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np

from graph_dynamics import GraphDynamics
from memory import Memory
from planner_models import SprintContext

logger = logging.getLogger("orgforge.ticket_assigner")


def _cosine(a: List[float], b: List[float]) -> float:
    """Safe cosine similarity — returns 0.0 if either vector is empty/zero."""
    if not a or not b:
        return 0.0
    va, vb = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom > 1e-9 else 0.0


class TicketAssigner:
    """
    Builds a SprintContext for one department before any LLM planning runs.

    Parameters
    ----------
    config         : the full OrgForge config dict
    graph_dynamics : live GraphDynamics instance (owns stress + betweenness)
    mem            : shared Memory instance (embedder + MongoDB collections)
    """

    def __init__(self, config: dict, graph_dynamics: GraphDynamics, mem: Memory):
        self._config = config
        self._gd = graph_dynamics
        self._mem = mem

        # MongoDB collection for cached ticket-title embeddings.
        # Separate from jira_tickets so we never pollute the ticket docs.
        self._skill_embed_cache = mem._db["ticket_skill_embeddings"]
        self._skill_embed_cache.create_index([("ticket_id", 1)], unique=True)

        # Engineer expertise vectors — computed once, reused every sprint.
        # Keys added lazily in _expertise_vector() so new-hire personas that
        # arrive mid-sim (org_lifecycle.scheduled_hires) are handled naturally.
        self._engineer_vectors: Dict[str, List[float]] = {}
        self._precompute_engineer_vectors()

    # ── Public ────────────────────────────────────────────────────────────────

    def build(
        self, state, dept_members: List[str], dept_name: str = ""
    ) -> SprintContext:
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
        open_tickets = self._mem.get_open_tickets_for_dept(
            dept_members, dept_name=dept_name
        )

        # Tickets in the sprint with no assignee yet (newly created this sprint)
        unassigned = list(
            self._mem._jira.find(
                {"assignee": None, "dept": dept_name, "status": {"$ne": "Done"}}
            )
        )

        # Already-owned tickets stay owned — we only re-assign unassigned ones
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
                    self._mem.upsert_ticket(ticket)

        in_progress = [
            t["id"] for t in open_tickets if t.get("status") == "In Progress"
        ]

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
        )

    # ── Capacity ──────────────────────────────────────────────────────────────

    def _compute_capacity(self, members: List[str], state) -> Dict[str, float]:
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

        Cost matrix  [engineers × tickets]
        Each cell = -(skill_score × stress_score × centrality_factor)
        Negative because linear_sum_assignment minimises cost.
        """
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
            cent_factor = 1.0 - (cent * 0.3)  # key players penalised 0–30%

            for j, ticket in enumerate(tickets):
                skill = self._skill_score(eng, ticket)
                recency = 1.2 if ticket["id"] in ticket_history.get(eng, set()) else 1.0
                score = skill * stress_score * cent_factor * recency
                cost[i][j] = -score  # negate → minimise

        row_ind, col_ind = linear_sum_assignment(cost)

        result: Dict[str, str] = {}
        assigned_eng_load: Dict[str, float] = {m: 0.0 for m in members}

        for i, j in zip(row_ind, col_ind):
            eng = members[i]
            tkt = tickets[j]
            pts = tkt.get("story_points", 2)
            est_hrs = pts * 0.75  # rough 0.75 hr/point heuristic

            if assigned_eng_load[eng] + est_hrs <= capacity[eng]:
                result[tkt["id"]] = eng
                assigned_eng_load[eng] += est_hrs
            else:
                # Over capacity — leave ticket unassigned this day
                logger.debug(f"[assigner] {eng} over capacity, skipping {tkt['id']}")

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
            # Try each member starting from current round-robin position
            for offset in range(len(members)):
                eng = members[(idx + offset) % len(members)]
                if load[eng] + est_hrs <= capacity[eng]:
                    result[ticket["id"]] = eng
                    load[eng] += est_hrs
                    idx = (idx + 1) % len(members)
                    break
        return result

    # ── Skill scoring (embedding-based) ──────────────────────────────────────

    def _skill_score(self, engineer: str, ticket: dict) -> float:
        """
        Returns a [0.5, 1.5] score representing how well the engineer's
        expertise matches the ticket title.

        Method
        ------
        1. Engineer expertise is embedded once at init (or lazily on first use
           for mid-sim hires) and stored in self._engineer_vectors.
        2. Ticket title is embedded on first encounter and cached in MongoDB
           ("ticket_skill_embeddings"), so re-scoring the same ticket in a
           later sprint costs zero embed calls.
        3. Cosine similarity ∈ [-1, 1] is linearly rescaled to [0.5, 1.5]
           so it's a drop-in replacement for the old keyword ratio.

        Neutral / empty-expertise engineers default to 1.0 (no preference).
        """
        eng_vec = self._expertise_vector(engineer)
        if not eng_vec:
            return 1.0  # no expertise info — treat as neutral

        tkt_vec = self._ticket_title_vector(ticket)
        if not tkt_vec:
            return 1.0  # un-embeddable title — treat as neutral

        similarity = _cosine(eng_vec, tkt_vec)  # [-1.0 … 1.0]
        # Map [-1, 1] → [0.5, 1.5]  (similarity=1 → 1.5, similarity=-1 → 0.5)
        return 0.5 + (similarity + 1.0) / 2.0

    def _precompute_engineer_vectors(self) -> None:
        """
        Embed every known persona's expertise at startup.
        Called once in __init__; new-hire personas picked up lazily via
        _expertise_vector() during the sim.
        """
        from config_loader import PERSONAS  # late import — avoids circular dep

        for name, persona in PERSONAS.items():
            self._engineer_vectors[name] = self._build_expertise_vector(name, persona)

    def _expertise_vector(self, engineer: str) -> List[float]:
        """
        Return (and lazily cache) the expertise embedding for an engineer.
        Handles mid-sim hires whose persona wasn't present at __init__ time.
        """
        if engineer not in self._engineer_vectors:
            from config_loader import PERSONAS, DEFAULT_PERSONA  # late import

            persona = PERSONAS.get(engineer, DEFAULT_PERSONA)
            self._engineer_vectors[engineer] = self._build_expertise_vector(
                engineer, persona
            )
        return self._engineer_vectors[engineer]

    def _build_expertise_vector(self, name: str, persona: dict) -> List[float]:
        """
        Produce a single embedding for an engineer by joining their expertise
        list and style into a short descriptive string.

        Example input  → "backend infra distributed-systems | methodical architect"
        This gives the embedder enough semantic context to differentiate
        a backend specialist from a mobile or design engineer.
        """
        expertise: List[str] = [e.lower() for e in persona.get("expertise", [])]
        style: str = persona.get("style", "").lower()

        if not expertise and not style:
            return []

        text_parts = []
        if expertise:
            text_parts.append(" ".join(expertise))
        if style:
            text_parts.append(style)

        text = " | ".join(text_parts)
        try:
            return self._mem._embed(
                text,
                input_type="search_document",
                caller="ticket_assigner.expertise",
                doc_id=name,
                doc_type="engineer_expertise",
            )
        except Exception as exc:
            logger.warning(f"[assigner] expertise embed failed for {name}: {exc}")
            return []

    def _ticket_title_vector(self, ticket: dict) -> List[float]:
        """
        Return the embedding for a ticket title, using MongoDB as a
        write-through cache to avoid re-embedding across sprints.

        Cache document schema:
          { ticket_id: str, title: str, embedding: List[float] }
        """
        ticket_id: str = ticket.get("id", "")
        title: str = ticket.get("title", "").strip()

        if not title:
            return []

        # ── Cache read ────────────────────────────────────────────────────────
        cached = self._skill_embed_cache.find_one(
            {"ticket_id": ticket_id}, {"embedding": 1, "_id": 0}
        )
        if cached and cached.get("embedding"):
            return cached["embedding"]

        # ── Cache miss: embed and persist ────────────────────────────────────
        try:
            vector = self._mem._embed(
                title,
                input_type="search_query",
                caller="ticket_assigner.ticket_title",
                doc_id=ticket_id,
                doc_type="ticket_title",
            )
        except Exception as exc:
            logger.warning(
                f"[assigner] ticket title embed failed for {ticket_id!r}: {exc}"
            )
            return []

        if vector:
            try:
                self._skill_embed_cache.update_one(
                    {"ticket_id": ticket_id},
                    {
                        "$set": {
                            "ticket_id": ticket_id,
                            "title": title,
                            "embedding": vector,
                        }
                    },
                    upsert=True,
                )
            except Exception as exc:
                # Non-fatal — we still return the vector for this call
                logger.warning(
                    f"[assigner] ticket embed cache write failed for {ticket_id!r}: {exc}"
                )

        return vector

    # ── History ───────────────────────────────────────────────────────────────

    def _ticket_history(self, state) -> Dict[str, set]:
        """
        Returns {engineer: {ticket_ids they've touched in prior days}}.
        Derived from ticket_actors_today which flow.py accumulates over the sim.
        Also checks jira_tickets assignee history for continuity.
        """
        history: Dict[str, set] = {}
        for ticket in self._mem._jira.find(
            {"assignee": {"$exists": True}}, {"id": 1, "assignee": 1}
        ):
            assignee = ticket.get("assignee")
            if assignee:
                history.setdefault(assignee, set()).add(ticket["id"])
        return history
