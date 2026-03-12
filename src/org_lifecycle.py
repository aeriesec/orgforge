"""
org_lifecycle.py
================
Dynamic hiring and firing for OrgForge.

Principle: The Python engine controls all mutations to org state, the social
graph, and the SimEvent log. LLMs only produce the *narrative prose* (Slack
announcements, onboarding docs, farewell messages). They never touch State.

Three public entry points called from flow.py:
  OrgLifecycleManager.process_departures(day, state, ...)
  OrgLifecycleManager.process_hires(day, state, ...)
  OrgLifecycleManager.get_roster_context()   ← for DepartmentPlanner prompts

Three departure side-effects handled deterministically (no LLM involvement):
  1. JIRA ticket reassignment — orphaned tickets are reassigned to the dept
     lead and transitioned back to "To Do" so they stay in the sprint backlog.
  2. Centrality vacuum — after node removal the betweenness cache is dirtied
     and an immediate recomputation is triggered; the resulting centrality
     shift is used to apply a one-time "vacuum stress" to the neighbours who
     absorbed the departing node's bridging load.
  3. Active incident handoff — if the departing engineer is the named
     responder on any live incident, ownership is forcibly transferred to the
     next person in the Dijkstra escalation chain before the node is removed.

Config schema (add to config.yaml under "org_lifecycle"):
  org_lifecycle:
    scheduled_departures:
      - name: "Jordan"
        day: 15
        reason: "voluntary"          # voluntary | layoff | performance
        role: "Senior Backend Engineer"
        knowledge_domains: ["auth-service", "redis-cache"]
        documented_pct: 0.30
    scheduled_hires:
      - name: "Taylor"
        day: 18
        dept: "Engineering"
        role: "Backend Engineer"
        expertise: ["Python", "Kafka"]
        style: "methodical, asks lots of questions"
        tenure: "new"
    enable_random_attrition: false
    random_attrition_daily_prob: 0.01
    # Stress points applied per unit of centrality gained after a departure
    centrality_vacuum_stress_multiplier: 40
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


from agent_factory import make_agent
from memory import Memory, SimEvent
from graph_dynamics import GraphDynamics

logger = logging.getLogger("orgforge.lifecycle")


# ─────────────────────────────────────────────────────────────────────────────
# DATA TYPES
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class DepartureRecord:
    """Immutable record of an engineer who has left the organisation."""

    name: str
    dept: str
    role: str
    day: int
    reason: str  # voluntary | layoff | performance
    knowledge_domains: List[str]
    documented_pct: float
    peak_stress: int
    edge_snapshot: Dict[str, float]  # {neighbour: weight} before removal
    centrality_at_departure: float = 0.0
    # Populated by the departure engine's side-effect methods
    reassigned_tickets: List[str] = field(default_factory=list)
    incident_handoffs: List[str] = field(default_factory=list)


@dataclass
class HireRecord:
    """An engineer who has joined. Enters the graph at edge_weight_floor."""

    name: str
    dept: str
    day: int
    role: str
    expertise: List[str]
    style: str
    tenure: str = "new"
    warmup_threshold: float = 2.0  # edges below this = "not yet collaborating"


@dataclass
class KnowledgeGapEvent:
    departed_name: str
    domain_hit: str
    triggered_by: str
    triggered_on_day: int
    documented_pct: float


# ─────────────────────────────────────────────────────────────────────────────
# MANAGER
# ─────────────────────────────────────────────────────────────────────────────


class OrgLifecycleManager:
    """
    Owns all mutations to org_chart, personas, GraphDynamics, and the
    State departure/hire ledgers.  N

    LLM usage is limited to a single call: _generate_backfill_name(), which
    asks the worker LLM for a unique name for an unplanned hire. All other
    mutations — graph changes, JIRA reassignment, incident handoff, stress
    propagation — are deterministic. The engine never delegates facts to the LLM.
    """

    def __init__(
        self,
        config: dict,
        graph_dynamics: GraphDynamics,
        mem: Memory,
        org_chart: Dict[str, List[str]],  # mutable — mutated in place
        personas: Dict[str, dict],  # mutable — mutated in place
        all_names: List[str],  # mutable — mutated in place
        leads: Dict[str, str],
        worker_llm=None,
        base_export_dir: str = "",
    ):
        self._cfg = config.get("org_lifecycle", {})
        self._gd = graph_dynamics
        self._mem = mem
        self._org_chart = org_chart
        self._personas = personas
        self._all_names = all_names
        self._leads = leads
        self._llm = worker_llm
        self._base = base_export_dir

        self._departed: List[DepartureRecord] = []
        self._hired: List[HireRecord] = []
        self._gap_events: List[KnowledgeGapEvent] = []
        self._domains_surfaced: Set[str] = set()

        # Build day-keyed lookup tables from config
        self._scheduled_departures: Dict[int, List[dict]] = {}
        for dep in self._cfg.get("scheduled_departures", []):
            self._scheduled_departures.setdefault(dep["day"], []).append(dep)

        self._scheduled_hires: Dict[int, List[dict]] = {}
        for hire in self._cfg.get("scheduled_hires", []):
            self._scheduled_hires.setdefault(hire["day"], []).append(hire)

    # ─── PUBLIC ───────────────────────────────────────────────────────────────

    def process_departures(
        self, day: int, date_str: str, state, clock
    ) -> List[DepartureRecord]:
        departures: List[DepartureRecord] = []

        for dep_cfg in self._scheduled_departures.get(day, []):
            record = self._execute_departure(
                dep_cfg, day, date_str, state, scheduled=True, clock=clock
            )
            if record:
                departures.append(record)

        if self._cfg.get("enable_random_attrition", False):
            prob = self._cfg.get("random_attrition_daily_prob", 0.01)
            candidates = [
                n
                for n in list(self._all_names)
                if n not in self._leads.values()
                and n not in [d.name for d in self._departed]
            ]
            for candidate in candidates:
                if random.random() < prob:
                    dept = next(
                        (d for d, m in self._org_chart.items() if candidate in m),
                        "Unknown",
                    )
                    dept_members = [
                        n
                        for n in self._org_chart.get(dept, [])
                        if n not in self._leads.values()
                    ]
                    min_size = self._cfg.get("min_dept_size", 2)
                    if len(dept_members) <= min_size:
                        continue

                    attrition_cfg = {
                        "name": candidate,
                        "reason": "voluntary",
                        "knowledge_domains": [],
                        "documented_pct": 0.5,
                        # role intentionally omitted — _execute_departure resolves it
                        # from personas so we don't hardcode anything here
                    }
                    record = self._execute_departure(
                        attrition_cfg,
                        day,
                        date_str,
                        state,
                        scheduled=False,
                        clock=clock,
                    )
                    if record:
                        departures.append(record)
                    break

        return departures

    def process_hires(self, day: int, date_str: str, state, clock) -> List[HireRecord]:
        hires: List[HireRecord] = []
        for hire_cfg in self._scheduled_hires.get(day, []):
            record = self._execute_hire(hire_cfg, day, date_str, state, clock)
            if record:
                hires.append(record)
        return hires

    def scan_for_knowledge_gaps(
        self,
        text: str,
        triggered_by: str,
        day: int,
        date_str: str,
        state,
        timestamp: str,
    ) -> List[KnowledgeGapEvent]:
        found: List[KnowledgeGapEvent] = []
        text_lower = text.lower()
        for record in self._departed:
            for domain in record.knowledge_domains:
                if domain.lower() not in text_lower:
                    continue
                gap_key = f"{record.name}:{domain}"
                if gap_key in self._domains_surfaced:
                    continue
                self._domains_surfaced.add(gap_key)
                gap_event = KnowledgeGapEvent(
                    departed_name=record.name,
                    domain_hit=domain,
                    triggered_by=triggered_by,
                    triggered_on_day=day,
                    documented_pct=record.documented_pct,
                )
                self._gap_events.append(gap_event)
                found.append(gap_event)
                self._mem.log_event(
                    SimEvent(
                        type="knowledge_gap_detected",
                        timestamp=timestamp,
                        day=day,
                        date=date_str,
                        actors=[record.name],
                        artifact_ids={"trigger": triggered_by},
                        facts={
                            "departed_employee": record.name,
                            "domain": domain,
                            "triggered_by": triggered_by,
                            "documented_pct": record.documented_pct,
                            "days_since_departure": day - record.day,
                            "escalation_harder": True,
                        },
                        summary=(
                            f"Knowledge gap: {domain} (owned by ex-{record.name}) "
                            f"surfaced in {triggered_by}. "
                            f"~{int(record.documented_pct * 100)}% documented."
                        ),
                        tags=["knowledge_gap", "departed_employee"],
                    )
                )
                logger.info(
                    f"    [yellow]⚠ Knowledge gap:[/yellow] {domain} "
                    f"(was {record.name}'s) surfaced in {triggered_by}"
                )
        return found

    def get_roster_context(self) -> str:
        lines: List[str] = []
        for d in self._departed[-3:]:
            gap_str = (
                f"Knowledge gaps: {', '.join(d.knowledge_domains)}. "
                f"~{int(d.documented_pct * 100)}% documented."
                if d.knowledge_domains
                else "No critical knowledge gaps."
            )
            ticket_str = (
                f" Reassigned: {', '.join(d.reassigned_tickets)}."
                if d.reassigned_tickets
                else ""
            )
            handoff_str = (
                f" Incident handoffs: {', '.join(d.incident_handoffs)}."
                if d.incident_handoffs
                else ""
            )
            if not lines:
                lines.append("RECENT DEPARTURES:")
            lines.append(
                f"  - {d.name} ({d.dept}) left Day {d.day} [{d.reason}]. "
                f"{gap_str}{ticket_str}{handoff_str}"
            )
        for h in self._hired[-3:]:
            warm = self._count_warm_edges(h)
            if not any("RECENT HIRES" in l for l in lines):
                lines.append("RECENT HIRES (still warming up):")
            lines.append(
                f"  - {h.name} ({h.dept}, {h.role}) joined Day {h.day}. "
                f"Warm collaborations so far: {warm}. Needs mentoring/1on1s."
            )
        return "\n".join(lines) if lines else ""

    def warm_up_edge(self, new_hire: str, colleague: str, boost: float) -> None:
        """Call from flow.py when onboarding_session or warmup_1on1 fires."""
        G = self._gd.G
        floor = self._gd.cfg.get("edge_weight_floor", 0.5)
        if not G.has_edge(new_hire, colleague):
            G.add_edge(new_hire, colleague, weight=floor)
        G[new_hire][colleague]["weight"] = round(
            G[new_hire][colleague].get("weight", floor) + boost, 4
        )
        self._gd._centrality_dirty = True

    def departed_names(self) -> List[str]:
        return [d.name for d in self._departed]

    def new_hire_names(self) -> List[str]:
        return [h.name for h in self._hired]

    def find_departure(self, name: str) -> Optional[DepartureRecord]:
        return next((d for d in self._departed if d.name == name), None)

    def find_hire(self, name: str) -> Optional[HireRecord]:
        return next((h for h in self._hired if h.name == name), None)

    # ─── DEPARTURE ENGINE ─────────────────────────────────────────────────────

    def _execute_departure(
        self, dep_cfg: dict, day: int, date_str: str, state, scheduled: bool, clock
    ) -> Optional[DepartureRecord]:
        name = dep_cfg["name"]
        G = self._gd.G

        if name not in G:
            logger.warning(f"[lifecycle] '{name}' not in graph — skipping departure.")
            return None

        dept = next((d for d, m in self._org_chart.items() if name in m), "Unknown")
        dept_lead = self._leads.get(dept) or next(iter(self._leads.values()))

        actual_role = (
            dep_cfg.get("role")
            or self._personas.get(name, {}).get("role")
            or f"{dept} Employee"
        )

        # Snapshot before any mutation
        edge_snapshot = {nb: G[name][nb].get("weight", 1.0) for nb in G.neighbors(name)}
        peak_stress = self._gd._stress.get(name, 30)
        centrality_before = dict(self._gd._get_centrality())
        departing_centrality = centrality_before.get(name, 0.0)

        record = DepartureRecord(
            name=name,
            dept=dept,
            role=actual_role,
            day=day,
            reason=dep_cfg.get("reason", "voluntary"),
            knowledge_domains=dep_cfg.get("knowledge_domains", []),
            documented_pct=float(dep_cfg.get("documented_pct", 0.5)),
            peak_stress=peak_stress,
            edge_snapshot=edge_snapshot,
            centrality_at_departure=departing_centrality,
        )

        departure_time = clock.schedule_meeting(
            [name], min_hour=9, max_hour=9, duration_mins=15
        )
        timestamp_iso = departure_time.isoformat()

        # Side-effects run in this exact order so each can still reference live graph:
        #   1. Incident handoff   — needs Dijkstra path through departing node
        #   2. JIRA reassignment  — reads ticket assignees from state
        #   3. Remove node        — graph mutation
        #   4. Centrality vacuum  — diff before/after centrality, apply stress

        self._handoff_active_incidents(
            name, dept_lead, record, day, date_str, state, timestamp_iso
        )
        self._reassign_jira_tickets(
            name, dept_lead, record, day, date_str, state, timestamp_iso
        )

        G.remove_node(name)
        self._gd._centrality_dirty = True
        self._gd._stress.pop(name, None)

        self._apply_centrality_vacuum(
            centrality_before, name, day, date_str, timestamp_iso
        )

        # Mutate org-level collections
        if dept in self._org_chart and name in self._org_chart[dept]:
            self._org_chart[dept].remove(name)
        if name in self._all_names:
            self._all_names.remove(name)
        self._personas.pop(name, None)
        self._departed.append(record)

        if not hasattr(state, "departed_employees"):
            state.departed_employees = {}
        state.departed_employees[name] = {
            "left": date_str,
            "role": dep_cfg.get("role", "Engineer"),
            "knew_about": record.knowledge_domains,
            "documented_pct": record.documented_pct,
        }

        self._schedule_backfill(record, day)

        self._mem.log_event(
            SimEvent(
                type="employee_departed",
                day=day,
                date=date_str,
                timestamp=departure_time.isoformat(),
                actors=[name],
                artifact_ids={},
                facts={
                    "name": name,
                    "dept": dept,
                    "reason": record.reason,
                    "knowledge_domains": record.knowledge_domains,
                    "documented_pct": record.documented_pct,
                    "peak_stress": peak_stress,
                    "centrality": round(departing_centrality, 4),
                    "strongest_bonds": sorted(
                        edge_snapshot.items(), key=lambda x: x[1], reverse=True
                    )[:3],
                    "tickets_reassigned": record.reassigned_tickets,
                    "incidents_handed_off": record.incident_handoffs,
                    "scheduled": scheduled,
                },
                summary=(
                    f"{name} ({dept}) departed Day {day} [{record.reason}]. "
                    + (
                        f"Gaps: {', '.join(record.knowledge_domains)}. "
                        if record.knowledge_domains
                        else ""
                    )
                    + f"~{int(record.documented_pct * 100)}% documented. "
                    + (
                        f"Reassigned {len(record.reassigned_tickets)} ticket(s). "
                        if record.reassigned_tickets
                        else ""
                    )
                    + (
                        f"Handed off {len(record.incident_handoffs)} incident(s)."
                        if record.incident_handoffs
                        else ""
                    )
                ),
                tags=["employee_departed", "lifecycle", dept.lower()],
            )
        )

        logger.info(
            f"  [red]👋 Departure:[/red] {name} ({dept}) [{record.reason}]. "
            f"{len(edge_snapshot)} edges severed. Centrality was {departing_centrality:.3f}."
            + (
                f" Undocumented domains: {record.knowledge_domains}"
                if record.knowledge_domains
                else ""
            )
        )
        return record

    # ── Side-effect 1: Incident handoff ──────────────────────────────────────

    def _handoff_active_incidents(
        self,
        name: str,
        dept_lead: str,
        record: DepartureRecord,
        day: int,
        date_str: str,
        state,
        timestamp_iso,
    ) -> None:
        """
        For every active incident whose linked JIRA ticket is assigned to the
        departing engineer, build a fresh Dijkstra path (while the node still
        exists) and transfer ownership to the next person in the chain.
        Falls back to dept_lead if no path exists.
        """
        for inc in state.active_incidents:
            jira = self._mem.get_ticket(inc.ticket_id)
            if not jira or jira.get("assignee") != name:
                continue

            # Build chain while departing node is still in the graph
            chain = self._gd.build_escalation_chain(
                first_responder=name,
                domain_keywords=record.knowledge_domains or None,
            )
            new_owner = next((n for n, _ in chain.chain if n != name), dept_lead)

            # Deterministic mutation — engine owns this, not the LLM
            jira["assignee"] = new_owner
            self._mem.upsert_ticket(jira)  # persist — get_ticket reads from MongoDB
            record.incident_handoffs.append(inc.ticket_id)

            self._mem.log_event(
                SimEvent(
                    type="escalation_chain",
                    timestamp=timestamp_iso,
                    day=day,
                    date=date_str,
                    actors=[name, new_owner],
                    artifact_ids={"jira": inc.ticket_id},
                    facts={
                        "trigger": "forced_handoff_on_departure",
                        "departed": name,
                        "new_owner": new_owner,
                        "incident": inc.ticket_id,
                        "incident_stage": inc.stage,
                        "chain_used": chain.chain,
                        "chain_narrative": self._gd.escalation_narrative(chain),
                    },
                    summary=(
                        f"Incident {inc.ticket_id} (stage={inc.stage}) handed off "
                        f"from departing {name} to {new_owner}. "
                        f"Chain: {self._gd.escalation_narrative(chain)}"
                    ),
                    tags=["escalation_chain", "incident_handoff", "lifecycle"],
                )
            )
            logger.info(
                f"    [cyan]🔀 Incident handoff:[/cyan] {inc.ticket_id} "
                f"(stage={inc.stage}) {name} → {new_owner}"
            )

    # ── Side-effect 2: JIRA ticket reassignment ───────────────────────────────

    def _reassign_jira_tickets(
        self,
        name: str,
        dept_lead: str,
        record: DepartureRecord,
        day: int,
        date_str: str,
        state,
        timestamp_iso,
    ) -> None:
        """
        Reassign all non-Done JIRA tickets owned by the departing engineer.

        Status logic:
          "To Do"        → stays "To Do", just new assignee
          "In Progress"  with no linked PR → reset to "To Do" (new owner starts fresh)
          "In Progress"  with linked PR    → keep status; PR review/merge closes it
        """
        open_tickets = list(
            self._mem._jira.find(
                {
                    "assignee": name,
                    "status": {"$ne": "Done"},
                }
            )
        )
        for ticket in open_tickets:
            if ticket.get("assignee") != name or ticket.get("status") == "Done":
                continue
            # Skip tickets that were already handed off via incident handoff above
            if ticket.get("id") in record.incident_handoffs:
                # Assignee already updated — just log and continue
                record.reassigned_tickets.append(ticket["id"])
                continue

            old_status = ticket["status"]
            ticket["assignee"] = dept_lead
            if old_status == "In Progress" and not ticket.get("linked_prs"):
                ticket["status"] = "To Do"
            self._mem.upsert_ticket(ticket)
            if self._base:
                import json as _json

                serialisable = {k: v for k, v in ticket.items() if k != "_id"}
                path = f"{self._base}/jira/{ticket['id']}.json"
                with open(path, "w") as f:
                    _json.dump(serialisable, f, indent=2)

            record.reassigned_tickets.append(ticket["id"])

            self._mem.log_event(
                SimEvent(
                    type="ticket_progress",
                    timestamp=timestamp_iso,
                    day=day,
                    date=date_str,
                    actors=[name, dept_lead],
                    artifact_ids={"jira": ticket["id"]},
                    facts={
                        "ticket_id": ticket["id"],
                        "title": ticket.get("title", ""),
                        "old_assignee": name,
                        "new_assignee": dept_lead,
                        "old_status": old_status,
                        "new_status": ticket["status"],
                        "reason": "departure_reassignment",
                    },
                    summary=(
                        f"Ticket {ticket['id']} reassigned: {name} → {dept_lead} "
                        f"(status: {old_status} → {ticket['status']})."
                    ),
                    tags=["ticket_reassignment", "lifecycle"],
                )
            )

        if record.reassigned_tickets:
            logger.info(
                f"    [yellow]📋 Reassigned:[/yellow] "
                f"{', '.join(record.reassigned_tickets)} → {dept_lead}"
            )

    # ── Side-effect 3: Centrality vacuum ─────────────────────────────────────

    def _apply_centrality_vacuum(
        self,
        centrality_before: Dict[str, float],
        departed_name: str,
        day: int,
        date_str: str,
        clock,
    ) -> None:
        """
        After the departed node is removed, force a fresh centrality computation.
        Any remaining node whose score increased has absorbed bridging load —
        it receives a proportional stress hit.

          stress_delta = (c_after - c_before) * multiplier   [capped at 20]

        This reflects the real organisational phenomenon: when a critical
        connector leaves, the people who were adjacent to them suddenly become
        the sole bridges across previously-separate clusters and feel the weight.
        """
        self._gd._centrality_dirty = True
        centrality_after = self._gd._get_centrality()

        multiplier = self._cfg.get("centrality_vacuum_stress_multiplier", 40)
        max_hit = 20

        vacuum_affected: List[Tuple[str, float, int]] = []

        for node, c_after in centrality_after.items():
            delta = c_after - centrality_before.get(node, 0.0)
            if delta <= 0:
                continue
            stress_hit = min(max_hit, int(delta * multiplier))
            if stress_hit <= 0:
                continue
            self._gd._stress[node] = min(
                100, self._gd._stress.get(node, 30) + stress_hit
            )
            vacuum_affected.append((node, round(delta, 4), stress_hit))

        if not vacuum_affected:
            return

        vacuum_affected.sort(key=lambda x: x[2], reverse=True)
        summary_str = ", ".join(
            f"{n} +{s}pts (Δc={d})" for n, d, s in vacuum_affected[:5]
        )

        self._mem.log_event(
            SimEvent(
                type="knowledge_gap_detected",  # closest existing type for RAG eval
                timestamp=clock,
                day=day,
                date=date_str,
                actors=[n for n, _, _ in vacuum_affected],
                artifact_ids={},
                facts={
                    "trigger": "centrality_vacuum",
                    "departed": departed_name,
                    "affected_nodes": [
                        {"name": n, "centrality_delta": d, "stress_added": s}
                        for n, d, s in vacuum_affected
                    ],
                },
                summary=(
                    f"Centrality vacuum after {departed_name}'s departure. "
                    f"Bridging load redistributed to: {summary_str}."
                ),
                tags=["centrality_vacuum", "lifecycle", "stress"],
            )
        )
        logger.info(
            f"    [magenta]📈 Centrality vacuum:[/magenta] "
            f"Stress absorbed by: {summary_str}"
        )

    # ─── HIRE ENGINE ──────────────────────────────────────────────────────────

    def _execute_hire(
        self, hire_cfg: dict, day: int, date_str: str, state, clock
    ) -> Optional[HireRecord]:
        name = hire_cfg["name"]
        dept = hire_cfg.get("dept", list(self._org_chart.keys())[0])
        role = hire_cfg.get("role", "Engineer")
        expertise = hire_cfg.get("expertise", ["general"])
        style = hire_cfg.get("style", "collaborative")
        tenure = hire_cfg.get("tenure", "new")
        floor = self._gd.cfg.get("edge_weight_floor", 0.5)
        G = self._gd.G

        self._personas[name] = {
            "style": style,
            "expertise": expertise,
            "tenure": tenure,
            "stress": 20,
            "social_role": hire_cfg.get(
                "social_role", "The Reliable Contributor"
            ),  # Capture from config
            "typing_quirks": hire_cfg.get(
                "typing_quirks", "Standard professional grammar."
            ),  # Capture from config
        }

        persona_data = self._personas.get(name, {})
        self._mem.embed_persona_skills(
            name,
            persona_data,
            dept,
            day=day,
            timestamp_iso=clock.now("system").isoformat(),
        )

        if name in G:
            logger.warning(f"[lifecycle] '{name}' already in graph — skipping hire.")
            return None

        if dept not in self._org_chart:
            self._org_chart[dept] = []
        self._org_chart[dept].append(name)
        if name not in self._all_names:
            self._all_names.append(name)

        self._personas[name] = {
            "style": style,
            "expertise": expertise,
            "tenure": tenure,
            "stress": 20,
        }

        G.add_node(name, dept=dept, is_lead=False, external=False, hire_day=day)
        self._gd._stress[name] = 20
        self._gd._centrality_dirty = True

        # Same-dept peers get 2× floor; cross-dept gets floor.
        # Both are below warmup_threshold so the planner naturally proposes 1on1s.
        for other in list(G.nodes()):
            if other == name:
                continue
            base_w = floor * 2.0 if G.nodes[other].get("dept") == dept else floor
            G.add_edge(name, other, weight=round(base_w, 4))

        record = HireRecord(
            name=name,
            dept=dept,
            day=day,
            role=role,
            expertise=expertise,
            style=style,
            tenure=tenure,
        )
        self._hired.append(record)

        if not hasattr(state, "new_hires"):
            state.new_hires = {}
        state.new_hires[name] = {
            "joined": date_str,
            "role": role,
            "dept": dept,
            "expertise": expertise,
        }

        hire_time = clock.schedule_meeting(
            [name], min_hour=9, max_hour=10, duration_mins=30
        )
        # Ensure it doesn't roll back before 09:30
        if hire_time.minute < 30 and hire_time.hour == 9:
            hire_time = hire_time.replace(minute=random.randint(30, 59))

        hr_email_id = hire_cfg.get("_hr_email_embed_id")

        self._mem.log_event(
            SimEvent(
                type="employee_hired",
                timestamp=hire_time.isoformat(),
                day=day,
                date=date_str,
                actors=[name],
                artifact_ids={
                    **({"hr_email": hr_email_id} if hr_email_id else {}),
                },
                facts={
                    "name": name,
                    "dept": dept,
                    "role": role,
                    "expertise": expertise,
                    "tenure": tenure,
                    "cold_start": True,
                    "edge_weight_floor": floor,
                    "hr_email_embed_id": hr_email_id,
                },
                summary=(
                    f"{name} joined {dept} as {role} on Day {day}. "
                    f"Cold-start edges at ω={floor}. Expertise: {', '.join(expertise)}."
                ),
                tags=["employee_hired", "lifecycle", dept.lower()],
            )
        )
        logger.info(
            f"  [green]🎉 New hire:[/green] {name} → {dept} ({role}). "
            f"Cold-start at ω={floor}. Expertise: {expertise}"
        )
        return record

    def _schedule_backfill(self, record: DepartureRecord, current_day: int) -> None:
        """
        If the departure reason warrants a backfill, inject a hire into
        _scheduled_hires at a future day based on configurable lag.
        """
        backfill_cfg = self._cfg.get("backfill", {})

        # Which departure reasons trigger a backfill attempt
        reasons_that_backfill = backfill_cfg.get("trigger_reasons", ["voluntary"])
        if record.reason not in reasons_that_backfill:
            return

        # Layoffs explicitly blocked from backfill
        if record.reason == "layoff":
            return

        lag_days = backfill_cfg.get("lag_days", 14)
        hire_day = current_day + lag_days

        name = self._generate_backfill_name(dept=record.dept, role=record.role)
        if name is None:
            return

        # Build a minimal hire config from the departing person's persona
        departed_persona = self._personas.get(record.name, {})
        backfill_hire = {
            "name": backfill_cfg.get("name_prefix", "NewHire") + f"_{hire_day}",
            "dept": record.dept,
            "role": record.role,
            "expertise": departed_persona.get("expertise", ["general"]),
            "style": "still ramping up, asks frequent questions",
            "tenure": "new",
            "day": hire_day,
            "_backfill_for": record.name,  # metadata only — for SimEvent logging
        }

        self._scheduled_hires.setdefault(hire_day, []).append(backfill_hire)

        logger.info(
            f"    [dim]📅 Backfill scheduled:[/dim] {record.dept} hire "
            f"queued for Day {hire_day} (replacing {record.name})"
        )

    def _generate_backfill_name(self, dept: str, role: str) -> Optional[str]:
        """
        Ask the LLM for a single realistic first+last name for a new hire.
        Retries up to 3 times if the name collides with an existing person.
        Returns None if a unique name can't be produced — backfill is skipped.
        """
        if self._llm is None:
            return None

        forbidden = set(self._all_names) | {d.name for d in self._departed}
        company = self._cfg.get("company_name", "the company")

        for attempt in range(3):
            try:
                from crewai import Task, Crew

                agent = make_agent(
                    role="HR Coordinator",
                    goal="Generate a realistic employee name.",
                    backstory=(
                        f"You work in HR at {company}. "
                        f"You are onboarding a new {role} for the {dept} team."
                    ),
                    llm=self._llm,
                )
                task = Task(
                    description=(
                        f"Generate ONE realistic full name (first and last) for a new "
                        f"{role} joining the {dept} team. "
                        f"The name must not be any of: {sorted(forbidden)}. "
                        f"Respond with ONLY the name — no punctuation, no explanation."
                    ),
                    expected_output="A single full name, e.g. 'Jordan Lee'.",
                    agent=agent,
                )
                raw = str(
                    Crew(agents=[agent], tasks=[task], verbose=False).kickoff()
                ).strip()

                # Strip any punctuation the LLM smuggled in
                name = " ".join(raw.split())
                name = "".join(c for c in name if c.isalpha() or c == " ").strip()

                if not name or len(name.split()) < 2:
                    continue
                if name in forbidden:
                    logger.info(
                        f"    [dim]LLM proposed existing name '{name}', retrying...[/dim]"
                    )
                    continue

                return name

            except Exception as e:
                logger.warning(
                    f"[lifecycle] Name generation attempt {attempt + 1} failed: {e}"
                )

        logger.warning(
            f"[lifecycle] Could not generate a unique backfill name for {dept} "
            f"after 3 attempts. Backfill skipped."
        )
        return None

    # ─── Private helpers ──────────────────────────────────────────────────────

    def _count_warm_edges(self, hire: HireRecord) -> int:
        G = self._gd.G
        if not G.has_node(hire.name):
            return 0
        return sum(
            1
            for nb in G.neighbors(hire.name)
            if G[hire.name][nb].get("weight", 0) >= hire.warmup_threshold
        )


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS — called from flow.py
# ─────────────────────────────────────────────────────────────────────────────


def patch_validator_for_lifecycle(
    validator, lifecycle_mgr: OrgLifecycleManager
) -> None:
    """Sync PlanValidator._valid_actors with the current live roster."""
    for name in lifecycle_mgr.departed_names():
        validator._valid_actors.discard(name)
    for name in lifecycle_mgr.new_hire_names():
        validator._valid_actors.add(name)


def recompute_escalation_after_departure(
    graph_dynamics: GraphDynamics,
    departed: DepartureRecord,
    first_responder: str,
) -> str:
    """
    Rebuild escalation chain post-departure and return a log-ready narrative.
    The node has already been removed, so Dijkstra routes around the gap.
    """
    chain = graph_dynamics.build_escalation_chain(
        first_responder=first_responder,
        domain_keywords=departed.knowledge_domains or None,
    )
    narrative = graph_dynamics.escalation_narrative(chain)
    note = f"[Post-departure re-route after {departed.name} left] Path: {narrative}"
    logger.info(f"    [cyan]🔀 Escalation re-routed:[/cyan] {note}")
    return note
