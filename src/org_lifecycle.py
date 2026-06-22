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
"""

from __future__ import annotations

from datetime import datetime
import logging
import json as _json
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


from agent_factory import make_agent
from config_loader import CONFIG
from crm_system import NullCRMSystem
from memory import Memory, SimEvent
from graph_dynamics import GraphDynamics

logger = logging.getLogger("orgforge.lifecycle")


@dataclass
class DepartureRecord:
    """Immutable record of an engineer who has left the organisation."""

    name: str
    dept: str
    role: str
    day: int
    reason: str
    knowledge_domains: List[str]
    documented_pct: float
    peak_stress: int
    edge_snapshot: Dict[str, float]
    centrality_at_departure: float = 0.0
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
    warmup_threshold: float = 2.0


@dataclass
class KnowledgeGapEvent:
    departed_name: str
    domain_hit: str
    triggered_by: str
    triggered_on_day: int
    documented_pct: float


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
        org_chart: Dict[str, List[str]],
        personas: Dict[str, dict],
        all_names: List[str],
        leads: Dict[str, str],
        worker_llm=None,
        base_export_dir: str = "",
        crm=None,
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
        self._crm = crm or NullCRMSystem()

        self._departed: List[DepartureRecord] = []
        self._hired: List[HireRecord] = []
        self._gap_events: List[KnowledgeGapEvent] = []
        self._domains_surfaced: Set[str] = set()

        self._scheduled_departures: Dict[int, List[dict]] = {}
        for dep in self._cfg.get("scheduled_departures", []):
            self._scheduled_departures.setdefault(dep["day"], []).append(dep)

        self._scheduled_hires: Dict[int, List[dict]] = {}
        for hire in self._cfg.get("scheduled_hires", []):
            self._scheduled_hires.setdefault(hire["day"], []).append(hire)

        sim_start_str = config.get("simulation", {}).get("start_date", "2024-01-01")
        sim_start = datetime.strptime(sim_start_str, "%Y-%m-%d")

        for gap in config.get("knowledge_gaps", []):
            left_dt = datetime.strptime(gap["left"], "%Y-%m")
            day = -(sim_start - left_dt).days

            self._departed.append(
                DepartureRecord(
                    name=gap["name"],
                    dept=gap.get("dept", "Unknown"),
                    role=gap.get("role", "Former Employee"),
                    day=day,
                    reason="voluntary",
                    knowledge_domains=gap.get("knew_about", []),
                    documented_pct=float(gap.get("documented_pct", 0.5)),
                    peak_stress=50,
                    edge_snapshot={},
                    centrality_at_departure=0.0,
                )
            )

    def process_departures(
        self, day: int, date_str: str, state, clock, ticket_assigner=None
    ) -> List[DepartureRecord]:
        departures: List[DepartureRecord] = []

        for dep_cfg in self._scheduled_departures.get(day, []):
            record = self._execute_departure(
                dep_cfg, day, date_str, state, scheduled=True, clock=clock
            )
            if record:
                departures.append(record)
                if ticket_assigner is not None:
                    ticket_assigner.evict_engineer(record.name)

        if self._cfg.get("enable_random_attrition", False):
            prob = self._cfg.get("random_attrition_daily_prob", 0.01)
            if random.random() < prob:
                candidates = [
                    n
                    for n in list(self._all_names)
                    if n not in self._leads.values()
                    and n not in [d.name for d in self._departed]
                ]
                if candidates:
                    candidate = random.choice(candidates)
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
                    if len(dept_members) > min_size:
                        attrition_cfg = {
                            "name": candidate,
                            "reason": "voluntary",
                            "knowledge_domains": self._personas.get(candidate, {}).get(
                                "expertise", []
                            ),
                            "documented_pct": 0.5,
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
                            if ticket_assigner is not None:
                                ticket_assigner.evict_engineer(record.name)

        return departures

    def process_hires(
        self, day: int, date_str: str, state, clock, ticket_assigner=None
    ) -> List[HireRecord]:
        hires: List[HireRecord] = []
        for hire_cfg in self._scheduled_hires.get(day, []):
            record = self._execute_hire(hire_cfg, day, date_str, state, clock)
            if record:
                hires.append(record)
                if ticket_assigner is not None:
                    ticket_assigner.register_hire(record.name)
        return hires

    def scan_for_knowledge_gaps(
        self,
        text: str,
        triggered_by: str,
        day: int,
        date_str: str,
        state,
        timestamp: str,
        author: Optional[str] = None,
    ) -> List[KnowledgeGapEvent]:
        """
        Detect knowledge gaps deterministically using BM25 text search against
        departed employee expertise profiles, then cross-reference the
        DomainRegistry for live documentation coverage and orphan status.

        Two-pass detection (fully deterministic given the same index state):

          Pass 1 -- BM25 text search: run the trigger text through
                    find_expert_by_skill() which calls search_persona_skills_text()
                    -> search_artifacts_text() with type_filter="persona_skill".
                    Atlas Search scores using alias-weighted BM25:
                      title boost 3x, aliases boost 2x, why_it_matters boost 1x.
                    This catches vocabulary overlap (e.g. "auth timeout" matches
                    aliases ["auth", "identity", "sso"] on a departed employee's
                    persona_skill record).

          Pass 2 -- DomainRegistry cross-reference: for each matched departed
                    employee, check whether their orphaned domains have a live
                    owner or whether the current author is in known_by. Pull
                    documentation_coverage from the registry so the SimEvent
                    carries an accurate, mutable coverage score rather than the
                    static documented_pct frozen at departure.

        Classification uses hardcoded thresholds on documentation_coverage:
          - coverage < 0.3 -> "likely"
          - coverage < 0.6 -> "possible"
          - else -> "none"

        No LLM involvement. Fully reproducible given the same Atlas Search
        index state and domain_registry contents.

        Args:
            text:          The content text to check for gap overlap.
            triggered_by:  The artifact ID (e.g., CONF page) that surfaced this.
            day:           Current simulation day.
            date_str:      Current date as ISO string.
            state:         Simulation state object.
            timestamp:     ISO timestamp of the triggering event.
            author:        Optional author name; if provided, gaps where the
                           author is in known_by for all affected domains are
                           suppressed (they already have coverage).
        """
        found: List[KnowledgeGapEvent] = []

        if not self._departed:
            return found

        registry = self._mem.get_domain_registry()
        orphaned_by_former_owner: Dict[str, List[dict]] = {}
        for rec in registry.values():
            if rec.get("primary_owner") is not None:
                continue
            former_owners = set(rec.get("former_owners", [rec.get("former_owner", "")]))
            for fo in former_owners:
                orphaned_by_former_owner.setdefault(fo, []).append(rec)

        expert_matches = self._mem.find_expert_by_skill(text, n=3)

        match_scores: Dict[str, float] = {}
        for match in expert_matches:
            name = match.get("name")
            score = match.get("score", 0.0)
            if not name:
                continue

            if name not in orphaned_by_former_owner:
                continue

            if author:
                owned_domains = orphaned_by_former_owner[name]
                all_covered = all(
                    author in rec.get("known_by", []) for rec in owned_domains
                )
                if all_covered:
                    continue

            if name not in match_scores or score > match_scores[name]:
                match_scores[name] = score

        for record in self._departed:
            if record.name not in match_scores:
                continue

            score = match_scores[record.name]

            gap_key = f"{record.name}:semantic:{triggered_by}"
            if gap_key in self._domains_surfaced:
                continue
            self._domains_surfaced.add(gap_key)

            gap_domains = record.knowledge_domains
            if not gap_domains:
                continue

            domain_label = ", ".join(gap_domains)

            registry_hits: List[dict] = []
            live_coverage: float = record.documented_pct

            for domain in gap_domains:
                key = domain.lower().replace(" ", "_")
                reg_doc = self._mem._db["domain_registry"].find_one({"_id": key})
                if reg_doc:
                    registry_hits.append(reg_doc)

            if registry_hits:
                live_coverage = min(r["documentation_coverage"] for r in registry_hits)
                known_by = list(
                    {name for r in registry_hits for name in r.get("known_by", [])}
                )
                orphaned_domains = [
                    r["domain"] for r in registry_hits if r.get("primary_owner") is None
                ]
            else:
                known_by = []
                orphaned_domains = gap_domains

            gap_event = KnowledgeGapEvent(
                departed_name=record.name,
                domain_hit=domain_label,
                triggered_by=triggered_by,
                triggered_on_day=day,
                documented_pct=live_coverage,
            )
            self._gap_events.append(gap_event)
            found.append(gap_event)

            if live_coverage < 0.3:
                gap_classification = "likely"
            elif live_coverage < 0.6:
                gap_classification = "possible"
            else:
                gap_classification = "none"

            self._mem.log_event(
                SimEvent(
                    type="knowledge_gap_detected",
                    timestamp=timestamp,
                    day=day,
                    date=date_str,
                    actors=[record.name],
                    artifact_ids={"jira": triggered_by},
                    facts={
                        "departed_employee": record.name,
                        "gap_areas": gap_domains,
                        "triggered_by": triggered_by,
                        "documented_pct": record.documented_pct,
                        "live_documentation_coverage": round(live_coverage, 3),
                        "days_since_departure": day - record.day,
                        "escalation_harder": True,
                        "semantic_score": round(score, 4),
                        "detection_method": "bm25_text_search",
                        "gap_classification": gap_classification,
                        "orphaned_domains": orphaned_domains,
                        "known_by": known_by,
                        "topics_beyond_author_expertise": gap_domains,
                        "author_domain_fit": (
                            "low"
                            if live_coverage < 0.3
                            else "medium"
                            if live_coverage < 0.6
                            else "high"
                        ),
                    },
                    summary=(
                        f"Knowledge gap: {domain_label} (owned by ex-{record.name}, "
                        f"score={score:.3f}) surfaced in {triggered_by}. "
                        f"Coverage at departure: {int(record.documented_pct * 100)}% -> "
                        f"live: {int(live_coverage * 100)}%."
                        + (
                            f" Orphaned: {orphaned_domains}."
                            if orphaned_domains
                            else ""
                        )
                        + (
                            f" Partial knowledge held by: {known_by}."
                            if known_by
                            else ""
                        )
                    ),
                    tags=["knowledge_gap", "departed_employee"]
                    + (["orphaned_domain"] if orphaned_domains else []),
                )
            )
            logger.info(
                f"    [yellow]⚠ Knowledge gap:[/yellow] {domain_label} "
                f"(was {record.name}'s, score={score:.3f}, "
                f"live_coverage={int(live_coverage * 100)}%) surfaced in {triggered_by}"
                + (f" | orphaned: {orphaned_domains}" if orphaned_domains else "")
            )

        return found

    def _load_departed_from_log(self, events: List[SimEvent]) -> None:
        """
        Reconstructs the _departed registry from the SimEvent log.
        Used by backfill scripts that run after the sim completes.
        """

        for e in events:
            if e.type != "employee_departed":
                continue
            name = next(iter(e.actors), None)
            if not name:
                continue
            if any(r.name == name for r in self._departed):
                continue
            self._departed.append(
                DepartureRecord(
                    name=name,
                    day=e.day,
                    knowledge_domains=e.facts.get("knowledge_domains", []),
                    documented_pct=e.facts.get("documented_pct", 0.0),
                    dept=e.facts.get("dept", ""),
                    role=e.facts.get("role", e.facts.get("dept", "")),
                    reason=e.facts.get("reason", "voluntary"),
                    peak_stress=e.facts.get("peak_stress", 50),
                    edge_snapshot={
                        bond[0]: bond[1] for bond in e.facts.get("strongest_bonds", [])
                    },
                )
            )
        logger.info(
            f"[lifecycle] Loaded {len(self._departed)} departed employee(s) from event log."
        )

    def get_roster_context(self, include_all_open_gaps: bool = False) -> str:
        lines: List[str] = []

        departed_to_show = (
            [
                d
                for d in self._departed
                if any(g.departed_name == d.name for g in self._gap_events)
            ]
            if include_all_open_gaps
            else self._departed[-3:]
        )
        for d in departed_to_show:
            live_coverages: List[str] = []
            orphaned: List[str] = []
            for domain in d.knowledge_domains:
                key = domain.lower().replace(" ", "_")
                reg = self._mem._db["domain_registry"].find_one({"_id": key})
                if reg:
                    pct = int(reg.get("documentation_coverage", d.documented_pct) * 100)
                    owner = reg.get("primary_owner")
                    known_by = reg.get("known_by", [])
                    status = (
                        f"owner={owner}"
                        if owner
                        else f"ORPHANED, known_by={known_by or 'nobody'}"
                    )
                    live_coverages.append(f"{domain} ({pct}%, {status})")
                    if not owner:
                        orphaned.append(domain)
                else:
                    live_coverages.append(
                        f"{domain} (~{int(d.documented_pct * 100)}%, no registry entry)"
                    )
                    orphaned.append(domain)

            if live_coverages:
                gap_str = "Knowledge gaps: " + "; ".join(live_coverages) + "."
            else:
                gap_str = "No critical knowledge gaps."

            orphan_str = (
                f" ⚠ Orphaned domains: {', '.join(orphaned)}." if orphaned else ""
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
                f"{gap_str}{orphan_str}{ticket_str}{handoff_str}"
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

        self._crm.handle_employee_departure(
            employee_name=name,
            role=actual_role,
            date_str=date_str,
            day=day,
        )

        self._orphan_domains_on_departure(name, record, day)
        self._register_domains_on_departure(name, record, day)

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

    def _register_domains_on_departure(
        self, name: str, record: DepartureRecord, day: int
    ) -> None:
        """
        Insert domain_registry entries for any knowledge domains this engineer
        owned that don't already have a record. Mirrors what seed_knowledge_gaps
        does for pre-sim (genesis) gaps so mid-sim departures are treated
        identically by scan_for_knowledge_gaps and get_roster_context.
        """
        from genesis import _domain_key, _build_system_tags

        for domain in record.knowledge_domains:
            key = _domain_key(domain)
            if self._mem._db["domain_registry"].find_one({"_id": key}):
                continue

            system_tags = _build_system_tags(domain, record.knowledge_domains)
            doc = {
                "_id": key,
                "domain": domain,
                "primary_owner": None,
                "former_owner": name,
                "former_owners": [name],
                "documentation_coverage": record.documented_pct,
                "coverage_at_last_departure": record.documented_pct,
                "last_updated_day": day,
                "known_by": [],
                "system_tags": system_tags,
                "dept": record.dept,
                "is_genesis_gap": False,
            }
            self._mem._db["domain_registry"].insert_one(doc)
            logger.info(
                f"    [dim]→ Domain registry: '{domain}' registered on departure of {name}. "
                f"Coverage={int(record.documented_pct * 100)}%, orphaned.[/dim]"
            )

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

            chain = self._gd.build_escalation_chain(
                first_responder=name,
                domain_keywords=record.knowledge_domains or None,
            )
            new_owner = next((n for n, _ in chain.chain if n != name), dept_lead)

            jira["assignee"] = new_owner
            self._mem.upsert_ticket(jira)

            path = f"{self._base}/jira/{jira['id']}.json"
            with open(path, "w") as f:
                _json.dump(jira, f, indent=2)
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
                },
                {"_id": 0},
            )
        )
        for ticket in open_tickets:
            if ticket.get("assignee") != name or ticket.get("status") == "Done":
                continue
            if ticket.get("id") in record.incident_handoffs:
                record.reassigned_tickets.append(ticket["id"])
                continue

            old_status = ticket["status"]
            ticket["assignee"] = dept_lead
            if old_status == "In Progress" and not ticket.get("linked_prs"):
                ticket["status"] = "To Do"
            self._mem.upsert_ticket(ticket)
            if self._base:
                path = f"{self._base}/jira/{ticket['id']}.json"
                with open(path, "w") as f:
                    _json.dump(ticket, f, indent=2)

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
                type="knowledge_gap_detected",
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

    def _orphan_domains_on_departure(
        self, name: str, record: DepartureRecord, day: int
    ) -> None:
        """
        Null out primary_owner on any DomainRegistry records this engineer
        owned. Called after _execute_departure so the registry stays in sync
        with the live org chart.

        Also appends the name to former_owners so audit trails are preserved,
        and records the coverage value at the time of departure so analysts can
        see how much degradation happened on their watch.
        """
        result = self._mem._db["domain_registry"].update_many(
            {"primary_owner": name},
            {
                "$set": {
                    "primary_owner": None,
                    "coverage_at_last_departure": None,
                },
                "$addToSet": {"former_owners": name},
            },
        )
        if result.modified_count == 0:
            return

        orphaned = self._mem._db["domain_registry"].find(
            {"former_owners": name, "primary_owner": None}
        )
        for doc in orphaned:
            self._mem._db["domain_registry"].update_one(
                {"_id": doc["_id"]},
                {
                    "$set": {
                        "coverage_at_last_departure": doc.get("documentation_coverage"),
                        "last_updated_day": day,
                    }
                },
            )
            logger.info(
                f"    [dim]→ Domain '{doc['domain']}' orphaned after {name}'s departure. "
                f"Coverage frozen at {int(doc.get('documentation_coverage', 0) * 100)}%.[/dim]"
            )

    def _claim_domains_on_hire(self, name: str, expertise: List[str], day: int) -> None:
        """
        When a new engineer is hired, check whether any orphaned DomainRegistry
        domains overlap with their stated expertise. If so, assign them as
        primary_owner and add them to known_by.

        This models the realistic scenario where a targeted backfill hire
        deliberately covers a known gap — the registry reflects the recovery
        immediately so planner prompts and gap detection see the updated state.
        """
        for domain_key_str in expertise:
            key = domain_key_str.lower().replace(" ", "_")
            doc = self._mem._db["domain_registry"].find_one(
                {"_id": key, "primary_owner": None}
            )
            if not doc:
                doc = self._mem._db["domain_registry"].find_one(
                    {"system_tags": key, "primary_owner": None}
                )
            if doc:
                self._mem._db["domain_registry"].update_one(
                    {"_id": doc["_id"]},
                    {
                        "$set": {
                            "primary_owner": name,
                            "last_updated_day": day,
                        },
                        "$addToSet": {"known_by": name},
                    },
                )
                logger.info(
                    f"    [dim]→ Domain '{doc['domain']}' claimed by new hire {name}. "
                    f"Coverage: {int(doc.get('documentation_coverage', 0) * 100)}%.[/dim]"
                )

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
            "social_role": hire_cfg.get("social_role", "The Reliable Contributor"),
            "typing_quirks": hire_cfg.get(
                "typing_quirks", "Standard professional grammar."
            ),
        }

        persona_data = self._personas.get(name, {})
        self._mem.embed_persona_skills(
            name,
            persona_data,
            dept,
            day=day,
            timestamp_iso=clock.now("system").isoformat(),
        )

        self._claim_domains_on_hire(name, expertise, day)

        if name in G:
            logger.warning(f"[lifecycle] '{name}' already in graph — skipping hire.")
            return None

        if dept not in self._org_chart:
            self._org_chart[dept] = []
        self._org_chart[dept].append(name)
        if name not in self._all_names:
            self._all_names.append(name)

        G.add_node(name, dept=dept, is_lead=False, external=False, hire_day=day)
        self._gd._stress[name] = 20
        self._gd._centrality_dirty = True

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

        reasons_that_backfill = backfill_cfg.get("trigger_reasons", ["voluntary"])
        if record.reason not in reasons_that_backfill:
            return

        if record.reason == "layoff":
            return

        lag_days = backfill_cfg.get("lag_days", 14)
        hire_day = current_day + lag_days

        persona = self._generate_backfill_persona(
            dept=record.dept, role=record.role, departed_name=record.name
        )
        if persona is None:
            return

        backfill_hire = {
            "name": persona["name"],
            "dept": record.dept,
            "role": record.role,
            "expertise": persona.get("expertise", ["general"]),
            "style": persona.get("style", "still ramping up, asks frequent questions"),
            "social_role": persona.get("social_role", "The Newcomer"),
            "typing_quirks": persona.get(
                "typing_quirks", "Standard professional grammar."
            ),
            "pet_peeves": persona.get("pet_peeves", ""),
            "tenure": "new",
            "day": hire_day,
            "_backfill_for": record.name,
        }

        self._scheduled_hires.setdefault(hire_day, []).append(backfill_hire)

        logger.info(
            f"    [dim]📅 Backfill scheduled:[/dim] {record.dept} hire "
            f"queued for Day {hire_day} (replacing {record.name})"
        )

    def _generate_backfill_persona(
        self, dept: str, role: str, departed_name: str
    ) -> Optional[dict]:
        """
        Ask the LLM to generate a full persona for a backfill hire.
        Includes name, style, social_role, typing_quirks, pet_peeves, and expertise.
        Retries up to 3 times if the generated name collides with an existing person.
        Returns None if a unique persona can't be produced — backfill is skipped.
        """
        if self._llm is None:
            return None

        forbidden = set(self._all_names) | {d.name for d in self._departed}
        company = CONFIG["simulation"]["company_name"]
        departed_persona = self._personas.get(departed_name, {})
        departed_expertise = departed_persona.get("expertise", ["general"])

        for attempt in range(3):
            try:
                import json
                from crewai import Task, Crew

                agent = make_agent(
                    role="HR Coordinator",
                    goal="Generate a realistic new hire persona.",
                    backstory=(
                        f"You work in HR at {company}. "
                        f"You are onboarding a new {role} for the {dept} team "
                        f"to replace {departed_name}, who recently left."
                    ),
                    llm=self._llm,
                )
                task = Task(
                    description=(
                        f"Generate a realistic persona for a new {role} joining the {dept} team.\n"
                        f"They are replacing {departed_name}, so their expertise should broadly "
                        f"cover: {departed_expertise}, but with their own background and gaps.\n"
                        f"The name must not be any of: {sorted(forbidden)}.\n\n"
                        f"Respond with ONLY a JSON object — no explanation, no markdown fences:\n"
                        f"{{\n"
                        f'  "name": "First Last",\n'
                        f'  "expertise": ["skill1", "skill2", "skill3"],\n'
                        f'  "style": "one sentence describing how they work",\n'
                        f'  "social_role": "The [Archetype]",\n'
                        f'  "typing_quirks": "one or two sentences describing their written voice",\n'
                        f'  "pet_peeves": "brief phrase"\n'
                        f"}}"
                    ),
                    expected_output="A JSON object with keys: name, expertise, style, social_role, typing_quirks, pet_peeves.",
                    agent=agent,
                )
                raw = str(
                    Crew(agents=[agent], tasks=[task], verbose=False).kickoff()
                ).strip()

                raw = (
                    raw.strip()
                    .removeprefix("```json")
                    .removeprefix("```")
                    .removesuffix("```")
                    .strip()
                )

                try:
                    persona = json.loads(raw)
                except json.JSONDecodeError:
                    logger.warning(
                        f"[lifecycle] Persona generation attempt {attempt + 1} "
                        f"returned invalid JSON: {raw[:200]}"
                    )
                    continue

                name = persona.get("name", "").strip()
                name = " ".join(name.split())
                name = "".join(c for c in name if c.isalpha() or c == " ").strip()

                if not name or len(name.split()) < 2:
                    logger.warning(
                        f"[lifecycle] Persona generation attempt {attempt + 1} "
                        f"produced an unusable name: '{name}'"
                    )
                    continue
                if name in forbidden:
                    logger.info(
                        f"    [dim]LLM proposed existing name '{name}', retrying...[/dim]"
                    )
                    continue

                persona["name"] = name
                return persona

            except Exception as e:
                logger.warning(
                    f"[lifecycle] Persona generation attempt {attempt + 1} failed: {e}"
                )

        logger.warning(
            f"[lifecycle] Could not generate a unique backfill persona for {dept} "
            f"after 3 attempts. Backfill skipped."
        )
        return None

    def _count_warm_edges(self, hire: HireRecord) -> int:
        G = self._gd.G
        if not G.has_node(hire.name):
            return 0
        return sum(
            1
            for nb in G.neighbors(hire.name)
            if G[hire.name][nb].get("weight", 0) >= hire.warmup_threshold
        )


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
    crm=None,
    root_cause: str = "",
) -> str:
    """
    Rebuild escalation chain post-departure and return a log-ready narrative.
    The node has already been removed, so Dijkstra routes around the gap.

    If crm is provided and the departed employee owned SF accounts, the chain
    prefers routing through whoever is best positioned to handle the customer
    relationship — not just the topologically nearest lead.
    """
    from crm_system import NullCRMSystem

    use_crm_aware = crm is not None and not isinstance(crm, NullCRMSystem)

    if use_crm_aware:
        owned_opps = list(
            crm._sf_o.find(
                {
                    "owner": departed.name,
                    "stage": {"$nin": ["Closed Won", "Closed Lost"]},
                },
                {"account_name": 1},
            )
        )
        for_org = owned_opps[0]["account_name"] if owned_opps else None

        chain = graph_dynamics.crm_aware_escalation_chain(
            first_responder=first_responder,
            crm=crm,
            root_cause=root_cause or "",
            for_org=for_org,
        )
    else:
        chain = graph_dynamics.build_escalation_chain(
            first_responder=first_responder,
            domain_keywords=departed.knowledge_domains or None,
        )

    narrative = graph_dynamics.escalation_narrative(chain)
    note = f"[Post-departure re-route after {departed.name} left] Path: {narrative}"
    logger.info(f"    [cyan]🔀 Escalation re-routed:[/cyan] {note}")
    return note
