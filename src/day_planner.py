"""
day_planner.py
==============
LLM-driven per-department planning layer for OrgForge.

Architecture:
  DepartmentPlanner  — one per dept, produces DepartmentDayPlan
  OrgCoordinator     — reads all dept plans, injects cross-dept collision events
  DayPlannerOrchestrator — top-level entry point called from flow.py daily_cycle

Engineering is the primary driver. Other departments react to Engineering's
plan before the OrgCoordinator looks for collision points.

Replace _generate_theme() in flow.py with:
    org_plan = self._day_planner.plan(self.state, self._mem, self.graph_dynamics)
    self.state.daily_theme = org_plan.org_theme
    self.state.org_day_plan = org_plan   # new State field — see note at bottom
"""

from __future__ import annotations

import json
import logging
import random
from typing import Dict, List, Optional

from crewai import Agent, Task, Crew

from memory import Memory, SimEvent
from graph_dynamics import GraphDynamics
from planner_models import (
    AgendaItem,
    CrossDeptSignal,
    DepartmentDayPlan,
    EngineerDayPlan,
    OrgDayPlan,
    ProposedEvent,
    KNOWN_EVENT_TYPES,
)
from plan_validator import PlanValidator

logger = logging.getLogger("orgforge.planner")


# ─────────────────────────────────────────────────────────────────────────────
# DEPARTMENT PLANNER
# ─────────────────────────────────────────────────────────────────────────────

class DepartmentPlanner:
    """
    Produces a DepartmentDayPlan for a single department.

    The LLM receives:
      - The org-level theme (from the previous _generate_theme() equivalent)
      - Last 7 day_summary facts filtered to this dept's actors
      - Cross-dept signals (facts from other depts' recent SimEvents)
      - Current roster with stress levels and assigned tickets

    The LLM produces a JSON plan. The engine parses and validates it.
    """

    # Prompt template — kept here so it's easy to tune without touching logic
    _PLAN_PROMPT = """
    You are the planning agent for the {dept} department at {company}.
    Today is Day {day} ({date}).

    ORG THEME: {org_theme}
    SYSTEM HEALTH: {system_health}/100
    TEAM MORALE: {morale_label}

    YOUR TEAM TODAY:
    {roster}

    OPEN JIRA TICKETS ASSIGNED TO YOUR TEAM:
    {open_tickets}

    RECENT DEPARTMENT HISTORY (last 7 days):
    {dept_history}

    CROSS-DEPARTMENT SIGNALS (what other teams are dealing with):
    {cross_signals}

    {lifecycle_context}

    KNOWN EVENT TYPES YOU CAN PROPOSE:
    {known_types}

    You may also propose NEW event types if the situation genuinely calls for it.
    If you propose a novel event, set "is_novel": true and specify "artifact_hint"
    as one of: "slack", "jira", "confluence", or "email".

    YOUR TASK:
    1. Write a department theme for today (one sentence, specific to {dept}).
    2. For each team member, write a 2-4 item agenda (what they plan to work on).
    3. Propose 1-3 events that should fire today, ordered by priority (1=must, 3=optional).
    4. Note your reasoning briefly.

    Engineering is the primary driver of the company. If you are Engineering,
    your plan shapes what other departments react to. Be specific — reference
    real ticket IDs, real names, and real system states.

    Respond ONLY with valid JSON matching this exact schema:
    {{
    "dept_theme": "string",
    "engineer_plans": [
        {{
        "name": "string",
        "focus_note": "string (one sentence about their headspace today)",
        "agenda": [
            {{
            "activity_type": "string",
            "description": "string",
            "related_id": "string or null",
            "collaborator": "string or null",
            "estimated_hrs": float
            }}
        ]
        }}
    ],
    "proposed_events": [
        {{
        "event_type": "string",
        "actors": ["string"],
        "rationale": "string",
        "facts_hint": {{}},
        "priority": int,
        "is_novel": false,
        "artifact_hint": "string or null"
        }}
    ],
    "planner_reasoning": "string"
    }}
    """

    def __init__(
        self,
        dept:        str,
        members:     List[str],
        config:      dict,
        worker_llm,
        is_primary:  bool = False,
    ):
        self.dept       = dept
        self.members    = members
        self.config     = config
        self._llm       = worker_llm
        self.is_primary = is_primary   # True for Engineering

    def plan(
        self,
        org_theme:       str,
        day:             int,
        date:            str,
        state,
        mem:             Memory,
        graph_dynamics:  GraphDynamics,
        cross_signals:   List[CrossDeptSignal],
        eng_plan:        Optional[DepartmentDayPlan] = None,  # None for Engineering itself
        lifecycle_context: str = "",
    ) -> DepartmentDayPlan:
        """
        Produce a DepartmentDayPlan. eng_plan is provided to non-Engineering
        departments so they can react to Engineering's agenda.
        """
        roster        = self._build_roster(graph_dynamics)
        open_tickets  = self._open_tickets(state)
        dept_history  = self._dept_history(mem, day)
        cross_str     = self._format_cross_signals(cross_signals, eng_plan)
        known_str     = ", ".join(sorted(KNOWN_EVENT_TYPES))
        morale_label  = (
            "low" if state.team_morale < 0.45 else
            "moderate" if state.team_morale < 0.70 else "healthy"
        )
        lifecycle_context=(
            f"\nROSTER CHANGES (recent hires/departures):\n{lifecycle_context}\n"
            if lifecycle_context else ""
        )

        prompt = self._PLAN_PROMPT.format(
            dept=self.dept,
            company=self.config["simulation"]["company_name"],
            day=day,
            date=date,
            org_theme=org_theme,
            system_health=state.system_health,
            morale_label=morale_label,
            roster=roster,
            open_tickets=open_tickets,
            dept_history=dept_history,
            cross_signals=cross_str,
            known_types=known_str,
            lifecycle_context=lifecycle_context,
        )

        agent = Agent(
            role=f"{self.dept} Department Planner",
            goal=f"Plan the {self.dept} team's day with realistic, grounded activities.",
            backstory=(
                f"You understand how {self.dept} teams work in a "
                f"{self.config['simulation'].get('industry', 'technology')} company. "
                f"You know the difference between a firefighting day and a productive one. "
                f"You reference real people, real tickets, and real system states."
            ),
            llm=self._llm,
        )
        task = Task(
            description=prompt,
            expected_output="Valid JSON only. No preamble, no markdown fences.",
            agent=agent,
        )

        raw = str(Crew(agents=[agent], tasks=[task], verbose=False).kickoff()).strip()

        return self._parse_plan(raw, org_theme, day, date, cross_signals)

    # ─── Parsing ─────────────────────────────────────────────────────────────

    def _parse_plan(
        self,
        raw:          str,
        org_theme:    str,
        day:          int,
        date:         str,
        cross_signals: List[CrossDeptSignal],
    ) -> DepartmentDayPlan:
        """
        Parse the LLM JSON response into a DepartmentDayPlan.
        Defensively handles partial or malformed responses.
        """
        # Strip any accidental markdown fences
        clean = raw.replace("```json", "").replace("```", "").strip()

        try:
            data = json.loads(clean)
        except json.JSONDecodeError as e:
            logger.warning(f"[planner] {self.dept} plan JSON parse failed: {e}. Using fallback.")
            return self._fallback_plan(org_theme, day, date, cross_signals)

        # ── Engineer plans ────────────────────────────────────────────────────
        eng_plans: List[EngineerDayPlan] = []
        for ep in data.get("engineer_plans", []):
            name = ep.get("name", "")
            if name not in self.members:
                continue   # LLM invented a name — skip silently

            agenda = [
                AgendaItem(
                    activity_type=a.get("activity_type", "ticket_progress"),
                    description=a.get("description", ""),
                    related_id=a.get("related_id"),
                    collaborator=a.get("collaborator"),
                    estimated_hrs=float(a.get("estimated_hrs", 2.0)),
                )
                for a in ep.get("agenda", [])
            ]

            # Fallback agenda if LLM returned nothing useful
            if not agenda:
                agenda = [AgendaItem(
                    activity_type="ticket_progress",
                    description=f"Continue assigned sprint work",
                    estimated_hrs=3.0,
                )]

            eng_plans.append(EngineerDayPlan(
                name=name,
                dept=self.dept,
                agenda=agenda,
                stress_level=0,     # will be patched by orchestrator after parse
                focus_note=ep.get("focus_note", ""),
            ))

        # Ensure every member has a plan (even if LLM missed them)
        planned_names = {p.name for p in eng_plans}
        for name in self.members:
            if name not in planned_names:
                eng_plans.append(self._default_engineer_plan(name))

        # ── Proposed events ───────────────────────────────────────────────────
        proposed: List[ProposedEvent] = []
        for pe in data.get("proposed_events", []):
            actors = [a for a in pe.get("actors", []) if a]
            if not actors:
                actors = self.members[:1]
            proposed.append(ProposedEvent(
                event_type=pe.get("event_type", "normal_day_slack"),
                actors=actors,
                rationale=pe.get("rationale", ""),
                facts_hint=pe.get("facts_hint", {}),
                priority=int(pe.get("priority", 2)),
                is_novel=bool(pe.get("is_novel", False)),
                artifact_hint=pe.get("artifact_hint"),
            ))

        return DepartmentDayPlan(
            dept=self.dept,
            theme=data.get("dept_theme", org_theme),
            engineer_plans=eng_plans,
            proposed_events=sorted(proposed, key=lambda e: e.priority),
            cross_dept_signals=cross_signals,
            planner_reasoning=data.get("planner_reasoning", ""),
            day=day,
            date=date,
        )

    # ─── Context builders ─────────────────────────────────────────────────────

    def _build_roster(self, graph_dynamics: GraphDynamics) -> str:
        lines = []
        for name in self.members:
            stress = graph_dynamics._stress.get(name, 30)
            tone   = graph_dynamics.stress_tone_hint(name)
            lines.append(f"  - {name}: stress={stress}/100. {tone}")
        return "\n".join(lines)

    def _open_tickets(self, state) -> str:
        tickets = [
            t for t in state.jira_tickets
            if t.get("assignee") in self.members and t.get("status") != "Done"
        ]
        if not tickets:
            return "  (no open tickets assigned to this team)"
        return "\n".join(
            f"  - [{t['id']}] {t['title']} — assigned to {t['assignee']}"
            for t in tickets[:8]   # cap at 8 to keep prompt tight
        )

    def _dept_history(self, mem: Memory, day: int) -> str:
        """Last 7 day_summary SimEvents filtered to this dept's actors."""
        summaries = [
            e for e in mem.get_event_log()
            if e.type == "day_summary" and e.day >= max(1, day - 7)
        ]
        if not summaries:
            return "  (no recent history)"
        lines = []
        for s in summaries[-7:]:
            dept_actors = [
                a for a in s.facts.get("active_actors", [])
                if a in self.members
            ]
            if not dept_actors and not self.is_primary:
                continue   # this dept was quiet that day — skip
            lines.append(
                f"  Day {s.day}: health={s.facts.get('system_health')} "
                f"morale={s.facts.get('morale_trend','?')} "
                f"dominant={s.facts.get('dominant_event','?')} "
                f"dept_actors={dept_actors}"
            )
        return "\n".join(lines) if lines else "  (dept was quiet recently)"

    def _format_cross_signals(
        self,
        signals:  List[CrossDeptSignal],
        eng_plan: Optional[DepartmentDayPlan],
    ) -> str:
        lines = []
        for s in signals:
            lines.append(
                f"  [{s.source_dept}] {s.event_type} (Day {s.day}): {s.summary} "
                f"[{s.relevance}]"
            )
        # Non-Engineering depts also see Engineering's proposed events for today
        if eng_plan and not self.is_primary:
            lines.append(f"\n  ENGINEERING'S PLAN TODAY:")
            for e in eng_plan.proposed_events[:3]:
                lines.append(f"    - {e.event_type}: {e.rationale}")
            for ep in eng_plan.engineer_plans[:3]:
                lines.append(
                    f"    - {ep.name} is focused on: "
                    f"{ep.agenda[0].description if ep.agenda else '?'}"
                )
        return "\n".join(lines) if lines else "  (no cross-dept signals today)"

    # ─── Fallbacks ────────────────────────────────────────────────────────────

    def _fallback_plan(
        self,
        org_theme:    str,
        day:          int,
        date:         str,
        cross_signals: List[CrossDeptSignal],
    ) -> DepartmentDayPlan:
        """Minimal valid plan when LLM output is unparseable."""
        return DepartmentDayPlan(
            dept=self.dept,
            theme=org_theme,
            engineer_plans=[self._default_engineer_plan(n) for n in self.members],
            proposed_events=[ProposedEvent(
                event_type="normal_day_slack",
                actors=self.members[:2],
                rationale="Fallback: LLM plan unparseable.",
                facts_hint={},
                priority=3,
            )],
            cross_dept_signals=cross_signals,
            planner_reasoning="Fallback plan — LLM response was not valid JSON.",
            day=day,
            date=date,
        )

    def _default_engineer_plan(self, name: str) -> EngineerDayPlan:
        return EngineerDayPlan(
            name=name,
            dept=self.dept,
            agenda=[AgendaItem(
                activity_type="ticket_progress",
                description="Continue assigned sprint work",
                estimated_hrs=3.0,
            )],
            stress_level=30,
            focus_note="",
        )


# ─────────────────────────────────────────────────────────────────────────────
# ORG COORDINATOR
# ─────────────────────────────────────────────────────────────────────────────

class OrgCoordinator:
    """
    Reads all DepartmentDayPlans and injects collision events —
    the natural interactions between departments that neither planned explicitly.

    Keeps its prompt narrow: it only looks for ONE collision per day,
    which is realistic. Real cross-dept interactions are rare and significant.
    """

    _COORD_PROMPT = """
    You are the org coordinator for {company} on Day {day}.

    Each department has produced its plan for today. Your job is to find
    ONE natural cross-department interaction that should happen given these plans.

    ENGINEERING'S PLAN:
    Theme: {eng_theme}
    Key events: {eng_events}
    Key focus areas: {eng_focus}

    OTHER DEPARTMENT PLANS:
    {other_plans}

    ORG STATE: health={health}, morale={morale_label}

    A collision event is something like:
    - Sales heard about an incident and needs a stability update from Engineering
    - A customer escalation lands in Sales that Engineering needs to know about
    - HR notices two burnt-out engineers and schedules a sync with the Eng lead
    - A feature request from Sales creates a new JIRA ticket in Engineering's backlog
    - Leadership calls a sync because health has been low for 3+ days

    Only propose a collision if it's genuinely motivated by the plans above.
    If nothing natural connects today, respond with {{"collision": null}}.

    Respond ONLY with valid JSON:
    {{
    "collision": {{
        "event_type": "string",
        "actors": ["string"],
        "rationale": "string",
        "facts_hint": {{}},
        "priority": 1,
        "artifact_hint": "string"
    }},
    "reasoning": "string"
    }}
    """

    def __init__(self, config: dict, planner_llm):
        self._config = config
        self._llm    = planner_llm

    def coordinate(
        self,
        dept_plans: Dict[str, DepartmentDayPlan],
        state,
        day:  int,
        date: str,
    ) -> OrgDayPlan:
        eng_key  = next((k for k in dept_plans if "eng" in k.lower()), None)
        eng_plan = dept_plans.get(eng_key) if eng_key else None

        org_theme = eng_plan.theme if eng_plan else "Normal operations"

        other_plans_str = self._format_other_plans(dept_plans, eng_key)
        eng_events_str  = (
            ", ".join(e.event_type for e in eng_plan.proposed_events[:3])
            if eng_plan else "none"
        )
        eng_focus_str = (
            " | ".join(
                f"{ep.name}: {ep.agenda[0].description}"
                for ep in (eng_plan.engineer_plans[:3] if eng_plan else [])
                if ep.agenda
            ) or "general sprint work"
        )
        morale_label = (
            "low" if state.team_morale < 0.45 else
            "moderate" if state.team_morale < 0.70 else "healthy"
        )

        prompt = self._COORD_PROMPT.format(
            company=self._config["simulation"]["company_name"],
            day=day,
            eng_theme=eng_plan.theme if eng_plan else org_theme,
            eng_events=eng_events_str,
            eng_focus=eng_focus_str,
            other_plans=other_plans_str,
            health=state.system_health,
            morale_label=morale_label,
        )

        agent = Agent(
            role="Org Coordinator",
            goal="Find one genuine cross-department interaction for today.",
            backstory=(
                "You understand how information flows between Engineering, Sales, "
                "and HR in a real company. You only flag interactions that are "
                "genuinely motivated by what each team is dealing with today."
            ),
            llm=self._llm,
        )
        task = Task(
            description=prompt,
            expected_output="Valid JSON only. No preamble.",
            agent=agent,
        )

        raw   = str(Crew(agents=[agent], tasks=[task], verbose=False).kickoff()).strip()
        clean = raw.replace("```json", "").replace("```", "").strip()

        collision_events: List[ProposedEvent] = []
        reasoning = ""

        try:
            data = json.loads(clean)
            reasoning = data.get("reasoning", "")
            col = data.get("collision")
            if col:
                actors = col.get("actors", [])
                if actors:
                    collision_events.append(ProposedEvent(
                        event_type=col.get("event_type", "leadership_sync"),
                        actors=actors,
                        rationale=col.get("rationale", ""),
                        facts_hint=col.get("facts_hint", {}),
                        priority=int(col.get("priority", 1)),
                        artifact_hint=col.get("artifact_hint"),
                    ))
                    logger.info(
                        f"  [magenta]🔀 Collision:[/magenta] "
                        f"{col.get('event_type')} — {col.get('rationale', '')[:60]}"
                    )
        except json.JSONDecodeError as e:
            logger.warning(f"[coordinator] JSON parse failed: {e}")

        return OrgDayPlan(
            org_theme=org_theme,
            dept_plans=dept_plans,
            collision_events=collision_events,
            coordinator_reasoning=reasoning,
            day=day,
            date=date,
        )

    def _format_other_plans(
        self,
        dept_plans: Dict[str, DepartmentDayPlan],
        eng_key:    Optional[str],
    ) -> str:
        lines = []
        for dept, plan in dept_plans.items():
            if dept == eng_key:
                continue
            events_str = ", ".join(e.event_type for e in plan.proposed_events[:2])
            lines.append(
                f"  {dept}: theme='{plan.theme}' "
                f"events=[{events_str}]"
            )
        return "\n".join(lines) if lines else "  (no other departments)"


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATOR — top-level entry point for flow.py
# ─────────────────────────────────────────────────────────────────────────────

class DayPlannerOrchestrator:
    """
    Called once per day from flow.py's daily_cycle(), replacing _generate_theme().

    Usage in flow.py:
        # In __init__:
        self._day_planner = DayPlannerOrchestrator(CONFIG, WORKER_MODEL, PLANNER_MODEL)

        # In daily_cycle(), replacing _generate_theme():
        org_plan = self._day_planner.plan(
            state=self.state,
            mem=self._mem,
            graph_dynamics=self.graph_dynamics,
        )
        self.state.daily_theme   = org_plan.org_theme
        self.state.org_day_plan  = org_plan   # add org_day_plan: Optional[Any] to State
    """

    def __init__(self, config: dict, worker_llm, planner_llm):
        self._config     = config
        self._worker_llm = worker_llm
        self._planner_llm = planner_llm

        org_chart: Dict[str, List[str]] = config["org_chart"]

        # Build one DepartmentPlanner per department
        self._dept_planners: Dict[str, DepartmentPlanner] = {}
        for dept, members in org_chart.items():
            is_primary = "eng" in dept.lower()
            self._dept_planners[dept] = DepartmentPlanner(
                dept=dept,
                members=members,
                config=config,
                worker_llm=worker_llm,
                is_primary=is_primary,
            )

        self._coordinator = OrgCoordinator(config, planner_llm)

        all_names = [n for members in org_chart.values() for n in members]
        external_names = [c["name"] for c in config.get("external_contacts", [])]
        self._validator = PlanValidator(
            all_names=all_names,
            external_contact_names=external_names,
            config=config,
        )

    def plan(
        self,
        state,
        mem:            Memory,
        graph_dynamics: GraphDynamics,
        lifecycle_context: str = "",
    ) -> OrgDayPlan:
        """
        Full planning pass for one day.
        Returns an OrgDayPlan the day loop executes against.
        """
        day  = state.day
        date = str(state.current_date.date())

        # ── Generate org theme (lightweight — replaces _generate_theme()) ─────
        org_theme = self._generate_org_theme(state, mem)

        # ── Build cross-dept signals from recent SimEvents ────────────────────
        cross_signals_by_dept = self._extract_cross_signals(mem, day)

        # ── Engineering plans first — it drives everyone else ─────────────────
        eng_key  = next((k for k in self._dept_planners if "eng" in k.lower()), None)
        eng_plan = None

        dept_plans: Dict[str, DepartmentDayPlan] = {}

        if eng_key:
            eng_plan = self._dept_planners[eng_key].plan(
                org_theme=org_theme,
                day=day, date=date,
                state=state, mem=mem,
                graph_dynamics=graph_dynamics,
                cross_signals=cross_signals_by_dept.get(eng_key, []),
                eng_plan=None,
            )
            self._patch_stress_levels(eng_plan, graph_dynamics)
            dept_plans[eng_key] = eng_plan
            logger.info(
                f"  [blue]📋 Eng plan:[/blue] {eng_plan.theme[:60]} "
                f"({len(eng_plan.proposed_events)} events)"
            )

        # ── Other departments react to Engineering ────────────────────────────
        for dept, planner in self._dept_planners.items():
            if dept == eng_key:
                continue
            plan = planner.plan(
                org_theme=org_theme,
                day=day, date=date,
                state=state, mem=mem,
                graph_dynamics=graph_dynamics,
                cross_signals=cross_signals_by_dept.get(dept, []),
                eng_plan=eng_plan,
                lifecycle_context=lifecycle_context,
            )
            self._patch_stress_levels(plan, graph_dynamics)
            dept_plans[dept] = plan
            logger.info(
                f"  [blue]📋 {dept} plan:[/blue] {plan.theme[:60]} "
                f"({len(plan.proposed_events)} events)"
            )

        # ── OrgCoordinator finds collisions ───────────────────────────────────
        org_plan = self._coordinator.coordinate(dept_plans, state, day, date)

        # ── Validate all proposed events ──────────────────────────────────────
        recent_summaries = self._recent_day_summaries(mem, day)
        all_proposed = org_plan.all_events_by_priority()
        results = self._validator.validate_plan(all_proposed, state, recent_summaries)

        # Log rejections as SimEvents so researchers can see what was blocked
        for r in self._validator.rejected(results):
            mem.log_event(SimEvent(
                type="proposed_event_rejected",
                day=day, date=date,
                actors=r.event.actors,
                artifact_ids={},
                facts={
                    "event_type":       r.event.event_type,
                    "rejection_reason": r.rejection_reason,
                    "rationale":        r.event.rationale,
                    "was_novel":        r.was_novel,
                },
                summary=f"Rejected: {r.event.event_type} — {r.rejection_reason}",
                tags=["validation", "rejected"],
            ))

        # Log novel events the community could implement
        for novel in self._validator.drain_novel_log():
            mem.log_event(SimEvent(
                type="novel_event_proposed",
                day=day, date=date,
                actors=novel.actors,
                artifact_ids={},
                facts={
                    "event_type":   novel.event_type,
                    "rationale":    novel.rationale,
                    "artifact_hint": novel.artifact_hint,
                    "facts_hint":   novel.facts_hint,
                },
                summary=f"Novel event proposed: {novel.event_type}. {novel.rationale}",
                tags=["novel", "proposed"],
            ))

        # Rebuild dept_plans with only approved events
        approved_set = {id(e) for e in self._validator.approved(results)}
        for dept, dplan in org_plan.dept_plans.items():
            dplan.proposed_events = [
                e for e in dplan.proposed_events if id(e) in approved_set
            ]
        org_plan.collision_events = [
            e for e in org_plan.collision_events if id(e) in approved_set
        ]

        return org_plan

    # ─── Helpers ─────────────────────────────────────────────────────────────

    def _generate_org_theme(self, state, mem: Memory) -> str:
        """Lightweight replacement for _generate_theme() in flow.py."""
        ctx = mem.context_for_prompt(
            f"day {state.day} system health sprint incidents",
            n=3, as_of_day=state.day
        )
        agent = Agent(
            role="CEO",
            goal="Decide today's dominant org theme.",
            backstory="You see patterns across the whole company.",
            llm=self._worker_llm,
        )
        task = Task(
            description=(
                f"Day {state.day}. Health: {state.system_health}. "
                f"Morale: {state.team_morale:.2f}.\n"
                f"Context:\n{ctx}\n"
                f"Write ONE sentence for today's org-wide theme."
            ),
            expected_output="One sentence.",
            agent=agent,
        )
        return str(Crew(agents=[agent], tasks=[task], verbose=False).kickoff()).strip()

    def _extract_cross_signals(
        self, mem: Memory, day: int
    ) -> Dict[str, List[CrossDeptSignal]]:
        """
        Reads recent SimEvents and produces cross-dept signals.
        Engineering incidents become signals for Sales and HR.
        Sales escalations become signals for Engineering.
        """
        signals: Dict[str, List[CrossDeptSignal]] = {}
        config_chart: Dict[str, List] = self._config["org_chart"]

        relevant_types = {
            "incident_resolved", "incident_opened", "postmortem_created",
            "feature_request_from_sales", "customer_escalation",
            "morale_intervention", "hr_checkin",
        }

        recent = [
            e for e in mem.get_event_log()
            if e.type in relevant_types and e.day >= max(1, day - 5)
        ]

        for event in recent:
            # Determine source dept from actors
            for actor in event.actors:
                source_dept = next(
                    (d for d, members in config_chart.items() if actor in members),
                    None,
                )
                if not source_dept:
                    continue

                signal = CrossDeptSignal(
                    source_dept=source_dept,
                    event_type=event.type,
                    summary=event.summary,
                    day=event.day,
                    relevance="direct" if day - event.day <= 2 else "indirect",
                )

                # Push signal to all OTHER departments
                for dept in config_chart:
                    if dept != source_dept:
                        signals.setdefault(dept, []).append(signal)
                break   # one signal per event

        return signals

    def _patch_stress_levels(
        self,
        plan:           DepartmentDayPlan,
        graph_dynamics: GraphDynamics,
    ):
        """Fills in stress_level on each EngineerDayPlan after parsing."""
        for ep in plan.engineer_plans:
            ep.stress_level = graph_dynamics._stress.get(ep.name, 30)

    def _recent_day_summaries(self, mem: Memory, day: int) -> List[dict]:
        """Last 7 day_summary facts dicts for the validator."""
        return [
            e.facts for e in mem.get_event_log()
            if e.type == "day_summary" and e.day >= max(1, day - 7)
        ]

