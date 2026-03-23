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
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
from dataclasses import asdict as _asdict

from agent_factory import make_agent
from crewai import Task, Crew

from memory import Memory, SimEvent
from graph_dynamics import GraphDynamics
from planner_models import (
    AgendaItem,
    CrossDeptSignal,
    DepartmentDayPlan,
    EngineerDayPlan,
    OrgDayPlan,
    ProposedEvent,
    SprintContext,
    KNOWN_EVENT_TYPES,
)
from plan_validator import PlanValidator
from ticket_assigner import TicketAssigner
from external_email_ingest import ExternalEmailSignal
from config_loader import (
    LEADS,
    LIVE_ORG_CHART,
    PERSONAS,
    DEFAULT_PERSONA,
    COMPANY_DESCRIPTION,
    resolve_role,
)

logger = logging.getLogger("orgforge.planner")


def _coerce_collaborators(raw) -> List[str]:
    """Normalize LLM collaborator output — handles str, list, or None."""
    if not raw:
        return []
    return raw if isinstance(raw, list) else [raw]


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
    You are the planning agent for the {dept} department at {company} which {company_description}.
    Today is Day {day} ({date}).

    ## ABSOLUTE RULES — apply these before writing a single agenda item

    1. TICKET OWNERSHIP IS LOCKED.
    - An engineer may only have ticket_progress items for tickets in their OWNED list below.
    - Do NOT assign owned tickets to anyone else.
    - Do NOT invent ticket IDs.
    - related_id MUST come from the acting engineer's OWNED list, or be null.

    2. CAPACITY LIMITS.
    - Stay within each engineer's CAPACITY hours listed below.
    - estimated_hrs across all agenda items must not exceed their available hours.

    3. COMPANY CONTEXT — {company} IS ESTABLISHED, NOT A STARTUP.
    - Day {day} is the first day we are observing them, NOT the founding day.
    - They have years of existing code, legacy systems, and established processes.
    - DO write about maintaining systems, paying down tech debt, iterating on
        existing features, and routine corporate work.

    4. NON-ENGINEERING TEAMS (applies if {dept} is not Engineering_Backend or Engineering_Mobile).
    - Do NOT propose ticket_progress, pr_review, or code-related activities.
    - DO use these activity types for your team:
        * deep_work        — focused individual work (analysis, writing, planning)
        * 1on1             — check-in with a team member
        * async_question   — pinging another department for info or a decision
        * design_discussion — collaborative session to align on approach
        * mentoring        — senior helping junior
        * confluence_page  — writing internal documentation, playbooks, runbooks,
                            or process guides relevant to your team's expertise.
                            Use this at least once per day per department.
    - Examples by department:
        * Design     → design_discussion, confluence_page (design system docs, UX guidelines)
        * Sales      → async_question (pinging PM for roadmap), confluence_page (sales playbook)
        * HR_Ops     → 1on1 (wellbeing check), confluence_page (onboarding guide, PTO policy)
        * QA_Support → design_discussion (test plan), confluence_page (QA runbook)

    5. NO EVENT REDUNDANCY (CRITICAL TO AVOID DUPLICATES).
    - NEVER put collaborative meetings (1on1, mentoring, design_discussion, async_question) in the individual agendas of BOTH participants.
    - Assign the meeting to the INITIATOR'S agenda ONLY. 
    - List the other participants in the `collaborator` array.

    6. EXPERTISE ALIGNMENT IS STRICT.
    - Every agenda item MUST map directly to the assigned engineer's specific `Expertise` tags listed in the roster below.

    ---
    ## YOUR TASK

    1. Write a department theme for today (max 10 words) that reflects what 
    YOUR {dept} team is specifically doing — NOT a restatement of the org 
    theme. The org theme is context, not your theme. A Sales team theme 
    should sound like sales work. An HR theme should sound like HR work.
    2. Provide a 1-sentence reasoning for the overall plan.
    3. For each team member, write a 1-3 item agenda (keep descriptions under 6 words).

    ## BEFORE YOU OUTPUT — verify each of these:
    [ ] No collaborative meeting appears in more than one engineer's agenda
    [ ] All related_ids are null or from that engineer's own owned ticket list
    [ ] No engineer's estimated_hrs total exceeds their listed capacity

    Only output the JSON after confirming all three.

    ---
    ## OUTPUT SCHEMA

    Respond ONLY with valid JSON matching this exact schema.
    Do not include any text before or after the JSON block.

    {{
        "dept_theme": "string — max 10 words",
        "planner_reasoning": "string — max 1 sentence",
        "engineer_plans": [
            {{
                "name": "string — must match a name from YOUR TEAM TODAY",
                "focus_note": "string — max 6 words about headspace",
                "agenda": [
                    {{
                        "activity_type": "exactly one of: ticket_progress | pr_review | 1on1 | async_question | design_discussion | mentoring | deep_work",
                        "description": "string — max 6 words",
                        "related_id": "string — MUST be from owned tickets or null",
                        "collaborator": ["string"],
                        "estimated_hrs": float
                    }}
                ]
            }}
        ]
    }}

    ---
    ## CONTEXT DATA

    ORG THEME (context only — do NOT copy this into your dept_theme): {org_theme}
    SPRINT THEME: {sprint_theme}
    SYSTEM HEALTH: {system_health}/100
    TEAM MORALE: {morale_label}

    ### YOUR TEAM TODAY
    {roster}

    ### TICKET OWNERSHIP — owned tickets are locked to their listed engineer
    {owned_tickets_section}

    ### AVAILABLE TICKETS — unowned, assign freely within capacity
    {available_tickets_section}

    ### ENGINEER CAPACITY TODAY (hours available after stress/on-call)
    {capacity_section}

    ### RECENT DEPARTMENT HISTORY (last 2 days)
    {dept_history}

    ### CROSS-DEPARTMENT SIGNALS
    {cross_signals}

    {lifecycle_context}

    ### ACTIVE INCIDENTS
    {open_chains_str}
    """

    def __init__(
        self,
        dept: str,
        members: List[str],
        config: dict,
        worker_llm,
        clock,
        is_primary: bool = False,
    ):
        self.dept = dept
        self.members = members
        self.config = config
        self._llm = worker_llm
        self.is_primary = is_primary  # True for Engineering
        self.clock = clock

    def plan(
        self,
        org_theme: str,
        day: int,
        date: str,
        state,
        mem: Memory,
        graph_dynamics: GraphDynamics,
        cross_signals: List[CrossDeptSignal],
        sprint_context: Optional[SprintContext] = None,
        eng_plan: Optional[DepartmentDayPlan] = None,  # None for Engineering itself
        lifecycle_context: str = "",
        email_signals: Optional[List["ExternalEmailSignal"]] = None,
    ) -> DepartmentDayPlan:
        """
        Produce a DepartmentDayPlan. eng_plan is provided to non-Engineering
        departments so they can react to Engineering's agenda.
        sprint_context is pre-built by TicketAssigner — ownership is locked
        before this method runs, so the LLM only sees its legal menu.
        """
        roster = self._build_roster(graph_dynamics)
        dept_history = self._dept_history(mem, day)
        cross_str = self._format_cross_signals(cross_signals, eng_plan)
        email_str = self._format_email_signals(email_signals or [], self.dept)
        if email_str:
            cross_str = cross_str + "\n\n" + email_str
        known_str = ", ".join(sorted(KNOWN_EVENT_TYPES))
        morale_label = (
            "low"
            if state.team_morale < 0.45
            else "moderate"
            if state.team_morale < 0.70
            else "healthy"
        )
        lifecycle_context = (
            f"\nROSTER CHANGES (recent hires/departures):\n{lifecycle_context}\n"
            if lifecycle_context
            else ""
        )

        # ── Render SprintContext sections ──────────────────────────────────────
        if sprint_context:
            owned_lines = [
                f"  - [{tid}] → {owner}"
                for tid, owner in sprint_context.owned_tickets.items()
            ] or ["  (none — all tickets unassigned)"]
            owned_section = "\n".join(owned_lines)

            avail_lines = [f"  - [{tid}]" for tid in sprint_context.available_tickets]
            avail_section = (
                "\n".join(avail_lines) if avail_lines else "  (none available)"
            )

            cap_lines = [
                f"  - {name}: {hrs:.1f}h"
                for name, hrs in sprint_context.capacity_by_member.items()
            ]
            capacity_section = "\n".join(cap_lines)
        else:
            # Non-engineering depts or fallback — use the old open_tickets path
            owned_section = self._open_tickets(state, mem)
            avail_section = "  (see owned tickets above)"
            capacity_section = "  (standard 6h per engineer)"

        open_chains = []
        for inc in state.active_incidents:
            if getattr(inc, "causal_chain", None):
                open_chains.append(
                    f'- {inc.ticket_id} ({inc.stage}): "{inc.root_cause[:80]}"\n'
                    f"  On-call: {inc.on_call}. Day {inc.days_active} active."
                )

        lead_name = self.config.get("leads", {}).get(self.dept, "")
        lead_persona = PERSONAS.get(lead_name, DEFAULT_PERSONA)
        lead_stress = state.persona_stress.get(lead_name, 30)
        lead_style = lead_persona.get("style", "pragmatic and direct")
        lead_expertise = ", ".join(lead_persona.get("expertise", []))
        on_call_name = self.config.get("on_call_engineer", "")

        lead_context = (
            f"You are {lead_name}, lead of {self.dept} at "
            f"{self.config['simulation']['company_name']} which {COMPANY_DESCRIPTION}. "
            f"Your current stress is {lead_stress}/100"
            + (" — you are on-call today." if lead_name == on_call_name else ".")
            + f" Your style: {lead_style}. "
            f"Your expertise: {lead_expertise}. "
            f"You plan your team's day from lived experience, not abstraction. "
            f"You know who is struggling, what is mid-flight, and what can wait."
        )

        agent = make_agent(
            role=f"{lead_name}, {self.dept} Lead",
            goal="Plan your team's day honestly, given your stress, their capacity, and what is actually on fire.",
            backstory=lead_context,
            llm=self._llm,
        )

        prompt = self._PLAN_PROMPT.format(
            dept=self.dept,
            company=self.config["simulation"]["company_name"],
            company_description=COMPANY_DESCRIPTION,
            day=day,
            date=date,
            org_theme=org_theme,
            system_health=state.system_health,
            morale_label=morale_label,
            roster=roster,
            owned_tickets_section=owned_section,
            available_tickets_section=avail_section,
            capacity_section=capacity_section,
            dept_history=dept_history,
            cross_signals=cross_str,
            known_types=known_str,
            lifecycle_context=lifecycle_context,
            sprint_theme=sprint_context.sprint_theme if sprint_context else "",
            open_chains_str=open_chains,
        )
        task = Task(
            description=prompt,
            expected_output="Valid JSON only. No preamble, no markdown fences.",
            agent=agent,
        )

        raw = str(Crew(agents=[agent], tasks=[task], verbose=False).kickoff()).strip()
        result, raw_data = self._parse_plan(
            raw, org_theme, day, date, cross_signals, sprint_context
        )

        mem.log_dept_plan(
            day=day,
            date=date,
            dept=self.dept,
            lead=lead_name,
            theme=result.theme,
            engineer_plans=[_asdict(ep) for ep in result.engineer_plans],
            proposed_events=[_asdict(e) for e in result.proposed_events],
            raw=raw_data,
        )

        system_time_iso = self.clock.now("system").isoformat()

        mem.log_event(
            SimEvent(
                type="dept_plan_created",
                day=day,
                date=date,
                timestamp=system_time_iso,
                actors=[lead_name],
                artifact_ids={"dept_plan": f"PLAN-{day}-{self.dept}"},
                facts={
                    "dept": self.dept,
                    "theme": result.theme,
                    "lead": lead_name,
                    "engineer_plans": [_asdict(ep) for ep in result.engineer_plans],
                },
                summary=f"{self.dept} plan created for Day {day}. Theme: {result.theme}.",
                tags=["dept_plan_created", self.dept.lower()],
            )
        )

        return result

    # ─── Parsing ─────────────────────────────────────────────────────────────

    def _parse_plan(
        self,
        raw: str,
        org_theme: str,
        day: int,
        date: str,
        cross_signals: List[CrossDeptSignal],
        sprint_context: Optional[SprintContext] = None,
    ) -> Tuple[DepartmentDayPlan, dict]:
        """
        Parse the LLM JSON response into a DepartmentDayPlan.
        Defensively handles partial or malformed responses.

        Ownership enforcement is upstream (TicketAssigner) — this parser
        trusts the LLM to have respected the menu it was given.  A lightweight
        integrity check still catches any stray violations, but does not need
        to be the primary guardrail.
        """
        # Strip any accidental markdown fences
        clean = raw.replace("```json", "").replace("```", "").strip()

        try:
            data = json.loads(clean)
        except json.JSONDecodeError as e:
            logger.warning(
                f"[planner] {self.dept} plan JSON parse failed: {e}. Using fallback."
            )
            return self._fallback_plan(org_theme, day, date, cross_signals), {}

        # ── Build a fast-lookup set of valid (engineer → ticket) pairs ────────
        # If SprintContext is present we can enforce ownership at parse time
        # as a belt-and-suspenders check.  This should rarely fire now that
        # the LLM receives the locked menu.
        owned_by: Dict[str, str] = (
            sprint_context.owned_tickets if sprint_context else {}
        )
        owner_of: Dict[str, str] = {tid: eng for tid, eng in owned_by.items()}

        eng_plans: List[EngineerDayPlan] = []
        for ep in data.get("engineer_plans", []):
            name = ep.get("name", "")
            if name not in self.members:
                continue  # LLM invented a name — skip silently

            agenda = []
            for a in ep.get("agenda", []):
                activity_type = a.get("activity_type", "ticket_progress")
                related_id = a.get("related_id")

                # Belt-and-suspenders ownership check — catches the rare LLM
                # slip-through.  Logs a warning instead of silently stripping.
                if activity_type == "ticket_progress" and related_id and sprint_context:
                    actual_owner = owner_of.get(related_id)
                    if actual_owner and actual_owner != name:
                        logger.warning(
                            f"[planner] {self.dept}: {name} tried to claim "
                            f"{related_id} owned by {actual_owner} — stripped. "
                            f"(SprintContext menu may need review)"
                        )
                        continue

                # Get the flat list of all valid employee names
                all_valid_names = {
                    n for members in LIVE_ORG_CHART.values() for n in members
                }

                # Coerce the raw LLM output, but strictly filter out hallucinated names
                raw_collabs = _coerce_collaborators(a.get("collaborator"))
                valid_collabs = [c for c in raw_collabs if c in all_valid_names]

                agenda.append(
                    AgendaItem(
                        activity_type=activity_type,
                        description=a.get("description", ""),
                        related_id=related_id,
                        collaborator=valid_collabs,
                        estimated_hrs=float(a.get("estimated_hrs", 2.0)),
                    )
                )

            # Fallback agenda if LLM returned nothing useful
            if not agenda:
                agenda = [
                    AgendaItem(
                        activity_type="ticket_progress",
                        description="Continue assigned sprint work",
                        estimated_hrs=3.0,
                    )
                ]

            eng_plans.append(
                EngineerDayPlan(
                    name=name,
                    dept=self.dept,
                    agenda=agenda,
                    stress_level=0,  # will be patched by orchestrator after parse
                    focus_note=ep.get("focus_note", ""),
                )
            )

        # Ensure every member has a plan (even if LLM missed them)
        planned_names = {p.name for p in eng_plans}
        for name in self.members:
            if name not in planned_names:
                eng_plans.append(self._default_engineer_plan(name))

        seen_collaborations = set()
        for ep in eng_plans:
            unique_agenda = []
            for a in ep.agenda:
                if a.activity_type in (
                    "1on1",
                    "mentoring",
                    "design_discussion",
                    "async_question",
                ):
                    # Create a normalized key of the participants
                    participants = frozenset([ep.name] + a.collaborator)
                    collab_key = (a.activity_type, participants)

                    if collab_key in seen_collaborations:
                        continue  # We already kept the initiator's version of this meeting
                    seen_collaborations.add(collab_key)

                unique_agenda.append(a)
            ep.agenda = unique_agenda

        # ── Proposed events ───────────────────────────────────────────────────
        proposed: List[ProposedEvent] = []
        for pe in data.get("proposed_events", []):
            actors = [a for a in pe.get("actors", []) if a]
            if not actors:
                actors = self.members[:1]
            proposed.append(
                ProposedEvent(
                    event_type=pe.get("event_type", "normal_day_slack"),
                    actors=actors,
                    rationale=pe.get("rationale", ""),
                    facts_hint=pe.get("facts_hint", {}),
                    priority=int(pe.get("priority", 2)),
                    is_novel=bool(pe.get("is_novel", False)),
                    artifact_hint=pe.get("artifact_hint"),
                )
            )

        return DepartmentDayPlan(
            dept=self.dept,
            theme=data.get("dept_theme", org_theme),
            engineer_plans=eng_plans,
            proposed_events=sorted(proposed, key=lambda e: e.priority),
            cross_dept_signals=cross_signals,
            planner_reasoning=data.get("planner_reasoning", ""),
            day=day,
            date=date,
            sprint_context=sprint_context,
        ), data

    # ─── Context builders ─────────────────────────────────────────────────────

    def _build_roster(self, graph_dynamics: GraphDynamics) -> str:
        lines = []
        for name in self.members:
            stress = graph_dynamics._stress.get(name, 30)
            tone = graph_dynamics.stress_tone_hint(name)

            # Fetch expertise from the persona config
            persona = self.config.get("personas", {}).get(name, {})
            expertise = ", ".join(persona.get("expertise", ["general operations"]))

            # Inject expertise into the LLM's view of the roster
            lines.append(
                f"  - {name} (Expertise: [{expertise}]): stress={stress}/100. {tone}"
            )
        return "\n".join(lines)

    def _open_tickets(self, state, mem: Memory) -> str:
        tickets = list(
            mem._jira.find(
                {
                    "assignee": {"$in": self.members},
                    "status": {"$ne": "Done"},
                }
            )
        )
        if not tickets:
            return "  (no open tickets assigned to this team)"
        return "\n".join(
            f"  - [{t['id']}] {t['title']} — assigned to {t['assignee']}"
            for t in tickets[:8]  # cap at 8 to keep prompt tight
        )

    def _dept_history(self, mem: Memory, day: int) -> str:
        """
        Last 2 day_summary SimEvents filtered to this dept's actors.

        Capped at 2 days (down from 7) — older history adds noise without
        changing planning decisions, and open_incidents is omitted here since
        it's already listed in the ACTIVE INCIDENTS section of the prompt.
        Non-engineering depts skip days where they had no active members.
        """
        summaries = [
            e
            for e in mem.get_event_log()
            if e.type == "day_summary" and e.day >= max(1, day - 2)
        ]
        if not summaries:
            return "  (no recent history)"
        lines = []
        for s in summaries[-2:]:
            dept_actors = [
                a for a in s.facts.get("active_actors", []) if a in self.members
            ]
            if not dept_actors and not self.is_primary:
                continue  # dept was quiet — skip rather than add empty line
            lines.append(
                f"  Day {s.day}: health={s.facts.get('system_health')} "
                f"morale={s.facts.get('morale_trend', '?')} "
                f"dominant={s.facts.get('dominant_event', '?')} "
                f"active={dept_actors}"
            )
        return "\n".join(lines) if lines else "  (dept was quiet recently)"

    def _format_cross_signals(
        self,
        signals: List[CrossDeptSignal],
        eng_plan: Optional[DepartmentDayPlan],
    ) -> str:
        lines = []

        # Non-engineering departments only receive [direct] signals — indirect
        # signals (e.g. older incidents from other teams) are noise for Sales,
        # Design, QA etc. and waste tokens without changing planning decisions.
        # Engineering (is_primary) still sees all signals ranked by priority.
        if not self.is_primary:
            signals = [s for s in signals if s.relevance == "direct"]

        # Prioritize high-signal events and cap at 2 for non-engineering,
        # 4 for engineering (primary driver needs fuller picture).
        priority_ranking = {
            "incident_opened": 0,
            "customer_escalation": 1,
            "incident_resolved": 2,
            "postmortem_created": 3,
        }
        sorted_signals = sorted(
            signals, key=lambda s: priority_ranking.get(s.event_type, 99)
        )
        cap = 4 if self.is_primary else 2
        capped_signals = sorted_signals[:cap]

        for s in capped_signals:
            lines.append(
                f"  [{s.source_dept}] {s.event_type} (Day {s.day}): {s.summary} [{s.relevance}]"
            )

        if eng_plan and not self.is_primary:
            eng_lines = [
                f"    - {ep.name}: {ep.agenda[0].description if ep.agenda else '?'}"
                for ep in eng_plan.engineer_plans[:3]
            ]
            if eng_lines:
                lines.append(
                    "\n  ENGINEERING TODAY (awareness only — do NOT mirror these topics; "
                    "plan your own department's work in response to this context, not in parallel to it):"
                )
                lines.extend(eng_lines)

        return "\n".join(lines) if lines else "  (no cross-dept signals today)"

    def _format_email_signals(
        self,
        signals: List[ExternalEmailSignal],
        liaison_dept: str,
    ) -> str:
        # Only vendor signals reach planner prompts;
        # customer signals carry forward as CrossDeptSignals tomorrow.
        relevant = [
            s
            for s in signals
            if not s.dropped
            and s.category == "vendor"
            and s.internal_liaison.lower() == liaison_dept.lower()
        ]
        if not relevant:
            return ""
        lines = ["### INBOUND VENDOR EMAILS THIS MORNING (act on these if relevant)"]
        for s in relevant:
            lines.append(
                f"  From: {s.source_name} ({s.source_org})\n"
                f'  Subject: "{s.subject}"\n'
                f"  Preview: {s.body_preview}"
            )
        return "\n".join(lines)

    # ─── Fallbacks ────────────────────────────────────────────────────────────

    def _fallback_plan(
        self,
        org_theme: str,
        day: int,
        date: str,
        cross_signals: List[CrossDeptSignal],
    ) -> DepartmentDayPlan:
        """Minimal valid plan when LLM output is unparseable."""
        return DepartmentDayPlan(
            dept=self.dept,
            theme=org_theme,
            engineer_plans=[self._default_engineer_plan(n) for n in self.members],
            proposed_events=[
                ProposedEvent(
                    event_type="normal_day_slack",
                    actors=self.members[:2],
                    rationale="Fallback: LLM plan unparseable.",
                    facts_hint={},
                    priority=3,
                )
            ],
            cross_dept_signals=cross_signals,
            planner_reasoning="Fallback plan — LLM response was not valid JSON.",
            day=day,
            date=date,
        )

    def _default_engineer_plan(self, name: str) -> EngineerDayPlan:
        return EngineerDayPlan(
            name=name,
            dept=self.dept,
            agenda=[
                AgendaItem(
                    activity_type="ticket_progress",
                    description="Continue assigned sprint work",
                    estimated_hrs=3.0,
                )
            ],
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
    You are the Org Coordinator for {company} which {company_description} on Day {day}.
    Your job is to find ONE realistic cross-department interaction that neither
    department planned for — a natural collision caused by stress, competing
    priorities, or misaligned timing.

    ## ABSOLUTE RULES

    1.VALID ACTOR NAMES — ONLY these names may appear in "actors":
    {all_names}

    2. {company} IS AN ESTABLISHED COMPANY.
    - Do not treat Day {day} as a founding day or early-stage moment.
    - Collisions should reflect mature org dynamics: competing priorities,
        resource contention, communication gaps — not greenfield chaos.

    3. PRIORITIZE FRICTION WHEN STRESS IS HIGH.
    - If a department lead has stress > 60, their department is a likely
        collision source. Use the stress levels below to pick realistic actors.

    4. IF NO COLLISION IS WARRANTED, SAY SO.
    - If morale is healthy and no department themes conflict, return:
        "collision": null
    - Do not invent friction just to fill the field.

    5. TENSION LEVEL must be exactly one of: "high", "medium", "low"

    ---
    ## COLLISION ARCHETYPES (use these as inspiration, not a checklist)

    - "The Accountability Check": A stressed PM pings Engineering lead for an ETA.
    - "The Scope Creep": Sales asks an Engineer for a "quick favor" mid-incident.
    - "The Wellness Friction": HR tries to hold a meeting while Eng is firefighting.
    - "The Knowledge Gap": Someone realizes a departed employee owned a critical service.

    ---
    ## OUTPUT SCHEMA

    Respond ONLY with valid JSON. No preamble, no markdown fences.

    {{
        "collision": {{
            "event_type": "string — a specific, descriptive name for this interaction",
            "actors": ["FirstName", "FirstName"],
            "rationale": "string — why these specific people are clashing TODAY given their stress and themes",
            "facts_hint": {{
                "tension_level": "high | medium | low"
            }},
            "priority": "int — 1=must fire, 2=should fire, 3=optional",
            "artifact_hint": "slack | email | confluence | jira"
        }}
    }}

    ---
    ## CONTEXT DATA

    ORG STATE: health={health}, morale={morale_label}

    DEPT PLANS & HEADSPACE:
    {other_plans_with_stress}
    """

    def __init__(self, config: dict, planner_llm):
        self._config = config
        self._llm = planner_llm
        self._all_names_str = "\n".join(
            f"  {dept}: {', '.join(members)}"
            for dept, members in LIVE_ORG_CHART.items()
        )

    def coordinate(
        self,
        dept_plans: Dict[str, DepartmentDayPlan],
        state,
        day: int,
        date: str,
        org_theme: str,
    ) -> OrgDayPlan:
        # Build a richer summary of other plans that includes lead stress and member names
        other_plans_str = ""
        for dept, plan in dept_plans.items():
            lead_name = self._config.get("leads", {}).get(dept)
            stress = state.persona_stress.get(lead_name, 50) if lead_name else 50
            members = self._config.get("org_chart", {}).get(dept, [])
            other_plans_str += (
                f"- {dept}: Theme='{plan.theme}'. Lead={lead_name} (Stress: {stress}/100). "
                f"Members: {members}. "
                f"Events planned: {[e.event_type for e in plan.proposed_events[:2]]}\n"
            )

        morale_label = "low" if state.team_morale < 0.45 else "healthy"

        prompt = self._COORD_PROMPT.format(
            company=self._config["simulation"]["company_name"],
            company_description=COMPANY_DESCRIPTION,
            day=day,
            other_plans_with_stress=other_plans_str,
            health=state.system_health,
            morale_label=morale_label,
            all_names=self._all_names_str,
        )
        agent = make_agent(
            role="Org Conflict Coordinator",
            goal="Identify realistic friction points between departments.",
            backstory="You understand that in high-growth companies, departments often have conflicting priorities and personalities.",
            llm=self._llm,
        )
        task = Task(
            description=prompt,
            expected_output="Valid JSON only. No preamble.",
            agent=agent,
        )

        raw = str(Crew(agents=[agent], tasks=[task], verbose=False).kickoff()).strip()
        clean = raw.replace("```json", "").replace("```", "").strip()

        collision_events: List[ProposedEvent] = []
        reasoning = ""

        try:
            raw_data = json.loads(clean)
            if isinstance(raw_data, list):
                data = raw_data[0] if len(raw_data) > 0 else {}
            elif isinstance(raw_data, dict):
                data = raw_data
            else:
                data = {}
            reasoning = data.get("reasoning", "")
            col = data.get("collision")
            if col:
                actors = col.get("actors", [])
                if actors:
                    collision_events.append(
                        ProposedEvent(
                            event_type=col.get("event_type", "leadership_sync"),
                            actors=actors,
                            rationale=col.get("rationale", ""),
                            facts_hint=col.get("facts_hint", {}),
                            priority=int(col.get("priority", 1)),
                            artifact_hint=col.get("artifact_hint"),
                        )
                    )
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
        eng_key: Optional[str],
    ) -> str:
        lines = []
        for dept, plan in dept_plans.items():
            if dept == eng_key:
                continue
            events_str = ", ".join(e.event_type for e in plan.proposed_events[:2])
            # Extract the names of the people in this department
            names = ", ".join(ep.name for ep in plan.engineer_plans)

            lines.append(
                f"  {dept} (Members: {names}): theme='{plan.theme}' "
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

    def __init__(self, config: dict, worker_llm, planner_llm, clock):
        self._config = config
        self._worker_llm = worker_llm
        self._planner_llm = planner_llm
        self._clock = clock

        # Build one DepartmentPlanner per department
        self._dept_planners: Dict[str, DepartmentPlanner] = {}
        for dept, members in LIVE_ORG_CHART.items():
            is_primary = "eng" in dept.lower()
            self._dept_planners[dept] = DepartmentPlanner(
                dept=dept,
                members=members,
                config=config,
                worker_llm=worker_llm,
                is_primary=is_primary,
                clock=self._clock,
            )

        self._coordinator = OrgCoordinator(config, planner_llm)

        all_names = [n for members in LIVE_ORG_CHART.values() for n in members]
        external_names = [c["name"] for c in config.get("external_contacts", [])]
        self._validator = PlanValidator(
            all_names=all_names,
            external_contact_names=external_names,
            config=config,
        )
        # TicketAssigner is stateless per-call — one instance for the whole sim
        self._ticket_assigner: Optional[TicketAssigner] = None  # set on first plan()

    def plan(
        self,
        state,
        mem: Memory,
        graph_dynamics: GraphDynamics,
        clock,
        lifecycle_context: str = "",
        email_signals: Optional[List["ExternalEmailSignal"]] = None,
    ) -> OrgDayPlan:
        """
        Full planning pass for one day.
        Returns an OrgDayPlan the day loop executes against.
        """

        import asyncio

        try:
            loop = asyncio.get_running_loop()
            logger.warning(
                f"[planner] Running event loop detected: {loop} — parallel dept plans will serialize"
            )
        except RuntimeError:
            logger.info(
                "[planner] No running event loop — ThreadPoolExecutor parallelism should work"
            )
        day = state.day
        date = str(state.current_date.date())
        system_time_iso = clock.now("system").isoformat()  # This will be 09:00:00

        # ── Pass 1: deterministic ticket assignment (Option C) ────────────────
        # Build SprintContext for every dept before any LLM call so the LLM
        # only ever sees the tickets it is legally allowed to plan against.
        if self._ticket_assigner is None:
            self._ticket_assigner = TicketAssigner(self._config, graph_dynamics, mem)

        sprint_contexts: Dict[str, SprintContext] = {}
        for dept, members in LIVE_ORG_CHART.items():
            if dept not in LEADS:
                continue
            sprint_contexts[dept] = self._ticket_assigner.build(
                state, members, dept_name=dept
            )

        # Seed ticket_actors_today from the locked assignments so the validator
        # has a populated map from the very start of the day.
        state.ticket_actors_today = {}
        for dept, ctx in sprint_contexts.items():
            for tid, owner in ctx.owned_tickets.items():
                state.ticket_actors_today.setdefault(tid, set()).add(owner)

        # ── Generate org theme (lightweight — replaces _generate_theme()) ─────
        org_theme = self._generate_org_theme(state, mem, clock)

        # ── Build cross-dept signals from recent SimEvents ────────────────────
        cross_signals_by_dept = self._extract_cross_signals(mem, day)

        # ── Engineering plans first — it drives everyone else ─────────────────
        eng_key = next((k for k in self._dept_planners if "eng" in k.lower()), None)
        eng_plan = None

        dept_plans: Dict[str, DepartmentDayPlan] = {}

        if eng_key:
            eng_plan = self._dept_planners[eng_key].plan(
                org_theme=org_theme,
                day=day,
                date=date,
                state=state,
                mem=mem,
                graph_dynamics=graph_dynamics,
                cross_signals=cross_signals_by_dept.get(eng_key, []),
                sprint_context=sprint_contexts.get(eng_key),
                eng_plan=None,
                email_signals=email_signals,
            )
            self._patch_stress_levels(eng_plan, graph_dynamics)
            dept_plans[eng_key] = eng_plan
            logger.info(
                f"  [blue]📋 Eng plan:[/blue] {eng_plan.theme[:60]} "
                f"({len(eng_plan.proposed_events)} events)"
            )

        # ── Other departments react to Engineering — run in parallel ─────────
        # Each non-eng dept plan is an independent Bedrock call with no shared
        # mutable state between departments. eng_plan is read-only at this point.
        # graph_dynamics._stress is also read-only here (patched after each
        # result comes in, under lock). MongoDB writes inside planner.plan()
        # (log_dept_plan, log_event) are thread-safe via PyMongo's pool.
        non_eng_depts = {
            dept: planner
            for dept, planner in self._dept_planners.items()
            if dept != eng_key
            and not (
                len(self._config["org_chart"].get(dept, [])) == 1
                and dept.upper() == "CEO"
            )
        }

        if non_eng_depts:
            lock = threading.Lock()
            with ThreadPoolExecutor(max_workers=len(non_eng_depts)) as ex:
                futures = {
                    ex.submit(
                        planner.plan,
                        org_theme=org_theme,
                        day=day,
                        date=date,
                        state=state,
                        mem=mem,
                        graph_dynamics=graph_dynamics,
                        cross_signals=cross_signals_by_dept.get(dept, []),
                        sprint_context=sprint_contexts.get(dept),
                        eng_plan=eng_plan,
                        lifecycle_context=lifecycle_context,
                        email_signals=email_signals,
                    ): dept
                    for dept, planner in non_eng_depts.items()
                }
                for future in as_completed(futures):
                    dept = futures[future]
                    try:
                        plan = future.result()
                        self._patch_stress_levels(plan, graph_dynamics)
                        with lock:
                            dept_plans[dept] = plan
                        logger.info(
                            f"  [blue]📋 {dept} plan:[/blue] {plan.theme[:60]} "
                            f"({len(plan.proposed_events)} events)"
                        )
                    except Exception as e:
                        logger.error(
                            f"  [red]✗ {dept} plan failed:[/red] {e} — using fallback"
                        )
                        # Fallback: empty plan so the rest of the day can proceed
                        with lock:
                            dept_plans[dept] = self._dept_planners[dept]._fallback_plan(
                                org_theme, day, date, []
                            )

        # ── OrgCoordinator finds collisions ───────────────────────────────────
        org_plan = self._coordinator.coordinate(dept_plans, state, day, date, org_theme)

        # ── Validate all proposed events ──────────────────────────────────────
        recent_summaries = self._recent_day_summaries(mem, day)
        all_proposed = org_plan.all_events_by_priority()
        results = self._validator.validate_plan(all_proposed, state, recent_summaries)

        # Log rejections as SimEvents so researchers can see what was blocked
        for r in self._validator.rejected(results):
            mem.log_event(
                SimEvent(
                    type="proposed_event_rejected",
                    timestamp=system_time_iso,
                    day=day,
                    date=date,
                    actors=r.event.actors,
                    artifact_ids={},
                    facts={
                        "event_type": r.event.event_type,
                        "rejection_reason": r.rejection_reason,
                        "rationale": r.event.rationale,
                        "was_novel": r.was_novel,
                    },
                    summary=f"Rejected: {r.event.event_type} — {r.rejection_reason}",
                    tags=["validation", "rejected"],
                )
            )

        # Log novel events the community could implement
        for novel in self._validator.drain_novel_log():
            mem.log_event(
                SimEvent(
                    type="novel_event_proposed",
                    timestamp=system_time_iso,
                    day=day,
                    date=date,
                    actors=novel.actors,
                    artifact_ids={},
                    facts={
                        "event_type": novel.event_type,
                        "rationale": novel.rationale,
                        "artifact_hint": novel.artifact_hint,
                        "facts_hint": novel.facts_hint,
                    },
                    summary=f"Novel event proposed: {novel.event_type}. {novel.rationale}",
                    tags=["novel", "proposed"],
                )
            )

        # Rebuild dept_plans with only approved events
        approved_set = {id(e) for e in self._validator.approved(results)}
        for dept, dplan in org_plan.dept_plans.items():
            dplan.proposed_events = [
                e for e in dplan.proposed_events if id(e) in approved_set
            ]
        org_plan.collision_events = [
            e for e in org_plan.collision_events if id(e) in approved_set
        ]

        org_plan.sprint_contexts = sprint_contexts

        return org_plan

    # ─── Helpers ─────────────────────────────────────────────────────────────

    def _generate_org_theme(self, state, mem: Memory, clock) -> str:
        ctx = mem.previous_day_context(state.day)

        # ── Resolve CEO persona ────────────────────────────────────────────
        from flow import PERSONAS, DEFAULT_PERSONA

        ceo_name = resolve_role("ceo")
        ceo_persona = PERSONAS.get(ceo_name, DEFAULT_PERSONA)
        ceo_stress = state.persona_stress.get(ceo_name, 30)
        ceo_style = ceo_persona.get("style", "strategic and decisive")

        agent = make_agent(
            role=f"{ceo_name}, CEO" if ceo_name else "CEO",
            goal="Decide today's dominant org theme based on what you know about your company right now.",
            backstory=(
                f"You are {ceo_name}, CEO of "
                f"{self._config['simulation']['company_name']} which {COMPANY_DESCRIPTION}. "
                f"Your current stress is {ceo_stress}/100. "
                f"Your style: {ceo_style}. "
                f"You set the tone for the whole org each morning based on "
                f"what kept you up last night and what needs to change today."
            ),
            llm=self._worker_llm,
        )
        task = Task(
            description=(
                f"Day {state.day}. "
                f"System health: {state.system_health}/100. "
                f"Team morale: {state.team_morale:.2f}.\n\n"
                f"Recent context (use this to make the theme specific and grounded):\n"
                f"{ctx}\n\n"
                f"Write exactly ONE sentence for today's org-wide theme. "
                f"The sentence must reflect the actual state above — reference a specific "
                f"pressure, milestone, or mood. "
                f"Output the sentence only, with no label, no quotes, no explanation."
            ),
            expected_output=(
                "A single plain sentence with no preamble, no quotes, and no label. "
                "Example: 'The team is heads-down on stabilizing auth after yesterday's outage.'"
            ),
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
            "incident_resolved",
            "incident_opened",
            "postmortem_created",
            "feature_request_from_sales",
            "customer_escalation",
            "morale_intervention",
            "hr_checkin",
            "customer_email_routed",
            "customer_escalation",
        }

        recent = [
            e
            for e in mem.get_event_log()
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
                break  # one signal per event

        return signals

    def _patch_stress_levels(
        self,
        plan: DepartmentDayPlan,
        graph_dynamics: GraphDynamics,
    ):
        """Fills in stress_level on each EngineerDayPlan after parsing."""
        for ep in plan.engineer_plans:
            ep.stress_level = graph_dynamics._stress.get(ep.name, 30)

    def _recent_day_summaries(self, mem: Memory, day: int) -> List[dict]:
        """Last 7 day_summary facts dicts for the validator. Queries MongoDB."""
        return mem.get_recent_day_summaries(current_day=day, window=7)
