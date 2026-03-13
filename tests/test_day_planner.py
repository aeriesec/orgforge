"""
test_day_planner.py
===================
Unit tests for day_planner.py — DepartmentPlanner, OrgCoordinator,
and DayPlannerOrchestrator.

Mirrors the patterns established in test_normal_day.py:
  - conftest.py supplies make_test_memory and mock_config_and_db
  - LLM calls (Agent / Task / Crew) are patched at the module level
  - Pure-logic helpers are tested without patching where possible
"""

from __future__ import annotations

import copy
import json
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch


from planner_models import (
    AgendaItem,
    CrossDeptSignal,
    DepartmentDayPlan,
    EngineerDayPlan,
    OrgDayPlan,
    ProposedEvent,
    SprintContext,
)
from memory import SimEvent
from day_planner import (
    DepartmentPlanner,
    OrgCoordinator,
    DayPlannerOrchestrator,
    _coerce_collaborators,
)


# ─────────────────────────────────────────────────────────────────────────────
# SHARED CONSTANTS & HELPERS
# ─────────────────────────────────────────────────────────────────────────────

ORG_CHART = {
    "Engineering": ["Alice", "Bob"],
    "Sales": ["Dave"],
}

LEADS = {"Engineering": "Alice", "Sales": "Dave"}

PERSONAS = {
    "Alice": {
        "style": "direct",
        "expertise": ["backend"],
        "tenure": "senior",
        "stress": 30,
    },
    "Bob": {"style": "casual", "expertise": ["infra"], "tenure": "mid", "stress": 25},
    "Dave": {
        "style": "assertive",
        "expertise": ["sales"],
        "tenure": "senior",
        "stress": 40,
    },
}

CONFIG = {
    "simulation": {
        "company_name": "TestCorp",
        "domain": "testcorp.com",
        "start_date": "2026-01-01",
        "max_days": 1,
    },
    "model_presets": {"local_cpu": {"planner": "mock", "worker": "mock"}},
    "quality_preset": "local_cpu",
    "org_chart": ORG_CHART,
    "leads": LEADS,
    "personas": PERSONAS,
    "default_persona": {
        "style": "standard",
        "expertise": [],
        "tenure": "1y",
        "stress": 10,
    },
    "legacy_system": {
        "name": "OldDB",
        "description": "Legacy",
        "project_name": "Modernize",
    },
    "morale": {"initial": 0.8, "daily_decay": 0.99, "good_day_recovery": 0.05},
    "roles": {
        "on_call_engineer": "Alice",
        "incident_commander": "Bob",
        "postmortem_writer": "Bob",
    },
    "incident_triggers": ["crash", "fail", "error"],
    "external_contacts": [],
}

VALID_PLAN_JSON = {
    "dept_theme": "Steady progress on reliability improvements",
    "engineer_plans": [
        {
            "name": "Alice",
            "focus_note": "Working on auth refactor",
            "agenda": [
                {
                    "activity_type": "ticket_progress",
                    "description": "Fix auth retry logic",
                    "related_id": "ENG-101",
                    "collaborator": [],
                    "estimated_hrs": 3.0,
                },
                {
                    "activity_type": "pr_review",
                    "description": "Review Bob's infra PR",
                    "related_id": None,
                    "collaborator": ["Bob"],
                    "estimated_hrs": 1.0,
                },
            ],
        },
        {
            "name": "Bob",
            "focus_note": "Infra hardening",
            "agenda": [
                {
                    "activity_type": "deep_work",
                    "description": "K8s node pool resizing",
                    "related_id": "ENG-102",
                    "collaborator": [],
                    "estimated_hrs": 4.0,
                }
            ],
        },
    ],
    "proposed_events": [
        {
            "event_type": "normal_day_slack",
            "actors": ["Alice", "Bob"],
            "rationale": "Daily standup",
            "facts_hint": {},
            "priority": 2,
            "is_novel": False,
            "artifact_hint": None,
        }
    ],
}


def _make_mock_state(day: int = 5, morale: float = 0.75, health: int = 80) -> MagicMock:
    state = MagicMock()
    state.day = day
    state.current_date = datetime(2026, 1, day)
    state.daily_theme = "Reliability focus"
    state.team_morale = morale
    state.system_health = health
    state.persona_stress = {"Alice": 30, "Bob": 25, "Dave": 40}
    state.active_incidents = []
    state.ticket_actors_today = {}
    return state


def _make_sprint_context(
    owned: dict | None = None,
    available: list | None = None,
    members: list | None = None,
) -> SprintContext:
    return SprintContext(
        owned_tickets=owned or {"ENG-101": "Alice", "ENG-102": "Bob"},
        available_tickets=available or [],
        capacity_by_member={"Alice": 6.0, "Bob": 6.0}
        if members is None
        else {m: 6.0 for m in members},
        sprint_theme="Q1 reliability hardening",
        in_progress_ids=[],
    )


def _make_dept_plan(
    dept: str = "Engineering",
    members: list | None = None,
    theme: str = "Steady",
    day: int = 5,
) -> DepartmentDayPlan:
    names = members or ["Alice", "Bob"]
    eng_plans = [
        EngineerDayPlan(
            name=n,
            dept=dept,
            agenda=[
                AgendaItem(
                    activity_type="ticket_progress",
                    description="Sprint work",
                    estimated_hrs=3.0,
                )
            ],
            stress_level=25,
        )
        for n in names
    ]
    return DepartmentDayPlan(
        dept=dept,
        theme=theme,
        engineer_plans=eng_plans,
        proposed_events=[
            ProposedEvent(
                event_type="normal_day_slack",
                actors=names,
                rationale="Standup",
                facts_hint={},
                priority=2,
            )
        ],
        cross_dept_signals=[],
        planner_reasoning="",
        day=day,
        date=f"2026-01-0{day}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_llm():
    return MagicMock()


@pytest.fixture
def dept_planner(mock_llm):
    return DepartmentPlanner(
        dept="Engineering",
        members=["Alice", "Bob"],
        config=CONFIG,
        worker_llm=mock_llm,
        is_primary=True,
        clock=MagicMock(),
    )


@pytest.fixture
def mock_graph_dynamics():
    gd = MagicMock()
    gd._stress = {"Alice": 30, "Bob": 25, "Dave": 40}
    gd.stress_tone_hint.return_value = "calm and focused"
    return gd


@pytest.fixture
def mock_mem(make_test_memory):
    mem = make_test_memory
    mem.context_for_prompt = MagicMock(return_value="some context")
    mem.get_event_log = MagicMock(return_value=[])
    mem.get_recent_day_summaries = MagicMock(return_value=[])
    return mem


# ─────────────────────────────────────────────────────────────────────────────
# 1. _coerce_collaborators utility
# ─────────────────────────────────────────────────────────────────────────────


class TestCoerceCollaborators:
    def test_none_returns_empty_list(self):
        assert _coerce_collaborators(None) == []

    def test_empty_string_returns_empty_list(self):
        assert _coerce_collaborators("") == []

    def test_list_passthrough(self):
        assert _coerce_collaborators(["Alice", "Bob"]) == ["Alice", "Bob"]

    def test_single_string_wrapped_in_list(self):
        assert _coerce_collaborators("Alice") == ["Alice"]

    def test_empty_list_returns_empty(self):
        assert _coerce_collaborators([]) == []


# ─────────────────────────────────────────────────────────────────────────────
# 2. DepartmentPlanner._parse_plan — JSON parsing & model construction
# ─────────────────────────────────────────────────────────────────────────────


class TestDepartmentPlannerParsePlan:
    def test_valid_json_produces_correct_theme(self, dept_planner):
        raw = json.dumps(VALID_PLAN_JSON)
        plan, _ = dept_planner._parse_plan(raw, "Focus day", 5, "2026-01-05", [])
        assert plan.theme == "Steady progress on reliability improvements"

    def test_valid_json_produces_engineer_plans_for_all_members(self, dept_planner):
        raw = json.dumps(VALID_PLAN_JSON)
        plan, _ = dept_planner._parse_plan(raw, "Focus day", 5, "2026-01-05", [])
        names = {ep.name for ep in plan.engineer_plans}
        assert names == {"Alice", "Bob"}

    def test_valid_json_produces_proposed_events(self, dept_planner):
        raw = json.dumps(VALID_PLAN_JSON)
        plan, _ = dept_planner._parse_plan(raw, "Focus day", 5, "2026-01-05", [])
        assert len(plan.proposed_events) == 1
        assert plan.proposed_events[0].event_type == "normal_day_slack"

    def test_proposed_events_sorted_by_priority(self, dept_planner):
        data = copy.deepcopy(VALID_PLAN_JSON)
        data["proposed_events"] = [
            {
                "event_type": "low_pri",
                "actors": ["Alice"],
                "rationale": "",
                "facts_hint": {},
                "priority": 3,
                "is_novel": False,
                "artifact_hint": None,
            },
            {
                "event_type": "high_pri",
                "actors": ["Bob"],
                "rationale": "",
                "facts_hint": {},
                "priority": 1,
                "is_novel": False,
                "artifact_hint": None,
            },
        ]
        plan, _ = dept_planner._parse_plan(
            json.dumps(data), "Theme", 5, "2026-01-05", []
        )
        assert plan.proposed_events[0].event_type == "high_pri"
        assert plan.proposed_events[1].event_type == "low_pri"

    def test_invalid_json_returns_fallback_plan(self, dept_planner):
        plan, _ = dept_planner._parse_plan(
            "{not valid json}", "Fallback theme", 5, "2026-01-05", []
        )
        assert plan.dept == "Engineering"
        assert len(plan.engineer_plans) == 2  # Alice + Bob
        assert any(e.event_type == "normal_day_slack" for e in plan.proposed_events)
        assert "Fallback" in plan.planner_reasoning

    def test_markdown_fences_stripped_before_parse(self, dept_planner):
        raw = "```json\n" + json.dumps(VALID_PLAN_JSON) + "\n```"
        plan, _ = dept_planner._parse_plan(raw, "Theme", 5, "2026-01-05", [])
        assert plan.theme == "Steady progress on reliability improvements"

    def test_invented_name_in_llm_response_is_dropped(self, dept_planner):
        data = copy.deepcopy(VALID_PLAN_JSON)
        data["engineer_plans"] = data["engineer_plans"] + [
            {
                "name": "InventedPerson",
                "focus_note": "Phantom work",
                "agenda": [
                    {
                        "activity_type": "deep_work",
                        "description": "Ghost task",
                        "related_id": None,
                        "collaborator": [],
                        "estimated_hrs": 2.0,
                    }
                ],
            }
        ]
        plan, _ = dept_planner._parse_plan(
            json.dumps(data), "Theme", 5, "2026-01-05", []
        )
        names = {ep.name for ep in plan.engineer_plans}
        assert "InventedPerson" not in names

    def test_missing_member_gets_default_fallback_plan(self, dept_planner):
        """If LLM omits a team member, they still receive a default agenda."""
        data = copy.deepcopy(VALID_PLAN_JSON)
        # Only Alice in the response — Bob is missing
        data["engineer_plans"] = [data["engineer_plans"][0]]
        plan, _ = dept_planner._parse_plan(
            json.dumps(data), "Theme", 5, "2026-01-05", []
        )
        names = {ep.name for ep in plan.engineer_plans}
        assert "Bob" in names

    def test_empty_agenda_for_member_gets_fallback_agenda(self, dept_planner):
        """If the LLM returns an empty agenda list for an engineer, a default item is inserted."""
        data = copy.deepcopy(VALID_PLAN_JSON)
        data["engineer_plans"][0]["agenda"] = []
        plan, _ = dept_planner._parse_plan(
            json.dumps(data), "Theme", 5, "2026-01-05", []
        )
        alice_plan = next(ep for ep in plan.engineer_plans if ep.name == "Alice")
        assert len(alice_plan.agenda) >= 1

    def test_collaborator_string_coerced_to_list(self, dept_planner):
        """LLM sometimes returns collaborator as a bare string — must be list."""
        data = copy.deepcopy(VALID_PLAN_JSON)
        data["engineer_plans"][0]["agenda"][0]["collaborator"] = "Bob"
        plan, _ = dept_planner._parse_plan(
            json.dumps(data), "Theme", 5, "2026-01-05", []
        )
        alice = next(ep for ep in plan.engineer_plans if ep.name == "Alice")
        assert isinstance(alice.agenda[0].collaborator, list)

    def test_ownership_violation_stripped_with_sprint_context(self, dept_planner):
        """Ticket claimed by wrong engineer is stripped when SprintContext is provided."""
        ctx = _make_sprint_context(owned={"ENG-101": "Alice", "ENG-102": "Bob"})
        data = copy.deepcopy(VALID_PLAN_JSON)
        # Bob tries to claim ENG-101 which belongs to Alice
        data["engineer_plans"][1]["agenda"] = [
            {
                "activity_type": "ticket_progress",
                "description": "Steal Alice's ticket",
                "related_id": "ENG-101",
                "collaborator": [],
                "estimated_hrs": 2.0,
            }
        ]
        plan, _ = dept_planner._parse_plan(
            json.dumps(data), "Theme", 5, "2026-01-05", [], sprint_context=ctx
        )
        bob_plan = next(ep for ep in plan.engineer_plans if ep.name == "Bob")
        related_ids = [a.related_id for a in bob_plan.agenda]
        assert "ENG-101" not in related_ids

    def test_cross_dept_signals_attached_to_plan(self, dept_planner):
        signals = [
            CrossDeptSignal(
                source_dept="Sales",
                event_type="customer_escalation",
                summary="Customer angry",
                day=4,
                relevance="direct",
            )
        ]
        plan, _ = dept_planner._parse_plan(
            json.dumps(VALID_PLAN_JSON), "Theme", 5, "2026-01-05", signals
        )
        assert len(plan.cross_dept_signals) == 1
        assert plan.cross_dept_signals[0].source_dept == "Sales"


# ─────────────────────────────────────────────────────────────────────────────
# 3. DepartmentPlanner._fallback_plan
# ─────────────────────────────────────────────────────────────────────────────


class TestDepartmentPlannerFallback:
    def test_fallback_includes_all_members(self, dept_planner):
        plan = dept_planner._fallback_plan("Fallback theme", 5, "2026-01-05", [])
        names = {ep.name for ep in plan.engineer_plans}
        assert names == {"Alice", "Bob"}

    def test_fallback_uses_org_theme_as_dept_theme(self, dept_planner):
        plan = dept_planner._fallback_plan("Org theme today", 5, "2026-01-05", [])
        assert plan.theme == "Org theme today"

    def test_fallback_proposes_normal_day_slack_event(self, dept_planner):
        plan = dept_planner._fallback_plan("Theme", 5, "2026-01-05", [])
        assert any(e.event_type == "normal_day_slack" for e in plan.proposed_events)

    def test_fallback_reasoning_mentions_fallback(self, dept_planner):
        plan = dept_planner._fallback_plan("Theme", 5, "2026-01-05", [])
        assert (
            "Fallback" in plan.planner_reasoning
            or "fallback" in plan.planner_reasoning.lower()
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4. DepartmentPlanner._default_engineer_plan
# ─────────────────────────────────────────────────────────────────────────────


class TestDefaultEngineerPlan:
    def test_default_plan_has_correct_name(self, dept_planner):
        ep = dept_planner._default_engineer_plan("Alice")
        assert ep.name == "Alice"

    def test_default_plan_has_one_ticket_progress_item(self, dept_planner):
        ep = dept_planner._default_engineer_plan("Bob")
        assert len(ep.agenda) == 1
        assert ep.agenda[0].activity_type == "ticket_progress"

    def test_default_plan_stress_level_is_30(self, dept_planner):
        ep = dept_planner._default_engineer_plan("Alice")
        assert ep.stress_level == 30


# ─────────────────────────────────────────────────────────────────────────────
# 5. DepartmentPlanner context-builder helpers
# ─────────────────────────────────────────────────────────────────────────────


class TestDepartmentPlannerHelpers:
    def test_build_roster_includes_all_members(self, dept_planner, mock_graph_dynamics):
        roster = dept_planner._build_roster(mock_graph_dynamics)
        assert "Alice" in roster
        assert "Bob" in roster

    def test_build_roster_contains_stress_values(
        self, dept_planner, mock_graph_dynamics
    ):
        mock_graph_dynamics._stress = {"Alice": 55, "Bob": 10}
        roster = dept_planner._build_roster(mock_graph_dynamics)
        assert "55" in roster
        assert "10" in roster

    def test_dept_history_returns_no_history_string_when_empty(
        self, dept_planner, mock_mem
    ):
        result = dept_planner._dept_history(mock_mem, day=5)
        assert "no recent history" in result

    def test_dept_history_filters_to_relevant_days(self, dept_planner, mock_mem):
        """Only events from the last 7 days should appear."""
        old_event = SimEvent(
            type="day_summary",
            timestamp="2026-01-01T09:00:00",
            day=1,
            date="2026-01-01",
            actors=["Alice"],
            artifact_ids={},
            facts={
                "active_actors": ["Alice"],
                "system_health": 80,
                "morale_trend": "stable",
                "dominant_event": "normal_day",
            },
            summary="Day 1",
        )
        mock_mem.get_event_log.return_value = [old_event]
        # day=10, window means day >= max(1, 10-7) = 3, so day=1 is excluded
        result = dept_planner._dept_history(mock_mem, day=10)
        assert "Day 1" not in result

    def test_format_cross_signals_no_signals(self, dept_planner):
        result = dept_planner._format_cross_signals([], None)
        assert "no cross-dept signals" in result

    def test_format_cross_signals_with_signals(self, dept_planner):
        signals = [
            CrossDeptSignal(
                source_dept="Sales",
                event_type="customer_escalation",
                summary="Big client unhappy",
                day=4,
                relevance="direct",
            )
        ]
        result = dept_planner._format_cross_signals(signals, None)
        # Change the assertion to look for the member name 'Dave' or the summary
        assert "Dave" in result
        assert "customer_escalation" in result
        assert "Big client unhappy" in result

    def test_format_cross_signals_non_primary_appends_eng_plan(self, dept_planner):
        """Non-primary departments should see Engineering's proposed events."""
        dept_planner.is_primary = False
        eng_plan = _make_dept_plan(dept="Engineering")
        result = dept_planner._format_cross_signals([], eng_plan)
        assert "ENGINEERING" in result.upper()

    def test_format_cross_signals_primary_does_not_append_eng_plan(self, dept_planner):
        """Engineering itself should NOT see its own plan injected."""
        dept_planner.is_primary = True
        eng_plan = _make_dept_plan(dept="Engineering")
        result = dept_planner._format_cross_signals([], eng_plan)
        assert "ENGINEERING'S PLAN TODAY" not in result


# ─────────────────────────────────────────────────────────────────────────────
# 6. DepartmentPlanner._extract_cross_signals (via Orchestrator helper)
# ─────────────────────────────────────────────────────────────────────────────


class TestExtractCrossSignals:
    def _make_orchestrator(self):
        with patch("day_planner.PlanValidator"), patch("day_planner.OrgCoordinator"):
            return DayPlannerOrchestrator(CONFIG, MagicMock(), MagicMock(), MagicMock())

    def test_no_events_returns_empty_signals(self, mock_mem):
        orch = self._make_orchestrator()
        mock_mem.get_event_log.return_value = []
        signals = orch._extract_cross_signals(mock_mem, day=5)
        assert signals == {}

    def test_incident_resolved_creates_signal_for_other_depts(self, mock_mem):
        orch = self._make_orchestrator()
        event = SimEvent(
            type="incident_resolved",
            timestamp="2026-01-05T10:00:00",
            day=5,
            date="2026-01-05",
            actors=["Alice"],
            artifact_ids={},
            facts={},
            summary="Incident resolved",
        )
        mock_mem.get_event_log.return_value = [event]
        signals = orch._extract_cross_signals(mock_mem, day=5)
        # Alice is in Engineering; Sales should receive a signal
        assert "Sales" in signals
        assert signals["Sales"][0].source_dept == "Engineering"

    def test_signal_relevance_direct_within_2_days(self, mock_mem):
        orch = self._make_orchestrator()
        event = SimEvent(
            type="incident_resolved",
            timestamp="2026-01-05T10:00:00",
            day=4,
            date="2026-01-04",
            actors=["Alice"],
            artifact_ids={},
            facts={},
            summary="Incident resolved",
        )
        mock_mem.get_event_log.return_value = [event]
        signals = orch._extract_cross_signals(mock_mem, day=5)
        sales_signals = signals.get("Sales", [])
        assert any(s.relevance == "direct" for s in sales_signals)

    def test_signal_relevance_indirect_beyond_2_days(self, mock_mem):
        orch = self._make_orchestrator()
        event = SimEvent(
            type="incident_resolved",
            timestamp="2026-01-01T10:00:00",
            day=2,
            date="2026-01-02",
            actors=["Alice"],
            artifact_ids={},
            facts={},
            summary="Old incident",
        )
        mock_mem.get_event_log.return_value = [event]
        signals = orch._extract_cross_signals(mock_mem, day=5)
        sales_signals = signals.get("Sales", [])
        assert any(s.relevance == "indirect" for s in sales_signals)

    def test_irrelevant_event_type_ignored(self, mock_mem):
        orch = self._make_orchestrator()
        event = SimEvent(
            type="some_untracked_event",
            timestamp="2026-01-05T10:00:00",
            day=5,
            date="2026-01-05",
            actors=["Alice"],
            artifact_ids={},
            facts={},
            summary="Ignored",
        )
        mock_mem.get_event_log.return_value = [event]
        signals = orch._extract_cross_signals(mock_mem, day=5)
        assert signals == {}

    def test_source_dept_not_included_in_its_own_signals(self, mock_mem):
        orch = self._make_orchestrator()
        event = SimEvent(
            type="incident_resolved",
            timestamp="2026-01-05T10:00:00",
            day=5,
            date="2026-01-05",
            actors=["Alice"],
            artifact_ids={},
            facts={},
            summary="eng incident",
        )
        mock_mem.get_event_log.return_value = [event]
        signals = orch._extract_cross_signals(mock_mem, day=5)
        # Engineering is the source; it should NOT see its own signal
        assert "Engineering" not in signals


# ─────────────────────────────────────────────────────────────────────────────
# 7. DepartmentPlanner._patch_stress_levels (via Orchestrator)
# ─────────────────────────────────────────────────────────────────────────────


class TestPatchStressLevels:
    def _make_orchestrator(self):
        with patch("day_planner.PlanValidator"), patch("day_planner.OrgCoordinator"):
            return DayPlannerOrchestrator(CONFIG, MagicMock(), MagicMock(), MagicMock())

    def test_stress_patched_from_graph_dynamics(self, mock_graph_dynamics):
        orch = self._make_orchestrator()
        mock_graph_dynamics._stress = {"Alice": 77, "Bob": 12}
        plan = _make_dept_plan()
        orch._patch_stress_levels(plan, mock_graph_dynamics)
        alice_plan = next(ep for ep in plan.engineer_plans if ep.name == "Alice")
        bob_plan = next(ep for ep in plan.engineer_plans if ep.name == "Bob")
        assert alice_plan.stress_level == 77
        assert bob_plan.stress_level == 12

    def test_missing_name_defaults_to_30(self, mock_graph_dynamics):
        orch = self._make_orchestrator()
        mock_graph_dynamics._stress = {}  # no entries
        plan = _make_dept_plan()
        orch._patch_stress_levels(plan, mock_graph_dynamics)
        for ep in plan.engineer_plans:
            assert ep.stress_level == 30


# ─────────────────────────────────────────────────────────────────────────────
# 8. OrgCoordinator.coordinate — JSON parsing & collision events
# ─────────────────────────────────────────────────────────────────────────────


class TestOrgCoordinatorCoordinate:
    def _make_coordinator(self):
        return OrgCoordinator(CONFIG, MagicMock())

    def _run_coordinate(self, coordinator, raw_json: str, dept_plans=None):
        state = _make_mock_state()
        if dept_plans is None:
            dept_plans = {
                "Engineering": _make_dept_plan("Engineering"),
                "Sales": _make_dept_plan("Sales", members=["Dave"]),
            }
        with (
            patch("agent_factory.Agent"),
            patch("day_planner.Task"),
            patch("day_planner.Crew") as mock_crew,
        ):
            mock_crew.return_value.kickoff.return_value = raw_json
            return coordinator.coordinate(
                dept_plans, state, day=5, date="2026-01-05", org_theme="Reliability"
            )

    def test_valid_collision_produces_event(self):
        coord = self._make_coordinator()
        raw = json.dumps(
            {
                "collision": {
                    "event_type": "accountability_check",
                    "actors": ["Alice", "Dave"],
                    "rationale": "Dave needs ETA from Alice",
                    "facts_hint": {"tension_level": "high"},
                    "priority": 1,
                    "artifact_hint": "slack",
                }
            }
        )
        org_plan = self._run_coordinate(coord, raw)
        assert len(org_plan.collision_events) == 1
        assert org_plan.collision_events[0].event_type == "accountability_check"

    def test_invalid_json_produces_no_collision_events(self):
        coord = self._make_coordinator()
        org_plan = self._run_coordinate(coord, "{not json}")
        assert org_plan.collision_events == []

    def test_collision_without_actors_skipped(self):
        coord = self._make_coordinator()
        raw = json.dumps(
            {
                "collision": {
                    "event_type": "scope_creep",
                    "actors": [],
                    "rationale": "No one involved",
                    "facts_hint": {},
                    "priority": 1,
                    "artifact_hint": None,
                }
            }
        )
        org_plan = self._run_coordinate(coord, raw)
        assert org_plan.collision_events == []

    def test_dept_plans_preserved_in_org_plan(self):
        coord = self._make_coordinator()
        raw = json.dumps({"collision": None})
        dept_plans = {"Engineering": _make_dept_plan("Engineering")}
        state = _make_mock_state()
        with (
            patch("agent_factory.Agent"),
            patch("day_planner.Task"),
            patch("day_planner.Crew") as mock_crew,
        ):
            mock_crew.return_value.kickoff.return_value = raw
            org_plan = coord.coordinate(dept_plans, state, 5, "2026-01-05", "Theme")
        assert "Engineering" in org_plan.dept_plans

    def test_org_theme_preserved_in_org_plan(self):
        coord = self._make_coordinator()
        raw = json.dumps({})
        dept_plans = {"Engineering": _make_dept_plan("Engineering")}
        state = _make_mock_state()
        with (
            patch("agent_factory.Agent"),
            patch("day_planner.Task"),
            patch("day_planner.Crew") as mock_crew,
        ):
            mock_crew.return_value.kickoff.return_value = raw
            org_plan = coord.coordinate(
                dept_plans, state, 5, "2026-01-05", "My Org Theme"
            )
        assert org_plan.org_theme == "My Org Theme"


# ─────────────────────────────────────────────────────────────────────────────
# 9. DepartmentPlanner.plan — LLM path integration (Crew patched)
# ─────────────────────────────────────────────────────────────────────────────


class TestDepartmentPlannerPlan:
    def test_plan_returns_department_day_plan(
        self, dept_planner, mock_graph_dynamics, mock_mem
    ):
        state = _make_mock_state()
        raw = json.dumps(VALID_PLAN_JSON)
        with (
            patch("agent_factory.Agent"),
            patch("day_planner.Task"),
            patch("day_planner.Crew") as mock_crew,
        ):
            mock_crew.return_value.kickoff.return_value = raw
            plan = dept_planner.plan(
                org_theme="Theme",
                day=5,
                date="2026-01-05",
                state=state,
                mem=mock_mem,
                graph_dynamics=mock_graph_dynamics,
                cross_signals=[],
                sprint_context=_make_sprint_context(),
            )
        assert isinstance(plan, DepartmentDayPlan)
        assert plan.dept == "Engineering"

    def test_plan_falls_back_on_llm_json_error(
        self, dept_planner, mock_graph_dynamics, mock_mem
    ):
        state = _make_mock_state()
        with (
            patch("agent_factory.Agent"),
            patch("day_planner.Task"),
            patch("day_planner.Crew") as mock_crew,
        ):
            mock_crew.return_value.kickoff.return_value = "not json at all"
            plan = dept_planner.plan(
                org_theme="Theme",
                day=5,
                date="2026-01-05",
                state=state,
                mem=mock_mem,
                graph_dynamics=mock_graph_dynamics,
                cross_signals=[],
            )
        assert "Fallback" in plan.planner_reasoning

    def test_morale_label_low_when_morale_below_45(
        self, dept_planner, mock_graph_dynamics, mock_mem
    ):
        """
        Low morale state must pass 'low' label into the prompt — verify by
        checking the plan completes without error in that regime.
        """
        state = _make_mock_state(morale=0.30)
        raw = json.dumps(VALID_PLAN_JSON)
        with (
            patch("agent_factory.Agent"),
            patch("day_planner.Task"),
            patch("day_planner.Crew") as mock_crew,
        ):
            mock_crew.return_value.kickoff.return_value = raw
            plan = dept_planner.plan(
                org_theme="Crisis",
                day=5,
                date="2026-01-05",
                state=state,
                mem=mock_mem,
                graph_dynamics=mock_graph_dynamics,
                cross_signals=[],
            )
        assert plan is not None

    def test_planner_deduplicates_mirrored_collaborations(self):
        # 1. Setup a minimal planner (we don't need real LLMs just to test parsing)
        config = {"simulation": {"company_name": "TestCorp"}}
        members = ["Jax", "Deepa"]

        planner = DepartmentPlanner(
            dept="engineering",
            members=members,
            config=config,
            worker_llm=None,  # Not needed for _parse_plan
            clock=MagicMock(),
        )

        # 2. Create a mock LLM output that exhibits the duplicate/mirrored bug
        # Jax lists Deepa for a 1on1, and Deepa lists Jax for the SAME 1on1.
        mock_llm_json = json.dumps(
            {
                "dept_theme": "Testing Deduplication",
                "engineer_plans": [
                    {
                        "name": "Jax",
                        "agenda": [
                            {
                                "activity_type": "1on1",
                                "description": "Catch up with Deepa",
                                "collaborator": ["Deepa"],
                            },
                            {
                                "activity_type": "ticket_progress",
                                "description": "Solo work on API",
                                "collaborator": [],
                            },
                        ],
                    },
                    {
                        "name": "Deepa",
                        "agenda": [
                            {
                                "activity_type": "1on1",
                                "description": "Weekly sync with Jax",
                                "collaborator": ["Jax"],
                            },
                            {
                                "activity_type": "design_discussion",
                                "description": "Review architecture with Jax",
                                "collaborator": ["Jax"],
                            },
                        ],
                    },
                ],
            }
        )

        # 3. Parse the plan
        plan, _ = planner._parse_plan(
            raw=mock_llm_json,
            org_theme="Global Theme",
            day=1,
            date="2026-03-11",
            cross_signals=[],
        )

        # 4. Extract all collaborative events from the parsed plan
        collaborative_events = []
        for ep in plan.engineer_plans:
            for item in ep.agenda:
                if item.activity_type in (
                    "1on1",
                    "mentoring",
                    "design_discussion",
                    "async_question",
                ):
                    # Normalize participants to a frozenset for easy comparison
                    participants = frozenset([ep.name] + item.collaborator)
                    collaborative_events.append((item.activity_type, participants))

        # 5. Assertions to ensure the fix stays sticky

        # A. The mirrored 1on1 should be collapsed into a single event
        one_on_ones = [e for e in collaborative_events if e[0] == "1on1"]
        assert len(one_on_ones) == 1, "Mirrored 1-on-1s were not deduplicated!"

        # B. Solo work should be completely unaffected
        jax_plan = next(ep for ep in plan.engineer_plans if ep.name == "Jax")
        solo_items = [
            a for a in jax_plan.agenda if a.activity_type == "ticket_progress"
        ]
        assert len(solo_items) == 1, "Solo work was accidentally stripped!"

        # C. Un-mirrored collaborative events should remain intact
        design_discussions = [
            e for e in collaborative_events if e[0] == "design_discussion"
        ]
        assert len(design_discussions) == 1, (
            "Valid single-sided collaborative events were lost!"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 10. DayPlannerOrchestrator construction
# ─────────────────────────────────────────────────────────────────────────────


class TestDayPlannerOrchestratorInit:
    def test_one_dept_planner_per_department(self):
        with (
            patch("day_planner.PlanValidator"),
            patch("day_planner.OrgCoordinator"),
            patch("day_planner.LIVE_ORG_CHART", ORG_CHART),
        ):
            orch = DayPlannerOrchestrator(CONFIG, MagicMock(), MagicMock(), MagicMock())
        assert set(orch._dept_planners.keys()) == {"Engineering", "Sales"}

    def test_engineering_is_primary(self):
        with (
            patch("day_planner.PlanValidator"),
            patch("day_planner.OrgCoordinator"),
            patch("day_planner.LIVE_ORG_CHART", ORG_CHART),
        ):
            orch = DayPlannerOrchestrator(CONFIG, MagicMock(), MagicMock(), MagicMock())
        assert orch._dept_planners["Engineering"].is_primary is True

    def test_sales_is_not_primary(self):
        with (
            patch("day_planner.PlanValidator"),
            patch("day_planner.OrgCoordinator"),
            patch("day_planner.LIVE_ORG_CHART", ORG_CHART),
        ):
            orch = DayPlannerOrchestrator(CONFIG, MagicMock(), MagicMock(), MagicMock())
        assert orch._dept_planners["Sales"].is_primary is False

    def test_ticket_assigner_starts_as_none(self):
        with (
            patch("day_planner.PlanValidator"),
            patch("day_planner.OrgCoordinator"),
            patch("day_planner.LIVE_ORG_CHART", ORG_CHART),
        ):
            orch = DayPlannerOrchestrator(CONFIG, MagicMock(), MagicMock(), MagicMock())
        assert orch._ticket_assigner is None

    def test_external_contacts_included_in_validator(self):
        cfg = dict(CONFIG)
        cfg["external_contacts"] = [{"name": "VendorPat"}]
        with (
            patch("day_planner.PlanValidator") as mock_validator_cls,
            patch("day_planner.OrgCoordinator"),
            patch("day_planner.LIVE_ORG_CHART", ORG_CHART),
        ):
            DayPlannerOrchestrator(cfg, MagicMock(), MagicMock(), MagicMock())
        call_kwargs = mock_validator_cls.call_args
        ext_names = (
            call_kwargs.kwargs.get("external_contact_names") or call_kwargs.args[1]
        )
        assert "VendorPat" in ext_names


# ─────────────────────────────────────────────────────────────────────────────
# 11. DayPlannerOrchestrator.plan — high-level wiring
# ─────────────────────────────────────────────────────────────────────────────


class TestDayPlannerOrchestratorPlan:
    """
    These tests patch out every LLM call and verify that the orchestrator
    correctly sequences: ticket assignment → org theme → dept plans →
    coordinator → validation → return OrgDayPlan.
    """

    def _make_orch_and_mocks(self):
        mock_worker = MagicMock()
        mock_planner = MagicMock()
        with (
            patch("day_planner.PlanValidator") as mock_validator_cls,
            patch("day_planner.OrgCoordinator") as mock_coord_cls,
        ):
            orch = DayPlannerOrchestrator(
                CONFIG, mock_worker, mock_planner, MagicMock()
            )
        return orch, mock_validator_cls, mock_coord_cls

    def _make_clock(self, state):
        clock = MagicMock()
        clock.now.return_value = datetime(2026, 1, 5, 9, 0, 0)
        return clock

    def test_plan_returns_org_day_plan(self, mock_mem, mock_graph_dynamics):
        with (
            patch("day_planner.PlanValidator"),
            patch("day_planner.LIVE_ORG_CHART", ORG_CHART),
            patch("day_planner.OrgCoordinator") as mock_coord_cls,
            patch("day_planner.TicketAssigner") as mock_ta_cls,
            patch.object(
                DayPlannerOrchestrator,
                "_generate_org_theme",
                return_value="Daily theme",
            ),
            patch.object(
                DayPlannerOrchestrator, "_extract_cross_signals", return_value={}
            ),
            patch.object(
                DayPlannerOrchestrator, "_recent_day_summaries", return_value=[]
            ),
        ):
            orch = DayPlannerOrchestrator(CONFIG, MagicMock(), MagicMock(), MagicMock())

            # Set up TicketAssigner stub
            mock_ta = MagicMock()
            mock_ta.build.return_value = _make_sprint_context()
            mock_ta_cls.return_value = mock_ta

            # Set up dept planners to return canned plans
            for dept, planner in orch._dept_planners.items():
                members = CONFIG["org_chart"][dept]
                planner.plan = MagicMock(return_value=_make_dept_plan(dept, members))

            # OrgCoordinator
            eng_plan = _make_dept_plan("Engineering")
            sales_plan = _make_dept_plan("Sales", members=["Dave"])
            org_plan = OrgDayPlan(
                org_theme="Daily theme",
                dept_plans={"Engineering": eng_plan, "Sales": sales_plan},
                collision_events=[],
                coordinator_reasoning="",
                day=5,
                date="2026-01-05",
            )
            mock_coord_cls.return_value.coordinate.return_value = org_plan

            # PlanValidator
            orch._validator.validate_plan.return_value = []
            orch._validator.rejected.return_value = []
            orch._validator.drain_novel_log.return_value = []
            orch._validator.approved.return_value = []

            state = _make_mock_state()
            clock = self._make_clock(state)
            result = orch.plan(
                state=state,
                mem=mock_mem,
                graph_dynamics=mock_graph_dynamics,
                clock=clock,
            )

        assert isinstance(result, OrgDayPlan)

    def test_plan_seeds_ticket_actors_today(self, mock_mem, mock_graph_dynamics):
        """state.ticket_actors_today must be populated from locked sprint assignments."""
        with (
            patch("day_planner.PlanValidator"),
            patch("day_planner.LIVE_ORG_CHART", ORG_CHART),
            patch("day_planner.LEADS", LEADS),
            patch("day_planner.OrgCoordinator") as mock_coord_cls,
            patch("day_planner.TicketAssigner") as mock_ta_cls,
            patch.object(
                DayPlannerOrchestrator, "_generate_org_theme", return_value="Theme"
            ),
            patch.object(
                DayPlannerOrchestrator, "_extract_cross_signals", return_value={}
            ),
            patch.object(
                DayPlannerOrchestrator, "_recent_day_summaries", return_value=[]
            ),
        ):
            orch = DayPlannerOrchestrator(CONFIG, MagicMock(), MagicMock(), MagicMock())

            mock_ta = MagicMock()
            mock_ta.build.return_value = _make_sprint_context(
                owned={"ENG-101": "Alice"}, members=["Alice", "Bob"]
            )
            mock_ta_cls.return_value = mock_ta

            for dept, planner in orch._dept_planners.items():
                members = CONFIG["org_chart"][dept]
                planner.plan = MagicMock(return_value=_make_dept_plan(dept, members))

            org_plan = OrgDayPlan(
                org_theme="Theme",
                dept_plans={
                    "Engineering": _make_dept_plan("Engineering"),
                    "Sales": _make_dept_plan("Sales", members=["Dave"]),
                },
                collision_events=[],
                coordinator_reasoning="",
                day=5,
                date="2026-01-05",
            )
            mock_coord_cls.return_value.coordinate.return_value = org_plan
            orch._validator.validate_plan.return_value = []
            orch._validator.rejected.return_value = []
            orch._validator.drain_novel_log.return_value = []
            orch._validator.approved.return_value = []

            state = _make_mock_state()
            clock = self._make_clock(state)
            orch.plan(
                state=state,
                mem=mock_mem,
                graph_dynamics=mock_graph_dynamics,
                clock=clock,
            )

        assert "ENG-101" in state.ticket_actors_today
        assert "Alice" in state.ticket_actors_today["ENG-101"]

    def test_plan_rejected_events_logged_as_sim_events(
        self, mock_mem, mock_graph_dynamics
    ):
        """Rejected proposed events must be persisted as proposed_event_rejected SimEvents."""
        rejected_event = ProposedEvent(
            event_type="scope_creep",
            actors=["Dave", "Alice"],
            rationale="Out of scope",
            facts_hint={},
            priority=2,
        )

        class FakeRejection:
            event = rejected_event
            rejection_reason = "duplicate"
            was_novel = False

        with (
            patch("day_planner.PlanValidator"),
            patch("day_planner.LIVE_ORG_CHART", ORG_CHART),
            patch("day_planner.OrgCoordinator") as mock_coord_cls,
            patch("day_planner.TicketAssigner") as mock_ta_cls,
            patch.object(
                DayPlannerOrchestrator, "_generate_org_theme", return_value="Theme"
            ),
            patch.object(
                DayPlannerOrchestrator, "_extract_cross_signals", return_value={}
            ),
            patch.object(
                DayPlannerOrchestrator, "_recent_day_summaries", return_value=[]
            ),
        ):
            orch = DayPlannerOrchestrator(CONFIG, MagicMock(), MagicMock(), MagicMock())

            mock_ta = MagicMock()
            mock_ta.build.return_value = _make_sprint_context()
            mock_ta_cls.return_value = mock_ta

            for dept, planner in orch._dept_planners.items():
                members = CONFIG["org_chart"][dept]
                planner.plan = MagicMock(return_value=_make_dept_plan(dept, members))

            org_plan = OrgDayPlan(
                org_theme="Theme",
                dept_plans={
                    "Engineering": _make_dept_plan("Engineering"),
                    "Sales": _make_dept_plan("Sales", members=["Dave"]),
                },
                collision_events=[],
                coordinator_reasoning="",
                day=5,
                date="2026-01-05",
            )
            mock_coord_cls.return_value.coordinate.return_value = org_plan
            orch._validator.validate_plan.return_value = [FakeRejection()]
            orch._validator.rejected.return_value = [FakeRejection()]
            orch._validator.drain_novel_log.return_value = []
            orch._validator.approved.return_value = []

            state = _make_mock_state()
            clock = self._make_clock(state)
            orch.plan(
                state=state,
                mem=mock_mem,
                graph_dynamics=mock_graph_dynamics,
                clock=clock,
            )

        logged_types = [c.args[0].type for c in mock_mem.log_event.call_args_list]
        assert "proposed_event_rejected" in logged_types

    def test_plan_novel_events_logged_as_sim_events(
        self, mock_mem, mock_graph_dynamics
    ):
        """Novel proposed events must be persisted as novel_event_proposed SimEvents."""
        novel_event = ProposedEvent(
            event_type="new_thing",
            actors=["Alice"],
            rationale="A genuinely new workflow",
            facts_hint={},
            priority=2,
            is_novel=True,
            artifact_hint="slack",
        )
        with (
            patch("day_planner.PlanValidator"),
            patch("day_planner.LIVE_ORG_CHART", ORG_CHART),
            patch("day_planner.LEADS", LEADS),
            patch("day_planner.OrgCoordinator") as mock_coord_cls,
            patch("day_planner.TicketAssigner") as mock_ta_cls,
            patch.object(
                DayPlannerOrchestrator, "_generate_org_theme", return_value="Theme"
            ),
            patch.object(
                DayPlannerOrchestrator, "_extract_cross_signals", return_value={}
            ),
            patch.object(
                DayPlannerOrchestrator, "_recent_day_summaries", return_value=[]
            ),
        ):
            orch = DayPlannerOrchestrator(CONFIG, MagicMock(), MagicMock(), MagicMock())

            mock_ta = MagicMock()
            mock_ta.build.return_value = _make_sprint_context()
            mock_ta_cls.return_value = mock_ta

            for dept, planner in orch._dept_planners.items():
                members = CONFIG["org_chart"][dept]
                planner.plan = MagicMock(return_value=_make_dept_plan(dept, members))

            org_plan = OrgDayPlan(
                org_theme="Theme",
                dept_plans={
                    "Engineering": _make_dept_plan("Engineering"),
                    "Sales": _make_dept_plan("Sales", members=["Dave"]),
                },
                collision_events=[],
                coordinator_reasoning="",
                day=5,
                date="2026-01-05",
            )
            mock_coord_cls.return_value.coordinate.return_value = org_plan
            orch._validator.validate_plan.return_value = []
            orch._validator.rejected.return_value = []
            orch._validator.drain_novel_log.return_value = [novel_event]
            orch._validator.approved.return_value = []

            state = _make_mock_state()
            clock = self._make_clock(state)
            orch.plan(
                state=state,
                mem=mock_mem,
                graph_dynamics=mock_graph_dynamics,
                clock=clock,
            )

        logged_types = [c.args[0].type for c in mock_mem.log_event.call_args_list]
        assert "novel_event_proposed" in logged_types

    def test_ceo_is_excluded_from_sprint_contexts(self, mock_mem, mock_graph_dynamics):
        """Departments missing from LEADS (like CEO) must be skipped during ticket assignment."""
        # 1. Mock the charts to include a CEO who is NOT in LEADS
        mock_org_chart = {"Engineering": ["Alice"], "CEO": ["John"]}
        mock_leads = {"Engineering": "Alice"}

        with (
            patch("day_planner.LIVE_ORG_CHART", mock_org_chart),
            patch("day_planner.LEADS", mock_leads),
            patch("day_planner.PlanValidator"),
            patch("day_planner.OrgCoordinator") as mock_coord_cls,
            patch("day_planner.TicketAssigner") as mock_ta_cls,
            patch.object(
                DayPlannerOrchestrator, "_generate_org_theme", return_value="Theme"
            ),
            patch.object(
                DayPlannerOrchestrator, "_extract_cross_signals", return_value={}
            ),
            patch.object(
                DayPlannerOrchestrator, "_recent_day_summaries", return_value=[]
            ),
        ):
            orch = DayPlannerOrchestrator(CONFIG, MagicMock(), MagicMock(), MagicMock())

            mock_ta = MagicMock()
            mock_ta.build.return_value = _make_sprint_context()
            mock_ta_cls.return_value = mock_ta

            # Stub the individual department planners
            for dept, planner in orch._dept_planners.items():
                members = CONFIG["org_chart"].get(dept, ["Alice"])
                planner.plan = MagicMock(return_value=_make_dept_plan(dept, members))

            # Stub the org coordinator
            mock_coord_cls.return_value.coordinate.return_value = OrgDayPlan(
                org_theme="Theme",
                dept_plans={},
                collision_events=[],
                coordinator_reasoning="",
                day=5,
                date="2026-01-05",
            )

            state = _make_mock_state()
            orch.plan(
                state=state,
                mem=mock_mem,
                graph_dynamics=mock_graph_dynamics,
                clock=self._make_clock(state),
            )

            # 2. Extract the department names TicketAssigner was asked to build
            called_depts = [
                call.kwargs.get("dept_name") or call.args[2]
                for call in mock_ta.build.call_args_list
            ]

            # 3. Assert Engineering got tickets, but CEO was completely skipped
            assert "Engineering" in called_depts
            assert "CEO" not in called_depts
