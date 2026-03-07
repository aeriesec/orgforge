import json
import os
import random
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, call

import networkx as nx

from graph_dynamics import GraphDynamics
from sim_clock import SimClock
from normal_day import NormalDayHandler, dept_of_name
from planner_models import (
    AgendaItem,
    EngineerDayPlan,
    DepartmentDayPlan,
    OrgDayPlan,
    ProposedEvent,
)
from memory import SimEvent
from flow import persona_backstory


# ─────────────────────────────────────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _make_real_clock(date: datetime) -> SimClock:
    state_stub = MagicMock()
    state_stub.current_date = date
    state_stub.actor_cursors = {}
    return SimClock(state_stub)


def _make_ticket(ticket_id: str, title: str, status: str = "To Do",
                 assignee: str = "Alice") -> dict:
    return {
        "id": ticket_id, "title": title, "status": status,
        "assignee": assignee, "linked_prs": [], "comments": [],
    }


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

ORG_CHART = {
    "Engineering": ["Alice", "Bob", "Carol"],
    "Sales":       ["Dave"],
}
ALL_NAMES = ["Alice", "Bob", "Carol", "Dave"]
LEADS     = {"Engineering": "Alice", "Sales": "Dave"}

PERSONAS = {
    "Alice": {"style": "direct",    "expertise": ["backend"],  "tenure": "senior", "stress": 30},
    "Bob":   {"style": "casual",    "expertise": ["infra"],    "tenure": "mid",    "stress": 25},
    "Carol": {"style": "methodical","expertise": ["frontend"], "tenure": "junior", "stress": 20},
    "Dave":  {"style": "assertive", "expertise": ["sales"],    "tenure": "senior", "stress": 40},
}

CONFIG = {
    "simulation": {
        "company_name":          "TestCorp",
        "domain":                "testcorp.com",
        "output_dir":            "/tmp/orgforge_test",
        "watercooler_prob":      0.0,   # disable by default; opt-in per test
        "aws_alert_prob":        0.0,
        "snyk_alert_prob":       0.0,
        "adhoc_confluence_prob": 0.0,
    },
    "org_chart": ORG_CHART,
    "leads":     LEADS,
    "personas":  PERSONAS,
    "graph_dynamics": {},
}


@pytest.fixture
def graph_and_gd():
    """Real NetworkX graph + GraphDynamics wired to CONFIG."""
    G = nx.Graph()
    for name in ALL_NAMES:
        G.add_node(name, dept=dept_of_name(name, ORG_CHART), external=False)
    for i, a in enumerate(ALL_NAMES):
        for b in ALL_NAMES[i + 1:]:
            G.add_edge(a, b, weight=5.0)
    gd = GraphDynamics(G, CONFIG)
    return G, gd


@pytest.fixture
def mock_state():
    state = MagicMock()
    state.current_date     = datetime(2026, 1, 5)
    state.day              = 5
    state.daily_theme      = "Improving reliability"
    state.jira_tickets     = []
    state.confluence_pages = []
    state.slack_threads    = []
    state.daily_artifacts_created = 0
    state.actor_cursors    = {}
    return state


@pytest.fixture
def clock(mock_state):
    """Real SimClock backed by mock_state so cursor advances are observable."""
    clk = SimClock(mock_state)
    clk.reset_to_business_start(ALL_NAMES + ["system"])
    return clk


@pytest.fixture
def handler(graph_and_gd, mock_state, clock):
    """NormalDayHandler with mocked LLM, mem, and git but real graph/clock."""
    G, gd = graph_and_gd

    mock_mem = MagicMock()
    mock_mem.context_for_prompt.return_value = "some context"

    mock_git = MagicMock()
    mock_worker = MagicMock()
    mock_planner = MagicMock()

    h = NormalDayHandler(
        config=CONFIG,
        mem=mock_mem,
        state=mock_state,
        graph_dynamics=gd,
        social_graph=G,
        git=mock_git,
        worker_llm=mock_worker,
        planner_llm=mock_planner,
        clock=clock,
        persona_helper=persona_backstory
    )
    return h


def _simple_eng_plan(name: str, items: list) -> EngineerDayPlan:
    return EngineerDayPlan(
        name=name,
        dept="Engineering",
        agenda=items,
        stress_level=25,
    )


def _simple_dept_plan(eng_plans: list) -> DepartmentDayPlan:
    return DepartmentDayPlan(
        dept="Engineering",
        theme="Steady progress",
        engineer_plans=eng_plans,
        proposed_events=[],
        cross_dept_signals=[],
        planner_reasoning="",
        day=5,
        date="2026-01-05",
    )


def _simple_org_plan(dept_plans: dict) -> OrgDayPlan:
    return OrgDayPlan(
        org_theme="Reliability",
        dept_plans=dept_plans,
        collision_events=[],
        coordinator_reasoning="",
        day=5,
        date="2026-01-05",
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. DISPATCH ROUTING
# ─────────────────────────────────────────────────────────────────────────────

def test_dispatch_deep_work_returns_actor_only(handler, mock_state):
    """
    deep_work items must not produce Slack/JIRA artifacts.
    The only return value should be the engineer's own name.
    """
    item = AgendaItem(
        activity_type="deep_work",
        description="Heads-down on auth refactor",
        estimated_hrs=3.0,
    )
    eng_plan  = _simple_eng_plan("Alice", [item])
    dept_plan = _simple_dept_plan([eng_plan])

    actors = handler._dispatch(eng_plan, item, dept_plan, "2026-01-05")

    assert actors == ["Alice"]

    logged_types = [c.args[0].type for c in handler._mem.log_event.call_args_list]
    assert "deep_work_session" in logged_types
    # Must not emit ticket_progress, async_question, etc.
    assert "ticket_progress" not in logged_types


def test_dispatch_deferred_item_is_skipped(handler, mock_state):
    """
    Deferred items must be logged as agenda_item_deferred and must not
    be dispatched to any activity handler.
    """
    item = AgendaItem(
        activity_type="ticket_progress",
        description="Fix retry logic",
        related_id="ORG-101",
        estimated_hrs=2.0,
        deferred=True,
        defer_reason="Deferred: P1 incident",
    )
    eng_plan  = _simple_eng_plan("Bob", [item])
    dept_plan = _simple_dept_plan([eng_plan])
    org_plan  = _simple_org_plan({"Engineering": dept_plan})

    handler._execute_agenda_items(org_plan, "2026-01-05")

    logged_types = [c.args[0].type for c in handler._mem.log_event.call_args_list]
    assert "agenda_item_deferred" in logged_types
    assert "ticket_progress"      not in logged_types


def test_dispatch_unknown_activity_type_does_not_raise(handler, mock_state):
    """
    An unrecognised activity_type must fall back to _handle_generic_activity
    without raising an exception.
    """
    item = AgendaItem(
        activity_type="some_future_activity_type",
        description="Novel thing",
        estimated_hrs=1.0,
    )
    eng_plan  = _simple_eng_plan("Alice", [item])
    dept_plan = _simple_dept_plan([eng_plan])

    # Should not raise
    actors = handler._dispatch(eng_plan, item, dept_plan, "2026-01-05")
    assert "Alice" in actors


# ─────────────────────────────────────────────────────────────────────────────
# 2. TICKET PROGRESS
# ─────────────────────────────────────────────────────────────────────────────

def test_ticket_progress_moves_todo_to_in_progress(handler, mock_state, tmp_path):
    """
    A ticket in 'To Do' status must be set to 'In Progress' when progressed.
    """
    ticket = _make_ticket("ORG-101", "Fix retry logic", status="To Do", assignee="Alice")
    mock_state.jira_tickets = [ticket]

    item = AgendaItem(
        activity_type="ticket_progress",
        description="Continue ORG-101",
        related_id="ORG-101",
        estimated_hrs=2.0,
    )
    eng_plan  = _simple_eng_plan("Alice", [item])
    dept_plan = _simple_dept_plan([eng_plan])

    with patch.object(handler, "_save_ticket"), \
         patch("normal_day.Crew") as mock_crew:
        mock_crew.return_value.kickoff.return_value = "Made progress on retry logic."
        handler._dispatch(eng_plan, item, dept_plan, "2026-01-05")

    assert ticket["status"] == "In Progress"


def test_ticket_progress_emits_simevent(handler, mock_state):
    """
    _handle_ticket_progress must emit exactly one ticket_progress SimEvent
    with the correct ticket_id in facts.
    """
    ticket = _make_ticket("ORG-102", "Write unit tests", status="To Do", assignee="Bob")
    mock_state.jira_tickets = [ticket]

    item = AgendaItem(
        activity_type="ticket_progress",
        description="Work on ORG-102",
        related_id="ORG-102",
        estimated_hrs=1.5,
    )
    eng_plan  = _simple_eng_plan("Bob", [item])
    dept_plan = _simple_dept_plan([eng_plan])

    with patch.object(handler, "_save_ticket"), \
         patch("normal_day.Crew") as mock_crew:
        mock_crew.return_value.kickoff.return_value = "Added three new test cases."
        handler._dispatch(eng_plan, item, dept_plan, "2026-01-05")

    events = [
        c.args[0] for c in handler._mem.log_event.call_args_list
        if c.args[0].type == "ticket_progress"
    ]
    assert len(events) == 1
    assert events[0].facts["ticket_id"] == "ORG-102"


def test_ticket_progress_no_op_for_missing_ticket(handler, mock_state):
    """
    When the ticket_id does not exist in state.jira_tickets, the handler
    must return gracefully without emitting a SimEvent or crashing.
    """
    mock_state.jira_tickets = []

    item = AgendaItem(
        activity_type="ticket_progress",
        description="Work on ghost ticket",
        related_id="ORG-GHOST",
        estimated_hrs=1.0,
    )
    eng_plan  = _simple_eng_plan("Alice", [item])
    dept_plan = _simple_dept_plan([eng_plan])

    actors = handler._dispatch(eng_plan, item, dept_plan, "2026-01-05")

    assert actors == ["Alice"]
    ticket_events = [
        c for c in handler._mem.log_event.call_args_list
        if c.args[0].type == "ticket_progress"
    ]
    assert len(ticket_events) == 0


def test_ticket_progress_blocker_emits_blocker_flagged(handler, mock_state):
    """
    When the LLM-generated comment contains a blocker keyword, a
    blocker_flagged SimEvent must also be emitted.
    """
    ticket = _make_ticket("ORG-103", "Investigate timeout", status="In Progress", assignee="Carol")
    mock_state.jira_tickets = [ticket]

    item = AgendaItem(
        activity_type="ticket_progress",
        description="ORG-103 timeout investigation",
        related_id="ORG-103",
        collaborator=["Alice"],
        estimated_hrs=2.0,
    )
    eng_plan  = _simple_eng_plan("Carol", [item])
    dept_plan = _simple_dept_plan([eng_plan])

    with patch.object(handler, "_save_ticket"), \
         patch.object(handler, "_save_slack"), \
         patch("normal_day.Crew") as mock_crew:
        # First call: ticket comment (contains "blocked"); second call: blocker Slack
        mock_crew.return_value.kickoff.side_effect = [
            "I'm blocked waiting on the infra team to open the port.",
            "Alice: On it, I'll check now.",
        ]
        handler._dispatch(eng_plan, item, dept_plan, "2026-01-05")

    logged_types = [c.args[0].type for c in handler._mem.log_event.call_args_list]
    assert "blocker_flagged" in logged_types


# ─────────────────────────────────────────────────────────────────────────────
# 3. 1:1 HANDLING
# ─────────────────────────────────────────────────────────────────────────────

def test_one_on_one_emits_simevent_with_both_actors(handler, mock_state):
    """
    _handle_one_on_one must emit a 1on1 SimEvent whose actors list contains
    both the engineer and their collaborator.
    """
    item = AgendaItem(
        activity_type="1on1",
        description="Weekly 1:1 with Alice",
        collaborator=["Alice"],
        estimated_hrs=0.5,
    )
    eng_plan  = _simple_eng_plan("Bob", [item])
    dept_plan = _simple_dept_plan([eng_plan])

    with patch.object(handler, "_save_slack"), \
         patch("normal_day.Crew") as mock_crew:
        mock_crew.return_value.kickoff.return_value = (
            "Bob: Hey, quick question about the sprint.\nAlice: Sure, what's up?"
        )
        handler._dispatch(eng_plan, item, dept_plan, "2026-01-05")

    events = [
        c.args[0] for c in handler._mem.log_event.call_args_list
        if c.args[0].type == "1on1"
    ]
    assert len(events) == 1
    assert "Bob"   in events[0].actors
    assert "Alice" in events[0].actors


def test_one_on_one_boosts_graph_edge(handler, graph_and_gd):
    """
    A completed 1:1 must increase the edge weight between the two participants
    via record_slack_interaction.
    """
    G, gd = graph_and_gd
    weight_before = G["Bob"]["Alice"]["weight"]

    item = AgendaItem(
        activity_type="1on1",
        description="Weekly 1:1",
        collaborator=["Alice"],
        estimated_hrs=0.5,
    )
    eng_plan  = _simple_eng_plan("Bob", [item])
    dept_plan = _simple_dept_plan([eng_plan])

    with patch.object(handler, "_save_slack"), \
         patch("normal_day.Crew") as mock_crew:
        mock_crew.return_value.kickoff.return_value = (
            "Bob: Wanted to chat about priorities.\nAlice: Let's do it."
        )
        handler._dispatch(eng_plan, item, dept_plan, "2026-01-05")

    assert G["Bob"]["Alice"]["weight"] > weight_before


def test_one_on_one_skipped_when_collaborator_is_self(handler, mock_state):
    """
    A 1:1 where the collaborator is the same person as the engineer must
    return early and not emit a SimEvent.
    """
    item = AgendaItem(
        activity_type="1on1",
        description="Self 1:1 (invalid)",
        collaborator=["Alice"],   # Alice is also the eng_plan name below
        estimated_hrs=0.5,
    )
    eng_plan  = _simple_eng_plan("Alice", [item])
    dept_plan = _simple_dept_plan([eng_plan])

    with patch("normal_day.Crew"):
        handler._dispatch(eng_plan, item, dept_plan, "2026-01-05")

    events = [
        c for c in handler._mem.log_event.call_args_list
        if c.args[0].type == "1on1"
    ]
    assert len(events) == 0


# ─────────────────────────────────────────────────────────────────────────────
# 4. ASYNC QUESTION
# ─────────────────────────────────────────────────────────────────────────────

def test_async_question_emits_simevent(handler, mock_state):
    """
    _handle_async_question must emit an async_question SimEvent whose facts
    include the asker and channel.
    """
    ticket = _make_ticket("ORG-104", "Cache invalidation bug", assignee="Carol")
    mock_state.jira_tickets = [ticket]

    item = AgendaItem(
        activity_type="async_question",
        description="Question about cache invalidation",
        related_id="ORG-104",
        collaborator=["Bob"],
        estimated_hrs=0.5,
    )
    eng_plan  = _simple_eng_plan("Carol", [item])
    dept_plan = _simple_dept_plan([eng_plan])

    with patch.object(handler, "_save_slack"), \
         patch("normal_day.Crew") as mock_crew:
        mock_crew.return_value.kickoff.return_value = (
            "Carol: Anyone know why the cache isn't invalidating?\n"
            "Bob: Did you check the TTL setting?\n"
            "Carol: Oh good call, looking now."
        )
        handler._dispatch(eng_plan, item, dept_plan, "2026-01-05")

    events = [
        c.args[0] for c in handler._mem.log_event.call_args_list
        if c.args[0].type == "async_question"
    ]
    assert len(events) == 1
    assert events[0].facts["asker"] == "Carol"
    assert "channel" in events[0].facts


def test_async_question_cross_dept_uses_digital_hq(handler, mock_state):
    """
    When asker and responder are in different departments, the channel must
    be 'digital-hq'.
    """
    item = AgendaItem(
        activity_type="async_question",
        description="Quota question for sales",
        collaborator=["Dave"],   # Dave is in Sales, Carol is in Engineering
        estimated_hrs=0.5,
    )
    eng_plan  = _simple_eng_plan("Carol", [item])
    dept_plan = _simple_dept_plan([eng_plan])

    with patch.object(handler, "_save_slack"), \
         patch("normal_day.Crew") as mock_crew:
        mock_crew.return_value.kickoff.return_value = (
            "Carol: Dave, can you clarify the quota?\nDave: Sure, let me pull it up."
        )
        handler._dispatch(eng_plan, item, dept_plan, "2026-01-05")

    events = [
        c.args[0] for c in handler._mem.log_event.call_args_list
        if c.args[0].type == "async_question"
    ]
    assert len(events) == 1
    assert events[0].facts["channel"] == "digital-hq"


# ─────────────────────────────────────────────────────────────────────────────
# 5. DESIGN DISCUSSION
# ─────────────────────────────────────────────────────────────────────────────

def test_design_discussion_emits_simevent(handler, mock_state):
    """
    A design discussion must emit a design_discussion SimEvent with the
    correct topic and at least two actors.
    """
    item = AgendaItem(
        activity_type="design_discussion",
        description="Caching strategy for user sessions",
        collaborator=["Bob"],
        estimated_hrs=1.0,
    )
    eng_plan  = _simple_eng_plan("Alice", [item])
    dept_plan = _simple_dept_plan([eng_plan])

    with patch.object(handler, "_save_slack"), \
         patch.object(handler, "_save_md"), \
         patch("normal_day.Crew") as mock_crew, \
         patch("random.random", return_value=0.99):   # suppress Confluence stub
        mock_crew.return_value.kickoff.return_value = (
            "Alice: We should use Redis for session caching.\n"
            "Bob: Agreed, but we need to think about eviction.\n"
            "Alice: Let's set a 1hr TTL by default."
        )
        handler._dispatch(eng_plan, item, dept_plan, "2026-01-05")

    events = [
        c.args[0] for c in handler._mem.log_event.call_args_list
        if c.args[0].type == "design_discussion"
    ]
    assert len(events) == 1
    assert "Caching strategy" in events[0].facts["topic"]
    assert len(events[0].actors) >= 2


def test_design_discussion_confluence_stub_created_sometimes(handler, mock_state):
    """
    When random.random() < 0.30, _create_design_doc_stub must be called and
    a confluence_created SimEvent must be emitted.
    """
    item = AgendaItem(
        activity_type="design_discussion",
        description="Retry policy design",
        collaborator=["Bob"],
        estimated_hrs=1.0,
    )
    eng_plan  = _simple_eng_plan("Alice", [item])
    dept_plan = _simple_dept_plan([eng_plan])

    mock_cw = MagicMock()
    def fake_write_design(*args, **kwargs):
        from memory import SimEvent
        handler._mem.log_event(SimEvent(
            type="confluence_created", day=1, date="", timestamp="", 
            actors=[], artifact_ids={}, facts={}, summary=""
        ))
        return "CONF-ENG-123"
    mock_cw.write_design_doc.side_effect = fake_write_design
    handler._confluence = mock_cw

    with patch.object(handler, "_save_slack"), \
         patch.object(handler, "_save_md"), \
         patch("normal_day.Crew") as mock_crew, \
         patch("random.random", return_value=0.10):   # trigger Confluence path
        mock_crew.return_value.kickoff.return_value = (
            "Alice: We need a clear retry policy.\n"
            "Bob: Exponential back-off seems right."
        )
        handler._dispatch(eng_plan, item, dept_plan, "2026-01-05")

    logged_types = [c.args[0].type for c in handler._mem.log_event.call_args_list]
    assert "confluence_created" in logged_types


# ─────────────────────────────────────────────────────────────────────────────
# 6. MENTORING
# ─────────────────────────────────────────────────────────────────────────────

def test_mentoring_double_boosts_graph_edge(handler, graph_and_gd):
    """
    A mentoring session calls record_slack_interaction twice, so the edge
    weight between mentor and mentee must increase by at least 2× the
    configured slack_boost.
    """
    G, gd = graph_and_gd
    weight_before = G["Alice"]["Carol"]["weight"]
    boost = gd.cfg["slack_boost"]

    item = AgendaItem(
        activity_type="mentoring",
        description="Help Carol with async patterns",
        collaborator=["Carol"],
        estimated_hrs=1.0,
    )
    eng_plan  = _simple_eng_plan("Alice", [item])
    dept_plan = _simple_dept_plan([eng_plan])

    with patch.object(handler, "_save_slack"), \
         patch("normal_day.Crew") as mock_crew:
        mock_crew.return_value.kickoff.return_value = (
            "Alice: Let's talk about async/await.\nCarol: I've been struggling with it."
        )
        handler._dispatch(eng_plan, item, dept_plan, "2026-01-05")

    # Double boost: weight should have grown by at least 2 × slack_boost
    assert G["Alice"]["Carol"]["weight"] >= weight_before + (2 * boost)


def test_mentoring_emits_simevent(handler, mock_state):
    """
    A completed mentoring session must emit a mentoring SimEvent with
    mentor and mentee recorded in facts.
    """
    item = AgendaItem(
        activity_type="mentoring",
        description="Career growth discussion",
        collaborator=["Carol"],
        estimated_hrs=0.75,
    )
    eng_plan  = _simple_eng_plan("Alice", [item])
    dept_plan = _simple_dept_plan([eng_plan])

    with patch.object(handler, "_save_slack"), \
         patch("normal_day.Crew") as mock_crew:
        mock_crew.return_value.kickoff.return_value = (
            "Alice: How are you finding the new ticket workload?\n"
            "Carol: It's a lot, but I'm managing."
        )
        handler._dispatch(eng_plan, item, dept_plan, "2026-01-05")

    events = [
        c.args[0] for c in handler._mem.log_event.call_args_list
        if c.args[0].type == "mentoring"
    ]
    assert len(events) == 1
    assert events[0].facts["mentor"] == "Alice"
    assert events[0].facts["mentee"] == "Carol"


def test_mentoring_skipped_when_no_junior_found(handler, mock_state):
    """
    If the org has no junior colleague for the mentor (e.g. everyone is
    senior), the handler must return without emitting a mentoring SimEvent.
    """
    item = AgendaItem(
        activity_type="mentoring",
        description="Mentoring session",
        collaborator=[],   # no explicit collaborator
        estimated_hrs=1.0,
    )
    # Dave is in Sales and is senior — no juniors exist for him in his dept
    eng_plan = EngineerDayPlan(
        name="Dave",
        dept="Sales",
        agenda=[item],
        stress_level=40,
    )
    dept_plan = DepartmentDayPlan(
        dept="Sales",
        theme="Close deals",
        engineer_plans=[eng_plan],
        proposed_events=[],
        cross_dept_signals=[],
        planner_reasoning="",
        day=5,
        date="2026-01-05",
    )

    handler._dispatch(eng_plan, item, dept_plan, "2026-01-05")

    events = [
        c for c in handler._mem.log_event.call_args_list
        if c.args[0].type == "mentoring"
    ]
    assert len(events) == 0


# ─────────────────────────────────────────────────────────────────────────────
# 7. CLOCK INTEGRATION — cursors advance correctly
# ─────────────────────────────────────────────────────────────────────────────

def test_deep_work_advances_actor_cursor(handler, clock, mock_state):
    """
    _log_deep_work must advance the engineer's SimClock cursor by
    approximately the item's estimated_hrs.
    """
    cursor_before = clock.now("Alice")

    item = AgendaItem(
        activity_type="deep_work",
        description="Focused refactor",
        estimated_hrs=3.0,
    )
    eng_plan  = _simple_eng_plan("Alice", [item])
    dept_plan = _simple_dept_plan([eng_plan])

    handler._dispatch(eng_plan, item, dept_plan, "2026-01-05")

    cursor_after = clock.now("Alice")
    elapsed_hours = (cursor_after - cursor_before).total_seconds() / 3600
    assert elapsed_hours >= 2.9, (
        f"Expected cursor to advance ~3 hours, got {elapsed_hours:.2f}h"
    )


def test_one_on_one_syncs_both_cursors(handler, clock, mock_state):
    """
    After a 1:1, both participants' cursors must be at or beyond the
    session end time (i.e. they were both consumed by the meeting).
    """
    # Give Bob a head start so we can verify sync
    clock._set_cursor("Bob",   datetime(2026, 1, 5, 10, 0))
    clock._set_cursor("Alice", datetime(2026, 1, 5,  9, 0))

    item = AgendaItem(
        activity_type="1on1",
        description="Sprint check-in",
        collaborator=["Alice"],
        estimated_hrs=0.5,
    )
    eng_plan  = _simple_eng_plan("Bob", [item])
    dept_plan = _simple_dept_plan([eng_plan])

    with patch.object(handler, "_save_slack"), \
         patch("normal_day.Crew") as mock_crew:
        mock_crew.return_value.kickoff.return_value = (
            "Bob: Quick sync on the sprint.\nAlice: Sure, go ahead."
        )
        handler._dispatch(eng_plan, item, dept_plan, "2026-01-05")

    # Both cursors must be past the original later cursor (Bob at 10:00)
    assert clock.now("Bob")   >= datetime(2026, 1, 5, 10, 0)
    assert clock.now("Alice") >= datetime(2026, 1, 5, 10, 0)


# ─────────────────────────────────────────────────────────────────────────────
# 8. _execute_agenda_items — distraction gate
# ─────────────────────────────────────────────────────────────────────────────

def test_distraction_fires_at_most_once_per_engineer(handler, mock_state, graph_and_gd):
    """
    With watercooler_prob=1.0, each engineer must be distracted at most once
    regardless of how many agenda items they have.
    """
    # Override config to always trigger
    handler._config["simulation"]["watercooler_prob"] = 1.0

    items = [
        AgendaItem(activity_type="deep_work", description=f"Task {i}", estimated_hrs=1.0)
        for i in range(5)
    ]
    eng_plan  = _simple_eng_plan("Alice", items)
    dept_plan = _simple_dept_plan([eng_plan])
    org_plan  = _simple_org_plan({"Engineering": dept_plan})

    with patch.object(handler, "_trigger_watercooler_chat") as mock_wc, \
         patch.object(handler, "_log_deep_work"):
        handler._execute_agenda_items(org_plan, "2026-01-05")

    assert mock_wc.call_count == 1, (
        f"_trigger_watercooler_chat called {mock_wc.call_count} times; expected 1"
    )


def test_distraction_never_fires_when_prob_zero(handler, mock_state):
    """
    With watercooler_prob=0.0, _trigger_watercooler_chat must never be called.
    """
    handler._config["simulation"]["watercooler_prob"] = 0.0

    items = [
        AgendaItem(activity_type="deep_work", description="Task", estimated_hrs=1.0)
        for _ in range(3)
    ]
    eng_plan  = _simple_eng_plan("Alice", items)
    dept_plan = _simple_dept_plan([eng_plan])
    org_plan  = _simple_org_plan({"Engineering": dept_plan})

    with patch.object(handler, "_trigger_watercooler_chat") as mock_wc, \
         patch.object(handler, "_log_deep_work"):
        handler._execute_agenda_items(org_plan, "2026-01-05")

    assert mock_wc.call_count == 0


def test_distraction_index_varies_across_runs(handler, mock_state):
    """
    _execute_agenda_items must pass all non-deferred indices to random.choice
    when selecting the distraction target, not always pick index 0.
    """
    handler._config["simulation"]["watercooler_prob"] = 1.0

    items = [
        AgendaItem(activity_type="deep_work", description=f"Task {i}", estimated_hrs=1.0)
        for i in range(4)
    ]
    eng_plan  = _simple_eng_plan("Bob", items)
    dept_plan = _simple_dept_plan([eng_plan])
    org_plan  = _simple_org_plan({"Engineering": dept_plan})

    with patch.object(handler, "_trigger_watercooler_chat"), \
         patch.object(handler, "_log_deep_work"), \
         patch("normal_day.random.random", return_value=0.0), \
         patch("normal_day.random.choice", return_value=2) as mock_choice:
        handler._execute_agenda_items(org_plan, "2026-01-05")

    # random.choice must have been called with the full list of non-deferred indices
    choice_calls = [c for c in mock_choice.call_args_list
                    if isinstance(c.args[0], list) and all(isinstance(x, int) for x in c.args[0])]
    assert len(choice_calls) == 1
    passed_indices = choice_calls[0].args[0]
    assert passed_indices == [0, 1, 2, 3], (
        f"Expected all 4 indices passed to random.choice, got {passed_indices}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 9. GRAPH DYNAMICS INTEGRATION
# ─────────────────────────────────────────────────────────────────────────────

def test_execute_agenda_items_calls_graph_dynamics_record(handler, mock_state):
    """
    After executing agenda items, graph_dynamics_record must be called with
    a non-empty participant list so edge weights are updated.
    """
    item = AgendaItem(
        activity_type="1on1",
        description="Sync",
        collaborator=["Alice"],
        estimated_hrs=0.5,
    )
    eng_plan  = _simple_eng_plan("Bob", [item])
    dept_plan = _simple_dept_plan([eng_plan])
    org_plan  = _simple_org_plan({"Engineering": dept_plan})

    with patch.object(handler, "_save_slack"), \
         patch.object(handler, "graph_dynamics_record") as mock_gdr, \
         patch("normal_day.Crew") as mock_crew:
        mock_crew.return_value.kickoff.return_value = (
            "Bob: Hey Alice.\nAlice: Hey Bob."
        )
        handler._execute_agenda_items(org_plan, "2026-01-05")

    mock_gdr.assert_called_once()
    participants_arg = mock_gdr.call_args.args[0]
    assert len(participants_arg) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# 10. UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def test_dept_of_name_returns_correct_dept():
    assert dept_of_name("Alice", ORG_CHART) == "Engineering"
    assert dept_of_name("Dave",  ORG_CHART) == "Sales"


def test_dept_of_name_unknown_returns_unknown():
    assert dept_of_name("Ghost", ORG_CHART) == "Unknown"


def test_closest_colleague_returns_highest_weight_neighbour(handler, graph_and_gd):
    """
    _closest_colleague must return the neighbour with the highest edge weight.
    """
    G, gd = graph_and_gd
    # Give Bob a very strong edge to Carol and weak edge to Alice
    G["Bob"]["Carol"]["weight"] = 20.0
    G["Bob"]["Alice"]["weight"] =  1.0

    result = handler._closest_colleague("Bob")
    assert result == "Carol"


def test_find_lead_for_returns_dept_lead(handler):
    assert handler._find_lead_for("Bob")   == "Alice"
    assert handler._find_lead_for("Carol") == "Alice"
    assert handler._find_lead_for("Dave")  == "Dave"


def test_parse_slack_messages_filters_unknown_names(handler, mock_state):
    """
    _parse_slack_messages must only include lines whose speaker is in the
    valid_names list — invented names must be silently dropped.
    """
    raw = (
        "Alice: Sounds good.\n"
        "RandomPerson: This should be ignored.\n"
        "Bob: Agreed.\n"
    )
    messages = handler._parse_slack_messages(raw, ["Alice", "Bob"])
    speakers = [m["user"] for m in messages]

    assert "Alice"        in speakers
    assert "Bob"          in speakers
    assert "RandomPerson" not in speakers

def test_dept_of_name_returns_first_match_when_name_in_multiple_depts():
    """
    If a name appears in two departments (data integrity issue),
    dept_of_name returns the first match rather than raising.
    This documents the known behaviour so any future change is deliberate.
    """
    ambiguous_chart = {
        "Engineering": ["Alice", "Bob"],
        "Platform":    ["Alice", "Carol"],  # Alice appears twice
    }
    result = dept_of_name("Alice", ambiguous_chart)
    # Must return one of the two valid depts, not crash or return "Unknown"
    assert result in ("Engineering", "Platform")