import pytest
from unittest.mock import MagicMock, patch
from flow import Flow, ActiveIncident
from memory import SimEvent
from org_lifecycle import OrgLifecycleManager, patch_validator_for_lifecycle


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_flow():
    """Flow instance with mocked LLMs and DB, extended with lifecycle manager."""
    with patch("flow.build_llm"), patch("flow.Memory"):
        flow = Flow()
        flow.state.day = 5
        flow.state.system_health = 80
        return flow


@pytest.fixture
def lifecycle(mock_flow):
    """
    Standalone OrgLifecycleManager wired to the flow's live graph and state.
    Mirrors the setup in Flow.__init__ per the patch guide.
    """
    org_chart = {"Engineering": ["Alice", "Bob", "Carol"]}
    personas  = {
        "Alice": {"style": "direct", "expertise": ["backend"], "tenure": "3y", "stress": 30},
        "Bob":   {"style": "casual", "expertise": ["infra"],   "tenure": "2y", "stress": 25},
        "Carol": {"style": "quiet",  "expertise": ["frontend"],"tenure": "1y", "stress": 20},
    }
    all_names = ["Alice", "Bob", "Carol"]
    leads     = {"Engineering": "Alice"}

    # Build a fresh graph that matches the org_chart above
    import networkx as nx
    from graph_dynamics import GraphDynamics
    G = nx.Graph()
    for name in all_names:
        G.add_node(name, dept="Engineering", is_lead=(name == "Alice"), external=False)
    for i, a in enumerate(all_names):
        for b in all_names[i + 1:]:
            G.add_edge(a, b, weight=5.0)

    config = {
        "org_lifecycle": {
            "centrality_vacuum_stress_multiplier": 40,
            "enable_random_attrition": False,
        },
        "graph_dynamics": {},
        "personas": {n: personas[n] for n in all_names},
        "org_chart": org_chart,
        "leads": leads,
    }
    gd = GraphDynamics(G, config)

    mgr = OrgLifecycleManager(
        config=config,
        graph_dynamics=gd,
        mem=mock_flow._mem,
        org_chart=org_chart,
        personas=personas,
        all_names=all_names,
        leads=leads,
    )
    return mgr, gd, org_chart, all_names, mock_flow.state


# ─────────────────────────────────────────────────────────────────────────────
# 1. JIRA TICKET REASSIGNMENT
# ─────────────────────────────────────────────────────────────────────────────

def test_departure_reassigns_open_tickets(lifecycle):
    """
    Verifies that non-Done JIRA tickets owned by a departing engineer are
    reassigned to the dept lead and reset to 'To Do' when no PR is linked.
    """
    mgr, gd, org_chart, all_names, state = lifecycle

    state.jira_tickets = [
        {"id": "ORG-101", "title": "Fix retry logic", "status": "In Progress",
         "assignee": "Bob", "linked_prs": []},
        {"id": "ORG-102", "title": "Write docs",      "status": "To Do",
         "assignee": "Bob", "linked_prs": []},
        {"id": "ORG-103", "title": "Already done",    "status": "Done",
         "assignee": "Bob", "linked_prs": []},
    ]
    state.active_incidents = []

    dep_cfg = {"name": "Bob", "reason": "voluntary", "role": "Engineer",
               "knowledge_domains": [], "documented_pct": 0.5, "day": 5}
    mgr._scheduled_departures = {5: [dep_cfg]}
    mgr.process_departures(day=5, date_str="2026-01-05", state=state)

    t101 = next(t for t in state.jira_tickets if t["id"] == "ORG-101")
    t102 = next(t for t in state.jira_tickets if t["id"] == "ORG-102")
    t103 = next(t for t in state.jira_tickets if t["id"] == "ORG-103")

    # In Progress with no PR → reset to To Do, reassigned to lead
    assert t101["assignee"] == "Alice"
    assert t101["status"]   == "To Do"

    # To Do → stays To Do, reassigned to lead
    assert t102["assignee"] == "Alice"
    assert t102["status"]   == "To Do"

    # Done → untouched
    assert t103["assignee"] == "Bob"


def test_departure_preserves_in_progress_ticket_with_pr(lifecycle):
    """
    An 'In Progress' ticket that already has a linked PR must keep its status
    so the existing PR review/merge flow can close it naturally.
    """
    mgr, gd, org_chart, all_names, state = lifecycle

    state.jira_tickets = [
        {"id": "ORG-200", "title": "Hot fix", "status": "In Progress",
         "assignee": "Bob", "linked_prs": ["PR-101"]},
    ]
    state.active_incidents = []

    dep_cfg = {"name": "Bob", "reason": "layoff", "role": "Engineer",
               "knowledge_domains": [], "documented_pct": 0.5, "day": 5}
    mgr._scheduled_departures = {5: [dep_cfg]}
    mgr.process_departures(day=5, date_str="2026-01-05", state=state)

    t200 = next(t for t in state.jira_tickets if t["id"] == "ORG-200")
    assert t200["assignee"] == "Alice"
    assert t200["status"]   == "In Progress"   # status preserved


# ─────────────────────────────────────────────────────────────────────────────
# 2. ACTIVE INCIDENT HANDOFF
# ─────────────────────────────────────────────────────────────────────────────

def test_departure_hands_off_active_incident(lifecycle):
    """
    When a departing engineer owns an active incident's JIRA ticket, ownership
    must transfer to another person before the node is removed.
    """
    mgr, gd, org_chart, all_names, state = lifecycle

    state.jira_tickets = [
        {"id": "ORG-300", "title": "DB outage", "status": "In Progress",
         "assignee": "Bob", "linked_prs": []},
    ]
    state.active_incidents = [
        ActiveIncident(ticket_id="ORG-300", title="DB outage",
                       day_started=4, stage="investigating", root_cause="OOM"),
    ]

    dep_cfg = {"name": "Bob", "reason": "voluntary", "role": "Engineer",
               "knowledge_domains": [], "documented_pct": 0.5, "day": 5}
    mgr._scheduled_departures = {5: [dep_cfg]}
    mgr.process_departures(day=5, date_str="2026-01-05", state=state)

    t300 = next(t for t in state.jira_tickets if t["id"] == "ORG-300")
    # Bob is gone — ticket must now belong to someone still in the graph
    assert t300["assignee"] != "Bob"
    assert t300["assignee"] in all_names or t300["assignee"] == "Alice"


def test_handoff_emits_escalation_chain_simevent(lifecycle):
    """
    The forced handoff must emit an escalation_chain SimEvent with
    trigger='forced_handoff_on_departure' so the ground-truth log is accurate.
    """
    mgr, gd, org_chart, all_names, state = lifecycle

    state.jira_tickets = [
        {"id": "ORG-301", "title": "API down", "status": "In Progress",
         "assignee": "Carol", "linked_prs": []},
    ]
    state.active_incidents = [
        ActiveIncident(ticket_id="ORG-301", title="API down",
                       day_started=4, stage="detected", root_cause="timeout"),
    ]

    dep_cfg = {"name": "Carol", "reason": "voluntary", "role": "Engineer",
               "knowledge_domains": [], "documented_pct": 0.8, "day": 5}
    mgr._scheduled_departures = {5: [dep_cfg]}
    mgr.process_departures(day=5, date_str="2026-01-05", state=state)

    logged_types = [call.args[0].type for call in mgr._mem.log_event.call_args_list]
    assert "escalation_chain" in logged_types

    escalation_event = next(
        call.args[0] for call in mgr._mem.log_event.call_args_list
        if call.args[0].type == "escalation_chain"
    )
    assert escalation_event.facts["trigger"] == "forced_handoff_on_departure"
    assert escalation_event.facts["departed"] == "Carol"


# ─────────────────────────────────────────────────────────────────────────────
# 3. CENTRALITY VACUUM
# ─────────────────────────────────────────────────────────────────────────────

def test_centrality_vacuum_stresses_neighbours(lifecycle):
    """
    Removing a bridge shortcut node should increase stress on remaining nodes
    that absorb its rerouted traffic.

    Topology: outer ring Alice-Carol-Dave-Eve-Alice, with Bob as an internal
    shortcut between Alice and Dave. Removing Bob keeps the ring intact but
    forces Alice-Dave traffic through Carol and Eve, increasing their
    betweenness and triggering vacuum stress.
    """
    import networkx as nx
    from graph_dynamics import GraphDynamics

    mgr, gd, org_chart, all_names, state = lifecycle
    state.jira_tickets     = []
    state.active_incidents = []

    ring_nodes = ["Alice", "Bob", "Carol", "Dave", "Eve"]
    G = nx.Graph()
    for name in ring_nodes:
        G.add_node(name, dept="Engineering", is_lead=(name == "Alice"), external=False)

    # Outer ring — stays connected after Bob is removed
    G.add_edge("Alice", "Carol", weight=5.0)
    G.add_edge("Carol", "Dave",  weight=5.0)
    G.add_edge("Dave",  "Eve",   weight=5.0)
    G.add_edge("Eve",   "Alice", weight=5.0)
    # Bob is an internal shortcut — high centrality, but not the only path
    G.add_edge("Alice", "Bob",   weight=5.0)
    G.add_edge("Bob",   "Dave",  weight=5.0)

    config = {
        "org_lifecycle": {"centrality_vacuum_stress_multiplier": 40},
        "graph_dynamics": {},
        "personas": {n: {"stress": 25} for n in ring_nodes},
        "org_chart": {"Engineering": ring_nodes},
        "leads": {"Engineering": "Alice"},
    }
    gd_ring = GraphDynamics(G, config)
    for name in ring_nodes:
        gd_ring._stress[name] = 25

    mgr._gd        = gd_ring
    mgr._org_chart = {"Engineering": list(ring_nodes)}
    mgr._all_names = list(ring_nodes)
    mgr._leads     = {"Engineering": "Alice"}

    remaining = ["Alice", "Carol", "Dave", "Eve"]
    stress_before = {n: gd_ring._stress.get(n, 25) for n in remaining}

    dep_cfg = {"name": "Bob", "reason": "voluntary", "role": "Engineer",
               "knowledge_domains": [], "documented_pct": 0.5, "day": 5}
    mgr._scheduled_departures = {5: [dep_cfg]}
    mgr.process_departures(day=5, date_str="2026-01-05", state=state)

    stress_increased = any(
        gd_ring._stress.get(n, 0) > stress_before[n]
        for n in remaining
        if gd_ring.G.has_node(n)
    )
    assert stress_increased


def test_centrality_vacuum_stress_capped_at_20(lifecycle):
    """
    The per-departure stress cap of 20 points must never be exceeded regardless
    of how extreme the centrality shift is.
    """
    mgr, gd, org_chart, all_names, state = lifecycle
    state.jira_tickets     = []
    state.active_incidents = []

    # Force a very high multiplier to stress-test the cap
    mgr._cfg["centrality_vacuum_stress_multiplier"] = 10_000

    stress_before = {n: gd._stress.get(n, 30) for n in all_names}

    dep_cfg = {"name": "Bob", "reason": "voluntary", "role": "Engineer",
               "knowledge_domains": [], "documented_pct": 0.5, "day": 5}
    mgr._scheduled_departures = {5: [dep_cfg]}
    mgr.process_departures(day=5, date_str="2026-01-05", state=state)

    for name in ["Alice", "Carol"]:
        if not gd.G.has_node(name):
            continue
        delta = gd._stress.get(name, 0) - stress_before.get(name, 30)
        assert delta <= 20, f"{name} stress delta {delta} exceeded cap of 20"


# ─────────────────────────────────────────────────────────────────────────────
# 4. NODE REMOVAL & GRAPH INTEGRITY
# ─────────────────────────────────────────────────────────────────────────────

def test_departed_node_removed_from_graph(lifecycle):
    """The departing engineer's node must not exist in the graph after departure."""
    mgr, gd, org_chart, all_names, state = lifecycle
    state.jira_tickets     = []
    state.active_incidents = []

    assert gd.G.has_node("Bob")

    dep_cfg = {"name": "Bob", "reason": "voluntary", "role": "Engineer",
               "knowledge_domains": [], "documented_pct": 0.5, "day": 5}
    mgr._scheduled_departures = {5: [dep_cfg]}
    mgr.process_departures(day=5, date_str="2026-01-05", state=state)

    assert not gd.G.has_node("Bob")
    assert "Bob" not in all_names
    assert "Bob" not in org_chart.get("Engineering", [])


def test_departed_node_stress_entry_removed(lifecycle):
    """The departing engineer's stress entry must be cleaned up from GraphDynamics."""
    mgr, gd, org_chart, all_names, state = lifecycle
    state.jira_tickets     = []
    state.active_incidents = []

    gd._stress["Bob"] = 55

    dep_cfg = {"name": "Bob", "reason": "voluntary", "role": "Engineer",
               "knowledge_domains": [], "documented_pct": 0.5, "day": 5}
    mgr._scheduled_departures = {5: [dep_cfg]}
    mgr.process_departures(day=5, date_str="2026-01-05", state=state)

    assert "Bob" not in gd._stress


def test_departure_emits_employee_departed_simevent(lifecycle):
    """A departure must emit exactly one employee_departed SimEvent."""
    mgr, gd, org_chart, all_names, state = lifecycle
    state.jira_tickets     = []
    state.active_incidents = []

    dep_cfg = {"name": "Bob", "reason": "layoff", "role": "Engineer",
               "knowledge_domains": ["auth-service"], "documented_pct": 0.2, "day": 5}
    mgr._scheduled_departures = {5: [dep_cfg]}
    mgr.process_departures(day=5, date_str="2026-01-05", state=state)

    departed_events = [
        call.args[0] for call in mgr._mem.log_event.call_args_list
        if call.args[0].type == "employee_departed"
    ]
    assert len(departed_events) == 1
    evt = departed_events[0]
    assert evt.facts["name"]   == "Bob"
    assert evt.facts["reason"] == "layoff"
    assert "auth-service" in evt.facts["knowledge_domains"]


def test_departure_record_stored_on_state(lifecycle):
    """state.departed_employees must be populated after a departure."""
    mgr, gd, org_chart, all_names, state = lifecycle
    state.jira_tickets     = []
    state.active_incidents = []

    dep_cfg = {"name": "Bob", "reason": "voluntary", "role": "Senior Engineer",
               "knowledge_domains": ["redis-cache"], "documented_pct": 0.4, "day": 5}
    mgr._scheduled_departures = {5: [dep_cfg]}
    mgr.process_departures(day=5, date_str="2026-01-05", state=state)

    assert "Bob" in state.departed_employees
    assert state.departed_employees["Bob"]["role"] == "Senior Engineer"
    assert "redis-cache" in state.departed_employees["Bob"]["knew_about"]


# ─────────────────────────────────────────────────────────────────────────────
# 5. KNOWLEDGE GAP SCANNING
# ─────────────────────────────────────────────────────────────────────────────

def test_knowledge_gap_scan_detects_domain_hit(lifecycle):
    """
    scan_for_knowledge_gaps must emit a knowledge_gap_detected SimEvent when
    the incident root cause mentions a departed employee's known domain.
    """
    mgr, gd, org_chart, all_names, state = lifecycle
    state.jira_tickets     = []
    state.active_incidents = []

    # First, register a departure with known domains
    dep_cfg = {"name": "Bob", "reason": "voluntary", "role": "Engineer",
               "knowledge_domains": ["auth-service", "redis-cache"],
               "documented_pct": 0.25, "day": 3}
    mgr._scheduled_departures = {3: [dep_cfg]}
    mgr.process_departures(day=3, date_str="2026-01-03", state=state)
    mgr._mem.log_event.reset_mock()

    # Now trigger a scan with text that mentions the domain
    gaps = mgr.scan_for_knowledge_gaps(
        text="Root cause: auth-service JWT validation failing after config change.",
        triggered_by="ORG-400",
        day=5,
        date_str="2026-01-05",
        state=state,
    )

    assert len(gaps) == 1
    assert gaps[0].domain_hit       == "auth-service"
    assert gaps[0].departed_name    == "Bob"
    assert gaps[0].triggered_by     == "ORG-400"
    assert gaps[0].documented_pct   == 0.25

    gap_events = [
        call.args[0] for call in mgr._mem.log_event.call_args_list
        if call.args[0].type == "knowledge_gap_detected"
    ]
    assert len(gap_events) == 1


def test_knowledge_gap_scan_deduplicates(lifecycle):
    """
    The same domain must only surface once per simulation run regardless of
    how many times the text is scanned.
    """
    mgr, gd, org_chart, all_names, state = lifecycle
    state.jira_tickets     = []
    state.active_incidents = []

    dep_cfg = {"name": "Bob", "reason": "voluntary", "role": "Engineer",
               "knowledge_domains": ["redis-cache"], "documented_pct": 0.5, "day": 3}
    mgr._scheduled_departures = {3: [dep_cfg]}
    mgr.process_departures(day=3, date_str="2026-01-03", state=state)
    mgr._mem.log_event.reset_mock()

    text = "redis-cache connection pool exhausted"
    mgr.scan_for_knowledge_gaps(text=text, triggered_by="ORG-401",
                                day=5, date_str="2026-01-05", state=state)
    mgr.scan_for_knowledge_gaps(text=text, triggered_by="ORG-402",
                                day=6, date_str="2026-01-06", state=state)

    gap_events = [
        call.args[0] for call in mgr._mem.log_event.call_args_list
        if call.args[0].type == "knowledge_gap_detected"
    ]
    assert len(gap_events) == 1   # second scan must be a no-op


def test_knowledge_gap_scan_no_false_positives(lifecycle):
    """Unrelated text must not trigger any knowledge gap events."""
    mgr, gd, org_chart, all_names, state = lifecycle
    state.jira_tickets     = []
    state.active_incidents = []

    dep_cfg = {"name": "Bob", "reason": "voluntary", "role": "Engineer",
               "knowledge_domains": ["auth-service"], "documented_pct": 0.5, "day": 3}
    mgr._scheduled_departures = {3: [dep_cfg]}
    mgr.process_departures(day=3, date_str="2026-01-03", state=state)
    mgr._mem.log_event.reset_mock()

    gaps = mgr.scan_for_knowledge_gaps(
        text="Disk I/O throughput degraded on worker-node-3.",
        triggered_by="ORG-403",
        day=5, date_str="2026-01-05", state=state,
    )
    assert len(gaps) == 0


# ─────────────────────────────────────────────────────────────────────────────
# 6. NEW HIRE COLD START
# ─────────────────────────────────────────────────────────────────────────────

def test_new_hire_added_to_graph(lifecycle):
    """A hired engineer must appear in the graph and org_chart after process_hires."""
    mgr, gd, org_chart, all_names, state = lifecycle

    hire_cfg = {"name": "Taylor", "dept": "Engineering", "role": "Backend Engineer",
                "expertise": ["Python", "Kafka"], "style": "methodical", "tenure": "new",
                "day": 5}
    mgr._scheduled_hires = {5: [hire_cfg]}
    mgr.process_hires(day=5, date_str="2026-01-05", state=state)

    assert gd.G.has_node("Taylor")
    assert "Taylor" in org_chart["Engineering"]
    assert "Taylor" in all_names


def test_new_hire_cold_start_edges(lifecycle):
    """
    All edges for a new hire must start at or below floor × 2, ensuring they
    sit below warmup_threshold (2.0) so the planner proposes onboarding events.
    """
    mgr, gd, org_chart, all_names, state = lifecycle
    floor = gd.cfg["edge_weight_floor"]

    hire_cfg = {"name": "Taylor", "dept": "Engineering", "role": "Backend Engineer",
                "expertise": ["Python"], "style": "methodical", "tenure": "new",
                "day": 5}
    mgr._scheduled_hires = {5: [hire_cfg]}
    mgr.process_hires(day=5, date_str="2026-01-05", state=state)

    for nb in gd.G.neighbors("Taylor"):
        weight = gd.G["Taylor"][nb]["weight"]
        assert weight <= floor * 2.0, (
            f"Taylor→{nb} edge weight {weight} exceeds cold-start ceiling {floor * 2.0}"
        )


def test_new_hire_warm_up_edge(lifecycle):
    """warm_up_edge must increase the edge weight between the hire and a colleague."""
    mgr, gd, org_chart, all_names, state = lifecycle

    hire_cfg = {"name": "Taylor", "dept": "Engineering", "role": "Backend Engineer",
                "expertise": ["Python"], "style": "methodical", "tenure": "new",
                "day": 5}
    mgr._scheduled_hires = {5: [hire_cfg]}
    mgr.process_hires(day=5, date_str="2026-01-05", state=state)

    weight_before = gd.G["Taylor"]["Alice"]["weight"]
    mgr.warm_up_edge("Taylor", "Alice", boost=1.5)
    weight_after  = gd.G["Taylor"]["Alice"]["weight"]

    assert weight_after == round(weight_before + 1.5, 4)


def test_new_hire_emits_simevent(lifecycle):
    """process_hires must emit an employee_hired SimEvent with correct facts."""
    mgr, gd, org_chart, all_names, state = lifecycle

    hire_cfg = {"name": "Taylor", "dept": "Engineering", "role": "Backend Engineer",
                "expertise": ["Python", "Kafka"], "style": "methodical", "tenure": "new",
                "day": 5}
    mgr._scheduled_hires = {5: [hire_cfg]}
    mgr.process_hires(day=5, date_str="2026-01-05", state=state)

    hired_events = [
        call.args[0] for call in mgr._mem.log_event.call_args_list
        if call.args[0].type == "employee_hired"
    ]
    assert len(hired_events) == 1
    evt = hired_events[0]
    assert evt.facts["name"]       == "Taylor"
    assert evt.facts["cold_start"] is True
    assert "Kafka" in evt.facts["expertise"]


def test_new_hire_stress_initialised_low(lifecycle):
    """New hires must start with a low stress score, not inherit the org average."""
    mgr, gd, org_chart, all_names, state = lifecycle

    hire_cfg = {"name": "Taylor", "dept": "Engineering", "role": "Backend Engineer",
                "expertise": ["Python"], "style": "methodical", "tenure": "new",
                "day": 5}
    mgr._scheduled_hires = {5: [hire_cfg]}
    mgr.process_hires(day=5, date_str="2026-01-05", state=state)

    assert gd._stress.get("Taylor", 0) == 20


def test_new_hire_record_stored_on_state(lifecycle):
    """state.new_hires must be populated with the hire's metadata."""
    mgr, gd, org_chart, all_names, state = lifecycle

    hire_cfg = {"name": "Taylor", "dept": "Engineering", "role": "Backend Engineer",
                "expertise": ["Python"], "style": "methodical", "tenure": "new",
                "day": 5}
    mgr._scheduled_hires = {5: [hire_cfg]}
    mgr.process_hires(day=5, date_str="2026-01-05", state=state)

    assert hasattr(state, "new_hires")
    assert "Taylor" in state.new_hires
    assert state.new_hires["Taylor"]["role"] == "Backend Engineer"


# ─────────────────────────────────────────────────────────────────────────────
# 7. VALIDATOR PATCH
# ─────────────────────────────────────────────────────────────────────────────

def test_patch_validator_removes_departed_actor(lifecycle):
    """
    After a departure, patch_validator_for_lifecycle must remove the departed
    name from PlanValidator._valid_actors so the actor integrity check holds.
    """
    from plan_validator import PlanValidator

    mgr, gd, org_chart, all_names, state = lifecycle
    state.jira_tickets     = []
    state.active_incidents = []

    validator = PlanValidator(
        all_names=list(all_names),
        external_contact_names=[],
        config={},
    )
    assert "Bob" in validator._valid_actors

    dep_cfg = {"name": "Bob", "reason": "voluntary", "role": "Engineer",
               "knowledge_domains": [], "documented_pct": 0.5, "day": 5}
    mgr._scheduled_departures = {5: [dep_cfg]}
    mgr.process_departures(day=5, date_str="2026-01-05", state=state)

    patch_validator_for_lifecycle(validator, mgr)

    assert "Bob" not in validator._valid_actors


def test_patch_validator_adds_new_hire(lifecycle):
    """
    After a hire, patch_validator_for_lifecycle must add the new name to
    PlanValidator._valid_actors so the planner can propose events with them.
    """
    from plan_validator import PlanValidator

    mgr, gd, org_chart, all_names, state = lifecycle

    validator = PlanValidator(
        all_names=list(all_names),
        external_contact_names=[],
        config={},
    )
    assert "Taylor" not in validator._valid_actors

    hire_cfg = {"name": "Taylor", "dept": "Engineering", "role": "Backend Engineer",
                "expertise": ["Python"], "style": "methodical", "tenure": "new",
                "day": 5}
    mgr._scheduled_hires = {5: [hire_cfg]}
    mgr.process_hires(day=5, date_str="2026-01-05", state=state)

    patch_validator_for_lifecycle(validator, mgr)

    assert "Taylor" in validator._valid_actors


# ─────────────────────────────────────────────────────────────────────────────
# 8. ROSTER CONTEXT
# ─────────────────────────────────────────────────────────────────────────────

def test_get_roster_context_reflects_departure_and_hire(lifecycle):
    """
    get_roster_context must surface both a recent departure and a recent hire
    so DepartmentPlanner prompts reflect actual roster state.
    """
    mgr, gd, org_chart, all_names, state = lifecycle
    state.jira_tickets     = []
    state.active_incidents = []

    dep_cfg = {"name": "Bob", "reason": "voluntary", "role": "Engineer",
               "knowledge_domains": ["redis-cache"], "documented_pct": 0.3, "day": 4}
    mgr._scheduled_departures = {4: [dep_cfg]}
    mgr.process_departures(day=4, date_str="2026-01-04", state=state)

    hire_cfg = {"name": "Taylor", "dept": "Engineering", "role": "Backend Engineer",
                "expertise": ["Python"], "style": "methodical", "tenure": "new",
                "day": 5}
    mgr._scheduled_hires = {5: [hire_cfg]}
    mgr.process_hires(day=5, date_str="2026-01-05", state=state)

    context = mgr.get_roster_context()

    assert "Bob"     in context
    assert "Taylor"  in context
    assert "redis-cache" in context
