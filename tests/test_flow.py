import pytest
from unittest.mock import MagicMock, patch
from memory import SimEvent
from flow import (
    GitSimulator,
    OrgForgeSimulation,
    SprintState,
    build_social_graph,
    dept_of,
    email_of,
    render_template,
    score_sentiment,
)

from datetime import datetime


@pytest.fixture
def mock_flow():
    """Fixture to initialize OrgForgeSimulation with mocked LLMs and DB."""
    with patch("flow.build_llm"), patch("flow.Memory"):
        sim = OrgForgeSimulation()
        sim.state.day = 1
        sim.state.system_health = 100
        sim._mem.log_slack_messages = MagicMock(return_value=("", ""))
        return sim


### --- Bug Catching Tests ---


def test_embed_and_count_recursion_fix(mock_flow):
    """
    Verifies that _embed_and_count enqueues to the embed worker
    rather than calling embed_artifact synchronously (which would
    previously cause recursion if wired incorrectly).
    """
    mock_flow._embed_worker.enqueue = MagicMock()

    mock_flow._embed_and_count(
        id="test", type="doc", title="T", content="C", day=1, date="2026-01-01"
    )

    assert mock_flow._embed_worker.enqueue.called
    assert mock_flow.state.daily_artifacts_created == 1


@patch("flow.Crew")
@patch("flow.Task")
@patch("agent_factory.Agent")
def test_incident_logic_variable_scope(
    mock_agent_class, mock_task_class, mock_crew_class, mock_flow
):
    """
    Verifies that _handle_incident uses correctly defined variables
    (involves_gap vs involves_bill).
    """
    # Setup the mock Crew to return a dummy string when kickoff() is called
    mock_crew_instance = MagicMock()
    mock_crew_instance.kickoff.return_value = "Mocked Root Cause"
    mock_crew_class.return_value = mock_crew_instance

    # Mock necessary components for incident creation
    mock_flow.graph_dynamics = MagicMock()
    mock_flow.graph_dynamics.build_escalation_chain.return_value = MagicMock(chain=[])
    mock_flow.graph_dynamics.escalation_narrative.return_value = ""
    mock_flow.graph_dynamics._stress = MagicMock()
    mock_flow.graph_dynamics._stress.get.return_value = 50

    # This test ensures the code actually runs without a NameError
    try:
        mock_flow._handle_incident()
    except NameError as e:
        pytest.fail(f"_handle_incident failed due to variable scope error: {e}")


def test_social_graph_burnout_propagation(mock_flow):
    """Tests if stress correctly propagates through the network."""
    person = list(mock_flow.social_graph.nodes)[0]

    # Dynamically grab the configured threshold and push them over it
    burnout_threshold = mock_flow.graph_dynamics.cfg["burnout_threshold"]
    mock_flow.graph_dynamics._stress[person] = burnout_threshold + 10

    result = mock_flow.graph_dynamics.propagate_stress()

    assert person in result.burnt_out
    assert len(result.stress_snapshot) > 0


def test_memory_context_retrieval(mock_flow):
    """Tests semantic context window construction."""
    from memory import Memory

    mem = Memory()  # Safe: MongoClient, build_embedder, and _init_vector_indexes are all mocked by autouse fixture

    # Wire up the instance so context_for_prompt has something to work with
    mem._embedder.embed = MagicMock(return_value=[0.1] * 1024)
    mem._artifacts.count_documents = MagicMock(return_value=10)
    mem._artifacts.aggregate = MagicMock(return_value=[])
    mem.recall_events = MagicMock(
        return_value=[
            SimEvent(
                type="test",
                day=1,
                date="2026-01-01",
                actors=[],
                artifact_ids={},
                facts={},
                summary="Test Event",
                timestamp="2026-03-05T13:33:51.027Z",
            )
        ]
    )

    context = mem.context_for_prompt("server crash")
    assert "RELEVANT EVENTS" in context


def test_edge_weight_decay(mock_flow):
    """Verifies that social edges decay over time without interaction."""
    u, v, data = next(iter(mock_flow.social_graph.edges(data=True)))

    floor = mock_flow.graph_dynamics.cfg["edge_weight_floor"]
    decay_rate = mock_flow.graph_dynamics.cfg["edge_decay_rate"]

    # Manually boost the weight safely above the floor
    initial_weight = floor + 5.0
    mock_flow.social_graph[u][v]["weight"] = initial_weight

    mock_flow.graph_dynamics.decay_edges()

    new_weight = mock_flow.social_graph[u][v]["weight"]

    # Dynamically calculate the exact expected result
    expected_weight = round(max(floor, initial_weight * decay_rate), 4)

    assert new_weight == expected_weight


def test_escalation_path_logic(mock_flow):
    """Ensures escalation prefers strong relationships over weak ones."""
    responder = "Alice"
    bestie = "Bob"
    lead = "Charlie"

    # Register the nodes
    mock_flow.social_graph.add_node(responder)
    mock_flow.social_graph.add_node(bestie)
    mock_flow.social_graph.add_node(lead)

    # Explicitly make Charlie a lead so the algorithm targets him
    mock_flow.graph_dynamics._leads = {"Engineering": lead}

    mock_flow.social_graph.add_edge(responder, bestie, weight=10.0)
    mock_flow.social_graph.add_edge(bestie, lead, weight=10.0)
    mock_flow.social_graph.add_edge(responder, lead, weight=0.1)

    chain = mock_flow.graph_dynamics.build_escalation_chain(responder)

    assert "Bob" in [node for node, role in chain.chain]


def test_temporal_memory_isolation(mock_flow):
    """Ensures context_for_prompt respects the day limit."""
    from memory import Memory

    mem = Memory()  # Safe: mocked by autouse fixture
    mem._embedder.embed = MagicMock(return_value=[0.1] * 1024)
    mem._artifacts.count_documents = MagicMock(return_value=10)
    mem._artifacts.aggregate = MagicMock(return_value=[])
    # recall_events chains .find().sort().limit() — mock the full cursor chain
    mock_cursor = MagicMock()
    mock_cursor.sort.return_value = mock_cursor
    mock_cursor.limit.return_value = iter([])
    mem._events.find = MagicMock(return_value=mock_cursor)

    mem.context_for_prompt("incident", as_of_time="2026-03-05T13:33:51.027Z")

    args, kwargs = mem._artifacts.aggregate.call_args
    pipeline = args[0]

    vector_search_stage = next(s for s in pipeline if "$vectorSearch" in s)
    assert "filter" in vector_search_stage["$vectorSearch"]
    assert "timestamp" in vector_search_stage["$vectorSearch"]["filter"]
    assert vector_search_stage["$vectorSearch"]["filter"]["timestamp"] == {
        "$lte": "2026-03-05T13:33:51.027Z"
    }


def test_graph_interaction_boost(mock_flow):
    """Verifies that Slack interactions boost edge weights between participants."""
    u, v = "Alice", "Bob"
    # Ensure nodes exist and set a baseline weight
    mock_flow.social_graph.add_node(u)
    mock_flow.social_graph.add_node(v)
    mock_flow.social_graph.add_edge(u, v, weight=1.0)

    # Simulate a Slack thread between them
    mock_flow.graph_dynamics.record_slack_interaction([u, v])

    new_weight = mock_flow.social_graph[u][v]["weight"]
    expected_boost = mock_flow.graph_dynamics.cfg["slack_boost"]

    assert new_weight > 1.0
    assert new_weight == 1.0 + expected_boost


def test_git_simulator_reviewer_selection(mock_flow):
    """Ensures GitSimulator picks the closest engineering colleague for PR reviews."""
    mock_flow.social_graph.add_node("Alice", dept="Engineering")
    mock_flow.social_graph.add_node("Bob", dept="Engineering")
    mock_flow.social_graph.add_node("Charlie", dept="Engineering")

    # Alice is tight with Bob, but barely knows Charlie
    mock_flow.social_graph.add_edge("Alice", "Bob", weight=10.0)
    mock_flow.social_graph.add_edge("Alice", "Charlie", weight=1.0)

    pr = mock_flow._git.create_pr(
        author="Alice",
        ticket_id="TKT-1",
        title="Fixing stuff",
        timestamp="2026-03-05T13:33:51.027Z",
    )

    assert pr["reviewers"][0] == "Bob"


def test_relevant_external_contacts(mock_flow):
    """Verifies external contacts are triggered by correct events and health thresholds."""
    contacts = [
        {
            "name": "AWS",
            "trigger_on": ["incident"],
            "trigger_health_threshold": 80,
        }
    ]

    mock_flow._mem._db["sim_config"].insert_one(
        {"_id": "inbound_email_sources", "sources": contacts}
    )
    mock_flow._mem.get_inbound_email_sources.return_value = contacts
    mock_flow._mem._db.__getitem__.return_value.find_one.return_value = {
        "sources": contacts
    }

    triggered_aws = mock_flow.graph_dynamics.relevant_external_contacts(
        "incident_opened", 75
    )
    assert len(triggered_aws) == 1
    assert triggered_aws[0]["name"] == "AWS"


def test_end_of_day_resets_and_morale(mock_flow):
    """Verifies EOD correctly decays morale and resets daily metric counters."""
    import flow

    decay = flow.CONFIG["morale"]["daily_decay"]
    recovery = flow.CONFIG["morale"]["good_day_recovery"]

    start_morale = 0.8
    mock_flow.state.daily_incidents_opened = 3
    mock_flow.state.daily_artifacts_created = 5
    mock_flow.state.team_morale = start_morale
    mock_flow.state.active_incidents = []

    mock_flow._end_of_day()

    assert mock_flow.state.daily_incidents_opened == 0
    assert mock_flow.state.daily_artifacts_created == 0
    assert len(mock_flow.state.morale_history) == 1

    expected_morale = round(min(1.0, (start_morale * decay) + recovery), 3)

    assert mock_flow.state.team_morale == expected_morale


def test_memory_log_event():
    """Verifies SimEvents are added to the in-memory log and formatted for MongoDB."""
    from memory import Memory
    from unittest.mock import MagicMock

    mem = Memory()  # Safe: mocked by autouse fixture
    mem._embedder.embed = MagicMock(return_value=[0.1] * 1024)
    mem._events.update_one = MagicMock()

    event = SimEvent(
        type="test_event",
        day=1,
        date="2026-01-01",
        actors=["Alice"],
        artifact_ids={},
        facts={"foo": "bar"},
        summary="Test",
        timestamp="2026-03-05T13:33:51.027Z",
    )

    mem.log_event(event)

    # Verify in-memory state
    assert len(mem._event_log) == 1
    assert mem._event_log[0].type == "test_event"

    # Verify DB call
    assert mem._events.update_one.called
    args, kwargs = mem._events.update_one.call_args

    # The ID generator logic is EVT-{day}-{type}-{index}
    assert args[0] == {"_id": "EVT-1-test_event-1"}


def test_end_of_day_summary_time_uses_latest_actor_cursor(mock_flow):
    """
    When any actor worked past 17:30, _end_of_day must use that actor's cursor
    as the housekeeping timestamp — not the 17:30 EOD baseline.

    The end_of_day SimEvent is the second log_event call in _end_of_day.
    Its timestamp must be strictly after 17:30.
    """
    import flow as flow_module

    # Push one actor's cursor well past EOD
    late_cursor = mock_flow.state.current_date.replace(hour=18, minute=45, second=0)
    for name in flow_module.ALL_NAMES:
        mock_flow._clock._set_cursor(name, late_cursor)

    mock_flow.state.active_incidents = []
    mock_flow._end_of_day()

    # The day_summary event is emitted after end_of_day — grab its timestamp
    all_calls = mock_flow._mem.log_event.call_args_list
    day_summary_evt = next(
        c.args[0] for c in all_calls if c.args[0].type == "day_summary"
    )

    stamped = datetime.fromisoformat(day_summary_evt.timestamp)
    eod_baseline = mock_flow.state.current_date.replace(hour=17, minute=30, second=0)
    assert stamped > eod_baseline, f"Expected timestamp after 17:30, got {stamped}"


def test_end_of_day_summary_time_uses_eod_baseline_when_all_actors_idle(mock_flow):
    """
    When every actor's cursor is before 17:30 (a quiet day), _end_of_day must
    floor the housekeeping timestamp to the 17:30 baseline, not an earlier time.
    """
    import flow as flow_module

    # Put every actor at 10:00 — well before EOD
    early_cursor = mock_flow.state.current_date.replace(hour=10, minute=0, second=0)
    for name in flow_module.ALL_NAMES:
        mock_flow._clock._set_cursor(name, early_cursor)

    mock_flow.state.active_incidents = []
    mock_flow._end_of_day()

    all_calls = mock_flow._mem.log_event.call_args_list
    day_summary_evt = next(
        c.args[0] for c in all_calls if c.args[0].type == "day_summary"
    )

    stamped = datetime.fromisoformat(day_summary_evt.timestamp)
    eod_baseline = mock_flow.state.current_date.replace(hour=17, minute=30, second=0)
    # Must be at or after the baseline, never before
    assert stamped >= eod_baseline, (
        f"Expected timestamp >= 17:30 baseline, got {stamped}"
    )


@patch("flow.Crew")
@patch("flow.Task")
@patch("agent_factory.Agent")
def test_incident_sync_to_system_advances_on_call_cursor(
    mock_agent_class, mock_task_class, mock_crew_class, mock_flow
):
    """
    After _handle_incident fires, the on-call engineer's cursor must sit at or
    after the system clock's incident_start time.

    This verifies that clock.sync_to_system([on_call]) actually ran and had
    an effect — not just that the code didn't crash.
    """
    import flow as flow_module

    mock_crew_instance = MagicMock()
    mock_crew_instance.kickoff.return_value = "Disk I/O timeout on worker node"
    mock_crew_class.return_value = mock_crew_instance

    mock_flow.graph_dynamics = MagicMock()
    mock_flow.graph_dynamics.build_escalation_chain.return_value = MagicMock(chain=[])
    mock_flow.graph_dynamics.escalation_narrative.return_value = ""
    mock_flow.graph_dynamics.relevant_external_contacts.return_value = []
    mock_flow.graph_dynamics._stress = MagicMock()
    mock_flow.graph_dynamics._stress.get.return_value = 40

    on_call = flow_module.resolve_role("on_call_engineer")

    # Start the on-call engineer well before the system clock will land
    mock_flow._clock.reset_to_business_start(flow_module.ALL_NAMES)
    system_before = mock_flow._clock.now("system")

    mock_flow._handle_incident()

    system_after = mock_flow._clock.now("system")
    on_call_after = mock_flow._clock.now(on_call)

    # The system clock must have advanced (tick_system was called)
    assert system_after > system_before, "tick_system did not advance system cursor"

    # The on-call engineer must be at or after the incident start time
    assert on_call_after >= system_after, (
        f"on-call cursor {on_call_after} is before system clock {system_after} "
        "after sync_to_system — sync had no effect"
    )


@patch("confluence_writer.Task")
@patch("confluence_writer.Crew")
@patch("flow.Crew")
@patch("flow.Task")
@patch("agent_factory.Agent")
def test_postmortem_artifact_timestamp_within_actor_work_block(
    mock_agent_class,  # Maps to @patch("agent_factory.Agent")
    mock_task_class,  # Maps to @patch("flow.Task")
    mock_crew_class,  # Maps to @patch("flow.Crew")
    mock_cw_crew,  # Maps to @patch("confluence_writer.Crew")
    mock_cw_task,  # Maps to @patch("confluence_writer.Task")
    mock_flow,  # FIXTURE: Must be the absolute last argument
):
    """
    _write_postmortem uses advance_actor to compute the artifact timestamp.
    That timestamp must fall inside the writer's work block — i.e. between
    their cursor before and after the call.
    """
    import flow as flow_module

    mock_crew_instance = MagicMock()
    mock_crew_instance.kickoff.return_value = "## Postmortem\n\nRoot cause: OOM."

    # Assign the mock instance to BOTH Crew mocks
    mock_crew_class.return_value = mock_crew_instance
    mock_cw_crew.return_value = mock_crew_instance

    writer = flow_module.resolve_role("postmortem_writer")
    mock_flow._clock.reset_to_business_start(flow_module.ALL_NAMES)
    cursor_before = mock_flow._clock.now(writer)

    inc = MagicMock()
    inc.ticket_id = "ORG-999"
    inc.title = "Test Incident"
    inc.root_cause = "OOM on worker node"
    inc.days_active = 2
    inc.pr_id = None
    inc.causal_chain = MagicMock()
    inc.causal_chain.snapshot.return_value = ["ORG-999"]
    mock_flow._mem.get_ticket.return_value = {"id": "ORG-999"}

    mock_flow._write_postmortem(inc)

    cursor_after = mock_flow._clock.now(writer)

    # Grab the confluence_created SimEvent with the postmortem tag to inspect its timestamp
    all_calls = mock_flow._mem.log_event.call_args_list
    pm_evt = next(
        c.args[0]
        for c in all_calls
        if c.args[0].type == "confluence_created" and "postmortem" in c.args[0].tags
    )
    stamped = datetime.fromisoformat(pm_evt.timestamp)

    assert cursor_before <= stamped <= cursor_after, (
        f"Postmortem timestamp {stamped} is outside the writer's work block "
        f"[{cursor_before}, {cursor_after}]"
    )


@patch("flow.Crew")
@patch("flow.Task")
@patch("agent_factory.Agent")
def test_sprint_planning_simevent_timestamp_within_scheduled_window(
    mock_agent_class, mock_task_class, mock_crew_class, mock_flow
):
    """
    _handle_sprint_planning calls clock.schedule_meeting(attendees, min_hour=9,
    max_hour=11) and stamps the sprint_planned SimEvent with that time.
    The timestamp must fall inside [09:00, 11:00) on the current sim date.
    """
    mock_crew_instance = MagicMock()
    mock_crew_instance.kickoff.return_value = "Sprint plan output"
    mock_crew_class.return_value = mock_crew_instance

    import flow as flow_module

    mock_flow._clock.reset_to_business_start(flow_module.ALL_NAMES)

    mock_flow._handle_sprint_planning()

    all_calls = mock_flow._mem.log_event.call_args_list
    sprint_evt = next(
        c.args[0] for c in all_calls if c.args[0].type == "sprint_planned"
    )
    stamped = datetime.fromisoformat(sprint_evt.timestamp)

    window_start = mock_flow.state.current_date.replace(hour=9, minute=0, second=0)
    window_end = mock_flow.state.current_date.replace(hour=11, minute=0, second=0)
    assert window_start <= stamped < window_end, (
        f"sprint_planned timestamp {stamped} outside expected window "
        f"[{window_start}, {window_end})"
    )


@patch("flow.Crew")
@patch("flow.Task")
@patch("agent_factory.Agent")
def test_retrospective_simevent_timestamp_within_scheduled_window(
    mock_agent_class, mock_task_class, mock_crew_class, mock_flow
):
    """
    _handle_retrospective calls clock.schedule_meeting(attendees, min_hour=14,
    max_hour=16) — the retrospective SimEvent timestamp must land in [14:00, 16:00).
    """
    mock_crew_instance = MagicMock()
    mock_crew_instance.kickoff.return_value = "## Retro\n\nWhat went well: everything."
    mock_crew_class.return_value = mock_crew_instance

    import flow as flow_module

    mock_flow._clock.reset_to_business_start(flow_module.ALL_NAMES)

    mock_flow._handle_retrospective()

    all_calls = mock_flow._mem.log_event.call_args_list
    retro_evt = next(c.args[0] for c in all_calls if c.args[0].type == "retrospective")
    stamped = datetime.fromisoformat(retro_evt.timestamp)

    window_start = mock_flow.state.current_date.replace(hour=14, minute=0, second=0)
    window_end = mock_flow.state.current_date.replace(hour=16, minute=0, second=0)
    assert window_start <= stamped < window_end, (
        f"retrospective timestamp {stamped} outside expected window "
        f"[{window_start}, {window_end})"
    )


def test_render_template():
    with (
        patch("flow.LEGACY", {"name": "OldDB", "project_name": "ProjectX"}),
        patch("flow.PRODUCT_PAGE", "Product1"),
        patch("flow.COMPANY_NAME", "TestCorp"),
        patch("flow.INDUSTRY", "Tech"),
    ):
        res = render_template(
            "{legacy_system} {project_name} {product_page} {company_name} {industry}"
        )
        assert res == "OldDB ProjectX Product1 TestCorp Tech"


def test_dept_of():
    with patch(
        "flow.ORG_CHART", {"Engineering": ["Alice", "Bob"], "Sales": ["Charlie"]}
    ):
        assert dept_of("Bob") == "Engineering"
        assert dept_of("Charlie") == "Sales"
        assert dept_of("UnknownPerson") == "Unknown"


def test_email_of():
    with patch("flow.COMPANY_DOMAIN", "example.com"):
        assert email_of("Alice") == "alice@example.com"
        assert email_of("Bob-Smith") == "bob-smith@example.com"


def test_sprint_state_start_date():
    sprint = SprintState(sprint_number=2)
    start_dt = datetime(2026, 1, 1)
    res = sprint.start_date(start_dt, 10)
    assert res.weekday() < 5
    assert (res - start_dt).days == 14


def test_score_sentiment():
    messages = [{"text": "This is terrible and I hate it!"}, {"text": "Awful."}]
    res = score_sentiment(messages)
    assert res < 0.5

    messages_good = [
        {"text": "This is absolutely wonderful and great!"},
        {"text": "Fantastic."},
    ]
    res_good = score_sentiment(messages_good)
    assert res_good > 0.5

    assert score_sentiment([]) == 0.5


def test_build_social_graph(make_test_memory):
    make_test_memory._db["sim_config"].insert_one(
        {
            "_id": "inbound_email_sources",
            "sources": [{"name": "VendorA", "internal_liaison": "Eng"}],
        }
    )

    with (
        patch("flow.ORG_CHART", {"Eng": ["Alice", "Bob"], "Sales": ["Charlie"]}),
        patch("flow.LEADS", {"Eng": "Alice", "Sales": "Charlie"}),
    ):
        G = build_social_graph(make_test_memory)
        assert G.has_node("Alice")
        assert G.has_node("VendorA")
        assert G.nodes["Alice"]["is_lead"] is True
        assert G.nodes["Bob"]["is_lead"] is False
        assert G.nodes["VendorA"]["external"] is True
        assert G.has_edge("VendorA", "Alice")
        assert G["Alice"]["Bob"]["weight"] > G["Alice"]["Charlie"]["weight"]


def test_git_simulator_merge_pr(mock_flow):
    mock_flow._mem._prs = MagicMock()
    git_sim = GitSimulator(
        mock_flow.state, mock_flow._mem, mock_flow.social_graph, MagicMock()
    )
    with patch("os.path.exists", return_value=False):
        git_sim.merge_pr("PR-100")
        mock_flow._mem._prs.update_one.assert_called_with(
            {"pr_id": "PR-100"}, {"$set": {"status": "merged"}}
        )


def test_is_sprint_planning_day(mock_flow):
    with patch("flow.CONFIG", {"simulation": {"sprint_length_days": 10}}):
        mock_flow.state.day = 1
        assert mock_flow._is_sprint_planning_day() is True
        mock_flow.state.day = 2
        assert mock_flow._is_sprint_planning_day() is False
        mock_flow.state.day = 11
        assert mock_flow._is_sprint_planning_day() is True


def test_is_retro_day(mock_flow):
    with patch("flow.CONFIG", {"simulation": {"sprint_length_days": 10}}):
        mock_flow.state.day = 9
        assert mock_flow._is_retro_day() is True
        mock_flow.state.day = 10
        assert mock_flow._is_retro_day() is False
        mock_flow.state.day = 19
        assert mock_flow._is_retro_day() is True


def test_is_standup_day(mock_flow):
    with patch("flow.CONFIG", {"simulation": {"sprint_length_days": 10}}):
        mock_flow.state.current_date = datetime(2026, 1, 5)
        mock_flow.state.day = 2
        assert mock_flow._is_standup_day() is True

        mock_flow.state.current_date = datetime(2026, 1, 6)
        assert mock_flow._is_standup_day() is False


def test_record_daily_actor_and_event(mock_flow):
    mock_flow.state.daily_active_actors = []
    mock_flow.state.daily_event_type_counts = {}

    mock_flow._record_daily_actor("Alice", "Bob")
    mock_flow._record_daily_actor("Alice")
    assert mock_flow.state.daily_active_actors == ["Alice", "Bob", "Alice"]

    mock_flow._record_daily_event("test_event")
    mock_flow._record_daily_event("test_event")
    assert mock_flow.state.daily_event_type_counts["test_event"] == 2


def test_select_domain_expert(mock_flow):
    mock_flow._mem.find_expert_by_skill = MagicMock(
        return_value=[
            {"name": "Alice", "dept": "Engineering_Backend", "score": 0.9},
            {"name": "Bob", "dept": "Engineering_Backend", "score": 0.8},
        ]
    )
    with patch("flow.LIVE_ORG_CHART", {"Engineering_Backend": ["Alice", "Bob"]}):
        res = mock_flow._select_domain_expert(
            "DB crash", exclude="Alice", search_depts={"Engineering_Backend"}
        )
        assert res == "Bob"

        res_no_exclude = mock_flow._select_domain_expert(
            "DB crash", exclude="Charlie", search_depts={"Engineering_Backend"}
        )
        assert res_no_exclude == "Alice"


def test_advance_incidents_state_machine(mock_flow):
    from flow import ActiveIncident

    inc = ActiveIncident(ticket_id="INC-1", title="Test", day_started=1)
    mock_flow.state.active_incidents = [inc]
    mock_flow._git.create_pr = MagicMock(
        return_value={"pr_id": "PR-1", "reviewers": ["Bob"]}
    )
    mock_flow._emit_bot_message = MagicMock()
    mock_flow._mem.get_ticket = MagicMock(return_value={})

    mock_flow._advance_incidents()
    assert inc.stage == "investigating"

    mock_flow._advance_incidents()
    assert inc.stage == "fix_in_progress"
    assert inc.pr_id == "PR-1"

    mock_flow._mem._prs.find_one = MagicMock(return_value={"reviewers": ["Bob"]})
    mock_flow._normal_day._handle_pr_review_for_incident = MagicMock()
    mock_flow._advance_incidents()
    assert inc.stage == "review_pending"
    mock_flow._normal_day._handle_pr_review_for_incident.assert_called()

    mock_flow._git.merge_pr = MagicMock()
    mock_flow._write_postmortem = MagicMock()
    mock_flow._advance_incidents()
    assert inc.stage == "resolved"
    assert "INC-1" in mock_flow.state.resolved_incidents


def test_build_llm_ollama_branch():
    with (
        patch("flow._PROVIDER", "ollama"),
        patch("flow._PRESET", {"planner": "mock-planner"}),
        patch("flow.OllamaLLM") as MockOllama,
    ):
        from flow import build_llm

        build_llm("planner")
        MockOllama.assert_called_with(
            model="mock-planner", base_url="http://localhost:11434"
        )
