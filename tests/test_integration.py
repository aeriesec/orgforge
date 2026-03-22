import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch
from planner_models import OrgDayPlan, DepartmentDayPlan, EngineerDayPlan, AgendaItem
from config_loader import ALL_NAMES  # <-- Import the real simulation roster


@pytest.fixture
def integration_flow(make_test_memory):
    """
    Creates a Flow instance that uses mongomock for real CRUD operations.
    """
    # We patch flow.Memory to return the mongomock instance from conftest.py
    with patch("flow.build_llm"), patch("flow.Memory", return_value=make_test_memory):
        from flow import Flow

        flow = Flow()
        # Initialize basic state for the simulation
        flow.state.day = 1
        flow.state.system_health = 100
        # Mock file I/O to prevent writing local JSON files during testing
        flow._mem.log_slack_messages = MagicMock(return_value=("", ""))
        return flow


@patch("normal_day.Crew")
@patch("normal_day.Task")
@patch("confluence_writer.Crew")
@patch("confluence_writer.Task")
@patch("flow.Crew")
@patch("flow.Task")
@patch("agent_factory.Agent")
def test_5_day_deep_integration(
    mock_flow_task,
    mock_flow_crew,
    mock_cw_agent,
    mock_cw_task,
    mock_cw_crew,
    mock_nd_task,
    mock_nd_crew,
    integration_flow,
):
    """
    DEEP SMOKE TEST: Verifies Normal Day logic, incident lifecycles, and
    database persistence without crashing.
    """
    # 1. Setup Robust LLM Mock
    mock_crew_instance = MagicMock()
    # Return a list for ticket/PR generation, or a string for Slack/Confluence
    mock_crew_instance.kickoff.return_value = '[{"title": "Deep Test Ticket", "story_points": 3, "description": "Verify logic"}]'

    mock_flow_crew.return_value = mock_crew_instance
    mock_cw_crew.return_value = mock_crew_instance
    mock_nd_crew.return_value = mock_crew_instance

    mock_flow_task.return_value.output.raw = "mocked message"
    mock_cw_task.return_value.output.raw = "mocked message"
    mock_nd_task.return_value.output.raw = "Alice: mocked message"

    # 2. Memory & State Setup
    integration_flow._mem.has_genesis_artifacts = MagicMock(return_value=True)
    integration_flow._mem.load_latest_checkpoint = MagicMock(return_value=None)

    integration_flow._mem.log_event = integration_flow._mem.__class__.log_event.__get__(
        integration_flow._mem
    )
    
    # Grab real names from the configuration so the social graph doesn't crash
    test_actor = ALL_NAMES[0]
    test_collab = ALL_NAMES[1] if len(ALL_NAMES) > 1 else test_actor

    # 3. The "Un-Mocked" Day Plan
    def dynamic_plan(*args, **kwargs):
        # Capture current state for the models
        current_day = integration_flow.state.day
        date_str = str(integration_flow.state.current_date.date())

        # 1. Handle Incident Branch
        if current_day == 2:
            return OrgDayPlan(
                org_theme="critical server crash detected",
                dept_plans={},
                collision_events=[],
                coordinator_reasoning="Forced incident for testing",
                day=current_day,
                date=date_str,
            )

        # 2. Handle Normal Day Branch
        # EngineerDayPlan requires: name, dept, agenda, stress_level
        test_actor_agenda = [
            AgendaItem(
                activity_type="ticket_progress",
                description="Working on ORG-101",
                related_id="ORG-101",
            ),
            AgendaItem(
                activity_type="async_question",
                description="Asking about API",
                collaborator=[test_collab],
            ),
        ]

        test_actor_plan = EngineerDayPlan(
            name=test_actor,
            dept="Engineering",  
            agenda=test_actor_agenda,
            stress_level=30,  
        )

        dept_plan = DepartmentDayPlan(
            dept="Engineering",  
            theme="Standard dev work",  
            engineer_plans=[test_actor_plan],
            proposed_events=[],  
            cross_dept_signals=[],  
            planner_reasoning="Test logic",  
            day=current_day,  
            date=date_str,  
        )

        return OrgDayPlan(
            org_theme="normal feature work",
            dept_plans={"Engineering": dept_plan},
            collision_events=[],
            coordinator_reasoning="Assembling test plans",  
            day=current_day,  
            date=date_str,  
        )

    with patch.object(integration_flow._day_planner, "plan", side_effect=dynamic_plan):
        integration_flow.state.max_days = 5
        integration_flow.state.day = 1
        integration_flow.state.current_date = datetime(2026, 3, 9)

        try:
            integration_flow.daily_cycle()
        except Exception as e:
            pytest.fail(f"Deep Smoke Test crashed! Error: {e}")

    # 4. Final Verifications
    assert integration_flow.state.day == 6
    # Verify our dynamic actor actually 'worked' - check if events were logged to mongomock
    events = list(integration_flow._mem._events.find({"actors": test_actor}))
    assert len(events) > 0, f"No activities were recorded for {test_actor} in the database."