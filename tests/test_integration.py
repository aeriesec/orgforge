import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch
from planner_models import OrgDayPlan, DepartmentDayPlan, EngineerDayPlan, AgendaItem

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

@patch("confluence_writer.Crew")
@patch("confluence_writer.Task")
@patch("confluence_writer.Agent")
@patch("flow.Crew")
@patch("flow.Task")
@patch("flow.Agent")
def test_5_day_deep_integration(mock_flow_agent, mock_flow_task, mock_flow_crew, 
                                mock_cw_agent, mock_cw_task, mock_cw_crew, integration_flow):
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

    # 2. Memory & State Setup
    integration_flow._mem.has_genesis_artifacts = MagicMock(return_value=True)
    integration_flow._mem.load_latest_checkpoint = MagicMock(return_value=None)

    integration_flow._mem.log_event = integration_flow._mem.__class__.log_event.__get__(integration_flow._mem)

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
                date=date_str
            )
        
        # 2. Handle Normal Day Branch
        # EngineerDayPlan requires: name, dept, agenda, stress_level
        alice_agenda = [
            AgendaItem(activity_type="ticket_progress", description="Working on ORG-101", related_id="ORG-101"),
            AgendaItem(activity_type="async_question", description="Asking about API", collaborator=["Bob"])
        ]
        
        alice_plan = EngineerDayPlan(
            name="Alice", 
            dept="Engineering", # Changed from 'role' to 'dept'
            agenda=alice_agenda,
            stress_level=30      # Added missing required field
        )
        
        # DepartmentDayPlan requires: dept, theme, engineer_plans, proposed_events, 
        # cross_dept_signals, planner_reasoning, day, date
        dept_plan = DepartmentDayPlan(
            dept="Engineering",               # Changed from 'dept_name'
            theme="Standard dev work",        # Changed from 'plan_summary'
            engineer_plans=[alice_plan],
            proposed_events=[],               # Added missing field
            cross_dept_signals=[],            # Added missing field
            planner_reasoning="Test logic",   # Added missing field
            day=current_day,                  # Added missing field
            date=date_str                     # Added missing field
        )
        
        # OrgDayPlan requires: org_theme, dept_plans, collision_events, 
        # coordinator_reasoning, day, date
        return OrgDayPlan(
            org_theme="normal feature work",
            dept_plans={"Engineering": dept_plan},
            collision_events=[],
            coordinator_reasoning="Assembling test plans", # Added missing field
            day=current_day,                               # Added missing field
            date=date_str                                  # Added missing field
        )
    
    with patch.object(integration_flow._day_planner, 'plan', side_effect=dynamic_plan):
        integration_flow.state.max_days = 5
        integration_flow.state.day = 1
        integration_flow.state.current_date = datetime(2026, 3, 9)

        try:
            integration_flow.daily_cycle()
        except Exception as e:
            # This will catch any AttributeError or NameError inside normal_day.py
            pytest.fail(f"Deep Smoke Test crashed! Error: {e}")

    # 4. Final Verifications
    assert integration_flow.state.day == 6
    # Verify Alice actually 'worked' - check if events were logged to mongomock
    events = list(integration_flow._mem._events.find({"actors": "Alice"}))
    assert len(events) > 0, "No activities were recorded for Alice in the database."