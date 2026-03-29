import pytest
import json
from unittest.mock import MagicMock, patch
from datetime import datetime
from normal_day import NormalDayHandler


@pytest.fixture
def mock_handler():
    # Safely bypass the persona voice engine so it doesn't query mocked graphs
    with patch(
        "utils.persona_utils.persona_utils.get_voice_card",
        return_value="Mock backstory",
    ):
        config = {
            "simulation": {
                "domain": "test.com",
                "company_name": "TestCorp",
                "output_dir": "/tmp",
            },
            "org_chart": {"Engineering": ["Alice", "Bob"]},
            "personas": {
                "Alice": {"social_role": "Engineer"},
                "Bob": {"social_role": "Engineer"},
            },
        }
        mem = MagicMock()
        mem.get_ticket.return_value = {
            "id": "ENG-101",
            "title": "Fix DB",
            "status": "In Progress",
            "dept_type": "eng",
            "comments": [],
        }

        state = MagicMock()
        state.current_date = datetime(2026, 1, 1)
        state.day = 1
        state.active_incidents = []
        state.ticket_actors_today = {}

        gd = MagicMock()
        gd._stress = {}  # Prevents MagicMock integer comparisons

        social_graph = MagicMock()
        git = MagicMock()
        git.create_pr.return_value = {"pr_id": "PR-999", "reviewers": ["Bob"]}

        clock = MagicMock()
        clock.now.return_value = datetime(2026, 1, 1, 10, 0)
        clock.advance_actor.return_value = (
            datetime(2026, 1, 1, 12, 0),
            datetime(2026, 1, 1, 12, 0),
        )

        handler = NormalDayHandler(
            config=config,
            mem=mem,
            state=state,
            graph_dynamics=gd,
            social_graph=social_graph,
            git=git,
            worker_llm=MagicMock(),
            planner_llm=MagicMock(),
            clock=clock,
            persona_helper=MagicMock(),
        )
        handler._save_slack = MagicMock(return_value=("path/to/slack", "thread-123"))
        handler._emit_bot_message = MagicMock(return_value="bot-thread-123")

        yield handler


@pytest.fixture
def dummy_eng_plan():
    plan = MagicMock()
    plan.name = "Alice"
    plan.is_on_call = False
    return plan


@pytest.fixture
def dummy_agenda_item():
    item = MagicMock()
    item.activity_type = "ticket_progress"
    item.description = "Work on DB"
    item.related_id = "ENG-101"
    item.estimated_hrs = 2.0
    item.collaborator = []
    return item


def test_dispatch_routing(mock_handler, dummy_eng_plan, dummy_agenda_item):
    mock_handler._handle_ticket_progress = MagicMock()
    mock_handler._handle_pr_review = MagicMock()
    mock_handler._handle_one_on_one = MagicMock()

    dummy_agenda_item.activity_type = "ticket_progress"
    mock_handler._dispatch(dummy_eng_plan, dummy_agenda_item, MagicMock(), "2026-01-01")
    mock_handler._handle_ticket_progress.assert_called_once()

    dummy_agenda_item.activity_type = "pr_review"
    mock_handler._dispatch(dummy_eng_plan, dummy_agenda_item, MagicMock(), "2026-01-01")
    mock_handler._handle_pr_review.assert_called_once()

    dummy_agenda_item.activity_type = "1on1"
    mock_handler._dispatch(dummy_eng_plan, dummy_agenda_item, MagicMock(), "2026-01-01")
    mock_handler._handle_one_on_one.assert_called_once()


def test_try_force_merge_stale_pr(mock_handler):
    ticket = {
        "id": "ENG-200",
        "status": "In Review",
        "linked_prs": ["PR-555"],
        "in_review_since": 1,
    }
    mock_handler._state.day = 7

    mock_handler._mem._prs.find_one = MagicMock(
        side_effect=lambda q, *a, **kw: (
            {"pr_id": "PR-555", "status": "open"}
            if q.get("pr_id") == "PR-555" and not q.get("changes_requested")
            else None
        )
    )
    mock_handler._handle_pr_review_for_incident = MagicMock()
    mock_handler._git.merge_pr = MagicMock()

    result = mock_handler._try_force_merge_stale_pr(ticket, "Alice", "2026-01-07")

    assert result is True
    assert ticket["status"] == "Done"
    mock_handler._git.merge_pr.assert_called_once_with("PR-555")
    mock_handler._emit_bot_message.assert_called()


def test_complete_non_eng_ticket(mock_handler):
    ticket = {
        "id": "HR-101",
        "title": "Update Handbook",
        "status": "In Progress",
        "completion_artifact": "confluence",
    }
    mock_handler._confluence = MagicMock()
    mock_handler._create_design_doc_stub = MagicMock(return_value="CONF-999")
    chain = MagicMock()

    comp_id = mock_handler._complete_non_eng_ticket(
        ticket,
        "Alice",
        "Wrote handbook",
        "Ctx",
        "2026-01-01",
        "2026-01-01T10:00:00",
        chain,
    )

    assert comp_id == "CONF-999"
    assert ticket["status"] == "Done"
    chain.append.assert_called_with("CONF-999")


@patch("normal_day.Crew")
def test_ticket_progress_incomplete_no_pr(
    mock_crew, mock_handler, dummy_eng_plan, dummy_agenda_item
):
    mock_crew_instance = MagicMock()
    mock_crew_instance.kickoff.return_value = (
        '{"comment": "Still working", "is_code_complete": false}'
    )
    mock_crew.return_value = mock_crew_instance

    mock_handler._handle_ticket_progress(
        dummy_eng_plan, dummy_agenda_item, "2026-01-01"
    )

    mock_handler._git.create_pr.assert_not_called()

    ticket = mock_handler._mem.get_ticket("ENG-101")
    assert any("Still working" in c["text"] for c in ticket["comments"])


@patch("normal_day.Crew")
def test_ticket_progress_complete_spawns_pr(
    mock_crew, mock_handler, dummy_eng_plan, dummy_agenda_item
):
    mock_crew_instance = MagicMock()
    mock_crew_instance.kickoff.return_value = (
        '{"comment": "Finished the API", "is_code_complete": true}'
    )
    mock_crew.return_value = mock_crew_instance

    mock_handler._handle_ticket_progress(
        dummy_eng_plan, dummy_agenda_item, "2026-01-01"
    )

    mock_handler._git.create_pr.assert_called_once()
    ticket = mock_handler._mem.get_ticket("ENG-101")
    assert "PR-999" in ticket.get("linked_prs", [])


@patch("normal_day.Crew")
def test_ticket_progress_detects_blocker(
    mock_crew, mock_handler, dummy_eng_plan, dummy_agenda_item
):
    mock_crew_instance = MagicMock()
    mock_crew_instance.kickoff.return_value = (
        '{"comment": "I am blocked on the DB access", "is_code_complete": false}'
    )
    mock_crew.return_value = mock_crew_instance

    mock_handler._handle_ticket_progress(
        dummy_eng_plan, dummy_agenda_item, "2026-01-01"
    )

    logged_events = [
        call.args[0].type for call in mock_handler._mem.log_event.call_args_list
    ]
    assert "blocker_flagged" in logged_events


@patch("normal_day.Crew")
def test_pr_review_approved_merges_pr(
    mock_crew, mock_handler, dummy_eng_plan, dummy_agenda_item
):
    dummy_agenda_item.activity_type = "pr_review"
    dummy_agenda_item.related_id = "PR-999"

    mock_pr = {"pr_id": "PR-999", "author": "Bob", "title": "Fix", "status": "open"}
    mock_handler._find_pr = MagicMock(return_value=mock_pr)

    mock_crew_instance = MagicMock()
    mock_crew_instance.kickoff.return_value = (
        '{"comment": "Looks great", "verdict": "approved"}'
    )
    mock_crew.return_value = mock_crew_instance

    mock_handler._handle_pr_review(dummy_eng_plan, dummy_agenda_item, "2026-01-01")

    assert mock_pr["status"] == "merged"
    mock_handler._git.merge_pr.assert_called_once_with("PR-999")
    mock_handler._emit_bot_message.assert_called()


@patch("normal_day.Crew")
def test_pr_review_changes_requested_triggers_reply(
    mock_crew, mock_handler, dummy_eng_plan, dummy_agenda_item
):
    dummy_agenda_item.activity_type = "pr_review"
    dummy_agenda_item.related_id = "PR-999"

    mock_pr = {"pr_id": "PR-999", "author": "Bob", "title": "Fix", "status": "open"}
    mock_handler._find_pr = MagicMock(return_value=mock_pr)
    mock_handler._emit_review_reply = MagicMock(
        return_value=(["Alice", "Bob"], "reply-thread-id")
    )

    mock_crew_instance = MagicMock()
    mock_crew_instance.kickoff.return_value = (
        '{"comment": "Missing tests", "verdict": "changes_requested"}'
    )
    mock_crew.return_value = mock_crew_instance

    mock_handler._handle_pr_review(dummy_eng_plan, dummy_agenda_item, "2026-01-01")

    assert mock_pr.get("changes_requested") is True
    assert mock_pr["status"] == "open"
    mock_handler._git.merge_pr.assert_not_called()
    mock_handler._emit_review_reply.assert_called_once()


@patch("normal_day.Crew")
def test_emit_blocker_slack(mock_crew, mock_handler):
    mock_crew_instance = MagicMock()
    mock_crew_instance.kickoff.return_value = json.dumps(
        [
            {"speaker": "Alice", "message": "I am blocked on DB access"},
            {"speaker": "Bob", "message": "I can grant you permissions now"},
        ]
    )
    mock_crew.return_value = mock_crew_instance

    mock_handler._save_slack = MagicMock(return_value=("slack_path", "thread-abc"))

    participants = mock_handler._emit_blocker_slack(
        "Alice",
        "Bob",
        "ENG-101",
        "DB Issue",
        "No access",
        "2026-01-01",
        "2026-01-01T10:00:00",
    )

    assert participants == ["Alice", "Bob"]
    mock_handler._save_slack.assert_called_once()
    logged_events = [c.args[0].type for c in mock_handler._mem.log_event.call_args_list]
    assert "blocker_flagged" in logged_events


@patch("normal_day.Crew")
def test_ticket_progress_timeout_force_complete(
    mock_crew, mock_handler, dummy_eng_plan, dummy_agenda_item
):
    mock_crew_instance = MagicMock()
    mock_crew_instance.kickoff.return_value = (
        '{"comment": "Still working, lots to do", "is_code_complete": false}'
    )
    mock_crew.return_value = mock_crew_instance

    ticket = mock_handler._mem.get_ticket.return_value
    ticket["in_progress_since"] = 1
    mock_handler._state.day = 5

    mock_handler._handle_ticket_progress(
        dummy_eng_plan, dummy_agenda_item, "2026-01-05"
    )

    mock_handler._git.create_pr.assert_called_once()
    assert ticket["status"] == "In Review"
