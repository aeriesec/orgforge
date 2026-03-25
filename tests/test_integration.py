"""
test_integration.py
===================
Integration tests for NormalDayHandler routing and the DayPlanner pipeline.

Philosophy:
  - The day_planner is bypassed via patch.object so we control exact agenda inputs.
  - NormalDayHandler._dispatch runs for real — that's the path being tested.
  - All LLM calls (Crew/Task/Agent) are mocked at the module level.
  - mongomock gives us a real in-memory DB so we can assert on persisted state.
  - Each test owns one clear behavior and asserts the specific DB/state change
    that proves it happened.
"""

import json
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from planner_models import (
    OrgDayPlan,
    DepartmentDayPlan,
    EngineerDayPlan,
    AgendaItem,
    SprintContext,
)
from config_loader import ALL_NAMES


def _crew_returning(payload: dict):
    """Return a mock Crew whose kickoff() returns a JSON string."""
    m = MagicMock()
    m.kickoff.return_value = json.dumps(payload)
    return m


def _make_ticket_progress_crew():
    return _crew_returning(
        {"comment": "Made progress today.", "is_code_complete": False}
    )


def _make_pr_review_crew():
    return _crew_returning({"comment": "Looks good.", "verdict": "approved"})


def _make_pr_review_changes_crew():
    return _crew_returning(
        {"comment": "Please fix the null check.", "verdict": "changes_requested"}
    )


@pytest.fixture
def sim(make_test_memory):
    """
    Minimal OrgForgeSimulation with mongomock memory and no real LLM.

    Key choices:
      - sim._normal_day._confluence is stubbed to a MagicMock so
        _maybe_adhoc_confluence() never reaches confluence_writer.Task.
        This avoids the pydantic ValidationError from Task receiving a
        MagicMock Agent rather than a real BaseAgent instance.
      - All other Crew/Task/Agent patches are applied per-test so each
        test controls exactly what the LLM returns.
    """
    with patch("flow.build_llm"), patch("flow.Memory", return_value=make_test_memory):
        from flow import OrgForgeSimulation

        s = OrgForgeSimulation()

    s.state.day = 1
    s.state.system_health = 100
    s.state.current_date = datetime(2026, 3, 9)
    s._mem.log_slack_messages = MagicMock(return_value=("slack/path", "thread-001"))
    s._mem.has_genesis_artifacts = MagicMock(return_value=True)
    s._mem.load_latest_checkpoint = MagicMock(return_value=None)

    s._normal_day._confluence = MagicMock()

    return s


def _seed_ticket(mem, ticket_id, assignee, status="To Do", linked_prs=None):
    mem._jira.insert_one(
        {
            "id": ticket_id,
            "title": f"Test ticket {ticket_id}",
            "description": "Test",
            "assignee": assignee,
            "status": status,
            "dept": "Engineering_Mobile",
            "dept_type": "eng",
            "sprint": 1,
            "story_points": 3,
            "comments": [],
            "linked_prs": linked_prs or [],
            "in_progress_since": 1,
        }
    )


def _seed_pr(mem, pr_id, ticket_id, author, reviewers):
    mem._prs.insert_one(
        {
            "pr_id": pr_id,
            "ticket_id": ticket_id,
            "linked_ticket": ticket_id,
            "title": f"[{ticket_id}] Test PR",
            "description": "Test change",
            "author": author,
            "author_email": f"{author.lower()}@test.com",
            "reviewers": reviewers,
            "status": "open",
            "comments": [],
            "created_at": "2026-03-06T10:00:00",
        }
    )


def _make_org_plan(sim, engineer_plans_by_dept: dict) -> OrgDayPlan:
    date_str = str(sim.state.current_date.date())
    dept_plans = {}
    for dept, eng_plans in engineer_plans_by_dept.items():
        dept_plans[dept] = DepartmentDayPlan(
            dept=dept,
            theme="Test theme",
            engineer_plans=eng_plans,
            proposed_events=[],
            cross_dept_signals=[],
            planner_reasoning="Test",
            day=sim.state.day,
            date=date_str,
        )
    return OrgDayPlan(
        org_theme="test run",
        dept_plans=dept_plans,
        collision_events=[],
        coordinator_reasoning="test",
        day=sim.state.day,
        date=date_str,
    )


class TestTicketProgressRouting:
    @patch("normal_day.Crew")
    @patch("normal_day.Task")
    @patch("agent_factory.Agent")
    def test_ticket_progress_adds_jira_comment(
        self, mock_agent, mock_task, mock_crew, sim
    ):
        """
        ticket_progress on an owned ticket must append a comment to the ticket
        and log a ticket_progress SimEvent.
        """
        _seed_ticket(sim._mem, "ENG-200", ALL_NAMES[0])
        mock_crew.return_value = _make_ticket_progress_crew()

        org_plan = _make_org_plan(
            sim,
            {
                "Engineering_Mobile": [
                    EngineerDayPlan(
                        name=ALL_NAMES[0],
                        dept="Engineering_Mobile",
                        agenda=[
                            AgendaItem(
                                activity_type="ticket_progress",
                                description="Work on ENG-200",
                                related_id="ENG-200",
                                estimated_hrs=2.0,
                            )
                        ],
                        stress_level=20,
                    )
                ]
            },
        )

        sim._normal_day.handle(org_plan)

        ticket = sim._mem.get_ticket("ENG-200")
        assert len(ticket["comments"]) == 1
        assert ticket["comments"][0]["author"] == ALL_NAMES[0]

        events = list(sim._mem._events.find({"type": "ticket_progress"}))
        assert len(events) == 1
        assert events[0]["facts"]["ticket_id"] == "ENG-200"

    @patch("normal_day.Crew")
    @patch("normal_day.Task")
    @patch("agent_factory.Agent")
    def test_ticket_status_transitions_to_in_progress(
        self, mock_agent, mock_task, mock_crew, sim
    ):
        """
        A ticket in 'To Do' must move to 'In Progress' after ticket_progress.
        """
        _seed_ticket(sim._mem, "ENG-201", ALL_NAMES[0], status="To Do")
        mock_crew.return_value = _make_ticket_progress_crew()

        org_plan = _make_org_plan(
            sim,
            {
                "Engineering_Mobile": [
                    EngineerDayPlan(
                        name=ALL_NAMES[0],
                        dept="Engineering_Mobile",
                        agenda=[
                            AgendaItem(
                                activity_type="ticket_progress",
                                description="Start ENG-201",
                                related_id="ENG-201",
                                estimated_hrs=2.0,
                            )
                        ],
                        stress_level=20,
                    )
                ]
            },
        )

        sim._normal_day.handle(org_plan)

        ticket = sim._mem.get_ticket("ENG-201")
        assert ticket["status"] == "In Progress"

    @patch("normal_day.Crew")
    @patch("normal_day.Task")
    @patch("agent_factory.Agent")
    def test_ticket_progress_missing_ticket_does_not_crash(
        self, mock_agent, mock_task, mock_crew, sim
    ):
        """
        ticket_progress with a related_id that doesn't exist in the DB
        must silently return without crashing or logging a broken event.
        """
        mock_crew.return_value = _make_ticket_progress_crew()

        org_plan = _make_org_plan(
            sim,
            {
                "Engineering_Mobile": [
                    EngineerDayPlan(
                        name=ALL_NAMES[0],
                        dept="Engineering_Mobile",
                        agenda=[
                            AgendaItem(
                                activity_type="ticket_progress",
                                description="Ghost ticket",
                                related_id="ENG-GHOST",
                                estimated_hrs=1.0,
                            )
                        ],
                        stress_level=20,
                    )
                ]
            },
        )

        sim._normal_day.handle(org_plan)

        events = list(sim._mem._events.find({"type": "ticket_progress"}))
        assert len(events) == 0


class TestPRReviewRouting:
    @patch("normal_day.Crew")
    @patch("normal_day.Task")
    @patch("agent_factory.Agent")
    def test_reviewer_gets_pr_review_event_not_author(
        self, mock_agent, mock_task, mock_crew, sim
    ):
        """
        A pr_review agenda item assigned to a reviewer must log a pr_review
        SimEvent where facts["reviewer"] is the reviewer, not the author.
        """
        author = ALL_NAMES[0]
        reviewer = ALL_NAMES[1] if len(ALL_NAMES) > 1 else ALL_NAMES[0]

        _seed_ticket(
            sim._mem, "ENG-103", author, status="In Review", linked_prs=["PR-107"]
        )
        _seed_pr(sim._mem, "PR-107", "ENG-103", author, [reviewer])
        mock_crew.return_value = _make_pr_review_crew()

        org_plan = _make_org_plan(
            sim,
            {
                "Engineering_Mobile": [
                    EngineerDayPlan(
                        name=reviewer,
                        dept="Engineering_Mobile",
                        agenda=[
                            AgendaItem(
                                activity_type="pr_review",
                                description="Review PR-107",
                                related_id="PR-107",
                                estimated_hrs=1.0,
                            )
                        ],
                        stress_level=20,
                    )
                ]
            },
        )

        sim._normal_day.handle(org_plan)

        events = list(sim._mem._events.find({"type": "pr_review"}))
        assert len(events) == 1, "Expected exactly one pr_review event"
        assert events[0]["facts"]["reviewer"] == reviewer, (
            "facts.reviewer must be the assigned reviewer"
        )
        assert events[0]["facts"]["author"] == author, (
            "facts.author must be the PR author"
        )
        assert reviewer in events[0]["actors"], "Reviewer must be listed in actors"

    @patch("normal_day.Crew")
    @patch("normal_day.Task")
    @patch("agent_factory.Agent")
    def test_pr_review_changes_requested_moves_ticket_back(
        self, mock_agent, mock_task, mock_crew, sim
    ):
        """
        When a reviewer requests changes, the linked ticket must move back
        to 'In Progress' and the PR must gain changes_requested=True.
        """
        author = ALL_NAMES[0]
        reviewer = ALL_NAMES[1] if len(ALL_NAMES) > 1 else ALL_NAMES[0]

        _seed_ticket(
            sim._mem, "ENG-103", author, status="In Review", linked_prs=["PR-107"]
        )
        _seed_pr(sim._mem, "PR-107", "ENG-103", author, [reviewer])
        mock_crew.return_value = _make_pr_review_changes_crew()

        org_plan = _make_org_plan(
            sim,
            {
                "Engineering_Mobile": [
                    EngineerDayPlan(
                        name=reviewer,
                        dept="Engineering_Mobile",
                        agenda=[
                            AgendaItem(
                                activity_type="pr_review",
                                description="Review PR-107",
                                related_id="PR-107",
                                estimated_hrs=1.0,
                            )
                        ],
                        stress_level=20,
                    )
                ]
            },
        )

        sim._normal_day.handle(org_plan)

        ticket = sim._mem.get_ticket("ENG-103")
        assert ticket["status"] == "In Progress", (
            "changes_requested must move ticket back to In Progress"
        )
        pr = sim._mem._prs.find_one({"pr_id": "PR-107"}, {"_id": 0})
        assert pr.get("changes_requested") is True

    @patch("normal_day.Crew")
    @patch("normal_day.Task")
    @patch("agent_factory.Agent")
    def test_in_review_ticket_with_young_pr_author_still_comments(
        self, mock_agent, mock_task, mock_crew, sim
    ):
        """

        This test exists to make any future accidental change to this contract
        visible and deliberate.
        """
        author = ALL_NAMES[0]
        _seed_ticket(
            sim._mem, "ENG-103", author, status="In Review", linked_prs=["PR-107"]
        )
        _seed_pr(sim._mem, "PR-107", "ENG-103", author, ["Jordan", "Sam"])
        sim._mem._jira.update_one(
            {"id": "ENG-103"},
            {"$set": {"in_review_since": sim.state.day}},  # age = 0
        )
        mock_crew.return_value = _make_ticket_progress_crew()

        org_plan = _make_org_plan(
            sim,
            {
                "Engineering_Mobile": [
                    EngineerDayPlan(
                        name=author,
                        dept="Engineering_Mobile",
                        agenda=[
                            AgendaItem(
                                activity_type="ticket_progress",
                                description="Author works on In Review ticket",
                                related_id="ENG-103",
                                estimated_hrs=2.0,
                            )
                        ],
                        stress_level=20,
                    )
                ]
            },
        )

        sim._normal_day.handle(org_plan)

        ticket = sim._mem.get_ticket("ENG-103")
        assert len(ticket.get("comments", [])) == 1, (
            "Handler allows author to comment on young In Review tickets — "
            "blocking this is the planner's responsibility, not the handler's."
        )

    @patch("normal_day.Crew")
    @patch("normal_day.Task")
    @patch("agent_factory.Agent")
    def test_stale_pr_force_merged_after_5_days(
        self, mock_agent, mock_task, mock_crew, sim
    ):
        """
        A PR that has been In Review for >= 5 days with no changes_requested
        must be force-merged when the author runs ticket_progress.
        """
        author = ALL_NAMES[0]
        _seed_ticket(
            sim._mem, "ENG-103", author, status="In Review", linked_prs=["PR-107"]
        )
        _seed_pr(sim._mem, "PR-107", "ENG-103", author, ["Jordan", "Sam"])
        sim._mem._jira.update_one(
            {"id": "ENG-103"},
            {"$set": {"in_review_since": sim.state.day - 5}},  # age = 5
        )
        mock_crew.return_value = _make_pr_review_crew()

        org_plan = _make_org_plan(
            sim,
            {
                "Engineering_Mobile": [
                    EngineerDayPlan(
                        name=author,
                        dept="Engineering_Mobile",
                        agenda=[
                            AgendaItem(
                                activity_type="ticket_progress",
                                description="Force merge stale PR",
                                related_id="ENG-103",
                                estimated_hrs=0.5,
                            )
                        ],
                        stress_level=20,
                    )
                ]
            },
        )

        sim._normal_day.handle(org_plan)

        pr = sim._mem._prs.find_one({"pr_id": "PR-107"}, {"_id": 0})
        assert pr["status"] == "merged", (
            "PR stale for >= 5 days with no changes_requested must be force-merged"
        )

    @patch("normal_day.Crew")
    @patch("normal_day.Task")
    @patch("agent_factory.Agent")
    def test_pr_review_fallback_finds_reviewable_pr(
        self, mock_agent, mock_task, mock_crew, sim
    ):
        """
        If a pr_review agenda item has no related_id, _find_reviewable_pr must
        find an open PR where this person is listed as a reviewer.
        """
        author = ALL_NAMES[0]
        reviewer = ALL_NAMES[1] if len(ALL_NAMES) > 1 else ALL_NAMES[0]

        _seed_ticket(
            sim._mem, "ENG-104", author, status="In Review", linked_prs=["PR-108"]
        )
        _seed_pr(sim._mem, "PR-108", "ENG-104", author, [reviewer])
        mock_crew.return_value = _make_pr_review_crew()

        org_plan = _make_org_plan(
            sim,
            {
                "Engineering_Mobile": [
                    EngineerDayPlan(
                        name=reviewer,
                        dept="Engineering_Mobile",
                        agenda=[
                            AgendaItem(
                                activity_type="pr_review",
                                description="Review open PRs",
                                related_id=None,
                                estimated_hrs=1.0,
                            )
                        ],
                        stress_level=20,
                    )
                ]
            },
        )

        sim._normal_day.handle(org_plan)

        events = list(sim._mem._events.find({"type": "pr_review"}))
        assert len(events) == 1
        assert events[0]["facts"]["reviewer"] == reviewer


class TestPlannerInReviewSection:
    """
    Unit tests for the prompt-building layer — no LLM, no NormalDayHandler.
    Catches the data-pipeline bug where in_review_section only emitted ticket
    IDs and omitted PR IDs and reviewer names.
    """

    def test_in_review_section_includes_pr_id_and_reviewers(self, sim):
        """
        The in_review_section string must contain the PR ID and reviewer names,
        not just the bare ticket ID.
        """
        author = ALL_NAMES[0]
        reviewer = ALL_NAMES[1] if len(ALL_NAMES) > 1 else ALL_NAMES[0]

        _seed_ticket(
            sim._mem, "ENG-103", author, status="In Review", linked_prs=["PR-107"]
        )
        _seed_pr(sim._mem, "PR-107", "ENG-103", author, [reviewer])

        sprint_ctx = SprintContext(
            owned_tickets={"ENG-103": author},
            available_tickets=[],
            in_progress_ids=[],
            capacity_by_member={author: 6.0},
            in_review=["ENG-103"],
            sprint_theme="test sprint",
        )

        in_review_lines = []
        for tid in sprint_ctx.in_review:
            pr = sim._mem._prs.find_one(
                {"ticket_id": tid, "status": "open"}, {"_id": 0}
            )
            if pr:
                reviewers_str = ", ".join(pr.get("reviewers", []))
                pr_id = pr.get("pr_id", "?")
                in_review_lines.append(
                    f"  - [{tid}] → {pr_id} | awaiting review from: {reviewers_str}"
                )
            else:
                in_review_lines.append(f"  - [{tid}] → no PR found")

        section = "\n".join(in_review_lines)

        assert "PR-107" in section, "PR ID must appear in in_review_section"
        assert reviewer in section, "Reviewer name must appear in in_review_section"

    def test_in_review_section_bare_ticket_id_is_insufficient(self, sim):
        """
        Documents the OLD broken behaviour: bare ticket ID gives the LLM
        nothing to route on. Asserts the old logic omits what it should include.
        This test should FAIL if someone reverts to the old one-liner.
        """
        author = ALL_NAMES[0]
        reviewer = ALL_NAMES[1] if len(ALL_NAMES) > 1 else ALL_NAMES[0]

        _seed_ticket(
            sim._mem, "ENG-103", author, status="In Review", linked_prs=["PR-107"]
        )
        _seed_pr(sim._mem, "PR-107", "ENG-103", author, [reviewer])

        sprint_ctx = SprintContext(
            owned_tickets={"ENG-103": author},
            available_tickets=[],
            in_progress_ids=[],
            capacity_by_member={author: 6.0},
            in_review=["ENG-103"],
            sprint_theme="test sprint",
        )

        old_section = "\n".join(f"  - [{tid}]" for tid in sprint_ctx.in_review)

        assert "PR-107" not in old_section, "Confirms old logic omits PR ID"
        assert reviewer not in old_section, "Confirms old logic omits reviewer names"


class TestDailyLoop:
    @patch("normal_day.Crew")
    @patch("normal_day.Task")
    @patch("confluence_writer.Crew")
    @patch("confluence_writer.Task")
    @patch("flow.Crew")
    @patch("flow.Task")
    @patch("agent_factory.Agent")
    def test_5_day_cycle_completes_and_advances_day(
        self, mock_agent, mock_ft, mock_fc, mock_cwt, mock_cwc, mock_ndt, mock_ndc, sim
    ):
        """
        Smoke test: daily_cycle() runs to completion over 5 days without
        raising, and state.day ends at 6.
        """
        crew_inst = _crew_returning({"comment": "Done.", "is_code_complete": False})
        mock_fc.return_value = crew_inst
        mock_cwc.return_value = crew_inst
        mock_ndc.return_value = crew_inst

        author = ALL_NAMES[0]
        _seed_ticket(sim._mem, "ENG-300", author)

        def dynamic_plan(*args, **kwargs):
            return _make_org_plan(
                sim,
                {
                    "Engineering_Mobile": [
                        EngineerDayPlan(
                            name=author,
                            dept="Engineering_Mobile",
                            agenda=[
                                AgendaItem(
                                    activity_type="ticket_progress",
                                    description="Sprint work",
                                    related_id="ENG-300",
                                    estimated_hrs=2.0,
                                )
                            ],
                            stress_level=20,
                        )
                    ]
                },
            )

        with patch.object(sim._day_planner, "plan", side_effect=dynamic_plan):
            sim.state.max_days = 5
            sim.state.day = 1
            sim.state.current_date = datetime(2026, 3, 9)
            sim.daily_cycle()

        assert sim.state.day == 6

    @patch("normal_day.Crew")
    @patch("normal_day.Task")
    @patch("confluence_writer.Crew")
    @patch("confluence_writer.Task")
    @patch("flow.Crew")
    @patch("flow.Task")
    @patch("agent_factory.Agent")
    def test_incident_probability_branch_opens_incident(
        self, mock_agent, mock_ft, mock_fc, mock_cwt, mock_cwc, mock_ndt, mock_ndc, sim
    ):
        crew_inst = _crew_returning({"comment": "Done.", "is_code_complete": False})
        mock_fc.return_value = crew_inst
        mock_cwc.return_value = crew_inst
        mock_ndc.return_value = crew_inst

        def dynamic_plan(*args, **kwargs):
            date_str = str(sim.state.current_date.date())
            return OrgDayPlan(
                org_theme="normal work",
                dept_plans={},
                collision_events=[],
                coordinator_reasoning="test",
                day=sim.state.day,
                date=date_str,
            )

        with patch.object(sim._day_planner, "plan", side_effect=dynamic_plan):
            with patch("flow.random.random", return_value=0.0):
                sim.state.max_days = 5
                sim.state.day = 5
                sim.state.last_incident_day = 0
                sim.state.current_date = datetime(2026, 3, 9)
                sim.daily_cycle()

        events = list(sim._mem._events.find({"type": "incident_opened"}))
        assert len(events) >= 1


class TestNonEngineeringLifecycle:
    @patch("normal_day.Crew")
    @patch("normal_day.Task")
    @patch("agent_factory.Agent")
    def test_non_engineering_ticket_lifecycle_creates_artifact(
        self, mock_agent, mock_task, mock_crew, sim
    ):
        author = ALL_NAMES[1] if len(ALL_NAMES) > 1 else ALL_NAMES[0]

        sim._mem._jira.insert_one(
            {
                "id": "MKT-100",
                "title": "Q3 Marketing Plan",
                "description": "Draft the plan",
                "assignee": author,
                "status": "To Do",
                "dept": "Sales_Marketing",
                "dept_type": "non_eng",
                "completion_artifact": "email",
                "sprint": 1,
                "story_points": 3,
                "comments": [],
                "in_progress_since": 1,
            }
        )

        mock_crew_inst = MagicMock()
        mock_crew_inst.kickoff.return_value = (
            '{"comment": "Drafted the plan", "is_task_complete": true}'
        )
        mock_crew.return_value = mock_crew_inst

        org_plan = _make_org_plan(
            sim,
            {
                "Sales_Marketing": [
                    EngineerDayPlan(
                        name=author,
                        dept="Sales_Marketing",
                        agenda=[
                            AgendaItem(
                                activity_type="ticket_progress",
                                description="Write Q3 Plan",
                                related_id="MKT-100",
                                estimated_hrs=2.0,
                            )
                        ],
                        stress_level=20,
                    )
                ]
            },
        )

        sim._normal_day.handle(org_plan)

        ticket = sim._mem.get_ticket("MKT-100")
        assert ticket["status"] == "Done"

        events = list(sim._mem._events.find({"type": "ticket_completion_email"}))
        assert len(events) == 1
        assert events[0]["facts"]["ticket_id"] == "MKT-100"

    @patch("normal_day.Crew")
    @patch("normal_day.Task")
    @patch("agent_factory.Agent")
    def test_causal_chain_preservation_on_incident_tickets(
        self, mock_agent, mock_task, mock_crew, sim
    ):
        author = ALL_NAMES[0]
        _seed_ticket(sim._mem, "ENG-999", author)

        from flow import ActiveIncident
        from causal_chain_handler import CausalChainHandler

        chain = CausalChainHandler("ENG-999")
        chain.append("slack_pagerduty_01")

        inc = ActiveIncident(
            ticket_id="ENG-999",
            title="P1: DB Down",
            day_started=1,
            root_cause="OOM",
            causal_chain=chain,
            on_call=author,
        )
        sim.state.active_incidents.append(inc)

        mock_crew_inst = MagicMock()
        mock_crew_inst.kickoff.return_value = (
            '{"comment": "Investigating logs", "is_code_complete": false}'
        )
        mock_crew.return_value = mock_crew_inst

        org_plan = _make_org_plan(
            sim,
            {
                "Engineering_Mobile": [
                    EngineerDayPlan(
                        name=author,
                        dept="Engineering_Mobile",
                        agenda=[
                            AgendaItem(
                                activity_type="ticket_progress",
                                description="Fix DB",
                                related_id="ENG-999",
                                estimated_hrs=1.0,
                            )
                        ],
                        stress_level=80,
                    )
                ]
            },
        )

        sim._normal_day.handle(org_plan)

        ticket = sim._mem.get_ticket("ENG-999")
        saved_chain = ticket.get("causal_chain", [])

        assert "slack_pagerduty_01" in saved_chain
        assert any("comment" in item for item in saved_chain)

    @patch("normal_day.Crew")
    @patch("normal_day.Task")
    @patch("agent_factory.Agent")
    def test_watercooler_distraction_applies_time_penalty(
        self, mock_agent, mock_task, mock_crew, sim
    ):
        author = ALL_NAMES[0]

        mock_crew_inst = MagicMock()
        mock_crew_inst.kickoff.return_value = (
            '{"speaker": "Alice", "message": "Did you see that?"}'
        )
        mock_crew.return_value = mock_crew_inst

        agenda_item = AgendaItem(
            activity_type="deep_work",
            description="Focus time",
            related_id=None,
            estimated_hrs=2.0,
        )

        org_plan = _make_org_plan(
            sim,
            {
                "Engineering_Mobile": [
                    EngineerDayPlan(
                        name=author,
                        dept="Engineering_Mobile",
                        agenda=[agenda_item],
                        stress_level=20,
                    )
                ]
            },
        )

        with patch("random.random", return_value=0.0):
            sim._normal_day.handle(org_plan)

        assert agenda_item.estimated_hrs > 2.0

        events = list(sim._mem._events.find({"type": "watercooler_chat"}))
        assert len(events) >= 1
