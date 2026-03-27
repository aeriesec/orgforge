"""
test_crm_system.py
==================
Unit tests for the CRM state machine.
Validates MongoDB writes, JSON exports, and SimEvent emissions.
"""

import pytest
from datetime import datetime
from crm_system import CRMSystem, NullCRMSystem


@pytest.fixture
def crm_config():
    return {
        "crm": {
            "salesforce": {"enabled": True, "seed_accounts": True},
            "zendesk": {"enabled": True, "link_to_incidents": True},
        }
    }


@pytest.fixture
def crm(crm_config, tmp_path, make_test_memory):
    return CRMSystem.from_config(crm_config, tmp_path, make_test_memory)


def test_factory_returns_null_when_disabled(tmp_path, make_test_memory):
    system = CRMSystem.from_config({}, tmp_path, make_test_memory)
    assert isinstance(system, NullCRMSystem)


def test_handle_inbound_complaint_creates_zd_ticket(crm, make_test_memory):
    facts = {"subject": "System is down", "sender_org": "Globex", "body": "Help!"}
    ticket_id = crm.handle_inbound_complaint(
        facts, "2026-03-09T10:00:00", "2026-03-09", 1
    )

    assert ticket_id.startswith("ZD-")
    ticket = make_test_memory._db["zd_tickets"].find_one({"ticket_id": ticket_id})
    assert ticket["status"] == "Open"
    assert ticket["org_name"] == "Globex"

    events = list(make_test_memory._events.find({"type": "zd_ticket_opened"}))
    assert len(events) == 1
    assert events[0]["facts"]["ticket_id"] == ticket_id


def test_incident_escalates_zd_and_flags_sf(crm, make_test_memory):

    crm.handle_inbound_complaint(
        {"subject": "Broken", "sender_org": "Initech"},
        "2026-03-09T10:00:00",
        "2026-03-09",
        1,
    )
    crm.process_outbound_email(
        {
            "recipient_org": "Initech",
            "subject": "Quote",
            "stage": "Proposal/Price Quote",
        },
        "2026-03-09T10:00:00",
        "2026-03-09",
        1,
    )

    crm.handle_incident_opened(
        "ENG-999", "Database", 40, "2026-03-09T12:00:00", "2026-03-09", 1
    )

    zd_ticket = make_test_memory._db["zd_tickets"].find_one({"org_name": "Initech"})
    assert zd_ticket["priority"] == "Urgent"
    assert zd_ticket["related_incident"] == "ENG-999"

    sf_opp = make_test_memory._db["sf_opps"].find_one({"account_name": "Initech"})
    assert len(sf_opp["risk_notes"]) == 1
    assert "ENG-999" in sf_opp["risk_notes"][0]

    context = crm.planner_context()
    assert "OPEN SUPPORT TICKETS" in context
    assert "[URGENT]" in context
    assert "AT-RISK DEALS" in context


def test_employee_departure_reassigns_crm_assets(crm, make_test_memory):

    crm.process_outbound_email(
        {"sender": "Bob", "recipient_org": "Stark Ind", "stage": "Negotiation/Review"},
        "2026-03-09T10:00:00",
        "2026-03-09",
        1,
    )

    crm.handle_employee_departure("Bob", "Sales Exec", "2026-03-10", 2)

    opp = make_test_memory._db["sf_opps"].find_one({"account_name": "Stark Ind"})
    assert opp["owner"] == "Pending Reassignment"
    assert any("departed" in note.lower() for note in opp["risk_notes"])


def test_partial_enablement_does_not_crash(tmp_path, make_test_memory):
    """If ZD is off but SF is on, ZD methods must silently no-op without crashing."""
    cfg = {"crm": {"salesforce": {"enabled": True}, "zendesk": {"enabled": False}}}
    crm = CRMSystem.from_config(cfg, tmp_path, make_test_memory)

    ticket_id = crm.handle_inbound_complaint({"subject": "Help"}, "T0", "D0", 1)
    assert ticket_id is None
    assert make_test_memory._db["zd_tickets"].count_documents({}) == 0


def test_process_outbound_email_never_downgrades_stage(crm, make_test_memory):
    """
    If an account is in 'Negotiation', a new 'Prospecting' email shouldn't
    downgrade the opportunity stage, but should still log the touchpoint.
    """

    crm.process_outbound_email(
        {"recipient_org": "Acme", "stage": "Negotiation/Review", "subject": "Contract"},
        "2026-03-08T10:00:00Z",
        "2026-03-08",
        1,
    )

    crm.process_outbound_email(
        {"recipient_org": "Acme", "stage": "Prospecting", "subject": "Checking in"},
        "2026-03-08T10:00:00Z",
        "2026-03-08",
        1,
    )

    opp = make_test_memory._db["sf_opps"].find_one({"account_name": "Acme"})
    assert opp["stage"] == "Negotiation/Review", "Stage was improperly downgraded"
    assert len(opp["touchpoints"]) == 2, "Touchpoint was not recorded"


def test_planner_context_enforces_token_limits(crm, make_test_memory):
    """Ensure the context string doesn't blow up the LLM prompt if there are 100 open tickets."""

    for i in range(10):
        crm.handle_inbound_complaint(
            {"subject": f"Issue {i}", "sender_org": "Org"}, "T0", "D0", 1
        )

    context = crm.planner_context()

    assert "OPEN SUPPORT TICKETS (10)" in context
    assert context.count("ZD-") == 5, "Context must cap at 5 displayed tickets"
    assert "... and 5 more." in context


def test_incident_resolved_closes_linked_zd_tickets(crm, make_test_memory):
    """When an incident is resolved, linked ZD tickets must transition to 'Solved'."""
    t_id = crm.handle_inbound_complaint(
        {"subject": "Down", "sender_org": "Org"}, "T0", "D0", 1
    )
    crm.handle_incident_opened("ENG-1", "DB", 40, "T1", "D0", 1)  # Links the ticket

    crm.handle_incident_resolved("ENG-1", "http://postmortem", "T2", "D0", 1)

    ticket = make_test_memory._db["zd_tickets"].find_one({"ticket_id": t_id})
    assert ticket["status"] == "Solved"
    assert "resolved" in ticket["comments"][-1]["text"].lower()
