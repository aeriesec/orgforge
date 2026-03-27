"""
test_post_sim_artifacts.py
==========================
Unit tests for post-simulation artifact generation.
Validates SLA credits, NPS scoring math, and Datadog interpolations.
"""

from unittest.mock import patch

import pytest
from datetime import datetime
from memory import SimEvent
from post_sim_artifacts import DatadogWriter, EventIndex, NPSWriter, InvoiceWriter


@pytest.fixture
def sim_start():
    return datetime(2026, 1, 1)


@pytest.fixture
def event_index(sim_start):
    events = [
        SimEvent(
            type="crm_touchpoint",
            day=1,
            date="2026-01-01",
            timestamp="T00",
            actors=[],
            artifact_ids={},
            facts={"account_name": "Acme Corp"},
            summary="",
        ),
        SimEvent(
            type="zd_ticket_opened",
            day=1,
            date="2026-01-01",
            timestamp="T00",
            actors=[],
            artifact_ids={},
            facts={"ticket_id": "ZD-1", "org_name": "Acme Corp"},
            summary="",
        ),
        SimEvent(
            type="incident_opened",
            day=2,
            date="2026-01-02",
            timestamp="T01",
            actors=[],
            artifact_ids={"jira": "ENG-100"},
            facts={"root_cause": "OOM"},
            summary="",
        ),
        SimEvent(
            type="zd_tickets_escalated",
            day=2,
            date="2026-01-02",
            timestamp="T01",
            actors=[],
            artifact_ids={},
            facts={"ticket_ids": ["ZD-1"], "incident_id": "ENG-100"},
            summary="",
        ),
        SimEvent(
            type="sf_deals_risk_flagged",
            day=2,
            date="2026-01-02",
            timestamp="T01",
            actors=[],
            artifact_ids={},
            facts={"account_names": ["Acme Corp"], "incident_id": "ENG-100"},
            summary="",
        ),
        SimEvent(
            type="incident_resolved",
            day=5,
            date="2026-01-05",
            timestamp="T02",
            actors=[],
            artifact_ids={"jira": "ENG-100"},
            facts={},
            summary="",
        ),
    ]
    return EventIndex(events, sim_start)


def test_nps_scoring_logic_penalizes_escalations_and_breaches(
    event_index, tmp_path, sim_start
):
    writer = NPSWriter(event_index, tmp_path, sim_start, sim_end_day=10)
    score, detail = writer._score("Acme Corp")

    assert detail["escalated_tickets"] == 1
    assert detail["unresolved_tickets"] == 1
    assert detail["sla_breach_days"] == 2
    assert score == 2
    assert writer._classify(score) == "detractor"


def test_invoice_writer_applies_sla_credits(
    event_index, tmp_path, sim_start, make_test_memory
):
    writer = InvoiceWriter(
        event_index, tmp_path, sim_start, sim_end_day=30, mem=make_test_memory
    )

    invoices = writer.build_invoices()
    assert len(invoices) == 1
    acme_inv = invoices[0]

    assert acme_inv["customer"]["org_name"] == "Acme Corp"

    line_items = acme_inv["line_items"]
    sla_lines = [item for item in line_items if item["line_item_type"] == "sla_credit"]

    assert len(sla_lines) == 1
    assert sla_lines[0]["incident_id"] == "ENG-100"
    assert sla_lines[0]["breach_days"] == 2
    assert sla_lines[0]["amount"] < 0  # Credits must be negative


def test_event_index_links_incidents_to_customers_via_sf_flags(event_index):

    assert "ENG-100" in event_index.customer_risk_flags["Acme Corp"]
    assert event_index.incidents["ENG-100"]["duration_days"] == 3


def test_invoice_writer_exact_sla_threshold_yields_no_credit(
    tmp_path, make_test_memory
):
    """If the threshold is 1 day, a 1-day incident should result in 0 credits."""
    from post_sim_artifacts import SLA_BREACH_THRESHOLD_DAYS

    events = [
        SimEvent(
            type="crm_touchpoint",
            day=1,
            date="D1",
            timestamp="T1",
            actors=[],
            artifact_ids={},
            facts={"account_name": "Acme"},
            summary="",
        ),
        SimEvent(
            type="sf_deals_risk_flagged",
            day=1,
            date="D1",
            timestamp="T1",
            actors=[],
            artifact_ids={},
            facts={"account_names": ["Acme"], "incident_id": "ENG-1"},
            summary="",
        ),
        # Incident lasts EXACTLY the threshold
        SimEvent(
            type="incident_opened",
            day=1,
            date="D1",
            timestamp="T1",
            actors=[],
            artifact_ids={"jira": "ENG-1"},
            facts={"root_cause": "OOM"},
            summary="",
        ),
        SimEvent(
            type="incident_resolved",
            day=1 + SLA_BREACH_THRESHOLD_DAYS,
            date="D2",
            timestamp="T2",
            actors=[],
            artifact_ids={"jira": "ENG-1"},
            facts={},
            summary="",
        ),
    ]
    idx = EventIndex(events, datetime(2026, 1, 1))
    writer = InvoiceWriter(idx, tmp_path, datetime(2026, 1, 1), 30, make_test_memory)

    invoices = writer.build_invoices()
    acme_inv = invoices[0]

    sla_lines = [
        item
        for item in acme_inv["line_items"]
        if item["line_item_type"] == "sla_credit"
    ]
    assert len(sla_lines) == 0, "Exact threshold duration should not trigger a credit"


def test_datadog_health_interpolation_bounds(tmp_path):
    """Health must strictly bound between the floor (15) and ceiling (100) during incidents."""
    events = [
        SimEvent(
            type="day_summary",
            day=1,
            date="D1",
            timestamp="T1",
            actors=[],
            artifact_ids={},
            facts={"system_health": 100},
            summary="",
        ),
        SimEvent(
            type="incident_opened",
            day=2,
            date="D2",
            timestamp="T2",
            actors=[],
            artifact_ids={"jira": "ENG-1"},
            facts={"root_cause": "OOM"},
            summary="",
        ),
        SimEvent(
            type="incident_resolved",
            day=4,
            date="D4",
            timestamp="T4",
            actors=[],
            artifact_ids={"jira": "ENG-1"},
            facts={},
            summary="",
        ),
    ]
    idx = EventIndex(events, datetime(2026, 1, 1))
    writer = DatadogWriter(idx, tmp_path, datetime(2026, 1, 1), 5)

    health_d3 = writer._health_at(day=3, minute_offset=120)
    assert 12 <= health_d3 <= 20, (
        f"Health {health_d3} did not hit the expected floor near 15"
    )

    health_d5 = writer._health_at(day=5, minute_offset=0)
    assert health_d5 > 80, "Health did not recover after incident resolution"


@patch("post_sim_artifacts._write_json")
def test_post_sim_survives_empty_simulation(
    mock_write_json, tmp_path, make_test_memory
):
    """If the sim did nothing (no events), post_sim should output empty summaries gracefully."""
    idx = EventIndex([], datetime(2026, 1, 1))

    nps_writer = NPSWriter(idx, tmp_path, datetime(2026, 1, 1), 5)
    responses = nps_writer.build_responses()
    assert len(responses) == 0
    nps_writer.write(responses)  # Should write a summary.json with 0s

    assert mock_write_json.called

    summary_data = mock_write_json.call_args[0][1]
    assert summary_data["response_count"] == 0
