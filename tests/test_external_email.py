import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime

from external_email_ingest import ExternalEmailIngestor, ExternalEmailSignal
from flow import ActiveIncident, State
from causal_chain_handler import CausalChainHandler


@pytest.fixture
def ingestor(make_test_memory):
    """Fixture providing a wired-up ExternalEmailIngestor with mocked LLMs."""
    org_chart = {"Engineering": ["Alice", "Bob", "Charlie"], "Sales": ["Diana"]}
    leads = {"Engineering": "Alice", "Sales": "Diana"}
    personas = {
        "Alice": {"expertise": ["backend", "database"]},
        "Bob": {"expertise": ["frontend", "react"]},
        "Charlie": {"expertise": ["aws", "infrastructure", "redis"]},
        "Diana": {"expertise": ["sales"]},
    }

    # Needs a scheduled hire for the HR test
    config = {
        "simulation": {"company_name": "TestCorp"},
        "org_lifecycle": {
            "scheduled_hires": [
                {"name": "Taylor", "day": 5, "dept": "Engineering", "role": "Engineer"}
            ]
        },
    }

    return ExternalEmailIngestor(
        config=config,
        mem=make_test_memory,
        worker_llm=MagicMock(),
        planner_llm=MagicMock(),
        export_dir="/tmp/export",
        leads=leads,
        org_chart=org_chart,
        personas=personas,
        registry=MagicMock(),
        clock=MagicMock(),
    )


@pytest.fixture
def mock_state():
    state = State()
    state.current_date = datetime(2026, 1, 1, 9, 0, 0)
    state.system_health = 100
    return state


### --- Tests ---


def test_find_expert_for_topic(ingestor):
    """
    Verifies that vendor emails are routed to the team member whose expertise
    tags best overlap with the email's topic, not just the department lead.
    """
    # Overlaps with Bob's "react" and "frontend"
    bob_topic = "Urgent: React frontend components are failing to render"
    assert ingestor._find_expert_for_topic(bob_topic, "Engineering") == "Bob"

    # Overlaps with Charlie's "aws" and "infrastructure"
    charlie_topic = "AWS infrastructure quota limit reached"
    assert ingestor._find_expert_for_topic(charlie_topic, "Engineering") == "Charlie"

    # Overlaps with nobody in particular, should default to the lead (Alice)
    unknown_topic = "General inquiry about your software"
    assert ingestor._find_expert_for_topic(unknown_topic, "Engineering") == "Alice"


def test_hr_outbound_window(ingestor, mock_state):
    """
    Verifies that HR outbound emails only fire 1-3 days before the hire's arrival date.
    """
    # Spy on the actual sending method
    ingestor._send_hr_outbound = MagicMock()

    # Taylor is scheduled for Day 5.

    # Day 1: 4 days out. Too early.
    mock_state.day = 1
    ingestor.generate_hr_outbound(mock_state)
    assert not ingestor._send_hr_outbound.called

    # Day 2: 3 days out. Should trigger (Offer Letter).
    mock_state.day = 2
    ingestor.generate_hr_outbound(mock_state)
    assert ingestor._send_hr_outbound.call_count == 1

    # Reset mock and internal flag
    ingestor._send_hr_outbound.reset_mock()
    ingestor._scheduled_hires[5][0]["_hr_email_sent"] = False

    # Day 5: Day of arrival. Too late (0 days out).
    mock_state.day = 5
    ingestor.generate_hr_outbound(mock_state)
    assert not ingestor._send_hr_outbound.called


def test_vendor_email_appends_to_active_incident(ingestor, mock_state):
    """
    Verifies that a vendor email whose topic overlaps with an active incident's
    root cause is appended to that incident's causal chain.
    """
    # Create an active incident
    chain_handler = CausalChainHandler("ORG-999")
    inc = ActiveIncident(
        ticket_id="ORG-999",
        title="DB Outage",
        day_started=1,
        root_cause="redis cache eviction failed due to OOM",
        causal_chain=chain_handler,
    )
    mock_state.active_incidents = [inc]

    # Create a vendor signal that mentions "redis"
    signal = ExternalEmailSignal(
        source_name="AWS",
        source_org="AWS",
        source_email="aws@amazon.com",
        internal_liaison="Engineering",
        subject="ElastiCache Alert",
        body_preview="Redis memory usage critical",
        full_body="...",
        tone="urgent",
        topic="redis memory exhaustion",
        timestamp_iso="2026-01-01T08:00:00Z",
        embed_id="email_aws_123",
        category="vendor",
        causal_chain=CausalChainHandler("email_aws_123"),
    )

    ingestor._engineer_opens_jira = MagicMock(return_value=None)
    ingestor._send_vendor_ack = MagicMock(return_value="mock_ack_id")
    ingestor._route_vendor_email(signal, mock_state)

    # The vendor email's embed_id should now be in the incident's causal chain
    assert "email_aws_123" in inc.causal_chain.snapshot()


@patch("external_email_ingest.random.random")
def test_customer_email_dropped_probability(mock_random, ingestor, mock_state):
    """
    Verifies that customer emails are dropped and logged correctly when they
    fall within the 15% drop probability window.
    """
    # Force random to return 0.10 (which is < _PROB_EMAIL_DROPPED's 0.15)
    mock_random.return_value = 0.10

    # Setup a dummy source so it generates one email
    ingestor._sources = [
        {
            "name": "Acme Corp",
            "category": "customer",
            "trigger_on": ["always"],
            "topics": ["complaint"],
        }
    ]

    # Mock out the LLM generation so it returns our dummy signal
    dummy_signal = ExternalEmailSignal(
        source_name="Acme",
        source_org="Acme",
        source_email="contact@acme.com",
        internal_liaison="Sales",
        subject="Help",
        body_preview="...",
        full_body="...",
        tone="frustrated",
        topic="bug",
        timestamp_iso="2026-01-01T10:00:00Z",
        embed_id="email_acme_1",
        category="customer",
        causal_chain=CausalChainHandler("email_acme_1"),
    )
    ingestor._generate_email = MagicMock(return_value=dummy_signal)

    # Spy on the specific routing and dropping methods
    ingestor._route_customer_email = MagicMock()
    ingestor._log_dropped_email = MagicMock()

    signals = ingestor.generate_business_hours(mock_state)

    assert len(signals) == 1
    assert signals[0].dropped is True
    assert ingestor._log_dropped_email.called
    assert not ingestor._route_customer_email.called
