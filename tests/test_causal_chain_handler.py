"""
test_causal_chain_handler.py
============================
Unit tests for causal chain construction and hybrid Recurrence Detection.
"""

import pytest
from unittest.mock import patch
from causal_chain_handler import (
    CausalChainHandler,
    RecurrenceDetector,
    RecurrenceMatchStore,
)
from memory import SimEvent


def test_causal_chain_handler_deduplicates_and_orders():
    """Causal chains must be append-only, ordered, and strictly deduplicated."""
    chain = CausalChainHandler("ENG-100")

    chain.append("slack_thread_01")
    chain.append("PR-202")
    chain.append("slack_thread_01")
    chain.append("CONF-99")

    snap = chain.snapshot()
    assert len(snap) == 4
    assert snap == ["ENG-100", "slack_thread_01", "PR-202", "CONF-99"]
    assert chain.root == "ENG-100"


@pytest.fixture
def detector(make_test_memory):
    return RecurrenceDetector(mem=make_test_memory)


def _mock_sim_event(jira_id: str, day: int) -> SimEvent:
    return SimEvent(
        type="incident_opened",
        day=day,
        date="2026-01-01",
        timestamp="T0",
        actors=[],
        artifact_ids={"jira": jira_id},
        facts={"root_cause": "test"},
        summary="",
    )
