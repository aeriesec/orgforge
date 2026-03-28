"""
test_causal_chain_handler.py
============================
Unit tests for causal chain construction and hybrid Recurrence Detection.
"""

import pytest
from unittest.mock import patch, MagicMock
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


@patch.object(RecurrenceDetector, "_text_search")
@patch.object(RecurrenceDetector, "_vector_search")
def test_recurrence_detector_rejects_below_thresholds(
    mock_vec, mock_text, detector, make_test_memory
):
    """
    If the best match falls below BOTH the text threshold (0.40) and the
    vector threshold (0.72), it must be rejected to prevent false positives.
    """

    mock_text.return_value = [{"score": 2.0}]  # Normalises to ~0.35 (below 0.40)
    mock_vec.return_value = [(_mock_sim_event("ENG-10", 1), 0.65)]  # Below 0.72

    match = detector.find_prior_incident(
        "weak issue", current_day=5, current_ticket_id="ENG-20"
    )

    assert match is None

    store = make_test_memory._db[RecurrenceMatchStore.COLLECTION]
    record = store.find_one({"current_ticket_id": "ENG-20"})
    assert record is not None
    assert record["matched"] is False
    assert record["confidence"] == "rejected"


@patch.object(RecurrenceDetector, "_text_search")
@patch.object(RecurrenceDetector, "_vector_search")
def test_recurrence_detector_accepts_strong_vector_only_match(
    mock_vec, mock_text, detector, make_test_memory
):
    """
    If the text search yields nothing (e.g. completely paraphrased root cause),
    but the vector similarity is very high, it must confidently match.
    """
    mock_text.return_value = []
    expected_match = _mock_sim_event("ENG-11", 2)
    mock_vec.return_value = [(expected_match, 0.85)]  # Well above 0.72

    match = detector.find_prior_incident(
        "paraphrased issue", current_day=5, current_ticket_id="ENG-21"
    )

    assert match is not None
    assert match.artifact_ids["jira"] == "ENG-11"

    record = make_test_memory._db[RecurrenceMatchStore.COLLECTION].find_one(
        {"current_ticket_id": "ENG-21"}
    )
    assert record["fusion_strategy"] == "vector_only"
    assert record["matched"] is True


@patch.object(RecurrenceDetector, "_text_search")
@patch.object(RecurrenceDetector, "_vector_search")
def test_rrf_fusion_favors_earliest_incident_on_tie(mock_vec, mock_text, detector):
    """
    If multiple incidents breach the threshold with identical scores (e.g., a recurring
    issue that has happened 3 times), the detector MUST return the earliest one
    to anchor the recurrence depth properly and avoid daisy-chaining.
    """

    match_day_1 = _mock_sim_event("ENG-01", 1)
    match_day_3 = _mock_sim_event("ENG-03", 3)
    match_day_5 = _mock_sim_event("ENG-05", 5)

    mock_text.return_value = []

    mock_vec.return_value = [
        (match_day_5, 0.90),
        (match_day_3, 0.90),
        (match_day_1, 0.90),
    ]

    match = detector.find_prior_incident(
        "recurring issue", current_day=10, current_ticket_id="ENG-99"
    )

    assert match.artifact_ids["jira"] == "ENG-01"
