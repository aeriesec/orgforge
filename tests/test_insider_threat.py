import io
import json
import os
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from insider_threat import InsiderThreatInjector, _NullInjector


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def base_config():
    """Standard enabled config with one active subject."""
    return {
        "simulation": {"domain": "testcorp.com"},
        "insider_threat": {
            "enabled": True,
            "mode": "passive",
            "dlp_noise_ratio": 0.0,
            "telemetry_dir": "security_telemetry",
            "subjects": [
                {
                    "name": "Mallory",
                    "threat_class": "malicious",
                    "onset_day": 3,
                    "behaviors": [
                        "secret_in_commit",
                        "unusual_hours_access",
                        "sentiment_drift",
                        "data_exfil_email",
                    ],
                }
            ],
        },
    }


@pytest.fixture
def injector(base_config, tmp_path):
    """Returns a fully configured InsiderThreatInjector."""
    all_names = ["Alice", "Bob", "Mallory"]
    return InsiderThreatInjector.from_config(base_config, tmp_path, all_names)


# ─────────────────────────────────────────────────────────────────────────────
# 1. FACTORY & NULL OBJECT
# ─────────────────────────────────────────────────────────────────────────────


def test_factory_returns_null_injector_when_disabled(base_config, tmp_path):
    """When enabled: false, the factory must return a _NullInjector."""
    base_config["insider_threat"]["enabled"] = False
    inj = InsiderThreatInjector.from_config(base_config, tmp_path, ["Alice"])
    assert isinstance(inj, _NullInjector)


def test_null_injector_is_safe_noop():
    """The _NullInjector must safely absorb all API calls without crashing."""
    inj = _NullInjector()

    # Must return original data unmodified
    assert inj.inject_pr({"desc": "test"}, "Bob", 1) == {"desc": "test"}
    assert inj.inject_slack([{"msg": "hi"}], "general", 1, datetime.now()) == [
        {"msg": "hi"}
    ]

    # Must return expected empty/None types
    assert inj.end_day(1, None, None, None, "2026-01-01") == []
    assert inj.inject_email("path", "Bob", [], "Subj", 1, datetime.now()) is None
    assert inj.is_active("Bob", "anything", 1) is False


# ─────────────────────────────────────────────────────────────────────────────
# 2. LIFECYCLE & TEMPORAL ONSET
# ─────────────────────────────────────────────────────────────────────────────


def test_subject_inactive_before_onset_day(injector):
    """Behaviors must not fire before the subject's onset_day."""
    injector.begin_day(day=2, state=None)

    # Mallory's onset_day is 3
    assert not injector.is_active("Mallory", "secret_in_commit", 2)
    assert "Mallory" not in injector.active_subject_names()


def test_subject_becomes_active_on_onset_day(injector):
    """Subject activates exactly on their onset_day."""
    injector.begin_day(day=3, state=None)

    assert injector.is_active("Mallory", "secret_in_commit", 3)
    assert "Mallory" in injector.active_subject_names()


# ─────────────────────────────────────────────────────────────────────────────
# 3. PR INJECTION (Secret in Commit)
# ─────────────────────────────────────────────────────────────────────────────


def test_inject_pr_mutates_description_and_logs_telemetry(injector):
    """
    secret_in_commit must append a fake credential to the PR description
    and queue a telemetry record.
    """
    injector.begin_day(3, None)

    base_pr = {"pr_id": "PR-100", "description": "Fix login bug."}

    mutated_pr = injector.inject_pr(base_pr.copy(), "Mallory", day=3)

    # Description should be longer due to injected secret
    assert len(mutated_pr["description"]) > len(base_pr["description"])
    assert "Fix login bug." in mutated_pr["description"]

    # Telemetry should be queued
    assert len(injector._pending_telemetry) == 1
    telemetry = injector._pending_telemetry[0]
    assert telemetry.actor == "Mallory"
    assert telemetry.record_type == "commit"
    assert telemetry._true_positive is True


def test_inject_pr_cooldown_prevents_spam(injector):
    """The same behavior must not fire twice within its cooldown window."""
    injector.begin_day(3, None)

    # First injection (Day 3)
    injector.inject_pr({"description": "First PR"}, "Mallory", day=3)
    assert len(injector._pending_telemetry) == 1

    # Second injection (Day 4) - should be blocked by the 4-day cooldown
    pr_2 = {"description": "Second PR"}
    mutated_2 = injector.inject_pr(pr_2.copy(), "Mallory", day=4)

    assert mutated_2 == pr_2  # No mutation
    assert len(injector._pending_telemetry) == 1  # No new telemetry


# ─────────────────────────────────────────────────────────────────────────────
# 4. SLACK INJECTION
# ─────────────────────────────────────────────────────────────────────────────


def test_inject_slack_unusual_hours_appends_message(injector):
    """
    unusual_hours_access must append a new message with an off-hours
    timestamp and the _security_injected flag.
    """
    injector.begin_day(3, None)
    messages = [{"user": "Alice", "text": "Morning!"}]
    current_date = datetime(2026, 1, 3, 10, 0, 0)  # 10:00 AM

    # Force the 0.35 probability check to pass
    with patch("random.random", return_value=0.1):
        result = injector.inject_slack(messages, "general", 3, current_date)

    assert len(result) == 2
    injected_msg = result[1]

    assert injected_msg["user"] == "Mallory"
    assert injected_msg["_security_injected"] is True

    # Verify the timestamp was shifted to off-hours (e.g., 1am, 2am, 23pm)
    injected_ts = datetime.fromisoformat(injected_msg["ts"])
    assert injected_ts.hour < 5 or injected_ts.hour > 20


def test_inject_slack_sentiment_drift_mutates_existing_message(injector):
    """
    sentiment_drift must modify the subject's existing message in the
    channel, leaving other messages alone.
    """
    injector.begin_day(3, None)
    messages = [
        {"user": "Alice", "text": "Looks good to me."},
        {"user": "Mallory", "text": "I will deploy it now."},
    ]

    result = injector.inject_slack(messages, "backend", 3, datetime.now())

    assert len(result) == 2
    assert result[0]["text"] == "Looks good to me."  # Alice untouched

    # Mallory's message should have malicious drift (prefix/suffix wrappers)
    # The malicious wrapper might be empty, so we test the telemetry side-effect
    assert len(injector._pending_telemetry) == 1
    assert injector._pending_telemetry[0]._behavior == "sentiment_drift"


# ─────────────────────────────────────────────────────────────────────────────
# 5. EMAIL INJECTION (Data Exfil)
# ─────────────────────────────────────────────────────────────────────────────


def test_inject_email_writes_exfil_file(injector, tmp_path):
    """
    data_exfil_email must write a separate .eml file representing a forward
    to a personal email address, and return the path to it.
    """
    injector.begin_day(3, None)

    # Create a dummy outbound email
    eml_dir = tmp_path / "emails"
    eml_dir.mkdir()
    original_eml_path = str(eml_dir / "outbound.eml")

    current_date = datetime(2026, 1, 3, 14, 0, 0)

    # Force the 0.5 probability check to pass, AND bypass the conftest mock
    with patch("random.random", return_value=0.1), patch("builtins.open", io.open):
        exfil_path = injector.inject_email(
            eml_path=original_eml_path,
            sender="Mallory",
            recipients=["client@corp.com"],
            subject_line="Project Specs",
            day=3,
            current_date=current_date,
        )

    assert exfil_path is not None
    assert exfil_path.endswith("_fwd_3.eml")
    assert os.path.exists(exfil_path)

    # Read the generated file using io.open to bypass the mock again
    with io.open(exfil_path, "r") as f:
        content = f.read()
        assert "mallory.personal@" in content


# ─────────────────────────────────────────────────────────────────────────────
# 6. TELEMETRY FLUSHING & SEPARATION
# ─────────────────────────────────────────────────────────────────────────────


def test_telemetry_flushed_to_separate_files_at_end_day(injector, tmp_path):
    """
    end_day must flush pending telemetry to access_log.jsonl and _ground_truth.jsonl.
    The observable log must NOT contain the _true_positive fields.
    """
    injector.begin_day(3, None)

    # Trigger an injection to queue telemetry
    injector.inject_pr({"description": "Test PR"}, "Mallory", 3)
    assert len(injector._pending_telemetry) == 1

    # Flush, bypassing the conftest mock so it writes to tmp_path
    with patch("builtins.open", io.open):
        injector.end_day(3, None, None, None, "2026-01-03")

    assert len(injector._pending_telemetry) == 0

    telemetry_dir = tmp_path / "security_telemetry"
    obs_path = telemetry_dir / "access_log.jsonl"
    gt_path = telemetry_dir / "_ground_truth.jsonl"

    assert obs_path.exists()
    assert gt_path.exists()

    # Read observable log
    with io.open(obs_path) as f:
        obs_data = json.loads(f.readline())
        assert "actor" in obs_data
        assert "secret_var" in obs_data
        assert "true_positive" not in obs_data  # MUST NOT LEAK
        assert "behavior" not in obs_data  # MUST NOT LEAK

    # Read ground truth log
    with io.open(gt_path) as f:
        gt_data = json.loads(f.readline())
        assert "actor" in gt_data
        assert "true_positive" in gt_data
        assert gt_data["true_positive"] is True
        assert gt_data["behavior"] == "secret_in_commit"
