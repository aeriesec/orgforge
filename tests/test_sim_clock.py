import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from sim_clock import SimClock, DAY_START_HOUR, DAY_START_MINUTE, DAY_END_HOUR, DAY_END_MINUTE


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_state():
    """Minimal stand-in for the simulation State object."""
    state = MagicMock()
    state.current_date = datetime(2026, 3, 10, 0, 0, 0)  # Monday
    state.actor_cursors = {}
    return state


@pytest.fixture
def clock(mock_state):
    """SimClock wired to the mock state."""
    return SimClock(mock_state)


# ─────────────────────────────────────────────────────────────────────────────
# 1. RESET TO BUSINESS START
# ─────────────────────────────────────────────────────────────────────────────

def test_reset_initialises_all_actors(clock, mock_state):
    """reset_to_business_start must set every actor and 'system' to 09:00 today."""
    actors = ["Alice", "Bob", "Carol"]
    clock.reset_to_business_start(actors)

    expected = mock_state.current_date.replace(
        hour=DAY_START_HOUR, minute=DAY_START_MINUTE, second=0, microsecond=0
    )
    for actor in actors:
        assert mock_state.actor_cursors[actor] == expected

    assert mock_state.actor_cursors["system"] == expected


def test_reset_overwrites_existing_cursors(clock, mock_state):
    """reset_to_business_start must overwrite previously advanced cursors."""
    mock_state.actor_cursors["Alice"] = datetime(2026, 3, 10, 15, 30, 0)
    clock.reset_to_business_start(["Alice"])

    expected_start = mock_state.current_date.replace(
        hour=DAY_START_HOUR, minute=DAY_START_MINUTE, second=0, microsecond=0
    )
    assert mock_state.actor_cursors["Alice"] == expected_start


# ─────────────────────────────────────────────────────────────────────────────
# 2. ADVANCE ACTOR
# ─────────────────────────────────────────────────────────────────────────────

def test_advance_actor_moves_cursor_forward(clock, mock_state):
    """advance_actor must push the cursor forward by the given hours."""
    clock.reset_to_business_start(["Alice"])
    start = mock_state.actor_cursors["Alice"]

    _, new_cursor = clock.advance_actor("Alice", 2.0)

    assert new_cursor > start
    assert new_cursor == start + timedelta(hours=2.0)


def test_advance_actor_artifact_timestamp_within_block(clock, mock_state):
    """The returned artifact timestamp must fall between the old and new cursor."""
    clock.reset_to_business_start(["Alice"])
    start = mock_state.actor_cursors["Alice"]

    artifact_time, new_cursor = clock.advance_actor("Alice", 3.0)

    assert start <= artifact_time <= new_cursor


def test_advance_actor_does_not_affect_other_actors(clock, mock_state):
    """Advancing one actor must leave all other cursors unchanged."""
    clock.reset_to_business_start(["Alice", "Bob"])
    bob_before = mock_state.actor_cursors["Bob"]

    clock.advance_actor("Alice", 2.0)

    assert mock_state.actor_cursors["Bob"] == bob_before


def test_advance_actor_rolls_over_to_next_business_day(clock, mock_state):
    """An advance that pushes past 17:30 must land at 09:00 the next business day."""
    # Put Alice at 16:30 — 2 hours will spill past EOD
    mock_state.actor_cursors["Alice"] = datetime(2026, 3, 10, 16, 30, 0)

    _, new_cursor = clock.advance_actor("Alice", 2.0)

    assert new_cursor.hour == DAY_START_HOUR
    assert new_cursor.minute == DAY_START_MINUTE
    assert new_cursor.date() > datetime(2026, 3, 10).date()


# ─────────────────────────────────────────────────────────────────────────────
# 3. SYNC AND TICK
# ─────────────────────────────────────────────────────────────────────────────

def test_sync_and_tick_brings_laggard_forward(clock, mock_state):
    """sync_and_tick must pull slower actors up to the max cursor before ticking."""
    mock_state.actor_cursors["Alice"] = datetime(2026, 3, 10, 11, 0, 0)
    mock_state.actor_cursors["Bob"]   = datetime(2026, 3, 10, 9,  0, 0)  # behind

    result = clock.sync_and_tick(["Alice", "Bob"], min_mins=5, max_mins=5)

    # Both must be at or past 11:00 + 5 min
    assert mock_state.actor_cursors["Alice"] == result
    assert mock_state.actor_cursors["Bob"]   == result
    assert result >= datetime(2026, 3, 10, 11, 5, 0)


def test_sync_and_tick_result_advances_beyond_sync_point(clock, mock_state):
    """The returned time must be strictly after the maximum pre-sync cursor."""
    mock_state.actor_cursors["Alice"] = datetime(2026, 3, 10, 10, 0, 0)
    mock_state.actor_cursors["Bob"]   = datetime(2026, 3, 10, 10, 0, 0)

    result = clock.sync_and_tick(["Alice", "Bob"], min_mins=5, max_mins=5)

    assert result > datetime(2026, 3, 10, 10, 0, 0)


def test_sync_and_tick_enforces_business_hours_by_default(clock, mock_state):
    """Without allow_after_hours, sync_and_tick must not land past 17:30."""
    # Put actors near end-of-day so the tick pushes them over
    mock_state.actor_cursors["Alice"] = datetime(2026, 3, 10, 17, 25, 0)
    mock_state.actor_cursors["Bob"]   = datetime(2026, 3, 10, 17, 25, 0)

    result = clock.sync_and_tick(["Alice", "Bob"], min_mins=10, max_mins=10)

    eod = datetime(2026, 3, 10, DAY_END_HOUR, DAY_END_MINUTE, 0)
    assert result != datetime(2026, 3, 10, 17, 35, 0), "Should not land after EOD"
    # Must roll to the next business day start
    assert result.hour == DAY_START_HOUR and result.minute == DAY_START_MINUTE


# ─────────────────────────────────────────────────────────────────────────────
# 4. TICK MESSAGE
# ─────────────────────────────────────────────────────────────────────────────

def test_tick_message_incident_cadence_is_faster_than_async(clock, mock_state):
    """
    Over many samples, incident cadence ticks should produce shorter average
    gaps than async cadence ticks, matching the CADENCE_RANGES config.
    """
    import statistics

    results_incident = []
    results_async    = []

    for _ in range(30):
        # Reset to a clean starting point each time
        mock_state.actor_cursors = {
            "Alice": datetime(2026, 3, 10, 10, 0, 0),
            "Bob":   datetime(2026, 3, 10, 10, 0, 0),
        }
        t = clock.tick_message(["Alice", "Bob"], cadence="incident")
        results_incident.append((t - datetime(2026, 3, 10, 10, 0, 0)).total_seconds())

        mock_state.actor_cursors = {
            "Alice": datetime(2026, 3, 10, 10, 0, 0),
            "Bob":   datetime(2026, 3, 10, 10, 0, 0),
        }
        t = clock.tick_message(["Alice", "Bob"], cadence="async")
        results_async.append((t - datetime(2026, 3, 10, 10, 0, 0)).total_seconds())

    assert statistics.mean(results_incident) < statistics.mean(results_async)


def test_tick_message_unknown_cadence_falls_back_to_normal(clock, mock_state):
    """tick_message with an unrecognised cadence must not raise; normal range applies."""
    mock_state.actor_cursors["Alice"] = datetime(2026, 3, 10, 10, 0, 0)
    mock_state.actor_cursors["Bob"]   = datetime(2026, 3, 10, 10, 0, 0)

    try:
        result = clock.tick_message(["Alice", "Bob"], cadence="made_up_cadence")
    except Exception as e:
        pytest.fail(f"tick_message raised unexpectedly with unknown cadence: {e}")

    assert result > datetime(2026, 3, 10, 10, 0, 0)


# ─────────────────────────────────────────────────────────────────────────────
# 5. TICK SYSTEM
# ─────────────────────────────────────────────────────────────────────────────

def test_tick_system_advances_only_system_cursor(clock, mock_state):
    """tick_system must only move the 'system' cursor, leaving human actors alone."""
    clock.reset_to_business_start(["Alice", "Bob"])
    alice_before = mock_state.actor_cursors["Alice"]

    clock.tick_system(min_mins=3, max_mins=3)

    assert mock_state.actor_cursors["Alice"] == alice_before
    assert mock_state.actor_cursors["system"] > datetime(2026, 3, 10, 9, 0, 0)


def test_tick_system_moves_forward_in_time(clock, mock_state):
    """Each tick_system call must produce a strictly later timestamp."""
    clock.reset_to_business_start(["Alice"])

    t1 = clock.tick_system(min_mins=2, max_mins=2)
    t2 = clock.tick_system(min_mins=2, max_mins=2)

    assert t2 > t1


# ─────────────────────────────────────────────────────────────────────────────
# 6. AT (PINNED MEETING)
# ─────────────────────────────────────────────────────────────────────────────

def test_at_stamps_artifact_at_exact_scheduled_time(clock, mock_state):
    """clock.at must return precisely the scheduled start time for artifact stamping."""
    clock.reset_to_business_start(["Alice", "Bob"])

    result = clock.at(["Alice", "Bob"], hour=10, minute=30, duration_mins=60)

    assert result == datetime(2026, 3, 10, 10, 30, 0)


def test_at_advances_cursors_to_meeting_end(clock, mock_state):
    """After clock.at, all actors' cursors must sit at the meeting's end time."""
    clock.reset_to_business_start(["Alice", "Bob"])

    clock.at(["Alice", "Bob"], hour=11, minute=0, duration_mins=45)
    expected_end = datetime(2026, 3, 10, 11, 45, 0)

    assert mock_state.actor_cursors["Alice"] == expected_end
    assert mock_state.actor_cursors["Bob"]   == expected_end


def test_at_does_not_time_travel_actors_already_past_meeting(clock, mock_state):
    """Actors already past the meeting end must not be rewound."""
    future_time = datetime(2026, 3, 10, 14, 0, 0)
    mock_state.actor_cursors["Alice"] = future_time
    mock_state.actor_cursors["Bob"]   = datetime(2026, 3, 10, 9, 0, 0)

    clock.at(["Alice", "Bob"], hour=10, minute=0, duration_mins=60)

    # Alice was already past 11:00, so she must stay at (or past) her original time
    assert mock_state.actor_cursors["Alice"] >= future_time


# ─────────────────────────────────────────────────────────────────────────────
# 7. NOW
# ─────────────────────────────────────────────────────────────────────────────

def test_now_returns_cursor_without_advancing(clock, mock_state):
    """clock.now must read an actor's current time without side-effects."""
    mock_state.actor_cursors["Alice"] = datetime(2026, 3, 10, 11, 0, 0)

    t = clock.now("Alice")

    assert t == datetime(2026, 3, 10, 11, 0, 0)
    assert mock_state.actor_cursors["Alice"] == datetime(2026, 3, 10, 11, 0, 0)


def test_now_defaults_unknown_actor_to_business_start(clock, mock_state):
    """clock.now for an actor not yet in the cursor map must return 09:00 today."""
    mock_state.actor_cursors = {}  # empty

    t = clock.now("UnknownPerson")

    expected = mock_state.current_date.replace(
        hour=DAY_START_HOUR, minute=DAY_START_MINUTE, second=0, microsecond=0
    )
    assert t == expected


# ─────────────────────────────────────────────────────────────────────────────
# 8. SYNC TO SYSTEM
# ─────────────────────────────────────────────────────────────────────────────

def test_sync_to_system_pulls_lagging_actors_forward(clock, mock_state):
    """sync_to_system must bring actors behind the system clock up to it."""
    sys_time = datetime(2026, 3, 10, 12, 0, 0)
    mock_state.actor_cursors["system"] = sys_time
    mock_state.actor_cursors["Alice"]  = datetime(2026, 3, 10, 9, 30, 0)  # behind

    clock.sync_to_system(["Alice"])

    assert mock_state.actor_cursors["Alice"] == sys_time


def test_sync_to_system_does_not_rewind_actors_ahead_of_system(clock, mock_state):
    """
    An actor already ahead of the system clock (busy with future work) must not
    be pulled backward — the incident interrupts them, not rewinds them.
    """
    sys_time = datetime(2026, 3, 10, 10, 0, 0)
    future   = datetime(2026, 3, 10, 14, 0, 0)
    mock_state.actor_cursors["system"] = sys_time
    mock_state.actor_cursors["Alice"]  = future

    clock.sync_to_system(["Alice"])

    assert mock_state.actor_cursors["Alice"] == future


# ─────────────────────────────────────────────────────────────────────────────
# 9. SCHEDULE MEETING
# ─────────────────────────────────────────────────────────────────────────────

def test_schedule_meeting_returns_time_within_window(clock, mock_state):
    """schedule_meeting must return a start time inside [min_hour, max_hour)."""
    clock.reset_to_business_start(["Alice", "Bob"])

    for _ in range(20):  # randomised — run multiple trials
        mock_state.actor_cursors = {
            "Alice": datetime(2026, 3, 10, 9, 0, 0),
            "Bob":   datetime(2026, 3, 10, 9, 0, 0),
        }
        result = clock.schedule_meeting(["Alice", "Bob"], min_hour=10, max_hour=12)
        assert 10 <= result.hour < 12, f"Meeting hour {result.hour} outside [10, 12)"


def test_schedule_meeting_advances_cursors_past_meeting_end(clock, mock_state):
    """After schedule_meeting, all participants must be at or after the meeting end."""
    mock_state.actor_cursors = {
        "Alice": datetime(2026, 3, 10, 9, 0, 0),
        "Bob":   datetime(2026, 3, 10, 9, 0, 0),
    }
    start = clock.schedule_meeting(["Alice", "Bob"], min_hour=10, max_hour=11, duration_mins=30)
    meeting_end = start + timedelta(minutes=30)

    assert mock_state.actor_cursors["Alice"] >= meeting_end
    assert mock_state.actor_cursors["Bob"]   >= meeting_end


# ─────────────────────────────────────────────────────────────────────────────
# 10. SYNC AND ADVANCE
# ─────────────────────────────────────────────────────────────────────────────

def test_sync_and_advance_returns_sync_start_and_end(clock, mock_state):
    """sync_and_advance must return (meeting_start, new_cursor_horizon)."""
    mock_state.actor_cursors["Alice"] = datetime(2026, 3, 10, 10, 0, 0)
    mock_state.actor_cursors["Bob"]   = datetime(2026, 3, 10, 9,  0, 0)  # behind

    start, end = clock.sync_and_advance(["Alice", "Bob"], hours=1.0)

    # Start must be the max of the pre-sync cursors (Alice's 10:00)
    assert start == datetime(2026, 3, 10, 10, 0, 0)
    assert end   == datetime(2026, 3, 10, 11, 0, 0)


def test_sync_and_advance_updates_all_participant_cursors(clock, mock_state):
    """Every actor in the list must end up at the returned end time."""
    mock_state.actor_cursors["Alice"] = datetime(2026, 3, 10, 10, 0, 0)
    mock_state.actor_cursors["Bob"]   = datetime(2026, 3, 10, 10, 0, 0)

    _, end = clock.sync_and_advance(["Alice", "Bob"], hours=2.0)

    assert mock_state.actor_cursors["Alice"] == end
    assert mock_state.actor_cursors["Bob"]   == end


# ─────────────────────────────────────────────────────────────────────────────
# 11. BUSINESS HOURS ENFORCEMENT
# ─────────────────────────────────────────────────────────────────────────────

def test_weekend_rolls_to_monday(clock, mock_state):
    """
    Advancing an actor into a Saturday must roll the cursor to Monday 09:00,
    skipping both weekend days.
    """
    # Friday 17:00 — one hour advance will push past EOD and into the weekend
    mock_state.current_date = datetime(2026, 3, 13, 0, 0, 0)  # Friday
    mock_state.actor_cursors["Alice"] = datetime(2026, 3, 13, 17, 0, 0)

    _, new_cursor = clock.advance_actor("Alice", 1.0)

    # Must land on Monday (weekday 0)
    assert new_cursor.weekday() == 0, f"Expected Monday, got weekday {new_cursor.weekday()}"
    assert new_cursor.hour   == DAY_START_HOUR
    assert new_cursor.minute == DAY_START_MINUTE


def test_no_overlapping_tasks_for_same_actor(clock, mock_state):
    """
    Two sequential advance_actor calls must never produce overlapping time blocks.
    The second block must start where the first ended.
    """
    clock.reset_to_business_start(["Alice"])
    _, end_of_first = clock.advance_actor("Alice", 1.0)
    start_of_second = mock_state.actor_cursors["Alice"]

    assert start_of_second == end_of_first
