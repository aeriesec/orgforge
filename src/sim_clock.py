"""
sim_clock.py
============
Actor-Local simulation clock for OrgForge.

This clock guarantees perfect forensic timelines by tracking time per-employee.
It prevents "Engineer Cloning" (an employee doing two things at the exact same
millisecond) and allows true parallel activity across the company.

Two core mechanisms:
  AMBIENT (advance_actor):
      Moves an individual's cursor forward based on the estimated hours of their
      AgendaItem task. Other employees are unaffected.

  CAUSAL (sync_and_tick / tick_message):
      Finds the latest cursor among all participating actors, brings everyone up
      to that time (simulating asynchronous delay/waiting), and then ticks forward.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    pass  # State imported at runtime to avoid circular imports

# ── Business hours config ─────────────────────────────────────────────────────
DAY_START_HOUR = 9
DAY_START_MINUTE = 0
DAY_END_HOUR = 17
DAY_END_MINUTE = 30

# ── Per-cadence reply spacing (minutes) ──────────────────────────────────────
CADENCE_RANGES = {
    "incident": (1, 4),  # urgent — tight back-and-forth
    "normal": (3, 12),  # standard conversation pace
    "async": (10, 35),  # heads-down work, slow replies
}


class SimClock:
    """
    Manages time cursors for every actor in the simulation.
    Guarantees no overlapping tasks for individuals while allowing the org to run in parallel.
    """

    def __init__(self, state):
        self._state = state
        if not hasattr(self._state, "actor_cursors"):
            self._state.actor_cursors = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def reset_to_business_start(self, all_actors: List[str]) -> None:
        """
        Call at the top of each day in daily_cycle().
        Resets every actor's cursor to 09:00 on the current simulation date,
        skipping weekends automatically.
        """
        base = self._get_default_start()

        # Reset all human actors
        self._state.actor_cursors = {a: base for a in all_actors}

        # Reset a dedicated system cursor for independent bot alerts
        self._state.actor_cursors["system"] = base

    def advance_actor(self, actor: str, hours: float) -> tuple[datetime, datetime]:
        """
        AMBIENT WORK: Advance an actor's horizon.
        Returns (artifact_timestamp, new_cursor_horizon).
        artifact_timestamp is randomly sampled from within the work block.
        """
        current = self._get_cursor(actor)
        delta = timedelta(hours=hours)
        end_time = self._enforce_business_hours(current + delta)

        # Sample a random time within the block for the actual JIRA/Confluence timestamp
        total_seconds = int((end_time - current).total_seconds())
        if total_seconds > 0:
            random_offset = random.randint(0, total_seconds)
            artifact_time = current + timedelta(seconds=random_offset)
        else:
            artifact_time = current

        self._set_cursor(actor, end_time)
        return artifact_time, end_time

    def sync_and_tick(
        self,
        actors: List[str],
        min_mins: int = 5,
        max_mins: int = 15,
        allow_after_hours: bool = False,
    ) -> datetime:
        """
        CAUSAL WORK: Synchronizes multiple actors, then ticks forward.
        Finds the latest time among participants (meaning the thread cannot
        start until the busiest person is free), moves everyone to that time,
        and advances by a random delta.
        """
        # 1. Sync everyone to the maximum cursor
        synced_time = self._sync_time(actors)

        # 2. Tick forward from the synced time
        delta = timedelta(minutes=random.randint(min_mins, max_mins))
        candidate = synced_time + delta

        if not allow_after_hours:
            candidate = self._enforce_business_hours(candidate)

        # 3. Update all participants
        for a in actors:
            self._set_cursor(a, candidate)

        return candidate

    def tick_message(
        self,
        actors: List[str],
        cadence: str = "normal",
        allow_after_hours: bool = False,
    ) -> datetime:
        """
        CAUSAL WORK: Advance sim_time for a specific group of actors by a
        cadence-appropriate amount. Use inside _parse_slack_messages().
        """
        min_mins, max_mins = CADENCE_RANGES.get(cadence, CADENCE_RANGES["normal"])
        return self.sync_and_tick(actors, min_mins, max_mins, allow_after_hours)

    def tick_system(self, min_mins: int = 2, max_mins: int = 5) -> datetime:
        """Advances the independent system clock for automated bot alerts."""
        current = self._get_cursor("system")
        candidate = current + timedelta(minutes=random.randint(min_mins, max_mins))
        candidate = self._enforce_business_hours(candidate)
        self._set_cursor("system", candidate)
        return candidate

    def at(
        self, actors: List[str], hour: int, minute: int, duration_mins: int = 30
    ) -> datetime:
        """
        Pins a scheduled meeting to a specific time.
        Returns the exact meeting start time for artifact stamping.
        """
        meeting_start = self._get_default_start().replace(hour=hour, minute=minute)
        meeting_end = meeting_start + timedelta(minutes=duration_mins)

        for a in actors:
            # Only advance people who are behind the meeting's end time.
            # If someone is already past it, they "missed" it, but we don't time-travel them back.
            if self._get_cursor(a) < meeting_end:
                self._set_cursor(a, meeting_end)

        # The artifact itself is stamped exactly when scheduled
        return meeting_start

    def now(self, actor: str) -> datetime:
        """Return a specific actor's current time without advancing it."""
        return self._get_cursor(actor)

    def sync_to_system(self, actors: List[str]) -> datetime:
        """
        INCIDENT RESPONSE: Forces specific actors to jump to the current system
        clock (e.g., when an incident fires).
        """
        sys_time = self._get_cursor("system")
        for a in actors:
            # If they are behind the system alert, pull them forward to it.
            # If they are somehow ahead, the incident interrupts their future task.
            if self._get_cursor(a) < sys_time:
                self._set_cursor(a, sys_time)
        return sys_time

    def schedule_meeting(
        self, actors: List[str], min_hour: int, max_hour: int, duration_mins: int = 45
    ) -> datetime:
        """
        Schedules a ceremony within a specific window and syncs all participants to it.
        Example: schedule_meeting(leads, 9, 11) for Sprint Planning.
        """
        # Pick a random top-of-the-hour or half-hour slot in the window
        hour = random.randint(min_hour, max(min_hour, max_hour - 1))
        minute = random.choice([0, 15, 30, 45])

        # Use the at() method we defined earlier to pin the actors
        return self.at(actors, hour, minute, duration_mins)

    def sync_and_advance(
        self, actors: List[str], hours: float
    ) -> tuple[datetime, datetime]:
        """
        Synchronizes multiple actors to the latest available cursor,
        then advances all of them by the specified hours.
        Returns (meeting_start_time, new_cursor_horizon).
        """
        # 1. Find the latest time among all participants
        start_time = self._sync_time(actors)

        # 2. Calculate the end time based on the agenda item's duration
        delta = timedelta(hours=hours)
        end_time = self._enforce_business_hours(start_time + delta)

        # 3. Fast-forward everyone to the end of the meeting
        for a in actors:
            self._set_cursor(a, end_time)

        return start_time, end_time

    # ── Private ───────────────────────────────────────────────────────────────

    def _get_default_start(self) -> datetime:
        """Returns 09:00 on the current State date."""
        return self._state.current_date.replace(
            hour=DAY_START_HOUR,
            minute=DAY_START_MINUTE,
            second=0,
            microsecond=0,
        )

    def _get_cursor(self, actor: str) -> datetime:
        """Safely fetch an actor's cursor, defaulting to 09:00 today."""
        if actor not in self._state.actor_cursors:
            self._state.actor_cursors[actor] = self._get_default_start()
        return self._state.actor_cursors[actor]

    def _set_cursor(self, actor: str, dt: datetime) -> None:
        self._state.actor_cursors[actor] = dt

    def _sync_time(self, actors: List[str]) -> datetime:
        """Finds the latest cursor among actors and pulls everyone up to it."""
        if not actors:
            return self._get_cursor("system")

        max_time = max(self._get_cursor(a) for a in actors)
        for a in actors:
            self._set_cursor(a, max_time)

        return max_time

    def _enforce_business_hours(self, dt: datetime) -> datetime:
        """
        If dt falls outside 09:00–17:30 Mon–Fri, return 09:00 the next
        business day. Weekends are skipped.
        """
        end_of_day = dt.replace(
            hour=DAY_END_HOUR,
            minute=DAY_END_MINUTE,
            second=0,
            microsecond=0,
        )
        if dt <= end_of_day and dt.weekday() < 5:
            return dt

        # Roll forward to next business day 09:00
        next_day = dt + timedelta(days=1)
        while next_day.weekday() >= 5:
            next_day += timedelta(days=1)

        return next_day.replace(
            hour=DAY_START_HOUR,
            minute=DAY_START_MINUTE,
            second=0,
            microsecond=0,
        )

    def tick_speaker(self, actor: str, cadence: str = "normal") -> datetime:
        """
        Advances only the current speaker's cursor by a cadence-appropriate delta.
        Use inside _parse_slack_messages() instead of tick_message().
        """
        min_mins, max_mins = CADENCE_RANGES.get(cadence, CADENCE_RANGES["normal"])
        current = self._get_cursor(actor)
        delta = timedelta(minutes=random.randint(min_mins, max_mins))
        candidate = self._enforce_business_hours(current + delta)
        self._set_cursor(actor, candidate)
        return candidate
