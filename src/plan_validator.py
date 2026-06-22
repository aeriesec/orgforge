"""
plan_validator.py
=================
The integrity enforcer between LLM proposals and the execution engine.

The LLM proposes. The engine decides. This is the boundary.

Checks every ProposedEvent against:
  1. Actor integrity    — named actors must exist in the org or external contacts
  2. Causal consistency — event can't contradict facts in the last N SimEvents
  3. State plausibility — health/morale thresholds make the event sensible
  4. Cooldown windows   — same event type can't fire too frequently
  5. Novel event triage — unknown event types are logged, not silently dropped
  6. Ticket dedup       — same ticket can't receive progress from multiple actors
                          on the same day (reads state.ticket_actors_today)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Set

from planner_models import (
    ProposedEvent,
    ValidationResult,
    KNOWN_EVENT_TYPES,
)

logger = logging.getLogger("orgforge.validator")

_DEPARTED_NAMES: set = set()


_BLOCKED_WHEN_CRITICAL = {
    "team_celebration",
    "hackathon",
    "offretreat",
    "deep_work_session",
}


_REQUIRES_PRIOR_INCIDENT = {
    "postmortem_created",
    "escalation_chain",
    "stability_update_to_sales",
    "customer_escalation",
}


_COOLDOWN_DAYS: Dict[str, int] = {
    "retrospective": 9,
    "sprint_planned": 9,
    "morale_intervention": 5,
    "hr_checkin": 3,
    "leadership_sync": 2,
    "vendor_meeting": 3,
    "onboarding_session": 1,
    "farewell_message": 999,
    "warmup_1on1": 2,
    "dlp_alert": 1,
    "secret_detected": 999,
}


class PlanValidator:
    """
    Validates a list of ProposedEvents before the engine executes them.

    Usage:
        validator = PlanValidator(
            all_names=ALL_NAMES,
            external_contact_names=external_names,
            config=CONFIG,
        )
        results = validator.validate_plan(proposed_events, state, recent_events)
    """

    def __init__(
        self,
        all_names: List[str],
        external_contact_names: List[str],
        config: dict,
    ):
        self._internal_names: Set[str] = set(all_names)
        self._external_names: Set[str] = set(external_contact_names)

        self._valid_actors: Set[str] = self._internal_names | self._external_names
        self._config = config
        self._novel_log: List[ProposedEvent] = []

    @property
    def external_contact_names(self) -> List[str]:
        """Returns the current list of valid external names."""
        return list(self._external_names)

    @external_contact_names.setter
    def external_contact_names(self, names: List[str]):
        """
        Refresh the whitelist of valid external actors.
        Called daily by the DayPlannerOrchestrator.
        """
        self._external_names = set(names)
        self._valid_actors = self._internal_names | self._external_names

    def validate_plan(
        self,
        proposed: List[ProposedEvent],
        state,
        recent_events: List[dict],
    ) -> List[ValidationResult]:
        """
        Validate every ProposedEvent in the plan.
        Returns ValidationResult for each — caller logs rejections as SimEvents.
        """

        recent_event_types = self._recent_event_types(recent_events)
        recent_incident_count = sum(e.get("incidents_opened", 0) for e in recent_events)
        ticket_actors_today = self._ticket_actors_today(state)

        results: List[ValidationResult] = []
        for event in proposed:
            result = self._validate_one(
                event,
                state,
                recent_event_types,
                recent_incident_count,
                ticket_actors_today,
            )
            if not result.approved and result.was_novel:
                self._novel_log.append(event)
            results.append(result)

        return results

    def approved(self, results: List[ValidationResult]) -> List[ProposedEvent]:
        """Convenience filter — returns only approved events."""
        return [r.event for r in results if r.approved]

    def rejected(self, results: List[ValidationResult]) -> List[ValidationResult]:
        """Convenience filter — returns only rejected results with reasons."""
        return [r for r in results if not r.approved]

    def drain_novel_log(self) -> List[ProposedEvent]:
        """
        Returns novel (unknown event type) proposals since last drain.
        Caller should log these as 'novel_event_proposed' SimEvents so
        researchers and contributors can see what the LLM wanted to do.
        """
        novel = list(self._novel_log)
        self._novel_log.clear()
        return novel

    def _validate_one(
        self,
        event: ProposedEvent,
        state,
        recent_event_types: Dict[str, int],
        recent_incident_count: int,
        ticket_actors_today: Dict[str, set],
    ) -> ValidationResult:

        unknown_actors = [a for a in event.actors if a not in self._valid_actors]
        if unknown_actors:
            return ValidationResult(
                approved=False,
                event=event,
                rejection_reason=f"Unknown actors: {unknown_actors}. "
                f"LLM invented names not in org_chart.",
            )

        departed_actors = [a for a in event.actors if a in _DEPARTED_NAMES]
        if departed_actors:
            return ValidationResult(
                approved=False,
                event=event,
                rejection_reason=(
                    f"Actors {departed_actors} have departed the organisation. "
                    f"Remove them from this event."
                ),
            )

        if event.event_type not in KNOWN_EVENT_TYPES:
            if event.artifact_hint in {"slack", "jira", "confluence", "email"}:
                logger.info(
                    f"  [cyan]✨ Novel event approved (fallback artifact):[/cyan] "
                    f"{event.event_type} → {event.artifact_hint}"
                )
                return ValidationResult(approved=True, event=event, was_novel=True)
            else:
                return ValidationResult(
                    approved=False,
                    event=event,
                    was_novel=True,
                    rejection_reason=(
                        f"Novel event type '{event.event_type}' has no known "
                        f"artifact_hint. Logged for future implementation."
                    ),
                )

        if event.event_type in _BLOCKED_WHEN_CRITICAL and state.system_health < 40:
            return ValidationResult(
                approved=False,
                event=event,
                rejection_reason=(
                    f"'{event.event_type}' blocked: system health critical "
                    f"({state.system_health}). Inappropriate tone for current state."
                ),
            )

        if event.event_type in _REQUIRES_PRIOR_INCIDENT and recent_incident_count == 0:
            return ValidationResult(
                approved=False,
                event=event,
                rejection_reason=(
                    f"'{event.event_type}' requires a prior incident in the "
                    f"recent window. None found."
                ),
            )

        cooldown = _COOLDOWN_DAYS.get(event.event_type)
        if cooldown:
            days_since = recent_event_types.get(event.event_type, 999)
            if days_since < cooldown:
                return ValidationResult(
                    approved=False,
                    event=event,
                    rejection_reason=(
                        f"'{event.event_type}' in cooldown. "
                        f"Last fired {days_since}d ago, cooldown is {cooldown}d."
                    ),
                )

        if event.event_type == "morale_intervention" and state.team_morale > 0.6:
            return ValidationResult(
                approved=False,
                event=event,
                rejection_reason=(
                    f"morale_intervention not warranted: morale={state.team_morale:.2f} "
                    f"is above intervention threshold."
                ),
            )

        if event.event_type == "ticket_progress":
            ticket_id = (event.facts_hint or {}).get("ticket_id")
            if ticket_id:
                actors_on_ticket = ticket_actors_today.get(ticket_id, set())
                overlap = [a for a in event.actors if a in actors_on_ticket]
                if overlap:
                    return ValidationResult(
                        approved=False,
                        event=event,
                        rejection_reason=(
                            f"Duplicate ticket work: {overlap} already logged "
                            f"progress on {ticket_id} today."
                        ),
                    )

        return ValidationResult(approved=True, event=event)

    def _recent_event_types(self, recent_summaries: List[dict]) -> Dict[str, int]:
        """
        Returns {event_type: days_since_last_occurrence} from day_summary facts.
        Uses dominant_event and event_type_counts from the enriched summary.
        """
        days_since: Dict[str, int] = {}
        for i, summary in enumerate(reversed(recent_summaries)):
            dominant = summary.get("dominant_event")
            if dominant and dominant not in days_since:
                days_since[dominant] = i + 1

            for etype in summary.get("event_type_counts", {}).keys():
                if etype not in days_since:
                    days_since[etype] = i + 1
        return days_since

    def _ticket_actors_today(self, state) -> Dict[str, set]:
        """
        Returns the live {ticket_id: {actor, ...}} map for today.
        Reads from state.ticket_actors_today, which flow.py owns:
          - Reset to {} at the top of each daily_cycle()
          - Updated after each ticket_progress event fires
        Defaults to {} safely if state doesn't have the attribute yet.
        """
        return getattr(state, "ticket_actors_today", {})
