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

# ─────────────────────────────────────────────────────────────────────────────
# PLAUSIBILITY RULES
# Each rule is a (condition, rejection_reason) pair.
# Rules are evaluated in order — first failure rejects the event.
# ─────────────────────────────────────────────────────────────────────────────

# Events that are inappropriate when system health is critically low
_BLOCKED_WHEN_CRITICAL = {
    "team_celebration",
    "hackathon",
    "offretreat",
    "deep_work_session",
}

# Events that require at least one incident to have occurred recently
_REQUIRES_PRIOR_INCIDENT = {
    "postmortem_created",
    "escalation_chain",
    "stability_update_to_sales",
    "customer_escalation",
}

# Minimum days between repeated firings of the same event type
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
        self._valid_actors: Set[str] = set(all_names) | set(external_contact_names)
        self._config = config
        self._novel_log: List[ProposedEvent] = []  # accumulates for SimEvent logging

    # ─── PUBLIC ──────────────────────────────────────────────────────────────

    def validate_plan(
        self,
        proposed: List[ProposedEvent],
        state,  # flow.State — avoids circular import
        recent_events: List[dict],  # last N day_summary facts dicts
    ) -> List[ValidationResult]:
        """
        Validate every ProposedEvent in the plan.
        Returns ValidationResult for each — caller logs rejections as SimEvents.
        """
        # Build cooldown tracker from recent events
        recent_event_types = self._recent_event_types(recent_events)
        recent_incident_count = sum(e.get("incidents_opened", 0) for e in recent_events)
        # Live per-ticket actor tracking for today — read from state, not summaries.
        # state.ticket_actors_today is populated by flow.py as ticket_progress
        # events execute, and reset to {} at the top of each daily_cycle().
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

    # ─── PRIVATE ─────────────────────────────────────────────────────────────

    def _validate_one(
        self,
        event: ProposedEvent,
        state,
        recent_event_types: Dict[str, int],  # {event_type: days_since_last}
        recent_incident_count: int,
        ticket_actors_today: Dict[
            str, set
        ],  # {ticket_id: {actors who touched it today}}
    ) -> ValidationResult:

        # ── 1. Actor integrity ────────────────────────────────────────────────
        unknown_actors = [a for a in event.actors if a not in self._valid_actors]
        if unknown_actors:
            return ValidationResult(
                approved=False,
                event=event,
                rejection_reason=f"Unknown actors: {unknown_actors}. "
                f"LLM invented names not in org_chart.",
            )

        # ── 1b. Departed-actor guard ──────────────────────────────────────────
        # patch_validator_for_lifecycle() keeps _valid_actors pruned,
        # but this explicit check gives a clearer rejection message.
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

        # ── 2. Novel event type ───────────────────────────────────────────────
        if event.event_type not in KNOWN_EVENT_TYPES:
            # Novel events are approved if they name a known artifact type.
            # This allows the engine to generate something even without
            # a bespoke handler — it falls back to a Slack summary.
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

        # ── 3. State plausibility ─────────────────────────────────────────────
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

        # ── 4. Cooldown window ────────────────────────────────────────────────
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

        # ── 5. Morale-gated events ────────────────────────────────────────────
        if event.event_type == "morale_intervention" and state.team_morale > 0.6:
            return ValidationResult(
                approved=False,
                event=event,
                rejection_reason=(
                    f"morale_intervention not warranted: morale={state.team_morale:.2f} "
                    f"is above intervention threshold."
                ),
            )

        # ── 6. Ticket dedup ───────────────────────────────────────────────────
        # Prevents multiple agents independently logging progress on the same
        # ticket on the same day. ticket_id is sourced from facts_hint, not
        # related_id (which lives on AgendaItem, not ProposedEvent).
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

        # ── All checks passed ─────────────────────────────────────────────────
        return ValidationResult(approved=True, event=event)

    def _recent_event_types(self, recent_summaries: List[dict]) -> Dict[str, int]:
        """
        Returns {event_type: days_since_last_occurrence} from day_summary facts.
        Uses dominant_event and event_type_counts from the enriched summary.
        """
        days_since: Dict[str, int] = {}
        for i, summary in enumerate(reversed(recent_summaries)):
            # dominant_event is the richest single signal per day
            dominant = summary.get("dominant_event")
            if dominant and dominant not in days_since:
                days_since[dominant] = i + 1
            # event_type_counts gives the full picture
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
