"""
Extract event context from OrgForge's agenda items / SimEvents.

This is the bridge between OrgForge's internal data structures and the
grounding pipeline. We only read — never mutate — OrgForge state.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional


# Heuristic component keywords drawn from velomind.yaml's incident_triggers
# and tech_stack notable_quirks. Used to enrich event_ctx.component when
# it isn't passed explicitly.
_COMPONENT_PATTERNS: list[tuple[str, str]] = [
    (r"\bts[- ]?17[ab]\b|torque[- ]?sensor", "torque_sensor"),
    (r"\bbat[- ]?0?42\b|cold impedance|battery|range estimate", "battery"),
    (r"\bota\b|firmware|smartassist|f_torque_stale", "firmware"),
    (r"dealer dashboard|held[- ]?unit", "dealer_dashboard"),
    (r"telemetry|incident upload|kafka|timescaledb", "telemetry"),
]


def _detect_component(text: str) -> Optional[str]:
    if not text:
        return None
    lc = text.lower()
    for pattern, label in _COMPONENT_PATTERNS:
        if re.search(pattern, lc):
            return label
    return None


@dataclass(frozen=True)
class EventContext:
    """Everything the grounding layer needs to know about an event in order
    to fetch matching real-world artifacts and look up the right genre profile.
    """

    activity_type: str          # OrgForge activity (ticket_progress, async_question, ...)
    actors: list[str]           # Named actors involved (engineer + collaborators)
    channel_kind: str           # 'slack' | 'email_internal' | 'email_external' | ...
    date: str                   # ISO calendar date
    sim_day: int                # OrgForge sim day index

    component: Optional[str] = None
    symptom: Optional[str] = None
    ticket_title: Optional[str] = None
    ticket_status: Optional[str] = None
    arc_id: Optional[str] = None
    knowledge_gap_hint: Optional[str] = None
    company_industry: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def from_ticket_progress(
    *,
    assignee: str,
    collaborators: list[str],
    ticket: dict,
    date_str: str,
    sim_day: int,
    is_non_eng: bool,
    completion_artifact: str,
    knowledge_gap_hint: str = "",
    company_industry: Optional[str] = None,
) -> EventContext:
    """Build an EventContext from a ticket-progress agenda item. Used by the
    NormalDayHandler._handle_ticket_progress site."""
    title = ticket.get("title", "") or ""
    desc = ticket.get("description", "") or ""
    component = _detect_component(title + " " + desc)
    # Channel kind is driven by completion_artifact when non-eng; engineering
    # work renders via the slack thread by default.
    if is_non_eng:
        channel_kind = {
            "email": "email_internal",
            "slack": "slack",
            "confluence": "confluence",
        }.get(completion_artifact, "slack")
    else:
        channel_kind = "slack"
    return EventContext(
        activity_type="ticket_progress",
        actors=[assignee] + list(collaborators),
        channel_kind=channel_kind,
        date=date_str,
        sim_day=sim_day,
        component=component,
        ticket_title=title,
        ticket_status=ticket.get("status"),
        knowledge_gap_hint=knowledge_gap_hint or None,
        company_industry=company_industry,
    )


def from_async_question(
    *,
    asker: str,
    answerers: list[str],
    item_description: str,
    date_str: str,
    sim_day: int,
    company_industry: Optional[str] = None,
) -> EventContext:
    return EventContext(
        activity_type="async_question",
        actors=[asker] + list(answerers),
        channel_kind="slack",
        date=date_str,
        sim_day=sim_day,
        component=_detect_component(item_description),
        ticket_title=item_description[:120] if item_description else None,
        company_industry=company_industry,
    )


def from_pr_review(
    *,
    reviewer: str,
    author: Optional[str],
    pr_title: str,
    date_str: str,
    sim_day: int,
    company_industry: Optional[str] = None,
) -> EventContext:
    actors = [reviewer]
    if author:
        actors.append(author)
    return EventContext(
        activity_type="pr_review",
        actors=actors,
        channel_kind="slack",
        date=date_str,
        sim_day=sim_day,
        component=_detect_component(pr_title),
        ticket_title=pr_title,
        company_industry=company_industry,
    )


def from_design_discussion(
    *,
    participants: list[str],
    description: str,
    date_str: str,
    sim_day: int,
    company_industry: Optional[str] = None,
) -> EventContext:
    return EventContext(
        activity_type="design_discussion",
        actors=list(participants),
        channel_kind="slack",
        date=date_str,
        sim_day=sim_day,
        component=_detect_component(description),
        ticket_title=description[:120] if description else None,
        company_industry=company_industry,
    )
