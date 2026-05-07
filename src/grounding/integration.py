"""
Single-import integration surface for OrgForge renderer code.

Each `_handle_*` method in normal_day.py adds ONE line right after building
the CrewAI Agent's backstory:

    backstory = augment(backstory, activity_type="ticket_progress",
                        assignee=assignee, ticket=ticket,
                        date_str=date_str, sim_day=self._state.day)

When ORGFORGE_GROUNDING_ENABLED is unset (the default), this function is a
strict no-op — it returns its input unchanged. The simulation runs identically
to vanilla OrgForge in that case, which gives us a clean A/B baseline.

Errors in the grounding pipeline NEVER raise into OrgForge code: any failure
falls back to returning the original backstory and logs a warning.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger("orgforge.grounding.integration")


def augment(backstory: str, **ctx: Any) -> str:
    """Universal backstory augmenter. Accepts any subset of:

      activity_type        : str (required)  e.g. "ticket_progress", "pr_review", "async_question", "design_discussion"
      assignee / reviewer  : str             primary actor
      actors               : list[str]       additional actors (collaborators)
      ticket               : dict            OrgForge ticket dict (will read title, description, status)
      item_description     : str             agenda item description
      pr_title             : str             PR title for pr_review
      date_str             : str             ISO date
      sim_day              : int             OrgForge sim day index
      company_industry     : str             velomind.yaml's simulation.industry
      channel_kind         : str             "slack" | "email_internal" | "email_external"
      completion_artifact  : str             from non-eng tickets — overrides channel_kind
      knowledge_gap_hint   : str             from orphaned-domain pass

    Returns the augmented backstory string (or the original on no-op / failure).
    """
    try:
        from . import GROUNDING_ENABLED
    except Exception:
        return backstory
    if not GROUNDING_ENABLED:
        return backstory
    try:
        from .event_context import EventContext, _detect_component
        from .prompt_injector import augment_backstory as _aug

        activity_type = ctx.get("activity_type") or "generic_activity"

        # Assemble actors
        primary = ctx.get("assignee") or ctx.get("reviewer") or ""
        extra_actors = ctx.get("actors") or []
        if not isinstance(extra_actors, list):
            extra_actors = [extra_actors] if extra_actors else []
        all_actors = ([primary] if primary else []) + list(extra_actors)

        # Component detection
        ticket = ctx.get("ticket") or {}
        text_parts = [
            (ticket.get("title") or "") if isinstance(ticket, dict) else "",
            (ticket.get("description") or "") if isinstance(ticket, dict) else "",
            ctx.get("item_description") or "",
            ctx.get("pr_title") or "",
        ]
        component = _detect_component(" ".join(p for p in text_parts if p))

        # Channel resolution
        channel_kind = ctx.get("channel_kind") or "slack"
        completion_artifact = ctx.get("completion_artifact")
        if completion_artifact == "email":
            channel_kind = "email_internal"
        elif completion_artifact == "confluence":
            channel_kind = "confluence"

        ticket_title = (
            ticket.get("title") if isinstance(ticket, dict) else None
        ) or (ctx.get("item_description") or ctx.get("pr_title") or "")[:200] or None
        ticket_status = ticket.get("status") if isinstance(ticket, dict) else None

        evt_ctx = EventContext(
            activity_type=activity_type,
            actors=all_actors,
            channel_kind=channel_kind,
            date=ctx.get("date_str") or "",
            sim_day=int(ctx.get("sim_day") or 0),
            component=component,
            ticket_title=ticket_title,
            ticket_status=ticket_status,
            knowledge_gap_hint=ctx.get("knowledge_gap_hint") or None,
            company_industry=ctx.get("company_industry"),
        )
        return _aug(backstory, evt_ctx)
    except Exception as exc:
        logger.warning("[integration] augment failed (returning original): %s", exc)
        return backstory
