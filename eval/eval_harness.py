"""
eval_harness.py
===============
OrgForge Eval Dataset Generator

Produces three novel eval tracks that require the deterministic state machine
to exist. No retrieval questions. Those are covered by other benchmarks.

Run after flow.py and post_sim_artifacts.py complete:
    python eval_harness.py

Produces in export/eval/:
    actor_visibility.json      — per-actor artifact visibility cones, time-indexed
    causal_link_index.json     — explicit causal links derived from sim flags
    absence_catalog.json       — expected-but-absent artifact pairs
    eval_questions.json        — PERSPECTIVE + COUNTERFACTUAL + SILENCE questions

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TRACK 1 — PERSPECTIVE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Questions scoped to what a specific actor could have known at a specific moment,
given their actual subsystem access and information horizon.

Ground truth is derived from the actor's visibility cone: the set of artifact IDs
reachable by that actor at or before as_of_time, filtered by subsystem access.

Cross-subsystem questions (e.g. engineer sees Slack + Zoom but not Salesforce)
are flagged difficulty="hard". Single-subsystem questions are "medium".

Example:
  "Based only on what Morgan had access to as of Day 9, should she have known
   that Acme Corp was at churn risk?"
  ground_truth: { "answer": False, "reason": "sf_deals_risk_flagged not in Morgan's visibility cone" }

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TRACK 2 — COUNTERFACTUAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Questions of the form "if X had been different, would Y have occurred?"
Only generated where the sim encodes an explicit causal link:
  - involves_gap / knowledge_gap_detected  → incident causation
  - recurrence_of                          → repeat incident prevention
  - spawned_doc                            → design discussion → documentation
  - email_dropped                          → unactioned communication
  - sf_ownership_lapsed                    → CRM ownership gap
  - zd_escalation_source                   → support ticket → incident

Ground truth is always derivable from the explicit link without inference.

Example:
  "If Jordan had documented auth-service before departing, would incident IT-108
   have been diagnosed faster?"
  ground_truth: { "outcome_changed": True, "mechanism": "knowledge_gap_detected",
                  "gap_domain": "auth-service", "causal_event": "evt_..." }

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TRACK 3 — SILENCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Questions about things that did NOT happen. The state machine is the arbiter:
if no event fired, absence is ground truth regardless of whether absence was
intentional.

Each SILENCE question includes an expected_search_space — the artifact IDs the
agent MUST check before concluding absence. A correct "no" reached without
searching the right places scores 0 on trajectory even if the boolean is right.

Example:
  "Was a postmortem written for the Zendesk escalation on Day 6?"
  ground_truth: False
  expected_search_space: ["confluence/postmortems/", "jira/IT-*", "slack/incidents"]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Design principles
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Ground truth is always derived from SimEvent log. LLMs only write prose.
- Question prose generation includes a structured validation loop.
- Actor visibility cones are first-class data structures, not a scoring afterthought.
- Subsystem access is explicitly modeled per actor per day.
- The absence catalog is built by pattern-matching expected event pairs, not heuristics.
"""

from __future__ import annotations

import json
import logging
import random
import re
from config_loader import CONFIG, DEPARTED_EMPLOYEES
import yaml
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from agent_factory import make_agent
from crewai import Crew, Task
from memory import Memory, SimEvent


logger = logging.getLogger("orgforge.eval")

with open(Path(__file__).resolve().parent.parent / "config" / "config.yaml") as f:
    _CFG = yaml.safe_load(f)

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

EXPORT_DIR = PROJECT_ROOT / "export"

raw_output = CONFIG["simulation"].get("output_dir")
if raw_output:
    BASE = (PROJECT_ROOT / raw_output).resolve()
else:
    BASE = EXPORT_DIR.resolve()

EXPORT_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR = BASE / "eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

_SIM_START = datetime.strptime(_CFG["simulation"]["start_date"], "%Y-%m-%d")


def _business_day_to_date(start: datetime, n: int) -> datetime:
    """Convert a 1-based business day counter to a calendar date."""
    current = start
    days_counted = 0
    while days_counted < n:
        current += timedelta(days=1)
        if current.weekday() < 5:
            days_counted += 1
    return current


_ROLE_SUBSYSTEM_ACCESS: Dict[str, Set[str]] = {
    "ceo": {
        "slack",
        "jira",
        "confluence",
        "zoom",
        "email",
        "salesforce",
        "zendesk",
        "datadog",
    },
    "product": {"slack", "jira", "confluence", "zoom", "email"},
    "engineering_backend": {
        "slack",
        "jira",
        "confluence",
        "git",
        "zoom",
        "email",
        "datadog",
    },
    "engineering_mobile": {
        "slack",
        "jira",
        "confluence",
        "git",
        "zoom",
        "email",
        "datadog",
    },
    "design": {"slack", "confluence", "zoom", "email"},
    "sales_marketing": {"slack", "salesforce", "email", "zoom", "confluence"},
    "hr_ops": {"slack", "email", "confluence", "zoom"},
    "qa_support": {"slack", "zendesk", "confluence", "email"},
    "external": set(),
}

_SYSTEM_ACTORS = {"John"}


_ARTIFACT_SUBSYSTEM: Dict[str, str] = {
    "jira": "jira",
    "confluence": "confluence",
    "slack": "slack",
    "pr": "git",
    "email": "email",
    "zd_ticket": "zendesk",
    "sf_opp": "salesforce",
    "sf_account": "salesforce",
    "datadog": "datadog",
    "zoom": "zoom",
    "invoice": "email",
    "nps": "salesforce",
    "hr_email": "email",
    "jira_comment": "jira",
    "slack_thread": "slack",
}

_EXPLICIT_CAUSAL_LINKS = {
    "involves_gap",  # incident ← knowledge gap
    "recurrence_of",  # incident ← prior unresolved incident
    "spawned_doc",  # confluence ← design discussion
    "email_dropped",  # communication failure ← routing gap
    "sf_ownership_lapsed",  # CRM gap ← employee departure
    "blocker_flagged",  # blocker → delayed progress
    "incident_coordination",  # incident → external contact
    "departure_reassignment",  # departure → ticket/escalation shift
    "assignment_domain_mismatch",  # planning mismatch → knowledge gap → incident
    "jira_from_customer_email",
    "jira_from_vendor_email",
    "customer_escalation_relayed",
    "incident_handoff",
    "pr_gap_detected",
    "async_gap_detected",
    "doc_gap_detected",
    "centrality_vacuum",
    "sf_stage_advanced_by_customer",
    "feature_request_fyi",
    "proactive_outreach_from_crm_signal",
    "ticket_completion_notifies_lead",
    "org_collision_tension",
    "postmortem_from_incident",
    "incident_triggers_risk_flag",
    "review_triggers_revision",
    "hire_fills_knowledge_gap",
    "escalation_from_zendesk",
}


_SILENCE_PAIRS: List[Tuple[str, str, str]] = [
    ("incident_opened", "postmortem_created", "jira"),
    ("incident_opened", "incident_resolved", "jira"),
    ("customer_escalation", "zd_ticket_opened", "email"),
    ("customer_email_routed", "zd_ticket_opened", "email"),
    ("inbound_external_email", "customer_email_routed", "email"),
    ("inbound_external_email", "jira_from_vendor_email", "email"),
    ("design_discussion", "confluence_created", "zoom_transcript"),
    ("knowledge_gap_detected", "confluence_created", "gap_domain"),
    ("zd_tickets_escalated", "incident_opened", "jira"),
    ("employee_departed", "sf_ownership_lapsed", "actor"),
    ("employee_departed", "ticket_reassigned", "actor"),
    ("pr_opened", "pr_merged", "pr"),
    ("incident_opened", "zd_tickets_escalated", "jira"),
    ("employee_hired", "onboarding_session", "name"),
    ("employee_hired", "warmup_1on1", "name"),
    ("incident_opened", "sf_deals_risk_flagged", "jira"),
    ("assignment_domain_mismatch", "knowledge_gap_detected", "ticket_id"),
    ("design_discussion", "confluence_created", "slack_thread"),
    ("email_dropped", "zd_ticket_opened", "email"),
]


_RESPONSE_TYPE_SUBSYSTEM: Dict[str, str] = {
    "postmortem_created": "confluence",
    "incident_resolved": "jira",
    "zd_ticket_opened": "zendesk",
    "customer_email_routed": "email",
    "jira_from_vendor_email": "jira",
    "confluence_created": "confluence",
    "sf_ownership_lapsed": "salesforce",
    "ticket_reassigned": "jira",
    "pr_merged": "git",
    "zd_tickets_escalated": "zendesk",
    "incident_opened": "jira",
    "onboarding_session": "slack",
    "warmup_1on1": "slack",
    "sf_deals_risk_flagged": "salesforce",
    "knowledge_gap_detected": "confluence",
}

_BROADCAST_CONFIG = {
    "incident_opened": ["slack", "datadog"],
    "incident_resolved": ["slack"],
    "postmortem_created": ["slack", "confluence"],
    "standup": ["slack"],
    "pr_opened": ["git"],
    "pr_merged": ["git"],
    "knowledge_gap_detected": ["slack", "confluence"],
}

_VAGUE_GAP_TERMS = {
    "undocumented expertise",
    "undocumented knowledge",
    "general knowledge",
    "tribal knowledge",
    "institutional knowledge",
    "unknown",
    "unspecified",
    "undocumented domain",
    "unknown issue",
}

_JIRA_PROJECT_ACCESS: Dict[str, Set[str]] = {
    "ENG": {"engineering_backend", "engineering_mobile", "ceo"},
    "HR": {"hr_ops", "ceo"},
    "SALES": {"sales_marketing", "ceo"},
    "PROD": {"product", "ceo"},
    "DES": {"design", "product", "ceo"},
    "QA": {"qa_support", "ceo"},
    "ORG": {
        "engineering_backend",
        "engineering_mobile",
        "product",
        "ceo",
        "hr_ops",
        "design",
        "sales_marketing",
        "qa_support",
    },
}

_MAX_QUESTIONS_PER_ACTOR = 5
_MAX_QUESTIONS_PER_EVENT_TYPE = 5

_NEGATIVE_COUNTERFACTUAL_STYLES: List[str] = [
    "Phrase it to suggest the cause was necessary: 'If X had not happened, would Y still have occurred?'",
    "Phrase it as a dependency question: 'Was Y dependent on X, or would it have happened regardless?'",
    "Phrase it from the effect perspective: 'Would Y have been prevented if only X had been different?'",
    "Phrase it as a challenge: 'Can we be sure X was actually required for Y, or would Y have happened anyway?'",
    "Phrase it as an alternate history: 'Had X not taken place, would the outcome still have been Y?'",
]


def _jira_project_visible(ticket_id: str, role: str) -> bool:
    prefix = ticket_id.split("-")[0] if "-" in ticket_id else ""
    allowed_roles = _JIRA_PROJECT_ACCESS.get(prefix)
    if allowed_roles is None:
        return True  # unknown prefix — don't restrict
    return role in allowed_roles


def _strip_root(v: str) -> str:
    """Strip absolute export or project root prefix from stored artifact paths."""
    for prefix in (str(BASE), str(BASE.parent)):
        if v.startswith(prefix + "/"):
            return v[len(prefix) + 1 :]
    return v


def _safe_artifact_values(artifact_ids: dict) -> Set[str]:
    """Flatten artifact_ids values — some may be lists. Skips disk-only keys."""
    vals: Set[str] = set()
    _DISK_ONLY_KEYS = {"eml_path", "zoom_path", "transcript_path"}
    for k, v in (artifact_ids or {}).items():
        if k in _DISK_ONLY_KEYS:
            continue
        if isinstance(v, list):
            vals.update(_strip_root(str(x)) for x in v)
        elif v:
            vals.add(_strip_root(str(v)))
    return vals


@dataclass
class ActorVisibilityCone:
    """
    The complete set of artifact IDs visible to a specific actor at a specific
    moment, partitioned by subsystem.

    Built from SimEvents: an actor can see an artifact if they appear in
    event.actors for that artifact's creation event, or if the artifact was
    broadcast to a channel/tool they have access to (e.g. an incident Slack
    message is visible to all engineers).
    """

    actor: str
    role: str
    as_of_time: str
    as_of_day: int
    subsystem_access: Set[str]  # subsystems this actor can query
    visible_artifacts: Dict[str, Set[str]]  # subsystem → set of artifact IDs
    directly_involved: Set[str]  # artifacts where actor appears in event.actors
    broadcast_visible: Set[str]  # artifacts visible via channel broadcast

    def all_visible(self) -> Set[str]:
        """Return artifact IDs visible to this actor, restricted to accessible subsystems.

        Only subsystems in self.subsystem_access are included so that
        all_visible() and can_see() encode the same access model. Previously
        all_visible() unioned across ALL subsystems regardless of role, which
        caused missed_artifacts in _find_asymmetry_events to be empty for
        artifacts the actor could not actually access.
        """
        all_ids: Set[str] = set()
        for subsystem, ids in self.visible_artifacts.items():
            if subsystem in self.subsystem_access:
                all_ids.update(ids)
        return all_ids

    def can_see(self, artifact_id: str, doc_type: str) -> bool:
        subsystem = _ARTIFACT_SUBSYSTEM.get(doc_type, "default")
        if subsystem not in self.subsystem_access:
            return False
        return artifact_id in self.all_visible()

    def to_dict(self) -> dict:
        return {
            "actor": self.actor,
            "role": self.role,
            "as_of_time": self.as_of_time,
            "as_of_day": self.as_of_day,
            "subsystem_access": sorted(self.subsystem_access),
            "visible_artifacts": {
                k: sorted(v) for k, v in self.visible_artifacts.items()
            },
            "directly_involved": sorted(self.directly_involved),
            "broadcast_visible": sorted(self.broadcast_visible),
        }


@dataclass
class CausalLink:
    """
    An explicit causal relationship encoded in the simulation.
    These are the only valid sources for COUNTERFACTUAL questions.
    """

    link_type: str  # one of _EXPLICIT_CAUSAL_LINKS
    cause_event_id: str
    cause_event_type: str
    effect_event_id: str
    effect_event_type: str
    actors: List[str]
    day: int
    link_field: str  # the fact key that carries the link
    link_value: str  # the value of that field
    subsystems_involved: Set[str]
    counterfactual_premise: str  # natural language "if X had been different"
    counterfactual_outcome: str  # natural language "then Y would have..."
    outcome_changed: bool  # does removing the cause change the effect?

    def to_dict(self) -> dict:
        d = asdict(self)
        d["subsystems_involved"] = sorted(self.subsystems_involved)
        return d


@dataclass
class AbsenceRecord:
    """
    A case where a trigger event fired but its expected response event did not.
    The state machine is the arbiter — no inference about intent.
    """

    trigger_event_id: str
    trigger_event_type: str
    expected_response_type: str
    trigger_day: int
    trigger_actors: List[str]
    trigger_artifact_ids: Dict[str, str]
    link_field: str
    link_value: str
    subsystem: str
    expected_search_space: List[str]  # artifact IDs the agent must check

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# ACTOR VISIBILITY BUILDER
# ─────────────────────────────────────────────────────────────────────────────


class ActorVisibilityBuilder:
    """
    Reconstructs the knowledge cone for every actor at every day boundary.

    Visibility rules:
    1. DIRECT: actor appears in event.actors for an artifact's creation event
    2. BROADCAST: artifact was created in a shared channel (Slack incidents,
       standups, engineering-wide announcements) — visible to all actors with
       that subsystem access
    3. ROLE-GATED: actor's role must include the artifact's subsystem
    4. TEMPORAL: artifact timestamp must be <= as_of_time

    Broadcast channels are inferred from event type:
    - standup, incident_alert, dept_announcement → broadcast to subsystem members
    - direct_message, email, zd_ticket → direct only
    """

    # Event types whose artifacts are broadcast to all actors with subsystem access
    _BROADCAST_EVENTS = {
        "standup",
        "incident_opened",
        "incident_resolved",
        "postmortem_created",
        "pr_opened",
        "pr_merged",
        "knowledge_gap_detected",
    }

    def __init__(self, mem: Memory):
        self._mem = mem
        self._events: List[SimEvent] = mem.get_event_log(from_db=True)
        self._actor_roles: Dict[str, str] = self._infer_actor_roles()

    def _infer_actor_roles(self) -> Dict[str, str]:
        roles: Dict[str, str] = {}

        for dept, members in CONFIG["org_chart"].items():
            for name in members:
                roles[name] = dept.lower().replace(" ", "_")

        for name, data in DEPARTED_EMPLOYEES.items():
            roles[name] = data["dept"].lower().replace(" ", "_")

        lifecycle = CONFIG.get("org_lifecycle", {})

        for hire in lifecycle.get("scheduled_hires", []):
            roles[hire["name"]] = hire["dept"].lower().replace(" ", "_")

        for dep in lifecycle.get("scheduled_departures", []):
            roles[dep["name"]] = dep["dept"].lower().replace(" ", "_")

        for actor in self._all_actors():
            if actor not in roles:
                roles[actor] = "external"

        return roles

    def _subsystem_access_for(self, actor: str) -> Set[str]:
        role = self._actor_roles.get(actor, "external")
        return set(_ROLE_SUBSYSTEM_ACCESS.get(role, _ROLE_SUBSYSTEM_ACCESS["external"]))

    def _all_actors(self) -> Set[str]:
        actors: Set[str] = set()
        for event in self._events:
            actors.update(event.actors)
        return actors

    def _artifact_subsystem(self, doc_type: str) -> str:
        return _ARTIFACT_SUBSYSTEM.get(doc_type, "default")

    def build_all(self) -> Dict[str, List[ActorVisibilityCone]]:
        all_actors = self._all_actors()
        result: Dict[str, List[ActorVisibilityCone]] = {}
        max_day = max((e.day for e in self._events), default=1)

        events_by_day = defaultdict(list)
        for event in self._events:
            events_by_day[event.day].append(event)

        for actor in all_actors:
            role = self._actor_roles.get(actor, "default")
            access = self._subsystem_access_for(actor)
            cones: List[ActorVisibilityCone] = []

            current_visible = defaultdict(set)
            current_directly_involved = set()
            current_broadcast_visible = set()

            for day in range(1, max_day + 1):
                for event in events_by_day.get(day, []):
                    is_direct = actor in (event.actors or [])

                    broadcast_channels = _BROADCAST_CONFIG.get(event.type)
                    is_broadcast = broadcast_channels is not None

                    for doc_type, artifact_id in (event.artifact_ids or {}).items():
                        if not artifact_id:
                            continue
                        subsystem = self._artifact_subsystem(doc_type)
                        if subsystem not in access:
                            continue

                        if is_direct:
                            current_visible[subsystem].add(artifact_id)
                            current_directly_involved.add(artifact_id)
                        elif is_broadcast and any(
                            sub in access for sub in broadcast_channels
                        ):
                            current_visible[subsystem].add(artifact_id)
                            current_broadcast_visible.add(artifact_id)

                as_of_dt = _business_day_to_date(_SIM_START, day).replace(
                    hour=23, minute=59
                )
                cones.append(
                    ActorVisibilityCone(
                        actor=actor,
                        role=role,
                        as_of_time=as_of_dt.isoformat(),
                        as_of_day=day,
                        subsystem_access=access,
                        visible_artifacts={
                            k: set(v) for k, v in current_visible.items()
                        },
                        directly_involved=set(current_directly_involved),
                        broadcast_visible=set(current_broadcast_visible),
                    )
                )

            result[actor] = cones

        return result


class CausalLinkIndexer:
    """
    Scans the SimEvent log for all explicit causal links.
    Only links in _EXPLICIT_CAUSAL_LINKS are indexed — no inference.

    Each link becomes a potential COUNTERFACTUAL question source.
    The counterfactual premise and outcome are templated deterministically
    from the link type and event facts; LLMs only rephrase them.
    """

    MAX_LINKS_PER_TYPE: int = 15

    def __init__(self, mem: Memory):
        self._mem = mem
        self._events: List[SimEvent] = mem.get_event_log(from_db=True)

    def _subsystems_for_event(self, event: SimEvent) -> Set[str]:
        subsystems: Set[str] = set()
        for doc_type in event.artifact_ids or {}:
            s = _ARTIFACT_SUBSYSTEM.get(doc_type, "default")
            if s != "default":
                subsystems.add(s)
        return subsystems

    def _find_effect_event(self, link_type: str, cause: SimEvent) -> Optional[SimEvent]:
        if link_type == "involves_gap":
            gap_areas = cause.facts.get("gap_areas", [])
            if not gap_areas:
                return None

            cause_domains: set[str] = set()
            for area in gap_areas:
                for part in str(area).split(","):
                    part = part.strip().lower()
                    if part:
                        cause_domains.add(part)

            if not cause_domains:
                return None

            for e in self._events:
                if e.day < cause.day:
                    continue

                if e.type == "incident_opened" and e.facts.get("involves_gap"):
                    incident_domains: set[str] = set()
                    for area in e.facts.get("gap_areas", []):
                        for part in str(area).split(","):
                            part = part.strip().lower()
                            if part:
                                incident_domains.add(part)

                    if cause_domains & incident_domains:
                        return e

                relevant_types = {
                    "async_question_asked",
                    "pr_review_comment",
                    "confluence_created",
                    "postmortem_created",
                }
                if e.type in relevant_types:
                    event_domains: set[str] = set()
                    for area in e.facts.get("gap_areas") or e.facts.get("domain") or []:
                        for part in str(area).split(","):
                            part = part.strip().lower()
                            if part:
                                event_domains.add(part)

                    if cause_domains & event_domains:
                        return e

        elif link_type == "recurrence_of":
            prior_jira_id = cause.facts.get("recurrence_of")
            if not prior_jira_id:
                return None
            for e in self._events:
                if (
                    e.type == "incident_opened"
                    and (e.artifact_ids or {}).get("jira") == prior_jira_id
                ):
                    return e
            return None

        elif link_type == "spawned_doc":
            cause_artifacts = _safe_artifact_values(cause.artifact_ids)
            conf_id = (cause.artifact_ids or {}).get("confluence")
            if conf_id:
                return next(
                    (
                        e
                        for e in self._events
                        if e.type == "confluence_created"
                        and (e.artifact_ids or {}).get("confluence") == conf_id
                    ),
                    None,
                )
            for e in self._events:
                if (
                    e.type == "confluence_created"
                    and e.facts.get("source_discussion") in cause_artifacts
                ):
                    return e
            return None

        elif link_type == "postmortem_from_incident":
            jira_id = (cause.artifact_ids or {}).get("jira", "")
            if not jira_id:
                return None
            for e in self._events:
                if (
                    e.type == "postmortem_created"
                    and jira_id in str(e.artifact_ids)
                    and e.day >= cause.day
                ):
                    return e
            return None

        elif link_type == "email_dropped":
            source = cause.facts.get("source")
            email_id = (cause.artifact_ids or {}).get("email", "")
            if not source:
                return None
            for e in self._events:
                if e.day <= cause.day:
                    continue
                if e.type == "customer_escalation" and source in str(e.facts):
                    return e
                if (
                    e.type == "inbound_external_email"
                    and e.facts.get("source") == source
                    and e.facts.get("tone") in ("frustrated", "urgent")
                ):
                    return e
                if e.type == "zd_ticket_opened" and email_id in str(
                    e.facts.get("causal_chain", [])
                ):
                    return e
            return None

        elif link_type == "sf_ownership_lapsed":
            actor = (cause.actors or [None])[0]
            if not actor:
                return None
            for e in self._events:
                if e.type == "sf_ownership_lapsed" and actor in (e.actors or []):
                    return e

        elif link_type == "zd_escalation_source":
            jira_id = (cause.artifact_ids or {}).get("jira")
            if isinstance(jira_id, list):
                jira_id = jira_id[0] if jira_id else None
            if not jira_id:
                return None
            for e in self._events:
                if e.type == "zd_tickets_escalated" and e.day >= cause.day:
                    escalated_jira = (e.artifact_ids or {}).get("jira")
                    if escalated_jira == jira_id:
                        return e
                    if jira_id in str(e.facts):
                        return e
            return None

        elif link_type == "blocker_flagged":
            jira_id = cause.artifact_ids.get("jira")
            return next(
                (
                    e
                    for e in self._events
                    if e.type == "ticket_progress"
                    and e.artifact_ids.get("jira") == jira_id
                    and e.day >= cause.day
                ),
                None,
            )

        elif link_type == "incident_coordination":
            jira_id = cause.artifact_ids.get("jira")
            return next(
                (
                    e
                    for e in self._events
                    if e.type == "external_contact_summarized"
                    and e.artifact_ids.get("jira") == jira_id
                ),
                None,
            )

        elif link_type == "departure_reassignment":
            departed_actor = (cause.actors or [None])[0]
            return next(
                (
                    e
                    for e in self._events
                    if e.type == "escalation_chain"
                    and e.facts.get("trigger") == "post_departure_reroute"
                    and e.facts.get("departed") == departed_actor
                ),
                None,
            )

        elif link_type == "deal_risk_propagation":
            return next(
                (
                    e
                    for e in self._events
                    if e.type == "sf_deals_risk_flagged" and e.day >= cause.day
                ),
                None,
            )

        elif link_type == "onboarding_path":
            new_hire = cause.facts.get("name")
            return next(
                (
                    e
                    for e in self._events
                    if e.type == "onboarding_session" and new_hire in e.actors
                ),
                None,
            )

        elif link_type == "assignment_domain_mismatch":
            # Look for a knowledge_gap_detected on the same ticket on a later day,
            # or an incident_opened whose gap_areas overlap with the mismatch domains.
            ticket_id = cause.facts.get("ticket_id")
            mismatch_actors = set(cause.actors or [])
            for e in self._events:
                if e.day < cause.day:
                    continue
                if e.type == "knowledge_gap_detected":
                    if ticket_id and e.artifact_ids.get("jira") == ticket_id:
                        return e
                    # Also match on overlapping actors (the assigned engineer surfaces the gap)
                    if mismatch_actors & set(e.actors or []):
                        return e
                if e.type == "incident_opened":
                    gap_areas = e.facts.get("gap_areas", [])
                    mismatch_domains = cause.facts.get("assignment_risk_domains", [])
                    if (
                        gap_areas
                        and mismatch_domains
                        and set(gap_areas) & set(mismatch_domains)
                    ):
                        return e

        elif link_type == "jira_from_vendor_email":
            email_id = (cause.artifact_ids or {}).get("email", "")
            if not email_id:
                return None
            for e in self._events:
                if (
                    e.type == "jira_ticket_created"
                    and e.facts.get("source") == "vendor_email"
                    and email_id
                    in (
                        (e.artifact_ids or {}).get("source_email", ""),
                        str(e.facts.get("causal_chain", [])),
                    )
                    and e.day >= cause.day
                ):
                    return e
            return None

        elif link_type == "jira_from_customer_email":
            email_id = (cause.artifact_ids or {}).get("email", "")
            if not email_id:
                return None
            for e in self._events:
                if (
                    e.type == "jira_ticket_created"
                    and e.facts.get("source") == "customer_email"
                    and (e.artifact_ids or {}).get("source_email") == email_id
                    and e.day >= cause.day
                ):
                    return e
            return None

        elif link_type == "customer_escalation_relayed":
            source_email = (cause.artifact_ids or {}).get("source_email", "")
            for e in self._events:
                if (
                    e.type == "customer_escalation"
                    and (e.artifact_ids or {}).get("source_email") == source_email
                ):
                    return e
            return None

        elif link_type == "incident_handoff":
            departed = (cause.actors or [None])[0]
            for e in self._events:
                if (
                    e.type == "escalation_chain"
                    and e.facts.get("trigger") == "forced_handoff_on_departure"
                    and departed in (e.actors or [])
                    and e.day >= cause.day
                ):
                    return e
            return None

        elif link_type == "pr_gap_detected":
            pr_id = (cause.artifact_ids or {}).get("pr")
            if not pr_id:
                return None
            for e in self._events:
                if (
                    e.type == "knowledge_gap_detected"
                    and e.facts.get("detection_method") == "reviewer_audit"
                    and (e.artifact_ids or {}).get("pr") == pr_id
                    and e.day >= cause.day
                ):
                    return e
            return None

        elif link_type == "async_gap_detected":
            for e in self._events:
                if (
                    e.type == "knowledge_gap_detected"
                    and e.facts.get("detection_method") == "async_thread_classification"
                    and e.day >= cause.day
                ):
                    slack_id = (cause.artifact_ids or {}).get("slack_thread")
                    if (
                        slack_id
                        and (e.artifact_ids or {}).get("slack_thread") == slack_id
                    ):
                        return e
            return None

        elif link_type == "doc_gap_detected":
            conf_id = (cause.artifact_ids or {}).get("confluence")
            if not conf_id:
                return None
            gap_domains = set(cause.facts.get("topics_beyond_expertise", []))
            if not gap_domains:
                return None
            for e in self._events:
                if e.type == "incident_opened" and e.day > cause.day:
                    incident_tokens = set()
                    for entry in e.facts.get("gap_areas", []):
                        for part in str(entry).split(","):
                            part = part.strip().lower()
                            if part:
                                incident_tokens.add(part)

                    gap_tokens = set()
                    for phrase in gap_domains:
                        gap_tokens.add(phrase.strip().lower())
                        for word in phrase.lower().replace("-", " ").split():
                            if len(word) > 4:
                                gap_tokens.add(word)

                    if gap_tokens & incident_tokens:
                        return e
            return None

        elif link_type == "centrality_vacuum":
            for e in self._events:
                if (
                    e.type == "knowledge_gap_detected"
                    and e.facts.get("trigger") == "centrality_vacuum"
                    and e.day >= cause.day
                    and e.mongo_id != cause.mongo_id
                ):
                    return e
            return None

        elif link_type == "sf_stage_advanced_by_customer":
            email_id = (cause.artifact_ids or {}).get("email", "")
            if not email_id:
                return None
            for e in self._events:
                if (
                    e.type == "crm_touchpoint"
                    and e.facts.get("triggered_by") == email_id
                    and e.day >= cause.day
                ):
                    return e
            return None

        elif link_type == "feature_request_fyi":
            email_id = (cause.artifact_ids or {}).get("email", "")
            if not email_id:
                return None
            for e in self._events:
                if (
                    e.type == "normal_day_slack"
                    and "feature_request" in (e.tags or [])
                    and "fyi" in (e.tags or [])
                    and email_id in str(e.artifact_ids)
                    and e.day >= cause.day
                ):
                    return e
            return None

        elif link_type == "proactive_outreach_from_crm_signal":
            opp_id = (cause.artifact_ids or {}).get("sf_opp", "")
            if not opp_id:
                return None
            for e in self._events:
                if (
                    e.type == "proactive_outreach_initiated"
                    and (e.artifact_ids or {}).get("sf_opp") == opp_id
                    and e.day >= cause.day
                ):
                    return e
            return None

        elif link_type == "ticket_completion_notifies_lead":
            jira_id = (cause.artifact_ids or {}).get("jira", "")
            if not jira_id:
                return None
            for e in self._events:
                if (
                    e.type == "ticket_progress"
                    and (e.artifact_ids or {}).get("jira") == jira_id
                    and e.facts.get("new_status") == "Done"
                    and e.day >= cause.day
                ):
                    return e
            return None

        elif link_type == "org_collision_tension":
            for e in self._events:
                if (
                    e.type == "org_collision"
                    and e.facts.get("tension") in ("conflict", "alignment")
                    and e.day >= cause.day
                    and set(e.actors or []) & set(cause.actors or [])
                ):
                    return e
            return None

        elif link_type == "incident_triggers_risk_flag":
            jira_id = (cause.artifact_ids or {}).get("jira", "")
            if not jira_id:
                return None
            for e in self._events:
                if (
                    e.type == "sf_deals_risk_flagged"
                    and e.day >= cause.day
                    and (jira_id in str(e.artifact_ids) or jira_id in str(e.facts))
                ):
                    return e
            return None

        elif link_type == "review_triggers_revision":
            pr_id = (cause.artifact_ids or {}).get("pr", "")
            if not pr_id:
                return None
            for e in self._events:
                if (
                    e.type == "pr_review"
                    and (e.artifact_ids or {}).get("pr") == pr_id
                    and e.facts.get("verdict") == "approved"
                    and e.day >= cause.day
                    and e is not cause
                ):
                    return e
            return None

        elif link_type == "hire_fills_knowledge_gap":
            departed_domains = cause.facts.get("knowledge_domains", [])
            if not departed_domains:
                return None
            for e in self._events:
                if e.type == "employee_hired" and e.day > cause.day:
                    expertise = e.facts.get("expertise", [])
                    if set(d.lower() for d in departed_domains) & set(
                        d.lower() for d in expertise
                    ):
                        return e
            return None

        elif link_type == "escalation_from_zendesk":
            jira_id = (cause.artifact_ids or {}).get("jira", "")
            ticket_ids = cause.facts.get("ticket_ids", [])
            if not (jira_id or ticket_ids):
                return None
            for e in self._events:
                if e.type == "incident_opened" and e.day <= cause.day:
                    incident_jira = (e.artifact_ids or {}).get("jira", "")
                    if incident_jira == jira_id:
                        return e
                    if any(tid in str(e.facts) for tid in ticket_ids):
                        return e
            return None

        return None

    def _counterfactual_template(
        self, link_type: str, cause: SimEvent, effect: SimEvent
    ) -> Tuple[str, str, bool]:
        """
        Returns (premise, outcome, outcome_changed) as deterministic strings.
        These become the ground_truth fields — no LLM involvement here.
        """
        if link_type == "involves_gap":
            gap_areas = cause.facts.get("gap_areas", ["unknown domain"])
            gap_str = ", ".join(gap_areas)
            actor = (
                cause.facts.get("departed_employee")
                or (cause.actors or ["the departing engineer"])[0]
            )
            return (
                f"{actor} had fully documented {gap_str} before departing",
                "the subsequent incident would have been diagnosed faster or avoided entirely",
                True,
            )

        elif link_type == "recurrence_of":
            original_jira = (effect.artifact_ids or {}).get(
                "jira", "the original incident"
            )
            recurring_jira = (cause.artifact_ids or {}).get(
                "jira", "the recurring incident"
            )
            gap_days = cause.facts.get("recurrence_gap_days", "")
            gap_str = f" {gap_days} days later" if gap_days else ""
            return (
                f"the root cause of {original_jira} had been fully addressed after it was first resolved",
                f"{recurring_jira} would not have occurred{gap_str}",
                True,
            )

        elif link_type == "postmortem_from_incident":
            jira_id = (cause.artifact_ids or {}).get("jira", "the incident")
            return (
                f"incident {jira_id} had not occurred",
                "the postmortem page would not have been written and the root cause analysis would not exist",
                True,
            )

        elif link_type == "spawned_doc":
            topic = cause.facts.get("topic", "the design discussion")
            conf_id = (effect.artifact_ids or {}).get(
                "confluence", "the Confluence doc"
            )
            premise = f"the discussion about '{topic}' had not been documented"
            outcome = f"{conf_id} would not exist and related decisions would remain undocumented"
            return premise, outcome, True

        elif link_type == "email_dropped":
            source = cause.facts.get("source", "the sender")
            subject = cause.facts.get("subject", "the email")[:80]
            reason = cause.facts.get("reason", "no_action_taken").replace("_", " ")
            return (
                f"the email from {source} ('{subject}') had been actioned rather than dropped due to {reason}",
                "a support ticket or escalation would have been opened and the issue tracked",
                True,
            )

        elif link_type == "sf_ownership_lapsed":
            actor = (cause.actors or ["the departed employee"])[0]
            accounts = effect.facts.get("lapsed_accounts", [])
            acc_str = ", ".join(accounts[:3]) if accounts else "affected accounts"
            premise = (
                f"{actor}'s Salesforce accounts had been reassigned before departure"
            )
            outcome = (
                f"{acc_str} would not have lost ownership and pipeline would be intact"
            )
            return premise, outcome, True

        elif link_type == "zd_escalation_source":
            ticket_ids = effect.facts.get("ticket_ids", ["the support ticket"])
            tickets_str = ", ".join(ticket_ids[:3])
            premise = f"{tickets_str} had been resolved at the support level"
            outcome = "the incident escalation would not have occurred"
            return premise, outcome, True

        elif link_type == "blocker_flagged":
            reason = cause.facts.get("blocker_reason", "a technical blocker")
            jira_id = effect.artifact_ids.get("jira", "the ticket")
            return (
                f"the blocker regarding '{reason}' had been resolved immediately",
                f"work on {jira_id} would have progressed without delay",
                True,
            )

        elif link_type == "incident_coordination":
            contact = effect.facts.get("external_party", "the external contact")
            return (
                f"the incident on Day {cause.day} had not occurred",
                f"no coordination with {contact} would have been needed at all",
                True,
            )

        elif link_type == "departure_reassignment":
            actor = (cause.actors or ["the employee"])[0]
            return (
                f"{actor} had not departed the company",
                "their active tickets and escalation responsibilities would not have been reassigned",
                True,
            )

        elif link_type == "deal_risk_propagation":
            jira_id = cause.artifact_ids.get("jira", "the incident")
            return (
                f"the incident {jira_id} had not occurred",
                "the associated Salesforce deals would not have been flagged as at-risk",
                True,
            )

        elif link_type == "onboarding_path":
            name = cause.facts.get("name", "the new hire")
            return (
                f"{name} had not been hired on Day {cause.day}",
                "the onboarding sessions and warmup meetings for them would not have taken place",
                True,
            )

        elif link_type == "assignment_domain_mismatch":
            actors = cause.actors or ["the engineer"]
            ticket_id = cause.facts.get("ticket_id", "the ticket")
            coverage = cause.facts.get("documentation_coverage")
            coverage_str = (
                f" (documentation coverage: {int(coverage * 100)}%)" if coverage else ""
            )
            return (
                f"{actors[0]} had been assigned to {ticket_id} with matching domain expertise",
                f"the knowledge gap{coverage_str} would likely not have been surfaced and the associated incident risk reduced",
                True,
            )

        elif link_type == "jira_from_vendor_email":
            vendor = cause.facts.get("source", (cause.actors or ["the vendor"])[0])
            org = cause.facts.get("org", "")
            org_str = f" ({org})" if org else ""
            return (
                f"the email from {vendor}{org_str} had not arrived",
                "a Jira ticket would not have been created to track the request",
                True,
            )

        elif link_type == "jira_from_customer_email":
            customer = cause.facts.get("source", (cause.actors or ["the customer"])[0])
            org = cause.facts.get("org", "")
            org_str = f" ({org})" if org else ""
            return (
                f"the email from {customer}{org_str} had not been received or had been dropped",
                "the resulting Jira ticket would not have been opened",
                True,
            )

        elif link_type == "customer_escalation_relayed":
            customer = cause.facts.get("source", "the customer")
            slack_id = (cause.artifact_ids or {}).get(
                "slack_thread", "the Slack thread"
            )
            return (
                f"the email from {customer} had been dropped rather than relayed",
                f"{slack_id} would not have been created and Product would not have been notified",
                True,
            )

        elif link_type == "incident_handoff":
            departed = (cause.actors or ["the employee"])[0]
            return (
                f"{departed} had not departed and a forced handoff had not been required",
                "the escalation chain would not have been transferred mid-incident",
                True,
            )

        elif link_type == "pr_gap_detected":
            actor = (cause.actors or ["the reviewer"])[0]
            pr_id = (cause.artifact_ids or {}).get("pr", "the PR")
            return (
                f"the author of {pr_id} had domain expertise matching the review scope",
                f"{actor} would not have flagged a knowledge gap during review",
                True,
            )

        elif link_type == "async_gap_detected":
            actor = (cause.actors or ["the engineer"])[0]
            return (
                f"{actor} had prior knowledge of the domain being discussed",
                "the async thread would not have surfaced a knowledge gap",
                True,
            )

        elif link_type == "doc_gap_detected":
            author = cause.facts.get("author", (cause.actors or ["the author"])[0])
            conf_id = (cause.artifact_ids or {}).get("confluence", "the design doc")
            domains = cause.facts.get("topics_beyond_author_expertise", ["the domain"])
            domain_str = " and ".join(domains[:2])
            return (
                f"{author} had expertise in {domain_str} when writing {conf_id}",
                "the knowledge gap would not have been embedded in the design doc, "
                "and the downstream incident would not have occurred",
                True,
            )

        elif link_type == "centrality_vacuum":
            actor = (cause.actors or ["the central actor"])[0]
            gap = cause.facts.get("gap_domain", "the domain")
            return (
                f"{actor} had not been the sole knowledge holder for {gap}",
                "the centrality vacuum would not have created a knowledge gap",
                True,
            )

        elif link_type == "sf_stage_advanced_by_customer":
            customer = cause.facts.get("source", (cause.actors or ["the customer"])[0])
            account = cause.facts.get("org", "the account")
            opp_id = (effect.artifact_ids or {}).get("sf_opp", "the opportunity")
            return (
                f"the email from {customer} ({account}) had not been received",
                f"{opp_id} would not have advanced to the next stage",
                True,
            )

        elif link_type == "feature_request_fyi":
            customer = cause.facts.get("source", (cause.actors or ["the customer"])[0])
            return (
                f"the feature request email from {customer} had not been received",
                "the FYI thread in #product would not have been created",
                True,
            )

        elif link_type == "proactive_outreach_from_crm_signal":
            account = cause.facts.get("account_name", "the account")
            opp_id = (cause.artifact_ids or {}).get("sf_opp", "the opportunity")
            return (
                f"the CRM signal for {account} ({opp_id}) had not been logged",
                "the proactive outreach email would not have been sent",
                True,
            )

        elif link_type == "ticket_completion_notifies_lead":
            jira_id = (effect.artifact_ids or {}).get("jira", "the ticket")
            recipient = effect.facts.get("to", "the lead")
            return (
                f"{jira_id} had not reached Done status",
                f"the completion notification to {recipient} would not have been sent",
                True,
            )

        elif link_type == "org_collision_tension":
            actors = cause.actors or ["the parties involved"]
            tension = cause.facts.get("tension", "tension")
            return (
                f"the {tension} between {' and '.join(actors[:2])} had not occurred",
                "the interpersonal collision event would not have been logged",
                True,
            )

        elif link_type == "incident_triggers_risk_flag":
            jira_id = (cause.artifact_ids or {}).get("jira", "the incident")
            accounts = effect.facts.get("affected_accounts", [])
            acc_str = ", ".join(accounts[:3]) if accounts else "associated accounts"
            return (
                f"incident {jira_id} had not occurred or had been resolved immediately",
                f"{acc_str} would not have been flagged as at-risk in Salesforce",
                True,
            )

        elif link_type == "review_triggers_revision":
            reviewer = cause.facts.get(
                "reviewer", (cause.actors or ["the reviewer"])[0]
            )
            author = cause.facts.get("author", (effect.actors or ["the author"])[0])
            return (
                f"{reviewer} had approved the pull request on Day {cause.day} without requesting changes",
                f"{author} would not have revised the implementation before merge",
                True,
            )

        elif link_type == "hire_fills_knowledge_gap":
            departed = (cause.actors or ["the former employee"])[0]
            domains = cause.facts.get("knowledge_domains", [])
            domain_str = ", ".join(domains[:3])
            name = effect.facts.get("name", (effect.actors or ["the new hire"])[0])
            return (
                f"{name} had not been hired to fill the gap in {domain_str}",
                f"the knowledge vacuum left by {departed}'s departure would remain unaddressed",
                True,
            )

        elif link_type == "escalation_from_zendesk":
            ticket_ids = cause.facts.get("ticket_ids", ["the support tickets"])
            tickets_str = ", ".join(ticket_ids[:3])
            jira_id = (effect.artifact_ids or {}).get("jira", "the incident")
            return (
                f"the Zendesk tickets ({tickets_str}) had been resolved at the support level",
                f"incident {jira_id} would not have been opened",
                True,
            )

        return (
            "the causal condition had been different",
            "the outcome would have changed",
            True,
        )

    def build(self) -> List[CausalLink]:
        links: List[CausalLink] = []

        for link_type in _EXPLICIT_CAUSAL_LINKS:
            if link_type == "involves_gap":
                cause_events = [
                    e
                    for e in self._events
                    if e.type == "knowledge_gap_detected"
                    and e.facts.get("trigger") != "centrality_vacuum"
                    and e.facts.get("detection_method")
                    not in (
                        "reviewer_audit",
                        "async_thread_classification",
                    )
                ]
            elif link_type == "recurrence_of":
                cause_events = [
                    e
                    for e in self._events
                    if e.type == "incident_opened" and e.facts.get("recurrence_of")
                ]
            elif link_type == "spawned_doc":
                cause_events = [
                    e
                    for e in self._events
                    if e.type == "design_discussion" and e.facts.get("spawned_doc")
                ]
            elif link_type == "email_dropped":
                cause_events = [e for e in self._events if e.type == "email_dropped"]
            elif link_type == "sf_ownership_lapsed":
                cause_events = [
                    e for e in self._events if e.type == "employee_departed"
                ]
            elif link_type == "zd_escalation_source":
                cause_events = [e for e in self._events if e.type == "incident_opened"]
            elif link_type == "blocker_flagged":
                cause_events = [e for e in self._events if e.type == "blocker_flagged"]
            elif link_type == "incident_coordination":
                cause_events = [e for e in self._events if e.type == "incident_opened"]
            elif link_type == "departure_reassignment":
                cause_events = [
                    e for e in self._events if e.type == "employee_departed"
                ]
            elif link_type == "assignment_domain_mismatch":
                cause_events = [
                    e for e in self._events if e.type == "assignment_domain_mismatch"
                ]
            elif link_type == "jira_from_vendor_email":
                cause_events = [
                    e
                    for e in self._events
                    if e.type == "inbound_external_email"
                    and e.facts.get("category") == "vendor"
                ]

            elif link_type == "jira_from_customer_email":
                cause_events = [
                    e
                    for e in self._events
                    if e.type == "inbound_external_email"
                    and e.facts.get("category") == "customer"
                    and e.facts.get("email_type") != "escalation"
                ]

            elif link_type == "customer_escalation_relayed":
                cause_events = [
                    e
                    for e in self._events
                    if e.type == "inbound_external_email"
                    and e.facts.get("category") == "customer"
                ]

            elif link_type == "incident_handoff":
                cause_events = [
                    e for e in self._events if e.type == "employee_departed"
                ]

            elif link_type == "pr_gap_detected":
                cause_events = [e for e in self._events if e.type == "pr_review"]

            elif link_type == "async_gap_detected":
                cause_events = [e for e in self._events if e.type == "async_question"]

            elif link_type == "doc_gap_detected":
                cause_events = [
                    e
                    for e in self._events
                    if e.type == "knowledge_gap_detected"
                    and e.facts.get("detection_method") == "author_self_audit"
                    and e.facts.get(
                        "topics_beyond_expertise"
                    )  # must have domains to match on
                ]

            elif link_type == "centrality_vacuum":
                cause_events = [
                    e
                    for e in self._events
                    if e.type == "knowledge_gap_detected"
                    and e.facts.get("trigger") == "centrality_vacuum"
                ]

            elif link_type == "sf_stage_advanced_by_customer":
                cause_events = [
                    e
                    for e in self._events
                    if e.type == "inbound_external_email"
                    and e.facts.get("category") == "customer"
                ]

            elif link_type == "feature_request_fyi":
                cause_events = [
                    e
                    for e in self._events
                    if e.type == "inbound_external_email"
                    and "feature" in " ".join(e.tags or []).lower()
                ]

            elif link_type == "proactive_outreach_from_crm_signal":
                cause_events = [
                    e
                    for e in self._events
                    if e.type == "crm_touchpoint"
                    and e.facts.get("triggered_by") is None
                ]

            elif link_type == "ticket_completion_notifies_lead":
                cause_events = [
                    e for e in self._events if e.type == "ticket_completion_email"
                ]

            elif link_type == "postmortem_from_incident":
                cause_events = [e for e in self._events if e.type == "incident_opened"]

            elif link_type == "org_collision_tension":
                cause_events = [e for e in self._events if e.type == "org_collision"]

            elif link_type == "incident_triggers_risk_flag":
                cause_events = [e for e in self._events if e.type == "incident_opened"]

            elif link_type == "review_triggers_revision":
                cause_events = [
                    e
                    for e in self._events
                    if e.type == "pr_review"
                    and e.facts.get("verdict") == "changes_requested"
                ]

            elif link_type == "hire_fills_knowledge_gap":
                cause_events = [
                    e
                    for e in self._events
                    if e.type == "employee_departed"
                    and e.facts.get("knowledge_domains")
                ]

            elif link_type == "escalation_from_zendesk":
                cause_events = [
                    e for e in self._events if e.type == "zd_tickets_escalated"
                ]

            else:
                logger.debug(
                    f"[causal_index] No cause-event selector for link_type={link_type!r} — skipping"
                )
                continue

            cause_events = list(cause_events)
            random.shuffle(cause_events)
            links_this_type = 0

            for cause in cause_events:
                if links_this_type >= self.MAX_LINKS_PER_TYPE:
                    break

                effect = self._find_effect_event(link_type, cause)
                if not effect:
                    continue

                if link_type == "involves_gap":
                    link_value = str(cause.facts.get("gap_areas", ""))
                    cleaned = link_value.strip("[]'\" ").lower()
                    if cleaned in _VAGUE_GAP_TERMS:
                        logger.debug(
                            f"[causal_index] Skipping vague involves_gap: {link_value}"
                        )
                        continue

                premise, outcome, changed = self._counterfactual_template(
                    link_type, cause, effect
                )

                subsystems = self._subsystems_for_event(
                    cause
                ) | self._subsystems_for_event(effect)

                link_field = {
                    "involves_gap": "gap_areas",
                    "recurrence_of": "prior_postmortem",
                    "spawned_doc": "spawned_doc",
                    "email_dropped": "email",
                    "sf_ownership_lapsed": "actor",
                    "zd_escalation_source": "jira",
                    "blocker_flagged": "jira",
                    "incident_coordination": "jira",
                    "departure_reassignment": "actor",
                    "assignment_domain_mismatch": "ticket_id",
                    "jira_from_customer_email": "email",
                    "jira_from_vendor_email": "email",
                    "customer_escalation_relayed": "email",
                    "incident_handoff": "actor",
                    "pr_gap_detected": "pr",
                    "async_gap_detected": "slack_thread",
                    "centrality_vacuum": "gap_domain",
                    "sf_stage_advanced_by_customer": "sf_opp",
                    "feature_request_fyi": "email",
                    "proactive_outreach_from_crm_signal": "sf_opp",
                    "ticket_completion_notifies_lead": "jira",
                    "org_collision_tension": "actor",
                    "postmortem_from_incident": "jira",
                    "incident_triggers_risk_flag": "jira",
                    "review_triggers_revision": "pr",
                    "hire_fills_knowledge_gap": "actor",
                    "escalation_from_zendesk": "jira",
                    "doc_gap_detected": "confluence",
                }.get(link_type, "")

                if link_type == "recurrence_of":
                    recurrence_jira_id = (cause.artifact_ids or {}).get("jira", "")
                    if recurrence_jira_id:
                        recurrence_ticket = self._mem._db["jira_tickets"].find_one(
                            {"id": recurrence_jira_id},
                            {"prior_postmortem": 1, "recurrence_of": 1},
                        )
                        if recurrence_ticket:
                            link_value = (
                                recurrence_ticket.get("prior_postmortem")
                                or recurrence_ticket.get("recurrence_of")
                                or ""
                            )
                        else:
                            link_value = ""
                    else:
                        link_value = ""
                else:
                    link_value = str(
                        cause.facts.get(link_field, "")
                        or (cause.artifact_ids or {}).get(link_field, "")
                        or (cause.actors or [""])[0]
                    )

                links_this_type += 1

                if link_type == "recurrence_of":
                    link_cause_id, link_cause_type = effect.mongo_id, effect.type
                    link_effect_id, link_effect_type = cause.mongo_id, cause.type
                    link_day = effect.day
                else:
                    link_cause_id, link_cause_type = cause.mongo_id, cause.type
                    link_effect_id, link_effect_type = effect.mongo_id, effect.type
                    link_day = cause.day

                links.append(
                    CausalLink(
                        link_type=link_type,
                        cause_event_id=link_cause_id,
                        cause_event_type=link_cause_type,
                        effect_event_id=link_effect_id,
                        effect_event_type=link_effect_type,
                        actors=list(set((cause.actors or []) + (effect.actors or []))),
                        day=link_day,
                        link_field=link_field,
                        link_value=link_value,
                        subsystems_involved=subsystems,
                        counterfactual_premise=premise,
                        counterfactual_outcome=outcome,
                        outcome_changed=changed,
                    )
                )
            logger.info(
                f"[causal_index] {link_type}: {len(cause_events)} candidates, "
                f"{links_this_type} linked, "
                f"{len(cause_events) - links_this_type} with no effect found"
            )

        by_type_counts = defaultdict(int)
        for lnk in links:
            by_type_counts[lnk.link_type] += 1
        logger.info(
            f"[causal_index] {len(links)} explicit causal links indexed "
            f"(cap={self.MAX_LINKS_PER_TYPE}/type, {len(by_type_counts)} types covered)"
        )
        return links


class AbsenceCatalogBuilder:
    """
    Builds the catalog of expected-but-absent artifact pairs.

    For each pair in _SILENCE_PAIRS, scans the event log for trigger events
    that have no matching response event. The state machine is the arbiter:
    if no response event fired, the absence is ground truth.

    Also derives expected_search_space: the set of artifact IDs the agent
    must check before concluding absence. This is what separates a well-reasoned
    "no" from a lucky guess.
    """

    def __init__(self, mem: Memory):
        self._mem = mem
        self._events: List[SimEvent] = mem.get_event_log(from_db=True)

    def _dept_of_actor(self, actor: str) -> Optional[str]:
        for dept, members in CONFIG["org_chart"].items():
            if actor in members:
                return dept.lower().replace(" ", "_")
        return None

    def _match_key(self, event: SimEvent, link_field: str) -> Optional[str]:
        val = (event.artifact_ids or {}).get(link_field)
        if val:
            return val
        val = event.facts.get(link_field)
        if val:
            return str(val)

        if link_field == "gap_domain" and event.type == "knowledge_gap_detected":
            method = event.facts.get("detection_method", "")

            if method in ("embedding_similarity", "centrality_vacuum"):
                areas = event.facts.get("gap_areas")
                if areas:
                    first = areas[0] if isinstance(areas, list) else areas
                    if str(first).strip().lower() not in _VAGUE_GAP_TERMS:
                        return str(first)

            elif method == "reviewer_audit":
                pr = (event.artifact_ids or {}).get("pr")
                if pr:
                    return str(pr)

            elif method == "author_self_audit":
                return None

        if link_field == "actor" and event.actors:
            return event.actors[0]

        return None

    def _expected_search_space(
        self,
        trigger: SimEvent,
        expected_response_type: str,
        artifact_ids_override: Optional[dict] = None,
    ) -> List[str]:
        artifact_ids = artifact_ids_override or trigger.artifact_ids
        search_space: List[str] = list(_safe_artifact_values(artifact_ids))

        if expected_response_type == "postmortem_created":
            pass

        elif expected_response_type == "incident_resolved":
            jira_id = (artifact_ids or {}).get("jira", "")
            if jira_id:
                search_space.append(jira_id)

        elif expected_response_type == "zd_ticket_opened":
            email_id = (artifact_ids or {}).get("email", "")
            if email_id:
                search_space.append(email_id)

        elif expected_response_type == "customer_email_routed":
            email_id = (artifact_ids or {}).get("email", "")
            if email_id:
                search_space.append(email_id)

        elif expected_response_type == "confluence_created":
            zoom_id = (artifact_ids or {}).get("zoom_transcript", "")
            if zoom_id:
                search_space.append(zoom_id)
            slack_id = (artifact_ids or {}).get("slack", "")
            if slack_id:
                search_space.append(slack_id)

        elif expected_response_type == "sf_ownership_lapsed":
            pass

        elif expected_response_type == "ticket_reassigned":
            pass

        elif expected_response_type == "pr_merged":
            pr_id = (artifact_ids or {}).get("pr", "")
            if pr_id:
                search_space.append(pr_id)

        elif expected_response_type == "zd_tickets_escalated":
            jira_id = (artifact_ids or {}).get("jira", "")
            if jira_id:
                search_space.append(jira_id)

        elif expected_response_type == "onboarding_session":
            slack_thread = (artifact_ids or {}).get("slack_thread", "")
            if slack_thread:
                search_space.append(slack_thread)

        elif expected_response_type == "warmup_1on1":
            slack_thread = (artifact_ids or {}).get("slack_thread", "")
            if slack_thread:
                search_space.append(slack_thread)

        elif expected_response_type == "sf_deals_risk_flagged":
            jira_id = (artifact_ids or {}).get("jira", "")
            if jira_id:
                search_space.append(jira_id)

        elif expected_response_type == "jira_from_vendor_email":
            email_id = (artifact_ids or {}).get("email", "")
            if email_id:
                search_space.append(email_id)

        elif expected_response_type == "knowledge_gap_detected":
            ticket_id = trigger.facts.get("ticket_id", "")
            if ticket_id:
                search_space.append(ticket_id)
            pr_id = (artifact_ids or {}).get("pr", "")
            confluence_id = (artifact_ids or {}).get("confluence", "")
            slack_thread = (artifact_ids or {}).get("slack_thread", "")
            if pr_id:
                search_space.append(pr_id)
            if confluence_id:
                search_space.append(confluence_id)
            if slack_thread:
                search_space.append(slack_thread)

        return list(dict.fromkeys(search_space))

    def build(self) -> Tuple[List[AbsenceRecord], List[AbsenceRecord]]:
        records: List[AbsenceRecord] = []
        confirmed: List[AbsenceRecord] = []

        for trigger_type, response_type, link_field in _SILENCE_PAIRS:
            trigger_events = [
                e for e in self._events if e.type == trigger_type and e.day >= 1
            ]

            for trigger in trigger_events:
                trigger_artifacts = _safe_artifact_values(trigger.artifact_ids)
                link_key = self._match_key(trigger, link_field)

                if trigger_type == "inbound_external_email":
                    category = trigger.facts.get("category", "")
                    if (
                        response_type == "customer_email_routed"
                        and category == "vendor"
                    ):
                        continue
                    if (
                        response_type == "jira_from_vendor_email"
                        and category != "vendor"
                    ):
                        continue

                detection_method = trigger.facts.get("detection_method", "")
                if (
                    trigger_type == "knowledge_gap_detected"
                    and detection_method == "author_self_audit"
                ):
                    audited_confluence = (trigger.artifact_ids or {}).get("confluence")
                    if audited_confluence:
                        trigger_artifacts = trigger_artifacts - {
                            str(audited_confluence)
                        }

                if response_type == "confluence_created" and (
                    trigger.facts.get("spawned_doc")
                    or "confluence" in (trigger.artifact_ids or {})
                ):
                    continue

                response_found = False

                for e in self._events:
                    if e.type != response_type or e.day < trigger.day:
                        continue

                    e_match_key = self._match_key(e, link_field)

                    if (
                        link_key
                        and e_match_key is not None
                        and (
                            e_match_key == link_key
                            or link_key in str(e.artifact_ids)
                            or link_key in str(e.facts)
                        )
                    ):
                        response_found = True
                        break

                    response_artifacts = _safe_artifact_values(e.artifact_ids)
                    if trigger_artifacts & response_artifacts:
                        response_found = True
                        break

                subsystem = _RESPONSE_TYPE_SUBSYSTEM.get(
                    response_type,
                    _ARTIFACT_SUBSYSTEM.get(
                        next(iter(trigger.artifact_ids or {}), ""), "default"
                    ),
                )
                search_space = self._expected_search_space(trigger, response_type)

                trigger_artifact_ids = dict(trigger.artifact_ids or {})

                if (
                    response_type == "sf_ownership_lapsed"
                    and trigger_type == "employee_departed"
                ):
                    actor = (trigger.actors or [""])[0]
                    if actor:
                        owned_accounts = [
                            doc["account_id"]
                            for doc in self._mem._db["sf_accounts"].find(
                                {"owner": actor}, {"account_id": 1, "_id": 0}
                            )
                            if doc.get("account_id")
                        ]
                        owned_opps = [
                            doc["opportunity_id"]
                            for doc in self._mem._db["sf_opps"].find(
                                {
                                    "owner": actor,
                                    "stage": {"$nin": ["Closed Won", "Closed Lost"]},
                                },
                                {"opportunity_id": 1, "_id": 0},
                            )
                            if doc.get("opportunity_id")
                        ]
                        if owned_accounts:
                            trigger_artifact_ids["sf_accounts"] = owned_accounts
                        if owned_opps:
                            trigger_artifact_ids["sf_opps"] = owned_opps

                if (
                    response_type == "zd_ticket_opened"
                    and trigger_type == "customer_email_routed"
                ):
                    email_id = (trigger.artifact_ids or {}).get("email", "")
                    if email_id:
                        linked_tickets = [
                            doc["id"]
                            for doc in self._mem._db["zd_tickets"].find(
                                {"source_email_id": email_id}, {"id": 1, "_id": 0}
                            )
                            if doc.get("id")
                        ]
                        if linked_tickets:
                            trigger_artifact_ids["zd_ticket"] = linked_tickets

                if response_found:
                    response_event_artifacts = sorted(
                        _safe_artifact_values(e.artifact_ids)
                    )
                    enriched_search_space = list(
                        dict.fromkeys(search_space + response_event_artifacts)
                    )

                    confirmed_rec = AbsenceRecord(
                        trigger_event_id=trigger.mongo_id,
                        trigger_event_type=trigger_type,
                        expected_response_type=response_type,
                        trigger_day=trigger.day,
                        trigger_actors=list(dict.fromkeys(trigger.actors or [])),
                        trigger_artifact_ids=dict(trigger.artifact_ids or {}),
                        link_field=link_field,
                        link_value=link_key or "N/A",
                        subsystem=subsystem,
                        expected_search_space=enriched_search_space,
                    )
                    confirmed.append(confirmed_rec)
                else:
                    absence_rec = AbsenceRecord(
                        trigger_event_id=trigger.mongo_id,
                        trigger_event_type=trigger_type,
                        expected_response_type=response_type,
                        trigger_day=trigger.day,
                        trigger_actors=list(dict.fromkeys(trigger.actors or [])),
                        trigger_artifact_ids=dict(trigger.artifact_ids or {}),
                        link_field=link_field,
                        link_value=link_key or "N/A",
                        subsystem=subsystem,
                        expected_search_space=search_space,
                    )
                    records.append(absence_rec)

        logger.info(f"[absence_catalog] {len(records)} absence records cataloged")
        logger.info(
            f"[absence_catalog] {len(confirmed)} confirmed response records cataloged"
        )

        MAX_PER_TRIGGER = 8
        capped: List[AbsenceRecord] = []
        trigger_counts: Dict[str, int] = defaultdict(int)
        random.shuffle(records)
        for r in records:
            if trigger_counts[r.trigger_event_type] < MAX_PER_TRIGGER:
                capped.append(r)
                trigger_counts[r.trigger_event_type] += 1
        logger.info(
            f"[absence_catalog] Capped to {len(capped)} records ({MAX_PER_TRIGGER}/trigger_type)"
        )

        confirmed_capped: List[AbsenceRecord] = []
        confirmed_counts: Dict[str, int] = defaultdict(int)
        random.shuffle(confirmed)
        for r in confirmed:
            if confirmed_counts[r.trigger_event_type] < MAX_PER_TRIGGER:
                confirmed_capped.append(r)
                confirmed_counts[r.trigger_event_type] += 1

        return capped, confirmed_capped


class EvalQuestionGenerator:
    """
    Generates PERSPECTIVE, COUNTERFACTUAL, and SILENCE questions.

    Ground truth is always derived deterministically from the three indexes.
    LLMs only write question prose, and every generated question is validated
    against a structured rubric before inclusion.

    Question prose validation checks:
    - Ends with a question mark
    - Does not contain the ground truth answer verbatim
    - Does not name an artifact ID directly (keeps questions natural-language)
    - Is unambiguous — references the actor/day/subsystem constraint explicitly
    """

    MAX_PERSPECTIVE = 30
    MAX_COUNTERFACTUAL = 40
    MAX_SILENCE = 30

    def __init__(
        self,
        mem: Memory,
        worker_llm,
        visibility_map: Dict[str, List[ActorVisibilityCone]],
        causal_links: List[CausalLink],
        absence_catalog: List[AbsenceRecord],
        confirmed_catalog: List[AbsenceRecord],
    ):
        self._mem = mem
        self._worker_llm = worker_llm
        self._visibility_map = visibility_map
        self._causal_links = causal_links
        self._absence_catalog = absence_catalog
        self._confirmed_catalog = confirmed_catalog
        self._events: List[SimEvent] = mem.get_event_log(from_db=True)

    def generate(self) -> List[dict]:
        questions: List[dict] = []

        logger.info("[eval] Generating PERSPECTIVE questions...")
        questions.extend(self._perspective_questions())

        logger.info("[eval] Generating COUNTERFACTUAL questions...")
        questions.extend(self._counterfactual_questions())

        logger.info("[eval] Generating SILENCE questions...")
        questions.extend(self._silence_questions())

        random.shuffle(questions)

        logger.info(f"[eval] {len(questions)} total questions generated")
        return questions

    def _build_negative_counterfactual(
        self,
        link: CausalLink,
        effect_cause_map: Dict[str, List[CausalLink]],
    ) -> Optional[dict]:
        """
        Generate a counterfactual where outcome_changed is False.

        Strategy: The effect event has MULTIPLE independent causes in the
        causal link index. Removing THIS cause would not have prevented the
        effect because at least one alternative cause is sufficient.

        Example: If three knowledge gaps (EVT-12, EVT-27, EVT-32) all
        contributed to incident EVT-42, removing any single gap would NOT
        have prevented the incident — the other gaps still exist.
        """
        sibling_causes = effect_cause_map.get(link.effect_event_id, [])
        alternatives = [
            s for s in sibling_causes if s.cause_event_id != link.cause_event_id
        ]

        if not alternatives:
            return None

        alternative = alternatives[0]

        cause_event = next(
            (e for e in self._events if e.mongo_id == link.cause_event_id), None
        )
        effect_event = next(
            (e for e in self._events if e.mongo_id == link.effect_event_id), None
        )
        alt_cause_event = next(
            (e for e in self._events if e.mongo_id == alternative.cause_event_id), None
        )

        if not (cause_event and effect_event and alt_cause_event):
            return None

        premise_parts = {
            "involves_gap": lambda: (
                f"{(cause_event.facts.get('departed_employee') or (cause_event.actors or ['the engineer'])[0])} "
                f"had fully documented {', '.join(cause_event.facts.get('gap_areas', ['the domain']))}"
            ),
            "doc_gap_detected": lambda: (
                f"{cause_event.facts.get('author', (cause_event.actors or ['the author'])[0])} "
                f"had expertise in {' and '.join(cause_event.facts.get('topics_beyond_expertise', ['the domain'])[:2])} "
                f"when writing {(cause_event.artifact_ids or {}).get('confluence', 'the design doc')}"
            ),
            "recurrence_of": lambda: (
                f"the root cause of {(cause_event.artifact_ids or {}).get('jira', 'the original incident')} "
                f"had been fully addressed"
            ),
        }

        premise_fn = premise_parts.get(
            link.link_type,
            lambda: (
                f"the {link.link_type.replace('_', ' ')} on Day {link.day} had not occurred"
            ),
        )
        premise = premise_fn()

        alt_mechanism = alternative.link_type.replace("_", " ")
        alt_actor = (alt_cause_event.actors or ["another source"])[0]
        alt_day = alternative.day

        outcome = (
            f"the effect ({link.effect_event_type.replace('_', ' ')}) would still have "
            f"occurred due to an independent {alt_mechanism} "
            f"involving {alt_actor} on Day {alt_day}"
        )

        cause_artifacts = sorted(
            _safe_artifact_values(cause_event.artifact_ids if cause_event else {})
        )
        effect_artifacts = sorted(
            _safe_artifact_values(effect_event.artifact_ids if effect_event else {})
        )
        alt_cause_artifacts = sorted(
            _safe_artifact_values(
                alt_cause_event.artifact_ids if alt_cause_event else {}
            )
        )

        ground_truth = {
            "outcome_changed": False,
            "causal_mechanism": link.link_type,
            "causal_link_field": link.link_field,
            "causal_link_value": link.link_value,
            "cause_event_id": link.cause_event_id,
            "cause_event_type": link.cause_event_type,
            "effect_event_id": link.effect_event_id,
            "effect_event_type": link.effect_event_type,
            "premise": premise,
            "outcome": outcome,
            "actors": link.actors,
            "as_of_time": (_SIM_START + timedelta(days=link.day)).isoformat(),
            "evidence_chain_artifacts": {
                "cause": cause_artifacts,
                "effect": effect_artifacts,
            },
            "alternative_cause_event_id": alternative.cause_event_id,
            "alternative_cause_type": alternative.cause_event_type,
            "alternative_mechanism": alternative.link_type,
            "alternative_cause_artifacts": alt_cause_artifacts,
            "alternative_cause_count": len(sibling_causes) - 1,
        }

        difficulty = "hard" if len(link.subsystems_involved) > 1 else "medium"
        actors_str = (
            ", ".join(link.actors[:3]) if link.actors else "the involved parties"
        )

        _NEGATIVE_CF_DOMAIN_HINTS: Dict[str, str] = {
            "involves_gap": (
                "a knowledge gap that contributed to an incident, BUT the incident "
                "had multiple contributing knowledge gaps — removing just this one "
                "would not have prevented it"
            ),
            "doc_gap_detected": (
                "a documentation gap in a design doc that contributed to an incident, "
                "BUT other documentation gaps also contributed — removing just this "
                "author's gap would not have prevented the incident"
            ),
            "recurrence_of": (
                "a recurring incident linked to a prior unresolved issue, BUT the "
                "same effect was also caused by an independent path"
            ),
        }

        domain_hint = _NEGATIVE_CF_DOMAIN_HINTS.get(
            link.link_type,
            f"a {link.link_type.replace('_', ' ')} event that contributed to "
            f"an outcome, BUT the outcome had multiple independent causes — "
            f"removing just this one would not have changed it",
        )

        NEGATIVE_QUESTION_STYLES = [
            "Phrase it to suggest the cause was necessary: 'If X had not happened, would Y still have occurred?'",
            "Phrase it as a dependency question: 'Was Y dependent on X, or would it have happened regardless?'",
            "Phrase it from the effect perspective: 'Would Y have been prevented if only X had been different?'",
        ]

        style = random.choice(NEGATIVE_QUESTION_STYLES)

        template = (
            f"Write a counterfactual yes/no question about events on Day {link.day} "
            f"in a simulated company. {style} "
            f"The question must name the Day and the actors involved: {actors_str}. "
            f"The question is about: {domain_hint}. "
            f"Do not name specific artifact IDs or the causal mechanism label. "
            f"Do not reveal the answer or make the causal link obvious. "
            f"The question should require investigation to answer. "
            f"Output only the question text."
        )

        question_text = self._generate_and_validate_prose(
            template=template,
            ground_truth_str=outcome,
            question_type="COUNTERFACTUAL",
        )
        if not question_text:
            return None

        return {
            "question_id": f"counterfactual_neg_{link.cause_event_id}_{link.link_type}",
            "question_type": "COUNTERFACTUAL",
            "difficulty": difficulty,
            "link_type": link.link_type,
            "day": link.day,
            "actors": link.actors,
            "subsystems_involved": sorted(link.subsystems_involved),
            "question_text": question_text,
            "ground_truth": ground_truth,
            "evidence_chain": [link.cause_event_id, link.effect_event_id],
            "requires_reasoning": True,
        }

    def _cap_by_key(
        self,
        items: list,
        key_fn,
        max_per_key: int,
        max_total: int,
    ) -> list:
        by_key: Dict[str, list] = defaultdict(list)
        for item in items:
            by_key[key_fn(item)].append(item)

        selected = []
        overflow = []
        for group in by_key.values():
            random.shuffle(group)
            selected.extend(group[:max_per_key])
            overflow.extend(group[max_per_key:])

        random.shuffle(overflow)
        if len(selected) < max_total:
            selected.extend(overflow[: max_total - len(selected)])

        random.shuffle(selected)
        return selected[:max_total]

    def _cap_by_keys(
        self,
        items: list,
        key_fns: list,
        max_per_keys: list,
        max_total: int,
    ) -> list:
        """Multi-key capping: an item is included only if ALL its keys
        are below their respective caps."""
        counters = [defaultdict(int) for _ in key_fns]
        selected = []
        random.shuffle(items)

        for item in items:
            if len(selected) >= max_total:
                break
            keys = [fn(item) for fn in key_fns]
            if all(counters[i][keys[i]] < max_per_keys[i] for i in range(len(key_fns))):
                selected.append(item)
                for i, k in enumerate(keys):
                    counters[i][k] += 1

        return selected

    def _perspective_questions(self) -> List[dict]:
        MAX_PER_EVENT_TYPE = 5
        MAX_PER_ACTOR = 5

        internal_actors = {
            actor
            for actor in self._visibility_map.keys()
            if self._visibility_map[actor][0].role != "external"
            and actor not in _SYSTEM_ACTORS
        }

        all_asymmetry = [
            ev for ev in self._find_asymmetry_events() if ev[0] in internal_actors
        ]

        true_candidates = [ev for ev in all_asymmetry if ev[5]]
        false_candidates = [ev for ev in all_asymmetry if not ev[5]]

        target_true = int(self.MAX_PERSPECTIVE * 0.38)

        true_capped = self._cap_by_keys(
            true_candidates,
            key_fns=[
                lambda ev: ev[2].type,
                lambda ev: ev[0],
            ],
            max_per_keys=[MAX_PER_EVENT_TYPE, MAX_PER_ACTOR],
            max_total=target_true,
        )
        false_capped = self._cap_by_keys(
            false_candidates,
            key_fns=[
                lambda ev: ev[2].type,
                lambda ev: ev[0],
            ],
            max_per_keys=[MAX_PER_EVENT_TYPE, MAX_PER_ACTOR],
            max_total=self.MAX_PERSPECTIVE - target_true,
        )

        pool = true_capped + false_capped
        random.shuffle(pool)

        questions: List[dict] = []
        for actor, cone, event, info_available, cross_subsystem, _ in pool:
            question = self._build_perspective_question(
                actor, cone, event, info_available, cross_subsystem
            )
            if question:
                questions.append(question)

        by_type: Dict[str, List[dict]] = defaultdict(list)
        for q in questions:
            by_type[q["event_type"]].append(q)

        final = []
        overflow = []
        for event_type, group in by_type.items():
            random.shuffle(group)
            final.extend(group[:MAX_PER_EVENT_TYPE])
            overflow.extend(group[MAX_PER_EVENT_TYPE:])

        if len(final) < self.MAX_PERSPECTIVE:
            random.shuffle(overflow)
            final.extend(overflow[: self.MAX_PERSPECTIVE - len(final)])

        true_qs = [q for q in final if q["ground_truth"]["could_actor_have_known"]]
        false_qs = [q for q in final if not q["ground_truth"]["could_actor_have_known"]]
        n_t = min(len(true_qs), target_true)
        n_f = min(len(false_qs), self.MAX_PERSPECTIVE - n_t)
        final = random.sample(true_qs, n_t) + random.sample(false_qs, n_f)
        random.shuffle(final)

        logger.info(
            f"[eval] {len(final)} PERSPECTIVE questions built ({n_t} true / {n_f} false)"
        )
        return final

    def _generate_from_ranked_pool(
        self,
        primary: List[Tuple],
        reserve: List[Tuple],
        target: int,
    ) -> List[dict]:
        questions: List[dict] = []
        reserve_iter = iter(reserve)

        for candidate in primary:
            if len(questions) >= target:
                break

            actor, cone, event, info_available, cross_subsystem, _ = (
                candidate  # ← unpack, discard 6th
            )
            question = self._build_perspective_question(
                actor, cone, event, info_available, cross_subsystem
            )
            if question:
                questions.append(question)
            else:
                backup = next(reserve_iter, None)
                if backup:
                    actor, cone, event, info_available, cross_subsystem, _ = backup
                    question = self._build_perspective_question(
                        actor, cone, event, info_available, cross_subsystem
                    )
                    if question:
                        questions.append(question)

        while len(questions) < target:
            backup = next(reserve_iter, None)
            if not backup:
                break
            actor, cone, event, info_available, cross_subsystem, _ = backup
            question = self._build_perspective_question(
                actor, cone, event, info_available, cross_subsystem
            )
            if question:
                questions.append(question)

        return questions

    def _find_asymmetry_events(self) -> List[Tuple]:
        results = []
        significant_types = {
            "incident_opened",
            "customer_escalation",
            "sf_deals_risk_flagged",
            "knowledge_gap_detected",
            "employee_departed",
            "design_discussion",
            "customer_email_routed",
            "zd_tickets_escalated",
            "sf_ownership_lapsed",
            "postmortem_created",
            "inbound_external_email",
            "assignment_domain_mismatch",
            "email_dropped",
            "vendor_email_routed",
            "sales_outbound_email",
            "vendor_ack_sent",
            "customer_reply_sent",
            "sf_stage_advanced_by_customer",
            "watercooler_chat",
            "blocker_flagged",
            "onboarding_session",
            "warmup_1on1",
            "jira_ticket_created",
            "feature_request_fyi",
            "zd_ticket_opened",
            "zd_tickets_resolved",
            "proactive_outreach_initiated",
            "crm_touchpoint",
            "incident_resolved",
            "pr_review",
            "confluence_created",
            "retrospective",
            "sprint_planned",
            "escalation_chain",
            "org_collision",
            "ticket_completion_email",
            "external_contact_summarized",
        }

        MAX_TUPLES_PER_TYPE = 5
        MAX_PER_EVENT_ID = 1
        MAX_PER_ACTOR = 4

        type_tuple_counts: Dict[str, int] = defaultdict(int)
        event_id_counts: Dict[str, int] = defaultdict(int)
        actor_tuple_counts: Dict[str, int] = defaultdict(int)

        artifact_timestamps: Dict[str, str] = {
            doc["_id"]: doc.get("timestamp", "")
            for doc in self._mem._db["artifacts"].find({}, {"_id": 1, "timestamp": 1})
        }

        email_to_ticket: Dict[str, str] = {}
        for doc in self._mem._db["jira_tickets"].find(
            {"source_email_id": {"$exists": True, "$ne": ""}},
            {"_id": 0, "id": 1, "source_email_id": 1},
        ):
            email_to_ticket[doc["source_email_id"]] = doc["id"]

        shuffled_events = list(self._events)
        random.shuffle(shuffled_events)

        for event in shuffled_events:
            if not event.mongo_id:
                continue
            if event.type not in significant_types:
                continue

            if type_tuple_counts[event.type] >= MAX_TUPLES_PER_TYPE:
                continue
            event_subsystems = set()
            for doc_type, aid in (event.artifact_ids or {}).items():
                if not aid:
                    continue
                s = _ARTIFACT_SUBSYSTEM.get(doc_type, "default")
                if s != "default":
                    event_subsystems.add(s)

            if not event_subsystems:
                continue

            event_artifacts = _safe_artifact_values(event.artifact_ids)

            for actor, cones in self._visibility_map.items():
                if type_tuple_counts[event.type] >= MAX_TUPLES_PER_TYPE:
                    break
                if actor_tuple_counts[actor] >= MAX_PER_ACTOR:  # <-- NEW
                    continue

                if actor in (event.actors or []):
                    continue

                cone = next((c for c in cones if c.as_of_day == event.day), None)
                if not cone:
                    continue

                all_visible = cone.all_visible()

                all_visible = {
                    aid
                    for aid in all_visible
                    if not (
                        any(aid.startswith(p + "-") for p in _JIRA_PROJECT_ACCESS)
                        and not _jira_project_visible(aid, cone.role)
                    )
                }

                missed_artifacts = event_artifacts - all_visible

                if missed_artifacts and event.type in (
                    "inbound_external_email",
                    "customer_email_routed",
                ):
                    for email_id in event_artifacts:
                        ticket_id = email_to_ticket.get(email_id)
                        if ticket_id and ticket_id in all_visible:
                            missed_artifacts = set()
                            break

                if missed_artifacts:
                    missed_artifacts = {
                        aid
                        for aid in missed_artifacts
                        if artifact_timestamps.get(aid, "") <= cone.as_of_time
                    }

                if not missed_artifacts:
                    if (
                        event.type not in ActorVisibilityBuilder._BROADCAST_EVENTS
                        or not (event_subsystems & cone.subsystem_access)
                    ):
                        continue
                    if event_id_counts[event.mongo_id] >= MAX_PER_EVENT_ID:
                        continue
                    event_id_counts[event.mongo_id] += 1
                    type_tuple_counts[event.type] += 1
                    results.append(
                        (
                            actor,
                            cone,
                            event,
                            {
                                "actor_visible_subsystems": sorted(
                                    cone.subsystem_access
                                ),
                                "event_subsystems": sorted(event_subsystems),
                                "blocked_by_role": [],
                                "missed_artifacts": [],
                                "related_artifacts_actor_saw": sorted(
                                    event_artifacts & all_visible
                                ),
                            },
                            False,
                            True,
                        )
                    )
                    continue

                if event_id_counts[event.mongo_id] >= MAX_PER_EVENT_ID:
                    continue
                event_id_counts[event.mongo_id] += 1

                blocked_by_role = event_subsystems - cone.subsystem_access
                cross_subsystem = len(blocked_by_role) > 0

                related_visible = []
                for e in self._events:
                    if e.day > event.day:
                        continue
                    if actor not in (e.actors or []):
                        continue
                    shared_artifacts = (
                        _safe_artifact_values(e.artifact_ids) & event_artifacts
                    )
                    if not shared_artifacts:
                        continue
                    for aid in _safe_artifact_values(e.artifact_ids):
                        if aid in all_visible:
                            ts = artifact_timestamps.get(aid, "")
                            if ts and ts <= cone.as_of_time:
                                related_visible.append(aid)

                info_available = {
                    "actor_visible_subsystems": sorted(cone.subsystem_access),
                    "event_subsystems": sorted(event_subsystems),
                    "blocked_by_role": sorted(blocked_by_role),
                    "missed_artifacts": sorted(missed_artifacts),
                    "related_artifacts_actor_saw": sorted(set(related_visible)),
                }

                _event_artifacts_for_approx = _safe_artifact_values(event.artifact_ids)
                _secondary = set(related_visible)
                if not missed_artifacts:
                    _approx_known = True
                elif _secondary & _event_artifacts_for_approx:
                    _approx_known = True
                else:
                    _approx_known = False

                type_tuple_counts[event.type] += 1
                actor_tuple_counts[actor] += 1
                results.append(
                    (actor, cone, event, info_available, cross_subsystem, _approx_known)
                )

        return results

    def _build_perspective_question(
        self,
        actor: str,
        cone: ActorVisibilityCone,
        event: SimEvent,
        info_available: dict,
        cross_subsystem: bool,
    ) -> Optional[dict]:

        missed = info_available["missed_artifacts"]
        blocked = info_available["blocked_by_role"]
        secondary_visible = set(info_available["related_artifacts_actor_saw"])
        event_artifacts = _safe_artifact_values(event.artifact_ids)

        if len(missed) == 0:
            could_have_known = True
        elif secondary_visible:
            propagated_event_artifacts = secondary_visible & event_artifacts
            could_have_known = len(propagated_event_artifacts) > 0
        else:
            could_have_known = False

        is_broadcast = event.type in ActorVisibilityBuilder._BROADCAST_EVENTS

        if event.type == "design_discussion" and not could_have_known:
            confluence_id = (event.artifact_ids or {}).get("confluence")
            spawned_doc = event.facts.get("spawned_doc", False)
            doc_in_corpus = bool(
                spawned_doc
                and confluence_id
                and self._mem._db["artifacts"].find_one({"_id": confluence_id})
            )
            if doc_in_corpus and "confluence" in cone.subsystem_access:
                could_have_known = True

        if not could_have_known:
            if blocked:
                reason = (
                    f"Actor had access to {sorted(cone.subsystem_access)} but event "
                    f"involved {sorted(info_available['event_subsystems'])}; "
                    f"blocked by role from: {sorted(blocked)}"
                )
            elif event.type == "design_discussion" and not (
                event.artifact_ids or {}
            ).get("confluence"):
                reason = (
                    "No Confluence page was generated for this design discussion; "
                    "awareness requires direct participation"
                )
            else:
                reason = (
                    f"Actor was not a direct participant in this "
                    f"{event.type.replace('_', ' ')} event, which is not broadcast "
                    f"org-wide; platform access alone is insufficient for private events"
                )
        else:
            if event.type == "design_discussion":
                reason = (
                    "Design discussion produced a Confluence page discoverable "
                    "by any actor with Confluence access, regardless of direct participation"
                )
            else:
                reason = (
                    f"All event artifacts were in actor's visibility cone via "
                    f"{'direct involvement' if info_available['related_artifacts_actor_saw'] else 'broadcast'}"
                )

        evidence = sorted(info_available["related_artifacts_actor_saw"])

        if event.type == "design_discussion" and event.facts.get("spawned_doc"):
            conf_id = (event.artifact_ids or {}).get("confluence", "")
            if conf_id and conf_id not in evidence:
                evidence.append(conf_id)

        ground_truth = {
            "actor": actor,
            "as_of_day": cone.as_of_day,
            "as_of_time": cone.as_of_time,
            "could_actor_have_known": could_have_known,
            "reason": reason,
            "evidence_artifacts": evidence,
            "missed_artifacts": sorted(missed),
            "blocked_subsystems": sorted(blocked),
        }

        difficulty = "hard" if cross_subsystem else "medium"
        event_desc = self._event_description(event)

        PERSPECTIVE_QUESTION_STYLES = {
            "positive_biased": [
                "Phrase it as an awareness question: 'Would X have been aware of Y as of Day Z, based only on what was accessible to them?'",
                "Phrase it as an information access question: 'Could X have encountered Y through the systems available to them by Day Z?'",
                "Phrase it as an organizational question: 'Would Y have reached X through normal channels by Day Z?'",
                "Phrase it as a knowledge question: 'Based on X's role and access, is it likely they knew about Y before Day Z ended?'",
            ],
            "null_biased": [
                "Phrase it as an exclusion question: 'Would Y have been outside X's visibility by Day Z?'",
                "Phrase it as a reach question: 'Was there any path by which Y could have reached X before Day Z ended?'",
                "Phrase it as a gap question: 'Would X have had a blind spot around Y as of Day Z?'",
            ],
        }

        if could_have_known:
            style = random.choice(PERSPECTIVE_QUESTION_STYLES["positive_biased"])
        else:
            style_bucket = random.choices(
                list(PERSPECTIVE_QUESTION_STYLES.keys()), weights=[0.6, 0.4]
            )[0]
            style = random.choice(PERSPECTIVE_QUESTION_STYLES[style_bucket])

        template = (
            f"Write a question asking whether {actor} would have known about "
            f"'{event_desc}' as of Day {cone.as_of_day}. "
            f"The question must name the actor and the Day {cone.as_of_day} time constraint. "
            f"Do not reveal the answer. Do not include artifact IDs. "
            f"Do not mention which systems the actor can or cannot access. "
            f"{style} "
            f"Output only the question text."
        )

        question_text = self._generate_and_validate_prose(
            template=template,
            ground_truth_str=str(could_have_known),
            question_type="PERSPECTIVE",
        )
        if not question_text:
            return None

        return {
            "question_id": f"perspective_{actor}_{event.mongo_id}",
            "question_type": "PERSPECTIVE",
            "difficulty": difficulty,
            "cross_subsystem": cross_subsystem,
            "actor": actor,
            "actor_role": cone.role,
            "as_of_day": cone.as_of_day,
            "as_of_time": cone.as_of_time,
            "subsystem_access": sorted(cone.subsystem_access),
            "blocked_subsystems": sorted(info_available["blocked_by_role"]),
            "event_id": event.mongo_id,
            "event_type": event.type,
            "event_day": event.day,
            "question_text": question_text,
            "ground_truth": ground_truth,
            "requires_reasoning": True,
        }

    MAX_PER_LINK_TYPE = 5

    def _counterfactual_questions(self) -> List[dict]:
        effect_cause_map: Dict[str, List[CausalLink]] = defaultdict(list)
        for link in self._causal_links:
            effect_cause_map[link.effect_event_id].append(link)

        target_negative = self.MAX_COUNTERFACTUAL // 3
        target_positive = self.MAX_COUNTERFACTUAL - target_negative

        by_type: Dict[str, List[CausalLink]] = defaultdict(list)
        for link in self._causal_links:
            by_type[link.link_type].append(link)

        n_types = len(by_type)
        per_type = max(2, target_positive // max(n_types, 1))

        stratified: List[CausalLink] = []
        overflow: List[CausalLink] = []
        for link_type, group in by_type.items():
            random.shuffle(group)
            stratified.extend(group[:per_type])
            overflow.extend(group[per_type:])

        if len(stratified) < target_positive:
            random.shuffle(overflow)
            stratified.extend(overflow[: target_positive - len(stratified)])

        guaranteed: List[CausalLink] = []
        seen_types: set = set()
        for link in stratified:
            if link.link_type not in seen_types:
                guaranteed.append(link)
                seen_types.add(link.link_type)

        already_guaranteed_ids = {id(lnk) for lnk in guaranteed}
        remaining = [lnk for lnk in stratified if id(lnk) not in already_guaranteed_ids]
        random.shuffle(remaining)
        remaining_budget = target_positive - len(guaranteed)
        positive_sampled = guaranteed + remaining[:remaining_budget]
        random.shuffle(positive_sampled)

        seen_effects: set = set()
        positive_questions: List[dict] = []
        for link in positive_sampled:
            if link.effect_event_id in seen_effects:
                continue
            question = self._build_counterfactual_question(link)
            if question:
                positive_questions.append(question)
                seen_effects.add(link.effect_event_id)

        negative_candidates = [
            link
            for link in self._causal_links
            if len(effect_cause_map.get(link.effect_event_id, [])) >= 2
        ]
        random.shuffle(negative_candidates)

        seen_negative_effects: set = set()
        negative_questions: List[dict] = []
        for link in negative_candidates:
            if len(negative_questions) >= target_negative:
                break
            if link.effect_event_id in seen_negative_effects:
                continue
            if link.effect_event_id in seen_effects:
                continue
            q = self._build_negative_counterfactual(link, effect_cause_map)
            if q:
                negative_questions.append(q)
                seen_negative_effects.add(link.effect_event_id)

        questions = positive_questions + negative_questions
        random.shuffle(questions)

        logger.info(
            f"[eval] {len(questions)} COUNTERFACTUAL questions built "
            f"({len(positive_questions)} positive / {len(negative_questions)} negative)"
        )
        return questions

    def _build_counterfactual_question(self, link: CausalLink) -> Optional[dict]:

        if link.link_type == "involves_gap":
            cleaned = link.link_value.strip("[]'\" ").lower()
            if cleaned in _VAGUE_GAP_TERMS:
                logger.debug(
                    f"[counterfactual] Skipping {link.cause_event_id} — "
                    f"vague link_value: {link.link_value}"
                )
                return None

        cause_event = next(
            (e for e in self._events if e.mongo_id == link.cause_event_id),
            None,
        )
        effect_event = next(
            (e for e in self._events if e.mongo_id == link.effect_event_id),
            None,
        )

        def _clean_artifact_ids(ids: List[str]) -> List[str]:
            """Remove absolute paths that survive _strip_root (scoring would never match them)."""
            cleaned = []
            for aid in ids:
                if aid.startswith("/"):
                    match = re.search(r"export/(.+)$", aid)
                    cleaned.append(match.group(1) if match else aid)
                else:
                    cleaned.append(aid)
            return cleaned

        cause_artifacts = _clean_artifact_ids(
            sorted(
                _safe_artifact_values(cause_event.artifact_ids if cause_event else {})
            )
        )
        effect_artifacts = _clean_artifact_ids(
            sorted(
                _safe_artifact_values(effect_event.artifact_ids if effect_event else {})
            )
        )

        if link.link_type == "recurrence_of":
            effect_jira_id = (
                (effect_event.artifact_ids or {}).get("jira", "")
                if effect_event
                else ""
            )
            if effect_jira_id:
                effect_ticket = self._mem._db["jira_tickets"].find_one(
                    {"id": effect_jira_id}, {"prior_postmortem": 1}
                )
                if effect_ticket and effect_ticket.get("prior_postmortem"):
                    prior_pm = effect_ticket["prior_postmortem"]
                    if prior_pm not in cause_artifacts:
                        cause_artifacts.append(prior_pm)

        if link.link_type == "departure_reassignment" and effect_event:
            reassigned = effect_event.facts.get("reassigned_tickets", [])
            if reassigned:
                effect_artifacts = sorted(set(effect_artifacts) | set(reassigned[:5]))
            handoff_jira = (effect_event.artifact_ids or {}).get("jira")
            if handoff_jira:
                effect_artifacts = sorted(set(effect_artifacts) | {handoff_jira})

            if not cause_artifacts and not effect_artifacts:
                logger.debug(
                    f"[counterfactual] Skipping {link.cause_event_id} — "
                    f"departure_reassignment has no resolvable evidence artifacts"
                )
                return None

        ground_truth = {
            "outcome_changed": link.outcome_changed,
            "causal_mechanism": link.link_type,
            "causal_link_field": link.link_field,
            "causal_link_value": link.link_value,
            "cause_event_id": link.cause_event_id,
            "cause_event_type": link.cause_event_type,
            "effect_event_id": link.effect_event_id,
            "effect_event_type": link.effect_event_type,
            "premise": link.counterfactual_premise,
            "outcome": link.counterfactual_outcome,
            "actors": link.actors,
            "as_of_time": (_SIM_START + timedelta(days=link.day)).isoformat(),
            "evidence_chain_artifacts": {
                "cause": cause_artifacts,
                "effect": effect_artifacts,
            },
        }

        if link.link_type == "recurrence_of" and effect_event:
            day_context = (
                f"The original incident was resolved on Day {link.day}. "
                f"The recurrence occurred on Day {effect_event.day}. "
                f"Refer to the follow-on as 'the subsequent incident' — "
                f"do not assign Day {link.day} to it."
            )
        else:
            day_context = ""

        difficulty = "hard" if len(link.subsystems_involved) > 1 else "medium"

        COUNTERFACTUAL_QUESTION_STYLES = [
            "Phrase it as a direct counterfactual: 'If X had not happened, would Y have occurred?'",
            "Phrase it as a hypothetical: 'Had X not taken place, would Y still have happened?'",
            "Phrase it from the outcome perspective: 'Would Y have occurred without X happening first?'",
            "Phrase it as an investigative question: 'Was Y a direct consequence of X, or would it have happened regardless?'",
            "Phrase it as a dependency question: 'Did Y depend on X occurring, or was it independently triggered?'",
        ]

        style = random.choice(COUNTERFACTUAL_QUESTION_STYLES)
        actors_str = (
            ", ".join(link.actors[:3]) if link.actors else "the involved parties"
        )

        _COUNTERFACTUAL_DOMAIN_HINTS: Dict[str, str] = {
            "postmortem_from_incident": "a system incident and its aftermath documentation",
            "recurrence_of": "a recurring system incident linked to a prior unresolved issue",
            "review_triggers_revision": "a code review that led to an implementation revision before merge",
            "zd_escalation_source": "a support ticket escalation that triggered an incident",
            "escalation_from_zendesk": "Zendesk support tickets escalating into an incident",
            "incident_coordination": "a system incident and resulting coordination with an external party",
            "jira_from_vendor_email": "a vendor email and the Jira ticket it triggered",
            "jira_from_customer_email": "a customer email and the Jira ticket it triggered",
            "email_dropped": "an unactioned email and its downstream follow-up",
            "departure_reassignment": "an employee departure and the ticket reassignments that followed",
            "involves_gap": "a knowledge gap that contributed to a subsequent incident",
            "spawned_doc": "a design discussion and the Confluence page it produced",
            "sf_ownership_lapsed": "an employee departure and the lapsed Salesforce account ownership",
            "blocker_flagged": "a technical blocker and its effect on ticket progress",
            "assignment_domain_mismatch": "a domain mismatch in ticket assignment and the resulting knowledge gap",
            "hire_fills_knowledge_gap": "a new hire's expertise filling a gap left by a departed employee",
            "incident_triggers_risk_flag": "an incident and the Salesforce risk flags it triggered",
            "pr_gap_detected": "a PR review that surfaced a knowledge gap",
            "customer_escalation_relayed": "a customer email relayed as an internal escalation",
            "async_gap_detected": "an async Slack Q&A thread that surfaced a knowledge gap in an undocumented domain",
            "doc_gap_detected": "a knowledge gap embedded in a Confluence design document that contributed to a downstream incident",
            "centrality_vacuum": "a key person's departure creating a knowledge vacuum in their domain",
            "sf_stage_advanced_by_customer": "an inbound customer email that advanced a Salesforce opportunity stage",
            "feature_request_fyi": "a customer feature request email relayed as an FYI to the Product team",
            "proactive_outreach_from_crm_signal": "a CRM at-risk signal that triggered proactive outbound sales outreach",
            "ticket_completion_notifies_lead": "a ticket reaching Done status and triggering a completion notification",
            "org_collision_tension": "an unplanned cross-department friction event between overlapping responsibilities",
            "customer_escalation_relayed": "a customer escalation email relayed internally via Slack to the Product team",
        }

        domain_hint = _COUNTERFACTUAL_DOMAIN_HINTS.get(
            link.link_type, "an organizational event and its downstream effects"
        )

        template = (
            f"Write a counterfactual yes/no question about events on Day {link.day} "
            f"in a simulated company. {style} "
            f"The question must name the Day and the actors involved: {actors_str}. "
            f"The question is about: {domain_hint}. "
            f"Do not name specific artifact IDs or the causal mechanism label. "
            f"Do not reveal the answer or make the causal link obvious from the question text. "
            f"The question should require investigation to answer. "
            + (f"{day_context} " if day_context else "")
            + "Output only the question text."
        )

        question_text = self._generate_and_validate_prose(
            template=template,
            ground_truth_str=link.counterfactual_outcome,
            question_type="COUNTERFACTUAL",
        )
        if not question_text:
            return None

        return {
            "question_id": f"counterfactual_{link.cause_event_id}_{link.link_type}",
            "question_type": "COUNTERFACTUAL",
            "difficulty": difficulty,
            "link_type": link.link_type,
            "day": link.day,
            "actors": link.actors,
            "subsystems_involved": sorted(link.subsystems_involved),
            "question_text": question_text,
            "ground_truth": ground_truth,
            "evidence_chain": [link.cause_event_id, link.effect_event_id],
            "requires_reasoning": True,
        }

    def _silence_questions(self) -> List[dict]:
        questions: List[dict] = []

        target_false = self.MAX_SILENCE // 2
        target_true = self.MAX_SILENCE - target_false

        by_trigger: Dict[str, List[AbsenceRecord]] = defaultdict(list)
        for record in self._absence_catalog:
            by_trigger[record.trigger_event_type].append(record)

        n_triggers = len(by_trigger)
        per_trigger = max(1, target_false // max(n_triggers, 1))

        stratified: List[AbsenceRecord] = []
        for records in by_trigger.values():
            random.shuffle(records)
            by_response: Dict[str, List[AbsenceRecord]] = defaultdict(list)
            for r in records:
                by_response[r.expected_response_type].append(r)
            chosen: List[AbsenceRecord] = []
            response_groups = list(by_response.values())
            random.shuffle(response_groups)
            for group in response_groups:
                if len(chosen) >= per_trigger:
                    break
                chosen.append(random.choice(group))
            stratified.extend(chosen)

        selected_ids = {id(r) for r in stratified}
        remainder = [r for r in self._absence_catalog if id(r) not in selected_ids]
        if len(stratified) < target_false and remainder:
            extra = random.sample(
                remainder, min(target_false - len(stratified), len(remainder))
            )
            stratified.extend(extra)

        random.shuffle(stratified)
        false_pool = stratified[: target_false + 5]

        confirmed_pool = list(self._confirmed_catalog)
        random.shuffle(confirmed_pool)
        confirmed_pool = confirmed_pool[: target_true + 5]

        false_questions: List[dict] = []
        for record in false_pool:
            if len(false_questions) >= target_false:
                break
            q = self._build_silence_question(record, exists=False)
            if q:
                false_questions.append(q)

        true_questions: List[dict] = []
        for record in confirmed_pool:
            if len(true_questions) >= target_true:
                break
            q = self._build_silence_question(record, exists=True)
            if q:
                true_questions.append(q)

        questions = false_questions + true_questions
        random.shuffle(questions)

        logger.info(
            f"[eval] {len(questions)} SILENCE questions built "
            f"({len(true_questions)} exists=true / {len(false_questions)} exists=false)"
        )
        return questions

    def _build_silence_question(
        self, record: AbsenceRecord, exists: bool = False
    ) -> Optional[dict]:

        ground_truth = {
            "answer": exists,
            "absence_type": "state_machine_confirmed"
            if not exists
            else "state_machine_present",
            "trigger_event_id": record.trigger_event_id,
            "trigger_event_type": record.trigger_event_type,
            "expected_response_type": record.expected_response_type,
            "trigger_day": record.trigger_day,
            "trigger_actors": record.trigger_actors,
            "expected_search_space": record.expected_search_space,
            "link_field": record.link_field,
            "link_value": record.link_value,
        }

        expected_desc = {
            "postmortem_created": "a postmortem document",
            "incident_resolved": "an incident resolution",
            "zd_ticket_opened": "a Zendesk support ticket",
            "customer_email_routed": "an internal routing of the customer email",
            "confluence_created": "a Confluence documentation page",
            "sf_ownership_lapsed": "a Salesforce ownership transfer",
            "ticket_reassigned": "a Jira ticket reassignment",
            "pr_merged": "a merged pull request",
            "zd_tickets_escalated": "a Zendesk escalation",
            "incident_opened": "an incident ticket",
            "onboarding_session": "an onboarding session",
            "warmup_1on1": "a warmup 1-on-1 meeting",
            "sf_deals_risk_flagged": "a Salesforce risk flag on related deals",
            "knowledge_gap_detected": "a formal knowledge gap detection event",
            "jira_from_vendor_email": "a Jira ticket created from the vendor email",
        }.get(record.expected_response_type, f"a {record.expected_response_type} event")

        trigger_desc = {
            "incident_opened": f"the incident on Day {record.trigger_day}",
            "customer_escalation": f"the customer escalation on Day {record.trigger_day}",
            "inbound_external_email": f"the inbound email on Day {record.trigger_day}",
            "design_discussion": f"the design discussion on Day {record.trigger_day}",
            "knowledge_gap_detected": f"the knowledge gap detected on Day {record.trigger_day}",
            "employee_departed": f"the employee departure on Day {record.trigger_day}",
            "zd_tickets_escalated": f"the Zendesk escalation on Day {record.trigger_day}",
            "pr_opened": f"the pull request opened on Day {record.trigger_day}",
            "customer_email_routed": f"the routing of the customer email on Day {record.trigger_day}",
            "sf_deals_risk_flagged": f"the CRM risk flagging on Day {record.trigger_day}",
            "employee_hired": f"the hiring of {record.link_value} on Day {record.trigger_day}",
            "assignment_domain_mismatch": f"the domain mismatch assignment flagged on Day {record.trigger_day}",
        }.get(record.trigger_event_type, f"the event on Day {record.trigger_day}")

        actors_str = (
            ", ".join(record.trigger_actors[:2])
            if record.trigger_actors
            else "the involved parties"
        )

        trigger_ev = next(
            (e for e in self._events if e.mongo_id == record.trigger_event_id),
            None,
        )

        if not trigger_ev:
            logger.warning(
                f"[eval] Skipping SILENCE question for unknown trigger type: {record.trigger_event_type}"
            )
            return None

        _TRIVIAL_ENTRIES = {"jira", "confluence", "slack", "email", "git", "zendesk"}
        effective_space = [
            e for e in record.expected_search_space if e not in _TRIVIAL_ENTRIES
        ]
        if not effective_space:
            logger.debug(
                f"[silence] Skipping {record.trigger_event_id} — "
                f"expected_search_space is empty or trivially satisfied"
            )
            return None

        if record.trigger_event_type == "knowledge_gap_detected":
            detection_method = trigger_ev.facts.get("detection_method", "")

            if detection_method == "author_self_audit":
                return None

            if trigger_ev.day < 1:
                return None

            if detection_method == "reviewer_audit":
                gap_topics = trigger_ev.facts.get("topics_beyond_author_expertise", [])
                pr_title = trigger_ev.facts.get("pr_title", "")
                gap_topic_str = (
                    " and ".join(gap_topics) if gap_topics else record.link_value
                )
                trigger_desc = (
                    f"the knowledge gap detected on Day {record.trigger_day} involving {actors_str}, "
                    f"specifically regarding {gap_topic_str}"
                    + (f" identified during review of '{pr_title}'" if pr_title else "")
                )

            elif detection_method == "async_thread_classification":
                gap_domain = trigger_ev.facts.get("gap_domain", record.link_value)
                topic = trigger_ev.facts.get("topic", "")
                trigger_desc = (
                    f"the knowledge gap detected on Day {record.trigger_day} involving {actors_str}, "
                    f"specifically regarding {gap_domain}"
                    + (f" (topic: '{topic}')" if topic else "")
                )

            elif detection_method in ("embedding_similarity", "centrality_vacuum"):
                gap_areas = trigger_ev.facts.get("gap_areas", [])
                gap_topic_str = ", ".join(gap_areas) if gap_areas else record.link_value
                trigger_desc = (
                    f"the knowledge gap detected on Day {record.trigger_day} involving {actors_str}, "
                    f"specifically regarding {gap_topic_str}"
                )

            else:
                trigger_desc = (
                    f"the knowledge gap detected on Day {record.trigger_day} involving {actors_str}, "
                    f"specifically regarding {record.link_value}"
                )

        SILENCE_QUESTION_STYLES = [
            "Phrase it as a process compliance question a manager would ask.",
            "Phrase it as an audit question checking whether a response was documented.",
            "Phrase it as an operational question about whether follow-up occurred.",
            "Phrase it as a gap analysis question about whether proper procedure was followed.",
        ]

        style = random.choice(SILENCE_QUESTION_STYLES)

        template = (
            f"Write a yes/no question asking whether {expected_desc} was created "
            f"in response to {trigger_desc} involving {actors_str}. "
            f"CRITICAL: Only refer to the event exactly as described ('{trigger_desc}'). "
            f"Do not call it an 'incident' or 'outage' unless those words are explicitly used. "
            f"The question should require investigation to confirm or deny — do not imply the answer either way. "
            f"Do not state or imply the answer. Do not include system IDs. "
            f"Do NOT use phrases like 'as required by', 'as expected by', 'should have been', "
            f"'was supposed to', or 'procedure requires' — these imply the answer. "
            f"{style} "
            f"Output only the question text."
        )

        question_text = self._generate_and_validate_prose(
            template=template,
            ground_truth_str=str(exists),
            question_type="SILENCE",
        )
        if not question_text:
            return None

        return {
            "question_id": f"silence_{record.trigger_event_id}_{record.expected_response_type}",
            "question_type": "SILENCE",
            "difficulty": "hard",
            "trigger_event_id": record.trigger_event_id,
            "trigger_event_type": record.trigger_event_type,
            "trigger_day": record.trigger_day,
            "expected_response_type": record.expected_response_type,
            "subsystem": record.subsystem,
            "question_text": question_text,
            "ground_truth": ground_truth,
            "expected_search_space": record.expected_search_space,
            "requires_reasoning": True,
        }

    def _event_description(self, event: SimEvent) -> str:
        """Natural language description of an event for use in question templates."""
        descs = {
            "incident_opened": lambda e: (
                f"a P1 incident ({e.facts.get('title', 'system incident')})"
            ),
            "customer_escalation": lambda e: (
                f"a customer escalation from {e.facts.get('customer', 'a customer')}"
            ),
            "sf_deals_risk_flagged": lambda e: (
                "Salesforce accounts being flagged at-risk"
            ),
            "knowledge_gap_detected": lambda e: (
                (
                    f"a knowledge gap in {', '.join(e.facts.get('topics_beyond_author_expertise', ['an undocumented domain']))} "
                    f"identified during review of '{e.facts.get('pr_title', 'a pull request')}'"
                )
                if e.facts.get("detection_method")
                in ("reviewer_audit", "author_self_audit")
                else (
                    f"a knowledge gap regarding {e.facts.get('gap_domain', 'an undocumented domain')} "
                    f"surfaced in an async discussion about '{e.facts.get('topic', 'an unspecified topic')}'"
                )
                if e.facts.get("detection_method") == "async_thread_classification"
                else (
                    f"a knowledge gap in {', '.join(e.facts.get('gap_areas', ['an undocumented domain']))} "
                    f"(triggered by {e.facts.get('triggered_by', 'unknown')}, "
                    f"left by {e.facts.get('departed_employee', 'a departed employee')}, "
                    f"detected via {e.facts.get('detection_method', 'automated analysis')})"
                )
            ),
            "employee_departed": lambda e: (
                f"the departure of {(e.actors or ['a team member'])[0]}"
            ),
            "design_discussion": lambda e: (
                f"a design discussion about {e.facts.get('topic', 'a technical topic')}"
            ),
            "customer_email_routed": lambda e: (
                "a customer email being routed to support"
            ),
            "zd_tickets_escalated": lambda e: (
                "Zendesk tickets being escalated to an incident"
            ),
            "sf_ownership_lapsed": lambda e: "Salesforce accounts losing their owner",
            "postmortem_created": lambda e: "a postmortem being written",
            "inbound_external_email": lambda e: (
                f"an inbound email from {e.facts.get('source', e.facts.get('sender', 'an external contact'))} "
                f"({e.facts.get('org', 'external')}) "
                f"regarding '{e.facts.get('subject', e.facts.get('topic', 'an unspecified topic'))[:80]}'"
            ),
            "pr_review": lambda e: (
                f"a {'change request' if e.facts.get('verdict') == 'changes_requested' else 'review'} "
                f"by {e.facts.get('reviewer', 'a reviewer')} on "
                f"'{e.facts.get('pr_title', 'an unnamed PR')[:80]}' "
                f"authored by {e.facts.get('author', 'unknown')}: "
                f'"{e.facts.get("review_text", "")[:120]}"'
            ),
            "employee_hired": lambda e: (
                f"{e.facts.get('name', 'a new hire')} joining {e.facts.get('dept', 'a department')} "
                f"as {e.facts.get('role', 'an engineer')} with expertise in "
                f"{', '.join(e.facts.get('expertise', ['unspecified']))}"
            ),
            "confluence_created": lambda e: (
                f"a {e.facts.get('type', 'document')} '{e.facts.get('title', 'untitled')}' "
                f"created by {(e.actors or ['unknown'])[0]}"
                + (
                    f", spawning tickets {', '.join(e.facts['spawned_tickets'][:3])}"
                    if e.facts.get("spawned_tickets")
                    else ""
                )
                + (
                    f", updating domains: {', '.join(e.facts['domains_updated'][:3])}"
                    if e.facts.get("domains_updated")
                    else ""
                )
            ),
            "jira_ticket_created": lambda e: (
                f"a Jira ticket '{e.facts.get('title', 'untitled')[:80]}' opened by "
                f"{(e.actors or ['unknown'])[0]}"
                + (
                    f" from a {e.facts.get('source', '').replace('_', ' ')}"
                    if e.facts.get("source")
                    else ""
                )
                + (
                    f" (vendor: {e.facts.get('vendor', '')})"
                    if e.facts.get("vendor")
                    else ""
                )
            ),
            "ticket_progress": lambda e: (
                f"ticket {e.facts.get('ticket_id', (e.artifact_ids or {}).get('jira', 'unknown'))} "
                f"moved to '{e.facts.get('status', 'unknown status')}' "
                f"by {(e.actors or ['unknown'])[0]}"
            ),
            "email_dropped": lambda e: (
                f"an unactioned email from {e.facts.get('source', 'an external contact')} "
                f"with subject '{e.facts.get('subject', 'unspecified')[:80]}' "
                f"(reason: {e.facts.get('reason', 'unknown').replace('_', ' ')})"
            ),
            "sales_outbound_email": lambda e: (
                f"an outbound sales email to {e.facts.get('account', 'a prospect')} "
                f"sent by {(e.actors or ['a sales rep'])[0]}"
            ),
            "watercooler_chat": lambda e: (
                f"an informal chat between {' and '.join((e.actors or ['colleagues'])[:2])}"
            ),
            "vendor_email_routed": lambda e: (
                f"a vendor email from {e.facts.get('source', 'an external vendor')} "
                f"being routed internally"
            ),
        }
        fn = descs.get(event.type)
        if fn:
            try:
                return fn(event)
            except Exception:
                pass
        return f"a {event.type.replace('_', ' ')} event"

    def _generate_and_validate_prose(
        self,
        template: str,
        ground_truth_str: str,
        question_type: str,
        max_attempts: int = 3,
    ) -> Optional[str]:
        """
        LLM writes question prose. Validates against structured rubric.
        Retries up to max_attempts if validation fails.

        Validation rules:
        1. Must end with '?'
        2. Must not contain ground_truth_str verbatim
        3. Must not contain raw artifact IDs (pattern: XX-\\d+ or [a-f0-9]{8,})
        4. Must be between 15 and 120 words
        5. Must contain at least one of: actor name, day reference, subsystem word
        """
        agent = make_agent(
            role="Eval Dataset Author",
            goal="Write natural-sounding evaluation questions for AI agent benchmarks.",
            backstory=(
                "You write clear, specific questions for evaluating AI agents on reasoning "
                "tasks. Questions must be unambiguous, naturally phrased, and answerable "
                "only through careful reasoning over a corporate document corpus."
            ),
            llm=self._worker_llm,
        )

        for attempt in range(max_attempts):
            retry_note = (
                " Previous attempt failed validation. Make sure the question: "
                "ends with '?', does not reveal the answer, avoids artifact IDs, "
                "and is 15-120 words long."
                if attempt > 0
                else ""
            )
            task = Task(
                description=template + retry_note,
                expected_output="One question ending with a question mark. No preamble or explanation.",
                agent=agent,
            )
            try:
                result = str(
                    Crew(agents=[agent], tasks=[task], verbose=False).kickoff()
                ).strip()

                if self._validate_prose(result, ground_truth_str, question_type):
                    return result
                else:
                    logger.debug(
                        f"[eval] Prose validation failed (attempt {attempt + 1}): {result[:80]}"
                    )
            except Exception as exc:
                logger.warning(
                    f"[eval] Prose generation error (attempt {attempt + 1}): {exc}"
                )

        return None

    def _validate_prose(
        self, text: str, ground_truth_str: str, question_type: str
    ) -> bool:
        normalized = (
            text.replace("\u2010", "-")
            .replace("\u2011", "-")
            .replace("\u2012", "-")
            .replace("\u2013", "-")
        )

        if not normalized.endswith("?"):
            return False

        words = normalized.split()
        if len(words) < 10 or len(words) > 150:
            return False

        gt_lower = ground_truth_str.lower()
        if gt_lower in normalized.lower() and len(gt_lower) > 4:
            return False

        if re.search(r"\b[A-Z]{1,4}-\d{2,6}\b", normalized):
            return False
        if re.search(r"\b[a-f0-9]{8,}\b", normalized):
            return False

        if question_type == "PERSPECTIVE":
            if not re.search(
                r"day\s+\d+|as of|by\s+[A-Z][a-z]+", normalized, re.IGNORECASE
            ):
                return False

        if question_type == "COUNTERFACTUAL":
            if not re.search(
                r"\b(if|had|would|could|might|hypothetically)\b",
                normalized,
                re.IGNORECASE,
            ):
                return False

        if question_type == "SILENCE":
            if not re.search(
                r"\b(was|were|did|has|have|is|are)\b", normalized, re.IGNORECASE
            ):
                return False

            _SILENCE_LEAK_PHRASES = (
                "as required by",
                "as expected by",
                "should have been",
                "was supposed to",
                "procedure requires",
                "protocol requires",
                "per the",
                "as mandated",
            )
            if any(phrase in normalized.lower() for phrase in _SILENCE_LEAK_PHRASES):
                return False

        return True


class EvalHarness:
    """
    Orchestrates the full eval dataset generation pipeline:
      1. Build actor visibility cones (ActorVisibilityBuilder)
      2. Index explicit causal links (CausalLinkIndexer)
      3. Catalog expected-but-absent artifacts (AbsenceCatalogBuilder)
      4. Generate eval questions (EvalQuestionGenerator)
      5. Write all intermediate structures + final questions to export/eval/
    """

    def __init__(self):
        from flow import build_llm

        self._mem = Memory()
        self._worker_llm = build_llm("worker")
        self._leads = list(_CFG.get("leads", {}).values())

    def run(self) -> None:
        logger.info("[bold cyan]🔬 Building OrgForge eval dataset...[/bold cyan]")

        logger.info("[eval] Building actor visibility cones...")
        vis_builder = ActorVisibilityBuilder(self._mem)
        visibility_map = vis_builder.build_all()
        vis_export = {
            actor: cones[-1].to_dict()
            for actor, cones in visibility_map.items()
            if cones
        }
        vis_path = EVAL_DIR / "actor_visibility.json"
        with open(vis_path, "w") as f:
            json.dump(vis_export, f, indent=2, default=str)
        logger.info(f"  → {vis_path} ({len(visibility_map)} actors)")

        logger.info("[eval] Indexing explicit causal links...")
        link_indexer = CausalLinkIndexer(self._mem)
        causal_links = link_indexer.build()
        links_path = EVAL_DIR / "causal_link_index.json"
        with open(links_path, "w") as f:
            json.dump([lnk.to_dict() for lnk in causal_links], f, indent=2, default=str)
        logger.info(f"  → {links_path} ({len(causal_links)} links)")

        logger.info("[eval] Building absence catalog...")
        absence_builder = AbsenceCatalogBuilder(self._mem)
        absence_catalog, confirmed_catalog = absence_builder.build()
        absence_path = EVAL_DIR / "absence_catalog.json"
        with open(absence_path, "w") as f:
            json.dump([r.to_dict() for r in absence_catalog], f, indent=2, default=str)
        logger.info(f"  → {absence_path} ({len(absence_catalog)} absence records)")

        confirmed_path = EVAL_DIR / "confirmed_catalog.json"
        with open(confirmed_path, "w") as f:
            json.dump(
                [r.to_dict() for r in confirmed_catalog], f, indent=2, default=str
            )
        logger.info(
            f"  → {confirmed_path} ({len(confirmed_catalog)} confirmed records)"
        )

        """ logger.info("[eval] Building GRAPH track...")
        graph_snapshots, graph_questions = build_graph_track(
            mem=self._mem,
            worker_llm=self._worker_llm,
            leads=self._leads,
            eval_dir=EVAL_DIR,
        )
        logger.info(
            f"  → {len(graph_snapshots)} graph snapshots, {len(graph_questions)} GRAPH questions"
        ) """

        logger.info("[eval] Generating eval questions...")
        generator = EvalQuestionGenerator(
            mem=self._mem,
            worker_llm=self._worker_llm,
            visibility_map=visibility_map,
            causal_links=causal_links,
            absence_catalog=absence_catalog,
            confirmed_catalog=confirmed_catalog,
        )
        questions = generator.generate()
        # questions.extend(graph_questions)
        random.shuffle(questions)

        by_type: Dict[str, int] = defaultdict(int)
        by_difficulty: Dict[str, int] = defaultdict(int)
        cross_subsystem_count = 0
        for q in questions:
            by_type[q["question_type"]] += 1
            by_difficulty[q["difficulty"]] += 1
            if q.get("cross_subsystem"):
                cross_subsystem_count += 1

        questions_path = EVAL_DIR / "eval_questions.json"
        with open(questions_path, "w") as f:
            json.dump(
                {
                    "metadata": {
                        "generated_at": datetime.now().isoformat(),
                        "version": "2.0",
                        "tracks": ["PERSPECTIVE", "COUNTERFACTUAL", "SILENCE", "GRAPH"],
                        "total_questions": len(questions),
                        "by_type": dict(by_type),
                        "by_difficulty": dict(by_difficulty),
                        "cross_subsystem_questions": cross_subsystem_count,
                        "actors_with_visibility_cones": len(visibility_map),
                        "causal_links_indexed": len(causal_links),
                        "absence_records": len(absence_catalog),
                        # "graph_snapshots": len(graph_snapshots),
                    },
                    "questions": questions,
                },
                f,
                indent=2,
                default=str,
            )

        logger.info(f"  → {questions_path}")
        logger.info(
            f"[green]✓ Eval dataset complete.[/green] "
            f"Types: {dict(by_type)} | Difficulty: {dict(by_difficulty)} | "
            f"Cross-subsystem: {cross_subsystem_count}"
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    EvalHarness().run()
