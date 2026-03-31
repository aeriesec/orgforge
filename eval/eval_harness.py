"""
eval_harness.py
===============
OrgForge Eval Dataset Generator — v2

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

BASE = Path(_CFG["simulation"].get("output_dir", "./export"))
EVAL_DIR = BASE / "eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

_SIM_START = datetime.strptime(_CFG["simulation"]["start_date"], "%Y-%m-%d")

# ── Subsystem access model ────────────────────────────────────────────────────
# Maps role patterns to the subsystems they have access to.
# Agents outside a subsystem cannot retrieve its artifacts.
# Extend this as new subsystems are added to the simulation.

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
    "engineering_backend": {"slack", "jira", "confluence", "git", "zoom", "datadog"},
    "engineering_mobile": {"slack", "jira", "confluence", "git", "zoom", "datadog"},
    "design": {"slack", "confluence", "zoom"},
    "sales_marketing": {"slack", "salesforce", "email", "zoom", "confluence"},
    "hr_ops": {"slack", "email", "confluence", "zoom"},
    "qa_support": {"slack", "zendesk", "confluence", "email"},
    "external": set(),
}

# Maps artifact ID prefixes / doc_types to their subsystem
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
    "invoice": "email",  # invoices are email artifacts for access purposes
    "nps": "salesforce",  # NPS lives in the CRM surface
}

# Explicit causal link types the sim encodes — COUNTERFACTUAL scope
_EXPLICIT_CAUSAL_LINKS = {
    "involves_gap",  # incident ← knowledge gap
    "recurrence_of",  # incident ← prior unresolved incident
    "spawned_doc",  # confluence ← design discussion
    "email_dropped",  # communication failure ← routing gap
    "sf_ownership_lapsed",  # CRM gap ← employee departure
    "zd_escalation_source",  # incident ← support ticket escalation
    "blocker_flagged",  # blocker → delayed progress
    "incident_coordination",  # incident → external contact
    "departure_reassignment",  # departure → ticket/escalation shift
    "assignment_domain_mismatch",  # planning mismatch → knowledge gap → incident
}

# Expected event pairs for SILENCE catalog:
# (trigger_event_type, expected_response_event_type, link_field)
# If trigger fired but response did not, that's a valid SILENCE question target.
_SILENCE_PAIRS: List[Tuple[str, str, str]] = [
    ("incident_opened", "postmortem_created", "jira"),
    ("incident_opened", "incident_resolved", "jira"),
    ("customer_escalation", "zd_ticket_opened", "email"),
    ("customer_email_routed", "zd_ticket_opened", "email"),
    ("inbound_external_email", "customer_email_routed", "email"),
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
]

_BROADCAST_CONFIG = {
    "incident_opened": ["slack", "datadog"],
    "incident_resolved": ["slack"],
    "postmortem_created": ["slack", "confluence"],
    "standup": ["slack"],
    "pr_opened": ["git", "slack"],
    "pr_merged": ["git", "slack"],
    "knowledge_gap_detected": ["slack", "confluence"],
}


def _safe_artifact_values(artifact_ids: dict) -> Set[str]:
    """Flatten artifact_ids values — some may be lists."""
    vals: Set[str] = set()
    for v in (artifact_ids or {}).values():
        if isinstance(v, list):
            vals.update(str(x) for x in v)
        elif v:
            vals.add(str(v))
    return vals


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────


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
    as_of_time: str  # ISO timestamp — the knowledge horizon
    as_of_day: int
    subsystem_access: Set[str]  # subsystems this actor can query
    visible_artifacts: Dict[str, Set[str]]  # subsystem → set of artifact IDs
    directly_involved: Set[str]  # artifacts where actor appears in event.actors
    broadcast_visible: Set[str]  # artifacts visible via channel broadcast

    def all_visible(self) -> Set[str]:
        all_ids: Set[str] = set()
        for ids in self.visible_artifacts.values():
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
        """
        Standardize role inference using config_loader ground truth.
        """
        roles: Dict[str, str] = {}

        for dept, members in CONFIG["org_chart"].items():
            role_slug = dept.lower().replace(" ", "_")
            for name in members:
                roles[name] = role_slug

        for name, data in DEPARTED_EMPLOYEES.items():
            roles[name] = data["role"].lower().replace(" ", "_")

        lifecycle = CONFIG.get("org_lifecycle", {})
        for hire in lifecycle.get("scheduled_hires", []):
            roles[hire["name"]] = hire["role"].lower().replace(" ", "_")
        for dep in lifecycle.get("scheduled_departures", []):
            roles[dep["name"]] = dep["role"].lower().replace(" ", "_")

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

    _BROADCAST_CONFIG = {
        "incident_opened": ["slack", "datadog"],
        "incident_resolved": ["slack"],
        "postmortem_created": ["slack", "confluence"],
        "standup": ["slack"],
        "pr_opened": ["git"],
        "pr_merged": ["git"],
        "knowledge_gap_detected": ["slack", "confluence"],
    }

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

                as_of_dt = _SIM_START + timedelta(days=day - 1, hours=23, minutes=59)
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


# ─────────────────────────────────────────────────────────────────────────────
# CAUSAL LINK INDEX
# ─────────────────────────────────────────────────────────────────────────────


class CausalLinkIndexer:
    """
    Scans the SimEvent log for all explicit causal links.
    Only links in _EXPLICIT_CAUSAL_LINKS are indexed — no inference.

    Each link becomes a potential COUNTERFACTUAL question source.
    The counterfactual premise and outcome are templated deterministically
    from the link type and event facts; LLMs only rephrase them.
    """

    def __init__(self, mem: Memory):
        self._mem = mem
        self._events: List[SimEvent] = mem.get_event_log(from_db=True)
        self._event_by_id: Dict[str, SimEvent] = {
            self._synthetic_event_id(e): e for e in self._events
        }

    def _synthetic_event_id(self, e: SimEvent) -> str:
        """Build a stable synthetic key since SimEvent has no event_id attr."""
        raw = next(iter((e.artifact_ids or {}).values()), "none")
        first_artifact = raw[0] if isinstance(raw, list) else (raw or "none")
        actor = (e.actors or ["unknown"])[0]
        return f"evt_{e.type}_{e.day}_{first_artifact}_{actor}"

    def _subsystems_for_event(self, event: SimEvent) -> Set[str]:
        subsystems: Set[str] = set()
        for doc_type in event.artifact_ids or {}:
            s = _ARTIFACT_SUBSYSTEM.get(doc_type, "default")
            if s != "default":
                subsystems.add(s)
        return subsystems

    def _find_effect_event(self, link_type: str, cause: SimEvent) -> Optional[SimEvent]:
        """Find the downstream event causally linked to cause."""
        if link_type == "involves_gap":
            gap_domain = (
                cause.facts.get("gap_areas", [None])[0]
                if cause.facts.get("gap_areas")
                else None
            )
            if not gap_domain:
                return None

            for e in self._events:
                if e.day < cause.day:
                    continue

                if e.type == "incident_opened" and gap_domain in e.facts.get(
                    "gap_areas", []
                ):
                    return e

                relevant_types = {
                    "async_question_asked",
                    "pr_review_comment",
                    "confluence_created",
                    "postmortem_created",
                }
                if e.type in relevant_types:
                    event_domains = (
                        e.facts.get("gap_areas") or e.facts.get("domain") or []
                    )
                    if gap_domain in (
                        event_domains
                        if isinstance(event_domains, list)
                        else [event_domains]
                    ):
                        return e

        elif link_type == "recurrence_of":
            cause_artifacts = _safe_artifact_values(cause.artifact_ids)
            for e in self._events:
                recurrence = e.facts.get("recurrence_of")
                if recurrence and recurrence in cause_artifacts:
                    return e

        elif link_type == "spawned_doc":
            cause_artifacts = _safe_artifact_values(cause.artifact_ids)
            for e in self._events:
                if (
                    e.type == "confluence_created"
                    and e.facts.get("source_discussion") in cause_artifacts
                ):
                    return e

        elif link_type == "email_dropped":
            email_id = (cause.artifact_ids or {}).get("email")
            if isinstance(email_id, list):
                email_id = email_id[0] if email_id else None
            if not email_id:
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
                    if gap_areas and mismatch_domains and set(gap_areas) & set(mismatch_domains):
                        return e

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
            actor = (cause.actors or ["the departing engineer"])[0]
            jira_id = (effect.artifact_ids or {}).get("jira", "the incident")
            premise = f"{actor} had fully documented {gap_str} before departing"
            outcome = f"{jira_id} would have been diagnosed faster or prevented"
            return premise, outcome, True

        elif link_type == "recurrence_of":
            orig = effect.facts.get("recurrence_of", "the original incident")
            jira_id = (effect.artifact_ids or {}).get("jira", "the recurrence")
            premise = f"the postmortem for {orig} had included preventive action items"
            outcome = f"{jira_id} would likely not have occurred"
            return premise, outcome, True

        elif link_type == "spawned_doc":
            topic = cause.facts.get("topic", "the design discussion")
            conf_id = (effect.artifact_ids or {}).get(
                "confluence", "the Confluence doc"
            )
            premise = f"the discussion about '{topic}' had not been documented"
            outcome = f"{conf_id} would not exist and related decisions would remain undocumented"
            return premise, outcome, True

        elif link_type == "email_dropped":
            sender = cause.facts.get("sender", "the customer")
            premise = f"the email from {sender} had been routed correctly"
            outcome = "a support ticket would have been opened and the issue tracked"
            return premise, outcome, True

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

        if link_type == "blocker_flagged":
            reason = cause.facts.get("blocker_reason", "a technical blocker")
            jira_id = effect.artifact_ids.get("jira", "the ticket")
            return (
                f"the blocker regarding '{reason}' had been resolved immediately",
                f"work on {jira_id} would have progressed without delay",
                True,
            )

        elif link_type == "incident_coordination":
            contact = effect.facts.get("external_party", "the external contact")
            jira_id = cause.artifact_ids.get("jira", "the incident")
            return (
                f"the incident {jira_id} had not occurred",
                f"the team would not have needed to coordinate with {contact}",
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
                    e for e in self._events if e.type == "knowledge_gap_detected"
                ]
            elif link_type == "recurrence_of":
                cause_events = [
                    e for e in self._events if e.type == "incident_resolved"
                ]
            elif link_type == "spawned_doc":
                cause_events = [
                    e
                    for e in self._events
                    if e.type == "design_discussion" and e.facts.get("spawned_doc")
                ]
            elif link_type == "email_dropped":
                cause_events = [
                    e for e in self._events if e.type == "inbound_external_email"
                ]
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
            else:
                continue

            for cause in cause_events:
                effect = self._find_effect_event(link_type, cause)
                if not effect:
                    continue

                premise, outcome, changed = self._counterfactual_template(
                    link_type, cause, effect
                )

                subsystems = self._subsystems_for_event(
                    cause
                ) | self._subsystems_for_event(effect)

                link_field = {
                    "involves_gap": "gap_areas",
                    "recurrence_of": "recurrence_of",
                    "spawned_doc": "spawned_doc",
                    "email_dropped": "email",
                    "sf_ownership_lapsed": "actor",
                    "zd_escalation_source": "jira",
                    "blocker_flagged": "jira",
                    "incident_coordination": "jira",
                    "departure_reassignment": "actor",
                    "assignment_domain_mismatch": "ticket_id",
                }.get(link_type, "")

                link_value = str(
                    cause.facts.get(link_field, "")
                    or (cause.artifact_ids or {}).get(link_field, "")
                    or (cause.actors or [""])[0]
                )

                links.append(
                    CausalLink(
                        link_type=link_type,
                        cause_event_id=self._synthetic_event_id(cause),
                        cause_event_type=cause.type,
                        effect_event_id=self._synthetic_event_id(effect),
                        effect_event_type=effect.type,
                        actors=list(set((cause.actors or []) + (effect.actors or []))),
                        day=cause.day,
                        link_field=link_field,
                        link_value=link_value,
                        subsystems_involved=subsystems,
                        counterfactual_premise=premise,
                        counterfactual_outcome=outcome,
                        outcome_changed=changed,
                    )
                )

        logger.info(f"[causal_index] {len(links)} explicit causal links indexed")
        return links


# ─────────────────────────────────────────────────────────────────────────────
# ABSENCE CATALOG
# ─────────────────────────────────────────────────────────────────────────────


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

    @staticmethod
    def _synthetic_event_id(e: SimEvent) -> str:
        first_artifact = next(iter((e.artifact_ids or {}).values()), "none")
        actor = (e.actors or ["unknown"])[0]
        return f"evt_{e.type}_{e.day}_{first_artifact}_{actor}"

    def _match_key(self, event: SimEvent, link_field: str) -> Optional[str]:
        """Extract the value that links a trigger to its expected response."""
        val = (event.artifact_ids or {}).get(link_field)
        if val:
            return val
        val = event.facts.get(link_field)
        if val:
            return str(val)
        if link_field == "actor" and event.actors:
            return event.actors[0]
        return None

    def _expected_search_space(
        self, trigger: SimEvent, expected_response_type: str
    ) -> List[str]:
        """
        Derive the artifact IDs the agent should check to confirm absence.
        These are artifacts that WOULD contain the response if it had occurred.
        """
        search_space: List[str] = []

        # Always include trigger artifacts as starting points
        for artifact_id in (trigger.artifact_ids or {}).values():
            if artifact_id:
                search_space.append(artifact_id)

        if expected_response_type == "postmortem_created":
            jira_id = (trigger.artifact_ids or {}).get("jira", "")
            if jira_id:
                search_space.append(f"confluence/postmortems/{jira_id}")
                search_space.append("slack/channels/incidents")

        elif expected_response_type == "incident_resolved":
            jira_id = (trigger.artifact_ids or {}).get("jira", "")
            if jira_id:
                search_space.append(jira_id)
                search_space.append("jira/incidents")

        elif expected_response_type == "zd_ticket_opened":
            email_id = (trigger.artifact_ids or {}).get("email", "")
            if email_id:
                search_space.append("zendesk/tickets")
                search_space.append(email_id)

        elif expected_response_type == "customer_email_routed":
            search_space.append("slack/channels/support")
            search_space.append("zendesk/queue")

        elif expected_response_type == "confluence_created":
            zoom_id = (trigger.artifact_ids or {}).get("zoom_transcript", "")
            if zoom_id:
                search_space.append(zoom_id)
            search_space.append("confluence/design-docs")
            search_space.append("confluence/decisions")

        elif expected_response_type == "sf_ownership_lapsed":
            actor = (trigger.actors or [""])[0]
            if actor:
                search_space.append(f"salesforce/accounts/{actor}")
                search_space.append("salesforce/ownership-log")

        elif expected_response_type == "ticket_reassigned":
            actor = (trigger.actors or [""])[0]
            if actor:
                search_space.append("jira/reassignments")
                search_space.append("slack/channels/engineering")

        elif expected_response_type == "pr_merged":
            pr_id = (trigger.artifact_ids or {}).get("pr", "")
            if pr_id:
                search_space.append(pr_id)
                search_space.append("git/merged-prs")

        elif expected_response_type == "zd_tickets_escalated":
            jira_id = (trigger.artifact_ids or {}).get("jira", "")
            if jira_id:
                search_space.append(jira_id)
                search_space.append("zendesk/escalations")

        elif expected_response_type == "onboarding_session":
            name = trigger.facts.get("name", "")
            search_space.append("slack/channels/general")
            search_space.append(f"confluence/onboarding/{name}")

        elif expected_response_type == "warmup_1on1":
            name = trigger.facts.get("name", "")
            search_space.append("slack/channels/engineering")
            search_space.append("zoom/transcripts")

        elif expected_response_type == "sf_deals_risk_flagged":
            jira_id = trigger.artifact_ids.get("jira", "")
            search_space.append("salesforce/opportunities")
            if jira_id:
                search_space.append(jira_id)

        elif expected_response_type == "knowledge_gap_detected":
            # Silence: assignment_domain_mismatch fired but no gap was ever formally detected
            ticket_id = trigger.facts.get("ticket_id", "")
            if ticket_id:
                search_space.append(f"jira/{ticket_id}")
            search_space.append("slack/channels/engineering")
            search_space.append("confluence/knowledge-gaps")

        return list(dict.fromkeys(search_space))  # dedupe, preserve order

    def build(self) -> List[AbsenceRecord]:
        records: List[AbsenceRecord] = []

        for trigger_type, response_type, link_field in _SILENCE_PAIRS:
            trigger_events = [e for e in self._events if e.type == trigger_type]

            for trigger in trigger_events:
                trigger_artifacts = _safe_artifact_values(trigger.artifact_ids)
                link_key = self._match_key(trigger, link_field)

                if response_type == "confluence_created" and (
                    trigger.facts.get("spawned_doc")
                    or "confluence" in (trigger.artifact_ids or {})
                ):
                    continue

                response_found = False
                for e in self._events:
                    if e.type != response_type or e.day < trigger.day:
                        continue

                    if link_key and (
                        self._match_key(e, link_field) == link_key
                        or link_key in str(e.artifact_ids)
                        or link_key in str(e.facts)
                    ):
                        response_found = True
                        break

                    response_artifacts = _safe_artifact_values(e.artifact_ids)
                    if trigger_artifacts & response_artifacts:
                        response_found = True
                        break

                if response_found:
                    continue

                subsystem = _ARTIFACT_SUBSYSTEM.get(
                    list((trigger.artifact_ids or {}).keys() or [""])[0], "default"
                )
                search_space = self._expected_search_space(trigger, response_type)

                records.append(
                    AbsenceRecord(
                        trigger_event_id=self._synthetic_event_id(trigger),
                        trigger_event_type=trigger_type,
                        expected_response_type=response_type,
                        trigger_day=trigger.day,
                        trigger_actors=trigger.actors or [],
                        trigger_artifact_ids=dict(trigger.artifact_ids or {}),
                        link_field=link_field,
                        link_value=link_key or "N/A",
                        subsystem=subsystem,
                        expected_search_space=search_space,
                    )
                )

        logger.info(f"[absence_catalog] {len(records)} absence records cataloged")
        return records


# ─────────────────────────────────────────────────────────────────────────────
# QUESTION GENERATOR
# ─────────────────────────────────────────────────────────────────────────────


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

    MAX_PERSPECTIVE = 40
    MAX_COUNTERFACTUAL = 30
    MAX_SILENCE = 30

    def __init__(
        self,
        mem: Memory,
        worker_llm,
        visibility_map: Dict[str, List[ActorVisibilityCone]],
        causal_links: List[CausalLink],
        absence_catalog: List[AbsenceRecord],
    ):
        self._mem = mem
        self._worker_llm = worker_llm
        self._visibility_map = visibility_map
        self._causal_links = causal_links
        self._absence_catalog = absence_catalog
        self._events: List[SimEvent] = mem.get_event_log(from_db=True)

    @staticmethod
    def _synthetic_event_id(e: SimEvent) -> str:
        """Build a stable synthetic key since SimEvent has no event_id attr."""
        raw = next(iter((e.artifact_ids or {}).values()), "none")
        first_artifact = raw[0] if isinstance(raw, list) else (raw or "none")
        actor = (e.actors or ["unknown"])[0]
        return f"evt_{e.type}_{e.day}_{first_artifact}_{actor}"

    def generate(self) -> List[dict]:
        questions: List[dict] = []

        logger.info("[eval] Generating PERSPECTIVE questions...")
        questions.extend(self._perspective_questions())

        logger.info("[eval] Generating COUNTERFACTUAL questions...")
        questions.extend(self._counterfactual_questions())

        logger.info("[eval] Generating SILENCE questions...")
        questions.extend(self._silence_questions())

        # Shuffle so question types are interleaved in the output
        random.shuffle(questions)

        logger.info(f"[eval] {len(questions)} total questions generated")
        return questions

    # ── TRACK 1: PERSPECTIVE ─────────────────────────────────────────────────

    def _perspective_questions(self) -> List[dict]:
        questions: List[dict] = []

        internal_actors = [
            actor
            for actor in self._visibility_map.keys()
            if self._visibility_map[actor][0].role != "external"
        ]

        # Find events that involve information asymmetry — where the actor was
        # NOT in event.actors but the event affected them (e.g. a customer
        # escalation that went to sales but not engineering)
        asymmetry_events = [
            ev for ev in self._find_asymmetry_events() if ev[0] in internal_actors
        ]

        candidates = random.sample(
            asymmetry_events, min(self.MAX_PERSPECTIVE, len(asymmetry_events))
        )

        for actor, cone, event, info_available, cross_subsystem in candidates:
            question = self._build_perspective_question(
                actor, cone, event, info_available, cross_subsystem
            )
            if question:
                questions.append(question)

        logger.info(f"[eval] {len(questions)} PERSPECTIVE questions built")
        return questions

    def _find_asymmetry_events(self) -> List[Tuple]:
        """
        Find (actor, cone, event, info_available, is_cross_subsystem) tuples
        where an actor had partial or no visibility into a significant event.

        Focuses on events with real decision-making consequence:
        - Customer escalations visible to support but not engineering
        - Incidents visible to engineering but not sales
        - Design decisions visible to eng but not the broader org
        - HR/departure events with asymmetric visibility
        - CRM risk flags invisible to non-sales actors
        """
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
        }

        for event in self._events:
            if event.type not in significant_types:
                continue

            event_subsystems = set()
            event_artifacts = set()

            for doc_type, aid in (event.artifact_ids or {}).items():
                if not aid:
                    continue
                event_artifacts.add(str(aid))
                s = _ARTIFACT_SUBSYSTEM.get(doc_type, "default")
                if s != "default":
                    event_subsystems.add(s)

            if not event_subsystems:
                continue

            # Look for actors NOT in the event who have relevant role-based access
            for actor, cones in self._visibility_map.items():
                if actor in (event.actors or []):
                    continue  # Actor was directly involved — not an asymmetry case

                # Find the cone at the event's day
                cone = next((c for c in cones if c.as_of_day == event.day), None)
                if not cone:
                    continue

                all_visible = cone.all_visible()
                event_artifacts = _safe_artifact_values(event.artifact_ids)

                # Check if actor missed this information
                missed_artifacts = event_artifacts - all_visible
                if not missed_artifacts:
                    continue  # Actor could already see everything

                # Determine if this spans subsystems the actor doesn't have
                blocked_by_role = event_subsystems - cone.subsystem_access
                cross_subsystem = len(blocked_by_role) > 0

                # What did the actor actually know that's related?
                related_visible = []
                for e in self._events:
                    if e.day > event.day:
                        continue
                    if actor not in (e.actors or []):
                        continue
                    shared_actors = set(e.actors or []) & set(event.actors or [])
                    shared_artifacts = _safe_artifact_values(
                        e.artifact_ids
                    ) & _safe_artifact_values(event.artifact_ids)
                    if shared_actors or shared_artifacts:
                        for aid in _safe_artifact_values(e.artifact_ids):
                            if aid in all_visible:
                                related_visible.append(aid)

                info_available = {
                    "actor_visible_subsystems": sorted(cone.subsystem_access),
                    "event_subsystems": sorted(event_subsystems),
                    "blocked_by_role": sorted(blocked_by_role),
                    "missed_artifacts": sorted(missed_artifacts),
                    "related_artifacts_actor_saw": sorted(set(related_visible)),
                }

                results.append((actor, cone, event, info_available, cross_subsystem))

        return results

    def _build_perspective_question(
        self,
        actor: str,
        cone: ActorVisibilityCone,
        event: SimEvent,
        info_available: dict,
        cross_subsystem: bool,
    ) -> Optional[dict]:

        # Derive ground truth deterministically
        missed = info_available["missed_artifacts"]
        blocked = info_available["blocked_by_role"]
        could_have_known = len(missed) == 0  # actor had access to all artifacts

        ground_truth = {
            "actor": actor,
            "as_of_day": cone.as_of_day,
            "as_of_time": cone.as_of_time,
            "could_actor_have_known": could_have_known,
            "reason": (
                f"Actor had access to {sorted(cone.subsystem_access)} but event "
                f"involved {sorted(info_available['event_subsystems'])}; "
                f"blocked by role from: {sorted(blocked)}"
                if not could_have_known
                else f"All event artifacts were in actor's visibility cone via "
                f"{'direct involvement' if info_available['related_artifacts_actor_saw'] else 'broadcast'}"
            ),
            "evidence_artifacts": sorted(info_available["related_artifacts_actor_saw"]),
            "missed_artifacts": sorted(missed),
            "blocked_subsystems": sorted(blocked),
        }

        difficulty = "hard" if cross_subsystem else "medium"

        # Build prose template
        event_desc = self._event_description(event)
        subsystem_constraint = (
            f"{actor} has access to {', '.join(sorted(cone.subsystem_access))} "
            f"but not {', '.join(sorted(blocked))}"
            if blocked
            else f"{actor} has access to {', '.join(sorted(cone.subsystem_access))}"
        )

        template = (
            f"Write a question asking whether {actor} would have known about "
            f"'{event_desc}' as of Day {cone.as_of_day}, given that "
            f"{subsystem_constraint}. "
            f"The question must name the actor, the approximate time constraint "
            f"(Day {cone.as_of_day}), and the subsystem limitation. "
            f"Do not reveal the answer. Do not include artifact IDs. "
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
            "question_id": f"perspective_{actor}_{self._synthetic_event_id(event)}",
            "question_type": "PERSPECTIVE",
            "difficulty": difficulty,
            "cross_subsystem": cross_subsystem,
            "actor": actor,
            "actor_role": cone.role,
            "as_of_day": cone.as_of_day,
            "as_of_time": cone.as_of_time,
            "subsystem_access": sorted(cone.subsystem_access),
            "blocked_subsystems": sorted(info_available["blocked_by_role"]),
            "event_id": self._synthetic_event_id(event),
            "event_type": event.type,
            "event_day": event.day,
            "question_text": question_text,
            "ground_truth": ground_truth,
            "actor_visible_artifacts": sorted(cone.all_visible()),
            "requires_reasoning": True,
        }

    # ── TRACK 2: COUNTERFACTUAL ───────────────────────────────────────────────

    def _counterfactual_questions(self) -> List[dict]:
        questions: List[dict] = []

        sampled = random.sample(
            self._causal_links, min(self.MAX_COUNTERFACTUAL, len(self._causal_links))
        )

        for link in sampled:
            question = self._build_counterfactual_question(link)
            if question:
                questions.append(question)

        logger.info(f"[eval] {len(questions)} COUNTERFACTUAL questions built")
        return questions

    def _build_counterfactual_question(self, link: CausalLink) -> Optional[dict]:

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
        }

        subsystems_str = ", ".join(sorted(link.subsystems_involved))
        difficulty = "hard" if len(link.subsystems_involved) > 1 else "medium"

        template = (
            f"Write a counterfactual question asking: if {link.counterfactual_premise}, "
            f"would the following have occurred: {link.counterfactual_outcome}? "
            f"The question should involve events from Day {link.day} in a simulated "
            f"company with systems including {subsystems_str}. "
            f"The question must be phrased as a hypothetical (use 'if', 'had', 'would'). "
            f"Do not reveal the answer. Do not include event IDs. "
            f"Output only the question text."
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

    # ── TRACK 3: SILENCE ─────────────────────────────────────────────────────

    def _silence_questions(self) -> List[dict]:
        questions: List[dict] = []

        sampled = random.sample(
            self._absence_catalog, min(self.MAX_SILENCE, len(self._absence_catalog))
        )

        for record in sampled:
            question = self._build_silence_question(record)
            if question:
                questions.append(question)

        logger.info(f"[eval] {len(questions)} SILENCE questions built")
        return questions

    def _build_silence_question(self, record: AbsenceRecord) -> Optional[dict]:

        ground_truth = {
            "answer": False,  # The expected artifact/event does NOT exist
            "absence_type": "state_machine_confirmed",
            "trigger_event_id": record.trigger_event_id,
            "trigger_event_type": record.trigger_event_type,
            "expected_response_type": record.expected_response_type,
            "trigger_day": record.trigger_day,
            "trigger_actors": record.trigger_actors,
            "expected_search_space": record.expected_search_space,
            "link_field": record.link_field,
            "link_value": record.link_value,
        }

        # Build a natural-language description of what should have existed
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
            (
                e
                for e in self._events
                if self._synthetic_event_id(e) == record.trigger_event_id
            ),
            None,
        )

        if not trigger_ev:
            logger.warning(
                f"[eval] Skipping SILENCE question for unknown trigger type: {record.trigger_event_type}"
            )
            return None

        template = (
            f"Write a yes/no question asking whether {expected_desc} was created "
            f"in response to {trigger_desc} involving {actors_str}. "
            f"CRITICAL: Only refer to the event exactly as described ('{trigger_desc}'). "
            f"Do not call it an 'incident' or 'outage' unless those words are explicitly used. "
            f"The question should be phrased so that the correct answer is 'no' — "
            f"the artifact does not exist — but the agent must investigate to confirm this. "
            f"Do not state or imply the answer. Do not include system IDs. "
            f"The question should sound like something a manager would ask when reviewing "
            f"process compliance. "
            f"Output only the question text."
        )

        question_text = self._generate_and_validate_prose(
            template=template,
            ground_truth_str="False",
            question_type="SILENCE",
        )
        if not question_text:
            return None

        return {
            "question_id": f"silence_{record.trigger_event_id}_{record.expected_response_type}",
            "question_type": "SILENCE",
            "difficulty": "hard",  # Absence reasoning is always hard
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

    # ── PROSE GENERATION + VALIDATION ────────────────────────────────────────

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
                f"a knowledge gap in {', '.join(e.facts.get('gap_areas', ['an undocumented domain']))}"
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
                f"an inbound email from {e.facts.get('sender', 'an external contact')}"
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
        if not text.endswith("?"):
            return False

        words = text.split()
        if len(words) < 10 or len(words) > 150:
            return False

        # Must not leak ground truth verbatim
        gt_lower = ground_truth_str.lower()
        if gt_lower in text.lower() and len(gt_lower) > 4:
            return False

        # Must not contain raw artifact IDs (e.g. IT-108, abc12345)
        if re.search(r"\b[A-Z]{1,4}-\d{2,6}\b", text):
            return False
        if re.search(r"\b[a-f0-9]{8,}\b", text):
            return False

        # PERSPECTIVE questions must reference an actor name or role
        if question_type == "PERSPECTIVE":
            if not re.search(r"day\s+\d+|as of|by\s+[A-Z][a-z]+", text, re.IGNORECASE):
                return False

        # COUNTERFACTUAL questions must use hypothetical language
        if question_type == "COUNTERFACTUAL":
            if not re.search(
                r"\b(if|had|would|could|might|hypothetically)\b", text, re.IGNORECASE
            ):
                return False

        # SILENCE questions must be yes/no answerable
        if question_type == "SILENCE":
            if not re.search(
                r"\b(was|were|did|has|have|is|are)\b", text, re.IGNORECASE
            ):
                return False

        return True


# ─────────────────────────────────────────────────────────────────────────────
# HARNESS ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────


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

    def run(self) -> None:
        logger.info("[bold cyan]🔬 Building OrgForge eval dataset v2...[/bold cyan]")

        # Step 1: Actor visibility
        logger.info("[eval] Building actor visibility cones...")
        vis_builder = ActorVisibilityBuilder(self._mem)
        visibility_map = vis_builder.build_all()
        vis_export = {
            actor: [cone.to_dict() for cone in cones]
            for actor, cones in visibility_map.items()
        }
        vis_path = EVAL_DIR / "actor_visibility.json"
        with open(vis_path, "w") as f:
            json.dump(vis_export, f, indent=2, default=str)
        logger.info(f"  → {vis_path} ({len(visibility_map)} actors)")

        # Step 2: Causal link index
        logger.info("[eval] Indexing explicit causal links...")
        link_indexer = CausalLinkIndexer(self._mem)
        causal_links = link_indexer.build()
        links_path = EVAL_DIR / "causal_link_index.json"
        with open(links_path, "w") as f:
            json.dump([lnk.to_dict() for lnk in causal_links], f, indent=2, default=str)
        logger.info(f"  → {links_path} ({len(causal_links)} links)")

        # Step 3: Absence catalog
        logger.info("[eval] Building absence catalog...")
        absence_builder = AbsenceCatalogBuilder(self._mem)
        absence_catalog = absence_builder.build()
        absence_path = EVAL_DIR / "absence_catalog.json"
        with open(absence_path, "w") as f:
            json.dump([r.to_dict() for r in absence_catalog], f, indent=2, default=str)
        logger.info(f"  → {absence_path} ({len(absence_catalog)} absence records)")

        # Step 4: Question generation
        logger.info("[eval] Generating eval questions...")
        generator = EvalQuestionGenerator(
            mem=self._mem,
            worker_llm=self._worker_llm,
            visibility_map=visibility_map,
            causal_links=causal_links,
            absence_catalog=absence_catalog,
        )
        questions = generator.generate()

        # Summary stats
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
                        "tracks": ["PERSPECTIVE", "COUNTERFACTUAL", "SILENCE"],
                        "total_questions": len(questions),
                        "by_type": dict(by_type),
                        "by_difficulty": dict(by_difficulty),
                        "cross_subsystem_questions": cross_subsystem_count,
                        "actors_with_visibility_cones": len(visibility_map),
                        "causal_links_indexed": len(causal_links),
                        "absence_records": len(absence_catalog),
                    },
                    "questions": questions,
                },
                f,
                indent=2,
                default=str,
            )

        logger.info(f"  → {questions_path}")
        logger.info(
            f"[green]✓ Eval dataset v2 complete.[/green] "
            f"Types: {dict(by_type)} | Difficulty: {dict(by_difficulty)} | "
            f"Cross-subsystem: {cross_subsystem_count}"
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    EvalHarness().run()
