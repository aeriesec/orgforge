"""
crm_system.py
=============
Active CRM state machine for OrgForge.

Salesforce and Zendesk records are live simulation state, not post-processing.
They live in MongoDB alongside JIRA tickets and are embedded into the vector
store so the DayPlanner can read open tickets during planning — exactly as it
reads open JIRA tickets.

Design principles (mirrors insider_threat.py)
----------------------------------------------
* NullCRMSystem is the default — completely inert when disabled, no callers
  need to guard with ``if crm is not None``.
* All writes go to MongoDB first, then export to disk. Disk layout mirrors
  the existing export/jira/ and export/slack/ conventions.
* Every state change emits a SimEvent. Ground truth is always in the event
  log — not inferred from disk files.
* The planner_context() method returns a compact string that DayPlannerOrchestrator
  can inject into its planning prompt alongside email_signals, so Product and
  Engineering planners "see" open support tickets and at-risk deals.

MongoDB collections
-------------------
  zd_tickets    — one doc per Zendesk ticket  (mirrors _jira collection schema)
  sf_accounts   — one doc per Salesforce account
  sf_opps       — one doc per Salesforce opportunity

Config schema (config.yaml)
---------------------------
  crm:
    salesforce:
      enabled: true
      seed_accounts: true       # generate from contacts at sim start
    zendesk:
      enabled: true
      link_to_incidents: true   # escalate open ZD tickets when incident fires

Public API (called from flow.py)
---------------------------------
  crm = CRMSystem.from_config(config, export_base, mem)

  # Sim start — seeds SF accounts from contacts
  crm.initialize_salesforce_accounts(contacts)

  # Pre-standup — called before DayPlannerOrchestrator.plan()
  crm_signals = crm.planner_context()        # injected alongside email_signals

  # Inbound email handling — called from ExternalEmailIngestor
  zd_id = crm.handle_inbound_complaint(event_facts, timestamp, date_str, day)

  # Incident lifecycle — called from _handle_incident() and _advance_incidents()
  crm.handle_incident_opened(incident_id, component, health, timestamp, date_str, day)
  crm.handle_incident_resolved(incident_id, postmortem_link, timestamp, date_str, day)

  # Outbound email classification — called from email_gen.py / HR outbound path
  crm.process_outbound_email(email_data, timestamp, date_str, day)

  # Org lifecycle — called from OrgLifecycleManager departure handler
  crm.handle_employee_departure(employee_name, role, date_str, day)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
import random
from typing import Any, Dict, List, Optional

from config_loader import CONFIG

logger = logging.getLogger("orgforge.crm")

_ZD_TYPES = ["question", "incident", "problem", "task"]
_ZD_TYPE_WEIGHTS = [0.6, 0.2, 0.1, 0.1]
_ZD_PRIORITIES = ["Low", "Normal", "High", "Urgent"]
_ZD_PRIORITY_WEIGHTS = [0.2, 0.6, 0.15, 0.05]
_ZD_CHANNELS = ["email", "web_widget", "api"]
_ZD_CHANNEL_WEIGHTS = [0.7, 0.2, 0.1]


_SF_TYPES = ["New Business", "Renewal", "Upsell/Cross-sell"]
_SF_TYPE_WEIGHTS = [0.6, 0.25, 0.15]
_SF_LEAD_SOURCES = ["Inbound Email", "Outbound", "Event", "Referral"]
_SF_LEAD_SOURCE_WEIGHTS = [0.4, 0.3, 0.2, 0.1]

_STAGE_PROBABILITIES = {
    "Prospecting": 10,
    "Value Proposition": 25,
    "Proposal/Price Quote": 50,
    "Negotiation/Review": 75,
    "Closed Won": 100,
    "Closed Lost": 0,
}


class NullCRMSystem:
    """
    Drop-in replacement when CRM is disabled.
    Every method is a no-op or returns an empty/None value so callers
    never need to check ``if crm is not None``.
    """

    def initialize_salesforce_accounts(self) -> None:
        pass

    def planner_context(self) -> str:
        return ""

    def handle_inbound_complaint(
        self,
        event_facts: Dict,
        timestamp: str,
        date_str: str,
        day: int,
    ) -> Optional[str]:
        return None

    def handle_incident_opened(
        self,
        incident_id: str,
        component: str,
        health: int,
        timestamp: str,
        date_str: str,
        day: int,
    ) -> None:
        pass

    def handle_incident_resolved(
        self,
        incident_id: str,
        postmortem_link: str,
        timestamp: str,
        date_str: str,
        day: int,
    ) -> None:
        pass

    def process_outbound_email(
        self,
        email_data: Dict,
        timestamp: str,
        date_str: str,
        day: int,
    ) -> Optional[Dict]:
        return None

    def handle_employee_departure(
        self,
        employee_name: str,
        role: str,
        date_str: str,
        day: int,
    ) -> None:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# LIVE SYSTEM
# ─────────────────────────────────────────────────────────────────────────────


class CRMSystem:
    """
    Active CRM state machine. Instantiated once in OrgForgeSimulation.__init__
    and passed as a dependency to NormalDayHandler, ExternalEmailIngestor, and
    the incident handlers in flow.py.
    """

    # Sales-intent keywords for outbound email classification.
    # Kept narrow to avoid false positives on engineering or HR mail.
    _SALES_KEYWORDS = {
        "contract",
        "renewal",
        "proposal",
        "following up",
        "pricing",
        "quote",
        "demo",
        "partnership",
        "commercial",
        "subscription",
        "onboarding",
        "account review",
    }

    def __init__(self, config: Dict, export_base: Path, mem, planner_llm=None):
        crm_cfg = config.get("crm", {})
        self._sf_cfg = crm_cfg.get("salesforce", {})
        self._zd_cfg = crm_cfg.get("zendesk", {})
        self._sf_on = self._sf_cfg.get("enabled", False)
        self._zd_on = self._zd_cfg.get("enabled", False)
        self._base = Path(export_base)
        self._mem = mem
        self._llm = planner_llm

        # Shorthand collections (created lazily below)
        self._zd = mem._db["zd_tickets"]
        self._sf_a = mem._db["sf_accounts"]
        self._sf_o = mem._db["sf_opps"]

        # Monotonic counters — restored from Mongo on resume
        self._zd_counter = (
            max(
                (d.get("_seq", 0) for d in self._zd.find({}, {"_seq": 1})),
                default=100,
            )
            + 1
        )
        self._opp_counter = (
            max(
                (d.get("_seq", 0) for d in self._sf_o.find({}, {"_seq": 1})),
                default=1000,
            )
            + 1
        )

        self._ensure_dirs()
        logger.info(
            f"[crm] CRMSystem online — SF={'on' if self._sf_on else 'off'} "
            f"ZD={'on' if self._zd_on else 'off'}"
        )

    # ── Factory ──────────────────────────────────────────────────────────────

    @classmethod
    def from_config(
        cls, config: Dict, export_base: Path, mem, planner_llm=None
    ) -> "CRMSystem | NullCRMSystem":
        """
        Returns a live CRMSystem if either SF or ZD is enabled,
        otherwise a NullCRMSystem. Same pattern as InsiderThreatInjector.from_config.
        """
        crm_cfg = config.get("crm", {})
        sf_on = crm_cfg.get("salesforce", {}).get("enabled", False)
        zd_on = crm_cfg.get("zendesk", {}).get("enabled", False)
        if sf_on or zd_on:
            return cls(config, export_base, mem, planner_llm)
        return NullCRMSystem()



    def _ensure_dirs(self):
        for sub in [
            "salesforce/accounts",
            "salesforce/opportunities",
            "zendesk/tickets",
            "zendesk/comments",
        ]:
            (self._base / sub).mkdir(parents=True, exist_ok=True)

    def _write(self, rel_path: str, data: Dict) -> None:
        path = self._base / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(data, fh, indent=2)

    def _embed(
        self,
        *,
        id: str,
        artifact_type: str,
        title: str,
        content: str,
        day: int,
        date: str,
        timestamp: str,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Enqueue an embedding so the artifact is searchable by the DayPlanner."""
        self._mem.embed_artifact(
            id=id,
            type=artifact_type,
            title=title,
            content=content,
            day=day,
            date=date,
            timestamp=timestamp,
            metadata=metadata or {},
        )

    def _emit(self, event) -> None:
        """Log a SimEvent via the shared memory bus."""
        self._mem.log_event(event)

    def planner_context(self) -> str:
        """
        Returns a compact, human-readable string summarising live CRM state.
        Called once per day before DayPlannerOrchestrator.plan() so Product and
        Engineering planners see open support tickets and at-risk deals the same
        way they see vendor email signals.

        Kept to ~150 tokens so it doesn't crowd out the sprint/incident context.
        """
        lines: List[str] = []

        if self._zd_on:
            open_tickets = list(
                self._zd.find({"status": {"$in": ["Open", "Pending"]}}, {"_id": 0})
            )
            if open_tickets:
                lines.append(f"OPEN SUPPORT TICKETS ({len(open_tickets)}):")
                for t in open_tickets[:5]:  # cap at 5 to control prompt size
                    urgency = " [URGENT]" if t.get("priority") == "Urgent" else ""
                    t_type = t.get("type", "ticket").upper()
                    lines.append(
                        f"  [{t['ticket_id']}]{urgency} {t_type}: {t['subject']} "
                        f"— {t.get('org_name', 'Unknown')} "
                        f"(linked incident: {t.get('related_incident', 'none')})"
                    )
                if len(open_tickets) > 5:
                    lines.append(f"  ... and {len(open_tickets) - 5} more.")

        if self._sf_on:
            at_risk = list(
                self._sf_o.find(
                    {
                        "stage": {"$nin": ["Closed Won", "Closed Lost"]},
                        "risk_notes": {"$not": {"$size": 0}},
                    },
                    {"_id": 0},
                )
            )
            if at_risk:
                lines.append(f"AT-RISK DEALS ({len(at_risk)}):")
                for opp in at_risk[:3]:
                    amt = opp.get("amount", 0)
                    prob = opp.get("probability", 0)
                    lines.append(
                        f"  [{opp['opportunity_id']}] {opp['account_name']} (${amt:,} / {prob}% prob) "
                        f"— stage: {opp['stage']} — "
                        f"risk: {opp['risk_notes'][-1][:80]}"
                    )

            active_pipeline = list(
                self._sf_o.find(
                    {
                        "stage": {"$nin": ["Closed Won", "Closed Lost"]},
                        "risk_notes": {"$size": 0},
                    },
                    {"_id": 0},
                ).sort("_seq", -1)
            )

            if active_pipeline:
                lines.append(
                    f"ACTIVE SALES PIPELINE (Target these for proactive outreach!):"
                )

                for stage in [
                    "Negotiation/Review",
                    "Proposal/Price Quote",
                    "Value Proposition",
                    "Prospecting",
                ]:
                    stage_deals = [
                        opp for opp in active_pipeline if opp.get("stage") == stage
                    ]
                    for opp in stage_deals[:2]:
                        lines.append(
                            f"  [{opp['opportunity_id']}] {opp['account_name']} "
                            f"— Stage: {stage} — Owner: {opp.get('owner', 'Unassigned')}"
                        )

        return "\n".join(lines) if lines else ""

    def initialize_salesforce_accounts(self) -> None:
        """
        Runs once during genesis_phase(), before the daily loop starts.
        Reads customer contacts from MongoDB and seeds SF accounts.
        """
        if not self._sf_on or not self._sf_cfg.get("seed_accounts", True):
            return

        contacts = list(
            self._mem._db["sim_config"].find(
                {"_id": "inbound_email_sources", "category": "customer"}, {"_id": 0}
            )
        )

        start_dt = datetime.strptime(CONFIG["simulation"]["start_date"], "%Y-%m-%d")

        for contact in contacts:
            org_name = contact.get("org", "Unknown")
            safe_id = org_name.upper().replace(" ", "").replace("-", "")
            account_id = f"ACC-{safe_id}"

            if self._sf_a.find_one({"account_id": account_id}):
                continue

            days_ago = random.randint(30, 730)
            hours_ago = random.randint(0, 23)
            mins_ago = random.randint(0, 59)
            created_dt = start_dt - timedelta(
                days=days_ago, hours=hours_ago, minutes=mins_ago
            )

            account = {
                "account_id": account_id,
                "name": org_name,
                "primary_contact": contact.get("name", "Unknown Contact"),
                "type": "Customer",
                "industry": contact.get("industry", "Technology"),
                "tier": contact.get(
                    "tier",
                    random.choices(
                        ["Enterprise", "Mid-Market", "SMB"], weights=[0.2, 0.5, 0.3]
                    )[0],
                ),
                "website": f"https://www.{org_name.lower().replace(' ', '')}.com",
                "billing_region": contact.get(
                    "billing_region",
                    random.choices(["NA", "EMEA", "APAC"], weights=[0.6, 0.3, 0.1])[0],
                ),
                "arr": contact.get(
                    "arr", random.choice([50000, 100000, 250000, 500000])
                ),
                "owner": contact.get("internal_liaison", "Unassigned"),
                "created_at": created_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "risk_flag": False,
            }
            self._sf_a.insert_one({**account, "_seq": 0})
            self._write(f"salesforce/accounts/{account_id}.json", account)
            logger.info(f"[crm] SF account seeded: {account_id} ({org_name})")

    def handle_inbound_complaint(
        self,
        event_facts: Dict,
        timestamp: str,
        date_str: str,
        day: int,
    ) -> Optional[str]:
        """
        Called by ExternalEmailIngestor when it processes a customer_complaint
        email. Creates a ZD ticket in MongoDB + disk and embeds it so Product
        planners see it the next morning.

        Returns the new ticket_id (e.g. 'ZD-101') or None if ZD is disabled.
        """
        if not self._zd_on:
            return None

        from memory import SimEvent

        seq = self._zd_counter
        self._zd_counter += 1
        ticket_id = f"ZD-{seq}"

        ticket = {
            "ticket_id": ticket_id,
            "type": "incident",
            "status": "Open",
            "priority": "Normal",
            "description": event_facts.get("body", "(See email body.)"),
            "requester": {
                "name": event_facts.get("sender_name", "Customer"),
                "email": event_facts.get("sender", "customer@unknown.com"),
                "org_name": event_facts.get("sender_org", "Unknown"),
            },
            "subject": event_facts.get("subject", "Customer complaint"),
            "org_name": event_facts.get("sender_org", "Unknown"),
            "channel": "email",
            "tags": ["support", "inbound", "needs_triage"],
            "satisfaction_rating": {"score": "unoffered"},
            "created_at": timestamp,
            "updated_at": timestamp,
            "related_incident": None,
            "comments": [
                {
                    "author": "Customer",
                    "text": event_facts.get("body", "(See email body.)"),
                    "timestamp": timestamp,
                }
            ],
            "_seq": seq,
        }

        self._zd.insert_one(ticket)
        self._write(
            f"zendesk/tickets/{ticket_id}.json",
            {k: v for k, v in ticket.items() if k != "_id"},
        )
        self._write_zd_comment(ticket_id, ticket["comments"][0])

        self._embed(
            id=ticket_id,
            artifact_type="zd_ticket",
            title=f"[{ticket_id}] {ticket['subject']}",
            content=f"Customer: {ticket['org_name']}\n{ticket['comments'][0]['text']}",
            day=day,
            date=date_str,
            timestamp=timestamp,
            metadata={
                "ticket_id": ticket_id,
                "org_name": ticket["org_name"],
                "status": "Open",
            },
        )

        self._emit(
            SimEvent(
                type="zd_ticket_opened",
                timestamp=timestamp,
                day=day,
                date=date_str,
                actors=[],
                artifact_ids={"zd_ticket": ticket_id},
                facts={
                    "ticket_id": ticket_id,
                    "subject": ticket["subject"],
                    "org_name": ticket["org_name"],
                    "channel": "email",
                },
                summary=f"Zendesk ticket {ticket_id} opened: {ticket['subject']} ({ticket['org_name']})",
                tags=["zendesk", "support", "customer_complaint"],
            )
        )

        logger.info(f"[crm] ZD ticket opened: {ticket_id} ({ticket['org_name']})")
        return ticket_id

    def _write_zd_comment(self, ticket_id: str, comment: Dict) -> None:
        """Write a single comment to disk under zendesk/comments/{ticket_id}/."""
        comment_dir = self._base / "zendesk" / "comments" / ticket_id
        comment_dir.mkdir(parents=True, exist_ok=True)

        safe_ts = comment["timestamp"].replace(":", "-").replace(".", "-")
        path = comment_dir / f"{safe_ts}.json"

        if path.exists():
            path = comment_dir / f"{safe_ts}_{id(comment)}.json"
        with open(path, "w") as fh:
            json.dump(comment, fh, indent=2)

    def _add_zd_comment(
        self,
        ticket_id: str,
        text: str,
        author: str,
        timestamp: str,
    ) -> None:
        """
        Append a comment to a live ZD ticket in Mongo + disk.
        Used by incident escalation and resolution handlers.
        """
        comment = {"author": author, "text": text, "timestamp": timestamp}
        self._zd.update_one(
            {"ticket_id": ticket_id},
            {"$push": {"comments": comment}, "$set": {"updated_at": timestamp}},
        )

        doc = self._zd.find_one({"ticket_id": ticket_id}, {"_id": 0, "_seq": 0})
        if doc:
            self._write(f"zendesk/tickets/{ticket_id}.json", doc)
        self._write_zd_comment(ticket_id, comment)

    def handle_incident_opened(
        self,
        incident_id: str,
        component: str,
        health: int,
        timestamp: str,
        date_str: str,
        day: int,
    ) -> None:
        """
        Called immediately after _handle_incident() logs the incident_opened
        SimEvent in flow.py.

        ZD path: escalates all currently-open tickets to Urgent and links them
        to the incident. These escalations will appear in planner_context() the
        next morning.

        SF path: when health < 60, appends a risk note to every open opportunity
        so Sales planners see the risk in their daily context.
        """
        from memory import SimEvent

        if self._zd_on and self._zd_cfg.get("link_to_incidents", True):
            open_tickets = list(self._zd.find({"status": "Open"}, {"_id": 0}))
            escalated_ids = []

            for t in open_tickets:
                tid = t["ticket_id"]
                self._zd.update_one(
                    {"ticket_id": tid},
                    {
                        "$set": {
                            "priority": "Urgent",
                            "related_incident": incident_id,
                            "updated_at": timestamp,
                        },
                        "$addToSet": {
                            "tags": {"$each": ["escalated", "incident_linked"]}
                        },
                    },
                )

                doc = self._zd.find_one({"ticket_id": tid}, {"_id": 0, "_seq": 0})
                if doc:
                    self._write(f"zendesk/tickets/{tid}.json", doc)

                self._add_zd_comment(
                    tid,
                    f"SYSTEM: Escalated — active infrastructure incident "
                    f"{incident_id} (component: {component}) may be related.",
                    "ZD Bot",
                    timestamp,
                )
                escalated_ids.append(tid)

            if escalated_ids:
                self._emit(
                    SimEvent(
                        type="zd_tickets_escalated",
                        timestamp=timestamp,
                        day=day,
                        date=date_str,
                        actors=[],
                        artifact_ids={"jira": incident_id, "zd_tickets": escalated_ids},
                        facts={
                            "incident_id": incident_id,
                            "component": component,
                            "escalated_count": len(escalated_ids),
                            "ticket_ids": escalated_ids,
                        },
                        summary=(
                            f"{len(escalated_ids)} ZD ticket(s) escalated to Urgent "
                            f"due to incident {incident_id}."
                        ),
                        tags=["zendesk", "escalation", "incident"],
                    )
                )
                logger.info(
                    f"[crm] {len(escalated_ids)} ZD ticket(s) escalated → {incident_id}"
                )

        if self._sf_on and health < 60:
            open_opps = list(
                self._sf_o.find(
                    {"stage": {"$nin": ["Closed Won", "Closed Lost"]}},
                    {"_id": 0},
                )
            )
            risk_note = (
                f"Active SEV on {component} ({incident_id}) — "
                f"system health {health}/100 — potential SLA impact."
            )
            flagged_ids = []
            flagged_orgs = []

            for opp in open_opps:
                oid = opp["opportunity_id"]
                org = opp.get("account_name")

                self._sf_o.update_one(
                    {"opportunity_id": oid},
                    {
                        "$push": {"risk_notes": risk_note},
                        "$set": {"updated_at": timestamp},
                    },
                )
                doc = self._sf_o.find_one(
                    {"opportunity_id": oid}, {"_id": 0, "_seq": 0}
                )
                if doc:
                    self._write(f"salesforce/opportunities/{oid}.json", doc)

                flagged_ids.append(oid)

                if org and org not in flagged_orgs:
                    flagged_orgs.append(org)

            if flagged_ids:
                self._emit(
                    SimEvent(
                        type="sf_deals_risk_flagged",
                        timestamp=timestamp,
                        day=day,
                        date=date_str,
                        actors=[],
                        artifact_ids={"jira": incident_id, "sf_opps": flagged_ids},
                        facts={
                            "incident_id": incident_id,
                            "health": health,
                            "flagged_count": len(flagged_ids),
                            "opp_ids": flagged_ids,
                            "account_names": flagged_orgs,
                            "risk_note": risk_note,
                        },
                        summary=(
                            f"{len(flagged_ids)} SF deal(s) flagged at-risk "
                            f"due to incident {incident_id} (health={health})."
                        ),
                        tags=["salesforce", "risk", "incident"],
                    )
                )
                logger.info(
                    f"[crm] {len(flagged_ids)} SF deal(s) flagged at-risk "
                    f"→ {incident_id}"
                )

    def handle_incident_resolved(
        self,
        incident_id: str,
        postmortem_link: str,
        timestamp: str,
        date_str: str,
        day: int,
    ) -> None:
        """
        Called from _advance_incidents() when inc.stage transitions to
        'resolved'. Closes any ZD tickets that were linked to this incident.
        """
        if not self._zd_on:
            return

        from memory import SimEvent

        linked = list(self._zd.find({"related_incident": incident_id}, {"_id": 0}))
        resolved_ids = []

        for t in linked:
            tid = t["ticket_id"]
            self._zd.update_one(
                {"ticket_id": tid},
                {"$set": {"status": "Solved", "updated_at": timestamp}},
            )
            doc = self._zd.find_one({"ticket_id": tid}, {"_id": 0, "_seq": 0})
            if doc:
                self._write(f"zendesk/tickets/{tid}.json", doc)

            self._add_zd_comment(
                tid,
                (
                    f"The underlying engineering issue has been resolved. "
                    f"Postmortem: {postmortem_link}. "
                    f"Thank you for your patience."
                ),
                "Support Agent",
                timestamp,
            )
            resolved_ids.append(tid)

        if resolved_ids:
            self._emit(
                SimEvent(
                    type="zd_tickets_resolved",
                    timestamp=timestamp,
                    day=day,
                    date=date_str,
                    actors=[],
                    artifact_ids={"jira": incident_id, "zd_tickets": resolved_ids},
                    facts={
                        "incident_id": incident_id,
                        "postmortem_link": postmortem_link,
                        "resolved_count": len(resolved_ids),
                        "ticket_ids": resolved_ids,
                    },
                    summary=(
                        f"{len(resolved_ids)} ZD ticket(s) resolved after "
                        f"incident {incident_id} closed."
                    ),
                    tags=["zendesk", "resolved", "incident"],
                )
            )
            logger.info(
                f"[crm] {len(resolved_ids)} ZD ticket(s) resolved → {incident_id}"
            )

    def process_outbound_email(
        self,
        email_data: Dict,
        timestamp: str,
        date_str: str,
        day: int,
    ) -> Optional[Dict]:
        """
        After each outbound email is emitted. If the subject line contains
        sales-intent keywords, creates or advances a Salesforce opportunity
        and emits a crm_touchpoint SimEvent.

        Returns the crm_touchpoint event facts dict (or None) so the caller
        can optionally log additional context.
        """
        if not self._sf_on:
            return None

        from memory import SimEvent

        sender = email_data.get("sender", "")
        recipient = email_data.get("recipient", "")
        sender_org = email_data.get("sender_org", "")
        recip_org = email_data.get("recipient_org", email_data.get("to_org", "Unknown"))

        stage = email_data.get("stage", "Prospecting")

        safe_org = recip_org.upper().replace(" ", "").replace("-", "")
        existing = self._sf_o.find_one(
            {
                "account_name": recip_org,
                "stage": {"$nin": ["Closed Won", "Closed Lost"]},
            },
            {"_id": 0},
        )

        if existing:
            oid = existing["opportunity_id"]
            _STAGE_RANK = {
                "Prospecting": 1,
                "Value Proposition": 2,
                "Proposal/Price Quote": 3,
                "Negotiation/Review": 4,
                "Closed Won": 5,
                "Closed Lost": 0,
            }
            if _STAGE_RANK.get(stage, 1) > _STAGE_RANK.get(
                existing.get("stage", "Prospecting"), 1
            ):
                self._sf_o.update_one(
                    {"opportunity_id": oid},
                    {
                        "$set": {
                            "stage": stage,
                            "probability": _STAGE_PROBABILITIES.get(stage, 10),
                            "updated_at": timestamp,
                        }
                    },
                )

            self._sf_o.update_one(
                {"opportunity_id": oid},
                {
                    "$push": {
                        "touchpoints": {
                            "sender": sender,
                            "subject": email_data.get("subject", ""),
                            "timestamp": timestamp,
                        }
                    }
                },
            )
            doc = self._sf_o.find_one({"opportunity_id": oid}, {"_id": 0, "_seq": 0})
            if doc:
                self._write(f"salesforce/opportunities/{oid}.json", doc)
        else:
            seq = self._opp_counter
            self._opp_counter += 1
            oid = f"OPP-{seq}"

            try:
                ts_dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except ValueError:
                ts_dt = datetime.utcnow()
            close_date = (ts_dt + timedelta(days=random.randint(30, 90))).strftime(
                "%Y-%m-%d"
            )

            opp = {
                "opportunity_id": oid,
                "account_name": recip_org,
                "type": random.choices(_SF_TYPES, weights=_SF_TYPE_WEIGHTS)[0],
                "stage": stage,
                "probability": _STAGE_PROBABILITIES.get(stage, 10),
                "amount": random.choice([15000, 35000, 50000, 85000, 120000]),
                "close_date": close_date,
                "owner": sender,
                "lead_source": random.choices(
                    _SF_LEAD_SOURCES, weights=_SF_LEAD_SOURCE_WEIGHTS
                )[0],
                "next_step": "Awaiting customer response",
                "created_at": timestamp,
                "updated_at": timestamp,
                "risk_notes": [],
                "touchpoints": [
                    {
                        "sender": sender,
                        "subject": email_data.get("subject", ""),
                        "timestamp": timestamp,
                    }
                ],
                "_seq": seq,
            }
            self._sf_o.insert_one(opp)
            self._write(
                f"salesforce/opportunities/{oid}.json",
                {k: v for k, v in opp.items() if k not in ("_id", "_seq")},
            )

       
            self._embed(
                id=oid,
                artifact_type="sf_opportunity",
                title=f"[{oid}] {recip_org} — {stage}",
                content=(
                    f"Account: {recip_org}\nStage: {stage}\n"
                    f"Owner: {sender}\nLast touchpoint: {email_data.get('subject', '')}"
                ),
                day=day,
                date=date_str,
                timestamp=timestamp,
                metadata={
                    "opportunity_id": oid,
                    "account_name": recip_org,
                    "stage": stage,
                },
            )
            logger.info(f"[crm] SF opportunity created: {oid} ({recip_org}, {stage})")

        touchpoint_facts = {
            "opportunity_id": oid,
            "account_name": recip_org,
            "stage": stage,
            "sender": sender,
            "subject": email_data.get("subject", ""),
        }

        self._emit(
            SimEvent(
                type="crm_touchpoint",
                timestamp=timestamp,
                day=day,
                date=date_str,
                actors=[sender],
                artifact_ids={"sf_opp": oid},
                facts=touchpoint_facts,
                summary=f"CRM touchpoint: {sender} → {recip_org} ({stage})",
                tags=["salesforce", "touchpoint", "sales"],
            )
        )

        return touchpoint_facts

    def handle_employee_departure(
        self,
        employee_name: str,
        role: str,
        date_str: str,
        day: int,
    ) -> None:
        """
        Called from OrgLifecycleManager when an employee departs. Mirrors the
        JIRA ticket reassignment cascade: if the departed employee owned any
        SF accounts or open opportunities, they are flagged for reassignment.

        This extends the departure cascade ground truth into the CRM layer
        so eval agents can answer cross-domain questions like:
        "Which customer accounts are currently without an owner?"
        """
        if not self._sf_on:
            return

        from memory import SimEvent

        reassigned_accounts = []
        reassigned_opps = []

        # Accounts
        for acc in self._sf_a.find({"owner": employee_name}, {"_id": 0}):
            aid = acc["account_id"]
            self._sf_a.update_one(
                {"account_id": aid},
                {"$set": {"owner": "Pending Reassignment", "risk_flag": True}},
            )
            doc = self._sf_a.find_one({"account_id": aid}, {"_id": 0, "_seq": 0})
            if doc:
                self._write(f"salesforce/accounts/{aid}.json", doc)
            reassigned_accounts.append(aid)

        for opp in self._sf_o.find(
            {
                "owner": employee_name,
                "stage": {"$nin": ["Closed Won", "Closed Lost"]},
            },
            {"_id": 0},
        ):
            oid = opp["opportunity_id"]
            risk_note = (
                f"Owner {employee_name} ({role}) departed on {date_str}. "
                f"Deal requires immediate reassignment."
            )
            self._sf_o.update_one(
                {"opportunity_id": oid},
                {
                    "$set": {"owner": "Pending Reassignment"},
                    "$push": {"risk_notes": risk_note},
                },
            )
            doc = self._sf_o.find_one({"opportunity_id": oid}, {"_id": 0, "_seq": 0})
            if doc:
                self._write(f"salesforce/opportunities/{oid}.json", doc)
            reassigned_opps.append(oid)

        if reassigned_accounts or reassigned_opps:
            ts = datetime.utcnow().isoformat()
            self._emit(
                SimEvent(
                    type="sf_ownership_lapsed",
                    timestamp=ts,
                    day=day,
                    date=date_str,
                    actors=[employee_name],
                    artifact_ids={
                        "sf_accounts": reassigned_accounts,
                        "sf_opps": reassigned_opps,
                    },
                    facts={
                        "departed_employee": employee_name,
                        "role": role,
                        "accounts_lapsed": reassigned_accounts,
                        "opportunities_lapsed": reassigned_opps,
                    },
                    summary=(
                        f"SF ownership lapsed after {employee_name} departure: "
                        f"{len(reassigned_accounts)} account(s), "
                        f"{len(reassigned_opps)} open deal(s) pending reassignment."
                    ),
                    tags=["salesforce", "lifecycle", "employee_departed"],
                )
            )
            logger.info(
                f"[crm] SF ownership lapsed: {employee_name} → "
                f"{len(reassigned_accounts)} accounts, "
                f"{len(reassigned_opps)} open opps"
            )
