"""
external_email_ingest.py
========================
Inbound and outbound email generation during the simulation.

Three email categories, each with distinct routing and causal tracing:

  1. TECH VENDOR INBOUND  (AWS, Stripe, Snyk, etc.)
     Arrive pre-standup (06:00–08:59). Routed to the Engineering member
     whose expertise best overlaps the topic — not just the dept lead.
     May produce a JIRA task. Appended to any live incident's causal chain
     if the topic overlaps the root cause.

  2. CUSTOMER / CLIENT INBOUND
     Arrive during business hours (09:00–16:30). Routed through the
     product gatekeeper chain:
       customer email → Sales Slack ping → Product decision → optional JIRA
     ~15 % of customer emails are dropped (no action taken). These are
     logged as "email_dropped" SimEvents — an eval agent should detect the
     gap between the email artifact and the absence of any downstream work.

  3. HR OUTBOUND  (offer letters, onboarding prep)
     Fired 1–3 days before a scheduled hire arrives. Karen (HR lead) sends
     to the prospect. Logged as an artifact and linked into the
     employee_hired causal chain on arrival day.

Causal tracing
--------------
Every email is assigned an embed_id and rooted in a CausalChainHandler.
Each downstream artifact (Slack thread, JIRA ticket) is appended in order.
Chain snapshots are written into every SimEvent's facts so the eval harness
can reconstruct the full thread — or notice when it terminates early.

Source generation
-----------------
Sources are generated once by LLM during genesis (same pattern as
generate_tech_stack in confluence_writer.py), persisted to
sim_config["inbound_email_sources"], and loaded on every subsequent run.
No config.yaml entries required.
"""

from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass, field
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent_factory import make_agent
from causal_chain_handler import CausalChainHandler
from config_loader import COMPANY_DESCRIPTION
from crm_system import NullCRMSystem
from crewai import Crew, Task
import json_repair
from memory import Memory, SimEvent
from insider_threat import _NullInjector
from utils.persona_utils import get_voice_card

logger = logging.getLogger("orgforge.external_email")


_PROB_ALWAYS = 0.40
_PROB_INCIDENT = 0.70
_PROB_INCIDENT_QUIET = 0.10
_HEALTH_THRESHOLD = 60
_PROB_EMAIL_DROPPED = 0.15  # customer emails dropped with no action
_PROB_CUSTOMER_JIRA = 0.55  # high-priority customer complaint → JIRA
_PROB_VENDOR_JIRA = 0.45  # vendor alert → JIRA task
_HR_EMAIL_WINDOW = (1, 3)  # days before hire arrival to send email


@dataclass
class ExternalEmailSignal:
    """
    Structured signal from one email.
    Carries a live CausalChainHandler so every downstream artifact
    (Slack ping, JIRA ticket) can be appended in order.
    Passed to DepartmentPlanner prompts for vendor emails only —
    dropped emails and out-of-hours customer emails are excluded.
    """

    source_name: str
    source_org: str
    source_email: str
    internal_liaison: str
    subject: str
    body_preview: str
    full_body: str
    tone: str
    topic: str
    timestamp_iso: str
    embed_id: str
    category: str
    dropped: bool = False
    eml_path: str = ""
    causal_chain: Optional[CausalChainHandler] = None
    facts: Dict[str, Any] = field(default_factory=dict)

    @property
    def as_cross_signal_text(self) -> str:
        """Only non-dropped vendor emails surface in planning prompts."""
        if self.dropped or self.category != "vendor":
            return ""
        return (
            f"[INBOUND EMAIL] From: {self.source_name} ({self.source_org}) "
            f'— Subject: "{self.subject}" — Preview: {self.body_preview}'
        )


class ExternalEmailIngestor:
    """
    Manages genesis-time source generation and all email flows during the sim.
    """

    _MONGO_KEY = "inbound_email_sources"

    def __init__(
        self,
        config: dict,
        mem: Memory,
        worker_llm,
        planner_llm,
        export_dir,
        leads: Dict[str, str],
        org_chart: Dict[str, List[str]],
        personas: Dict[str, dict],
        registry,
        clock,
        threat_injector=None,
        crm=None,
    ):
        self._config = config
        self._mem = mem
        self._worker_llm = worker_llm
        self._planner_llm = planner_llm
        self._export_dir = Path(export_dir)
        self._leads = leads
        self._org_chart = org_chart
        self._personas = personas
        self._registry = registry
        self._clock = clock
        self._company_name: str = config.get("simulation", {}).get(
            "company_name", "the company"
        )
        self._industry: str = config.get("simulation", {}).get("industry", "technology")
        self._domain: str = config.get("simulation", {}).get("domain", "company.com")
        self._company_desc: str = config.get("simulation", {}).get(
            "company_description", f"a {self._industry} company"
        )
        self._threat = threat_injector or _NullInjector()
        self._crm = crm or NullCRMSystem()

        self._scheduled_hires: Dict[int, List[dict]] = {}
        for hire in config.get("org_lifecycle", {}).get("scheduled_hires", []):
            self._scheduled_hires.setdefault(hire["day"], []).append(hire)

        @property
        def _sources(self):
            """Lazy lookup: pull sources from MongoDB only when needed."""
            doc = self._mem._db["sim_config"].find_one({"_id": "inbound_email_sources"})
            return doc.get("sources", []) if doc else []


    def generate_pre_standup(self, state) -> List[ExternalEmailSignal]:
        """
        Vendor / automated alerts arriving 06:00–08:59.
        Returns signals injected into DepartmentPlanner prompts.
        Also routes each signal to the best-matched engineer and may open a JIRA.
        """
        self._ensure_sources_loaded()
        signals: List[ExternalEmailSignal] = []
        has_incident = bool(state.active_incidents)

        for source in self._sources or []:
            if source.get("category") != "vendor":
                continue
            if not self._should_fire(
                source.get("trigger_on", ["always"]), state.system_health, has_incident
            ):
                continue
            topic = random.choice(source.get("topics", ["general update"]))
            signal = self._generate_email(
                source, topic, state, hour_range=(6, 8), category="vendor"
            )
            if not signal:
                continue
            self._route_vendor_email(signal, state)
            signals.append(signal)

        if signals:
            logger.info(f"  [cyan]📬 {len(signals)} vendor alert(s) pre-standup[/cyan]")
        return signals

    def generate_business_hours(self, state) -> List[ExternalEmailSignal]:
        """
        Customer emails arriving 09:00–16:30.
        Each non-dropped email triggers: Sales Slack ping → Product decision → optional JIRA.
        Dropped emails (~15%) are logged as "email_dropped" SimEvents.
        Returns all signals (dropped + routed) for tomorrow's CrossDeptSignal extraction.
        """
        self._ensure_sources_loaded()
        signals: List[ExternalEmailSignal] = []
        has_incident = bool(state.active_incidents)

        for source in self._sources or []:
            if source.get("category") != "customer":
                continue
            if not self._should_fire(
                source.get("trigger_on", ["always"]), state.system_health, has_incident
            ):
                continue
            topic = random.choice(source.get("topics", ["general update"]))
            signal = self._generate_email(
                source, topic, state, hour_range=(9, 16), category="customer"
            )
            if not signal:
                continue

            if random.random() < _PROB_EMAIL_DROPPED:
                signal.dropped = True
                self._log_dropped_email(signal, state)
                logger.info(
                    f"    [dim yellow]📭 Dropped (no action): "
                    f'{signal.source_name} → "{signal.subject[:50]}"[/dim yellow]'
                )
            else:
                self._route_customer_email(signal, state)

            signals.append(signal)

        if signals:
            n_routed = sum(1 for s in signals if not s.dropped)
            n_dropped = len(signals) - n_routed
            logger.info(
                f"  [cyan]📬 {n_routed} customer email(s) routed, "
                f"{n_dropped} dropped[/cyan]"
            )
        return signals

    def generate_hr_outbound(self, state) -> None:
        """
        Fires 1–3 days before a scheduled hire arrives.
        Sends offer letter (3 days out) or onboarding prep (1-2 days out).
        Stores embed_id on hire config so employee_hired can reference it.
        """
        hr_lead = self._leads.get("HR_Ops", next(iter(self._leads.values())))
        date_str = str(state.current_date.date())

        for target_day, hires in self._scheduled_hires.items():
            days_until = target_day - state.day
            if days_until not in range(_HR_EMAIL_WINDOW[0], _HR_EMAIL_WINDOW[1] + 1):
                continue
            for hire in hires:
                if hire.get("_hr_email_sent"):
                    continue
                self._send_hr_outbound(hire, hr_lead, days_until, state, date_str)
                hire["_hr_email_sent"] = True

    def _route_customer_email(self, signal: ExternalEmailSignal, state) -> None:
        date_str = str(state.current_date.date())
        sales_lead = self._leads.get(
            signal.internal_liaison, next(iter(self._leads.values()))
        )
        product_dept = next((d for d in self._leads if "product" in d.lower()), None)
        product_lead = self._leads.get(product_dept, sales_lead)

        thread_id = self._sales_pings_product(
            signal, sales_lead, product_lead, state, date_str
        )
        if thread_id:
            signal.causal_chain.append(thread_id)

        is_high = signal.tone in ("frustrated", "urgent") or (
            state.system_health < 70 and "stability" in signal.topic.lower()
        )
        if is_high and random.random() < _PROB_CUSTOMER_JIRA:
            ticket_id = self._product_opens_jira(signal, product_lead, state, date_str)
            if ticket_id:
                signal.causal_chain.append(ticket_id)

        reply_id = self._send_customer_reply(
            signal, sales_lead, is_high, state, date_str
        )
        if reply_id:
            signal.causal_chain.append(reply_id)

        self._mem.log_event(
            SimEvent(
                type="customer_email_routed",
                timestamp=signal.timestamp_iso,
                day=state.day,
                date=date_str,
                actors=[signal.source_name, sales_lead, product_lead],
                artifact_ids={"email": signal.embed_id},
                facts={
                    "source": signal.source_name,
                    "subject": signal.subject,
                    "high_priority": is_high,
                    "causal_chain": signal.causal_chain.snapshot(),
                },
                summary=(
                    f"Customer email from {signal.source_name} routed: "
                    f"{sales_lead} → {product_lead}"
                    + (" [JIRA opened]" if len(signal.causal_chain) > 2 else "")
                ),
                tags=["email", "customer", "routed", "causal_chain"],
            )
        )

    def _sales_pings_product(
        self, signal, sales_lead, product_lead, state, date_str
    ) -> Optional[str]:
        participants = [sales_lead, product_lead]
        ping_time, _ = self._clock.sync_and_advance(participants, hours=0.25)

        backstory = get_voice_card(
            sales_lead, "async", graph_dynamics=None, mem=self._mem
        )
        p = self._personas.get(sales_lead, {})

        agent = make_agent(
            role=f"{sales_lead} — {p.get('social_role', 'Sales Lead')}",
            goal="Summarise a customer email for Product on Slack naturally in your own voice.",
            backstory=backstory,
            llm=self._worker_llm,
        )
        task = Task(
            description=(
                f"You just read this email from {signal.source_name}:\n"
                f"Subject: {signal.subject}\n{signal.full_body}\n\n"
                f"Write a Slack message to {product_lead} (Product) that:\n"
                f"1. Summarises what {signal.source_name} is asking (2 sentences)\n"
                f"2. States urgency: high / medium / low\n"
                f"3. Ends with a concrete ask for {product_lead}\n"
                f"Under 80 words. No bullets. Write as {sales_lead} using your typing quirks."
            ),
            expected_output="Slack message under 80 words.",
            agent=agent,
        )
        text = str(Crew(agents=[agent], tasks=[task], verbose=False).kickoff()).strip()

        _, thread_id = self._mem.log_slack_messages(
            channel="product",
            messages=[
                {
                    "user": sales_lead,
                    "email": self._email_of(sales_lead),
                    "text": text,
                    "ts": ping_time.isoformat(),
                    "date": date_str,
                    "is_bot": False,
                    "metadata": {
                        "type": "customer_email_relay",
                        "source_email_id": signal.embed_id,
                        "customer": signal.source_name,
                    },
                }
            ],
            export_dir=self._export_dir,
        )

        self._mem.log_event(
            SimEvent(
                type="customer_escalation",
                timestamp=ping_time.isoformat(),
                day=state.day,
                date=date_str,
                actors=[sales_lead, product_lead],
                artifact_ids={
                    "slack_thread": thread_id,
                    "email": signal.embed_id,
                    "source_email": signal.embed_id,
                },
                facts={
                    "customer": signal.source_name,
                    "subject": signal.subject,
                    "relayed_by": sales_lead,
                    "product_gatekeeper": product_lead,
                    "causal_chain": signal.causal_chain.snapshot(),
                },
                summary=f"{sales_lead} relayed {signal.source_name} email to {product_lead} in #product",
                tags=["customer_escalation", "slack", "causal_chain"],
            )
        )
        return thread_id

    def _product_opens_jira(
        self, signal, product_lead, state, date_str
    ) -> Optional[str]:
        ticket_id = self._registry.next_jira_id("PROD")
        self._registry.register_jira(ticket_id)
        jira_time, _ = self._clock.sync_and_advance([product_lead], hours=0.3)

        backstory = get_voice_card(
            product_lead, "async", graph_dynamics=None, mem=self._mem
        )
        p = self._personas.get(product_lead, {})

        agent = make_agent(
            role=f"{product_lead} — {p.get('social_role', 'Product Manager')}",
            goal="Write a JIRA ticket from a customer complaint.",
            backstory=backstory,
            llm=self._worker_llm,
        )
        task = Task(
            description=(
                f"You are {product_lead}. Customer {signal.source_name} emailed:\n"
                f"Subject: {signal.subject}\n{signal.full_body}\n\n"
                f"Write a JIRA description under 80 words covering:\n"
                f"  - The customer issue\n  - Customer name + urgency\n"
                f"  - One acceptance criterion\nNo preamble."
            ),
            expected_output="JIRA description under 80 words.",
            agent=agent,
        )
        description = str(
            Crew(agents=[agent], tasks=[task], verbose=False).kickoff()
        ).strip()

        ticket = {
            "id": ticket_id,
            "title": f"[Customer] {signal.subject[:80]}",
            "description": description,
            "status": "To Do",
            "assignee": product_lead,
            "type": "task",
            "priority": "high" if signal.tone in ("frustrated", "urgent") else "medium",
            "source": "customer_email",
            "source_email_id": signal.embed_id,
            "customer": signal.source_name,
            "causal_chain": signal.causal_chain.snapshot(),
            "created_at": jira_time.isoformat(),
            "updated_at": jira_time.isoformat(),
        }
        self._mem.upsert_ticket(ticket)
        self._mem.embed_artifact(
            id=ticket_id,
            type="jira",
            title=ticket["title"],
            content=json.dumps(ticket),
            day=state.day,
            date=date_str,
            timestamp=jira_time.isoformat(),
            metadata={
                "source": "customer_email",
                "customer": signal.source_name,
                "causal_chain": signal.causal_chain.snapshot(),
            },
        )
        self._mem.log_event(
            SimEvent(
                type="jira_ticket_created",
                timestamp=jira_time.isoformat(),
                day=state.day,
                date=date_str,
                actors=[product_lead],
                artifact_ids={"jira": ticket_id, "source_email": signal.embed_id},
                facts={
                    "title": ticket["title"],
                    "source": "customer_email",
                    "customer": signal.source_name,
                    "causal_chain": signal.causal_chain.snapshot(),
                },
                summary=f"{product_lead} opened {ticket_id} from {signal.source_name} email",
                tags=["jira", "customer", "causal_chain"],
            )
        )
        logger.info(
            f"    [green]🎫 {ticket_id}[/green] opened by {product_lead} "
            f"from {signal.source_name} complaint"
        )
        return ticket_id

    def _route_vendor_email(self, signal: ExternalEmailSignal, state) -> None:
        date_str = str(state.current_date.date())
        recipient = self._find_expert_for_topic(signal.topic, signal.internal_liaison)

        for inc in state.active_incidents:
            if any(
                kw in signal.topic.lower()
                for kw in inc.root_cause.lower().split()
                if len(kw) > 4
            ):
                inc.causal_chain.append(signal.embed_id)
                logger.info(
                    f"    [dim]🔗 Vendor email appended to {inc.ticket_id} chain[/dim]"
                )
                break

        if signal.tone == "urgent" or random.random() < _PROB_VENDOR_JIRA:
            ticket_id = self._engineer_opens_jira(signal, recipient, state, date_str)
            if ticket_id:
                signal.causal_chain.append(ticket_id)

        ack_id = self._send_vendor_ack(signal, recipient, state, date_str)
        if ack_id:
            signal.causal_chain.append(ack_id)

        self._mem.log_event(
            SimEvent(
                type="vendor_email_routed",
                timestamp=signal.timestamp_iso,
                day=state.day,
                date=date_str,
                actors=[signal.source_name, recipient],
                artifact_ids={"email": signal.embed_id},
                facts={
                    "vendor": signal.source_name,
                    "topic": signal.topic,
                    "routed_to": recipient,
                    "causal_chain": signal.causal_chain.snapshot(),
                },
                summary=f"Vendor email from {signal.source_name} routed to {recipient}",
                tags=["email", "vendor", "routed"],
            )
        )

    def _find_expert_for_topic(self, topic: str, liaison_dept: str) -> str:
        """Pick the dept member whose expertise tags best overlap the email topic."""
        topic_lower = topic.lower()
        members = self._org_chart.get(liaison_dept, [])
        lead = self._leads.get(liaison_dept, members[0] if members else "")
        best, best_score = lead, 0
        for name in members:
            tags = [
                e.lower() for e in self._personas.get(name, {}).get("expertise", [])
            ]
            score = sum(1 for t in tags if t in topic_lower)
            if score > best_score:
                best, best_score = name, score
        return best

    def _engineer_opens_jira(self, signal, assignee, state, date_str) -> Optional[str]:
        ticket_id = self._registry.next_jira_id()
        self._registry.register_jira(ticket_id)
        jira_time, _ = self._clock.sync_and_advance([assignee], hours=0.25)

        ticket = {
            "id": ticket_id,
            "title": f"[Vendor] {signal.subject[:80]}",
            "description": f"From {signal.source_name}: {signal.body_preview}",
            "status": "To Do",
            "assignee": assignee,
            "type": "task",
            "priority": "high" if signal.tone == "urgent" else "medium",
            "source": "vendor_email",
            "source_email_id": signal.embed_id,
            "vendor": signal.source_name,
            "causal_chain": signal.causal_chain.snapshot(),
            "created_at": jira_time.isoformat(),
            "updated_at": jira_time.isoformat(),
        }
        self._mem.upsert_ticket(ticket)
        self._mem.embed_artifact(
            id=ticket_id,
            type="jira",
            title=ticket["title"],
            content=json.dumps(ticket),
            day=state.day,
            date=date_str,
            timestamp=jira_time.isoformat(),
            metadata={"source": "vendor_email", "vendor": signal.source_name},
        )
        self._mem.log_event(
            SimEvent(
                type="jira_ticket_created",
                timestamp=jira_time.isoformat(),
                day=state.day,
                date=date_str,
                actors=[assignee],
                artifact_ids={"jira": ticket_id, "source_email": signal.embed_id},
                facts={
                    "title": ticket["title"],
                    "source": "vendor_email",
                    "vendor": signal.source_name,
                    "causal_chain": signal.causal_chain.snapshot(),
                },
                summary=f"{assignee} opened {ticket_id} from {signal.source_name} alert",
                tags=["jira", "vendor", "causal_chain"],
            )
        )
        logger.info(
            f"    [green]🎫 {ticket_id}[/green] opened by {assignee} "
            f"from {signal.source_name} alert"
        )
        return ticket_id

    def _send_hr_outbound(self, hire, hr_lead, days_until, state, date_str) -> None:
        name = hire["name"]
        role = hire.get("role", "team member")
        dept = hire.get("dept", "Engineering")
        email_type = (
            "offer_letter" if days_until == _HR_EMAIL_WINDOW[1] else "onboarding_prep"
        )
        subject = (
            f"Your offer — {role} at {self._company_name}"
            if email_type == "offer_letter"
            else f"Getting ready for Day 1 — {self._company_name}"
        )

        hr_time, _ = self._clock.sync_and_advance([hr_lead], hours=0.5)

        backstory = get_voice_card(hr_lead, "async", graph_dynamics=None, mem=self._mem)
        p = self._personas.get(hr_lead, {})

        agent = make_agent(
            role=f"{hr_lead} — {p.get('social_role', 'HR Lead')}",
            goal=f"Write a {email_type.replace('_', ' ')} to {name}.",
            backstory=backstory,
            llm=self._worker_llm,
        )
        task = Task(
            description=(
                f"You are {hr_lead} at {self._company_name}. Write a "
                f"{'warm offer letter' if email_type == 'offer_letter' else 'friendly onboarding prep email'} "
                f"to {name}, joining as {role} in {dept} in {days_until} day(s).\n"
                f"Under 120 words. Warm, professional. Use your typing quirks. No [PLACEHOLDER] tokens. "
                f"Output body only."
            ),
            expected_output="Email body under 120 words.",
            agent=agent,
        )
        body = str(Crew(agents=[agent], tasks=[task], verbose=False).kickoff()).strip()

        embed_id = f"hr_outbound_{name.lower()}_{state.day}_{email_type}"
        eml_path = self._write_eml(
            date_str=date_str,
            from_name=hr_lead,
            from_addr=self._email_of(hr_lead),
            to_name=name,
            to_addr=f"{name.lower()}@personal.email",
            subject=subject,
            body=body,
            timestamp_iso=hr_time.isoformat(),
            direction="outbound",
        )

        _exfil_path = self._threat.inject_email(
            eml_path=str(eml_path),
            sender=hr_lead,
            recipients=[f"{name.lower()}@personal.email"],
            subject_line=subject,
            day=state.day,
            current_date=state.current_date,
        )
        if _exfil_path:
            self._mem.embed_artifact(
                id=f"exfil_{state.day}_{hr_lead}_hr",
                type="email",
                title=f"Outbound email: {hr_lead} (day {state.day})",
                content=open(_exfil_path).read(),
                day=state.day,
                date=date_str,
                timestamp=hr_time.isoformat(),
                metadata={"sender": hr_lead, "exfil": True},
            )

        self._mem.embed_artifact(
            id=embed_id,
            type="email",
            title=subject,
            content=f"To: {name}\n\n{body}",
            day=state.day,
            date=date_str,
            timestamp=hr_time.isoformat(),
            metadata={
                "hr_lead": hr_lead,
                "prospect": name,
                "role": role,
                "dept": dept,
                "email_type": email_type,
                "direction": "outbound",
                "hire_day": hire["day"],
            },
        )

        hire["_hr_email_embed_id"] = embed_id

        self._mem.log_event(
            SimEvent(
                type="hr_outbound_email",
                timestamp=hr_time.isoformat(),
                day=state.day,
                date=date_str,
                actors=[hr_lead, name],
                artifact_ids={"email": embed_id, "eml_path": str(eml_path)},
                facts={
                    "email_type": email_type,
                    "prospect": name,
                    "role": role,
                    "dept": dept,
                    "hire_day": hire["day"],
                    "days_until_arrival": days_until,
                },
                summary=(
                    f"{hr_lead} sent {email_type.replace('_', ' ')} to "
                    f"{name} ({days_until}d before arrival)"
                ),
                tags=["email", "hr", "outbound", email_type],
            )
        )
        logger.info(
            f"  [magenta]📤 HR → {name}:[/magenta] {subject} "
            f"({days_until}d before Day {hire['day']})"
        )

    def _send_customer_reply(
        self,
        signal: ExternalEmailSignal,
        sales_lead: str,
        is_high: bool,
        state,
        date_str: str,
    ) -> Optional[str]:
        """
        Sales lead sends an acknowledgment / follow-up reply to the customer.
        Always fires for routed (non-dropped) customer emails.
        High-priority emails get a more substantive response that references
        the internal escalation and promises a follow-up timeline.
        """
        reply_time, _ = self._clock.sync_and_advance([sales_lead], hours=0.4)

        urgency_hint = (
            "This is high priority — acknowledge urgency, mention it has been escalated "
            "internally, and commit to a follow-up within 24 hours."
            if is_high
            else "This is routine — thank them, confirm receipt, and say the team will be in touch."
        )

        backstory = get_voice_card(
            sales_lead, "async", graph_dynamics=None, mem=self._mem
        )
        p = self._personas.get(sales_lead, {})

        agent = make_agent(
            role=f"{sales_lead} — {p.get('social_role', 'Sales Lead')}",
            goal=f"Reply to a customer email on behalf of {self._company_name}.",
            backstory=backstory,
            llm=self._worker_llm,
        )
        task = Task(
            description=(
                f"You are {sales_lead} at {self._company_name}. "
                f"You received this email from {signal.source_name}:\n"
                f"Subject: {signal.subject}\n{signal.full_body}\n\n"
                f"{urgency_hint}\n"
                f"Under 80 words. Professional, warm. Use your typing quirks.\n\n"
                f"Respond ONLY with a JSON object in this exact format:\n"
                f"{{\n"
                f'  "body": "<the email text>",\n'
                f'  "crm_stage": "Choose EXACTLY ONE: Prospecting | Value Proposition | Proposal/Price Quote | Negotiation/Review"\n'
                f"}}"
            ),
            expected_output="Valid JSON object with 'body' and 'crm_stage' keys.",
            agent=agent,
        )
        try:
            raw = str(
                Crew(agents=[agent], tasks=[task], verbose=False).kickoff()
            ).strip()
            parsed = json_repair.loads(raw)

            if isinstance(parsed, dict):
                data = parsed
            elif (
                isinstance(parsed, list)
                and len(parsed) > 0
                and isinstance(parsed[0], dict)
            ):
                data = parsed[0]
            else:
                data = {}

            body = data.get(
                "body", "Thank you for reaching out. We will be in touch shortly."
            )
            crm_stage = data.get("crm_stage", "Prospecting")
        except Exception as exc:
            logger.warning(f"[external_email] Customer reply LLM failed: {exc}")
            return None

        embed_id = (
            f"reply_customer_{signal.source_name.lower().replace(' ', '_')}_{state.day}"
        )
        subject = f"Re: {signal.subject}"

        eml_path = self._write_eml(
            date_str=date_str,
            from_name=sales_lead,
            from_addr=self._email_of(sales_lead),
            to_name=signal.source_name,
            to_addr=signal.source_email,
            subject=subject,
            body=body,
            timestamp_iso=reply_time.isoformat(),
            direction="outbound",
        )

        self._crm.process_outbound_email(
            email_data={
                "sender": sales_lead,
                "recipient": signal.source_name,
                "sender_org": self._company_name,
                "recipient_org": signal.source_org,
                "subject": subject,
                "stage": crm_stage,
            },
            timestamp=reply_time.isoformat(),
            date_str=date_str,
            day=state.day,
        )

        _exfil_path = self._threat.inject_email(
            eml_path=str(eml_path),
            sender=sales_lead,
            recipients=[signal.source_email],
            subject_line=subject,
            day=state.day,
            current_date=state.current_date,
        )
        if _exfil_path:
            self._mem.embed_artifact(
                id=f"exfil_{state.day}_{sales_lead}",
                type="email",
                title=f"Outbound email: {sales_lead} (day {state.day})",
                content=open(_exfil_path).read(),
                day=state.day,
                date=date_str,
                timestamp=reply_time.isoformat(),
                metadata={"sender": sales_lead, "exfil": True},
            )

        self._mem.embed_artifact(
            id=embed_id,
            type="email",
            title=subject,
            content=f"To: {signal.source_name}\n\n{body}",
            day=state.day,
            date=date_str,
            timestamp=reply_time.isoformat(),
            metadata={
                "direction": "outbound",
                "reply_to_email_id": signal.embed_id,
                "customer": signal.source_name,
                "sent_by": sales_lead,
                "high_priority": is_high,
            },
        )

        self._mem.log_event(
            SimEvent(
                type="customer_reply_sent",
                timestamp=reply_time.isoformat(),
                day=state.day,
                date=date_str,
                actors=[sales_lead, signal.source_name],
                artifact_ids={
                    "eml_path": str(eml_path),
                    "embed_id": embed_id,
                    "source_email": signal.embed_id,
                },
                facts={
                    "customer": signal.source_name,
                    "subject": subject,
                    "sent_by": sales_lead,
                    "high_priority": is_high,
                    "causal_chain": signal.causal_chain.snapshot(),
                },
                summary=(
                    f'{sales_lead} replied to {signal.source_name}: "{subject[:80]}"'
                ),
                tags=["email", "outbound", "customer_reply", "causal_chain"],
            )
        )
        logger.info(f"    [cyan]📤 Reply → {signal.source_name}:[/cyan] {subject[:80]}")
        return embed_id

    def _send_vendor_ack(
        self,
        signal: ExternalEmailSignal,
        recipient: str,
        state,
        date_str: str,
    ) -> Optional[str]:
        """
        The assigned engineer sends a brief acknowledgment to the vendor.
        Confirms receipt, states the issue is being investigated, and
        references any JIRA ticket already in the causal chain.
        """
        ack_time, _ = self._clock.sync_and_advance([recipient], hours=0.3)

        jira_ref = next(
            (
                a
                for a in signal.causal_chain.snapshot()
                if str(a).startswith("JIRA-") or str(a).upper().startswith("PROJ-")
            ),
            None,
        )
        jira_hint = (
            f"Reference ticket {jira_ref} as the tracking issue."
            if jira_ref
            else "No ticket number yet — just say it is being investigated."
        )

        backstory = get_voice_card(
            recipient, "async", graph_dynamics=None, mem=self._mem
        )
        p = self._personas.get(recipient, {})

        agent = make_agent(
            role=f"{recipient} — {p.get('social_role', 'Engineer')}",
            goal=f"Acknowledge a vendor alert email on behalf of {self._company_name}.",
            backstory=backstory,
            llm=self._worker_llm,
        )
        task = Task(
            description=(
                f"You are {recipient} at {self._company_name}. "
                f"You received this alert from {signal.source_org}:\n"
                f"Subject: {signal.subject}\n{signal.full_body}\n\n"
                f"Write a short acknowledgment email. Confirm receipt, state "
                f"you are investigating. {jira_hint}\n"
                f"Under 60 words. Technical, professional. Use your typing quirks. Output body only."
            ),
            expected_output="Acknowledgment email body under 60 words.",
            agent=agent,
        )
        try:
            body = str(
                Crew(agents=[agent], tasks=[task], verbose=False).kickoff()
            ).strip()
        except Exception as exc:
            logger.warning(f"[external_email] Vendor ack LLM failed: {exc}")
            return None

        embed_id = (
            f"ack_vendor_{signal.source_name.lower().replace(' ', '_')}_{state.day}"
        )
        subject = f"Re: {signal.subject}"

        eml_path = self._write_eml(
            date_str=date_str,
            from_name=recipient,
            from_addr=self._email_of(recipient),
            to_name=signal.source_name,
            to_addr=signal.source_email,
            subject=subject,
            body=body,
            timestamp_iso=ack_time.isoformat(),
            direction="outbound",
        )

        _exfil_path = self._threat.inject_email(
            eml_path=str(eml_path),
            sender=recipient,
            recipients=[signal.source_email],
            subject_line=subject,
            day=state.day,
            current_date=state.current_date,
        )
        if _exfil_path:
            self._mem.embed_artifact(
                id=f"exfil_{state.day}_{recipient}",
                type="email",
                title=f"Outbound email: {recipient} (day {state.day})",
                content=open(_exfil_path).read(),
                day=state.day,
                date=date_str,
                timestamp=ack_time.isoformat(),
                metadata={"sender": recipient, "exfil": True},
            )

        self._mem.embed_artifact(
            id=embed_id,
            type="email",
            title=subject,
            content=f"To: {signal.source_name} ({signal.source_org})\n\n{body}",
            day=state.day,
            date=date_str,
            timestamp=ack_time.isoformat(),
            metadata={
                "direction": "outbound",
                "reply_to_email_id": signal.embed_id,
                "vendor": signal.source_name,
                "sent_by": recipient,
                "jira_ref": jira_ref,
            },
        )

        self._mem.log_event(
            SimEvent(
                type="vendor_ack_sent",
                timestamp=ack_time.isoformat(),
                day=state.day,
                date=date_str,
                actors=[recipient, signal.source_name],
                artifact_ids={
                    "eml_path": str(eml_path),
                    "embed_id": embed_id,
                    "source_email": signal.embed_id,
                },
                facts={
                    "vendor": signal.source_name,
                    "subject": subject,
                    "sent_by": recipient,
                    "jira_ref": jira_ref,
                    "causal_chain": signal.causal_chain.snapshot(),
                },
                summary=(
                    f'{recipient} acknowledged {signal.source_name}: "{subject[:80]}"'
                ),
                tags=["email", "outbound", "vendor_ack", "causal_chain"],
            )
        )
        logger.info(f"    [cyan]📤 Ack → {signal.source_name}:[/cyan] {subject[:80]}")
        return embed_id

    def _log_dropped_email(self, signal: ExternalEmailSignal, state) -> None:
        """
        Email artifact exists in the store but causal chain has no children.
        Eval agents should detect this gap.
        """

        liaison_name = self._leads.get(
            signal.internal_liaison, next(iter(self._leads.values()))
        )

        self._mem.log_event(
            SimEvent(
                type="email_dropped",
                timestamp=signal.timestamp_iso,
                day=state.day,
                date=str(state.current_date.date()),
                actors=[signal.source_name, liaison_name],
                artifact_ids={"email": signal.embed_id},
                facts={
                    "source": signal.source_name,
                    "subject": signal.subject,
                    "reason": "no_action_taken",
                    "causal_chain": signal.causal_chain.snapshot(),
                },
                summary=(
                    f"Email from {signal.source_name} received but not actioned: "
                    f'"{signal.subject[:60]}"'
                ),
                tags=["email", "dropped", "eval_signal"],
            )
        )

    # ─────────────────────────────────────────────────────────────────────────
    # CORE EMAIL GENERATION
    # ─────────────────────────────────────────────────────────────────────────

    def _generate_email(
        self,
        source: dict,
        topic: str,
        state,
        hour_range: Tuple[int, int],
        category: str,
    ) -> Optional[ExternalEmailSignal]:
        source_name = source["name"]
        source_org = source.get("org", source_name)
        source_addr = source.get("email", f"contact@{source_name.lower()}.com")
        liaison_dept = source.get("internal_liaison", list(self._leads.keys())[0])
        liaison_name = self._leads.get(liaison_dept, next(iter(self._leads.values())))
        tone = source.get("tone", "professional")
        date_str = str(state.current_date.date())

        incident_ctx = ""
        if state.active_incidents:
            inc = state.active_incidents[0]
            incident_ctx = (
                f"\nActive incident: {inc.ticket_id} — {inc.root_cause}. "
                f"Reference naturally if relevant."
            )

        # 1. Fetch the actual company tech stack from memory
        tech_stack = self._mem.tech_stack_for_prompt()
        tech_ctx = (
            (
                f"\nCOMPANY TECH STACK:\n{tech_stack}\n"
                f"CONSTRAINT: If referencing the company's current infrastructure or code, restrict it to the stack above. "
                f"You may reference outside technologies ONLY if suggesting a migration, offering a new service, or making a competitive recommendation."
            )
            if tech_stack
            else ""
        )

        email_ts = state.current_date.replace(
            hour=random.randint(*hour_range),
            minute=random.randint(0, 59),
            second=random.randint(0, 59),
        )

        agent = make_agent(
            role=f"Representative from {source_org}",
            goal=f"Write a realistic email about: {topic}.",
            backstory=(
                f"You represent {source_org}. Tone: {tone}. "
                f"Specific, concise. Never break character."
            ),
            llm=self._worker_llm,
        )
        task = Task(
            description=(
                f"Email from {source_org} to {liaison_name} at {self._company_name} "
                f"about: {topic}.\nTone: {tone}. Health: {state.system_health}/100."
                f"{incident_ctx}"
                f"{tech_ctx}\n\n"
                f"COMPANY CONTEXT: {self._company_name} is {self._company_desc}. "
                f"Ground your email in this reality.\n\n"
                f"Format:\nSUBJECT: <subject>\n---\n<body, 3-6 sentences, under 120 words>"
            ),
            expected_output="SUBJECT: <subject>\n---\n<body>",
            agent=agent,
        )

        try:
            raw = str(
                Crew(agents=[agent], tasks=[task], verbose=False).kickoff()
            ).strip()
        except Exception as exc:
            logger.warning(f"[external_email] LLM failed for {source_name}: {exc}")
            return None

        subject, body = self._parse_email_output(raw, topic)
        embed_id = (
            f"ext_email_{source_name.lower().replace(' ', '_')}"
            f"_{state.day}_{hour_range[0]}"
        )

        eml_path = self._write_eml(
            date_str=date_str,
            from_name=source_name,
            from_addr=source_addr,
            to_name=liaison_name,
            to_addr=self._email_of(liaison_name),
            subject=subject,
            body=body,
            timestamp_iso=email_ts.isoformat(),
        )

        self._mem.embed_artifact(
            id=embed_id,
            type="email",
            title=subject,
            content=f"From: {source_name} ({source_org})\n\n{body}",
            day=state.day,
            date=date_str,
            timestamp=email_ts.isoformat(),
            metadata={
                "source": source_name,
                "org": source_org,
                "category": category,
                "topic": topic,
                "liaison": liaison_name,
                "tone": tone,
                "direction": "inbound",
            },
        )

        self._mem.log_event(
            SimEvent(
                type="inbound_external_email",
                timestamp=email_ts.isoformat(),
                day=state.day,
                date=date_str,
                actors=[source_name, liaison_name],
                artifact_ids={"email": embed_id, "eml_path": str(eml_path)},
                facts={
                    "source": source_name,
                    "org": source_org,
                    "category": category,
                    "topic": topic,
                    "subject": subject,
                    "liaison": liaison_name,
                    "liaison_dept": liaison_dept,
                    "tone": tone,
                    "body_preview": body[:200],
                },
                summary=f'Inbound [{category}] email from {source_name}: "{subject}"',
                tags=["email", "inbound", category, source_name.lower()],
            )
        )

        zd_ticket_id = None
        if category == "customer":
            zd_ticket_id = self._crm.handle_inbound_complaint(
                event_facts={
                    "subject": subject,
                    "body": body[:500],
                    "sender_org": source_org,
                },
                timestamp=email_ts.isoformat(),
                date_str=date_str,
                day=state.day,
            )
            if zd_ticket_id:
                logger.info(
                    f"    [dim]🔗 ZD ticket {zd_ticket_id} linked to inbound complaint[/dim]"
                )

        artifact_ids = {"email": embed_id, "eml_path": str(eml_path)}
        if zd_ticket_id:
            artifact_ids["zd_ticket"] = zd_ticket_id

        body_preview = body[:200].rstrip() + ("…" if len(body) > 200 else "")

        return ExternalEmailSignal(
            source_name=source_name,
            source_org=source_org,
            source_email=source_addr,
            internal_liaison=liaison_dept,
            subject=subject,
            body_preview=body_preview,
            full_body=body,
            tone=tone,
            topic=topic,
            timestamp_iso=email_ts.isoformat(),
            embed_id=embed_id,
            category=category,
            eml_path=str(eml_path),
            causal_chain=CausalChainHandler(root_id=embed_id),
            facts={"subject": subject, "topic": topic, "org": source_org},
        )

    # ─────────────────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _ensure_sources_loaded(self) -> None:
        if self._sources is None:
            self._sources = self._mem.get_inbound_email_sources() or []

    @staticmethod
    def _should_fire(triggers, system_health, has_incident) -> bool:
        for t in triggers:
            if t == "always" and random.random() < _PROB_ALWAYS:
                return True
            if t == "incident":
                if random.random() < (
                    _PROB_INCIDENT if has_incident else _PROB_INCIDENT_QUIET
                ):
                    return True
            if t == "low_health" and system_health < _HEALTH_THRESHOLD:
                return True
        return False

    def _persona_hint(self, name: str) -> str:
        p = self._personas.get(name, {})
        return (
            f"You are {name}. {p.get('typing_quirks', 'Professional tone.')} "
            f"Never acknowledge being an AI."
        )

    def _email_of(self, name: str) -> str:
        return f"{name.lower()}@{self._domain}"

    @staticmethod
    def _parse_email_output(raw: str, topic: str) -> Tuple[str, str]:
        subject, body = f"Re: {topic}", raw
        for i, line in enumerate(raw.splitlines()):
            if line.upper().startswith("SUBJECT:"):
                subject = line[8:].strip()
                rest = "\n".join(raw.splitlines()[i + 1 :]).lstrip("-").strip()
                if rest:
                    body = rest
                break
        return subject, body

    @staticmethod
    def _parse_sources(raw: str) -> List[dict]:
        cleaned = re.sub(r"^```[a-z]*\n?", "", raw.strip()).rstrip("` \n")
        try:
            parsed = json.loads(cleaned)
            if not isinstance(parsed, list):
                raise ValueError("not a list")
            required = {
                "name",
                "org",
                "email",
                "category",
                "internal_liaison",
                "trigger_on",
                "topics",
            }
            return [s for s in parsed if required.issubset(s.keys())]
        except Exception as exc:
            logger.warning(f"[external_email] Source parse failed: {exc}")
            return []

    def _fallback_sources(self) -> List[dict]:
        eng = next(
            (d for d in self._leads if "eng" in d.lower()), list(self._leads.keys())[0]
        )
        sales = next(
            (d for d in self._leads if "sales" in d.lower()),
            list(self._leads.keys())[-1],
        )
        accounts = self._config.get("sales_accounts", ["Key Customer"])
        return [
            {
                "name": "AWS",
                "org": "Amazon Web Services",
                "email": "billing-alerts@aws.amazon.com",
                "category": "vendor",
                "internal_liaison": eng,
                "trigger_on": ["always", "low_health"],
                "tone": "formal",
                "topics": ["EC2 cost spike", "RDS storage warning", "quota limit"],
            },
            {
                "name": accounts[0],
                "org": accounts[0],
                "email": f"cto@{accounts[0].lower().replace(' ', '')}.com",
                "category": "customer",
                "internal_liaison": sales,
                "trigger_on": ["always", "incident"],
                "tone": "frustrated",
                "topics": [
                    "platform stability concerns",
                    "SLA questions",
                    "feature request",
                ],
            },
        ]

    def _write_eml(
        self,
        date_str,
        from_name,
        from_addr,
        to_name,
        to_addr,
        subject,
        body,
        timestamp_iso,
        direction="inbound",
    ) -> Path:
        out_dir = self._export_dir / "emails" / direction / date_str
        out_dir.mkdir(parents=True, exist_ok=True)
        safe = from_name.lower().replace(" ", "_").replace("/", "_")
        path = out_dir / f"{safe}.eml"
        msg = MIMEMultipart("alternative")
        msg["From"] = f"{from_name} <{from_addr}>"
        msg["To"] = f"{to_name} <{to_addr}>"
        msg["Subject"] = subject
        msg["Date"] = timestamp_iso
        msg["X-OrgForge-Direction"] = direction
        msg.attach(MIMEText(body, "plain"))
        with open(path, "w") as fh:
            fh.write(msg.as_string())
        return path
