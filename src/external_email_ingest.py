"""
external_email_ingest.py
========================
Inbound and outbound email generation during the simulation.
"""

from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass, field
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent_factory import make_agent
from causal_chain_handler import CausalChainHandler
from crm_system import NullCRMSystem
from crewai import Crew, Task
import json_repair
from memory import Memory, SimEvent
from insider_threat import _NullInjector
from utils.persona_utils import persona_utils

logger = logging.getLogger("orgforge.external_email")


_PROB_ALWAYS = 0.40
_PROB_INCIDENT = 0.70
_PROB_INCIDENT_QUIET = 0.10
_HEALTH_THRESHOLD = 60
_PROB_EMAIL_DROPPED = 0.15
_PROB_CUSTOMER_JIRA = 0.55
_PROB_VENDOR_JIRA = 0.45
_HR_EMAIL_WINDOW = (1, 3)
_VALID_EMAIL_TYPES = frozenset(
    ["complaint", "question", "feature_request", "positive_feedback", "general_inquiry"]
)

_VALID_CRM_STAGES = frozenset(
    [
        "Prospecting",
        "Value Proposition",
        "Proposal/Price Quote",
        "Negotiation/Review",
        "Closed Won",
        "Closed Lost",
    ]
)


_STAGE_RANK = {
    "Prospecting": 1,
    "Value Proposition": 2,
    "Proposal/Price Quote": 3,
    "Negotiation/Review": 4,
    "Closed Won": 5,
    "Closed Lost": 0,
}


def _get_stage_probability(stage: str) -> int:
    return {
        "Prospecting": 10,
        "Value Proposition": 25,
        "Proposal/Price Quote": 50,
        "Negotiation/Review": 75,
        "Closed Won": 100,
        "Closed Lost": 0,
    }.get(stage, 10)


_PROB_CUSTOMER_REPLY = 0.30

_PROB_NON_COMPLAINT_SALES_FYI = 0.35

_ZD_TICKET_TYPES = frozenset(["complaint", "question", "general_inquiry"])

_COMPLAINT_EMAIL_TYPES = frozenset(["complaint"])

_ZD_TICKET_PROB: dict = {
    "complaint": 1.0,
    "question": 0.70,
    "general_inquiry": 0.30,
}


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
        graph_dynamics=None,
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
        self._sources = None
        self._gd = graph_dynamics

        self._scheduled_hires: Dict[int, List[dict]] = {}
        for hire in config.get("org_lifecycle", {}).get("scheduled_hires", []):
            self._scheduled_hires.setdefault(hire["day"], []).append(hire)

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
        Customer emails arriving 09:00-16:30, driven entirely by simulation state.

        Emails are only generated when there is a real reason for a customer to
        reach out -- active incidents affecting their capabilities, stale deals,
        upcoming renewals, or expansion interest. Random probability firing is
        intentionally removed; silence is the correct output when nothing warrants
        an email.

        Dropped emails (~15%) are still modelled for eval ground truth.
        Returns all signals (dropped + routed).
        """
        self._ensure_sources_loaded()

        derived = self._derive_customer_email_signals(state)
        signals: List[ExternalEmailSignal] = []

        for item in derived:
            signal = self._generate_email(
                source=item["source"],
                topic=item["topic"],
                state=state,
                hour_range=(9, 16),
                category="customer",
                email_type=item["email_type"],
                symptom=item.get("symptom", ""),
                trigger_context=item.get("trigger", ""),
            )
            if not signal:
                continue

            if random.random() < _PROB_EMAIL_DROPPED:
                signal.dropped = True
                self._log_dropped_email(signal, state)
                logger.info(
                    f"    [dim yellow]📭 Dropped (no action): "
                    f'{signal.source_name} -> "{signal.subject[:50]}"[/dim yellow]'
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
        elif derived:
            logger.info(
                "  [dim]📭 No customer emails fired today (all signals suppressed)[/dim]"
            )
        else:
            logger.info(
                "  [dim]📭 No customer email signals derived from state today[/dim]"
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

    def _sales_pings_product(
        self, signal, sales_lead, product_lead, state, date_str
    ) -> Optional[str]:
        participants = [sales_lead, product_lead]
        ping_time, _ = self._clock.sync_and_advance(participants, hours=0.25)

        backstory = persona_utils.get_voice_card(
            sales_lead, "async", self._gd, mem=self._mem
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

        backstory = persona_utils.get_voice_card(
            product_lead, "async", self._gd, mem=self._mem
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

        linked_incident_ticket_id = None
        for inc in state.active_incidents:
            if any(
                kw in signal.topic.lower()
                for kw in inc.root_cause.lower().split()
                if len(kw) > 4
            ):
                inc.causal_chain.append(signal.embed_id)
                linked_incident_ticket_id = inc.ticket_id
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

        facts = {
            "vendor": signal.source_name,
            "topic": signal.topic,
            "routed_to": recipient,
            "causal_chain": signal.causal_chain.snapshot(),
        }

        if linked_incident_ticket_id:
            facts["linked_incident"] = linked_incident_ticket_id

        self._mem.log_event(
            SimEvent(
                type="vendor_email_routed",
                timestamp=signal.timestamp_iso,
                day=state.day,
                date=date_str,
                actors=[signal.source_name, recipient],
                artifact_ids={"email": signal.embed_id},
                facts=facts,
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

        backstory = persona_utils.get_voice_card(
            hr_lead, "async", self._gd, mem=self._mem
        )
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
            embed_id=embed_id,
            day=date_str,
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

        backstory = persona_utils.get_voice_card(
            sales_lead, "async", self._gd, mem=self._mem
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
            embed_id=embed_id,
            day=state.day,
        )

        if not is_high:
            self._crm.process_outbound_email(
                email_data={
                    "sender": sales_lead,
                    "recipient": signal.source_name,
                    "sender_org": self._company_name,
                    "recipient_org": signal.source_org,
                    "subject": subject,
                    "stage": crm_stage,
                    "embed_id": embed_id,
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

        backstory = persona_utils.get_voice_card(
            recipient, "async", self._gd, mem=self._mem
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
            embed_id=embed_id,
            day=state.day,
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

    def _generate_email(
        self,
        source: dict,
        topic: str,
        state,
        hour_range: Tuple[int, int],
        category: str,
        email_type: str = "general_inquiry",
        symptom: str = "",
        trigger_context: str = "",
    ) -> Optional[Any]:
        """
        Generates a single inbound email from an external contact.

        For customers: outputs JSON so email_type is declared at generation time
        (not classified post-hoc), no tech_ctx is injected, and the prompt uses
        first-person sender framing grounded in the derived signal context.

        For vendors: keeps plain SUBJECT/---/body format with tech_ctx so engineers
        can reference infrastructure specifics.
        """
        source_first_name = source["first_name"]
        source_name = source_first_name
        source_last_name = source["last_name"]
        source_org = source.get("org", source_name)
        source_addr = source.get("email", f"contact@{source_name.lower()}.com")
        liaison_dept = source.get("internal_liaison", list(self._leads.keys())[0])
        liaison_name = self._leads.get(liaison_dept, next(iter(self._leads.values())))
        tone = source.get("tone", "professional")
        date_str = str(state.current_date.date())

        email_ts = state.current_date.replace(
            hour=random.randint(*hour_range),
            minute=random.randint(0, 59),
            second=random.randint(0, 59),
        )

        backstory = persona_utils.get_voice_card(
            source_first_name, "async", self._gd, mem=self._mem, internal=False
        )

        agent = make_agent(
            role=f"{source_first_name} {source_last_name}, {source.get('contact_role', 'representative')} at {source_org}",
            goal=f"Write a realistic inbound email to {self._company_name}.",
            backstory=backstory,
            llm=self._worker_llm,
        )

        if category == "customer":
            symptom_hint = f"\nSITUATION: {symptom}" if symptom else ""
            email_type_hint = {
                "complaint": "You are writing to report a problem you are experiencing. Describe the business impact on your organisation. Do NOT name or guess at internal systems.",
                "question": "You are following up on a business matter or asking for clarification. Be specific to your situation.",
                "feature_request": "You are requesting a capability or improvement that would benefit your team.",
                "positive_feedback": "You are writing to share positive feedback or a success story.",
                "general_inquiry": "You have a general question or comment.",
            }.get(
                email_type, "Write a professional email relevant to your relationship."
            )

            task = Task(
                description=(
                    f"You are {source_first_name} {source_last_name}, {source.get('contact_role', 'a representative')} at {source_org}.\n"
                    f"You are writing an email to {liaison_name} at {self._company_name}.\n"
                    f"Tone: {tone}.{symptom_hint}\n\n"
                    f"INTENT: {email_type_hint}\n\n"
                    f"IMPORTANT: Write entirely from your perspective as a customer. "
                    f"Describe only what you observe or experience — never reference {self._company_name}'s internal systems, "
                    f"infrastructure, or technology by name. You don't know what's running under the hood.\n\n"
                    f"Respond ONLY with a JSON object. No preamble, no markdown fences:\n"
                    f"{{\n"
                    f'  "subject": "<email subject>",\n'
                    f'  "body": "<email body, 3-6 sentences, under 120 words>",\n'
                    f'  "email_type": "<exactly one of: complaint, question, feature_request, positive_feedback, general_inquiry>"\n'
                    f"}}"
                ),
                expected_output='JSON with "subject", "body", and "email_type" keys.',
                agent=agent,
            )

            try:
                raw = str(
                    Crew(agents=[agent], tasks=[task], verbose=False).kickoff()
                ).strip()
                parsed = json_repair.loads(raw)
                if isinstance(parsed, list) and parsed:
                    parsed = parsed[0]
                if not isinstance(parsed, dict):
                    raise ValueError("LLM did not return a dict")
                subject = parsed.get("subject", f"Re: {topic}").strip()
                body = parsed.get("body", "").strip()
                resolved_email_type = (
                    parsed.get("email_type", email_type).strip().lower()
                )
                if resolved_email_type not in _VALID_EMAIL_TYPES:
                    resolved_email_type = email_type
                if not body:
                    raise ValueError("Empty body")
            except Exception as exc:
                logger.warning(
                    f"[external_email] Customer email LLM failed for {source_name}: {exc}"
                )
                return None

        else:
            tech_stack = self._mem.tech_stack_for_prompt()
            tech_ctx = (
                (
                    f"\nCOMPANY TECH STACK (for your reference):\n{tech_stack}\n"
                    f"Restrict references to the company's infrastructure to this stack only. "
                    f"You may reference outside technologies only if alerting about an integration issue, "
                    f"suggesting a migration, or offering a new service."
                )
                if tech_stack
                else ""
            )
            resolved_email_type = "general_inquiry"

            task = Task(
                description=(
                    f"You are {source_first_name} {source_last_name} from {source_org}.\n"
                    f"Write an email to {liaison_name} at {self._company_name} about: {topic}.\n"
                    f"Tone: {tone}. Their system health: {state.system_health}/100."
                    f"{tech_ctx}\n\n"
                    f"Write as yourself — do not describe the email, write it.\n"
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
                logger.warning(
                    f"[external_email] Vendor email LLM failed for {source_name}: {exc}"
                )
                return None

            subject, body = self._parse_email_output(raw, topic)

        embed_id = (
            f"ext_email_{source_name.lower().replace(' ', '_')}"
            f"_{state.day}_{hour_range[0]}"
        )

        eml_path = self._write_eml(
            date_str=date_str,
            from_name=f"{source_first_name} {source_last_name}",
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
            content=f"From: {source_first_name} {source_last_name} ({source_org})\n\n{body}",
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
                "email_type": resolved_email_type,
            },
        )

        facts = {
            "source": source_name,
            "org": source_org,
            "category": category,
            "topic": topic,
            "subject": subject,
            "liaison": liaison_name,
            "liaison_dept": liaison_dept,
            "tone": tone,
            "email_type": resolved_email_type,
            "body_preview": body[:200],
        }

        chain = CausalChainHandler(root_id=embed_id)
        zd_ticket_id = None

        if category == "customer" and resolved_email_type in _ZD_TICKET_TYPES:
            zd_ticket_id = self._crm.handle_inbound_customer_email(
                event_facts={
                    "subject": subject,
                    "body": body[:500],
                    "sender_org": source_org,
                    "sender": source_addr,
                    "sender_name": f"{source_first_name} {source_last_name}",
                    "email": embed_id,
                    "liaison_email": self._email_of(liaison_name),
                },
                email_type=resolved_email_type,
                timestamp=email_ts.isoformat(),
                date_str=date_str,
                day=state.day,
            )
            if zd_ticket_id:
                logger.info(
                    f"    [dim]🔗 ZD ticket {zd_ticket_id} [{resolved_email_type}] "
                    f"linked to email from {source_name}[/dim]"
                )

        facts["causal_chain"] = chain.snapshot()
        if zd_ticket_id:
            chain.append(zd_ticket_id)

        self._mem.log_event(
            SimEvent(
                type="inbound_external_email",
                timestamp=email_ts.isoformat(),
                day=state.day,
                date=date_str,
                actors=[source_name, liaison_name],
                artifact_ids={"email": embed_id, "eml_path": str(eml_path)},
                facts=facts,
                summary=f'Inbound [{category}/{resolved_email_type}] email from {source_name}: "{subject}"',
                tags=[
                    "email",
                    "inbound",
                    category,
                    resolved_email_type,
                    source_name.lower(),
                ],
            )
        )

        artifact_ids: Dict[str, Any] = {"email": embed_id, "eml_path": str(eml_path)}
        if zd_ticket_id:
            artifact_ids["zd_ticket"] = zd_ticket_id

        body_preview = body[:200].rstrip() + ("…" if len(body) > 200 else "")

        signal = ExternalEmailSignal(
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
            facts={
                "subject": subject,
                "topic": topic,
                "org": source_org,
                "email_type": resolved_email_type,
            },
        )

        signal.facts["email_type"] = resolved_email_type
        if zd_ticket_id:
            signal.facts["zd_ticket_id"] = zd_ticket_id

        return signal

    def _generate_customer_reply_email(
        self,
        contact_name: str,
        account_name: str,
        owner: str,
        prior_subject: str,
        prior_body: str,
        current_stage: str,
        opp_id: str,
        state,
    ) -> Optional[Tuple[str, str, str]]:
        """
        Generates a customer reply using an LLM.

        The customer is given the full body of the prior outbound email and their
        deal stage context, then asked to reply as themselves. The prompt
        explicitly instructs the LLM not to force a stage advance — only output
        one if the reply honestly warrants it.

        Returns (body, email_type, crm_stage) or None on failure.
        """
        prior_email_ctx = (
            f"The email you are replying to:\n"
            f"Subject: {prior_subject}\n"
            f"---\n{prior_body[:800]}\n\n"
            if prior_body
            else (
                f"You are replying to a recent email from {owner} "
                f'with subject: "{prior_subject}".\n\n'
            )
        )

        stage_guidance = {
            "Prospecting": (
                "You are early in conversations — curious but not yet committed. "
                "You might ask questions, request more information, or express cautious interest."
            ),
            "Value Proposition": (
                "You've seen some value but haven't committed. You might push back on pricing, "
                "ask about integrations, or request a demo or case study."
            ),
            "Proposal/Price Quote": (
                "You have a proposal in hand. You might negotiate terms, ask for clarification "
                "on scope, or flag internal approval steps you need to complete."
            ),
            "Negotiation/Review": (
                "You're close to a decision. You might raise a final objection, request a small "
                "concession, confirm timeline, or — if everything looks right — signal readiness to proceed."
            ),
        }.get(current_stage, "Respond naturally based on the conversation so far.")

        agent = make_agent(
            role=f"{contact_name} — {account_name}",
            goal=f"Reply to an email from {owner} at {self._company_name}.",
            backstory=(
                f"You are {contact_name}, a decision-maker at {account_name}. "
                f"You are in active conversations with {self._company_name}. "
                f"You communicate professionally but directly. "
                f"Never acknowledge being an AI."
            ),
            llm=self._worker_llm,
        )

        task = Task(
            description=(
                f"{prior_email_ctx}"
                f"Deal context: You are currently at the '{current_stage}' stage "
                f"with {self._company_name}.\n"
                f"{stage_guidance}\n\n"
                f"Write a reply from {contact_name} to {owner}.\n\n"
                f"Rules:\n"
                f"- Reply specifically to what was said above. Reference it directly.\n"
                f"- Do NOT force positivity or urgency. Let your reply honestly reflect "
                f"  where you are. If you are uncertain or have concerns, say so.\n"
                f"- Under 120 words. Professional but human.\n\n"
                f"Then classify your reply and assess the deal stage it implies.\n\n"
                f"Respond ONLY with a JSON object. No preamble, no markdown fences.\n"
                f"{{\n"
                f'  "body": "<your reply email text>",\n'
                f'  "email_type": "<exactly one of: complaint, question, '
                f'feature_request, positive_feedback, general_inquiry>",\n'
                f'  "crm_stage": "<the SF stage this reply honestly implies. '
                f"Return the CURRENT stage ('{current_stage}') if the reply does not "
                f"move things forward — only advance the stage if the reply genuinely "
                f"signals it. Must be exactly one of: Prospecting, Value Proposition, "
                f'Proposal/Price Quote, Negotiation/Review, Closed Won, Closed Lost>"\n'
                f"}}"
            ),
            expected_output='JSON with "body", "email_type", and "crm_stage" keys.',
            agent=agent,
        )

        try:
            raw = str(
                Crew(agents=[agent], tasks=[task], verbose=False).kickoff()
            ).strip()
            parsed = json_repair.loads(raw)

            if isinstance(parsed, list) and parsed:
                parsed = parsed[0]
            if not isinstance(parsed, dict):
                raise ValueError("LLM did not return a dict")

            body = parsed.get("body", "").strip()
            email_type = parsed.get("email_type", "general_inquiry").strip().lower()
            crm_stage = parsed.get("crm_stage", current_stage).strip()

            if email_type not in _VALID_EMAIL_TYPES:
                email_type = "general_inquiry"
            if crm_stage not in _VALID_CRM_STAGES:
                crm_stage = current_stage
            if not body:
                raise ValueError("Empty body in LLM reply")

            return body, email_type, crm_stage

        except Exception as exc:
            logger.warning(
                f"[external_email] Customer reply generation failed for "
                f"{contact_name} ({account_name}): {exc}"
            )
            return None

    def generate_customer_replies(self, state) -> List[Any]:
        """
        For each open SF opportunity, probabilistically generate a customer reply
        to the most recent outbound email sent by the account owner.
        """

        if not self._crm or not hasattr(self._crm, "_sf_o"):
            return []

        signals: List[Any] = []
        date_str = str(state.current_date.date())

        open_opps = list(
            self._crm._sf_o.find(
                {"stage": {"$nin": ["Closed Won", "Closed Lost"]}},
                {"_id": 0, "_seq": 0},
            )
        )

        if not open_opps:
            return []

        for opp in open_opps:
            if random.random() > _PROB_CUSTOMER_REPLY:
                continue

            opp_id = opp.get("opportunity_id", "")
            account_name = opp.get("account_name", "Unknown")
            current_stage = opp.get("stage", "Prospecting")
            owner = opp.get("owner", "")
            touchpoints = opp.get("touchpoints", [])

            if not touchpoints:
                continue

            last_touchpoint = touchpoints[-1]
            last_subject = last_touchpoint.get("subject", "")
            last_embed_id = last_touchpoint.get("embed_id", "")

            prior_body = ""
            if last_embed_id:
                prior_doc = self._mem._db["emails"].find_one(
                    {"embed_id": last_embed_id}, {"body": 1, "_id": 0}
                )
                if prior_doc:
                    prior_body = prior_doc.get("body", "")

            acc = self._crm._sf_a.find_one(
                {"name": account_name}, {"_id": 0, "_seq": 0}
            )
            contact_name = (acc or {}).get("primary_contact", account_name)
            contact_email = (acc or {}).get(
                "primary_contact_email",
                (
                    f"{contact_name.lower().replace(' ', '.')}"
                    f"@{account_name.lower().replace(' ', '')}.com"
                ),
            )

            result = self._generate_customer_reply_email(
                contact_name=contact_name,
                account_name=account_name,
                owner=owner,
                prior_subject=last_subject,
                prior_body=prior_body,
                current_stage=current_stage,
                opp_id=opp_id,
                state=state,
            )
            if not result:
                continue

            reply_body, email_type, suggested_stage = result

            reply_ts = state.current_date.replace(
                hour=random.randint(9, 16),
                minute=random.randint(0, 59),
                second=random.randint(0, 59),
            )

            reply_subject = (
                f"Re: {last_subject}" if last_subject else f"Re: {account_name}"
            )
            embed_id = (
                f"customer_reply_{account_name.lower().replace(' ', '_')}"
                f"_{opp_id}_{state.day}"
            )

            sales_dept = next((d for d in self._leads if "sales" in d.lower()), None)
            sales_lead = self._leads.get(sales_dept, owner) if sales_dept else owner
            owner_addr = self._email_of(owner) if owner else self._email_of(sales_lead)

            eml_path = self._write_eml(
                date_str=date_str,
                from_name=contact_name,
                from_addr=contact_email,
                to_name=owner if owner else sales_lead,
                to_addr=owner_addr,
                subject=reply_subject,
                body=reply_body,
                timestamp_iso=reply_ts.isoformat(),
                direction="inbound",
                embed_id=embed_id,
                day=state.day,
            )

            self._mem.embed_artifact(
                id=embed_id,
                type="email",
                title=reply_subject,
                content=f"From: {contact_name} ({account_name})\n\n{reply_body}",
                day=state.day,
                date=date_str,
                timestamp=reply_ts.isoformat(),
                metadata={
                    "source": contact_name,
                    "org": account_name,
                    "category": "customer",
                    "direction": "inbound",
                    "opportunity_id": opp_id,
                    "reply_to_subject": last_subject,
                    "email_type": email_type,
                },
            )

            self._mem.log_event(
                SimEvent(
                    type="inbound_external_email",
                    timestamp=reply_ts.isoformat(),
                    day=state.day,
                    date=date_str,
                    actors=[contact_name, sales_lead],
                    artifact_ids={"email": embed_id, "eml_path": str(eml_path)},
                    facts={
                        "source": contact_name,
                        "org": account_name,
                        "category": "customer",
                        "topic": current_stage,
                        "subject": reply_subject,
                        "liaison": sales_lead,
                        "tone": "professional",
                        "body_preview": reply_body[:200],
                        "opportunity_id": opp_id,
                        "is_customer_reply": True,
                    },
                    summary=(
                        f"Inbound [customer_reply] from {contact_name} "
                        f'({account_name}): "{reply_subject}"'
                    ),
                    tags=["email", "inbound", "customer", "customer_reply"],
                )
            )

            body_preview = reply_body[:200].rstrip() + (
                "…" if len(reply_body) > 200 else ""
            )
            internal_liaison_dept = sales_dept or next(iter(self._leads.keys()))

            signal = ExternalEmailSignal(
                source_name=contact_name,
                source_org=account_name,
                source_email=contact_email,
                internal_liaison=internal_liaison_dept,
                subject=reply_subject,
                body_preview=body_preview,
                full_body=reply_body,
                tone="professional",
                topic=current_stage,
                timestamp_iso=reply_ts.isoformat(),
                embed_id=embed_id,
                category="customer",
                eml_path=str(eml_path),
                causal_chain=CausalChainHandler(root_id=embed_id),
                facts={
                    "subject": reply_subject,
                    "topic": current_stage,
                    "org": account_name,
                    "email_type": email_type,
                    "opportunity_id": opp_id,
                    "is_customer_reply": True,
                },
            )

            self._route_customer_email(signal, state)

            if (
                suggested_stage
                and suggested_stage in _VALID_CRM_STAGES
                and _STAGE_RANK.get(suggested_stage, 0)
                > _STAGE_RANK.get(current_stage, 0)
            ):
                self._crm._sf_o.update_one(
                    {"opportunity_id": opp_id},
                    {
                        "$set": {
                            "stage": suggested_stage,
                            "probability": _get_stage_probability(suggested_stage),
                            "updated_at": reply_ts.isoformat(),
                        }
                    },
                )
                updated_doc = self._crm._sf_o.find_one(
                    {"opportunity_id": opp_id}, {"_id": 0, "_seq": 0}
                )
                if updated_doc:
                    self._crm._write(
                        f"salesforce/opportunities/{opp_id}.json", updated_doc
                    )

                self._mem.log_event(
                    SimEvent(
                        type="sf_stage_advanced_by_customer",
                        timestamp=reply_ts.isoformat(),
                        day=state.day,
                        date=date_str,
                        actors=[contact_name, owner],
                        artifact_ids={"email": embed_id, "sf_opp": opp_id},
                        facts={
                            "opportunity_id": opp_id,
                            "account_name": account_name,
                            "previous_stage": current_stage,
                            "new_stage": suggested_stage,
                            "triggered_by": embed_id,
                        },
                        summary=(
                            f"SF opp {opp_id} ({account_name}) advanced "
                            f"{current_stage} → {suggested_stage} by customer reply"
                        ),
                        tags=["salesforce", "stage_advanced", "customer_reply"],
                    )
                )
                logger.info(
                    f"  [green]📈 {opp_id} ({account_name}): "
                    f"{current_stage} → {suggested_stage}[/green]"
                )

            signals.append(signal)
            logger.info(
                f"  [cyan]📬 Customer reply: {contact_name} ({account_name})[/cyan] "
                f"[{email_type}]"
            )

        if signals:
            logger.info(f"  [cyan]📬 {len(signals)} customer reply(s) generated[/cyan]")

        return signals

    def _route_customer_email(self, signal: Any, state) -> None:
        """
        Patched router. Dispatches based on email_type rather than assuming
        every inbound customer email is a complaint.
        """
        email_type = signal.facts.get("email_type", "general_inquiry")

        if email_type in _COMPLAINT_EMAIL_TYPES:
            self._route_complaint_email(signal, state)
        else:
            self._route_non_complaint_email(signal, state, email_type=email_type)

    def _route_non_complaint_email(
        self, signal: Any, state, email_type: str = "general_inquiry"
    ) -> None:
        """
        Routing for non-complaint customer emails.

        Questions and general_inquiries may produce a ZD ticket (probability-
        gated via _ZD_TICKET_PROB in crm_system — 70% and 30% respectively).
        The ticket is already created upstream in _generate_email before this
        method is called, so we just append it to the causal chain if present.

        Feature requests get a low-probability (~35%) FYI to #product.
        Positive feedback gets a direct reply only — no ticket, no escalation.

        Sales replies to all non-complaint emails directly.
        """
        date_str = str(state.current_date.date())
        sales_lead = self._leads.get(
            signal.internal_liaison, next(iter(self._leads.values()))
        )

        zd_ticket_id = signal.facts.get("zd_ticket_id")
        if zd_ticket_id:
            signal.causal_chain.append(zd_ticket_id)

        fyi_thread_id = None
        if (
            email_type == "feature_request"
            and random.random() < _PROB_NON_COMPLAINT_SALES_FYI
        ):
            fyi_thread_id = self._sales_fyi_to_product(
                signal, sales_lead, state, date_str
            )
            if fyi_thread_id:
                signal.causal_chain.append(fyi_thread_id)

        reply_id = self._send_customer_reply(
            signal, sales_lead, is_high=False, state=state, date_str=date_str
        )
        if reply_id:
            signal.causal_chain.append(reply_id)

        self._mem.log_event(
            SimEvent(
                type="customer_email_routed",
                timestamp=signal.timestamp_iso,
                day=state.day,
                date=date_str,
                actors=[signal.source_name, sales_lead],
                artifact_ids={"email": signal.embed_id},
                facts={
                    "source": signal.source_name,
                    "subject": signal.subject,
                    "email_type": email_type,
                    "high_priority": False,
                    "zd_ticket_id": zd_ticket_id,
                    "fyi_sent": fyi_thread_id is not None,
                    "causal_chain": signal.causal_chain.snapshot(),
                },
                summary=(
                    f"{email_type.replace('_', ' ').title()} from {signal.source_name} "
                    f"handled by {sales_lead}"
                    + (f" [ZD-{zd_ticket_id}]" if zd_ticket_id else "")
                    + (" [FYI sent to Product]" if fyi_thread_id else "")
                ),
                tags=["email", "customer", email_type, "routed", "causal_chain"],
            )
        )

    def _sales_fyi_to_product(
        self, signal: Any, sales_lead: str, state, date_str: str
    ) -> Optional[str]:
        """
        Posts a low-friction FYI in #product when a customer sends a feature
        request. This is deliberately lighter than _sales_pings_product():
        no explicit ask, no urgency label — just awareness.

        Returns the Slack thread_id, or None on failure.
        """
        product_dept = next((d for d in self._leads if "product" in d.lower()), None)
        product_lead = self._leads.get(product_dept, sales_lead)

        participants = [sales_lead, product_lead]
        fyi_time, _ = self._clock.sync_and_advance(participants, hours=0.1)

        backstory = persona_utils.get_voice_card(
            sales_lead, "async", graph_dynamics=None, mem=self._mem
        )
        p = self._personas.get(sales_lead, {})

        agent = make_agent(
            role=f"{sales_lead} — {p.get('social_role', 'Sales Lead')}",
            goal="Share a brief, low-priority customer feature request with Product.",
            backstory=backstory,
            llm=self._worker_llm,
        )
        task = Task(
            description=(
                f"You just read a feature request email from {signal.source_name}:\n"
                f"Subject: {signal.subject}\n{signal.full_body}\n\n"
                f"Write a short, casual Slack FYI to {product_lead} (Product). This is "
                f"NOT urgent — you're just sharing it for awareness. No action required.\n"
                f"Under 60 words. No bullets. Write as {sales_lead} using your typing quirks."
            ),
            expected_output="Casual Slack FYI under 60 words.",
            agent=agent,
        )
        try:
            text = str(
                Crew(agents=[agent], tasks=[task], verbose=False).kickoff()
            ).strip()
        except Exception as exc:
            logger.warning(f"[external_email] Sales FYI LLM failed: {exc}")
            return None

        _, thread_id = self._mem.log_slack_messages(
            channel="product",
            messages=[
                {
                    "user": sales_lead,
                    "email": self._email_of(sales_lead),
                    "text": text,
                    "ts": fyi_time.isoformat(),
                    "date": date_str,
                    "is_bot": False,
                    "metadata": {
                        "type": "customer_feature_request_fyi",
                        "source_email_id": signal.embed_id,
                        "customer": signal.source_name,
                    },
                }
            ],
            export_dir=self._export_dir,
        )

        self._mem.log_event(
            SimEvent(
                type="feature_request_fyi",
                timestamp=fyi_time.isoformat(),
                day=state.day,
                date=date_str,
                actors=[sales_lead, product_lead],
                artifact_ids={
                    "slack_thread": thread_id,
                    "email": signal.embed_id,
                },
                facts={
                    "customer": signal.source_name,
                    "subject": signal.subject,
                    "relayed_by": sales_lead,
                    "product_gatekeeper": product_lead,
                    "causal_chain": signal.causal_chain.snapshot(),
                },
                summary=(
                    f"{sales_lead} shared feature request FYI from "
                    f"{signal.source_name} in #product (no action required)"
                ),
                tags=["feature_request", "fyi", "slack", "causal_chain"],
            )
        )
        logger.info(
            f"    [dim]💬 FYI → #product: feature request from {signal.source_name}[/dim]"
        )
        return thread_id

    def _route_complaint_email(self, signal: Any, state) -> None:
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
                    "email_type": "complaint",
                    "high_priority": is_high,
                    "causal_chain": signal.causal_chain.snapshot(),
                },
                summary=(
                    f"Complaint from {signal.source_name} routed: "
                    f"{sales_lead} → {product_lead}"
                    + (" [JIRA opened]" if len(signal.causal_chain) > 2 else "")
                ),
                tags=["email", "customer", "complaint", "routed", "causal_chain"],
            )
        )

    def _get_stage_probability(self, stage: str) -> int:
        """Returns the default SF probability for a given stage."""
        return {
            "Prospecting": 10,
            "Value Proposition": 25,
            "Proposal/Price Quote": 50,
            "Negotiation/Review": 75,
            "Closed Won": 100,
            "Closed Lost": 0,
        }.get(stage, 10)

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

    def _incident_affects_customer(self, incident, source: dict) -> bool:
        return self._gd._incident_affects_customer(incident, source)

    def _derive_customer_email_signals(self, state) -> List[dict]:
        """
        Inspects simulation state and CRM data to derive a list of grounded
        customer email signals. Each signal represents a real reason a customer
        would reach out — not a random probability fire.

        Returns a list of dicts, each with:
            source        — the full source record from inbound_email_sources
            email_type    — complaint | question | feature_request | positive_feedback | general_inquiry
            trigger       — human-readable reason string for LLM context
            symptom       — customer-facing symptom description (no internal tech names)
            topic         — topic string passed to _generate_email

        Signal priority (highest to lowest):
          1. Active incident that affects this customer → complaint
          2. Open opp at Negotiation/Review stale > 3 days → question (customer follows up)
          3. Contract renewal within 60 days → question (renewal conversation)
          4. Opp has risk_notes → question or complaint depending on sentiment
          5. High expansion_potential (>= 8) + healthy system → feature_request
        """
        self._ensure_sources_loaded()
        signals: List[dict] = []
        date_str = str(state.current_date.date())

        customer_sources = [
            s for s in (self._sources or []) if s.get("category") == "customer"
        ]

        for source in customer_sources:
            org_name = source.get("org", "")
            sentiment = source.get("sentiment_baseline", 0.8)
            tone = source.get("tone", "formal")

            for incident in state.active_incidents:
                if org_name in getattr(incident, "contacted_customers", []):
                    continue
                if not self._gd._incident_affects_customer(incident, source):
                    continue

                symptom = source.get(
                    "symptom_language",
                    "We are experiencing issues accessing your platform and wanted to follow up.",
                )
                signals.append(
                    {
                        "source": source,
                        "email_type": "complaint",
                        "trigger": f"Active incident {incident.ticket_id} affecting platform capabilities this customer depends on",
                        "symptom": symptom,
                        "topic": symptom,
                        "incident_id": incident.ticket_id,
                    }
                )
                break

            else:
                opp = None
                if hasattr(self._crm, "_sf_o"):
                    opp = self._crm._sf_o.find_one(
                        {
                            "account_name": org_name,
                            "stage": "Negotiation/Review",
                        },
                        {"_id": 0, "_seq": 0},
                    )

                if opp:
                    touchpoints = opp.get("touchpoints", [])
                    last_touch = (
                        touchpoints[-1].get("timestamp", "") if touchpoints else ""
                    )
                    days_stale = 0
                    if last_touch:
                        try:
                            last_dt = datetime.fromisoformat(
                                last_touch.replace("Z", "+00:00")
                            ).replace(tzinfo=None)
                            days_stale = (
                                state.current_date.replace(tzinfo=None) - last_dt
                            ).days
                        except ValueError:
                            pass

                    if days_stale >= 3:
                        topic = (
                            "Following up on our proposal — checking in on next steps"
                        )
                        signals.append(
                            {
                                "source": source,
                                "email_type": "question",
                                "trigger": f"Open deal {opp['opportunity_id']} at Negotiation/Review, no touchpoint in {days_stale} days",
                                "symptom": "",
                                "topic": topic,
                            }
                        )
                        continue

                renewal_str = source.get("contract_renewal_date", "")
                if renewal_str:
                    try:
                        renewal_dt = datetime.fromisoformat(
                            renewal_str.replace("Z", "+00:00")
                        ).replace(tzinfo=None)
                        days_to_renewal = (
                            renewal_dt - state.current_date.replace(tzinfo=None)
                        ).days
                        if 0 < days_to_renewal <= 60:
                            topic = "Upcoming contract renewal — wanted to discuss terms and our roadmap needs"
                            signals.append(
                                {
                                    "source": source,
                                    "email_type": "question",
                                    "trigger": f"Contract renewal in {days_to_renewal} days",
                                    "symptom": "",
                                    "topic": topic,
                                }
                            )
                            continue
                    except ValueError:
                        pass

                if hasattr(self._crm, "_sf_o"):
                    risky_opp = self._crm._sf_o.find_one(
                        {
                            "account_name": org_name,
                            "stage": {"$nin": ["Closed Won", "Closed Lost"]},
                            "risk_notes": {"$not": {"$size": 0}},
                        },
                        {"_id": 0, "_seq": 0},
                    )
                    if risky_opp and sentiment < 0.6:
                        topic = "Wanted to discuss some concerns we have about platform reliability"
                        signals.append(
                            {
                                "source": source,
                                "email_type": "complaint"
                                if sentiment < 0.45
                                else "question",
                                "trigger": f"Risky deal {risky_opp['opportunity_id']} + low sentiment ({sentiment})",
                                "symptom": "",
                                "topic": topic,
                            }
                        )
                        continue

                if (
                    source.get("expansion_potential", 0) >= 8
                    and state.system_health >= 80
                    and random.random() < 0.25
                ):
                    topic = "Exploring additional use cases and features for our team"
                    signals.append(
                        {
                            "source": source,
                            "email_type": "feature_request",
                            "trigger": f"High expansion potential ({source.get('expansion_potential')}) + healthy system",
                            "symptom": "",
                            "topic": topic,
                        }
                    )

                elif sentiment < 0.45 and random.random() < 0.15:
                    topic = source.get("topics", ["platform reliability concerns"])[0]
                    signals.append(
                        {
                            "source": source,
                            "email_type": "complaint",
                            "trigger": f"Chronically low sentiment ({sentiment:.2f}) — unprompted complaint",
                            "symptom": source.get(
                                "symptom_language",
                                "We've been experiencing ongoing issues and wanted to follow up.",
                            ),
                            "topic": topic,
                        }
                    )

        return signals

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
        date_str: str,
        from_name: str,
        from_addr: str,
        to_name: str,
        to_addr: str,
        subject: str,
        body: str,
        timestamp_iso: str,
        direction: str = "inbound",
        embed_id: str = "",
        day: int = 0,
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

        doc_id = embed_id or f"{from_name.lower().replace(' ', '_')}_{timestamp_iso}"
        try:
            self._mem._db["emails"].update_one(
                {"embed_id": doc_id},
                {
                    "$setOnInsert": {
                        "embed_id": doc_id,
                        "direction": direction,
                        "from_name": from_name,
                        "from_addr": from_addr,
                        "to_name": to_name,
                        "to_addr": to_addr,
                        "subject": subject,
                        "body": body,
                        "timestamp": timestamp_iso,
                        "day": day,
                        "date": date_str,
                        "eml_path": str(path),
                    }
                },
                upsert=True,
            )
        except Exception as exc:
            logger.warning(f"[external_email] emails collection insert failed: {exc}")

        return path
