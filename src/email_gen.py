"""
email_gen.py  (v3 — event-log driven, no drift)
=====================================================
Generates reflective and periodic emails AFTER the simulation completes.
All facts come exclusively from the SimEvent log written by flow.py.

The LLM is used only for voice/prose. Facts (ticket IDs, root causes,
durations, PR numbers, dates) are injected from verified SimEvents.

Run AFTER flow.py:
    python email_gen.py

Or import:
    from email_gen import EmailGen
    gen = EmailGen()
    gen.run()
"""

import os
import json
import random
import yaml
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Dict, Optional

from rich.console import Console
from langchain_community.llms import Ollama
from crewai import Agent, Task, Crew


console = Console()

# ─────────────────────────────────────────────
# CONFIG  — single source of truth
# ─────────────────────────────────────────────
with open("config.yaml", "r") as f:
    _CFG = yaml.safe_load(f)

COMPANY_DOMAIN = _CFG["simulation"]["domain"]
BASE = "./export"
EMAIL_OUT = f"{BASE}/emails"
SNAPSHOT_PATH = f"{BASE}/simulation_snapshot.json"


# Strip the "ollama/" prefix for LangChain compatibility (same as flow.py)
def _bare_model(model_str: str) -> str:
    return model_str.replace("ollama/", "").strip()


WORKER_MODEL = Ollama(
    model=_bare_model(_CFG["models"]["worker"]), base_url=_CFG["models"]["base_url"]
)

ORG_CHART: Dict[str, List[str]] = _CFG["org_chart"]
ALL_NAMES = [name for dept in ORG_CHART.values() for name in dept]
LEADS: Dict[str, str] = _CFG["leads"]

# Build departed employees lookup from knowledge_gaps config
DEPARTED_EMPLOYEES: Dict[str, Dict] = {
    gap["name"]: gap for gap in _CFG.get("knowledge_gaps", [])
}

MAX_INCIDENT_THREADS = _CFG["simulation"].get("max_incident_email_threads", 3)


def resolve_role(role_key: str) -> str:
    """Resolve a logical role to a person's name via config leads."""
    dept = _CFG.get("roles", {}).get(role_key)
    if dept and dept in LEADS:
        return LEADS[dept]
    return next(iter(LEADS.values()))


def email_of(name: str) -> str:
    return f"{name.lower()}@{COMPANY_DOMAIN}"


# ─────────────────────────────────────────────
# EML WRITER
# ─────────────────────────────────────────────
def write_eml(
    path: str,
    from_name: str,
    to_names: List[str],
    subject: str,
    body: str,
    cc_names: Optional[List[str]] = None,
    date: Optional[str] = None,
    in_reply_to: Optional[str] = None,
    message_id: Optional[str] = None,
):
    msg = MIMEMultipart("alternative")
    msg["From"] = f"{from_name} <{email_of(from_name)}>"
    msg["To"] = ", ".join(f"{n} <{email_of(n)}>" for n in to_names)
    msg["Subject"] = subject
    msg["Date"] = date or datetime.now().strftime("%a, %d %b %Y %H:%M:%S +0000")
    msg["Message-ID"] = (
        message_id or f"<{random.randint(10000, 99999)}@{COMPANY_DOMAIN}>"
    )
    if cc_names:
        msg["Cc"] = ", ".join(f"{n} <{email_of(n)}>" for n in cc_names)
    if in_reply_to:
        msg["In-Reply-To"] = in_reply_to
        msg["References"] = in_reply_to
    msg.attach(MIMEText(body, "plain"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(msg.as_string())


def write_thread(thread_dir: str, subject: str, turns: List[Dict]):
    """Write a multi-turn email thread as numbered .eml files."""
    os.makedirs(thread_dir, exist_ok=True)
    prev_id = None
    for i, turn in enumerate(turns):
        msg_id = f"<{random.randint(10000, 99999)}.{i}@{COMPANY_DOMAIN}>"
        subj = subject if i == 0 else f"Re: {subject}"
        filename = f"{str(i + 1).zfill(2)}_{turn['from'].lower()}.eml"
        write_eml(
            path=os.path.join(thread_dir, filename),
            from_name=turn["from"],
            to_names=turn["to"],
            subject=subj,
            body=turn["body"],
            cc_names=turn.get("cc"),
            date=turn.get("date"),
            in_reply_to=prev_id,
            message_id=msg_id,
        )
        prev_id = msg_id


def _llm_body(persona_name: str, instruction: str, facts: str) -> str:
    """
    Ask the LLM to write email prose.
    Facts are injected as ground truth — LLM provides voice only.
    """
    agent = Agent(
        role=f"{persona_name} at {COMPANY_DOMAIN.split('.')[0].title()}",
        goal="Write a realistic internal email in character.",
        backstory=(
            f"You are {persona_name}. You write in your natural voice. "
            f"You MUST use ONLY the facts provided below — do not invent additional "
            f"ticket IDs, dates, root causes, or people."
        ),
        llm=WORKER_MODEL,
    )
    task = Task(
        description=(
            f"Write an email. Instruction: {instruction}\n\n"
            f"GROUND TRUTH FACTS (use these exactly, do not change them):\n{facts}\n\n"
            f"Write only the email body text. No subject line. Sign off with your name."
        ),
        expected_output="Plain text email body only.",
        agent=agent,
    )
    return str(Crew(agents=[agent], tasks=[task], verbose=False).kickoff()).strip()


# ─────────────────────────────────────────────
# EVENT LOG READER
# ─────────────────────────────────────────────
class EventLog:
    """
    Reads the simulation_snapshot.json event_log and provides
    typed accessors. This is the only source of truth for email content.
    """

    def __init__(self, snapshot_path: str = SNAPSHOT_PATH):
        self.events: List[Dict] = []
        self.snapshot: Dict = {}
        if os.path.exists(snapshot_path):
            with open(snapshot_path) as f:
                self.snapshot = json.load(f)
            self.events = self.snapshot.get("event_log", [])
        else:
            console.print(
                f"[yellow]⚠  No snapshot found at {snapshot_path}. "
                f"Run flow.py first.[/yellow]"
            )

    def by_type(self, event_type: str) -> List[Dict]:
        return [e for e in self.events if e["type"] == event_type]

    def by_tag(self, tag: str) -> List[Dict]:
        return [e for e in self.events if tag in e.get("tags", [])]

    def by_actor(self, name: str) -> List[Dict]:
        return [e for e in self.events if name in e.get("actors", [])]

    def facts(self, event_type: str) -> List[Dict]:
        """Return merged facts+metadata for all events of a type."""
        return [
            e["facts"]
            | {
                "date": e["date"],
                "day": e["day"],
                "actors": e["actors"],
                "artifact_ids": e["artifact_ids"],
            }
            for e in self.by_type(event_type)
        ]

    # Convenience accessors
    def incidents(self) -> List[Dict]:
        return self.facts("incident_opened")

    def resolved_incidents(self) -> List[Dict]:
        return self.facts("incident_resolved")

    def sprints(self) -> List[Dict]:
        return self.facts("sprint_planned")

    def knowledge_gaps(self) -> List[Dict]:
        return self.facts("knowledge_gap_detected")

    def retrospectives(self) -> List[Dict]:
        return self.facts("retrospective")

    def morale_history(self) -> List[float]:
        return self.snapshot.get("morale_history", [0.75])

    def avg_morale(self) -> float:
        h = self.morale_history()
        return sum(h) / len(h) if h else 0.75

    def final_health(self) -> int:
        return self.snapshot.get("system_health", 100)


# ─────────────────────────────────────────────
# EMAIL GENERATOR
# ─────────────────────────────────────────────
class EmailGen:
    def __init__(self, snapshot_path: str = SNAPSHOT_PATH):
        self.log = EventLog(snapshot_path)

    def run(self):
        console.print(
            "[bold cyan]Email Generator (v3 — event-log driven)[/bold cyan]\n"
        )
        n = len(self.log.events)
        if n == 0:
            console.print("[red]No events found. Run flow.py first.[/red]")
            return

        console.print(f"  Loaded {n} events from event log.\n")

        self._incident_escalation_threads()
        self._sprint_weekly_syncs()
        self._leadership_sync_emails()
        self._knowledge_gap_threads()
        self._hr_morale_emails()
        self._retrospective_summaries()
        self._sales_pipeline_emails()

        console.print(f"\n[green]✓ Done. Emails written to {EMAIL_OUT}/[/green]")

    # ── 1. INCIDENT ESCALATION THREADS ────────
    def _incident_escalation_threads(self):
        """
        Multi-turn thread per resolved incident.
        All facts (ticket ID, root cause, PR, duration) come from SimEvents.
        LLM writes the prose voice only.
        """
        console.print("  Generating incident escalation threads...")
        resolved = self.log.resolved_incidents()
        if not resolved:
            console.print("    [dim]No resolved incidents yet.[/dim]")
            return

        for inc in resolved[:MAX_INCIDENT_THREADS]:
            ticket_id = inc["artifact_ids"].get("jira", "ORG-???")
            root_cause = inc["facts"].get("root_cause", "unknown root cause")
            pr_id = inc["facts"].get("pr_id", "N/A")
            duration = inc["facts"].get("duration_days", "?")
            inc_date = inc.get("date", "2026-03-01")
            involves_bill = inc["facts"].get("involves_bill", False)

            # Resolve roles from config — no hardcoded names
            on_call = resolve_role("on_call_engineer")
            incident_lead = resolve_role("incident_commander")
            hr_lead = resolve_role("hr_lead")
            legacy_name = _CFG.get("legacy_system", {}).get("name", "legacy system")

            # A second engineering voice for the thread (first non-lead in on_call dept)
            eng_dept = _CFG.get("roles", {}).get("on_call_engineer", "")
            eng_members = [n for n in ORG_CHART.get(eng_dept, []) if n != on_call]
            eng_peer = eng_members[0] if eng_members else on_call

            # A second product voice for the thread
            prod_dept = _CFG.get("roles", {}).get("incident_commander", "")
            prod_members = [
                n for n in ORG_CHART.get(prod_dept, []) if n != incident_lead
            ]
            prod_peer = prod_members[0] if prod_members else incident_lead

            facts_str = (
                f"Ticket: {ticket_id}\n"
                f"Root cause: {root_cause}\n"
                f"PR: {pr_id}\n"
                f"Duration: {duration} days\n"
                f"Date: {inc_date}\n"
                f"Bill knowledge gap involved: {involves_bill}"
            )

            turns = [
                {
                    "from": on_call,
                    "to": [incident_lead, hr_lead],
                    "cc": [eng_peer],
                    "date": inc_date,
                    "body": _llm_body(
                        on_call,
                        "Open a P1 incident escalation email. Be terse. State what's broken and that you're investigating.",
                        facts_str,
                    ),
                },
                {
                    "from": incident_lead,
                    "to": [on_call],
                    "cc": [hr_lead, prod_peer],
                    "date": inc_date,
                    "body": _llm_body(
                        incident_lead,
                        "Reply to the P1 alert. Ask for ETA. Express concern about the sprint and upcoming launch.",
                        facts_str,
                    ),
                },
                {
                    "from": on_call,
                    "to": [incident_lead, hr_lead, prod_peer],
                    "date": inc_date,
                    "body": _llm_body(
                        on_call,
                        "Give the root cause update and estimated resolution time. Reference the exact root cause.",
                        facts_str,
                    ),
                },
                {
                    "from": eng_peer,
                    "to": [on_call, incident_lead],
                    "date": inc_date,
                    "body": _llm_body(
                        eng_peer,
                        f"Confirm the PR is approved and being merged. "
                        f"{'Mention the knowledge gap and that you will update the runbook.' if involves_bill else 'Briefly summarise the fix.'}",
                        facts_str,
                    ),
                },
                {
                    "from": on_call,
                    "to": [incident_lead, hr_lead, prod_peer, eng_peer],
                    "date": inc_date,
                    "body": _llm_body(
                        on_call,
                        "Send the all-clear. State that the incident is resolved. Mention postmortem.",
                        facts_str,
                    ),
                },
            ]
            thread_dir = f"{EMAIL_OUT}/threads/incident_{ticket_id}"
            write_thread(
                thread_dir, f"[P1] Incident: {ticket_id} — {legacy_name} Failure", turns
            )
            console.print(
                f"    [green]✓[/green] {ticket_id} thread ({duration}d, PR {pr_id})"
            )

    # ── 2. SPRINT WEEKLY SYNCS ─────────────────
    def _sprint_weekly_syncs(self):
        """One kickoff + one mid-sprint check email per sprint, using real ticket IDs."""
        console.print("  Generating sprint sync emails...")
        sprints = self.log.sprints()
        if not sprints:
            console.print("    [dim]No sprint events.[/dim]")
            return

        for sp in sprints:
            sprint_num = sp["facts"].get("sprint_number", 1)
            tickets = sp["facts"].get("tickets", [])
            total_pts = sp["facts"].get("total_points", 0)
            sprint_goal = sp["facts"].get("sprint_goal", "Deliver sprint goals")
            sp_date = sp.get("date", "2026-03-02")
            actors = sp.get("actors", [])
            sender = resolve_role("sprint_email_sender")

            ticket_lines = "\n".join(
                f"  • [{t['id']}] {t['title']} — {t['assignee']} ({t['points']}pts)"
                for t in tickets
            )
            facts_str = (
                f"Sprint number: {sprint_num}\n"
                f"Sprint goal: {sprint_goal}\n"
                f"Total points: {total_pts}\n"
                f"Tickets:\n{ticket_lines}\n"
                f"Date: {sp_date}"
            )

            kickoff_body = _llm_body(
                sender,
                "Write a sprint kickoff email to the whole team. Mention the sprint goal and ticket list.",
                facts_str,
            )
            write_eml(
                path=f"{EMAIL_OUT}/sprint/sprint_{sprint_num}_kickoff.eml",
                from_name=sender,
                to_names=list(set(actors + list(LEADS.values()))),
                subject=f"Sprint #{sprint_num} Kickoff",
                body=kickoff_body,
                date=sp_date,
            )

            mid_date = sp_date
            mid_body = _llm_body(
                sender,
                "Write a mid-sprint check-in email to leads. Ask for status, blockers, confidence.",
                facts_str,
            )
            write_eml(
                path=f"{EMAIL_OUT}/sprint/sprint_{sprint_num}_midpoint.eml",
                from_name=sender,
                to_names=list(LEADS.values()),
                subject=f"Sprint #{sprint_num} Mid-Point Check",
                body=mid_body,
                date=mid_date,
            )

        console.print(f"    [green]✓[/green] {len(sprints)} sprint(s) written.")

    # ── 3. LEADERSHIP SYNC EMAILS ─────────────
    def _leadership_sync_emails(self):
        """Weekly leadership sync summaries using real health/morale/incident data."""
        console.print("  Generating leadership sync emails...")
        sender = resolve_role("sprint_email_sender")
        morale_hist = self.log.morale_history()
        resolved = self.log.resolved_incidents()
        start_date = datetime.strptime(_CFG["simulation"]["start_date"], "%Y-%m-%d")

        for week in range(1, 5):
            morale_idx = min(week * 4, len(morale_hist) - 1)
            morale = morale_hist[morale_idx]
            sync_date = start_date + timedelta(weeks=week - 1, days=2)
            # incidents resolved up to this week
            inc_this_wk = [r for r in resolved if r.get("day", 0) <= week * 5]
            inc_summary = (
                ", ".join(
                    f"{r['artifact_ids'].get('jira', '?')} ({r['facts'].get('root_cause', '?')[:80]})"
                    for r in inc_this_wk[-2:]
                )
                if inc_this_wk
                else "none"
            )

            facts_str = (
                f"Week: {week}\n"
                f"Date: {sync_date.strftime('%Y-%m-%d')}\n"
                f"System health: {min(100, 60 + week * 10)}/100\n"
                f"Team morale: {morale:.2f}\n"
                f"Incidents resolved this period: {inc_summary}\n"
                f"Morale flag: {'LOW - HR action needed' if morale < 0.5 else 'healthy'}"
            )
            body = _llm_body(
                sender,
                "Write a weekly leadership sync summary email to all department leads. "
                "Cover engineering, product, sales, and team morale. End with 3 action items.",
                facts_str,
            )
            write_eml(
                path=f"{EMAIL_OUT}/leadership/week_{week}_sync.eml",
                from_name=sender,
                to_names=list(LEADS.values()),
                subject=f"Week {week} Leadership Sync — Notes & Actions",
                body=body,
                date=sync_date.strftime("%a, %d %b %Y 15:00:00 +0000"),
            )

        console.print("    [green]✓[/green] 4 leadership sync emails.")

    # ── 4. KNOWLEDGE GAP THREADS ───────────────
    def _knowledge_gap_threads(self):
        """Thread about departed-employee knowledge gaps.

        Checks for explicit `knowledge_gap_detected` events first (emitted by
        flow.py when an incident touches a gap area).  Falls back to
        scanning `incident_opened` events for the `involves_bill` flag so the
        thread is generated even if the dedicated event was never fired.
        """
        console.print("  Generating knowledge gap threads...")

        gaps = self.log.facts("knowledge_gap_detected")

        # Fallback: derive gap info from incident_opened events that flagged involves_bill
        if not gaps:
            bill_incidents = [
                e
                for e in self.log.by_type("incident_opened")
                if e.get("facts", {}).get("involves_bill")
            ]
            if bill_incidents:
                inc = bill_incidents[0]
                _legacy = _CFG.get("legacy_system", {}).get("name", "legacy system")
                gaps = [
                    {
                        "facts": {"gap_area": [_legacy], "involves_bill": True},
                        "artifact_ids": inc.get("artifact_ids", {}),
                        "date": inc.get("date", _CFG["simulation"]["start_date"]),
                        "day": inc.get("day", 1),
                        "actors": inc.get("actors", []),
                    }
                ]

        if not gaps:
            console.print("    [dim]No knowledge gap events — skipping.[/dim]")
            return

        gap = gaps[0]
        gap_area = gap["facts"].get(
            "gap_area", [_CFG.get("legacy_system", {}).get("name", "legacy system")]
        )
        ticket = gap["artifact_ids"].get("jira", "ORG-???")
        gap_date = gap.get("date", "2026-03-05")

        # Pull departed employee details from config for richer context
        departed_info = next(iter(DEPARTED_EMPLOYEES.values()), {})
        departed_name = departed_info.get("name", "Bill")
        departed_role = departed_info.get("role", "ex-CTO")
        departed_left = departed_info.get("left", "unknown")
        doc_pct = int(departed_info.get("documented_pct", 0.2) * 100)

        # Resolve participants from config roles
        on_call = resolve_role("on_call_engineer")
        incident_lead = resolve_role("incident_commander")
        hr_lead = resolve_role("hr_lead")
        eng_dept = _CFG.get("roles", {}).get("on_call_engineer", "")
        eng_peers = [n for n in ORG_CHART.get(eng_dept, []) if n != on_call]
        eng_peer = eng_peers[0] if eng_peers else on_call
        legacy_name = _CFG.get("legacy_system", {}).get("name", "legacy system")

        facts_str = (
            f"Incident that triggered gap discovery: {ticket}\n"
            f"Systems with missing documentation: {gap_area}\n"
            f"Departed employee: {departed_name} ({departed_role}, left {departed_left})\n"
            f"Documented percentage: ~{doc_pct}%\n"
            f"Date discovered: {gap_date}"
        )
        turns = [
            {
                "from": on_call,
                "to": [incident_lead, hr_lead, eng_peer],
                "date": gap_date,
                "body": _llm_body(
                    on_call,
                    f"Raise the alarm about a critical knowledge gap in systems {departed_name} owned. "
                    "Propose a 'knowledge excavation sprint'.",
                    facts_str,
                ),
            },
            {
                "from": incident_lead,
                "to": [on_call, hr_lead, eng_peer],
                "date": gap_date,
                "body": _llm_body(
                    incident_lead,
                    "Respond to the alarm. Acknowledge the risk but push back on a full sprint pause. "
                    "Counter-propose 20% time allocation.",
                    facts_str,
                ),
            },
            {
                "from": hr_lead,
                "to": [incident_lead, on_call, eng_peer],
                "date": gap_date,
                "body": _llm_body(
                    hr_lead,
                    f"Propose a Confluence documentation template. Suggest reaching out to {departed_name} for a paid consulting session.",
                    facts_str,
                ),
            },
            {
                "from": on_call,
                "to": [hr_lead, incident_lead],
                "date": gap_date,
                "body": _llm_body(
                    on_call,
                    f"Confirm you and {eng_peer} have started the {legacy_name} documentation page. Agree to reach out to {departed_name}.",
                    facts_str,
                ),
            },
        ]
        write_thread(
            f"{EMAIL_OUT}/threads/knowledge_gap_{departed_name.lower()}",
            f"Knowledge Gap: {departed_name}'s Systems — Action Plan",
            turns,
        )
        console.print(
            f"    [green]✓[/green] Knowledge gap thread written ({departed_name} / {gap_area})."
        )

    # ── 5. HR MORALE EMAILS ────────────────────
    def _hr_morale_emails(self):
        """HR emails driven by actual morale data from the event log."""
        console.print("  Generating HR emails...")
        avg = self.log.avg_morale()

        hr_lead = resolve_role("hr_lead")
        prod_lead = resolve_role("sprint_email_sender")
        on_call = resolve_role("on_call_engineer")
        new_hire = (
            _CFG["simulation"].get("new_hire")
            or (
                ORG_CHART.get(_CFG["simulation"].get("new_hire_dept", "Product"), [""])[
                    0
                ]
            )
        )
        new_hire_dept = _CFG["simulation"].get("new_hire_dept", "Product")
        legacy_name = _CFG.get("legacy_system", {}).get("name", "legacy system")
        legacy_proj = _CFG.get("legacy_system", {}).get("project_name", legacy_name)
        company = _CFG["simulation"]["company_name"]
        avg = self.log.avg_morale()

        write_eml(
            path=f"{EMAIL_OUT}/hr/welcome_{new_hire.lower()}.eml",
            from_name=hr_lead,
            to_names=[new_hire],
            cc_names=[prod_lead],
            subject=f"Welcome to {company}, {new_hire}!",
            body=(
                f"Hi {new_hire},\n\nWe're so glad to have you on the {new_hire_dept} team!\n\n"
                f"Your first week includes team introductions ({prod_lead} will set these up), "
                f"JIRA and Confluence access (Tom will send credentials), and an engineering "
                f"onboarding session with {on_call} on Thursday.\n\n"
                f"One heads-up: our legacy system ({legacy_proj} / {legacy_name}) is in a transition phase. "
                f"Don't be alarmed if you see incident tickets — we're actively stabilising it.\n\n"
                f"Welcome!\n\n{hr_lead}"
            ),
            date=f"Mon, {_CFG['simulation']['start_date'].replace('-', ' ').split()[2]} {_CFG['simulation']['start_date'].split('-')[1]} {_CFG['simulation']['start_date'].split('-')[0]} 08:30:00 +0000",
        )

        intervention_threshold = _CFG["morale"].get("intervention_threshold", 0.55)
        if avg < intervention_threshold:
            facts_str = f"Average team morale during simulation: {avg:.2f} (scale 0-1, threshold {intervention_threshold})"
            body = _llm_body(
                hr_lead,
                "Write a warm, non-alarmist check-in email to team leads about low morale. "
                "Offer 1:1s and remind them of the EAP. Don't mention the metric directly.",
                facts_str,
            )
            write_eml(
                path=f"{EMAIL_OUT}/hr/morale_intervention.eml",
                from_name=hr_lead,
                to_names=list(LEADS.values()),
                subject="Team Pulse — Let's Talk",
                body=body,
            )
            console.print(
                f"    [yellow]Morale intervention email written (avg={avg:.2f})[/yellow]"
            )

        start_year = _CFG["simulation"]["start_date"].split("-")[0]
        write_eml(
            path=f"{EMAIL_OUT}/hr/remote_policy_{start_year}.eml",
            from_name=hr_lead,
            to_names=ALL_NAMES,
            subject=f"Updated: Remote Work Policy {start_year}",
            body=(
                f"Team,\n\nWe've updated our remote work policy for {start_year}. Key changes:\n\n"
                "  • Core hours: 10am–3pm in your local timezone\n"
                "  • Monthly in-person anchor day (first Monday)\n"
                "  • All-hands meetings: Tuesdays 2pm EST\n\n"
                f"Full policy on Confluence. Please review and acknowledge by Friday.\n\n{hr_lead}"
            ),
        )
        console.print("    [green]✓[/green] HR emails written.")

    # ── 6. RETROSPECTIVE SUMMARIES ─────────────
    def _retrospective_summaries(self):
        """Post-retro summary emails to all leads, referencing the actual retro Confluence page."""
        console.print("  Generating retrospective summaries...")
        retros = self.log.retrospectives()
        if not retros:
            console.print("    [dim]No retrospective events.[/dim]")
            return

        sender = resolve_role("sprint_email_sender")
        for retro in retros:
            sprint_num = retro["facts"].get("sprint_number", 1)
            conf_id = retro["artifact_ids"].get("confluence", "CONF-RETRO-???")
            resolved = retro["facts"].get("resolved_incidents", [])
            retro_date = retro.get("date", "2026-03-06")

            facts_str = (
                f"Sprint number: {sprint_num}\n"
                f"Retrospective page: {conf_id}\n"
                f"Incidents resolved this sprint: {resolved}\n"
                f"Date: {retro_date}"
            )
            body = _llm_body(
                sender,
                "Send a brief post-retro summary to all leads. Reference the Confluence page ID. "
                "List 2-3 key takeaways and thank the team.",
                facts_str,
            )
            write_eml(
                path=f"{EMAIL_OUT}/retros/sprint_{sprint_num}_retro_summary.eml",
                from_name=sender,
                to_names=list(LEADS.values()),
                subject=f"Sprint #{sprint_num} Retro Summary — {conf_id}",
                body=body,
                date=retro_date,
            )

        console.print(f"    [green]✓[/green] {len(retros)} retro summaries.")

    # ── 7. SALES PIPELINE EMAILS ───────────────
    def _sales_pipeline_emails(self):
        """Sales pipeline emails — uses real health/incident data for context."""
        console.print("  Generating sales pipeline emails...")
        resolved = self.log.resolved_incidents()
        start_date = datetime.strptime(_CFG["simulation"]["start_date"], "%Y-%m-%d")
        accounts = _CFG.get(
            "sales_accounts", ["Acme Corp", "Beta LLC", "Gamma Inc", "Delta Co"]
        )
        sender = resolve_role("sales_email_sender")
        prod_lead = resolve_role("sprint_email_sender")
        # Second person on product to copy
        prod_dept = _CFG.get("roles", {}).get("sprint_email_sender", "")
        prod_peers = [n for n in ORG_CHART.get(prod_dept, []) if n != prod_lead]
        prod_peer = prod_peers[0] if prod_peers else prod_lead
        # Sales dept peers to CC
        sales_dept = _CFG.get("roles", {}).get("sales_email_sender", "")
        sales_peers = [n for n in ORG_CHART.get(sales_dept, []) if n != sender][:2]

        for week in range(1, 4):
            deals = random.sample(accounts, min(3, len(accounts)))
            report_date = start_date + timedelta(weeks=week - 1, days=4)
            inc_this_wk = [r for r in resolved if r.get("day", 0) <= week * 5]
            stability = (
                "stable"
                if not inc_this_wk
                else f"recovering ({len(inc_this_wk)} incident(s) this period)"
            )

            facts_str = (
                f"Week: {week}\n"
                f"Deals in pipeline: {', '.join(deals)}\n"
                f"Platform stability status: {stability}\n"
                f"Q1 attainment estimate: {50 + week * 12}%\n"
                f"Date: {report_date.strftime('%Y-%m-%d')}"
            )
            body = _llm_body(
                sender,
                f"Write a weekly pipeline update to {prod_lead} and {prod_peer}. "
                "Mention deal statuses, attainment, and any platform stability concerns. "
                "Keep it punchy.",
                facts_str,
            )
            write_eml(
                path=f"{EMAIL_OUT}/sales/week_{week}_pipeline.eml",
                from_name=sender,
                to_names=[prod_lead, prod_peer],
                cc_names=sales_peers,
                subject=f"Week {week} Sales Pipeline Update",
                body=body,
                date=report_date.strftime("%a, %d %b %Y 17:00:00 +0000"),
            )

        console.print("    [green]✓[/green] 3 pipeline emails.")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    gen = EmailGen()
    gen.run()
