from __future__ import annotations

from datetime import datetime, timedelta
from email.message import EmailMessage
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from typing import Any

import yaml

from memory import Memory, SimEvent


BASELINE_MARKER_ID = "velomind_baseline_seeded"
NON_ENG_DEPTS = {"HR_Ops", "Sales_Marketing", "Design", "QA_Support", "Product"}
COMPLETION_ARTIFACT = {
    "HR_Ops": "confluence",
    "Sales_Marketing": "email",
    "Design": "confluence",
    "QA_Support": "confluence",
    "Product": "confluence",
}


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.title_parts: list[str] = []
        self.h1_parts: list[str] = []
        self._in_title = False
        self._in_h1 = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"p", "div", "section", "article", "li", "tr", "h1", "h2", "h3"}:
            self.parts.append("\n")
        if tag == "title":
            self._in_title = True
        if tag == "h1":
            self._in_h1 = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        if tag == "h1":
            self._in_h1 = False
        if tag in {"p", "li", "tr", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if not text:
            return
        self.parts.append(text)
        self.parts.append(" ")
        if self._in_title:
            self.title_parts.append(text)
        if self._in_h1:
            self.h1_parts.append(text)

    @property
    def text(self) -> str:
        raw = "".join(self.parts)
        return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", raw)).strip()

    @property
    def title(self) -> str:
        return " ".join(self.title_parts).strip()

    @property
    def h1(self) -> str:
        return " ".join(self.h1_parts).strip()


def seed_baseline(mem: Memory, config: dict[str, Any], export_dir: Path) -> dict[str, int]:
    """Import the VeloMind worldsim fixture into OrgForge before day 1."""

    marker = mem._db["sim_config"].find_one({"_id": BASELINE_MARKER_ID})
    if marker:
        return marker.get("summary", {})

    source_dir = Path(config.get("velomind_source_dir", "")).expanduser()
    if not source_dir.exists():
        raise FileNotFoundError(f"VeloMind source directory not found: {source_dir}")

    export_dir = export_dir.expanduser().resolve()
    export_dir.mkdir(parents=True, exist_ok=True)

    start_dt = datetime.strptime(config["simulation"]["start_date"], "%Y-%m-%d")
    baseline_dt = start_dt - timedelta(days=1)
    clock = _Clock(baseline_dt)
    name_to_dept = _name_to_dept(config)
    leads = config.get("leads", {})

    _ensure_indexes(mem)

    counts: dict[str, int] = {
        "bookstack_pages": _seed_bookstack(mem, source_dir, export_dir, clock),
        "plane_issues": _seed_plane(mem, source_dir, export_dir, clock, name_to_dept),
        "matrix_messages": _seed_matrix(mem, source_dir, export_dir, clock),
        "mailpit_emails": _seed_mailpit(mem, source_dir, export_dir, clock),
        "twenty_crm_records": _seed_twenty_crm(
            mem, source_dir, export_dir, clock, config, leads
        ),
        "trudesk_tickets": _seed_trudesk(mem, source_dir, export_dir, clock, leads),
        "gitea_commits": _seed_gitea(mem, source_dir, export_dir, clock),
        "scenario_docs": _seed_scenario_docs(mem, source_dir, export_dir, clock),
    }

    timestamp = clock.peek().isoformat()
    date_str = timestamp[:10]
    mem.log_event(
        SimEvent(
            type="velomind_baseline_seeded",
            day=0,
            date=date_str,
            timestamp=timestamp,
            actors=["Maya", "Amara", "Zoe", "Ben"],
            artifact_ids={"source_dir": str(source_dir)},
            facts=counts,
            summary=(
                "Imported VeloMind baseline artifacts across BookStack, Plane, "
                "Matrix, Mailpit, Twenty CRM, Trudesk, and Gitea."
            ),
            tags=["genesis", "velomind", "baseline_import"],
        )
    )

    mem._db["sim_config"].update_one(
        {"_id": BASELINE_MARKER_ID},
        {
            "$set": {
                "_id": BASELINE_MARKER_ID,
                "source_dir": str(source_dir),
                "export_dir": str(export_dir),
                "summary": counts,
                "created_at": timestamp,
            }
        },
        upsert=True,
    )
    return counts


class _Clock:
    def __init__(self, day: datetime) -> None:
        self._dt = day.replace(hour=8, minute=0, second=0, microsecond=0)

    def tick(self, minutes: int = 7) -> datetime:
        value = self._dt
        self._dt = self._dt + timedelta(minutes=minutes)
        return value

    def peek(self) -> datetime:
        return self._dt


def _ensure_indexes(mem: Memory) -> None:
    mem._db["emails"].create_index([("embed_id", 1)], unique=True)
    mem._db["sf_accounts"].create_index([("account_id", 1)], unique=True)
    mem._db["sf_opps"].create_index([("opportunity_id", 1)], unique=True)
    mem._db["zd_tickets"].create_index([("ticket_id", 1)], unique=True)
    mem._db["git_commits"].create_index([("repo", 1), ("branch", 1)])
    mem._db["crm_notes"].create_index([("title", 1)])
    mem._db["crm_tasks"].create_index([("title", 1), ("due", 1)])


def _load_yaml(path: Path) -> Any:
    with open(path, "r") as fh:
        return yaml.safe_load(fh) or {}


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2, default=str)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)


def _slug(text: str, max_len: int = 64) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return (value or "item")[:max_len].strip("-")


def _compact_id(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper()) or "UNKNOWN"


def _display_user(value: str) -> str:
    raw = (value or "").split("@", 1)[0].replace("-", ".").replace("_", ".")
    if not raw:
        return "Unknown"
    parts = [p for p in raw.split(".") if p]
    if not parts:
        return raw.capitalize()
    if len(parts) == 1:
        return parts[0].capitalize()
    return " ".join(part.capitalize() for part in parts)


def _first_name(value: str) -> str:
    return _display_user(value).split()[0]


def _email_addr(value: str) -> str:
    if "@" in value:
        return value
    return f"{value.replace(' ', '.').lower()}@velomind.co"


def _parse_html(path: Path) -> tuple[str, str]:
    parser = _HTMLTextExtractor()
    parser.feed(path.read_text())
    title = parser.title or parser.h1 or path.stem.replace("-", " ").title()
    return title, parser.text


def _name_to_dept(config: dict[str, Any]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for dept, members in config.get("org_chart", {}).items():
        for name in members:
            mapping[name] = dept
    return mapping


def _seed_bookstack(
    mem: Memory, source_dir: Path, export_dir: Path, clock: _Clock
) -> int:
    pages_dir = source_dir / "data" / "bookstack" / "pages"
    count = 0
    for page_path in sorted(pages_dir.glob("*.html")):
        title, body = _parse_html(page_path)
        slug = _slug(page_path.stem)
        artifact_id = f"BOOKSTACK-{slug.upper()}"
        timestamp = clock.tick().isoformat()
        date_str = timestamp[:10]
        markdown = f"# {title}\n\n{body}\n"
        out_path = export_dir / "confluence" / "velomind" / "bookstack" / f"{slug}.md"
        _write_text(out_path, markdown)
        mem.embed_artifact(
            id=artifact_id,
            type="confluence",
            title=title,
            content=markdown,
            day=0,
            date=date_str,
            timestamp=timestamp,
            metadata={
                "source_system": "bookstack",
                "source_path": str(page_path),
                "file_path": str(out_path),
                "author": "VeloMind import",
                "topic": "Atlas One launch readiness",
            },
        )
        count += 1
    return count


def _seed_plane(
    mem: Memory,
    source_dir: Path,
    export_dir: Path,
    clock: _Clock,
    name_to_dept: dict[str, str],
) -> int:
    issues = _load_yaml(source_dir / "data" / "plane" / "issues.yaml")
    count = 0
    for project, project_issues in issues.items():
        for idx, issue in enumerate(project_issues or [], start=100):
            ticket_id = f"{project}-{idx}"
            assignee = _first_name(issue.get("assignee", ""))
            dept = name_to_dept.get(assignee, _dept_for_project(project))
            timestamp = clock.tick().isoformat()
            date_str = timestamp[:10]
            status = _status(issue.get("state", "Todo"))
            ticket = {
                "id": ticket_id,
                "title": issue.get("title", "Untitled VeloMind issue"),
                "description": issue.get("desc", ""),
                "status": status,
                "assignee": assignee,
                "dept": dept,
                "dept_type": "non_eng" if dept in NON_ENG_DEPTS else "eng",
                "completion_artifact": COMPLETION_ARTIFACT.get(dept, "slack"),
                "sprint": 0,
                "sprint_theme": "Atlas One launch readiness baseline",
                "story_points": _story_points(issue.get("priority", "medium")),
                "priority": issue.get("priority", "medium"),
                "labels": issue.get("labels", []),
                "linked_prs": [],
                "comments": [
                    {
                        "author": "Plane import",
                        "text": "Imported from the VeloMind Plane fixture.",
                        "timestamp": timestamp,
                    }
                ],
                "created_at": timestamp,
                "updated_at": timestamp,
                "source_system": "plane",
                "source_project": project,
            }
            mem.upsert_ticket(ticket)
            _write_json(export_dir / "jira" / f"{ticket_id}.json", ticket)
            mem.embed_artifact(
                id=ticket_id,
                type="jira",
                title=ticket["title"],
                content=json.dumps(ticket, indent=2),
                day=0,
                date=date_str,
                timestamp=timestamp,
                metadata={
                    "source_system": "plane",
                    "project": project,
                    "dept": dept,
                    "priority": issue.get("priority", "medium"),
                },
            )
            count += 1
    return count


def _status(value: str) -> str:
    mapping = {
        "todo": "To Do",
        "backlog": "To Do",
        "in progress": "In Progress",
        "in review": "In Review",
        "done": "Done",
        "closed": "Closed",
    }
    return mapping.get(str(value).strip().lower(), str(value))


def _story_points(priority: str) -> int:
    return {"urgent": 5, "high": 3, "medium": 2, "low": 1}.get(
        str(priority).lower(), 2
    )


def _dept_for_project(project: str) -> str:
    return {
        "BIKE": "Engineering_Hardware",
        "ML": "Engineering_Backend",
        "APP": "Engineering_Mobile",
        "PILOT": "QA_Support",
        "GTM": "Sales_Marketing",
    }.get(project, "Product")


def _seed_matrix(
    mem: Memory, source_dir: Path, export_dir: Path, clock: _Clock
) -> int:
    conversations = _load_yaml(source_dir / "data" / "matrix" / "conversations.yaml")
    count = 0
    for channel, messages in conversations.items():
        channel_docs: list[dict[str, Any]] = []
        threads: dict[str, list[dict[str, Any]]] = {}
        for msg in messages or []:
            timestamp = clock.tick(minutes=3).isoformat()
            thread_key = msg.get("thread") or msg.get("id")
            thread_id = f"matrix_{channel}_{thread_key}"
            doc = {
                "_id": f"matrix:{channel}:{msg.get('id')}",
                "channel": channel,
                "user": _first_name(msg.get("user", "")),
                "source_user": msg.get("user"),
                "text": msg.get("text", ""),
                "ts": timestamp,
                "date": timestamp[:10],
                "thread_id": thread_id,
                "source_id": msg.get("id"),
                "reactions": msg.get("reactions", []),
                "source_system": "matrix",
            }
            mem._db["slack_messages"].replace_one({"_id": doc["_id"]}, doc, upsert=True)
            channel_docs.append(doc)
            threads.setdefault(thread_id, []).append(doc)
            count += 1
        _write_json(
            export_dir / "slack" / "channels" / channel / f"{clock.peek().date()}.json",
            channel_docs,
        )
        for thread_id, thread_msgs in threads.items():
            first = thread_msgs[0]
            title = f"#{channel}: {first['text'][:70]}"
            content = "\n".join(
                f"{m['user']}: {m['text']}" for m in sorted(thread_msgs, key=lambda x: x["ts"])
            )
            mem.embed_artifact(
                id=f"MATRIX-{_slug(thread_id).upper()}",
                type="slack",
                title=title,
                content=content,
                day=0,
                date=first["date"],
                timestamp=first["ts"],
                metadata={
                    "source_system": "matrix",
                    "channel": channel,
                    "thread_id": thread_id,
                },
            )
    return count


def _seed_mailpit(
    mem: Memory, source_dir: Path, export_dir: Path, clock: _Clock
) -> int:
    emails = _load_yaml(source_dir / "data" / "mailpit" / "emails.yaml")
    for idx, email in enumerate(emails or [], start=1):
        timestamp = clock.tick(minutes=11).isoformat()
        sender = email.get("from", "")
        recipients = email.get("to", [])
        if isinstance(recipients, str):
            recipients = [recipients]
        body_html = email.get("body", "")
        body_text = _html_to_text(body_html)
        direction = _email_direction(sender)
        embed_id = f"mailpit-{idx:03d}"
        from_addr = _email_addr(sender)
        to_addrs = [_email_addr(r) for r in recipients]
        doc = {
            "_id": embed_id,
            "embed_id": embed_id,
            "direction": direction,
            "from_name": _display_user(sender),
            "from_addr": from_addr,
            "to_name": ", ".join(_display_user(r) for r in recipients),
            "to_addr": ", ".join(to_addrs),
            "to_addrs": to_addrs,
            "subject": email.get("subject", "(no subject)"),
            "body": body_text,
            "body_html": body_html,
            "timestamp": timestamp,
            "date": timestamp[:10],
            "source_system": "mailpit",
        }
        mem._db["emails"].replace_one({"embed_id": embed_id}, doc, upsert=True)
        eml_path = (
            export_dir
            / "emails"
            / direction
            / timestamp[:10]
            / f"{embed_id}.eml"
        )
        _write_email(eml_path, doc)
        mem.embed_artifact(
            id=embed_id,
            type="email",
            title=doc["subject"],
            content=f"From: {doc['from_name']} <{from_addr}>\nTo: {doc['to_addr']}\n\n{body_text}",
            day=0,
            date=timestamp[:10],
            timestamp=timestamp,
            metadata={
                "source_system": "mailpit",
                "direction": direction,
                "from_addr": from_addr,
                "to_addrs": to_addrs,
                "file_path": str(eml_path),
            },
        )
    return len(emails or [])


def _html_to_text(raw: str) -> str:
    parser = _HTMLTextExtractor()
    parser.feed(raw or "")
    return parser.text


def _email_direction(sender: str) -> str:
    if "@" not in sender:
        return "internal"
    return "internal" if sender.endswith("@velomind.co") else "inbound"


def _write_email(path: Path, doc: dict[str, Any]) -> None:
    msg = EmailMessage()
    msg["From"] = f"{doc['from_name']} <{doc['from_addr']}>"
    msg["To"] = doc["to_addr"]
    msg["Subject"] = doc["subject"]
    msg["Date"] = doc["timestamp"]
    msg.set_content(doc["body"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(bytes(msg))


def _seed_twenty_crm(
    mem: Memory,
    source_dir: Path,
    export_dir: Path,
    clock: _Clock,
    config: dict[str, Any],
    leads: dict[str, str],
) -> int:
    crm = _load_yaml(source_dir / "data" / "twenty" / "crm.yaml")
    contacts_by_company: dict[str, list[dict[str, Any]]] = {}
    for contact in crm.get("contacts", []):
        contacts_by_company.setdefault(contact.get("company", ""), []).append(contact)

    configured_contacts = {
        contact.get("org"): contact for contact in config.get("external_contacts", [])
    }
    count = 0
    for company in crm.get("companies", []):
        timestamp = clock.tick().isoformat()
        company_name = company.get("name", "Unknown")
        configured = configured_contacts.get(company_name, {})
        liaison = configured.get("internal_liaison", "Sales_Marketing")
        primary = (contacts_by_company.get(company_name) or [{}])[0]
        account_id = f"ACC-{_compact_id(company_name)}"
        account = {
            "account_id": account_id,
            "name": company_name,
            "type": "Vendor" if configured.get("category") == "vendor" else "Customer",
            "primary_contact_name": _contact_name(primary),
            "primary_contact_email": primary.get("email", configured.get("email", "")),
            "contact_role": primary.get("title", configured.get("contact_role", "")),
            "owner": leads.get(liaison, liaison),
            "industry": configured.get("industry", "Micromobility"),
            "tier": configured.get("tier"),
            "employee_count": company.get("employees"),
            "website": f"https://{company.get('domain', '')}".rstrip("/"),
            "billing_region": configured.get("billing_region", "NA"),
            "billing_city": configured.get("billing_city", company.get("city", "")),
            "billing_state": configured.get("billing_state"),
            "billing_country": configured.get("billing_country", "US"),
            "arr": configured.get("arr", 0),
            "is_lighthouse": configured.get("is_lighthouse", False),
            "expansion_potential": configured.get("expansion_potential", 0),
            "status": "Active",
            "sentiment_baseline": configured.get("sentiment_baseline", 0.7),
            "risk_flag": configured.get("sentiment_baseline", 0.7) < 0.5,
            "contract_renewal_date": configured.get("contract_renewal_date"),
            "last_activity_date": timestamp,
            "created_at": timestamp,
            "source_system": "twenty",
        }
        mem._db["sf_accounts"].update_one(
            {"account_id": account_id},
            {"$set": {k: v for k, v in account.items() if v is not None}},
            upsert=True,
        )
        _write_json(export_dir / "salesforce" / "accounts" / f"{account_id}.json", account)
        count += 1

    for idx, deal in enumerate(crm.get("deals", []), start=100):
        timestamp = clock.tick().isoformat()
        opp_id = f"OPP-VM-{idx}"
        company_name = deal.get("company", "Unknown")
        configured = configured_contacts.get(company_name, {})
        liaison = configured.get("internal_liaison", "Sales_Marketing")
        stage = _sf_stage(deal.get("stage", "New"))
        opp = {
            "opportunity_id": opp_id,
            "name": deal.get("name", company_name),
            "account_name": company_name,
            "stage": stage,
            "probability": _stage_probability(stage),
            "amount": deal.get("amount", 0),
            "owner": leads.get(liaison, liaison),
            "type": "New Business",
            "lead_source": "VeloMind scenario seed",
            "close_date": "2026-06-15T00:00:00Z",
            "description": deal.get("description", ""),
            "primary_contact_email": deal.get("contact", ""),
            "risk_notes": _risk_notes(deal.get("description", "")),
            "touchpoints": [],
            "created_at": timestamp,
            "updated_at": timestamp,
            "_seq": idx,
            "source_system": "twenty",
        }
        mem._db["sf_opps"].update_one(
            {"opportunity_id": opp_id}, {"$set": opp}, upsert=True
        )
        _write_json(export_dir / "salesforce" / "opportunities" / f"{opp_id}.json", opp)
        mem.embed_artifact(
            id=opp_id,
            type="sf_opportunity",
            title=f"{deal.get('name', company_name)} — {stage}",
            content=json.dumps(opp, indent=2),
            day=0,
            date=timestamp[:10],
            timestamp=timestamp,
            metadata={"source_system": "twenty", "account_name": company_name},
        )
        count += 1

    for idx, note in enumerate(crm.get("notes", []), start=1):
        timestamp = clock.tick().isoformat()
        note_id = f"CRM-NOTE-{idx:03d}"
        doc = {
            "_id": note_id,
            "note_id": note_id,
            "title": note.get("title", "CRM note"),
            "body": note.get("body", ""),
            "targets": note.get("targets", []),
            "created_at": timestamp,
            "source_system": "twenty",
        }
        mem._db["crm_notes"].replace_one({"_id": note_id}, doc, upsert=True)
        _write_json(export_dir / "salesforce" / "notes" / f"{note_id}.json", doc)
        mem.embed_artifact(
            id=note_id,
            type="crm_note",
            title=doc["title"],
            content=doc["body"],
            day=0,
            date=timestamp[:10],
            timestamp=timestamp,
            metadata={"source_system": "twenty", "targets": note.get("targets", [])},
        )
        count += 1

    for idx, task in enumerate(crm.get("tasks", []), start=1):
        timestamp = clock.tick().isoformat()
        task_id = f"CRM-TASK-{idx:03d}"
        doc = {
            "_id": task_id,
            "task_id": task_id,
            "title": task.get("title", "CRM task"),
            "body": task.get("body", ""),
            "due": task.get("due"),
            "status": task.get("status", "TODO"),
            "targets": task.get("targets", []),
            "created_at": timestamp,
            "source_system": "twenty",
        }
        mem._db["crm_tasks"].replace_one({"_id": task_id}, doc, upsert=True)
        _write_json(export_dir / "salesforce" / "tasks" / f"{task_id}.json", doc)
        count += 1

    return count


def _contact_name(contact: dict[str, Any]) -> str:
    return " ".join(part for part in [contact.get("first"), contact.get("last")] if part)


def _sf_stage(stage: str) -> str:
    return {
        "New": "Prospecting",
        "Qualification": "Prospecting",
        "Meeting": "Value Proposition",
        "Proposal": "Proposal/Price Quote",
        "Customer": "Closed Won",
    }.get(stage, stage)


def _stage_probability(stage: str) -> int:
    return {
        "Prospecting": 10,
        "Value Proposition": 25,
        "Proposal/Price Quote": 50,
        "Negotiation/Review": 75,
        "Closed Won": 100,
        "Closed Lost": 0,
    }.get(stage, 10)


def _risk_notes(description: str) -> list[str]:
    lower = description.lower()
    if any(term in lower for term in ["risk", "held", "ts-17b", "bat-042", "safety"]):
        return [description]
    return []


def _seed_trudesk(
    mem: Memory, source_dir: Path, export_dir: Path, clock: _Clock, leads: dict[str, str]
) -> int:
    tickets = _load_yaml(source_dir / "data" / "trudesk" / "tickets.yaml")
    for idx, item in enumerate(tickets or [], start=100):
        timestamp = clock.tick(minutes=9).isoformat()
        ticket_id = f"ZD-VM-{idx}"
        owner = _first_name(item.get("owner", "zoe.kline"))
        priority = int(item.get("priority", 1))
        requester_org = _support_org(item.get("subject", ""))
        doc = {
            "ticket_id": ticket_id,
            "type": "incident" if "[Safety]" in item.get("subject", "") else "question",
            "status": "Open" if priority >= 2 else "Pending",
            "priority": {3: "urgent", 2: "high", 1: "normal", 0: "low"}.get(
                priority, "normal"
            ),
            "description": item.get("issue", ""),
            "assignee_email": f"{owner.lower()}@velomind.co",
            "assignee": owner,
            "requester": {
                "name": requester_org,
                "email": f"{_slug(requester_org)}@example.com",
                "org_name": requester_org,
            },
            "subject": item.get("subject", "Support ticket"),
            "org_name": requester_org,
            "channel": "support_portal",
            "email_type": "complaint" if priority >= 2 else "question",
            "tags": ["support", "trudesk", "velomind"],
            "satisfaction_rating": {"score": "unoffered"},
            "created_at": timestamp,
            "updated_at": timestamp,
            "related_incident": None,
            "comments": [
                {
                    "author": requester_org,
                    "text": item.get("issue", ""),
                    "timestamp": timestamp,
                }
            ],
            "_seq": idx,
            "source_system": "trudesk",
        }
        mem._db["zd_tickets"].update_one(
            {"ticket_id": ticket_id}, {"$set": doc}, upsert=True
        )
        _write_json(export_dir / "zendesk" / "tickets" / f"{ticket_id}.json", doc)
        mem.embed_artifact(
            id=ticket_id,
            type="zd_ticket",
            title=f"[{ticket_id}] {doc['subject']}",
            content=f"{doc['org_name']}\n\n{doc['description']}",
            day=0,
            date=timestamp[:10],
            timestamp=timestamp,
            metadata={
                "source_system": "trudesk",
                "ticket_id": ticket_id,
                "priority": doc["priority"],
            },
        )
    return len(tickets or [])


def _support_org(subject: str) -> str:
    match = re.match(r"\[([^\]]+)\]\s*(.*)", subject or "")
    if not match:
        return "Unknown"
    tag = match.group(1)
    rest = match.group(2)
    if tag in {"Rider", "Safety", "App"}:
        return "East Bay Commuter Pilot"
    if tag == "Dealer":
        if "Bay Trail" in rest:
            return "Bay Trail Outfitters"
        return "Golden Gate Cycles"
    return tag


def _seed_gitea(
    mem: Memory, source_dir: Path, export_dir: Path, clock: _Clock
) -> int:
    gitea_dir = source_dir / "data" / "gitea"
    count = 0
    pr_count = 0
    for yaml_path in sorted(gitea_dir.glob("*.commits.yaml")):
        repo = yaml_path.name.replace(".commits.yaml", "")
        data = _load_yaml(yaml_path)
        repo_records: list[dict[str, Any]] = []
        for commit in data.get("commits", []):
            timestamp = clock.tick(minutes=5).isoformat()
            record = _commit_doc(repo, "main", commit, timestamp, count)
            mem._db["git_commits"].replace_one({"_id": record["_id"]}, record, upsert=True)
            repo_records.append(record)
            count += 1
        for branch in data.get("branches", []):
            branch_name = branch.get("name", "branch")
            branch_commits = branch.get("commits", [])
            for commit in branch_commits:
                timestamp = clock.tick(minutes=5).isoformat()
                record = _commit_doc(repo, branch_name, commit, timestamp, count)
                mem._db["git_commits"].replace_one(
                    {"_id": record["_id"]}, record, upsert=True
                )
                repo_records.append(record)
                count += 1
            if branch.get("pull_request"):
                pr_count += 1
                pr = branch["pull_request"]
                timestamp = clock.tick().isoformat()
                pr_id = f"PR-{99 + pr_count}"
                author = _first_name(
                    (branch_commits or data.get("commits", [{}]))[-1].get("author", "")
                )
                pr_doc = {
                    "pr_id": pr_id,
                    "ticket_id": _match_ticket(pr.get("title", ""), pr.get("body", "")),
                    "linked_ticket": _match_ticket(pr.get("title", ""), pr.get("body", "")),
                    "title": pr.get("title", "Gitea PR"),
                    "description": pr.get("body", ""),
                    "author": author,
                    "author_email": f"{author.lower()}@velomind.co",
                    "reviewers": ["Amara", "Grant"] if author != "Amara" else ["Kenji", "Grant"],
                    "status": "open",
                    "comments": [],
                    "created_at": timestamp,
                    "repo": repo,
                    "branch": branch_name,
                    "source_system": "gitea",
                }
                mem.upsert_pr(pr_doc)
                _write_json(export_dir / "git" / "prs" / f"{pr_id}.json", pr_doc)
                mem.embed_artifact(
                    id=pr_id,
                    type="pr",
                    title=pr_doc["title"],
                    content=json.dumps(pr_doc, indent=2),
                    day=0,
                    date=timestamp[:10],
                    timestamp=timestamp,
                    metadata={"source_system": "gitea", "repo": repo, "branch": branch_name},
                )
        _write_json(export_dir / "git" / "gitea" / f"{repo}.json", repo_records)
    return count


def _commit_doc(
    repo: str, branch: str, commit: dict[str, Any], timestamp: str, seq: int
) -> dict[str, Any]:
    return {
        "_id": f"gitea:{repo}:{branch}:{seq:04d}",
        "repo": repo,
        "branch": branch,
        "message": commit.get("message", ""),
        "author": _first_name(commit.get("author", "")),
        "author_username": commit.get("author", ""),
        "files": commit.get("files", []),
        "created_at": timestamp,
        "source_system": "gitea",
    }


def _match_ticket(title: str, body: str) -> str:
    text = f"{title} {body}".lower()
    if "stale torque" in text:
        return "BIKE-104"
    if "rollback" in text:
        return "BIKE-106"
    if "range" in text:
        return "ML-100"
    if "dealer dashboard" in text:
        return "APP-101"
    return ""


def _seed_scenario_docs(
    mem: Memory, source_dir: Path, export_dir: Path, clock: _Clock
) -> int:
    count = 0
    for source_path in [source_dir / "README.md", source_dir / "scenario.yaml"]:
        if not source_path.exists():
            continue
        timestamp = clock.tick().isoformat()
        content = source_path.read_text()
        artifact_id = f"VELOMIND-{source_path.stem.upper()}"
        out_path = export_dir / "confluence" / "velomind" / source_path.name
        _write_text(out_path, content)
        mem.embed_artifact(
            id=artifact_id,
            type="confluence",
            title=f"VeloMind {source_path.name}",
            content=content,
            day=0,
            date=timestamp[:10],
            timestamp=timestamp,
            metadata={
                "source_system": "worldsim",
                "source_path": str(source_path),
                "file_path": str(out_path),
                "author": "VeloMind import",
                "topic": "scenario ground truth",
            },
        )
        count += 1
    return count
