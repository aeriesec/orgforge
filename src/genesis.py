# genesis.py
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
import random
import re
from typing import List
from config_loader import (
    BASE,
    COMPANY_DESCRIPTION,
    COMPANY_NAME,
    CONFIG,
    INDUSTRY,
    LEADS,
    LEGACY,
    ORG_CHART,
)
from memory import Memory, SimEvent
from agent_factory import make_agent
from crewai import Task, Crew

logger = logging.getLogger("orgforge.genesis")

_DEFAULT_SOURCE_COUNT = 15
_MAX_RETRIES = 3


def initialize(config, planner_llm, reset=False):
    """
    The 'Valet' function: handles setup, reset, and seeding in one go.
    Returns the initialized Memory object.
    """
    from memory import Memory

    mem = Memory()

    if reset:
        mem.reset(export_dir=config.get("base_dir", "export"))
        logger.info("[genesis] 🧹 Database and exports wiped.")

    logger.info("[genesis] 🚀 Seeding corporate ground truth...")
    seed_tech_stack(mem, planner_llm)
    seed_external_sources(mem, planner_llm)
    seed_crm_accounts(mem)
    seed_knowledge_gaps(mem)
    logger.info("[genesis] ✅ Seeding complete.")

    return mem


def seed_external_sources(mem: Memory, planner_llm):
    """Generates the 15 external vendors/customers and saves to MongoDB."""
    if mem.get_inbound_email_sources():
        return

    logger.info("[cyan]🌐 Generating inbound email sources...[/cyan]")

    tech_stack = mem.tech_stack_for_prompt()
    dept_str = ", ".join(LEADS.keys())

    all_names = [name for members in ORG_CHART.values() for name in members]

    agent = make_agent(
        role="Enterprise IT Architect",
        goal=f"Design the realistic external email ecosystem for {COMPANY_NAME} which {COMPANY_DESCRIPTION}.",
        backstory=(
            f"You are an experienced enterprise architect who understands "
            f"communication patterns between a {INDUSTRY} company and its "
            f"vendors, customers, and partners."
        ),
        llm=planner_llm,
    )

    task = Task(
        description=(
            f"Generate 15 realistic inbound email sources. EXACTLY 8 must be 'customer' category, and 7 must be 'vendor' category.\n"
            f"TECH STACK: {tech_stack}\n"
            f"DEPARTMENTS: {dept_str}\n"
            f"DEPARTMENTAL LIAISON LOGIC (Assign Liaisons Based on These Rules):\n"
            f"  - Engineering_Backend: Responsible for Infrastructure (AWS), Databases (TitanDB), Source Control (GitHub), and Monitoring.\n"
            f"  - Engineering_Mobile: Responsible for React Native and mobile platform issues.\n"
            f"  - Product: Responsible for project management (Jira) and feature roadmaps.\n"
            f"  - Sales_Marketing: Responsible for payment/data vendors (e.g., Stripe) and Customer communication.\n"
            f"  - QA_Support: Responsible for CI/CD (Jenkins) and testing tool alerts.\n"
            f"  - HR_Ops: Responsible for legal, compliance, and payroll vendors.\n\n"
            f"Rules:\n"
            f"  - HUMAN NAMES: The 'first_name' and 'last_name' field MUST be a realistic human name representing the Point of Contact (e.g., 'Marcus Thorne').\n"
            f"  - NO DUPLICATE NAMES: Ensure no new generated names overlap with these: {all_names}.\n"
            f"  - PERSONA DICT: Include a nested 'persona' object with 'typing_quirks' (string), 'social_role' (string, matching contact_role), and 'expertise' (array of strings).\n"
            f"  - ADHERENCE: Use ONLY vendors that appear in the TECH STACK above. If Jira is listed, never use Trello.\n"
            f"  - FIRMOGRAPHICS (Customers ONLY): Include 'industry' (e.g. Financial Services), 'tier' (Enterprise, Mid-Market, SMB), 'billing_region' (NA, EMEA, APAC), 'billing_city', 'billing_state' (2-letter code if US), 'billing_country', and 'arr' (e.g. 50000, 120000, 350000).\n"
            f"  - STRATEGIC (Customers ONLY): Include 'is_lighthouse' (bool), 'expansion_potential' (int 1-10), and 'contract_renewal_date' (ISO Date string).\n"
            f"  - TECHNICAL (Vendors ONLY): Include 'integration_complexity' (Low, Med, High) and 'version_in_use' (e.g., 'v2 Beta', 'Legacy').\n"
            f"  - HEALTH SENSITIVITY: Include 'trigger_health_threshold' (int 0-100). Scale: Infrastructure/Enterprise (85-98), SMB/Standard Vendors (70-85).\n"
            f"  - PERSONA: Include 'contact_role' (e.g. VP Engineering, Procurement) and 'persona_archetype' (e.g. The Champion, The Skeptic, The Bureaucrat).\n"
            f"  - DYNAMICS: Include 'expected_sla_hours' (int: 2, 4, 24, 48), 'cadence' (daily, weekly, bi-weekly, reactive), and 'timezone_offset' (int: -8 to +8).\n"
            f"  - RELATIONSHIP: Include 'sentiment_baseline' (float 0.0 to 1.0) and 'history_summary' (1 short sentence mapping the history).\n"
            f"  - TOPICS: Provide 3-5 hyper-specific topics (e.g., 'GitHub Actions Runner Timeout' or 'Stripe API 402 Payment Required').\n"
            f"  - CATEGORY: exactly 'vendor' or 'customer'.\n"
            f"  - TRIGGER_ON: array of 'always', 'incident', 'low_health'.\n"
            f"  - TONE: formal | technical | frustrated | urgent | friendly.\n\n"
            f"Raw JSON array only — no preamble, no markdown fences:\n"
            f"[\n"
            f'  {{"name":"GitHub","org":"GitHub Inc.","first_name":"Jake","last_name": "Smith","org":"GitHub Inc.","email":"j.smith@github.com",'
            f'"category":"vendor","internal_liaison":"Engineering_Backend",'
            f'"contact_role":"Senior Technical Account Manager","persona_archetype":"The Technical Expert",'
            f'"trigger_on":["incident", "low_health"],"trigger_health_threshold":95,'
            f'"expected_sla_hours":4,"cadence":"reactive","timezone_offset":-8,'
            f'"integration_complexity":"High","version_in_use":"Enterprise Cloud",'
            f'"sentiment_baseline":0.8,"history_summary":"Solid uptime, but API rate limits frequently cause friction.",'
            f'"tone":"technical","topics":["Webhooks failing with 5xx","Pull Request comment API latency"]}},\n'
            f'  {{"name":"GlobalFinance","org":"GlobalFinance Corp","email":"cto@globalfinance.com",'
            f'"category":"customer","internal_liaison":"Sales_Marketing",'
            f'"contact_role":"CTO","persona_archetype":"The Skeptic",'
            f'"persona": {{"typing_quirks": "terse, lowercase heavy, fast responses", "social_role": "CTO", "expertise": ["enterprise architecture", "security compliance"]}},'
            f'"trigger_on":["always","incident"],"trigger_health_threshold":90,'
            f'"expected_sla_hours":2,"cadence":"weekly","timezone_offset":-5,'
            f'"is_lighthouse":true,"expansion_potential":8,"contract_renewal_date":"2026-12-01T00:00:00Z",'
            f'"sentiment_baseline":0.4,"history_summary":"Demanding enterprise client, currently evaluating competitors for next year.",'
            f'"tone":"formal","topics":["SLA reporting","Contract renewal"],"industry":"Financial Services",'
            f'"tier":"Enterprise","billing_region":"NA","billing_city":"New York","billing_state":"NY","billing_country":"USA","arr":250000}}\n'
            f"]"
        ),
        expected_output=f"Raw JSON array of {_DEFAULT_SOURCE_COUNT} source objects.",
        agent=agent,
    )

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            logger.info(
                f"[genesis] Generating external sources (Attempt {attempt}/{_MAX_RETRIES})..."
            )
            result = str(Crew(agents=[agent], tasks=[task]).kickoff()).strip()

            sources = _parse_sources(result)

            if isinstance(sources, list) and len(sources) >= 10:
                mem.save_inbound_email_sources(sources)

                logger.info(f"[genesis] ✅ Successfully seeded {len(sources)} sources.")
                for s in sources:
                    logger.info(
                        f"    [dim]→ [{s['category']}] {s['name']} "
                        f"({s['internal_liaison']}) triggers={s['trigger_on']}[/dim]"
                    )
                return

            raise ValueError("Incomplete or malformed list returned.")

        except Exception as e:
            logger.warning(f"[genesis] ⚠ Attempt {attempt} failed: {e}")
            if attempt == _MAX_RETRIES:
                logger.error(
                    "[genesis] ❌ All retries failed. Simulation cannot start without ground truth."
                )
                raise SystemExit(1)

    pass


def seed_tech_stack(mem: Memory, planner_llm):
    """Generates the tech stack ground truth and saves to Confluence."""
    if mem._artifacts.find_one({"type": "tech_stack"}):
        return

    logger.info("[genesis] Generating tech stack...")

    agent = make_agent(
        role="Principal Engineer",
        goal="Define the canonical technology stack for this company.",
        backstory=(
            f"You are a principal engineer at {COMPANY_NAME}, "
            f"a {INDUSTRY} company. You are documenting the actual "
            f"technologies in use — not aspirational, not greenfield. "
            f"This is a company with years of history and legacy decisions."
        ),
        llm=planner_llm,
    )
    task = Task(
        description=(
            f"Define the canonical tech stack for {COMPANY_NAME} "
            f"which {COMPANY_DESCRIPTION}\n\n"
            f"The legacy system is called '{LEGACY['name']}' "
            f"({LEGACY['description']}).\n\n"
            f"Respond ONLY with a JSON object with these keys:\n"
            f"  database, backend_language, frontend_language, mobile, "
            f"  infra, message_queue, source_control, ci_cd, "
            f"  monitoring, notable_quirks\n\n"
            f"Each value is a short string (1-2 sentences max). "
            f"Include at least one legacy wart or technical debt item. "
            f"No preamble, no markdown fences."
        ),
        expected_output="A single JSON object. No preamble.",
        agent=agent,
    )

    raw = str(Crew(agents=[agent], tasks=[task], verbose=False).kickoff()).strip()

    try:
        stack = json.loads(raw.replace("```json", "").replace("```", "").strip())
    except json.JSONDecodeError:
        logger.warning(
            "[confluence] Tech stack JSON parse failed — using minimal fallback."
        )
        stack = {
            "notable_quirks": "Stack unknown — legacy system predates documentation."
        }

    mem.save_tech_stack(stack)
    logger.info(f"[confluence] ✓ Tech stack established: {list(stack.keys())}")

    pass


def seed_crm_accounts(mem: Memory):
    """Seeds Salesforce accounts from the external sources in MongoDB."""
    logger.info("[genesis] Seeding CRM accounts...")

    _zd = mem._db["zd_tickets"]
    _sf_a = mem._db["sf_accounts"]
    _sf_o = mem._db["sf_opps"]
    _emails = mem._db["emails"]

    _zd.create_index([("ticket_id", 1)], unique=True)
    _zd.create_index([("status", 1)])
    _zd.create_index([("related_incident", 1)])
    _sf_a.create_index([("account_id", 1)], unique=True)
    _sf_a.create_index([("name", 1)])
    _sf_a.create_index([("owner", 1)])
    _sf_o.create_index([("opportunity_id", 1)], unique=True)
    _sf_o.create_index([("stage", 1), ("_seq", -1)])
    _sf_o.create_index([("account_name", 1), ("stage", 1)])
    _sf_o.create_index([("owner", 1), ("stage", 1)])
    _emails.create_index([("embed_id", 1)], unique=True)

    doc = mem._db["sim_config"].find_one({"_id": "inbound_email_sources"})
    if not doc:
        return

    if (
        not CONFIG["crm"]["salesforce"]["enabled"]
        or not CONFIG["crm"]["salesforce"]["seed_accounts"]
    ):
        logger.info("[genesis] Salesforce not enabled, continuing...")
        return

    db_contacts = mem._db["sim_config"].find_one(
        {"_id": "inbound_email_sources"}, {"_id": 0}
    )

    contacts = db_contacts["sources"] if db_contacts else []

    logger.info(f"[genesis] Found {len(contacts)} contacts to process into CRM.")

    start_dt = datetime.strptime(CONFIG["simulation"]["start_date"], "%Y-%m-%d")

    tier_config = {
        "Enterprise": (5001, 50000),
        "Mid-Market": (101, 5000),
        "SMB": (1, 100),
        "Unknown": (1, 500),
    }

    for contact in contacts:
        if contact.get("category", "").lower() != "customer":
            continue

        org_name = contact.get("org", "Unknown")
        safe_id = org_name.upper().replace(" ", "").replace("-", "")
        account_id = f"ACC-{safe_id}"

        if mem._db["sf_accounts"].find_one({"account_id": account_id}):
            continue

        days_ago = random.randint(30, 730)
        hours_ago = random.randint(0, 23)
        mins_ago = random.randint(0, 59)
        created_dt = start_dt - timedelta(
            days=days_ago, hours=hours_ago, minutes=mins_ago
        )

        delta_seconds = int((start_dt - created_dt).total_seconds())
        last_activity_dt = created_dt + timedelta(
            seconds=random.randint(0, delta_seconds)
        )

        default_renewal = (created_dt + timedelta(days=365)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

        category = contact.get("category", "customer").capitalize()
        tier = contact.get("tier", "Unknown") if category == "Customer" else "Unknown"
        emp_range = tier_config.get(tier, tier_config["Unknown"])

        sentiment = contact.get("sentiment_baseline", 0.8)
        is_risky = True if sentiment < 0.5 else False

        account = {
            "account_id": account_id,
            "name": org_name,
            "type": category,
            "primary_contact_name": f"{contact.get('first_name', 'First Name')} {contact.get('last_name', 'Last Name')}",
            "primary_contact_email": contact.get("email", ""),
            "contact_role": contact.get("contact_role", "Unknown"),
            "owner": contact.get("internal_liaison", "Unassigned"),
            "industry": contact.get("industry", "Technology"),
            "tier": tier if tier != "Unknown" else None,
            "employee_count": random.randint(*emp_range),
            "website": f"https://www.{org_name.lower().replace(' ', '')}.com",
            "billing_region": contact.get("billing_region", "NA"),
            "arr": contact.get("arr", 0),
            "is_lighthouse": contact.get("is_lighthouse", False),
            "expansion_potential": contact.get("expansion_potential", 0),
            "status": "Active",
            "sentiment_baseline": sentiment,
            "risk_flag": is_risky,
            "contract_renewal_date": contact.get(
                "contract_renewal_date", default_renewal
            ),
            "last_activity_date": last_activity_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "created_at": created_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

        account = {k: v for k, v in account.items() if v is not None}

        mem._db["sf_accounts"].insert_one({**account, "_seq": 0})

        path = Path(BASE) / f"salesforce/accounts/{account_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(account, fh, indent=2)

        logger.info(
            f"[crm] SF account seeded: {account_id} ({org_name}) | Type: {category} | Risk: {is_risky}"
        )

    pass


def _domain_key(domain: str) -> str:
    """Normalise a domain name to a stable MongoDB key."""
    return domain.lower().replace(" ", "_")


def _build_system_tags(domain: str, knew_about: List[str]) -> List[str]:
    """
    Derive search-friendly system tags from a domain name and the full
    knew_about list so ticket tagging has multiple match surfaces.

    e.g. "TitanDB" → ["titandb", "titan", "db"]
         "legacy auth service" → ["legacy auth service", "legacy auth", "auth service", "auth"]
    """
    tags = set()
    raw = domain.lower()
    tags.add(raw)

    for token in re.split(r"[\s_\-]+", raw):
        if len(token) >= 3:
            tags.add(token)

    for sibling in knew_about:
        for token in re.split(r"[\s_\-]+", sibling.lower()):
            if len(token) >= 3 and token in raw:
                tags.add(token)

    return sorted(tags)


def seed_knowledge_gaps(mem: Memory):
    """
    Embeds skills, logs departure events, and seeds the DomainRegistry for
    every pre-simulation employee defined under knowledge_gaps in config.

    DomainRegistry documents live in mem._db["domain_registry"] with _id
    equal to the normalised domain key.  Each document schema:

        {
            "_id":                  "titandb",          # normalised key
            "domain":               "TitanDB",          # display name
            "primary_owner":        None,               # None = orphaned
            "former_owner":         "Bill",
            "documentation_coverage": 0.20,
            "last_updated_day":     -180,               # day relative to sim start
            "known_by":             [],                 # engineers with partial knowledge
            "system_tags":          ["titandb", "titan", "db"],
            "dept":                 "Engineering_Backend",
            "is_genesis_gap":       True,
        }

    Callers (DepartmentPlanner, PR reviewer) should call
    mem.get_domain_registry() to receive the full registry dict keyed by
    domain display name.
    """
    if not CONFIG.get("knowledge_gaps"):
        return

    logger.info("[genesis] Seeding pre-simulation knowledge gaps...")

    sim_start = datetime.strptime(CONFIG["simulation"]["start_date"], "%Y-%m-%d")

    mem._db["domain_registry"].create_index([("system_tags", 1)])
    mem._db["domain_registry"].create_index([("primary_owner", 1)])

    for gap in CONFIG.get("knowledge_gaps", []):
        name = gap["name"]
        left_date = gap["left"]
        left_dt = datetime.strptime(left_date, "%Y-%m")
        departure_day = -(sim_start - left_dt).days
        dept = gap.get("dept", "Engineering")
        role = gap.get("role", "Former Employee")
        knew_about = gap.get("knew_about", [])
        doc_pct = gap.get("documented_pct", 0.5)

        mem.embed_persona_skills(
            name=name,
            data={
                "expertise": knew_about,
                "social_role": role,
            },
            dept=dept,
            day=departure_day,
            timestamp_iso=f"{left_date}-01T09:00:00",
        )

        mem.log_event(
            SimEvent(
                type="employee_departed",
                day=departure_day,
                date=f"{left_date}-01",
                timestamp=f"{left_date}-01T09:00:00",
                actors=[name],
                artifact_ids={},
                facts={
                    "name": name,
                    "role": role,
                    "knowledge_domains": knew_about,
                    "documented_pct": doc_pct,
                    "is_genesis_gap": True,
                },
                summary=f"Genesis Gap: {name} ({role}) left Day {departure_day}.",
                tags=["employee_departed", "lifecycle", "genesis"],
            )
        )

        for domain in knew_about:
            key = _domain_key(domain)
            system_tags = _build_system_tags(domain, knew_about)

            existing = mem._db["domain_registry"].find_one({"_id": key})
            if existing:
                # Domain already registered (e.g. two departures knew the
                # same system).  Just ensure former_owners is a list and
                # append — don't overwrite coverage.
                mem._db["domain_registry"].update_one(
                    {"_id": key},
                    {
                        "$addToSet": {
                            "former_owners": name,
                            "system_tags": {"$each": system_tags},
                        }
                    },
                )
                logger.info(
                    f"    [dim]→ Domain '{domain}' already registered — "
                    f"appended {name} as former owner.[/dim]"
                )
                continue

            record = {
                "_id": key,
                "domain": domain,
                "primary_owner": None,  # orphaned from day 0
                "former_owner": name,
                "former_owners": [name],
                "documentation_coverage": doc_pct,
                "last_updated_day": departure_day,
                "known_by": [],  # no current engineers
                "system_tags": system_tags,
                "dept": dept,
                "is_genesis_gap": True,
            }

            mem._db["domain_registry"].insert_one(record)

            logger.info(
                f"    [dim]→ Domain registry: '{domain}' "
                f"(owner=None, coverage={int(doc_pct * 100)}%, "
                f"tags={system_tags})[/dim]"
            )

        mem.log_event(
            SimEvent(
                type="knowledge_gap_detected",
                day=departure_day,
                date=f"{left_date}-01",
                timestamp=f"{left_date}-01T09:00:00",
                actors=[name],
                artifact_ids={},
                facts={
                    "detection_method": "genesis_seed",
                    "former_owner": name,
                    "role": role,
                    "dept": dept,
                    "domains": knew_about,
                    "documentation_coverage": doc_pct,
                    "gap_classification": "likely",
                    "author_domain_fit": "low",
                    "topics_beyond_author_expertise": knew_about,
                    "is_genesis_gap": True,
                },
                summary=(
                    f"Genesis gap seeded: {name} ({role}) departed with sole "
                    f"ownership of {knew_about}. "
                    f"Documentation coverage: {int(doc_pct * 100)}%."
                ),
                tags=["knowledge_gap", "genesis", "orphaned_domain"],
            )
        )

        logger.info(
            f"    [dim]→ Seeded Genesis Gap: {name} | domains: {knew_about}[/dim]"
        )


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
