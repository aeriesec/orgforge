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


# genesis.py — replace seed_external_sources entirely


def seed_external_sources(mem: Memory, planner_llm):
    if mem.get_inbound_email_sources():
        return

    logger.info("[cyan]🌐 Generating inbound email sources...[/cyan]")
    tech_stack = mem.tech_stack_for_prompt()

    vendors = _generate_vendor_sources(mem, planner_llm, tech_stack)
    customers = _generate_customer_sources(mem, planner_llm, tech_stack)

    sources = vendors + customers
    if len(sources) < 10:
        raise SystemExit("[genesis] ❌ Too few sources generated. Aborting.")

    mem.save_inbound_email_sources(sources)
    logger.info(
        f"[genesis] ✅ Seeded {len(sources)} sources ({len(vendors)}V + {len(customers)}C)."
    )
    for s in sources:
        logger.info(
            f"    [dim]→ [{s['category']}] {s['name']} "
            f"({s['internal_liaison']}) triggers={s['trigger_on']}[/dim]"
        )


def _generate_vendor_sources(mem: Memory, planner_llm, tech_stack: str) -> List[dict]:
    dept_str = ", ".join(LEADS.keys())
    all_names = [name for members in ORG_CHART.values() for name in members]

    agent = make_agent(
        role="Enterprise IT Architect",
        goal=f"Design the vendor email ecosystem for {COMPANY_NAME}.",
        backstory=(
            f"You map communication patterns between a {INDUSTRY} company "
            f"and its technology vendors."
        ),
        llm=planner_llm,
    )
    task = Task(
        description=(
            f"Generate exactly 7 vendor email sources for {COMPANY_NAME}, "
            f"a {INDUSTRY} company that {COMPANY_DESCRIPTION}.\n\n"
            f"TECH STACK (use ONLY vendors that appear here):\n{tech_stack}\n\n"
            f"DEPARTMENTS: {dept_str}\n\n"
            f"LIAISON RULES — assign internal_liaison based on what the vendor provides:\n"
            f"  - Infrastructure, cloud, hosting, databases, source control, monitoring → Engineering_Backend\n"
            f"  - Mobile platform tools, SDKs → Engineering_Mobile\n"
            f"  - Project management tools (Jira, etc.) → Product\n"
            f"  - CI/CD, testing tools → QA_Support\n"
            f"  - Payment processing, billing → Sales_Marketing\n"
            f"  - Legal, compliance, payroll → HR_Ops\n\n"
            f"NO DUPLICATE NAMES with: {all_names}\n\n"
            f"Each vendor must include:\n"
            f"  - name, org, first_name, last_name, email\n"
            f'  - category: exactly "vendor"\n'
            f"  - internal_liaison: one of [{dept_str}] per rules above\n"
            f"  - contact_role, persona_archetype\n"
            f"  - persona: {{typing_quirks, social_role, expertise[]}}\n"
            f"  - trigger_on: array of 'always', 'incident', 'low_health'\n"
            f"  - trigger_health_threshold: int 85-98 for infra, 70-85 for standard\n"
            f"  - tone: formal | technical | urgent\n"
            f"  - topics: 3-5 specific to what this vendor provides\n"
            f"  - integration_complexity: Low | Med | High\n"
            f"  - version_in_use: e.g. 'Enterprise Cloud', 'v2 Beta'\n"
            f"  - expected_sla_hours, cadence, timezone_offset\n"
            f"  - sentiment_baseline: float 0.0-1.0\n"
            f"  - history_summary: 1 short sentence\n\n"
            f"Raw JSON array only — no preamble, no markdown fences."
        ),
        expected_output="Raw JSON array of 7 vendor objects.",
        agent=agent,
    )

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            raw = str(Crew(agents=[agent], tasks=[task]).kickoff()).strip()
            sources = _parse_sources(raw)
            vendors = [s for s in sources if s.get("category") == "vendor"]
            if len(vendors) >= 5:
                return vendors
            raise ValueError(f"Only {len(vendors)} vendors parsed")
        except Exception as e:
            logger.warning(f"[genesis] Vendor attempt {attempt} failed: {e}")
            if attempt == _MAX_RETRIES:
                raise SystemExit("[genesis] ❌ Vendor generation failed.")
    return []


def _generate_customer_sources(mem: Memory, planner_llm, tech_stack: str) -> List[dict]:
    all_names = [name for members in ORG_CHART.values() for name in members]

    agent = make_agent(
        role="VP of Customer Success",
        goal=f"Design the customer ecosystem for {COMPANY_NAME}.",
        backstory=(
            f"You understand how {INDUSTRY} customers use "
            f"{COMPANY_NAME}'s platform and what they depend on."
        ),
        llm=planner_llm,
    )
    task = Task(
        description=(
            f"Generate exactly 8 customer email sources for {COMPANY_NAME}, "
            f"a {INDUSTRY} company that {COMPANY_DESCRIPTION}.\n\n"
            f"TECH STACK (for depends_on_components only — customers never see this):\n{tech_stack}\n\n"
            f"INTERNAL_LIAISON: For ALL customers, set to 'Sales_Marketing'. No exceptions.\n\n"
            f"NO DUPLICATE NAMES with: {all_names}\n\n"
            f"Each customer must include:\n"
            f"  - name (a realistic human name), org, first_name, last_name, email\n"
            f'  - category: exactly "customer"\n'
            f'  - internal_liaison: "Sales_Marketing"\n'
            f"  - contact_role, persona_archetype (The Champion, The Skeptic, The Bureaucrat, etc.)\n"
            f"  - persona: {{typing_quirks, social_role, expertise[]}}\n"
            f"  - trigger_on: array of 'always', 'incident', 'low_health'\n"
            f"  - trigger_health_threshold: int (Enterprise=88-98, Mid-Market=80-90, SMB=70-85)\n"
            f"  - tone: formal | friendly | frustrated | urgent\n"
            f"  - topics: 3-5 hyper-specific to what THIS customer uses the platform for — "
            f"written from their perspective, no internal tech names\n"
            f"  - industry, tier (Enterprise|Mid-Market|SMB), billing_region (NA|EMEA|APAC), "
            f"billing_city, billing_state, billing_country, arr\n"
            f"  - is_lighthouse (bool), expansion_potential (1-10), contract_renewal_date (ISO)\n"
            f"  - expected_sla_hours, cadence, timezone_offset\n"
            f"  - sentiment_baseline: float 0.0-1.0\n"
            f"  - history_summary: 1 short sentence\n\n"
            f"  - DEPENDS_ON_COMPONENTS: Array of 2-4 exact technology/component names "
            f"extracted from the TECH STACK above that power what this customer uses. "
            f"Use the specific product names as they appear in the stack "
            f"(e.g., 'Kafka', 'PostgreSQL', 'TitanDB', 'React Native', 'Redis'). "
            f"NOT category keys like 'database' or 'infra'. "
            f"A cycling team relying on live data might depend on ['Kafka', 'TitanDB', 'React Native']. "
            f"A clinic using historical reports might depend on ['PostgreSQL', 'S3']. "
            f"These MUST match real names from the tech stack.\n\n"
            f"  - AFFECTED_BY: Array of 2-4 capability strings describing end-user outcomes "
            f"this customer depends on. NOT internal tech names. "
            f"e.g., ['real-time athlete metrics', 'GPS tracking sync', 'historical performance reports']\n\n"
            f"  - SYMPTOM_LANGUAGE: 1-2 sentences in the customer's own voice describing "
            f"how an outage would affect THEM. Reflects their industry, persona_archetype, and tone. "
            f"NEVER mention internal system names.\n\n"
            f"Ensure diversity: mix tiers, regions, industries, and sentiment levels. "
            f"At least 2 should have sentiment_baseline < 0.6.\n\n"
            f"Raw JSON array only — no preamble, no markdown fences."
        ),
        expected_output="Raw JSON array of 8 customer objects.",
        agent=agent,
    )

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            raw = str(Crew(agents=[agent], tasks=[task]).kickoff()).strip()
            sources = _parse_sources(raw)
            customers = [s for s in sources if s.get("category") == "customer"]
            if len(customers) >= 6:
                return customers
            raise ValueError(f"Only {len(customers)} customers parsed")
        except Exception as e:
            logger.warning(f"[genesis] Customer attempt {attempt} failed: {e}")
            if attempt == _MAX_RETRIES:
                raise SystemExit("[genesis] ❌ Customer generation failed.")
    return []


def seed_tech_stack(mem: Memory, planner_llm):
    """Generates the tech stack ground truth."""
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

    mem._db["artifacts"].create_index(
        [("title", "text"), ("content", "text")],
        name="artifacts_text_search",
        weights={"title": 3, "content": 1},
    )

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

        liaison_dept = contact.get("internal_liaison", "Unassigned")
        liaison_person = LEADS.get(liaison_dept, liaison_dept)

        account = {
            "account_id": account_id,
            "name": org_name,
            "type": category,
            "primary_contact_name": f"{contact.get('first_name', 'First Name')} {contact.get('last_name', 'Last Name')}",
            "primary_contact_email": contact.get("email", ""),
            "contact_role": contact.get("contact_role", "Unknown"),
            "owner": liaison_person,
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
        valid = []
        for s in parsed:
            if not required.issubset(s.keys()):
                continue
            # Force-correct customer liaison at parse time
            if s.get("category", "").lower() == "customer":
                s["internal_liaison"] = "Sales_Marketing"
            valid.append(s)
        return valid
    except Exception as exc:
        logger.warning(f"[external_email] Source parse failed: {exc}")
        return []
