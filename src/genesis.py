# genesis.py
from datetime import datetime, timedelta
import json
import logging
import random
import re
from typing import Dict, List
from config_loader import (
    BASE,
    COMPANY_DESCRIPTION,
    COMPANY_NAME,
    CONFIG,
    INDUSTRY,
    LEADS,
    LEGACY,
)
from memory import Memory
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
    logger.info("[genesis] ✅ Seeding complete.")

    return mem


def seed_external_sources(mem: Memory, planner_llm):
    """Generates the 15 external vendors/customers and saves to MongoDB."""
    if mem.get_inbound_email_sources():
        return

    logger.info("[cyan]🌐 Generating inbound email sources...[/cyan]")

    tech_stack = mem.tech_stack_for_prompt()
    dept_str = ", ".join(LEADS.keys())
    db_accounts = list(mem._db["sf_accounts"].find({"type": "Customer"}, {"name": 1}))

    accounts = [doc["name"] for doc in db_accounts]
    random.shuffle(accounts)

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
            f"  - ADHERENCE: Use ONLY vendors that appear in the TECH STACK above. If Jira is listed, never use Trello.\n"
            f"  - CUSTOMERS: All category:'customer' entries must be from the KNOWN CUSTOMERS list.\n"
            f"  - FIRMOGRAPHICS (Customers ONLY): For customer entries, include 'industry' (e.g. Financial Services, Healthcare), 'tier' (Enterprise, Mid-Market, or SMB), 'billing_region' (NA, EMEA, APAC), and 'arr' (realistic numeric annual revenue like 50000, 120000, 350000).\n"
            f"  - HEALTH SENSITIVITY: Include 'trigger_health_threshold' (int 0-100). Scale: Infrastructure/Enterprise (85-98), SMB/Standard Vendors (70-85).\n"  # New Rule
            f"  - TOPICS: Provide 3-5 hyper-specific topics (e.g., 'GitHub Actions Runner Timeout' or 'Stripe API 402 Payment Required').\n"
            f"  - CATEGORY: exactly 'vendor' or 'customer'.\n"
            f"  - TRIGGER_ON: array of 'always', 'incident', 'low_health'.\n"
            f"  - TONE: formal | technical | frustrated | urgent | friendly.\n\n"
            f"Raw JSON array only — no preamble, no markdown fences:\n"
            f"[\n"
            f'  {{"name":"GitHub","org":"GitHub Inc.","email":"support@github.com",'
            f'"category":"vendor","internal_liaison":"Engineering_Backend",'
            f'"trigger_on":["incident", "low_health"],"trigger_health_threshold":95,"tone":"technical",'
            f'"topics":["Webhooks failing with 5xx","Pull Request comment API latency"]}},\n'
            f'  {{"name":"GlobalFinance","org":"GlobalFinance Corp","email":"cto@globalfinance.com",'
            f'"category":"customer","internal_liaison":"Sales_Marketing",'
            f'"trigger_on":["always","incident"],"trigger_health_threshold":90,"tone":"formal",'
            f'"topics":["SLA reporting","Contract renewal"],"industry":"Financial Services",'
            f'"tier":"Enterprise","billing_region":"NA","arr":250000}}\n'
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
    doc = mem._db["sim_config"].find_one({"_id": "inbound_email_sources"})
    if not doc:
        return

    if (
        not CONFIG["crm"]["salesforce"]["enabled"]
        or not CONFIG["crm"]["salesforce"]["seed_accounts"]
    ):
        return

    contacts = list(
        mem._db["sim_config"].find(
            {"_id": "inbound_email_sources", "category": "customer"}, {"_id": 0}
        )
    )

    start_dt = datetime.strptime(CONFIG["simulation"]["start_date"], "%Y-%m-%d")

    for contact in contacts:
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
            "arr": contact.get("arr", random.choice([50000, 100000, 250000, 500000])),
            "owner": contact.get("internal_liaison", "Unassigned"),
            "created_at": created_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "risk_flag": False,
        }
        mem._db["sf_accounts"].insert_one({**account, "_seq": 0})

        path = BASE / "salesforce/accounts/{account_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(account, fh, indent=2)

        logger.info(f"[crm] SF account seeded: {account_id} ({org_name})")

    pass


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
