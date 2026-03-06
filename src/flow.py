"""
flow.py (MongoDB + YAML + NetworkX Edition)
=================================================
OrgForge simulation engine. Reads from config.yaml.
Uses NetworkX for social graphs. Uses MongoDB for vector/artifact storage.
"""

import os
import logging
import json
import random
import re
from pathlib import Path
import yaml
import networkx as nx
from datetime import datetime, timedelta
from typing import Any, List, Dict, Optional
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dataclasses import field

from day_planner import DayPlannerOrchestrator
from normal_day import NormalDayHandler
from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.logging import RichHandler
from rich import box
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from crewai import Agent, Task, Crew
from crewai.flow.flow import Flow, listen, start
from langchain_ollama import OllamaLLM

from memory import Memory, SimEvent
from graph_dynamics import GraphDynamics
from sim_clock import SimClock
from org_lifecycle import (
         OrgLifecycleManager,
         patch_validator_for_lifecycle,
         recompute_escalation_after_departure,
     )

os.makedirs("./export", exist_ok=True)

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
EXPORT_DIR = PROJECT_ROOT / "export"

EXPORT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        # Writes all logs to a permanent file
        logging.FileHandler(EXPORT_DIR / "simulation.log", mode='a'),
        RichHandler(rich_tracebacks=True, show_time=False, show_path=False)
    ]
)

logger = logging.getLogger("orgforge.flow")

# ─────────────────────────────────────────────
# 1. LOAD CONFIG
# ─────────────────────────────────────────────
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

COMPANY_NAME    = CONFIG["simulation"]["company_name"]
COMPANY_DOMAIN  = CONFIG["simulation"]["domain"]
INDUSTRY        = CONFIG["simulation"].get("industry", "technology")
BASE            = CONFIG["simulation"].get("output_dir", str(EXPORT_DIR))
ORG_CHART       = CONFIG["org_chart"]
LEADS           = CONFIG["leads"]
PERSONAS        = CONFIG["personas"]
DEFAULT_PERSONA = CONFIG["default_persona"]
LEGACY          = CONFIG["legacy_system"]
PRODUCT_PAGE    = CONFIG.get("product_page", "Product Launch")

DEPARTED_EMPLOYEES: Dict[str, Dict] = {
    gap["name"]: {
        "left":           gap["left"],
        "role":           gap["role"],
        "knew_about":     gap["knew_about"],
        "documented_pct": gap["documented_pct"],
    }
    for gap in CONFIG.get("knowledge_gaps", [])
}

ALL_NAMES = [name for dept in ORG_CHART.values() for name in dept]
LIVE_ORG_CHART  = {dept: list(members) for dept, members in ORG_CHART.items()}
LIVE_PERSONAS   = {k: dict(v) for k, v in PERSONAS.items()}

# ── Active preset ─────────────────────────────
_PRESET_NAME = CONFIG.get("quality_preset", "local_cpu")
_PRESET      = CONFIG["model_presets"][_PRESET_NAME]
_PROVIDER    = _PRESET.get("provider", "ollama")

def _bare_model(model_str: str) -> str:
    return model_str.strip()

def build_llm(model_key: str):
    """
    Return the correct LangChain LLM for the active quality_preset.

    preset provider values:
      "ollama"  → langchain_community.llms.Ollama         (local_cpu / local_gpu)
      "bedrock" → langchain_aws.ChatBedrock                (cloud — AWS Bedrock)

    model_key: "planner" or "worker"
    """
    model_str = _PRESET[model_key]
    model     = _bare_model(model_str)

    if _PROVIDER == "bedrock":
        try:
            from crewai import LLM
            region = _PRESET.get("aws_region", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
            llm = LLM(
                model=model,
                region_name=region,
                max_tokens=4096,
                temperature=0.7,
            )
            logger.info(f"[config] {model_key} → Bedrock/{model} (region={region})")
            return llm
        except ImportError:
            raise ImportError(
                "langchain-aws is required for the cloud preset. "
                "Run: pip install langchain-aws"
            )

    # Default: Ollama (local_cpu / local_gpu)
    
    # 1. Check environment variable first (injected by Docker)
    # 2. Fall back to config.yaml if no env var exists
    # 3. Fall back to localhost if neither exists
    env_base_url = os.environ.get("OLLAMA_BASE_URL")
    config_base_url = _PRESET.get("base_url", "http://localhost:11434")
    
    base_url = env_base_url if env_base_url else config_base_url
    
    logger.info(f"[config] {model_key} → Ollama/{model} ({base_url})")
    return OllamaLLM(model=model, base_url=base_url, timeout=1200)

PLANNER_MODEL = build_llm("planner")
WORKER_MODEL  = build_llm("worker")

# Propagate embedding config to memory via environment
os.environ.setdefault("EMBED_PROVIDER", _PRESET.get("embed_provider", "ollama"))
os.environ.setdefault("EMBED_MODEL",    _PRESET.get("embed_model", "mxbai-embed-large"))
os.environ.setdefault("EMBED_DIMS",     str(_PRESET.get("embed_dims", 1024)))
os.environ.setdefault("DB_NAME",        CONFIG["simulation"].get("db_name", "orgforge"))
if _PROVIDER == "bedrock":
    os.environ.setdefault("AWS_DEFAULT_REGION", _PRESET.get("aws_region", "us-east-1"))

def resolve_role(role_key: str) -> str:
    """Resolve a logical role (e.g. 'on_call_engineer') to a person's name via config."""
    dept = CONFIG.get("roles", {}).get(role_key)
    if dept and dept in LEADS:
        return LEADS[dept]
    # Fallback: first lead alphabetically
    return next(iter(LEADS.values()))

def render_template(template: str) -> str:
    """Replace {legacy_system}, {project_name}, {product_page} placeholders in config strings."""
    return (template
        .replace("{legacy_system}",  LEGACY["name"])
        .replace("{project_name}",   LEGACY["project_name"])
        .replace("{product_page}",   PRODUCT_PAGE)
        .replace("{company_name}",   COMPANY_NAME)
        .replace("{industry}",       INDUSTRY)
    )

console = Console()
vader   = SentimentIntensityAnalyzer()

def dept_of(name: str) -> str:
    for dept, members in ORG_CHART.items():
        if name in members:
            return dept
    return "Unknown"

def email_of(name: str) -> str:
    return f"{name.lower()}@{COMPANY_DOMAIN}"

def build_social_graph() -> nx.Graph:
    """Builds a weighted social graph of employees and external contacts."""
    G = nx.Graph()

    # Internal nodes (unchanged)
    for dept, members in ORG_CHART.items():
        for member in members:
            G.add_node(member,
                dept=dept,
                is_lead=(member in LEADS.values()),
                external=False,
            )

    for n1 in G.nodes():
        for n2 in G.nodes():
            if n1 >= n2: continue
            weight = 0.5
            if G.nodes[n1]['dept'] == G.nodes[n2]['dept']:
                weight += 10.0
            if G.nodes[n1]['is_lead'] and G.nodes[n2]['is_lead']:
                weight += 5.0
            G.add_edge(n1, n2, weight=weight)

    # External nodes — cold edges to their liaison department
    for contact in CONFIG.get("external_contacts", []):
        node_id = contact["name"]
        liaison_dept = contact.get("internal_liaison", list(LEADS.keys())[0])
        liaison_lead = LEADS.get(liaison_dept, next(iter(LEADS.values())))

        G.add_node(node_id,
            dept="External",
            org=contact.get("org", "External"),
            role=contact.get("role", "Contact"),
            display_name=contact.get("display_name", node_id),
            is_lead=False,
            external=True,
        )

        # Cold starting edge to liaison lead only — warms up via incidents
        G.add_edge(node_id, liaison_lead, weight=0.5)

    return G

# ─────────────────────────────────────────────
# 2. STATE
# ─────────────────────────────────────────────
class ActiveIncident(BaseModel):
    ticket_id: str
    title: str
    day_started: int
    stage: str = "detected"
    days_active: int = 0
    involves_gap_knowledge: bool = False
    pr_id: Optional[str] = None
    root_cause: str = ""

class SprintState(BaseModel):
    sprint_number: int = 1
    start_day: int = 1
    tickets_in_sprint: List[str] = []
    velocity: int = 0

class State(BaseModel):
    day: int = 1
    max_days: int = Field(default_factory=lambda: CONFIG["simulation"]["max_days"])
    current_date: datetime = Field(default_factory=lambda: datetime.strptime(CONFIG["simulation"]["start_date"], "%Y-%m-%d"))
    system_health: int = 100
    team_morale: float = Field(default_factory=lambda: CONFIG["morale"]["initial"])
    morale_history: List[float] = []
    is_researching: bool = False
    confluence_pages: List[Dict] = []
    jira_tickets: List[Dict] = []
    slack_threads: List[Dict] = []
    pr_registry: List[Dict] = []
    active_incidents: List[ActiveIncident] = []
    resolved_incidents: List[str] = []
    sprint: SprintState = Field(default_factory=SprintState)
    daily_theme: str = ""
    persona_stress: Dict[str, int] = {}
    actor_cursors: Dict[str, Any] = Field(default_factory=dict)

    # Daily counters — reset each morning, read at end of day
    daily_incidents_opened:   int = 0
    daily_incidents_resolved: int = 0
    daily_artifacts_created:  int = 0
    daily_external_contacts:  int = 0

    org_day_plan: Optional[Any] = None
    daily_active_actors: List[str] = []
    daily_event_type_counts: Dict[str, int] = {}
    departed_employees: Dict[str, Dict] = {}   # name → {left, role, knew_about, documented_pct}
    new_hires: Dict[str, Dict] = {}   # name → {joined, role, dept, expertise}
    ticket_actors_today: Dict[str, List[str]] = field(default_factory=dict)


# ─────────────────────────────────────────────
# 3. FILE I/O
# ─────────────────────────────────────────────
def _mkdir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def save_json(path: str, data):
    _mkdir(path)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def save_md(path: str, content: str):
    _mkdir(path)
    with open(path, "w") as f:
        f.write(content)

def append_log(path: str, line: str):
    _mkdir(path)
    with open(path, "a") as f:
        f.write(line + "\n")

def save_eml(path: str, from_name: str, to_names: List[str], subject: str, body: str, cc_names: Optional[List[str]] = None, in_reply_to: Optional[str] = None, date_str: Optional[str] = None):
    msg = MIMEMultipart("alternative")
    msg["From"] = f"{from_name} <{email_of(from_name)}>"
    msg["To"] = ", ".join(f"{n} <{email_of(n)}>" for n in to_names)
    msg["Subject"] = subject
    msg["Date"] = date_str or datetime.now().strftime("%a, %d %b %Y %H:%M:%S +0000")
    msg["Message-ID"] = f"<{random.randint(10000,99999)}@{COMPANY_DOMAIN}>"
    if cc_names: msg["Cc"] = ", ".join(f"{n} <{email_of(n)}>" for n in cc_names)
    if in_reply_to:
        msg["In-Reply-To"] = in_reply_to
        msg["References"] = in_reply_to
    msg.attach(MIMEText(body, "plain"))
    _mkdir(path)
    with open(path, "w") as f:
        f.write(msg.as_string())

# ─────────────────────────────────────────────
# 4. GIT SIMULATOR (NetworkX Aware)
# ─────────────────────────────────────────────
class GitSimulator:
    def __init__(self, state: State, mem: Memory, social_graph: nx.Graph, worker_llm):
        self._state = state
        self._mem = mem
        self._graph = social_graph
        self._worker_llm = worker_llm

    def create_pr(self, author: str, ticket_id: str, title: str, timestamp: str, reviewers: Optional[List[str]] = None) -> Dict:
        pr_id = f"PR-{len(self._state.pr_registry) + 100}"
        
        if not reviewers:
            edges = self._graph[author]
            
            eng_colleagues = {
                n: edges[n].get('weight', 1.0) 
                for n in edges 
                if "engineer" in self._graph.nodes[n].get('dept', '').lower() and n != author
            }
            
            if eng_colleagues:
                sorted_eng = sorted(eng_colleagues.items(), key=lambda x: x[1], reverse=True)
                reviewers = [sorted_eng[0][0]]
                if len(sorted_eng) > 1:
                    reviewers.append(sorted_eng[1][0])
            else:
                eng_dept   = next((d for d in ORG_CHART if "engineer" in d.lower()), list(ORG_CHART.keys())[0])
                fallback   = next((n for n in ORG_CHART[eng_dept] if n != author), ORG_CHART[eng_dept][0])
                reviewers = [fallback]

        try:
            ctx = self._mem.context_for_prompt(title, n=2, as_of_time=timestamp)
            agent = Agent(role="Software Engineer", goal="Write a PR description.", backstory=f"You are {author}.", llm=self._worker_llm)
            task = Task(
                description=(
                    f"Write a GitHub Pull Request description for ticket [{ticket_id}]: {title}.\n"
                    f"Context: {ctx}\n"
                    f"Include a short 'What Changed' and 'Why' section. Keep it under 100 words. Format as Markdown."
                ),
                expected_output="Markdown PR body.", agent=agent
            )
            description = str(Crew(agents=[agent], tasks=[task], verbose=False).kickoff()).strip()
        except Exception as e:
            description = f"Auto-generated PR for [{ticket_id}]: {title}"

        pr = {
            "pr_id": pr_id, "ticket_id": ticket_id, "title": title,
            "description": description,
            "author": author, "author_email": email_of(author),
            "reviewers": reviewers, "status": "open", "comments": [],
            "created_at": timestamp,
        }
        
        path = f"{BASE}/git/prs/{pr_id}.json"
        save_json(path, pr)
        self._state.pr_registry.append({"pr_id": pr_id, "ticket_id": ticket_id, "author": author, "status": "open"})
        self._mem.embed_artifact(
            id=pr_id, type="pr", title=title, content=json.dumps(pr),
            day=self._state.day, date=str(self._state.current_date.date()),
            timestamp=timestamp,
            metadata={"author": author, "ticket_id": ticket_id},
        )
        self._state.daily_artifacts_created += 1
        return pr

    def merge_pr(self, pr_id: str):
        for pr in self._state.pr_registry:
            if pr["pr_id"] == pr_id:
                pr["status"] = "merged"
                path = f"{BASE}/git/prs/{pr_id}.json"
                if os.path.exists(path):
                    with open(path) as f:
                        data = json.load(f)
                    data["status"] = "merged"
                    save_json(path, data)
                break

# ─────────────────────────────────────────────
# 5. HELPERS
# ─────────────────────────────────────────────
def persona_backstory(name: str, mem: Optional[Memory] = None, extra: str = "", graph_dynamics=None) -> str:
    p = PERSONAS.get(name, DEFAULT_PERSONA)
    stress_label = graph_dynamics.stress_label(name) if graph_dynamics else (
        "low" if p["stress"] < 40 else ("moderate" if p["stress"] < 70 else "high")
    )
    tone_hint = graph_dynamics.stress_tone_hint(name) if graph_dynamics else ""
    history = ""
    if mem:
        past = mem.persona_history(name, n=3)
        if past:
            history = " Recent actions: " + " | ".join(f"Day {e.day}: {e.summary}" for e in past)
    return (
        f"You are {name}, working in {dept_of(name)} at {COMPANY_NAME}. "
        f"Style: {p['style']}. Stress: {stress_label}. "
        f"Expertise: {', '.join(p['expertise'])}. Tenure: {p['tenure']}."
        f"{history} {extra}"
    )

def next_conf_id(state: State, prefix: str = "ENG") -> str:
    n = len([p for p in state.confluence_pages if prefix in p["id"]]) + 1
    return f"CONF-{prefix}-{n:03d}"

def next_jira_id(state: State) -> str:
    return f"ORG-{len(state.jira_tickets) + 100}"

def bill_gap_warning(topic: str) -> str:
    """Scans all departed employees (not just Bill) for knowledge gap warnings."""
    for emp_name, emp in DEPARTED_EMPLOYEES.items():
        hits = [k for k in emp["knew_about"] if k.lower() in topic.lower()]
        if hits:
            return (
                f"\n\n> ⚠️ **Knowledge Gap**: This area ({', '.join(hits)}) was owned by "
                f"{emp_name} (ex-{emp['role']}, left {emp['left']}). "
                f"Only ~{int(emp['documented_pct']*100)}% documented."
            )
    return ""

def score_sentiment(messages: List[Dict]) -> float:
    if not messages: return 0.5
    scores = [vader.polarity_scores(m.get("text", m.get("body", "")))["compound"] for m in messages]
    return round((sum(scores) / len(scores) + 1) / 2, 3)

# ─────────────────────────────────────────────
# 6. THE FLOW
# ─────────────────────────────────────────────
class Flow(Flow[State]):

    def __init__(self):
        super().__init__()
        self._mem = Memory()
        self.graph_dynamics = GraphDynamics(build_social_graph(), CONFIG)
        self.social_graph   = self.graph_dynamics.G
        self._git = GitSimulator(self.state, self._mem, self.social_graph, WORKER_MODEL)
        self._day_planner = DayPlannerOrchestrator(CONFIG, WORKER_MODEL, PLANNER_MODEL)
        self._clock = SimClock(self.state)
        self._normal_day = NormalDayHandler(
            config=CONFIG, mem=self._mem, state=self.state,
            graph_dynamics=self.graph_dynamics, social_graph=self.social_graph,
            git=self._git, worker_llm=WORKER_MODEL, planner_llm=PLANNER_MODEL,
            clock=self._clock
        )
        self._lifecycle = OrgLifecycleManager(
            config=CONFIG,
            graph_dynamics=self.graph_dynamics,
            mem=self._mem,
            org_chart=LIVE_ORG_CHART,
            personas=LIVE_PERSONAS,
            all_names=ALL_NAMES,
            leads=LEADS,
            worker_llm=WORKER_MODEL,
        )

        stats = self._mem.stats()
        logger.info(f"[dim]Memory: provider={stats['embed_provider']} model={stats['embed_model']} dims={stats['embed_dims']} MongoDB={'✓' if stats['mongodb_ok'] else '⚠'}[/dim]")

    def _is_sprint_planning_day(self) -> bool:
        return self.state.current_date.weekday() == 0 and self.state.day % 10 == 1

    def _is_retro_day(self) -> bool:
        return self.state.current_date.weekday() == 4 and self.state.day % 10 == 9

    def _is_standup_day(self) -> bool:
        return self.state.current_date.weekday() in (0, 2, 4)

    def _embed_and_count(self, **kwargs):
        self._mem.embed_artifact(**kwargs)
        self.state.daily_artifacts_created += 1

    def _record_daily_actor(self, *names: str):
        """
        Call this whenever a named actor participates in any event today.
        Appends to daily_active_actors; dedup happens at EOD.

        Usage (sprinkle at existing event-firing sites):
            self._record_daily_actor(on_call, incident_lead)
            self._record_daily_actor(*attendees)
        """
        self.state.daily_active_actors.extend(names)


    def _record_daily_event(self, event_type: str):
        """
        Call this once per event fired during the day loop.
        Drives dominant_event and event_type_counts in the summary.

        Usage (one line added wherever a SimEvent is logged):
            self._record_daily_event("incident_opened")
            self._record_daily_event("standup")
        """
        counts = self.state.daily_event_type_counts
        counts[event_type] = counts.get(event_type, 0) + 1

    # ─── GENESIS ─────────────────────────────
    @start()
    def genesis_phase(self):
        logger.info(Panel.fit(
            f"[bold cyan]{COMPANY_NAME.upper()} — ORGFORGE SIMULATION[/bold cyan]\n"
            f"[dim]Preset: {_PRESET_NAME} | Provider: {_PROVIDER} | Seeding corporate archives...[/dim]",
            box=box.DOUBLE_EDGE,
        ))

        # Build historian backstory from config — no hardcoded company lore
        gap_summary = "; ".join(
            f"{n} (ex-{e['role']}, left {e['left']}) owned {', '.join(e['knew_about'][:2])}"
            for n, e in DEPARTED_EMPLOYEES.items()
        )
        historian = Agent(
            role="Corporate Historian",
            goal="Write authentic internal technical and business documents.",
            backstory=(
                f"You work at {COMPANY_NAME}, a {INDUSTRY} company. "
                f"The legacy system '{LEGACY['name']}' ({LEGACY['description']}) is known to be unstable. "
                + (f"Departed employees with knowledge gaps: {gap_summary}. " if gap_summary else "")
                + "Write with real detail and insider knowledge."
            ),
            llm=PLANNER_MODEL,
        )

        # Resolve departments dynamically — fall back gracefully if dept doesn't exist
        eng_dept   = next((d for d in ORG_CHART if "engineer" in d.lower() or "eng" in d.lower()), list(ORG_CHART.keys())[0])
        sales_dept = next((d for d in ORG_CHART if "sales" in d.lower() or "market" in d.lower()), list(ORG_CHART.keys())[-1])
        eng_members  = random.sample(ORG_CHART[eng_dept],   min(3, len(ORG_CHART[eng_dept])))
        sale_members = random.sample(ORG_CHART[sales_dept], min(2, len(ORG_CHART[sales_dept])))

        # Pull genesis doc config and render templates
        tech_cfg = CONFIG.get("genesis_docs", {}).get("technical", {})
        biz_cfg  = CONFIG.get("genesis_docs", {}).get("business",  {})
        tech_count  = tech_cfg.get("count", 3)
        tech_prefix = tech_cfg.get("id_prefix", "CONF-ENG")
        tech_ids    = [f"{tech_prefix}-{str(i+1).zfill(3)}" for i in range(tech_count)]

        tech_prompt = render_template(tech_cfg.get("prompt", (
            "Write {count} separate Confluence pages about {project_name} and {legacy_system}. "
            "Reference {engineers}. IDs: {ids}. Separate with '---PAGE BREAK---'."
        ))).replace("{count}", str(tech_count)).replace("{engineers}", str(eng_members)).replace("{ids}", ", ".join(tech_ids))

        biz_prompt = render_template(biz_cfg.get("prompt", (
            "Write a {product_page} Campaign Brief (CONF-MKT-001) and a Sales OKR doc (CONF-MKT-002). "
            "Reference {sales_members}. Separate with '---PAGE BREAK---'."
        ))).replace("{sales_members}", str(sale_members))

        tech_task = Task(description=tech_prompt, expected_output=f"{tech_count} full Markdown pages separated by '---PAGE BREAK---'.", agent=historian)
        biz_task  = Task(description=biz_prompt,  expected_output="Two full Markdown pages separated by '---PAGE BREAK---'.", agent=historian)

        crew   = Crew(agents=[historian], tasks=[tech_task, biz_task], verbose=False)
        result = str(crew.kickoff())

        from datetime import datetime, time
        genesis_time = datetime.combine(self.state.current_date.date(), time(8, 0)).isoformat()

        for raw in result.split("---PAGE BREAK---"):
            ids = re.findall(r"CONF-[A-Z]+-\d+", raw)
            if not ids: continue
            doc_id = ids[0]
            title_match = re.search(r"#\s+(.+)", raw)
            title = title_match.group(1).strip() if title_match else f"Archive: {doc_id}"
            path = f"{BASE}/confluence/archives/{doc_id}.md"
            content = raw + bill_gap_warning(raw)
            save_md(path, content)

            entry = {"id": doc_id, "title": title, "summary": title, "path": path}
            self.state.confluence_pages.append(entry)
            self._embed_and_count(
                id=doc_id, type="confluence", title=title, content=content,
                day=self.state.day, date=str(self.state.current_date.date()),
                metadata={"authors": str(eng_members)}, timestamp=genesis_time
            )

            
            self._mem.log_event(SimEvent(
                type="confluence_created", timestamp=genesis_time,
                day=self.state.day, date=str(self.state.current_date.date()),
                actors=eng_members, artifact_ids={"confluence": doc_id},
                facts={"title": title, "phase": "genesis"},
                summary=f"Archive page {doc_id} created: {title}", tags=["genesis", "confluence"],
            ))

        logger.info(f"[green]✓ Genesis complete.[/green] Memory: {self._mem.stats()['artifact_count']} artifacts embedded.\n")

    # ─── DAILY LOOP ───────────────────────────
    @listen(genesis_phase)
    def daily_cycle(self):
        while self.state.day <= self.state.max_days:
            dow = self.state.current_date.weekday()
            if dow >= 5:
                logger.info(f"[dim]  ↷ Weekend ({self.state.current_date.date()})[/dim]")
                self.state.current_date += timedelta(days=1)
                continue

            self._state.ticket_actors_today = {}
            self._clock.reset_to_business_start(ALL_NAMES)
            date_str = str(self.state.current_date.date())
            departures = self._lifecycle.process_departures(self.state.day, date_str, self.state, self._clock)
            hires      = self._lifecycle.process_hires(self.state.day, date_str, self.state, self._clock)

            if departures or hires:
                # Patch the day planner's validator to reflect the new roster
                patch_validator_for_lifecycle(self._day_planner._validator, self._lifecycle)

            org_plan = self._day_planner.plan(
                self.state, self._mem, self.graph_dynamics,
                lifecycle_context=self._lifecycle.get_roster_context(),
                clock=self._clock
            )
            self.state.daily_theme  = org_plan.org_theme
            self.state.org_day_plan = org_plan
            self._print_day_header()

            for inc in self.state.active_incidents: inc.days_active += 1

            if self._is_sprint_planning_day(): self._handle_sprint_planning()
            if self._is_standup_day(): self._handle_standup()
            if self._is_retro_day(): self._handle_retrospective()

            self._advance_incidents()

            theme_lower = self.state.daily_theme.lower()
            incident_triggers = CONFIG.get("incident_triggers", ["crash", "fail", "error", "latency", "timeout", "outage"])
            if any(x in theme_lower for x in incident_triggers):
                self._handle_incident()
            else:
                self._normal_day.handle(self.state.org_day_plan)
                if random.random() < CONFIG["simulation"].get("adhoc_confluence_prob", 0.3):
                    self._generate_adhoc_confluence_page()

            self._end_of_day()
            self.state.day += 1
            self.state.current_date += timedelta(days=1)

        self._print_final_report()

    # ─── SPRINT PLANNING ──────────────────────
    def _handle_sprint_planning(self):
        logger.info(f"  [bold blue]📋 Sprint #{self.state.sprint.sprint_number} Planning[/bold blue]")
        attendees = list(set(list(LEADS.values()) + random.sample(ALL_NAMES, 3)))
        raw_themes = CONFIG.get("sprint_ticket_themes", ["Refactor legacy system", "Add retry logic", "Fix errors", "QA regression"])
        ticket_themes = [render_template(t) for t in raw_themes]
        n_tickets = CONFIG["simulation"].get("sprint_tickets_per_planning", 4)

        meeting_time = self._clock.schedule_meeting(attendees, min_hour=9, max_hour=11)
        timestamp_str = meeting_time.isoformat()
        
        new_tickets = []
        for theme in random.sample(ticket_themes, min(n_tickets, len(ticket_themes))):
            tid = next_jira_id(self.state)
            assignee = random.choice(ALL_NAMES)
            pts = random.choice([1, 2, 3, 5, 8])
            ticket = {
                "id": tid, "title": theme, "status": "To Do", "assignee": assignee,
                "sprint": self.state.sprint.sprint_number, "story_points": pts, "linked_prs": [],
                "created_at": timestamp_str,
                "updated_at": timestamp_str
            }
            self.state.jira_tickets.append(ticket)
            self.state.sprint.tickets_in_sprint.append(tid)
            new_tickets.append(ticket)
            save_json(f"{BASE}/jira/{tid}.json", ticket)
            self._embed_and_count(id=tid, type="jira", title=theme, content=json.dumps(ticket), day=self.state.day, 
                                  date=str(self.state.current_date.date()), metadata={"assignee": assignee}, timestamp=timestamp_str)

        sprint_facts = {
            "sprint_number": self.state.sprint.sprint_number,
            "tickets": [{"id": t["id"], "title": t["title"], "assignee": t["assignee"], "points": t["story_points"]} for t in new_tickets],
            "total_points": sum(t["story_points"] for t in new_tickets),
            "sprint_goal": render_template(CONFIG.get("sprint_goal_template", "Stabilize {legacy_system} and deliver sprint features")),
        }

        self._mem.log_event(SimEvent(
            type="sprint_planned", day=self.state.day, date=str(self.state.current_date.date()), timestamp=timestamp_str,
            actors=attendees, artifact_ids={"jira_tickets": json.dumps([t["id"] for t in new_tickets])},
            facts=sprint_facts, summary=f"Sprint #{sprint_facts['sprint_number']} planned.", tags=["sprint", "planning"],
        ))

        self._record_daily_actor(*attendees)
        self._record_daily_event("sprint_planned")

        logger.info(f"    [green]✓[/green] {[t['id'] for t in new_tickets]}")

    # ─── STANDUP (LLM) ────────────────────────
    def _handle_standup(self):
        logger.info(f"  [bold blue]☕ Standup[/bold blue]")
        attendees = random.sample(ALL_NAMES, min(8, len(ALL_NAMES)))
        meeting_time = self._clock.schedule_meeting(attendees, min_hour=9, max_hour=10, duration_mins=15)
        meeting_time_iso = meeting_time.isoformat()
        ctx = self._mem.context_for_prompt(
            "current sprint active tickets and incidents", 
            n=4, 
            as_of_time=meeting_time_iso
        )
        
        profiles = []
        for name in attendees:
            p = PERSONAS.get(name, DEFAULT_PERSONA)
            past = self._mem.persona_history(name, n=1)
            recent = f" Recently did: {past[0].summary}" if past else ""
            profiles.append(f"- {name} ({dept_of(name)}, Expert in: {', '.join(p['expertise'])}).{recent}")

        standup_agent = Agent(role="Tech Lead", goal="Simulate a realistic daily standup thread.", backstory="You observe the daily standup.", llm=WORKER_MODEL)
        task = Task(
            description=f"Write a Slack standup update for:\n{chr(10).join(profiles)}\n\nContext:\n{ctx}\nFormat EXACTLY: Name: [Message]",
            expected_output="A transcript of standup updates.", agent=standup_agent
        )
        result = str(Crew(agents=[standup_agent], tasks=[task], verbose=False).kickoff())
        
        messages = []
        for line in result.split("\n"):
            if ":" in line:
                name, text = line.split(":", 1)
                name = name.strip()
                if name in ALL_NAMES:
                    messages.append({"user": name, "text": text.strip(), "ts": self.state.current_date.replace(hour=9, minute=30).isoformat()})

        if not messages: messages = [{"user": "System", "text": "Standup notes corrupted.", "ts": self.state.current_date.isoformat()}]
        date_str = str(self.state.current_date.date())
        slack_path = f"{BASE}/slack/channels/standup/{date_str}.json"
        save_json(slack_path, messages)
        self.state.slack_threads.append({"date": date_str, "channel": "standup", "message_count": len(messages)})
        self._mem.log_event(SimEvent(type="standup", timestamp=meeting_time_iso, day=self.state.day, date=date_str, actors=[m["user"] for m in messages], 
                                     artifact_ids={"slack": slack_path}, facts={"attendee_count": len(messages)}, 
                                     summary=f"Standup: {len(messages)} attendees shared updates.", tags=["standup"]))

        self._record_daily_actor(*[m["user"] for m in messages])
        self._record_daily_event("standup")

    # ─── RETROSPECTIVE ────────────────────────
    def _handle_retrospective(self):
        logger.info(f"  [bold blue]🔄 Retro — Sprint #{self.state.sprint.sprint_number}[/bold blue]")
        conf_id = next_conf_id(self.state, "RETRO")

        attendees = ALL_NAMES
        meeting_time = self._clock.schedule_meeting(attendees, min_hour=14, max_hour=16, duration_mins=60)
        meeting_time_iso = meeting_time.isoformat()
        ctx = self._mem.context_for_prompt(f"sprint {self.state.sprint.sprint_number} incidents velocity", n=4, as_of_time=meeting_time_iso)
        scrum_master = resolve_role("scrum_master")
        historian = Agent(role="Scrum Master", goal="Write retro.", backstory=persona_backstory(scrum_master, self._mem, graph_dynamics=self.graph_dynamics), llm=PLANNER_MODEL)
        task = Task(description=f"Write retro Confluence {conf_id} for Sprint #{self.state.sprint.sprint_number}.\nContext:\n{ctx}\nSections: What went well, What didn't, Action items.", expected_output="Markdown.", agent=historian)
        content = str(Crew(agents=[historian], tasks=[task], verbose=False).kickoff())
        path = f"{BASE}/confluence/retros/{conf_id}.md"
        save_md(path, content)
        entry = {"id": conf_id, "title": f"Retro Sprint #{self.state.sprint.sprint_number}", "summary": "Sprint Retrospective", "path": path}
        self.state.confluence_pages.append(entry)
        self._embed_and_count(id=conf_id, type="confluence", title=entry["title"], content=content, day=self.state.day, date=str(self.state.current_date.date()),
                              timestamp=meeting_time_iso)
        self._mem.log_event(SimEvent(type="retrospective", timestamp=meeting_time_iso, day=self.state.day, date=str(self.state.current_date.date()), 
                            actors=list(LEADS.values()), artifact_ids={"confluence": conf_id}, facts={"sprint_number": self.state.sprint.sprint_number}, 
                            summary=f"Sprint #{self.state.sprint.sprint_number} retrospective.", tags=["retrospective", "sprint"]))
        self.state.sprint.sprint_number += 1
        self.state.sprint.tickets_in_sprint = []

        self._record_daily_actor(*list(LEADS.values()))
        self._record_daily_event("retrospective")
        logger.info(f"    [green]✓[/green] {conf_id}")

    # ─── INCIDENT DETECTION ───────────────────
    def _handle_incident(self):
        ticket_id     = next_jira_id(self.state)
        on_call       = resolve_role("on_call_engineer")
        incident_lead = resolve_role("incident_commander")
        eng_peer      = next((n for n in ORG_CHART.get(CONFIG["roles"].get("on_call_engineer", ""), []) if n != on_call), on_call)

        incident_start = self._clock.tick_system(min_mins=30, max_mins=240)
        incident_start_iso = incident_start.isoformat()

        self._clock.sync_to_system([on_call])

        rc_agent = Agent(role="Senior Engineer", goal="Diagnose root cause.", backstory=persona_backstory(on_call, self._mem, graph_dynamics=self.graph_dynamics), llm=PLANNER_MODEL)
        rc_task  = Task(description=f"Theme: {self.state.daily_theme}\nWrite ONE specific technical root cause (max 20 words).", expected_output="One sentence.", agent=rc_agent)
        root_cause = str(Crew(agents=[rc_agent], tasks=[rc_task], verbose=False).kickoff()).strip()

        involves_gap = any(
            k.lower() in root_cause.lower()
            for emp in DEPARTED_EMPLOYEES.values()
            for k in emp["knew_about"]
        )

        self._lifecycle.scan_for_knowledge_gaps(
            text=root_cause,
            triggered_by=ticket_id,
            day=self.state.day,
            date_str=str(self.state.current_date.date()),
            state=self.state,
            timestamp=incident_start_iso
        )

        title  = f"{LEGACY['name']}: {self.state.daily_theme[:60]}"
        ticket = {"id": ticket_id, "title": title, "status": "In Progress", "assignee": on_call, "root_cause": root_cause, "linked_prs": []}
        self.state.jira_tickets.append(ticket)
        save_json(f"{BASE}/jira/{ticket_id}.json", ticket)
        self._embed_and_count(id=ticket_id, type="jira", title=title, content=json.dumps(ticket), day=self.state.day, 
                              date=str(self.state.current_date.date()), timestamp=incident_start_iso)

        inc = ActiveIncident(ticket_id=ticket_id, title=title, day_started=self.state.day, involves_gap_knowledge=involves_gap, root_cause=root_cause)
        self.state.active_incidents.append(inc)

        self.state.daily_incidents_opened += 1

        self._emit_bot_message(
            "system-alerts", 
            "Datadog", 
            f"🚨 [CRITICAL] Anomaly detected: {root_cause[:40]}... Error rate spiked 400%. System health dropped to {self.state.system_health}.",
            incident_start_iso
        )
        self._emit_bot_message(
            "incidents", 
            "PagerDuty", 
            f"📞 Paging on-call engineer: {on_call}. Incident linked to [{ticket_id}].",
            incident_start_iso
        )
        self.state.system_health = max(0, self.state.system_health - 15)

        triggered_contacts = self.graph_dynamics.relevant_external_contacts(
            event_type="incident_opened",
            system_health=self.state.system_health,
            config=CONFIG,
        )
        for contact in triggered_contacts:
            self._handle_external_contact(inc, contact)

        self._mem.log_event(SimEvent(
            type="incident_opened", timestamp=incident_start_iso,
            day=self.state.day, date=str(self.state.current_date.date()),
            actors=[on_call, incident_lead],
            artifact_ids={"jira": ticket_id},
            facts={"title": title, "root_cause": root_cause, "involves_gap": involves_gap},
            summary=f"P1 incident {ticket_id}: {root_cause}", tags=["incident", "P1"]
        ))

        self._record_daily_actor(on_call, incident_lead)
        self._record_daily_event("incident_opened")

        self.graph_dynamics.apply_incident_stress([on_call, incident_lead])
        self.graph_dynamics.record_incident_collaboration([on_call, incident_lead])

        gap_kw = [k for emp in DEPARTED_EMPLOYEES.values() for k in emp["knew_about"]]
        chain  = self.graph_dynamics.build_escalation_chain(
            first_responder=on_call,
            domain_keywords=gap_kw if involves_gap else None,
        )
        self._mem.log_event(SimEvent(
            type="escalation_chain", 
            timestamp=incident_start_iso,
            day=self.state.day,
            date=str(self.state.current_date.date()),
            actors=[n for n, _ in chain.chain],
            artifact_ids={"jira": ticket_id},
            facts={"chain": chain.chain,
                "narrative": self.graph_dynamics.escalation_narrative(chain)},
            summary=self.graph_dynamics.escalation_narrative(chain),
            tags=["escalation", "incident"],
        ))

        # Emit knowledge_gap_detected if this incident touches a departed employee's systems
        if involves_gap:
            gap_areas = [
                k for emp in DEPARTED_EMPLOYEES.values()
                for k in emp["knew_about"]
                if k.lower() in self.state.daily_theme.lower()
            ]
            self._mem.log_event(SimEvent(
                type="knowledge_gap_detected", 
                timestamp=incident_start_iso,
                day=self.state.day, date=str(self.state.current_date.date()),
                actors=[on_call, eng_peer],
                artifact_ids={"jira": ticket_id},
                facts={"gap_area": gap_areas or [LEGACY["name"]], "involves_gap": True},
                summary=f"Knowledge gap detected during {ticket_id}: systems owned by departed employee.",
                tags=["knowledge_gap"]
            ))

        if self.state.org_day_plan:
           eng_key = next((k for k in self.state.org_day_plan.dept_plans
                           if "eng" in k.lower()), None)
           if eng_key:
               eng_dept_plan = self.state.org_day_plan.dept_plans[eng_key]

               # 1. Primary rolls their independent time (2.0 to 5.5 hours)
               primary_hrs_lost = round(random.uniform(2.0, 5.5), 1)
               
               # 2. Peer rolls a DEPENDENT time (e.g., 20% to 60% of the primary's time)
               # This guarantees the peer never works longer than the primary.
               peer_fraction = random.uniform(0.2, 0.6)
               peer_hrs_lost = round(primary_hrs_lost * peer_fraction, 1)

               for ep in eng_dept_plan.engineer_plans:
                   if ep.name in [on_call, incident_lead]:
                       ep.apply_incident_pressure(inc.title, hrs_lost=primary_hrs_lost)
                   elif ep.name == eng_peer:
                       ep.apply_incident_pressure(inc.title, hrs_lost=peer_hrs_lost)

        logger.info(f"    [red]🚨 {ticket_id}:[/red] {root_cause[:65]}")

    def _advance_incidents(self):
        still_active = []
        on_call  = resolve_role("on_call_engineer")
        eng_peer = next((n for n in ORG_CHART.get(CONFIG["roles"].get("on_call_engineer", ""), []) if n != on_call), on_call)

        cron_time_iso = self._clock.now("system").isoformat()

        for inc in self.state.active_incidents:
            if inc.stage == "detected":
                inc.stage = "investigating"
                still_active.append(inc)

            elif inc.stage == "investigating":
                inc.stage = "fix_in_progress"
                pr = self._git.create_pr(author=on_call, ticket_id=inc.ticket_id, title=f"[{inc.ticket_id}] Fix: {inc.root_cause[:60]}",
                                          timestamp=cron_time_iso)

                for t in self.state.jira_tickets:
                    if t["id"] == inc.ticket_id:
                        if pr["pr_id"] not in t.get("linked_prs", []):
                            t.setdefault("linked_prs", []).append(pr["pr_id"])
                        
                        # Update the timestamp to match the PR creation time!
                        t["updated_at"] = cron_time_iso 
                        
                        save_json(f"{BASE}/jira/{inc.ticket_id}.json", t) # Update the JSON on disk
                        break

                self._emit_bot_message(
                    "engineering",
                    "GitHub",
                    f"🛠️ {on_call} opened PR {pr['pr_id']}: [{inc.ticket_id}] Fix. Reviewers requested: {', '.join(pr['reviewers'])}.",
                    cron_time_iso
                )
                inc.pr_id = pr["pr_id"]
                logger.info(f"    [yellow]🔧 {inc.ticket_id}:[/yellow] {pr['pr_id']} opened.")
                still_active.append(inc)

                # Check whether any external contacts should be triggered
                triggered_contacts = self.graph_dynamics.relevant_external_contacts(
                    event_type="fix_in_progress",
                    system_health=self.state.system_health,
                    config=CONFIG,
                )
                for contact in triggered_contacts:
                    self._handle_external_contact(inc, contact)

            elif inc.stage == "fix_in_progress":
                # Resolve: merge PR, write postmortem, log event — do NOT append to still_active
                inc.stage = "resolved"
                if inc.pr_id:
                    self._git.merge_pr(inc.pr_id)
                self._emit_bot_message(
                    "engineering",
                    "GitHub Actions",
                    f"✅ Build passed for PR {inc.pr_id}. Deploying to production...",
                    cron_time_iso
                )
                self.state.system_health = min(100, self.state.system_health + 20)
                self._write_postmortem(inc)
                self.state.resolved_incidents.append(inc.ticket_id)
                self.state.daily_incidents_resolved += 1
                self._mem.log_event(SimEvent(
                    type="incident_resolved", timestamp=cron_time_iso,
                    day=self.state.day, date=str(self.state.current_date.date()),
                    actors=[on_call, eng_peer], artifact_ids={"jira": inc.ticket_id, "pr": inc.pr_id or ""},
                    facts={"root_cause": inc.root_cause, "duration_days": inc.days_active},
                    summary=f"{inc.ticket_id} resolved in {inc.days_active}d.", tags=["incident_resolved"]
                ))
                logger.info(f"    [green]✅ {inc.ticket_id} resolved.[/green]")
                # Resolved — intentionally not appended to still_active

            else:
                # Unknown stage — keep active to avoid silent data loss
                still_active.append(inc)

        self.state.active_incidents = still_active

    def _write_postmortem(self, inc: ActiveIncident):
        on_call  = resolve_role("postmortem_writer")
        eng_peer = next((n for n in ORG_CHART.get(CONFIG["roles"].get("postmortem_writer", ""), []) if n != on_call), on_call)
        conf_id  = next_conf_id(self.state, "ENG")

        pm_duration_hours = random.randint(60, 180) / 60.0
        artifact_time, new_cursor = self._clock.advance_actor(on_call, hours=pm_duration_hours)
        artifact_time_iso = artifact_time.isoformat()

        writer   = Agent(role="Senior Engineer", goal="Write a postmortem.", backstory=persona_backstory(on_call, self._mem, graph_dynamics=self.graph_dynamics), llm=PLANNER_MODEL)
        task = Task(
            description=f"Write Confluence postmortem for {inc.ticket_id}.\nActual Root Cause: {inc.root_cause}\nDuration: {inc.days_active} days.\nFormat as Markdown.",
            expected_output="A full Markdown document.", agent=writer
        )
        content      = str(Crew(agents=[writer], tasks=[task], verbose=False).kickoff())
        full_content = f"# Postmortem: {inc.title}\n**ID:** {conf_id}\n**JIRA:** [{inc.ticket_id}]\n\n" + content
        path         = f"{BASE}/confluence/postmortems/{conf_id}.md"
        save_md(path, full_content)
        entry = {"id": conf_id, "title": f"Postmortem: {inc.ticket_id}", "path": path}
        self.state.confluence_pages.append(entry)
        self._embed_and_count(id=conf_id, type="confluence", title=entry["title"], content=full_content, day=self.state.day, date=str(self.state.current_date.date()),
                               timestamp=artifact_time_iso)
        self._mem.log_event(SimEvent(
            type="postmortem_created", 
            timestamp=artifact_time_iso,
            day=self.state.day, date=str(self.state.current_date.date()),
            actors=[on_call, eng_peer],
            artifact_ids={"confluence": conf_id, "jira": inc.ticket_id}, facts={"root_cause": inc.root_cause},
            summary=f"Postmortem {conf_id} written for {inc.ticket_id}.", tags=["postmortem"]
        ))
        logger.info(f"    [green]📄 Postmortem Generated:[/green] {conf_id}")

    def _emit_bot_message(self, channel: str, bot_name: str, text: str, timestamp: str):
        """Injects a contextual bot message into a specific Slack channel."""
        date_str = str(self.state.current_date.date())
        slack_path = f"{BASE}/slack/channels/{channel}/{date_str}_bots.json"
        
        # Load existing bot messages for the day if they exist
        messages = []
        if os.path.exists(slack_path):
            with open(slack_path, "r") as f:
                messages = json.load(f)
                
        messages.append({
            "user": bot_name, 
            "email": f"{bot_name.lower()}@bot.{COMPANY_DOMAIN}",
            "text": text, 
            "ts": timestamp,
            "is_bot": True
        })
        
        save_json(slack_path, messages)

    def _generate_adhoc_confluence_page(self):
        raw_topics = CONFIG.get("adhoc_confluence_topics", [["ENG", "Documentation"], ["HR", "Policy Update"]])
        rendered   = [(t[0], render_template(t[1])) for t in raw_topics]
        prefix, title = random.choice(rendered)
        conf_id = next_conf_id(self.state, prefix)
        author  = random.choice(ALL_NAMES)
        session_mins = random.randint(30, 90)
        session_hours = session_mins / 60.0
        artifact_time, new_cursor = self._clock.advance_actor(author, hours=session_hours)
        artifact_time_iso = artifact_time.isoformat()
        ctx     = self._mem.context_for_prompt(title, n=3, as_of_time=artifact_time_iso)
        
        writer = Agent(role="Corporate Writer", goal="Write documentation.", backstory=f"You are {author}.", llm=PLANNER_MODEL)
        task = Task(description=f"Write Confluence page titled '{title}'.\nContext:\n{ctx}\nFormat as Markdown.", expected_output="Markdown.", agent=writer)
        content = str(Crew(agents=[writer], tasks=[task], verbose=False).kickoff())

        full_content = f"# {title}\n**ID:** {conf_id}  \n**Author:** {author}\n\n" + content + bill_gap_warning(title)

        self._lifecycle.scan_for_knowledge_gaps(
            text=full_content,
            triggered_by=conf_id,
            day=self.state.day,
            date_str=str(self.state.current_date.date()),
            state=self.state,
            timestamp=artifact_time_iso
        )
        
        path = f"{BASE}/confluence/general/{conf_id}.md"
        save_md(path, full_content)
        entry = {"id": conf_id, "title": title, "path": path}
        self.state.confluence_pages.append(entry)
        self._embed_and_count(id=conf_id, type="confluence", title=title, content=full_content, day=self.state.day, date=str(self.state.current_date.date()),
                              timestamp=artifact_time_iso)
        self._mem.log_event(SimEvent(type="confluence_created", timestamp=artifact_time_iso,
                                     day=self.state.day, date=str(self.state.current_date.date()), 
                                     actors=[author], artifact_ids={"confluence": conf_id}, facts={"title": title}, summary=f"{author} created {conf_id}.", tags=["confluence"]))
        logger.info(f"    [dim]📝 Generated Confluence Page: {conf_id} — {title}[/dim]")

    # ─── END OF DAY ───────────────────────────
    def _end_of_day(self):
        date_str = str(self.state.current_date.date())
        decay    = CONFIG["morale"]["daily_decay"]
        recovery = CONFIG["morale"]["good_day_recovery"]

        self.state.team_morale = round(self.state.team_morale * decay, 3)
        if not self.state.active_incidents:
            self.state.team_morale = round(min(1.0, self.state.team_morale + recovery), 3)

        self.state.morale_history.append(self.state.team_morale)

        all_cursors = [self._clock.now(a) for a in ALL_NAMES]
        latest_time_worked = max(all_cursors) if all_cursors else self.state.current_date

        # Ensure the summary doesn't happen before 17:30
        eod_baseline = self.state.current_date.replace(hour=17, minute=30, second=0)
        summary_time = max(latest_time_worked, eod_baseline)
        housekeeping_time = summary_time + timedelta(minutes=1)

        # ── end_of_day event (unchanged) ─────────────────────────────────────────
        self._mem.log_event(SimEvent(
            type="end_of_day", timestamp=housekeeping_time.isoformat(), day=self.state.day, date=date_str, actors=[], artifact_ids={},
            facts={"morale": self.state.team_morale, "system_health": self.state.system_health},
            summary=f"Day {self.state.day} end.", tags=["eod"]
        ))

        # ── Derive enrichment fields from accumulated daily state ─────────────────

        # Deduplicated actors seen in any event today, ordered by frequency
        unique_actors = list(dict.fromkeys(self.state.daily_active_actors))

        # Dominant event type fired most often today (e.g. "incident_opened")
        event_counts  = self.state.daily_event_type_counts
        dominant_event = max(event_counts, key=event_counts.get) if event_counts else "normal_day"

        # Departments represented by today's active actors
        departments_involved = list({
            dept_of(name)
            for name in unique_actors
            if dept_of(name) != "Unknown"
        })

        # Still-open incidents at EOD (not yet resolved)
        open_incident_ids = [inc.ticket_id for inc in self.state.active_incidents]

        # Stress snapshot for today's active actors only — keeps the summary tight
        stress_today = dict({
            name: self.graph_dynamics._stress.get(name, 0)
            for name in unique_actors
        })

        # ── Enriched day_summary SimEvent ────────────────────────────────────────
        self._mem.log_event(SimEvent(
            type="day_summary",
            timestamp=housekeeping_time.isoformat(),
            day=self.state.day,
            date=date_str,
            actors=unique_actors,                          # populated — was always []
            artifact_ids={},
            facts={
                # ── Original numeric fields (unchanged) ──
                "incidents_opened":   self.state.daily_incidents_opened,
                "incidents_resolved": self.state.daily_incidents_resolved,
                "artifacts_created":  self.state.daily_artifacts_created,
                "external_contacts":  self.state.daily_external_contacts,
                "morale":             self.state.team_morale,
                "system_health":      self.state.system_health,
                "theme":              self.state.daily_theme,

                # ── New enrichment fields ──
                "active_actors":        unique_actors,
                "dominant_event":       dominant_event,
                "event_type_counts":    dict(self.state.daily_event_type_counts),
                "departments_involved": departments_involved,
                "open_incidents":       open_incident_ids,
                "stress_snapshot":      stress_today,

                # Trajectory signal — gives DayPlanner a one-glance health picture
                "health_trend":  (
                    "declining"  if self.state.system_health < 60 else
                    "recovering" if self.state.daily_incidents_resolved > self.state.daily_incidents_opened else
                    "stable"
                ),
                "morale_trend": (
                    "low"      if self.state.team_morale < 0.45 else
                    "moderate" if self.state.team_morale < 0.70 else
                    "healthy"
                ),
            },
            summary=(
                f"Day {self.state.day} ({date_str}): "
                f"{self.state.daily_incidents_opened} incident(s) opened, "
                f"{self.state.daily_incidents_resolved} resolved. "
                f"Health: {self.state.system_health} "
                f"({'declining' if self.state.system_health < 60 else 'recovering' if self.state.daily_incidents_resolved > self.state.daily_incidents_opened else 'stable'}). "
                f"Morale: {self.state.team_morale:.2f}. "
                f"Active actors: {', '.join(unique_actors) or 'none'}. "
                f"Depts: {', '.join(departments_involved) or 'none'}. "
                f"Dominant event: {dominant_event}."
            ),
            tags=["day_summary"],
        ))

        # ── Reset all daily counters ──────────────────────────────────────────────
        self.state.daily_incidents_opened   = 0
        self.state.daily_incidents_resolved = 0
        self.state.daily_artifacts_created  = 0
        self.state.daily_external_contacts  = 0
        self.state.daily_active_actors      = []   # new
        self.state.daily_event_type_counts  = {}   # new

        date_str = str(self.state.current_date.date())
        for dep in self._lifecycle._departed:
            if dep.day != self.state.day:
                continue   # only process today's departures
            # Pick the on-call engineer or first dept member as first responder
            dept_members = [
                n for n in LIVE_ORG_CHART.get(dep.dept, [])
                if n != dep.name
            ]
            if dept_members:
                note = recompute_escalation_after_departure(
                    self.graph_dynamics,
                    departed=dep,
                    first_responder=dept_members[0],
                )
                self._mem.log_event(SimEvent(
                    type="escalation_chain",
                    timestamp=housekeeping_time.isoformat(),
                    day=self.state.day,
                    date=date_str,
                    actors=dept_members[:2],
                    artifact_ids={},
                    facts={
                        "trigger":       "post_departure_reroute",
                        "departed":      dep.name,
                        "new_path_note": note,
                    },
                    summary=f"Escalation path updated after {dep.name} departure. {note}",
                    tags=["escalation_chain", "lifecycle"],
                ))

        self.graph_dynamics.decay_edges()
        self._last_stress_prop = self.graph_dynamics.propagate_stress()
        prop = self._last_stress_prop
        if prop.burnt_out:
            logger.info(
                f"    [red]🔥 Burnout spreading:[/red] "
                f"{', '.join(prop.burnt_out)} stressed; "
                f"neighbours affected: {', '.join(prop.affected) or 'none'}"
            )


    def _print_day_header(self):
        m = int(self.state.team_morale * 10)
        h = self.state.system_health // 10
        logger.info(f"\n[bold]Day {self.state.day}[/bold] [dim]({self.state.current_date.strftime('%a %b %d')})[/dim]  ❤️  {'█'*h}{'░'*(10-h)} {self.state.system_health}   😊 {'█'*m}{'░'*(10-m)} {self.state.team_morale:.2f}\n  [italic]{self.state.daily_theme}[/italic]")

    def _print_final_report(self):
        s = self._mem.stats()
        table = Table(title="Simulation Complete", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        for row in [
            ("Confluence Pages", str(len(self.state.confluence_pages))),
            ("JIRA Tickets", str(len(self.state.jira_tickets))),
            ("Slack Threads", str(len(self.state.slack_threads))),
            ("Git PRs", str(len(self.state.pr_registry))),
            ("Incidents Resolved", str(len(self.state.resolved_incidents))),
            ("Embedded Artifacts", str(s["artifact_count"])),
            ("Employees Departed", str(len(self._lifecycle._departed))),
            ("Employees Hired",    str(len(self._lifecycle._hired))),
            ("Knowledge Gaps Surfaced", str(len(self._lifecycle._gap_events))),
            ("MongoDB Active", "✓" if s["mongodb_ok"] else "⚠"),
        ]:
            table.add_row(*row)
        logger.info("\n")
        logger.info(table)
        
        snapshot = {
            "confluence_pages": self.state.confluence_pages, "jira_tickets": self.state.jira_tickets,
            "slack_threads": self.state.slack_threads, "pr_registry": self.state.pr_registry,
            "resolved_incidents": self.state.resolved_incidents, "morale_history": self.state.morale_history,
            "system_health": self.state.system_health, "event_log": [e.to_dict() for e in self._mem.get_event_log()],
        }
        snapshot["top_relationships"] = self.graph_dynamics.relationship_summary(10)
        snapshot["estranged_pairs"] = self.graph_dynamics.estranged_pairs()
        snapshot["departed_employees"] = [
            {
                "name":              d.name,
                "dept":              d.dept,
                "day":               d.day,
                "reason":            d.reason,
                "knowledge_domains": d.knowledge_domains,
                "documented_pct":    d.documented_pct,
                "peak_stress":       d.peak_stress,
            }
            for d in self._lifecycle._departed
        ]
        snapshot["new_hires"] = [
            {
                "name":      h.name,
                "dept":      h.dept,
                "day":       h.day,
                "role":      h.role,
                "expertise": h.expertise,
                "warm_edges_at_end": sum(
                    1 for nb in self.social_graph.neighbors(h.name)
                    if self.social_graph.has_node(h.name)
                    and self.social_graph[h.name][nb].get("weight", 0) >= h.warmup_threshold
                ) if self.social_graph.has_node(h.name) else 0,
            }
            for h in self._lifecycle._hired
        ]
        snapshot["knowledge_gap_events"] = [
            {
                "departed":       g.departed_name,
                "domain":         g.domain_hit,
                "triggered_by":   g.triggered_by,
                "day":            g.triggered_on_day,
                "documented_pct": g.documented_pct,
            }
            for g in self._lifecycle._gap_events
        ]
        snapshot["stress_snapshot"] = self._last_stress_prop.stress_snapshot if hasattr(self, '_last_stress_prop') else {}
        save_json(f"{BASE}/simulation_snapshot.json", snapshot)

    def _handle_external_contact(self, inc: ActiveIncident, contact: dict) -> None:
        """
        Generates a Slack message where an employee summarizes what an
        external party (AWS, customer, vendor) communicated about an incident.
        Logs a SimEvent for ground-truth retrieval evaluation.
        """
        liaison_dept = contact.get("internal_liaison", list(LEADS.keys())[0])
        liaison_name = LEADS.get(liaison_dept, next(iter(LEADS.values())))
        display_name = contact.get("display_name", contact["name"])
        tone         = contact.get("summary_tone", "professional")
        date_str     = str(self.state.current_date.date())

        participants = [liaison_name, display_name]
        interaction_mins = random.randint(15, 45)
        interaction_hours = interaction_mins / 60.0

        start_time, end_time = self._clock.sync_and_advance(participants, hours=interaction_hours)
        interaction_start_iso = start_time.isoformat()

        # Boost the edge between liaison and external node — they just talked
        external_node = contact["name"]
        if self.social_graph.has_edge(liaison_name, external_node):
            self.graph_dynamics.record_incident_collaboration(
                [liaison_name, external_node]
            )

        # Generate the Slack summary message
        ctx = self._mem.context_for_prompt(inc.root_cause, n=2, as_of_time=interaction_start_iso)

        agent = Agent(
            role="Employee",
            goal="Summarize an external conversation for your team on Slack.",
            backstory=persona_backstory(liaison_name, self._mem,
                extra=self.graph_dynamics.stress_tone_hint(liaison_name)),
            llm=WORKER_MODEL,
        )
        task = Task(
            description=(
                f"You just got off a call/email with {display_name} "
                f"({contact.get('org', 'external party')}) regarding incident "
                f"{inc.ticket_id}: {inc.root_cause}.\n"
                f"Their tone was: {tone}.\n"
                f"Context: {ctx}\n\n"
                f"Write a single Slack message to your team that:\n"
                f"1. Summarizes what {display_name} told you (2-3 sentences)\n"
                f"2. Ends with one concrete action item or next step\n"
                f"Keep it under 100 words. Do not use bullet points."
            ),
            expected_output="A single Slack message under 100 words.",
            agent=agent,
        )
        summary_text = str(
            Crew(agents=[agent], tasks=[task], verbose=False).kickoff()
        ).strip()

        # Write to the incidents Slack channel
        message = {
            "user":     liaison_name,
            "email":    email_of(liaison_name),
            "text":     summary_text,
            "ts":       self.state.current_date.replace(
                            hour=random.randint(10, 16),
                            minute=random.randint(0, 59)
                        ).isoformat(),
            "is_bot":   False,
            "metadata": {
                "type":           "external_contact_summary",
                "external_party": display_name,
                "org":            contact.get("org", "External"),
                "incident":       inc.ticket_id,
            }
        }

        slack_path = f"{BASE}/slack/channels/incidents/{date_str}_{external_node}.json"
        save_json(slack_path, [message])
        self.state.slack_threads.append({
            "date":          date_str,
            "channel":       "incidents",
            "message_count": 1,
        })

        # SimEvent — this is what makes it retrievable as ground truth
        self._mem.log_event(SimEvent(
            type="external_contact_summarized",
            timestamp=interaction_start_iso,
            day=self.state.day,
            date=date_str,
            actors=[liaison_name, external_node],
            artifact_ids={"slack": slack_path, "jira": inc.ticket_id},
            facts={
                "external_party":  display_name,
                "org":             contact.get("org", "External"),
                "incident":        inc.ticket_id,
                "root_cause":      inc.root_cause,
                "liaison":         liaison_name,
                "summary_tone":    tone,
            },
            summary=(
                f"{liaison_name} summarized {display_name} contact re "
                f"{inc.ticket_id} in #incidents."
            ),
            tags=["external", "slack", "incident"],
        ))
        self.state.daily_external_contacts += 1

        self._embed_and_count(
            id=f"ext_{external_node}_{inc.ticket_id}",
            type="slack",
            title=f"External contact summary: {display_name} re {inc.ticket_id}",
            content=summary_text,
            day=self.state.day,
            date=date_str,
            metadata={
                "external_party": display_name,
                "liaison":        liaison_name,
                "incident":       inc.ticket_id,
                "causal_parent":   inc.ticket_id,
            },
            timestamp=interaction_start_iso
        )

        self._record_daily_actor(liaison_name)
        self._record_daily_event("external_contact_summarized")

        logger.info(
            f"    [cyan]🌐 External contact:[/cyan] {liaison_name} summarized "
            f"{display_name} re {inc.ticket_id} in #incidents"
        )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Wipe MongoDB and export dir before running")
    args = parser.parse_args()

    flow = Flow()
    if args.reset:
        flow._mem.reset(export_dir=BASE)
    flow.kickoff()