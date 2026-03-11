"""
flow.py (MongoDB + YAML + NetworkX Edition)
=================================================
OrgForge simulation engine. Reads from config.yaml.
Uses NetworkX for social graphs. Uses MongoDB for vector/artifact storage.
"""

from config_loader import (
    EXPORT_DIR,
    CONFIG,
    COMPANY_NAME,
    COMPANY_DOMAIN,
    INDUSTRY,
    BASE,
    ORG_CHART,
    LEADS,
    PERSONAS,
    DEFAULT_PERSONA,
    LEGACY,
    PRODUCT_PAGE,
    DEPARTED_EMPLOYEES,
    ALL_NAMES,
    LIVE_ORG_CHART,
    LIVE_PERSONAS,
    _PRESET_NAME,
    _PRESET,
    _PROVIDER,
    resolve_role,
)

import os
import logging
import json
import random
import re
from agent_factory import make_agent
import networkx as nx
from datetime import datetime, timedelta
from typing import Any, List, Dict, Optional
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from day_planner import DayPlannerOrchestrator
from normal_day import NormalDayHandler, dept_of_name
from artifact_registry import ArtifactRegistry
from confluence_writer import ConfluenceWriter
from token_tracker import orgforge_token_listener
from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.logging import RichHandler
from rich import box
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from crewai import Process, Task, Crew
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
from causal_chain_handler import (
    CausalChainHandler,
    ARTIFACT_KEY_JIRA,
    ARTIFACT_KEY_SLACK_THREAD,
    RecurrenceDetector,
)

os.makedirs("./export", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    force=True,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(EXPORT_DIR / "simulation.log", mode="a"),
        RichHandler(rich_tracebacks=True, show_time=False, show_path=False),
    ],
)

logger = logging.getLogger("orgforge.flow")


def _patch_crewai_bedrock():
    try:
        from crewai.llms.providers.bedrock import completion as _bedrock_mod

        original = _bedrock_mod.BedrockCompletion._get_inference_config

        def patched_get_inference_config(self):
            config = original(self)
            config.pop("stopSequences", None)
            return config

        _bedrock_mod.BedrockCompletion._get_inference_config = (
            patched_get_inference_config
        )
        logger.info("[patch] crewAI Bedrock stopSequences patch applied")

    except (ImportError, AttributeError) as e:
        logger.warning(f"[patch] Could not patch crewAI Bedrock provider: {e}")


_patch_crewai_bedrock()


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
    model = _bare_model(model_str)

    if _PROVIDER == "bedrock":
        try:
            from crewai import LLM

            region = _PRESET.get(
                "aws_region", os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
            )

            llm_args = {
                "model": model,
                "region_name": region,
                "temperature": 0.7,
            }

            if "deepseek" in model.lower():
                llm_args["drop_params"] = True
                llm_args["additional_drop_params"] = [
                    "stop",
                    "temperature",
                    "stopSequences",
                ]
                logger.info(
                    f"[config] Adding drop_params=True for Bedrock model: {model}"
                )

            llm = LLM(**llm_args)

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
    return OllamaLLM(model=model, base_url=base_url)


PLANNER_MODEL = build_llm("planner")
WORKER_MODEL = build_llm("worker")

# Propagate embedding config to memory via environment
os.environ.setdefault("EMBED_PROVIDER", _PRESET.get("embed_provider", "ollama"))
os.environ.setdefault("EMBED_MODEL", _PRESET.get("embed_model", "mxbai-embed-large"))
os.environ.setdefault("EMBED_DIMS", str(_PRESET.get("embed_dims", 1024)))
os.environ.setdefault("DB_NAME", CONFIG["simulation"].get("db_name", "orgforge"))
if _PROVIDER == "bedrock":
    os.environ.setdefault("AWS_DEFAULT_REGION", _PRESET.get("aws_region", "us-east-1"))


def render_template(template: str) -> str:
    """Replace {legacy_system}, {project_name}, {product_page} placeholders in config strings."""
    return (
        template.replace("{legacy_system}", LEGACY["name"])
        .replace("{project_name}", LEGACY["project_name"])
        .replace("{product_page}", PRODUCT_PAGE)
        .replace("{company_name}", COMPANY_NAME)
        .replace("{industry}", INDUSTRY)
    )


console = Console()
vader = SentimentIntensityAnalyzer()


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
            G.add_node(
                member,
                dept=dept,
                is_lead=(member in LEADS.values()),
                external=False,
            )

    for n1 in G.nodes():
        for n2 in G.nodes():
            if n1 >= n2:
                continue
            weight = 0.5
            if G.nodes[n1]["dept"] == G.nodes[n2]["dept"]:
                weight += 10.0
            if G.nodes[n1]["is_lead"] and G.nodes[n2]["is_lead"]:
                weight += 5.0
            G.add_edge(n1, n2, weight=weight)

    # External nodes — cold edges to their liaison department
    for contact in CONFIG.get("external_contacts", []):
        node_id = contact["name"]
        liaison_dept = contact.get("internal_liaison", list(LEADS.keys())[0])
        liaison_lead = LEADS.get(liaison_dept, next(iter(LEADS.values())))

        G.add_node(
            node_id,
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
    causal_chain: Any = None
    recurrence_of: Optional[str] = None


class SprintState(BaseModel):
    sprint_number: int = 1
    start_day: int = 1
    tickets_in_sprint: List[str] = []
    velocity: int = 0
    sprint_theme: str = ""

    def start_date(self, sim_start_date: datetime, sprint_length: int) -> datetime:
        """
        Derive sprint start date by counting only business days forward
        from sim start — mirrors how state.day advances in the main loop.
        """
        target_day = (self.sprint_number - 1) * sprint_length + 1
        current = sim_start_date
        biz_day = 1
        while biz_day < target_day:
            current += timedelta(days=1)
            if current.weekday() < 5:
                biz_day += 1
        return current


class State(BaseModel):
    day: int = 1
    max_days: int = Field(default_factory=lambda: CONFIG["simulation"]["max_days"])
    current_date: datetime = Field(
        default_factory=lambda: datetime.strptime(
            CONFIG["simulation"]["start_date"], "%Y-%m-%d"
        )
    )
    system_health: int = 100
    team_morale: float = Field(default_factory=lambda: CONFIG["morale"]["initial"])
    morale_history: List[float] = []
    is_researching: bool = False
    active_incidents: List[ActiveIncident] = []
    resolved_incidents: List[str] = []
    sprint: SprintState = Field(default_factory=SprintState)
    daily_theme: str = ""
    persona_stress: Dict[str, int] = {}
    actor_cursors: Dict[str, Any] = Field(default_factory=dict)

    # Daily counters — reset each morning, read at end of day
    daily_incidents_opened: int = 0
    daily_incidents_resolved: int = 0
    daily_artifacts_created: int = 0
    daily_external_contacts: int = 0

    org_day_plan: Optional[Any] = None
    daily_active_actors: List[str] = []
    daily_event_type_counts: Dict[str, int] = {}
    departed_employees: Dict[
        str, Dict
    ] = {}  # name → {left, role, knew_about, documented_pct}
    new_hires: Dict[str, Dict] = {}  # name → {joined, role, dept, expertise}
    ticket_actors_today: Dict[str, List[str]] = Field(default_factory=dict)


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


def save_eml(
    path: str,
    from_name: str,
    to_names: List[str],
    subject: str,
    body: str,
    cc_names: Optional[List[str]] = None,
    in_reply_to: Optional[str] = None,
    date_str: Optional[str] = None,
):
    msg = MIMEMultipart("alternative")
    msg["From"] = f"{from_name} <{email_of(from_name)}>"
    msg["To"] = ", ".join(f"{n} <{email_of(n)}>" for n in to_names)
    msg["Subject"] = subject
    msg["Date"] = date_str or datetime.now().strftime("%a, %d %b %Y %H:%M:%S +0000")
    msg["Message-ID"] = f"<{random.randint(10000, 99999)}@{COMPANY_DOMAIN}>"
    if cc_names:
        msg["Cc"] = ", ".join(f"{n} <{email_of(n)}>" for n in cc_names)
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
        self.graph_dynamics = GraphDynamics(build_social_graph(), CONFIG)

    def create_pr(
        self,
        author: str,
        ticket_id: str,
        title: str,
        timestamp: str,
        reviewers: Optional[List[str]] = None,
    ) -> Dict:
        pr_id = f"PR-{self._mem._prs.count_documents({}) + 100}"

        if not reviewers:
            edges = self._graph[author]

            eng_colleagues = {
                n: edges[n].get("weight", 1.0)
                for n in edges
                if "engineer" in self._graph.nodes[n].get("dept", "").lower()
                and n != author
            }

            if eng_colleagues:
                sorted_eng = sorted(
                    eng_colleagues.items(), key=lambda x: x[1], reverse=True
                )
                reviewers = [sorted_eng[0][0]]
                if len(sorted_eng) > 1:
                    reviewers.append(sorted_eng[1][0])
            else:
                eng_dept = next(
                    (d for d in ORG_CHART if "engineer" in d.lower()),
                    list(ORG_CHART.keys())[0],
                )
                fallback = next(
                    (n for n in ORG_CHART[eng_dept] if n != author),
                    ORG_CHART[eng_dept][0],
                )
                reviewers = [fallback]

        try:
            ctx = self._mem.context_for_prompt(title, n=2, as_of_time=timestamp)
            agent = make_agent(
                role=f"{author}, Software Engineer",
                goal=f"Write a PR description for your own code as {author} would.",
                backstory=persona_backstory(
                    author, self._mem, graph_dynamics=self.graph_dynamics
                ),
                llm=self._worker_llm,
            )
            task = Task(
                description=(
                    f"You are {author}. Write a GitHub Pull Request description.\n\n"
                    f"FORMAT — respond in Markdown with exactly these two sections:\n"
                    f"## What Changed\n"
                    f"[1-2 sentences]\n\n"
                    f"## Why\n"
                    f"[1-2 sentences]\n\n"
                    f"Total length: under 100 words. No preamble, no extra sections.\n\n"
                    f"--- TICKET ---\n"
                    f"[{ticket_id}]: {title}\n\n"
                    f"--- CONTEXT ---\n"
                    f"{ctx}"
                ),
                expected_output=(
                    "Markdown PR body with exactly two sections: '## What Changed' and '## Why'. "
                    "Under 100 words. No preamble."
                ),
                agent=agent,
            )
            description = str(
                Crew(agents=[agent], tasks=[task], verbose=False).kickoff()
            ).strip()
        except Exception:
            description = f"Auto-generated PR for [{ticket_id}]: {title}"

        pr = {
            "pr_id": pr_id,
            "ticket_id": ticket_id,
            "linked_ticket": ticket_id,
            "title": title,
            "description": description,
            "author": author,
            "author_email": email_of(author),
            "reviewers": reviewers,
            "status": "open",
            "comments": [],
            "created_at": timestamp,
        }

        path = f"{BASE}/git/prs/{pr_id}.json"
        save_json(path, pr)
        self._mem.upsert_pr(pr)
        self._mem.embed_artifact(
            id=pr_id,
            type="pr",
            title=title,
            content=json.dumps(pr),
            day=self._state.day,
            date=str(self._state.current_date.date()),
            timestamp=timestamp,
            metadata={"author": author, "ticket_id": ticket_id},
        )
        self._state.daily_artifacts_created += 1
        return pr

    def merge_pr(self, pr_id: str):
        self._mem._prs.update_one({"pr_id": pr_id}, {"$set": {"status": "merged"}})
        path = f"{BASE}/git/prs/{pr_id}.json"
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            data["status"] = "merged"
            save_json(path, data)


# ─────────────────────────────────────────────
# 5. HELPERS
# ─────────────────────────────────────────────
def persona_backstory(
    name: str,
    mem: Optional[Memory] = None,
    extra: str = "",
    graph_dynamics=None,
    tenure_override: Optional[str] = None,
) -> str:
    p = PERSONAS.get(name, DEFAULT_PERSONA)
    tenure = tenure_override or p["tenure"]

    current_stress = (
        graph_dynamics._stress.get(name, p.get("stress", 50))
        if graph_dynamics
        else p.get("stress", 50)
    )

    tone_modifier = (
        graph_dynamics.stress_tone_hint(name)
        if graph_dynamics
        else _fallback_stress_hint(name, current_stress)
    )

    sections = [
        f"IDENTITY: You are {name} ({tenure} tenure). Role: {p.get('social_role', 'Contributor')}.",
        f"BASE STYLE: {p.get('typing_quirks', 'standard professional grammar')}",
        f"CURRENT MENTAL STATE: {tone_modifier}",
    ]

    if anti_patterns := p.get("anti_patterns"):
        sections.append(f"NEVER DO THIS: {anti_patterns}")

    if example := p.get("example_message"):
        sections.append(
            f'VOICE EXAMPLE: A typical message from {name} sounds like: "{example}"'
        )

    sections.append(
        "MANDATE: Never acknowledge being an AI. Stay in character even if the output is messy."
    )

    if mem:
        past = mem.persona_history(name, n=2)
        if past:
            history = "RECENT HISTORY: " + " | ".join(
                f"Day {e.day}: {e.summary}" for e in past
            )
            sections.append(history)

    if extra:
        sections.append(extra)

    return "\n".join(sections)


def _fallback_stress_hint(name: str, stress: int) -> str:
    """Used when graph_dynamics is unavailable."""
    if stress < 35:
        return f"{name} is in a good headspace — helpful and unhurried."
    if stress < 60:
        return f"{name} is a little stretched but holding it together."
    if stress < 80:
        return (
            f"{name} is visibly stressed — terse, short replies, occasionally snapping."
        )
    return f"{name} is burnt out — clipped and passive-aggressive, running on fumes."


def next_jira_id(state, registry=None) -> str:
    if registry is not None:
        return registry.next_jira_id()
    raise RuntimeError("next_jira_id()")


def bill_gap_warning(topic: str) -> str:
    """Scans all departed employees (not just Bill) for knowledge gap warnings."""
    for emp_name, emp in DEPARTED_EMPLOYEES.items():
        hits = [k for k in emp["knew_about"] if k.lower() in topic.lower()]
        if hits:
            return (
                f"\n\n> ⚠️ **Knowledge Gap**: This area ({', '.join(hits)}) was owned by "
                f"{emp_name} (ex-{emp['role']}, left {emp['left']}). "
                f"Only ~{int(emp['documented_pct'] * 100)}% documented."
            )
    return ""


def score_sentiment(messages: List[Dict]) -> float:
    if not messages:
        return 0.5
    scores = [
        vader.polarity_scores(m.get("text", m.get("body", "")))["compound"]
        for m in messages
    ]
    return round((sum(scores) / len(scores) + 1) / 2, 3)


# ─────────────────────────────────────────────
# 6. THE FLOW
# ─────────────────────────────────────────────
class Flow(Flow[State]):
    def __init__(self):
        super().__init__()
        self._mem = Memory()
        self.graph_dynamics = GraphDynamics(build_social_graph(), CONFIG)
        self.social_graph = self.graph_dynamics.G
        self._git = GitSimulator(self.state, self._mem, self.social_graph, WORKER_MODEL)
        self._day_planner = DayPlannerOrchestrator(CONFIG, WORKER_MODEL, PLANNER_MODEL)
        self._clock = SimClock(self.state)
        self._lifecycle = OrgLifecycleManager(
            config=CONFIG,
            graph_dynamics=self.graph_dynamics,
            mem=self._mem,
            org_chart=LIVE_ORG_CHART,
            personas=LIVE_PERSONAS,
            all_names=ALL_NAMES,
            leads=LEADS,
            worker_llm=WORKER_MODEL,
            base_export_dir=BASE,
        )
        self._registry = ArtifactRegistry(self._mem, base_export_dir=BASE)
        self._confluence = ConfluenceWriter(
            mem=self._mem,
            registry=self._registry,
            state=self.state,
            config=CONFIG,
            worker_llm=WORKER_MODEL,
            planner_llm=PLANNER_MODEL,
            clock=self._clock,
            lifecycle=self._lifecycle,
            persona_helper=persona_backstory,
            graph_dynamics=self.graph_dynamics,
        )
        self._normal_day = NormalDayHandler(
            config=CONFIG,
            mem=self._mem,
            state=self.state,
            graph_dynamics=self.graph_dynamics,
            social_graph=self.social_graph,
            git=self._git,
            worker_llm=WORKER_MODEL,
            planner_llm=PLANNER_MODEL,
            clock=self._clock,
            persona_helper=persona_backstory,
            confluence_writer=self._confluence,
            vader=vader,
        )
        self._recurrence_detector = RecurrenceDetector(self._mem)
        orgforge_token_listener.attach(self._mem)

        stats = self._mem.stats()
        logger.info(
            f"[dim]Memory: provider={stats['embed_provider']} model={stats['embed_model']} dims={stats['embed_dims']} MongoDB={'✓' if stats['mongodb_ok'] else '⚠'}[/dim]"
        )

    def _is_sprint_planning_day(self) -> bool:
        sprint_length = CONFIG["simulation"].get("sprint_length_days", 10)
        return self.state.day % sprint_length == 1

    def _is_retro_day(self) -> bool:
        sprint_length = CONFIG["simulation"].get("sprint_length_days", 10)
        return self.state.day % sprint_length == (sprint_length - 1)

    def _is_standup_day(self) -> bool:
        # Skip standup if the team is already doing sprint planning today
        if self._is_sprint_planning_day():
            return False
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
        if self._mem.has_genesis_artifacts():
            logger.info(
                "[bold green]⏩ Genesis Guard: Corporate history exists. Skipping LLM generation.[/bold green]"
            )
            # The Registry seeds itself from Mongo in __init__, so IDs are already synced.
            return

        logger.info(
            Panel.fit(
                f"[bold cyan]{COMPANY_NAME.upper()} — ORGFORGE SIMULATION[/bold cyan]\n"
                f"[dim]Preset: {_PRESET_NAME} | Provider: {_PROVIDER} | Seeding corporate archives...[/dim]",
                box=box.DOUBLE_EDGE,
            )
        )

        eng_dept = next(
            (d for d in ORG_CHART if "engineer" in d.lower() or "eng" in d.lower()),
            list(ORG_CHART.keys())[0],
        )
        sales_dept = next(
            (d for d in ORG_CHART if "sales" in d.lower() or "market" in d.lower()),
            list(ORG_CHART.keys())[-1],
        )
        eng_member = random.choice(ORG_CHART[eng_dept])
        sale_member = random.choice(ORG_CHART[sales_dept])

        tech_cfg = CONFIG.get("genesis_docs", {}).get("technical", {})
        biz_cfg = CONFIG.get("genesis_docs", {}).get("business", {})

        self._confluence.generate_tech_stack()
        tech_context = self._mem.tech_stack_for_prompt()

        # Technical pages — one LLM call per page, no PAGE BREAK parsing
        self._confluence.write_genesis_batch(
            prefix=tech_cfg.get("id_prefix", "CONF-ENG").replace("CONF-", ""),
            count=tech_cfg.get("count", 3),
            prompt_tpl=(
                "You are {author}. Write a single Confluence page with ID {id} about {project_name} "
                "and {legacy_system}. "
                "Existing related pages you may reference: {related_pages}. "
                "Use only the following canonical tech stack — never invent or substitute alternatives:\n{tech_stack}\n"
                "Output only Markdown. Do not include an author block, contributor list, "
                "or metadata section in your output."
            ),
            author=eng_member,
            subdir="archives",
            extra_vars={"tech_stack": tech_context},
        )

        # Business pages
        self._confluence.write_genesis_batch(
            prefix=biz_cfg.get("id_prefix", "CONF-MKT").replace("CONF-", ""),
            count=biz_cfg.get("count", 2),
            prompt_tpl=(
                "You are {author}. Write a single Confluence page with ID {id} for {company} "
                "about {product_page} campaign planning and go-to-market strategy. "
                "Existing related pages you may reference: {related_pages}. "
                "Output only Markdown. Do not include an author block, contributor list, "
                "or metadata section in your output."
            ),
            author=sale_member,
            extra_vars={"product_page": PRODUCT_PAGE},
            subdir="archives",
            tags=["genesis"],
        )

        logger.info(
            f"[green]✓ Genesis complete.[/green] "
            f"Memory: {self._mem.stats()['artifact_count']} artifacts embedded.\n"
        )

    # ─── DAILY LOOP ───────────────────────────
    @listen(genesis_phase)
    def daily_cycle(self):
        latest = self._mem.load_latest_checkpoint()
        if latest:
            logger.info(
                f"[bold cyan]♻️ Resuming simulation from Day {latest['day']}...[/bold cyan]"
            )
            self.state.day = latest["day"] + 1
            self.state.team_morale = latest["state_vars"]["morale"]
            self.state.system_health = latest["state_vars"]["health"]

            # Restore the 'Live' state of the secondary systems
            self.graph_dynamics._stress = latest["stress"]
            self.state.actor_cursors = latest["cursors"]

            # Re-sync current_date string back to a datetime object
            self.state.current_date = datetime.strptime(
                latest["state_vars"]["date"], "%Y-%m-%d"
            )

            if "graph" in latest:
                restored_graph = nx.node_link_graph(latest["graph"])
                self.social_graph.clear()
                self.social_graph.add_nodes_from(restored_graph.nodes(data=True))
                self.social_graph.add_edges_from(restored_graph.edges(data=True))
                # Force graph_dynamics to recalculate betweenness centrality
                self.graph_dynamics._centrality_dirty = True

        while self.state.day <= self.state.max_days:
            dow = self.state.current_date.weekday()
            if dow >= 5:
                logger.info(
                    f"[dim]  ↷ Weekend ({self.state.current_date.date()})[/dim]"
                )
                self.state.current_date += timedelta(days=1)
                continue

            self._state.ticket_actors_today = {}  # cleared here; orchestrator re-seeds from SprintContext
            self._clock.reset_to_business_start(ALL_NAMES)
            date_str = str(self.state.current_date.date())
            departures = self._lifecycle.process_departures(
                self.state.day, date_str, self.state, self._clock
            )
            hires = self._lifecycle.process_hires(
                self.state.day, date_str, self.state, self._clock
            )

            for inc in self.state.active_incidents:
                inc.days_active += 1

            if departures or hires:
                # Patch the day planner's validator to reflect the new roster
                patch_validator_for_lifecycle(
                    self._day_planner._validator, self._lifecycle
                )

            org_plan = self._day_planner.plan(
                self.state,
                self._mem,
                self.graph_dynamics,
                lifecycle_context=self._lifecycle.get_roster_context(),
                clock=self._clock,
            )
            if org_plan is None:
                logger.error(
                    f"[flow] Day {self.state.day}: DayPlanner returned None — skipping normal day"
                )
                continue  # or raise, depending on how strict you want to be

            self._mem._current_day = self.state.day
            self.state.daily_theme = org_plan.org_theme
            self.state.org_day_plan = org_plan
            self._print_day_header()

            if self._is_sprint_planning_day():
                self._handle_sprint_planning()

            if self._is_standup_day():
                self._handle_standup()
            if self._is_retro_day():
                self._handle_retrospective()

            self._advance_incidents()

            theme_lower = self.state.daily_theme.lower()
            incident_triggers = CONFIG.get(
                "incident_triggers",
                ["crash", "fail", "error", "latency", "timeout", "outage"],
            )
            if any(x in theme_lower for x in incident_triggers):
                self._handle_incident()
            else:
                self._normal_day.handle(self.state.org_day_plan)
                if random.random() < CONFIG["simulation"].get(
                    "adhoc_confluence_prob", 0.3
                ):
                    self._generate_adhoc_confluence_page()

            self._mem.save_checkpoint(
                day=self.state.day,
                state_vars={
                    "morale": self.state.team_morale,
                    "health": self.state.system_health,
                    "date": str(self.state.current_date.date()),
                },
                stress=self.graph_dynamics._stress,
                cursors=self.state.actor_cursors,
                graph_data=nx.node_link_data(self.social_graph),
            )

            self._end_of_day()
            self.state.day += 1
            self.state.current_date += timedelta(days=1)

        self._print_final_report()

    # ─── SPRINT PLANNING ──────────────────────
    def _handle_sprint_planning(self):
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed

        sprint_num = self.state.sprint.sprint_number
        logger.info(
            f"  [bold blue]📋 Sprint #{sprint_num} Planning (LLM-driven)[/bold blue]"
        )

        active = list(dict.fromkeys(getattr(self.state, "daily_active_actors", [])))
        leads = list(LEADS.values())
        if len(active) >= 4:
            attendees = list(
                dict.fromkeys(leads + random.sample(active, min(3, len(active))))
            )
        else:
            attendees = list(
                dict.fromkeys(leads + random.sample(ALL_NAMES, min(3, len(ALL_NAMES))))
            )
        meeting_time = self._clock.schedule_meeting(attendees, min_hour=9, max_hour=11)
        timestamp_str = meeting_time.isoformat()
        date_str = str(self.state.current_date.date())

        # ── Step 1: LLM generates sprint theme (one call, cheap) ──────────────
        # Tier 1: structured MongoDB query — no embedding needed here.
        # sprint_theme omitted intentionally; it hasn't been decided yet.
        ctx = self._mem.context_for_sprint_planning(
            sprint_num=sprint_num,
            dept="",  # all depts for the theme step
            as_of_time=timestamp_str,
        )
        product_dept = next((d for d in LEADS if "product" in d.lower()), None)
        product_lead = LEADS.get(product_dept, LEADS[next(iter(LEADS))])

        product_agent = make_agent(
            role="Product Manager",
            goal="Propose a sprint theme grounded in business priorities.",
            backstory=persona_backstory(
                product_lead, self._mem, graph_dynamics=self.graph_dynamics
            ),
            llm=PLANNER_MODEL,
        )
        product_task = Task(
            description=(
                f"It's Sprint #{sprint_num} planning at {COMPANY_NAME}.\n"
                f"System health: {self.state.system_health}/100. Morale: {self.state.team_morale:.2f}.\n"
                f"Recent context:\n{ctx}\n\n"
                f"Propose a sprint theme as a single sentence grounded in customer needs, "
                f"roadmap priorities, or recent sales signals. Be specific to {INDUSTRY}. "
                f"Output only the theme sentence."
            ),
            expected_output="One sentence sprint theme proposal.",
            agent=product_agent,
        )

        eng_dept = next((d for d in LEADS if "eng" in d.lower()), None)
        eng_lead = LEADS.get(eng_dept, LEADS[next(iter(LEADS))])

        eng_agent = make_agent(
            role="Engineering Lead",
            goal="Ratify or amend the sprint theme based on technical reality.",
            backstory=persona_backstory(
                eng_lead, self._mem, graph_dynamics=self.graph_dynamics
            ),
            llm=PLANNER_MODEL,
        )
        eng_task = Task(
            description=(
                f"The Product Manager has just proposed a sprint theme (see context above).\n"
                f"System health: {self.state.system_health}/100. "
                f"Morale: {self.state.team_morale:.2f}.\n\n"
                f"Either accept it as-is or amend it to reflect technical constraints "
                f"or current system reality. Output only the final one-sentence theme. "
                f"No preamble, no explanation."
            ),
            expected_output="One sentence. No preamble, no label, no quotes.",
            agent=eng_agent,
            context=[product_task],
        )

        sprint_theme = str(
            Crew(
                agents=[product_agent, eng_agent],
                tasks=[product_task, eng_task],
                verbose=True,
            ).kickoff()
        ).strip()
        logger.info(f"    [cyan]🎯 Sprint theme:[/cyan] {sprint_theme}")

        self.state.sprint.sprint_theme = sprint_theme

        # ── Step 2: Per-dept ticket generation in parallel ─────────────────────
        n_per_dept = CONFIG["simulation"].get("sprint_tickets_per_planning", 4)
        all_new_tickets: list = []
        lock = threading.Lock()

        def _generate_dept_tickets(dept: str, members: list) -> list:
            # Tier 1: structured query scoped to this dept — no embedding.
            dept_ctx = self._mem.context_for_sprint_planning(
                sprint_num=sprint_num,
                dept=dept,
                sprint_theme=sprint_theme,
                as_of_time=timestamp_str,
            )
            lead_name = LEADS.get(dept, members[0])
            agent = make_agent(
                role=f"{dept} Lead",
                goal=f"Create realistic sprint tickets for the {dept} team.",
                backstory=persona_backstory(
                    lead_name, self._mem, graph_dynamics=self.graph_dynamics
                ),
                llm=WORKER_MODEL,
            )
            # Before building the task, collect the dept's expertise tags
            member_expertise = {}
            for name in members:
                persona = CONFIG.get("personas", {}).get(name, {})
                tags = persona.get("expertise", [])
                if tags:
                    member_expertise[name] = tags

            expertise_str = "\n".join(
                f"  - {name}: {', '.join(tags)}"
                for name, tags in member_expertise.items()
            )

            task = Task(
                description=(
                    f'Sprint #{sprint_num} theme: "{sprint_theme}"\n'
                    f"Department: {dept}\n"
                    f"Team members and their expertise:\n{expertise_str}\n"
                    f"Recent dept context:\n{dept_ctx}\n\n"
                    f"Create exactly {n_per_dept} Jira tickets for this sprint.\n"
                    f"DOMAIN CONSTRAINT: Every ticket must be work that maps directly to "
                    f"at least one expertise tag listed above. If a ticket topic does not "
                    f"appear in any team member's expertise, it belongs to a different "
                    f"department — do not create it.\n\n"
                    f"Each ticket must:\n"
                    f"  - Be concrete and specific (not 'improve performance' but "
                    f"'reduce /search endpoint p99 from 800ms to 300ms')\n"
                    f"  - Reference real systems, endpoints, or workflows from the "
                    f"context where possible\n"
                    f"  - Have story points (1, 2, 3, 5, or 8) based on complexity\n\n"
                    f"Respond ONLY with a JSON array:\n"
                    f"[\n"
                    f'  {{"title": "string", "story_points": int, '
                    f'"description": "string (2 sentences)"}},\n'
                    f"  ...\n"
                    f"]\n"
                    f"No preamble. No markdown fences. Raw JSON only."
                ),
                expected_output=f"JSON array of {n_per_dept} ticket objects.",
                agent=agent,
            )
            raw = str(
                Crew(agents=[agent], tasks=[task], verbose=True).kickoff()
            ).strip()

            # Parse — strip accidental fences
            raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("` \n")
            try:
                proposals = json.loads(raw)
                if not isinstance(proposals, list):
                    raise ValueError("Not a list")
            except Exception as e:
                logger.warning(
                    f"[sprint] {dept} ticket parse failed: {e}. Raw: {raw[:200]}"
                )
                # Graceful fallback: one generic ticket so the sprint isn't empty
                proposals = [
                    {
                        "title": f"{sprint_theme} — {dept} work",
                        "story_points": 2,
                        "description": "",
                    }
                ]

            dept_tickets = []
            with lock:
                for proposal in proposals[:n_per_dept]:
                    tid = self._registry.next_jira_id()
                    self._registry.register_jira(tid)
                    ticket = {
                        "id": tid,
                        "title": proposal.get("title", f"{sprint_theme} — {dept}"),
                        "description": proposal.get("description", ""),
                        "status": "To Do",
                        "assignee": None,  # TicketAssigner owns this next morning
                        "dept": dept,
                        "sprint": sprint_num,
                        "sprint_theme": sprint_theme,
                        "story_points": proposal.get("story_points", 2),
                        "linked_prs": [],
                        "created_at": timestamp_str,
                        "updated_at": timestamp_str,
                    }
                    self._mem.upsert_ticket(ticket)
                    save_json(f"{BASE}/jira/{tid}.json", ticket)
                    self.state.sprint.tickets_in_sprint.append(tid)
                    dept_tickets.append(ticket)
                    self._embed_and_count(
                        id=tid,
                        type="jira",
                        title=ticket["title"],
                        content=json.dumps(ticket),
                        day=self.state.day,
                        date=date_str,
                        metadata={"dept": dept, "sprint_theme": sprint_theme},
                        timestamp=timestamp_str,
                    )
            return dept_tickets

        depts = [
            (dept, members) for dept, members in LIVE_ORG_CHART.items() if dept in LEADS
        ]
        with ThreadPoolExecutor(max_workers=len(depts)) as pool:
            futures = {
                pool.submit(_generate_dept_tickets, dept, members): dept
                for dept, members in depts
            }
            for future in as_completed(futures):
                dept = futures[future]
                try:
                    tickets = future.result()
                    all_new_tickets.extend(tickets)
                    logger.info(
                        f"    [green]✓ {dept}:[/green] {[t['id'] for t in tickets]}"
                    )
                except Exception as e:
                    logger.error(f"    [red]✗ {dept} ticket gen failed:[/red] {e}")

        for ticket in all_new_tickets:
            self._mem.log_event(
                SimEvent(
                    type="jira_ticket_created",
                    day=self.state.day,
                    date=date_str,
                    timestamp=timestamp_str,
                    actors=[LEADS.get(ticket["dept"], attendees[0])],
                    artifact_ids={"jira": ticket["id"]},
                    facts={
                        "sprint_number": sprint_num,
                        "sprint_theme": sprint_theme,
                        "title": ticket["title"],
                        "points": ticket["story_points"],
                        "dept": ticket["dept"],
                        "status": "To Do",
                    },
                    summary=f"[{ticket['id']}] {ticket['title']} ({ticket['dept']})",
                    tags=["jira", "sprint_backlog", ticket["dept"].lower()],
                )
            )

        self._mem.log_event(
            SimEvent(
                type="sprint_planned",
                day=self.state.day,
                date=date_str,
                timestamp=timestamp_str,
                actors=attendees,
                artifact_ids={"jira_tickets": [t["id"] for t in all_new_tickets]},
                facts={
                    "sprint_number": sprint_num,
                    "sprint_theme": sprint_theme,
                    "tickets": [
                        {
                            "id": t["id"],
                            "title": t["title"],
                            "dept": t["dept"],
                            "points": t["story_points"],
                        }
                        for t in all_new_tickets
                    ],
                    "total_points": sum(t["story_points"] for t in all_new_tickets),
                    "depts": list({t["dept"] for t in all_new_tickets}),
                },
                summary=f'Sprint #{sprint_num} planned: "{sprint_theme}" — {len(all_new_tickets)} tickets across {len(LIVE_ORG_CHART)} depts.',
                tags=["sprint", "planning"],
            )
        )

        self._record_daily_actor(*attendees)
        self._record_daily_event("sprint_planned")
        logger.info(
            f"    [bold green]✓ Sprint #{sprint_num} — {len(all_new_tickets)} tickets. Theme: {sprint_theme}[/bold green]"
        )

    def _handle_standup(self):
        logger.info("  [bold blue]☕ Multi-Agent Standup[/bold blue]")

        # Deriving all_names from the config provided in your files
        all_names = [n for dept in CONFIG["org_chart"].values() for n in dept]
        attendees = random.sample(all_names, min(8, len(all_names)))

        meeting_time = self._clock.schedule_meeting(
            attendees, min_hour=9, max_hour=10, duration_mins=15
        )
        meeting_time_iso = meeting_time.isoformat()
        date_str = str(self.state.current_date.date())

        messages = []
        for name in attendees:
            backstory = persona_backstory(
                name, mem=self._mem, graph_dynamics=self.graph_dynamics
            )

            # Tier 1: structured per-person query — no embedding.
            personal_ctx = self._mem.context_for_person(
                name=name,
                as_of_time=meeting_time_iso,
                n=3,
            )

            # Pull this engineer's owned tickets from SprintContext
            dept = dept_of_name(name, CONFIG["org_chart"])
            sprint_ctx = self.state.org_day_plan.sprint_contexts.get(dept)
            owned = (
                [
                    tid
                    for tid, owner in sprint_ctx.owned_tickets.items()
                    if owner == name
                ]
                if sprint_ctx
                else []
            )
            owned_str = (
                "\n".join(f"  - {tid}" for tid in owned)
                if owned
                else "  (no tickets assigned yet)"
            )

            standup_agent = make_agent(
                role=f"{name}, {dept} Team Member",
                goal=f"Write your morning standup update as {name} would.",
                backstory=backstory,
                llm=WORKER_MODEL,
            )
            task = Task(
                description=(
                    f"You are {name}. It's the morning standup.\n\n"
                    f"YOUR ASSIGNED TICKETS FOR THIS SPRINT:\n"
                    f"{owned_str}\n\n"
                    f"RULE: Only reference tickets from your assigned list above. "
                    f"Do not mention tickets owned by other people.\n\n"
                    f"Write your Slack update in 1-3 sentences. "
                    f"Use your typing quirks. Reflect your current stress and mood. "
                    f"Output the message only — no name prefix, no label.\n\n"
                    f"--- YOUR RECENT CONTEXT ---\n"
                    f"{personal_ctx}"
                ),
                expected_output=(
                    "A single Slack message, 1-3 sentences, no name prefix, no preamble."
                ),
                agent=standup_agent,
            )

            response = str(
                Crew(agents=[standup_agent], tasks=[task], verbose=False).kickoff()
            ).strip()

            messages.append(
                {
                    "user": name,
                    "text": response,
                    "ts": meeting_time_iso,
                    "thread_ts": meeting_time_iso,
                    "day": self.state.day,
                    "date": date_str,
                }
            )

        slack_path, thread_id = self._mem.log_slack_messages(
            "standup", messages, export_dir=EXPORT_DIR
        )

        self._mem.log_event(
            SimEvent(
                type="standup",
                timestamp=meeting_time_iso,
                day=self.state.day,
                date=date_str,
                actors=attendees,
                artifact_ids={"slack_path": slack_path, "slack_thread": thread_id},
                facts={"attendee_count": len(messages)},
                summary=f"Standup: {len(messages)} voices shared updates.",
                tags=["standup"],
            )
        )

        self._record_daily_actor(*attendees)
        self._record_daily_event("standup")

    # ─── RETROSPECTIVE ────────────────────────
    def _handle_retrospective(self):
        logger.info(
            f"  [bold blue]🔄 Retro — Sprint #{self.state.sprint.sprint_number}[/bold blue]"
        )

        sprint_num = self.state.sprint.sprint_number
        date_str = str(self.state.current_date.date())
        conf_id = self._registry.next_id("RETRO")
        self._registry.register_confluence(conf_id, f"Retro Sprint #{sprint_num}")

        # ── Attendees: engineering + product only ────────────────────────────────
        sprint_depts = {"engineering", "product"}
        attendees = [
            n for n in ALL_NAMES if dept_of_name(n, ORG_CHART).lower() in sprint_depts
        ]

        meeting_time = self._clock.schedule_meeting(
            attendees, min_hour=14, max_hour=16, duration_mins=60
        )
        meeting_time_iso = meeting_time.isoformat()

        # ── Sprint-bounded context only ──────────────────────────────────────────
        sprint_length = CONFIG["simulation"].get("sprint_length_days", 10)
        sprint_start_iso = self.state.sprint.start_date(
            self.state.current_date
            - timedelta(days=self.state.day - self.state.sprint.start_day),
            sprint_length,
        ).isoformat()
        # Tier 1: structured sprint-window query — no embedding.
        ctx = self._mem.context_for_retrospective(
            sprint_num=sprint_num,
            since_iso=sprint_start_iso,
            as_of_iso=meeting_time_iso,
        )

        # ── Participants ─────────────────────────────────────────────────────────
        scrum_master = resolve_role("scrum_master")
        eng_dept = next((d for d in LEADS if "engineering" in d.lower()), None)
        eng_lead = LEADS.get(eng_dept)
        product_dept = next((d for d in LEADS if "product" in d.lower()), None)
        product_lead = LEADS.get(product_dept)

        sprint_leads = [p for p in [scrum_master, eng_lead, product_lead] if p]

        # ── Per-voice agents ─────────────────────────────────────────────────────
        agents = []
        tasks = []
        prev_task = None

        role_prompts = {
            eng_lead: (
                "Engineering Lead",
                f"Reflect on Sprint #{sprint_num} from an engineering perspective. "
                f"Cover: velocity, incidents, technical debt surfaced, and any process "
                f"friction the team hit. Be specific — reference actual events where "
                f"possible. 3-5 bullet points.",
            ),
            product_lead: (
                "Product Manager",
                f"Reflect on Sprint #{sprint_num} from a product perspective. "
                f"Cover: whether sprint goals were met, any scope changes mid-sprint, "
                f"and customer or stakeholder signals that should shape the next sprint. "
                f"3-5 bullet points.",
            ),
            scrum_master: (
                "Scrum Master",
                f"You have heard from engineering and product. Now synthesize their input "
                f"into a Confluence retrospective document ({conf_id}) for Sprint #{sprint_num}.\n\n"
                f"Sections:\n"
                f"## What Went Well\n"
                f"## What Didn't\n"
                f"## Action Items (owner + due sprint)\n\n"
                f"System health: {self.state.system_health}/100. "
                f"Team morale: {self.state.team_morale:.2f}.\n"
                f"Ground action items in the problems raised — no generic platitudes.",
            ),
        }

        for name in sprint_leads:
            role_label, desc = role_prompts.get(
                name, ("Team Member", "Share your sprint reflections.")
            )

            agent = make_agent(
                role=role_label,
                goal=f"Contribute authentically to the Sprint #{sprint_num} retrospective.",
                backstory=persona_backstory(
                    name, self._mem, graph_dynamics=self.graph_dynamics
                ),
                llm=PLANNER_MODEL,
            )
            task = Task(
                description=f"Context from this sprint:\n{ctx}\n\n{desc}",
                expected_output="Markdown contribution to the retrospective.",
                agent=agent,
                context=[prev_task] if prev_task else [],
            )

            agents.append(agent)
            tasks.append(task)
            prev_task = task

        content = str(
            Crew(
                agents=agents, tasks=tasks, process=Process.sequential, verbose=False
            ).kickoff()
        )

        # ── Persist ──────────────────────────────────────────────────────────────
        path = f"{BASE}/confluence/retros/{conf_id}.md"
        save_md(path, content)

        self._embed_and_count(
            id=conf_id,
            type="confluence",
            title=f"Retro Sprint #{sprint_num}",
            content=content,
            day=self.state.day,
            date=date_str,
            timestamp=meeting_time_iso,
        )

        self._mem.log_event(
            SimEvent(
                type="retrospective",
                timestamp=meeting_time_iso,
                day=self.state.day,
                date=date_str,
                actors=attendees,
                artifact_ids={"confluence": conf_id},
                facts={
                    "sprint_number": sprint_num,
                    "system_health": self.state.system_health,
                    "team_morale": self.state.team_morale,
                    "sprint_start": sprint_start_iso,
                },
                summary=f"Sprint #{sprint_num} retrospective.",
                tags=["retrospective", "sprint"],
            )
        )

        # ── Close sprint ─────────────────────────────────────────────────────────
        self._close_sprint()
        self._record_daily_actor(*attendees)
        self._record_daily_event("retrospective")
        logger.info(f"    [green]✓[/green] {conf_id}")

    def _close_sprint(self) -> None:
        """Advance sprint counter and reset per-sprint state."""
        self.state.sprint.sprint_number += 1
        self.state.sprint.tickets_in_sprint = []

    def _handle_incident(self):
        ticket_id = next_jira_id(self.state, self._registry)
        on_call = resolve_role("on_call_engineer")
        incident_lead = resolve_role("incident_commander")
        eng_peer = next(
            (
                n
                for n in ORG_CHART.get(CONFIG["roles"].get("on_call_engineer", ""), [])
                if n != on_call
            ),
            on_call,
        )

        incident_start = self._clock.tick_system(min_mins=30, max_mins=240)
        incident_start_iso = incident_start.isoformat()
        date_str = str(self.state.current_date.date())

        self._clock.sync_to_system([on_call])

        # ── 1. Root cause ─────────────────────────────────────────────────────
        rc_agent = make_agent(
            role=f"{on_call}, Senior On-Call Engineer",
            goal=f"Diagnose the root cause of today's incident as {on_call}, given what you know about this system.",
            backstory=persona_backstory(
                on_call, self._mem, graph_dynamics=self.graph_dynamics
            ),
            llm=PLANNER_MODEL,
        )

        if self.state.system_health < 40:
            length_instruction = (
                "2-3 sentences. The system is in bad shape — "
                "the root cause may involve multiple interacting failures."
            )
        elif self.state.system_health < 70:
            length_instruction = (
                "1-2 sentences. Describe the specific failure and what triggered it."
            )
        else:
            length_instruction = (
                "1 sentence, 25 words or fewer. "
                "This is likely an isolated, straightforward failure."
            )
        rc_task = Task(
            description=(
                f"You are {on_call}. You are on-call and an incident just fired.\n\n"
                f"System health: {self.state.system_health}/100\n"
                f"Today's org theme: {self.state.daily_theme}\n\n"
                f"Write the root cause. Length: {length_instruction}\n"
                f"Reference a real system component, endpoint, or dependency. "
                f"No preamble, no label — just the root cause."
            ),
            expected_output=(
                f"{length_instruction} No label like 'Root cause:'. No preamble. "
                f"Example (1 sentence): 'Redis TTL misconfiguration caused auth token cache stampede under load.' "
                f"Example (2 sentences): 'A deploy at 14:32 introduced a missing null check in the payments service. "
                f"Under load, this caused a cascade of 500s that overwhelmed the retry queue.'"
            ),
            agent=rc_agent,
        )
        root_cause = str(
            Crew(agents=[rc_agent], tasks=[rc_task], verbose=False).kickoff()
        ).strip()

        # ── 2. Knowledge gap detection ────────────────────────────────────────
        involves_gap = any(
            k.lower() in root_cause.lower()
            for emp in DEPARTED_EMPLOYEES.values()
            for k in emp["knew_about"]
        )

        # Build detailed gap context for description + embedding
        gap_areas: List[str] = []
        gap_context_str: str = ""
        if involves_gap:
            departed_details: List[str] = []
            for emp_name, emp in DEPARTED_EMPLOYEES.items():
                hits = [k for k in emp["knew_about"] if k.lower() in root_cause.lower()]
                if hits:
                    gap_areas.extend(hits)
                    departed_details.append(
                        f"{emp_name} (ex-{emp['role']}, left {emp['left']}, "
                        f"{int(emp['documented_pct'] * 100)}% documented) "
                        f"owned: {', '.join(hits)}"
                    )
            if departed_details:
                gap_context_str = (
                    f"KNOWLEDGE GAP FLAG: This incident touches underdocumented systems. "
                    f"{' | '.join(departed_details)}. "
                    f"Resolution may be blocked pending knowledge recovery."
                )

        self._lifecycle.scan_for_knowledge_gaps(
            text=root_cause,
            triggered_by=ticket_id,
            day=self.state.day,
            date_str=date_str,
            state=self.state,
            timestamp=incident_start_iso,
        )

        # ── 3. Recurrence detection ───────────────────────────────────────────
        prior = self._recurrence_detector.find_prior_incident(
            root_cause, self.state.day, ticket_id
        )
        recurrence_of = prior.artifact_ids.get("jira") if prior else None
        recurrence_gap = (self.state.day - prior.day) if prior else None
        prior_postmortem = (
            self._recurrence_detector.find_postmortem_for_ticket(recurrence_of)
            if recurrence_of
            else None
        )

        recurrence_str = (
            f"Prior occurrence: {recurrence_of} — {recurrence_gap} days ago."
            if recurrence_of
            else "First occurrence of this issue."
        )

        # ── 4. Escalation chain ───────────────────────────────────────────────
        gap_kw = [k for emp in DEPARTED_EMPLOYEES.values() for k in emp["knew_about"]]
        chain = self.graph_dynamics.build_escalation_chain(
            first_responder=on_call,
            domain_keywords=gap_kw if involves_gap else None,
        )
        escalation_narrative = self.graph_dynamics.escalation_narrative(chain)
        escalation_actors = [n for n, _ in chain.chain]

        # ── 5. Bot alerts — capture thread ids before ticket is built ─────────
        datadog_text = (
            f"🚨 [CRITICAL] Anomaly detected: {root_cause[:40]}... "
            f"Error rate spiked 400%. System health dropped to {self.state.system_health}."
        )
        pagerduty_text = (
            f"📞 Paging on-call engineer: {on_call}. Incident linked to [{ticket_id}]."
        )
        datadog_thread = self._emit_bot_message(
            "system-alerts", "Datadog", datadog_text, incident_start_iso
        )
        pagerduty_thread = self._emit_bot_message(
            "incidents", "PagerDuty", pagerduty_text, incident_start_iso
        )

        self.state.system_health = max(0, self.state.system_health - 15)

        # ── 6. Causal chain — start it now, append as artifacts are created ───
        chain_handler = CausalChainHandler(ticket_id)
        chain_handler.append(datadog_thread)
        chain_handler.append(pagerduty_thread)

        # ── 7. Generate ticket description — all context is available now ─────
        title = f"{LEGACY['name']}: {self.state.daily_theme[:60]}"
        desc_agent = make_agent(
            role="Senior Engineer",
            goal="Write a concise Jira ticket description for an incident.",
            backstory=persona_backstory(
                on_call, self._mem, graph_dynamics=self.graph_dynamics
            ),
            llm=WORKER_MODEL,
        )
        desc_task = Task(
            description=(
                f"Write a Jira ticket description for this incident.\n\n"
                f"Title: {title}\n"
                f"Root cause: {root_cause}\n"
                f"Escalation path: {escalation_narrative}\n"
                f"System health at incident open: {self.state.system_health}/100\n"
                f"{gap_context_str}\n"
                f"{recurrence_str}\n\n"
                f"Include:\n"
                f"  - What is broken and how it manifests (1-2 sentences)\n"
                f"  - Which system or component is affected\n"
                f"  - User or business impact\n"
                f"{'  - Note the knowledge gap and documentation risk explicitly\n' if gap_areas else ''}"
                f"  - One acceptance criterion for resolution\n\n"
                f"Keep it under 100 words. Write as {on_call} would in Jira."
            ),
            expected_output="A Jira ticket description under 100 words.",
            agent=desc_agent,
        )
        description = str(
            Crew(agents=[desc_agent], tasks=[desc_task], verbose=False).kickoff()
        ).strip()

        # ── 8. Build ticket with full context ─────────────────────────────────
        ticket = {
            "id": ticket_id,
            "title": title,
            "description": description,
            "status": "In Progress",
            "assignee": on_call,
            "root_cause": root_cause,
            "linked_prs": [],
            "dept": dept_of(on_call),
            "sprint": self.state.sprint.sprint_number,
            # Causal chain
            "causal_chain": chain_handler.snapshot(),
            "bot_threads": [datadog_thread, pagerduty_thread],
            # Recurrence
            "recurrence_of": recurrence_of,
            "recurrence_gap_days": recurrence_gap,
            "prior_postmortem": prior_postmortem,
            # Knowledge gap
            "gap_areas": gap_areas,
            # Escalation
            "escalation_actors": escalation_actors,
            "escalation_narrative": escalation_narrative,
        }

        self._mem.upsert_ticket(ticket)
        save_json(f"{BASE}/jira/{ticket_id}.json", ticket)

        # ── 9. Embed — now maximally rich, called last ────────────────────────
        embed_content = "\n\n".join(
            filter(
                None,
                [
                    title,
                    description,
                    f"Root cause: {root_cause}",
                    f"Escalation: {escalation_narrative}",
                    f"Knowledge gap areas: {', '.join(gap_areas)}" if gap_areas else "",
                    f"Recurrence of: {recurrence_of} ({recurrence_gap} days ago)"
                    if recurrence_of
                    else "",
                ],
            )
        )

        self._embed_and_count(
            id=ticket_id,
            type="jira",
            title=title,
            content=embed_content,
            day=self.state.day,
            date=date_str,
            timestamp=incident_start_iso,
            metadata={
                "assignee": on_call,
                "dept": dept_of(on_call),
                "recurrence_of": recurrence_of,
                "has_recurrence": recurrence_of is not None,
                "has_gap": bool(gap_areas),
                "gap_areas": gap_areas,
            },
        )

        # ── 10. Activate incident — chain_handler travels with it ─────────────
        inc = ActiveIncident(
            ticket_id=ticket_id,
            title=title,
            day_started=self.state.day,
            involves_gap_knowledge=involves_gap,
            root_cause=root_cause,
            causal_chain=chain_handler,
            recurrence_of=recurrence_of,
        )
        self.state.active_incidents.append(inc)
        self.state.daily_incidents_opened += 1

        # ── 11. External contacts ─────────────────────────────────────────────
        triggered_contacts = self.graph_dynamics.relevant_external_contacts(
            event_type="incident_opened",
            system_health=self.state.system_health,
            config=CONFIG,
        )
        for contact in triggered_contacts:
            self._handle_external_contact(inc, contact)

        # ── 12. SimEvents ─────────────────────────────────────────────────────
        self._mem.log_event(
            SimEvent(
                type="incident_opened",
                timestamp=incident_start_iso,
                day=self.state.day,
                date=date_str,
                actors=[on_call, incident_lead],
                artifact_ids={
                    ARTIFACT_KEY_JIRA: ticket_id,
                    ARTIFACT_KEY_SLACK_THREAD: pagerduty_thread,
                },
                facts={
                    "title": title,
                    "root_cause": root_cause,
                    "involves_gap": involves_gap,
                    "gap_areas": gap_areas,
                    "causal_chain": chain_handler.snapshot(),
                    "recurrence_of": recurrence_of,
                    "recurrence_gap_days": recurrence_gap,
                    "prior_postmortem": prior_postmortem,
                    "bot_threads": [datadog_thread, pagerduty_thread],
                },
                summary=f"P1 incident {ticket_id}: {root_cause}",
                tags=["incident", "P1"]
                + (["knowledge_gap"] if involves_gap else [])
                + (["recurrence"] if recurrence_of else []),
            )
        )

        self._mem.log_event(
            SimEvent(
                type="escalation_chain",
                timestamp=incident_start_iso,
                day=self.state.day,
                date=date_str,
                actors=escalation_actors,
                artifact_ids={ARTIFACT_KEY_JIRA: ticket_id},
                facts={
                    "chain": chain.chain,
                    "narrative": escalation_narrative,
                },
                summary=escalation_narrative,
                tags=["escalation", "incident"],
            )
        )

        if involves_gap:
            self._mem.log_event(
                SimEvent(
                    type="knowledge_gap_detected",
                    timestamp=incident_start_iso,
                    day=self.state.day,
                    date=date_str,
                    actors=[on_call, eng_peer],
                    artifact_ids={ARTIFACT_KEY_JIRA: ticket_id},
                    facts={
                        "gap_area": gap_areas or [LEGACY["name"]],
                        "involves_gap": True,
                        "gap_context": gap_context_str,
                    },
                    summary=f"Knowledge gap detected during {ticket_id}: {gap_context_str[:80]}",
                    tags=["knowledge_gap"],
                )
            )

        # ── 13. Graph dynamics ────────────────────────────────────────────────
        self._record_daily_actor(on_call, incident_lead)
        self._record_daily_event("incident_opened")
        self.graph_dynamics.apply_incident_stress([on_call, incident_lead])
        self.graph_dynamics.record_incident_collaboration([on_call, incident_lead])

        # ── 14. Dept plan pressure ────────────────────────────────────────────
        if self.state.org_day_plan:
            eng_key = next(
                (k for k in self.state.org_day_plan.dept_plans if "eng" in k.lower()),
                None,
            )
            if eng_key:
                eng_dept_plan = self.state.org_day_plan.dept_plans[eng_key]
                primary_hrs_lost = round(random.uniform(2.0, 5.5), 1)
                peer_hrs_lost = round(primary_hrs_lost * random.uniform(0.2, 0.6), 1)

                for ep in eng_dept_plan.engineer_plans:
                    if ep.name in [on_call, incident_lead]:
                        ep.apply_incident_pressure(inc.title, hrs_lost=primary_hrs_lost)
                    elif ep.name == eng_peer:
                        ep.apply_incident_pressure(inc.title, hrs_lost=peer_hrs_lost)

        logger.info(f"    [red]🚨 {ticket_id}:[/red] {root_cause[:65]}")

    def _advance_incidents(self):
        still_active = []
        on_call = resolve_role("on_call_engineer")
        eng_peer = next(
            (
                n
                for n in ORG_CHART.get(CONFIG["roles"].get("on_call_engineer", ""), [])
                if n != on_call
            ),
            on_call,
        )

        cron_time_iso = self._clock.now("system").isoformat()

        for inc in self.state.active_incidents:
            if inc.stage == "detected":
                inc.stage = "investigating"
                still_active.append(inc)

            elif inc.stage == "investigating":
                inc.stage = "fix_in_progress"
                pr = self._git.create_pr(
                    author=on_call,
                    ticket_id=inc.ticket_id,
                    title=f"[{inc.ticket_id}] Fix: {inc.root_cause[:60]}",
                    timestamp=cron_time_iso,
                )

                t = self._mem.get_ticket(inc.ticket_id)
                if t:
                    if pr["pr_id"] not in t.get("linked_prs", []):
                        t.setdefault("linked_prs", []).append(pr["pr_id"])
                    t["updated_at"] = cron_time_iso
                    self._mem.upsert_ticket(t)
                    save_json(f"{BASE}/jira/{inc.ticket_id}.json", t)

                self._emit_bot_message(
                    "engineering",
                    "GitHub",
                    f"🛠️ {on_call} opened PR {pr['pr_id']}: [{inc.ticket_id}] Fix. Reviewers requested: {', '.join(pr['reviewers'])}.",
                    cron_time_iso,
                )
                inc.pr_id = pr["pr_id"]
                if getattr(inc, "causal_chain", None):
                    inc.causal_chain.append(pr["pr_id"])

                logger.info(
                    f"    [yellow]🔧 {inc.ticket_id}:[/yellow] {pr['pr_id']} opened."
                )
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
                    cron_time_iso,
                )
                self.state.system_health = min(100, self.state.system_health + 20)
                self._write_postmortem(inc)
                self.state.resolved_incidents.append(inc.ticket_id)
                self.state.daily_incidents_resolved += 1
                self._mem.log_event(
                    SimEvent(
                        type="incident_resolved",
                        timestamp=cron_time_iso,
                        day=self.state.day,
                        date=str(self.state.current_date.date()),
                        actors=[on_call, eng_peer],
                        artifact_ids={"jira": inc.ticket_id, "pr": inc.pr_id or ""},
                        facts={
                            "root_cause": inc.root_cause,
                            "duration_days": inc.days_active,
                            "causal_chain": inc.causal_chain.snapshot()
                            if getattr(inc, "causal_chain", None)
                            else [],
                        },
                        summary=f"{inc.ticket_id} resolved in {inc.days_active}d.",
                        tags=["incident_resolved"],
                    )
                )
                logger.info(f"    [green]✅ {inc.ticket_id} resolved.[/green]")
                # Resolved — intentionally not appended to still_active

            else:
                # Unknown stage — keep active to avoid silent data loss
                still_active.append(inc)

        self.state.active_incidents = still_active

    def _write_postmortem(self, inc: ActiveIncident):
        on_call = resolve_role("postmortem_writer")
        eng_peer = next(
            (
                n
                for n in ORG_CHART.get(CONFIG["roles"].get("postmortem_writer", ""), [])
                if n != on_call
            ),
            on_call,
        )
        conf_id = self._confluence.write_postmortem(
            incident_id=inc.ticket_id,
            incident_title=inc.title,
            root_cause=inc.root_cause,
            days_active=inc.days_active,
            on_call=on_call,
            eng_peer=eng_peer,
        )

        if conf_id and getattr(inc, "causal_chain", None):
            inc.causal_chain.append(conf_id)

    def _emit_bot_message(self, channel: str, bot_name: str, text: str, timestamp: str):
        date_str = str(self.state.current_date.date())
        message = {
            "user": bot_name,
            "email": f"{bot_name.lower()}@bot.{COMPANY_DOMAIN}",
            "text": text,
            "ts": timestamp,
            "date": date_str,
            "is_bot": True,
        }

        slack_path, thread_id = self._mem.log_slack_messages(
            channel=channel,
            messages=[message],
            export_dir=EXPORT_DIR,
        )

        return thread_id

    def _generate_adhoc_confluence_page(
        self,
        author: Optional[str] = None,
        backstory: Optional[str] = None,
    ):
        self._confluence.write_adhoc_page(author=author, backstory=backstory)

    # ─── END OF DAY ───────────────────────────
    def _end_of_day(self):
        date_str = str(self.state.current_date.date())
        decay = CONFIG["morale"]["daily_decay"]
        recovery = CONFIG["morale"]["good_day_recovery"]

        self.state.team_morale = round(self.state.team_morale * decay, 3)
        if not self.state.active_incidents:
            self.state.team_morale = round(
                min(1.0, self.state.team_morale + recovery), 3
            )

        self.state.morale_history.append(self.state.team_morale)

        all_cursors = [self._clock.now(a) for a in ALL_NAMES]
        latest_time_worked = (
            max(all_cursors) if all_cursors else self.state.current_date
        )

        # Ensure the summary doesn't happen before 17:30
        eod_baseline = self.state.current_date.replace(hour=17, minute=30, second=0)
        summary_time = max(latest_time_worked, eod_baseline)
        housekeeping_time = summary_time + timedelta(minutes=1)

        # ── end_of_day event (unchanged) ─────────────────────────────────────────
        self._mem.log_event(
            SimEvent(
                type="end_of_day",
                timestamp=housekeeping_time.isoformat(),
                day=self.state.day,
                date=date_str,
                actors=[],
                artifact_ids={},
                facts={
                    "morale": self.state.team_morale,
                    "system_health": self.state.system_health,
                },
                summary=f"Day {self.state.day} end.",
                tags=["eod"],
            )
        )

        # ── Derive enrichment fields from accumulated daily state ─────────────────

        # Deduplicated actors seen in any event today, ordered by frequency
        unique_actors = list(dict.fromkeys(self.state.daily_active_actors))

        # Dominant event type fired most often today (e.g. "incident_opened")
        event_counts = self.state.daily_event_type_counts
        dominant_event = (
            max(event_counts, key=event_counts.get) if event_counts else "normal_day"
        )

        # Departments represented by today's active actors
        departments_involved = list(
            {dept_of(name) for name in unique_actors if dept_of(name) != "Unknown"}
        )

        # Still-open incidents at EOD (not yet resolved)
        open_incident_ids = [inc.ticket_id for inc in self.state.active_incidents]

        # Stress snapshot for today's active actors only — keeps the summary tight
        stress_today = dict(
            {name: self.graph_dynamics._stress.get(name, 0) for name in unique_actors}
        )

        # ── Enriched day_summary SimEvent ────────────────────────────────────────
        self._mem.log_event(
            SimEvent(
                type="day_summary",
                timestamp=housekeeping_time.isoformat(),
                day=self.state.day,
                date=date_str,
                actors=unique_actors,  # populated — was always []
                artifact_ids={},
                facts={
                    # ── Original numeric fields (unchanged) ──
                    "incidents_opened": self.state.daily_incidents_opened,
                    "incidents_resolved": self.state.daily_incidents_resolved,
                    "artifacts_created": self.state.daily_artifacts_created,
                    "external_contacts": self.state.daily_external_contacts,
                    "morale": self.state.team_morale,
                    "system_health": self.state.system_health,
                    "theme": self.state.daily_theme,
                    # ── New enrichment fields ──
                    "active_actors": unique_actors,
                    "dominant_event": dominant_event,
                    "event_type_counts": dict(self.state.daily_event_type_counts),
                    "departments_involved": departments_involved,
                    "open_incidents": open_incident_ids,
                    "stress_snapshot": stress_today,
                    # Trajectory signal — gives DayPlanner a one-glance health picture
                    "health_trend": (
                        "declining"
                        if self.state.system_health < 60
                        else "recovering"
                        if self.state.daily_incidents_resolved
                        > self.state.daily_incidents_opened
                        else "stable"
                    ),
                    "morale_trend": (
                        "low"
                        if self.state.team_morale < 0.45
                        else "moderate"
                        if self.state.team_morale < 0.70
                        else "healthy"
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
            )
        )

        # ── Reset all daily counters ──────────────────────────────────────────────
        self.state.daily_incidents_opened = 0
        self.state.daily_incidents_resolved = 0
        self.state.daily_artifacts_created = 0
        self.state.daily_external_contacts = 0
        self.state.daily_active_actors = []
        self.state.daily_event_type_counts = {}

        date_str = str(self.state.current_date.date())
        for dep in self._lifecycle._departed:
            if dep.day != self.state.day:
                continue  # only process today's departures
            # Pick the on-call engineer or first dept member as first responder
            dept_members = [
                n for n in LIVE_ORG_CHART.get(dep.dept, []) if n != dep.name
            ]
            if dept_members:
                note = recompute_escalation_after_departure(
                    self.graph_dynamics,
                    departed=dep,
                    first_responder=dept_members[0],
                )
                self._mem.log_event(
                    SimEvent(
                        type="escalation_chain",
                        timestamp=housekeeping_time.isoformat(),
                        day=self.state.day,
                        date=date_str,
                        actors=dept_members[:2],
                        artifact_ids={},
                        facts={
                            "trigger": "post_departure_reroute",
                            "departed": dep.name,
                            "new_path_note": note,
                        },
                        summary=f"Escalation path updated after {dep.name} departure. {note}",
                        tags=["escalation_chain", "lifecycle"],
                    )
                )

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
        logger.info(
            f"\n[bold]Day {self.state.day}[/bold] [dim]({self.state.current_date.strftime('%a %b %d')})[/dim]  ❤️  {'█' * h}{'░' * (10 - h)} {self.state.system_health}   😊 {'█' * m}{'░' * (10 - m)} {self.state.team_morale:.2f}\n  [italic]{self.state.daily_theme}[/italic]"
        )

    def _print_final_report(self):
        s = self._mem.stats()
        table = Table(title="Simulation Complete", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        for row in [
            (
                "Confluence Pages",
                str(self._mem._artifacts.count_documents({"type": "confluence"})),
            ),
            ("JIRA Tickets", str(self._mem._jira.count_documents({}))),
            ("Slack Threads", str(self._mem._slack.count_documents({}))),
            ("Git PRs", str(self._mem._prs.count_documents({}))),
            ("Incidents Resolved", str(len(self.state.resolved_incidents))),
            ("Embedded Artifacts", str(s["artifact_count"])),
            ("Employees Departed", str(len(self._lifecycle._departed))),
            ("Employees Hired", str(len(self._lifecycle._hired))),
            ("Knowledge Gaps Surfaced", str(len(self._lifecycle._gap_events))),
            ("MongoDB Active", "✓" if s["mongodb_ok"] else "⚠"),
        ]:
            table.add_row(*row)
        logger.info("\n")
        logger.info(table)

        _proj = {"_id": 0, "embedding": 0}
        snapshot = {
            "confluence_pages": list(
                self._mem._artifacts.find({"type": "confluence"}, _proj)
            ),
            "jira_tickets": list(self._mem._jira.find({}, {"_id": 0})),
            "slack_threads": list(self._mem._slack.find({}, {"_id": 0})),
            "pr_registry": list(self._mem._prs.find({}, {"_id": 0})),
            "resolved_incidents": self.state.resolved_incidents,
            "morale_history": self.state.morale_history,
            "system_health": self.state.system_health,
            "event_log": [e.to_dict() for e in self._mem.get_event_log()],
        }
        snapshot["top_relationships"] = self.graph_dynamics.relationship_summary(10)
        snapshot["estranged_pairs"] = self.graph_dynamics.estranged_pairs()
        snapshot["departed_employees"] = [
            {
                "name": d.name,
                "dept": d.dept,
                "day": d.day,
                "reason": d.reason,
                "knowledge_domains": d.knowledge_domains,
                "documented_pct": d.documented_pct,
                "peak_stress": d.peak_stress,
            }
            for d in self._lifecycle._departed
        ]
        snapshot["new_hires"] = [
            {
                "name": h.name,
                "dept": h.dept,
                "day": h.day,
                "role": h.role,
                "expertise": h.expertise,
                "warm_edges_at_end": sum(
                    1
                    for nb in self.social_graph.neighbors(h.name)
                    if self.social_graph.has_node(h.name)
                    and self.social_graph[h.name][nb].get("weight", 0)
                    >= h.warmup_threshold
                )
                if self.social_graph.has_node(h.name)
                else 0,
            }
            for h in self._lifecycle._hired
        ]
        snapshot["knowledge_gap_events"] = [
            {
                "departed": g.departed_name,
                "domain": g.domain_hit,
                "triggered_by": g.triggered_by,
                "day": g.triggered_on_day,
                "documented_pct": g.documented_pct,
            }
            for g in self._lifecycle._gap_events
        ]
        snapshot["stress_snapshot"] = (
            self._last_stress_prop.stress_snapshot
            if hasattr(self, "_last_stress_prop")
            else {}
        )
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
        tone = contact.get("summary_tone", "professional")
        date_str = str(self.state.current_date.date())

        participants = [liaison_name, display_name]
        interaction_mins = random.randint(15, 45)
        interaction_hours = interaction_mins / 60.0

        start_time, end_time = self._clock.sync_and_advance(
            participants, hours=interaction_hours
        )
        interaction_start_iso = start_time.isoformat()

        # Boost the edge between liaison and external node — they just talked
        external_node = contact["name"]
        if self.social_graph.has_edge(liaison_name, external_node):
            self.graph_dynamics.record_incident_collaboration(
                [liaison_name, external_node]
            )

        # Tier 1: structured incident fetch — no embedding.
        ctx = self._mem.context_for_incident(
            ticket_id=inc.ticket_id,
            as_of_time=interaction_start_iso,
        )

        agent = make_agent(
            role="Employee",
            goal="Summarize an external conversation for your team on Slack.",
            backstory=persona_backstory(
                liaison_name,
                self._mem,
                extra=self.graph_dynamics.stress_tone_hint(liaison_name),
            ),
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
            "user": liaison_name,
            "email": email_of(liaison_name),
            "text": summary_text,
            "ts": self.state.current_date.replace(
                hour=random.randint(10, 16), minute=random.randint(0, 59)
            ).isoformat(),
            "is_bot": False,
            "metadata": {
                "type": "external_contact_summary",
                "external_party": display_name,
                "org": contact.get("org", "External"),
                "incident": inc.ticket_id,
            },
        }

        message["date"] = date_str
        slack_path, thread_id = self._mem.log_slack_messages(
            channel="incidents",
            messages=[message],
            export_dir=EXPORT_DIR,
        )

        if thread_id and getattr(inc, "causal_chain", None):
            inc.causal_chain.append(thread_id)

        # SimEvent — this is what makes it retrievable as ground truth
        self._mem.log_event(
            SimEvent(
                type="external_contact_summarized",
                timestamp=interaction_start_iso,
                day=self.state.day,
                date=date_str,
                actors=[liaison_name, external_node],
                artifact_ids={
                    "slack_thread": thread_id,
                    "slack_path": slack_path,
                    "jira": inc.ticket_id,
                },
                facts={
                    "external_party": display_name,
                    "org": contact.get("org", "External"),
                    "incident": inc.ticket_id,
                    "root_cause": inc.root_cause,
                    "liaison": liaison_name,
                    "summary_tone": tone,
                },
                summary=(
                    f"{liaison_name} summarized {display_name} contact re "
                    f"{inc.ticket_id} in #incidents."
                ),
                tags=["external", "slack", "incident"],
            )
        )
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
                "liaison": liaison_name,
                "incident": inc.ticket_id,
            },
            timestamp=interaction_start_iso,
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
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe MongoDB and export dir before running",
    )
    args = parser.parse_args()

    flow = Flow()
    if args.reset:
        flow._mem.reset(export_dir=BASE)
    flow.kickoff()
