"""
confluence_writer.py
Single source of truth for all Confluence page generation in OrgForge.

Every path that produces a Confluence artifact (genesis, postmortems,
design doc stubs, ad-hoc pages) runs through this module.

Knowledge gap detection is fully deterministic and engine-controlled:
  - _compute_domain_fit() checks the author's expertise against orphaned
    domains in the registry using system_tags and documentation_coverage
    thresholds. No LLM involvement.
  - scan_for_knowledge_gaps() (on org_lifecycle.py) uses BM25 text search
    over persona_skill artifacts cross-referenced with the DomainRegistry.
    No LLM involvement.

The LLM's only job is to produce prose and alias vocabulary. It never
assesses its own knowledge gaps, domain fit, or confidence.

Alias vocabulary is emitted by the LLM and stored as a List[str] of domain
terms. Each term is indexed by Atlas Search for BM25 retrieval so vocabulary
captured at write time grows with each artifact.
"""

from __future__ import annotations

from datetime import datetime
import json
import logging
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from agent_factory import make_agent
from causal_chain_handler import CausalChainHandler
from config_loader import COMPANY_DESCRIPTION, PERSONAS
from crewai import Task, Crew
from memory import Memory, SimEvent
from artifact_registry import ArtifactRegistry, ConfluencePage
from utils.persona_utils import persona_utils

if TYPE_CHECKING:
    from graph_dynamics import GraphDynamics

logger = logging.getLogger("orgforge.confluence")

_CONF_PREFIX_MAP = {
    "Engineering_Backend": "ENG",
    "Engineering_Mobile": "ENG",
    "Product": "PROD",
    "Design": "DESIGN",
    "Sales_Marketing": "SALES",
    "HR_Ops": "HR",
    "QA_Support": "QA",
}

_ALIAS_INSTRUCTION = (
    "\n\n### ALIAS VOCABULARY\n"
    "List 4-10 short lowercase domain terms that someone searching for this \n"
    "page's topic would use (system names, component names, protocol names,\n"
    "abbreviated names. NOT generic words like 'overview' or 'documentation').\n"
)

_ALIAS_JSON_FIELDS = '  "aliases": ["string"]\n'


def _extract_aliases(parsed: dict) -> Optional[List[str]]:
    """
    Pull the aliases array from a parsed LLM JSON response.
    Returns a list of lowercase terms, or None if absent/malformed.
    """
    raw = parsed.get("aliases")
    if not isinstance(raw, list):
        return None
    clean_terms = [t.lower().strip() for t in raw if isinstance(t, str) and t.strip()]
    return clean_terms if clean_terms else None


class ConfluenceWriter:
    def __init__(
        self,
        mem: Memory,
        registry: ArtifactRegistry,
        state,
        config: Dict,
        worker_llm,
        planner_llm,
        clock,
        lifecycle,
        persona_helper,
        graph_dynamics: "GraphDynamics",
        base_export_dir: str = "./export",
    ):
        self._mem = mem
        self._registry = registry
        self._state = state
        self._config = config
        self._worker = worker_llm
        self._planner = planner_llm
        self._clock = clock
        self._lifecycle = lifecycle
        self._persona = persona_helper
        self._gd = graph_dynamics
        self._base = base_export_dir
        self._company = config["simulation"]["company_name"]
        self._company_description = config["simulation"]["company_description"]
        self._industry = config["simulation"].get("industry", "technology")
        self._legacy = config.get("legacy_system", {})
        self._all_names = [n for dept in config["org_chart"].values() for n in dept]
        self._org_chart = config["org_chart"]

    def write_genesis_batch(
        self,
        prefix: str,
        count: int,
        prompt_tpl: str,
        author: str,
        extra_vars: Optional[Dict[str, str]] = None,
        subdir: str = "archives",
        tags: Optional[List[str]] = None,
        page_date: Optional[datetime] = None,
    ) -> List[str]:
        """
        Generate *count* independent genesis Confluence pages for a given prefix.
        Each page is generated in a separate LLM call. The alias vocabulary is
        extracted from the same JSON output as the page content.
        """
        genesis_time = self._clock.now("system").isoformat()
        queue = [self._registry.next_id(prefix) for _ in range(count)]
        registered_ids: List[str] = []
        processed: set = set()

        while queue:
            conf_id = queue.pop(0)
            if conf_id in processed:
                continue
            processed.add(conf_id)

            related = ", ".join(registered_ids) if registered_ids else "None yet."

            vars_ = {
                "id": conf_id,
                "company": self._company,
                "industry": self._industry,
                "legacy_system": self._legacy.get("name", ""),
                "project_name": self._legacy.get("project_name", ""),
                "author": author,
                "related_pages": related,
                **(extra_vars or {}),
            }
            prompt = self._render(prompt_tpl, vars_)
            prompt += (
                f"\n\nThis page's ID is {conf_id}. "
                f"You may ONLY reference pages that already exist: {related}. "
                f"Do NOT invent or reference any other CONF-* IDs."
            )

            if not author or author == "system":
                author = self._pick_dept_author(prefix)

            historian = make_agent(
                role=f"{author}, {prefix} Department",
                goal="Write one authentic internal Confluence page as yourself.",
                backstory=self._persona(author, mem=self._mem, graph_dynamics=self._gd),
                llm=self._planner,
            )
            task = Task(
                description=(
                    f"{prompt}\n\n"
                    f"{_ALIAS_INSTRUCTION}"
                    f"\nRespond ONLY with valid JSON:\n"
                    f"{{\n"
                    f'  "markdown_doc": "full Markdown page content, no # title",\n'
                    f"{_ALIAS_JSON_FIELDS}"
                    f"}}"
                ),
                expected_output=(f"Valid JSON with markdown_doc and aliases keys."),
                agent=historian,
            )

            raw_output = str(
                Crew(agents=[historian], tasks=[task], verbose=False).kickoff()
            ).strip()

            clean = raw_output.replace("```json", "").replace("```", "").strip()
            content = raw_output
            aliases: Optional[List[str]] = None

            try:
                parsed = json.loads(clean)
                content = parsed.get("markdown_doc", raw_output)
                aliases = _extract_aliases(parsed)
            except (json.JSONDecodeError, ValueError):
                content = raw_output

            resolved_tags = tags or ["genesis", "confluence"]

            conf_ids = self._finalize_page(
                raw_content=content,
                conf_id=conf_id,
                title=self._extract_title(content, conf_id),
                author=author,
                date_str=str(self._state.current_date.date()),
                timestamp=genesis_time,
                subdir=subdir,
                tags=resolved_tags,
                facts={"phase": "genesis"},
                aliases=aliases,
            )
            registered_ids.extend(conf_ids)

        logger.info(
            f"[confluence] Genesis batch complete ({prefix}): "
            f"{len(registered_ids)} page(s) registered."
        )
        return registered_ids

    def write_genesis_batches_parallel(
        self,
        batches: List[Dict],
    ) -> Dict[str, List[str]]:
        """
        Run multiple independent genesis batches concurrently.
        Pages within a batch remain sequential; batches across prefixes are parallel.
        """
        prefixes = [b["prefix"] for b in batches]
        if len(prefixes) != len(set(prefixes)):
            raise ValueError(
                f"[confluence] Duplicate prefixes in parallel genesis batches: {prefixes}."
            )
        results: Dict[str, List[str]] = {}
        lock = threading.Lock()

        with ThreadPoolExecutor(max_workers=len(batches)) as ex:
            futures = {
                ex.submit(self.write_genesis_batch, **batch): batch["prefix"]
                for batch in batches
            }
            for future in as_completed(futures):
                prefix = futures[future]
                try:
                    ids = future.result()
                    with lock:
                        results[prefix] = ids
                    logger.info(
                        f"[confluence] Parallel genesis batch done ({prefix}): "
                        f"{len(ids)} page(s)"
                    )
                except Exception as e:
                    logger.error(
                        f"[confluence] Parallel genesis batch failed ({prefix}): {e}"
                    )
                    with lock:
                        results[prefix] = []

        return results

    def write_postmortem(
        self,
        incident_id: str,
        incident_title: str,
        root_cause: str,
        days_active: int,
        on_call: str,
        eng_peer: str,
    ) -> tuple:
        """
        Generate a postmortem Confluence page for a resolved incident.
        The LLM emits alias vocabulary alongside Markdown in a single JSON response.
        """
        conf_id = self._registry.next_id("ENG")
        date_str = str(self._state.current_date.date())

        pm_hours = random.randint(60, 180) / 60.0
        artifact_time, _ = self._clock.advance_actor(on_call, hours=pm_hours)
        timestamp = artifact_time.isoformat()

        backstory = persona_utils.get_voice_card(
            on_call, "design", mem=self._mem, graph_dynamics=self._gd
        )
        related = self._registry.related_context(topic=root_cause, n=3)
        _qa_lead = next(
            (name for name, p in PERSONAS.items() if "QA" in p.get("expertise", [])),
            eng_peer,
        )
        _infra_lead = next(
            (name for name, p in PERSONAS.items() if "infra" in p.get("expertise", [])),
            eng_peer,
        )
        _action_owners = (
            f"Use these real names as action item owners:\n"
            f"  - Engineering fixes: {on_call}\n"
            f"  - Code review / PR: {eng_peer}\n"
            f"  - Test coverage / QA: {_qa_lead}\n"
            f"  - Infra / alerting: {_infra_lead}\n"
            f"Do NOT use role labels. Use the names above."
        )

        writer = make_agent(
            role="Senior Engineer",
            goal="Write a thorough incident postmortem.",
            backstory=backstory,
            llm=self._planner,
        )
        task = Task(
            description=(
                f"Write a postmortem for incident {incident_id}.\n"
                f"Title: Postmortem: {incident_title}\n"
                f"Root Cause: {root_cause}\n"
                f"Duration: {days_active} days.\n"
                f"You may reference these existing pages if relevant:\n{related}\n"
                f"{_action_owners}\n\n"
                f"Format as Markdown. Do NOT write a main # title. "
                f"Start directly with ## Executive Summary. "
                f"Include: Executive Summary, Timeline, Root Cause, Impact, "
                f"What Went Wrong, What Went Right, Action Items.\n\n"
                f"{_ALIAS_INSTRUCTION}"
                f"\nRespond ONLY with valid JSON:\n"
                f"{{\n"
                f'  "markdown_doc": "full postmortem Markdown, no # title",\n'
                f"{_ALIAS_JSON_FIELDS}"
                f"}}"
            ),
            expected_output="Valid JSON with markdown_doc and aliases keys.",
            agent=writer,
        )
        raw_output = str(
            Crew(agents=[writer], tasks=[task], verbose=False).kickoff()
        ).strip()

        clean = raw_output.replace("```json", "").replace("```", "").strip()
        content = raw_output
        aliases: Optional[List[str]] = None
        try:
            parsed = json.loads(clean)
            content = parsed.get("markdown_doc", raw_output)
            aliases = _extract_aliases(parsed)
        except (json.JSONDecodeError, ValueError):
            content = raw_output

        self._lifecycle.scan_for_knowledge_gaps(
            text=content,
            triggered_by=conf_id,
            day=self._state.day,
            date_str=date_str,
            state=self._state,
            timestamp=timestamp,
            author=on_call,
        )

        conf_ids = self._finalize_page(
            raw_content=content,
            conf_id=conf_id,
            title=f"Postmortem: {incident_title}",
            author=on_call,
            date_str=date_str,
            timestamp=timestamp,
            subdir="postmortems",
            tags=["postmortem", "confluence"],
            facts={"root_cause": root_cause, "incident_id": incident_id},
            extra_artifact_ids={"jira": incident_id},
            aliases=aliases,
        )
        logger.info(f"    [green]Postmortem:[/green] {conf_ids[0]}")

        for inc in self._state.active_incidents:
            if inc.ticket_id == incident_id and getattr(inc, "causal_chain", None):
                inc.causal_chain.append(conf_ids[0])
                break

        return conf_ids[0], timestamp

    def write_design_doc(
        self,
        author: str,
        participants: List[str],
        topic: str,
        slack_transcript: List[Dict],
        date_str: str,
    ) -> Optional[str]:
        """
        Generate a design doc Confluence page from a Slack discussion.

        The LLM produces prose and a self_audit block containing raw
        observations only. All gap classification and domain_fit scoring
        is computed deterministically by the engine from those observations,
        consistent with the physics-cognition boundary.
        """
        conf_id = self._registry.next_id("ENG")
        write_delay_hours = random.uniform(0.5, 1.5)
        artifact_time, _ = self._clock.advance_actor(author, hours=write_delay_hours)
        timestamp = artifact_time.isoformat()

        chat_log = "\n".join(f"{m['user']}: {m['text']}" for m in slack_transcript)

        backstory = persona_utils.get_voice_card(
            author, "design", mem=self._mem, graph_dynamics=self._gd, include_expertise=False
        )

        related_pages = self._mem.search_artifacts_text(
            query=topic,
            n=5,
            type_filter="confluence",
            as_of_time=timestamp,
        )
        related = "\n".join(
            f"- {r['id']}: {r['title']}"
            for r in related_pages
        ) or "None yet."

        expertise_tokens = self._mem.get_author_domain_tokens(author)
        expertise_str = ", ".join(sorted(expertise_tokens))

        agent = make_agent(
            role="Technical Lead",
            goal="Document technical decisions and extract an actionable ticket.",
            backstory=backstory,
            llm=self._planner,
        )
        task = Task(
            description=(
                f"You just had this Slack discussion about '{topic}':\n\n{chat_log}\n\n"
                f"Existing pages you may reference:\n{related}\n\n"
                f"Write a design doc Confluence page with ID {conf_id}.\n"
                f"Also extract 1 concrete next step as a JIRA ticket.\n\n"
                f"\n\n### SELF-AUDIT\n"
                f"Your expertise on record: [{expertise_str}]\n"
                f"After writing the doc, fill the self_audit block objectively, not in character.\n"
                f"  topics_in_doc: every distinct technical domain, system, or component this doc discusses.\n"
                f"  topics_outside_my_expertise: copy terms from topics_in_doc only — "
                f"do not add new terms — where the topic is NOT in your expertise list above.\n"
                f"  claims_i_approximated: specific sentences or values where you inferred, "
                f"generalized, or were uncertain rather than stating a known fact.\n"
                f"  sections_i_left_thin: section headers (## only) where you wrote less "
                f"than the section warrants because you lacked detail.\n\n"
                f"Respond ONLY with valid JSON:\n"
                f"{{\n"
                f'  "markdown_doc": "full Markdown, no # title, start with ## Problem Statement",\n'
                f'  "new_tickets": [\n'
                f'    {{"title": "string", "assignee": "{author}", "story_points": 1}}\n'
                f"  ],\n"
                f"{_ALIAS_JSON_FIELDS}"
                f'  "self_audit": {{\n'
                f'    "topics_in_doc": ["string"],\n'
                f'    "topics_outside_my_expertise": ["subset of topics_in_doc only"],\n'
                f'    "claims_i_approximated": ["string"],\n'
                f'    "sections_i_left_thin": ["string"]\n'
                f"  }}\n"
                f"}}"
            ),
            expected_output="Valid JSON only. No markdown fences.",
            agent=agent,
        )
        raw = str(Crew(agents=[agent], tasks=[task], verbose=False).kickoff()).strip()
        clean = raw.replace("```json", "").replace("```", "").strip()

        try:
            brace_start = clean.index("{")
            brace_end = clean.rindex("}") + 1
            clean = clean[brace_start:brace_end]
        except ValueError:
            clean = "{}"

        try:
            parsed = json.loads(clean)
            content = parsed.get("markdown_doc", "Draft pending.")
            new_tickets = parsed.get("new_tickets", [])
            aliases = _extract_aliases(parsed)
            audit = parsed.get("self_audit", {})
        except json.JSONDecodeError as e:
            logger.warning(f"[confluence] JSON parse failed for design doc: {e}")
            content = raw
            new_tickets = []
            aliases = None
            audit = {}

        topics_in_doc = set(audit.get("topics_in_doc", []))
        topics_outside = [
            t for t in audit.get("topics_outside_my_expertise", [])
            if t in topics_in_doc and t not in expertise_tokens
        ]
        claims_approximated = audit.get("claims_i_approximated", [])
        sections_thin = audit.get("sections_i_left_thin", [])



        conf_ids = self._finalize_page(
            raw_content=content,
            conf_id=conf_id,
            title=f"Design: {topic[:80]}",
            author=author,
            date_str=date_str,
            timestamp=timestamp,
            subdir="design",
            tags=["confluence", "design_doc"],
            facts={"title": f"Design: {topic[:80]}", "type": "design_doc"},
            skip_event=True,
            aliases=aliases,
        )

        _updated_domains = [
            rec["domain"]
            for rec in self._mem._db["domain_registry"].find(
                {"known_by": author, "last_updated_day": self._state.day}
            )
        ]

        self._lifecycle.scan_for_knowledge_gaps(
            text=content,
            triggered_by=conf_id,
            day=self._state.day,
            date_str=date_str,
            state=self._state,
            timestamp=timestamp,
            author=author,
            topics_outside_expertise=topics_outside,
            claims_approximated=claims_approximated,
        )

        created_ticket_ids = self._spawn_tickets(
            new_tickets, author, participants, date_str, timestamp
        )

        chain = CausalChainHandler(root_id=conf_ids[0])
        for tid in created_ticket_ids:
            chain.append(tid)

        self._mem.log_event(
            SimEvent(
                type="confluence_created",
                timestamp=timestamp,
                day=self._state.day,
                date=date_str,
                actors=participants,
                artifact_ids={
                    "confluence": conf_ids[0],
                    "spawned_tickets": created_ticket_ids,
                },
                facts={
                    "title": f"Design: {topic[:80]}",
                    "type": "design_doc",
                    "spawned_tickets": created_ticket_ids,
                    "causal_chain": chain.snapshot(),
                    "author_domain_fit": domain_fit,
                    "gap_classification": gap_classification,
                    "topics_outside_author_expertise": topics_outside,
                    "claims_approximated": claims_approximated,
                    "sections_left_thin": sections_thin,
                    "domains_updated": _updated_domains,
                },
                summary=(
                    f"{author} created {conf_ids[0]} and spawned "
                    f"{len(created_ticket_ids)} ticket(s): {', '.join(created_ticket_ids)}"
                ),
                tags=["confluence", "design_doc", "jira", "causal_chain"],
            )
        )

        logger.info(
            f"    [dim]Design doc: {conf_ids[0]} "
            f"(spawned {len(created_ticket_ids)} ticket(s), "
            f"domain_fit={domain_fit}, gap={gap_classification})[/dim]"
        )
        return conf_ids[0]

    def write_adhoc_page(
        self,
        author: Optional[str] = None,
        backstory: Optional[str] = None,
    ) -> Optional[tuple]:
        """
        Generate a character-accurate ad-hoc Confluence page.
        The writer task emits alias vocabulary alongside the page content.
        """
        active_today: List[str] = list(
            dict.fromkeys(getattr(self._state, "daily_active_actors", []))
        )
        resolved_author: str = author or (
            random.choice(active_today)
            if active_today
            else random.choice(self._all_names)
        )

        dept = next(
            (d for d, members in self._org_chart.items() if resolved_author in members),
            "ENG",
        )

        _PREFIX_MAP = {
            "Engineering_Backend": "ENG",
            "Engineering_Mobile": "ENG",
            "Product": "PROD",
            "Design": "DESIGN",
            "Sales_Marketing": "MKT",
            "HR_Ops": "HR",
            "QA_Support": "QA",
        }
        prefix = _PREFIX_MAP.get(dept, dept[:3].upper())

        daily_theme: str = getattr(self._state, "daily_theme", "")

        doc_history = list(
            self._mem._events.find(
                {
                    "type": "confluence_created",
                    "timestamp": {"$lte": self._clock.now(resolved_author).isoformat()},
                },
                {"facts.title": 1, "actors": 1},
            )
            .sort("timestamp", -1)
            .limit(20)
        )

        history_str = (
            "\n".join(
                [
                    f"  - {e['facts'].get('title')} (by {e['actors'][0]})"
                    for e in doc_history
                ]
            )
            if doc_history
            else "No documentation exists yet."
        )

        persona = self._config.get("personas", {}).get(resolved_author, {})
        expertise_list = persona.get("expertise", ["general tasks"])
        expertise_str = ", ".join(expertise_list)

        seed_query = f"{resolved_author} {expertise_str} {daily_theme}"
        topic_ctx = self._mem.context_for_prompt(
            seed_query,
            n=5,
            as_of_time=self._clock.now(resolved_author).isoformat(),
        )
        backstory = persona_utils.get_voice_card(
            resolved_author, "design", mem=self._mem, graph_dynamics=self._gd
        )

        topic_agent = make_agent(
            role="Content Planner",
            goal="Identify a unique documentation gap based on your expertise and org history.",
            backstory=backstory,
            llm=self._worker,
        )

        topic_task = Task(
            description=(
                f"Today's Org Theme: {daily_theme}\n"
                f"Recent Activity Context: {topic_ctx}\n\n"
                f"EXISTING DOCUMENTATION (Do NOT duplicate or overlap significantly):\n"
                f"{history_str}\n\n"
                f"TASK:\n"
                f"Based on your expertise ({expertise_str}), propose ONE specific "
                f"Confluence page title you would plausibly write TODAY.\n\n"
                f"Rules:\n"
                f"- The topic MUST fall within your area of expertise ({expertise_str}).\n"
                f"- Find a specific gap. If a topic is already documented, "
                f"look for a sub-topic or angle not yet covered.\n"
                f"- Be specific and realistic based on the current Org Theme.\n"
                f"- Return ONLY the page title string. No explanation. No quotes."
            ),
            expected_output="A single page title string. Nothing else.",
            agent=topic_agent,
        )

        title = (
            str(Crew(agents=[topic_agent], tasks=[topic_task], verbose=False).kickoff())
            .strip()
            .strip('"')
            .strip("'")
        )
        title = title.splitlines()[0][:120]

        conf_id = self._registry.next_id(prefix)

        session_hours = random.randint(30, 90) / 60.0
        artifact_time, _ = self._clock.advance_actor(
            resolved_author, hours=session_hours
        )
        timestamp = artifact_time.isoformat()
        date_str = str(self._state.current_date.date())

        ctx = self._mem.context_for_prompt(title, n=3, as_of_time=timestamp)
        related = self._registry.related_context(topic=title, n=4)
        backstory = persona_utils.get_voice_card(
            resolved_author, "design", mem=self._mem, graph_dynamics=self._gd
        )

        writer_agent = make_agent(
            role="Corporate Writer",
            goal=f"Draft a {title} Confluence page.",
            backstory=backstory,
            llm=self._planner,
        )
        task = Task(
            description=(
                f"Write a single Confluence page with ID {conf_id} titled '{title}'.\n"
                f"Context from memory: {ctx}\n"
                f"Existing pages you may reference (and ONLY these):\n{related}\n\n"
                f"Rules:\n"
                f"- Use your specific technical expertise and typing style.\n"
                f"- Do not invent any CONF-* IDs not listed above.\n"
                f"- Format as Markdown. Do not write a main # title or metadata block.\n"
                f"- Start directly with the first paragraph or ## section.\n\n"
                f"{_ALIAS_INSTRUCTION}"
                f"\nRespond ONLY with valid JSON:\n"
                f"{{\n"
                f'  "markdown_doc": "full Markdown page content",\n'
                f"{_ALIAS_JSON_FIELDS}"
                f"}}"
            ),
            expected_output="Valid JSON with markdown_doc and aliases keys.",
            agent=writer_agent,
        )
        raw = str(
            Crew(agents=[writer_agent], tasks=[task], verbose=False).kickoff()
        ).strip()

        clean = raw.replace("```json", "").replace("```", "").strip()
        content = raw
        aliases: Optional[List[str]] = None
        try:
            parsed = json.loads(clean)
            content = parsed.get("markdown_doc", raw)
            aliases = _extract_aliases(parsed)
        except (json.JSONDecodeError, ValueError):
            content = raw

        self._lifecycle.scan_for_knowledge_gaps(
            text=content,
            triggered_by=conf_id,
            day=self._state.day,
            date_str=date_str,
            state=self._state,
            timestamp=timestamp,
            author=resolved_author,
        )

        content += self._knowledge_gap_warning(title)

        self._finalize_page(
            raw_content=content,
            conf_id=conf_id,
            title=title,
            author=resolved_author,
            date_str=date_str,
            timestamp=timestamp,
            subdir="general",
            tags=["confluence", "adhoc"],
            facts={"title": title, "adhoc": True},
            aliases=aliases,
        )

        return (conf_id, resolved_author, title)

    def _finalize_page(
        self,
        raw_content: str,
        conf_id: str,
        title: str,
        author: str,
        date_str: str,
        timestamp: str,
        subdir: str,
        tags: List[str],
        facts: Dict,
        extra_artifact_ids: Optional[Dict[str, str]] = None,
        skip_event: bool = False,
        aliases: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Common finalization pipeline for every Confluence page.
        Writes aliases as a list so Atlas Search can index them for BM25 retrieval.
        """
        clean_content = self._registry.strip_broken_references(raw_content)

        pages: List[ConfluencePage] = self._registry.chunk_into_pages(
            parent_id=conf_id,
            parent_title=title,
            content=clean_content,
            prefix=self._id_prefix_from_id(conf_id),
            state=self._state,
            author=author,
            date_str=date_str,
        )

        created_ids: List[str] = []
        for page in pages:
            try:
                final_content = self._registry.strip_broken_references(page.content)
            except Exception as e:
                logger.warning(f"[finalize] strip_broken_references failed: {e}")
                final_content = page.content

            self._save_md(page.path, final_content)

            meta = {
                "author": author,
                "parent_id": page.parent_id or "",
                "is_chunk": page.parent_id is not None,
                "tags": tags or [],
            }
            if tags and "genesis" in tags:
                meta["phase"] = "genesis"

            page_aliases = aliases if page.parent_id is None else None

            self._mem.embed_artifact(
                id=page.id,
                type="confluence",
                title=page.title,
                content=final_content,
                day=self._state.day,
                date=date_str,
                timestamp=timestamp,
                metadata=meta,
                aliases=page_aliases,
            )

            if page.parent_id is None and author and "genesis" not in (tags or []):
                domains_updated = self._update_domain_registry_on_write(
                    author=author,
                    title=page.title,
                    content=final_content,
                    day=self._state.day,
                )
                if domains_updated:
                    meta["domains_updated"] = domains_updated

            self._state.daily_artifacts_created += 1

            page_facts = dict(facts)
            page_facts.update(
                {
                    "parent_id": page.parent_id or "",
                    "is_chunk": page.parent_id is not None,
                    "title": page.title,
                    "author": author,
                }
            )

            artifact_ids = {"confluence": page.id}
            if extra_artifact_ids:
                artifact_ids.update(extra_artifact_ids)

            if not skip_event:
                self._mem.log_event(
                    SimEvent(
                        type="confluence_created",
                        timestamp=timestamp,
                        day=self._state.day,
                        date=date_str,
                        actors=[author],
                        artifact_ids=artifact_ids,
                        facts=page_facts,
                        summary=(
                            f"{'Child' if page.parent_id else 'Page'} "
                            f"{page.id} created: {page.title}"
                        ),
                        tags=tags,
                    )
                )

            created_ids.append(page.id)

        return created_ids

    def generate_tech_stack(self) -> dict:
        """
        Ask the LLM to invent a plausible tech stack for this company and industry.
        Persists to MongoDB immediately so every subsequent LLM call can reference it.
        """
        existing = self._mem.get_tech_stack()
        if existing:
            logger.info("[confluence] Tech stack already exists. Skipping generation.")
            return existing

        agent = make_agent(
            role="Principal Engineer",
            goal="Define the canonical technology stack for this company.",
            backstory=(
                f"You are a principal engineer at {self._company}, "
                f"a {self._industry} company. You are documenting the actual "
                f"technologies in use. Not aspirational, not greenfield."
            ),
            llm=self._planner,
        )
        task = Task(
            description=(
                f"Define the canonical tech stack for {self._company} "
                f"which {COMPANY_DESCRIPTION}\n\n"
                f"The legacy system is called '{self._legacy.get('name', '')}' "
                f"({self._legacy.get('description', '')}).\n\n"
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
                "[confluence] Tech stack JSON parse failed. Using minimal fallback."
            )
            stack = {
                "notable_quirks": "Stack unknown. Legacy system predates documentation."
            }

        self._mem.save_tech_stack(stack)
        logger.info(f"[confluence] Tech stack established: {list(stack.keys())}")
        return stack

    def _spawn_tickets(
        self,
        new_tickets: List[Dict],
        fallback_author: str,
        valid_names: List[str],
        date_str: str,
        timestamp: str,
    ) -> List[str]:
        """Create JIRA tickets from a list of LLM-extracted action items."""
        created_ids: List[str] = []
        from artifact_registry import JIRA_DEPT_PREFIX

        for tk in new_tickets:
            assignee = tk.get("assignee", fallback_author)
            if assignee not in self._all_names:
                assignee = fallback_author

            author_dept = next(
                (
                    d
                    for d, members in self._org_chart.items()
                    if fallback_author in members
                ),
                "",
            )
            prefix = JIRA_DEPT_PREFIX.get(
                author_dept, "ENG" if author_dept.startswith("Engineering") else "ORG"
            )
            is_eng = author_dept.startswith("Engineering")

            tid = self._registry.next_jira_id(prefix)
            self._registry.register_jira(tid)

            ticket = {
                "id": tid,
                "title": tk.get("title", "Generated Task"),
                "status": "To Do",
                "assignee": assignee,
                "sprint": getattr(self._state.sprint, "sprint_number", 1),
                "story_points": tk.get("story_points", 2),
                "linked_prs": [],
                "created_at": timestamp,
                "updated_at": timestamp,
                "dept": author_dept,
                "dept_type": "eng" if is_eng else "non_eng",
                "completion_artifact": "slack" if not is_eng else None,
            }

            self._save_json(f"{self._base}/jira/{tid}.json", ticket)

            self._mem.embed_artifact(
                id=tid,
                type="jira",
                title=ticket["title"],
                content=json.dumps(ticket),
                day=self._state.day,
                date=date_str,
                metadata={"assignee": assignee},
                timestamp=timestamp,
            )
            self._state.daily_artifacts_created += 1
            created_ids.append(tid)
        return created_ids

    def _update_domain_registry_on_write(
        self,
        author: str,
        title: str,
        content: str,
        day: int,
        coverage_delta: float = 0.10,
        known_by_tag_threshold: int = 2,
    ) -> List[str]:
        """
        After any Confluence page is finalised, increment documentation_coverage
        for any domain whose system_tags appear in the page content.

        known_by is only updated when the domain is clearly a primary topic:
        either the domain name itself appears in the content, or at least
        known_by_tag_threshold distinct system_tags match. A single incidental
        tag mention (e.g. 'auth' in a page about something else) does not
        qualify.
        """
        updated: List[str] = []
        all_domains = list(self._mem._db["domain_registry"].find({}))
        if not all_domains:
            return updated

        search_text = f"{title} {content}".lower()
        title_lower = title.lower()

        for rec in all_domains:
            tags = rec.get("system_tags", [])
            matched_tags = [tag for tag in tags if tag in search_text]
            if not matched_tags:
                continue

            old_coverage = rec.get("documentation_coverage", 0.0)
            new_coverage = min(1.0, old_coverage + coverage_delta)


            is_primary_topic = (
                rec["domain"].lower() in title_lower
                or len(matched_tags) >= known_by_tag_threshold
            )

            update: dict = {
                "$set": {
                    "documentation_coverage": round(new_coverage, 3),
                    "last_updated_day": day,
                }
            }
            if is_primary_topic:
                update["$addToSet"] = {"known_by": author}

            self._mem._db["domain_registry"].update_one(
                {"_id": rec["_id"]},
                update,
            )
            updated.append(rec["domain"])
            logger.info(
                f"    [dim]Domain registry: '{rec['domain']}' coverage "
                f"{int(old_coverage * 100)}% to {int(new_coverage * 100)}%"
                f"{' (known_by: ' + author + ')' if is_primary_topic else ''}"
                f" (author: {author})[/dim]"
            )

        return updated

    def _pick_dept_author(self, prefix: str) -> str:
        for dept, members in self._org_chart.items():
            if prefix.upper() in dept.upper() and members:
                return random.choice(members)
        return random.choice(self._all_names)

    def _conf_prefix_for(self, author: str) -> str:
        dept = next(
            (d for d, members in self._org_chart.items() if author in members),
            "",
        )
        return _CONF_PREFIX_MAP.get(dept, "ENG")

    def _compute_domain_fit(self, author: str, topic: str) -> str:
        """
        Deterministically compute how well the author's expertise covers the
        topic, using the domain_registry's orphan status and system_tags.

        Returns "high", "medium", or "low" based on documentation_coverage
        thresholds (same thresholds used in scan_for_knowledge_gaps):
          - No orphaned domains touched by topic -> "high"
          - Author is in known_by for all touched orphans -> "high"
          - avg coverage < 0.3 -> "low"
          - avg coverage < 0.6 -> "medium"
          - else -> "high"

        """
        orphaned = list(self._mem._db["domain_registry"].find({"primary_owner": None}))
        topic_lower = topic.lower()
        touched_orphans = [
            rec
            for rec in orphaned
            if any(tag in topic_lower for tag in rec.get("system_tags", []))
        ]
        if not touched_orphans:
            return "high"
        covered = all(author in rec.get("known_by", []) for rec in touched_orphans)
        if covered:
            return "high"
        avg_coverage = sum(
            r.get("documentation_coverage", 0) for r in touched_orphans
        ) / len(touched_orphans)
        if avg_coverage < 0.3:
            return "low"
        if avg_coverage < 0.6:
            return "medium"
        return "high"

    def _compute_gap_classification(self, author: str, topic: str) -> str:
        """
        Deterministically classify whether this topic/author combination
        represents a knowledge gap, using the same thresholds as
        scan_for_knowledge_gaps in org_lifecycle.py:
          - live_coverage < 0.3 -> "likely"
          - live_coverage < 0.6 -> "possible"
          - else -> "none"

        This runs BEFORE the LLM writes anything, so the engine knows at
        planning time whether this document enters a gap zone.
        """
        orphaned = list(self._mem._db["domain_registry"].find({"primary_owner": None}))
        topic_lower = topic.lower()
        touched_orphans = [
            rec
            for rec in orphaned
            if any(tag in topic_lower for tag in rec.get("system_tags", []))
        ]
        if not touched_orphans:
            return "none"
        covered = all(author in rec.get("known_by", []) for rec in touched_orphans)
        if covered:
            return "none"
        min_coverage = min(r.get("documentation_coverage", 0) for r in touched_orphans)
        if min_coverage < 0.3:
            return "likely"
        if min_coverage < 0.6:
            return "possible"
        return "none"

    @staticmethod
    def _render(template: str, vars_: Dict[str, str]) -> str:
        for key, value in vars_.items():
            template = template.replace(f"{{{key}}}", str(value))
        return template

    def _render_template(self, template: str) -> str:
        return (
            template.replace("{legacy_system}", self._legacy.get("name", ""))
            .replace("{project_name}", self._legacy.get("project_name", ""))
            .replace("{company_name}", self._company)
            .replace("{industry}", self._industry)
        )

    @staticmethod
    def _extract_title(content: str, fallback: str) -> str:
        clean = re.sub(r"```.*?```", "", content, flags=re.DOTALL)
        m = re.search(r"^#\s+(.+)", clean, re.MULTILINE)
        if m:
            return m.group(1).strip()
        m = re.search(r"^##\s+(.+)", clean, re.MULTILINE)
        if m:
            return m.group(1).strip()
        return f"Archive: {fallback}"

    @staticmethod
    def _id_prefix_from_id(conf_id: str) -> str:
        parts = conf_id.split("-")
        return parts[1] if len(parts) >= 3 else "GEN"

    def _knowledge_gap_warning(self, topic: str) -> str:
        topic_lower = topic.lower()
        all_domains = list(
            self._mem._db["domain_registry"].find({"primary_owner": None})
        )
        for rec in all_domains:
            tags = rec.get("system_tags", [])
            if any(tag in topic_lower for tag in tags):
                pct = int(rec.get("documentation_coverage", 0.2) * 100)
                former = rec.get("former_owner", "a former employee")
                known_by = rec.get("known_by", [])
                known_str = (
                    f" Partial knowledge held by: {', '.join(known_by)}."
                    if known_by
                    else " No current owner."
                )
                return (
                    f"\n\n> **Knowledge Gap**: This area ({rec['domain']}) was owned by "
                    f"{former}. Only ~{pct}% documented.{known_str}"
                )

        departed = self._config.get("knowledge_gaps", [])
        for emp in departed:
            hits = [k for k in emp.get("knew_about", []) if k.lower() in topic_lower]
            if hits:
                return (
                    f"\n\n> **Knowledge Gap**: This area ({', '.join(hits)}) was owned by "
                    f"{emp['name']} (ex-{emp['role']}, left {emp['left']}). "
                    f"Only ~{int(emp.get('documented_pct', 0.2) * 100)}% documented."
                )
        return ""

    def _save_md(self, path: str, content: str) -> None:
        import os

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(content)

    def _save_json(self, path: str, data: Any) -> None:
        import os

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def _tenure_at_date(
        tenure_str: str, sim_start: datetime, page_date: datetime
    ) -> str:
        import re
        from dateutil.relativedelta import relativedelta

        m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(yr|mo)", tenure_str.strip())
        if not m:
            return tenure_str

        value, unit = float(m.group(1)), m.group(2)
        months_at_sim_start = int(value * 12) if unit == "yr" else int(value)

        delta = relativedelta(sim_start, page_date)
        months_offset = delta.years * 12 + delta.months

        months_at_page = months_at_sim_start - months_offset
        if months_at_page <= 0:
            return "new"

        if months_at_page < 12:
            return f"{months_at_page}mo"
        years = months_at_page // 12
        remainder = months_at_page % 12
        return f"{years}yr" if remainder == 0 else f"{years}.{remainder // 1}yr"
