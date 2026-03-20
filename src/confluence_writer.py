"""
confluence_writer.py
=====================
Single source of truth for all Confluence page generation in OrgForge.

Every path that produces a Confluence artifact — genesis, postmortems,
design doc stubs, ad-hoc pages — runs through this module.

Responsibilities:
  - ID allocation (Python owns the namespace, never the LLM)
  - Single-page LLM generation (one task per page, no PAGE BREAK parsing)
  - Reference injection (LLM is told which pages already exist)
  - Reference validation + broken-ref stripping (via ArtifactRegistry)
  - Chunking of long content into focused child pages
  - Embedding and SimEvent logging

Callers (flow.py, normal_day.py) import ConfluenceWriter and call the
appropriate method. They no longer manage conf_id allocation or embedding
directly for Confluence artifacts.

Usage:
    from confluence_writer import ConfluenceWriter

    writer = ConfluenceWriter(
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
        base_export_dir=BASE,
    )
"""

from __future__ import annotations

from datetime import datetime
import json
import logging
import random
import re
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from agent_factory import make_agent
from causal_chain_handler import CausalChainHandler
from config_loader import COMPANY_DESCRIPTION, PERSONAS
from crewai import Task, Crew
from memory import Memory, SimEvent
from artifact_registry import ArtifactRegistry, ConfluencePage

if TYPE_CHECKING:
    from graph_dynamics import GraphDynamics

logger = logging.getLogger("orgforge.confluence")


# ─────────────────────────────────────────────────────────────────────────────
# CONFLUENCE WRITER
# ─────────────────────────────────────────────────────────────────────────────


class ConfluenceWriter:
    def __init__(
        self,
        mem: Memory,
        registry: ArtifactRegistry,
        state,  # flow.State — avoid circular import
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

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────────────────────

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

        Python allocates all IDs upfront. Each page is generated in a separate
        LLM call so max_tokens truncation only ever affects one page, not the
        whole batch. Later pages in the batch receive the IDs of earlier ones as
        allowed references so cross-links are always resolvable.

        Args:
            prefix:      ID namespace, e.g. "ENG" or "MKT".
            count:       Number of pages to generate.
            prompt_tpl:  Single-page prompt template. Available placeholders:
                           {id}, {company}, {industry}, {legacy_system},
                           {project_name}, {author}, {related_pages}
            author:      Author of the page.
            extra_vars:  Any additional {placeholder} → value substitutions.
            subdir:      Export subdirectory under confluence/.

        Returns:
            List of registered conf_ids in generation order.
        """
        # 1. Allocate all IDs before any LLM call
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
                goal="Write one authentic internal Confluence page as yourself. Write with real insider detail.",
                backstory=self._persona(author, mem=self._mem, graph_dynamics=self._gd),
                llm=self._planner,
            )
            task = Task(
                description=prompt,
                expected_output=(
                    f"A single Markdown Confluence page with ID {conf_id}. "
                    f"No separators. No preamble. "
                    f"Do not include a main # title or a metadata block at the top. "
                    f"Start directly with the first paragraph or ## section."
                ),
                agent=historian,
            )
            raw = str(
                Crew(agents=[historian], tasks=[task], verbose=False).kickoff()
            ).strip()

            resolved_tags = tags or ["genesis", "confluence"]

            conf_ids = self._finalise_page(
                raw_content=raw,
                conf_id=conf_id,
                title=self._extract_title(raw, conf_id),
                author=author,
                date_str=str(self._state.current_date.date()),
                timestamp=genesis_time,
                subdir=subdir,
                tags=resolved_tags,
                facts={"phase": "genesis"},
            )
            registered_ids.extend(conf_ids)

        logger.info(
            f"[confluence] ✓ Genesis batch complete ({prefix}): "
            f"{len(registered_ids)} page(s) registered."
        )
        return registered_ids

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

        Returns the registered conf_id.
        """
        conf_id = self._registry.next_id("ENG")
        date_str = str(self._state.current_date.date())

        pm_hours = random.randint(60, 180) / 60.0
        artifact_time, _ = self._clock.advance_actor(on_call, hours=pm_hours)
        timestamp = artifact_time.isoformat()

        backstory = self._persona(on_call, mem=self._mem, graph_dynamics=self._gd)
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
            f"Do NOT use role labels like 'backend lead' or 'qa lead' — use the names above."
        )

        writer = make_agent(
            role="Senior Engineer",
            goal="Write a thorough incident postmortem.",
            backstory=backstory,
            llm=self._planner,
        )
        task = Task(
            description=(
                f"Write a Confluence postmortem page with ID {conf_id} "
                f"for incident {incident_id}.\n"
                f"Title: Postmortem: {incident_title}\n"
                f"Root Cause: {root_cause}\n"
                f"Duration: {days_active} days.\n"
                f"You may reference these existing pages if relevant:\n{related}\n"
                f"{_action_owners}\n"
                f"Format as Markdown. Do NOT write a main # title — start directly with ## Executive Summary. "
                f"Include: Executive Summary, Timeline, Root Cause, Impact, "
                f"What Went Wrong, What Went Right, Action Items."
            ),
            expected_output=f"A single Markdown postmortem page with ID {conf_id}.",
            agent=writer,
        )
        raw = str(Crew(agents=[writer], tasks=[task], verbose=False).kickoff()).strip()

        self._lifecycle.scan_for_knowledge_gaps(
            text=raw,
            triggered_by=conf_id,
            day=self._state.day,
            date_str=date_str,
            state=self._state,
            timestamp=timestamp,
        )

        conf_ids = self._finalise_page(
            raw_content=raw,
            conf_id=conf_id,
            title=f"Postmortem: {incident_title}",
            author=on_call,
            date_str=date_str,
            timestamp=timestamp,
            subdir="postmortems",
            tags=["postmortem", "confluence"],
            facts={"root_cause": root_cause, "incident_id": incident_id},
            extra_artifact_ids={"jira": incident_id},
        )
        logger.info(f"    [green]📄 Postmortem:[/green] {conf_ids[0]}")

        for inc in self._state.active_incidents:
            if inc.ticket_id == incident_id and getattr(inc, "causal_chain", None):
                inc.causal_chain.append(conf_ids[0])
                logger.info(
                    f"    [dim]🔗 Postmortem {conf_ids[0]} appended to "
                    f"{incident_id} causal chain[/dim]"
                )
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
        Also spawns 1-3 JIRA tickets from the action items in the chat.

        Returns the registered conf_id, or None on failure.
        """
        conf_id = self._registry.next_id("ENG")
        artifact_time, _ = self._clock.advance_actor(author, hours=0.5)
        timestamp = artifact_time.isoformat()

        chat_log = "\n".join(f"{m['user']}: {m['text']}" for m in slack_transcript)
        # Tier 3: topic is a free-form Slack thread subject — HyDE rewrite
        # produces a better embedding target than the raw topic string alone.
        ctx = self._mem.recall_with_rewrite(raw_query=topic, n=3, as_of_time=timestamp)
        related = self._registry.related_context(topic=topic, n=3)

        agent = make_agent(
            role="Technical Lead",
            goal="Document technical decisions and extract actionable tickets.",
            backstory=self._persona(
                author,
                mem=self._mem,
                graph_dynamics=self._gd,
                extra="You just finished a Slack discussion and need to document decisions and assign follow-up work.",
            ),
            llm=self._planner,
        )
        task = Task(
            description=(
                f"You just had this Slack discussion about '{topic}':\n\n{chat_log}\n\n"
                f"Background context: {ctx}\n"
                f"Existing pages you may reference:\n{related}\n\n"
                f"Write a design doc Confluence page with ID {conf_id}.\n"
                f"Also extract 1-3 concrete next steps as JIRA ticket definitions.\n"
                f"Respond ONLY with valid JSON matching this exact schema:\n"
                f"{{\n"
                f'  "markdown_doc": "string (full Markdown, no main # title, start directly with ## Problem Statement)",\n'
                f'  "new_tickets": [\n'
                f'    {{"title": "string", '
                f'"assignee": "string (must be exactly: {author})", '
                f'"story_points": 1|2|3|5|8}}\n'
                f"  ]\n"
                f"}}"
            ),
            expected_output="Valid JSON only. No markdown fences.",
            agent=agent,
        )
        raw = str(Crew(agents=[agent], tasks=[task], verbose=False).kickoff()).strip()
        clean = raw.replace("```json", "").replace("```", "").strip()

        # Extract the JSON object robustly: find the outermost { ... } block
        # so any leading/trailing prose from the LLM doesn't break the parse.
        try:
            brace_start = clean.index("{")
            brace_end = clean.rindex("}") + 1
            clean = clean[brace_start:brace_end]
        except ValueError:
            logger.warning(
                f"[confluence] No JSON object found in design doc response — "
                f"raw output: {clean[:200]!r}"
            )
            clean = "{}"

        try:
            parsed = json.loads(clean)
            content = parsed.get("markdown_doc", "Draft pending.")
            new_tickets = parsed.get("new_tickets", [])
        except json.JSONDecodeError as e:
            logger.warning(
                f"[confluence] JSON parse failed for design doc: {e} — "
                f"raw JSON attempt: {clean[:200]!r}"
            )
            content = raw  # save the original unmodified LLM output as the doc
            new_tickets = []

        conf_ids = self._finalise_page(
            raw_content=content,
            conf_id=conf_id,
            title=f"Design: {topic[:70]}",
            author=author,
            date_str=date_str,
            timestamp=timestamp,
            subdir="design",
            tags=["confluence", "design_doc"],
            facts={"title": f"Design: {topic[:80]}", "type": "design_doc"},
        )

        # Spawn JIRA tickets from action items
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
                    "spawned_tickets": json.dumps(created_ticket_ids),
                },
                facts={
                    "title": f"Design: {topic[:80]}",
                    "type": "design_doc",
                    "spawned_tickets": created_ticket_ids,
                    "causal_chain": chain.snapshot(),  # ← add this
                },
                summary=(
                    f"{author} created {conf_ids[0]} and spawned "
                    f"{len(created_ticket_ids)} ticket(s): {', '.join(created_ticket_ids)}"
                ),
                tags=[
                    "confluence",
                    "design_doc",
                    "jira",
                    "causal_chain",
                ],  # ← add causal_chain
            )
        )

        logger.info(
            f"    [dim]📄 Design doc: {conf_ids[0]} "
            f"(spawned {len(created_ticket_ids)} ticket(s))[/dim]"
        )
        return conf_ids[0]

    def write_adhoc_page(
        self,
        author: Optional[str] = None,
        backstory: Optional[str] = None,
    ) -> None:
        """
        Generate a character-accurate ad-hoc Confluence page.

        Topic and ID prefix are derived from the author's persona expertise via
        a fast LLM call — no hardcoded topic lists required. The author is drawn
        from state.daily_active_actors so every page is organically tied to
        someone who was actually working today.

        Falls back to a random org member only if no active actors exist yet.
        """
        # ── 1. Pick an author from today's active workers ─────────────────────
        active_today: List[str] = list(
            dict.fromkeys(getattr(self._state, "daily_active_actors", []))
        )
        resolved_author: str = author or (
            random.choice(active_today)
            if active_today
            else random.choice(self._all_names)
        )

        # ── 2. Determine prefix from author's department ──────────────────────
        dept = next(
            (d for d, members in self._org_chart.items() if resolved_author in members),
            "ENG",
        )
        # Map department names to short Confluence ID prefixes
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
                {"type": "confluence_created"}, {"facts.title": 1, "actors": 1}
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
        # Tier 3: LLM-generated seed query benefits from HyDE rewrite.
        # recall_with_rewrite degrades to context_for_prompt if no llm_callable
        # is passed, so this is safe to land before the callable is wired in.
        topic_ctx = self._mem.recall_with_rewrite(
            raw_query=seed_query,
            n=3,
            as_of_time=self._clock.now(resolved_author).isoformat(),
        )

        topic_agent = make_agent(
            role="Content Planner",
            goal="Identify a unique documentation gap based on your expertise and org history.",
            backstory=self._persona(
                resolved_author,
                mem=self._mem,
                graph_dynamics=self._gd,
                extra=(
                    f"You are {resolved_author} from the {dept} department. "
                    f"Your expertise includes: {expertise_str}. "
                    f"You are deciding what to document today to provide the most value to the team."
                ),
            ),
            llm=self._worker,
        )

        topic_task = Task(
            description=(
                f"Today's Org Theme: {daily_theme}\n"
                f"Recent Activity Context: {topic_ctx}\n\n"
                f"EXISTING DOCUMENTATION (Do NOT duplicate or overlap significantly):\n"
                f"{history_str}\n\n"
                f"TASK:\n"
                f"Based on your expertise ({expertise_str}), propose ONE specific Confluence page title "
                f"you would plausibly write TODAY. \n\n"
                f"Rules:\n"
                f"- The topic MUST fall within your area of expertise ({expertise_str}). "
                f"Do NOT propose engineering, infrastructure, or backend topics unless that is your expertise.\n"
                f"- Find a specific 'gap'. If a topic is already documented, look for a sub-topic or angle not yet covered.\n"
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

        # Sanity-trim in case the LLM adds extra prose
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

        writer_agent = make_agent(
            role="Corporate Writer",
            goal=f"Draft a {title} Confluence page.",
            backstory=self._persona(
                resolved_author,
                mem=self._mem,
                graph_dynamics=self._gd,
                extra=(
                    f"You are {resolved_author} from the {dept} department. "
                    f"Your expertise includes: {expertise_str}. "
                ),
            ),
            llm=self._planner,
        )
        task = Task(
            description=(
                f"Write a single Confluence page with ID {conf_id} titled '{title}'.\n"
                f"Context from memory: {ctx}\n"
                f"Existing pages you may reference (and ONLY these):\n{related}\n\n"
                f"Rules:\n"
                f"- Use your specific technical expertise and typing style.\n"
                f"- If stressed, the doc may be shorter or more blunt.\n"
                f"- Do not invent any CONF-* IDs not listed above.\n"
                f"- Format as Markdown. Do not write a main # title or metadata block (like Author/Date) at the top.\n"
                f"- Start directly with the first paragraph or ## section."
            ),
            expected_output=f"A single Markdown Confluence page with ID {conf_id}.",
            agent=writer_agent,
        )
        raw = str(
            Crew(agents=[writer_agent], tasks=[task], verbose=False).kickoff()
        ).strip()
        raw += self._bill_gap_warning(title)

        # Lifecycle scan before chunking
        self._lifecycle.scan_for_knowledge_gaps(
            text=raw,
            triggered_by=conf_id,
            day=self._state.day,
            date_str=date_str,
            state=self._state,
            timestamp=timestamp,
        )

        self._finalise_page(
            raw_content=raw,
            conf_id=conf_id,
            title=title,
            author=resolved_author,
            date_str=date_str,
            timestamp=timestamp,
            subdir="general",
            tags=["confluence", "adhoc"],
            facts={"title": title, "adhoc": True},
        )

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE — PAGE FINALISATION PIPELINE
    # ─────────────────────────────────────────────────────────────────────────

    def _finalise_page(
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
    ) -> List[str]:
        """
        Common finalisation pipeline for every Confluence page:
          1. Strip broken cross-references
          2. Register ID (raises DuplicateArtifactError — caller handles)
          3. Chunk into child pages if content is long
          4. Save .md files, embed, log SimEvents

        Returns list of all conf_ids created (parent + children).
        """
        # 1. Strip any CONF-* references that aren't registered yet
        clean_content = self._registry.strip_broken_references(raw_content)

        # 2. Chunk into focused child pages (or single page if short enough)
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
            # _registry.chunk_into_pages already registered IDs —
            # catch the rare race where an ID was registered externally
            logger.info(
                f"[finalise] embedding page.id={page.id} parent={page.parent_id or 'ROOT'} content_len={len(page.content)}"
            )
            try:
                # strip broken refs one more time after chunking added headers
                final_content = self._registry.strip_broken_references(page.content)
            except Exception as e:
                logger.info(f"[finalise] Caught exception {e}")
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

            self._mem.embed_artifact(
                id=page.id,
                type="confluence",
                title=page.title,
                content=final_content,
                day=self._state.day,
                date=date_str,
                timestamp=timestamp,
                metadata=meta,
            )

            if page.parent_id is None and author:
                self._mem._db["author_expertise"].update_one(
                    {"author": author},
                    {"$addToSet": {"topics": page.title.lower().strip()}},
                    upsert=True,
                )

            self._state.daily_artifacts_created += 1

            logger.debug(f"[finalise] pre-facts page.id={page.id}")
            page_facts = dict(facts)

            page_facts.update(
                {
                    "parent_id": page.parent_id or "",
                    "is_chunk": page.parent_id is not None,
                    "title": page.title,
                    "author": author,
                }
            )

            logger.info(f"[finalise] page facts {page_facts}")
            logger.debug(f"[finalise] pre-artifact-ids page.id={page.id}")
            artifact_ids = {"confluence": page.id}
            if extra_artifact_ids:
                artifact_ids.update(extra_artifact_ids)

            logger.debug(f"[finalise] pre-log-event page.id={page.id}")
            logger.debug(
                f"[finalise] SimEvent fields — "
                f"type=confluence_created "
                f"timestamp={timestamp} "
                f"day={self._state.day} "
                f"date={date_str} "
                f"actors={[author]} "
                f"artifact_ids={artifact_ids} "
                f"facts={page_facts} "
                f"summary={'Child' if page.parent_id else 'Page'} {page.id} created: {page.title} "
                f"tags={tags}"
            )
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
                        f"{'Child' if page.parent_id else 'Page'} {page.id} created: {page.title}"
                    ),
                    tags=tags,
                )
            )
            logger.debug(f"[finalise] post-log-event page.id={page.id}")

            logger.info(f"[confluence] _finalise_page complete: {page.id}")

            created_ids.append(page.id)

        return created_ids

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE — JIRA TICKET SPAWNING
    # ─────────────────────────────────────────────────────────────────────────

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
        for tk in new_tickets:
            tid = self._registry.next_jira_id()
            self._registry.register_jira(tid)
            assignee = tk.get("assignee", fallback_author)
            if assignee not in self._all_names:
                assignee = fallback_author

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
            }
            self._mem.upsert_ticket(ticket)
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

    def _pick_dept_author(self, prefix: str) -> str:
        """Return a random member of the department matching prefix, fallback to any employee."""
        for dept, members in self._org_chart.items():
            if prefix.upper() in dept.upper() and members:
                return random.choice(members)
        # Fallback: any employee at all
        return random.choice(self._all_names)

    def generate_tech_stack(self) -> dict:
        """
        Ask the LLM to invent a plausible tech stack for this company and industry.
        Persists to MongoDB immediately so every subsequent LLM call can reference it.
        Returns the stack as a dict.
        """
        # Check if already generated (warm restart guard)
        existing = self._mem.get_tech_stack()
        if existing:
            logger.info("[confluence] Tech stack already exists — skipping generation.")
            return existing

        agent = make_agent(
            role="Principal Engineer",
            goal="Define the canonical technology stack for this company.",
            backstory=(
                f"You are a principal engineer at {self._company}, "
                f"a {self._industry} company. You are documenting the actual "
                f"technologies in use — not aspirational, not greenfield. "
                f"This is a company with years of history and legacy decisions."
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
                "[confluence] Tech stack JSON parse failed — using minimal fallback."
            )
            stack = {
                "notable_quirks": "Stack unknown — legacy system predates documentation."
            }

        self._mem.save_tech_stack(stack)
        logger.info(f"[confluence] ✓ Tech stack established: {list(stack.keys())}")
        return stack

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE — UTILITIES
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _render(template: str, vars_: Dict[str, str]) -> str:
        for key, value in vars_.items():
            template = template.replace(f"{{{key}}}", str(value))
        return template

    def _render_template(self, template: str) -> str:
        """Apply simulation-level placeholder substitutions."""
        return (
            template.replace("{legacy_system}", self._legacy.get("name", ""))
            .replace("{project_name}", self._legacy.get("project_name", ""))
            .replace("{company_name}", self._company)
            .replace("{industry}", self._industry)
        )

    @staticmethod
    def _extract_title(content: str, fallback: str) -> str:
        # Strip code blocks before searching for headings
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
        """Extract prefix from a conf_id like CONF-ENG-003 → ENG."""
        parts = conf_id.split("-")
        return parts[1] if len(parts) >= 3 else "GEN"

    def _bill_gap_warning(self, topic: str) -> str:
        """Append a knowledge-gap warning if the topic touches a departed employee's domain."""
        departed = self._config.get("knowledge_gaps", [])
        for emp in departed:
            hits = [k for k in emp.get("knew_about", []) if k.lower() in topic.lower()]
            if hits:
                return (
                    f"\n\n> ⚠️ **Knowledge Gap**: This area ({', '.join(hits)}) was owned by "
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
        """
        Back-calculate what an employee's tenure label should read
        on a historical page_date, given their tenure string at sim_start.

        "5yr" at 2026-03-02, page dated 2024-03-02  →  "3yr"
        "2yr" at 2026-03-02, page dated 2025-09-02  →  "6mo"
        "new" / unparseable                          →  returned unchanged
        """
        import re
        from dateutil.relativedelta import relativedelta

        m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(yr|mo)", tenure_str.strip())
        if not m:
            return tenure_str  # "new" or anything non-standard — leave as-is

        value, unit = float(m.group(1)), m.group(2)
        months_at_sim_start = int(value * 12) if unit == "yr" else int(value)

        # How many months before sim_start is this page?
        delta = relativedelta(sim_start, page_date)
        months_offset = delta.years * 12 + delta.months

        months_at_page = months_at_sim_start - months_offset
        if months_at_page <= 0:
            return "new"  # hadn't joined yet or just started

        if months_at_page < 12:
            return f"{months_at_page}mo"
        years = months_at_page // 12
        remainder = months_at_page % 12
        return f"{years}yr" if remainder == 0 else f"{years}.{remainder // 1}yr"
