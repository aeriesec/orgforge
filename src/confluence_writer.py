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

import json
import logging
import random
import re
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from crewai import Agent, Task, Crew
from memory import Memory, SimEvent
from artifact_registry import ArtifactRegistry, DuplicateArtifactError, ConfluencePage

if TYPE_CHECKING:
    from graph_dynamics import GraphDynamics

logger = logging.getLogger("orgforge.confluence")


# ─────────────────────────────────────────────────────────────────────────────
# CONFLUENCE WRITER
# ─────────────────────────────────────────────────────────────────────────────

class ConfluenceWriter:

    def __init__(
        self,
        mem:              Memory,
        registry:         ArtifactRegistry,
        state,                              # flow.State — avoid circular import
        config:           Dict,
        worker_llm,
        planner_llm,
        clock,
        lifecycle,
        persona_helper,
        graph_dynamics:   "GraphDynamics",
        base_export_dir:  str = "./export",
    ):
        self._mem        = mem
        self._registry   = registry
        self._state      = state
        self._config     = config
        self._worker     = worker_llm
        self._planner    = planner_llm
        self._clock      = clock
        self._lifecycle  = lifecycle
        self._persona    = persona_helper
        self._gd         = graph_dynamics
        self._base       = base_export_dir
        self._company    = config["simulation"]["company_name"]
        self._industry   = config["simulation"].get("industry", "technology")
        self._legacy     = config.get("legacy_system", {})
        self._all_names  = [n for dept in config["org_chart"].values() for n in dept]
        self._org_chart  = config["org_chart"]

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────────────────────

    def write_genesis_batch(
        self,
        prefix:     str,
        count:      int,
        prompt_tpl: str,
        authors:    List[str],
        extra_vars: Optional[Dict[str, str]] = None,
        subdir:     str = "archives",
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
                           {project_name}, {authors}, {related_pages}
            authors:     List of persona names to reference as authors.
            extra_vars:  Any additional {placeholder} → value substitutions.
            subdir:      Export subdirectory under confluence/.

        Returns:
            List of registered conf_ids in generation order.
        """
        # 1. Allocate all IDs before any LLM call
        ids = [self._registry.next_id(prefix) for _ in range(count)]
        logger.info(f"[confluence] Genesis batch allocated: {ids}")

        genesis_time = self._clock.now("system").isoformat()
        registered_ids: List[str] = []

        for conf_id in ids:
            # 2. Build related-pages context from pages registered so far
            related = self._registry.related_context(topic=conf_id, n=5)

            # 3. Render prompt — one page, all constraints explicit
            vars_ = {
                "id":           conf_id,
                "company":      self._company,
                "industry":     self._industry,
                "legacy_system": self._legacy.get("name", ""),
                "project_name": self._legacy.get("project_name", ""),
                "authors":      ", ".join(authors),
                "related_pages": related or "None yet.",
                **(extra_vars or {}),
            }
            prompt = self._render(prompt_tpl, vars_)

            # 4. Single-page LLM call
            historian = Agent(
                role="Corporate Historian",
                goal="Write one authentic internal Confluence page.",
                backstory=(
                    f"You work at {self._company}, a {self._industry} company. "
                    f"The legacy system '{self._legacy.get('name', '')}' is known to be unstable. "
                    f"Write with real insider detail."
                ),
                llm=self._planner,
            )
            task = Task(
                description=prompt,
                expected_output=(
                    f"A single Markdown Confluence page with ID {conf_id}. "
                    f"No separators. No preamble. Start directly with the # heading."
                ),
                agent=historian,
            )
            raw = str(Crew(agents=[historian], tasks=[task], verbose=False).kickoff()).strip()

            # 5. Register, validate, chunk, save, embed
            conf_ids = self._finalise_page(
                raw_content=raw,
                conf_id=conf_id,
                title=self._extract_title(raw, conf_id),
                authors=authors,
                date_str=str(self._state.current_date.date()),
                timestamp=genesis_time,
                subdir=subdir,
                tags=["genesis", "confluence"],
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
        incident_id:   str,
        incident_title: str,
        root_cause:    str,
        days_active:   int,
        on_call:       str,
        eng_peer:      str,
    ) -> str:
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

        writer = Agent(
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
                f"Format as Markdown. Start directly with the # heading. "
                f"Include: Executive Summary, Timeline, Root Cause, Impact, "
                f"What Went Wrong, What Went Right, Action Items."
            ),
            expected_output=f"A single Markdown postmortem page with ID {conf_id}.",
            agent=writer,
        )
        raw = str(Crew(agents=[writer], tasks=[task], verbose=False).kickoff()).strip()

        conf_ids = self._finalise_page(
            raw_content=raw,
            conf_id=conf_id,
            title=f"Postmortem: {incident_title}",
            authors=[on_call, eng_peer],
            date_str=date_str,
            timestamp=timestamp,
            subdir="postmortems",
            tags=["postmortem", "confluence"],
            facts={"root_cause": root_cause, "incident_id": incident_id},
            extra_artifact_ids={"jira": incident_id},
        )
        logger.info(f"    [green]📄 Postmortem:[/green] {conf_ids[0]}")
        return conf_ids[0]

    def write_design_doc(
        self,
        author:           str,
        participants:     List[str],
        topic:            str,
        slack_transcript: List[Dict],
        date_str:         str,
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
        ctx = self._mem.context_for_prompt(topic, n=3, as_of_time=timestamp)
        related = self._registry.related_context(topic=topic, n=3)

        agent = Agent(
            role="Technical Lead",
            goal="Document technical decisions and extract actionable tickets.",
            backstory=(
                f"You are {author} at {self._company}. You just finished a Slack "
                f"discussion and need to document decisions and assign follow-up work."
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
                f"  \"markdown_doc\": \"string (full Markdown, starting with ## Problem Statement)\",\n"
                f"  \"new_tickets\": [\n"
                f"    {{\"title\": \"string\", "
                f"\"assignee\": \"string (must be one of: {', '.join(participants)})\", "
                f"\"story_points\": 1|2|3|5|8}}\n"
                f"  ]\n"
                f"}}"
            ),
            expected_output="Valid JSON only. No markdown fences.",
            agent=agent,
        )
        raw = str(Crew(agents=[agent], tasks=[task], verbose=False).kickoff()).strip()
        clean = raw.replace("```json", "").replace("```", "").strip()

        try:
            parsed = json.loads(clean)
            content = parsed.get("markdown_doc", "Draft pending.")
            new_tickets = parsed.get("new_tickets", [])
        except json.JSONDecodeError as e:
            logger.warning(f"[confluence] JSON parse failed for design doc: {e}")
            content = clean   # save whatever the LLM produced
            new_tickets = []

        conf_ids = self._finalise_page(
            raw_content=content,
            conf_id=conf_id,
            title=f"Design: {topic[:50]}",
            authors=participants,
            date_str=date_str,
            timestamp=timestamp,
            subdir="design",
            tags=["confluence", "design_doc"],
            facts={"title": f"Design: {topic[:50]}", "type": "design_doc"},
        )

        # Spawn JIRA tickets from action items
        created_ticket_ids = self._spawn_tickets(
            new_tickets, author, participants, date_str, timestamp
        )

        # Update SimEvent to include spawned tickets
        self._mem.log_event(SimEvent(
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
                "title": f"Design: {topic[:50]}",
                "type": "design_doc",
                "spawned_tickets": created_ticket_ids,
            },
            summary=(
                f"{author} created {conf_ids[0]} and spawned "
                f"{len(created_ticket_ids)} ticket(s): {', '.join(created_ticket_ids)}"
            ),
            tags=["confluence", "design_doc", "jira"],
        ))

        logger.info(
            f"    [dim]📄 Design doc: {conf_ids[0]} "
            f"(spawned {len(created_ticket_ids)} ticket(s))[/dim]"
        )
        return conf_ids[0]

    def write_adhoc_page(
        self,
        author:    Optional[str] = None,
        backstory: Optional[str] = None,
    ) -> None:
        """
        Generate a character-accurate ad-hoc Confluence page.

        Mirrors the original _generate_adhoc_confluence_page() in flow.py but
        with Python-allocated IDs, reference injection, and chunking.
        """
        raw_topics = self._config.get(
            "adhoc_confluence_topics",
            [["ENG", "Documentation"], ["HR", "Policy Update"]],
        )
        rendered = [(t[0], self._render_template(t[1])) for t in raw_topics]
        prefix, title = random.choice(rendered)

        conf_id = self._registry.next_id(prefix)
        resolved_author = author or random.choice(self._all_names)

        if backstory is None:
            backstory = self._persona(
                resolved_author, mem=self._mem, graph_dynamics=self._gd
            )

        session_hours = random.randint(30, 90) / 60.0
        artifact_time, _ = self._clock.advance_actor(resolved_author, hours=session_hours)
        timestamp = artifact_time.isoformat()
        date_str = str(self._state.current_date.date())

        ctx = self._mem.context_for_prompt(title, n=3, as_of_time=timestamp)
        related = self._registry.related_context(topic=title, n=4)

        writer_agent = Agent(
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
                f"- If stressed, the doc may be shorter or more blunt.\n"
                f"- Do not invent any CONF-* IDs not listed above.\n"
                f"- Start directly with the # heading. Format as Markdown."
            ),
            expected_output=f"A single Markdown Confluence page with ID {conf_id}.",
            agent=writer_agent,
        )
        raw = str(Crew(agents=[writer_agent], tasks=[task], verbose=False).kickoff()).strip()
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
            authors=[resolved_author],
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
        raw_content:        str,
        conf_id:            str,
        title:              str,
        authors:            List[str],
        date_str:           str,
        timestamp:          str,
        subdir:             str,
        tags:               List[str],
        facts:              Dict,
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
            author=authors[0] if authors else "",
            date_str=date_str,
        )

        created_ids: List[str] = []
        for page in pages:
            # _registry.chunk_into_pages already registered IDs —
            # catch the rare race where an ID was registered externally
            try:
                # strip broken refs one more time after chunking added headers
                final_content = self._registry.strip_broken_references(page.content)
            except Exception:
                final_content = page.content

            self._save_md(page.path, final_content)

            entry = {"id": page.id, "title": page.title, "path": page.path}
            self._state.confluence_pages.append(entry)

            self._mem.embed_artifact(
                id=page.id,
                type="confluence",
                title=page.title,
                content=final_content,
                day=self._state.day,
                date=date_str,
                timestamp=timestamp,
                metadata={
                    "authors":   str(authors),
                    "parent_id": page.parent_id or "",
                    "is_chunk":  page.parent_id is not None,
                },
            )
            self._state.daily_artifacts_created += 1

            page_facts = dict(facts)
            page_facts.update({
                "parent_id": page.parent_id or "",
                "is_chunk":  page.parent_id is not None,
            })

            artifact_ids = {"confluence": page.id}
            if extra_artifact_ids:
                artifact_ids.update(extra_artifact_ids)

            self._mem.log_event(SimEvent(
                type="confluence_created",
                timestamp=timestamp,
                day=self._state.day,
                date=date_str,
                actors=authors,
                artifact_ids=artifact_ids,
                facts=page_facts,
                summary=(
                    f"{'Child' if page.parent_id else 'Page'} {page.id} created: {page.title}"
                ),
                tags=tags,
            ))

            created_ids.append(page.id)

        return created_ids

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE — JIRA TICKET SPAWNING
    # ─────────────────────────────────────────────────────────────────────────

    def _spawn_tickets(
        self,
        new_tickets:  List[Dict],
        fallback_author: str,
        valid_names:  List[str],
        date_str:     str,
        timestamp:    str,
    ) -> List[str]:
        """Create JIRA tickets from a list of LLM-extracted action items."""
        created_ids: List[str] = []
        for tk in new_tickets:
            tid = f"ORG-{len(self._state.jira_tickets) + 100}"
            assignee = tk.get("assignee", fallback_author)
            if assignee not in self._all_names:
                assignee = fallback_author

            ticket = {
                "id":           tid,
                "title":        tk.get("title", "Generated Task"),
                "status":       "To Do",
                "assignee":     assignee,
                "sprint":       getattr(self._state.sprint, "sprint_number", 1),
                "story_points": tk.get("story_points", 2),
                "linked_prs":   [],
                "created_at":   timestamp,
                "updated_at":   timestamp,
            }
            self._state.jira_tickets.append(ticket)
            self._save_json(f"{self._base}/jira/{tid}.json", ticket)
            self._mem.embed_artifact(
                id=tid, type="jira", title=ticket["title"],
                content=json.dumps(ticket),
                day=self._state.day, date=date_str,
                metadata={"assignee": assignee},
                timestamp=timestamp,
            )
            self._state.daily_artifacts_created += 1
            created_ids.append(tid)
        return created_ids

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
        return (template
            .replace("{legacy_system}",  self._legacy.get("name", ""))
            .replace("{project_name}",   self._legacy.get("project_name", ""))
            .replace("{company_name}",   self._company)
            .replace("{industry}",       self._industry)
        )

    @staticmethod
    def _extract_title(content: str, fallback: str) -> str:
        m = re.search(r"^#\s+(.+)", content, re.MULTILINE)
        return m.group(1).strip() if m else f"Archive: {fallback}"

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
