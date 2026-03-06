"""
artifact_registry.py
=====================
Central ID allocation and integrity registry for all OrgForge artifacts.

The registry is the single authority for artifact IDs. No caller should
construct an ID manually — always call next_id() or next_jira_id() first.
This eliminates duplicate IDs across Confluence pages, JIRA tickets, and
any future artifact type by using the same underlying allocator pattern.

Responsibilities:
  1. ID allocation     — deterministic, pre-reserved, warm-restart safe
  2. ID uniqueness     — raises DuplicateArtifactError on collision
  3. Reference validation — scans Markdown for CONF-* citations, reports broken refs
  4. Page chunking     — splits long Confluence content into focused child pages
  5. Ticket summary    — builds a structured TicketSummary context object so the
                         LLM always has title + full state, never just raw comments

Supported namespaces (extensible via the same _allocate() core):
  Confluence  →  next_id("ENG")      → "CONF-ENG-001"
                 next_id("MKT")      → "CONF-MKT-001"
  JIRA        →  next_jira_id()      → "ORG-100"

Callers that previously constructed IDs manually:

  flow.py          next_conf_id()         → registry.next_id(prefix)
  flow.py          next_jira_id()         → registry.next_jira_id()
  normal_day.py    f"CONF-ENG-{len...}"   → registry.next_id("ENG")
  normal_day.py    f"ORG-{len...}"        → registry.next_jira_id()
  ticket_assigner  (no ID creation)       — unaffected
  org_lifecycle    (no ID creation)       — unaffected
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from memory import Memory
    from flow import State

logger = logging.getLogger("orgforge.registry")

# ── Patterns ──────────────────────────────────────────────────────────────────
_CONF_REF_RE = re.compile(r"\bCONF-[A-Z]+-\d+\b")

# ── Chunking defaults ─────────────────────────────────────────────────────────
# Cohere Embed v4 supports 128K tokens, but focused pages improve retrieval
# precision. ~3 000 chars ≈ ~750 tokens — well inside any embedder's window.
DEFAULT_CHUNK_CHARS   = 3_000
DEFAULT_CHUNK_OVERLAP = 200


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ConfluencePage:
    """A single generated (or chunked) Confluence page ready to be saved."""
    id:        str
    title:     str
    content:   str
    path:      str
    parent_id: Optional[str] = None
    child_ids: List[str]     = field(default_factory=list)


@dataclass
class TicketSummary:
    """
    Structured JIRA ticket context for LLM prompt injection.

    Carries full identity (id + title) alongside live state so the LLM
    never has to infer what a ticket is about from an ID alone, and never
    loses track of whether the ticket was previously blocked or how many
    days it has been active.

    Usage:
        summary = registry.ticket_summary(ticket, current_day)
        prompt  = f"Work on this ticket:\\n{summary.for_prompt()}"
    """
    id:            str
    title:         str
    status:        str
    assignee:      str
    story_points:  Optional[int]
    days_active:   int
    comment_count: int
    last_comments: List[Dict]    # last 3 comments, normalised to author/date/text
    linked_prs:    List[str]
    was_blocked:   bool
    sprint:        Optional[int]

    def for_prompt(self) -> str:
        """Compact, human-readable block for LLM prompt injection."""
        lines = [
            f"Ticket:       [{self.id}] {self.title}",
            (
                f"Status:       {self.status}"
                f"  |  Assignee: {self.assignee}"
                + (f"  |  Sprint: {self.sprint}" if self.sprint else "")
            ),
            (
                f"Story points: {self.story_points or '?'}"
                f"  |  Days active: {self.days_active}"
                f"  |  Total comments: {self.comment_count}"
            ),
            f"Blocked previously: {'Yes' if self.was_blocked else 'No'}",
        ]
        if self.linked_prs:
            lines.append(f"Linked PRs:   {', '.join(self.linked_prs)}")
        if self.last_comments:
            lines.append("Recent comments:")
            for c in self.last_comments:
                lines.append(
                    f"  - {c.get('author', '?')} "
                    f"({c.get('date', '?')}): "
                    f"{c.get('text', '').strip()}"
                )
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# EXCEPTIONS
# ─────────────────────────────────────────────────────────────────────────────

class DuplicateArtifactError(Exception):
    """Raised when an artifact ID is registered more than once."""


# ─────────────────────────────────────────────────────────────────────────────
# REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

class ArtifactRegistry:
    """
    Central registry for all OrgForge artifact IDs.

    Two namespaces, one allocation pattern:

      _confluence : Dict[str, str]   CONF-ENG-001 → title (or "__reserved__")
      _jira       : Dict[str, int]   ORG-100      → 100  (or 0 = reserved)

    Both use _allocate() which:
      1. Finds the current high-water mark in the dict
      2. Increments by 1
      3. Pre-reserves the slot so concurrent same-batch calls can't
         allocate the same ID before register() is called
      4. Returns the new ID

    Both caches are seeded from MongoDB on startup so warm restarts never
    reuse IDs from a previous simulation run.
    """

    def __init__(self, mem: "Memory", base_export_dir: str = "./export"):
        self._mem  = mem
        self._base = base_export_dir

        self._confluence: Dict[str, str] = {}   # CONF-ENG-001 → title
        self._jira:       Dict[str, int] = {}   # ORG-100      → 100

        self._seed_from_mongo()

    # ─────────────────────────────────────────────
    # SEEDING
    # ─────────────────────────────────────────────

    def _seed_from_mongo(self) -> None:
        """Populate both caches from MongoDB on startup."""
        try:
            for doc in self._mem._artifacts.find(
                {"type": "confluence"}, {"_id": 1, "title": 1}
            ):
                self._confluence[doc["_id"]] = doc.get("title", "")

            for doc in self._mem._artifacts.find(
                {"type": "jira"}, {"_id": 1}
            ):
                jid    = doc["_id"]
                suffix = jid.replace("ORG-", "")
                if suffix.isdigit():
                    self._jira[jid] = int(suffix)

            logger.info(
                f"[registry] Seeded {len(self._confluence)} Confluence + "
                f"{len(self._jira)} JIRA IDs from MongoDB."
            )
        except Exception as e:
            logger.warning(f"[registry] Could not seed from MongoDB: {e}")

    # ─────────────────────────────────────────────
    # SHARED ALLOCATOR CORE
    # ─────────────────────────────────────────────

    @staticmethod
    def _allocate(
        store:             Dict,
        existing_nums_fn,       # callable(store) → List[int]
        make_id_fn,             # callable(n: int) → str
        reserve_value:     Any,
    ) -> str:
        """
        Generic ID allocator used by both namespaces.

        Adding a new artifact type means passing two small lambdas —
        the sequencing logic itself never needs to be duplicated.
        """
        nums   = existing_nums_fn(store)
        next_n = max(nums, default=0) + 1
        new_id = make_id_fn(next_n)
        store[new_id] = reserve_value
        return new_id

    # ─────────────────────────────────────────────
    # CONFLUENCE  —  allocation & registration
    # ─────────────────────────────────────────────

    def next_id(self, prefix: str) -> str:
        """
        Allocate the next available Confluence ID for a given prefix.

        next_id("ENG") → "CONF-ENG-001"
        next_id("ENG") → "CONF-ENG-002"   (second call)
        next_id("MKT") → "CONF-MKT-001"   (independent sequence)
        """
        pat    = f"CONF-{prefix}-"
        new_id = self._allocate(
            store             = self._confluence,
            existing_nums_fn  = lambda s: [
                int(k[len(pat):])
                for k in s
                if k.startswith(pat) and k[len(pat):].isdigit()
            ],
            make_id_fn        = lambda n: f"CONF-{prefix}-{n:03d}",
            reserve_value     = "__reserved__",
        )
        logger.debug(f"[registry] Allocated Confluence ID: {new_id}")
        return new_id

    def register_confluence(self, conf_id: str, title: str) -> None:
        """
        Confirm a pre-allocated Confluence ID with its final title.
        Raises DuplicateArtifactError if already confirmed.
        """
        current = self._confluence.get(conf_id)
        if current is not None and current != "__reserved__":
            raise DuplicateArtifactError(
                f"[registry] Duplicate Confluence ID '{conf_id}'. "
                f"Existing title: '{current}'."
            )
        self._confluence[conf_id] = title
        logger.debug(f"[registry] Confirmed Confluence {conf_id}: {title}")

    # Backward-compat alias kept so old callers and chunk_into_pages don't break
    def register(self, conf_id: str, title: str, content: str = "") -> None:
        self.register_confluence(conf_id, title)

    def confluence_exists(self, conf_id: str) -> bool:
        val = self._confluence.get(conf_id)
        return val is not None and val != "__reserved__"

    def all_confluence_ids(self) -> List[str]:
        return [k for k, v in self._confluence.items() if v != "__reserved__"]

    # ─────────────────────────────────────────────
    # JIRA  —  allocation & registration
    # ─────────────────────────────────────────────

    def next_jira_id(self) -> str:
        """
        Allocate the next available JIRA ticket ID.

        Starts at ORG-100 (matching the original convention) and increments
        from there. Each call pre-reserves the slot immediately.
        """
        new_id = self._allocate(
            store             = self._jira,
            # Parse the highest number from the keys, not the values
            existing_nums_fn  = lambda s: [
                int(k.replace("ORG-", "")) 
                for k in s.keys() 
                if k.startswith("ORG-") and k.replace("ORG-", "").isdigit()
            ] if s else [99],
            make_id_fn        = lambda n: f"ORG-{n}",
            reserve_value     = 0,   # 0 = reserved sentinel; confirmed in register_jira
        )
        logger.debug(f"[registry] Allocated JIRA ID: {new_id}")
        return new_id

    def register_jira(self, jira_id: str) -> None:
        """
        Confirm a pre-allocated JIRA ID.
        Raises DuplicateArtifactError if already confirmed.
        """
        suffix = jira_id.replace("ORG-", "")
        if not suffix.isdigit():
            raise ValueError(f"[registry] Malformed JIRA ID: '{jira_id}'")
        n       = int(suffix)
        current = self._jira.get(jira_id)
        if current is not None and current != 0:
            raise DuplicateArtifactError(
                f"[registry] Duplicate JIRA ID '{jira_id}'."
            )
        self._jira[jira_id] = n
        logger.debug(f"[registry] Confirmed JIRA {jira_id}")

    def jira_exists(self, jira_id: str) -> bool:
        val = self._jira.get(jira_id)
        return val is not None and val != 0

    def all_jira_ids(self) -> List[str]:
        return [k for k, v in self._jira.items() if v != 0]

    # ─────────────────────────────────────────────
    # TICKET SUMMARY
    # ─────────────────────────────────────────────

    def ticket_summary(self, ticket: Dict, current_day: int) -> TicketSummary:
        """
        Build a TicketSummary from a raw JIRA ticket dict.

        This is the single place that decides what ticket context the LLM
        sees. Centralising it here means every caller — normal_day,
        ticket_assigner, postmortem writer — gets identical, consistent
        context with no risk of different code paths slicing comments
        differently or forgetting the title.

        Args:
            ticket:      Raw ticket dict from state.jira_tickets
            current_day: state.day — used to compute days_active

        Returns:
            TicketSummary with a .for_prompt() method for prompt injection
        """
        comments    = ticket.get("comments", [])
        created_day = ticket.get("created_day", current_day)
        was_blocked = any(
            any(kw in c.get("text", "").lower()
                for kw in ("blocked", "blocker", "waiting on", "stuck", "can't proceed"))
            for c in comments
        )

        # Normalise comment dicts — strip internal engine fields before
        # passing to the LLM, and remove quote-wrapping on stored text.
        clean_comments = [
            {
                "author": c.get("author", "?"),
                "date":   c.get("date",   "?"),
                "text":   c.get("text",   "").strip('"'),
            }
            for c in comments[-3:]
        ]

        return TicketSummary(
            id            = ticket["id"],
            title         = ticket.get("title", "Untitled"),
            status        = ticket.get("status", "To Do"),
            assignee      = ticket.get("assignee", "Unassigned"),
            story_points  = ticket.get("story_points"),
            days_active   = max(0, current_day - created_day),
            comment_count = len(comments),
            last_comments = clean_comments,
            linked_prs    = ticket.get("linked_prs", []),
            was_blocked   = was_blocked,
            sprint        = ticket.get("sprint"),
        )

    # ─────────────────────────────────────────────
    # CONTEXT FOR PROMPTS
    # ─────────────────────────────────────────────

    def related_context(self, topic: str, n: int = 5) -> str:
        """
        Return a bullet list of existing confirmed Confluence page IDs +
        titles for injection into LLM prompts as the allowed reference set.

        Returns "None yet." if no pages have been confirmed yet.
        """
        candidates = [
            (id_, title)
            for id_, title in self._confluence.items()
            if title not in ("__reserved__", "")
        ]
        if not candidates:
            return "None yet."
        return "\n".join(
            f"- {id_}: {title}" for id_, title in candidates[-n:]
        )

    # ─────────────────────────────────────────────
    # REFERENCE VALIDATION
    # ─────────────────────────────────────────────

    def validate_references(self, content: str) -> List[str]:
        """
        Scan Markdown for CONF-* references.
        Returns sorted list of any IDs not yet confirmed.
        """
        found = set(_CONF_REF_RE.findall(content))
        return sorted(ref for ref in found if not self.confluence_exists(ref))

    def strip_broken_references(self, content: str) -> str:
        """
        Replace unresolved CONF-* citations with a clear placeholder.

        "See CONF-ENG-099"  →  "See [CONF-ENG-099 — not yet created]"
        """
        broken = set(self.validate_references(content))
        if not broken:
            return content

        def _replace(m: re.Match) -> str:
            ref = m.group(0)
            return f"[{ref} — not yet created]" if ref in broken else ref

        result = _CONF_REF_RE.sub(_replace, content)
        logger.warning(
            f"[registry] Replaced {len(broken)} unresolved ref(s): {sorted(broken)}"
        )
        return result

    # ─────────────────────────────────────────────
    # PAGE CHUNKING
    # ─────────────────────────────────────────────

    def chunk_into_pages(
        self,
        parent_id:    str,
        parent_title: str,
        content:      str,
        prefix:       str,
        state:        "State",
        author:       str = "",
        date_str:     str = "",
        subdir:       str = "general",
        max_chars:    int = DEFAULT_CHUNK_CHARS,
        overlap:      int = DEFAULT_CHUNK_OVERLAP,
    ) -> List[ConfluencePage]:
        """
        Split *content* into a family of focused child pages plus a parent
        index page that links to all of them.

        Splits on ## headings first; hard-splits oversized sections with
        overlap so context is not completely lost at boundaries.

        Returns [parent_index, child_1, child_2, ...].
        All IDs are registered immediately so later pages in the same
        generation batch can safely reference earlier siblings.
        """
        sections = self._split_on_headings(content, max_chars, overlap)

        if len(sections) <= 1:
            page = self._make_page(
                conf_id=parent_id, title=parent_title,
                content=content, author=author,
                date_str=date_str, subdir=subdir,
            )
            self.register_confluence(parent_id, parent_title)
            return [page]

        # Build child pages
        child_pages: List[ConfluencePage] = []
        for i, section_text in enumerate(sections):
            child_id    = f"{parent_id}-{str(i + 1).zfill(2)}"
            sec_title   = self._extract_first_heading(section_text) or f"Part {i + 1}"
            child_title = f"{parent_title} — {sec_title}"
            child_pages.append(self._make_page(
                conf_id=child_id, title=child_title,
                content=section_text, author=author,
                date_str=date_str, subdir=subdir,
                parent_id=parent_id, parent_title=parent_title,
            ))

        # Build parent index
        child_links = "\n".join(f"- [{p.title}]({p.id})" for p in child_pages)
        index_page  = self._make_page(
            conf_id=parent_id, title=parent_title,
            content=(
                f"This page is an index for **{parent_title}**.\n\n"
                f"## Contents\n\n{child_links}\n"
            ),
            author=author, date_str=date_str, subdir=subdir,
            child_ids=[p.id for p in child_pages],
        )

        # Register parent first, then children
        self.register_confluence(parent_id, parent_title)
        for cp in child_pages:
            self.register_confluence(cp.id, cp.title)

        logger.info(
            f"[registry] Chunked '{parent_title}' → "
            f"{len(child_pages)} child page(s) under {parent_id}."
        )
        return [index_page] + child_pages

    # ─────────────────────────────────────────────
    # PRIVATE HELPERS
    # ─────────────────────────────────────────────

    def _make_page(
        self,
        conf_id:      str,
        title:        str,
        content:      str,
        author:       str,
        date_str:     str,
        subdir:       str,
        parent_id:    Optional[str] = None,
        parent_title: Optional[str] = None,
        child_ids:    Optional[List[str]] = None,
    ) -> ConfluencePage:
        header_lines = [f"# {title}", f"**ID:** {conf_id}"]
        if author:
            header_lines.append(f"**Author:** {author}")
        if date_str:
            header_lines.append(f"**Date:** {date_str}")
        if parent_id:
            header_lines.append(
                f"**Parent:** [{parent_title or parent_id}]({parent_id})"
            )
        return ConfluencePage(
            id        = conf_id,
            title     = title,
            content   = "\n".join(header_lines) + "\n\n" + content.lstrip(),
            path      = f"{self._base}/confluence/{subdir}/{conf_id}.md",
            parent_id = parent_id,
            child_ids = child_ids or [],
        )

    @staticmethod
    def _split_on_headings(content: str, max_chars: int, overlap: int) -> List[str]:
        raw    = [s for s in re.split(r"(?=\n## )", content) if s.strip()]
        result: List[str] = []
        buffer = ""
        for section in raw:
            if len(buffer) + len(section) <= max_chars:
                buffer += section
            else:
                if buffer:
                    result.append(buffer)
                if len(section) > max_chars:
                    result.extend(ArtifactRegistry._hard_split(section, max_chars, overlap))
                    buffer = ""
                else:
                    buffer = section
        if buffer:
            result.append(buffer)
        return result if result else [content]

    @staticmethod
    def _hard_split(text: str, max_chars: int, overlap: int) -> List[str]:
        chunks, start = [], 0
        while start < len(text):
            chunks.append(text[start: start + max_chars])
            start += max_chars - overlap
        return chunks

    @staticmethod
    def _extract_first_heading(text: str) -> Optional[str]:
        m = re.search(r"^#{1,3}\s+(.+)", text, re.MULTILINE)
        return m.group(1).strip() if m else None
