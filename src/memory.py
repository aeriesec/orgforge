"""
memory.py
=========
Structured retrieval layer for OrgForge.

Usage:
    from memory import Memory
    mem = Memory()
"""

from datetime import datetime, timezone
import os
import json
import logging
from dataclasses import dataclass, field, asdict
import re
from typing import List, Dict, Optional, Any, Tuple, Union
import shutil
from pathlib import Path

from pymongo import MongoClient
from pymongo.operations import SearchIndexModel

logger = logging.getLogger("orgforge.memory")

MONGO_URI = os.environ.get(
    "MONGO_URI", "mongodb://localhost:27017/?directConnection=true"
)
DB_NAME = os.environ.get("DB_NAME", "orgforge")

_TICKET_PROGRESS_PROJECTION = {
    "_id": 0,
    "id": 1,
    "title": 1,
    "status": 1,
    "assignee": 1,
    "root_cause": 1,
    "description": 1,
    "linked_prs": 1,
    "gap_areas": 1,
    "sprint": 1,
    "story_points": 1,
    "comments": {"$slice": -3},
}

_EVENT_LOG_MAX_DAYS = 7


@dataclass
class SimEvent:
    type: str
    day: int
    date: str
    timestamp: str
    actors: List[str]
    artifact_ids: Dict[str, Any]
    facts: Dict[str, Any]
    summary: str
    tags: List[str] = field(default_factory=list)
    mongo_id: Optional[str] = field(default=None)

    def to_embed_text(self) -> str:
        return (
            f"[{self.type}] Day {self.day} ({self.date}). "
            f"Actors: {', '.join(self.actors)}. "
            f"Artifacts: {json.dumps(self.artifact_ids)}. "
            f"Facts: {json.dumps(self.facts)}. "
            f"Tags: {', '.join(self.tags)}. "
            f"Summary: {self.summary}"
        )

    def to_dict(self) -> Dict:
        return json.loads(json.dumps(asdict(self), default=str))

    @classmethod
    def from_dict(cls, d: Dict) -> "SimEvent":
        return cls(
            type=d.get("type", ""),
            day=d.get("day", 0),
            date=d.get("date", ""),
            timestamp=d.get("timestamp", ""),
            actors=d.get("actors", []),
            artifact_ids=d.get("artifact_ids", {}),
            facts=d.get("facts", {}),
            summary=d.get("summary", ""),
            tags=d.get("tags", []),
            mongo_id=d.get("_id"),
        )


class Memory:
    def __init__(
        self,
        mongo_uri: str = MONGO_URI,
        mongo_client=None,
    ):
        self._client = mongo_client or MongoClient(mongo_uri)
        self._db = self._client[DB_NAME]

        self._artifacts = self._db["artifacts"]
        self._events = self._db["events"]
        self._jira = self._db["jira_tickets"]
        self._prs = self._db["pull_requests"]
        self._checkpoints = self._db["checkpoints"]
        self._slack = self._db["slack_messages"]
        self._plans = self._db["dept_plans"]
        self._conversation_summaries = self._db["conversation_summaries"]
        self._confluence_pages = self._db["confluence_pages"]
        self._zoom_transcripts = self._db["zoom_transcripts"]
        
        self._jira.create_index([("id", 1)], unique=True)
        self._jira.create_index([("assignee", 1), ("status", 1)])
        self._prs.create_index([("pr_id", 1)], unique=True)
        self._prs.create_index([("reviewers", 1), ("status", 1)])
        self._slack.create_index([("channel", 1), ("ts", 1)])
        self._plans.create_index([("day", 1), ("dept", 1)])
        self._conversation_summaries.create_index(
            [("participants", 1), ("type", 1), ("day", -1)]
        )
        self._conversation_summaries.create_index([("day", -1)])
        self._events.create_index([("timestamp", 1)])
        self._events.create_index([("type", 1), ("day", 1)])
        self._events.create_index([("type", 1), ("timestamp", -1)])
        self._events.create_index([("actors", 1), ("timestamp", -1)])
        self._events.create_index([("type", 1), ("artifact_ids.jira", 1)])
        self._events.create_index([("tags", 1)])
        self._events.create_index([("type", 1), ("facts.participants", 1)])
        self._checkpoints.create_index([("day", -1)])
        self._jira.create_index([("dept", 1), ("status", 1)])
        self._events.create_index(
            [
                ("type", 1),
                ("facts.gap_classification", 1),
                ("timestamp", -1),
            ]
        )
        self._confluence_pages.create_index([("day", 1)])
        self._confluence_pages.create_index([("metadata.author", 1)])
        self._confluence_pages.create_index([("metadata.dept", 1)])
        self._zoom_transcripts.create_index([("day", 1)])
        self._zoom_transcripts.create_index([("metadata.participants", 1)])
        
        self._current_day: int = 0

        self._event_log: List[SimEvent] = []

        for coll_name in ("confluence_pages", "zoom_transcripts", "events"):
            if coll_name not in self._db.list_collection_names():
                self._db.create_collection(coll_name)

        self._init_text_indexes()

    def _init_text_indexes(self) -> None:
        """
        Create the Atlas Search text index on artifacts
        and a lightweight text index on events.

        The artifacts index follows the analyzer pattern from indexes.ts [16]:
          - canonical_name: whole_name_analyzer on index, lucene.standard on
            search, with a phrase multi-path using canonical_query_search_analyzer
          - aliases: whole_name_analyzer on index, aliases_light on search,
            with a shingle multi-path using alias_search_analyzer
          - type, day: filter tokens for scoped queries

        The events index is a lightweight standard-analyzer index over
        facts.root_cause and summary for the RecurrenceDetector text search leg.
        """
        artifacts_index_definition = {
            "analyzer": "lucene.standard",
            "mappings": {
                "dynamic": False,
                "fields": {
                    "_id": {"type": "token"},
                    "title": {
                        "type": "string",
                        "analyzer": "lucene.standard",
                    },
                    "aliases": {
                        "type": "string",
                        "analyzer": "lucene.standard",
                    },
                    "why_it_matters": {
                        "type": "string",
                        "analyzer": "lucene.standard",
                    },
                    "type": {"type": "token"},
                    "day": {"type": "number"},
                    "metadata.author": {"type": "token"},
                    "metadata.dept": {"type": "token"},
                },
            },
        }

        events_index_definition = {
            "analyzer": "lucene.standard",
            "mappings": {
                "dynamic": False,
                "fields": {
                    "type": {"type": "token"},
                    "day": {"type": "number"},
                    "facts.root_cause": {
                        "type": "string",
                        "analyzer": "lucene.standard",
                    },
                    "summary": {
                        "type": "string",
                        "analyzer": "lucene.standard",
                    },
                },
            },
        }

        index_specs = [
            ("confluence_pages", "confluence_search", artifacts_index_definition),
            ("events", "event_text_search", events_index_definition),
        ]

        for coll_name, index_name, definition in index_specs:
            coll = self._db[coll_name]
            try:
                existing = list(coll.list_search_indexes())
                found = next((i for i in existing if i.get("name") == index_name), None)
                if found:
                    status = found.get("status", "")
                    if status in ("FAILED", "DOES_NOT_EXIST"):
                        logger.warning(
                            f"[memory] Search index {index_name} on {coll_name} "
                            f"is {status}. Dropping and recreating."
                        )
                        coll.drop_search_index(index_name)
                    else:
                        logger.debug(
                            f"[memory] Search index {index_name} on {coll_name} "
                            f"already exists (status={status})."
                        )
                        continue

                model = SearchIndexModel(
                    definition=definition,
                    name=index_name,
                    type="search",
                )
                coll.create_search_index(model=model)
                logger.info(
                    f"[memory] Created search index {index_name} on {coll_name}."
                )
            except Exception as e:
                logger.error(
                    f"[memory] Failed to create search index {index_name} "
                    f"on {coll_name}: {e}"
                )

        try:
            existing_idx = self._events.index_information()
            if not any("text" in str(v.get("key")) for v in existing_idx.values()):
                self._events.create_index(
                    [("facts.root_cause", "text"), ("summary", "text")],
                    name="event_text_search_legacy",
                )
                logger.info("[memory] Created $text index on events collection.")
        except Exception as e:
            logger.warning(f"[memory] Could not create $text index on events: {e}")

    def store_artifact(
        self,
        id: str,
        type: str,
        title: str,
        content: str,
        day: int,
        date: str,
        timestamp: str,
        metadata: Optional[Dict] = None,
        aliases: Optional[List[str]] = None,
        why_it_matters: Optional[str] = None,
    ) -> None:
        doc = {
            "_id": id,
            "type": type,
            "title": title,
            "content": content,
            "day": day,
            "date": date,
            "timestamp": timestamp,
            "aliases": aliases or [],
            "why_it_matters": why_it_matters or "",
            "metadata": metadata or {},
        }
        self._artifacts.update_one({"_id": id}, {"$set": doc}, upsert=True)


    def store_confluence_page(
        self,
        id: str,
        title: str,
        content: str,
        day: int,
        date: str,
        timestamp: str,
        metadata: Optional[Dict] = None,
        aliases: Optional[List[str]] = None,
        why_it_matters: Optional[str] = None,
    ) -> None:
        """Store a Confluence page in its dedicated collection."""
        doc = {
            "_id": id,
            "title": title,
            "content": content,
            "day": day,
            "date": date,
            "timestamp": timestamp,
            "aliases": aliases or [],
            "why_it_matters": why_it_matters or "",
            "metadata": metadata or {},
        }
        self._confluence_pages.update_one({"_id": id}, {"$set": doc}, upsert=True)

    def store_zoom_transcript(
        self,
        id: str,
        title: str,
        content: str,
        day: int,
        date: str,
        timestamp: str,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Store a Zoom transcript in its dedicated collection."""
        doc = {
            "_id": id,
            "title": title,
            "content": content,
            "day": day,
            "date": date,
            "timestamp": timestamp,
            "metadata": metadata or {},
        }
        self._zoom_transcripts.update_one({"_id": id}, {"$set": doc}, upsert=True)

    def embed_artifact(
            self,
            id: str,
            type: str,
            title: str,
            content: str,
            day: int,
            date: str,
            timestamp: str,
            metadata: Optional[Dict] = None,
            aliases: Optional[List[str]] = None,
            why_it_matters: Optional[str] = None,
        ) -> None:
            """
            Routing function. Sends documents to their dedicated collections.
            Replaces the old pattern of dumping everything into 'artifacts'.
            
            Types that already live in their own collection (jira, pr, email,
            slack_thread, zd_ticket, sf_opportunity) are no-ops here since
            they are persisted by their own upsert methods.
            """
            if type == "confluence":
                self.store_confluence_page(
                    id=id,
                    title=title,
                    content=content,
                    day=day,
                    date=date,
                    timestamp=timestamp,
                    metadata=metadata,
                    aliases=aliases,
                    why_it_matters=why_it_matters,
                )
            elif type == "persona_skill":
                name = (metadata or {}).get("name", id)
                dept = (metadata or {}).get("dept", "")
                self.store_persona_skills(
                    name=name,
                    data={"expertise": aliases or [], "social_role": "", "style": ""},
                    dept=dept,
                    day=day,
                    timestamp_iso=timestamp,
                )
            elif type == "zoom_transcript":
                self.store_zoom_transcript(
                    id=id,
                    title=title,
                    content=content,
                    day=day,
                    date=date,
                    timestamp=timestamp,
                    metadata=metadata,
                )
            elif type in ("jira", "jira_comment", "pr", "email", "slack_thread",
                        "slack", "zd_ticket", "sf_opportunity"):
                pass
            else:
                logger.warning(
                    f"[memory] embed_artifact called with unrouted type='{type}' "
                    f"(id={id}). Skipping storage."
                )

    def store_persona_skills(
        self,
        name: str,
        data: Dict[str, Any],
        dept: str,
        day: int,
        timestamp_iso: str,
    ) -> None:
        raw_expertise = data.get("expertise", [])
        if isinstance(raw_expertise, str):
            raw_expertise = [raw_expertise]
        expertise_tags = [e.lower() for e in raw_expertise]
        alias_terms = list(dict.fromkeys(expertise_tags + [dept.lower()]))

        expertise_val = data.get("expertise", [])
        if isinstance(expertise_val, str):
            expertise_val = [expertise_val]
        skill_text = (
            f"Employee: {name}. Dept: {dept}. "
            f"Expertise: {', '.join(expertise_val)}. "
            f"Role: {data.get('social_role', '')}. "
            f"Style: {data.get('style', '')}"
        )

        self.store_artifact(
            id=f"skill_{name.lower().replace(' ', '_')}",
            type="persona_skill",
            title=f"Expertise Profile: {name}",
            content=skill_text,
            day=day,
            date=timestamp_iso,
            timestamp=timestamp_iso,
            metadata={"name": name, "dept": dept},
            aliases=alias_terms,
            why_it_matters=f"{name} owns {dept} domain knowledge: {', '.join(expertise_tags[:4])}",
        )

    def embed_persona_skills(
        self,
        name: str,
        data: Dict[str, Any],
        dept: str,
        day: int,
        timestamp_iso: str,
    ) -> None:
        """
        Backward-compatible alias for store_persona_skills().
        Called from flow.py genesis phase and org_lifecycle.py hire path.
        """
        self.store_persona_skills(
            name=name,
            data=data,
            dept=dept,
            day=day,
            timestamp_iso=timestamp_iso,
        )

    def log_event(self, event: SimEvent) -> None:
        """
        Insert SimEvent into in-memory log and MongoDB.
        No embedding. The event_text_search Atlas Search index and the
        $text index on events.facts.root_cause handle text retrieval.
        """
        self._event_log.append(event)

        if len(self._event_log) % 100 == 0:
            cutoff_day = (self._current_day or 0) - _EVENT_LOG_MAX_DAYS
            if cutoff_day > 0:
                self._event_log = [e for e in self._event_log if e.day >= cutoff_day]

        event_id = f"EVT-{event.day}-{event.type}-{len(self._event_log)}"

        doc = event.to_dict()
        doc["_id"] = event_id

        self._events.update_one({"_id": event_id}, {"$set": doc}, upsert=True)

    def search_artifacts_text(
        self,
        query: str,
        n: int = 5,
        type_filter: Optional[str] = None,
        type_exclude: Optional[List[str]] = None,
        day_range: Optional[Tuple[int, int]] = None,
        as_of_time: Optional[Any] = None,
        min_score: float = 1.0,
    ) -> List[Dict]:
        """
        Alias-weighted BM25 text search over the artifacts collection using
        the beliefs_search Atlas Search index.

        Scoring mirrors beliefsReader.searchText() [17]:
          - canonical_name exact/fuzzy: boost 14
          - canonical_name phrase:      boost 14
          - aliases fuzzy:              boost  5
          - aliases shingle:            boost 14

        This score separation ensures a query for "auth token expiry" ranks
        Jordan's persona skill record (aliases: ["auth", "identity", "sso"])
        well above unrelated artifacts, replacing cosine similarity entirely.

        Args:
            query:       The search string. Noise-stripped by callers where
                         appropriate (context_for_prompt strips via the same
                         pattern as queryExpander.ts [19]).
            n:           Maximum results to return.
            type_filter: Restrict to a single artifact type (e.g. "confluence").
            type_exclude: Exclude one or more artifact types.
            day_range:   (min_day, max_day) inclusive filter.
            as_of_time:  Causal ceiling -- only artifacts with timestamp <=
                         this value are eligible.
            min_score:   Minimum Atlas Search score threshold (default 1.0).

        Returns:
            List of dicts with keys: id, title, type, day, timestamp,
            metadata, canonical_name, aliases, score.
        """
        if not query or not query.strip():
            return []

        if type_filter and type_exclude:
            raise ValueError(
                "search_artifacts_text(): type_filter and type_exclude "
                "are mutually exclusive."
            )

        fuzzy_opts = {"maxEdits": 1, "prefixLength": 2}

        filter_clauses: List[Dict] = []

        if type_filter:
            filter_clauses.append({"equals": {"path": "type", "value": type_filter}})
        elif type_exclude:
            for t in type_exclude:
                filter_clauses.append({"equals": {"path": "type", "value": t}})

        iso_ceiling = self._to_iso(as_of_time)

        if type_filter == "persona_skill":
            _index_name = "persona_skill_search"
        else:
            _index_name = "confluence_search"

        search_stage: Dict[str, Any] = {
            "index": _index_name,
            "compound": {
                "should": [
                    {
                        "text": {
                            "query": query,
                            "path": "title",
                            "score": {"boost": {"value": 3}},
                        }
                    },
                    {
                        "text": {
                            "query": query,
                            "path": "aliases",
                            "score": {"boost": {"value": 2}},
                        }
                    },
                    {
                        "text": {
                            "query": query,
                            "path": "why_it_matters",
                            "score": {"boost": {"value": 1}},
                        }
                    },
                ],
                "minimumShouldMatch": 1,
            },
        }

        if type_exclude:
            search_stage["compound"]["mustNot"] = [
                {"equals": {"path": "type", "value": t}} for t in type_exclude
            ]
            filter_clauses = [
                fc
                for fc in filter_clauses
                if not ("equals" in fc and fc["equals"].get("path") == "type")
            ]

        if filter_clauses:
            search_stage["compound"]["filter"] = filter_clauses

        pipeline: List[Dict] = [
            {"$search": search_stage},
            {"$addFields": {"score": {"$meta": "searchScore"}}},
            {"$match": {"score": {"$gte": min_score}}},
        ]

        match_stage: Dict[str, Any] = {}
        if day_range:
            match_stage["day"] = {
                "$gte": day_range[0],
                "$lte": day_range[1],
            }
        if iso_ceiling:
            match_stage["timestamp"] = {"$lte": iso_ceiling}
        if match_stage:
            pipeline.append({"$match": match_stage})

        pipeline.append({"$limit": n})
        pipeline.append(
            {
                "$project": {
                    "id": "$_id",
                    "title": 1,
                    "type": 1,
                    "day": 1,
                    "timestamp": 1,
                    "metadata": 1,
                    "aliases": 1,
                    "why_it_matters": 1,
                    "score": 1,
                }
            }
        )

        if type_filter == "persona_skill":
            target_coll = self._persona_skills
        elif type_filter == "confluence":
            target_coll = self._confluence_pages
        else:
            # Default: query confluence_pages (the most common search target
            # after persona_skill is excluded via type_exclude)
            target_coll = self._confluence_pages

        try:
            return list(target_coll.aggregate(pipeline))
        except Exception as e:
            logger.error(f"[memory] search_artifacts_text failed: {e}")
            return []

    def search_persona_skills_text(
        self,
        query: str,
        n: int = 5,
    ) -> List[Dict]:
        """
        Find the best-matched engineers for a topic by querying domain_registry
        directly.
        """
        if not query or not query.strip():
            return []

        query_tokens = {
            t.lower()
            for t in re.split(r"[\s_\-/]+", query.lower())
            if len(t) >= 3
        }
        if not query_tokens:
            return []

        matched_domains = list(
            self._db["domain_registry"].find(
                {"system_tags": {"$in": list(query_tokens)}},
                {
                    "domain": 1,
                    "primary_owner": 1,
                    "known_by": 1,
                    "dept": 1,
                    "system_tags": 1,
                    "documentation_coverage": 1,
                    "_id": 0,
                },
            )
        )

        if not matched_domains:
            return []

        candidate_scores: Dict[str, float] = {}
        candidate_dept: Dict[str, str] = {}

        for rec in matched_domains:
            tags = set(rec.get("system_tags", []))
            overlap = len(query_tokens & tags)
            if overlap == 0:
                continue

            dept = rec.get("dept", "")
            coverage = rec.get("documentation_coverage", 0.0)
            base_score = overlap * (1.0 + coverage)

            primary = rec.get("primary_owner")
            if primary:
                candidate_scores[primary] = (
                    candidate_scores.get(primary, 0.0) + base_score * 1.5
                )
                candidate_dept.setdefault(primary, dept)

            for name in rec.get("known_by", []):
                if name == primary:
                    continue
                candidate_scores[name] = (
                    candidate_scores.get(name, 0.0) + base_score
                )
                candidate_dept.setdefault(name, dept)

        if not candidate_scores:
            return []

        ranked = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)

        return [
            {
                "name": name,
                "dept": candidate_dept.get(name, ""),
                "score": round(score, 4),
            }
            for name, score in ranked[:n]
        ]

    def record_author_expertise_signals(
        self,
        author: str,
        artifact_id: str,
        artifact_type: str,
        day: int,
        timestamp_iso: str,
        topics_in_doc: List[str],
        topics_outside_my_expertise: Optional[List[str]] = [],
        claims_approximated: Optional[List[str]] = [],
        sections_left_thin: Optional[List[str]] = [],
    ) -> None:
        """
        Persist LLM-produced skill signals into the author_expertise collection.

        Args:
            author:                      The engineer who authored the artifact.
            artifact_id:                 ID of the source artifact (e.g. "CONF-ENG-017").
            artifact_type:               Type of the source artifact (e.g. "confluence").
            day:                         Current simulation day.
            timestamp_iso:               ISO 8601 timestamp for the update.
            topics_in_doc:               All topics the LLM identified in the artifact.
            topics_outside_my_expertise: Topics the LLM flagged as beyond the
                                        author's known expertise.
            claims_approximated:         Statements the LLM produced with low
                                        confidence.
            sections_left_thin:          Sections the LLM identified as
                                        under-documented.
        """
        if not topics_in_doc:
            return

        doc: Dict[str, Any] = {
            "author": author,
            "artifact_id": artifact_id,
            "artifact_type": artifact_type,
            "day": day,
            "timestamp": timestamp_iso,
            "topics_in_doc": topics_in_doc,
            "topics_outside_my_expertise": topics_outside_my_expertise,
            "claims_approximated": claims_approximated,
            "sections_left_thin": sections_left_thin,
        }

        try:
            self._db["author_expertise"].insert_one(doc)
        except Exception as e:
            logger.warning(
                f"[memory] record_author_expertise_signals failed for "
                f"author={author}, artifact_id={artifact_id}: {e}"
            )
            return

        logger.debug(
            f"[memory] record_author_expertise_signals: author={author}, "
            f"artifact_type={artifact_type}, artifact_id={artifact_id}"
        )

    def find_expert_by_skill(self, topic: str, n: int = 1) -> List[Dict]:
        """
        Backward-compatible alias for search_persona_skills_text().
        Called from flow.py _select_domain_expert() and org_lifecycle.py
        scan_for_knowledge_gaps(). Returns [{name, dept, score}].
        """
        return self.search_persona_skills_text(query=topic, n=n)
    

    def domain_context_for_topic(
        self,
        topic: str,
        as_hint: bool = False,
    ) -> Tuple[List[Dict], str]:
        """
        Query domain_registry for all domains whose system_tags overlap the
        topic string. Returns both the raw docs and a pre-formatted string
        so callers never duplicate tokenisation or formatting logic.
        """
        tokens = [
            t.lower()
            for t in re.split(r"[\s_\-/]+", topic.lower())
            if len(t) >= 3
        ]
        if not tokens:
            return [], ""

        docs = list(
            self._db["domain_registry"].find(
                {"system_tags": {"$in": tokens}},
                {
                    "domain": 1,
                    "primary_owner": 1,
                    "known_by": 1,
                    "documentation_coverage": 1,
                    "_id": 0,
                },
            )
        )
        if not docs:
            return [], ""

        if as_hint:
            lines = [
                "Note: the following domain knowledge exists and may be referenced naturally:"
            ]
            for rec in docs:
                pct = int(rec.get("documentation_coverage", 0) * 100)
                owner = rec.get("primary_owner")
                known_by = rec.get("known_by", [])
                if owner:
                    lines.append(
                        f"  - '{rec['domain']}': owned by {owner} ({pct}% documented)"
                    )
                elif known_by:
                    lines.append(
                        f"  - '{rec['domain']}': no primary owner, "
                        f"partial knowledge held by {', '.join(known_by)} ({pct}% documented)"
                    )
                else:
                    lines.append(
                        f"  - '{rec['domain']}': orphaned domain ({pct}% documented, no known experts)"
                    )
        else:
            lines = ["DOMAIN CONTEXT (from domain_registry):"]
            for rec in docs:
                pct = int(rec.get("documentation_coverage", 0) * 100)
                owner = rec.get("primary_owner", "none")
                lines.append(
                    f"  - {rec['domain']}: owner={owner}, coverage={pct}%"
                )

        return docs, "\n".join(lines)
    
    def get_author_domain_tokens(self, author: str) -> set[str]:
        """
        Returns the set of lowercase tokens representing an author's live
        domain knowledge. Primary source is domain_registry (earned through
        PRs, incidents, Confluence authorship). Falls back to PERSONAS config
        on Day 0 before any registry signal exists.

        Mirrors TicketAssigner._skill_score's vocabulary strategy so both
        assignment scoring and self-audit expertise comparison use identical
        token sets.
        """
        domain_docs = list(
            self._db["domain_registry"].find(
                {"$or": [{"known_by": author}, {"primary_owner": author}]},
                {"system_tags": 1, "_id": 0},
            )
        )

        if domain_docs:
            all_tags = [tag for doc in domain_docs for tag in doc.get("system_tags", [])]
            return {t.lower() for t in all_tags}

        from config_loader import PERSONAS, DEFAULT_PERSONA
        persona = PERSONAS.get(author, DEFAULT_PERSONA)
        return {e.lower() for e in persona.get("expertise", [])}
    
    def get_author_domain_tokens(self, author: str) -> set[str]:
        """
        Returns the set of lowercase tokens representing an author's live
        domain knowledge. Primary source is domain_registry (earned through
        PRs, incidents, Confluence authorship). Falls back to PERSONAS config
        on Day 0 before any registry signal exists.

        Mirrors TicketAssigner._skill_score's vocabulary strategy so both
        assignment scoring and self-audit expertise comparison use identical
        token sets.
        """
        domain_docs = list(
            self._db["domain_registry"].find(
                {"$or": [{"known_by": author}, {"primary_owner": author}]},
                {"system_tags": 1, "_id": 0},
            )
        )

        if domain_docs:
            all_tags = [tag for doc in domain_docs for tag in doc.get("system_tags", [])]
            return {t.lower() for t in all_tags}

        from config_loader import PERSONAS, DEFAULT_PERSONA
        persona = PERSONAS.get(author, DEFAULT_PERSONA)
        return {e.lower() for e in persona.get("expertise", [])}

    def context_for_prompt(
        self,
        query: str,
        n: int = 4,
        as_of_time: Optional[Any] = None,
        since: Optional[Any] = None,
    ) -> str:
        """
        Three-tier context block for LLM prompts, replacing vector RAG.

        Follows the contextBuilder.ts tiered pattern [18]:

        Tier 1 -- Pinned / always-on (injected unconditionally):
            Tech stack doc + any active knowledge_gap_detected events whose
            domain tags overlap the query. These are structural facts the LLM
            must always have regardless of query similarity.

        Tier 2 -- Text search (alias-weighted BM25):
            search_artifacts_text() over artifacts, noise-stripped query,
            excludes persona_skill type. The score separation from canonical
            and alias boosting means the top results are genuinely relevant
            rather than the semantically proximate blob that vector search
            returns.

        Tier 3 -- Open questions (scoped by type, not search score):
            Unresolved knowledge_gap_detected events fetched by type from
            MongoDB. These are injected so the LLM is aware of documented
            uncertainty in the current scope, matching the open_questions tier
            in contextBuilder.ts [18].

        as_of_time and since enforce causal ceilings/floors on Tier 2 results.
        """
        stripped_query = self._strip_query_noise(query)
        iso_ceiling = self._to_iso(as_of_time)
        iso_floor = self._to_iso(since)
        lines: List[str] = []

        tech = self.tech_stack_for_prompt()
        if tech:
            lines.append(tech)

        gap_filter: Dict[str, Any] = {"type": "knowledge_gap_detected"}
        if iso_ceiling:
            gap_filter["timestamp"] = {"$lte": iso_ceiling}

        active_gaps = list(
            self._events.find(gap_filter, {"_id": 0}).sort("timestamp", -1).limit(3)
        )
        if active_gaps:
            query_lower = stripped_query.lower()
            relevant_gaps = [
                g
                for g in active_gaps
                if any(
                    domain.lower() in query_lower
                    for domain in g.get("facts", {}).get("gap_areas", [])
                    + g.get("facts", {}).get("topics_beyond_author_expertise", [])
                )
            ]
            if relevant_gaps:
                lines.append("=== ACTIVE KNOWLEDGE GAPS (awareness only) ===")
                for g in relevant_gaps:
                    facts = g.get("facts", {})
                    domains = facts.get("gap_areas") or facts.get(
                        "topics_beyond_author_expertise", []
                    )
                    lines.append(
                        f"  Day {g.get('day', '?')} — gap in: "
                        f"{', '.join(domains)} "
                        f"(detection: {facts.get('detection_method', '?')})"
                    )

        if stripped_query:
            day_range: Optional[Tuple[int, int]] = None
            if iso_floor:
                pass

            artifacts = self.search_artifacts_text(
                query=stripped_query,
                n=n,
                type_exclude=["persona_skill"],
                as_of_time=as_of_time,
                min_score=1.0,
            )

            if artifacts:
                lines.append("=== RELEVANT ARTIFACTS ===")
                for a in artifacts:
                    day_label = f"Day {a.get('day')}" if "day" in a else "?"
                    lines.append(
                        f"  [{a.get('type', '').upper()}] {a.get('id')} -- "
                        f"{a.get('title')} ({day_label}, "
                        f"score {a.get('score', 0):.2f})"
                    )

        open_q_filter: Dict[str, Any] = {
            "type": "knowledge_gap_detected",
            "facts.gap_classification": {"$in": ["likely", "possible"]},
        }
        if iso_ceiling:
            open_q_filter["timestamp"] = {"$lte": iso_ceiling}

        open_questions = list(
            self._events.find(open_q_filter, {"_id": 0}).sort("timestamp", -1).limit(3)
        )
        if open_questions:
            lines.append("=== OPEN QUESTIONS / UNRESOLVED GAPS ===")
            for q in open_questions:
                facts = q.get("facts", {})
                topic = (
                    facts.get("topic")
                    or facts.get("gap_domain")
                    or ", ".join(facts.get("gap_areas", []))
                    or "unknown"
                )
                classification = facts.get("gap_classification", "?")
                lines.append(f"  Day {q.get('day', '?')} — [{classification}] {topic}")

        if not lines:
            return "No relevant prior context found."

        return "\n".join(lines)

    def recall_with_rewrite(
        self,
        raw_query: str,
        n: int = 4,
        as_of_time: Optional[Any] = None,
        since: Optional[Any] = None,
    ) -> str:
        """
        Backward-compatible alias for context_for_prompt().

        Previously performed HyDE query rewriting before embedding. That LLM
        call and the embedding are now gone. The noise-stripping inside
        context_for_prompt() handles the same cleanup that HyDE was compensating
        for -- stripping code blocks, stack traces, and markdown before BM25
        search produces cleaner results than HyDE + cosine did.

        Called from confluence_writer.py write_design_doc() and
        write_adhoc_page(). No callers need to change.
        """
        return self.context_for_prompt(
            query=raw_query,
            n=n,
            as_of_time=as_of_time,
            since=since,
        )

    @staticmethod
    def _strip_query_noise(text: str) -> str:
        """
        Remove noise from a query string before BM25 search.

        Mirrors the STRIP_PATTERNS pipeline from queryExpander.ts [19]:
          - Fenced code blocks
          - Inline code
          - Stack trace lines (starting with "at ")
          - URLs
          - File paths
          - Markdown headings and bold/italic markers
          - HTML tags
          - Long hex strings (commit hashes, UUIDs)
          - Pure numbers
          - Trailing punctuation

        Returns the cleaned string, or the original if cleaning produces
        something too short to be useful (< 4 chars).
        """
        import re

        patterns = [
            r"```[\s\S]*?```",
            r"~~~[\s\S]*?~~~",
            r"^\s+at\s+.+$",
            r"https?://[^\s]+",
            r"`[^`]+`",
            r"^#{1,6}\s+",
            r"\*{1,2}([^*]+)\*{1,2}",
            r"_{1,2}([^_]+)_{1,2}",
            r"<[^>]+>",
            r"\b[0-9a-f]{8,}\b",
            r"\b\d+\b",
            r"[?!.,;:]+(?=\s|$)",
        ]

        out = text
        for pattern in patterns:
            out = re.sub(pattern, " ", out, flags=re.MULTILINE | re.IGNORECASE)

        out = re.sub(r"\s{2,}", " ", out).strip()

        if len(out) < 4:
            return text.strip()

        return out

    @staticmethod
    def _to_iso(as_of_time: Optional[Any]) -> Optional[str]:
        """
        Normalise as_of_time to an ISO 8601 string for MongoDB comparisons.
        Accepts datetime objects or pre-formatted ISO strings.
        Returns None unchanged so optional semantics are preserved.
        """
        if as_of_time is None:
            return None
        if isinstance(as_of_time, str):
            return as_of_time
        return as_of_time.isoformat()

    _NOISY_PERSONAL_EVENTS = {
        "sprint_planned",
        "retrospective",
        "sprint_goal_updated",
        "day_summary",
        "leadership_sync",
        "normal_day_slack",
        "watercooler_chat",
        "standup",
        "onboarding_session",
    }

    def persona_history(self, name: str, n: int = 4) -> List[SimEvent]:
        relevant = [
            e
            for e in self._event_log
            if name in e.actors and e.type not in self._NOISY_PERSONAL_EVENTS
        ]
        return relevant[-n:]

    def get_event_log(
        self, from_db: bool = False, as_of_time: Optional[str] = None
    ) -> List[SimEvent]:
        if from_db:
            query: Dict[str, Any] = {}
            if as_of_time:
                query["timestamp"] = {"$lte": as_of_time}
            raw = self._events.find(query).sort("timestamp", 1)
            return [SimEvent.from_dict(r) for r in raw]

        log = self._event_log
        if as_of_time:
            log = [e for e in log if e.timestamp <= as_of_time]
        return log

    def events_by_type(self, event_type: str) -> List[SimEvent]:
        return [e for e in self._event_log if e.type == event_type]

    def previous_day_context(self, current_day: int) -> str:
        if current_day == 1:
            return (
                "This is the first observed day. The company has existing systems, "
                "legacy debt, and established teams. Key pressures already in play:\n"
                + self._known_pressures_summary()
            )

        prev_day = current_day - 1

        summary_doc = self._events.find_one(
            {"type": "day_summary", "day": prev_day},
            {"facts": 1, "date": 1, "_id": 0},
            sort=[("timestamp", -1)],
        )
        header = ""
        if summary_doc:
            f = summary_doc.get("facts", {})
            header = (
                f"Yesterday (Day {prev_day}): "
                f"system health {f.get('system_health', '?')}, "
                f"morale {f.get('morale', '?'):.2f}, "
                f"{f.get('incidents_opened', 0)} incident(s) opened, "
                f"{f.get('incidents_resolved', 0)} resolved. "
                f"Health trend: {f.get('health_trend', 'unknown')}."
            )

        _ALLOW = {
            "incident_opened",
            "incident_resolved",
            "postmortem_created",
            "sprint_planned",
            "customer_escalation",
            "customer_email_routed",
            "knowledge_gap_detected",
            "morale_intervention",
            "employee_departed",
            "employee_hired",
            "zd_ticket_opened",
            "zd_tickets_escalated",
            "zd_tickets_resolved",
            "sf_deals_risk_flagged",
            "sf_ownership_lapsed",
        }

        docs = list(
            self._events.find(
                {"day": prev_day, "type": {"$in": list(_ALLOW)}},
                {"type": 1, "actors": 1, "facts": 1, "artifact_ids": 1, "_id": 0},
            ).sort("timestamp", 1)
        )

        lines = []
        for doc in docs:
            event_type = doc.get("type", "")
            facts = doc.get("facts", {})
            artifact_ids = doc.get("artifact_ids", {})
            title = (
                facts.get("root_cause")
                or facts.get("title")
                or facts.get("sprint_theme")
                or facts.get("subject")
                or artifact_ids.get("jira")
                or ""
            )
            label = event_type.replace("_", " ").title()
            line = f"- {label}"
            if title:
                line += f": {title[:120]}"
            lines.append(line)

        if not header and not lines:
            return f"No significant events on Day {prev_day}."

        parts = []
        if header:
            parts.append(header)
        if lines:
            parts.append("\n".join(lines))
        return "\n".join(parts)

    def context_for_sprint_planning(
        self,
        sprint_num: int,
        dept: str,
        sprint_theme: str = "",
        as_of_time: Optional[Any] = None,
    ) -> str:
        iso = self._to_iso(as_of_time)
        lines: List[str] = []

        header = f"=== SPRINT #{sprint_num} PLANNING CONTEXT"
        if sprint_theme:
            header += f" -- {sprint_theme}"
        if dept:
            header += f" ({dept})"
        header += " ==="
        lines.append(header)

        ticket_filter: Dict[str, Any] = {"status": {"$ne": "Done"}}
        _SPRINT_TICKET_PROJECTION = {
            "_id": 0,
            "id": 1,
            "title": 1,
            "status": 1,
            "assignee": 1,
            "dept": 1,
            "story_points": 1,
            "priority": 1,
        }
        if dept:
            ticket_filter["dept"] = dept
        open_tickets = list(
            self._jira.find(ticket_filter, _SPRINT_TICKET_PROJECTION)
            .sort("priority", 1)
            .limit(10)
        )
        if open_tickets:
            lines.append(f"  Open tickets ({dept or 'all depts'}):")
            for t in open_tickets:
                assignee = t.get("assignee", "unassigned")
                priority = t.get("priority", "medium")
                lines.append(
                    f"    [{t['id']}] {t.get('title', '')} "
                    f"(status={t.get('status', '?')}, priority={priority}, assignee={assignee})"
                )
        else:
            lines.append(f"  No open tickets for {dept or 'any dept'}.")

        incident_filter: Dict[str, Any] = {"type": "incident_detected"}
        if iso:
            incident_filter["timestamp"] = {"$lte": iso}
        recent_incidents = list(
            self._events.find(incident_filter, {"_id": 0})
            .sort("timestamp", -1)
            .limit(5)
        )
        if recent_incidents:
            lines.append("  Recent incidents:")
            for inc in recent_incidents:
                facts = inc.get("facts", {})
                lines.append(
                    f"    Day {inc.get('day', '?')} -- "
                    f"{facts.get('title', facts.get('root_cause', 'Unknown'))}"
                )

        checkpoint = self._checkpoints.find_one(sort=[("day", -1)])
        if checkpoint:
            state = checkpoint.get("state", {})
            velocity = state.get("velocity") or checkpoint.get("velocity")
            sys_health = state.get("system_health")
            if velocity is not None:
                lines.append(f"  Last recorded velocity: {velocity} points")
            if sys_health is not None:
                lines.append(f"  System health at last checkpoint: {sys_health}/100")

        return (
            "\n".join(lines)
            if len(lines) > 1
            else f"No sprint planning context found for {dept}."
        )

    def get_domain_registry(self) -> dict:
        return {rec["domain"]: rec for rec in self._db["domain_registry"].find({})}

    def get_orphaned_domains(self) -> list:
        return list(self._db["domain_registry"].find({"primary_owner": None}))

    def update_domain_coverage(self, domain: str, delta: float, author: str, day: int):
        key = domain.lower().replace(" ", "_")
        rec = self._db["domain_registry"].find_one({"_id": key})
        if rec:
            new_coverage = min(1.0, rec["documentation_coverage"] + delta)
            self._db["domain_registry"].update_one(
                {"_id": key},
                {
                    "$set": {
                        "documentation_coverage": new_coverage,
                        "last_updated_day": day,
                    },
                    "$addToSet": {"known_by": author},
                },
            )

    def context_for_retrospective(
        self,
        sprint_num: int,
        since_iso: str,
        as_of_iso: str,
    ) -> str:
        lines: List[str] = [f"=== SPRINT #{sprint_num} RETROSPECTIVE CONTEXT ==="]

        done_tickets = list(
            self._jira.find(
                {"status": "Done"},
                {
                    "_id": 0,
                    "id": 1,
                    "title": 1,
                    "status": 1,
                    "assignee": 1,
                    "story_points": 1,
                },
            ).limit(15)
        )
        carried_tickets = list(
            self._jira.find(
                {"status": {"$ne": "Done"}},
                {
                    "_id": 0,
                    "id": 1,
                    "title": 1,
                    "status": 1,
                    "assignee": 1,
                    "story_points": 1,
                },
            ).limit(10)
        )

        if done_tickets:
            lines.append("  Completed this sprint:")
            for t in done_tickets:
                lines.append(
                    f"    [{t['id']}] {t.get('title', '')} (assignee={t.get('assignee', '?')})"
                )

        if carried_tickets:
            lines.append("  Carried over / incomplete:")
            for t in carried_tickets:
                lines.append(
                    f"    [{t['id']}] {t.get('title', '')} "
                    f"(status={t.get('status', '?')}, assignee={t.get('assignee', '?')})"
                )

        _RETRO_TYPES = {
            "incident_detected",
            "incident_resolved",
            "postmortem_published",
            "deploy",
            "sprint_started",
            "retrospective",
        }
        event_filter: Dict[str, Any] = {
            "type": {"$in": list(_RETRO_TYPES)},
            "timestamp": {"$gte": since_iso, "$lte": as_of_iso},
        }
        sprint_events = list(
            self._events.find(event_filter, {"_id": 0}).sort("timestamp", 1).limit(20)
        )
        if sprint_events:
            lines.append("  Notable events this sprint:")
            for e in sprint_events:
                facts = e.get("facts", {})
                label = e.get("type", "").replace("_", " ").title()
                detail = (
                    facts.get("title")
                    or facts.get("root_cause")
                    or facts.get("summary")
                    or ""
                )
                actors = e.get("actors", [])
                actor_str = f" ({actors[0]})" if actors else ""
                lines.append(
                    f"    Day {e.get('day', '?')} -- {label}{actor_str}"
                    + (f": {detail}" if detail else "")
                )

        checkpoint = self._checkpoints.find_one(sort=[("day", -1)])
        if checkpoint:
            state = checkpoint.get("state", {})
            velocity = state.get("velocity") or checkpoint.get("velocity")
            sys_health = state.get("system_health")
            morale = state.get("team_morale")
            if velocity is not None:
                lines.append(f"  Sprint velocity: {velocity} points")
            if sys_health is not None:
                lines.append(f"  System health: {sys_health}/100")
            if morale is not None:
                lines.append(f"  Team morale: {morale:.2f}")

        return (
            "\n".join(lines)
            if len(lines) > 1
            else f"No retrospective context found for Sprint #{sprint_num}."
        )

    def context_for_incident(
        self,
        ticket_id: str,
        as_of_time: Optional[Any] = None,
    ) -> str:
        iso = self._to_iso(as_of_time)
        lines: List[str] = [f"=== INCIDENT CONTEXT: {ticket_id} ==="]

        ticket = self._jira.find_one(
            {"id": ticket_id},
            {
                "_id": 0,
                "id": 1,
                "title": 1,
                "description": 1,
                "status": 1,
                "assignee": 1,
                "root_cause": 1,
                "recurrence_of": 1,
                "recurrence_gap_days": 1,
                "gap_areas": 1,
                "escalation_narrative": 1,
                "linked_prs": 1,
            },
        )
        if ticket:
            lines.append(
                f"  [{ticket_id}] {ticket.get('title', '')} "
                f"(status={ticket.get('status', '?')}, "
                f"priority={ticket.get('priority', '?')}, "
                f"assignee={ticket.get('assignee', 'unassigned')})"
            )
            root_cause = ticket.get("root_cause") or ticket.get("facts", {}).get(
                "root_cause"
            )
            if root_cause:
                lines.append(f"  Root cause: {root_cause}")
        else:
            lines.append(f"  Ticket {ticket_id} not found in jira_tickets.")

        pm_filter: Dict[str, Any] = {
            "type": "postmortem_published",
            "artifact_ids.jira": ticket_id,
        }
        if iso:
            pm_filter["timestamp"] = {"$lte": iso}
        postmortems = list(
            self._events.find(pm_filter, {"_id": 0}).sort("timestamp", -1).limit(3)
        )
        if postmortems:
            lines.append("  Postmortems:")
            for pm in postmortems:
                facts = pm.get("facts", {})
                lines.append(
                    f"    Day {pm.get('day', '?')} -- "
                    f"{facts.get('title', facts.get('summary', 'Postmortem'))}"
                )

        prior_filter: Dict[str, Any] = {
            "type": "incident_detected",
            "_id": {"$ne": ticket_id},
        }
        if iso:
            prior_filter["timestamp"] = {"$lte": iso}
        prior = self._events.find_one(
            prior_filter,
            {"_id": 0},
            sort=[("timestamp", -1)],
        )
        if prior:
            facts = prior.get("facts", {})
            lines.append(
                f"  Prior incident: Day {prior.get('day', '?')} -- "
                f"{facts.get('title', facts.get('root_cause', 'Unknown'))}"
            )

        return "\n".join(lines)

    def context_for_person(
        self,
        name: str,
        as_of_time: Optional[Any] = None,
        n: int = 3,
    ) -> str:
        iso = self._to_iso(as_of_time)
        lines: List[str] = [f"=== RECENT CONTEXT: {name} ==="]

        ticket_filter: Dict[str, Any] = {
            "assignee": name,
            "status": {"$ne": "Done"},
        }
        if iso:
            ticket_filter["created_at"] = {"$lte": iso}

        open_tickets = list(
            self._jira.find(ticket_filter, {"_id": 0}).sort("priority", 1).limit(5)
        )
        if open_tickets:
            lines.append("  Open tickets:")
            for t in open_tickets:
                lines.append(
                    f"    [{t['id']}] {t.get('title', '')} "
                    f"(status={t.get('status', '?')})"
                )
        else:
            lines.append("  No open tickets assigned.")

        event_filter: Dict[str, Any] = {"actors": name}
        if iso:
            event_filter["timestamp"] = {"$lte": iso}
        recent_events = list(
            self._events.find(event_filter, {"_id": 0}).sort("timestamp", -1).limit(n)
        )
        if recent_events:
            lines.append("  Recent activity:")
            for e in recent_events:
                facts = e.get("facts", {})
                label = e.get("type", "").replace("_", " ").title()
                detail = (
                    facts.get("title")
                    or facts.get("summary")
                    or e.get("summary", "")
                    or ""
                )
                lines.append(
                    f"    Day {e.get('day', '?')} -- {label}"
                    + (f": {detail}" if detail else "")
                )

        return (
            "\n".join(lines)
            if len(lines) > 1
            else f"No recent context found for {name}."
        )

    def context_for_pr_review(
        self,
        pr_id: str,
        ticket_id: Optional[str] = None,
        as_of_time: Optional[Any] = None,
        n: int = 2,
    ) -> str:
        lines = []
        iso = self._to_iso(as_of_time)

        pr = self._prs.find_one({"pr_id": pr_id}, {"_id": 0})
        if pr:
            pr_comments = pr.get("comments", [])
            if iso:
                pr_comments = [c for c in pr_comments if c.get("timestamp", "") <= iso]
            if pr_comments:
                lines.append("=== PR REVIEW HISTORY ===")
                for c in pr_comments[-3:]:
                    lines.append(
                        f"  [{c['date']}] {c['author']} [{c.get('verdict', '?')}]: {c['text'][:200]}"
                    )

        if ticket_id:
            ticket = self._jira.find_one({"id": ticket_id}, {"_id": 0, "comments": 1})
            if ticket:
                t_comments = ticket.get("comments", [])
                if iso:
                    t_comments = [c for c in t_comments if c.get("created", "") <= iso]
                author = pr.get("author") if pr else None
                author_updates = [
                    c for c in t_comments if not author or c.get("author") == author
                ]
                if author_updates:
                    lines.append("=== AUTHOR TICKET UPDATES ===")
                    for c in author_updates[-3:]:
                        lines.append(
                            f"  [{c['date']}] {c['author']}: {c['text'][:200]}"
                        )

        return "\n".join(lines) if lines else "No prior context found."

    def context_for_ticket(
        self,
        ticket_id: str,
        as_of_time: Optional[Any] = None,
    ) -> str:
        iso = self._to_iso(as_of_time)
        lines: List[str] = [f"=== TICKET CONTEXT: {ticket_id} ==="]

        ticket = self._jira.find_one(
            {"id": ticket_id},
            {
                "_id": 0,
                "id": 1,
                "title": 1,
                "status": 1,
                "assignee": 1,
                "description": 1,
                "story_points": 1,
                "linked_prs": 1,
                "root_cause": 1,
                "recurrence_of": 1,
                "recurrence_gap_days": 1,
                "gap_areas": 1,
                "comments": {"$slice": -3},
            },
        )
        if not ticket:
            lines.append(f"  Ticket {ticket_id} not found.")
            return "\n".join(lines)

        lines.append(
            f"  [{ticket_id}] {ticket.get('title', '')} "
            f"(status={ticket.get('status', '?')}, "
            f"assignee={ticket.get('assignee', 'unassigned')}, "
            f"points={ticket.get('story_points', '?')})"
        )

        description = ticket.get("description", "").strip()
        if description:
            lines.append(f"  Description: {description[:200]}")

        if ticket.get("root_cause"):
            lines.append(f"  Root cause: {ticket['root_cause'][:150]}")

        if ticket.get("gap_areas"):
            lines.append(f"  Knowledge gap areas: {', '.join(ticket['gap_areas'])}")

        linked_prs = ticket.get("linked_prs", [])
        if linked_prs:
            lines.append(f"  Linked PRs: {', '.join(linked_prs)}")

        comments = ticket.get("comments", [])
        if comments:
            lines.append("  Recent comments:")
            for c in comments:
                author = c.get("author", "?")
                date = c.get("date", "")
                text = c.get("text", "").strip().strip('"')[:150]
                lines.append(f"    {author} ({date}): {text}")

        recurrence_of = ticket.get("recurrence_of")
        if recurrence_of:
            gap = ticket.get("recurrence_gap_days", "?")
            ancestor = self._jira.find_one(
                {"id": recurrence_of},
                {"_id": 0, "root_cause": 1, "title": 1},
            )
            ancestor_root_cause = (
                ancestor.get("root_cause", "")[:120] if ancestor else ""
            )
            lines.append(
                f"  Recurrence: this is a repeat of {recurrence_of} "
                f"({gap} days ago)"
                + (
                    f" -- prior root cause: {ancestor_root_cause}"
                    if ancestor_root_cause
                    else ""
                )
            )

        blocker_filter: Dict[str, Any] = {
            "type": "blocker_flagged",
            "artifact_ids.jira": ticket_id,
        }
        if iso:
            blocker_filter["timestamp"] = {"$lte": iso}
        blockers = list(
            self._events.find(
                blocker_filter,
                {"_id": 0, "day": 1, "actors": 1, "facts.blocker_reason": 1},
            )
            .sort("timestamp", -1)
            .limit(2)
        )
        if blockers:
            lines.append("  Active blockers:")
            for b in blockers:
                reason = b.get("facts", {}).get("blocker_reason", "")[:120]
                actor = b.get("actors", ["?"])[0]
                lines.append(f"    Day {b.get('day', '?')} -- {actor}: {reason}")

        discussion_filter: Dict[str, Any] = {
            "type": {"$in": ["async_question", "design_discussion"]},
            "artifact_ids.jira": ticket_id,
        }
        if iso:
            discussion_filter["timestamp"] = {"$lte": iso}
        prior_discussions = list(
            self._events.find(
                discussion_filter,
                {
                    "_id": 0,
                    "day": 1,
                    "type": 1,
                    "facts.topic": 1,
                    "facts.asker": 1,
                    "summary": 1,
                },
            )
            .sort("timestamp", -1)
            .limit(3)
        )
        if prior_discussions:
            lines.append("  Prior discussions (do not rehash these):")
            for d in prior_discussions:
                label = d.get("type", "").replace("_", " ").title()
                topic = d.get("facts", {}).get("topic") or d.get("summary", "")
                lines.append(f"    Day {d.get('day', '?')} [{label}]: {topic[:120]}")

        return "\n".join(lines)

    def context_for_ticket_progress(
        self,
        ticket_id: str,
        assignee: str,
        as_of_time: Optional[Any] = None,
    ) -> str:
        iso = self._to_iso(as_of_time)
        lines: List[str] = [f"=== TICKET CONTEXT: {ticket_id} ==="]

        ticket = self._jira.find_one({"id": ticket_id}, _TICKET_PROGRESS_PROJECTION)
        if not ticket:
            lines.append(f"  Ticket {ticket_id} not found.")
            return "\n".join(lines)

        lines.append(
            f"  [{ticket_id}] {ticket.get('title', '')} "
            f"(status={ticket.get('status', '?')}, "
            f"assignee={ticket.get('assignee', 'unassigned')}, "
            f"points={ticket.get('story_points', '?')})"
        )

        description = ticket.get("description", "").strip()
        if description:
            lines.append(f"  Description: {description[:200]}")

        if ticket.get("root_cause"):
            lines.append(f"  Root cause: {ticket['root_cause']}")

        if ticket.get("gap_areas"):
            lines.append(f"  Knowledge gap areas: {', '.join(ticket['gap_areas'])}")

        comments = ticket.get("comments", [])
        if comments:
            lines.append("  Recent comments:")
            for c in comments:
                author = c.get("author", "?")
                date = c.get("date", "")
                text = c.get("text", "").strip().strip('"')[:150]
                lines.append(f"    {author} ({date}): {text}")

        linked_prs = ticket.get("linked_prs", [])
        if linked_prs:
            lines.append(f"  Linked PRs: {', '.join(linked_prs)}")

        blocker_filter: Dict[str, Any] = {
            "type": "blocker_flagged",
            "artifact_ids.jira": ticket_id,
        }
        if iso:
            blocker_filter["timestamp"] = {"$lte": iso}
        blockers = list(
            self._events.find(
                blocker_filter,
                {"_id": 0, "day": 1, "actors": 1, "facts.blocker_reason": 1},
            )
            .sort("timestamp", -1)
            .limit(2)
        )
        if blockers:
            lines.append("  Recent blockers:")
            for b in blockers:
                reason = b.get("facts", {}).get("blocker_reason", "")[:120]
                actors = b.get("actors", [])
                actor_str = actors[0] if actors else "?"
                lines.append(f"    Day {b.get('day', '?')} -- {actor_str}: {reason}")

        incident_filter: Dict[str, Any] = {
            "type": "incident_opened",
            "artifact_ids.jira": ticket_id,
        }
        if iso:
            incident_filter["timestamp"] = {"$lte": iso}
        incident_origin = self._events.find_one(
            incident_filter,
            {"_id": 0, "day": 1, "facts.root_cause": 1},
        )
        if incident_origin:
            root_cause = incident_origin.get("facts", {}).get("root_cause", "")
            lines.append(
                f"  Incident origin (Day {incident_origin.get('day', '?')}): {root_cause[:120]}"
            )

        progress_filter: Dict[str, Any] = {
            "type": "ticket_progress",
            "actors": assignee,
            "artifact_ids.jira": ticket_id,
        }
        if iso:
            progress_filter["timestamp"] = {"$lte": iso}
        prior_progress = list(
            self._events.find(
                progress_filter,
                {"_id": 0, "day": 1, "summary": 1},
            )
            .sort("timestamp", -1)
            .limit(3)
        )
        if prior_progress:
            lines.append("  Prior progress:")
            for p in prior_progress:
                lines.append(f"    Day {p.get('day', '?')} -- {p.get('summary', '')}")

        return "\n".join(lines)

    def context_for_person_conversations(
        self,
        name: str,
        conv_type: Optional[str] = None,
        as_of_time: Optional[Any] = None,
        n: int = 3,
    ) -> str:
        iso = self._to_iso(as_of_time)

        query: Dict[str, Any] = {"participants": name}
        if conv_type:
            query["type"] = conv_type
        if iso:
            query["timestamp"] = {"$lte": iso}

        docs = list(
            self._conversation_summaries.find(
                query,
                {
                    "_id": 0,
                    "type": 1,
                    "participants": 1,
                    "summary": 1,
                    "day": 1,
                    "slack_thread_id": 1,
                },
            )
            .sort("day", -1)
            .limit(n)
        )

        if not docs:
            return ""

        lines = [f"=== PAST CONVERSATIONS: {name} ==="]
        for d in docs:
            other = next((p for p in d["participants"] if p != name), "?")
            label = d.get("type", "conversation").replace("_", " ").title()
            lines.append(
                f"  Day {d.get('day', '?')} [{label} with {other}]: {d.get('summary', '')}"
            )

        return "\n".join(lines)

    def design_discussions_for_ticket(
        self,
        ticket_id: str,
        actors: List[str],
        as_of_time: Optional[Any] = None,
        n: int = 2,
    ) -> List[Dict]:
        iso = self._to_iso(as_of_time)

        query: Dict[str, Any] = {
            "type": "design_discussion",
            "$or": [
                {"facts.participants": {"$in": actors}},
                {"artifact_ids.jira": ticket_id},
            ],
        }
        if iso:
            query["timestamp"] = {"$lte": iso}

        docs = list(
            self._events.find(
                query,
                {
                    "_id": 0,
                    "day": 1,
                    "facts.topic": 1,
                    "facts.participants": 1,
                    "artifact_ids.slack_thread": 1,
                    "artifact_ids.confluence": 1,
                },
            )
            .sort("timestamp", -1)
            .limit(n * 3)
        )

        actor_set = set(actors)
        scored = []
        for d in docs:
            discussion_participants = set(d.get("facts", {}).get("participants", []))
            overlap = len(actor_set & discussion_participants)
            if overlap >= 1:
                scored.append((overlap, d))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for _, d in scored[:n]:
            results.append(
                {
                    "day": d.get("day", "?"),
                    "topic": d.get("facts", {}).get("topic", ""),
                    "participants": d.get("facts", {}).get("participants", []),
                    "slack_thread_id": d.get("artifact_ids", {}).get(
                        "slack_thread", ""
                    ),
                    "confluence_id": d.get("artifact_ids", {}).get("confluence", ""),
                }
            )
        return results

    def format_design_discussions_hint(self, discussions: List[Dict]) -> str:
        if not discussions:
            return ""

        lines = [
            "Note: related design discussions already happened -- reference if relevant:"
        ]
        for d in discussions:
            parts = [f"  Day {d['day']}"]
            if d["topic"]:
                parts.append(f"topic: '{d['topic'][:80]}'")
            if d["participants"]:
                parts.append(f"participants: {', '.join(d['participants'])}")
            if d["confluence_id"]:
                parts.append(f"doc: {d['confluence_id']}")
            lines.append(" -- ".join(parts))

        return "\n".join(lines)

    def stats(self) -> Dict:
        return {
            "confluence_page_count": self._confluence_pages.count_documents({}),
            "event_count": self._events.count_documents({}),
            "event_log_len": len(self._event_log),
            "artifact_count": (
                self._confluence_pages.count_documents({})
            ),
            "mongodb_ok": True,
        }

    def reset(self, export_dir: Optional[str] = None):
        db_name = self._db.name
        self._client.drop_database(db_name)
        self._db = self._client[db_name]

        self._db.create_collection("artifacts")
        self._db.create_collection("events")

        self._artifacts = self._db["artifacts"]
        self._events = self._db["events"]
        self._jira = self._db["jira_tickets"]
        self._prs = self._db["pull_requests"]
        self._checkpoints = self._db["checkpoints"]
        self._slack = self._db["slack_messages"]
        self._plans = self._db["dept_plans"]
        self._conversation_summaries = self._db["conversation_summaries"]
        self._confluence_pages = self._db["confluence_pages"]
        self._zoom_transcripts = self._db["zoom_transcripts"]

        self._event_log = []
        self._init_text_indexes()
        logger.info("[memory] Database reset.")

        if export_dir:
            export_path = Path(export_dir)
            if export_path.exists():
                shutil.rmtree(export_path)
            export_path.mkdir(parents=True, exist_ok=True)
            log_path = export_path / "simulation.log"
            root_logger = logging.getLogger()
            for handler in root_logger.handlers[:]:
                if isinstance(handler, logging.FileHandler):
                    handler.close()
                    root_logger.removeHandler(handler)
            new_handler = logging.FileHandler(log_path, mode="a")
            new_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            root_logger.addHandler(new_handler)
            logger.info(f"[memory] Export directory cleared: {export_path}")

    def has_genesis_artifacts(self) -> bool:
        return self._events.count_documents({"tags": "genesis"}) > 0

    def save_checkpoint(
        self,
        day: int,
        state_vars: Dict,
        stress: Dict,
        cursors: Dict,
        graph_data: Dict,
        active_incidents: Optional[List[Dict]] = None,
        sprint: Optional[Dict] = None,
        resolved_incidents: Optional[List[str]] = None,
        morale_history: Optional[List[float]] = None,
    ):
        self._checkpoints.update_one(
            {"day": day},
            {
                "$set": {
                    "day": day,
                    "state": state_vars,
                    "stress": stress,
                    "cursors": cursors,
                    "graph": graph_data,
                    "active_incidents": active_incidents or [],
                    "sprint": sprint or {},
                    "resolved_incidents": resolved_incidents or [],
                    "morale_history": morale_history or [],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            },
            upsert=True,
        )

    def load_latest_checkpoint(self) -> Optional[Dict]:
        return self._db["checkpoints"].find_one(sort=[("day", -1)])

    def upsert_ticket(self, ticket: Dict):
        self._jira.update_one({"id": ticket["id"]}, {"$set": ticket}, upsert=True)

    def get_ticket(
        self, ticket_id: str, as_of_time: Optional[str] = None
    ) -> Optional[Dict]:
        query: dict = {"id": ticket_id}
        if as_of_time:
            query["$or"] = [
                {"timestamp": {"$lte": as_of_time}},
                {"timestamp": {"$exists": False}},
            ]
        return self._jira.find_one(query, {"_id": 0})

    def get_open_tickets_for_dept(
        self, members: List[str], dept_name: str = ""
    ) -> List[Dict]:
        query: Dict[str, Any] = {"status": {"$ne": "Done"}}
        if dept_name:
            query["dept"] = dept_name
        else:
            query["assignee"] = {"$in": members}
        return list(self._jira.find(query, {"_id": 0}))

    def upsert_pr(self, pr: Dict):
        self._prs.update_one({"pr_id": pr["pr_id"]}, {"$set": pr}, upsert=True)

    def get_reviewable_prs_for(
        self, name: str, as_of_time: Optional[str] = None
    ) -> List[Dict]:
        query: dict = {"reviewers": name, "status": "open"}
        if as_of_time:
            query["timestamp"] = {"$lte": as_of_time}
        return list(self._prs.find(query, {"_id": 0}))

    def get_pr_by_ticket_id(self, ticket_id: str) -> Optional[Dict]:
        return self._prs.find_one({"ticket_id": ticket_id}, {"_id": 0})

    def log_slack_messages(
        self, channel: str, messages: List[Dict], export_dir: Path
    ) -> Tuple[str, str]:
        if not messages:
            return ("", "")

        date_str = messages[0].get("date")
        thread_id = f"slack_{channel}_{messages[0].get('ts', datetime.now(timezone.utc).isoformat())}"

        for m in messages:
            m["thread_id"] = thread_id

        channel_dir = export_dir / "slack" / "channels" / channel
        channel_dir.mkdir(parents=True, exist_ok=True)
        file_path = channel_dir / f"{date_str}.json"

        history = []
        if file_path.exists():
            with open(file_path, "r") as f:
                try:
                    history = json.load(f)
                except json.JSONDecodeError:
                    pass

        history.extend(messages)
        with open(file_path, "w") as f:
            json.dump(history, f, indent=2)

        db_docs = [
            {**m, "channel": channel, "file_path": str(file_path)} for m in messages
        ]
        self._slack.insert_many(db_docs)
        return (str(file_path), thread_id)

    def get_slack_history(self, channel: str, limit: int = 10) -> List[Dict]:
        return list(self._slack.find({"channel": channel}).sort("ts", -1).limit(limit))

    def get_recent_day_summaries(self, current_day: int, window: int = 7) -> List[dict]:
        cutoff = max(1, current_day - window)
        docs = self._events.find(
            {"type": "day_summary", "day": {"$gte": cutoff}}, {"facts": 1, "_id": 0}
        ).sort("day", 1)
        return [d["facts"] for d in docs if "facts" in d]

    def log_dept_plan(
        self,
        day: int,
        date: str,
        dept: str,
        lead: str,
        theme: str,
        engineer_plans: List[Dict],
        proposed_events: List[Dict],
        raw: dict,
    ) -> None:
        doc = {
            "day": day,
            "date": date,
            "dept": dept,
            "lead": lead,
            "theme": theme,
            "engineer_plans": engineer_plans,
            "proposed_events": proposed_events,
            "raw": raw,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        try:
            self._plans.insert_one(doc)
        except Exception as e:
            logger.warning(f"[memory] dept_plan insert failed: {e}")

    def save_conversation_summary(
        self,
        conv_type: str,
        participants: List[str],
        summary: str,
        day: int,
        date: str,
        timestamp: str,
        slack_thread_id: str = "",
        extra_facts: Optional[Dict] = None,
    ) -> None:
        doc = {
            "type": conv_type,
            "participants": sorted(participants),
            "summary": summary,
            "day": day,
            "date": date,
            "timestamp": timestamp,
            "slack_thread_id": slack_thread_id,
            **(extra_facts or {}),
        }
        try:
            self._conversation_summaries.insert_one(doc)
        except Exception as e:
            logger.warning(f"[memory] conversation_summary insert failed: {e}")

    def facts_for_event_type(self, event_type: str) -> List[Dict]:
        return [
            e.facts
            | {"date": e.date, "actors": e.actors, "artifact_ids": e.artifact_ids}
            for e in self.events_by_type(event_type)
        ]

    def _known_pressures_summary(self) -> str:
        lines = []
        for doc in self._confluence_pages.find(
            {"metadata.phase": "genesis"}, {"title": 1, "_id": 0}
        ).limit(5):
            lines.append(f"  - Existing doc: {doc['title']}")
        return "\n".join(lines) if lines else "  - No prior artifacts found."

    def save_tech_stack(self, stack: dict) -> None:
        self._db["sim_config"].update_one(
            {"_id": "tech_stack"},
            {
                "$set": {
                    "_id": "tech_stack",
                    "stack": stack,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            },
            upsert=True,
        )

    def get_tech_stack(self) -> Optional[dict]:
        doc = self._db["sim_config"].find_one({"_id": "tech_stack"})
        return doc["stack"] if doc else None

    def tech_stack_for_prompt(self) -> str:
        stack = self.get_tech_stack()
        if not stack:
            return ""
        lines = [
            "CANONICAL TECH STACK -- always reference these, never invent alternatives:"
        ]
        for key, value in stack.items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)

    def save_inbound_email_sources(self, sources: list) -> None:
        self._db["sim_config"].update_one(
            {"_id": "inbound_email_sources"},
            {
                "$set": {
                    "_id": "inbound_email_sources",
                    "sources": sources,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            },
            upsert=True,
        )

    def get_inbound_email_sources(self) -> Optional[list]:
        doc = self._db["sim_config"].find_one({"_id": "inbound_email_sources"})
        return doc["sources"] if doc else None
