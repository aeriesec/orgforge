"""
export_to_hf.py
===============
Normalises all OrgForge simulation artifacts into a flat HuggingFace-ready
corpus, runs BM25 and dense-retrieval baselines, produces Parquet files, and
writes a dataset card (README.md) to export/hf_dataset/.

Run after flow.py + eval_harness.py:
    python export_to_hf.py

Output layout
-------------
export/hf_dataset/
  corpus/
    corpus-00000.parquet        — flat document corpus (one row per artifact)
  questions/
    questions-00000.parquet     — eval questions with ground truth
  threads/
    threads-00000.parquet       — causal thread graphs (JSON-serialised)
  baselines/
    bm25_results.json           — BM25 retrieval scores per question
    dense_results.json          — dense retrieval scores per question
    baseline_summary.json       — aggregate numbers for the dataset card
  README.md                     — HuggingFace dataset card

Corpus schema (one row per document)
-------------------------------------
  doc_id          str   — globally unique, e.g. "ORG-42", "CONF-ENG-007", "EMAIL-001"
  doc_type        str   — "jira" | "confluence" | "slack" | "email" | "pr" | "sim_event"
  title           str   — human-readable title or subject line
  body            str   — full text content for retrieval
  day             int   — simulation day this artifact was created
  date            str   — ISO date string
  timestamp       str   — ISO datetime string (business-hours accurate)
  actors          str   — JSON array of actor names involved
  tags            str   — JSON array of tags from SimEvent
  artifact_ids    str   — JSON dict mapping type→id (for cross-referencing)
  dept            str   — owning department, empty if cross-dept
  is_incident     bool  — True if this artifact is part of an incident thread
  is_external     bool  — True for emails from outside the org

Question schema
---------------
  question_id         str
  question_type       str   — RETRIEVAL | CAUSAL | TEMPORAL | GAP_DETECTION | ROUTING
  question_text       str
  ground_truth        str   — JSON-serialised ground_truth dict
  evidence_chain      str   — JSON array of artifact IDs
  difficulty          str   — easy | medium | hard
  requires_reasoning  bool
  chain_id            str

Baseline methodology
---------------------
BM25   — rank_bm25 (Okapi BM25) over the body field.
          For each retrieval / causal / routing / gap question the top-10
          returned doc_ids are compared against evidence_chain.
          MRR@10 and Recall@10 are reported per question type.

Dense  — sentence-transformers "Losspost/stella_en_1.5b_v5" (1536-dim).
          Cosine similarity between question_text embedding and body embeddings.
          Same MRR@10 / Recall@10 reported for comparison.
          If sentence-transformers is not installed, this section is skipped
          gracefully and the dataset card notes the omission.

Temporal and GAP_DETECTION questions require boolean answers, not retrieval,
so baselines report evidence recall only (whether the right artifacts are
surfaced, not whether the final answer is correct).
"""

from __future__ import annotations

import json
import logging
import re
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger("orgforge.export_hf")

# ── Config ────────────────────────────────────────────────────────────────────
with open(Path(__file__).resolve().parent.parent / "config" / "config.yaml") as f:
    _CFG = yaml.safe_load(f)

_SIM_CFG = _CFG.get("simulation", {})
_ORG_CFG = _CFG.get("org", {})

# Build actor -> dept lookup from org_chart
# Handles org_chart nested under 'org' or at top level
_ORG_CHART = (
    _ORG_CFG.get("org_chart")
    or _CFG.get("org_chart")
    or {}
)
_ACTOR_TO_DEPT: Dict[str, str] = {}
for _dept, _members in _ORG_CHART.items():
    if isinstance(_members, list):
        for _name in _members:
            _ACTOR_TO_DEPT[str(_name).strip()] = _dept

BASE = Path(_SIM_CFG.get("output_dir", "./export"))
EVAL_DIR = BASE / "eval"
HF_DIR = BASE / "hf_dataset"
CORPUS_DIR = HF_DIR / "corpus"
QUES_DIR = HF_DIR / "questions"
THREAD_DIR = HF_DIR / "threads"
BASELINE_DIR = HF_DIR / "baselines"
_DENSE_MODEL_NAME = "Losspost/stella_en_1.5b_v5"

for d in (CORPUS_DIR, QUES_DIR, THREAD_DIR, BASELINE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── Optional imports (degrade gracefully) ────────────────────────────────────
try:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    _PARQUET_AVAILABLE = True
except ImportError:
    _PARQUET_AVAILABLE = False
    logger.warning(
        "pandas/pyarrow not installed — Parquet output disabled. "
        "pip install pandas pyarrow"
    )

try:
    from rank_bm25 import BM25Okapi

    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False
    logger.warning(
        "rank_bm25 not installed — BM25 baseline disabled. pip install rank-bm25"
    )

try:
    import ollama
    import numpy as np

    _DENSE_AVAILABLE = True
    _DENSE_MODEL_NAME = (
        "Losspost/stella_en_1.5b_v5"  # or whatever your memory.py uses
    )
except ImportError:
    _DENSE_AVAILABLE = False
    logger.warning("ollama not installed — dense baseline disabled.")


# ─────────────────────────────────────────────────────────────────────────────
# CORPUS BUILDER
# ─────────────────────────────────────────────────────────────────────────────


def _dept_from_artifact_id(artifact_id: str) -> str:
    """Derive department from artifact ID prefix e.g. CONF-ENG-019 -> Engineering."""
    parts = artifact_id.split("-")
    if len(parts) < 2:
        return ""
    code = parts[1].upper()
    return {
        "ENG":   "Engineering",
        "PRD":   "Product",
        "MKT":   "Sales_Marketing",
        "QA":    "QA_Support",
        "RETRO": "",
    }.get(code, "")


class CorpusBuilder:
    """
    Reads the MongoDB-persisted artifacts (via Memory) and the SimEvent log,
    then normalises every artifact into a flat list of corpus rows.

    Falls back to reconstructing from eval JSON if MongoDB is unavailable,
    which allows the exporter to run in offline/CI environments.
    """

    def __init__(self, mem=None):
        self._mem = mem
        self._events: List[dict] = []
        if mem is not None:
            try:
                raw = mem.get_event_log(from_db=True)
                self._events = [
                    e.to_dict() if hasattr(e, "to_dict") else e for e in raw
                ]
            except Exception as exc:
                logger.warning(f"Could not load SimEvent log from Memory: {exc}")

    # ── PUBLIC ────────────────────────────────────────────────────────────────

    def build(self) -> List[dict]:
        rows: List[dict] = []

        # 1. One or more rows per SimEvent (always available)
        # Events with both jira + confluence keys emit a row for each artifact
        for evt in self._events:
            evt_rows = self._sim_event_to_row(evt)
            if evt_rows:
                rows.extend(evt_rows)

        # 2. Supplement with richer artifact bodies from MongoDB if available
        if self._mem is not None:
            rows = self._enrich_from_mongo(rows)
            rows.extend(self._plans_to_corpus_rows())

        # Deduplicate: keep the row with the longest body for each doc_id
        seen: Dict[str, dict] = {}
        for row in rows:
            did = row["doc_id"]
            if did not in seen or len(row.get("body", "")) > len(seen[did].get("body", "")):
                seen[did] = row
        rows = list(seen.values())

        logger.info(f"  corpus: {len(rows)} documents")
        return rows

    # ── PRIVATE ───────────────────────────────────────────────────────────────

    def _sim_event_to_row(self, evt: dict) -> List[dict]:
        """
        Convert a SimEvent to one or more corpus rows.
        Events that reference multiple artifact types (e.g. both jira and
        confluence) emit one row per artifact so no artifact is silently dropped.
        """
        event_type = evt.get("type", "")
        artifact_ids = evt.get("artifact_ids", {})
        facts = evt.get("facts", {})

        # Shared fields derived once per event
        evt_actors = evt.get("actors", [])
        dept_val = str(facts.get("dept", "")).strip()
        if not dept_val and evt_actors:
            for _actor in evt_actors:
                _d = _ACTOR_TO_DEPT.get(str(_actor).strip(), "")
                if _d:
                    dept_val = _d
                    break

        is_incident = event_type in (
            "incident_opened", "incident_resolved",
            "escalation_chain", "postmortem_created",
        )
        is_external = event_type in (
            "inbound_external_email", "customer_email_routed",
            "vendor_email_routed", "email_dropped",
        )

        shared = {
            "day":          int(evt.get("day", 0)),
            "date":         str(evt.get("date", "")),
            "timestamp":    str(evt.get("timestamp", "")),
            "actors":       json.dumps(evt_actors),
            "tags":         json.dumps(evt.get("tags", [])),
            "artifact_ids": json.dumps(artifact_ids),
            "dept":         dept_val,
            "is_incident":  is_incident,
            "is_external":  is_external,
        }

        rows: List[dict] = []

        # ── JIRA ──────────────────────────────────────────────────────────────
        jira_id = artifact_ids.get("jira", "")
        if jira_id:
            rows.append({
                **shared,
                "doc_id":   jira_id,
                "doc_type": "jira",
                "title":    str(facts.get("title", facts.get("root_cause", jira_id)))[:512],
                "body":     self._jira_body(facts),
            })

        # ── CONFLUENCE ────────────────────────────────────────────────────────
        conf_id = artifact_ids.get("confluence", "") or next(
            (v for v in artifact_ids.values()
             if isinstance(v, str) and str(v).startswith("CONF-")), ""
        )
        if conf_id:
            body = facts.get("content", facts.get("summary", "")) or evt.get("summary", "")
            rows.append({
                **shared,
                "doc_id":   conf_id,
                "doc_type": "confluence",
                "title":    str(facts.get("title", conf_id))[:512],
                "body":     body,
                "dept":     dept_val or _dept_from_artifact_id(conf_id),
            })

        # ── EMAIL ─────────────────────────────────────────────────────────────
        email_id = artifact_ids.get("email", "")
        if email_id or event_type in (
            "inbound_external_email", "hr_outbound_email",
            "customer_email_routed", "vendor_email_routed", "email_dropped",
        ):
            rows.append({
                **shared,
                "doc_id":   email_id or f"EMAIL-{evt.get('day', 0)}-{id(evt)}",
                "doc_type": "email",
                "title":    str(facts.get("subject", facts.get("summary", email_id)))[:512],
                "body":     self._email_body(facts, evt),
            })

        # ── SLACK ─────────────────────────────────────────────────────────────
        # Only use slack_thread as the canonical ID — slack and slack_path
        # are file paths or message fragments, not retrievable thread documents
        slack_id = artifact_ids.get("slack_thread", "")
        if slack_id:
            channel = facts.get("channel", "#general")
            rows.append({
                **shared,
                "doc_id":   slack_id,
                "doc_type": "slack",
                "title":    str(channel + ": " + facts.get("summary", "")[:80])[:512],
                "body":     facts.get("content", facts.get("summary", "")),
            })

        # ── PR ────────────────────────────────────────────────────────────────
        pr_id = artifact_ids.get("pr", "")
        if pr_id:
            rows.append({
                **shared,
                "doc_id":   pr_id,
                "doc_type": "pr",
                "title":    str(facts.get("title", pr_id))[:512],
                "body":     facts.get("description", facts.get("summary", "")),
            })

        # ── FALLBACK ──────────────────────────────────────────────────────────
        if not rows:
            rows.append({
                **shared,
                "doc_id":   f"EVENT-{evt.get('day', 0)}-{event_type}",
                "doc_type": "sim_event",
                "title":    event_type.replace("_", " ").title(),
                "body":     evt.get("summary", ""),
            })

        # Ensure every row has a non-empty body
        for row in rows:
            if not row.get("body"):
                row["body"] = evt.get("summary", "")

        return rows

    def _jira_body(self, facts: dict) -> str:
        parts = []
        for key in (
            "title",
            "description",
            "root_cause",
            "fix_summary",
            "gap_areas",
            "comments",
        ):
            val = facts.get(key)
            if val:
                if isinstance(val, list):
                    parts.append(f"{key}: " + "; ".join(str(v) for v in val))
                else:
                    parts.append(f"{key}: {val}")
        return "\n".join(parts)

    def _email_body(self, facts: dict, evt: dict) -> str:
        parts = []
        for key in ("subject", "body", "from", "to", "summary", "source", "prospect"):
            val = facts.get(key)
            if val:
                parts.append(f"{key}: {val}")
        if not parts:
            parts.append(evt.get("summary", ""))
        return "\n".join(parts)

    def _plans_to_corpus_rows(self) -> List[dict]:
        rows = []
        for plan in self._mem._db["dept_plans"].find({}, {"_id": 0}):
            dept = plan["dept"]
            day = plan["day"]
            lead = plan.get("lead", "")
            theme = plan.get("theme", "")
            plan_id = f"PLAN-{day}-{dept}"

            # One row per engineer — body now includes theme + full agenda
            for ep in plan.get("engineer_plans", []):
                agenda_text = "\n".join(
                    f"{'[DEFERRED] ' if item.get('deferred') else ''}"
                    f"{item.get('activity_type')}: {item.get('description')}"
                    + (
                        f" (reason: {item.get('defer_reason')})"
                        if item.get("defer_reason")
                        else ""
                    )
                    for item in ep.get("agenda", [])
                )
                body = (
                    f"Dept: {dept}. Theme: {theme}. Lead: {lead}. "
                    f"Engineer: {ep.get('name', '')}.\n{agenda_text}"
                )
                rows.append(
                    {
                        "doc_id": plan_id,
                        "doc_type": "dept_plan",
                        "title": f"{dept} plan — Day {day}",
                        "body": body,
                        "day": day,
                        "date": plan["date"],
                        "timestamp": plan.get("timestamp", f"{plan['date']}T09:00:00"),
                        "actors": json.dumps([lead]),
                        "tags": json.dumps(["dept_plan", dept]),
                        "artifact_ids": json.dumps({"dept_plan": plan_id}),
                        "dept": dept,
                        "is_incident": False,
                        "is_external": False,
                    }
                )

            # Dept-level rationale row — fall back to theme if planner_reasoning absent
            reasoning = (
                plan.get("raw", {}).get("planner_reasoning", "")
                or plan.get("planner_reasoning", "")
                or f"Dept: {dept}. Theme: {theme}. Lead: {lead}."
            )
            rows.append(
                {
                    "doc_id": f"{plan_id}-reasoning",
                    "doc_type": "dept_plan_reasoning",
                    "title": f"{dept} planner reasoning — Day {day}",
                    "body": reasoning,
                    "day": day,
                    "date": plan["date"],
                    "timestamp": plan.get("timestamp", f"{plan['date']}T09:00:00"),
                    "actors": json.dumps([lead]),
                    "tags": json.dumps(["planner_reasoning", dept]),
                    "artifact_ids": json.dumps({}),
                    "dept": dept,
                    "is_incident": False,
                    "is_external": False,
                }
            )
        return rows

    def _enrich_from_mongo(self, rows: List[dict]) -> List[dict]:
        """
        Attempt to replace thin SimEvent body text with richer MongoDB content.
        Silently skips if the collection is unavailable.
        """
        try:
            rich_map: Dict[str, str] = {}

            # Confluence pages
            # Also build a snippet index so CONF-UNKNOWN rows can be
            # re-identified by matching their thin body against MongoDB content.
            conf_id_map: Dict[str, str] = {}   # content_snippet_or_title -> page_id
            for page in self._mem._db["confluence_pages"].find(
                {}, {"_id": 0, "id": 1, "content": 1, "title": 1}
            ):
                if page.get("id") and page.get("content"):
                    rich_map[page["id"]] = page["content"]
                    snippet = page["content"][:120].strip()
                    if snippet:
                        conf_id_map[snippet] = page["id"]
                    title_key = page.get("title", "").strip()
                    if title_key:
                        conf_id_map[title_key] = page["id"]

            # JIRA tickets + jira_comment artifacts folded into parent body
            comment_map: Dict[str, List[str]] = defaultdict(list)
            for comment in self._mem._db["artifacts"].find(
                {"type": "jira_comment"},
                {"_id": 0, "parent_id": 1, "body": 1, "author": 1}
            ):
                parent = comment.get("parent_id", "")
                cbody = comment.get("body", "")
                cauthor = comment.get("author", "")
                if parent and cbody:
                    comment_map[parent].append(
                        f"comment ({cauthor}): {cbody}" if cauthor else f"comment: {cbody}"
                    )

            for ticket in self._mem._db["jira_tickets"].find(
                {},
                {
                    "_id": 0,
                    "id": 1,
                    "title": 1,
                    "description": 1,
                    "root_cause": 1,
                    "comments": 1,
                },
            ):
                tid = ticket.get("id")
                if not tid:
                    continue
                parts = [
                    ticket.get("title", ""),
                    ticket.get("description", ""),
                    ticket.get("root_cause", ""),
                ]
                for c in ticket.get("comments") or []:
                    parts.append(str(c.get("body", "")))
                for c in comment_map.get(tid, []):
                    parts.append(c)
                rich_map[tid] = "\n".join(p for p in parts if p)

            for row in rows:
                if row["doc_id"] == "CONF-UNKNOWN" and row["doc_type"] == "confluence":
                    # Try to resolve the real ID via body snippet or title match
                    body_snippet = (row.get("body") or "")[:120].strip()
                    title_key = (row.get("title") or "").strip()
                    resolved_id = (
                        conf_id_map.get(body_snippet)
                        or conf_id_map.get(title_key)
                    )
                    if resolved_id:
                        row["doc_id"] = resolved_id
                        row["title"] = resolved_id  # was also CONF-UNKNOWN
                        row["body"] = rich_map[resolved_id]
                        if not row.get("dept"):
                            row["dept"] = _dept_from_artifact_id(resolved_id)
                    else:
                        # No real Confluence page exists — upstream emitted a
                        # confluence key with an empty value on a Slack-style
                        # social interaction. Reclassify correctly.
                        row["doc_type"] = "slack"
                        row["doc_id"] = (
                            f"SLACK-SOCIAL-{row.get('day', 0)}-"
                            f"{abs(hash(body_snippet)) % 10000:04d}"
                        )
                        row["title"] = (row.get("body") or "")[:80].strip()
                        logger.debug(
                            f"Reclassified CONF-UNKNOWN social event as slack: "
                            f"{row['doc_id']}"
                        )
                elif row["doc_id"] in rich_map:
                    row["body"] = rich_map[row["doc_id"]]
                    if row["doc_type"] == "confluence" and not row.get("dept"):
                        row["dept"] = _dept_from_artifact_id(row["doc_id"])
            # ── Orphan sweep ──────────────────────────────────────────────
            # Create corpus rows for any artifacts in MongoDB not yet in corpus.
            # Covers all retrievable types; excludes jira_comment (folded above)
            # and persona_skill (not a corpus artifact).
            # slack_thread = full thread document (correct corpus unit)
            # slack = individual message fragments — excluded, same as jira_comment
            # slack_messages collection also excluded for same reason
            _TYPE_MAP = {
                "confluence":   "confluence",
                "slack_thread": "slack",
                "email":        "email",
                "pr":           "pr",
                "jira":         "jira",
            }
            existing_ids = {row["doc_id"] for row in rows}
            for artifact in self._mem._db["artifacts"].find(
                {"type": {"$in": list(_TYPE_MAP.keys())}},
                {"_id": 1, "type": 1, "content": 1, "body": 1, "title": 1,
                 "subject": 1, "day": 1, "date": 1, "timestamp": 1,
                 "metadata": 1, "author": 1, "actors": 1}
            ):
                art_id = str(artifact.get("_id", ""))
                art_type = artifact.get("type", "")
                doc_type = _TYPE_MAP.get(art_type, "sim_event")
                if not art_id or art_id in existing_ids:
                    continue
                meta = artifact.get("metadata", {})
                author = artifact.get("author") or meta.get("author", "")
                actors = artifact.get("actors") or ([author] if author else [])
                tags = meta.get("tags", [art_type])
                body = (
                    artifact.get("content")
                    or artifact.get("body")
                    or artifact.get("subject")
                    or ""
                )
                title = artifact.get("title") or artifact.get("subject") or art_id
                dept = (
                    _dept_from_artifact_id(art_id)
                    or next(
                        (_ACTOR_TO_DEPT.get(str(a), "") for a in actors
                         if _ACTOR_TO_DEPT.get(str(a))), ""
                    )
                )
                rows.append({
                    "doc_id":       art_id,
                    "doc_type":     doc_type,
                    "title":        str(title)[:512],
                    "body":         str(body),
                    "day":          int(artifact.get("day", 0)),
                    "date":         str(artifact.get("date", "")),
                    "timestamp":    str(artifact.get("timestamp", "")),
                    "actors":       json.dumps(actors),
                    "tags":         json.dumps(tags),
                    "artifact_ids": json.dumps({art_type: art_id}),
                    "dept":         dept,
                    "is_incident":  any(t in tags for t in ("postmortem", "incident")),
                    "is_external":  art_type == "email",
                })
                logger.debug(f"  orphan artifact added: {art_id} ({doc_type})")

        except Exception as exc:
            logger.debug(f"MongoDB enrichment skipped: {exc}")
        return rows


# ─────────────────────────────────────────────────────────────────────────────
# BASELINE RUNNER
# ─────────────────────────────────────────────────────────────────────────────

# Question types that rely on evidence retrieval (vs. boolean reasoning)
_RETRIEVAL_TYPES = {
    "RETRIEVAL", "CAUSAL", "ROUTING", "GAP_DETECTION", "TEMPORAL",
    "ESCALATION", "KNOWLEDGE_GAP", "POSTMORTEM", "STANDUP", "CUSTOMER_ESC",
}


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokeniser."""
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


def _mrr_at_k(ranked_ids: List[str], relevant_ids: List[str], k: int = 10) -> float:
    for i, did in enumerate(ranked_ids[:k], 1):
        if did in set(relevant_ids):
            return 1.0 / i
    return 0.0


def _recall_at_k(ranked_ids: List[str], relevant_ids: List[str], k: int = 10) -> float:
    if not relevant_ids:
        return 1.0
    hits = sum(1 for did in ranked_ids[:k] if did in set(relevant_ids))
    return hits / len(relevant_ids)


class BaselineRunner:
    """
    Runs BM25 and (optionally) dense retrieval baselines against the
    eval questions and returns per-question and aggregate metrics.
    """

    def __init__(self, corpus: List[dict], questions: List[dict], mem=None):
        self._corpus = corpus
        self._questions = questions
        self._mem = mem
        self._doc_ids = [row["doc_id"] for row in corpus]
        self._bodies = [row["body"] or "" for row in corpus]

        # BM25 index
        if _BM25_AVAILABLE:
            tokenised = [_tokenize(b) for b in self._bodies]
            self._bm25 = BM25Okapi(tokenised)
        else:
            self._bm25 = None

    # ── PUBLIC ────────────────────────────────────────────────────────────────

    def run_bm25(self) -> Tuple[List[dict], Dict[str, Any]]:
        if self._bm25 is None:
            return [], {"error": "rank_bm25 not installed"}
        return self._run_retrieval(use_dense=False)

    def run_dense(self) -> Tuple[List[dict], Dict[str, Any]]:
        if self._mem is None:
            return [], {"error": "Memory unavailable — dense baseline requires MongoDB"}
        return self._run_retrieval(use_dense=True)

    def _rank(self, query: str, use_dense: bool, top_k: int = 10) -> List[str]:
        if use_dense and self._mem is not None:
            # Use the same collection and pattern as Memory.recall()
            results = self._mem.recall(query=query, n=top_k)
            # recall() returns {"id": artifact._id, ...} — _id IS the artifact doc_id
            return [r["id"] for r in results if r.get("id")]
        elif self._bm25 is not None:
            scores = self._bm25.get_scores(_tokenize(query))
            indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            return [self._doc_ids[i] for i in indices[:top_k]]
        return []

    # ── PRIVATE ───────────────────────────────────────────────────────────────

    def _run_retrieval(self, use_dense: bool) -> Tuple[List[dict], Dict[str, Any]]:
        per_question: List[dict] = []
        by_type: Dict[str, List[float]] = defaultdict(list)

        for q in self._questions:
            qtype = q.get("question_type", "")
            evidence = q.get("evidence_chain", [])
            if not evidence:
                continue  # no reference artifacts — skip

            q_text = q.get("question_text", "")
            ranked_ids = self._rank(q_text, use_dense=use_dense)

            mrr = _mrr_at_k(ranked_ids, evidence, k=10)
            recall = _recall_at_k(ranked_ids, evidence, k=10)

            per_question.append(
                {
                    "question_id": q.get("question_id"),
                    "question_type": qtype,
                    "difficulty": q.get("difficulty"),
                    "mrr_at_10": round(mrr, 4),
                    "recall_at_10": round(recall, 4),
                    "top10": ranked_ids[:10],
                }
            )

            by_type[qtype].append((mrr, recall))

        # Aggregate
        def _mean(vals):
            return round(sum(vals) / len(vals), 4) if vals else 0.0

        aggregate = {
            "method": "dense" if use_dense else "bm25",
            "model": _DENSE_MODEL_NAME if use_dense else "BM25Okapi (rank-bm25)",
            "overall": {
                "mrr_at_10": _mean([r["mrr_at_10"] for r in per_question]),
                "recall_at_10": _mean([r["recall_at_10"] for r in per_question]),
                "n": len(per_question),
            },
            "by_type": {
                qtype: {
                    "mrr_at_10": _mean([v[0] for v in vals]),
                    "recall_at_10": _mean([v[1] for v in vals]),
                    "n": len(vals),
                }
                for qtype, vals in by_type.items()
            },
        }
        return per_question, aggregate

    """ def _rank(self, query: str, use_dense: bool, top_k: int = 10) -> List[str]:
        if use_dense and self._dense_matrix is not None:
            q_vec = self._dense_model.encode([query], normalize_embeddings=True)[0]
            scores = self._dense_matrix @ q_vec
            indices = scores.argsort()[::-1][:top_k]
            return [self._doc_ids[i] for i in indices]
        elif self._bm25 is not None:
            scores = self._bm25.get_scores(_tokenize(query))
            indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            return [self._doc_ids[i] for i in indices[:top_k]]
        return [] """


# ─────────────────────────────────────────────────────────────────────────────
# DATASET CARD WRITER
# ─────────────────────────────────────────────────────────────────────────────


class DatasetCardWriter:
    """Produces the HuggingFace README.md dataset card."""

    def write(
        self,
        out_path: Path,
        corpus: List[dict],
        questions: List[dict],
        threads: List[dict],
        baseline_summary: dict,
        cfg: dict,
    ) -> None:
        card = self._render(corpus, questions, threads, baseline_summary, cfg)
        out_path.write_text(card, encoding="utf-8")
        logger.info(f"  → {out_path}")

    # ── PRIVATE ───────────────────────────────────────────────────────────────

    def _render(
        self,
        corpus: List[dict],
        questions: List[dict],
        threads: List[dict],
        baseline_summary: dict,
        cfg: dict,
    ) -> str:
        sim_cfg = cfg.get("simulation", {})
        org_cfg = cfg.get("org", {})

        num_days = sim_cfg.get("num_days", "?")
        org_chart = cfg.get("org_chart", {})
        org_size = sum(len(v) for v in org_chart.values() if isinstance(v, list))
        company = sim_cfg.get("company_name", "OrgForge Simulated Corp")
        industry = sim_cfg.get("industry", "Software")
        num_sprints = sim_cfg.get("num_sprints", "?")

        # Corpus breakdown
        by_type: Dict[str, int] = defaultdict(int)
        for row in corpus:
            by_type[row["doc_type"]] += 1

        # Question breakdown
        by_qtype: Dict[str, int] = defaultdict(int)
        by_diff: Dict[str, int] = defaultdict(int)
        for q in questions:
            by_qtype[q.get("question_type", "?")] += 1
            by_diff[q.get("difficulty", "?")] += 1

        # Thread breakdown
        by_chain: Dict[str, int] = defaultdict(int)
        for t in threads:
            by_chain[t.get("chain_type", "?")] += 1

        # Baseline tables
        bm25_section = self._baseline_table(baseline_summary.get("bm25", {}))
        dense_section = self._baseline_table(baseline_summary.get("dense", {}))

        return textwrap.dedent(f"""\
        ---
        language:
          - en
        license: apache-2.0
        task_categories:
          - question-answering
          - text-retrieval
        task_ids:
          - extractive-qa
          - document-retrieval
          - multi-hop-retrieval
        tags:
          - rag
          - enterprise
          - synthetic
          - orgforge
          - causal-reasoning
          - temporal-reasoning
        pretty_name: "OrgForge Enterprise RAG Benchmark"
        size_categories:
          - 1K<n<10K
        ---

        # OrgForge Enterprise RAG Benchmark

        > A synthetic but causally-grounded benchmark for evaluating RAG systems
        > against realistic enterprise knowledge bases.

        ## Dataset Summary

        This dataset was produced by **OrgForge**, an event-driven organisation
        simulator that generates weeks of realistic enterprise activity — JIRA
        tickets, Confluence pages, Slack threads, emails, and PRs — in a
        controlled, reproducible way. All ground-truth answers are derived
        deterministically from the simulation's event log; no LLM invented any
        answer.

        | Property | Value |
        |---|---|
        | Company | {company} |
        | Industry | {industry} |
        | Simulation days | {num_days} |
        | Sprints simulated | {num_sprints} |
        | Org size (engineers + staff) | ~{org_size} |
        | Total corpus documents | {len(corpus):,} |
        | Total eval questions | {len(questions):,} |
        | Causal threads | {len(threads):,} |

        ## Corpus

        Each document in the corpus represents a real artifact produced by the
        simulation (ticket, page, thread, email, PR). Documents are stored in
        `corpus/corpus-00000.parquet`.

        | Artifact type | Count |
        |---|---|
        {self._table_rows(by_type)}

        ### Schema

        | Column | Type | Description |
        |---|---|---|
        | `doc_id` | str | Unique artifact ID (e.g. `ORG-42`, `CONF-ENG-007`) |
        | `doc_type` | str | `jira`, `confluence`, `slack`, `email`, `pr`, `sim_event` |
        | `title` | str | Human-readable title or subject |
        | `body` | str | Full retrievable text |
        | `day` | int | Simulation day (1-indexed) |
        | `date` | str | ISO date |
        | `timestamp` | str | ISO datetime (business-hours-accurate) |
        | `actors` | str | JSON list of actor names |
        | `tags` | str | JSON list of semantic tags |
        | `artifact_ids` | str | JSON dict of cross-references |
        | `dept` | str | Owning department |
        | `is_incident` | bool | True if part of an incident thread |
        | `is_external` | bool | True for inbound external emails |

        ## Eval Questions

        Questions are in `questions/questions-00000.parquet`.

        | Question type | Count |
        |---|---|
        {self._table_rows(by_qtype)}

        | Difficulty | Count |
        |---|---|
        {self._table_rows(by_diff)}

        ### Question Types

        | Type | Description | Requires multi-hop? |
        |---|---|---|
        | `RETRIEVAL` | Which artifact first documented a specific fact? | No |
        | `CAUSAL` | What artifact or action directly followed event X? | Yes (2-hop) |
        | `TEMPORAL` | Did person P have access to domain D before incident I? | Yes (cross-thread) |
        | `GAP_DETECTION` | Was email E ever actioned? | Yes (absence-of-evidence) |
        | `ROUTING` | Who was the first internal person to see an inbound email? | No |
        | `PLAN` | What was department X focused on during Day N? | No |
        | `ESCALATION` | Who was involved in the escalation chain for incident X? | No |
        | `KNOWLEDGE_GAP` | What domain was undocumented when incident X fired? | No |
        | `POSTMORTEM` | Which Confluence doc captured the postmortem for incident X? | Yes (2-hop) |
        | `STANDUP` | What did person X report at standup on Day N? | No |
        | `CUSTOMER_ESC` | Who handled the escalation from customer X and what action was taken? | Yes (2-hop) |

        ### Question Schema

        | Column | Type | Description |
        |---|---|---|
        | `question_id` | str | Unique question identifier |
        | `question_type` | str | One of the five types above |
        | `question_text` | str | Natural-language question |
        | `ground_truth` | str | JSON-serialised answer dict |
        | `evidence_chain` | str | JSON list of artifact IDs that support the answer |
        | `difficulty` | str | `easy`, `medium`, `hard` |
        | `requires_reasoning` | bool | Multi-hop traversal required? |
        | `chain_id` | str | Causal thread this question derives from |

        ## Causal Threads

        Causal threads are in `threads/threads-00000.parquet`.
        Each thread is a directed artifact graph with actor knowledge annotations.

        | Chain type | Count |
        |---|---|
        {self._table_rows(by_chain)}

        ## Baselines

        All baselines evaluate **evidence retrieval** — whether the correct
        artifacts from `evidence_chain` are surfaced in the top-10 results —
        rather than final answer accuracy. MRR@10 and Recall@10 are reported.

        TEMPORAL and GAP_DETECTION questions test absence-of-evidence reasoning;
        retrieval metrics for those types measure whether the relevant artifacts
        are found, not whether the boolean conclusion is correct.

        ### BM25 (Okapi BM25 via rank-bm25)

        {bm25_section}

        ### Dense Retrieval (sentence-transformers `{_DENSE_MODEL_NAME}`)

        {dense_section}

        ## Scoring

        Use `scorer.py` (included in this repository) to evaluate agent answers
        against the ground truth. `scorer.py` implements per-type comparison
        logic with partial credit and returns a `ScorerResult` per question.

        ```python
        from scorer import OrgForgeScorer
        scorer = OrgForgeScorer()
        result = scorer.score(question, agent_answer)
        report = scorer.report(scorer.score_all(questions, answers))
        ```

        Scores are in [0.0, 1.0]. A score ≥ 0.9 is considered correct.
        Partial credit (0.2–0.9) is awarded when the agent retrieves the right
        artifacts but draws an incorrect conclusion.

        ## Citation

        ```bibtex
        @misc{{orgforge2026,
          title  = {{OrgForge: A Multi-Agent Simulation Framework for Verifiable Synthetic Corporate Corpora}},
          author = {{Jeffrey Flynt}},
          year   = {{2026}},
          note   = {{Synthetic benchmark generated by the OrgForge simulator}}
        }}
        ```

        ## Licence

        Apache 2.0. The simulation engine that produced this dataset is
        independently licensed; see the OrgForge repository for details.
        """)

    def _table_rows(self, d: Dict[str, int]) -> str:
        return "\n        ".join(
            f"| `{k}` | {v:,} |" for k, v in sorted(d.items(), key=lambda x: -x[1])
        )

    def _baseline_table(self, summary: dict) -> str:
        if "error" in summary:
            return f"> ⚠️ Baseline unavailable: {summary['error']}"
        if not summary:
            return "> Baseline not run."

        model = summary.get("model", "?")
        overall = summary.get("overall", {})
        by_type = summary.get("by_type", {})

        lines = [f"Model: `{model}`\n"]
        lines.append("| Question type | MRR@10 | Recall@10 | N |")
        lines.append("|---|---|---|---|")
        lines.append(
            f"| **Overall** | **{overall.get('mrr_at_10', '?')}** "
            f"| **{overall.get('recall_at_10', '?')}** "
            f"| **{overall.get('n', '?')}** |"
        )
        for qtype, metrics in sorted(by_type.items()):
            lines.append(
                f"| {qtype} | {metrics.get('mrr_at_10', '?')} "
                f"| {metrics.get('recall_at_10', '?')} "
                f"| {metrics.get('n', '?')} |"
            )
        return "\n        ".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# PARQUET WRITER
# ─────────────────────────────────────────────────────────────────────────────


def _write_parquet(rows: List[dict], out_dir: Path, stem: str = "part-00000") -> None:
    if not _PARQUET_AVAILABLE:
        out_path = out_dir / f"{stem}.json"
        with open(out_path, "w") as f:
            json.dump(rows, f, indent=2, default=str)
        logger.info(
            f"  → {out_path} (JSON fallback — install pandas+pyarrow for Parquet)"
        )
        return

    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    df = pd.DataFrame(rows)
    tbl = pa.Table.from_pandas(df)
    out_path = out_dir / f"{stem}.parquet"
    pq.write_table(tbl, out_path, compression="snappy")
    logger.info(
        f"  → {out_path} ({len(rows):,} rows, {out_path.stat().st_size // 1024} KB)"
    )


def _questions_to_rows(questions: List[dict]) -> List[dict]:
    rows = []
    for q in questions:
        rows.append(
            {
                "question_id": q.get("question_id", ""),
                "question_type": q.get("question_type", ""),
                "question_text": q.get("question_text", ""),
                "ground_truth": json.dumps(q.get("ground_truth", {}), default=str),
                "evidence_chain": json.dumps(q.get("evidence_chain", []), default=str),
                "difficulty": q.get("difficulty", ""),
                "requires_reasoning": bool(q.get("requires_reasoning", False)),
                "chain_id": q.get("chain_id", ""),
            }
        )
    return rows


def _threads_to_rows(threads: List[dict]) -> List[dict]:
    rows = []
    for t in threads:
        rows.append(
            {
                "chain_id": t.get("chain_id", ""),
                "chain_type": t.get("chain_type", ""),
                "root_artifact": t.get("root_artifact", ""),
                "root_event_type": t.get("root_event_type", ""),
                "day": int(t.get("day", 0)),
                "date": str(t.get("date", "")),
                "terminal_artifact": t.get("terminal_artifact", ""),
                "complete": bool(t.get("complete", False)),
                "nodes": json.dumps(t.get("nodes", []), default=str),
                # type-specific extras
                "high_priority": bool(t.get("high_priority", False)),
                "source": str(t.get("source", "")),
                "prospect": str(t.get("prospect", "")),
                "confluence_id": str(t.get("confluence_id", "")),
                "root_cause": str(t.get("root_cause", "")),
            }
        )
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────


class HFExporter:
    """
    Orchestrates the full export pipeline:
      1. Build corpus from SimEvent log + MongoDB
      2. Load causal threads and eval questions
      3. Run BM25 and dense baselines
      4. Write Parquet files + dataset card
    """

    def run(self) -> None:
        logger.info("[bold cyan]📦 HuggingFace dataset export starting…[/bold cyan]")

        # 1. Memory (optional — degrade gracefully)
        mem = None
        try:
            from memory import Memory

            mem = Memory()
            logger.info("  Connected to MongoDB Memory.")
        except Exception as exc:
            logger.warning(
                f"  Memory unavailable ({exc}). Corpus will derive from eval JSON only."
            )

        # 2. Corpus
        corpus_builder = CorpusBuilder(mem)
        corpus = corpus_builder.build()

        # If Memory was unavailable, try to reconstruct doc stubs from eval questions
        if not corpus:
            logger.warning("  Empty corpus — check that flow.py has run first.")

        # 3. Load eval data
        threads_path = EVAL_DIR / "causal_threads.json"
        questions_path = EVAL_DIR / "eval_questions.json"

        threads = json.loads(threads_path.read_text()) if threads_path.exists() else []
        q_data = (
            json.loads(questions_path.read_text()) if questions_path.exists() else {}
        )
        questions = q_data.get("questions", []) if isinstance(q_data, dict) else q_data

        logger.info(
            f"  {len(threads)} causal threads, {len(questions)} eval questions loaded"
        )

        # 4. Baselines
        baseline_runner = BaselineRunner(corpus, questions, mem=mem)
        bm25_per_q, bm25_agg = baseline_runner.run_bm25()
        dense_per_q, dense_agg = baseline_runner.run_dense()

        baseline_summary = {"bm25": bm25_agg, "dense": dense_agg}

        # Write per-question baseline results
        with open(BASELINE_DIR / "bm25_results.json", "w") as f:
            json.dump(bm25_per_q, f, indent=2, default=str)
        with open(BASELINE_DIR / "dense_results.json", "w") as f:
            json.dump(dense_per_q, f, indent=2, default=str)
        with open(BASELINE_DIR / "baseline_summary.json", "w") as f:
            json.dump(baseline_summary, f, indent=2, default=str)
        logger.info(f"  → baselines written to {BASELINE_DIR}")

        # 5. Parquet
        _write_parquet(corpus, CORPUS_DIR, "corpus-00000")
        _write_parquet(_questions_to_rows(questions), QUES_DIR, "questions-00000")
        _write_parquet(_threads_to_rows(threads), THREAD_DIR, "threads-00000")

        # 6. Dataset card
        DatasetCardWriter().write(
            out_path=HF_DIR / "README.md",
            corpus=corpus,
            questions=questions,
            threads=threads,
            baseline_summary=baseline_summary,
            cfg=_CFG,
        )

        logger.info(
            f"[green]✓ Export complete.[/green] "
            f"Output: {HF_DIR}  |  "
            f"BM25 overall MRR@10: {bm25_agg.get('overall', {}).get('mrr_at_10', 'n/a')}  |  "
            f"Dense overall MRR@10: {dense_agg.get('overall', {}).get('mrr_at_10', 'n/a')}"
        )


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    HFExporter().run()
