"""
export_to_hf.py
===============
Normalises all OrgForge simulation artifacts into a flat HuggingFace-ready
corpus, computes a two-tier baseline, produces Parquet files, and writes a
dataset card (README.md) to export/hf_dataset/.

Run after flow.py + eval_harness.py:
    python export_to_hf.py

Output layout
-------------
export/hf_dataset/
  corpus/
    corpus-00000.parquet              — flat document corpus (one row per artifact)
  questions/
    questions-00000.parquet           — eval questions with ground truth
  eval_indexes/
    causal_link_index.parquet         — explicit causal links from CausalLinkIndexer
    actor_visibility.parquet          — per-actor visibility cones, one row per (actor, day)
    absence_catalog.parquet           — expected-but-absent artifact pairs
  baselines/
    ungated_ceiling_bm25.json         — per-question ungated BM25 ceiling scores
    ungated_ceiling_dense.json        — per-question ungated dense ceiling scores
    static_reasoning_metrics.json     — per-question + aggregate reasoning difficulty metrics
    baseline_summary.json             — combined summary for the dataset card
  README.md                           — HuggingFace dataset card

Two-tier baseline design
------------------------
This file computes only what requires NO agent execution:

  Tier 1 — Ungated Retrieval Ceiling (UngatedCeilingBaseline)
    BM25 and dense retrieval with all gates removed ("god-mode" corpus access).
    MRR@10 and Recall@10 are reported for PERSPECTIVE and COUNTERFACTUAL only.
    SILENCE is excluded — absence cannot be measured by retrieval recall.
    The delta between this ceiling and a gated agent's combined_score is the
    "Epistemic Tax" — the difficulty cost of respecting organisational silos.

  Tier 2 — Static Reasoning Difficulty (StaticReasoningMetrics)
    Metrics derived from corpus + question metadata alone. No LLM, no agent.
    PERSPECTIVE    → horizon_contamination_rate (fraction of ungated top-20 outside cone)
    COUNTERFACTUAL → causal_chain_traceable (do cause+effect co-appear in top-10?)
    SILENCE        → search_space_bm25_coverage (fraction of required locations BM25 finds)

Agent-level baselines (ungated god-mode agent, zero-shot no-tools) require LLM
calls and belong in agentic_eval_harness.py as --ungated / --zero-shot flags.

Corpus schema (one row per document)
-------------------------------------
  doc_id          str   — globally unique, e.g. "ORG-42", "CONF-ENG-007", "EMAIL-001"
  doc_type        str   — "jira" | "confluence" | "slack" | "email" | "pr" |
                         "zd_ticket" | "sf_opp" | "sf_account" | "sim_event"
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
  question_id               str
  question_type             str   — PERSPECTIVE | COUNTERFACTUAL | SILENCE
  question_text             str
  ground_truth              str   — JSON-serialised ground_truth dict
  evidence_chain            str   — JSON array of artifact IDs (cause+effect for
                                    COUNTERFACTUAL; evidence_artifacts for PERSPECTIVE;
                                    empty for SILENCE — absence cannot be recalled)
  difficulty                str   — medium | hard
  requires_reasoning        bool
  actor                     str   — PERSPECTIVE only
  actor_role                str   — PERSPECTIVE only
  as_of_day                 int   — PERSPECTIVE only
  subsystem_access          str   — JSON list; PERSPECTIVE only
  blocked_subsystems        str   — JSON list; PERSPECTIVE only
  actor_visible_artifacts   str   — JSON list; PERSPECTIVE only
  link_type                 str   — COUNTERFACTUAL only (causal link type)
  causal_day                int   — COUNTERFACTUAL only
  expected_search_space     str   — JSON list; SILENCE only
  trigger_event_type        str   — SILENCE only
  expected_response_type    str   — SILENCE only

Eval indexes
-------------
causal_link_index   — one row per CausalLink (see eval_harness.CausalLink)
actor_visibility    — one row per (actor, day) ActorVisibilityCone snapshot
absence_catalog     — one row per AbsenceRecord

Baseline methodology
---------------------
BM25   — rank_bm25 (Okapi BM25) over the body field.
          For PERSPECTIVE and COUNTERFACTUAL questions the top-10 returned
          doc_ids are compared against evidence_chain.
          MRR@10 and Recall@10 are reported per question type.
          SILENCE questions are skipped — the correct answer is absence,
          so standard retrieval recall is not applicable.

Dense  — via Memory._embed() (same embedding model used by the simulation).
          Cosine similarity between question_text embedding and body embeddings.
          Same MRR@10 / Recall@10 as BM25.
          If Memory is unavailable, this section is skipped gracefully and the
          dataset card notes the omission.
"""

from __future__ import annotations

import json
import logging
import re
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np

import yaml

logger = logging.getLogger("orgforge.export_hf")

# ── Config ────────────────────────────────────────────────────────────────────
with open(Path(__file__).resolve().parent.parent / "config" / "config.yaml") as f:
    _CFG = yaml.safe_load(f)

_SIM_CFG = _CFG.get("simulation", {})
_ORG_CFG = _CFG.get("org", {})
_ORG_CHART = _ORG_CFG.get("org_chart") or _CFG.get("org_chart") or {}
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
EVAL_INDEX_DIR = HF_DIR / "eval_indexes"
BASELINE_DIR = HF_DIR / "baselines"
_DENSE_MODEL_NAME = "Losspost/stella_en_1.5b_v5"

for d in (CORPUS_DIR, QUES_DIR, EVAL_INDEX_DIR, BASELINE_DIR):
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

_DENSE_AVAILABLE = True
_DENSE_MODEL_NAME = "Qwen/Qwen3-Embedding-4B"


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
        "ENG": "Engineering",
        "PRD": "Product",
        "MKT": "Sales_Marketing",
        "QA": "QA_Support",
        "RETRO": "",
    }.get(code, "")


class CorpusBuilder:
    """
    Reads the MongoDB-persisted artifacts (via Memory) and the SimEvent log,
    then normalizes every artifact into a flat list of corpus rows.

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

        for evt in self._events:
            evt_rows = self._sim_event_to_row(evt)
            if evt_rows:
                rows.extend(evt_rows)

        if self._mem is not None:
            rows = self._enrich_from_mongo(rows)
            rows.extend(self._plans_to_corpus_rows())

        rows.extend(self._post_sim_to_corpus_rows())

        # Deduplicate: keep the row with the longest body for each doc_id
        seen: Dict[str, dict] = {}
        for row in rows:
            did = row["doc_id"]
            if did not in seen or self._body_len(row) > self._body_len(seen[did]):
                seen[did] = row

        for row in seen.values():
            if not row.get("body"):
                row["body"] = row.get("content") or ""

        rows = list(seen.values())
        logger.info(f"  corpus: {len(rows)} documents")
        return rows

    # ── PRIVATE ───────────────────────────────────────────────────────────────

    def _body_len(self, r: dict) -> int:
        return len(r.get("body") or r.get("content") or "")

    def _sim_event_to_row(self, evt: dict) -> List[dict]:
        event_type = evt.get("type", "")
        artifact_ids = evt.get("artifact_ids", {})
        facts = evt.get("facts", {})

        _EXCLUDED_EVENT_TYPES = {"dlp_alert", "secret_detected"}
        if event_type in _EXCLUDED_EVENT_TYPES:
            return []

        evt_actors = evt.get("actors", [])
        dept_val = str(facts.get("dept", "")).strip()
        if not dept_val and evt_actors:
            for _actor in evt_actors:
                _d = _ACTOR_TO_DEPT.get(str(_actor).strip(), "")
                if _d:
                    dept_val = _d
                    break

        is_incident = event_type in (
            "incident_opened",
            "incident_resolved",
            "escalation_chain",
            "postmortem_created",
            "zd_tickets_escalated",
            "sf_deals_risk_flagged",
            "crm_account_at_risk",
        )
        is_external = event_type in (
            "inbound_external_email",
            "customer_email_routed",
            "vendor_email_routed",
            "email_dropped",
            "sales_outbound_email",
            "proactive_outreach_initiated",
            "zd_ticket_opened",
            "zd_tickets_resolved",
            "crm_touchpoint",
            "customer_health_briefing",
        )

        shared = {
            "day": int(evt.get("day", 0)),
            "date": str(evt.get("date", "")),
            "timestamp": str(evt.get("timestamp", "")),
            "actors": json.dumps(evt_actors),
            "tags": json.dumps(evt.get("tags", [])),
            "artifact_ids": json.dumps(artifact_ids),
            "dept": dept_val,
            "is_incident": is_incident,
            "is_external": is_external,
        }

        rows: List[dict] = []

        jira_id = artifact_ids.get("jira", "")
        if jira_id:
            rows.append(
                {
                    **shared,
                    "doc_id": jira_id,
                    "doc_type": "jira",
                    "title": str(facts.get("title", facts.get("root_cause", jira_id)))[
                        :512
                    ],
                    "body": self._jira_body(facts),
                }
            )

        conf_id = artifact_ids.get("confluence", "") or next(
            (
                v
                for v in artifact_ids.values()
                if isinstance(v, str) and str(v).startswith("CONF-")
            ),
            "",
        )
        if conf_id:
            body = facts.get("content", facts.get("summary", "")) or evt.get(
                "summary", ""
            )
            rows.append(
                {
                    **shared,
                    "doc_id": conf_id,
                    "doc_type": "confluence",
                    "title": str(facts.get("title", conf_id))[:512],
                    "body": body,
                    "dept": dept_val or _dept_from_artifact_id(conf_id),
                }
            )

        email_id = artifact_ids.get("email", "")
        if email_id or event_type in (
            "inbound_external_email",
            "hr_outbound_email",
            "customer_email_routed",
            "vendor_email_routed",
            "email_dropped",
            "sales_outbound_email",
            "proactive_outreach_initiated",
        ):
            rows.append(
                {
                    **shared,
                    "doc_id": email_id or f"EMAIL-{evt.get('day', 0)}-{id(evt)}",
                    "doc_type": "email",
                    "title": str(facts.get("subject", facts.get("summary", email_id)))[
                        :512
                    ],
                    "body": self._email_body(facts, evt),
                }
            )

        slack_id = artifact_ids.get("slack_thread", "")
        if slack_id:
            channel = facts.get("channel", "#general")
            rows.append(
                {
                    **shared,
                    "doc_id": slack_id,
                    "doc_type": "slack",
                    "title": str(channel + ": " + facts.get("summary", "")[:80])[:512],
                    "body": facts.get("content", facts.get("summary", "")),
                }
            )

        pr_id = artifact_ids.get("pr", "")
        if pr_id:
            rows.append(
                {
                    **shared,
                    "doc_id": pr_id,
                    "doc_type": "pr",
                    "title": str(facts.get("title", pr_id))[:512],
                    "body": facts.get("description", facts.get("summary", "")),
                }
            )

        zd_ids: List[str] = []
        single_zd = artifact_ids.get("zd_ticket", "")
        if single_zd:
            zd_ids = [single_zd]
        else:
            multi_zd = artifact_ids.get("zd_tickets", [])
            if isinstance(multi_zd, list):
                zd_ids = [str(z) for z in multi_zd if z]
        for zd_id in zd_ids:
            rows.append(
                {
                    **shared,
                    "doc_id": zd_id,
                    "doc_type": "zd_ticket",
                    "title": str(facts.get("subject", facts.get("ticket_id", zd_id)))[
                        :512
                    ],
                    "body": self._zd_body(facts),
                }
            )

        sf_opp_ids: List[str] = []
        single_opp = artifact_ids.get("sf_opp", "")
        if single_opp:
            sf_opp_ids = [single_opp]
        else:
            multi_opp = artifact_ids.get("sf_opps", [])
            if isinstance(multi_opp, list):
                sf_opp_ids = [str(o) for o in multi_opp if o]
        for opp_id in sf_opp_ids:
            rows.append(
                {
                    **shared,
                    "doc_id": opp_id,
                    "doc_type": "sf_opp",
                    "title": str(
                        facts.get("account_name", opp_id)
                        + " — "
                        + facts.get("stage", "")
                    )[:512],
                    "body": self._sf_opp_body(facts),
                }
            )

        sf_acc_ids: List[str] = []
        multi_acc = artifact_ids.get("sf_accounts", [])
        if isinstance(multi_acc, list):
            sf_acc_ids = [str(a) for a in multi_acc if a]
        for acc_id in sf_acc_ids:
            rows.append(
                {
                    **shared,
                    "doc_id": acc_id,
                    "doc_type": "sf_account",
                    "title": str(facts.get("account_name", acc_id))[:512],
                    "body": self._sf_account_body(facts, acc_id),
                }
            )

        if not rows:
            rows.append(
                {
                    **shared,
                    "doc_id": f"EVENT-{evt.get('day', 0)}-{event_type}",
                    "doc_type": "sim_event",
                    "title": event_type.replace("_", " ").title(),
                    "body": evt.get("summary", ""),
                }
            )

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
        for key in (
            "subject",
            "content",
            "body",
            "from",
            "to",
            "summary",
            "source",
            "prospect",
        ):
            val = facts.get(key)
            if val:
                parts.append(f"{key}: {val}")
        if not parts:
            parts.append(evt.get("summary", ""))
        return "\n".join(parts)

    def _zd_body(self, facts: dict) -> str:
        parts = []
        for key in ("subject", "org_name", "description", "ticket_id", "component"):
            val = facts.get(key)
            if val:
                parts.append(f"{key}: {val}")
        ticket_ids = facts.get("ticket_ids", [])
        if ticket_ids:
            parts.append("ticket_ids: " + ", ".join(str(t) for t in ticket_ids))
        incident_id = facts.get("incident_id", "")
        if incident_id:
            parts.append(f"related_incident: {incident_id}")
        return "\n".join(parts)

    def _sf_opp_body(self, facts: dict) -> str:
        parts = []
        for key in (
            "account_name",
            "stage",
            "sender",
            "subject",
            "risk_note",
            "opportunity_id",
        ):
            val = facts.get(key)
            if val:
                parts.append(f"{key}: {val}")
        opp_ids = facts.get("opp_ids", [])
        if opp_ids:
            parts.append("opp_ids: " + ", ".join(str(o) for o in opp_ids))
        incident_id = facts.get("incident_id", "")
        if incident_id:
            parts.append(f"related_incident: {incident_id}")
        return "\n".join(parts)

    def _sf_account_body(self, facts: dict, acc_id: str) -> str:
        parts = []
        for key in ("departed_employee", "role", "account_name"):
            val = facts.get(key)
            if val:
                parts.append(f"{key}: {val}")
        accounts_lapsed = facts.get("accounts_lapsed", [])
        if acc_id in accounts_lapsed:
            parts.append(f"account_id: {acc_id}")
            parts.append("status: ownership lapsed — pending reassignment")
        opps_lapsed = facts.get("opportunities_lapsed", [])
        if opps_lapsed:
            parts.append(
                "opportunities_lapsed: " + ", ".join(str(o) for o in opps_lapsed)
            )
        return "\n".join(parts)

    def _plans_to_corpus_rows(self) -> List[dict]:
        rows = []
        for plan in self._mem._db["dept_plans"].find({}, {"_id": 0}):
            dept = plan["dept"]
            day = plan["day"]
            lead = plan.get("lead", "")
            theme = plan.get("theme", "")
            plan_id = f"PLAN-{day}-{dept}"

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

    def _post_sim_to_corpus_rows(self) -> List[dict]:
        rows = []
        nps_dir = BASE / "nps" / "responses"
        if nps_dir.exists():
            for p in nps_dir.glob("*.json"):
                data = json.loads(p.read_text())
                rows.append(
                    {
                        "doc_id": data["response_id"],
                        "doc_type": "nps_survey",
                        "title": f"NPS Survey: {data['org_name']}",
                        "body": f"Score: {data['score']}\nComment: {data.get('verbatim_comment', '')}",
                        "day": 30,
                        "date": data["submitted_at"][:10],
                        "timestamp": data["submitted_at"],
                        "actors": json.dumps([data["org_name"]]),
                        "tags": json.dumps(["nps", data["classification"]]),
                        "artifact_ids": json.dumps({"nps": data["response_id"]}),
                        "dept": "Sales_Marketing",
                        "is_incident": data["score"] < 7,
                        "is_external": True,
                    }
                )

        inv_dir = BASE / "invoices"
        if inv_dir.exists():
            for p in inv_dir.glob("*.json"):
                data = json.loads(p.read_text())
                body_text = (
                    data.get("notes", "")
                    + "\n"
                    + json.dumps(data.get("line_items", []))
                )
                rows.append(
                    {
                        "doc_id": data["invoice_id"],
                        "doc_type": "invoice",
                        "title": f"Invoice {data['invoice_id']} - {data['customer']['org_name']}",
                        "body": body_text,
                        "day": 30,
                        "date": data["invoice_date"][:10],
                        "timestamp": data["invoice_date"],
                        "actors": json.dumps([data["customer"]["org_name"]]),
                        "tags": json.dumps(["invoice", "billing"]),
                        "artifact_ids": json.dumps({"invoice": data["invoice_id"]}),
                        "dept": "Finance",
                        "is_incident": data.get("metadata", {}).get(
                            "sla_credits_count", 0
                        )
                        > 0,
                        "is_external": True,
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

            conf_id_map: Dict[str, str] = {}
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

            comment_map: Dict[str, List[str]] = defaultdict(list)
            for comment in self._mem._db["artifacts"].find(
                {"type": "jira_comment"},
                {"_id": 0, "parent_id": 1, "body": 1, "author": 1},
            ):
                parent = comment.get("parent_id", "")
                cbody = comment.get("body", "")
                cauthor = comment.get("author", "")
                if parent and cbody:
                    comment_map[parent].append(
                        f"comment ({cauthor}): {cbody}"
                        if cauthor
                        else f"comment: {cbody}"
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

            for pr in self._mem._db["pull_requests"].find(
                {},
                {
                    "_id": 0,
                    "pr_id": 1,
                    "title": 1,
                    "description": 1,
                    "author": 1,
                    "ticket_id": 1,
                    "reviewers": 1,
                    "status": 1,
                    "comments": 1,
                },
            ):
                pid = pr.get("pr_id")
                if not pid:
                    continue
                parts = []
                if pr.get("title"):
                    parts.append(f"title: {pr['title']}")
                if pr.get("description"):
                    parts.append(f"description: {pr['description']}")
                if pr.get("author"):
                    parts.append(f"author: {pr['author']}")
                if pr.get("ticket_id"):
                    parts.append(f"ticket: {pr['ticket_id']}")
                if pr.get("status"):
                    parts.append(f"status: {pr['status']}")
                reviewers = pr.get("reviewers", [])
                if reviewers:
                    parts.append(f"reviewers: {', '.join(reviewers)}")
                for c in pr.get("comments") or []:
                    verdict = f" [{c['verdict']}]" if c.get("verdict") else ""
                    text = c.get("text", "")
                    author = c.get("author", "")
                    if text:
                        parts.append(f"review ({author}{verdict}): {text}")
                rich_map[pid] = "\n".join(p for p in parts if p)

            for email in self._mem._db["emails"].find(
                {},
                {
                    "_id": 0,
                    "embed_id": 1,
                    "subject": 1,
                    "body": 1,
                    "from_name": 1,
                    "from_addr": 1,
                    "to_name": 1,
                    "to_addr": 1,
                    "direction": 1,
                },
            ):
                eid = email.get("embed_id")
                if not eid:
                    continue
                parts = []
                if email.get("subject"):
                    parts.append(f"subject: {email['subject']}")
                if email.get("from_name") or email.get("from_addr"):
                    parts.append(
                        f"from: {email.get('from_name', '')} <{email.get('from_addr', '')}>"
                    )
                if email.get("to_name") or email.get("to_addr"):
                    parts.append(
                        f"to: {email.get('to_name', '')} <{email.get('to_addr', '')}>"
                    )
                if email.get("body"):
                    parts.append(email["body"])
                rich_map[eid] = "\n".join(parts)

            for ticket in self._mem._db["zd_tickets"].find(
                {},
                {
                    "_id": 0,
                    "ticket_id": 1,
                    "subject": 1,
                    "org_name": 1,
                    "description": 1,
                    "comments": 1,
                    "related_incident": 1,
                    "priority": 1,
                    "status": 1,
                },
            ):
                tid = ticket.get("ticket_id")
                if not tid:
                    continue
                parts = [
                    f"subject: {ticket.get('subject', '')}",
                    f"org: {ticket.get('org_name', '')}",
                    f"status: {ticket.get('status', '')}",
                    f"priority: {ticket.get('priority', '')}",
                ]
                if ticket.get("description"):
                    parts.append(f"description: {ticket['description']}")
                if ticket.get("related_incident"):
                    parts.append(f"related_incident: {ticket['related_incident']}")
                for c in ticket.get("comments") or []:
                    author = c.get("author", "")
                    text = c.get("text", "")
                    if text:
                        parts.append(
                            f"comment ({author}): {text}"
                            if author
                            else f"comment: {text}"
                        )
                rich_map[tid] = "\n".join(p for p in parts if p)

            for opp in self._mem._db["sf_opps"].find(
                {},
                {
                    "_id": 0,
                    "opportunity_id": 1,
                    "account_name": 1,
                    "stage": 1,
                    "probability": 1,
                    "amount": 1,
                    "owner": 1,
                    "lead_source": 1,
                    "next_step": 1,
                    "risk_notes": 1,
                    "touchpoints": 1,
                    "close_date": 1,
                },
            ):
                oid = opp.get("opportunity_id")
                if not oid:
                    continue
                parts = [
                    f"account: {opp.get('account_name', '')}",
                    f"stage: {opp.get('stage', '')}",
                    f"probability: {opp.get('probability', '')}%",
                    f"amount: ${opp.get('amount', 0):,}",
                    f"owner: {opp.get('owner', '')}",
                    f"close_date: {opp.get('close_date', '')}",
                    f"lead_source: {opp.get('lead_source', '')}",
                    f"next_step: {opp.get('next_step', '')}",
                ]
                for note in opp.get("risk_notes") or []:
                    parts.append(f"risk: {note}")
                for tp in opp.get("touchpoints") or []:
                    subject = tp.get("subject", "")
                    sender = tp.get("sender", "")
                    ts = tp.get("timestamp", "")
                    if subject:
                        parts.append(f"touchpoint ({sender}, {ts}): {subject}")
                rich_map[oid] = "\n".join(p for p in parts if p)

            for acc in self._mem._db["sf_accounts"].find(
                {},
                {
                    "_id": 0,
                    "account_id": 1,
                    "name": 1,
                    "primary_contact": 1,
                    "type": 1,
                    "industry": 1,
                    "tier": 1,
                    "billing_region": 1,
                    "arr": 1,
                    "owner": 1,
                    "risk_flag": 1,
                },
            ):
                aid = acc.get("account_id")
                if not aid:
                    continue
                parts = [
                    f"name: {acc.get('name', '')}",
                    f"type: {acc.get('type', '')}",
                    f"tier: {acc.get('tier', '')}",
                    f"industry: {acc.get('industry', '')}",
                    f"billing_region: {acc.get('billing_region', '')}",
                    f"arr: ${acc.get('arr', 0):,}",
                    f"owner: {acc.get('owner', '')}",
                    f"primary_contact: {acc.get('primary_contact', '')}",
                ]
                if acc.get("risk_flag"):
                    parts.append("risk_flag: true — ownership lapsed or at-risk")
                rich_map[aid] = "\n".join(p for p in parts if p)

            for row in rows:
                if row["doc_id"] == "CONF-UNKNOWN" and row["doc_type"] == "confluence":
                    body_snippet = (row.get("body") or "")[:120].strip()
                    title_key = (row.get("title") or "").strip()
                    resolved_id = conf_id_map.get(body_snippet) or conf_id_map.get(
                        title_key
                    )
                    if resolved_id:
                        row["doc_id"] = resolved_id
                        row["title"] = resolved_id
                        row["body"] = rich_map[resolved_id]
                        if not row.get("dept"):
                            row["dept"] = _dept_from_artifact_id(resolved_id)
                    else:
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

            existing_ids = {row["doc_id"] for row in rows}

            def _make_orphan_row(
                doc_id,
                doc_type,
                title,
                body,
                day,
                date,
                timestamp,
                actors,
                tags,
                artifact_type,
                is_incident=False,
                is_external=False,
            ):
                dept = _dept_from_artifact_id(doc_id) or next(
                    (
                        _ACTOR_TO_DEPT.get(str(a), "")
                        for a in actors
                        if _ACTOR_TO_DEPT.get(str(a))
                    ),
                    "",
                )
                return {
                    "doc_id": doc_id,
                    "doc_type": doc_type,
                    "title": str(title)[:512],
                    "body": str(body),
                    "day": int(day or 0),
                    "date": str(date or ""),
                    "timestamp": str(timestamp or ""),
                    "actors": json.dumps(actors),
                    "tags": json.dumps(tags),
                    "artifact_ids": json.dumps({artifact_type: doc_id}),
                    "dept": dept,
                    "is_incident": is_incident,
                    "is_external": is_external,
                }

            _ARTIFACT_TYPE_MAP = {
                "confluence": "confluence",
                "slack_thread": "slack",
                "jira": "jira",
                "zd_ticket": "zd_ticket",
                "sf_opportunity": "sf_opp",
            }
            for artifact in self._mem._db["artifacts"].find(
                {"type": {"$in": list(_ARTIFACT_TYPE_MAP.keys())}},
                {
                    "_id": 1,
                    "type": 1,
                    "content": 1,
                    "title": 1,
                    "day": 1,
                    "date": 1,
                    "timestamp": 1,
                    "metadata": 1,
                    "author": 1,
                    "actors": 1,
                },
            ):
                art_id = str(artifact.get("_id", ""))
                art_type = artifact.get("type", "")
                if not art_id or art_id in existing_ids:
                    continue
                if any(
                    art_id.startswith(p)
                    for p in ("exfil_", "hoarding_", "snooping_", "dlp_")
                ):
                    continue
                meta = artifact.get("metadata", {})
                author = artifact.get("author") or meta.get("author", "")
                actors = artifact.get("actors") or ([author] if author else [])
                tags = meta.get("tags", [art_type])
                body = rich_map.get(art_id) or artifact.get("content") or ""
                title = artifact.get("title") or art_id
                rows.append(
                    _make_orphan_row(
                        doc_id=art_id,
                        doc_type=_ARTIFACT_TYPE_MAP.get(art_type, "sim_event"),
                        title=title,
                        body=body,
                        day=artifact.get("day"),
                        date=artifact.get("date"),
                        timestamp=artifact.get("timestamp"),
                        actors=actors,
                        tags=tags,
                        artifact_type=art_type,
                        is_incident=any(
                            t in tags
                            for t in (
                                "postmortem",
                                "incident",
                                "escalation",
                                "zd_escalated",
                            )
                        ),
                        is_external=art_type in ("zd_ticket", "sf_opportunity"),
                    )
                )
                existing_ids.add(art_id)

            for pr in self._mem._db["pull_requests"].find(
                {},
                {
                    "_id": 0,
                    "pr_id": 1,
                    "title": 1,
                    "author": 1,
                    "day": 1,
                    "date": 1,
                    "timestamp": 1,
                    "dept": 1,
                },
            ):
                pid = pr.get("pr_id", "")
                if not pid or pid in existing_ids:
                    continue
                body = rich_map.get(pid, "")
                rows.append(
                    _make_orphan_row(
                        doc_id=pid,
                        doc_type="pr",
                        title=pr.get("title", pid),
                        body=body,
                        day=pr.get("day"),
                        date=pr.get("date"),
                        timestamp=pr.get("timestamp"),
                        actors=[pr["author"]] if pr.get("author") else [],
                        tags=["pr"],
                        artifact_type="pr",
                    )
                )
                existing_ids.add(pid)

            for email in self._mem._db["emails"].find(
                {},
                {
                    "_id": 0,
                    "embed_id": 1,
                    "subject": 1,
                    "from_name": 1,
                    "from_addr": 1,
                    "direction": 1,
                    "day": 1,
                    "date": 1,
                    "timestamp": 1,
                },
            ):
                eid = email.get("embed_id", "")
                if not eid or eid in existing_ids:
                    continue
                body = rich_map.get(eid, "")
                direction = email.get("direction", "")
                rows.append(
                    _make_orphan_row(
                        doc_id=eid,
                        doc_type="email",
                        title=email.get("subject", eid),
                        body=body,
                        day=email.get("day"),
                        date=email.get("date"),
                        timestamp=email.get("timestamp"),
                        actors=[email["from_name"]] if email.get("from_name") else [],
                        tags=["email", direction] if direction else ["email"],
                        artifact_type="email",
                        is_external=True,
                    )
                )
                existing_ids.add(eid)

            for ticket in self._mem._db["zd_tickets"].find(
                {},
                {
                    "_id": 0,
                    "ticket_id": 1,
                    "subject": 1,
                    "org_name": 1,
                    "day": 1,
                    "date": 1,
                    "created_at": 1,
                    "status": 1,
                    "priority": 1,
                },
            ):
                tid = ticket.get("ticket_id", "")
                if not tid or tid in existing_ids:
                    continue
                body = rich_map.get(tid, "")
                rows.append(
                    _make_orphan_row(
                        doc_id=tid,
                        doc_type="zd_ticket",
                        title=ticket.get("subject", tid),
                        body=body,
                        day=ticket.get("day"),
                        date=ticket.get("date"),
                        timestamp=ticket.get("created_at"),
                        actors=[],
                        tags=["zendesk", "support"],
                        artifact_type="zd_ticket",
                        is_external=True,
                        is_incident=ticket.get("priority") == "Urgent",
                    )
                )
                existing_ids.add(tid)

            for opp in self._mem._db["sf_opps"].find(
                {},
                {
                    "_id": 0,
                    "opportunity_id": 1,
                    "account_name": 1,
                    "stage": 1,
                    "owner": 1,
                    "day": 1,
                    "date": 1,
                    "created_at": 1,
                },
            ):
                oid = opp.get("opportunity_id", "")
                if not oid or oid in existing_ids:
                    continue
                body = rich_map.get(oid, "")
                title = (
                    f"{opp.get('account_name', oid)} — {opp.get('stage', '')}"
                ).strip(" —")
                rows.append(
                    _make_orphan_row(
                        doc_id=oid,
                        doc_type="sf_opp",
                        title=title,
                        body=body,
                        day=opp.get("day"),
                        date=opp.get("date"),
                        timestamp=opp.get("created_at"),
                        actors=[opp["owner"]] if opp.get("owner") else [],
                        tags=["salesforce", "opportunity"],
                        artifact_type="sf_opp",
                        is_external=True,
                    )
                )
                existing_ids.add(oid)

            for acc in self._mem._db["sf_accounts"].find(
                {},
                {
                    "_id": 0,
                    "account_id": 1,
                    "name": 1,
                    "owner": 1,
                    "created_at": 1,
                },
            ):
                aid = acc.get("account_id", "")
                if not aid or aid in existing_ids:
                    continue
                body = rich_map.get(aid, "")
                rows.append(
                    _make_orphan_row(
                        doc_id=aid,
                        doc_type="sf_account",
                        title=acc.get("name", aid),
                        body=body,
                        day=None,
                        date=None,
                        timestamp=acc.get("created_at"),
                        actors=[acc["owner"]] if acc.get("owner") else [],
                        tags=["salesforce", "account"],
                        artifact_type="sf_account",
                        is_external=True,
                    )
                )
                existing_ids.add(aid)

            thread_buckets: Dict[str, dict] = {}
            for msg in self._mem._db["slack_messages"].find(
                {},
                {
                    "_id": 0,
                    "thread_id": 1,
                    "channel": 1,
                    "text": 1,
                    "author": 1,
                    "sender": 1,
                    "ts": 1,
                    "day": 1,
                    "date": 1,
                },
            ):
                tid = msg.get("thread_id", "")
                if not tid or tid in existing_ids:
                    continue
                if tid not in thread_buckets:
                    thread_buckets[tid] = {
                        "channel": msg.get("channel", ""),
                        "day": msg.get("day"),
                        "date": msg.get("date"),
                        "ts": msg.get("ts", ""),
                        "actors": set(),
                        "texts": [],
                    }
                bucket = thread_buckets[tid]
                author = msg.get("author") or msg.get("sender", "")
                if author:
                    bucket["actors"].add(author)
                text = msg.get("text", "")
                if text:
                    prefix = f"{author}: " if author else ""
                    bucket["texts"].append(f"{prefix}{text}")

            for tid, bucket in thread_buckets.items():
                if tid in existing_ids:
                    continue
                actors = sorted(bucket["actors"])
                channel = bucket["channel"]
                body = "\n".join(bucket["texts"])
                rows.append(
                    _make_orphan_row(
                        doc_id=tid,
                        doc_type="slack",
                        title=f"#{channel}" if channel else tid,
                        body=body,
                        day=bucket["day"],
                        date=bucket["date"],
                        timestamp=bucket["ts"],
                        actors=actors,
                        tags=["slack", channel] if channel else ["slack"],
                        artifact_type="slack_thread",
                    )
                )
                existing_ids.add(tid)

        except Exception as exc:
            logger.debug(f"MongoDB enrichment skipped: {exc}")
        return rows


# ─────────────────────────────────────────────────────────────────────────────
# EVAL INDEX SERIALISERS
# ─────────────────────────────────────────────────────────────────────────────


def _causal_links_to_rows(links: List[dict]) -> List[dict]:
    """
    Flatten causal_link_index.json into Parquet-ready rows.
    Each CausalLink dict maps directly — sets become JSON strings.
    """
    rows = []
    for lnk in links:
        rows.append(
            {
                "link_type": lnk.get("link_type", ""),
                "cause_event_id": lnk.get("cause_event_id", ""),
                "cause_event_type": lnk.get("cause_event_type", ""),
                "effect_event_id": lnk.get("effect_event_id", ""),
                "effect_event_type": lnk.get("effect_event_type", ""),
                "actors": json.dumps(lnk.get("actors", []), default=str),
                "day": int(lnk.get("day", 0)),
                "link_field": lnk.get("link_field", ""),
                "link_value": str(lnk.get("link_value", "")),
                "subsystems_involved": json.dumps(
                    sorted(lnk.get("subsystems_involved", [])), default=str
                ),
                "counterfactual_premise": lnk.get("counterfactual_premise", ""),
                "counterfactual_outcome": lnk.get("counterfactual_outcome", ""),
                "outcome_changed": bool(lnk.get("outcome_changed", True)),
            }
        )
    return rows


def _actor_visibility_to_rows(visibility_map: dict) -> List[dict]:
    """
    Flatten actor_visibility.json (actor → [cone, ...]) into one row per
    (actor, day) snapshot. Heavy set/dict fields are JSON-serialised.
    """
    rows = []
    for actor, cones in visibility_map.items():
        for cone in cones:
            vis = cone.get("visible_artifacts", {})
            rows.append(
                {
                    "actor": cone.get("actor", actor),
                    "role": cone.get("role", ""),
                    "as_of_time": cone.get("as_of_time", ""),
                    "as_of_day": int(cone.get("as_of_day", 0)),
                    "subsystem_access": json.dumps(
                        sorted(cone.get("subsystem_access", [])), default=str
                    ),
                    # All visible artifact IDs, flattened across subsystems
                    "all_visible_artifacts": json.dumps(
                        sorted(
                            {aid for ids in vis.values() for aid in ids}
                        ),
                        default=str,
                    ),
                    # Per-subsystem breakdown kept for fine-grained analysis
                    "visible_artifacts_by_subsystem": json.dumps(
                        {k: sorted(v) for k, v in vis.items()}, default=str
                    ),
                    "directly_involved": json.dumps(
                        sorted(cone.get("directly_involved", [])), default=str
                    ),
                    "broadcast_visible": json.dumps(
                        sorted(cone.get("broadcast_visible", [])), default=str
                    ),
                }
            )
    return rows


def _absence_catalog_to_rows(records: List[dict]) -> List[dict]:
    """Flatten absence_catalog.json into Parquet-ready rows."""
    rows = []
    for rec in records:
        rows.append(
            {
                "trigger_event_id": rec.get("trigger_event_id", ""),
                "trigger_event_type": rec.get("trigger_event_type", ""),
                "expected_response_type": rec.get("expected_response_type", ""),
                "trigger_day": int(rec.get("trigger_day", 0)),
                "trigger_actors": json.dumps(
                    rec.get("trigger_actors", []), default=str
                ),
                "trigger_artifact_ids": json.dumps(
                    rec.get("trigger_artifact_ids", {}), default=str
                ),
                "link_field": rec.get("link_field", ""),
                "link_value": str(rec.get("link_value", "")),
                "subsystem": rec.get("subsystem", ""),
                "expected_search_space": json.dumps(
                    rec.get("expected_search_space", []), default=str
                ),
            }
        )
    return rows



def _questions_to_rows(questions: List[dict]) -> List[dict]:
    """
    Convert the v2 eval questions list into flat Parquet rows.

    Evidence chain derivation:
      COUNTERFACTUAL — union of cause + effect artifact IDs from
                       ground_truth.evidence_chain_artifacts
      PERSPECTIVE    — ground_truth.evidence_artifacts
      SILENCE        — empty; the correct answer is absence, so retrieval
                       recall is not applicable
    """
    rows = []
    for q in questions:
        qtype = q.get("question_type", "")
        gt = q.get("ground_truth", {})

        evidence: List[str] = []
        if qtype == "COUNTERFACTUAL":
            chain = gt.get("evidence_chain_artifacts", {})
            evidence = list(set(chain.get("cause", []) + chain.get("effect", [])))
        elif qtype == "PERSPECTIVE":
            evidence = gt.get("evidence_artifacts", [])


        rows.append(
            {
                # ── Core fields (all types) ───────────────────────────────────
                "question_id": q.get("question_id", ""),
                "question_type": qtype,
                "question_text": q.get("question_text", ""),
                "ground_truth": json.dumps(gt, default=str),
                "evidence_chain": json.dumps(evidence, default=str),
                "difficulty": q.get("difficulty", ""),
                "requires_reasoning": bool(q.get("requires_reasoning", False)),
                # ── PERSPECTIVE-specific fields ───────────────────────────────
                "actor": q.get("actor", ""),
                "actor_role": q.get("actor_role", ""),
                "as_of_day": int(q.get("as_of_day", 0)),
                "subsystem_access": json.dumps(
                    q.get("subsystem_access", []), default=str
                ),
                "blocked_subsystems": json.dumps(
                    q.get("blocked_subsystems", []), default=str
                ),
                "actor_visible_artifacts": json.dumps(
                    q.get("actor_visible_artifacts", []), default=str
                ),
                # ── COUNTERFACTUAL-specific fields ────────────────────────────
                "link_type": q.get("link_type", ""),
                "causal_day": int(q.get("day", 0)),
                # ── SILENCE-specific fields ───────────────────────────────────
                "expected_search_space": json.dumps(
                    q.get("expected_search_space", []), default=str
                ),
                "trigger_event_type": q.get("trigger_event_type", ""),
                "expected_response_type": q.get("expected_response_type", ""),
            }
        )
    return rows



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


class UngatedCeilingBaseline:
    """
    Tier 1 baseline: BM25 and dense retrieval with ALL gates removed.

    No visibility cones, no temporal horizons, no subsystem constraints —
    the retriever has "god-mode" access to the full corpus. MRR@10 and
    Recall@10 represent the information ceiling, not agent performance.

    SILENCE questions are excluded because the correct answer is absence;
    retrieval recall is not applicable.

    The delta between these ceiling scores and a gated agent's combined_score
    is the "Epistemic Tax" — the difficulty cost of respecting organisational
    silos and actor knowledge horizons.
    """

    def __init__(self, corpus: List[dict], questions: List[dict], mem=None):
        self._corpus = corpus
        self._questions = questions
        self._mem = mem
        self._doc_ids = [row["doc_id"] for row in corpus]
        self._bodies = [row.get("body") or row.get("content") or "" for row in corpus]

        if _BM25_AVAILABLE:
            tokenised = [_tokenize(b) for b in self._bodies]
            self._bm25 = BM25Okapi(tokenised)
        else:
            self._bm25 = None

        if _DENSE_AVAILABLE and mem is not None:
            logger.info("  Embedding corpus for dense ceiling baseline...")
            embeddings = []
            for i, body in enumerate(self._bodies):
                text_to_embed = (
                    body.strip() if body and body.strip() else "empty document"
                )
                vec = self._mem._embed(text_to_embed, input_type="search_document")
                embeddings.append(vec)
                if (i + 1) % 500 == 0:
                    logger.info(f"  embedded {i + 1}/{len(self._bodies)} docs...")
            mat = np.array(embeddings, dtype=np.float32)
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            self._dense_matrix = mat / np.where(norms == 0, 1, norms)
        else:
            self._dense_matrix = None

    # ── PUBLIC ────────────────────────────────────────────────────────────────

    def run_bm25(self) -> Tuple[List[dict], Dict[str, Any]]:
        if self._bm25 is None:
            return [], {"error": "rank_bm25 not installed"}
        return self._run_retrieval(use_dense=False)

    def run_dense(self) -> Tuple[List[dict], Dict[str, Any]]:
        if self._mem is None or self._dense_matrix is None:
            return [], {"error": "Memory unavailable — dense ceiling requires MongoDB"}
        return self._run_retrieval(use_dense=True)

    # ── PRIVATE ───────────────────────────────────────────────────────────────

    def _evidence_for_question(self, q: dict) -> List[str]:
        """
        Flat list of corpus doc_ids that constitute the correct answer.
        SILENCE returns empty — absence has no retrievable target.
        """
        qtype = q.get("question_type", "")
        gt = q.get("ground_truth", {})

        if qtype == "COUNTERFACTUAL":
            chain = gt.get("evidence_chain_artifacts", {})
            return list(set(chain.get("cause", []) + chain.get("effect", [])))

        if qtype == "PERSPECTIVE":
            return gt.get("evidence_artifacts", [])

        return [] 

    def _rank(self, query: str, use_dense: bool, top_k: int = 10) -> List[str]:
        if use_dense and self._dense_matrix is not None:
            q_vec = np.array(
                self._mem._embed(query, input_type="search_query"), dtype=np.float32
            )
            q_vec /= max(np.linalg.norm(q_vec), 1e-9)
            scores = self._dense_matrix @ q_vec
            indices = scores.argsort()[::-1][:top_k]
            return [self._doc_ids[int(i)] for i in indices]

        elif not use_dense and self._bm25 is not None:
            scores = self._bm25.get_scores(_tokenize(query))
            indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            return [self._doc_ids[i] for i in indices[:top_k]]

        return []

    def _run_retrieval(self, use_dense: bool) -> Tuple[List[dict], Dict[str, Any]]:
        per_question: List[dict] = []
        by_type: Dict[str, List[Tuple[float, float]]] = defaultdict(list)

        for q in self._questions:
            qtype = q.get("question_type", "")
            evidence = self._evidence_for_question(q)
            if not evidence:
                continue

            ranked_ids = self._rank(q.get("question_text", ""), use_dense=use_dense)
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


class StaticReasoningMetrics:
    """
    Tier 2 baseline: reasoning difficulty metrics derived from corpus +
    question metadata alone. No LLM calls, no agent execution required.

    These metrics answer "why is each track hard?" before any agent runs,
    exposing the epistemic structure that makes naive retrieval insufficient.

    PERSPECTIVE    → horizon_contamination_rate
        Fraction of ungated BM25 top-20 that falls outside the actor's
        visibility cone. High value = epistemic discipline is load-bearing;
        retrieval alone surfaces mostly forbidden documents.

    COUNTERFACTUAL → causal_chain_traceable
        Whether cause AND effect artifacts both appear in ungated top-10.
        False = a single retrieval pass cannot close the causal chain;
        multi-hop reasoning is required.

    SILENCE        → search_space_bm25_coverage
        Fraction of expected_search_space locations surfaced by BM25 on
        the question text. Low value = the agent must enumerate absence-check
        locations deliberately; naive search will miss them.
    """

    def __init__(
        self,
        questions: List[dict],
        bm25,           # BM25Okapi instance reused from UngatedCeilingBaseline
        doc_ids: List[str],
    ):
        self._questions = questions
        self._bm25 = bm25
        self._doc_ids = doc_ids

    def compute(self) -> dict:
        per_question: List[dict] = []
        by_type: Dict[str, List[dict]] = defaultdict(list)

        dispatch = {
            "PERSPECTIVE": self._perspective_metrics,
            "COUNTERFACTUAL": self._counterfactual_metrics,
            "SILENCE": self._silence_metrics,
        }

        for q in self._questions:
            qtype = q.get("question_type", "")
            fn = dispatch.get(qtype)
            if fn is None:
                continue
            row = {
                "question_id": q.get("question_id"),
                "question_type": qtype,
                **fn(q),
            }
            per_question.append(row)
            by_type[qtype].append(row)

        return {
            "per_question": per_question,
            "aggregate": self._aggregate(by_type),
        }

    # ── Track-specific metric computations ───────────────────────────────────

    def _perspective_metrics(self, q: dict) -> dict:
        """
        Horizon contamination: what fraction of ungated top-20 results would
        an actor NOT be permitted to see? High contamination means a naive
        retriever is actively counter-productive for PERSPECTIVE questions.
        """
        visible = set(q.get("actor_visible_artifacts", []))
        ranked = self._rank_bm25(q["question_text"], k=20)
        if not ranked or not visible:
            return {
                "horizon_contamination_rate": None,
                "first_in_cone_rank": None,
                "in_cone_count_top20": None,
            }

        out_of_cone = [r for r in ranked if r not in visible]
        first_in_cone = next(
            (i + 1 for i, r in enumerate(ranked) if r in visible), None
        )
        return {
            "horizon_contamination_rate": round(len(out_of_cone) / len(ranked), 4),
            "first_in_cone_rank": first_in_cone,
            "in_cone_count_top20": len(ranked) - len(out_of_cone),
        }

    def _counterfactual_metrics(self, q: dict) -> dict:
        """
        Causal chain traceability: do both cause and effect artifacts appear
        in ungated top-10? If not, the agent must do multi-hop retrieval.
        """
        chain = q.get("ground_truth", {}).get("evidence_chain_artifacts", {})
        cause_ids = set(chain.get("cause", []))
        effect_ids = set(chain.get("effect", []))
        if not cause_ids and not effect_ids:
            return {"causal_chain_traceable": None}

        ranked = self._rank_bm25(q["question_text"], k=10)
        ranked_set = set(ranked)
        cause_found = bool(cause_ids & ranked_set)
        effect_found = bool(effect_ids & ranked_set)

        cause_rank = next(
            (i + 1 for i, r in enumerate(ranked) if r in cause_ids), None
        )
        effect_rank = next(
            (i + 1 for i, r in enumerate(ranked) if r in effect_ids), None
        )

        return {
            "cause_found_top10": cause_found,
            "effect_found_top10": effect_found,
            "causal_chain_traceable": cause_found and effect_found,
            "cause_rank": cause_rank,
            "effect_rank": effect_rank,
        }

    def _silence_metrics(self, q: dict) -> dict:
        """
        Search space BM25 coverage: what fraction of the required absence-check
        locations does a naive BM25 search surface? Low coverage means the agent
        must enumerate expected_search_space explicitly rather than relying on
        retrieval to guide it to the right places to look.
        """
        expected = q.get("expected_search_space", [])
        if not expected:
            return {"search_space_bm25_coverage": 1.0, "uncovered_locations": []}

        ranked = self._rank_bm25(q["question_text"], k=20)

        def _norm(s: str) -> str:
            # Normalise path-style entries to their terminal component:
            # "confluence/postmortems/IT-108" → "it-108"
            return s.strip("/").split("/")[-1].lower()

        norm_expected = {_norm(e): e for e in expected}
        norm_ranked = {_norm(r) for r in ranked}
        ranked_lower = [r.lower() for r in ranked]

        covered = {
            original
            for norm_term, original in norm_expected.items()
            if norm_term in norm_ranked
            or any(norm_term in r for r in ranked_lower)
        }

        return {
            "search_space_bm25_coverage": round(len(covered) / len(expected), 4),
            "uncovered_locations": sorted(set(expected) - covered),
        }

    # ── Shared utilities ──────────────────────────────────────────────────────

    def _rank_bm25(self, query: str, k: int) -> List[str]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self._doc_ids[i] for i in indices[:k]]

    def _aggregate(self, by_type: Dict[str, List[dict]]) -> dict:
        def _mean(vals: list) -> float:
            filtered = [v for v in vals if v is not None]
            return round(sum(filtered) / len(filtered), 4) if filtered else 0.0

        agg: dict = {}

        if rows := by_type.get("PERSPECTIVE", []):
            agg["PERSPECTIVE"] = {
                "n": len(rows),
                "avg_horizon_contamination_rate": _mean(
                    [r.get("horizon_contamination_rate") for r in rows]
                ),
                "avg_first_in_cone_rank": _mean(
                    [r.get("first_in_cone_rank") for r in rows]
                ),
                "interpretation": (
                    "High contamination = retrieval surfaces many out-of-cone docs; "
                    "epistemic discipline is load-bearing, not optional."
                ),
            }

        if rows := by_type.get("COUNTERFACTUAL", []):
            pct_traceable = _mean(
                [1.0 if r.get("causal_chain_traceable") else 0.0 for r in rows]
            )
            agg["COUNTERFACTUAL"] = {
                "n": len(rows),
                "pct_causal_chain_traceable_top10": pct_traceable,
                "pct_requires_multi_hop": round(1.0 - pct_traceable, 4),
                "interpretation": (
                    "Low traceability = causal link cannot be closed by a single "
                    "retrieval pass; agent must traverse cause → effect explicitly."
                ),
            }

        if rows := by_type.get("SILENCE", []):
            agg["SILENCE"] = {
                "n": len(rows),
                "avg_search_space_bm25_coverage": _mean(
                    [r.get("search_space_bm25_coverage") for r in rows]
                ),
                "pct_fully_covered": _mean(
                    [
                        1.0
                        if r.get("search_space_bm25_coverage", 0) >= 1.0
                        else 0.0
                        for r in rows
                    ]
                ),
                "interpretation": (
                    "Low coverage = BM25 misses required absence-check locations; "
                    "agent must enumerate expected_search_space explicitly."
                ),
            }

        return agg


# ─────────────────────────────────────────────────────────────────────────────
# DATASET CARD WRITER
# ─────────────────────────────────────────────────────────────────────────────


class DatasetCardWriter:
    """Produces the HuggingFace README.md dataset card for the v2 eval."""

    def write(
        self,
        out_path: Path,
        corpus: List[dict],
        questions: List[dict],
        causal_links: List[dict],
        actor_visibility: dict,
        absence_catalog: List[dict],
        baseline_summary: dict,
        cfg: dict,
    ) -> None:
        card = self._render(
            corpus,
            questions,
            causal_links,
            actor_visibility,
            absence_catalog,
            baseline_summary,
            cfg,
        )
        out_path.write_text(card, encoding="utf-8")
        logger.info(f"  → {out_path}")

    # ── PRIVATE ───────────────────────────────────────────────────────────────

    def _render(
        self,
        corpus: List[dict],
        questions: List[dict],
        causal_links: List[dict],
        actor_visibility: dict,
        absence_catalog: List[dict],
        baseline_summary: dict,
        cfg: dict,
    ) -> str:
        sim_cfg = cfg.get("simulation", {})
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

        # Causal link breakdown
        by_link: Dict[str, int] = defaultdict(int)
        for lnk in causal_links:
            by_link[lnk.get("link_type", "?")] += 1

        # Eval index counts
        n_actors = len(actor_visibility)
        n_cone_snapshots = sum(len(v) for v in actor_visibility.values())

        # Baseline tables — two-tier system
        ceiling = baseline_summary.get("ungated_ceiling", {})
        bm25_section = self._ungated_ceiling_table(ceiling.get("bm25", {}))
        dense_section = self._ungated_ceiling_table(ceiling.get("dense", {}))
        reasoning_section = self._reasoning_metrics_table(
            baseline_summary.get("static_reasoning_metrics", {})
        )

        return textwrap.dedent(f"""\
        ---
        language:
        - en
        license: mit
        configs:
        - config_name: default
            data_files:
            - split: train
                path: "**/*.parquet"
        task_categories:
        - question-answering
        - text-retrieval
        task_ids:
        - extractive-qa
        - document-retrieval
        tags:
        - rag
        - enterprise
        - synthetic
        - orgforge
        - causal-reasoning
        - temporal-reasoning
        - epistemic-reasoning
        - agentic-eval
        pretty_name: "OrgForge Enterprise Agentic RAG Benchmark"
        size_categories:
        - 1K<n<10K
        ---

        # OrgForge Enterprise Agentic RAG Benchmark

        > A synthetic but causally-grounded benchmark for evaluating agentic RAG
        > systems against realistic enterprise knowledge bases — with explicit
        > trajectory scoring for epistemic discipline, causal reasoning, and
        > absence verification.

        ## Dataset Summary

        This dataset was produced by **OrgForge**, an event-driven organisation
        simulator that generates weeks of realistic enterprise activity — JIRA
        tickets, Confluence pages, Slack threads, zoom transcripts, emails, PRs, Zendesk tickets,
        and Salesforce records — in a controlled, reproducible way.

        All ground-truth answers are derived **deterministically** from the
        simulation's event log via three purpose-built indexes:
        - **Actor visibility cones** — what each actor could have known at each moment
        - **Causal link index** — explicit cause→effect relationships encoded by the sim
        - **Absence catalog** — expected-but-absent artifacts confirmed by the state machine

        No LLM invented any answer. LLMs only wrote question prose.

        | Property | Value |
        |---|---|
        | Company | {company} |
        | Industry | {industry} |
        | Simulation days | {num_days} |
        | Sprints simulated | {num_sprints} |
        | Org size (engineers + staff) | ~{org_size} |
        | Total corpus documents | {len(corpus):,} |
        | Total eval questions | {len(questions):,} |
        | Causal links indexed | {len(causal_links):,} |
        | Actors with visibility cones | {n_actors} |
        | Visibility cone snapshots | {n_cone_snapshots:,} |
        | Absence records | {len(absence_catalog):,} |

        ## Corpus

        Each document represents a real artifact produced by the simulation.
        Stored in `corpus/corpus-00000.parquet`.

        | Artifact type | Count |
        |---|---|
        {self._table_rows(by_type)}

        ### Corpus Schema

        | Column | Type | Description |
        |---|---|---|
        | `doc_id` | str | Unique artifact ID (e.g. `IT-042`, `CONF-ENG-007`) |
        | `doc_type` | str | `jira`, `confluence`, `slack`, `email`, `pr`, `zd_ticket`, `sf_opp`, `sf_account`, `sim_event` |
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
        | `is_external` | bool | True for inbound external content |

        ## Eval Questions

        Questions are in `questions/questions-00000.parquet`.

        | Question type | Count |
        |---|---|
        {self._table_rows(by_qtype)}

        | Difficulty | Count |
        |---|---|
        {self._table_rows(by_diff)}

        ### Question Tracks

        | Track | Description | Score weights (answer / trajectory) |
        |---|---|---|
        | `PERSPECTIVE` | Could actor X have known about event Y as of Day N, given their subsystem access? | 0.40 / 0.60 |
        | `COUNTERFACTUAL` | If condition X had been different, would outcome Y have occurred? | 0.50 / 0.50 |
        | `SILENCE` | Was artifact X actually created in response to trigger Y? (correct answer: no) | 0.30 / 0.70 |

        #### PERSPECTIVE
        Scored primarily on **epistemic discipline**: did the agent stay within the
        actor's visibility cone and access only permitted subsystems? Trajectory
        weight (0.60) exceeds answer weight (0.40) because using out-of-cone
        artifacts to reach the correct answer is still a failure mode.

        #### COUNTERFACTUAL
        Requires identifying the **explicit causal link** encoded by the simulation
        (`involves_gap`, `recurrence_of`, `spawned_doc`, `email_dropped`,
        `sf_ownership_lapsed`, `zd_escalation_source`, `blocker_flagged`,
        `incident_coordination`, `departure_reassignment`). No inference — the
        link must be traceable to real artifacts.

        #### SILENCE
        Tests **absence-of-evidence reasoning**. Trajectory weight is highest
        (0.70) because a correct "no" answer reached without searching
        `expected_search_space` scores 0 on trajectory even if the boolean
        is right. The agent must demonstrate it checked the right places.

        > **Retrieval baselines** are reported for PERSPECTIVE and COUNTERFACTUAL
        > only. SILENCE questions test absence — standard retrieval recall is not
        > applicable because the correct answer is that the artifact does not exist.

        ### Question Schema

        | Column | Type | Description |
        |---|---|---|
        | `question_id` | str | Unique question identifier |
        | `question_type` | str | `PERSPECTIVE`, `COUNTERFACTUAL`, or `SILENCE` |
        | `question_text` | str | Natural-language question |
        | `ground_truth` | str | JSON-serialised answer dict |
        | `evidence_chain` | str | JSON list of artifact IDs (empty for SILENCE) |
        | `difficulty` | str | `medium` or `hard` |
        | `requires_reasoning` | bool | Always True — all tracks require multi-step reasoning |
        | `actor` | str | PERSPECTIVE: actor whose knowledge horizon is tested |
        | `actor_role` | str | PERSPECTIVE: actor's role slug |
        | `as_of_day` | int | PERSPECTIVE: knowledge horizon day |
        | `subsystem_access` | str | PERSPECTIVE: JSON list of accessible subsystems |
        | `blocked_subsystems` | str | PERSPECTIVE: JSON list of blocked subsystems |
        | `actor_visible_artifacts` | str | PERSPECTIVE: JSON list of all visible artifact IDs |
        | `link_type` | str | COUNTERFACTUAL: causal link type |
        | `causal_day` | int | COUNTERFACTUAL: day the causal link was established |
        | `expected_search_space` | str | SILENCE: JSON list of artifact paths agent must check |
        | `trigger_event_type` | str | SILENCE: event type that should have triggered the response |
        | `expected_response_type` | str | SILENCE: artifact/event type that was never created |

        ## Eval Indexes

        Stored in `eval_indexes/`. These are the ground-truth indexes that back
        question generation — useful for building custom eval harnesses.

        ### causal_link_index.parquet

        One row per explicit causal link found in the simulation.

        | Column | Type | Description |
        |---|---|---|
        | `link_type` | str | One of the causal link types above |
        | `cause_event_id` | str | Synthetic event ID of the cause |
        | `cause_event_type` | str | Event type of the cause |
        | `effect_event_id` | str | Synthetic event ID of the effect |
        | `effect_event_type` | str | Event type of the effect |
        | `actors` | str | JSON list of involved actor names |
        | `day` | int | Simulation day the link was established |
        | `counterfactual_premise` | str | Natural-language "if X had been different" |
        | `counterfactual_outcome` | str | Natural-language "then Y would have..." |
        | `outcome_changed` | bool | Always True — removing cause changes outcome |

        | Link type | Count |
        |---|---|
        {self._table_rows(by_link)}

        ### actor_visibility.parquet

        One row per (actor, day) snapshot.

        | Column | Type | Description |
        |---|---|---|
        | `actor` | str | Actor name |
        | `role` | str | Role slug |
        | `as_of_day` | int | Snapshot day |
        | `subsystem_access` | str | JSON list of accessible subsystems |
        | `all_visible_artifacts` | str | JSON list of all artifact IDs visible to this actor on this day |
        | `visible_artifacts_by_subsystem` | str | JSON dict (subsystem → artifact IDs) |
        | `directly_involved` | str | JSON list of artifacts where actor was in event.actors |
        | `broadcast_visible` | str | JSON list of artifacts visible via broadcast channel |

        ### absence_catalog.parquet

        One row per expected-but-absent artifact pair.

        | Column | Type | Description |
        |---|---|---|
        | `trigger_event_id` | str | Event that should have triggered a response |
        | `trigger_event_type` | str | Type of trigger event |
        | `expected_response_type` | str | Type of artifact that was never created |
        | `trigger_day` | int | Day the trigger event fired |
        | `expected_search_space` | str | JSON list of artifact paths agent must check |

        ## Baselines and Reasoning Difficulty

        OrgForge uses a **two-tier baseline** system, computed entirely from
        corpus metadata — no agent execution required.

        - **Tier 1 (Ungated Retrieval Ceiling):** What is the information ceiling
          if all gates are removed? This is "god-mode" retrieval, and the gap
          between it and a gated agent's score is the **Epistemic Tax**.
        - **Tier 2 (Static Reasoning Difficulty):** Why is each track hard,
          independent of any agent? These metrics characterise the epistemic
          structure before any model runs.

        Agent-level baselines (ungated god-mode agent, zero-shot no-tools)
        require LLM calls and are available as flags in `agentic_eval_harness.py`:
        `--ungated` and `--zero-shot`.

        ---

        ### Tier 1 — Ungated Retrieval Ceiling

        BM25 and dense retrieval with **no gates**: no visibility cones, no
        temporal horizons, no subsystem constraints ("god-mode" corpus access).

        The **Epistemic Tax** for a track is:

        ```
        epistemic_tax = ceiling_mrr@10 − gated_agent_combined_score
        ```

        A high tax on `PERSPECTIVE` means the question set heavily penalises
        using information the actor was never supposed to have.

        #### BM25 (Okapi BM25 via rank-bm25)

        {bm25_section}

        #### Dense Retrieval (`{_DENSE_MODEL_NAME}`)

        {dense_section}

        ---

        ### Tier 2 — Static Reasoning Difficulty Metrics

        Computed from corpus metadata and question ground-truth alone — no agents,
        no LLM calls required. These metrics characterise the epistemic structure
        of each question before any agent touches it.

        {reasoning_section}

        ---

        ### How to beat these baselines

        | Track | To beat the ceiling... |
        |---|---|
        | `PERSPECTIVE` | Achieve a `violation_adjusted_combined_score` above the ceiling MRR@10 **while** keeping `avg_actor_gate_violations` near 0. High score + high violations = the agent is cheating. |
        | `COUNTERFACTUAL` | Correctly identify the `causal_mechanism` for questions where `pct_requires_multi_hop = 1.0` — these cannot be answered by retrieval alone. |
        | `SILENCE` | Cover `expected_search_space` exhaustively before concluding. `avg_search_space_bm25_coverage` shows how little a naive search covers — the agent must enumerate the rest deliberately. |

        ---

        ## Agentic Evaluation

        Use `agentic_eval_harness.py` to run a gated agent against the full
        question set. The harness enforces temporal and actor visibility gates
        per question type, logs the complete tool-call trajectory, and scores
        both answer quality and trajectory quality.

        ```bash
        # Standard gated evaluation
        python agentic_eval_harness.py \\
            --questions export/eval/eval_questions.json \\
            --out export/eval/agentic_results.json \\
            --model claude-sonnet-4-6 \\
            --max-steps 15

        # Ungated god-mode agent — establishes the Epistemic Tax denominator
        python agentic_eval_harness.py \\
            --ungated \\
            --out export/eval/ungated_results.json

        # Zero-shot — no tools, no corpus — establishes the hallucination floor
        python agentic_eval_harness.py \\
            --zero-shot \\
            --out export/eval/zero_shot_results.json
        ```

        ## Leaderboard

        Submissions are ranked by `violation_adjusted_combined_score` on the
        **PERSPECTIVE** track. This is the primary axis because PERSPECTIVE is
        the only track with a hard behavioral constraint (actor visibility cone)
        that a capable-but-undisciplined agent can violate while still scoring
        high on raw accuracy.

        ### Ranking Formula

        ```
        violation_rate                = total_actor_gate_violations / total_tool_calls
        compliance_factor             = max(0, 1 − violation_rate) ** 2
        violation_adjusted_score      = combined_score × compliance_factor
        ```

        The quadratic exponent means violations compound non-linearly:

        | Violation rate | Compliance factor | Effective score discount |
        |---|---|---|
        | 0% (fully compliant) | 1.00 | None |
        | 10% | 0.81 | 19% |
        | 25% | 0.56 | 44% |
        | 50% | 0.25 | 75% |
        | 75% | 0.06 | 94% |

        ### Compliance Tiers

        | Tier | Violation rate | Meaning |
        |---|---|---|
        | `compliant` | < 5% | Agent demonstrates genuine epistemic discipline |
        | `borderline` | 5–20% | Agent occasionally accesses out-of-cone information |
        | `non_compliant` | > 20% | Agent is effectively operating in god-mode on PERSPECTIVE |

        > `combined_score` is still reported for reference but **must not** be
        > used as the primary ranking key. A model scoring 0.90 combined with a
        > 50% violation rate has a `violation_adjusted_combined_score` of 0.225
        > and belongs in `non_compliant` — below a model scoring 0.70 combined
        > with 0% violations (`violation_adjusted_combined_score` = 0.70, tier:
        > `compliant`).

        ## Citation

        ```bibtex
        @misc{{orgforge2026,
          title  = {{OrgForge: A Multi-Agent Simulation Framework for Verifiable Synthetic Corporate Corpora}},
          author = {{Jeffrey Flynt}},
          year   = {{2026}},
          note   = {{Synthetic benchmark generated by the OrgForge simulator v2}}
        }}
        ```

        ## License

        MIT. The simulation engine that produced this dataset is independently
        licensed; see the OrgForge repository for details.
        """)

    def _table_rows(self, d: Dict[str, int]) -> str:
        return "\n        ".join(
            f"| `{k}` | {v:,} |" for k, v in sorted(d.items(), key=lambda x: -x[1])
        )

    def _ungated_ceiling_table(self, summary: dict) -> str:
        """Renders the Tier 1 ungated retrieval ceiling table."""
        if "error" in summary:
            return f"> ⚠️ Ceiling unavailable: {summary['error']}"
        if not summary:
            return "> Ceiling not run."

        model = summary.get("model", "?")
        overall = summary.get("overall", {})
        by_type = summary.get("by_type", {})

        lines = [
            f"Model: `{model}`",
            "",
            "| Question type | MRR@10 | Recall@10 | N |",
            "|---|---|---|---|",
            (
                f"| **Overall** | **{overall.get('mrr_at_10', '?')}** "
                f"| **{overall.get('recall_at_10', '?')}** "
                f"| **{overall.get('n', '?')}** |"
            ),
        ]
        for qtype, metrics in sorted(by_type.items()):
            lines.append(
                f"| {qtype} | {metrics.get('mrr_at_10', '?')} "
                f"| {metrics.get('recall_at_10', '?')} "
                f"| {metrics.get('n', '?')} |"
            )
        lines += [
            "",
            "> SILENCE questions excluded — absence cannot be measured by retrieval recall.",
            "> **Epistemic Tax** = this ceiling MRR@10 − your gated agent's `violation_adjusted_combined_score`.",
        ]
        return "\n        ".join(lines)

    def _reasoning_metrics_table(self, static_metrics: dict) -> str:
        """Renders the Tier 2 static reasoning difficulty table."""
        if not static_metrics:
            return "> Static reasoning metrics not computed."

        lines = [
            "| Track | Metric | Value | Interpretation |",
            "|---|---|---|---|",
        ]

        if p := static_metrics.get("PERSPECTIVE"):
            lines += [
                (
                    f"| `PERSPECTIVE` | `avg_horizon_contamination_rate` "
                    f"| {p.get('avg_horizon_contamination_rate', '?')} "
                    f"| Fraction of ungated top-20 outside actor's visibility cone |"
                ),
                (
                    f"| `PERSPECTIVE` | `avg_first_in_cone_rank` "
                    f"| {p.get('avg_first_in_cone_rank', '?')} "
                    f"| Mean rank of first permitted doc — lower is easier |"
                ),
            ]

        if cf := static_metrics.get("COUNTERFACTUAL"):
            lines += [
                (
                    f"| `COUNTERFACTUAL` | `pct_causal_chain_traceable_top10` "
                    f"| {cf.get('pct_causal_chain_traceable_top10', '?')} "
                    f"| Fraction where cause+effect co-appear in ungated top-10 |"
                ),
                (
                    f"| `COUNTERFACTUAL` | `pct_requires_multi_hop` "
                    f"| {cf.get('pct_requires_multi_hop', '?')} "
                    f"| Fraction unreachable by a single retrieval pass |"
                ),
            ]

        if s := static_metrics.get("SILENCE"):
            lines += [
                (
                    f"| `SILENCE` | `avg_search_space_bm25_coverage` "
                    f"| {s.get('avg_search_space_bm25_coverage', '?')} "
                    f"| Fraction of required absence-check locations BM25 surfaces |"
                ),
                (
                    f"| `SILENCE` | `pct_fully_covered` "
                    f"| {s.get('pct_fully_covered', '?')} "
                    f"| Questions where BM25 covers the entire search space |"
                ),
            ]

        lines += [
            "",
            "> **Reading these metrics:** High contamination + low traceability + low coverage",
            "> means the question set demands genuine reasoning over retrieval luck. A gated",
            "> agent that outperforms the ungated ceiling on `PERSPECTIVE` questions is actively",
            "> exercising epistemic discipline — it is refusing correct-but-forbidden information.",
        ]
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


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────


class HFExporter:
    """
    Orchestrates the full v2 export pipeline:
      1. Build corpus from SimEvent log + MongoDB
      2. Load v2 eval data (eval_questions.json + the three eval indexes)
      3. Compute two-tier baselines (ungated retrieval ceiling + static reasoning metrics)
      4. Write Parquet files for corpus, questions, and eval indexes
      5. Write dataset card (README.md)
    """

    def run(self) -> None:
        logger.info("[bold cyan]📦 HuggingFace dataset export v2 starting…[/bold cyan]")

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
        if not corpus:
            logger.warning("  Empty corpus — check that flow.py has run first.")

        # 3. Load v2 eval data
        questions_path = EVAL_DIR / "eval_questions.json"
        causal_links_path = EVAL_DIR / "causal_link_index.json"
        actor_vis_path = EVAL_DIR / "actor_visibility.json"
        absence_path = EVAL_DIR / "absence_catalog.json"

        q_data = (
            json.loads(questions_path.read_text()) if questions_path.exists() else {}
        )
        raw_questions = (
            q_data.get("questions", []) if isinstance(q_data, dict) else q_data
        )
        # Filter to the three v2 tracks only (guard against mixed-version files)
        questions = [
            q
            for q in raw_questions
            if q.get("question_type") in ("PERSPECTIVE", "COUNTERFACTUAL", "SILENCE")
        ]

        causal_links = (
            json.loads(causal_links_path.read_text())
            if causal_links_path.exists()
            else []
        )
        actor_visibility = (
            json.loads(actor_vis_path.read_text()) if actor_vis_path.exists() else {}
        )
        absence_catalog = (
            json.loads(absence_path.read_text()) if absence_path.exists() else []
        )

        logger.info(
            f"  {len(questions)} eval questions loaded "
            f"({sum(1 for q in questions if q.get('question_type') == 'PERSPECTIVE')} PERSPECTIVE, "
            f"{sum(1 for q in questions if q.get('question_type') == 'COUNTERFACTUAL')} COUNTERFACTUAL, "
            f"{sum(1 for q in questions if q.get('question_type') == 'SILENCE')} SILENCE)"
        )
        logger.info(
            f"  {len(causal_links)} causal links, "
            f"{len(actor_visibility)} actors, "
            f"{len(absence_catalog)} absence records loaded"
        )

        # 4. Two-tier baselines
        # ──────────────────────────────────────────────────────────────────────
        # Tier 1: ungated retrieval ceiling — BM25 and dense with no gates.
        # Tier 2: static reasoning difficulty — computed from metadata alone,
        #         reuses the already-built BM25 index to avoid double work.
        ceiling_runner = UngatedCeilingBaseline(corpus, questions, mem=mem)
        bm25_per_q, bm25_agg = ceiling_runner.run_bm25()
        dense_per_q, dense_agg = ceiling_runner.run_dense()

        static_metrics = StaticReasoningMetrics(
            questions=questions,
            bm25=ceiling_runner._bm25,       # reuse the already-built index
            doc_ids=ceiling_runner._doc_ids,
        )
        reasoning_output = static_metrics.compute()

        baseline_summary = {
            "ungated_ceiling": {"bm25": bm25_agg, "dense": dense_agg},
            "static_reasoning_metrics": reasoning_output["aggregate"],
        }

        (BASELINE_DIR / "ungated_ceiling_bm25.json").write_text(
            json.dumps(bm25_per_q, indent=2, default=str)
        )
        (BASELINE_DIR / "ungated_ceiling_dense.json").write_text(
            json.dumps(dense_per_q, indent=2, default=str)
        )
        (BASELINE_DIR / "static_reasoning_metrics.json").write_text(
            json.dumps(reasoning_output, indent=2, default=str)
        )
        (BASELINE_DIR / "baseline_summary.json").write_text(
            json.dumps(baseline_summary, indent=2, default=str)
        )
        logger.info(f"  → baselines written to {BASELINE_DIR}")

        # 5. Parquet — corpus + questions + eval indexes
        _write_parquet(corpus, CORPUS_DIR, "corpus-00000")
        _write_parquet(_questions_to_rows(questions), QUES_DIR, "questions-00000")
        _write_parquet(
            _causal_links_to_rows(causal_links),
            EVAL_INDEX_DIR,
            "causal_link_index",
        )
        _write_parquet(
            _actor_visibility_to_rows(actor_visibility),
            EVAL_INDEX_DIR,
            "actor_visibility",
        )
        _write_parquet(
            _absence_catalog_to_rows(absence_catalog),
            EVAL_INDEX_DIR,
            "absence_catalog",
        )

        # 6. Dataset card
        DatasetCardWriter().write(
            out_path=HF_DIR / "README.md",
            corpus=corpus,
            questions=questions,
            causal_links=causal_links,
            actor_visibility=actor_visibility,
            absence_catalog=absence_catalog,
            baseline_summary=baseline_summary,
            cfg=_CFG,
        )

        bm25_overall = bm25_agg.get("overall", {})
        dense_overall = dense_agg.get("overall", {})
        srm = reasoning_output["aggregate"]
        logger.info(
            f"[green]✓ Export v2 complete.[/green] "
            f"Output: {HF_DIR}  |  "
            f"Ceiling BM25 MRR@10: {bm25_overall.get('mrr_at_10', 'n/a')}  |  "
            f"Ceiling Dense MRR@10: {dense_overall.get('mrr_at_10', 'n/a')}  |  "
            f"PERSPECTIVE contamination: "
            f"{srm.get('PERSPECTIVE', {}).get('avg_horizon_contamination_rate', 'n/a')}  |  "
            f"COUNTERFACTUAL multi-hop: "
            f"{srm.get('COUNTERFACTUAL', {}).get('pct_requires_multi_hop', 'n/a')}  |  "
            f"SILENCE BM25 coverage: "
            f"{srm.get('SILENCE', {}).get('avg_search_space_bm25_coverage', 'n/a')}"
        )


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    HFExporter().run()
