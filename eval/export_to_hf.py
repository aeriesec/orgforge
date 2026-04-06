"""
export_to_hf.py
===============
Normalises all OrgForge simulation artifacts into a flat HuggingFace-ready
corpus, produces Parquet files, and writes a dataset card (README.md).

Run after flow.py completes:
    python export_to_hf.py

Output layout
-------------
export/hf_dataset/
  corpus/
    corpus-00000.parquet    — flat document corpus (one row per artifact)
  README.md                 — HuggingFace dataset card
"""

from __future__ import annotations

import json
import logging
import textwrap
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
import email as email_lib
from email.header import decode_header
import shutil

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
HF_DIR = BASE / "hf_dataset"
CORPUS_DIR = HF_DIR / "corpus"


_ARTIFACT_DOC_TYPES = frozenset(
    {
        "confluence",
        "dept_plans",
        "jira",
        "slack",
        "email",
        "pr",
        "zd_ticket",
        "sf_opp",
        "sf_account",
        "nps_survey",
        "invoice",
        "datadog_alert",
        "zoom_transcript",
    }
)


_EXCLUDE_FIELDS = {"_id", "timestamp", "embedding"}


for d in (CORPUS_DIR,):
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
        "ENG": "",
        "PROD": "Product",
        "MKT": "Sales_Marketing",
        "QA": "QA_Support",
        "HR": "HR_Ops",
        "RETRO": "",
    }.get(code, "")


import email as email_lib
from email.header import decode_header


def _parse_eml(eml_path: Path) -> dict:
    """Parse a .eml file and return headers + decoded body."""
    raw = eml_path.read_text(encoding="utf-8", errors="replace")
    msg = email_lib.message_from_string(raw)

    subject_parts = decode_header(msg.get("Subject", ""))
    subject = ""
    for part, charset in subject_parts:
        if isinstance(part, bytes):
            subject += part.decode(charset or "utf-8", errors="replace")
        else:
            subject += part

    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    body = payload.decode(
                        part.get_content_charset() or "utf-8", errors="replace"
                    )
                    break
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            body = payload.decode(
                msg.get_content_charset() or "utf-8", errors="replace"
            )

    return {
        "subject": subject,
        "from_addr": msg.get("From", ""),
        "to_addr": msg.get("To", ""),
        "direction": msg.get("X-OrgForge-Direction", ""),
        "body": body,
    }


def _load_confluence_from_disk() -> List[dict]:
    confluence_dir = BASE / "confluence"
    if not confluence_dir.exists():
        return []

    rows = []
    for p in confluence_dir.rglob("*.md"):
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()

            # Parse header fields
            doc_id = p.stem
            title = ""
            author = ""
            date = ""

            for line in lines[:6]:
                if line.startswith("# "):
                    title = line[2:].strip()
                elif line.startswith("**ID:**"):
                    doc_id = line.replace("**ID:**", "").strip()
                elif line.startswith("**Author:**"):
                    author = line.replace("**Author:**", "").strip()
                elif line.startswith("**Date:**"):
                    date = line.replace("**Date:**", "").strip()

            rows.append(
                {
                    "doc_id": doc_id,
                    "doc_type": "confluence",
                    "category": "artifact",
                    "title": title,
                    "body": text,
                    "day": 0,
                    "date": date,
                    "timestamp": f"{date}T09:00:00" if date else "",
                    "actors": json.dumps([author] if author else []),
                    "tags": json.dumps(["confluence"]),
                    "artifact_ids": json.dumps({}),
                    "dept": _dept_from_artifact_id(doc_id),
                    "is_incident": False,
                    "is_external": False,
                    "facts": "",
                }
            )
        except Exception as exc:
            logger.warning(f"  confluence disk read failed: {p} — {exc}")

    logger.info(f"  confluence disk fallback: {len(rows)} pages loaded")
    return rows


def _load_slack_from_disk() -> List[dict]:
    slack_dir = BASE / "slack"
    if not slack_dir.exists():
        return []

    buckets: Dict[str, dict] = {}
    for p in slack_dir.rglob("*.json"):
        try:
            messages = json.loads(p.read_text(encoding="utf-8", errors="replace"))
            for msg in messages:
                tid = msg.get("thread_id", "")
                if not tid:
                    continue
                if tid not in buckets:
                    buckets[tid] = {
                        "date": msg.get("date", ""),
                        "ts": msg.get("ts", ""),
                        "actors": set(),
                        "texts": [],
                    }
                bucket = buckets[tid]
                user = msg.get("user", "")
                if user:
                    bucket["actors"].add(user)
                text = msg.get("text", "")
                if text:
                    bucket["texts"].append(f"{user}: {text}" if user else text)
        except Exception as exc:
            logger.warning(f"  slack disk read failed: {p} — {exc}")

    rows = []
    for tid, bucket in buckets.items():
        rows.append(
            {
                "doc_id": tid,
                "doc_type": "slack",
                "category": "artifact",
                "title": tid.split("_2026")[0].replace("slack_", "#"),
                "body": "\n".join(bucket["texts"]),
                "day": 0,
                "date": bucket["date"],
                "timestamp": bucket["ts"],
                "actors": json.dumps(sorted(bucket["actors"])),
                "tags": json.dumps(["slack"]),
                "artifact_ids": json.dumps({}),
                "dept": "",
                "is_incident": False,
                "is_external": False,
                "facts": "",
            }
        )
    return rows


class CorpusBuilder:
    """
    Reads the MongoDB-persisted artifacts (via Memory) and the SimEvent log,
    then normalizes every artifact into a flat list of corpus rows.

    Falls back to reconstructing from the export directory if MongoDB is
    unavailable, which allows the exporter to run in offline/CI environments.
    """

    def __init__(self, mem=None, insider_threat_enabled: bool = False):
        self._mem = mem
        self._insider_threat_enabled = insider_threat_enabled
        self._events: List[dict] = []
        if mem is not None:
            for collection_name in ("sim_events", "events", "simevents"):
                try:
                    coll = mem._db[collection_name]
                    raw = list(coll.find({}, {"embedding": 0}))
                    if raw:
                        self._events = raw
                        logger.info(
                            f"  Loaded {len(self._events):,} SimEvents "
                            f"from '{collection_name}'."
                        )
                        break
                except Exception as exc:
                    logger.debug(f"  Collection '{collection_name}' unavailable: {exc}")
            if not self._events:
                try:
                    raw = mem.get_event_log(from_db=True)
                    self._events = [
                        e.to_dict() if hasattr(e, "to_dict") else e for e in raw
                    ]
                    logger.info(
                        f"  Loaded {len(self._events):,} SimEvents via get_event_log."
                    )
                except Exception as exc:
                    logger.warning(f"Could not load SimEvent log: {exc}")

    # ── PUBLIC ────────────────────────────────────────────────────────────────

    def _build_meta_map(self, collection: str, id_field: str) -> Dict[str, dict]:
        """Fetch all docs from a collection, strip noise fields, key by id_field."""
        meta_map = {}
        try:
            for doc in self._mem._db[collection].find({}, {"embedding": 0}):
                doc_id = str(doc.get(id_field, ""))
                if not doc_id:
                    continue
                clean = {k: v for k, v in doc.items() if k not in _EXCLUDE_FIELDS}
                meta_map[doc_id] = clean
        except Exception as exc:
            logger.debug(f"  meta_map failed for {collection}: {exc}")
        return meta_map

    def build(self) -> List[dict]:
        rows: List[dict] = []

        for evt in self._events:
            evt_rows = self._sim_event_to_row(
                evt, insider_threat_enabled=self._insider_threat_enabled
            )
            if evt_rows:
                rows.extend(evt_rows)

        if self._mem is not None:
            rows = self._enrich_from_mongo(rows)
            rows.extend(self._plans_to_corpus_rows())

        existing_ids = {row["doc_id"] for row in rows}
        rows.extend(_load_slack_from_disk())
        rows.extend(_load_confluence_from_disk())

        rows.extend(self._post_sim_to_corpus_rows())

        zoom_rows = [r for r in rows if r["doc_type"] == "zoom_transcript"]
        logger.info(f"  zoom rows before dedup: {len(zoom_rows)}")
        for r in zoom_rows[:3]:
            logger.info(f"    {r['doc_id']} body_len={len(r.get('body', ''))}")

        # Deduplication strategy:
        #   - For artifact doc_ids (jira, confluence, slack, etc.): keep the row
        #     with the longest body — the MongoDB-enriched version wins over the
        #     thin SimEvent version.
        #   - Internal event rows (EVT-* doc_ids) are unique by construction and
        #     never conflict with artifact rows, so they pass through intact.
        seen: Dict[str, dict] = {}
        for row in rows:
            did = row["doc_id"]
            if did not in seen or self._body_len(row) > self._body_len(seen[did]):
                seen[did] = row

        for row in seen.values():
            if not row.get("body"):
                row["body"] = row.get("content") or ""

        zoom_rows = [r for r in seen.values() if r["doc_type"] == "zoom_transcript"]
        logger.info(f"  zoom rows after dedup: {len(zoom_rows)}")

        rows = list(seen.values())

        for r in zoom_rows[:3]:
            logger.info(f"    {r['doc_id']} body_len={len(r.get('body', ''))}")

        by_type: Dict[str, int] = defaultdict(int)
        for row in rows:
            by_type[row["doc_type"]] += 1
        logger.info(
            f"  corpus: {len(rows):,} documents "
            f"({len(self._events):,} SimEvents → {len(rows):,} corpus rows)"
        )
        for doc_type, count in sorted(by_type.items(), key=lambda x: -x[1]):
            logger.info(f"    {doc_type:30s} {count:,}")
        return rows

    def artifact_counts(self, rows: List[dict]) -> Dict[str, int]:
        """Return counts by doc_type, sorted descending."""
        counts: Dict[str, int] = defaultdict(int)
        for row in rows:
            counts[row["doc_type"]] += 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))

    # ── PRIVATE ───────────────────────────────────────────────────────────────

    def _body_len(self, r: dict) -> int:
        return len(r.get("body") or r.get("content") or "")

    _IS_INCIDENT_TYPES = frozenset(
        {
            "incident_opened",
            "incident_resolved",
            "escalation_chain",
            "postmortem_created",
            "fix_in_progress",
            "zd_tickets_escalated",
            "sf_deals_risk_flagged",
            "crm_account_at_risk",
        }
    )

    _IS_EXTERNAL_TYPES = frozenset(
        {
            "inbound_external_email",
            "customer_email_routed",
            "vendor_email_routed",
            "email_dropped",
            "sales_outbound_email",
            "proactive_outreach_initiated",
            "zd_ticket_opened",
            "zd_tickets_escalated",
            "zd_tickets_resolved",
            "crm_touchpoint",
            "customer_health_briefing",
            "external_contact_summarized",
        }
    )

    def _facts_body(self, event_type: str, facts: dict, summary: str) -> str:
        """
        Render a structured plain-text body from SimEvent facts.
        Used for internal events that carry no separate artifact.
        Every key-value pair is included so the full ground truth is retrievable.
        """
        parts = [f"event_type: {event_type}"]
        if summary:
            parts.append(f"summary: {summary}")
        for key, val in facts.items():
            if val is None or val == "" or val == [] or val == {}:
                continue
            if isinstance(val, (dict, list)):
                parts.append(f"{key}: {json.dumps(val, default=str)}")
            else:
                parts.append(f"{key}: {val}")
        return "\n".join(parts)

    def _sim_event_to_row(
        self, evt: dict, insider_threat_enabled: bool = False
    ) -> List[dict]:
        event_type = evt.get("type", "")
        artifact_ids = evt.get("artifact_ids", {}) or {}
        facts = evt.get("facts", {}) or {}
        summary = evt.get("summary", "")

        if (
            event_type in ("dlp_alert", "secret_detected")
            and not insider_threat_enabled
        ):
            return []

        evt_actors = evt.get("actors", [])
        dept_val = ""
        if evt_actors:
            for _actor in evt_actors:
                _d = _ACTOR_TO_DEPT.get(str(_actor).strip(), "")
                if _d:
                    dept_val = _d
                    break

        if not dept_val:
            dept_val = str(facts.get("dept", "")).strip()

        if not dept_val:
            for aid in artifact_ids.values():
                _d = _dept_from_artifact_id(str(aid))
                if _d:
                    dept_val = _d
                    break

        is_incident = event_type in self._IS_INCIDENT_TYPES
        is_external = event_type in self._IS_EXTERNAL_TYPES

        evt_id = str(evt.get("_id", "")).strip()
        if not evt_id:
            actor_fp = (evt_actors[0] if evt_actors else "sys").replace(" ", "_")
            evt_id = f"EVT-{evt.get('day', 0):04d}-{event_type}-{actor_fp}"

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
            "facts": json.dumps(facts),
            "category": "sim_event",
        }

        rows: List[dict] = []

        jira_id = artifact_ids.get("jira", "")
        if jira_id:
            rows.append(
                {
                    **shared,
                    "doc_id": jira_id,
                    "doc_type": "jira",
                    "category": "artifact",
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
            rows.append(
                {
                    **shared,
                    "doc_id": conf_id,
                    "doc_type": "confluence",
                    "category": "artifact",
                    "title": str(facts.get("title", conf_id))[:512],
                    "body": facts.get("content", facts.get("summary", "")) or summary,
                    "dept": dept_val or _dept_from_artifact_id(conf_id),
                }
            )

        email_id = artifact_ids.get("email", "") or artifact_ids.get("embed_id", "")
        eml_path_str = artifact_ids.get("eml_path", "")
        if email_id:
            body = self._email_body(facts, evt)
            if eml_path_str:
                eml_path = Path(eml_path_str.lstrip("./"))
                if eml_path.exists():
                    parsed = _parse_eml(eml_path)
                    body = parsed.pop("body")
                    email_meta = parsed
                else:
                    body = self._email_body(facts, evt)
                    email_meta = {}
                    logger.warning(f"  eml not found on disk: {eml_path}")

                rows.append(
                    {
                        **shared,
                        "doc_id": email_id,
                        "doc_type": "email",
                        "category": "artifact",
                        "title": str(
                            email_meta.get("subject") or facts.get("subject", email_id)
                        )[:512],
                        "body": body,
                        "facts": json.dumps(
                            {**json.loads(shared["facts"]), **email_meta}
                        ),
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
                    "category": "artifact",
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
                    "category": "artifact",
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
                    "category": "artifact",
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
                    "category": "artifact",
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
                    "category": "artifact",
                    "title": str(facts.get("account_name", acc_id))[:512],
                    "body": self._sf_account_body(facts, acc_id),
                }
            )

        # ── Internal event row — ALWAYS emitted, even when artifact rows exist ─
        # Preserves the full ground-truth facts (stress snapshots, similarity
        # scores, coverage percentages, departure edge snapshots, etc.) as a
        # separately retrievable document. Filter by category == "sim_event"
        # to get the state-machine view independently from prose artifacts.
        rows.append(
            {
                **shared,
                "doc_id": evt_id,
                "doc_type": event_type,
                "title": event_type.replace("_", " ").title(),
                "body": self._facts_body(event_type, facts, summary),
                "facts": json.dumps(facts),
            }
        )

        zoom_path_str = artifact_ids.get("artifact_path", "")
        zoom_id = artifact_ids.get("zoom_transcript", "")
        if zoom_id and zoom_path_str and zoom_id.startswith("zoom_"):
            zoom_path = Path(zoom_path_str)
            if not zoom_path.is_absolute():
                zoom_path = Path(zoom_path_str.lstrip("./"))

            if zoom_path.exists():
                zoom_body = zoom_path.read_text(encoding="utf-8", errors="replace")
            else:
                zoom_body = summary
            rows.append(
                {
                    **shared,
                    "doc_id": zoom_id,
                    "doc_type": "zoom_transcript",
                    "category": "artifact",
                    "title": f"Zoom: {facts.get('topic', event_type)} ({evt.get('date', '')})",
                    "body": zoom_body,
                    "facts": json.dumps(facts),
                }
            )

        # Ensure no row has an empty body
        for row in rows:
            if not row.get("body"):
                row["body"] = summary

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

        dd_dir = BASE / "datadog"
        metrics_path = dd_dir / "metrics.jsonl"
        if metrics_path.exists():
            for line in metrics_path.read_text().splitlines():
                if not line.strip():
                    continue
                data = json.loads(line)
                rows.append(
                    {
                        "doc_id": f"DD-METRIC-{data.get('metric_name', '').replace('.', '_')}-{data.get('timestamp', '')}",
                        "doc_type": "datadog_metric",
                        "title": data.get("metric_name", "datadog metric"),
                        "body": json.dumps(data),
                        "day": data.get("day", 0),
                        "date": str(data.get("timestamp", ""))[:10],
                        "timestamp": str(data.get("timestamp", "")),
                        "actors": json.dumps([]),
                        "tags": json.dumps(["datadog", "metric"]),
                        "artifact_ids": json.dumps({}),
                        "dept": "Engineering_Backend",
                        "is_incident": data.get("alert_firing", False),
                        "is_external": False,
                    }
                )

        alerts_path = dd_dir / "alerts.jsonl"
        if alerts_path.exists():
            for line in alerts_path.read_text().splitlines():
                if not line.strip():
                    continue
                data = json.loads(line)
                alert_id = f"DD-{data.get('id', data.get('alert_id', ''))}"
                rows.append(
                    {
                        "doc_id": alert_id,
                        "doc_type": "datadog_alert",
                        "title": data.get("monitor_name", alert_id),
                        "body": json.dumps(data),
                        "day": data.get("day", 0),
                        "date": str(data.get("fired_at", ""))[:10],
                        "timestamp": str(data.get("fired_at", "")),
                        "actors": json.dumps([]),
                        "tags": json.dumps(
                            ["datadog", "alert", data.get("incident_id", "")]
                        ),
                        "artifact_ids": json.dumps(
                            {
                                "jira": data.get("attributes", {}).get(
                                    "jira_ticket", data.get("jira_ticket", "")
                                )
                            }
                        ),
                        "dept": "Engineering_Backend",
                        "is_incident": True,
                        "is_external": False,
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

        tech_doc = (
            self._mem._db["sim_config"].find_one({"_id": "tech_stack"})
            if self._mem
            else None
        )
        if tech_doc:
            stack = tech_doc.get("stack", {})
            body = "\n".join(f"{k}: {v}" for k, v in stack.items())
            rows.append(
                {
                    "doc_id": "SIM-CONFIG-tech_stack",
                    "doc_type": "sim_config",
                    "title": "Tech Stack",
                    "body": body,
                    "day": 0,
                    "date": str(tech_doc.get("created_at", ""))[:10],
                    "timestamp": str(tech_doc.get("created_at", "")),
                    "actors": json.dumps([]),
                    "tags": json.dumps(["tech_stack", "sim_config"]),
                    "artifact_ids": json.dumps({}),
                    "dept": "",
                    "is_incident": False,
                    "is_external": False,
                    "facts": "",
                    "category": "sim_config",
                }
            )

        return rows

    def _enrich_from_mongo(self, rows: List[dict]) -> List[dict]:
        """
        Attempt to replace thin SimEvent body text with richer MongoDB content.
        Silently skips if the collection is unavailable.
        """
        try:
            conf_rich_map: Dict[str, str] = {}
            conf_id_map: Dict[str, str] = {}
            for page in self._mem._db["confluence_pages"].find(
                {}, {"_id": 0, "id": 1, "content": 1, "title": 1}
            ):
                if page.get("id") and page.get("content"):
                    conf_rich_map[page["id"]] = page["content"]
                    snippet = page["content"][:120].strip()
                    if snippet:
                        conf_id_map[snippet] = page["id"]
                    title_key = page.get("title", "").strip()
                    if title_key:
                        conf_id_map[title_key] = page["id"]

            # ── Jira comments (used when building jira body) ──────────────────
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

            def _jira_body(doc):
                parts = [
                    doc.get("title", ""),
                    doc.get("description", ""),
                    doc.get("root_cause", ""),
                ]
                for c in doc.get("comments") or []:
                    parts.append(str(c.get("text", "") or c.get("body", "")))
                for c in comment_map.get(doc.get("id", ""), []):
                    parts.append(c)
                return "\n".join(p for p in parts if p)

            def _pr_body(doc):
                parts = []
                for key in ("title", "description"):
                    if doc.get(key):
                        parts.append(f"{key}: {doc[key]}")
                if doc.get("author"):
                    parts.append(f"author: {doc['author']}")
                if doc.get("ticket_id"):
                    parts.append(f"ticket: {doc['ticket_id']}")
                if doc.get("status"):
                    parts.append(f"status: {doc['status']}")
                reviewers = doc.get("reviewers", [])
                if reviewers:
                    parts.append(f"reviewers: {', '.join(reviewers)}")
                for c in doc.get("comments") or []:
                    verdict = f" [{c['verdict']}]" if c.get("verdict") else ""
                    text = c.get("text", "")
                    author = c.get("author", "")
                    if text:
                        parts.append(f"review ({author}{verdict}): {text}")
                return "\n".join(p for p in parts if p)

            def _email_body(doc):
                parts = []
                if doc.get("subject"):
                    parts.append(f"subject: {doc['subject']}")
                if doc.get("from_name") or doc.get("from_addr"):
                    parts.append(
                        f"from: {doc.get('from_name', '')} <{doc.get('from_addr', '')}>"
                    )
                if doc.get("to_name") or doc.get("to_addr"):
                    parts.append(
                        f"to: {doc.get('to_name', '')} <{doc.get('to_addr', '')}>"
                    )
                if doc.get("body"):
                    parts.append(doc["body"])
                return "\n".join(parts)

            def _zd_body(doc):
                parts = [
                    f"subject: {doc.get('subject', '')}",
                    f"org: {doc.get('org_name', '')}",
                    f"status: {doc.get('status', '')}",
                    f"priority: {doc.get('priority', '')}",
                ]
                if doc.get("description"):
                    parts.append(f"description: {doc['description']}")
                if doc.get("related_incident"):
                    parts.append(f"related_incident: {doc['related_incident']}")
                for c in doc.get("comments") or []:
                    author = c.get("author", "")
                    text = c.get("text", "")
                    if text:
                        parts.append(
                            f"comment ({author}): {text}"
                            if author
                            else f"comment: {text}"
                        )
                return "\n".join(p for p in parts if p)

            def _sf_opp_body(doc):
                parts = [
                    f"account: {doc.get('account_name', '')}",
                    f"stage: {doc.get('stage', '')}",
                    f"probability: {doc.get('probability', '')}%",
                    f"amount: ${doc.get('amount', 0):,}",
                    f"owner: {doc.get('owner', '')}",
                    f"close_date: {doc.get('close_date', '')}",
                    f"lead_source: {doc.get('lead_source', '')}",
                    f"next_step: {doc.get('next_step', '')}",
                ]
                for note in doc.get("risk_notes") or []:
                    parts.append(f"risk: {note}")
                for tp in doc.get("touchpoints") or []:
                    subject = tp.get("subject", "")
                    sender = tp.get("sender", "")
                    ts = tp.get("timestamp", "")
                    if subject:
                        parts.append(f"touchpoint ({sender}, {ts}): {subject}")
                return "\n".join(p for p in parts if p)

            def _sf_account_body(doc):
                parts = [
                    f"name: {doc.get('name', '')}",
                    f"type: {doc.get('type', '')}",
                    f"tier: {doc.get('tier', '')}",
                    f"industry: {doc.get('industry', '')}",
                    f"billing_region: {doc.get('billing_region', '')}",
                    f"arr: ${doc.get('arr', 0):,}",
                    f"owner: {doc.get('owner', '')}",
                    f"primary_contact: {doc.get('primary_contact', '')}",
                ]
                if doc.get("risk_flag"):
                    parts.append("risk_flag: true — ownership lapsed or at-risk")
                return "\n".join(p for p in parts if p)

            jira_rich, jira_meta = self._build_rich_and_meta(
                "jira_tickets", "id", _jira_body
            )
            pr_rich, pr_meta = self._build_rich_and_meta(
                "pull_requests", "pr_id", _pr_body
            )
            email_rich, email_meta = self._build_rich_and_meta(
                "emails", "embed_id", _email_body
            )
            zd_rich, zd_meta = self._build_rich_and_meta(
                "zd_tickets", "ticket_id", _zd_body
            )
            sf_opp_rich, sf_opp_meta = self._build_rich_and_meta(
                "sf_opps", "opportunity_id", _sf_opp_body
            )
            sf_acc_rich, sf_acc_meta = self._build_rich_and_meta(
                "sf_accounts", "account_id", _sf_account_body
            )

            # Merge all rich maps into one lookup
            rich_map: Dict[str, str] = {
                **conf_rich_map,
                **jira_rich,
                **pr_rich,
                **email_rich,
                **zd_rich,
                **sf_opp_rich,
                **sf_acc_rich,
            }

            # Map doc_type -> meta map for facts enrichment
            _DOC_TYPE_TO_META: Dict[str, Dict[str, dict]] = {
                "jira": jira_meta,
                "pr": pr_meta,
                "email": email_meta,
                "zd_ticket": zd_meta,
                "sf_opp": sf_opp_meta,
                "sf_account": sf_acc_meta,
            }

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
                            f"Reclassified CONF-UNKNOWN social event as slack: {row['doc_id']}"
                        )
                elif row["doc_id"] in rich_map:
                    row["body"] = rich_map[row["doc_id"]]
                    if row["doc_type"] == "confluence" and not row.get("dept"):
                        row["dept"] = _dept_from_artifact_id(row["doc_id"])

                # Enrich facts for all artifact types in one pass
                meta_map = _DOC_TYPE_TO_META.get(row["doc_type"])
                if meta_map and row["doc_id"] in meta_map:
                    existing = json.loads(row.get("facts") or "{}")
                    existing.update(meta_map[row["doc_id"]])
                    row["facts"] = json.dumps(existing, default=str)

            # ── Orphan sweep: artifacts in MongoDB not yet in corpus ───────────
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
                    "category": "artifact",
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
                    "facts": "",
                }

            # artifacts collection
            _ARTIFACT_TYPE_MAP = {
                "confluence": "confluence",
                "jira": "jira",
                "zd_ticket": "zd_ticket",
                "sf_opportunity": "sf_opp",
                "zoom_transcript": "zoom_transcript",
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

            # pull_requests
            for doc in self._mem._db["pull_requests"].find({}, {"embedding": 0}):
                pid = doc.get("pr_id", "")
                if not pid or pid in existing_ids:
                    continue
                rows.append(
                    _make_orphan_row(
                        doc_id=pid,
                        doc_type="pr",
                        title=doc.get("title", pid),
                        body=pr_rich.get(pid, ""),
                        day=doc.get("day"),
                        date=doc.get("date"),
                        timestamp=doc.get("timestamp"),
                        actors=[doc["author"]] if doc.get("author") else [],
                        tags=["pr"],
                        artifact_type="pr",
                    )
                )
                existing_ids.add(pid)

            # emails
            for doc in self._mem._db["emails"].find({}, {"embedding": 0}):
                eid = doc.get("embed_id")
                if not eid or eid in existing_ids:
                    continue
                rows.append(
                    _make_orphan_row(
                        doc_id=eid,
                        doc_type="email",
                        title=doc.get("subject", eid),
                        body=email_rich.get(eid, ""),
                        day=doc.get("day"),
                        date=doc.get("date"),
                        timestamp=doc.get("timestamp"),
                        actors=[doc["from_name"]] if doc.get("from_name") else [],
                        tags=["email", doc.get("direction", "")],
                        artifact_type="email",
                        is_external=True,
                    )
                )
                existing_ids.add(eid)

            # sf_accounts
            for doc in self._mem._db["sf_accounts"].find({}, {"embedding": 0}):
                aid = doc.get("account_id", "")
                if not aid or aid in existing_ids:
                    continue
                rows.append(
                    _make_orphan_row(
                        doc_id=aid,
                        doc_type="sf_account",
                        title=doc.get("name", aid),
                        body=sf_acc_rich.get(aid, ""),
                        day=None,
                        date=None,
                        timestamp=doc.get("created_at"),
                        actors=[doc["owner"]] if doc.get("owner") else [],
                        tags=["salesforce", "account"],
                        artifact_type="sf_account",
                        is_external=True,
                    )
                )
                existing_ids.add(aid)

            # slack_messages — bucket by thread_id
            thread_buckets: Dict[str, dict] = {}
            for msg in self._mem._db["slack_messages"].find({}, {"embedding": 0}):
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
                    bucket["texts"].append(f"{author}: {text}" if author else text)

            for tid, bucket in thread_buckets.items():
                if tid in existing_ids:
                    continue
                actors = sorted(bucket["actors"])
                channel = bucket["channel"]
                rows.append(
                    _make_orphan_row(
                        doc_id=tid,
                        doc_type="slack",
                        title=f"#{channel}" if channel else tid,
                        body="\n".join(bucket["texts"]),
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
# CORPUS STATS
# ─────────────────────────────────────────────────────────────────────────────


def _compute_corpus_stats(corpus: List[dict], cfg: dict, mem=None) -> dict:
    """
    Derives everything the dataset card needs from the corpus + config.

    mem is optional — if provided, the raw SimEvent count is read from MongoDB
    so the card shows total events alongside deduplicated corpus documents.
    """
    sim_cfg = cfg.get("simulation", {})
    org_chart = cfg.get("org_chart", {})
    knowledge_gaps = cfg.get("knowledge_gaps", [])

    artifact_counts: Dict[str, int] = defaultdict(int)
    sim_event_counts: Dict[str, int] = defaultdict(int)
    incident_docs = 0
    external_docs = 0
    dept_counts: Dict[str, int] = defaultdict(int)
    actors_seen: set = set()
    days_seen: set = set()

    for row in corpus:
        dt = row["doc_type"]
        if dt in _ARTIFACT_DOC_TYPES:
            artifact_counts[dt] += 1
        else:
            sim_event_counts[dt] += 1
        if row.get("is_incident"):
            incident_docs += 1
        if row.get("is_external"):
            external_docs += 1
        if row.get("dept"):
            dept_counts[row["dept"]] += 1
        for actor in json.loads(row.get("actors") or "[]"):
            if isinstance(actor, str):
                actors_seen.add(actor)
        if row.get("day"):
            days_seen.add(int(row["day"]))

    # Raw SimEvent count from MongoDB.
    # Most SimEvents are internal state-machine events (day_summary,
    # knowledge_gap_detected, proposed_event_rejected, etc.) that do not
    # map 1:1 to a corpus artifact. The corpus is the deduplicated set of
    # *artifacts* those events produced — which is why corpus doc count
    # will always be much smaller than the raw event count.
    sim_events_total = None
    sim_days_actual = None
    if mem is not None:
        for _coll in ("sim_events", "events", "simevents"):
            try:
                _n = mem._db[_coll].count_documents({})
                if _n:
                    sim_events_total = _n
                    last_event = mem._db[_coll].find_one(
                        {}, sort=[("day", -1)], projection={"day": 1}
                    )
                    if last_event:
                        sim_days_actual = last_event["day"]
                    break
            except Exception:
                continue

    genesis_gaps = []
    sim_start_str = sim_cfg.get("start_date", "")
    for gap in knowledge_gaps:
        departed_name = gap.get("name", "Unknown")
        left_str = gap.get("left", "")
        doc_pct = gap.get("documented_pct", 0.5)
        domains = gap.get("knew_about", [])

        days_before = 0
        if sim_start_str and left_str:
            try:
                sim_start = datetime.strptime(sim_start_str, "%Y-%m-%d")
                left_dt = datetime.strptime(left_str, "%Y-%m")
                days_before = (sim_start - left_dt).days
            except ValueError:
                pass

        genesis_gaps.append(
            {
                "name": departed_name,
                "left": left_str,
                "days_before_sim": days_before,
                "documented_pct": doc_pct,
                "domains": domains,
                "role": gap.get("role", ""),
                "dept": gap.get("dept", ""),
            }
        )

    customers = []
    vendors = []

    domain_registry_count = 0
    company_description = sim_cfg.get("company_description", "")
    domain = sim_cfg.get("domain", "")
    legacy_system = cfg.get("legacy_system", {}).get("name", "")
    insider_threat = cfg.get("insider_threat", {}).get("enabled", False)

    if mem is not None:
        try:
            sources_doc = mem._db["sim_config"].find_one(
                {"_id": "inbound_email_sources"}
            )
            sources = sources_doc.get("sources", []) if sources_doc else []
            customers = [s for s in sources if s.get("category") == "customer"]
            vendors = [s for s in sources if s.get("category") == "vendor"]
        except Exception:
            pass
        try:
            domain_registry_count = mem._db["domain_registry"].count_documents({})
        except Exception:
            pass

    return {
        "artifact_counts": dict(sorted(artifact_counts.items(), key=lambda x: -x[1])),
        "sim_event_counts": dict(sorted(sim_event_counts.items(), key=lambda x: -x[1])),
        "by_type": ...,
        "total": len(corpus),
        "incident_docs": incident_docs,
        "external_docs": external_docs,
        "dept_counts": dict(sorted(dept_counts.items(), key=lambda x: -x[1])),
        "unique_actors": len(actors_seen),
        "sim_days_covered": len(days_seen),
        "sim_events_total": sim_events_total,
        "org_size": sum(len(v) for v in org_chart.values() if isinstance(v, list)),
        "company": sim_cfg.get("company_name", "OrgForge Simulated Corp"),
        "domain": domain,
        "insider_threat": insider_threat,
        "legacy_system": legacy_system,
        "company_description": company_description,
        "industry": sim_cfg.get("industry", "Software"),
        "genesis_gaps": genesis_gaps,
        "customers": customers,
        "vendors": vendors,
        "domain_registry_count": domain_registry_count,
        "num_days": sim_days_actual or sim_cfg.get("num_days", "?"),
    }


class DatasetCardWriter:
    """
    Produces the HuggingFace README.md dataset card.

    Tells the story of the corpus first — what it is, why the ground truth
    is trustworthy, and what makes this dataset structurally different from
    other synthetic benchmarks. Artifact counts and schema follow.
    """

    def write(self, out_path: Path, corpus: List[dict], cfg: dict, mem=None) -> None:
        stats = _compute_corpus_stats(corpus, cfg, mem=mem)
        card = self._render(stats, cfg)
        out_path.write_text(card, encoding="utf-8")
        logger.info(f"  → {out_path}")

    # ── PRIVATE ───────────────────────────────────────────────────────────────

    def _render(self, stats: dict, cfg: dict) -> str:
        sim_cfg = cfg.get("simulation", {})
        org_lifecycle = cfg.get("org_lifecycle", {})

        company = stats["company"]
        industry = stats["industry"]
        domain = stats["domain"]
        legacy_system = stats["legacy_system"]
        insider_threat = stats["insider_threat"]
        company_description = stats["company_description"]
        num_days = str(stats["num_days"])
        org_size = str(stats["org_size"])
        total_docs = f"{stats['total']:,}"
        incident_docs = f"{stats['incident_docs']:,}"
        external_docs = f"{stats['external_docs']:,}"
        unique_actors = str(stats["unique_actors"])
        sim_events_total = (
            f"{stats['sim_events_total']:,}" if stats.get("sim_events_total") else "n/a"
        )
        genesis_gaps = stats["genesis_gaps"]
        artifact_counts = stats.get("artifact_counts", stats["by_type"])
        sim_event_counts = stats.get("sim_event_counts", {})
        n_customers = len(stats.get("customers", []))
        n_vendors = len(stats.get("vendors", []))
        tech_stack = stats.get("tech_stack", [])
        domain_reg_count = stats.get("domain_registry_count", 0)

        gap_table = self._genesis_gap_table(genesis_gaps)
        artifact_table = self._artifact_count_table(artifact_counts, "Artifact")
        sim_event_table = self._artifact_count_table(sim_event_counts, "SimEvent")
        dept_table = self._dept_count_table(stats["dept_counts"])
        schema_table = self._corpus_schema_table(sim_event_counts)

        scheduled_departures = org_lifecycle.get("scheduled_departures", [])
        scheduled_hires = org_lifecycle.get("scheduled_hires", [])
        enable_attrition = org_lifecycle.get("enable_random_attrition", False)
        attrition_prob = org_lifecycle.get("random_attrition_daily_prob", 0.0)

        lifecycle_lines = []
        if scheduled_departures:
            lifecycle_lines.append(
                f"- **{len(scheduled_departures)} scheduled departure(s)** during the sim"
            )
        if scheduled_hires:
            lifecycle_lines.append(
                f"- **{len(scheduled_hires)} scheduled hire(s)** during the sim "
                "(backfill hires are generated with deliberate expertise gaps, "
                "creating second-order knowledge problems that play out over subsequent days)"
            )
        if enable_attrition:
            lifecycle_lines.append(
                f"- **Random attrition** enabled at {attrition_prob:.1%} daily probability"
            )
        lifecycle_summary = (
            "\n".join(lifecycle_lines)
            if lifecycle_lines
            else "- No mid-sim departures or hires configured"
        )

        _fm = (
            "---\n"
            "language:\n"
            "- en\n"
            "license: mit\n"
            "configs:\n"
            "- config_name: default\n"
            "  data_files:\n"
            "  - split: train\n"
            '    path: "corpus/*.parquet"\n'
            "task_categories:\n"
            "- question-answering\n"
            "- text-retrieval\n"
            "- text-generation\n"
            "- summarization\n"
            "- text-classification\n"
            "task_ids:\n"
            "- open-domain-qa\n"
            "- closed-domain-qa\n"
            "- abstractive-qa\n"
            "- open-domain-abstractive-qa\n"
            "- document-retrieval\n"
            "- fact-checking-retrieval\n"
            "- dialogue-modeling\n"
            "- explanation-generation\n"
            "- multi-label-classification\n"
            "- fact-checking\n"
            "tags:\n"
            "- rag\n"
            "- enterprise\n"
            "- synthetic\n"
            "- orgforge\n"
            "- causal-reasoning\n"
            "- temporal-reasoning\n"
            "- knowledge-graphs\n"
            "- agentic-eval\n"
            f'pretty_name: "OrgForge — {company} Enterprise Corpus"\n'
            "size_categories:\n"
            "- 1K<n<10K\n"
            "---"
        )

        sections = []

        # ── Title + pitch ──────────────────────────────────────────────────────
        sections.append(f"# OrgForge — {company} Enterprise Corpus")
        sections.append("")
        sections.append("![OrgForge corpus overview](orgforge_dataset_hero.png)")
        sections.append("")
        sections.append(
            "OrgForge generates synthetic but causally grounded enterprise corpora from a\n"
            "deterministic simulation engine. Every artifact in this dataset — Jira tickets,\n"
            "Slack threads, Confluence pages, customer emails, Zendesk tickets, invoices, Zoom\n"
            "transcripts, Datadog alerts — traces back to a single event log. No LLM invented\n"
            "any facts. The state machine controls what happened; LLMs only wrote the prose."
        )
        sections.append("")

        # ── Why it exists ──────────────────────────────────────────────────────
        sections.append("## Why it exists")
        sections.append("")
        sections.append(
            "Evaluating agents that reason over institutional knowledge requires a corpus where\n"
            "the ground truth is not just *present* but *verifiable*. You need to know not just\n"
            "what the correct answer is, but why it is correct, when it became correct, and what\n"
            "changed it. Existing synthetic datasets generate plausible-looking documents with no\n"
            "guarantee of consistency across artifacts or time. OrgForge produces something\n"
            "structurally different: a corpus where every fact has a cause, every cause has a\n"
            "timestamp, and every timestamp connects to a retrievable artifact."
        )
        sections.append("")
        sections.append(
            f"This dataset is the output of a **{num_days}-day simulation** of **{company}**, a\n"
            f"{industry} company which {company_description} with ~{org_size} employees. It is not a random walk through\n"
            "enterprise activity — it was seeded with specific organizational crises and simulated\n"
            "through to their resolution."
        )
        sections.append("")

        # ── What makes it different ────────────────────────────────────────────
        sections.append("## What makes this corpus structurally different")
        sections.append("")
        sections.append(
            "**Causal grounding.** Every artifact is downstream of a SimEvent. A Jira ticket,\n"
            "the Slack thread that opened alongside it, the Confluence postmortem written the\n"
            "next day, and the Zendesk tickets that escalated from the same incident all share\n"
            "a causal ancestor. Cross-referencing between artifact types is not coincidental —\n"
            "it reflects the actual information flow the simulation produced."
        )
        sections.append("")
        sections.append(
            "**Temporal coherence.** Facts change over the simulation. An engineer present on\n"
            "Day 1 is gone on Day 12. Ticket ownership, domain coverage scores, relationship\n"
            "graph edge weights, and customer sentiment all evolve. The correct answer to a\n"
            "question about org state depends on what day it is asked relative to the timeline.\n"
            "Every corpus row carries a day, date, and timestamp accurate to the millisecond\n"
            "the underlying event fired."
        )
        sections.append("")
        sections.append(
            "**Verifiable ground truth.** The simulation snapshot and domain registry ship\n"
            "alongside the corpus as structured reference files (see Supplemental Files). For\n"
            "any question the corpus can raise — who owned this domain when this incident fired,\n"
            "which customer was affected, what was the system health on the day this postmortem\n"
            "was written — the answer exists as a queryable record independent of the text. You\n"
            "do not need to parse the corpus to build your eval set."
        )
        sections.append("")
        sections.append(
            "**Pre-simulation history.** The genesis knowledge gaps in this corpus pre-date the\n"
            "simulation by months or years. An agent asked why a Day 15 postmortem surfaces a\n"
            "specific knowledge gap must trace: current incident → semantic similarity match →\n"
            "departed employee persona → genesis event dated before sim start. That causal chain\n"
            "crosses a temporal boundary that does not exist in any other synthetic enterprise\n"
            "dataset we are aware of."
        )
        sections.append("")
        sections.append(
            "**State-driven external communication.** Customer emails, vendor alerts, and\n"
            "Zendesk tickets are generated from actual simulation conditions, not randomly\n"
            "sampled. Each external contact has a `depends_on_components` list mapped to the\n"
            "tech stack — an outage to a component a customer depends on is what triggers their\n"
            "email. Approximately 15% of customer emails are deliberately dropped with no action,\n"
            "leaving ground-truth absences in the event log that an agent must detect through\n"
            "negative evidence rather than positive retrieval."
        )
        sections.append("")
        sections.append(
            "**Persona-consistent prose.** Every artifact is written by a character with a\n"
            "specific tenure, stress level, writing style, and live CRM context. A Slack message\n"
            "from an engineer during a contract negotiation reads differently from one written by\n"
            "the same person on a quiet day. Stylometric and behavioral signals in the text\n"
            "reflect the org's state at the moment of writing, not random LLM variation."
        )
        sections.append("")

        # ── Use cases ──────────────────────────────────────────────────────────
        sections.append("## Use cases")
        sections.append("")
        sections.append(
            "- **Agentic reasoning** — tasks that require traversing causal chains across\n"
            "  artifact types, time, and org boundaries rather than finding a single relevant\n"
            "  document\n"
            "- **Multi-hop question answering** — questions whose correct answer requires\n"
            "  joining facts from Jira, Confluence, Slack, CRM, and the simulation ground truth\n"
            "- **Temporal reasoning** — questions where the correct answer depends on what day\n"
            "  they are asked relative to the simulation timeline\n"
            "- **RAG pipeline evaluation** — a corpus with known causal structure allows\n"
            "  precise measurement of what a retrieval system found versus what it needed to\n"
            "  find to answer correctly\n"
            "- **Org dynamics and knowledge loss research** — the simulation snapshot exposes\n"
            "  how knowledge concentration, engineer departure, and incident causation interact\n"
            "  over time in a controlled, reproducible setting"
        )
        sections.append("")

        # ── Scope ──────────────────────────────────────────────────────────────
        sections.append("## Scope and limitations")
        sections.append("")
        sections.append(
            "This is not a dataset of real corporate communications. The company, employees,\n"
            "customers, and vendors are entirely fictional. The simulation models organizational\n"
            "behavior at the structural level — stress, knowledge concentration, incident\n"
            "causation, relationship graph dynamics — but does not model everything. Affect,\n"
            "politics, ambiguity, and the texture of real human communication are present only\n"
            "to the extent that the persona and mood system introduces them through LLM-generated\n"
            "prose. Researchers should treat this as a controlled benchmark environment, not a\n"
            "proxy for real enterprise data."
        )
        sections.append("")
        sections.append("---")
        sections.append("")

        # ── Genesis knowledge gaps ─────────────────────────────────────────────
        sections.append("## Genesis Knowledge Gaps")
        sections.append("")
        sections.append(
            "These gaps pre-date the simulation. They are the structural cause of the\n"
            "organizational narrative in this corpus. Each departed employee's domains entered\n"
            "Day 1 as orphaned — undocumented, unowned, and detectable only through semantic\n"
            "similarity when new incidents touch the same systems."
        )
        sections.append("")
        if legacy_system:
            sections.append(
                "\n\n"
                f"The primary technical fault line in this corpus is **{legacy_system}**, a "
                f"{cfg.get('legacy_system', {}).get('description', 'legacy system')} whose "
                f"instability is the proximate cause of most incidents during the simulation."
            )
            sections.append("")
        sections.append(gap_table)
        sections.append("")
        sections.append("---")
        sections.append("")

        # ── Org lifecycle ──────────────────────────────────────────────────────
        sections.append("## Org Lifecycle")
        sections.append("")
        sections.append(lifecycle_summary)
        sections.append("")
        sections.append("---")
        sections.append("")

        # ── Corpus summary ─────────────────────────────────────────────────────
        sections.append("## Corpus Summary")
        sections.append("")
        sections.append("| Property | Value |")
        sections.append("|---|---|")
        sections.append(f"| Company | {company} |")
        sections.append(f"| Description | {company} {company_description} |")
        sections.append(f"| Domain | {domain} |")
        sections.append(f"| Industry | {industry} |")
        sections.append(f"| Simulation days | {num_days} |")
        sections.append(f"| Org size | ~{org_size} employees |")
        sections.append(f"| Customers | {n_customers} |")
        sections.append(f"| Vendors | {n_vendors} |")
        sections.append(f"| Total corpus documents | {total_docs} |")
        sections.append(f"| Total SimEvents | {sim_events_total} |")
        sections.append(f"| Incident-related documents | {incident_docs} |")
        sections.append(f"| External-origin documents | {external_docs} |")
        sections.append(f"| Unique actors | {unique_actors} |")
        sections.append(f"| Domain registry entries | {domain_reg_count} |")
        if tech_stack:
            sections.append(f"| Tech stack | {', '.join(str(t) for t in tech_stack)} |")
        sections.append("")
        sections.append("### Artifacts")
        sections.append("")
        sections.append(artifact_table)
        sections.append("")
        sections.append("### SimEvents (internal state-machine records)")
        sections.append("")
        sections.append(
            "SimEvents are the ground-truth event log entries that produced the artifacts above.\n"
            "They are included in the corpus as separately retrievable records for researchers\n"
            "who want the state-machine view alongside the prose artifacts."
        )
        sections.append("")
        sections.append(sim_event_table)
        sections.append("")
        sections.append("### By department")
        sections.append("")
        sections.append(dept_table)
        sections.append("")
        sections.append("---")
        sections.append("")

        # ── Supplemental files ─────────────────────────────────────────────────
        sections.append("## Supplemental Files")
        sections.append("")
        sections.append(
            "The corpus parquet contains the retrievable text artifacts. The following files\n"
            "ship alongside it for eval construction, ground-truth lookups, and time-series\n"
            "analysis. They are in `supplemental/`."
        )
        sections.append("")
        sections.append(
            "**`simulation_snapshot.json`** — Full org state at simulation end: incidents with\n"
            "open/resolve timestamps, morale curve, daily system health scores, relationship\n"
            "graph edge weights, departed employees, new hires, and knowledge gap events. This\n"
            "is the oracle for eval construction. Use it to build questions with verifiable\n"
            "answers without parsing the corpus."
        )
        sections.append("")
        sections.append(
            "**`assignment_scores.parquet`** — Per-sprint ticket assignment decisions with full\n"
            "scoring breakdown: skill match (embedding cosine similarity), inverse stress, \n"
            "betweenness centrality penalty, recency bonus, and composite score. One row per\n"
            "(engineer, ticket, day) triple. Useful for eval questions about whether assignments\n"
            "were optimal given org state at the time."
        )
        sections.append("")
        sections.append(
            "**`domain_registry.json`** — Snapshot of all knowledge domains: owner history,\n"
            "documentation coverage scores at each sim day, orphan status, and which incidents\n"
            "triggered semantic similarity matches against each domain. Joinable to corpus rows\n"
            "via the Confluence `doc_id` values that cover each domain."
        )
        sections.append("")
        sections.append(
            "**`sim_config.json`** — Reference record for the org configuration: full customer\n"
            "and vendor profiles (including `depends_on_components`, `sentiment_baseline`,\n"
            "`trigger_on` conditions, and `persona_archetype`), tech stack, and org structure.\n"
            "Useful for understanding why specific external communications were generated."
        )
        sections.append("")
        sections.append(
            "**`datadog_metrics.parquet`** — Time-series telemetry at 15-minute intervals\n"
            "across the simulation. Schema: `timestamp`, `metric_name`, `value`, `day`,\n"
            "`alert_firing` (bool). Kept separate from the corpus because individual metric\n"
            "ticks are not retrievable text documents. Datadog *alerts* are in the main corpus\n"
            "as `doc_type: datadog_alert` and link back to incidents via `artifact_ids`."
        )
        sections.append("")
        sections.append("---")
        sections.append("")

        # ── Schema ─────────────────────────────────────────────────────────────
        sections.append("## Corpus Schema")
        sections.append("")
        sections.append(
            "Stored in `corpus/corpus-00000.parquet`. One row per document."
        )
        sections.append("")
        sections.append(schema_table)
        sections.append("")
        sections.append("---")
        sections.append("")

        # ── Usage ──────────────────────────────────────────────────────────────
        sections.append("## Usage")
        sections.append("")
        sections.append("```python")
        sections.append("from datasets import load_dataset")
        sections.append("import json")
        sections.append("")
        sections.append('ds = load_dataset("aeriesec/orgforge")')
        sections.append('corpus = ds["train"]')
        sections.append("")
        sections.append("# All incident-related documents")
        sections.append('incidents = corpus.filter(lambda x: x["is_incident"])')
        sections.append("")
        sections.append("# All artifacts of a specific type")
        sections.append('jira    = corpus.filter(lambda x: x["doc_type"] == "jira")')
        sections.append(
            'zoom    = corpus.filter(lambda x: x["doc_type"] == "zoom_transcript")'
        )
        sections.append(
            'alerts  = corpus.filter(lambda x: x["doc_type"] == "datadog_alert")'
        )
        sections.append("")
        sections.append("# All documents involving a specific actor")
        sections.append("actor_docs = corpus.filter(")
        sections.append('    lambda x: "Jordan" in json.loads(x["actors"])')
        sections.append(")")
        sections.append("")
        sections.append("# All documents from a specific sim day")
        sections.append('day_5 = corpus.filter(lambda x: x["day"] == 5)')
        sections.append("")
        sections.append(
            "# Cross-reference: find the Confluence postmortem linked to a Jira ticket"
        )
        sections.append("def get_linked(corpus, doc_id, link_type):")
        sections.append('    source = [r for r in corpus if r["doc_id"] == doc_id][0]')
        sections.append(
            '    linked_id = json.loads(source["artifact_ids"]).get(link_type, "")'
        )
        sections.append('    return [r for r in corpus if r["doc_id"] == linked_id]')
        sections.append("```")
        sections.append("")
        sections.append(
            "The `artifact_ids` column is a JSON dict linking each document to related\n"
            "artifacts produced from the same SimEvent. An incident ticket will carry\n"
            "references to the Slack thread, PR, Confluence postmortem, and Datadog alert\n"
            "that share its causal ancestor, allowing full chain reconstruction without\n"
            "text matching."
        )
        sections.append("")
        sections.append("---")
        sections.append("")

        # ── Citation + license ─────────────────────────────────────────────────
        sections.append("## Citation")
        sections.append("")
        sections.append("If you use the OrgForge methodology or simulator, cite the paper:")
        sections.append("")
        sections.append("```bibtex")
        sections.append("@misc{flynt2026orgforge,")
        sections.append("  title     = {OrgForge: A Multi-Agent Simulation Framework for Verifiable Synthetic Corporate Corpora},")
        sections.append("  author    = {Jeffrey Flynt},")
        sections.append("  year      = {2026},")
        sections.append("  url       = {https://arxiv.org/abs/2603.14997},")
        sections.append("  note      = {arXiv:2603.14997}")
        sections.append("}")
        sections.append("```")
        sections.append("")
        sections.append("If you use this dataset directly, cite the dataset:")
        sections.append("")
        sections.append("```bibtex")
        sections.append("@misc{flynt2026orgforgedata,")
        sections.append(f"  title     = {{OrgForge — {company} Enterprise Corpus}},")
        sections.append("  author    = {Jeffrey Flynt},")
        sections.append("  year      = {2026},")
        sections.append("  url       = {https://huggingface.co/datasets/aeriesec/orgforge},")
        sections.append("  note      = {Dataset generated by the OrgForge simulator}")
        sections.append("}")
        sections.append("```")
        sections.append("")
        sections.append("## License")
        sections.append("")
        sections.append(
            "MIT. The simulation engine that produced this dataset is independently\n"
            "licensed under MIT; see the [OrgForge repository](https://github.com/aeriesec/orgforge)\n"
            "for details."
        )

        _body = "\n".join(sections)
        return _fm + "\n" + _body

    def _genesis_gap_table(self, gaps: List[dict]) -> str:
        if not gaps:
            return "> No genesis knowledge gaps configured for this simulation."

        header = (
            "| Former owner | Role | Departed | Days before sim | "
            "Documented at departure | Domains |\n"
            "|---|---|---|---|---|---|\n"
        )
        rows = []
        for gap in gaps:
            domains_str = ", ".join(f"`{d}`" for d in gap["domains"])
            rows.append(
                f"| {gap['name']} | {gap.get('role', '')} | {gap.get('left', '')} "
                f"| {gap['days_before_sim']} | {int(gap['documented_pct'] * 100)}% "
                f"| {domains_str} |"
            )
        return header + "\n".join(rows)

    def _artifact_count_table(self, by_type: Dict[str, int], type: str) -> str:
        header = f"| {type} | Count |\n|---|---|\n"
        rows = [f"| `{doc_type}` | {count:,} |" for doc_type, count in by_type.items()]
        return header + "\n".join(rows)

    def _dept_count_table(self, dept_counts: Dict[str, int]) -> str:
        if not dept_counts:
            return "> Department breakdown unavailable."
        header = "| Department | Documents |\n|---|---|\n"
        rows = [
            f"| {dept} | {count:,} |" for dept, count in dept_counts.items() if dept
        ]
        return header + "\n".join(rows)

    def _corpus_schema_table(self, sim_event_counts: Dict[str, int] = None) -> str:
        if sim_event_counts:
            sim_event_types = " \\| ".join(f"`{t}`" for t in sim_event_counts)
        else:
            sim_event_types = "`sim_event` \\| *(see corpus for full list)*"

        artifact_types = " \\| ".join(f"`{t}`" for t in sorted(_ARTIFACT_DOC_TYPES))

        return textwrap.dedent(f"""\
        | Column | Type | Description |
        |---|---|---|
        | `doc_id` | str | Unique artifact ID (e.g. `IT-042`, `CONF-ENG-007`, `PR-031`) |
        | `doc_type` | str | Artifact: {artifact_types} — SimEvent: {sim_event_types} |
        | `category` | str | `artifact` \\| `sim_event` \\| `sim_config` |
        | `title` | str | Human-readable title or subject line |
        | `body` | str | Full text content |
        | `day` | int | Simulation day this artifact was created (1-indexed) |
        | `date` | str | ISO date string |
        | `timestamp` | str | ISO datetime, business-hours-accurate to the millisecond |
        | `actors` | str | JSON list of actor names involved |
        | `tags` | str | JSON list of semantic tags from the SimEvent |
        | `artifact_ids` | str | JSON dict of cross-references to related artifacts by type |
        | `dept` | str | Owning department; empty if cross-department |
        | `is_incident` | bool | True if this artifact is part of a P1/P2 incident thread |
        | `is_external` | bool | True for artifacts originating outside the org (emails, Zendesk, NPS, invoices) |
        | `facts` | str | JSON dict of raw SimEvent facts; populated for SimEvent rows, empty string for artifact rows |""")


# ─────────────────────────────────────────────────────────────────────────────
# PARQUET WRITER
# ─────────────────────────────────────────────────────────────────────────────


def _write_parquet(rows: List[dict], out_dir: Path, stem: str = "corpus-00000") -> None:
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
    Orchestrates the corpus export pipeline:
      1. Build corpus from SimEvent log + MongoDB
      2. Write corpus Parquet
      3. Write dataset card (README.md)
    """

    def _write_hero_image(self, out_dir: Path) -> None:
        src = Path(__file__).resolve().parent / "orgforge_dataset_hero.png"
        if src.exists():
            shutil.copy2(src, HF_DIR / "orgforge_dataset_hero.png")
            logger.info("  → orgforge_dataset_hero.png")
        else:
            logger.warning(
                "  orgforge_dataset_hero.png not found next to script — skipping"
            )

    def _write_supplemental(self, mem, out_dir: Path) -> None:
        supp_dir = out_dir / "supplemental"
        supp_dir.mkdir(parents=True, exist_ok=True)

        snap_src = BASE / "simulation_snapshot.json"
        if snap_src.exists():
            shutil.copy2(snap_src, supp_dir / "simulation_snapshot.json")
            logger.info(f"  → supplemental/simulation_snapshot.json")
        else:
            logger.warning("  simulation_snapshot.json not found — skipping")

        if mem is not None:
            try:
                registry = list(mem._db["domain_registry"].find({}, {"embedding": 0}))
                if registry:
                    (supp_dir / "domain_registry.json").write_text(
                        json.dumps(registry, indent=2, default=str)
                    )
                    logger.info(
                        f"  → supplemental/domain_registry.json ({len(registry)} domains)"
                    )
            except Exception as exc:
                logger.warning(f"  domain_registry export failed: {exc}")

        dd_metrics = BASE / "datadog" / "metrics.jsonl"
        if dd_metrics.exists() and _PARQUET_AVAILABLE:
            rows = [
                json.loads(l) for l in dd_metrics.read_text().splitlines() if l.strip()
            ]
            if rows:
                _write_parquet(rows, supp_dir, stem="datadog_metrics")
                logger.info(
                    f"  → supplemental/datadog_metrics.parquet ({len(rows):,} rows)"
                )
        elif dd_metrics.exists():
            shutil.copy2(dd_metrics, supp_dir / "datadog_metrics.jsonl")
            logger.info(f"  → supplemental/datadog_metrics.jsonl (parquet unavailable)")

        try:
            scores = list(
                mem._db["assignment_scores"].find({}, {"_id": 0, "embedding": 0})
            )
            if scores and _PARQUET_AVAILABLE:
                _write_parquet(scores, supp_dir, stem="assignment_scores")
                logger.info(
                    f"  → supplemental/assignment_scores.parquet ({len(scores):,} rows)"
                )
        except Exception as exc:
            logger.warning(f"  assignment_scores export failed: {exc}")

        try:
            sim_config_docs = list(mem._db["sim_config"].find({}, {"_id": 1}))
            if sim_config_docs:
                sim_config_out = {}
                for doc in mem._db["sim_config"].find({}):
                    key = str(doc.pop("_id"))
                    sim_config_out[key] = doc
                (supp_dir / "sim_config.json").write_text(
                    json.dumps(sim_config_out, indent=2, default=str)
                )
                logger.info(f"  → supplemental/sim_config.json")
        except Exception as exc:
            logger.warning(f"  sim_config export failed: {exc}")

    def run(self) -> None:
        logger.info("📦 OrgForge HuggingFace export starting…")

        mem = None
        try:
            from memory import Memory

            mem = Memory()
            logger.info("  Connected to MongoDB.")
        except Exception as exc:
            logger.warning(
                f"  Memory unavailable ({exc}). Corpus will derive from SimEvent log only."
            )

        insider_threat_enabled = _CFG.get("insider_threat", {}).get("enabled", False)
        corpus_builder = CorpusBuilder(
            mem, insider_threat_enabled=insider_threat_enabled
        )
        corpus = corpus_builder.build()
        if not corpus:
            logger.warning("  Empty corpus — check that flow.py has run first.")
            return

        counts = corpus_builder.artifact_counts(corpus)
        logger.info("  Artifact counts:")
        for doc_type, count in counts.items():
            logger.info(f"    {doc_type:30s} {count:,}")

        _write_parquet(corpus, CORPUS_DIR, "corpus-00000")

        self._write_supplemental(mem, HF_DIR)

        self._write_hero_image(HF_DIR)

        DatasetCardWriter().write(
            out_path=HF_DIR / "README.md",
            corpus=corpus,
            cfg=_CFG,
            mem=mem,
        )

        logger.info(
            f"✓ Export complete. Output: {HF_DIR}  |  "
            f"Total documents: {len(corpus):,}  |  "
            f"Types: {len(counts)}"
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    HFExporter().run()
