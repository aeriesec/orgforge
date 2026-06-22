"""
eval_consistency.py
===================
Cross-document consistency comparison: OrgForge vs. LLM-only baselines.

Usage:
    python eval_consistency.py --model anthropic.claude-3-5-sonnet-20241022-v2:0
    python eval_consistency.py --incidents 3 --region us-west-2
    python eval_consistency.py --db orgforge --skip-baselines
    python eval_consistency.py --skip-nli  # skip prose-SimEvent divergence
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations
from statistics import mean
import time
from typing import Any

import boto3
from config_loader import COMPANY_NAME
from pymongo import MongoClient

from eval_divergence import NLIScorer, measure_artifact_divergence

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("eval_consistency")


class Bedrock:
    _RETRYABLE_ERRORS = (
        "ThrottlingException",
        "InternalServerException",
        "ServiceUnavailableException",
        "ModelTimeoutException",
        "Too Many Requests",
    )
    _MAX_RETRIES = 5
    _RETRY_BASE_DELAY = 2.0

    def __init__(self, model_id: str, region: str = "us-east-1"):
        self.model_id = model_id
        self._client = boto3.client("bedrock-runtime", region_name=region)

    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        messages = [{"role": "user", "content": [{"text": prompt}]}]
        system_list = [{"text": system}] if system else []
        kwargs: dict[str, Any] = {
            "modelId": self.model_id,
            "messages": messages,
            "inferenceConfig": {
                "maxTokens": max_tokens,
                "temperature": temperature,
            },
            "serviceTier": {"type": "flex"},
        }
        if system_list:
            kwargs["system"] = system_list

        resp = self._converse_with_retry(kwargs)
        content_blocks = resp["output"]["message"]["content"]
        for block in content_blocks:
            if block.get("type") == "text" or "text" in block:
                return block["text"]
        raise ValueError(f"No text block in Bedrock response: {content_blocks}")

    def _converse_with_retry(self, kwargs: dict[str, Any]) -> dict:
        delay = self._RETRY_BASE_DELAY
        last_exc: Exception = RuntimeError("No attempts made")
        for attempt in range(self._MAX_RETRIES):
            try:
                return self._client.converse(**kwargs)
            except Exception as exc:
                last_exc = exc
                retryable = any(tag in str(exc) for tag in self._RETRYABLE_ERRORS)
                if not retryable:
                    raise
                jitter = random.uniform(0, delay * 0.25)
                wait = delay + jitter
                logger.warning(
                    f"Bedrock transient error (attempt {attempt + 1}/{self._MAX_RETRIES}), "
                    f"retrying in {wait:.1f}s: {exc}"
                )
                time.sleep(wait)
                delay = min(delay * 2, 60.0)
        raise last_exc


@dataclass
class ArtifactDoc:
    artifact_id: str
    artifact_type: str
    title: str
    content: str
    timestamp: str
    author: str = ""
    metadata: dict = field(default_factory=dict)
    sim_event_type: str = ""
    sim_event_facts: dict = field(default_factory=dict)


@dataclass
class IncidentBundle:
    incident_id: str
    root_cause: str
    system_fault: str
    on_call: str
    day: int
    duration_days: int
    health_at_open: int
    actors: list[str] = field(default_factory=list)
    artifacts: list[ArtifactDoc] = field(default_factory=list)
    sim_event_facts: dict = field(default_factory=dict)


@dataclass
class EntitySet:
    artifact_id: str
    tech_components: set[str] = field(default_factory=set)
    person_names: set[str] = field(default_factory=set)
    ticket_ids: set[str] = field(default_factory=set)
    root_cause_tokens: set[str] = field(default_factory=set)


@dataclass
class EvalResult:
    arm: str
    entity_agreement: float
    temporal_violations: float
    contradictions: float
    prose_divergence: float
    n_incidents: int
    per_incident: list[dict] = field(default_factory=list)


class MongoExtractor:
    def __init__(
        self,
        db_name: str = "orgforge",
        uri: str = "mongodb://localhost:27017?directConnection=true",
    ):
        self._client = MongoClient(uri)
        self._db = self._client[db_name]

    def _resolve_sim_event_facts(self, artifact_id: str) -> tuple[str, dict]:
        ev = None
        if re.match(r"^[A-Z]+-\d+$", artifact_id):
            ev = self._db["events"].find_one(
                {"type": "incident_opened", "artifact_ids.jira": artifact_id}
            )

        if not ev:
            ev = self._db["events"].find_one(
                {
                    "$or": [
                        {"artifact_ids.jira": artifact_id},
                        {"artifact_ids.confluence": artifact_id},
                        {"artifact_ids.slack_thread": artifact_id},
                        {"artifact_ids.pr": artifact_id},
                        {"artifact_ids.zendesk": artifact_id},
                    ]
                },
                sort=[("day", 1)],
            )
        if not ev:
            ev = (
                self._db["events"].find_one(
                    {
                        "$or": [
                            {"artifact_ids.jira": {"$elemMatch": {"$eq": artifact_id}}},
                            {
                                "artifact_ids.confluence": {
                                    "$elemMatch": {"$eq": artifact_id}
                                }
                            },
                            {
                                "artifact_ids.slack_thread": {
                                    "$elemMatch": {"$eq": artifact_id}
                                }
                            },
                        ]
                    },
                    sort=[("day", 1)],
                )
                or {}
            )

        facts = dict(ev.get("facts", {}))

        if (
            ev.get("type", "") != "knowledge_gap_detected"
            and "actors" in ev
            and "actors" not in facts
        ):
            facts["actors"] = ev["actors"]

        if re.match(r"^[A-Z]+-\d+$", artifact_id):
            jira_doc = self._db["jira_tickets"].find_one(
                {"id": artifact_id}, {"assignee": 1, "escalation_actors": 1}
            )
            if jira_doc:
                doc_actors = list(
                    dict.fromkeys(
                        filter(
                            None,
                            [
                                jira_doc.get("assignee", ""),
                                *jira_doc.get("escalation_actors", []),
                            ],
                        )
                    )
                )
                if doc_actors:
                    facts["actors"] = doc_actors

        if "vendor_org" not in facts:
            vendor = (
                facts.get("external_party")
                or facts.get("org")
                or facts.get("vendor")
                or facts.get("vendor_name")
            )
            if vendor:
                facts["vendor_org"] = vendor

        artifact_ids = ev.get("artifact_ids", {})
        for src_key, dst_key in [
            ("jira", "ticket_id"),
            ("pr", "pr_id"),
            ("confluence", "confluence_id"),
            ("zendesk", "zd_ticket_id"),
            ("salesforce", "sf_opportunity_id"),
        ]:
            if src_key in artifact_ids and dst_key not in facts:
                facts[dst_key] = artifact_ids[src_key]

        if "assigned_to" not in facts:
            jira_doc = self._db["jira_tickets"].find_one(
                {"id": artifact_ids.get("jira", "")}, {"assignee": 1}
            )
            if jira_doc and jira_doc.get("assignee"):
                facts["assigned_to"] = jira_doc["assignee"]

        return ev.get("type", ""), facts

    _CAMEL_RE = re.compile(r"[A-Z][a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)|[a-z]+")

    def _camel_split(self, word: str) -> list[str]:
        parts = self._CAMEL_RE.findall(word)
        return parts if len(parts) > 1 else []

    def vendor_aliases(self) -> dict[str, set[str]]:
        doc = self._db["sim_config"].find_one({"_id": "inbound_email_sources"})
        if not doc:
            return {}

        tech = self.tech_stack_components()
        aliases: dict[str, set[str]] = {}

        for source in doc.get("sources", []):
            if source.get("category") != "vendor":
                continue

            name = source["name"]
            alias_set: set[str] = {name}

            org = source.get("org", "")
            if org:
                alias_set.add(org)

            words = name.split()
            if len(words) >= 2:
                acronym = "".join(w[0] for w in words if w[0].isupper())
                if len(acronym) >= 3:
                    alias_set.add(acronym)

            for word in words:
                parts = self._camel_split(word)
                if parts:
                    alias_set.add(" ".join(p.lower() for p in parts))

            for exp in source.get("persona", {}).get("expertise", []):
                if exp in tech:
                    alias_set.add(exp)

            alias_set.discard("")
            aliases[name] = alias_set

        return aliases

    def load_cached_baseline(
        self, arm: str, incident_id: str, model_id: str
    ) -> list[ArtifactDoc] | None:
        doc = self._db["baseline_cache"].find_one(
            {"_id": f"{arm}_{incident_id}_{model_id}"}
        )
        if not doc:
            return None
        return [ArtifactDoc(**a) for a in doc["artifacts"]]

    def save_cached_baseline(
        self, arm: str, incident_id: str, model_id: str, artifacts: list[ArtifactDoc]
    ) -> None:
        self._db["baseline_cache"].replace_one(
            {"_id": f"{arm}_{incident_id}_{model_id}"},
            {
                "_id": f"{arm}_{incident_id}_{model_id}",
                "arm": arm,
                "incident_id": incident_id,
                "model_id": model_id,
                "artifacts": [vars(a) for a in artifacts],
                "created_at": datetime.utcnow().isoformat(),
            },
            upsert=True,
        )

    def tech_stack_components(self) -> set[str]:
        doc = self._db["sim_config"].find_one({"_id": "tech_stack"})
        if not doc:
            return set()
        explicit = {
            "PostgreSQL",
            "TitanDB",
            "Java",
            "Spring",
            "Boot",
            "Python",
            "FastAPI",
            "React",
            "TypeScript",
            "Vite",
            "Kafka",
            "RabbitMQ",
            "AWS",
            "EC2",
            "EKS",
            "RDS",
            "S3",
            "CloudFront",
            "Terraform",
            "GitHub",
            "Jenkins",
            "Docker",
            "ECR",
            "Datadog",
            "Prometheus",
            "Grafana",
            "PagerDuty",
            "Swift",
            "Kotlin",
        }
        return explicit

    def org_names(self) -> set[str]:
        names: set[str] = set()
        for doc in self._db["events"].find(
            {"type": "day_summary"}, {"facts.active_actors": 1}
        ):
            for name in doc.get("facts", {}).get("active_actors", []):
                names.add(name)
        if not names:
            for doc in self._db["artifacts"].find(
                {"type": "persona_skill"}, {"metadata": 1}
            ):
                name = doc.get("metadata", {}).get("name")
                if name:
                    names.add(name)
        return names

    def extract_incidents(self, max_n: int = 5) -> list[IncidentBundle]:
        events = list(
            self._db["events"]
            .find({"type": "incident_opened"})
            .sort("day", 1)
            .limit(max_n)
        )
        bundles: list[IncidentBundle] = []
        for ev in events:
            facts = ev.get("facts", {})
            ticket_id = ev.get("artifact_ids", {}).get("jira", "")
            if not ticket_id:
                continue

            resolved_ev = self._db["events"].find_one(
                {"type": "incident_resolved", "artifact_ids.jira": ticket_id}
            )
            duration = (
                resolved_ev.get("facts", {}).get("duration_days", 0)
                if resolved_ev
                else 0
            )

            bundle = IncidentBundle(
                incident_id=ticket_id,
                root_cause=facts.get("root_cause", ""),
                system_fault=facts.get("system_fault", ""),
                on_call=ev.get("actors", [""])[0],
                day=ev.get("day", 0),
                duration_days=duration,
                health_at_open=facts.get("system_health", 100),
                actors=ev.get("actors", []),
                sim_event_facts=facts,
            )

            jira_doc = self._db["jira_tickets"].find_one({"id": ticket_id}, {"_id": 0})
            if jira_doc:
                bundle.actors = list(
                    set(
                        bundle.actors
                        + jira_doc.get("escalation_actors", [])
                        + [c.get("author", "") for c in jira_doc.get("comments", [])]
                    )
                    - {""}
                )

            causal_chain = (
                jira_doc.get("causal_chain", facts.get("causal_chain", [ticket_id]))
                if jira_doc
                else facts.get("causal_chain", [ticket_id])
            )
            bot_threads = set(
                jira_doc.get("bot_threads", facts.get("bot_threads", []))
                if jira_doc
                else facts.get("bot_threads", [])
            )
            bundle.artifacts = self._collect_artifacts(
                ticket_id, causal_chain, bot_threads
            )
            if bundle.artifacts:
                bundles.append(bundle)

        logger.info(f"Extracted {len(bundles)} incidents from MongoDB")
        return bundles

    def _collect_artifacts(
        self, ticket_id: str, causal_chain: list[str], bot_threads: set[str] = set()
    ) -> list[ArtifactDoc]:
        docs: list[ArtifactDoc] = []
        seen: set[str] = set()

        def _is_noise(art_doc: ArtifactDoc) -> bool:
            return art_doc.content.startswith(
                "GitHub: 💬"
            ) or art_doc.content.startswith("AWS Cost Explorer:")

        def _try_append(art_doc: ArtifactDoc | None) -> None:
            if art_doc and not _is_noise(art_doc):
                docs.append(art_doc)

        def _stamp_facts(doc: ArtifactDoc) -> ArtifactDoc:
            if not doc.sim_event_facts:
                ev_type, facts = self._resolve_sim_event_facts(doc.artifact_id)
                doc.sim_event_type = ev_type
                doc.sim_event_facts = facts
            return doc

        jira = self._db["jira_tickets"].find_one({"id": ticket_id})
        if jira:
            assignee = jira.get("assignee", "")
            escalation = " ".join(jira.get("escalation_actors", []))
            body_parts = [
                jira.get("title", ""),
                f"Assignee: {assignee}" if assignee else "",
                f"Escalation: {escalation}" if escalation else "",
                jira.get("description", ""),
                jira.get("root_cause", ""),
            ]

            ev_type, facts = self._resolve_sim_event_facts(ticket_id)

            docs.append(
                ArtifactDoc(
                    artifact_id=ticket_id,
                    artifact_type="jira",
                    title=jira.get("title", ticket_id),
                    content="\n".join(filter(None, body_parts)),
                    timestamp=jira.get("created_at", ""),
                    author=jira.get("assignee", ""),
                    sim_event_type=ev_type,
                    sim_event_facts=facts,
                )
            )
            seen.add(ticket_id)

            for comment in jira.get("comments", []):
                comment_author = comment.get("author", "")
                comment_text = comment.get("text", "")
                comment_ts = comment.get("created", "")
                if not comment_text:
                    continue
                comment_id = f"{ticket_id}_comment_{comment.get('day', comment_ts)}"
                if comment_id in seen:
                    continue
                seen.add(comment_id)
                docs.append(
                    ArtifactDoc(
                        artifact_id=comment_id,
                        artifact_type="jira_comment",
                        title=f"Comment on {ticket_id} by {comment_author}",
                        content=f"{comment_author}: {comment_text}",
                        timestamp=comment_ts,
                        author=comment_author,
                        sim_event_type=ev_type,
                        sim_event_facts=facts,
                    )
                )

        for chain_id in causal_chain:
            if chain_id in seen or chain_id in bot_threads:
                continue
            if re.match(r"^[A-Z]+-\d+_comment_\d+$", chain_id):
                seen.add(chain_id)
                continue
            seen.add(chain_id)
            resolved = self._resolve_artifact_by_id(chain_id)
            if resolved:
                _try_append(_stamp_facts(resolved))

        related_events = list(
            self._db["events"].find(
                {
                    "$or": [
                        {"artifact_ids.jira": ticket_id},
                        {"facts.causal_chain": ticket_id},
                    ]
                }
            )
        )
        for ev in related_events:
            ev_facts = ev.get("facts", {})
            ev_type = ev.get("type", "")
            for key, raw_aid in ev.get("artifact_ids", {}).items():
                aids = raw_aid if isinstance(raw_aid, list) else [raw_aid]
                for aid in aids:
                    if not aid or aid in seen or key in ("jira",):
                        continue
                    seen.add(aid)
                    art_doc = self._resolve_artifact(aid, key, ev)
                    if art_doc and not _is_noise(art_doc):
                        art_doc.sim_event_type = ev_type
                        art_doc.sim_event_facts = ev_facts
                        docs.append(art_doc)

        postmortem_ev = self._db["events"].find_one(
            {"type": "postmortem_created", "artifact_ids.jira": ticket_id}
        )
        if postmortem_ev:
            pm_facts = postmortem_ev.get("facts", {})
            raw_conf = postmortem_ev.get("artifact_ids", {}).get("confluence", "")
            conf_ids = (
                raw_conf
                if isinstance(raw_conf, list)
                else [raw_conf]
                if raw_conf
                else []
            )
            for conf_id in conf_ids:
                if conf_id and conf_id not in seen:
                    seen.add(conf_id)
                    resolved = self._resolve_artifact_by_id(conf_id)
                    if resolved:
                        resolved.sim_event_type = "postmortem_created"
                        resolved.sim_event_facts = pm_facts
                        _try_append(resolved)

        seen_content: set[str] = set()
        deduped: list[ArtifactDoc] = []
        for doc in docs:
            fingerprint = doc.content[:100].strip()
            if fingerprint not in seen_content:
                seen_content.add(fingerprint)
                deduped.append(doc)
        return deduped

    def _resolve_artifact(
        self, aid: str, key_hint: str, event: dict
    ) -> ArtifactDoc | None:
        if "slack" in key_hint:
            doc = self._db["artifacts"].find_one({"_id": aid, "type": "slack_thread"})
            if doc:
                return ArtifactDoc(
                    artifact_id=aid,
                    artifact_type="slack",
                    title=doc.get("title", f"Slack thread {aid}"),
                    content=doc.get("content", ""),
                    timestamp=doc.get("timestamp", ""),
                    author=doc.get("metadata", {}).get("participants", [""])[0],
                )
        if "confluence" in key_hint:
            doc = self._db["artifacts"].find_one({"_id": aid, "type": "confluence"})
            if doc:
                return ArtifactDoc(
                    artifact_id=aid,
                    artifact_type="confluence",
                    title=doc.get("title", aid),
                    content=doc.get("content", ""),
                    timestamp=doc.get("timestamp", ""),
                    author=doc.get("metadata", {}).get("author", ""),
                )
        if "pr" in key_hint:
            doc = self._db["pull_requests"].find_one({"pr_id": aid})
            if doc:
                author = doc.get("author", "")
                reviewers = " ".join(doc.get("reviewers", []))
                comment_parts = []
                for c in doc.get("comments", []):
                    c_author = c.get("author", "")
                    c_text = c.get("text", "")
                    c_verdict = c.get("verdict", "")
                    line = f"{c_author}: {c_text}"
                    if c_verdict:
                        line += f"\nReview verdict: {c_verdict.replace('_', ' ')}"
                    comment_parts.append(line)
                content = "\n".join(
                    filter(
                        None,
                        [
                            doc.get("title", ""),
                            f"Author: {author}" if author else "",
                            f"Reviewers: {reviewers}" if reviewers else "",
                            f"Status: {doc.get('status', '')}"
                            if doc.get("status")
                            else "",
                            doc.get("description", ""),
                            *comment_parts,
                        ],
                    )
                )
                return ArtifactDoc(
                    artifact_id=aid,
                    artifact_type="pr",
                    title=doc.get("title", aid),
                    content=content,
                    timestamp=doc.get("created_at", ""),
                    author=author,
                )
        return None

    def _resolve_artifact_by_id(self, aid: str) -> ArtifactDoc | None:
        for key_hint in ("confluence", "slack", "pr"):
            result = self._resolve_artifact(aid, key_hint, {})
            if result:
                return result

        messages = list(
            self._db["slack_messages"].find({"thread_id": aid}).sort("ts", 1)
        )
        if messages:
            content = "\n".join(
                f"{m.get('user', '?')}: {m.get('text', '')}"
                for m in messages
                if not m.get("is_bot", False)
            )
            participants = list({m.get("user", "") for m in messages if m.get("user")})
            if content:
                return ArtifactDoc(
                    artifact_id=aid,
                    artifact_type="slack",
                    title=f"Slack thread {messages[0].get('channel', '')} {aid}",
                    content=content,
                    timestamp=messages[0].get("ts", ""),
                    author=messages[0].get("user", ""),
                    metadata={"participants": participants},
                )

        doc = self._db["artifacts"].find_one({"_id": aid})
        if doc:
            return ArtifactDoc(
                artifact_id=aid,
                artifact_type=doc.get("type", "unknown"),
                title=doc.get("title", aid),
                content=doc.get("content", ""),
                timestamp=doc.get("timestamp", ""),
            )
        return None


class EntityExtractor:
    def __init__(self, tech_components: set[str], org_names: set[str]):
        self.canonical_tech = tech_components
        self.canonical_names = org_names
        self._tech = {
            re.compile(rf"\b{re.escape(c)}\b", re.IGNORECASE): c
            for c in tech_components
        }
        self._names = {
            re.compile(rf"\b{re.escape(n)}\b", re.IGNORECASE): n for n in org_names
        }
        self._ticket_re = re.compile(
            r"(?:^|(?<=\s))(?:ENG|HR|SALES|PROD|DES|QA|ORG)-\d+\b"
        )

    def extract(self, artifact: ArtifactDoc) -> EntitySet:
        text = artifact.content
        text_lower = text.lower()
        tech_hits = {
            canonical
            for pattern, canonical in self._tech.items()
            if pattern.search(text)
        }
        name_hits = {
            canonical
            for pattern, canonical in self._names.items()
            if pattern.search(text)
        }
        ticket_hits = set(self._ticket_re.findall(text))
        rc_tokens: set[str] = set()
        for word in re.split(r"\s+", text_lower):
            cleaned = word.strip(".,;:!?'\"()[]{}").lower()
            if len(cleaned) >= 4 and cleaned.isalpha():
                rc_tokens.add(cleaned)
        return EntitySet(
            artifact_id=artifact.artifact_id,
            tech_components=tech_hits,
            person_names=name_hits,
            ticket_ids=ticket_hits,
            root_cause_tokens=rc_tokens,
        )


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def entity_agreement_grounded(
    entity_sets: list[EntitySet],
    incident: IncidentBundle,
    canonical_tech: set[str],
    canonical_names: set[str],
) -> dict[str, float]:
    rc_text = " ".join(
        filter(
            None,
            [
                incident.root_cause,
                incident.system_fault,
                incident.sim_event_facts.get("title", ""),
            ],
        )
    ).lower()

    incident_tech = canonical_tech
    incident_names = set(incident.actors)
    incident_tickets = {incident.incident_id}
    for cid in incident.sim_event_facts.get("causal_chain", []):
        if re.match(r"^[A-Z]+-\d+$", cid):
            incident_tickets.add(cid)

    precision_scores: list[float] = []
    hallucination_scores: list[float] = []

    for es in entity_sets:
        checks: list[float] = []

        if es.tech_components:
            known = es.tech_components & canonical_tech
            if known:
                correct = known & incident_tech
                checks.append(len(correct) / len(known))

        if es.person_names:
            correct = es.person_names & (incident_names | canonical_names)
            checks.append(len(correct) / len(es.person_names))

        if es.ticket_ids:
            correct = es.ticket_ids & incident_tickets
            checks.append(len(correct) / len(es.ticket_ids))

        precision_scores.append(mean(checks) if checks else 1.0)
        hallucinated_names = es.person_names - canonical_names
        if es.person_names:
            hallucination_scores.append(
                1.0 - len(hallucinated_names) / len(es.person_names)
            )
        else:
            hallucination_scores.append(1.0)

    avg_precision = mean(precision_scores) if precision_scores else 1.0
    avg_no_hallucination = mean(hallucination_scores) if hallucination_scores else 1.0

    return {
        "tech": round(avg_precision, 4),
        "names": round(avg_no_hallucination, 4),
        "tickets": 1.0,
        "overall": round(mean([avg_precision, avg_no_hallucination]), 4),
    }


def entity_agreement(entity_sets: list[EntitySet]) -> dict[str, float]:
    if len(entity_sets) < 2:
        return {"tech": 1.0, "names": 1.0, "tickets": 1.0, "overall": 1.0}
    tech_scores = []
    name_scores = []
    ticket_scores = []
    for s1, s2 in combinations(entity_sets, 2):
        tech_scores.append(_jaccard(s1.tech_components, s2.tech_components))
        name_scores.append(_jaccard(s1.person_names, s2.person_names))
        if s1.ticket_ids or s2.ticket_ids:
            ticket_scores.append(_jaccard(s1.ticket_ids, s2.ticket_ids))
    avg_tickets = mean(ticket_scores) if ticket_scores else 1.0
    avg_tech = mean(tech_scores)
    avg_names = mean(name_scores)
    return {
        "tech": avg_tech,
        "names": avg_names,
        "tickets": avg_tickets,
        "overall": mean([avg_tech, avg_names, avg_tickets]),
    }


_AFTER_OPEN_SIM_TYPES: frozenset[str] = frozenset(
    {"postmortem_created", "incident_resolved"}
)


def _stratified_pairs(
    artifacts: list[ArtifactDoc],
    max_pairs: int = 10,
    seed: int = 42,
) -> list[tuple[ArtifactDoc, ArtifactDoc]]:
    by_type: dict[str, list[ArtifactDoc]] = {}
    for a in artifacts:
        by_type.setdefault(a.artifact_type, []).append(a)

    cross_type: list[tuple[ArtifactDoc, ArtifactDoc]] = []
    types = list(by_type.keys())
    for i, t1 in enumerate(types):
        for t2 in types[i + 1 :]:
            for a in by_type[t1]:
                for b in by_type[t2]:
                    cross_type.append((a, b))

    same_type: list[tuple[ArtifactDoc, ArtifactDoc]] = []
    for docs in by_type.values():
        same_type.extend(combinations(docs, 2))

    rng = random.Random(seed)
    rng.shuffle(cross_type)
    rng.shuffle(same_type)

    pairs = cross_type[:max_pairs]
    if len(pairs) < max_pairs:
        pairs += same_type[: max_pairs - len(pairs)]
    return pairs


def temporal_violations(artifacts: list[ArtifactDoc]) -> int:
    parsed: list[tuple[datetime, ArtifactDoc]] = []

    for a in artifacts:
        if not a.timestamp:
            continue
        try:
            ts = datetime.fromisoformat(a.timestamp.replace("Z", "+00:00"))
            parsed.append((ts, a))
        except ValueError:
            continue

    if not parsed:
        return 0

    anchor_ts = next(
        (ts for ts, a in parsed if a.artifact_type == "jira"),
        min(ts for ts, _ in parsed),
    )

    violations = 0
    for ts, art in parsed:
        title_lower = art.title.lower()
        is_causal_after_open = (
            art.sim_event_type in _AFTER_OPEN_SIM_TYPES
            or "postmortem" in title_lower
            or "resolv" in title_lower
        )
        if is_causal_after_open and ts < anchor_ts:
            violations += 1

    return violations


JUDGE_PROMPT = """You are a factual consistency auditor comparing two documents about the same incident.

Your task: identify FACTUAL CONTRADICTIONS where the documents make mutually exclusive claims 
about the same concrete fact.

STRICT RULES:
- "doc_a_quote" and "doc_b_quote" must be VERBATIM text copied from the document.
  Copy the exact words. Do not paraphrase, summarize, or introduce any term not present 
  in the source text. If you cannot find an exact quote that supports the contradiction, 
  do not report it.
- Only flag claims that DIRECTLY CONTRADICT each other (both cannot be true simultaneously).
- Do NOT flag: omissions, different levels of detail, different perspectives, different 
  writing styles, or additional context in one document that the other lacks.
- Do NOT flag different specific values (e.g., permission names like s3:PutObject vs 
  s3:PutObjectAcl) when both are plausible specifics of the same underlying concept and 
  neither document claims the other is wrong.
- Specific technical identifiers (IAM actions, error codes, API names) vary legitimately 
  across document types. Only flag if one document explicitly contradicts the other's 
  specific claim.

WHAT TO FLAG:
- Person A credited with an action in Doc A, Person B credited with the same action in Doc B
- "resolved in 2 days" vs "resolved in 5 days"  
- "service was down" vs "service was degraded but available"
- Explicit root cause statements that are mutually exclusive

Document A ({id_a}):
{content_a}

Document B ({id_b}):
{content_b}

Respond with a JSON object. The "contradictions" array must contain only entries where 
you can provide a verbatim quote from each document:

{{
  "contradictions": [
    {{
      "field": "brief label for what fact is contradicted",
      "doc_a_quote": "exact verbatim text from Document A",
      "doc_b_quote": "exact verbatim text from Document B"
    }}
  ]
}}

If no qualifying contradictions exist, return {{"contradictions": []}}.
JSON only. No preamble."""


def judge_contradictions(
    bedrock: Bedrock,
    bedrock_judge: Bedrock,
    artifacts: list[ArtifactDoc],
    max_pairs: int = 10,
    sleep_between_calls: float = 2.0,
) -> int:
    if len(artifacts) < 2:
        return 0
    pairs = _stratified_pairs(artifacts, max_pairs=max_pairs)
    total = 0
    for a, b in pairs:
        content_a = a.content[:3000]
        content_b = b.content[:3000]
        prompt = JUDGE_PROMPT.format(
            id_a=a.artifact_id,
            content_a=content_a,
            id_b=b.artifact_id,
            content_b=content_b,
        )
        try:
            raw = bedrock_judge.generate(prompt, temperature=0.0, max_tokens=2048)
            clean = raw.strip()
            if clean.startswith("```"):
                clean = re.sub(r"^```\w*\n?", "", clean).rstrip("`\n ")
            parsed = json.loads(clean)
            count = parsed.get("count", 0)
            total += count
            if count > 0:
                for c in parsed.get("contradictions", []):
                    logger.info(
                        f"  Contradiction: {c.get('field', '?')} — "
                        f"A says '{c.get('doc_a_says', '?')}', "
                        f"B says '{c.get('doc_b_says', '?')}'"
                    )
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning(
                f"  Judge parse failed for {a.artifact_id}↔{b.artifact_id}: {exc}"
            )
        except Exception as exc:
            logger.warning(f"  Judge call failed: {exc}")
        time.sleep(sleep_between_calls)
    return total


ARTIFACT_SEQUENCE = [
    ("jira", "JIRA Ticket", "Write a JIRA ticket for this incident."),
    ("slack", "Slack Thread", "Write a Slack #incidents thread about this incident."),
    (
        "pr",
        "Pull Request",
        "Write a GitHub PR description for the fix to this incident.",
    ),
    (
        "postmortem",
        "Postmortem",
        "Write a Confluence postmortem document for this incident.",
    ),
    (
        "email",
        "Customer Email",
        "Write an inbound customer complaint email triggered by this incident.",
    ),
]


def _baseline_system_prompt(tech_stack: str, org_chart_str: str, company: str) -> str:
    return (
        f"You are generating realistic organizational documents for {company}.\n"
        f"Tech stack:\n{tech_stack}\n\n"
        f"Team members:\n{org_chart_str}\n\n"
        f"Use ONLY the system names and person names provided above. "
        f"Do not invent any names not in this list."
    )


_BASELINE_SIM_EVENT_TYPES: dict[str, str] = {
    "jira": "incident_opened",
    "slack": "async_question",
    "pr": "pr_review",
    "postmortem": "postmortem_created",
    "email": "inbound_external_email",
}

_SCORING_TYPE_MAP: dict[str, str] = {
    "jira": "jira",
    "slack": "slack",
    "pr": "pr",
    "postmortem": "confluence",
    "email": "email",
}


def generate_baseline_artifacts(
    bedrock: Bedrock,
    incident: IncidentBundle,
    tech_stack_str: str,
    org_chart_str: str,
    company: str,
    chained: bool,
    rng: random.Random | None = None,
) -> list[ArtifactDoc]:
    _rng = rng or random.Random()
    system = _baseline_system_prompt(tech_stack_str, org_chart_str, company)
    context_prefix = (
        f"Incident: {incident.root_cause}\n"
        f"System fault: {incident.system_fault}\n"
        f"On-call engineer: {incident.on_call}\n"
        f"Duration: {incident.duration_days} days\n"
        f"System health at open: {incident.health_at_open}/100\n\n"
    )

    baseline_facts: dict[str, Any] = {
        "root_cause": incident.root_cause,
        "affected_system": incident.system_fault,
        "assigned_to": incident.on_call,
        "actors": incident.actors,
        "incident_id": incident.incident_id,
    }
    if incident.duration_days:
        baseline_facts["incident_duration_hours"] = incident.duration_days * 24
    baseline_facts.update(
        {k: v for k, v in incident.sim_event_facts.items() if k not in baseline_facts}
    )

    JIRA_ANCHOR_TS = "2026-03-10T09:00:00"  # fixed incident open anchor

    prior_artifacts: list[str] = []
    results: list[ArtifactDoc] = []

    for idx, (art_type, art_label, instruction) in enumerate(ARTIFACT_SEQUENCE):
        prompt = f"{context_prefix}{instruction}\n\n"
        if chained and prior_artifacts:
            prompt += "Previously generated documents for this incident:\n\n"
            for prior in prior_artifacts:
                prompt += f"{prior}\n\n---\n\n"
            prompt += (
                "Your document must be consistent with all of the above. "
                "Use the same names, systems, and timeline.\n\n"
            )
        prompt += (
            f"Write the {art_label} now. Output only the document content, "
            f"no preamble or meta-commentary."
        )
        try:
            content = bedrock.generate(prompt, system=system, max_tokens=2048)
        except Exception as exc:
            logger.warning(f"  Baseline generation failed for {art_type}: {exc}")
            content = f"[Generation failed: {exc}]"

        if art_type == "jira":
            fake_ts = JIRA_ANCHOR_TS
        elif chained:
            fake_ts = f"2026-03-{10 + idx:02d}T{9 + idx:02d}:00:00"
        else:
            day_delta = _rng.randint(-2, 5)
            hour = _rng.randint(8, 20)
            minute = _rng.randint(0, 59)
            fake_ts = f"2026-03-{10 + day_delta:02d}T{hour:02d}:{minute:02d}:00"

        doc = ArtifactDoc(
            artifact_id=f"baseline_{art_type}_{incident.incident_id}",
            artifact_type=_SCORING_TYPE_MAP[art_type],
            title=f"{art_label}: {incident.root_cause[:60]}",
            content=content,
            timestamp=fake_ts,
            author=incident.on_call,
            sim_event_type=_BASELINE_SIM_EVENT_TYPES[art_type],
            sim_event_facts=baseline_facts,
        )
        results.append(doc)
        if chained:
            prior_artifacts.append(f"[{art_label}]\n{content}")

    return results


SCORING_ARTIFACT_TYPES = {"jira", "slack", "pr", "confluence"}


def evaluate_arm(
    arm_name: str,
    incidents_with_artifacts: list[tuple[IncidentBundle, list[ArtifactDoc]]],
    extractor: EntityExtractor,
    bedrock: Bedrock | None,
    bedrock_judge: Bedrock | None,
    nli: NLIScorer | None = None,
    vendor_aliases: dict[str, set[str]] | None = None,
    skip_judge: bool = False,
    skip_nli: bool = False,
) -> EvalResult:
    all_agreement: list[float] = []
    all_violations: list[int] = []
    all_contradictions: list[int] = []
    all_divergence: list[float] = []
    per_incident: list[dict] = []

    for incident, artifacts in incidents_with_artifacts:
        if not artifacts:
            continue

        scoring_artifacts = [
            a for a in artifacts if a.artifact_type in SCORING_ARTIFACT_TYPES
        ]
        entity_sets = [extractor.extract(a) for a in scoring_artifacts]

        if arm_name == "OrgForge":
            agreement = entity_agreement_grounded(
                entity_sets,
                incident,
                extractor.canonical_tech,
                extractor.canonical_names,
            )
        else:
            agreement = entity_agreement(entity_sets)
        violations = temporal_violations(scoring_artifacts)

        contradictions = 0
        if not skip_judge and bedrock_judge and bedrock and len(scoring_artifacts) >= 2:
            logger.info(
                f"  Judging {arm_name}/{incident.incident_id} "
                f"({len(artifacts)} artifacts)..."
            )
            contradictions = judge_contradictions(
                bedrock,
                bedrock_judge,
                scoring_artifacts,
            )

        divergence_score = 1.0
        divergence_detail: list[dict] = []
        per_type_divergence: dict = {}

        DIVERGENCE_ARTIFACT_TYPES = {"jira", "slack", "pr", "confluence"}

        if not skip_nli:
            facts_artifacts = [
                a
                for a in scoring_artifacts
                if a.sim_event_facts and a.artifact_type in DIVERGENCE_ARTIFACT_TYPES
            ]
            if facts_artifacts:
                logger.info(
                    f"  NLI divergence {arm_name}/{incident.incident_id} "
                    f"({len(facts_artifacts)} artifacts with facts)..."
                )
                reports = [
                    measure_artifact_divergence(
                        a.artifact_id,
                        a.artifact_type,
                        a.content,
                        a.sim_event_facts,
                        nli,
                        vendor_aliases=vendor_aliases,
                        sim_event_type=a.sim_event_type,
                    )
                    for a in facts_artifacts
                ]
                divergence_score = mean(r.composite_score for r in reports)
                for r in reports:
                    status = "⚠" if r.composite_score < 0.7 else "✓"
                    logger.info(
                        f"    {status} {r.artifact_id}: "
                        f"composite={r.composite_score:.3f} "
                        f"entity={r.entity_score:.3f} "
                        f"nli={r.nli_score:.3f} "
                        f"numeric={r.numeric_score:.3f} "
                        f"({len(r.divergences)} divergences)"
                    )
                    for d in r.divergences:
                        logger.info(f"      → {d}")
                type_scores: dict[str, list[dict]] = {}
                for r in reports:
                    atype = next(
                        (
                            a.artifact_type
                            for a in facts_artifacts
                            if a.artifact_id == r.artifact_id
                        ),
                        "unknown",
                    )
                    type_scores.setdefault(atype, []).append(
                        {
                            "s_ent": r.entity_score,
                            "s_nli": r.nli_score,
                            "s_num": r.numeric_score,
                            "composite": r.composite_score,
                        }
                    )

                per_type_divergence = {
                    atype: {
                        "s_ent": round(mean(v["s_ent"] for v in scores), 4),
                        "s_nli": round(mean(v["s_nli"] for v in scores), 4),
                        "s_num": round(mean(v["s_num"] for v in scores), 4),
                        "composite": round(mean(v["composite"] for v in scores), 4),
                        "n": len(scores),
                    }
                    for atype, scores in type_scores.items()
                }

                flagged = [r for r in reports if r.composite_score < 0.7]
                divergence_detail = [
                    {
                        "artifact_id": r.artifact_id,
                        "composite": round(r.composite_score, 3),
                        "entity": round(r.entity_score, 3),
                        "nli": round(r.nli_score, 3),
                        "numeric": round(r.numeric_score, 3),
                        "n_divergences": len(r.divergences),
                    }
                    for r in flagged
                ]
                logger.info(
                    f"  Divergence {arm_name}/{incident.incident_id}: "
                    f"{divergence_score:.3f} ({len(facts_artifacts)} scored, "
                    f"{len(flagged)} flagged)"
                )
            else:
                logger.warning(
                    f"  No sim_event_facts found for {arm_name}/{incident.incident_id} "
                    f"— skipping divergence check"
                )

        all_agreement.append(agreement["overall"])
        all_violations.append(violations)
        all_contradictions.append(contradictions)
        all_divergence.append(divergence_score)

        per_incident.append(
            {
                "incident_id": incident.incident_id,
                "n_artifacts": len(artifacts),
                "agreement": agreement,
                "temporal_violations": violations,
                "contradictions": contradictions,
                "prose_divergence": round(divergence_score, 4),
                "divergence_flagged": divergence_detail,
                "divergence_by_type": per_type_divergence,
            }
        )

    return EvalResult(
        arm=arm_name,
        entity_agreement=round(mean(all_agreement), 4) if all_agreement else 0.0,
        temporal_violations=round(mean(all_violations), 2) if all_violations else 0.0,
        contradictions=round(mean(all_contradictions), 2)
        if all_contradictions
        else 0.0,
        prose_divergence=round(mean(all_divergence), 4) if all_divergence else 1.0,
        n_incidents=len(per_incident),
        per_incident=per_incident,
    )


def print_results(results: list[EvalResult]) -> None:
    col_w = 22
    header = f"{'Metric':<35}"
    for r in results:
        header += f"{r.arm:>{col_w}}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    for label, attr, fmt in [
        ("Entity agreement rate", "entity_agreement", ".4f"),
        ("Temporal violations / incident", "temporal_violations", ".2f"),
        ("Contradictions / incident (judge)", "contradictions", ".2f"),
        ("Prose-SimEvent divergence", "prose_divergence", ".4f"),
    ]:
        row = f"{label:<35}"
        for r in results:
            row += f"{getattr(r, attr):>{col_w}{fmt}}"
        print(row)

    row_n = f"{'Incidents evaluated':<35}"
    for r in results:
        row_n += f"{r.n_incidents:>{col_w}d}"
    print(row_n)

    print("=" * len(header))

    for r in results:
        print(f"\n── {r.arm} per-incident detail ──")
        for p in r.per_incident:
            print(
                f"  {p['incident_id']}: "
                f"agreement={p['agreement']['overall']:.3f}  "
                f"violations={p['temporal_violations']}  "
                f"contradictions={p['contradictions']}  "
                f"divergence={p['prose_divergence']:.3f}  "
                f"artifacts={p['n_artifacts']}"
            )
            for d in p.get("divergence_flagged", []):
                print(
                    f"    ⚠ {d['artifact_id']}: "
                    f"composite={d['composite']:.3f} "
                    f"entity={d['entity']:.3f} "
                    f"nli={d['nli']:.3f} "
                    f"numeric={d['numeric']:.3f} "
                    f"({d['n_divergences']} divergences)"
                )


def save_results(results: list[EvalResult], path: str = "eval_results.json") -> None:
    out = []
    for r in results:
        out.append(
            {
                "arm": r.arm,
                "entity_agreement": r.entity_agreement,
                "temporal_violations": r.temporal_violations,
                "contradictions": r.contradictions,
                "prose_divergence": r.prose_divergence,
                "n_incidents": r.n_incidents,
                "per_incident": r.per_incident,
            }
        )
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"Results saved to {path}")


def debug_entity_extraction(
    incidents: list[IncidentBundle],
    extractor: EntityExtractor,
) -> None:
    for inc in incidents:
        print(f"\n{'=' * 80}")
        print(f"INCIDENT: {inc.incident_id} ({len(inc.artifacts)} artifacts)")
        print(f"{'=' * 80}")
        print(
            f"\nTECH COMPONENTS ({len(extractor._tech)}): {sorted(extractor._tech.values())[:20]}"
        )
        print(
            f"ORG NAMES ({len(extractor._names)}): {sorted(extractor._names.values())}"
        )
        for art in inc.artifacts:
            entity_set = extractor.extract(art)
            print(f"\n  [{art.artifact_type.upper()}] {art.artifact_id}")
            print(f"  sim_event_type : {art.sim_event_type}")
            print(f"  sim_event_facts: {len(art.sim_event_facts)} keys")
            print(f"  title    : {art.title[:80]}")
            print(f"  timestamp: {art.timestamp}")
            print(f"  content  : {art.content[:200].replace(chr(10), ' ')!r}")
            print(f"  tech     : {sorted(entity_set.tech_components)}")
            print(f"  names    : {sorted(entity_set.person_names)}")
            print(f"  tickets  : {sorted(entity_set.ticket_ids)}")


def main():
    parser = argparse.ArgumentParser(
        description="OrgForge cross-document consistency evaluation",
    )
    parser.add_argument("--model", default="openai.gpt-oss-120b-1:0")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--db", default="orgforge")
    parser.add_argument(
        "--mongo-uri", default="mongodb://localhost:27017?directConnection=true"
    )
    parser.add_argument("--incidents", type=int, default=5)
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument(
        "--skip-nli",
        action="store_true",
        help="Skip NLI-based prose-SimEvent divergence checks",
    )
    parser.add_argument(
        "--nli-model",
        default="cross-encoder/nli-deberta-v3-base",
        help="HuggingFace NLI model for divergence scoring",
    )
    parser.add_argument("--output", default="eval_results.json")
    parser.add_argument("--judge-model", default="us.anthropic.claude-opus-4-6-v1")
    parser.add_argument("--debug-entities", action="store_true")
    args = parser.parse_args()

    # ── NLI model ────────────────────────────────────────────────────────
    nli: NLIScorer | None = None
    if not args.skip_nli:
        logger.info(f"Loading NLI model: {args.nli_model}")
        nli = NLIScorer(model_name=args.nli_model)
        logger.info("NLI model loaded")
    else:
        logger.info("NLI divergence checks disabled (--skip-nli)")

    mongo = MongoExtractor(db_name=args.db, uri=args.mongo_uri)
    bedrock = Bedrock(model_id=args.model, region=args.region)
    bedrock_judge = Bedrock(model_id=args.judge_model or args.model, region=args.region)

    logger.info("Loading canonical entities from MongoDB...")
    tech_components = mongo.tech_stack_components()
    org_names = mongo.org_names()
    logger.info(
        f"  Tech components: {len(tech_components)} | Org names: {len(org_names)}"
    )

    extractor = EntityExtractor(tech_components, org_names)

    logger.info(f"Extracting up to {args.incidents} incidents from OrgForge...")
    incidents = mongo.extract_incidents(max_n=args.incidents)

    if args.debug_entities:
        debug_entity_extraction(incidents, extractor)
        sys.exit(0)

    if not incidents:
        logger.error("No incidents found in MongoDB. Run OrgForge first.")
        sys.exit(1)

    for inc in incidents:
        facts_count = sum(1 for a in inc.artifacts if a.sim_event_facts)
        logger.info(
            f"  {inc.incident_id}: {inc.root_cause[:80]} "
            f"({len(inc.artifacts)} artifacts, {facts_count} with facts, "
            f"{inc.duration_days}d)"
        )

    # ── Arm 1: OrgForge ──────────────────────────────────────────────────
    logger.info("\n━━━ Evaluating OrgForge arm ━━━")
    orgforge_data = [(inc, inc.artifacts) for inc in incidents]
    vendor_aliases_map = mongo.vendor_aliases()

    orgforge_result = evaluate_arm(
        "OrgForge",
        orgforge_data,
        extractor,
        bedrock,
        bedrock_judge,
        nli=nli,
        vendor_aliases=vendor_aliases_map,
        skip_judge=args.skip_judge,
        skip_nli=args.skip_nli,
    )
    results = [orgforge_result]

    if not args.skip_baselines:
        tech_stack_str = ""
        ts_doc = mongo._db["artifacts"].find_one({"type": "tech_stack"})
        if ts_doc:
            content = ts_doc.get("content", "")
            tech_stack_str = (
                json.dumps(content, indent=2)
                if isinstance(content, dict)
                else str(content)
            )
        org_chart_str = "\n".join(sorted(org_names))
        company = COMPANY_NAME

        # ── Arm 2: Chained baseline ──────────────────────────────────────
        logger.info("\n━━━ Generating chained baseline ━━━")
        chained_data: list[tuple[IncidentBundle, list[ArtifactDoc]]] = []
        for inc in incidents:
            logger.info(f"  Generating chained artifacts for {inc.incident_id}...")
            arts = mongo.load_cached_baseline("chained", inc.incident_id, args.model)
            if arts is None:
                arts = generate_baseline_artifacts(
                    bedrock,
                    inc,
                    tech_stack_str,
                    org_chart_str,
                    company,
                    chained=True,
                )
                mongo.save_cached_baseline("chained", inc.incident_id, args.model, arts)
            else:
                logger.info(f"  Using cached chained baseline for {inc.incident_id}")
            chained_data.append((inc, arts))

        logger.info("\n━━━ Evaluating chained baseline ━━━")
        chained_result = evaluate_arm(
            "Chained",
            chained_data,
            extractor,
            bedrock,
            bedrock_judge,
            nli=nli,
            skip_judge=args.skip_judge,
            skip_nli=True,
        )
        results.append(chained_result)

        # ── Arm 3: Parallel baseline ─────────────────────────────────────
        logger.info("\n━━━ Generating parallel baseline ━━━")
        parallel_data: list[tuple[IncidentBundle, list[ArtifactDoc]]] = []
        for inc in incidents:
            logger.info(f"  Generating parallel artifacts for {inc.incident_id}...")
            arts = mongo.load_cached_baseline("parallel", inc.incident_id, args.model)
            if arts is None:
                arts = generate_baseline_artifacts(
                    bedrock,
                    inc,
                    tech_stack_str,
                    org_chart_str,
                    company,
                    chained=False,
                )
                mongo.save_cached_baseline(
                    "parallel", inc.incident_id, args.model, arts
                )
            else:
                logger.info(f"  Using cached parallel baseline for {inc.incident_id}")
            parallel_data.append((inc, arts))

        logger.info("\n━━━ Evaluating parallel baseline ━━━")
        parallel_result = evaluate_arm(
            "Parallel",
            parallel_data,
            extractor,
            bedrock,
            bedrock_judge,
            nli=nli,
            skip_judge=args.skip_judge,
            skip_nli=True,
        )
        results.append(parallel_result)

    print_results(results)
    save_results(results, args.output)


if __name__ == "__main__":
    main()
