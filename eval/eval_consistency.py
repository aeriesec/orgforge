"""
eval_consistency.py
===================
Cross-document consistency comparison: OrgForge vs. LLM-only baselines.

Metric: Ground Truth Fidelity
------------------------------
Instead of comparing artifacts against each other (Jaccard overlap), each
artifact is scored against the SimEvent ground truth that produced it.

For each ground truth field (on_call, root_cause, system_fault, ticket_id)
we compute TOKEN RECALL: what fraction of the field's significant tokens
appear in the artifact prose. This catches paraphrase while still failing
on genuine drift or hallucination.

Results are broken down per artifact type (jira, slack, pr, confluence, email)
so document-type coverage differences don't pollute the signal.

Usage:
    python eval_consistency.py --model anthropic.claude-3-5-sonnet-20241022-v2:0
    python eval_consistency.py --incidents 3 --region us-west-2
    python eval_consistency.py --db orgforge --skip-baselines
    python eval_consistency.py --skip-judge   # fidelity only, no contradiction judge
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations
from statistics import mean
import time
from typing import Any

import boto3
from config_loader import COMPANY_NAME
from pymongo import MongoClient

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("eval_consistency")


# ── Stopwords (excluded from token recall) ───────────────────────────────────

_STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "from",
    "is",
    "was",
    "are",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "that",
    "this",
    "these",
    "those",
    "it",
    "its",
    "which",
    "who",
    "what",
    "when",
    "where",
    "how",
    "not",
    "no",
    "nor",
    "so",
    "yet",
    "both",
    "either",
    "neither",
    "each",
    "than",
    "such",
    "as",
    "if",
    "then",
    "than",
    "too",
    "very",
    "just",
    "also",
}


# ── Bedrock wrapper ──────────────────────────────────────────────────────────


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


# ── Data model ───────────────────────────────────────────────────────────────


@dataclass
class ArtifactDoc:
    artifact_id: str
    artifact_type: str
    title: str
    content: str
    timestamp: str
    author: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class IncidentBundle:
    incident_id: str
    root_cause: str
    system_fault: str
    on_call: str
    day: int
    duration_days: int
    health_at_open: int
    artifacts: list[ArtifactDoc] = field(default_factory=list)
    sim_event_facts: dict = field(default_factory=dict)


@dataclass
class FieldFidelity:
    """Token recall score for one ground truth field in one artifact."""

    field_name: str
    ground_truth: str
    recall: float  # fraction of GT tokens found in artifact prose
    tokens_expected: int
    tokens_found: int


@dataclass
class ArtifactFidelity:
    """Fidelity scores for a single artifact against its incident ground truth."""

    artifact_id: str
    artifact_type: str
    fields: list[FieldFidelity] = field(default_factory=list)

    @property
    def overall(self) -> float:
        if not self.fields:
            return 1.0
        return mean(f.recall for f in self.fields)

    def by_field(self) -> dict[str, float]:
        return {f.field_name: f.recall for f in self.fields}


@dataclass
class IncidentFidelityResult:
    incident_id: str
    artifacts: list[ArtifactFidelity] = field(default_factory=list)

    @property
    def overall(self) -> float:
        if not self.artifacts:
            return 1.0
        return mean(a.overall for a in self.artifacts)

    def by_artifact_type(self) -> dict[str, float]:
        """Mean fidelity grouped by artifact type."""
        buckets: dict[str, list[float]] = defaultdict(list)
        for a in self.artifacts:
            buckets[a.artifact_type].append(a.overall)
        return {t: mean(scores) for t, scores in buckets.items()}

    def by_field(self) -> dict[str, float]:
        """Mean recall per ground truth field across all artifacts."""
        buckets: dict[str, list[float]] = defaultdict(list)
        for a in self.artifacts:
            for f in a.fields:
                buckets[f.field_name].append(f.recall)
        return {fname: mean(scores) for fname, scores in buckets.items()}


@dataclass
class EvalResult:
    arm: str
    n_incidents: int
    overall_fidelity: float
    fidelity_by_field: dict[str, float]
    fidelity_by_artifact_type: dict[str, float]
    contradictions: float  # mean per incident, 0.0 if judge skipped
    per_incident: list[dict] = field(default_factory=list)


# ── Ground truth fidelity ────────────────────────────────────────────────────


def _tokenize(text: str) -> set[str]:
    """Lower-case alpha tokens, length >= 3, excluding stopwords."""
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    return {t for t in tokens if len(t) >= 3 and t not in _STOPWORDS}


def _token_recall(ground_truth: str, artifact_text: str) -> tuple[float, int, int]:
    """
    What fraction of significant GT tokens appear in the artifact?
    Returns (recall, n_expected, n_found).
    Empty GT is treated as perfect (field not applicable).
    """
    gt_tokens = _tokenize(ground_truth)
    if not gt_tokens:
        return 1.0, 0, 0
    art_tokens = _tokenize(artifact_text)
    found = gt_tokens & art_tokens
    return len(found) / len(gt_tokens), len(gt_tokens), len(found)


# Ground truth fields to check, and which artifact types are expected to
# mention them. Artifact types NOT in the set are skipped for that field
# (e.g. a customer email is not expected to name the on-call engineer).
#
# Rationale for per-type scoping:
#   on_call      — operational docs (jira, slack, pr) name the responder;
#                  postmortems sometimes do, emails almost never should.
#   root_cause   — all internal docs should reflect it; customer email gets
#                  a sanitised version so we allow it but weight it lower.
#   system_fault — same as root_cause.
#   ticket_id    — internal cross-references; email unlikely to contain it.

FIELD_SCOPE: dict[str, set[str]] = {
    "on_call": {"jira", "jira_comment", "slack", "pr", "confluence"},
    "root_cause": {
        "jira",
        "jira_comment",
        "slack",
        "pr",
        "confluence",
        "postmortem",
        "email",
    },
    "system_fault": {
        "jira",
        "jira_comment",
        "slack",
        "pr",
        "confluence",
        "postmortem",
        "email",
    },
    "ticket_id": {"jira", "jira_comment", "slack", "pr", "confluence", "postmortem"},
}


def score_artifact_fidelity(
    artifact: ArtifactDoc,
    incident: IncidentBundle,
) -> ArtifactFidelity:
    """
    Score a single artifact against the incident's SimEvent ground truth.
    Only fields scoped to this artifact type are evaluated.
    """
    gt_fields = {
        "on_call": incident.on_call,
        "root_cause": incident.root_cause,
        "system_fault": incident.system_fault,
        "ticket_id": incident.incident_id,
    }

    result = ArtifactFidelity(
        artifact_id=artifact.artifact_id,
        artifact_type=artifact.artifact_type,
    )

    for field_name, gt_value in gt_fields.items():
        scoped_types = FIELD_SCOPE.get(field_name, set())
        if artifact.artifact_type not in scoped_types:
            continue
        if not gt_value:
            continue

        recall, n_expected, n_found = _token_recall(gt_value, artifact.content)
        result.fields.append(
            FieldFidelity(
                field_name=field_name,
                ground_truth=gt_value,
                recall=recall,
                tokens_expected=n_expected,
                tokens_found=n_found,
            )
        )

    return result


def score_incident_fidelity(
    incident: IncidentBundle,
    artifacts: list[ArtifactDoc],
) -> IncidentFidelityResult:
    result = IncidentFidelityResult(incident_id=incident.incident_id)
    for artifact in artifacts:
        af = score_artifact_fidelity(artifact, incident)
        if af.fields:  # skip artifacts with no applicable fields
            result.artifacts.append(af)
    return result


# ── MongoDB extraction ───────────────────────────────────────────────────────


class MongoExtractor:
    def __init__(
        self,
        db_name: str = "orgforge",
        uri: str = "mongodb://localhost:27017?directConnection=true",
    ):
        self._client = MongoClient(uri)
        self._db = self._client[db_name]

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
                sim_event_facts=facts,
            )

            jira_doc = self._db["jira_tickets"].find_one(
                {"id": ticket_id}, {"causal_chain": 1, "bot_threads": 1}
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

        # ── Jira ticket + comments ────────────────────────────────────────────
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
            docs.append(
                ArtifactDoc(
                    artifact_id=ticket_id,
                    artifact_type="jira",
                    title=jira.get("title", ticket_id),
                    content="\n".join(filter(None, body_parts)),
                    timestamp=jira.get("created_at", ""),
                    author=jira.get("assignee", ""),
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
                    )
                )

        # ── Causal chain ──────────────────────────────────────────────────────
        for chain_id in causal_chain:
            if chain_id in seen or chain_id in bot_threads:
                continue
            if re.match(r"^[A-Z]+-\d+_comment_\d+$", chain_id):
                seen.add(chain_id)
                continue
            seen.add(chain_id)
            _try_append(self._resolve_artifact_by_id(chain_id))

        # ── Related events ────────────────────────────────────────────────────
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
            for key, raw_aid in ev.get("artifact_ids", {}).items():
                aids = raw_aid if isinstance(raw_aid, list) else [raw_aid]
                for aid in aids:
                    if not aid or aid in seen or key in ("jira",):
                        continue
                    seen.add(aid)
                    art_doc = self._resolve_artifact(aid, key, ev)
                    if art_doc and not _is_noise(art_doc):
                        docs.append(art_doc)

        # ── Postmortem ────────────────────────────────────────────────────────
        postmortem_ev = self._db["events"].find_one(
            {"type": "postmortem_created", "artifact_ids.jira": ticket_id}
        )
        if postmortem_ev:
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
                    _try_append(self._resolve_artifact_by_id(conf_id))

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
                comment_parts = [
                    f"{c.get('author', '')}: {c.get('text', '')}"
                    for c in doc.get("comments", [])
                ]
                content = "\n".join(
                    filter(
                        None,
                        [
                            doc.get("title", ""),
                            f"Author: {author}" if author else "",
                            f"Reviewers: {reviewers}" if reviewers else "",
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


# ── LLM-as-judge (contradiction detection) ───────────────────────────────────
# Now used as a *second layer* on top of fidelity: only run if fidelity flags
# a potential issue, or on demand. Still available for full pairwise sweeps.

JUDGE_PROMPT = """You are a factual consistency auditor. You will be given two documents
that describe the same organizational incident. Your job is to identify
FACTUAL CONTRADICTIONS — places where the two documents disagree on concrete
facts (names, systems, timelines, root causes, outcomes).

Do NOT flag:
- Different levels of detail (one doc says more than the other)
- Different writing styles or perspectives
- Omissions (one doc doesn't mention something the other does)

DO flag:
- Different system/component names for the same failure
- Different people credited with the same action
- Contradictory timelines (e.g., "resolved in 2 days" vs. "resolved in 5 days")
- Different root causes for the same incident

Document A ({id_a}):
{content_a}

Document B ({id_b}):
{content_b}

Respond with a JSON object:
{{
  "contradictions": [
    {{"field": "string", "doc_a_says": "string", "doc_b_says": "string"}}
  ],
  "count": int
}}

If no contradictions, return {{"contradictions": [], "count": 0}}.
JSON only. No preamble."""


def judge_contradictions(
    bedrock_judge: Bedrock,
    artifacts: list[ArtifactDoc],
    max_pairs: int = 10,
    sleep_between_calls: float = 2.0,
) -> int:
    if len(artifacts) < 2:
        return 0

    pairs = list(combinations(artifacts, 2))[:max_pairs]
    total = 0

    for a, b in pairs:
        prompt = JUDGE_PROMPT.format(
            id_a=a.artifact_id,
            content_a=a.content[:3000],
            id_b=b.artifact_id,
            content_b=b.content[:3000],
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


# ── Baseline generation ──────────────────────────────────────────────────────

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


def generate_baseline_artifacts(
    bedrock: Bedrock,
    incident: IncidentBundle,
    tech_stack_str: str,
    org_chart_str: str,
    company: str,
    chained: bool,
) -> list[ArtifactDoc]:
    system = _baseline_system_prompt(tech_stack_str, org_chart_str, company)
    context_prefix = (
        f"Incident: {incident.root_cause}\n"
        f"System fault: {incident.system_fault}\n"
        f"On-call engineer: {incident.on_call}\n"
        f"Duration: {incident.duration_days} days\n"
        f"System health at open: {incident.health_at_open}/100\n\n"
    )

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

        fake_ts = f"2026-03-{10 + idx:02d}T{9 + idx:02d}:00:00"
        doc = ArtifactDoc(
            artifact_id=f"baseline_{art_type}_{incident.incident_id}",
            artifact_type=art_type,
            title=f"{art_label}: {incident.root_cause[:60]}",
            content=content,
            timestamp=fake_ts,
            author=incident.on_call,
        )
        results.append(doc)

        if chained:
            prior_artifacts.append(f"[{art_label}]\n{content}")

    return results


# ── Evaluation orchestrator ──────────────────────────────────────────────────

SCORING_ARTIFACT_TYPES = {
    "jira",
    "jira_comment",
    "slack",
    "pr",
    "confluence",
    "postmortem",
    "email",
}


def evaluate_arm(
    arm_name: str,
    incidents_with_artifacts: list[tuple[IncidentBundle, list[ArtifactDoc]]],
    bedrock_judge: Bedrock | None,
    skip_judge: bool = False,
) -> EvalResult:
    all_fidelity: list[float] = []
    all_contradictions: list[int] = []
    field_buckets: dict[str, list[float]] = defaultdict(list)
    type_buckets: dict[str, list[float]] = defaultdict(list)
    per_incident: list[dict] = []

    for incident, artifacts in incidents_with_artifacts:
        if not artifacts:
            continue

        scoring_artifacts = [
            a for a in artifacts if a.artifact_type in SCORING_ARTIFACT_TYPES
        ]
        if not scoring_artifacts:
            continue

        # ── Ground truth fidelity ─────────────────────────────────────────
        inc_fidelity = score_incident_fidelity(incident, scoring_artifacts)
        all_fidelity.append(inc_fidelity.overall)

        for fname, score in inc_fidelity.by_field().items():
            field_buckets[fname].append(score)
        for atype, score in inc_fidelity.by_artifact_type().items():
            type_buckets[atype].append(score)

        # ── Optional contradiction judge ───────────────────────────────────
        contradictions = 0
        if not skip_judge and bedrock_judge and len(scoring_artifacts) >= 2:
            logger.info(
                f"  Judging {arm_name}/{incident.incident_id} "
                f"({len(scoring_artifacts)} artifacts)..."
            )
            contradictions = judge_contradictions(bedrock_judge, scoring_artifacts)
        all_contradictions.append(contradictions)

        # ── Per-artifact detail for output ────────────────────────────────
        artifact_detail = []
        for af in inc_fidelity.artifacts:
            artifact_detail.append(
                {
                    "artifact_id": af.artifact_id,
                    "artifact_type": af.artifact_type,
                    "overall": round(af.overall, 4),
                    "fields": {f.field_name: round(f.recall, 4) for f in af.fields},
                }
            )

        per_incident.append(
            {
                "incident_id": incident.incident_id,
                "n_artifacts": len(scoring_artifacts),
                "fidelity_overall": round(inc_fidelity.overall, 4),
                "fidelity_by_field": {
                    k: round(v, 4) for k, v in inc_fidelity.by_field().items()
                },
                "fidelity_by_artifact_type": {
                    k: round(v, 4) for k, v in inc_fidelity.by_artifact_type().items()
                },
                "contradictions": contradictions,
                "artifacts": artifact_detail,
            }
        )

        logger.info(
            f"  {incident.incident_id}: fidelity={inc_fidelity.overall:.3f}  "
            f"contradictions={contradictions}  artifacts={len(scoring_artifacts)}"
        )

    return EvalResult(
        arm=arm_name,
        n_incidents=len(per_incident),
        overall_fidelity=round(mean(all_fidelity), 4) if all_fidelity else 0.0,
        fidelity_by_field={k: round(mean(v), 4) for k, v in field_buckets.items()},
        fidelity_by_artifact_type={
            k: round(mean(v), 4) for k, v in type_buckets.items()
        },
        contradictions=round(mean(all_contradictions), 2)
        if all_contradictions
        else 0.0,
        per_incident=per_incident,
    )


# ── Output formatting ────────────────────────────────────────────────────────


def print_results(results: list[EvalResult]) -> None:
    col_w = 18
    arms = [r.arm for r in results]
    header = f"{'Metric':<40}" + "".join(f"{a:>{col_w}}" for a in arms)

    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    # Overall fidelity
    row = f"{'GT fidelity (overall)':<40}"
    for r in results:
        row += f"{r.overall_fidelity:>{col_w}.4f}"
    print(row)

    # Per-field fidelity
    all_fields = sorted({f for r in results for f in r.fidelity_by_field})
    for fname in all_fields:
        row = f"  fidelity:{fname:<30}"
        for r in results:
            val = r.fidelity_by_field.get(fname, float("nan"))
            row += f"{val:>{col_w}.4f}"
        print(row)

    print("-" * len(header))

    # Per-artifact-type fidelity
    all_types = sorted({t for r in results for t in r.fidelity_by_artifact_type})
    for atype in all_types:
        row = f"  fidelity:{atype:<30}"
        for r in results:
            val = r.fidelity_by_artifact_type.get(atype, float("nan"))
            row += f"{val:>{col_w}.4f}"
        print(row)

    print("-" * len(header))

    row = f"{'Contradictions / incident (judge)':<40}"
    for r in results:
        row += f"{r.contradictions:>{col_w}.2f}"
    print(row)

    row = f"{'Incidents evaluated':<40}"
    for r in results:
        row += f"{r.n_incidents:>{col_w}d}"
    print(row)

    print("=" * len(header))

    # Per-incident detail
    for r in results:
        print(f"\n── {r.arm} per-incident detail ──")
        for p in r.per_incident:
            field_str = "  ".join(
                f"{k}={v:.3f}" for k, v in p["fidelity_by_field"].items()
            )
            print(
                f"  {p['incident_id']}: "
                f"fidelity={p['fidelity_overall']:.3f}  "
                f"contradictions={p['contradictions']}  "
                f"artifacts={p['n_artifacts']}"
            )
            if field_str:
                print(f"    fields: {field_str}")
            for a in p.get("artifacts", []):
                field_detail = "  ".join(f"{k}={v:.3f}" for k, v in a["fields"].items())
                print(
                    f"    [{a['artifact_type']:12s}] {a['artifact_id']}: {field_detail}"
                )


def save_results(results: list[EvalResult], path: str = "eval_results.json") -> None:
    out = []
    for r in results:
        out.append(
            {
                "arm": r.arm,
                "overall_fidelity": r.overall_fidelity,
                "fidelity_by_field": r.fidelity_by_field,
                "fidelity_by_artifact_type": r.fidelity_by_artifact_type,
                "contradictions": r.contradictions,
                "n_incidents": r.n_incidents,
                "per_incident": r.per_incident,
            }
        )
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"Results saved to {path}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="OrgForge cross-document consistency evaluation"
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
    parser.add_argument("--output", default="eval_results.json")
    parser.add_argument(
        "--judge-model",
        default="",
        help="Model for contradiction judge (defaults to --model)",
    )
    args = parser.parse_args()

    mongo = MongoExtractor(db_name=args.db, uri=args.mongo_uri)
    bedrock = Bedrock(model_id=args.model, region=args.region)
    bedrock_judge = (
        Bedrock(
            model_id=args.judge_model or args.model,
            region=args.region,
        )
        if not args.skip_judge
        else None
    )

    logger.info(f"Extracting up to {args.incidents} incidents from OrgForge...")
    incidents = mongo.extract_incidents(max_n=args.incidents)

    if not incidents:
        logger.error("No incidents found in MongoDB. Run OrgForge first.")
        sys.exit(1)

    for inc in incidents:
        logger.info(
            f"  {inc.incident_id}: on_call={inc.on_call!r}  "
            f"root_cause={inc.root_cause[:60]!r}  "
            f"({len(inc.artifacts)} artifacts)"
        )

    # ── Arm 1: OrgForge ──────────────────────────────────────────────────
    logger.info("\n━━━ Evaluating OrgForge arm ━━━")
    orgforge_result = evaluate_arm(
        "OrgForge",
        [(inc, inc.artifacts) for inc in incidents],
        bedrock_judge,
        skip_judge=args.skip_judge,
    )
    results = [orgforge_result]

    if not args.skip_baselines:
        org_names = mongo.org_names()
        org_chart_str = "\n".join(sorted(org_names))

        ts_doc = mongo._db["artifacts"].find_one({"type": "tech_stack"})
        tech_stack_str = ""
        if ts_doc:
            content = ts_doc.get("content", "")
            tech_stack_str = (
                json.dumps(content, indent=2)
                if isinstance(content, dict)
                else str(content)
            )

        company = COMPANY_NAME

        # ── Arm 2: Chained baseline ──────────────────────────────────────
        logger.info("\n━━━ Generating chained baseline ━━━")
        chained_data = []
        for inc in incidents:
            arts = mongo.load_cached_baseline("chained", inc.incident_id, args.model)
            if arts is None:
                logger.info(f"  Generating chained artifacts for {inc.incident_id}...")
                arts = generate_baseline_artifacts(
                    bedrock, inc, tech_stack_str, org_chart_str, company, chained=True
                )
                mongo.save_cached_baseline("chained", inc.incident_id, args.model, arts)
            else:
                logger.info(f"  Using cached chained baseline for {inc.incident_id}")
            chained_data.append((inc, arts))

        logger.info("\n━━━ Evaluating chained baseline ━━━")
        chained_result = evaluate_arm(
            "Chained",
            chained_data,
            bedrock_judge,
            skip_judge=args.skip_judge,
        )
        results.append(chained_result)

        # ── Arm 3: Parallel baseline ─────────────────────────────────────
        logger.info("\n━━━ Generating parallel baseline ━━━")
        parallel_data = []
        for inc in incidents:
            arts = mongo.load_cached_baseline("parallel", inc.incident_id, args.model)
            if arts is None:
                logger.info(f"  Generating parallel artifacts for {inc.incident_id}...")
                arts = generate_baseline_artifacts(
                    bedrock, inc, tech_stack_str, org_chart_str, company, chained=False
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
            bedrock_judge,
            skip_judge=args.skip_judge,
        )
        results.append(parallel_result)

    print_results(results)
    save_results(results, args.output)


if __name__ == "__main__":
    main()
