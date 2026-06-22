from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from statistics import mean
from typing import Any

import torch
from rapidfuzz import fuzz
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class DivergenceType(Enum):
    ENTITY_MISSING = "entity_missing"
    ENTITY_CONTRADICTED = "entity_contradicted"
    FACT_CONTRADICTED = "fact_contradicted"
    NUMERIC_MISMATCH = "numeric_mismatch"


@dataclass
class Divergence:
    divergence_type: DivergenceType
    sim_event_field: str
    sim_event_value: str
    prose_value: str | None
    artifact_id: str
    confidence: float
    detail: str = ""


@dataclass
class ProseSimEventReport:
    artifact_id: str
    entity_score: float
    nli_score: float
    numeric_score: float
    composite_score: float
    divergences: list[Divergence] = field(default_factory=list)


ENTITY_FIELDS: dict[str, list[str]] = {
    "actors": ["actors", "assigned_to", "author", "reviewer", "responders"],
    "identifiers": [
        "ticket_id",
        "pr_id",
        "incident_id",
        "confluence_id",
        "zd_ticket_id",
        "sf_opportunity_id",
        "invoice_id",
    ],
    "components": [
        "affected_system",
        "tech_stack",
        "depends_on_components",
        "system_tags",
    ],
    "organizations": ["customer_org", "vendor_org", "org_name"],
    "statuses": ["status", "verdict", "stage", "priority", "severity"],
}

FACT_TEMPLATES: dict[str, str] = {
    "root_cause": "The root cause of the incident is described as follows: {value}",
    "affected_system": "{value} was involved in the incident.",
    "status": "The status is {value}.",
    "verdict": "The verdict was {value}.",
    "assigned_to": "{value} worked on this.",
    "author": "{value} authored this.",
    "reviewer": "{value} reviewed this.",
    "customer_org": "This involves {value}.",
    "priority": "The priority is {value}.",
    "severity": "The severity is {value}.",
    "stage": "The stage is {value}.",
    "incident_duration_hours": "The duration was {value} hours.",
    "resolution_summary": "The resolution is described as follows: {value}",
    "title": "This document is about {value}.",
    "vendor_org": "This involves {value} as an external party.",
}

_SHORT_ALIAS_RE_CACHE: dict[str, re.Pattern] = {}


def _alias_in_prose(alias: str, prose_lower: str) -> bool:
    lowered = alias.lower()
    if len(lowered) <= 4:
        pattern = _SHORT_ALIAS_RE_CACHE.get(lowered)
        if pattern is None:
            pattern = re.compile(rf"\b{re.escape(lowered)}\b")
            _SHORT_ALIAS_RE_CACHE[lowered] = pattern
        return bool(pattern.search(prose_lower))
    return lowered in prose_lower


class NLIScorer:
    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self._label_map = {
            i: label.lower() for i, label in self.model.config.id2label.items()
        }

    def score(self, premise: str, hypothesis: str) -> dict[str, float]:
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        return {self._label_map[i]: probs[i].item() for i in range(3)}


def _flatten_fact_value(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, dict):
        return [str(v) for v in value.values()]
    return [str(value)]


def _clean_prose_for_nli(content: str, max_chars: int = 3500) -> str:
    lines = content.splitlines()
    filtered = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("|"):
            continue
        if stripped.startswith("#"):
            filtered.append(stripped.lstrip("#").strip())
            continue
        if re.match(r"^\*\*\w+[:：]\*\*\s*\w+-\w+$", stripped):
            continue
        filtered.append(stripped)
    return " ".join(filtered)[:max_chars]


# Fields whose numeric value is a boolean flag (0/1) with no prose counterpart.
_NUMERIC_BOOL_FIELDS: frozenset[str] = frozenset(
    {
        "has_question",
        "incident_review",
        "involves_gap",
        "recurrence_chain_depth",
        "is_escalated",
        "is_resolved",
        "has_postmortem",
        "has_followup",
        "escalation_harder",
    }
)

# Fields that are dimensionless ratios / percentages (0.0–1.0 scale).
# Prose never writes "0.25" for 25 % coverage — it writes "25 %" or "25 percent",
# so raw-number matching against these will always be off by 4×.
# They are skipped here and can be validated by NLI or a dedicated % check instead.
_NUMERIC_RATIO_FIELDS: frozenset[str] = frozenset(
    {
        "documented_pct",
        "live_documentation_coverage",
        "coverage",
        "score_pct",
        "completion_rate",
        "pass_rate",
    }
)

_NUMERIC_INTERNAL_FIELDS: frozenset[str] = frozenset(
    {
        "semantic_score",
        "days_since_departure",
        "recurrence_gap_days",
        "health_at_open",
        "health_at_close",
        "sim_score",
        "quality_score",
    }
)


def _extract_numeric_claims(facts: dict[str, Any]) -> dict[str, float]:
    numeric_fields = {}
    for key, value in facts.items():
        if (
            key in _NUMERIC_BOOL_FIELDS
            or key in _NUMERIC_RATIO_FIELDS
            or key in _NUMERIC_INTERNAL_FIELDS
        ):
            continue
        if isinstance(value, int | float):
            numeric_fields[key] = float(value)
        elif isinstance(value, str):
            try:
                numeric_fields[key] = float(value)
            except ValueError:
                pass
    return numeric_fields


def _find_numbers_in_prose(prose: str) -> list[tuple[float, str]]:
    patterns = [
        (r"(\d+\.?\d*)\s*hours?", "hours"),
        (r"(\d+\.?\d*)\s*minutes?", "minutes"),
        (r"\$\s*([\d,]+\.?\d*)", "dollars"),
        (r"(\d+\.?\d*)\s*%", "percent"),
        (r"(\d+\.?\d*)\s*days?", "days"),
        (r"(\d+)", "raw_number"),
    ]
    found: list[tuple[float, str]] = []
    for pattern, unit in patterns:
        for match in re.finditer(pattern, prose, re.IGNORECASE):
            raw = match.group(1).replace(",", "")
            found.append((float(raw), unit))
    return found


_ORG_INDICATORS = re.compile(
    r"\b(?:"
    r"inc|llc|corp|ltd|services|web services|cloud|platform|"
    r"aws|gcp|azure|hashicorp|terraform|"
    r"jenkins|cloudbees|github|gitlab|circleci|buildkite|"
    r"datadog|pagerduty|opsgenie|sentry|grafana|prometheus|"
    r"docker|kubernetes|kafka|rabbitmq|redis|"
    r"bot|ci[/-]cd|pipeline|automation|cron"
    r")\b",
    re.IGNORECASE,
)

FIELD_ARTIFACT_SCOPE: dict[str, set[str]] = {
    "assigned_to": {"jira", "jira_comment"},
    "reviewer": {"pr"},
    "verdict": {"pr"},
    "author": {"pr", "confluence", "jira"},
    "responders": {"jira", "jira_comment", "slack"},
    "actors": {"jira", "jira_comment", "slack"},
    "ticket_id": {"jira", "jira_comment", "pr", "confluence"},
    "vendor_org": {"jira", "jira_comment", "slack", "confluence"},
    "priority": {"jira", "jira_comment"},
    "severity": {"jira", "jira_comment"},
    "stage": {"jira", "jira_comment"},
    "resolution_summary": {"jira", "confluence"},
}


def _tokens_contained(expected_normalized: str, prose_lower: str) -> bool:
    expected_tokens = {t for t in expected_normalized.split() if len(t) > 2}
    if not expected_tokens:
        return False
    prose_tokens = set(prose_lower.split())
    return expected_tokens.issubset(prose_tokens)


def _all_tokens_present(
    expected: str, prose_lower: str, min_token_len: int = 3
) -> bool:
    tokens = [t.lower() for t in expected.split() if len(t) >= min_token_len]
    if len(tokens) < 2:
        return False
    return all(t in prose_lower for t in tokens)


def check_entity_divergence(
    artifact_id: str,
    artifact_type: str,
    content: str,
    sim_event_facts: dict[str, Any],
    fuzzy_threshold: int = 80,
    vendor_aliases: dict[str, set[str]] | None = None,
) -> tuple[float, list[Divergence]]:
    prose_lower = content.lower()
    divergences: list[Divergence] = []
    checks = 0
    matches = 0
    checked_org_values: set[str] = set()

    for category, field_names in ENTITY_FIELDS.items():
        for field_name in field_names:
            if field_name not in sim_event_facts:
                continue

            allowed_types = FIELD_ARTIFACT_SCOPE.get(field_name)
            if allowed_types and artifact_type not in allowed_types:
                continue

            expected_values = _flatten_fact_value(sim_event_facts[field_name])
            for expected in expected_values:
                if not expected or expected.lower() in ("none", "null", ""):
                    continue

                if category == "identifiers" and expected == artifact_id:
                    continue

                is_org = category in ("actors", "organizations") and (
                    _ORG_INDICATORS.search(expected)
                    or (vendor_aliases and expected in vendor_aliases)
                )

                if is_org and vendor_aliases and expected in vendor_aliases:
                    if expected in checked_org_values:
                        continue
                    checked_org_values.add(expected)
                    checks += 1
                    aliases = vendor_aliases[expected]
                    if any(_alias_in_prose(a, prose_lower) for a in aliases):
                        matches += 1
                    else:
                        divergences.append(
                            Divergence(
                                divergence_type=DivergenceType.ENTITY_MISSING,
                                sim_event_field=field_name,
                                sim_event_value=expected,
                                prose_value=None,
                                artifact_id=artifact_id,
                                confidence=0.6,
                                detail=(
                                    f"'{expected}' (vendor) not found via aliases "
                                    f"{sorted(aliases)} (category: {category})"
                                ),
                            )
                        )
                    continue

                if is_org:
                    continue

                normalized = expected.lower().replace("_", " ")
                checks += 1

                exact_hit = normalized in prose_lower
                fuzzy_hit = (
                    fuzz.partial_ratio(normalized, prose_lower) >= fuzzy_threshold
                )

                if category == "statuses":
                    token_hit = _tokens_contained(normalized, prose_lower)
                else:
                    token_hit = _all_tokens_present(expected, prose_lower)

                if exact_hit or fuzzy_hit or token_hit:
                    matches += 1
                else:
                    divergences.append(
                        Divergence(
                            divergence_type=DivergenceType.ENTITY_MISSING,
                            sim_event_field=field_name,
                            sim_event_value=expected,
                            prose_value=None,
                            artifact_id=artifact_id,
                            confidence=0.7,
                            detail=f"'{expected}' not found in prose (category: {category})",
                        )
                    )

    score = matches / checks if checks > 0 else 1.0
    return score, divergences


FACT_TEMPLATE_SCOPE: dict[str, set[str]] = {
    "root_cause": {"jira", "jira_comment", "confluence", "slack", "pr"},
    "affected_system": {"jira", "jira_comment", "confluence", "slack", "pr"},
    "status": {"jira", "jira_comment"},
    "verdict": {"pr"},
    "assigned_to": {"jira", "jira_comment"},
    "author": {"pr", "confluence"},
    "reviewer": {"pr"},
    "customer_org": {"jira", "confluence"},
    "priority": {"jira", "jira_comment", "confluence"},
    "severity": {"jira", "jira_comment", "confluence"},
    "stage": {"jira"},
    "incident_duration_hours": {"jira", "confluence"},
    "resolution_summary": {"jira", "confluence"},
    "title": {"jira", "confluence", "pr"},
}

_NLI_SKIP_FIELDS_BY_EVENT_TYPE: dict[str, frozenset[str]] = {
    "external_contact_summarized": frozenset({"root_cause", "title"}),
    "inbound_external_email": frozenset({"root_cause"}),
}


def check_nli_divergence(
    artifact_id: str,
    artifact_type: str,
    content: str,
    sim_event_facts: dict[str, Any],
    nli: NLIScorer,
    contradiction_threshold: float = 0.70,
    sim_event_type: str = "",
) -> tuple[float, list[Divergence]]:
    _MAX_HYPOTHESIS_VALUE_CHARS = 200

    divergences: list[Divergence] = []
    scores: list[float] = []
    prose_truncated = _clean_prose_for_nli(content)
    skip_fields = _NLI_SKIP_FIELDS_BY_EVENT_TYPE.get(sim_event_type, frozenset())

    for field_name, template in FACT_TEMPLATES.items():
        if field_name in skip_fields:
            continue

        if field_name not in sim_event_facts:
            continue

        allowed = FACT_TEMPLATE_SCOPE.get(field_name)
        if allowed and artifact_type not in allowed:
            continue

        allowed_types = FIELD_ARTIFACT_SCOPE.get(field_name)
        if allowed_types and artifact_type not in allowed_types:
            continue

        value = sim_event_facts[field_name]
        if value is None or str(value).lower() in ("none", "null", ""):
            continue

        if field_name == "vendor_org" and isinstance(value, str):
            words = value.split()
            if len(words) >= 2 and _ORG_INDICATORS.search(words[-1]):
                value = words[0]

        if isinstance(value, list):
            hypothesis = template.format(value=", ".join(str(v) for v in value))
        else:
            hypothesis = template.format(value=str(value))

        result = nli.score(premise=prose_truncated, hypothesis=hypothesis)

        if result["contradiction"] >= contradiction_threshold:
            divergences.append(
                Divergence(
                    divergence_type=DivergenceType.FACT_CONTRADICTED,
                    sim_event_field=field_name,
                    sim_event_value=str(value),
                    prose_value=f"NLI contradiction={result['contradiction']:.2f}",
                    artifact_id=artifact_id,
                    confidence=result["contradiction"],
                    detail=f"Hypothesis: '{hypothesis}'",
                )
            )
            scores.append(0.0)
        elif result["contradiction"] < 0.3:
            scores.append(1.0)
        elif result["entailment"] >= 0.5:
            scores.append(1.0)
        else:
            scores.append(1.0 - result["contradiction"])

    return mean(scores) if scores else 1.0, divergences


def check_numeric_divergence(
    artifact_id: str,
    content: str,
    sim_event_facts: dict[str, Any],
    relative_tolerance: float = 0.15,
) -> tuple[float, list[Divergence]]:
    numeric_facts = _extract_numeric_claims(sim_event_facts)
    if not numeric_facts:
        return 1.0, []
    prose_numbers = _find_numbers_in_prose(content)
    if not prose_numbers:
        return 1.0, []

    divergences: list[Divergence] = []
    checks = 0
    matches = 0

    unit_mapping: dict[str, list[str]] = {
        "duration_hours": ["hours"],
        "incident_duration_hours": ["hours"],
        "duration_minutes": ["minutes"],
        "sla_credit": ["dollars"],
        "credit_amount": ["dollars"],
        "coverage": ["percent"],
        "score": ["raw_number"],
    }

    for field_name, expected_value in numeric_facts.items():
        relevant_units = unit_mapping.get(field_name, ["raw_number"])
        candidate_numbers = [
            num for num, unit in prose_numbers if unit in relevant_units
        ]
        if not candidate_numbers:
            candidate_numbers = [num for num, _ in prose_numbers]
        if not candidate_numbers:
            continue

        checks += 1
        closest = min(candidate_numbers, key=lambda x: abs(x - expected_value))

        if expected_value == 0:
            is_match = closest == 0
        else:
            is_match = (
                abs(closest - expected_value) / abs(expected_value)
                <= relative_tolerance
            )

        if is_match:
            matches += 1
        else:
            divergences.append(
                Divergence(
                    divergence_type=DivergenceType.NUMERIC_MISMATCH,
                    sim_event_field=field_name,
                    sim_event_value=str(expected_value),
                    prose_value=str(closest),
                    artifact_id=artifact_id,
                    confidence=min(
                        1.0,
                        abs(closest - expected_value) / max(abs(expected_value), 1e-6),
                    ),
                    detail=f"Expected ~{expected_value}, closest prose number: {closest}",
                )
            )

    return (matches / checks if checks > 0 else 1.0), divergences


def measure_artifact_divergence(
    artifact_id: str,
    artifact_type: str,
    artifact_content: str,
    sim_event_facts: dict[str, Any],
    nli: NLIScorer | None = None,
    vendor_aliases: dict[str, set[str]] | None = None,
    weights: tuple[float, float, float] = (0.35, 0.45, 0.20),
    sim_event_type: str = "",
) -> ProseSimEventReport:
    entity_score, entity_divs = check_entity_divergence(
        artifact_id,
        artifact_type,
        artifact_content,
        sim_event_facts,
        vendor_aliases=vendor_aliases,
    )

    if nli is not None:
        nli_score, nli_divs = check_nli_divergence(
            artifact_id,
            artifact_type,
            artifact_content,
            sim_event_facts,
            nli,
            sim_event_type=sim_event_type,
        )
    else:
        nli_score, nli_divs = 1.0, []

    numeric_score, numeric_divs = check_numeric_divergence(
        artifact_id,
        artifact_content,
        sim_event_facts,
    )

    w_ent, w_nli, w_num = weights
    composite = w_ent * entity_score + w_nli * nli_score + w_num * numeric_score

    return ProseSimEventReport(
        artifact_id=artifact_id,
        entity_score=entity_score,
        nli_score=nli_score,
        numeric_score=numeric_score,
        composite_score=composite,
        divergences=entity_divs + nli_divs + numeric_divs,
    )
