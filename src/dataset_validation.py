"""
Dataset integrity checks for generated OrgForge runs.

The validator is intentionally read-only. It catches invariants that make the
viewer, downstream evals, and computer-use environments harder to trust.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


def _parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _date_to_day(db) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for checkpoint in db.checkpoints.find({}, {"day": 1, "state.date": 1, "_id": 0}):
        day = checkpoint.get("day")
        date_value = (checkpoint.get("state") or {}).get("date")
        if day is not None and date_value:
            mapping[str(date_value)[:10]] = int(day)
    return mapping


def _known_actor_sets(db, config: dict[str, Any] | None = None) -> tuple[set[str], set[str], dict[str, str]]:
    internal: set[str] = set()
    external: set[str] = set()
    aliases: dict[str, str] = {}

    if config:
        for members in (config.get("org_chart") or {}).values():
            internal.update(members or [])
        for source in config.get("external_contacts", []) or []:
            name = source.get("name")
            if name:
                external.add(name)
            first = source.get("first_name")
            if first and name and first != name:
                aliases[first] = name

    for event in db.events.find(
        {"type": {"$in": ["employee_hired", "employee_departed"]}},
        {"facts": 1, "actors": 1, "_id": 0},
    ):
        internal.update(a for a in event.get("actors", []) if a)
        facts = event.get("facts") or {}
        if facts.get("name"):
            internal.add(facts["name"])

    for doc in db.dept_plans.find({}, {"engineer_plans.name": 1, "lead": 1, "_id": 0}):
        if doc.get("lead"):
            internal.add(doc["lead"])
        for plan in doc.get("engineer_plans", []) or []:
            if plan.get("name"):
                internal.add(plan["name"])

    source_doc = db.sim_config.find_one({"_id": "inbound_email_sources"}) or {}
    for source in source_doc.get("sources", []) or []:
        name = source.get("name")
        if name:
            external.add(name)
        first = source.get("first_name")
        if first and name and first != name:
            aliases[first] = name

    for account in db.sf_accounts.find(
        {}, {"primary_contact_name": 1, "primary_contact": 1, "name": 1, "_id": 0}
    ):
        for key in ("primary_contact_name", "primary_contact"):
            name = account.get(key)
            if name:
                external.add(name)
                first = str(name).split()[0]
                if first and first != name:
                    aliases[first] = name
        if account.get("name"):
            external.add(account["name"])

    for opp in db.sf_opps.find(
        {}, {"account_name": 1, "primary_contact": 1, "primary_contact_name": 1, "_id": 0}
    ):
        for key in ("account_name", "primary_contact", "primary_contact_name"):
            name = opp.get(key)
            if name:
                external.add(name)

    return internal, external, aliases


def validate_dataset(
    db,
    export_dir: str | Path | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {}

    def add_error(kind: str, **fields: Any) -> None:
        errors.append({"kind": kind, **fields})

    def add_warning(kind: str, **fields: Any) -> None:
        warnings.append({"kind": kind, **fields})

    email_ids = {
        doc.get("embed_id") or str(doc.get("_id")): doc
        for doc in db.emails.find({}, {"embedding": 0})
    }
    artifact_ids = {str(doc["_id"]) for doc in db.artifacts.find({}, {"_id": 1})}
    metrics["emails"] = len(email_ids)

    paths = [
        doc.get("eml_path")
        for doc in email_ids.values()
        if doc.get("eml_path")
    ]
    duplicate_paths = sorted(path for path, count in Counter(paths).items() if count > 1)
    metrics["duplicate_email_paths"] = len(duplicate_paths)
    for path in duplicate_paths[:25]:
        add_error("duplicate_email_path", path=path)

    for embed_id, doc in email_ids.items():
        parent_id = (doc.get("reply_to_email_id") or "").strip()
        if parent_id == embed_id:
            add_error("email_self_parent", embed_id=embed_id)
        elif parent_id and parent_id not in email_ids:
            add_error("missing_email_parent", embed_id=embed_id, parent_id=parent_id)
        elif parent_id:
            parent_ts = _parse_ts(email_ids[parent_id].get("timestamp"))
            child_ts = _parse_ts(doc.get("timestamp"))
            if parent_ts and child_ts and child_ts < parent_ts:
                add_error(
                    "email_parent_after_child",
                    embed_id=embed_id,
                    parent_id=parent_id,
                    parent_ts=parent_ts.isoformat(),
                    child_ts=child_ts.isoformat(),
                )
        if embed_id not in artifact_ids:
            add_warning("email_missing_artifact", embed_id=embed_id)

    export_path = Path(export_dir).expanduser().resolve() if export_dir else None
    missing_exports = 0
    if export_path:
        for doc in email_ids.values():
            eml_path = doc.get("eml_path")
            if eml_path and not Path(eml_path).expanduser().exists():
                missing_exports += 1
                if missing_exports <= 25:
                    add_error("missing_email_export", embed_id=doc.get("embed_id"), path=eml_path)
    metrics["missing_email_exports"] = missing_exports

    for collection, field in (
        ("jira_tickets", "id"),
        ("pull_requests", "pr_id"),
        ("emails", "embed_id"),
    ):
        values = [
            doc.get(field)
            for doc in db[collection].find({field: {"$exists": True}}, {field: 1, "_id": 0})
            if doc.get(field)
        ]
        duplicates = sorted(value for value, count in Counter(values).items() if count > 1)
        metrics[f"duplicate_{collection}_{field}"] = len(duplicates)
        for value in duplicates[:25]:
            add_error("duplicate_logical_id", collection=collection, field=field, value=value)

    date_to_day = _date_to_day(db)
    mismatches = 0
    if date_to_day:
        for collection in (
            "emails",
            "events",
            "artifacts",
            "jira_tickets",
            "slack_messages",
            "pull_requests",
            "zd_tickets",
        ):
            if collection not in db.list_collection_names():
                continue
            for doc in db[collection].find(
                {"date": {"$exists": True}, "day": {"$exists": True}},
                {"date": 1, "day": 1, "embed_id": 1, "id": 1, "pr_id": 1},
            ):
                expected = date_to_day.get(str(doc.get("date"))[:10])
                if expected is not None and int(doc.get("day")) != expected:
                    mismatches += 1
                    if mismatches <= 25:
                        add_error(
                            "day_date_mismatch",
                            collection=collection,
                            id=doc.get("embed_id") or doc.get("id") or doc.get("pr_id") or str(doc.get("_id")),
                            date=doc.get("date"),
                            day=doc.get("day"),
                            expected_day=expected,
                        )
    metrics["day_date_mismatches"] = mismatches

    internal, external, aliases = _known_actor_sets(db, config)
    alias_hits: dict[str, set[str]] = defaultdict(set)
    unknown_actors = Counter()
    for event in db.events.find({}, {"actors": 1, "type": 1, "_id": 0}):
        for actor in event.get("actors", []) or []:
            if actor in internal or actor in external:
                continue
            if actor in aliases:
                alias_hits[actor].add(aliases[actor])
            elif actor:
                unknown_actors[actor] += 1
    for alias, full_names in sorted(alias_hits.items()):
        add_warning("actor_alias_used", actor=alias, should_be=sorted(full_names))
    for actor, count in unknown_actors.most_common(25):
        add_warning("unknown_actor", actor=actor, count=count)
    metrics["actor_aliases"] = len(alias_hits)
    metrics["unknown_actors"] = len(unknown_actors)

    return {
        "ok": not errors,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "metrics": metrics,
        "errors": errors,
        "warnings": warnings,
    }
