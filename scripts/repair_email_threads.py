#!/usr/bin/env python
"""
Repair generated email threading metadata for an existing OrgForge dataset.

The generator now writes stable embed/thread IDs and causally ordered reply
timestamps. This script backfills those fields into datasets produced before
that fix and rewrites unique .eml files so the viewer can show email threads
without relying on subject-only grouping.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

from pymongo import MongoClient


DEFAULT_MONGO_URI = "mongodb://localhost:27017/?directConnection=true"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mongo-uri", default=os.environ.get("MONGO_URI", DEFAULT_MONGO_URI))
    parser.add_argument("--db", default=os.environ.get("DB_NAME", "orgforge"))
    parser.add_argument(
        "--export-dir",
        default=os.environ.get("ORGFORGE_EXPORT_DIR", "export"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def business_time(dt: datetime) -> datetime:
    end = dt.replace(hour=17, minute=30, second=0, microsecond=0)
    if dt.weekday() < 5 and dt <= end:
        return dt
    next_day = dt + timedelta(days=1)
    while next_day.weekday() >= 5:
        next_day += timedelta(days=1)
    return next_day.replace(hour=9, minute=0, second=0, microsecond=0)


def safe_slug(value: str, limit: int = 110) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")[:limit] or "email"


def strip_one_reply_prefix(subject: str) -> str:
    return re.sub(r"^(re|fw|fwd):\s*", "", subject or "", flags=re.IGNORECASE).strip()


def normalize_subject(subject: str) -> str:
    cleaned = (subject or "").strip()
    while True:
        updated = strip_one_reply_prefix(cleaned)
        if updated == cleaned:
            break
        cleaned = updated
    return re.sub(r"\s+", " ", cleaned).strip().lower()


def reply_depth(subject: str) -> int:
    depth = 0
    cleaned = (subject or "").strip()
    while True:
        updated = strip_one_reply_prefix(cleaned)
        if updated == cleaned:
            return depth
        depth += 1
        cleaned = updated


def generated_day_suffix(embed_id: str) -> str:
    match = re.search(r"_(\d+)$", str(embed_id))
    return match.group(1) if match else ""


def write_eml(export_dir: Path, doc: dict[str, Any]) -> str:
    direction = doc.get("direction") or "inbound"
    date_str = doc.get("date") or str(doc.get("timestamp", ""))[:10] or "unknown"
    from_name = doc.get("from_name") or doc.get("from_addr") or "unknown"
    from_addr = doc.get("from_addr") or "unknown@example.com"
    to_name = doc.get("to_name") or doc.get("to_addr") or "unknown"
    to_addr = doc.get("to_addr") or "unknown@example.com"
    embed_id = doc.get("embed_id") or str(doc.get("_id"))

    out_dir = export_dir / "emails" / direction / date_str
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{safe_slug(from_name.lower().replace(' ', '_'))}_{safe_slug(embed_id)}.eml"

    msg = MIMEMultipart("alternative")
    msg["From"] = f"{from_name} <{from_addr}>"
    msg["To"] = f"{to_name} <{to_addr}>"
    msg["Subject"] = doc.get("subject", "")
    msg["Date"] = doc.get("timestamp", "")
    msg["Message-ID"] = f"<{safe_slug(embed_id)}@orgforge.local>"
    parent = doc.get("reply_to_email_id")
    if parent:
        msg["In-Reply-To"] = f"<{safe_slug(parent)}@orgforge.local>"
        msg["References"] = msg["In-Reply-To"]
    msg["X-OrgForge-Direction"] = direction
    msg.attach(MIMEText(doc.get("body", ""), "plain"))
    path.write_text(msg.as_string())
    return str(path)


def _email_addr_for_name(name: str, domain: str = "example.com") -> str:
    return f"{safe_slug(name.lower().replace(' ', '.'))}@{domain}"


def _doc_from_parent_artifact(db, parent_id: str, export_dir: Path) -> dict[str, Any] | None:
    artifact = db.artifacts.find_one({"_id": parent_id}, {"embedding": 0})
    if not artifact:
        return None

    metadata = artifact.get("metadata") or {}
    event = db.events.find_one(
        {
            "$or": [
                {"artifact_ids.email": parent_id},
                {"artifact_ids.embed_id": parent_id},
                {"artifact_ids.email_thread": parent_id},
            ]
        },
        {"facts": 1, "actors": 1, "date": 1, "day": 1, "timestamp": 1, "_id": 0},
    )
    facts = (event or {}).get("facts") or {}
    actors = (event or {}).get("actors") or []
    direction = metadata.get("direction") or facts.get("direction") or "inbound"
    from_name = (
        facts.get("source")
        or facts.get("from")
        or metadata.get("source")
        or (actors[0] if actors else "unknown")
    )
    to_name = (
        facts.get("to")
        or facts.get("liaison")
        or metadata.get("liaison")
        or (actors[1] if len(actors) > 1 else "unknown")
    )
    from_addr = facts.get("source_email") or _email_addr_for_name(from_name)
    to_addr = facts.get("liaison_email") or _email_addr_for_name(to_name, "orgforge.local")
    doc = {
        "embed_id": parent_id,
        "direction": direction,
        "from_name": from_name,
        "from_addr": from_addr,
        "to_name": to_name,
        "to_addr": to_addr,
        "subject": facts.get("subject") or artifact.get("title") or parent_id,
        "body": artifact.get("content", ""),
        "timestamp": artifact.get("timestamp") or (event or {}).get("timestamp"),
        "day": artifact.get("day") or (event or {}).get("day"),
        "date": artifact.get("date") or (event or {}).get("date"),
        "thread_id": parent_id,
        "reply_to_email_id": "",
        "thread_order": 0,
        "metadata": {
            "repaired_from_artifact": True,
            "artifact_id": parent_id,
        },
    }
    doc["eml_path"] = write_eml(export_dir, doc)
    return doc


def repair_missing_parent_emails(
    db,
    emails: dict[str, dict[str, Any]],
    parents: dict[str, str],
    export_dir: Path,
) -> tuple[int, int]:
    created = 0
    dropped_edges = 0
    for child_id, parent_id in list(parents.items()):
        if parent_id in emails:
            continue
        parent_doc = _doc_from_parent_artifact(db, parent_id, export_dir)
        if parent_doc:
            db.emails.update_one(
                {"embed_id": parent_id},
                {"$set": parent_doc},
                upsert=True,
            )
            emails[parent_id] = parent_doc
            created += 1
        else:
            parents.pop(child_id, None)
            dropped_edges += 1
    return created, dropped_edges


def load_date_to_day(db) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for checkpoint in db.checkpoints.find({}, {"day": 1, "state.date": 1, "_id": 0}):
        day = checkpoint.get("day")
        date_value = (checkpoint.get("state") or {}).get("date")
        if day is not None and date_value:
            mapping[str(date_value)[:10]] = int(day)
    return mapping


def align_inbound_embed_ids(db) -> int:
    changed = 0
    events = db.events.find(
        {
            "type": "inbound_external_email",
            "artifact_ids.email": {"$exists": True},
            "facts.subject": {"$exists": True},
        },
        {"artifact_ids": 1, "facts": 1, "timestamp": 1},
    )
    for event in events:
        embed_id = (event.get("artifact_ids") or {}).get("email")
        subject = (event.get("facts") or {}).get("subject")
        timestamp = event.get("timestamp")
        if not embed_id or not subject or db.emails.find_one({"embed_id": embed_id}):
            continue
        doc = db.emails.find_one(
            {
                "timestamp": timestamp,
                "subject": subject,
                "direction": "inbound",
            }
        )
        if not doc:
            continue
        db.emails.update_one({"_id": doc["_id"]}, {"$set": {"embed_id": embed_id}})
        changed += 1
    return changed


def build_parent_map(db, emails: dict[str, dict[str, Any]]) -> dict[str, str]:
    parents: dict[str, str] = {}

    for artifact in db.artifacts.find(
        {"metadata.reply_to_email_id": {"$exists": True}},
        {"_id": 1, "metadata.reply_to_email_id": 1},
    ):
        child = str(artifact.get("_id"))
        parent = ((artifact.get("metadata") or {}).get("reply_to_email_id") or "").strip()
        if child and parent and child != parent:
            parents[child] = parent

    for event in db.events.find(
        {"artifact_ids.source_email": {"$exists": True}},
        {"artifact_ids": 1},
    ):
        artifact_ids = event.get("artifact_ids") or {}
        child = artifact_ids.get("embed_id") or artifact_ids.get("email") or artifact_ids.get("email_thread")
        parent = artifact_ids.get("source_email")
        if child and parent and str(child) != str(parent):
            parents[str(child)] = str(parent)

    opps = {
        opp.get("opportunity_id"): opp
        for opp in db.sf_opps.find({}, {"_id": 0, "opportunity_id": 1, "touchpoints": 1})
    }
    for embed_id, doc in emails.items():
        if not str(embed_id).startswith("customer_reply_"):
            continue
        artifact = db.artifacts.find_one({"_id": embed_id}, {"metadata": 1})
        metadata = (artifact or {}).get("metadata") or {}
        opp = opps.get(metadata.get("opportunity_id"))
        if not opp:
            continue
        target_subject = metadata.get("reply_to_subject") or strip_one_reply_prefix(doc.get("subject", ""))
        target_depth = max(0, reply_depth(doc.get("subject", "")) - 1)
        candidates = [
            touchpoint
            for touchpoint in opp.get("touchpoints", [])
            if touchpoint.get("embed_id")
            and reply_depth(touchpoint.get("subject", "")) == target_depth
            and (
                re.sub(r"\s+", " ", touchpoint.get("subject", "")).strip().lower()
                == re.sub(r"\s+", " ", target_subject).strip().lower()
                or normalize_subject(touchpoint.get("subject", ""))
                == normalize_subject(target_subject)
            )
        ]
        if not candidates:
            continue
        candidates.sort(key=lambda item: item.get("timestamp", ""))
        parents[embed_id] = candidates[-1]["embed_id"]

    return parents


def resolve_thread_ids(
    emails: dict[str, dict[str, Any]], parents: dict[str, str]
) -> tuple[dict[str, str], dict[str, int]]:
    roots: dict[str, str] = {}
    depths: dict[str, int] = {}

    def root_for(embed_id: str, seen: set[str] | None = None) -> str:
        seen = seen or set()
        if embed_id in roots:
            return roots[embed_id]
        if embed_id in seen:
            roots[embed_id] = embed_id
            depths[embed_id] = 0
            return embed_id
        seen.add(embed_id)
        parent = parents.get(embed_id)
        if parent and parent in emails:
            root = root_for(parent, seen)
            roots[embed_id] = root
            depths[embed_id] = depths.get(parent, 0) + 1
        elif parent:
            roots[embed_id] = parent
            depths[embed_id] = 1
        else:
            roots[embed_id] = embed_id
            depths[embed_id] = 0
        return roots[embed_id]

    for embed_id in emails:
        root_for(embed_id)
    return roots, depths


def repair_timestamps(
    db,
    emails: dict[str, dict[str, Any]],
    parents: dict[str, str],
    date_to_day: dict[str, int],
) -> int:
    changed = 0
    artifact_times = {
        str(artifact["_id"]): parse_ts(artifact.get("timestamp"))
        for artifact in db.artifacts.find({}, {"_id": 1, "timestamp": 1})
    }

    for _ in range(10):
        pass_changed = 0
        for child_id, parent_id in parents.items():
            child = emails.get(child_id)
            if not child:
                continue
            parent = emails.get(parent_id)
            parent_ts = parse_ts(parent.get("timestamp")) if parent else artifact_times.get(parent_id)
            child_ts = parse_ts(child.get("timestamp"))
            if not parent_ts or not child_ts:
                continue
            delay_mins = 45 if str(child_id).startswith("customer_reply_") else 12
            target = business_time(parent_ts + timedelta(minutes=delay_mins))
            too_early = child_ts < target
            same_generated_day = (
                generated_day_suffix(child_id)
                and generated_day_suffix(child_id) == generated_day_suffix(parent_id)
            )
            too_late_same_generated_day = (
                same_generated_day and child_ts > parent_ts + timedelta(hours=4)
            )
            if not too_early and not too_late_same_generated_day:
                continue
            new_ts = target.isoformat()
            if child.get("timestamp") == new_ts:
                continue
            new_date = new_ts[:10]
            child["timestamp"] = new_ts
            child["date"] = new_date
            if new_date in date_to_day:
                child["day"] = date_to_day[new_date]
            pass_changed += 1
        changed += pass_changed
        if pass_changed == 0:
            break
    return changed


def update_snapshot(export_dir: Path, db) -> bool:
    snapshot_path = export_dir / "simulation_snapshot.json"
    if not snapshot_path.exists():
        return False
    try:
        snapshot = json.loads(snapshot_path.read_text())
    except json.JSONDecodeError:
        return False
    snapshot["emails"] = list(db.emails.find({}, {"_id": 0}).sort("timestamp", 1))
    snapshot["artifacts"] = list(
        db.artifacts.find({}, {"_id": 0, "embedding": 0}).sort("timestamp", 1)
    )
    snapshot["events"] = list(
        db.events.find({}, {"_id": 0, "embedding": 0}).sort("timestamp", 1)
    )
    snapshot_path.write_text(json.dumps(snapshot, indent=2, default=str))
    return True


def main() -> None:
    args = parse_args()
    export_dir = Path(args.export_dir).expanduser().resolve()
    client = MongoClient(args.mongo_uri)
    db = client[args.db]

    aligned = 0 if args.dry_run else align_inbound_embed_ids(db)
    emails = {
        doc.get("embed_id") or str(doc["_id"]): doc
        for doc in db.emails.find({}, {"embedding": 0})
    }
    parents = build_parent_map(db, emails)
    synthesized_parents, dropped_parent_edges = (
        (0, 0)
        if args.dry_run
        else repair_missing_parent_emails(db, emails, parents, export_dir)
    )
    roots, depths = resolve_thread_ids(emails, parents)
    date_to_day = load_date_to_day(db)
    ts_repairs = repair_timestamps(db, emails, parents, date_to_day)

    written_files = 0
    updated_docs = 0
    if not args.dry_run:
        for embed_id, doc in emails.items():
            if doc.get("timestamp"):
                doc["date"] = str(doc["timestamp"])[:10]
            if doc.get("date") in date_to_day:
                doc["day"] = date_to_day[doc["date"]]
            doc["thread_id"] = roots.get(embed_id, embed_id)
            doc["reply_to_email_id"] = parents.get(embed_id, "")
            doc["thread_order"] = depths.get(embed_id, reply_depth(doc.get("subject", "")))
            doc["eml_path"] = write_eml(export_dir, doc)
            written_files += 1

            update = {
                "embed_id": embed_id,
                "timestamp": doc.get("timestamp"),
                "date": doc.get("date"),
                "day": doc.get("day"),
                "thread_id": doc.get("thread_id"),
                "reply_to_email_id": doc.get("reply_to_email_id"),
                "thread_order": doc.get("thread_order"),
                "eml_path": doc.get("eml_path"),
            }
            email_filter = {"_id": doc["_id"]} if "_id" in doc else {"embed_id": embed_id}
            db.emails.update_one(email_filter, {"$set": update}, upsert=True)

            artifact_update = {
                "timestamp": doc.get("timestamp"),
                "date": doc.get("date"),
                "day": doc.get("day"),
                "metadata.thread_id": doc.get("thread_id"),
                "metadata.reply_to_email_id": doc.get("reply_to_email_id"),
                "metadata.file_path": doc.get("eml_path"),
            }
            db.artifacts.update_one(
                {"_id": embed_id},
                {
                    "$set": artifact_update,
                    "$setOnInsert": {
                        "_id": embed_id,
                        "type": "email",
                        "title": doc.get("subject", embed_id),
                        "content": (
                            f"From: {doc.get('from_name', '')}\n"
                            f"To: {doc.get('to_name', '')}\n"
                            f"Subject: {doc.get('subject', '')}\n\n"
                            f"{doc.get('body', '')}"
                        ),
                        "embedding": None,
                    },
                },
                upsert=True,
            )

            event_update = {
                "timestamp": doc.get("timestamp"),
                "date": doc.get("date"),
                "day": doc.get("day"),
                "artifact_ids.eml_path": doc.get("eml_path"),
            }
            db.events.update_many(
                {
                    "$or": [
                        {"artifact_ids.email": embed_id},
                        {"artifact_ids.embed_id": embed_id},
                        {"artifact_ids.email_thread": embed_id},
                    ]
                },
                {"$set": event_update},
            )
            db.sf_opps.update_many(
                {"touchpoints.embed_id": embed_id},
                {
                    "$set": {
                        "touchpoints.$.timestamp": doc.get("timestamp"),
                        "updated_at": doc.get("timestamp"),
                    }
                },
            )
            updated_docs += 1

        snapshot_updated = update_snapshot(export_dir, db)
    else:
        snapshot_updated = False

    print(f"aligned_embed_ids={aligned}")
    print(f"synthesized_parent_emails={synthesized_parents}")
    print(f"dropped_missing_parent_edges={dropped_parent_edges}")
    print(f"parent_edges={len(parents)}")
    print(f"timestamp_repairs={ts_repairs}")
    print(f"updated_email_docs={updated_docs}")
    print(f"rewritten_eml_files={written_files}")
    print(f"snapshot_updated={snapshot_updated}")


if __name__ == "__main__":
    main()
