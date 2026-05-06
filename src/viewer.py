"""
OrgForge dataset viewer.

Run locally:
    python src/viewer.py --db orgforge --export-dir export

The server intentionally uses the Python standard library for HTTP/static
serving so the viewer does not add another app framework to the project.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
from datetime import date, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from bson import ObjectId
from pymongo import MongoClient
from pymongo.errors import PyMongoError


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_ROOT = Path(__file__).resolve().parent / "viewer_static"
DEFAULT_MONGO_URI = "mongodb://localhost:27017/?directConnection=true"
DEFAULT_EXPORT_DIR = PROJECT_ROOT / "export"

TEXT_EXTENSIONS = {
    ".json",
    ".md",
    ".txt",
    ".log",
    ".eml",
    ".csv",
    ".yaml",
    ".yml",
}

COMMON_TEXT_FIELDS = [
    "id",
    "title",
    "summary",
    "content",
    "description",
    "text",
    "body",
    "subject",
    "type",
    "status",
    "assignee",
    "dept",
    "channel",
    "from_name",
    "from_addr",
    "to_name",
    "to_addr",
    "account_name",
    "opportunity_id",
    "vendor",
    "source",
    "date",
    "timestamp",
]

PREFERRED_COLLECTIONS = [
    "artifacts",
    "events",
    "jira_tickets",
    "slack_messages",
    "emails",
    "sf_accounts",
    "sf_opps",
    "dept_plans",
    "conversation_summaries",
    "domain_registry",
]


class ViewerState:
    def __init__(self, mongo_uri: str, db_name: str, export_dir: Path):
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.export_dir = export_dir.expanduser().resolve()
        self.client = MongoClient(mongo_uri, serverSelectionTimeoutMS=1500)


STATE: ViewerState


def _json_default(value: Any) -> Any:
    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    return str(value)


def _sanitize(value: Any, key: str | None = None) -> Any:
    if key == "embedding" and isinstance(value, list):
        sample = [round(float(x), 5) for x in value[:8] if isinstance(x, (int, float))]
        return {"kind": "vector", "dimensions": len(value), "sample": sample}
    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    if isinstance(value, dict):
        return {str(k): _sanitize(v, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    return value


def _truncate(text: Any, limit: int = 260) -> str:
    if text is None:
        return ""
    raw = str(text).replace("\n", " ").strip()
    if len(raw) <= limit:
        return raw
    return raw[: limit - 1].rstrip() + "..."


def _doc_label(doc: dict[str, Any]) -> str:
    for field in (
        "title",
        "subject",
        "summary",
        "id",
        "opportunity_id",
        "account_name",
        "name",
        "type",
        "_id",
    ):
        value = doc.get(field)
        if value:
            return _truncate(value, 90)
    return "Untitled document"


def _doc_summary(doc: dict[str, Any]) -> str:
    for field in ("summary", "description", "content", "body", "text", "theme"):
        value = doc.get(field)
        if value:
            return _truncate(value)
    facts = doc.get("facts")
    if isinstance(facts, dict):
        return _truncate(json.dumps(facts, default=_json_default))
    return ""


def _doc_meta(doc: dict[str, Any]) -> list[str]:
    meta = []
    for field in (
        "type",
        "status",
        "dept",
        "assignee",
        "date",
        "timestamp",
        "created_at",
        "channel",
        "direction",
    ):
        value = doc.get(field)
        if value not in (None, "", []):
            meta.append(f"{field}: {value}")
    return meta[:6]


def _summarize_doc(doc: dict[str, Any]) -> dict[str, Any]:
    sanitized = _sanitize(doc)
    doc_id = sanitized.get("_id") or sanitized.get("id") or _doc_label(sanitized)
    return {
        "_id": str(doc_id),
        "label": _doc_label(sanitized),
        "summary": _doc_summary(sanitized),
        "meta": _doc_meta(sanitized),
        "doc": sanitized,
    }


def _mask_mongo_uri(uri: str) -> str:
    match = re.match(r"^(mongodb(?:\\+srv)?://)([^/@]+)@(.+)$", uri)
    if not match:
        return uri
    return f"{match.group(1)}***:***@{match.group(3)}"


def _query_value(query: dict[str, list[str]], name: str, default: str = "") -> str:
    values = query.get(name)
    if not values:
        return default
    return values[0]


def _query_int(
    query: dict[str, list[str]], name: str, default: int, minimum: int, maximum: int
) -> int:
    try:
        value = int(_query_value(query, name, str(default)))
    except ValueError:
        return default
    return max(minimum, min(maximum, value))


def _db(name: str | None = None):
    return STATE.client[name or STATE.db_name]


def _collection_names(db_name: str | None = None) -> list[str]:
    names = _db(db_name).list_collection_names()
    preferred = [name for name in PREFERRED_COLLECTIONS if name in names]
    rest = sorted(name for name in names if name not in preferred)
    return preferred + rest


def _count_by_field(collection: str, field: str, limit: int = 12) -> list[dict[str, Any]]:
    try:
        rows = list(
            _db()[collection].aggregate(
                [
                    {"$match": {field: {"$exists": True, "$ne": None}}},
                    {"$group": {"_id": f"${field}", "count": {"$sum": 1}}},
                    {"$sort": {"count": -1, "_id": 1}},
                    {"$limit": limit},
                ]
            )
        )
    except PyMongoError:
        return []
    return [{"label": str(row["_id"]), "count": row["count"]} for row in rows]


def _file_kind(path: Path) -> str:
    try:
        rel = path.relative_to(STATE.export_dir)
    except ValueError:
        return path.suffix.lstrip(".") or "file"
    parts = rel.parts
    if not parts:
        return "file"
    if parts[0] == "salesforce" and len(parts) > 1:
        return f"salesforce/{parts[1]}"
    return parts[0] if len(parts) > 1 else path.suffix.lstrip(".") or "file"


def _iter_files() -> list[Path]:
    if not STATE.export_dir.exists():
        return []
    return sorted(
        p
        for p in STATE.export_dir.rglob("*")
        if p.is_file() and not p.name.startswith(".")
    )


def _file_record(path: Path) -> dict[str, Any]:
    rel = path.relative_to(STATE.export_dir)
    stat = path.stat()
    return {
        "path": rel.as_posix(),
        "name": path.name,
        "kind": _file_kind(path),
        "extension": path.suffix.lower() or "none",
        "size": stat.st_size,
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
    }


def _read_text_file(path: Path, max_bytes: int = 750_000) -> str:
    raw = path.read_bytes()[:max_bytes]
    return raw.decode("utf-8", errors="replace")


def _safe_export_path(relative_path: str) -> Path:
    candidate = (STATE.export_dir / unquote(relative_path)).resolve()
    if STATE.export_dir not in candidate.parents and candidate != STATE.export_dir:
        raise ValueError("path is outside export directory")
    if not candidate.is_file():
        raise FileNotFoundError(relative_path)
    return candidate


def _search_filter(raw_query: str) -> dict[str, Any]:
    if not raw_query:
        return {}
    regex = re.compile(re.escape(raw_query), re.IGNORECASE)
    return {"$or": [{field: regex} for field in COMMON_TEXT_FIELDS]}


def _default_sort_field(collection_name: str) -> str:
    collection = _db()[collection_name]
    for field in ("timestamp", "created_at", "updated_at", "ts", "date", "day"):
        if collection.find_one({field: {"$exists": True}}, {"_id": 1}):
            return field
    return "_id"


def _collection_exists(name: str) -> bool:
    return name in _collection_names()


def _find_docs(
    collection: str,
    filter_doc: dict[str, Any] | None = None,
    sort: list[tuple[str, int]] | None = None,
    limit: int = 2000,
) -> list[dict[str, Any]]:
    if not _collection_exists(collection):
        return []
    cursor = _db()[collection].find(filter_doc or {}, {"embedding": 0})
    if sort:
        cursor = cursor.sort(sort)
    if limit:
        cursor = cursor.limit(limit)
    return [_sanitize(doc) for doc in cursor]


def _normalize_subject(subject: str) -> str:
    cleaned = subject.strip()
    while True:
        updated = re.sub(r"^(re|fw|fwd):\s*", "", cleaned, flags=re.IGNORECASE)
        if updated == cleaned:
            break
        cleaned = updated
    return re.sub(r"\s+", " ", cleaned).strip()


def _thread_id(subject: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", _normalize_subject(subject).lower()).strip("-")


def _reply_depth(subject: str) -> int:
    depth = 0
    cleaned = subject.strip()
    while True:
        match = re.match(r"^(re|fw|fwd):\s*", cleaned, flags=re.IGNORECASE)
        if not match:
            return depth
        depth += 1
        cleaned = cleaned[match.end() :]


def _contains_query(value: Any, query: str) -> bool:
    if not query:
        return True
    return query.lower() in json.dumps(value, default=_json_default).lower()


def _load_snapshot() -> dict[str, Any]:
    path = STATE.export_dir / "simulation_snapshot.json"
    if not path.exists():
        return {}
    try:
        return _sanitize(json.loads(path.read_text()))
    except (OSError, json.JSONDecodeError):
        return {}


def _artifact_by_id(artifact_id: str) -> dict[str, Any] | None:
    if not artifact_id or not _collection_exists("artifacts"):
        return None
    doc = _db()["artifacts"].find_one({"_id": artifact_id}, {"embedding": 0})
    return _sanitize(doc) if doc else None


def _event_refs_for_artifact(artifact_id: str, limit: int = 20) -> list[dict[str, Any]]:
    if not artifact_id or not _collection_exists("events"):
        return []
    cursor = (
        _db()["events"]
        .find(
            {
                "$or": [
                    {"artifact_ids": artifact_id},
                    {"artifact_ids.jira": artifact_id},
                    {"artifact_ids.confluence": artifact_id},
                    {"artifact_ids.email": artifact_id},
                    {"artifact_ids.slack_thread": artifact_id},
                ]
            },
            {"embedding": 0},
        )
        .sort("timestamp", 1)
        .limit(limit)
    )
    return [_summarize_doc(doc) for doc in cursor]


def api_config(_: dict[str, list[str]]) -> dict[str, Any]:
    try:
        db_names = sorted(
            name
            for name in STATE.client.list_database_names()
            if name not in {"admin", "config", "local"}
        )
    except PyMongoError:
        db_names = []
    return {
        "mongo_uri": _mask_mongo_uri(STATE.mongo_uri),
        "db_name": STATE.db_name,
        "databases": db_names,
        "export_dir": str(STATE.export_dir),
        "export_exists": STATE.export_dir.exists(),
    }


def api_select_db(query: dict[str, list[str]]) -> dict[str, Any]:
    db_name = _query_value(query, "db").strip()
    if not db_name:
        raise KeyError("db is required")
    db_names = STATE.client.list_database_names()
    if db_name not in db_names:
        raise FileNotFoundError(db_name)
    STATE.db_name = db_name
    return api_config(query)


def api_overview(_: dict[str, list[str]]) -> dict[str, Any]:
    collections = []
    for name in _collection_names():
        collection = _db()[name]
        count = collection.count_documents({})
        sample = collection.find_one({}, {"embedding": 0}) or {}
        collections.append(
            {
                "name": name,
                "count": count,
                "fields": sorted(str(field) for field in sample.keys())[:18],
            }
        )

    files = [_file_record(path) for path in _iter_files()]
    by_kind: dict[str, int] = {}
    by_ext: dict[str, int] = {}
    for record in files:
        by_kind[record["kind"]] = by_kind.get(record["kind"], 0) + 1
        by_ext[record["extension"]] = by_ext.get(record["extension"], 0) + 1

    return {
        "collections": collections,
        "metrics": {
            "artifacts": _db()["artifacts"].count_documents({})
            if "artifacts" in [c["name"] for c in collections]
            else 0,
            "events": _db()["events"].count_documents({})
            if "events" in [c["name"] for c in collections]
            else 0,
            "jira_tickets": _db()["jira_tickets"].count_documents({})
            if "jira_tickets" in [c["name"] for c in collections]
            else 0,
            "slack_messages": _db()["slack_messages"].count_documents({})
            if "slack_messages" in [c["name"] for c in collections]
            else 0,
            "emails": _db()["emails"].count_documents({})
            if "emails" in [c["name"] for c in collections]
            else 0,
            "files": len(files),
        },
        "breakdowns": {
            "event_types": _count_by_field("events", "type"),
            "artifact_types": _count_by_field("artifacts", "type"),
            "jira_status": _count_by_field("jira_tickets", "status"),
            "slack_channels": _count_by_field("slack_messages", "channel"),
            "email_direction": _count_by_field("emails", "direction"),
            "file_kinds": [
                {"label": key, "count": value}
                for key, value in sorted(by_kind.items(), key=lambda item: (-item[1], item[0]))
            ],
            "file_extensions": [
                {"label": key, "count": value}
                for key, value in sorted(by_ext.items(), key=lambda item: (-item[1], item[0]))
            ],
        },
    }


def api_collections(_: dict[str, list[str]]) -> dict[str, Any]:
    return {
        "collections": [
            {"name": name, "count": _db()[name].count_documents({})}
            for name in _collection_names()
        ]
    }


def api_documents(query: dict[str, list[str]]) -> dict[str, Any]:
    name = _query_value(query, "collection", "artifacts")
    if name not in _collection_names():
        raise KeyError(f"unknown collection: {name}")

    search = _query_value(query, "q", "").strip()
    limit = _query_int(query, "limit", 40, 1, 250)
    skip = _query_int(query, "skip", 0, 0, 100_000)
    sort = _query_value(query, "sort", _default_sort_field(name))
    direction = -1 if _query_value(query, "direction", "desc") != "asc" else 1
    collection = _db()[name]
    filter_doc = _search_filter(search)

    cursor = (
        collection.find(filter_doc, {"embedding": 0})
        .sort(sort, direction)
        .skip(skip)
        .limit(limit)
    )
    docs = [_summarize_doc(doc) for doc in cursor]
    total = collection.count_documents(filter_doc)
    return {
        "collection": name,
        "total": total,
        "skip": skip,
        "limit": limit,
        "documents": docs,
        "sort": sort,
        "direction": "desc" if direction == -1 else "asc",
    }


def api_document(query: dict[str, list[str]]) -> dict[str, Any]:
    name = _query_value(query, "collection")
    doc_id = _query_value(query, "id")
    if not name or not doc_id:
        raise KeyError("collection and id are required")
    if name not in _collection_names():
        raise KeyError(f"unknown collection: {name}")

    candidates: list[dict[str, Any]] = [
        {"_id": doc_id},
        {"id": doc_id},
        {"embed_id": doc_id},
        {"opportunity_id": doc_id},
        {"account_id": doc_id},
    ]
    if ObjectId.is_valid(doc_id):
        candidates.insert(0, {"_id": ObjectId(doc_id)})

    doc = _db()[name].find_one({"$or": candidates}, {"embedding": 0})
    if not doc:
        raise FileNotFoundError(doc_id)
    return _summarize_doc(doc)


def api_files(query: dict[str, list[str]]) -> dict[str, Any]:
    search = _query_value(query, "q", "").strip().lower()
    kind = _query_value(query, "kind", "").strip()
    limit = _query_int(query, "limit", 200, 1, 1000)

    records = []
    kinds = set()
    for path in _iter_files():
        record = _file_record(path)
        kinds.add(record["kind"])
        haystack = f"{record['path']} {record['kind']} {record['extension']}".lower()
        if kind and record["kind"] != kind:
            continue
        if search and search not in haystack:
            continue
        records.append(record)

    return {
        "files": records[:limit],
        "kinds": sorted(kinds),
        "total": len(records),
        "export_dir": str(STATE.export_dir),
    }


def api_file(query: dict[str, list[str]]) -> dict[str, Any]:
    relative_path = _query_value(query, "path")
    path = _safe_export_path(relative_path)
    ext = path.suffix.lower()
    if ext not in TEXT_EXTENSIONS:
        return {**_file_record(path), "text": "", "binary": True}

    text = _read_text_file(path)
    parsed = None
    if ext == ".json":
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
    return {**_file_record(path), "text": text, "json": _sanitize(parsed)}


def api_search(query: dict[str, list[str]]) -> dict[str, Any]:
    raw = _query_value(query, "q", "").strip()
    limit = _query_int(query, "limit", 8, 1, 50)
    if not raw:
        return {"query": raw, "mongo": [], "files": []}

    mongo_hits = []
    for name in _collection_names():
        try:
            cursor = _db()[name].find(_search_filter(raw), {"embedding": 0}).limit(limit)
            for doc in cursor:
                hit = _summarize_doc(doc)
                hit["collection"] = name
                mongo_hits.append(hit)
                if len(mongo_hits) >= limit * 3:
                    break
        except PyMongoError:
            continue
        if len(mongo_hits) >= limit * 3:
            break

    file_hits = []
    lowered = raw.lower()
    for path in _iter_files():
        record = _file_record(path)
        haystack = record["path"].lower()
        snippet = ""
        if path.suffix.lower() in TEXT_EXTENSIONS:
            try:
                text = _read_text_file(path, max_bytes=120_000)
                idx = text.lower().find(lowered)
                if idx >= 0:
                    start = max(0, idx - 90)
                    end = min(len(text), idx + len(raw) + 180)
                    snippet = text[start:end].replace("\n", " ").strip()
                    haystack += " " + text.lower()
            except OSError:
                pass
        if lowered in haystack:
            record["snippet"] = _truncate(snippet or record["path"], 320)
            file_hits.append(record)
            if len(file_hits) >= limit * 3:
                break

    return {"query": raw, "mongo": mongo_hits, "files": file_hits}


def api_inbox(query: dict[str, list[str]]) -> dict[str, Any]:
    raw_query = _query_value(query, "q", "").strip()
    emails = _find_docs("emails", sort=[("timestamp", 1)], limit=5000)
    threads_by_id: dict[str, dict[str, Any]] = {}

    for email in emails:
        subject = email.get("subject", "No subject")
        tid = email.get("thread_id") or _thread_id(subject) or "no-subject"
        base_subject = _normalize_subject(subject) or subject
        thread = threads_by_id.setdefault(
            tid,
            {
                "id": tid,
                "subject": base_subject,
                "participants": [],
                "participant_emails": [],
                "directions": set(),
                "messages": [],
                "first": email.get("timestamp") or email.get("date"),
                "last": email.get("timestamp") or email.get("date"),
            },
        )
        thread["messages"].append(email)
        thread["directions"].add(email.get("direction", "unknown"))
        for name_key, email_key in (("from_name", "from_addr"), ("to_name", "to_addr")):
            label = email.get(name_key) or email.get(email_key)
            addr = email.get(email_key)
            if label and label not in thread["participants"]:
                thread["participants"].append(label)
            if addr and addr not in thread["participant_emails"]:
                thread["participant_emails"].append(addr)
        ts = email.get("timestamp") or email.get("date")
        if ts:
            thread["first"] = min(thread["first"], ts) if thread["first"] else ts
            thread["last"] = max(thread["last"], ts) if thread["last"] else ts

    threads = []
    for thread in threads_by_id.values():
        thread["messages"] = sorted(
            thread["messages"],
            key=lambda msg: (
                msg.get("timestamp", ""),
                int(msg.get("thread_order", 0) or 0),
                _reply_depth(msg.get("subject", "")),
            ),
        )
        thread["count"] = len(thread["messages"])
        thread["directions"] = sorted(thread["directions"])
        thread["preview"] = _truncate(thread["messages"][-1].get("body", ""), 220)
        if _contains_query(thread, raw_query):
            threads.append(thread)

    threads.sort(key=lambda item: item.get("last", ""), reverse=True)
    return {
        "threads": threads,
        "total_messages": len(emails),
        "total_threads": len(threads),
    }


def api_slack_app(query: dict[str, list[str]]) -> dict[str, Any]:
    raw_query = _query_value(query, "q", "").strip()
    messages = _find_docs("slack_messages", sort=[("ts", 1)], limit=20000)
    channels_by_name: dict[str, dict[str, Any]] = {}

    for message in messages:
        channel_name = message.get("channel", "unknown")
        thread_id = message.get("thread_id") or f"{channel_name}:{message.get('ts', '')}"
        channel = channels_by_name.setdefault(
            channel_name,
            {
                "name": channel_name,
                "kind": "dm" if channel_name.startswith("dm_") else "channel",
                "message_count": 0,
                "thread_count": 0,
                "participants": [],
                "threads_by_id": {},
                "last": "",
            },
        )
        channel["message_count"] += 1
        if message.get("user") and message["user"] not in channel["participants"]:
            channel["participants"].append(message["user"])
        if message.get("ts"):
            channel["last"] = max(channel["last"], message["ts"])

        thread = channel["threads_by_id"].setdefault(
            thread_id,
            {
                "id": thread_id,
                "channel": channel_name,
                "messages": [],
                "participants": [],
                "first": message.get("ts", ""),
                "last": message.get("ts", ""),
                "preview": "",
            },
        )
        thread["messages"].append(message)
        if message.get("user") and message["user"] not in thread["participants"]:
            thread["participants"].append(message["user"])
        if message.get("ts"):
            thread["first"] = min(thread["first"], message["ts"]) if thread["first"] else message["ts"]
            thread["last"] = max(thread["last"], message["ts"]) if thread["last"] else message["ts"]

    channels = []
    for channel in channels_by_name.values():
        threads = []
        for thread in channel.pop("threads_by_id").values():
            thread["messages"] = sorted(thread["messages"], key=lambda msg: msg.get("ts", ""))
            thread["count"] = len(thread["messages"])
            thread["preview"] = _truncate(thread["messages"][0].get("text", ""), 180)
            if _contains_query(thread, raw_query):
                threads.append(thread)
        threads.sort(key=lambda item: item.get("last", ""), reverse=True)
        channel["threads"] = threads
        channel["thread_count"] = len(threads)
        if threads:
            channels.append(channel)

    channels.sort(key=lambda item: (item["kind"] != "channel", item["name"]))
    return {
        "channels": channels,
        "total_messages": len(messages),
        "total_threads": sum(channel["thread_count"] for channel in channels),
    }


def api_jira_app(query: dict[str, list[str]]) -> dict[str, Any]:
    raw_query = _query_value(query, "q", "").strip()
    tickets = [
        ticket
        for ticket in _find_docs("jira_tickets", sort=[("created_at", -1)], limit=10000)
        if _contains_query(ticket, raw_query)
    ]
    status_order = ["To Do", "In Progress", "In Review", "Done", "Closed"]
    statuses = sorted({ticket.get("status", "Unknown") for ticket in tickets})
    ordered_statuses = [status for status in status_order if status in statuses] + [
        status for status in statuses if status not in status_order
    ]
    columns = []
    for status in ordered_statuses:
        status_tickets = [ticket for ticket in tickets if ticket.get("status", "Unknown") == status]
        columns.append({"status": status, "tickets": status_tickets, "count": len(status_tickets)})

    by_dept: dict[str, int] = {}
    by_completion: dict[str, int] = {}
    for ticket in tickets:
        by_dept[ticket.get("dept") or "Unassigned"] = by_dept.get(ticket.get("dept") or "Unassigned", 0) + 1
        by_completion[ticket.get("completion_artifact") or ticket.get("type") or "task"] = (
            by_completion.get(ticket.get("completion_artifact") or ticket.get("type") or "task", 0)
            + 1
        )

    return {
        "columns": columns,
        "tickets": tickets,
        "by_dept": [{"label": k, "count": v} for k, v in sorted(by_dept.items())],
        "by_completion": [
            {"label": k, "count": v} for k, v in sorted(by_completion.items())
        ],
    }


def api_docs_app(query: dict[str, list[str]]) -> dict[str, Any]:
    raw_query = _query_value(query, "q", "").strip()
    artifact_docs = []
    for doc in _find_docs(
        "artifacts",
        {"type": {"$in": ["confluence", "zoom_transcript"]}},
        sort=[("timestamp", -1)],
        limit=5000,
    ):
        if not _contains_query(doc, raw_query):
            continue
        doc_id = str(doc.get("_id", ""))
        artifact_docs.append(
            {
                "id": doc_id,
                "title": doc.get("title", doc_id),
                "type": doc.get("type", "document"),
                "date": doc.get("date"),
                "timestamp": doc.get("timestamp"),
                "author": (doc.get("metadata") or {}).get("author"),
                "participants": (doc.get("metadata") or {}).get("participants", []),
                "topic": (doc.get("metadata") or {}).get("topic"),
                "content": doc.get("content", ""),
                "metadata": doc.get("metadata", {}),
                "events": _event_refs_for_artifact(doc_id, limit=8),
            }
        )

    domains = _find_docs("domain_registry", sort=[("domain", 1)], limit=1000)
    return {
        "documents": artifact_docs,
        "domains": domains,
        "counts": {
            "confluence": sum(1 for doc in artifact_docs if doc["type"] == "confluence"),
            "meetings": sum(1 for doc in artifact_docs if doc["type"] == "zoom_transcript"),
            "domains": len(domains),
        },
    }


def api_crm_app(query: dict[str, list[str]]) -> dict[str, Any]:
    raw_query = _query_value(query, "q", "").strip()
    accounts = _find_docs("sf_accounts", sort=[("arr", -1)], limit=5000)
    opps = _find_docs("sf_opps", sort=[("created_at", -1)], limit=5000)
    emails = _find_docs("emails", sort=[("timestamp", -1)], limit=5000)

    opps_by_account: dict[str, list[dict[str, Any]]] = {}
    for opp in opps:
        opps_by_account.setdefault(opp.get("account_name", ""), []).append(opp)

    account_rows = []
    for account in accounts:
        account_opps = opps_by_account.get(account.get("name", ""), [])
        contact_email = account.get("primary_contact_email", "")
        related_emails = [
            email
            for email in emails
            if contact_email
            and contact_email in {email.get("from_addr"), email.get("to_addr")}
        ]
        row = {
            **account,
            "opportunities": account_opps,
            "related_emails": related_emails,
            "open_pipeline": sum(float(opp.get("amount") or 0) for opp in account_opps),
        }
        if _contains_query(row, raw_query):
            account_rows.append(row)

    return {
        "accounts": account_rows,
        "opportunities": opps,
        "totals": {
            "accounts": len(account_rows),
            "opportunities": len(opps),
            "arr": sum(float(account.get("arr") or 0) for account in account_rows),
            "pipeline": sum(float(opp.get("amount") or 0) for opp in opps),
        },
    }


def api_org_app(query: dict[str, list[str]]) -> dict[str, Any]:
    raw_query = _query_value(query, "q", "").strip()
    snapshot = _load_snapshot()
    dept_plans = [
        plan
        for plan in _find_docs("dept_plans", sort=[("dept", 1)], limit=1000)
        if _contains_query(plan, raw_query)
    ]
    domains = [
        domain
        for domain in _find_docs("domain_registry", sort=[("domain", 1)], limit=1000)
        if _contains_query(domain, raw_query)
    ]
    config_docs = _find_docs("sim_config", limit=10)

    stress_snapshot = snapshot.get("stress_snapshot", {})
    top_stress = []
    if isinstance(stress_snapshot, dict):
        top_stress = [
            {"name": name, "stress": value}
            for name, value in sorted(
                stress_snapshot.items(), key=lambda item: item[1], reverse=True
            )[:16]
        ]

    return {
        "dept_plans": dept_plans,
        "domains": domains,
        "config": config_docs,
        "top_relationships": snapshot.get("top_relationships", []),
        "estranged_pairs": snapshot.get("estranged_pairs", [])[:16],
        "departed_employees": snapshot.get("departed_employees", []),
        "new_hires": snapshot.get("new_hires", []),
        "top_stress": top_stress,
        "morale_history": snapshot.get("morale_history", []),
        "system_health": snapshot.get("system_health"),
    }


def api_timeline_app(query: dict[str, list[str]]) -> dict[str, Any]:
    raw_query = _query_value(query, "q", "").strip()
    event_type = _query_value(query, "type", "").strip()
    filter_doc: dict[str, Any] = {}
    if event_type:
        filter_doc["type"] = event_type
    events = [
        event
        for event in _find_docs("events", filter_doc, sort=[("timestamp", 1)], limit=10000)
        if _contains_query(event, raw_query)
    ]
    type_counts = _count_by_field("events", "type", limit=100)
    days: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        days.setdefault(str(event.get("date") or "unknown"), []).append(event)
    return {
        "events": events,
        "days": [{"date": key, "events": value} for key, value in days.items()],
        "types": type_counts,
    }


API_ROUTES = {
    "/api/config": api_config,
    "/api/select-db": api_select_db,
    "/api/overview": api_overview,
    "/api/collections": api_collections,
    "/api/documents": api_documents,
    "/api/document": api_document,
    "/api/files": api_files,
    "/api/file": api_file,
    "/api/search": api_search,
    "/api/app/inbox": api_inbox,
    "/api/app/slack": api_slack_app,
    "/api/app/jira": api_jira_app,
    "/api/app/docs": api_docs_app,
    "/api/app/crm": api_crm_app,
    "/api/app/org": api_org_app,
    "/api/app/timeline": api_timeline_app,
}


class ViewerHandler(BaseHTTPRequestHandler):
    server_version = "OrgForgeViewer/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[viewer] {self.address_string()} - {fmt % args}")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/"):
            self._handle_api(parsed.path, parse_qs(parsed.query))
            return
        self._handle_static(parsed.path)

    def _handle_api(self, path: str, query: dict[str, list[str]]) -> None:
        route = API_ROUTES.get(path)
        if not route:
            self._write_json({"error": "unknown endpoint"}, HTTPStatus.NOT_FOUND)
            return
        try:
            data = route(query)
            self._write_json(data)
        except (KeyError, FileNotFoundError, ValueError) as exc:
            self._write_json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
        except PyMongoError as exc:
            self._write_json({"error": f"MongoDB error: {exc}"}, HTTPStatus.BAD_GATEWAY)
        except Exception as exc:
            self._write_json({"error": f"{type(exc).__name__}: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def _write_json(self, data: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        raw = json.dumps(data, default=_json_default).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _handle_static(self, path: str) -> None:
        if path in {"", "/"}:
            relative = "index.html"
        else:
            relative = unquote(path.lstrip("/"))

        candidate = (STATIC_ROOT / relative).resolve()
        if STATIC_ROOT not in candidate.parents and candidate != STATIC_ROOT:
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        if not candidate.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        raw = candidate.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Browse OrgForge generated data.")
    parser.add_argument("--host", default=os.environ.get("VIEWER_HOST", "127.0.0.1"))
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("VIEWER_PORT", "8765"))
    )
    parser.add_argument(
        "--mongo-uri", default=os.environ.get("MONGO_URI", DEFAULT_MONGO_URI)
    )
    parser.add_argument("--db", default=os.environ.get("DB_NAME", "orgforge"))
    parser.add_argument(
        "--export-dir",
        default=os.environ.get("ORGFORGE_EXPORT_DIR", str(DEFAULT_EXPORT_DIR)),
    )
    return parser.parse_args()


def main() -> None:
    global STATE
    args = parse_args()
    STATE = ViewerState(
        mongo_uri=args.mongo_uri,
        db_name=args.db,
        export_dir=Path(args.export_dir),
    )

    server = ThreadingHTTPServer((args.host, args.port), ViewerHandler)
    url_host = "localhost" if args.host in {"0.0.0.0", "127.0.0.1"} else args.host
    print(f"OrgForge viewer: http://{url_host}:{args.port}")
    print(f"MongoDB database: {STATE.db_name}")
    print(f"Export directory: {STATE.export_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping viewer")
    finally:
        server.server_close()
        STATE.client.close()


if __name__ == "__main__":
    main()
