#!/usr/bin/env python
"""Import an OrgForge export snapshot and launch the local viewer."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dataset_validation import validate_dataset


DEFAULT_MONGO_URI = "mongodb://localhost:27017/?directConnection=true"

SNAPSHOT_COLLECTIONS = {
    "artifacts": "artifacts",
    "events": "events",
    "jira_tickets": "jira_tickets",
    "emails": "emails",
    "slack_threads": "slack_messages",
    "pr_registry": "pull_requests",
    "sf_accounts": "sf_accounts",
    "sf_opps": "sf_opps",
    "zd_tickets": "zd_tickets",
    "confluence_pages": "confluence_pages",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "One-command setup for browsing an exported OrgForge dataset. "
            "Starts MongoDB with docker compose when needed, imports "
            "simulation_snapshot.json, validates the dataset, then launches the viewer."
        )
    )
    parser.add_argument(
        "export_dir",
        nargs="?",
        default="export/velomind_30d",
        help="Export directory containing simulation_snapshot.json.",
    )
    parser.add_argument("--db", default="", help="MongoDB database name to load.")
    parser.add_argument("--mongo-uri", default=os.environ.get("MONGO_URI", DEFAULT_MONGO_URI))
    parser.add_argument("--host", default=os.environ.get("VIEWER_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("VIEWER_PORT", "8765")))
    parser.add_argument(
        "--config",
        default="",
        help="Optional config YAML for validation. Defaults to config/velomind.yaml when present.",
    )
    parser.add_argument("--skip-import", action="store_true", help="Do not import the snapshot.")
    parser.add_argument("--skip-validate", action="store_true", help="Do not run dataset validation.")
    parser.add_argument("--import-only", action="store_true", help="Import/validate, then exit.")
    parser.add_argument(
        "--no-docker",
        action="store_true",
        help="Do not try to start docker compose mongodb when MongoDB is unreachable.",
    )
    return parser.parse_args()


def _derive_db_name(export_dir: Path) -> str:
    slug = re.sub(r"[^A-Za-z0-9_]+", "_", export_dir.name).strip("_").lower()
    return f"orgforge_{slug or 'export'}"


def _snapshot_path(export_dir: Path) -> Path:
    return export_dir / "simulation_snapshot.json"


def _mongo_client(uri: str) -> MongoClient:
    return MongoClient(uri, serverSelectionTimeoutMS=1000)


def _ping(uri: str) -> bool:
    client = _mongo_client(uri)
    try:
        client.admin.command("ping")
        return True
    except ServerSelectionTimeoutError:
        return False
    finally:
        client.close()


def ensure_mongodb(uri: str, start_docker: bool) -> None:
    if _ping(uri):
        return
    if not start_docker:
        raise RuntimeError(
            "MongoDB is not reachable. Start it with `docker compose up -d mongodb` "
            "or pass a reachable --mongo-uri."
        )

    print("MongoDB is not reachable. Starting docker compose mongodb...")
    subprocess.run(["docker", "compose", "up", "-d", "mongodb"], cwd=ROOT, check=True)

    deadline = time.time() + 60
    while time.time() < deadline:
        if _ping(uri):
            return
        time.sleep(1)
    raise RuntimeError("MongoDB did not become reachable within 60 seconds.")


def _insert_many(collection, docs: list[dict[str, Any]], batch_size: int = 1000) -> None:
    for start in range(0, len(docs), batch_size):
        collection.insert_many(docs[start : start + batch_size])


def import_snapshot(export_dir: Path, uri: str, db_name: str) -> dict[str, int]:
    path = _snapshot_path(export_dir)
    if not path.exists():
        raise FileNotFoundError(f"Missing snapshot: {path}")
    snapshot = json.loads(path.read_text())
    client = _mongo_client(uri)
    try:
        db = client[db_name]
        counts: dict[str, int] = {}
        for snapshot_key, collection_name in SNAPSHOT_COLLECTIONS.items():
            docs = snapshot.get(snapshot_key, [])
            collection = db[collection_name]
            collection.delete_many({})
            if docs:
                _insert_many(collection, docs)
            counts[collection_name] = len(docs)

        _ensure_viewer_indexes(db)
        db.sim_config.update_one(
            {"_id": "viewer_import"},
            {
                "$set": {
                    "export_dir": str(export_dir),
                    "snapshot": str(path),
                    "import_counts": counts,
                    "imported_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
            },
            upsert=True,
        )
        return counts
    finally:
        client.close()


def _ensure_viewer_indexes(db) -> None:
    db.artifacts.create_index([("type", 1), ("timestamp", -1)])
    db.events.create_index([("type", 1), ("timestamp", -1)])
    db.jira_tickets.create_index([("id", 1)])
    db.emails.create_index([("embed_id", 1)])
    db.emails.create_index([("thread_id", 1), ("timestamp", 1)])
    db.slack_messages.create_index([("channel", 1), ("thread_id", 1), ("ts", 1)])
    db.pull_requests.create_index([("pr_id", 1)])


def _load_config(path: str, export_dir: Path) -> dict[str, Any] | None:
    candidates = []
    if path:
        candidates.append(Path(path).expanduser())
    candidates.append(ROOT / "config" / "velomind.yaml")
    for candidate in candidates:
        if candidate.exists():
            with open(candidate, "r") as fh:
                return yaml.safe_load(fh) or {}
    return None


def validate_import(export_dir: Path, uri: str, db_name: str, config_path: str) -> None:
    client = _mongo_client(uri)
    try:
        report = validate_dataset(
            client[db_name],
            export_dir=export_dir,
            config=_load_config(config_path, export_dir),
        )
    finally:
        client.close()

    status = "ok" if report["ok"] else "failed"
    print(
        f"dataset_validation={status} "
        f"errors={report['error_count']} warnings={report['warning_count']}"
    )
    if not report["ok"]:
        for item in report["errors"][:10]:
            print("ERROR", json.dumps(item, default=str))
        raise RuntimeError("Dataset validation failed.")


def launch_viewer(export_dir: Path, uri: str, db_name: str, host: str, port: int) -> int:
    url_host = "localhost" if host in {"0.0.0.0", "127.0.0.1"} else host
    print(f"Launching viewer: http://{url_host}:{port}")
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "src" / "viewer.py"),
            "--mongo-uri",
            uri,
            "--db",
            db_name,
            "--export-dir",
            str(export_dir),
            "--host",
            host,
            "--port",
            str(port),
        ],
        cwd=ROOT,
    ).returncode


def main() -> int:
    args = parse_args()
    export_dir = Path(args.export_dir).expanduser()
    if not export_dir.is_absolute():
        export_dir = ROOT / export_dir
    export_dir = export_dir.resolve()
    db_name = args.db or _derive_db_name(export_dir)

    ensure_mongodb(args.mongo_uri, start_docker=not args.no_docker)

    if not args.skip_import:
        counts = import_snapshot(export_dir, args.mongo_uri, db_name)
        summary = ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))
        print(f"Imported {db_name}: {summary}")

    if not args.skip_validate:
        validate_import(export_dir, args.mongo_uri, db_name, args.config)

    if args.import_only:
        print(f"Import complete. Start viewer later with --skip-import --db {db_name}.")
        return 0
    return launch_viewer(export_dir, args.mongo_uri, db_name, args.host, args.port)


if __name__ == "__main__":
    raise SystemExit(main())
