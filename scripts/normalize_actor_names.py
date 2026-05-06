#!/usr/bin/env python
"""Normalize event actor aliases to configured full names."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from pymongo import MongoClient
import yaml


DEFAULT_MONGO_URI = "mongodb://localhost:27017/?directConnection=true"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mongo-uri", default=os.environ.get("MONGO_URI", DEFAULT_MONGO_URI))
    parser.add_argument("--db", default=os.environ.get("DB_NAME", "orgforge"))
    parser.add_argument("--config", default=os.environ.get("ORGFORGE_CONFIG_PATH", ""))
    parser.add_argument("--export-dir", default=os.environ.get("ORGFORGE_EXPORT_DIR", "export"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _load_config(path: str) -> dict:
    if not path:
        return {}
    with open(Path(path).expanduser(), "r") as fh:
        return yaml.safe_load(fh) or {}


def _alias_map(config: dict, db) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for source in config.get("external_contacts", []) or []:
        name = source.get("name")
        first = source.get("first_name")
        if name and first and first != name:
            aliases[first] = name

    source_doc = db.sim_config.find_one({"_id": "inbound_email_sources"}) or {}
    for source in source_doc.get("sources", []) or []:
        name = source.get("name")
        first = source.get("first_name")
        if name and first and first != name:
            aliases[first] = name

    for account in db.sf_accounts.find({}, {"primary_contact_name": 1, "primary_contact": 1, "_id": 0}):
        for key in ("primary_contact_name", "primary_contact"):
            name = account.get(key)
            if name:
                first = str(name).split()[0]
                if first and first != name:
                    aliases[first] = name
    return aliases


def _replace_aliases(value, aliases: dict[str, str]):
    if isinstance(value, list):
        changed = False
        updated = []
        for item in value:
            replacement = aliases.get(item, item)
            changed = changed or replacement != item
            updated.append(replacement)
        return updated, changed
    if isinstance(value, str):
        replacement = aliases.get(value, value)
        return replacement, replacement != value
    return value, False


def _update_snapshot(export_dir: Path, db) -> bool:
    snapshot_path = export_dir / "simulation_snapshot.json"
    if not snapshot_path.exists():
        return False
    try:
        snapshot = json.loads(snapshot_path.read_text())
    except json.JSONDecodeError:
        return False
    snapshot["events"] = list(db.events.find({}, {"_id": 0, "embedding": 0}).sort("timestamp", 1))
    snapshot_path.write_text(json.dumps(snapshot, indent=2, default=str))
    return True


def main() -> int:
    args = parse_args()
    config = _load_config(args.config)
    client = MongoClient(args.mongo_uri)
    db = client[args.db]
    aliases = _alias_map(config, db)

    updated_events = 0
    for event in db.events.find({"actors": {"$exists": True}}, {"actors": 1}):
        actors, changed = _replace_aliases(event.get("actors", []), aliases)
        if changed:
            updated_events += 1
            if not args.dry_run:
                db.events.update_one({"_id": event["_id"]}, {"$set": {"actors": actors}})

    snapshot_updated = False
    if not args.dry_run:
        snapshot_updated = _update_snapshot(Path(args.export_dir).expanduser().resolve(), db)

    print(f"aliases={len(aliases)}")
    print(f"updated_events={updated_events}")
    print(f"snapshot_updated={snapshot_updated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
