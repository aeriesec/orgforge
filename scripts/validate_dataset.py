#!/usr/bin/env python
"""Validate an OrgForge MongoDB/export dataset."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from pymongo import MongoClient
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dataset_validation import validate_dataset


DEFAULT_MONGO_URI = "mongodb://localhost:27017/?directConnection=true"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mongo-uri", default=os.environ.get("MONGO_URI", DEFAULT_MONGO_URI))
    parser.add_argument("--db", default=os.environ.get("DB_NAME", "orgforge"))
    parser.add_argument("--export-dir", default=os.environ.get("ORGFORGE_EXPORT_DIR", "export"))
    parser.add_argument("--config", default=os.environ.get("ORGFORGE_CONFIG_PATH", ""))
    parser.add_argument("--json", action="store_true", help="Print full JSON report")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    client = MongoClient(args.mongo_uri)
    config = None
    if args.config:
        with open(Path(args.config).expanduser(), "r") as fh:
            config = yaml.safe_load(fh) or {}
    report = validate_dataset(client[args.db], export_dir=args.export_dir, config=config)
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        status = "ok" if report["ok"] else "failed"
        print(
            f"dataset_validation={status} "
            f"errors={report['error_count']} warnings={report['warning_count']}"
        )
        for key, value in sorted(report["metrics"].items()):
            print(f"{key}={value}")
        for item in report["errors"][:20]:
            print("ERROR", json.dumps(item, default=str))
        for item in report["warnings"][:20]:
            print("WARNING", json.dumps(item, default=str))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
