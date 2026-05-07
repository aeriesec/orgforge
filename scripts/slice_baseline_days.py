"""
Slice a fixed day-range out of an existing OrgForge export so it can be used
as a frozen baseline for A/B comparison against a re-run.

Usage:
    python scripts/slice_baseline_days.py \
        --source export/velomind_30d \
        --out    export/velomind_30d_baseline_days_1_10 \
        --start-day 1 --end-day 10

The slicer:
  1. Reads simulation_snapshot.json and writes a snapshot containing only the
     artifacts whose `day` field falls within [start_day, end_day].
  2. Computes the calendar-date range covered by those days from the
     timestamp/date fields in the snapshot itself (not from a config), so it
     works for any company config.
  3. Copies per-app subdirectories (slack, emails, jira, confluence, ...),
     keeping only files whose filename or parent-directory name is a date
     within the day range.

The output directory is a self-contained read-only baseline. The original
30-day export is untouched.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Iterable

DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")


def _walk_for_dates(value, start_day: int, end_day: int, dates: set[str]) -> None:
    """Recursively walk JSON-shaped data. Whenever a dict has a `day` field in
    [start_day, end_day], harvest every YYYY-MM-DD substring in any of its
    string values."""
    if isinstance(value, dict):
        day = value.get("day")
        if isinstance(day, int) and start_day <= day <= end_day:
            for v in value.values():
                if isinstance(v, str):
                    for m in DATE_RE.finditer(v):
                        dates.add(m.group(1))
        for v in value.values():
            _walk_for_dates(v, start_day, end_day, dates)
    elif isinstance(value, list):
        for v in value:
            _walk_for_dates(v, start_day, end_day, dates)


def _collect_dates_for_day_range(
    snapshot: dict, start_day: int, end_day: int
) -> set[str]:
    """Return every calendar date referenced by any artifact in the snapshot
    whose day falls within [start_day, end_day]. Walks the snapshot recursively
    rather than guessing which top-level lists matter, because OrgForge stores
    relevant timestamps in many places (snapshot lists, event_log, artifacts,
    nested comment threads)."""
    dates: set[str] = set()
    _walk_for_dates(snapshot, start_day, end_day, dates)
    return dates


def _filter_snapshot(snapshot: dict, start_day: int, end_day: int) -> dict:
    """Return a new snapshot dict containing only items in the day range. List
    fields of dicts are filtered by the `day` field; dicts and scalars pass
    through unchanged."""
    out = {}
    for key, value in snapshot.items():
        if isinstance(value, list) and value and isinstance(value[0], dict):
            out[key] = [
                item
                for item in value
                if not isinstance(item.get("day"), int)
                or start_day <= item["day"] <= end_day
            ]
        else:
            out[key] = value
    return out


def _path_dates(path: Path) -> Iterable[str]:
    """Yield every YYYY-MM-DD substring found in this path's name or any
    ancestor directory name relative to the export root."""
    for part in (path.name, *(p.name for p in path.parents)):
        m = DATE_RE.search(part)
        if m:
            yield m.group(1)


def _copy_dated_files(
    src_root: Path, dst_root: Path, allowed_dates: set[str]
) -> dict[str, int]:
    """Walk the source export. For files whose path contains a date, copy only
    if the date is in allowed_dates. For files with no date in the path, always
    copy (they are app-level metadata, not per-day output)."""
    counts: dict[str, int] = defaultdict(int)
    for src_file in src_root.rglob("*"):
        if not src_file.is_file():
            continue
        if src_file.name.startswith("."):
            continue
        rel = src_file.relative_to(src_root)
        dates_in_path = list(_path_dates(src_file))
        if dates_in_path and not any(d in allowed_dates for d in dates_in_path):
            continue
        dst_file = dst_root / rel
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst_file)
        counts[rel.parts[0] if len(rel.parts) > 1 else "<root>"] += 1
    return dict(counts)


def slice_baseline(
    source: Path, out: Path, start_day: int, end_day: int
) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Source export not found: {source}")
    if out.exists():
        raise FileExistsError(
            f"Output directory already exists: {out} (refusing to overwrite)"
        )
    snapshot_path = source / "simulation_snapshot.json"
    if not snapshot_path.exists():
        raise FileNotFoundError(
            f"No simulation_snapshot.json under {source}"
        )

    print(f"[slice] reading snapshot: {snapshot_path}")
    with snapshot_path.open() as f:
        snapshot = json.load(f)

    allowed_dates = _collect_dates_for_day_range(snapshot, start_day, end_day)
    print(
        f"[slice] day {start_day}..{end_day} maps to "
        f"{len(allowed_dates)} calendar dates: "
        f"{sorted(allowed_dates)}"
    )

    out.mkdir(parents=True)

    sliced_snapshot = _filter_snapshot(snapshot, start_day, end_day)
    sliced_path = out / "simulation_snapshot.json"
    with sliced_path.open("w") as f:
        json.dump(sliced_snapshot, f, indent=2)
    print(f"[slice] wrote sliced snapshot: {sliced_path}")

    counts = _copy_dated_files(source, out, allowed_dates)
    print("[slice] per-subdir file counts copied:")
    for k in sorted(counts.keys()):
        print(f"  {k:<20} {counts[k]}")

    manifest = {
        "source_export": str(source),
        "day_range": [start_day, end_day],
        "calendar_dates": sorted(allowed_dates),
        "subdir_counts": counts,
    }
    manifest_path = out / "slice_manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[slice] wrote manifest: {manifest_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--start-day", type=int, required=True)
    ap.add_argument("--end-day", type=int, required=True)
    args = ap.parse_args()

    slice_baseline(args.source, args.out, args.start_day, args.end_day)


if __name__ == "__main__":
    main()
