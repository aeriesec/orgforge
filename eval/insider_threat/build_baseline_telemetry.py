"""
build_baseline_telemetry.py
============================
Bridges the gap between OrgForge's normal simulation artifacts and the
security_telemetry format expected by the BASELINE_AGENT.

Reads Slack, PR, and email artifacts from the export directory for days
BEFORE any subject's onset_day, and emits them as clean telemetry records
into security_telemetry/baseline_telemetry.jsonl.

All records produced here are TRUE NEGATIVES by construction — no insider
threat behaviors were active during this window. The BASELINE_AGENT uses
them to establish a false-positive detection threshold.

Usage:
    python build_baseline_telemetry.py [--export-dir ./export] [--config config/config.yaml]

Output:
    export/security_telemetry/baseline_telemetry.jsonl

Design notes:
  - Skips bot messages (is_bot: true) — bots are not behavioral signals
  - Skips _security_injected messages — these are threat artifacts, not baseline
  - Skips threat subjects entirely for the pre-onset window (belt-and-suspenders;
    subjects should have no telemetry before onset_day anyway)
  - Uses the sim's start_date + day offsets to compute day numbers
  - PR records use created_at date; email records use directory date
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date, datetime
from email import message_from_file
from pathlib import Path

import yaml


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG LOADER
# ─────────────────────────────────────────────────────────────────────────────


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_onset_days(cfg: dict) -> dict[str, int]:
    """Return {subject_name: onset_day} for all configured subjects."""
    it = cfg.get("insider_threat", {})
    return {s["name"]: s.get("onset_day", 1) for s in it.get("subjects", [])}


def get_subject_names(cfg: dict) -> set[str]:
    it = cfg.get("insider_threat", {})
    return {s["name"] for s in it.get("subjects", [])}


def get_start_date(cfg: dict) -> date:
    raw = cfg["simulation"].get("start_date", "2026-03-02")
    return datetime.strptime(str(raw), "%Y-%m-%d").date()


def date_to_day(d: date, start_date: date) -> int:
    """Convert a calendar date to a simulation day number (day 1 = start_date)."""
    return (d - start_date).days + 1


def min_onset_day(cfg: dict) -> int:
    """The earliest onset_day across all subjects — baseline ends before this."""
    onset = get_onset_days(cfg)
    return min(onset.values()) if onset else 999


# ─────────────────────────────────────────────────────────────────────────────
# SLACK READER
# ─────────────────────────────────────────────────────────────────────────────


def read_slack_records(
    export_dir: Path,
    start_date: date,
    cutoff_day: int,
    subject_names: set[str],
) -> list[dict]:
    """
    Walk slack/channels/*/YYYY-MM-DD.json and emit one record per
    non-bot, non-injected human message that falls before cutoff_day.
    """
    records = []
    channels_dir = export_dir / "slack" / "channels"
    if not channels_dir.exists():
        print(f"  [slack] No channels dir found at {channels_dir}", file=sys.stderr)
        return records

    for channel_dir in sorted(channels_dir.iterdir()):
        if not channel_dir.is_dir():
            continue
        channel = channel_dir.name

        for json_file in sorted(channel_dir.glob("*.json")):
            # Parse date from filename
            try:
                file_date = datetime.strptime(json_file.stem, "%Y-%m-%d").date()
            except ValueError:
                continue

            day = date_to_day(file_date, start_date)
            if day >= cutoff_day:
                continue  # Only baseline days

            try:
                messages = json.loads(json_file.read_text())
            except Exception as e:
                print(f"  [slack] Failed to parse {json_file}: {e}", file=sys.stderr)
                continue

            for msg in messages:
                # Skip bots
                if msg.get("is_bot"):
                    continue
                # Skip security-injected messages (should not exist pre-onset, but defensive)
                if msg.get("_security_injected"):
                    continue

                user = msg.get("user", "")
                if not user:
                    continue

                # Skip threat subjects (belt-and-suspenders)
                if user in subject_names:
                    continue

                ts_raw = msg.get("ts", "")
                if not ts_raw:
                    continue

                # Parse the actual timestamp for access_hour
                try:
                    ts_dt = datetime.fromisoformat(ts_raw)
                    access_hour = ts_dt.hour
                    outside_business_hours = access_hour < 9 or access_hour >= 18
                    msg_date = ts_dt.date()
                    # Use the message's actual date for day calc (may differ from file date)
                    msg_day = date_to_day(msg_date, start_date)
                    if msg_day >= cutoff_day:
                        continue
                except Exception:
                    ts_dt = None
                    access_hour = None
                    outside_business_hours = False
                    msg_day = day

                record = {
                    "record_type": "slack_message",
                    "day": msg_day,
                    "date": str(file_date),
                    "timestamp": ts_raw,
                    "actor": user,
                    "channel": channel,
                    "sentiment": "neutral",  # baseline — no drift
                }
                if access_hour is not None:
                    record["access_hour"] = access_hour
                    record["outside_business_hours"] = outside_business_hours

                records.append(record)

    return records


# ─────────────────────────────────────────────────────────────────────────────
# PR READER
# ─────────────────────────────────────────────────────────────────────────────


def read_pr_records(
    export_dir: Path,
    start_date: date,
    cutoff_day: int,
    subject_names: set[str],
) -> list[dict]:
    """
    Walk git/prs/PR-*.json and emit one repo_access record per PR
    that was created before cutoff_day.
    """
    records = []
    prs_dir = export_dir / "git" / "prs"
    if not prs_dir.exists():
        print(f"  [prs] No prs dir found at {prs_dir}", file=sys.stderr)
        return records

    for pr_file in sorted(prs_dir.glob("PR-*.json")):
        try:
            pr = json.loads(pr_file.read_text())
        except Exception as e:
            print(f"  [prs] Failed to parse {pr_file}: {e}", file=sys.stderr)
            continue

        created_at = pr.get("created_at", "")
        if not created_at:
            continue

        try:
            pr_dt = datetime.fromisoformat(created_at)
            pr_date = pr_dt.date()
            pr_day = date_to_day(pr_date, start_date)
        except Exception:
            continue

        if pr_day >= cutoff_day:
            continue

        author = pr.get("author", pr.get("created_by", ""))
        if not author or author in subject_names:
            continue

        access_hour = pr_dt.hour
        record = {
            "record_type": "repo_access",
            "day": pr_day,
            "date": str(pr_date),
            "timestamp": created_at,
            "actor": author,
            "pr_id": pr.get("pr_id", pr_file.stem),
            "ticket_id": pr.get("ticket_id", ""),
            "access_hour": access_hour,
            "outside_business_hours": access_hour < 9 or access_hour >= 18,
            "repos_cloned": 1,
            "anomalous": False,
        }
        records.append(record)

    return records


# ─────────────────────────────────────────────────────────────────────────────
# EMAIL READER
# ─────────────────────────────────────────────────────────────────────────────


def _parse_sender_name(from_header: str) -> str:
    """Extract the display name from a From: header, lower-cased."""
    # "Jax <jax@apexathletics.io>" → "Jax"
    m = re.match(r"^([^<]+)<", from_header)
    if m:
        return m.group(1).strip()
    # "jax@apexathletics.io" → "jax"
    m2 = re.match(r"([^@]+)@", from_header)
    if m2:
        return m2.group(1).strip()
    return from_header.strip()


def read_email_records(
    export_dir: Path,
    start_date: date,
    cutoff_day: int,
    subject_names: set[str],
) -> list[dict]:
    """
    Walk emails/outbound/YYYY-MM-DD/*.eml and emit one email_send record
    per email sent before cutoff_day by a non-subject.

    Skips _fwd_ files — those are injected exfil artifacts.
    """
    records = []
    outbound_dir = export_dir / "emails" / "outbound"
    if not outbound_dir.exists():
        print(f"  [email] No outbound dir at {outbound_dir}", file=sys.stderr)
        return records

    for date_dir in sorted(outbound_dir.iterdir()):
        if not date_dir.is_dir():
            continue
        try:
            dir_date = datetime.strptime(date_dir.name, "%Y-%m-%d").date()
        except ValueError:
            continue

        day = date_to_day(dir_date, start_date)
        if day >= cutoff_day:
            continue

        for eml_file in sorted(date_dir.glob("*.eml")):
            # Skip injected exfil emails
            if "_fwd_" in eml_file.name:
                continue

            try:
                with open(eml_file) as f:
                    msg = message_from_file(f)
            except Exception as e:
                print(f"  [email] Failed to parse {eml_file}: {e}", file=sys.stderr)
                continue

            from_hdr = msg.get("From", "")
            sender_name = _parse_sender_name(from_hdr)

            if not sender_name or sender_name in subject_names:
                continue

            to_hdr = msg.get("To", "")
            subject_hdr = msg.get("Subject", "")
            # Determine if external (to address is outside company domain)
            is_external = "@apexathletics.io" not in to_hdr.lower()

            record = {
                "record_type": "email_send",
                "day": day,
                "date": str(dir_date),
                "timestamp": f"{dir_date.isoformat()}T09:00:00",  # no precise ts in eml path
                "actor": sender_name,
                "to": to_hdr,
                "subject": subject_hdr,
                "is_external": is_external,
                "off_hours": False,  # normal outbound — not flagged
                "eml_path": str(eml_file),
            }
            records.append(record)

    return records


# ─────────────────────────────────────────────────────────────────────────────
# WRITER
# ─────────────────────────────────────────────────────────────────────────────


def write_baseline(records: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records_sorted = sorted(records, key=lambda r: (r["day"], r["timestamp"]))
    with open(output_path, "w") as f:
        for rec in records_sorted:
            f.write(json.dumps(rec) + "\n")
    print(f"  Wrote {len(records_sorted)} baseline records → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────


def print_summary(records: list[dict], cutoff_day: int, onset_days: dict) -> None:
    by_type: dict[str, int] = {}
    by_day: dict[int, int] = {}
    actors: set[str] = set()

    for r in records:
        by_type[r["record_type"]] = by_type.get(r["record_type"], 0) + 1
        by_day[r["day"]] = by_day.get(r["day"], 0) + 1
        actors.add(r["actor"])

    print(f"\n{'─' * 60}")
    print("  Baseline telemetry summary")
    print(f"{'─' * 60}")
    print(f"  Cutoff: before day {cutoff_day}  (onset days: {onset_days})")
    print(f"  Total records : {len(records)}")
    print(f"  Unique actors : {len(actors)}")
    print(f"  By type       : {by_type}")
    print(f"  By day        : {dict(sorted(by_day.items()))}")
    print(f"  Actors        : {sorted(actors)}")
    print(f"{'─' * 60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Build baseline telemetry from OrgForge artifacts"
    )
    parser.add_argument(
        "--export-dir", default="./export", help="Path to OrgForge export dir"
    )
    parser.add_argument(
        "--config", default="config/config.yaml", help="Path to config.yaml"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path (default: <export-dir>/security_telemetry/baseline_telemetry.jsonl)",
    )
    args = parser.parse_args()

    export_dir = Path(args.export_dir).resolve()
    config_path = Path(args.config).resolve()

    if not config_path.exists():
        print(f"ERROR: config not found at {config_path}", file=sys.stderr)
        sys.exit(1)

    cfg = load_config(config_path)
    start_date = get_start_date(cfg)
    onset_days = get_onset_days(cfg)
    subject_names = get_subject_names(cfg)
    cutoff_day = min_onset_day(cfg)  # baseline = days strictly before earliest onset

    output_path = (
        Path(args.output)
        if args.output
        else export_dir / "security_telemetry" / "baseline_telemetry.jsonl"
    )

    print("\nOrgForge Baseline Telemetry Builder")
    print(f"  Export dir   : {export_dir}")
    print(f"  Start date   : {start_date}")
    print(f"  Subjects     : {onset_days}")
    print(f"  Cutoff day   : {cutoff_day} (baseline = days 1–{cutoff_day - 1})")
    print(f"  Output       : {output_path}\n")

    all_records = []

    print("Reading Slack messages...")
    slack = read_slack_records(export_dir, start_date, cutoff_day, subject_names)
    print(f"  {len(slack)} slack_message records")
    all_records.extend(slack)

    print("Reading PR records...")
    prs = read_pr_records(export_dir, start_date, cutoff_day, subject_names)
    print(f"  {len(prs)} repo_access (PR) records")
    all_records.extend(prs)

    print("Reading email records...")
    emails = read_email_records(export_dir, start_date, cutoff_day, subject_names)
    print(f"  {len(emails)} email_send records")
    all_records.extend(emails)

    if not all_records:
        print(
            "\nWARNING: No baseline records found. Check --export-dir path and "
            "that artifacts exist before the cutoff day.",
            file=sys.stderr,
        )
        sys.exit(1)

    print_summary(all_records, cutoff_day, onset_days)
    write_baseline(all_records, output_path)
    print("Done.")


if __name__ == "__main__":
    main()
