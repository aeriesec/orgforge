#!/usr/bin/env python3
"""
rescore.py
==========
Rescore existing verdict files against a new ground truth without re-running models.

Walks results/insider_threat/ (or --results-dir) for every run that has both
verdicts.json and scores.json, recomputes score_verdicts() using the new ground
truth file, and writes:
  - <run_dir>/scores_v2.json       updated scores for that run
  - rescore_summary.json           side-by-side diff of old vs new for every run
  - rescore_summary.csv            same, CSV-friendly

Usage
-----
    python rescore.py \\
        --ground-truth  _ground_truth_v2.jsonl \\
        --results-dir   results/insider_threat \\
        --access-log    export/access_log.jsonl

    # Dry run — print diffs but don't write scores_v2.json
    python rescore.py --ground-truth _ground_truth_v2.jsonl --dry-run

    # Also rescore sensitivity runs
    python rescore.py --ground-truth _ground_truth_v2.jsonl \\
        --results-dir results/sensitivity
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set


# ─────────────────────────────────────────────────────────────────────────────
# Copied verbatim from eval_insider_threat.py so this script is self-contained
# ─────────────────────────────────────────────────────────────────────────────

_ALL_BEHAVIORS = [
    "secret_in_commit",
    "unusual_hours_access",
    "excessive_repo_cloning",
    "sentiment_drift",
    "cross_dept_snooping",
    "data_exfil_email",
    "host_data_hoarding",
    "social_engineering",
    "idp_anomaly",
]

_ALL_CLASSES = ["negligent", "disgruntled", "malicious"]


def load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def load_ground_truth(path: Path) -> Dict[str, Set[str]]:
    gt: Dict[str, Set[str]] = {}
    for rec in load_jsonl(path):
        if rec.get("true_positive"):
            name = rec.get("actor", "")
            behavior = rec.get("behavior", "")
            if name and behavior:
                gt.setdefault(name, set()).add(behavior)
    return gt


def load_ground_truth_full(path: Path) -> Dict[str, dict]:
    gt: Dict[str, dict] = {}
    for rec in load_jsonl(path):
        if rec.get("true_positive"):
            name = rec.get("actor", "")
            if not name:
                continue
            if name not in gt:
                gt[name] = {
                    "behaviors": set(),
                    "threat_class": rec.get("threat_class", "unknown"),
                }
            b = rec.get("behavior", "")
            if b:
                gt[name]["behaviors"].add(b)
    return gt


def score_verdicts(
    verdicts: List[dict],
    ground_truth: Dict[str, Set[str]],
    ground_truth_full: Dict[str, dict],
    access_log: List[dict],
) -> dict:
    threat_actors = set(ground_truth.keys())
    verdict_map = {v.get("name", ""): v for v in verdicts}

    tp = fp = fn = 0
    by_class: Dict[str, Dict] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    by_behavior: Dict[str, Dict] = defaultdict(lambda: {"tp": 0, "fp": 0})

    for name, verdict in verdict_map.items():
        is_flagged = verdict.get("verdict") in ("suspicious", "likely_threat")
        is_threat = name in threat_actors
        actual_behaviors = ground_truth.get(name, set())
        tc = ground_truth_full.get(name, {}).get("threat_class", "unknown")

        if is_flagged and is_threat:
            tp += 1
            by_class[tc]["tp"] += 1
            for b in verdict.get("behaviors_observed", []):
                if b in actual_behaviors:
                    by_behavior[b]["tp"] += 1
                else:
                    by_behavior[b]["fp"] += 1
        elif is_flagged:
            fp += 1
            by_class["innocent"]["fp"] += 1

    for actor in threat_actors:
        v = verdict_map.get(actor)
        if v is None or v.get("verdict") not in ("suspicious", "likely_threat"):
            fn += 1
            tc = ground_truth_full.get(actor, {}).get("threat_class", "unknown")
            by_class[tc]["fn"] += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    # Vishing detection
    vishing_detected = False
    phone_call_records = [r for r in access_log if r.get("record_type") == "phone_call"]
    if phone_call_records:
        for v in verdicts:
            if "social_engineering" in v.get("behaviors_observed", []):
                for e in v.get("evidence", []):
                    desc = e.get("description", "").lower()
                    if any(
                        kw in desc
                        for kw in ["phone", "call", "vish", "auth", "preceded"]
                    ):
                        vishing_detected = True
                        break

    # Host trail reconstruction
    host_trail_reconstructed = False
    hoarding_records = [r for r in access_log if r.get("record_type") == "host_event"]
    if hoarding_records:
        has_phase1 = any(r.get("action") == "bulk_file_copy" for r in hoarding_records)
        has_phase2 = any(
            r.get("action") == "archive_creation" for r in hoarding_records
        )
        has_phase3 = any(r.get("action") == "archive_move" for r in hoarding_records)
        if has_phase1 and has_phase2 and has_phase3:
            for v in verdicts:
                if "host_data_hoarding" in v.get("behaviors_observed", []):
                    evidence_descs = " ".join(
                        e.get("description", "").lower() for e in v.get("evidence", [])
                    )
                    phase_hits = sum(
                        [
                            any(
                                kw in evidence_descs
                                for kw in ["copy", "bulk", "staged"]
                            ),
                            any(
                                kw in evidence_descs
                                for kw in ["compress", "archive", "zip", "7z"]
                            ),
                            any(
                                kw in evidence_descs
                                for kw in [
                                    "move",
                                    "dropbox",
                                    "drive",
                                    "onedrive",
                                    "usb",
                                    "sync",
                                ]
                            ),
                        ]
                    )
                    if phase_hits >= 3:
                        host_trail_reconstructed = True
                        break

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "by_class": {k: dict(v) for k, v in by_class.items()},
        "by_behavior": {k: dict(v) for k, v in by_behavior.items()},
        "vishing_detected": vishing_detected,
        "host_trail_reconstructed": host_trail_reconstructed,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Diff helpers
# ─────────────────────────────────────────────────────────────────────────────


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "✓" if v else "✗"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _diff_scores(old: dict, new: dict) -> List[str]:
    """Return human-readable lines describing what changed."""
    lines = []
    scalar_keys = [
        "precision",
        "recall",
        "f1",
        "tp",
        "fp",
        "fn",
        "vishing_detected",
        "host_trail_reconstructed",
    ]
    for k in scalar_keys:
        o, n = old.get(k), new.get(k)
        if o != n:
            lines.append(f"  {k}: {_fmt(o)} → {_fmt(n)}")

    # by_behavior diffs
    old_bb = old.get("by_behavior", {})
    new_bb = new.get("by_behavior", {})
    all_b = sorted(set(old_bb) | set(new_bb))
    for b in all_b:
        o_tp = old_bb.get(b, {}).get("tp", 0)
        o_fp = old_bb.get(b, {}).get("fp", 0)
        n_tp = new_bb.get(b, {}).get("tp", 0)
        n_fp = new_bb.get(b, {}).get("fp", 0)
        if o_tp != n_tp or o_fp != n_fp:
            lines.append(f"  by_behavior[{b}]: TP {o_tp}→{n_tp}  FP {o_fp}→{n_fp}")
    return lines


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def rescore(
    gt_path: Path,
    results_dir: Path,
    access_log_path: Path,
    dry_run: bool,
) -> None:
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth not found: {gt_path}")
    if not results_dir.exists():
        raise FileNotFoundError(f"Results dir not found: {results_dir}")

    ground_truth = load_ground_truth(gt_path)
    ground_truth_full = load_ground_truth_full(gt_path)
    access_log = load_jsonl(access_log_path) if access_log_path.exists() else []

    if not access_log:
        print(
            f"WARNING: access_log not found at {access_log_path} — "
            f"vishing and host_trail scoring will be approximate."
        )

    print(f"\nGround truth  : {gt_path}  ({len(ground_truth)} threat actors)")
    print(f"Results dir   : {results_dir}")
    print(f"Access log    : {access_log_path} ({len(access_log)} records)")
    print(f"Dry run       : {dry_run}\n")

    summary_rows = []
    run_dirs = sorted(p for p in results_dir.iterdir() if p.is_dir())

    for run_dir in run_dirs:
        verdicts_path = run_dir / "verdicts.json"
        scores_path = run_dir / "scores.json"

        if not verdicts_path.exists() or not scores_path.exists():
            continue

        verdicts = json.loads(verdicts_path.read_text())
        old_scores_full = json.loads(scores_path.read_text())
        old_verdict_scores = old_scores_full.get("verdicts") or {}

        if not verdicts:
            print(f"  {run_dir.name}: no verdicts — skipping")
            continue

        new_verdict_scores = score_verdicts(
            verdicts, ground_truth, ground_truth_full, access_log
        )

        diffs = _diff_scores(old_verdict_scores, new_verdict_scores)
        changed = bool(diffs)

        model = old_scores_full.get("model", run_dir.name)
        variant = old_scores_full.get("prompt_variant", "unknown")

        print(f"{'CHANGED' if changed else 'unchanged':>9}  {run_dir.name}")
        for line in diffs:
            print(line)

        summary_rows.append(
            {
                "run_dir": run_dir.name,
                "model": model,
                "variant": variant,
                "changed": changed,
                "old_f1": old_verdict_scores.get("f1", ""),
                "new_f1": new_verdict_scores["f1"],
                "old_precision": old_verdict_scores.get("precision", ""),
                "new_precision": new_verdict_scores["precision"],
                "old_recall": old_verdict_scores.get("recall", ""),
                "new_recall": new_verdict_scores["recall"],
                "old_idp_anomaly_tp": old_verdict_scores.get("by_behavior", {})
                .get("idp_anomaly", {})
                .get("tp", 0),
                "new_idp_anomaly_tp": new_verdict_scores["by_behavior"]
                .get("idp_anomaly", {})
                .get("tp", 0),
                "old_idp_anomaly_fp": old_verdict_scores.get("by_behavior", {})
                .get("idp_anomaly", {})
                .get("fp", 0),
                "new_idp_anomaly_fp": new_verdict_scores["by_behavior"]
                .get("idp_anomaly", {})
                .get("fp", 0),
                "diffs": "; ".join(diffs) if diffs else "",
            }
        )

        if not dry_run:
            new_scores_full = dict(old_scores_full)
            new_scores_full["verdicts"] = new_verdict_scores
            new_scores_full["ground_truth_version"] = "v2"
            new_scores_full["rescore_note"] = (
                "idp_anomaly split from unusual_hours_access for IDP records "
                "with anomalous_ip=True or new_device=True"
            )
            (run_dir / "scores_v2.json").write_text(
                json.dumps(new_scores_full, indent=2, default=str)
            )

    # Write summary
    if not dry_run and summary_rows:
        summary_path = results_dir.parent / "rescore_summary.json"
        summary_path.write_text(json.dumps(summary_rows, indent=2, default=str))
        print(f"\nSummary JSON → {summary_path}")

        csv_path = results_dir.parent / "rescore_summary.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Summary CSV  → {csv_path}")

    changed_count = sum(1 for r in summary_rows if r["changed"])
    print(f"\n{len(summary_rows)} runs processed, {changed_count} changed.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--ground-truth",
        required=True,
        type=Path,
        help="Path to updated ground truth JSONL (e.g. _ground_truth_v2.jsonl)",
    )
    p.add_argument(
        "--results-dir",
        default="results/insider_threat",
        type=Path,
        help="Directory containing run subdirectories (default: results/insider_threat)",
    )
    p.add_argument(
        "--access-log",
        default="export/access_log.jsonl",
        type=Path,
        help="Path to access_log.jsonl for vishing/host-trail scoring "
        "(default: export/access_log.jsonl)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print diffs but do not write scores_v2.json files",
    )
    args = p.parse_args()
    rescore(args.ground_truth, args.results_dir, args.access_log, args.dry_run)


if __name__ == "__main__":
    main()
