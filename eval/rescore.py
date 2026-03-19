"""
rescore.py
==========
Re-scores an existing per_question.json against the updated scorer.py
without making any LLM or retrieval calls.

Usage
-----
python rescore.py \
    --results  results/<run_id>/per_question.json \
    --questions /path/to/hf_dataset/questions/questions-00000.parquet \
    --scorer    scorer.py

Writes updated per_question.json and summary.json to the same directory,
and appends/updates leaderboard.json + leaderboard.csv.
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from collections import defaultdict
from datetime import datetime, timezone
from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import Dict, List, Optional


# ── Scoring helpers (duplicated from eval_e2e to keep this script standalone) ─


def _mean(vals: List[float]) -> float:
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def aggregate(per_question: List[dict]) -> dict:
    by_type: Dict[str, list] = defaultdict(list)
    by_diff: Dict[str, list] = defaultdict(list)

    for r in per_question:
        by_type[r.get("question_type", "UNKNOWN")].append(r)
        by_diff[r.get("difficulty", "unknown")].append(r)

    def _agg(rows):
        mrr = [r["scores"]["retrieval_mrr"] for r in rows]
        rec = [r["scores"]["retrieval_recall"] for r in rows]
        score = [
            r["scores"]["answer_score"]
            for r in rows
            if r["scores"]["answer_score"] is not None
        ]
        corr = [
            r["scores"]["correct"] for r in rows if r["scores"]["correct"] is not None
        ]
        return {
            "n": len(rows),
            "mrr_at_10": _mean(mrr),
            "recall_at_10": _mean(rec),
            "answer_score": _mean(score) if score else None,
            "accuracy": _mean([float(v) for v in corr]) if corr else None,
        }

    return {
        "overall": _agg(per_question),
        "by_type": {k: _agg(v) for k, v in sorted(by_type.items())},
        "by_difficulty": {k: _agg(v) for k, v in sorted(by_diff.items())},
    }


def _print_summary(summary: dict, run_id: str) -> None:
    overall = summary.get("overall", {})

    def _f(v):
        return f"{v:.4f}" if v is not None else "  n/a "

    print(f"\n{'=' * 64}")
    print(f"  Re-scored run: {run_id}")
    print(f"{'=' * 64}")
    print(
        f"  {'Type':<18} {'MRR@10':>8} {'Recall@10':>10} {'Score':>8} {'Acc':>6} {'N':>4}"
    )
    print(f"  {'-' * 58}")
    print(
        f"  {'OVERALL':<18} {_f(overall.get('mrr_at_10')):>8} "
        f"{_f(overall.get('recall_at_10')):>10} "
        f"{_f(overall.get('answer_score')):>8} "
        f"{_f(overall.get('accuracy')):>6} "
        f"{overall.get('n', 0):>4}"
    )
    print(f"  {'-' * 58}")
    for qtype, m in sorted(summary.get("by_type", {}).items()):
        print(
            f"  {qtype:<18} {_f(m.get('mrr_at_10')):>8} "
            f"{_f(m.get('recall_at_10')):>10} "
            f"{_f(m.get('answer_score')):>8} "
            f"{_f(m.get('accuracy')):>6} "
            f"{m.get('n', 0):>4}"
        )
    print(f"{'=' * 64}\n")


# ── Loader ────────────────────────────────────────────────────────────────────


def load_scorer(scorer_path: str):
    p = Path(scorer_path)
    if not p.exists():
        raise FileNotFoundError(f"scorer.py not found: {p}")
    mod = types.ModuleType("orgforge_scorer")
    mod.__file__ = str(p)
    sys.modules["orgforge_scorer"] = mod
    SourceFileLoader("orgforge_scorer", str(p)).exec_module(mod)
    return mod.OrgForgeScorer()


def load_questions(parquet_path: str) -> Dict[str, dict]:
    """Returns {question_id: question_dict} with ground_truth deserialised."""
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    questions = {}
    for row in df.to_dict("records"):
        for field in ("ground_truth", "evidence_chain"):
            val = row.get(field)
            if isinstance(val, str):
                try:
                    row[field] = json.loads(val)
                except Exception:
                    pass
        questions[row["question_id"]] = row
    return questions


# ── Main ──────────────────────────────────────────────────────────────────────


def rescore(args: argparse.Namespace) -> None:
    results_path = Path(args.results)
    run_dir = results_path.parent

    print(f"Loading results from {results_path}")
    per_question = json.loads(results_path.read_text())

    print(f"Loading questions from {args.questions}")
    questions = load_questions(args.questions)

    print(f"Loading scorer from {args.scorer}")
    scorer = load_scorer(args.scorer)

    # Re-score each question
    changed = 0
    for entry in per_question:
        qid = entry["question_id"]
        q = questions.get(qid)
        if q is None:
            print(f"  WARNING: {qid} not found in questions parquet — skipping")
            continue

        agent_answer = entry["agent_answer"]

        # Inject retrieved IDs for evidence scoring if missing
        if not agent_answer.get("retrieved_artifact_ids"):
            agent_answer["retrieved_artifact_ids"] = entry.get("top_k_ids", [])

        try:
            result = scorer.score(q, agent_answer)
            new_score = round(result.score, 4)
        except Exception as exc:
            print(f"  WARNING: scorer failed on {qid}: {exc}")
            new_score = None

        old_score = entry["scores"].get("answer_score")
        if old_score != new_score:
            changed += 1
            print(f"  {qid} ({entry['question_type']}): {old_score} → {new_score}")

        entry["scores"]["answer_score"] = new_score
        entry["scores"]["correct"] = (
            (new_score >= 0.9) if new_score is not None else None
        )

    print(f"\n{changed} scores changed out of {len(per_question)}")

    # Write updated per_question.json
    results_path.write_text(json.dumps(per_question, indent=2))
    print(f"Updated {results_path}")

    # Write updated summary.json
    summary = aggregate(per_question)
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Updated {summary_path}")

    # Print table
    run_id = run_dir.name
    _print_summary(summary, run_id)

    # Update leaderboard if requested
    if args.leaderboard:
        _update_leaderboard(run_id, run_dir, summary, Path(args.leaderboard))


def _update_leaderboard(
    run_id: str, run_dir: Path, summary: dict, leaderboard_dir: Path
) -> None:
    import csv
    from datetime import datetime, timezone

    lb_json = leaderboard_dir / "leaderboard.json"
    lb_csv = leaderboard_dir / "leaderboard.csv"

    # Parse retriever and generator from run_id.
    # New format: <retriever>__<generator-model>__<timestamp>
    # Old format: <retriever>__bedrock__<timestamp>  (legacy, generator unknown)
    parts = run_id.split("__")
    retriever = parts[0] if len(parts) > 0 else "unknown"
    generator = parts[1].replace("-", "/", 1) if len(parts) > 1 else "unknown"
    tier = "1" if generator == "none" else "1+2"

    overall = summary.get("overall", {})

    # Load existing leaderboard
    leaderboard = []
    if lb_json.exists():
        leaderboard = json.loads(lb_json.read_text())

    new_row = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tier": tier,
        "retriever": retriever,
        "generator": generator,
        "n": overall.get("n"),
        "mrr_at_10": overall.get("mrr_at_10"),
        "recall_at_10": overall.get("recall_at_10"),
        "answer_score": overall.get("answer_score"),
        "accuracy": overall.get("accuracy"),
        "by_type": {
            qtype: {
                "mrr_at_10": m.get("mrr_at_10"),
                "answer_score": m.get("answer_score"),
            }
            for qtype, m in summary.get("by_type", {}).items()
        },
    }

    leaderboard = [r for r in leaderboard if r.get("run_id") != run_id]
    leaderboard.append(new_row)
    leaderboard.sort(
        key=lambda r: (
            0 if r.get("tier") == "1+2" else 1,
            -(r.get("answer_score") or 0.0),
            -(r.get("mrr_at_10") or 0.0),
        )
    )

    lb_json.write_text(json.dumps(leaderboard, indent=2))
    print(f"Updated {lb_json}")

    # CSV — flatten by_type into columns
    _QTYPES = [
        "CAUSAL",
        "ESCALATION",
        "GAP_DETECTION",
        "PLAN",
        "RETRIEVAL",
        "ROUTING",
        "TEMPORAL",
    ]

    def _f(v):
        return "" if v is None else v

    fieldnames = [
        "run_id",
        "timestamp",
        "tier",
        "retriever",
        "generator",
        "n",
        "mrr_at_10",
        "recall_at_10",
        "answer_score",
        "accuracy",
    ]
    for qt in _QTYPES:
        fieldnames += [f"mrr_{qt}", f"score_{qt}"]

    with open(lb_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in leaderboard:
            flat = {
                k: _f(row.get(k))
                for k in [
                    "run_id",
                    "timestamp",
                    "tier",
                    "retriever",
                    "generator",
                    "n",
                    "mrr_at_10",
                    "recall_at_10",
                    "answer_score",
                    "accuracy",
                ]
            }
            by_type = row.get("by_type", {})
            for qt in _QTYPES:
                m = by_type.get(qt, {})
                flat[f"mrr_{qt}"] = _f(m.get("mrr_at_10"))
                flat[f"score_{qt}"] = _f(m.get("answer_score"))
            writer.writerow(flat)

    print(f"Updated {lb_csv}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Re-score existing eval results with updated scorer.py"
    )
    p.add_argument("--results", required=True, help="Path to per_question.json")
    p.add_argument("--questions", required=True, help="Path to questions-00000.parquet")
    p.add_argument("--scorer", required=True, help="Path to scorer.py")
    p.add_argument(
        "--leaderboard",
        default=None,
        metavar="DIR",
        help="Directory containing leaderboard.json/csv to update (optional). "
        "Defaults to the directory where you run this script.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.leaderboard is None:
        args.leaderboard = str(Path.cwd())
    rescore(args)
