"""
A/B comparison harness for the realworld-grounding-poc.

Reads the baseline (pure-synthetic, gpt-4o) and the grounded (gpt-5.5+high)
exports for the same day-range, computes structural metrics over Slack
messages and emails, and emits:

  comparison/<run_tag>/metrics.json        machine-readable metric dump
  comparison/<run_tag>/report.md           human report comparing both runs
  comparison/<run_tag>/paired_threads.md   N side-by-side thread pairs for
                                            human eyeball ratings

Tolerates the grounded export not yet existing — produces a "baseline only"
report so you can verify the metrics pipeline before the live grounded run.

Usage:
    python scripts/compare_baseline_grounded.py \
        --baseline export/velomind_30d_baseline_days_1_10 \
        --grounded export/velomind_grounded_days_1_10 \
        --out      comparison/days_1_10
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
)
logger = logging.getLogger("compare")


# ── Heuristic vocabularies ────────────────────────────────────────────────
HEDGING_MARKERS = {
    "let me check", "i think", "not sure", "i'm not sure", "maybe",
    "probably", "kind of", "sort of", "if i remember", "iirc",
    "i'd guess", "tbh", "ish", "for now", "honestly", "approximately",
    "mostly", "afaik", "i guess",
}
DRIFT_MARKERS = {
    "off topic", "anyway", "by the way", "btw", "unrelated",
    "switching gears", "lol", "haha", "tangent",
}
QUOTE_PREFIX = re.compile(r"^\s*>\s", re.MULTILINE)
MENTION = re.compile(r"@[A-Za-z][A-Za-z0-9._-]*")
CODE_FENCE = re.compile(r"```[\s\S]*?```")
EMOJI_LITE = re.compile(r":[a-z0-9_+-]+:")


# ── Loaders ───────────────────────────────────────────────────────────────
def _load_slack_messages(root: Path) -> list[dict]:
    """Walk root/slack/channels/<channel>/<date>.json and return a flat list of
    {channel, date, ts, user, text} dicts."""
    out: list[dict] = []
    chans = root / "slack" / "channels"
    if not chans.is_dir():
        logger.warning("[compare] no slack/channels under %s", root)
        return out
    for chan_dir in sorted(chans.iterdir()):
        if not chan_dir.is_dir():
            continue
        for jf in sorted(chan_dir.glob("*.json")):
            try:
                payload = json.loads(jf.read_text())
            except Exception as exc:
                logger.debug("[compare] %s parse failed: %s", jf, exc)
                continue
            if isinstance(payload, list):
                msgs = payload
            elif isinstance(payload, dict):
                msgs = payload.get("messages") or payload.get("thread") or []
            else:
                continue
            for m in msgs:
                if not isinstance(m, dict):
                    continue
                out.append({
                    "channel": chan_dir.name,
                    "date": jf.stem,
                    "ts": m.get("ts") or m.get("timestamp") or "",
                    "user": m.get("user") or m.get("from") or m.get("author") or "",
                    "text": m.get("text") or m.get("body") or "",
                })
    return out


def _load_emails(root: Path) -> list[dict]:
    """Walk root/emails/{inbound,internal,outbound}/<date>/*.eml and emit a
    flat list. OrgForge writes emails in MIME .eml format."""
    import email as _email
    from email import policy

    out: list[dict] = []
    edir = root / "emails"
    if not edir.is_dir():
        logger.warning("[compare] no emails/ under %s", root)
        return out
    for kind_dir in sorted(edir.iterdir()):
        if not kind_dir.is_dir():
            continue
        for date_dir in sorted(kind_dir.iterdir()):
            if not date_dir.is_dir():
                continue
            for ef in sorted(date_dir.glob("*.eml")):
                try:
                    msg = _email.message_from_bytes(
                        ef.read_bytes(), policy=policy.default
                    )
                except Exception:
                    continue
                # Walk to text/plain part for the body
                body = ""
                if msg.is_multipart():
                    for part in msg.walk():
                        ctype = part.get_content_type()
                        if ctype == "text/plain":
                            try:
                                body = part.get_content()
                            except Exception:
                                body = part.get_payload(decode=True) or ""
                                if isinstance(body, bytes):
                                    body = body.decode("utf-8", errors="replace")
                            break
                else:
                    try:
                        body = msg.get_content()
                    except Exception:
                        body = msg.get_payload(decode=True) or ""
                        if isinstance(body, bytes):
                            body = body.decode("utf-8", errors="replace")
                out.append({
                    "kind": kind_dir.name,
                    "date": date_dir.name,
                    "subject": str(msg.get("subject") or ""),
                    "from": str(msg.get("from") or ""),
                    "to": str(msg.get("to") or ""),
                    "body": body or "",
                })
    return out


# ── Metric primitives ─────────────────────────────────────────────────────
def _percentile(xs: list[int], p: float) -> int:
    if not xs:
        return 0
    xs = sorted(xs)
    return xs[int(round((len(xs) - 1) * p))]


def _length_stats(texts: list[str]) -> dict:
    chars = [len(t) for t in texts]
    words = [len((t or "").split()) for t in texts]
    return {
        "n": len(texts),
        "char_mean": int(statistics.mean(chars)) if chars else 0,
        "char_median": int(statistics.median(chars)) if chars else 0,
        "char_p95": _percentile(chars, 0.95),
        "char_max": max(chars) if chars else 0,
        "word_mean": int(statistics.mean(words)) if words else 0,
        "word_median": int(statistics.median(words)) if words else 0,
        "word_p95": _percentile(words, 0.95),
    }


def _heuristic_marker_rate(texts: list[str], markers: Iterable[str]) -> float:
    """Return percent of messages containing at least one marker (case-insensitive)."""
    if not texts:
        return 0.0
    hits = 0
    markers_lower = {m.lower() for m in markers}
    for t in texts:
        lc = (t or "").lower()
        if any(m in lc for m in markers_lower):
            hits += 1
    return round(100.0 * hits / len(texts), 2)


def _per_message_markups(texts: list[str]) -> dict:
    if not texts:
        return {"mention_rate_pct": 0, "quote_rate_pct": 0,
                "code_block_rate_pct": 0, "emoji_rate_pct": 0}
    n = len(texts)
    return {
        "mention_rate_pct": round(
            100 * sum(1 for t in texts if MENTION.search(t or "")) / n, 2
        ),
        "quote_rate_pct": round(
            100 * sum(1 for t in texts if QUOTE_PREFIX.search(t or "")) / n, 2
        ),
        "code_block_rate_pct": round(
            100 * sum(1 for t in texts if CODE_FENCE.search(t or "")) / n, 2
        ),
        "emoji_rate_pct": round(
            100 * sum(1 for t in texts if EMOJI_LITE.search(t or "")) / n, 2
        ),
    }


# ── Comparison ────────────────────────────────────────────────────────────
def metrics_for_run(label: str, root: Path) -> dict:
    slack = _load_slack_messages(root)
    emails = _load_emails(root)
    slack_texts = [m["text"] for m in slack]
    email_bodies = [e["body"] for e in emails]

    return {
        "label": label,
        "root": str(root),
        "exists": root.exists(),
        "slack": {
            "count": len(slack),
            "channel_count": len({m["channel"] for m in slack}),
            "date_range": sorted({m["date"] for m in slack})[:1] + sorted({m["date"] for m in slack})[-1:],
            "length": _length_stats(slack_texts),
            "vagueness_pct": _heuristic_marker_rate(slack_texts, HEDGING_MARKERS),
            "drift_pct": _heuristic_marker_rate(slack_texts, DRIFT_MARKERS),
            "markup": _per_message_markups(slack_texts),
        },
        "email": {
            "count": len(emails),
            "kinds": Counter(e["kind"] for e in emails),
            "length": _length_stats(email_bodies),
            "vagueness_pct": _heuristic_marker_rate(email_bodies, HEDGING_MARKERS),
            "drift_pct": _heuristic_marker_rate(email_bodies, DRIFT_MARKERS),
            "markup": _per_message_markups(email_bodies),
        },
    }


def _diff_pct(a: float | int, b: float | int) -> str:
    """Render b vs a as a delta string."""
    if a == 0 and b == 0:
        return "no change"
    if a == 0:
        return f"new ({b})"
    delta = (b - a) / a * 100.0
    arrow = "↑" if b > a else "↓" if b < a else "="
    return f"{arrow} {abs(delta):.1f}% (a={a}, b={b})"


def render_report(metrics_a: dict, metrics_b: Optional[dict], out_md: Path) -> None:
    lines: list[str] = []
    lines.append(f"# Comparison report: {metrics_a['label']} vs {metrics_b['label'] if metrics_b else '(grounded run not yet present)'}")
    lines.append("")

    def _section(name: str, key_app: str) -> None:
        a = metrics_a[key_app]
        lines.append(f"## {name}")
        lines.append("")
        lines.append(f"| metric | {metrics_a['label']} | "
                     + (f"{metrics_b['label']} | delta |" if metrics_b else " |"))
        lines.append("|---|---|" + ("---|---|" if metrics_b else "---|"))
        rows = [
            ("count", a["count"], (metrics_b[key_app]["count"] if metrics_b else None)),
            ("char_mean", a["length"]["char_mean"], (metrics_b[key_app]["length"]["char_mean"] if metrics_b else None)),
            ("char_p95",  a["length"]["char_p95"],  (metrics_b[key_app]["length"]["char_p95"]  if metrics_b else None)),
            ("word_mean", a["length"]["word_mean"], (metrics_b[key_app]["length"]["word_mean"] if metrics_b else None)),
            ("word_p95",  a["length"]["word_p95"],  (metrics_b[key_app]["length"]["word_p95"]  if metrics_b else None)),
            ("vagueness_pct", a["vagueness_pct"], (metrics_b[key_app]["vagueness_pct"] if metrics_b else None)),
            ("drift_pct",     a["drift_pct"],     (metrics_b[key_app]["drift_pct"]     if metrics_b else None)),
            ("mention_rate_pct", a["markup"]["mention_rate_pct"], (metrics_b[key_app]["markup"]["mention_rate_pct"] if metrics_b else None)),
            ("quote_rate_pct",   a["markup"]["quote_rate_pct"],   (metrics_b[key_app]["markup"]["quote_rate_pct"]   if metrics_b else None)),
            ("code_block_rate_pct", a["markup"]["code_block_rate_pct"], (metrics_b[key_app]["markup"]["code_block_rate_pct"] if metrics_b else None)),
            ("emoji_rate_pct",   a["markup"]["emoji_rate_pct"],   (metrics_b[key_app]["markup"]["emoji_rate_pct"]   if metrics_b else None)),
        ]
        for name_, av, bv in rows:
            if metrics_b:
                lines.append(f"| {name_} | {av} | {bv} | {_diff_pct(av, bv)} |")
            else:
                lines.append(f"| {name_} | {av} | |")
        lines.append("")

    _section("Slack", "slack")
    _section("Email", "email")

    lines.append("## Interpretation hints")
    lines.append("")
    lines.append(
        "- `vagueness_pct` rising in the grounded run is a positive signal — "
        "real workplace chat carries hedges (\"i think\", \"let me check\") that "
        "pure-synthetic LLM output tends to smooth away.")
    lines.append(
        "- `drift_pct` rising is a positive signal for chat realism (off-topic "
        "chatter is real). For email it should stay low.")
    lines.append(
        "- `char_p95` shrinking and `mention_rate_pct` rising in the grounded "
        "Slack run = closer to real channel patterns (shorter messages, more @-tags).")
    lines.append("")
    out_md.write_text("\n".join(lines))


# ── Paired threads for human eyeball ──────────────────────────────────────
def _pair_threads(baseline_root: Path, grounded_root: Optional[Path], n: int) -> str:
    """Pick n channels and dump the first day's worth side-by-side."""
    a_chans = sorted((baseline_root / "slack" / "channels").glob("*"))[:n]
    out_lines = ["# Paired threads — blind human eyeball comparison", ""]
    for ch in a_chans:
        if not ch.is_dir():
            continue
        a_files = sorted(ch.glob("*.json"))
        if not a_files:
            continue
        afile = a_files[0]
        try:
            apayload = json.loads(afile.read_text())
        except Exception:
            continue
        amsgs = apayload if isinstance(apayload, list) else (apayload.get("messages") or [])
        out_lines.append(f"## Channel `{ch.name}`  date={afile.stem}")
        out_lines.append("")
        out_lines.append(f"### A — {baseline_root.name}")
        out_lines.append("```")
        for m in amsgs[:8]:
            if isinstance(m, dict):
                out_lines.append(f"[{m.get('user','')}] {m.get('text','')}")
        out_lines.append("```")
        if grounded_root:
            bfile = grounded_root / "slack" / "channels" / ch.name / afile.name
            if bfile.exists():
                try:
                    bpayload = json.loads(bfile.read_text())
                except Exception:
                    bpayload = []
                bmsgs = bpayload if isinstance(bpayload, list) else (bpayload.get("messages") or [])
                out_lines.append(f"### B — {grounded_root.name}")
                out_lines.append("```")
                for m in bmsgs[:8]:
                    if isinstance(m, dict):
                        out_lines.append(f"[{m.get('user','')}] {m.get('text','')}")
                out_lines.append("```")
            else:
                out_lines.append(f"### B — {grounded_root.name}: (file missing — grounded run not produced for this channel)")
        out_lines.append("")
    return "\n".join(out_lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--grounded", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-paired-threads", type=int, default=20)
    args = ap.parse_args()

    if not args.baseline.exists():
        logger.error("[compare] baseline not found: %s", args.baseline)
        return 2

    args.out.mkdir(parents=True, exist_ok=True)

    metrics_a = metrics_for_run("baseline", args.baseline)
    metrics_b = None
    if args.grounded and args.grounded.exists():
        metrics_b = metrics_for_run("grounded", args.grounded)
    else:
        logger.warning(
            "[compare] grounded export not present at %s — "
            "writing baseline-only report. Re-run after grounded simulation completes.",
            args.grounded,
        )

    metrics_path = args.out / "metrics.json"
    metrics_path.write_text(json.dumps(
        {"baseline": metrics_a, "grounded": metrics_b},
        indent=2, default=str,
    ))
    logger.info("[compare] wrote %s", metrics_path)

    report_md = args.out / "report.md"
    render_report(metrics_a, metrics_b, report_md)
    logger.info("[compare] wrote %s", report_md)

    paired = args.out / "paired_threads.md"
    paired.write_text(_pair_threads(args.baseline, args.grounded, args.n_paired_threads))
    logger.info("[compare] wrote %s", paired)

    print()
    print(f"=== compare summary ({metrics_a['label']}"
          + (f" vs {metrics_b['label']}" if metrics_b else "") + ") ===")
    print(f"  Slack messages:  baseline={metrics_a['slack']['count']}"
          + (f"  grounded={metrics_b['slack']['count']}" if metrics_b else ""))
    print(f"  Email messages:  baseline={metrics_a['email']['count']}"
          + (f"  grounded={metrics_b['email']['count']}" if metrics_b else ""))
    print(f"  Slack vagueness: baseline={metrics_a['slack']['vagueness_pct']}%"
          + (f"  grounded={metrics_b['slack']['vagueness_pct']}%" if metrics_b else ""))
    print(f"  Slack drift:     baseline={metrics_a['slack']['drift_pct']}%"
          + (f"  grounded={metrics_b['slack']['drift_pct']}%" if metrics_b else ""))
    print(f"  Output dir:      {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
