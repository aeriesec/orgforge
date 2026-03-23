"""
export_insider_threat_to_hf.py
==============================
Packages the OrgForge insider threat simulation output into a
HuggingFace-ready dataset and writes a dataset card.

Run after flow.py (with insider_threat.enabled: true) and
build_baseline_telemetry.py:

    python export_insider_threat_to_hf.py

Output layout
-------------
export/hf_insider_threat/
  telemetry/
    access_log-00000.parquet       — observable telemetry stream (no labels)
    baseline_telemetry-00000.parquet — clean pre-onset baseline records
    idp_auth-00000.parquet         — IDP authentication events only
    host_events-00000.parquet      — host-level staging events only
  ground_truth/
    ground_truth-00000.parquet     — full records WITH labels (held out)
  leaderboard/
    insider_threat_leaderboard.csv — frozen leaderboard snapshot
    insider_threat_leaderboard.json
  README.md                        — HuggingFace dataset card

Parquet schema — telemetry (access_log / baseline / idp / host splits)
-----------------------------------------------------------------------
  record_type     str   — commit | repo_access | email_send | idp_auth |
                          host_event | slack_message | phone_call | dlp_alert
  day             int   — simulation day (1-indexed)
  date            str   — ISO date
  timestamp       str   — ISO datetime
  actor           str   — employee name (no threat annotation)
  + all detail columns from the original JSONL record (flattened one level)

Ground truth adds three columns:
  true_positive   bool
  threat_class    str   — negligent | disgruntled | malicious | null
  behavior        str   — behavior name | null

Requirements
------------
    pip install pandas pyarrow pyyaml
"""

from __future__ import annotations

import json
import logging
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import yaml

logger = logging.getLogger("orgforge.it_export_hf")

# ── Config ─────────────────────────────────────────────────────────────────────

_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "config.yaml"

with open(_CONFIG_PATH) as f:
    _CFG = yaml.safe_load(f)

_SIM_CFG = _CFG.get("simulation", {})
_IT_CFG = _CFG.get("insider_threat", {})

BASE = Path(_SIM_CFG.get("output_dir", "./export"))
TELEMETRY_DIR = BASE / _IT_CFG.get("telemetry_dir", "security_telemetry")
LEADERBOARD_JSON = Path("insider_threat_leaderboard.json")
LEADERBOARD_CSV = Path("insider_threat_leaderboard.csv")

HF_DIR = BASE / "hf_insider_threat"
HF_TELEMETRY_DIR = HF_DIR / "telemetry"
HF_GT_DIR = HF_DIR / "ground_truth"
HF_LB_DIR = HF_DIR / "leaderboard"

for d in (HF_TELEMETRY_DIR, HF_GT_DIR, HF_LB_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── Optional Parquet support ────────────────────────────────────────────────────

try:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    _PARQUET_AVAILABLE = True
except ImportError:
    _PARQUET_AVAILABLE = False
    logger.warning(
        "pandas/pyarrow not installed — JSON fallback enabled. pip install pandas pyarrow"
    )


# ─────────────────────────────────────────────────────────────────────────────
# JSONL READER
# ─────────────────────────────────────────────────────────────────────────────


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


# ─────────────────────────────────────────────────────────────────────────────
# RECORD NORMALISER
# ─────────────────────────────────────────────────────────────────────────────


# Top-level fields that are always present — kept as dedicated columns.
_CORE_FIELDS = {"record_type", "day", "date", "timestamp", "actor"}

# Ground-truth-only fields — stripped from the observable stream.
_GT_FIELDS = {"true_positive", "threat_class", "behavior"}

# Detail fields that warrant their own column rather than being buried in JSON.
# Everything else is left as a nested JSON string under "extra".
_PROMOTED_FIELDS = {
    # IDP auth
    "auth_result",
    "dst_app",
    "src_ip",
    "device_id",
    "device_os",
    "mfa_method",
    "new_device",
    "anomalous_ip",
    "ip_type",
    "access_hour",
    "outside_business_hours",
    "corroborating_activity_expected",
    "ghost_login",
    "preceded_by_call_record",
    "call_to_auth_gap_minutes",
    # Host events
    "action",
    "source_shares",
    "file_count",
    "staged_path",
    "total_bytes",
    "archive_name",
    "compressed_bytes",
    "destination_type",
    "destination_path",
    "cloud_sync_dir",
    "removable_media",
    "hoarding_trail_start_day",
    # Commits / repo access
    "pr_id",
    "ticket_id",
    "secret_var",
    "repos_cloned",
    "anomalous",
    "cross_dept",
    "ticket_dept",
    "accessor_dept",
    # Email
    "to",
    "subject",
    "is_external",
    "off_hours",
    # Social engineering
    "pattern",
    "target",
    "spoofed_sender",
    "reply_to_domain_mismatch",
    "sender_in_known_contacts",
    "followup_due_day",
    # DLP / general
    "policy_trigger",
    "channel",
    "sentiment",
    # Phone call
    "call_direction",
    "call_duration_seconds",
    "caller_id_in_known_contacts",
    "spoofed_caller_id",
}


def normalise_record(raw: dict, include_gt: bool = False) -> dict:
    """
    Flatten a raw JSONL record into a schema-stable row.

    Core fields become top-level columns.
    Promoted detail fields are lifted from the nested details dict.
    Everything remaining is serialised under `extra` as a JSON string.
    Ground-truth fields are included only when include_gt=True.
    """
    row: Dict[str, Any] = {}

    # Core fields
    for f in _CORE_FIELDS:
        row[f] = raw.get(f, None)

    # Ground truth fields (gt file only)
    if include_gt:
        for f in _GT_FIELDS:
            row[f] = raw.get(f, None)

    # Promoted detail fields — may appear at top level or be nested
    # (older JSONL has them at top level; newer has them inside details)
    detail_keys = set(raw.keys()) - _CORE_FIELDS - _GT_FIELDS
    remaining = {}

    for k in detail_keys:
        if k in _PROMOTED_FIELDS:
            row[k] = raw[k]
        else:
            remaining[k] = raw[k]

    # Anything left over goes into extra
    row["extra"] = json.dumps(remaining, default=str) if remaining else None

    return row


# ─────────────────────────────────────────────────────────────────────────────
# PARQUET WRITER
# ─────────────────────────────────────────────────────────────────────────────


def write_parquet(rows: List[dict], out_dir: Path, stem: str) -> Path:
    """Write rows to Parquet (or JSON fallback). Returns the output path."""
    if not rows:
        logger.warning(f"  Skipping {stem} — no rows to write")
        return out_dir / f"{stem}.parquet"

    if not _PARQUET_AVAILABLE:
        out_path = out_dir / f"{stem}.json"
        with open(out_path, "w") as f:
            json.dump(rows, f, indent=2, default=str)
        logger.info(f"  → {out_path} ({len(rows):,} rows — JSON fallback)")
        return out_path

    df = pd.DataFrame(rows)

    # Coerce bool columns that may have come in as strings
    for col in (
        "true_positive",
        "new_device",
        "anomalous_ip",
        "outside_business_hours",
        "is_external",
        "off_hours",
        "cross_dept",
        "anomalous",
        "ghost_login",
        "preceded_by_call_record",
        "removable_media",
        "corroborating_activity_expected",
        "caller_id_in_known_contacts",
        "reply_to_domain_mismatch",
        "sender_in_known_contacts",
    ):
        if col in df.columns:
            df[col] = df[col].map(
                lambda v: (
                    True
                    if str(v).lower() in ("true", "1", "yes")
                    else False
                    if str(v).lower() in ("false", "0", "no")
                    else None
                )
            )

    # Coerce numeric columns
    for col in (
        "day",
        "access_hour",
        "file_count",
        "total_bytes",
        "compressed_bytes",
        "call_duration_seconds",
        "call_to_auth_gap_minutes",
        "hoarding_trail_start_day",
        "repos_cloned",
        "followup_due_day",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    tbl = pa.Table.from_pandas(df, preserve_index=False)
    out_path = out_dir / f"{stem}.parquet"
    pq.write_table(tbl, out_path, compression="snappy")
    logger.info(
        f"  → {out_path} ({len(rows):,} rows, {out_path.stat().st_size // 1024} KB)"
    )
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# DATASET CARD WRITER
# ─────────────────────────────────────────────────────────────────────────────


def _table_rows(d: Dict[str, int]) -> str:
    return "\n".join(
        f"| `{k}` | {v:,} |" for k, v in sorted(d.items(), key=lambda x: -x[1])
    )


def write_dataset_card(
    out_path: Path,
    obs_records: List[dict],
    gt_records: List[dict],
    baseline_records: List[dict],
    leaderboard: List[dict],
    cfg: dict,
) -> None:
    sim_cfg = cfg.get("simulation", {})
    it_cfg = cfg.get("insider_threat", {})
    subjects = it_cfg.get("subjects", [])

    company = sim_cfg.get("company_name", "OrgForge Simulated Corp")
    industry = sim_cfg.get("industry", "Software")
    num_days = sim_cfg.get("max_days", "?")
    noise_ratio = it_cfg.get("dlp_noise_ratio", 0.4)
    log_format = it_cfg.get("log_format", "jsonl")

    # Summary stats
    by_type: Dict[str, int] = defaultdict(int)
    for r in obs_records:
        by_type[r.get("record_type", "unknown")] += 1

    by_class: Dict[str, int] = defaultdict(int)
    by_behavior: Dict[str, int] = defaultdict(int)
    for r in gt_records:
        if r.get("true_positive"):
            by_class[r.get("threat_class", "unknown")] += 1
            if r.get("behavior"):
                by_behavior[r["behavior"]] += 1

    tp_count = sum(1 for r in gt_records if r.get("true_positive"))
    fp_count = sum(1 for r in gt_records if not r.get("true_positive"))

    # Best leaderboard row
    best_row = max(leaderboard, key=lambda r: r.get("verdict_f1") or 0.0, default={})
    best_model = best_row.get("model", "n/a")
    best_f1 = best_row.get("verdict_f1", "n/a")

    subject_table = "\n".join(
        f"| {s['name']} | {s.get('threat_class', '?')} | {s.get('onset_day', '?')} | "
        f"{', '.join(s.get('behaviors', []))} |"
        for s in subjects
    )

    card = textwrap.dedent(f"""\
    ---
    language:
      - en
    license: mit
    task_categories:
      - text-classification
      - token-classification
    tags:
      - security
      - insider-threat
      - synthetic
      - orgforge
      - siem
      - detection
      - behavioral-analytics
      - anomaly-detection
      - time-series
      - security-analytics
    pretty_name: "OrgForge Insider Threat Detection Benchmark"
    size_categories:
      - 1K<n<10K
    ---

    # OrgForge Insider Threat Detection Benchmark

    > Structured security telemetry for benchmarking LLM-based insider threat
    > detection. No embedder required — the corpus is pre-structured JSONL/Parquet.
    > Ground truth is derived deterministically from the simulation's event log.

    ## Dataset Summary

    This dataset was produced by **OrgForge**, an event-driven organisation
    simulator, with the insider threat module enabled. A corporate simulation runs
    for {num_days} days; configured threat subjects exhibit realistic anomalous
    behaviors across multiple artifact surfaces. Detection agents read the
    observable telemetry stream and must correlate signals across days and record
    types without access to ground truth labels.

    | Property | Value |
    |---|---|
    | Company | {company} |
    | Industry | {industry} |
    | Simulation days | {num_days} |
    | Threat subjects | {len(subjects)} |
    | DLP noise ratio | {noise_ratio} |
    | Log format | {log_format} |
    | Total observable records | {len(obs_records):,} |
    | True positive records | {tp_count:,} |
    | False positive (noise) records | {fp_count:,} |
    | Baseline (clean) records | {len(baseline_records):,} |
    | Best leaderboard verdict F1 | {best_f1} ({best_model}) |

    ## Threat Subjects

    | Name | Class | Onset day | Behaviors |
    |---|---|---|---|
    {subject_table}

    Subjects behave normally before their onset day. Pre-onset records are clean
    true negatives and form the detection baseline.

    ## Telemetry Files

    All files are in `telemetry/`. The observable stream contains no ground truth
    labels — `true_positive`, `threat_class`, and `behavior` fields appear only in
    `ground_truth/`.

    | File | Description | Rows |
    |---|---|---|
    | `access_log-00000.parquet` | Full observable stream | {len(obs_records):,} |
    | `baseline_telemetry-00000.parquet` | Pre-onset clean records only | {len(baseline_records):,} |
    | `idp_auth-00000.parquet` | IDP authentication events | {by_type.get("idp_auth", 0):,} |
    | `host_events-00000.parquet` | Host-level staging events | {by_type.get("host_event", 0):,} |

    ### Record types in the observable stream

    {_table_rows(dict(by_type))}

    ### Telemetry schema

    Core fields present on every record:

    | Column | Type | Description |
    |---|---|---|
    | `record_type` | str | Event category (see table above) |
    | `day` | int | Simulation day (1-indexed) |
    | `date` | str | ISO date |
    | `timestamp` | str | ISO datetime (SimClock-accurate) |
    | `actor` | str | Employee name — no threat annotation |
    | `extra` | str | JSON string of additional detail fields |

    Selected promoted detail columns (present when applicable):

    | Column | Type | Description |
    |---|---|---|
    | `auth_result` | str | IDP: `success` or `mfa_failure` |
    | `src_ip` | str | IDP: source IP address |
    | `new_device` | bool | IDP: device not in employee's known profile |
    | `anomalous_ip` | bool | IDP: IP outside corporate range |
    | `ghost_login` | bool | IDP: disgruntled ghost login pattern |
    | `preceded_by_call_record` | bool | IDP: auth preceded by phone_call (vishing) |
    | `action` | str | Host: `bulk_file_copy`, `archive_creation`, `archive_move` |
    | `hoarding_trail_start_day` | int | Host: links phase 3 back to phase 1 |
    | `destination_type` | str | Host: `cloud_sync_dir` or `removable_media` |
    | `outside_business_hours` | bool | All: access outside 09:00–18:00 |
    | `pattern` | str | Social engineering: `spear_phishing`, `slack_pretexting`, `vishing_breadcrumb`, `trust_building` |
    | `preceded_by_call_record` | bool | IDP: marks vishing post-auth events |

    ## Ground Truth

    `ground_truth/ground_truth-00000.parquet` contains all records from the
    observable stream with three additional columns:

    | Column | Type | Description |
    |---|---|---|
    | `true_positive` | bool | True if this record is a genuine threat signal |
    | `threat_class` | str | `negligent`, `disgruntled`, `malicious`, or null |
    | `behavior` | str | Behavior name, or null for false positive records |

    **This file is held out during evaluation.** Detection agents read only the
    telemetry files. Ground truth is used for scoring only.

    ### Behaviors present in ground truth

    {_table_rows(dict(by_behavior))}

    ### Threat class distribution

    {_table_rows(dict(by_class))}

    ## Detection Scenarios

    The dataset is designed to benchmark reasoning under noise, not keyword
    matching. Key scenarios:

    **Ghost login (disgruntled):** `idp_auth` success present but no Slack,
    Jira, or email artifacts follow in the same time window.
    `corroborating_activity_expected: false` marks these records.

    **Vishing (malicious):** A `phone_call` record is followed within minutes by
    an `idp_auth` success filed under the **target's** actor name, not the
    attacker's. `preceded_by_call_record: true` and `call_to_auth_gap_minutes`
    link them. Per-actor triage agents miss this entirely — cross-actor
    correlation is required.

    **Host data hoarding (malicious/disgruntled):** Three-phase trail spread
    across consecutive days: `bulk_file_copy` → `archive_creation` →
    `archive_move`. The `hoarding_trail_start_day` field in phase 3 records
    links back to phase 1. Single-day triage agents will always miss the
    complete trail.

    **Trust building (social engineering):** A benign inbound email with
    `sender_in_known_contacts: false` precedes a follow-up attack 3–5 days
    later. The first contact is a clean false negative in isolation.

    ## Leaderboard

    `leaderboard/insider_threat_leaderboard.csv` contains a frozen snapshot of
    all model runs against this export. The full JSON version is also included.

    Columns:

    | Column | Description | Better |
    |---|---|---|
    | `triage_f1` | F1 on escalation decisions | ↑ |
    | `verdict_f1` | F1 on full case verdicts | ↑ |
    | `baseline_fp_rate` | FP rate on clean baseline period | ↓ |
    | `onset_sensitivity` | Fraction of pre-onset escalations | ↓ |
    | `vishing_detected` | Did the agent correlate phone_call → idp_auth? | ✓ |
    | `host_trail_reconstructed` | Did the agent cite all 3 hoarding phases? | ✓ |

    To add a row, run `eval_insider_threat.py` against this export and append
    the output to these files.

    ## Evaluation Pipeline

    ```bash
    # Build baseline (pre-onset clean records)
    python build_baseline_telemetry.py --export-dir ./export

    # Run detection pipeline (one command per model)
    python eval_insider_threat.py \\
        --model anthropic.claude-opus-4-5-20251101-v1:0 \\
        --export-dir ./export

    # Launch leaderboard UI
    python app.py   # (insider threat Gradio app)
    ```

    No embedder, no MongoDB, no vector database required. Credentials: AWS
    Bedrock (standard credential chain).

    ## Citation

    ```bibtex
    @misc{{orgforge_it2026,
      title  = {{OrgForge Insider Threat Detection Benchmark}},
      author = {{Jeffrey Flynt}},
      year   = {{2026}},
      note   = {{Synthetic benchmark generated by the OrgForge insider threat simulator}}
    }}
    ```

    ## Related Paper

    https://arxiv.org/abs/2603.14997

    ## Licence

    MIT. The simulation engine that produced this dataset is independently licensed; see the OrgForge repository for details.
    """)

    out_path.write_text(card, encoding="utf-8")
    logger.info(f"  → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXPORTER
# ─────────────────────────────────────────────────────────────────────────────


class InsiderThreatHFExporter:
    """
    Orchestrates the full insider threat HF export:
      1. Load telemetry JSONL files
      2. Normalise and split into logical subsets
      3. Write Parquet files
      4. Copy leaderboard snapshots
      5. Write dataset card
    """

    def run(self) -> None:
        logger.info("📦  Insider threat HuggingFace dataset export starting…")

        # ── 1. Load raw JSONL ─────────────────────────────────────────────────
        obs_raw = load_jsonl(TELEMETRY_DIR / "access_log.jsonl")
        gt_raw = load_jsonl(TELEMETRY_DIR / "_ground_truth.jsonl")
        baseline_raw = load_jsonl(TELEMETRY_DIR / "baseline_telemetry.jsonl")

        if not obs_raw:
            logger.error(
                f"No access_log.jsonl found at {TELEMETRY_DIR}. "
                f"Run flow.py with insider_threat.enabled: true first."
            )
            return

        if not gt_raw:
            logger.warning(
                f"No _ground_truth.jsonl found at {TELEMETRY_DIR}. "
                f"Ground truth parquet will be empty."
            )

        logger.info(
            f"  Loaded: {len(obs_raw):,} observable, "
            f"{len(gt_raw):,} ground truth, "
            f"{len(baseline_raw):,} baseline records"
        )

        # ── 2. Normalise ──────────────────────────────────────────────────────
        obs_rows = [normalise_record(r, include_gt=False) for r in obs_raw]
        gt_rows = [normalise_record(r, include_gt=True) for r in gt_raw]
        baseline_rows = [normalise_record(r, include_gt=False) for r in baseline_raw]

        # Subsets for convenience splits
        idp_rows = [r for r in obs_rows if r.get("record_type") == "idp_auth"]
        host_rows = [r for r in obs_rows if r.get("record_type") == "host_event"]

        # ── 3. Write Parquet ──────────────────────────────────────────────────
        logger.info("  Writing telemetry Parquet files…")
        write_parquet(obs_rows, HF_TELEMETRY_DIR, "access_log-00000")
        write_parquet(baseline_rows, HF_TELEMETRY_DIR, "baseline_telemetry-00000")
        write_parquet(idp_rows, HF_TELEMETRY_DIR, "idp_auth-00000")
        write_parquet(host_rows, HF_TELEMETRY_DIR, "host_events-00000")

        logger.info("  Writing ground truth Parquet file…")
        write_parquet(gt_rows, HF_GT_DIR, "ground_truth-00000")

        # ── 4. Leaderboard snapshot ───────────────────────────────────────────
        leaderboard: List[dict] = []
        if LEADERBOARD_JSON.exists():
            import shutil

            shutil.copy(LEADERBOARD_JSON, HF_LB_DIR / "insider_threat_leaderboard.json")
            leaderboard = json.loads(LEADERBOARD_JSON.read_text())
            logger.info(
                f"  → {HF_LB_DIR / 'insider_threat_leaderboard.json'} ({len(leaderboard)} rows)"
            )
        else:
            logger.warning(
                f"  No leaderboard JSON at {LEADERBOARD_JSON} — skipping snapshot"
            )

        if LEADERBOARD_CSV.exists():
            import shutil

            shutil.copy(LEADERBOARD_CSV, HF_LB_DIR / "insider_threat_leaderboard.csv")
            logger.info(f"  → {HF_LB_DIR / 'insider_threat_leaderboard.csv'}")
        else:
            logger.warning(
                f"  No leaderboard CSV at {LEADERBOARD_CSV} — skipping snapshot"
            )

        # ── 5. Dataset card ───────────────────────────────────────────────────
        logger.info("  Writing dataset card…")
        write_dataset_card(
            out_path=HF_DIR / "README.md",
            obs_records=obs_raw,
            gt_records=gt_raw,
            baseline_records=baseline_raw,
            leaderboard=leaderboard,
            cfg=_CFG,
        )

        # ── Summary ───────────────────────────────────────────────────────────
        tp = sum(1 for r in gt_raw if r.get("true_positive"))
        fp = sum(1 for r in gt_raw if not r.get("true_positive"))
        print(f"\n{'─' * 60}")
        print("  ✓  Insider threat HF export complete")
        print(f"{'─' * 60}")
        print(f"  Output dir         : {HF_DIR}")
        print(f"  Observable records : {len(obs_rows):,}")
        print(f"  Ground truth rows  : {len(gt_rows):,}  (TP={tp}, FP={fp})")
        print(f"  Baseline records   : {len(baseline_rows):,}")
        print(f"  IDP auth records   : {len(idp_rows):,}")
        print(f"  Host event records : {len(host_rows):,}")
        print(f"  Leaderboard rows   : {len(leaderboard)}")
        print(f"{'─' * 60}")
        print("\n  Upload to HuggingFace:")
        print(f"    cd {HF_DIR}")
        print("    huggingface-cli upload <your-org>/orgforge-insider-threat .")
        print()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    InsiderThreatHFExporter().run()
