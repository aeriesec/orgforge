"""
eval_insider_threat.py
======================
Insider threat detection leaderboard for OrgForge.

Runs the full 3-stage detection pipeline (Baseline → Triage → Correlation)
against a frozen simulation export using any Bedrock model, then appends a
scored row to insider_threat_leaderboard.json.

No embedder required — the telemetry stream is pre-structured JSONL.
No MongoDB required — Slack artifacts are read directly from the export dir.

Usage
-----
    # Single model run (official leaderboard)
    python eval_insider_threat.py \\
        --model anthropic.claude-opus-4-6-20260401-v1:0 \\
        --export-dir ./export \\
        --config config/config.yaml

    # Prompt sensitivity run (never touches leaderboard)
    python eval_insider_threat.py \\
        --model anthropic.claude-sonnet-4-6-20260401-v1:0 \\
        --prompt-variant v2_natural

    # Run all variants for one model
    for VARIANT in official v2_natural v3_examples_first; do
        python eval_insider_threat.py \\
            --model anthropic.claude-sonnet-4-6-20260401-v1:0 \\
            --prompt-variant $VARIANT
    done

    # Quick smoke-test (triage only, skip baseline and correlation)
    python eval_insider_threat.py --model amazon.nova-pro-v1:0 --triage-only

    # Full leaderboard sweep (run once per model, leaderboard auto-sorts)
    for MODEL in \\
        anthropic.claude-opus-4-6-20260401-v1:0 \\
        anthropic.claude-sonnet-4-6-20260401-v1:0 \\
        anthropic.claude-haiku-4-5-20251001-v1:0 \\
        meta.llama3-3-70b-instruct-v1:0 \\
        amazon.nova-pro-v1:0 \\
        mistral.mistral-large-2402-v1:0; do
        python eval_insider_threat.py --model $MODEL
    done

Output
------
Official runs  (--prompt-variant omitted or "official"):
    results/insider_threat/<run_id>/
        triage_decisions.json     — per-window triage decisions
        verdicts.json             — correlation agent verdicts
        scores.json               — precision / recall / F1 + breakdowns
    insider_threat_leaderboard.json   — append-only leaderboard (JSON)
    insider_threat_leaderboard.csv    — same, CSV-friendly

Sensitivity runs  (--prompt-variant v2_natural | v3_examples_first):
    results/sensitivity/<run_id>/
        triage_decisions.json
        verdicts.json
        scores.json
        prompt_variant.txt        — records which variant was used
    Leaderboard files are NEVER written.

Leaderboard columns
-------------------
Tier 1  (--triage-only)
    triage_precision, triage_recall, triage_f1
    baseline_fp_rate          — FP rate on clean pre-onset period
    onset_sensitivity         — fraction of subjects flagged BEFORE onset_day (bad)

Tier 2  (full pipeline, default)
    All Tier 1 columns, plus:
    verdict_precision, verdict_recall, verdict_f1
    by_class                  — per threat-class breakdown (negligent/disgruntled/malicious)
    by_behavior               — per behavior detection rate
    vishing_detected          — bool: did the agent correlate the phone_call → idp_auth pair
    host_trail_reconstructed  — bool: did the agent cite all 3 hoarding phases
    secret_in_commit_detected — bool: did the credential scan + correlation stage flag
                                the negligent insider's credential leak

Requirements
------------
    pip install boto3 pyyaml
    AWS credentials via env vars, ~/.aws/credentials, or IAM role.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from botocore.config import Config as BotocoreConfig

import yaml

logger = logging.getLogger("orgforge.it_eval")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

RESULTS_DIR = Path("results") / "insider_threat"
SENSITIVITY_DIR = Path("results") / "sensitivity"
LEADERBOARD_JSON = Path("insider_threat_leaderboard.json")
LEADERBOARD_CSV = Path("insider_threat_leaderboard.csv")

# Sentinel value meaning "use the canonical production prompts"
OFFICIAL_VARIANT = "official"

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

# Bedrock model IDs available for the leaderboard.
# Cross-region inference profiles are also accepted (us.anthropic.* etc.)
_SUGGESTED_MODELS = [
    "anthropic.claude-opus-4-6-20260401-v1:0",
    "anthropic.claude-sonnet-4-6-20260401-v1:0",
    "anthropic.claude-haiku-4-5-20251001-v1:0",
    "meta.llama3-3-70b-instruct-v1:0",
    "amazon.nova-pro-v1:0",
    "mistral.mistral-large-2402-v1:0",
]


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG HELPERS
# ─────────────────────────────────────────────────────────────────────────────


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_start_date(cfg: dict) -> datetime:
    raw = cfg["simulation"].get("start_date", "2026-03-02")
    return datetime.strptime(str(raw), "%Y-%m-%d")


def get_subjects(cfg: dict) -> List[dict]:
    return cfg.get("insider_threat", {}).get("subjects", [])


def get_subject_names(cfg: dict) -> set:
    return {s["name"] for s in get_subjects(cfg)}


def get_onset_days(cfg: dict) -> Dict[str, int]:
    return {s["name"]: s.get("onset_day", 1) for s in get_subjects(cfg)}


def get_company_name(cfg: dict) -> str:
    return cfg.get("simulation", {}).get("company_name", "the company")


def get_employee_list(cfg: dict) -> str:
    org = cfg.get("org_chart", {})
    lines = []
    for dept, members in org.items():
        for m in members:
            lines.append(f"  {m} ({dept})")
    return "\n".join(lines)


def get_max_day(cfg: dict) -> int:
    return cfg.get("simulation", {}).get("max_days", 30)


def date_to_day(d: datetime, start: datetime) -> int:
    return (d.date() - start.date()).days + 1


# ─────────────────────────────────────────────────────────────────────────────
# TELEMETRY I/O
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


def records_in_window(records: List[dict], start: int, end: int) -> List[dict]:
    return [r for r in records if start <= r.get("day", 0) <= end]


def jsonl_str(records: List[dict]) -> str:
    return "\n".join(json.dumps(r) for r in records)


def load_ground_truth(path: Path) -> Dict[str, set]:
    """Returns {actor: {behavior, ...}} for true positives only."""
    gt: Dict[str, set] = {}
    for rec in load_jsonl(path):
        if rec.get("true_positive"):
            name = rec.get("actor", "")
            behavior = rec.get("behavior", "")
            if name and behavior:
                gt.setdefault(name, set()).add(behavior)
    return gt


def load_ground_truth_full(path: Path) -> Dict[str, dict]:
    """Returns {actor: {behaviors: set, threat_class: str}} for true positives."""
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


# ─────────────────────────────────────────────────────────────────────────────
# SLACK READER (direct from export — no MongoDB)
# ─────────────────────────────────────────────────────────────────────────────


def read_slack_for_actors(
    export_dir: Path,
    actors: List[str],
    limit_per_actor: int = 10,
) -> List[dict]:
    """
    Read Slack messages for the given actors directly from the export directory.
    Skips bots and _security_injected messages.
    Returns messages sorted oldest-first so tone progression is visible.
    """
    actor_set = set(actors)
    messages: List[dict] = []

    channels_dir = export_dir / "slack" / "channels"
    if not channels_dir.exists():
        return messages

    per_actor: Dict[str, List[dict]] = defaultdict(list)

    for channel_dir in sorted(channels_dir.iterdir()):
        if not channel_dir.is_dir():
            continue
        channel = channel_dir.name

        for json_file in sorted(channel_dir.glob("*.json")):
            try:
                msgs = json.loads(json_file.read_text())
            except Exception:
                continue

            for msg in msgs:
                if msg.get("is_bot"):
                    continue
                if msg.get("_security_injected"):
                    continue
                user = msg.get("user", "")
                if user not in actor_set:
                    continue
                per_actor[user].append(
                    {
                        "user": user,
                        "text": msg.get("text", "")[:300],
                        "ts": msg.get("ts", ""),
                        "channel": channel,
                        "date": msg.get("date", ""),
                    }
                )

    for actor, msgs in per_actor.items():
        msgs.sort(key=lambda m: m.get("ts", ""))
        messages.extend(msgs[:limit_per_actor])

    return messages


def format_slack(messages: List[dict]) -> str:
    if not messages:
        return "  (none)"
    return "\n".join(
        f"  [{m.get('date') or m.get('ts', '')[:10]} #{m.get('channel', '')}] "
        f"{m.get('user', '')}: {m.get('text', '')}"
        for m in messages
    )


# ─────────────────────────────────────────────────────────────────────────────
# BEDROCK LLM WRAPPER
# ─────────────────────────────────────────────────────────────────────────────


class BedrockLLM:
    """
    Thin Bedrock converse() wrapper.

    Authentication uses the standard AWS credential chain:
      - AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY env vars
      - ~/.aws/credentials profile
      - IAM role (EC2 / ECS / Lambda)
    """

    def __init__(
        self,
        model: str,
        region: str = "us-east-1",
        call_delay: float = 1.0,
        max_retries: int = 6,
        retry_base_delay: float = 5.0,
    ):
        try:
            import boto3
        except ImportError:
            raise SystemExit("pip install boto3")

        if not region or len(region.split("-")) < 3:
            raise ValueError(
                f"Invalid AWS region: {region!r}. Expected format: us-east-1"
            )

        self.model = model
        self._call_delay = call_delay
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay
        self._client = boto3.client(
            "bedrock-runtime",
            region_name=region,
            config=BotocoreConfig(
                read_timeout=500,  # 5 minutes — DeepSeek and large models need this
                connect_timeout=10,
                retries={"max_attempts": 0},  # we handle retries ourselves
            ),
        )
        logger.info(f"Bedrock — model: {model}  region: {region}")

    def call(self, system: str, user: str, max_tokens: int = 4096) -> str:
        if self._call_delay > 0:
            time.sleep(self._call_delay)

        last_exc = None
        for attempt in range(self._max_retries + 1):
            try:
                resp = self._client.converse(
                    modelId=self.model,
                    system=[{"text": system}],
                    messages=[{"role": "user", "content": [{"text": user}]}],
                    inferenceConfig={
                        "maxTokens": max_tokens,
                        "temperature": 0.0,
                    },
                )
                for block in resp["output"]["message"]["content"]:
                    if "text" in block:
                        return block["text"]
                return ""

            except Exception as exc:
                # Safe extraction — response may be None for network errors
                response_meta = getattr(exc, "response", None) or {}
                code = response_meta.get("Error", {}).get("Code", "")

                is_throttle = code == "ThrottlingException"
                is_timeout = (
                    "ReadTimeoutError" in type(exc).__name__
                    or "ReadTimeout" in type(exc).__name__
                    or isinstance(exc, TimeoutError)
                )

                if (is_throttle or is_timeout) and attempt < self._max_retries:
                    delay = self._retry_base_delay * (2**attempt)
                    delay *= 0.8 + 0.4 * random.random()
                    reason = "Throttled" if is_throttle else "Timeout"
                    logger.warning(
                        f"{reason} (attempt {attempt + 1}/{self._max_retries}), "
                        f"retrying in {delay:.1f}s"
                    )
                    time.sleep(delay)
                    last_exc = exc
                else:
                    raise

        raise last_exc

    def call_json(self, system: str, user: str, max_tokens: int = 4096) -> Any:
        """Call and parse response as JSON. Strips markdown fences."""
        raw = self.call(system, user, max_tokens)
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [l for l in lines[1:] if not l.strip().startswith("```")]
            cleaned = "\n".join(lines).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Try extracting a complete JSON array or object
            m = re.search(r"(\[.*\]|\{.*\})", cleaned, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(1))
                except Exception:
                    pass
            # Last resort: if response looks truncated (no closing bracket),
            # log the full raw response at DEBUG level for diagnosis
            if len(raw) >= max_tokens * 3:  # rough chars-per-token estimate
                logger.warning(
                    f"Response may be truncated (raw length {len(raw)} chars, "
                    f"max_tokens={max_tokens}). Consider increasing max_tokens."
                )
            logger.warning(f"JSON parse failed. Raw:\n{raw[:500]}")
            return None


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────
#
# OFFICIAL prompts are the canonical leaderboard prompts. Do not edit these
# without bumping the leaderboard version and re-running all models.
#
# SENSITIVITY variants share the same semantic intent but differ in structure
# and phrasing. They are used only by --prompt-variant runs and are never
# written to the leaderboard. The goal is to test whether model rankings are
# stable across prompt phrasings or are artifacts of the official wording.
#
# To add a new variant:
#   1. Add entries to _SENSITIVITY_VARIANTS below.
#   2. Add the key to the --prompt-variant choices list in _parse_args().
#   3. Run with --prompt-variant <key>; results go to results/sensitivity/.
# ─────────────────────────────────────────────────────────────────────────────

# ── Official (production) prompts ────────────────────────────────────────────

_BASELINE_SYSTEM_OFFICIAL = """\
You are a security analyst calibrating an insider threat detection system.
You are reading telemetry from the CLEAN BASELINE period — no threat behaviors
were active during this window. Every employee here is innocent by construction.

Your job: flag any employees whose behavior looks statistically anomalous compared
to their peers. These detections count as false positives and set our FP threshold.

Rules:
  - Be conservative. One off-hours login is not anomalous.
  - Look for patterns: repeated off-hours access, unusual message volume,
    access to unexpected resources.
  - IDP records showing new devices or residential IPs are worth noting.
  - Return an empty array [] if nothing stands out.

Return ONLY a JSON array (may be empty):
[{"actor": "<n>", "anomaly_type": "<brief>", "fp_likelihood": "low|medium|high"}]
No other text."""

_BASELINE_USER_OFFICIAL = """\
Baseline telemetry (clean period — no threats active):
{telemetry}

Flag any statistically anomalous employees. Return JSON array only."""

_TRIAGE_SYSTEM_OFFICIAL = """\
You are a security analyst at {company}.
Triage the telemetry for this 7-day window and decide which employees warrant
escalation to Tier 2 investigation.

Telemetry record types you may see:
  repo_access    — git activity, clone events, cross-dept ticket reads
  commit         — PR events; may contain credential patterns (secret_in_commit)
  email_send     — outbound email; flag external recipients (gmail/protonmail/yahoo)
  slack_message  — Slack activity; read tone in the message text carefully
  idp_auth       — SSO authentication; note off-hours, new devices, residential IPs
  host_event     — local file system staging activity (bulk copy, compression, archive move)
  phone_call     — telephony record; correlate with idp_auth events close in time
  dlp_alert      — SIEM policy alert (may be false positive — treat as one signal only)

Escalation rules:
  - Single low-severity signal alone: do NOT escalate.
  - 2+ distinct signal types for the same employee in this window: escalate.
  - idp_auth with new_device AND off-hours in same window: escalate.
  - host_event bulk_file_copy + any external email: escalate.
  - phone_call followed within 30 min by idp_auth success on ANY employee account: escalate BOTH.
  - Passive-aggressive or disengaged Slack tone + off-hours access: escalate.
  - External email to gmail/protonmail/yahoo/outlook from corporate account: escalate.

Return ONE decision object per unique employee seen in the telemetry.
Return ONLY a JSON array:
[{
  "escalate": true|false,
  "actor": "<n>",
  "reason": "<one sentence>",
  "confidence": "low|medium|high",
  "signals_seen": ["<type>", ...]
}]
No other text."""

_TRIAGE_USER_OFFICIAL = """\
Window: days {start}–{end}

Telemetry:
{telemetry}

Employee directory:
{employees}
{slack_section}
Return triage decisions for every employee in the telemetry."""

_CORRELATION_SYSTEM_OFFICIAL = """\
You are a Tier 2 security investigator at {company}.
Build a case FOR or AGAINST each flagged employee by correlating all signals
across the full telemetry timeline.

Behavior definitions:
  secret_in_commit       — credential or API key in a PR description or code diff
  unusual_hours_access   — repo/channel/system access outside 09:00–18:00
  excessive_repo_cloning — cloning more than 3 repos in one session
  sentiment_drift        — Slack tone that is passive-aggressive, bitter, or
                           deliberately over-neutral (malicious actors suppress
                           affect; flat tone + other signals = sentiment_drift)
  cross_dept_snooping    — reading JIRA tickets outside the accessor's own dept
  data_exfil_email       — email forwarded to personal address (gmail/yahoo/protonmail)
  host_data_hoarding     — 3-phase trail: bulk_file_copy → archive_creation →
                           archive_move to cloud_sync_dir or removable_media.
                           Cite all three phases if present. The hoarding_trail_start_day
                           field in archive_move records links back to phase 1.
  social_engineering     — crafted inbound email impersonating IT/HR, Slack pretext
                           from a low-relationship sender, or phone_call record
                           followed within 30 min by idp_auth on a DIFFERENT actor's
                           account (vishing — the auth is filed under the victim, not
                           the attacker)
  idp_anomaly            — authentication from unrecognised device, residential IP,
                           or off-hours with no corroborating work activity after

Important:
  - Single off-hours access: NOT sufficient for likely_threat.
  - External email to personal domain: high severity.
  - host_data_hoarding phase 3 (archive_move) + data_exfil_email in same 3-day
    window = strong exfiltration case.
  - For vishing: look for phone_call records then idp_auth on ANY employee within
    30 min — the attacker's name is on the phone_call, the victim's name is on
    the auth. Report the ATTACKER as the social_engineering subject.
  - Base rate: most escalations are innocent. Only flag when evidence is compelling.

Return ONE verdict object per flagged employee.
Return ONLY a JSON array:
[{
  "name": "<employee>",
  "verdict": "innocent|suspicious|likely_threat",
  "threat_class": "negligent|disgruntled|malicious|unknown",
  "behaviors_observed": ["<behavior>", ...],
  "evidence": [
    {"source": "<record_type>", "day": <day>, "description": "<what you saw>"}
  ],
  "recommended_action": "<one sentence>",
  "confidence": "low|medium|high"
}]
No other text."""

_CORRELATION_USER_OFFICIAL = """\
Flagged employees: {flagged}

Full telemetry for these employees (all days):
{telemetry}
{slack_section}
Build your case and return verdict objects."""

# ── Sensitivity variant: v2_natural ──────────────────────────────────────────
# Same semantic intent as official prompts, but written in plain conversational
# English with no explicit numbered rules or taxonomy headers. Tests whether
# models rely on structured formatting cues or on the underlying reasoning task.

_BASELINE_SYSTEM_V2 = """\
You are a security analyst reviewing employee activity during a quiet baseline
period. Nothing suspicious is supposed to be happening here — every employee
is behaving normally by design.

Have a look at the activity and flag anyone who seems statistically out of step
with their colleagues. You're trying to understand what "normal" looks like, so
err on the side of caution — don't flag things that are mildly unusual. Look
for genuine patterns: someone repeatedly accessing systems at 2am, or showing
up on an unrecognised device over and over.

If nothing stands out, return an empty list.

Return ONLY a JSON array (may be empty):
[{"actor": "<name>", "anomaly_type": "<brief description>", "fp_likelihood": "low|medium|high"}]
No other text."""

_BASELINE_USER_V2 = """\
Here is the baseline telemetry. Everyone here is innocent — your job is to
calibrate what normal looks like and catch anything that seems genuinely off.

{telemetry}

Return a JSON array of flagged employees, or [] if nothing stands out."""

_TRIAGE_SYSTEM_V2 = """\
You are a security analyst at {company}. You're doing first-pass triage on
a week of telemetry. Your job is to decide who needs a closer look by a
senior investigator.

The telemetry includes things like git activity, authentication logs, email
records, Slack messages, file system events, phone call records, and DLP alerts.

A single odd event is usually not enough to escalate — you're looking for
combinations. Someone sending an email to their personal Gmail is interesting;
someone doing that while also logging in at 3am on an unrecognised device is
worth escalating. A passive-aggressive Slack tone alone isn't escalation-worthy,
but paired with off-hours access it matters.

Pay particular attention to: off-hours authentication especially on new devices
or unusual IPs; email to personal domains like Gmail, ProtonMail, or Yahoo;
bulk file operations; and phone call records that are quickly followed by
an authentication event — the person who made the call and the person who
authenticated may be different employees.

Return one decision per employee seen in the telemetry.
Return ONLY a JSON array:
[{
  "escalate": true|false,
  "actor": "<name>",
  "reason": "<one sentence explaining your decision>",
  "confidence": "low|medium|high",
  "signals_seen": ["<signal type>", ...]
}]
No other text."""

_TRIAGE_USER_V2 = """\
Week: days {start} through {end}

Telemetry for this window:
{telemetry}

Employees at {company}:
{employees}
{slack_section}
Return your triage decisions for every employee who appears in the telemetry."""

_CORRELATION_SYSTEM_V2 = """\
You are a senior security investigator at {company}. Some employees have been
flagged by first-pass triage and you need to build a full case for or against
each one, using everything available: the complete telemetry timeline, Slack
message history, and any cross-employee correlations you can find.

You're looking for patterns that indicate one of these threat profiles:

  Negligent insider — accidentally leaves credentials in code, authenticates
  normally, no concealment, no malicious intent.

  Disgruntled insider — sentiment in Slack messages turns passive-aggressive
  or bitter over time; accesses resources outside their normal scope; unusual
  hours. Typically not trying to hide.

  Malicious insider — deliberately conceals activity; Slack tone goes flat
  and over-neutral (suppressing affect is a signal); stages data across
  multiple days before exfiltrating; uses social engineering.

When you see host file system events, look for a three-phase sequence: bulk
file copy to a staging directory, then compression into an archive, then the
archive moving to a cloud sync folder or removable media. The archive_move
record will have a hoarding_trail_start_day field pointing back to when the
copying began.

For phone call records: if a call is closely followed by an authentication
event on a different employee's account, the caller may have tricked the
other employee into providing access. The attacker's name is on the call
record; the victim's name is on the auth record.

Most escalations will turn out to be innocent. Only return suspicious or
likely_threat when the evidence is genuinely compelling across multiple signals.

Return ONE verdict per flagged employee as a JSON array:
[{
  "name": "<employee>",
  "verdict": "innocent|suspicious|likely_threat",
  "threat_class": "negligent|disgruntled|malicious|unknown",
  "behaviors_observed": ["<behavior name>", ...],
  "evidence": [
    {"source": "<record type>", "day": <day>, "description": "<what you observed>"}
  ],
  "recommended_action": "<one sentence>",
  "confidence": "low|medium|high"
}]
No other text."""

_CORRELATION_USER_V2 = """\
Employees to investigate: {flagged}

Complete telemetry timeline:
{telemetry}
{slack_section}
Build your case and return verdict objects."""

# ── Sensitivity variant: v3_examples_first ───────────────────────────────────
# Leads with a worked positive and negative example before stating rules.
# Tests whether grounding the model with concrete cases changes performance,
# particularly for models that triage adequately but fail at verdict synthesis.

_BASELINE_SYSTEM_V3 = """\
You are a security analyst calibrating an insider threat detection system.
This is the CLEAN BASELINE period — no threats are active. Every employee
is innocent by construction.

Your task is to flag statistical outliers so we can measure our false-positive
rate before the active monitoring period begins.

Example of what IS worth flagging:
  Alice: 14 off-hours authentications in 10 days, 3 from unrecognised devices
  → flag as anomalous (repeated pattern, hardware anomaly)

Example of what is NOT worth flagging:
  Bob: 1 off-hours authentication on day 3
  → do not flag (single event, no pattern)

Be conservative. Patterns matter; isolated events do not.

Return ONLY a JSON array (may be empty):
[{"actor": "<name>", "anomaly_type": "<brief>", "fp_likelihood": "low|medium|high"}]
No other text."""

_BASELINE_USER_V3 = """\
Baseline telemetry (clean period — all employees innocent):
{telemetry}

Flag statistical outliers only. Return JSON array (empty [] if none)."""

_TRIAGE_SYSTEM_V3 = """\
You are a security analyst at {company} doing first-pass triage on a 7-day
window of telemetry.

EXAMPLE — escalate this:
  Employee: Morgan
  Day 14: idp_auth at 02:17, unrecognised_device=true, source_ip=residential
  Day 15: email_send to morgan.personal@gmail.com, subject "Q3 data"
  → Two distinct signal types (off-hours auth + external email). Escalate.

EXAMPLE — do not escalate this:
  Employee: Casey
  Day 14: dlp_alert (low severity)
  → Single signal, low severity. Do not escalate.

EXAMPLE — escalate both parties:
  Employee A: phone_call to Employee B at 14:32
  Employee B: idp_auth success at 14:44 (12 min after call)
  → Phone call followed by auth on a different account within 30 min.
    Escalate the caller (possible vishing attacker) and the authenticator
    (possible victim).

Now apply this logic to the following window.

Telemetry record types: repo_access, commit, email_send, slack_message,
idp_auth, host_event, phone_call, dlp_alert.

Return ONE decision per employee in the telemetry.
Return ONLY a JSON array:
[{
  "escalate": true|false,
  "actor": "<name>",
  "reason": "<one sentence>",
  "confidence": "low|medium|high",
  "signals_seen": ["<type>", ...]
}]
No other text."""

_TRIAGE_USER_V3 = """\
Window: days {start}–{end}

Telemetry:
{telemetry}

Employee directory:
{employees}
{slack_section}
Return triage decisions for every employee in the telemetry."""

_CORRELATION_SYSTEM_V3 = """\
You are a Tier 2 security investigator at {company}. Build a full case for
or against each flagged employee using all available evidence.

EXAMPLE VERDICT — malicious insider correctly identified:
{{
  "name": "Alex",
  "verdict": "likely_threat",
  "threat_class": "malicious",
  "behaviors_observed": ["host_data_hoarding", "data_exfil_email", "unusual_hours_access"],
  "evidence": [
    {{"source": "host_event", "day": 23, "description": "bulk_file_copy: 67 files to /tmp/staging"}},
    {{"source": "host_event", "day": 24, "description": "archive_creation: staging.7z, hoarding_trail_start_day=23"}},
    {{"source": "host_event", "day": 25, "description": "archive_move to Dropbox sync dir"}},
    {{"source": "email_send", "day": 25, "description": "outbound email to alex.home@protonmail.com at 23:41"}}
  ],
  "recommended_action": "Suspend account pending HR and legal review.",
  "confidence": "high"
}}

EXAMPLE VERDICT — innocent employee correctly cleared:
{{
  "name": "Jordan",
  "verdict": "innocent",
  "threat_class": "unknown",
  "behaviors_observed": [],
  "evidence": [
    {{"source": "idp_auth", "day": 12, "description": "single off-hours auth, no downstream activity"}}
  ],
  "recommended_action": "No action required.",
  "confidence": "medium"
}}

Behavior reference:
  secret_in_commit       — credential or key in a PR description
  unusual_hours_access   — access outside 09:00–18:00
  excessive_repo_cloning — more than 3 repos cloned in one session
  sentiment_drift        — passive-aggressive OR deliberately flat/neutral Slack tone
  cross_dept_snooping    — JIRA reads outside own department
  data_exfil_email       — email to personal address (gmail/yahoo/protonmail)
  host_data_hoarding     — 3-phase trail: bulk_file_copy → archive_creation → archive_move
                           Use hoarding_trail_start_day to link phase 3 back to phase 1.
  social_engineering     — impersonation email, Slack pretext, or phone_call followed
                           within 30 min by idp_auth on a DIFFERENT actor's account
                           (vishing: attacker on the call, victim authenticates)
  idp_anomaly            — unrecognised device, residential IP, or off-hours auth
                           with no corroborating downstream activity

Key rules:
  - Off-hours access alone: NOT sufficient for likely_threat.
  - External email to personal domain: high severity signal.
  - host_data_hoarding phase 3 + data_exfil_email within 3 days = strong exfil case.
  - Vishing: the ATTACKER is the social_engineering subject, not the victim.
  - Most escalations are innocent. Only flag when multi-signal evidence is compelling.

Return ONE verdict per flagged employee.
Return ONLY a JSON array (no other text):
[{{
  "name": "<employee>",
  "verdict": "innocent|suspicious|likely_threat",
  "threat_class": "negligent|disgruntled|malicious|unknown",
  "behaviors_observed": ["<behavior>", ...],
  "evidence": [{{"source": "<type>", "day": <n>, "description": "<observation>"}}],
  "recommended_action": "<one sentence>",
  "confidence": "low|medium|high"
}}]"""

_CORRELATION_USER_V3 = """\
Flagged employees: {flagged}

Full telemetry timeline:
{telemetry}
{slack_section}
Build your case and return verdict objects."""

# ── Variant registry ──────────────────────────────────────────────────────────
# Maps variant key → prompt triple (baseline_system, triage_system, correlation_system)
# and the corresponding user-message templates.

_SENSITIVITY_VARIANTS: Dict[str, dict] = {
    "v2_natural": {
        "label": "Natural language — no explicit rule list or taxonomy headers",
        "baseline_system": _BASELINE_SYSTEM_V2,
        "baseline_user": _BASELINE_USER_V2,
        "triage_system": _TRIAGE_SYSTEM_V2,
        "triage_user": _TRIAGE_USER_V2,
        "correlation_system": _CORRELATION_SYSTEM_V2,
        "correlation_user": _CORRELATION_USER_V2,
    },
    "v3_examples_first": {
        "label": "Examples-first — worked positive/negative examples before rules",
        "baseline_system": _BASELINE_SYSTEM_V3,
        "baseline_user": _BASELINE_USER_V3,
        "triage_system": _TRIAGE_SYSTEM_V3,
        "triage_user": _TRIAGE_USER_V3,
        "correlation_system": _CORRELATION_SYSTEM_V3,
        "correlation_user": _CORRELATION_USER_V3,
    },
}

# ── Prompt selector ───────────────────────────────────────────────────────────


def resolve_prompts(variant: str) -> dict:
    """
    Return the prompt set for the given variant key.
    Raises ValueError for unrecognised variants.

    'official' (or None) always returns the canonical production prompts
    and is the only variant that writes to the leaderboard.
    """
    if variant in (OFFICIAL_VARIANT, None):
        return {
            "label": "Official (production)",
            "baseline_system": _BASELINE_SYSTEM_OFFICIAL,
            "baseline_user": _BASELINE_USER_OFFICIAL,
            "triage_system": _TRIAGE_SYSTEM_OFFICIAL,
            "triage_user": _TRIAGE_USER_OFFICIAL,
            "correlation_system": _CORRELATION_SYSTEM_OFFICIAL,
            "correlation_user": _CORRELATION_USER_OFFICIAL,
        }
    if variant not in _SENSITIVITY_VARIANTS:
        raise ValueError(
            f"Unknown prompt variant: {variant!r}. "
            f"Available: {list(_SENSITIVITY_VARIANTS.keys())}"
        )
    return _SENSITIVITY_VARIANTS[variant]


def is_sensitivity_run(variant: str) -> bool:
    """True for any variant that must NOT touch the leaderboard."""
    return variant not in (OFFICIAL_VARIANT, None)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — BASELINE CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────


def run_baseline(
    llm: BedrockLLM,
    baseline_path: Path,
    prompts: dict,
) -> List[dict]:
    print("\n" + "═" * 64)
    print("  STAGE 1 — BASELINE CALIBRATION")
    print("═" * 64)

    records = load_jsonl(baseline_path)
    if not records:
        print(f"  WARNING: No baseline telemetry at {baseline_path}")
        return []

    day_min = min(r.get("day", 0) for r in records)
    day_max = max(r.get("day", 0) for r in records)
    print(f"  {len(records):,} baseline records  (days {day_min}–{day_max})")

    # Collapse raw records into a compact per-actor profile.
    # Preserves all statistical signals while reducing token load.
    summary = _summarise_baseline(records)

    result = llm.call_json(
        prompts["baseline_system"],
        prompts["baseline_user"].format(telemetry=summary),
        max_tokens=1024,
    )

    if result is None:
        print("  WARNING: Unparseable response from baseline agent")
        return []
    if isinstance(result, dict):
        result = [result]
    if not isinstance(result, list):
        result = []

    print(f"  Flagged {len(result)} potential FP actors in clean period:")
    for r in result:
        print(
            f"    • {r.get('actor', '?')} — {r.get('anomaly_type', '?')} "
            f"[{r.get('fp_likelihood', '?')}]"
        )

    return result


def _summarise_baseline(records: List[dict]) -> str:
    """
    Collapse raw baseline JSONL into a compact per-actor profile.
    Produces a fraction of the tokens while preserving the statistical
    signals the baseline agent needs to flag anomalies.
    """
    from collections import defaultdict

    actors: Dict[str, dict] = defaultdict(
        lambda: {
            "record_types": defaultdict(int),
            "off_hours_count": 0,
            "new_device_count": 0,
            "anomalous_ip_count": 0,
            "external_email_count": 0,
            "days_active": set(),
            "apps_accessed": set(),
            "channels": set(),
        }
    )

    for r in records:
        actor = r.get("actor", "unknown")
        rt = r.get("record_type", "unknown")
        a = actors[actor]

        a["record_types"][rt] += 1
        a["days_active"].add(r.get("day", 0))

        if (
            r.get("outside_business_hours")
            or r.get("access_hour", 12) < 8
            or r.get("access_hour", 12) > 18
        ):
            a["off_hours_count"] += 1
        if r.get("new_device"):
            a["new_device_count"] += 1
        if r.get("anomalous_ip"):
            a["anomalous_ip_count"] += 1
        if r.get("is_external"):
            a["external_email_count"] += 1
        if r.get("dst_app"):
            a["apps_accessed"].add(r["dst_app"])
        if r.get("channel"):
            a["channels"].add(r["channel"])

    lines = ["Per-actor baseline profile (clean period — no threats active):\n"]
    for actor, stats in sorted(actors.items()):
        type_summary = ", ".join(
            f"{k}:{v}" for k, v in sorted(stats["record_types"].items())
        )
        lines.append(
            f"  {actor}:"
            f"  days_active={len(stats['days_active'])}"
            f"  {type_summary}"
            f"  off_hours={stats['off_hours_count']}"
            f"  new_device={stats['new_device_count']}"
            f"  anomalous_ip={stats['anomalous_ip_count']}"
            f"  external_email={stats['external_email_count']}"
            f"  apps={len(stats['apps_accessed'])}"
        )

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — TRIAGE (7-day sliding windows)
# ─────────────────────────────────────────────────────────────────────────────


def run_triage(
    llm: BedrockLLM,
    access_log: List[dict],
    company: str,
    employees: str,
    max_day: int,
    export_dir: Path,
    prompts: dict,
    window_size: int = 7,
) -> Dict[str, List[dict]]:
    print("\n" + "═" * 64)
    print("  STAGE 2 — TRIAGE  (7-day sliding windows)")
    print("═" * 64)

    system = prompts["triage_system"].replace("{company}", company)
    all_decisions: Dict[str, List[dict]] = defaultdict(list)

    if not access_log:
        print("  WARNING: access_log is empty")
        return dict(all_decisions)

    for window_start in range(1, max_day + 1, window_size):
        window_end = min(window_start + window_size - 1, max_day)
        window_records = records_in_window(access_log, window_start, window_end)

        if not window_records:
            continue

        window_actors = list({r.get("actor") for r in window_records if r.get("actor")})

        # Read Slack directly from export for tone context
        slack_msgs = read_slack_for_actors(export_dir, window_actors, limit_per_actor=8)
        slack_section = ""
        if slack_msgs:
            slack_section = (
                "\n\nSlack messages for actors in this window "
                "(tone is a signal — passive-aggressive language matters):\n"
                + format_slack(slack_msgs)
            )

        print(
            f"  Days {window_start}–{window_end}: "
            f"{len(window_records)} records, "
            f"{len(slack_msgs)} Slack msgs, "
            f"actors: {window_actors}"
        )

        user = prompts["triage_user"].format(
            start=window_start,
            end=window_end,
            telemetry=jsonl_str(window_records),
            employees=employees,
            slack_section=slack_section,
            company=company,
        )

        decisions = llm.call_json(system, user, max_tokens=4096)

        if decisions is None:
            print(
                f"    WARNING: Unparseable triage response for window {window_start}–{window_end}"
            )
            continue
        if isinstance(decisions, dict):
            decisions = [decisions]
        if not isinstance(decisions, list):
            continue

        for d in decisions:
            actor = d.get("actor", "")
            if not actor:
                continue
            all_decisions[actor].append(d)
            icon = "🔴" if d.get("escalate") else "⚪"
            print(
                f"    {icon} {actor:<18} {d.get('reason', '')[:60]} "
                f"[{d.get('confidence', '')}]"
            )

    return dict(all_decisions)


def get_escalated(triage: Dict[str, List[dict]]) -> List[str]:
    return [actor for actor, ds in triage.items() if any(d.get("escalate") for d in ds)]


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2b — CREDENTIAL SCAN (parallel single-surface path for negligent class)
# ─────────────────────────────────────────────────────────────────────────────
#
# The main triage stage requires 2+ distinct signal types in a 7-day window
# before escalating. This threshold is well-calibrated for disgruntled and
# malicious subjects but is architecturally exclusionary for the negligent
# insider, whose entire threat profile may be a single high-confidence event
# on a single surface (e.g. secret_in_commit).
#
# This stage runs independently of the sliding-window threshold. It scans all
# commit/PR records for the secret_in_commit field set by the simulation engine
# and auto-escalates any matching actors directly into the correlation stage.
# A single confirmed credential leak is sufficient; no corroborating signal
# is required.
#
# Actors surfaced here are merged with triage escalations before Stage 3.
# The merge is de-duplicated: if triage already escalated the same actor,
# the credential scan result is redundant and produces no double-counting.

_CREDENTIAL_SCAN_SYSTEM = """\
You are a security engineer reviewing pull request and commit records for
accidental credential exposure.

A record has secret_in_commit: true when the simulation engine confirmed
a realistic synthetic credential (AWS key, GitHub token, database password,
etc.) was embedded in the artifact. All such values are synthetically
generated and are not real credentials.

Your task is simple: for every record with secret_in_commit: true, return
the actor name and the day. Do not second-guess the flag — if the engine
set it, treat it as confirmed.

Return ONLY a JSON array (empty [] if no records have secret_in_commit: true):
[{
  "actor": "<employee name>",
  "day": <day number>,
  "record_type": "<commit|pr_description|etc>",
  "confidence": "high"
}]
No other text."""

_CREDENTIAL_SCAN_USER = """\
PR and commit records to scan:
{records}

Return JSON array of confirmed secret_in_commit hits."""


def run_credential_scan(
    llm: BedrockLLM,
    access_log: List[dict],
    prompts: dict,
) -> List[str]:
    """
    Parallel single-surface stage for negligent insider detection.

    Scans commit/PR records for secret_in_commit: true (set deterministically
    by the simulation engine) and returns a list of actor names to escalate
    directly to the correlation stage, bypassing the 2-signal triage threshold.

    Uses its own fixed prompt regardless of the active prompt variant; the
    credential scan is a structural feature of the pipeline, not a prompt
    sensitivity variable.

    Returns a (possibly empty) list of actor names confirmed as credential
    leakers.
    """
    print("\n" + "═" * 64)
    print("  STAGE 2b — CREDENTIAL SCAN  (single-surface negligent path)")
    print("═" * 64)

    # Pull all commit and PR records — these are the only surfaces that carry
    # secret_in_commit events in the current corpus.
    commit_records = [
        r
        for r in access_log
        if r.get("record_type")
        in ("commit", "pr_description", "pr_event", "repo_access")
    ]

    if not commit_records:
        print("  No commit/PR records found — skipping.")
        return []

    # First pass: deterministic pre-filter.
    # The simulation engine sets secret_in_commit: true on confirmed records.
    # If any records carry the flag, we already know the answer — we don't
    # need the LLM for detection, only for generating the correlation verdict.
    # We still run the LLM call for consistency and to produce the evidence
    # object the scoring stage expects.
    engine_confirmed = [r for r in commit_records if r.get("secret_in_commit")]
    if engine_confirmed:
        confirmed_actors = list(
            {r.get("actor") for r in engine_confirmed if r.get("actor")}
        )
        print(
            f"  Engine-confirmed secret_in_commit on {len(engine_confirmed)} record(s): "
            f"{confirmed_actors}"
        )
    else:
        confirmed_actors = []
        print("  No engine-confirmed secret_in_commit records found.")

    # LLM verification pass — runs over ALL commit records, not just flagged
    # ones, so the model has context for its confidence assessment.
    print(
        f"  Sending {len(commit_records)} commit/PR records to credential scan agent..."
    )
    result = llm.call_json(
        _CREDENTIAL_SCAN_SYSTEM,
        _CREDENTIAL_SCAN_USER.format(records=jsonl_str(commit_records)),
        max_tokens=1024,
    )

    llm_actors: List[str] = []
    if isinstance(result, list):
        for hit in result:
            actor = hit.get("actor", "")
            if actor:
                llm_actors.append(actor)
                icon = "🔴"
                print(
                    f"    {icon} {actor:<18} day={hit.get('day', '?')}  "
                    f"type={hit.get('record_type', '?')}  "
                    f"conf={hit.get('confidence', '?')}"
                )
    elif result is None:
        print("  WARNING: Unparseable response from credential scan agent.")

    if not llm_actors and not engine_confirmed:
        print("  No credential leaks detected.")
        return []

    # Merge: engine-confirmed actors take precedence; LLM hits are additive.
    # Both sets should agree on any flagged corpus — disagreement is logged.
    engine_set = set(confirmed_actors)
    llm_set = set(llm_actors)

    if engine_set and llm_set and engine_set != llm_set:
        print(
            f"  NOTE: Engine-confirmed actors {engine_set} differ from "
            f"LLM-detected actors {llm_set}. "
            f"Engine ground truth takes precedence for scoring; "
            f"both sets are escalated for correlation."
        )

    escalate = list(engine_set | llm_set)
    print(f"  Credential scan escalating: {escalate}")
    return escalate


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — CORRELATION (full verdict per flagged actor)
# ─────────────────────────────────────────────────────────────────────────────


def run_correlation(
    llm: BedrockLLM,
    escalated: List[str],
    access_log: List[dict],
    company: str,
    export_dir: Path,
    prompts: dict,
) -> List[dict]:
    print("\n" + "═" * 64)
    print("  STAGE 3 — CORRELATION")
    print("═" * 64)

    if not escalated:
        print("  No escalated actors — skipping.")
        return []

    print(f"  Investigating: {escalated}")
    verdicts = []

    for actor in escalated:
        print(f"\n  → {actor}")

        # Telemetry scoped to this actor only
        relevant = [r for r in access_log if r.get("actor") == actor]

        # Also pull phone_call records where this actor made the call,
        # and any vishing auth records filed under other actors that
        # originated from a call by this actor
        phone_calls = [
            r
            for r in access_log
            if r.get("record_type") == "phone_call" and r.get("actor") != actor
            # include calls made by others only if they precede an
            # idp_auth on this actor's account (they may be the victim)
        ]
        vishing_on_actor = [
            r
            for r in access_log
            if r.get("record_type") == "idp_auth"
            and r.get("actor") == actor
            and r.get("preceded_by_call_record")
        ]
        # If this actor has vishing auths, include the phone_call records
        # from that day so the agent can see who made the call
        if vishing_on_actor:
            vishing_days = {r.get("day") for r in vishing_on_actor}
            extra_calls = [
                r
                for r in access_log
                if r.get("record_type") == "phone_call" and r.get("day") in vishing_days
            ]
            relevant = relevant + [r for r in extra_calls if r not in relevant]

        print(f"     {len(relevant)} records")

        slack_msgs = read_slack_for_actors(export_dir, [actor], limit_per_actor=15)
        slack_section = ""
        if slack_msgs:
            slack_section = (
                "\n\nSlack messages (tone progression — oldest to newest):\n"
                + format_slack(slack_msgs)
            )

        system = prompts["correlation_system"].replace("{company}", company)
        user = prompts["correlation_user"].format(
            flagged=actor,
            telemetry=jsonl_str(relevant),
            slack_section=slack_section,
        )

        verdict = llm.call_json(system, user, max_tokens=4096)

        if verdict is None:
            print(f"     WARNING: Unparseable response for {actor}")
            continue

        # Model may return a list with one item or a single object
        if isinstance(verdict, list):
            verdicts.extend(verdict)
        elif isinstance(verdict, dict):
            verdicts.append(verdict)

    # Print summary
    print("\n  Verdicts:")
    for v in verdicts:
        icon = (
            "🔴"
            if v.get("verdict") == "likely_threat"
            else "🟡"
            if v.get("verdict") == "suspicious"
            else "🟢"
        )
        print(
            f"    {icon} {v.get('name', '?'):<18} "
            f"{v.get('verdict', '?'):<14} "
            f"class={v.get('threat_class', '?'):<12} "
            f"conf={v.get('confidence', '?')}"
        )
        for b in v.get("behaviors_observed", []):
            print(f"       • {b}")

    return verdicts


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4 — SCORING
# ─────────────────────────────────────────────────────────────────────────────


def score_triage(
    triage: Dict[str, List[dict]],
    ground_truth: Dict[str, set],
    onset_days: Dict[str, int],
    access_log: List[dict],
) -> dict:
    """
    Score triage decisions:
      precision   — of escalated actors, fraction that are actual threats
      recall      — of actual threats, fraction that were escalated
      f1
      baseline_fp_rate  — fraction of innocent employees flagged in any window
      onset_sensitivity — fraction of subjects escalated BEFORE their onset_day
                          (escalating before onset means the model is guessing,
                          not detecting — lower is better)
    """
    threat_actors = set(ground_truth.keys())
    all_actors_seen = set(triage.keys())
    innocent_seen = all_actors_seen - threat_actors

    escalated = set(get_escalated(triage))
    tp = len(escalated & threat_actors)
    fp = len(escalated - threat_actors)
    fn = len(threat_actors - escalated)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    # Baseline FP rate: fraction of innocent actors that were escalated
    baseline_fp_rate = (
        len(escalated & innocent_seen) / len(innocent_seen) if innocent_seen else 0.0
    )

    # Onset sensitivity: did the model escalate a subject in a window that
    # ended BEFORE that subject's onset_day?
    premature = 0
    total_subjects_escalated = 0
    for actor in escalated & threat_actors:
        onset = onset_days.get(actor, 1)
        total_subjects_escalated += 1
        actor_records = [r for r in access_log if r.get("actor") == actor]
        pre_onset_records = [r for r in actor_records if r.get("day", 999) < onset]
        if pre_onset_records and actor in escalated:
            premature += 1

    onset_sensitivity = (
        premature / total_subjects_escalated if total_subjects_escalated else 0.0
    )

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "baseline_fp_rate": round(baseline_fp_rate, 4),
        "onset_sensitivity": round(onset_sensitivity, 4),
    }


def score_verdicts(
    verdicts: List[dict],
    ground_truth: Dict[str, set],
    ground_truth_full: Dict[str, dict],
    access_log: List[dict],
) -> dict:
    """
    Score correlation verdicts:
      precision / recall / f1 at the actor level
      by_class    — per threat class breakdown
      by_behavior — per behavior TP/FP
      vishing_detected          — bool
      host_trail_reconstructed  — bool
    """
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
        # fn handled below

    for actor in threat_actors:
        v = verdict_map.get(actor)
        if v is None or v.get("verdict") not in ("suspicious", "likely_threat"):
            fn += 1
            tc = ground_truth_full.get(actor, {}).get("threat_class", "unknown")
            by_class[tc]["fn"] += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    # Vishing detection: was social_engineering cited AND did the agent
    # describe a phone_call → idp_auth correlation anywhere in evidence?
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

    # Host trail reconstruction: did the agent cite all 3 hoarding phases?
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

    # secret_in_commit detection: did the correlation agent correctly flag a
    # negligent actor whose commit records carry secret_in_commit: true?
    secret_in_commit_detected = False
    commit_leak_records = [r for r in access_log if r.get("secret_in_commit")]
    if commit_leak_records:
        leaking_actors = {r.get("actor") for r in commit_leak_records if r.get("actor")}
        for v in verdicts:
            if (
                v.get("name") in leaking_actors
                and v.get("verdict") in ("suspicious", "likely_threat")
                and "secret_in_commit" in v.get("behaviors_observed", [])
            ):
                secret_in_commit_detected = True
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
        "secret_in_commit_detected": secret_in_commit_detected,
    }


# ─────────────────────────────────────────────────────────────────────────────
# LEADERBOARD  (official runs only)
# ─────────────────────────────────────────────────────────────────────────────


def update_leaderboard(
    run_id: str,
    model: str,
    tier: str,
    triage_scores: dict,
    verdict_scores: Optional[dict],
    cfg: dict,
) -> None:
    leaderboard = []
    if LEADERBOARD_JSON.exists():
        leaderboard = json.loads(LEADERBOARD_JSON.read_text())

    subjects = get_subjects(cfg)
    row = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "tier": tier,
        "sim_days": get_max_day(cfg),
        "subjects": len(subjects),
        "subject_classes": [s.get("threat_class") for s in subjects],
        # Tier 1 — triage
        "triage_precision": triage_scores["precision"],
        "triage_recall": triage_scores["recall"],
        "triage_f1": triage_scores["f1"],
        "triage_tp": triage_scores["tp"],
        "triage_fp": triage_scores["fp"],
        "triage_fn": triage_scores["fn"],
        "baseline_fp_rate": triage_scores["baseline_fp_rate"],
        "onset_sensitivity": triage_scores["onset_sensitivity"],
        # Tier 2 — verdict (None for triage-only runs)
        "verdict_precision": verdict_scores["precision"] if verdict_scores else None,
        "verdict_recall": verdict_scores["recall"] if verdict_scores else None,
        "verdict_f1": verdict_scores["f1"] if verdict_scores else None,
        "verdict_tp": verdict_scores["tp"] if verdict_scores else None,
        "verdict_fp": verdict_scores["fp"] if verdict_scores else None,
        "verdict_fn": verdict_scores["fn"] if verdict_scores else None,
        "vishing_detected": verdict_scores["vishing_detected"]
        if verdict_scores
        else None,
        "host_trail_reconstructed": verdict_scores["host_trail_reconstructed"]
        if verdict_scores
        else None,
        "secret_in_commit_detected": verdict_scores["secret_in_commit_detected"]
        if verdict_scores
        else None,
        # Per-behavior and per-class (Tier 2 only)
        "by_behavior": verdict_scores["by_behavior"] if verdict_scores else {},
        "by_class": verdict_scores["by_class"] if verdict_scores else {},
    }

    # Replace existing run with same id, else append
    leaderboard = [r for r in leaderboard if r.get("run_id") != run_id]
    leaderboard.append(row)

    # Sort: Tier 2 rows first, then by verdict_f1 desc, then triage_f1 desc
    leaderboard.sort(
        key=lambda r: (
            0 if r.get("tier") == "2" else 1,
            -(r.get("verdict_f1") or 0.0),
            -(r.get("triage_f1") or 0.0),
            r.get("baseline_fp_rate") or 1.0,  # lower FP rate = better
        )
    )

    LEADERBOARD_JSON.write_text(json.dumps(leaderboard, indent=2))
    logger.info(f"Leaderboard JSON updated: {LEADERBOARD_JSON}")

    _write_leaderboard_csv(leaderboard)


def _flatten_row(row: dict) -> dict:
    flat = {
        "run_id": row.get("run_id", ""),
        "timestamp": row.get("timestamp", ""),
        "model": row.get("model", ""),
        "tier": row.get("tier", ""),
        "sim_days": row.get("sim_days", ""),
        "subjects": row.get("subjects", ""),
        # Tier 1
        "triage_precision": row.get("triage_precision", ""),
        "triage_recall": row.get("triage_recall", ""),
        "triage_f1": row.get("triage_f1", ""),
        "baseline_fp_rate": row.get("baseline_fp_rate", ""),
        "onset_sensitivity": row.get("onset_sensitivity", ""),
        # Tier 2
        "verdict_precision": row.get("verdict_precision", ""),
        "verdict_recall": row.get("verdict_recall", ""),
        "verdict_f1": row.get("verdict_f1", ""),
        "vishing_detected": row.get("vishing_detected", ""),
        "host_trail_reconstructed": row.get("host_trail_reconstructed", ""),
        "secret_in_commit_detected": row.get("secret_in_commit_detected", ""),
    }
    # Per-behavior TP columns
    by_behavior = row.get("by_behavior", {})
    for b in _ALL_BEHAVIORS:
        flat[f"tp_{b}"] = by_behavior.get(b, {}).get("tp", "")
        flat[f"fp_{b}"] = by_behavior.get(b, {}).get("fp", "")
    # Per-class columns
    by_class = row.get("by_class", {})
    for c in _ALL_CLASSES:
        for metric in ("tp", "fp", "fn"):
            flat[f"{c}_{metric}"] = by_class.get(c, {}).get(metric, "")
    return flat


def _write_leaderboard_csv(leaderboard: List[dict]) -> None:
    if not leaderboard:
        return

    all_keys: list = []
    seen: set = set()
    for row in leaderboard:
        for k in _flatten_row(row):
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    with open(LEADERBOARD_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for row in leaderboard:
            writer.writerow(_flatten_row(row))

    logger.info(f"Leaderboard CSV updated: {LEADERBOARD_CSV}")


# ─────────────────────────────────────────────────────────────────────────────
# PRINT SUMMARY
# ─────────────────────────────────────────────────────────────────────────────


def print_summary(
    model: str,
    variant: str,
    triage_scores: dict,
    verdict_scores: Optional[dict],
    ground_truth_full: Dict[str, dict],
    verdicts: List[dict],
) -> None:
    def _fmt(v):
        if v is None:
            return "  n/a "
        if isinstance(v, bool):
            return "  yes " if v else "  no  "
        return f"{v:.4f}"

    print("\n" + "═" * 64)
    print(f"  RESULTS — {model}")
    if variant != OFFICIAL_VARIANT:
        print(f"  PROMPT VARIANT — {variant}  [sensitivity run, not on leaderboard]")
    print("═" * 64)

    print("\n  Triage (Tier 1)")
    print(f"    {'Precision':<22} {_fmt(triage_scores['precision'])}")
    print(f"    {'Recall':<22} {_fmt(triage_scores['recall'])}")
    print(f"    {'F1':<22} {_fmt(triage_scores['f1'])}")
    print(
        f"    {'TP/FP/FN':<22} {triage_scores['tp']}/{triage_scores['fp']}/{triage_scores['fn']}"
    )
    print(
        f"    {'Baseline FP rate':<22} {_fmt(triage_scores['baseline_fp_rate'])}  ← lower is better"
    )
    print(
        f"    {'Onset sensitivity':<22} {_fmt(triage_scores['onset_sensitivity'])}  ← lower is better"
    )

    if verdict_scores:
        print("\n  Verdicts (Tier 2)")
        print(f"    {'Precision':<22} {_fmt(verdict_scores['precision'])}")
        print(f"    {'Recall':<22} {_fmt(verdict_scores['recall'])}")
        print(f"    {'F1':<22} {_fmt(verdict_scores['f1'])}")
        print(
            f"    {'TP/FP/FN':<22} {verdict_scores['tp']}/{verdict_scores['fp']}/{verdict_scores['fn']}"
        )
        print(
            f"    {'Vishing detected':<22} {_fmt(verdict_scores['vishing_detected'])}"
        )
        print(
            f"    {'Host trail (3 phases)':<22} {_fmt(verdict_scores['host_trail_reconstructed'])}"
        )
        print(
            f"    {'Secret in commit':<22} {_fmt(verdict_scores['secret_in_commit_detected'])}"
        )

        if verdict_scores["by_class"]:
            print("\n  Per threat class:")
            for cls, counts in sorted(verdict_scores["by_class"].items()):
                tp_ = counts.get("tp", 0)
                fp_ = counts.get("fp", 0)
                fn_ = counts.get("fn", 0)
                p = tp_ / max(tp_ + fp_, 1)
                r = tp_ / max(tp_ + fn_, 1)
                print(
                    f"    {cls:<16}  TP={tp_}  FP={fp_}  FN={fn_}  P={p:.2f}  R={r:.2f}"
                )

        if verdict_scores["by_behavior"]:
            print("\n  Per behavior (correctly cited):")
            for b, counts in sorted(verdict_scores["by_behavior"].items()):
                print(f"    {b:<30}  TP={counts['tp']}  FP={counts['fp']}")

    print(f"\n  Ground truth actors: {list(ground_truth_full.keys())}")
    detected = [
        v["name"]
        for v in verdicts
        if v.get("verdict") in ("suspicious", "likely_threat")
    ]
    print(f"  Detected as threat:  {detected}")
    print("═" * 64 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────


def run_eval(args: argparse.Namespace) -> None:
    export_dir = Path(args.export_dir).resolve()
    config_path = Path(args.config).resolve()

    if not config_path.exists():
        print(f"ERROR: config not found at {config_path}", file=sys.stderr)
        sys.exit(1)

    cfg = load_config(config_path)
    it_cfg = cfg.get("insider_threat", {})

    # Validate log format — this script requires JSONL
    log_format = it_cfg.get("log_format", "jsonl")
    if log_format not in ("jsonl", "all"):
        print(
            f"ERROR: log_format is '{log_format}' in config.yaml. "
            f"eval_insider_threat.py requires log_format: jsonl or all.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Resolve prompt variant ────────────────────────────────────────────────
    variant = args.prompt_variant or OFFICIAL_VARIANT
    sensitivity = is_sensitivity_run(variant)
    try:
        prompts = resolve_prompts(variant)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    company = get_company_name(cfg)
    employees = get_employee_list(cfg)
    max_day = get_max_day(cfg)
    onset_days = get_onset_days(cfg)
    subject_names = get_subject_names(cfg)

    telemetry_dir = export_dir / it_cfg.get("telemetry_dir", "security_telemetry")
    access_log_path = telemetry_dir / "access_log.jsonl"
    gt_path = telemetry_dir / "_ground_truth.jsonl"
    baseline_path = telemetry_dir / "baseline_telemetry.jsonl"

    print(
        f"\nOrgForge Insider Threat Detection{'  [SENSITIVITY RUN]' if sensitivity else ''}"
    )
    print(f"  Company       : {company}")
    print(f"  Model         : {args.model}")
    print(
        f"  Prompt variant: {variant}{' — ' + prompts['label'] if sensitivity else ''}"
    )
    print(f"  Region        : {args.region}")
    print(f"  Export dir    : {export_dir}")
    print(f"  Max days      : {max_day}")
    print(f"  Subjects      : {onset_days}")
    if sensitivity:
        print("  ⚠  Leaderboard will NOT be updated for sensitivity runs.")

    access_log = load_jsonl(access_log_path)
    ground_truth = load_ground_truth(gt_path)
    ground_truth_full = load_ground_truth_full(gt_path)

    print(f"\n  Telemetry records  : {len(access_log)}")
    print(f"  Ground truth actors: {list(ground_truth_full.keys())}")

    if not access_log:
        print(f"\nERROR: No telemetry at {access_log_path}", file=sys.stderr)
        sys.exit(1)

    if not ground_truth:
        print(f"\nERROR: No ground truth at {gt_path}", file=sys.stderr)
        sys.exit(1)

    # ── Run ID and output directory ───────────────────────────────────────────
    safe_model = re.sub(r"[^a-zA-Z0-9._-]", "_", args.model)[:48]
    run_id = f"{safe_model}__{variant}__{datetime.now().strftime('%Y%m%dT%H%M%S')}"

    if sensitivity:
        run_dir = SENSITIVITY_DIR / run_id
    else:
        run_dir = RESULTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    llm = BedrockLLM(
        model=args.model,
        region=args.region,
        call_delay=args.call_delay,
    )

    # ── Validate flag combinations ────────────────────────────────────────────
    if args.correlation_only and args.triage_only:
        print(
            "ERROR: --correlation-only and --triage-only are mutually exclusive.",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.correlation_only and not args.actors:
        print(
            "ERROR: --correlation-only requires --actors <name> [<name> ...].",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Stage 1: Baseline ─────────────────────────────────────────────────────
    baseline_fps: List[dict] = []
    if args.correlation_only:
        print("\n  [Baseline stage skipped — correlation-only run]")
    elif not args.skip_baseline:
        baseline_fps = run_baseline(llm, baseline_path, prompts)
    else:
        print("\n  [Baseline stage skipped]")

    # ── Stage 2: Triage ───────────────────────────────────────────────────────
    triage_decisions: Dict[str, List[dict]] = {}
    triage_scores: dict

    if args.correlation_only:
        print("\n  [Triage stage skipped — correlation-only run]")
        # Optionally reload triage decisions from a previous run so scores
        # are preserved rather than recorded as null.
        if args.resume_run:
            resume_dir = Path(args.resume_run).resolve()
            prev_triage_path = resume_dir / "triage_decisions.json"
            if prev_triage_path.exists():
                triage_decisions = json.load(open(prev_triage_path))
                print(f"  Loaded triage decisions from {prev_triage_path}")
            else:
                print(f"  WARNING: --resume-run set but {prev_triage_path} not found.")
        triage_scores = score_triage(
            triage_decisions, ground_truth, onset_days, access_log
        )
    else:
        triage_decisions = run_triage(
            llm,
            access_log,
            company,
            employees,
            max_day,
            export_dir,
            prompts,
            args.window_size,
        )
        triage_scores = score_triage(
            triage_decisions, ground_truth, onset_days, access_log
        )

    # ── Stage 2b: Credential scan ─────────────────────────────────────────────
    credential_escalated: List[str] = []
    if args.correlation_only:
        print("\n  [Credential scan skipped — correlation-only run]")
    elif not args.skip_credential_scan and not args.triage_only:
        credential_escalated = run_credential_scan(llm, access_log, prompts)
    elif args.skip_credential_scan:
        print("\n  [Credential scan stage skipped]")

    # ── Resolve escalated actor list ──────────────────────────────────────────
    if args.correlation_only:
        # Caller supplies actors explicitly — no triage or scan needed.
        all_escalated = list(args.actors)
        print(f"\n  Actors (from --actors): {all_escalated}")
    else:
        escalated = get_escalated(triage_decisions)
        print(f"\n  Escalated (triage): {escalated}")
        all_escalated = list(
            dict.fromkeys(
                escalated + [a for a in credential_escalated if a not in escalated]
            )
        )
        if credential_escalated:
            new_from_scan = [a for a in credential_escalated if a not in escalated]
            if new_from_scan:
                print(f"\n  Credential scan added new escalations: {new_from_scan}")
        print(f"\n  Escalated (all stages): {all_escalated}")

    triage_scores = score_triage(triage_decisions, ground_truth, onset_days, access_log)

    # ── Stage 3: Correlation ──────────────────────────────────────────────────
    verdicts: List[dict] = []
    verdict_scores: Optional[dict] = None

    if not args.triage_only:
        verdicts = run_correlation(
            llm, all_escalated, access_log, company, export_dir, prompts
        )
        verdict_scores = score_verdicts(
            verdicts, ground_truth, ground_truth_full, access_log
        )
    else:
        print("\n  [Correlation stage skipped — triage-only run]")

    # ── Write per-run results ─────────────────────────────────────────────────
    with open(run_dir / "triage_decisions.json", "w") as f:
        json.dump(triage_decisions, f, indent=2, default=str)
    with open(run_dir / "verdicts.json", "w") as f:
        json.dump(verdicts, f, indent=2, default=str)
    with open(run_dir / "scores.json", "w") as f:
        json.dump(
            {
                "triage": triage_scores,
                "verdicts": verdict_scores,
                "baseline_false_positives": baseline_fps,
                "prompt_variant": variant,
                "prompt_label": prompts["label"],
                "correlation_only": args.correlation_only,
                "correlation_only_actors": args.actors
                if args.correlation_only
                else None,
                "ground_truth": {
                    actor: {
                        "behaviors": list(info["behaviors"]),
                        "threat_class": info["threat_class"],
                    }
                    for actor, info in ground_truth_full.items()
                },
            },
            f,
            indent=2,
            default=str,
        )

    # Sensitivity runs get an extra file recording exactly which prompt was used.
    if sensitivity:
        with open(run_dir / "prompt_variant.txt", "w") as f:
            f.write(f"variant: {variant}\n")
            f.write(f"label: {prompts['label']}\n\n")
            f.write("=== BASELINE SYSTEM ===\n")
            f.write(prompts["baseline_system"] + "\n\n")
            f.write("=== TRIAGE SYSTEM ===\n")
            f.write(prompts["triage_system"] + "\n\n")
            f.write("=== CORRELATION SYSTEM ===\n")
            f.write(prompts["correlation_system"] + "\n")

    print(f"\n  Run artifacts → {run_dir}")

    # ── Update leaderboard (official runs only) ───────────────────────────────
    # Correlation-only runs are excluded: triage scores are null or borrowed
    # from a prior run, so the leaderboard row would be misleading.
    if not sensitivity and not args.correlation_only:
        tier = "1" if args.triage_only else "2"
        update_leaderboard(run_id, args.model, tier, triage_scores, verdict_scores, cfg)
        print(f"  Leaderboard   → {LEADERBOARD_JSON}")
        print(f"  Leaderboard   → {LEADERBOARD_CSV}")
    elif args.correlation_only:
        print("  Leaderboard      skipped (correlation-only run)")
    else:
        print("  Leaderboard      skipped (sensitivity run)")

    # ── Print summary ─────────────────────────────────────────────────────────
    print_summary(
        args.model, variant, triage_scores, verdict_scores, ground_truth_full, verdicts
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    sensitivity_choices = [OFFICIAL_VARIANT] + list(_SENSITIVITY_VARIANTS.keys())

    p = argparse.ArgumentParser(
        description="OrgForge insider threat detection leaderboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Suggested models:\n"
            + "\n".join(f"  {m}" for m in _SUGGESTED_MODELS)
            + "\n\nPrompt variants (sensitivity analysis):\n"
            + "\n".join(
                f"  {k}  — {v['label']}" for k, v in _SENSITIVITY_VARIANTS.items()
            )
            + "\n\nSensitivity runs write to results/sensitivity/ and never touch the leaderboard."
        ),
    )
    p.add_argument(
        "--model",
        required=True,
        help=(
            "Bedrock model ID. Examples:\n"
            "  anthropic.claude-opus-4-6-20260401-v1:0\n"
            "  anthropic.claude-sonnet-4-6-20260401-v1:0\n"
            "  meta.llama3-3-70b-instruct-v1:0\n"
            "  amazon.nova-pro-v1:0\n"
            "  mistral.mistral-large-2402-v1:0\n"
            "Cross-region inference profiles (us.anthropic.*) also accepted."
        ),
    )
    p.add_argument(
        "--prompt-variant",
        default=OFFICIAL_VARIANT,
        choices=sensitivity_choices,
        metavar="VARIANT",
        help=(
            f"Prompt variant to use. '{OFFICIAL_VARIANT}' (default) uses the canonical "
            f"production prompts and writes to the leaderboard. "
            f"Any other variant writes to results/sensitivity/ only and never "
            f"touches the leaderboard. Choices: {sensitivity_choices}"
        ),
    )
    p.add_argument(
        "--region",
        default="us-east-2",
        help="AWS region for Bedrock (default: us-east-2)",
    )
    p.add_argument(
        "--export-dir",
        default="./export",
        help="Path to OrgForge export directory (default: ./export)",
    )
    p.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config.yaml (default: config/config.yaml)",
    )
    p.add_argument(
        "--window-size",
        type=int,
        default=7,
        help="Triage sliding window size in days (default: 7)",
    )
    p.add_argument(
        "--call-delay",
        type=float,
        default=1.0,
        help=(
            "Seconds to sleep between Bedrock calls (default: 1.0). "
            "Increase to 2–3 for Opus or if you hit ThrottlingException."
        ),
    )
    p.add_argument(
        "--triage-only",
        action="store_true",
        help="Run triage stage only (Tier 1). Skips correlation and verdict scoring.",
    )
    p.add_argument(
        "--correlation-only",
        action="store_true",
        help=(
            "Skip baseline, triage, and credential scan. Run correlation (Stage 3) "
            "directly against the actors supplied via --actors. "
            "Loads existing triage_decisions.json from --resume-run if provided, "
            "otherwise triage scores are recorded as null. "
            "Useful for rescoring secret_in_commit without re-running triage."
        ),
    )
    p.add_argument(
        "--actors",
        nargs="+",
        metavar="NAME",
        default=None,
        help=(
            "Space-separated list of actor names to pass directly to the correlation "
            "stage. Required when --correlation-only is set. "
            "Example: --actors Jordan"
        ),
    )
    p.add_argument(
        "--resume-run",
        default=None,
        metavar="RUN_DIR",
        help=(
            "Path to an existing run directory. When set with --correlation-only, "
            "triage_decisions.json from that run is loaded so triage scores are "
            "preserved in the output rather than recorded as null."
        ),
    )
    p.add_argument(
        "--skip-credential-scan",
        action="store_true",
        help=(
            "Skip the parallel credential scan stage (Stage 2b). "
            "This stage detects negligent insiders via single-surface secret_in_commit "
            "events that fall below the 2-signal triage threshold. "
            "Use this flag to reproduce pre-Stage-2b leaderboard results."
        ),
    )
    p.add_argument(
        "--skip-baseline",
        action="store_true",
        help=(
            "Skip baseline calibration stage. "
            "Use if baseline_telemetry.jsonl has not been built yet."
        ),
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Override results output directory (default: results/insider_threat/<run_id>/)",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Debug logging",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    run_eval(args)
