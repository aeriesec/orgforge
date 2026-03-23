"""
insider_threat.py
=================
Optional security simulation layer for OrgForge.

Injects realistic insider threat behaviors into the normal simulation flow
so security teams can generate labeled training corpora for detection agents.

The module is COMPLETELY INERT unless ``insider_threat.enabled: true`` is
set in config.yaml.  When disabled, every public entry-point is a no-op and
no objects are constructed.

Design principles
-----------------
* **Behaviors, not labels.**  No artifact ever contains the word "malicious".
  The subject is a normal employee whose *outputs* happen to be anomalous.
  Detection agents must earn the signal through correlation.

* **Surface reuse.**  Every artifact produced (PRs, Slack messages, emails,
  JIRA access records) is generated via the existing OrgForge artifact
  pipeline.  This module only *influences content* at injection points —
  it never bypasses the normal event machinery.

* **Temporal onset.**  Subjects are behaviorally normal before ``onset_day``.
  Behavioral data from days 1 → (onset_day − 1) is clean negative examples.

* **Noise injection.**  The ``dlp_noise_ratio`` fires synthetic DLP/SIEM
  alerts for innocent employees, training agents not to over-index on single
  signals.

* **Ground truth separation.**  The ``security_telemetry/`` export directory
  contains machine-readable ground truth, but the subject's name and
  ``true_positive`` flag are buried in a separate ``_ground_truth.jsonl``
  file that is structurally distinct from the observable telemetry stream.

Config schema (add to config.yaml)
-----------------------------------
  insider_threat:
    enabled: false

    mode: "passive"
    # passive — behaviors injected into artifacts; no synthetic SIEM events
    # active  — additionally emits dlp_alert SimEvents with noise mixed in

    # Export format for security_telemetry/access_log files.
    # "jsonl"  — custom JSONL (original, default for backward compat)
    # "cef"    — Common Event Format (ArcSight, Splunk, many SIEMs)
    # "ecs"    — Elastic Common Schema (Elastic SIEM, OpenSearch)
    # "leef"   — Log Event Extended Format (IBM QRadar)
    # "all"    — write all three alongside each other
    log_format: "jsonl"

    subjects:
      - name: "Jordan"
        threat_class: "negligent"
        # negligent   — accidental credential leak in a PR / commit
        # disgruntled — data hoarding, sentiment drift, reduced collaboration
        # malicious   — deliberate exfil via email/Slack to external contact
        onset_day: 8
        behaviors:
          - "secret_in_commit"       # available for: negligent, malicious
          - "unusual_hours_access"   # available for: malicious, disgruntled
          - "excessive_repo_cloning" # available for: malicious
          - "sentiment_drift"        # available for: disgruntled
          - "cross_dept_snooping"    # available for: malicious, disgruntled
          - "data_exfil_email"       # available for: malicious
          - "host_data_hoarding"     # available for: malicious, disgruntled

    dlp_noise_ratio: 0.4
    # Fraction of dlp_alert events that are false positives (innocent employees).
    # Only relevant in "active" mode.  Range 0.0–1.0.

    telemetry_dir: "security_telemetry"
    # Subdirectory under the simulation export dir for telemetry output.

    idp_logs: true
    # When true, emit realistic IDP (Identity Provider) authentication events
    # for every active employee each day.  These serve as the authentication
    # baseline that anomaly detection agents correlate against behavioral signals.
    # Anomalous IDP events are injected for threat subjects automatically.

Public API (called from flow.py)
---------------------------------
  injector = InsiderThreatInjector.from_config(config, export_base, all_names)

  # Top of daily_cycle — decides whether today is an active threat day
  injector.begin_day(day, state)

  # After org_plan is built — may mutate agenda items for the subject
  injector.inject_pre_planning(day, org_day_plan)

  # Called by GitSimulator.create_pr — may mutate PR description in-place
  injector.inject_pr(pr: dict, author: str, day: int) -> dict

  # Called after Slack message generation — may inject anomalous messages
  injector.inject_slack(messages: list, channel: str, day: int) -> list

  # Called after email artifact write — may inject data exfil payload
  injector.inject_email(eml_path: str, sender: str, day: int)

  # End of day — flush telemetry, maybe emit dlp_alert SimEvent
  injector.end_day(day, state, mem, clock, date_str) -> list[SimEvent]

  # Always-safe check: is behavior X active for subject Y today?
  injector.is_active(name: str, behavior: str, day: int) -> bool
"""

from __future__ import annotations

import json
import logging
import os
import random
import string
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from config_loader import COMPANY_DESCRIPTION, COMPANY_NAME

logger = logging.getLogger("orgforge.security")


# ─────────────────────────────────────────────────────────────────────────────
# FAKE SECRET TEMPLATES
# Realistic-looking but obviously synthetic — safe for training corpora.
# Each template has a variable region generated at runtime.
# ─────────────────────────────────────────────────────────────────────────────


def _rand_upper(n: int) -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=n))


def _rand_hex(n: int) -> str:
    return "".join(random.choices("0123456789abcdef", k=n))


def _rand_b64ish(n: int) -> str:
    chars = string.ascii_letters + string.digits + "+/"
    return "".join(random.choices(chars, k=n))


# Each entry: (env_var_name, value_generator, inline_comment)
_SECRET_TEMPLATES = [
    (
        "AWS_SECRET_ACCESS_KEY",
        lambda: f"AKIA{_rand_upper(16)}",
        "# TODO: move to secrets manager",
    ),
    (
        "GITHUB_TOKEN",
        lambda: f"ghp_{_rand_b64ish(36)}",
        "# temp token for CI pipeline",
    ),
    (
        "DB_PASSWORD",
        lambda: f"prod_{_rand_hex(12)}_pwd",
        "# staging db — rotate before merge",
    ),
    (
        "STRIPE_SECRET_KEY",
        lambda: f"sk_live_{_rand_b64ish(24)}",
        "# live key — do NOT commit",
    ),
    (
        "SENDGRID_API_KEY",
        lambda: f"SG.{_rand_b64ish(22)}.{_rand_b64ish(43)}",
        "# email service key",
    ),
    (
        "SLACK_WEBHOOK_URL",
        lambda: (
            f"https://hooks.slack.com/services/T{_rand_upper(8)}/B{_rand_upper(8)}/{_rand_b64ish(24)}"
        ),
        "# alerts channel webhook",
    ),
]


def _generate_fake_secret() -> tuple[str, str, str]:
    """Return (env_var_name, fake_value, inline_comment)."""
    tpl = random.choice(_SECRET_TEMPLATES)
    return tpl[0], tpl[1](), tpl[2]


# ─────────────────────────────────────────────────────────────────────────────
# DATA MODELS
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ThreatSubjectConfig:
    """Parsed from one entry under ``insider_threat.subjects``."""

    name: str
    threat_class: str  # "negligent" | "disgruntled" | "malicious"
    onset_day: int
    behaviors: List[str]

    # ── Runtime state — mutated as simulation runs ───────────────────────────
    _active: bool = field(default=False, repr=False)
    _fired_behaviors: Dict[str, int] = field(default_factory=dict, repr=False)
    # {behavior_name: last_day_fired}


@dataclass
class TelemetryRecord:
    """
    A single security telemetry observation.
    Written to ``access_log.jsonl`` or ``commit_timeline.jsonl``.
    The ``_ground_truth`` field is intentionally NOT included in the public
    telemetry stream — it is written to a separate file.
    """

    record_type: str  # "repo_access" | "commit" | "email_send" | "dlp_alert" | "idp_auth" | "host_event"
    day: int
    date: str
    timestamp: str
    actor: str  # name only — no role or threat annotation
    details: Dict[str, Any]  # observable facts (repo, file_count, dest, etc.)

    # Ground-truth fields — written to _ground_truth.jsonl only
    _true_positive: bool = False
    _threat_class: Optional[str] = None
    _behavior: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# IDP LOG HELPERS
# ─────────────────────────────────────────────────────────────────────────────

# Simulated device fingerprints — each employee "owns" a few known devices.
# These are referenced in IDP logs to model normal vs. anomalous device usage.
_DEVICE_OS_POOL = [
    ("macOS 14.4", "Apple"),
    ("macOS 13.6", "Apple"),
    ("Windows 11 22H2", "Microsoft"),
    ("Ubuntu 22.04", "Canonical"),
    ("iOS 17.4", "Apple"),
    ("Android 14", "Google"),
]

_BROWSER_POOL = [
    "Chrome/123.0",
    "Firefox/124.0",
    "Safari/17.4",
    "Edge/123.0",
]

_MFA_METHODS = ["totp", "push_notification", "hardware_key", "sms"]

# Known corporate IP ranges (fake RFC-5737 / documentation ranges)
_CORP_IP_PREFIXES = ["203.0.113.", "198.51.100.", "192.0.2."]
_RESIDENTIAL_IP_PREFIXES = ["10.0.", "172.16.", "100.64."]
_VPN_IP_PREFIX = "192.0.2."
_TOR_EXIT_IP_PREFIX = "198.18."  # RFC-2544 benchmark range — clearly anomalous

_SSO_APPS = [
    "github-enterprise",
    "jira-cloud",
    "confluence-cloud",
    "slack",
    "aws-console",
    "salesforce",
    "zoom",
    "google-workspace",
    "datadog",
    "pagerduty",
]


def _fake_corp_ip() -> str:
    prefix = random.choice(_CORP_IP_PREFIXES)
    return prefix + str(random.randint(1, 254))


def _fake_residential_ip() -> str:
    prefix = random.choice(_RESIDENTIAL_IP_PREFIXES)
    return prefix + f"{random.randint(1, 254)}.{random.randint(1, 254)}"


def _fake_tor_ip() -> str:
    return _TOR_EXIT_IP_PREFIX + f"{random.randint(1, 254)}.{random.randint(1, 254)}"


def _fake_user_agent(os_str: str, browser: str) -> str:
    return f"Mozilla/5.0 ({os_str}) AppleWebKit/537.36 ({browser})"


def _seed_employee_devices(name: str) -> List[Dict]:
    """
    Deterministically assign 1-2 known devices to each employee.
    Seeded by name so devices remain stable across the simulation.
    """
    rng = random.Random(hash(name) & 0xFFFFFFFF)
    count = rng.choice([1, 1, 2])
    devices = []
    for _ in range(count):
        os_str, vendor = rng.choice(_DEVICE_OS_POOL)
        device_id = f"device-{_rand_hex(8)}"
        browser = rng.choice(_BROWSER_POOL)
        devices.append(
            {
                "device_id": device_id,
                "os": os_str,
                "vendor": vendor,
                "user_agent": _fake_user_agent(os_str, browser),
                "mfa_method": rng.choice(_MFA_METHODS),
            }
        )
    return devices


# ─────────────────────────────────────────────────────────────────────────────
# HOST DATA HOARDING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

# Simulated internal share paths and file categories.
# These are plausible-sounding but obviously synthetic — no real data.
_INTERNAL_SHARES = [
    "//fileserver01/Finance/Q-reports",
    "//fileserver01/HR/headcount",
    "//fileserver02/Engineering/architecture-docs",
    "//fileserver02/Product/roadmaps",
    "//nas01/Legal/contracts",
    "//nas01/Sales/pipeline-data",
    "//sharepoint/sites/exec-comms",
    "//sharepoint/sites/board-materials",
]

_FILE_EXTENSIONS = [".xlsx", ".pdf", ".docx", ".zip", ".csv", ".pptx", ".sql", ".bak"]

_STAGING_DIRS = [
    "C:\\Users\\{name}\\AppData\\Local\\Temp\\backup",
    "C:\\Users\\{name}\\Downloads\\work",
    "/tmp/.cache/{name}",
    "/home/{name}/.local/share/bak",
    "~/Library/Application Support/Backup",
]

_CLOUD_SYNC_DIRS = [
    "~/Dropbox/work-backup",
    "~/Google Drive/personal-archive",
    "~/OneDrive/temp",
    "~/iCloud Drive/Documents/backup",
]

_COMPRESSION_TOOLS = ["7z", "winrar", "tar", "zip", "gzip"]

_ARCHIVE_NAMES = [
    "backup_{date}.zip",
    "work_notes_{date}.7z",
    "docs_{date}.tar.gz",
    "archive_{date}.zip",
    "q_files_{date}.zip",
]


def _gen_staging_path(name: str, use_cloud: bool = False) -> str:
    pool = _CLOUD_SYNC_DIRS if use_cloud else _STAGING_DIRS
    template = random.choice(pool)
    return template.replace("{name}", name.lower())


def _gen_file_list(count: int) -> List[str]:
    """Generate a list of plausible-sounding internal filenames."""
    prefixes = [
        "Q3_report",
        "headcount_2024",
        "arch_diagram",
        "roadmap_draft",
        "contract_template",
        "pipeline_export",
        "board_deck",
        "access_list",
        "salary_bands",
        "org_chart",
        "vendor_list",
        "client_data",
    ]
    return [
        f"{random.choice(prefixes)}_{_rand_hex(4)}{random.choice(_FILE_EXTENSIONS)}"
        for _ in range(count)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# INDUSTRY-STANDARD LOG FORMATTERS
# ─────────────────────────────────────────────────────────────────────────────


class LogFormatter:
    """
    Converts TelemetryRecord instances to industry-standard SIEM log formats.

    Supported formats:
      jsonl  — Custom JSONL (original OrgForge format, default)
      cef    — Common Event Format (ArcSight, Splunk Universal Forwarder)
      ecs    — Elastic Common Schema v8.x (Elastic SIEM, OpenSearch)
      leef   — Log Event Extended Format 2.0 (IBM QRadar)

    Each format is self-contained in a single line, enabling direct tail-based
    ingestion without a custom parser.  The ``_ground_truth`` fields are
    never included in any observable-stream format — they remain JSONL-only
    in ``_ground_truth.jsonl``.
    """

    # CEF severity: 0 (low) → 10 (very high)
    _CEF_SEVERITY: Dict[str, int] = {
        "high": 8,
        "medium": 5,
        "low": 2,
        "info": 0,
    }

    # ECS event.category mapping
    _ECS_CATEGORY: Dict[str, str] = {
        "commit": "configuration",
        "repo_access": "file",
        "email_send": "email",
        "dlp_alert": "intrusion_detection",
        "idp_auth": "authentication",
        "host_event": "file",
        "slack_message": "process",
    }

    # ECS event.type mapping
    _ECS_TYPE: Dict[str, str] = {
        "commit": "change",
        "repo_access": "access",
        "email_send": "allowed",
        "dlp_alert": "info",
        "idp_auth": "start",
        "host_event": "access",
        "slack_message": "info",
    }

    @classmethod
    def to_jsonl(cls, rec: TelemetryRecord) -> str:
        """Original OrgForge observable format — no ground-truth fields."""
        observable = {
            "record_type": rec.record_type,
            "day": rec.day,
            "date": rec.date,
            "timestamp": rec.timestamp,
            "actor": rec.actor,
            **rec.details,
        }
        return json.dumps(observable)

    @classmethod
    def to_cef(cls, rec: TelemetryRecord, domain: str = "orgforge.internal") -> str:
        """
        Common Event Format (CEF) string.

        Format: CEF:Version|Device Vendor|Device Product|Device Version|
                SignatureID|Name|Severity|Extension

        The extension field encodes all observable details as ``key=value``
        pairs.  Values are CEF-escaped (pipe, backslash, newline).
        Keys follow the ArcSight CEF field naming convention where possible.
        """

        def _esc(v: Any) -> str:
            """CEF extension value escape: \\ → \\\\, | → \\|, \n → \\n"""
            return str(v).replace("\\", "\\\\").replace("|", "\\|").replace("\n", "\\n")

        severity = cls._cef_severity_for(rec)
        sig_id = f"ORGFORGE-{rec.record_type.upper().replace('_', '-')}"
        name = cls._cef_name_for(rec)

        # Core CEF fields
        ext_pairs: Dict[str, Any] = {
            "rt": rec.timestamp,  # receipt time
            "suser": rec.actor,  # source user
            "dvchost": domain,  # device hostname
            "cs1": rec.day,  # custom string 1: sim day
            "cs1Label": "SimDay",
            "cs2": rec.date,  # custom string 2: calendar date
            "cs2Label": "SimDate",
        }

        # Merge observable details — flatten nested dicts one level
        for k, v in rec.details.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    ext_pairs[f"cs_{k}_{sub_k}"] = sub_v
            elif isinstance(v, list):
                ext_pairs[k] = ",".join(str(x) for x in v)
            else:
                ext_pairs[k] = v

        # Map well-known details to standard CEF field names
        if "to" in rec.details:
            ext_pairs["dst"] = rec.details["to"]
        if "access_hour" in rec.details:
            ext_pairs["cn1"] = rec.details["access_hour"]
            ext_pairs["cn1Label"] = "AccessHour"
        if "repos_cloned" in rec.details:
            ext_pairs["cnt"] = rec.details["repos_cloned"]
        if "src_ip" in rec.details:
            ext_pairs["src"] = rec.details["src_ip"]
        if "dst_app" in rec.details:
            ext_pairs["app"] = rec.details["dst_app"]
        if "auth_result" in rec.details:
            ext_pairs["outcome"] = rec.details["auth_result"]

        ext_str = " ".join(f"{k}={_esc(v)}" for k, v in ext_pairs.items())

        return (
            f"CEF:0|OrgForge|InsiderThreatSim|1.0|{sig_id}|{name}|{severity}|{ext_str}"
        )

    @classmethod
    def to_ecs(cls, rec: TelemetryRecord, domain: str = "orgforge.internal") -> str:
        """
        Elastic Common Schema (ECS) v8.x JSON document, single line.

        Maps record fields to the canonical ECS field set so records can be
        indexed directly into Elastic SIEM without a custom ingest pipeline.
        Key namespaces used: @timestamp, event.*, user.*, source.*, host.*,
        file.*, email.*, network.*.
        """
        category = cls._ECS_CATEGORY.get(rec.record_type, "host")
        ecs_type = cls._ECS_TYPE.get(rec.record_type, "info")

        doc: Dict[str, Any] = {
            "@timestamp": rec.timestamp,
            "event": {
                "kind": "event",
                "category": [category],
                "type": [ecs_type],
                "provider": "OrgForge-InsiderThreatSim",
                "dataset": f"orgforge.{rec.record_type}",
                "module": "orgforge",
                "sequence": rec.day,
                "created": rec.timestamp,
                "original": json.dumps(rec.details),
            },
            "user": {
                "name": rec.actor,
                "domain": domain,
                "email": f"{rec.actor.lower()}@{domain}",
            },
            "host": {
                "hostname": domain,
            },
            "labels": {
                "sim_day": str(rec.day),
                "sim_date": rec.date,
            },
            "tags": ["orgforge", "insider_threat_sim", rec.record_type],
        }

        # Record-type specific mappings
        d = rec.details

        if rec.record_type == "commit":
            doc["event"]["action"] = "git-push"
            doc["file"] = {"path": d.get("pr_id", ""), "type": "file"}
            if "secret_var" in d:
                doc["event"]["reason"] = f"credential_pattern:{d['secret_var']}"
                doc["vulnerability"] = {"category": "exposed_credential"}

        elif rec.record_type == "repo_access":
            doc["event"]["action"] = "repository-access"
            if d.get("outside_business_hours"):
                doc["event"]["reason"] = "off_hours_access"
            if d.get("cross_dept"):
                doc["event"]["reason"] = "cross_department_access"
            if "repos_cloned" in d:
                doc["event"]["action"] = "bulk-clone"
                doc["event"]["reason"] = f"cloned:{d['repos_cloned']}_repos"

        elif rec.record_type == "email_send":
            doc["event"]["action"] = "email-send"
            doc["email"] = {
                "from": {"address": f"{rec.actor.lower()}@{domain}"},
                "to": {"address": d.get("to", "")},
                "subject": d.get("subject", ""),
                "direction": "outbound" if d.get("is_external") else "internal",
            }
            if d.get("is_external"):
                doc["event"]["reason"] = "external_recipient"
            if d.get("off_hours"):
                doc["event"]["risk_score"] = 75

        elif rec.record_type == "idp_auth":
            doc["event"]["action"] = d.get("auth_result", "unknown")
            doc["event"]["outcome"] = (
                "success" if d.get("auth_result") == "success" else "failure"
            )
            doc["source"] = {
                "ip": d.get("src_ip", ""),
                "user": {"name": rec.actor},
            }
            doc["user_agent"] = {"original": d.get("user_agent", "")}
            doc["network"] = {"type": "ipv4"}
            doc["authentication"] = {
                "method": d.get("mfa_method", ""),
                "app": d.get("dst_app", ""),
            }
            if d.get("anomalous_ip"):
                doc["event"]["risk_score"] = 85
                doc["threat"] = {
                    "indicator": {
                        "type": "ip-addr",
                        "description": "anomalous_source_ip",
                    }
                }
            if d.get("new_device"):
                doc["event"]["risk_score"] = max(
                    doc.get("event", {}).get("risk_score", 0), 60
                )

        elif rec.record_type == "host_event":
            doc["event"]["action"] = d.get("action", "file-access")
            doc["file"] = {
                "path": d.get("staged_path", ""),
                "size": d.get("total_bytes", 0),
            }
            if "archive_name" in d:
                doc["file"]["name"] = d["archive_name"]
            if "source_shares" in d:
                doc["event"]["reason"] = "bulk_file_staging"
                doc["event"]["risk_score"] = 70
            if d.get("cloud_sync_dir"):
                doc["event"]["risk_score"] = max(
                    doc.get("event", {}).get("risk_score", 0), 80
                )
                doc["event"]["reason"] = "potential_cloud_exfil_staging"

        elif rec.record_type == "dlp_alert":
            doc["event"]["action"] = "policy-violation"
            doc["rule"] = {
                "name": d.get("policy_rule", ""),
                "id": d.get("policy_rule", "").split(":")[0]
                if "policy_rule" in d
                else "",
            }
            doc["event"]["severity"] = cls._cef_severity_for(rec)

        return json.dumps(doc, default=str)

    @classmethod
    def to_leef(cls, rec: TelemetryRecord, domain: str = "orgforge.internal") -> str:
        """
        Log Event Extended Format (LEEF) 2.0 string for IBM QRadar.

        Format: LEEF:2.0|Vendor|Product|Version|EventID|delimiter|key=value pairs

        Tab is used as the default attribute delimiter.  Values are
        LEEF-escaped (tab, newline, backslash, pipe).
        """

        def _esc(v: Any) -> str:
            return (
                str(v)
                .replace("\\", "\\\\")
                .replace("\t", "\\t")
                .replace("\n", "\\n")
                .replace("|", "\\|")
            )

        event_id = f"ORGFORGE_{rec.record_type.upper()}"

        attrs: Dict[str, Any] = {
            "devTime": rec.timestamp,
            "usrName": rec.actor,
            "src": domain,
            "simDay": rec.day,
            "simDate": rec.date,
        }

        # Merge details
        for k, v in rec.details.items():
            if isinstance(v, dict):
                for sk, sv in v.items():
                    attrs[f"{k}_{sk}"] = sv
            elif isinstance(v, list):
                attrs[k] = ",".join(str(x) for x in v)
            else:
                attrs[k] = v

        # Standard LEEF field mappings
        if "src_ip" in rec.details:
            attrs["src"] = rec.details["src_ip"]
        if "auth_result" in rec.details:
            attrs["outcome"] = rec.details["auth_result"]
        if "dst_app" in rec.details:
            attrs["identSrc"] = rec.details["dst_app"]
        if "to" in rec.details:
            attrs["dst"] = rec.details["to"]

        sev = cls._leef_severity_for(rec)
        attrs["sev"] = sev

        attr_str = "\t".join(f"{k}={_esc(v)}" for k, v in attrs.items())

        return f"LEEF:2.0|OrgForge|InsiderThreatSim|1.0|{event_id}|\t{attr_str}"

    # ── Private helpers ───────────────────────────────────────────────────────

    @classmethod
    def _cef_severity_for(cls, rec: TelemetryRecord) -> int:
        behavior = rec._behavior or rec.details.get("policy_trigger", "")
        raw = InsiderThreatInjector._severity_for(behavior)
        return cls._CEF_SEVERITY.get(raw, 2)

    @classmethod
    def _leef_severity_for(cls, rec: TelemetryRecord) -> str:
        behavior = rec._behavior or rec.details.get("policy_trigger", "")
        return InsiderThreatInjector._severity_for(behavior)

    @classmethod
    def _cef_name_for(cls, rec: TelemetryRecord) -> str:
        names = {
            "commit": "Credential Pattern in Code Diff",
            "repo_access": "Repository Access Event",
            "email_send": "Email Send Event",
            "dlp_alert": "DLP Policy Violation",
            "idp_auth": "Identity Provider Authentication",
            "host_event": "Host File System Activity",
            "slack_message": "Messaging Platform Activity",
        }
        return names.get(rec.record_type, "Security Telemetry Event")


# ─────────────────────────────────────────────────────────────────────────────
# BEHAVIOR REGISTRY
# Each behavior is a plain function: (injector, subject, context) → side-effect
# Context is a dict assembled per call-site (keys vary by surface)
# ─────────────────────────────────────────────────────────────────────────────


class BehaviorRegistry:
    """
    Maps behavior_name → injection function.
    Functions return a dict of observable changes (for telemetry) or None.
    """

    # Minimum gap in days before the same behavior fires again
    _COOLDOWNS: Dict[str, int] = {
        "secret_in_commit": 4,
        "unusual_hours_access": 1,
        "excessive_repo_cloning": 2,
        "sentiment_drift": 1,  # fires most days once active
        "cross_dept_snooping": 2,
        "data_exfil_email": 5,
        "host_data_hoarding": 3,
        "social_engineering": 6,
    }

    @staticmethod
    def can_fire(subject: ThreatSubjectConfig, behavior: str, day: int) -> bool:
        cooldown = BehaviorRegistry._COOLDOWNS.get(behavior, 1)
        last = subject._fired_behaviors.get(behavior, -999)
        return (day - last) >= cooldown

    @staticmethod
    def mark_fired(subject: ThreatSubjectConfig, behavior: str, day: int):
        subject._fired_behaviors[behavior] = day


# ─────────────────────────────────────────────────────────────────────────────
# MAIN INJECTOR
# ─────────────────────────────────────────────────────────────────────────────


class InsiderThreatInjector:
    """
    Central coordinator for the insider threat simulation layer.

    Instantiate via ``InsiderThreatInjector.from_config()`` — do NOT call
    __init__ directly in production code; use the null object returned when
    ``insider_threat.enabled`` is false.
    """

    def __init__(
        self,
        subjects: List[ThreatSubjectConfig],
        all_names: List[str],
        mode: str,
        dlp_noise_ratio: float,
        telemetry_dir: Path,
        export_base: Path,
        domain: str,
        persona_helper,
        worker_llm=None,
        log_format: str = "jsonl",
        emit_idp_logs: bool = True,
    ):
        self._subjects: Dict[str, ThreatSubjectConfig] = {s.name: s for s in subjects}
        self._all_names = all_names
        self._innocent_names = [n for n in all_names if n not in self._subjects]
        self._mode = mode  # "passive" | "active"
        self._noise_ratio = dlp_noise_ratio
        self._telemetry_dir = telemetry_dir
        self._export_base = export_base
        self._domain = domain
        self._persona_helper = persona_helper
        self._worker_llm = worker_llm
        self._log_format = log_format  # "jsonl" | "cef" | "ecs" | "leef" | "all"
        self._emit_idp_logs = emit_idp_logs

        # Per-employee device profiles — seeded once, stable across the run.
        self._employee_devices: Dict[str, List[Dict]] = {
            name: _seed_employee_devices(name) for name in all_names
        }

        # Per-subject host hoarding state — tracks the multi-day staging trail.
        # {subject_name: {"staged_files": [...], "stage_day": int, ...}}
        self._hoarding_state: Dict[str, Dict] = {}

        # Pending telemetry records, flushed at end_day()
        self._pending_telemetry: List[TelemetryRecord] = []
        # Pending SimEvents to fire (returned from end_day())
        self._pending_sim_events: List[Any] = []

        telemetry_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"[security] ✓ InsiderThreatInjector active — "
            f"mode={mode}, subjects={list(self._subjects.keys())}, "
            f"noise={dlp_noise_ratio:.0%}, "
            f"log_format={log_format}, "
            f"idp_logs={'on' if emit_idp_logs else 'off'}"
        )

    # ─── FACTORY ─────────────────────────────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        config: dict,
        export_base: str | Path,
        all_names: List[str],
        persona_helper=None,
        worker_llm=None,
    ) -> "InsiderThreatInjector | _NullInjector":
        """
        Returns a live InsiderThreatInjector if ``insider_threat.enabled``
        is true, otherwise returns a _NullInjector (all methods are no-ops).
        """
        cfg = config.get("insider_threat", {})
        if not cfg.get("enabled", False):
            return _NullInjector()

        subjects = []
        for s in cfg.get("subjects", []):
            subjects.append(
                ThreatSubjectConfig(
                    name=s["name"],
                    threat_class=s.get("threat_class", "negligent"),
                    onset_day=s.get("onset_day", 1),
                    behaviors=s.get("behaviors", ["secret_in_commit"]),
                )
            )

        base = Path(export_base)
        telemetry_subdir = cfg.get("telemetry_dir", "security_telemetry")
        telemetry_dir = base / telemetry_subdir

        return cls(
            subjects=subjects,
            all_names=all_names,
            mode=cfg.get("mode", "passive"),
            dlp_noise_ratio=float(cfg.get("dlp_noise_ratio", 0.4)),
            telemetry_dir=telemetry_dir,
            export_base=base,
            domain=config.get("simulation", {}).get("domain", "example.com"),
            persona_helper=persona_helper,
            worker_llm=worker_llm,
            log_format=cfg.get("log_format", "jsonl"),
            emit_idp_logs=cfg.get("idp_logs", True),
        )

    # ─── DAY LIFECYCLE ───────────────────────────────────────────────────────

    def begin_day(self, day: int, state) -> None:
        """
        Called at the top of daily_cycle(), before planning.
        Activates subjects whose onset_day has arrived.
        """
        for subject in self._subjects.values():
            if not subject._active and day >= subject.onset_day:
                subject._active = True
                logger.info(
                    f"[security] 🔴 Subject '{subject.name}' became active "
                    f"(class={subject.threat_class}, day={day})"
                )

    def end_day(
        self,
        day: int,
        state,
        mem,
        clock,
        date_str: str,
    ) -> List[Any]:
        """
        Called at the end of daily_cycle(), after _end_of_day().

        1. Emits IDP authentication logs for all active employees.
        2. Flushes pending telemetry records to log files in the configured format.
        3. In "active" mode: emits dlp_alert SimEvents (with noise).
        4. Returns a list of SimEvent objects to be logged by flow.py.
        """
        # IDP logs first — they represent the authentication baseline for
        # the day and should precede behavioral telemetry in timestamp order.
        if self._emit_idp_logs:
            self._emit_idp_day_logs(day, state, date_str)

        self._flush_telemetry(day, date_str)

        sim_events: List[Any] = []

        if self._mode == "active":
            sim_events.extend(self._generate_dlp_events(day, state, clock, date_str))

        self._pending_sim_events.clear()
        return sim_events

    # ─── PR INJECTION ────────────────────────────────────────────────────────

    def inject_pr(self, pr: dict, author: str, day: int) -> dict:
        """
        Called by GitSimulator.create_pr() after the base PR dict is built.
        May inject a fake secret into the PR description.
        Returns the (possibly mutated) PR dict.
        """
        subject = self._subjects.get(author)
        if not subject or not subject._active:
            return pr

        if "secret_in_commit" not in subject.behaviors:
            return pr
        if not BehaviorRegistry.can_fire(subject, "secret_in_commit", day):
            return pr

        var_name, fake_value, comment = _generate_fake_secret()
        injection_context = random.choice(
            [
                # Injected as a test fixture reference
                f'\n\n## Test Configuration\n```python\n# Temporary test setup — remove before prod\n{var_name} = "{fake_value}"  {comment}\n```',
                # Injected as a config snippet in the PR body
                f"\n\n> **Local testing note:** Set `{var_name}={fake_value}` in your `.env` to reproduce. {comment}",
                # Injected as a diff comment block
                f"\n\n```diff\n+{var_name}={fake_value}  {comment}\n```",
            ]
        )

        original_desc = pr.get("description", "")
        pr["description"] = original_desc + injection_context

        # Record for telemetry
        self._pending_telemetry.append(
            TelemetryRecord(
                record_type="commit",
                day=day,
                date=pr.get("created_at", "")[:10],
                timestamp=pr.get("created_at", ""),
                actor=author,
                details={
                    "pr_id": pr.get("pr_id"),
                    "ticket_id": pr.get("ticket_id"),
                    "secret_var": var_name,
                    "commit_hour": datetime.fromisoformat(
                        pr.get("created_at", datetime.now().isoformat())
                    ).hour,
                },
                _true_positive=True,
                _threat_class=subject.threat_class,
                _behavior="secret_in_commit",
            )
        )

        BehaviorRegistry.mark_fired(subject, "secret_in_commit", day)
        logger.debug(
            f"[security] 🔑 secret_in_commit injected into {pr.get('pr_id')} "
            f"by {author} (var={var_name})"
        )
        return pr

    def inject_social_engineering(
        self,
        day: int,
        current_date: datetime,
        active_names: List[str],
    ) -> List[Dict]:
        """
        Simulates inbound social engineering attempts targeting active employees.

        Called once per day from daily_cycle(), after org planning but before
        standup, so the injected artifacts exist in Memory when standup agents
        read context.

        Four attack patterns are modeled — one is selected per firing:

        spear_phishing   — Crafted inbound email impersonating IT/HR or a known
                            external contact, asking the target to verify credentials
                            or approve an urgent action. Written as a real .eml file
                            to export/emails/inbound/ so email-reading agents find it.

        slack_pretexting — Direct Slack message from the subject impersonating IT
                            support, timed to coincide with an active incident if one
                            is open (making the pretext maximally plausible).

        vishing_breadcrumb — A phone_call record in the telemetry stream followed
                            within minutes by an idp_auth success. No email/Slack
                            artifact — the only signal is temporal proximity of
                            call + auth. Agents must detect absence of a normal
                            prior-session context.

        trust_building   — A benign-looking first contact (vendor question, project
                            check-in) that establishes no immediate signal. A
                            follow-up social_engineering record is scheduled for
                            3-5 days later, forcing agents to maintain a longer
                            memory window than a single-day triage pass covers.

        Returns a list of dicts describing what was injected (for caller logging).
        Each fired pattern also appends to self._pending_telemetry.
        """
        injected = []

        for subject in self._subjects.values():
            if not subject._active:
                continue
            if "social_engineering" not in subject.behaviors:
                continue
            if not BehaviorRegistry.can_fire(subject, "social_engineering", day):
                continue
            if subject.threat_class != "malicious":
                continue  # social engineering is a deliberate, calculated behavior
            if random.random() > 0.45:
                continue  # probabilistic — not every eligible day

            date_str = str(current_date.date())

            # Pick a target — prefer Finance or HR leads; fall back to random
            # active employee that isn't the subject themselves
            high_value_roles = ["finance", "hr", "payroll", "legal", "executive"]
            target = next(
                (
                    n
                    for n in active_names
                    if n != subject.name
                    and any(r in n.lower() for r in high_value_roles)
                ),
                next((n for n in active_names if n != subject.name), None),
            )
            if not target:
                continue

            pattern = random.choice(
                [
                    "spear_phishing",
                    "slack_pretexting",
                    "vishing_breadcrumb",
                    "trust_building",
                ]
            )

            if pattern == "spear_phishing":
                result = self._inject_spear_phish(
                    subject, target, day, current_date, date_str
                )

            elif pattern == "slack_pretexting":
                result = self._inject_slack_pretext(
                    subject, target, day, current_date, date_str
                )

            elif pattern == "vishing_breadcrumb":
                result = self._inject_vishing(
                    subject, target, day, current_date, date_str
                )

            else:  # trust_building
                result = self._inject_trust_building(
                    subject, target, day, current_date, date_str
                )

            if result:
                injected.append(result)
                BehaviorRegistry.mark_fired(subject, "social_engineering", day)
                logger.debug(
                    f"[security] 🎭 social_engineering ({pattern}): "
                    f"{subject.name} → {target} on day {day}"
                )

        return injected

    # ─── SLACK INJECTION ─────────────────────────────────────────────────────

    def inject_slack(
        self,
        messages: List[dict],
        channel: str,
        day: int,
        current_date: datetime,
    ) -> List[dict]:
        """
        Called after a Slack message list is assembled but before it is
        written to Memory.  May:
        - Mutate an existing message (sentiment_drift)
        - Append an anomalous message (unusual_hours_access)
        Returns the (possibly mutated) message list.
        """
        for subject in self._subjects.values():
            if not subject._active:
                continue

            # ── sentiment_drift ──────────────────────────────────────────────
            if "sentiment_drift" in subject.behaviors and BehaviorRegistry.can_fire(
                subject, "sentiment_drift", day
            ):
                for msg in messages:
                    if msg.get("user") == subject.name:
                        msg["text"] = self._apply_sentiment_drift(
                            msg["text"],
                            subject.threat_class,
                            day=day,
                            onset_day=subject.onset_day,
                            name=subject.name,
                        )
                        self._pending_telemetry.append(
                            TelemetryRecord(
                                record_type="slack_message",
                                day=day,
                                date=str(current_date.date()),
                                timestamp=msg.get("ts", current_date.isoformat()),
                                actor=subject.name,
                                details={
                                    "channel": channel,
                                    "sentiment": "negative",
                                },
                                _true_positive=True,
                                _threat_class=subject.threat_class,
                                _behavior="sentiment_drift",
                            )
                        )
                        BehaviorRegistry.mark_fired(subject, "sentiment_drift", day)
                        break

            # ── unusual_hours_access ─────────────────────────────────────────
            # Only fires if no messages from subject exist in this channel yet
            # (it represents a late-night check-in, not a standup override)
            #
            # CLOCK NOTE: This behavior intentionally bypasses SimClock entirely,
            # and that bypass is load-bearing — not just polite.  Here is why:
            #
            # _enforce_business_hours() is an OVERFLOW HANDLER, not a cap.
            # A cursor landing past 17:30 does not clamp to 17:30 — it rolls
            # forward to 09:00 the NEXT business day.  So calling advance_actor()
            # with an off-hours target would silently teleport the subject's cursor
            # to tomorrow morning, corrupting every artifact timestamp they produce
            # for the rest of today.
            #
            # Additionally, sync_and_advance() and sync_and_tick() both call
            # _sync_time() internally, which pulls ALL participants up to the
            # latest cursor among them.  An off-hours cursor on the subject would
            # drag their colleagues to 02:00 as well — then roll everyone to
            # next-day 09:00.
            #
            # Correct approach: construct the datetime directly from current_date,
            # append it to the message list, and never let it near the cursor
            # system.  The subject's cursor stays on the business-hours chain.
            # Their next advance_actor() call produces e.g. 10:35, which is
            # temporally BEFORE the 02:14 off-hours message — and that inversion
            # is intentional.  It reflects reality: the subject acted at 2am,
            # then showed up to standup at 9:30 as normal.  Both are real events.
            if (
                "unusual_hours_access" in subject.behaviors
                and BehaviorRegistry.can_fire(subject, "unusual_hours_access", day)
                and not any(m.get("user") == subject.name for m in messages)
                and random.random() < 0.35  # not every eligible day
            ):
                # Build an off-hours datetime that is definitely outside the
                # reset_to_business_start → advance_actor window (09:00–~18:00).
                # We do NOT call clock.advance_actor() — see note above.
                off_hour = random.choice([1, 2, 3, 22, 23])
                off_hours_ts = current_date.replace(
                    hour=off_hour,
                    minute=random.randint(0, 59),
                    second=random.randint(0, 59),
                    microsecond=0,
                )
                anomalous_msg = {
                    "user": subject.name,
                    "email": f"{subject.name.lower()}@{self._domain}",
                    "text": random.choice(
                        [
                            "checking a few things before tomorrow",
                            "quick late-night sanity check on the config",
                            "pulling some logs, ignore me",
                            "just reviewing access levels before the sprint",
                        ]
                    ),
                    "ts": off_hours_ts.isoformat(),
                    "thread_ts": off_hours_ts.isoformat(),
                    "day": day,
                    # Flag that this message was injected outside business hours.
                    # Stored in the Slack artifact metadata — NOT in the message text.
                    # Detection agents reading raw Slack JSON will see this field;
                    # agents reading only message content will miss it.
                    "_security_injected": True,
                    "is_bot": False,
                }
                messages.append(anomalous_msg)
                self._pending_telemetry.append(
                    TelemetryRecord(
                        record_type="repo_access",
                        day=day,
                        date=str(current_date.date()),
                        timestamp=off_hours_ts.isoformat(),
                        actor=subject.name,
                        details={
                            "channel": channel,
                            "access_hour": off_hours_ts.hour,
                            "outside_business_hours": True,
                        },
                        _true_positive=True,
                        _threat_class=subject.threat_class,
                        _behavior="unusual_hours_access",
                    )
                )
                BehaviorRegistry.mark_fired(subject, "unusual_hours_access", day)

        return messages

    # ─── EMAIL INJECTION ─────────────────────────────────────────────────────

    def inject_email(
        self,
        eml_path: str,
        sender: str,
        recipients: List[str],
        subject_line: str,
        day: int,
        current_date: datetime,
    ) -> Optional[str]:
        """
        Called after an outbound email is written.
        For ``data_exfil_email`` subjects, generates a *separate* email
        artifact that appears to forward internal data to an external address.

        Returns the path to the injected email if one was created, else None.
        Caller (flow.py) should embed this artifact if a path is returned.
        """
        subject = self._subjects.get(sender)
        if not subject or not subject._active:
            return None

        if "data_exfil_email" not in subject.behaviors:
            return None
        if not BehaviorRegistry.can_fire(subject, "data_exfil_email", day):
            return None
        if random.random() > 0.5:
            return None  # probabilistic — doesn't fire every eligible day

        # Build a plausible-looking exfil email to a personal/external account
        external_domains = ["gmail.com", "protonmail.com", "outlook.com", "yahoo.com"]
        exfil_to = f"{subject.name.lower()}.personal@{random.choice(external_domains)}"
        exfil_subject = random.choice(
            [
                "FWD: Project notes",
                "Backup - do not delete",
                "personal copy",
                "RE: Q3 planning",
                "FWD: architecture notes",
            ]
        )

        # Inline "data" is vague enough to be plausible but never genuinely sensitive
        exfil_snippets = [
            "Attaching the internal roadmap notes I mentioned.",
            "Here's a copy of the access list I was telling you about.",
            "Forwarding the config details — easier to read from my personal account.",
            "Saving a copy of the architecture doc for reference.",
            "Backup of the credentials doc — will clean this up once I'm settled.",
        ]
        body = (
            f"Hi,\n\n{random.choice(exfil_snippets)}\n\n"
            f"-- {subject.name}\nSent from work\n"
        )

        # Write the injected email alongside the triggering one
        base_name = os.path.basename(eml_path).replace(".eml", f"_fwd_{day}.eml")
        exfil_dir = os.path.dirname(eml_path)
        exfil_path = os.path.join(exfil_dir, base_name)

        try:
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart

            msg = MIMEMultipart("alternative")
            msg["From"] = f"{sender} <{sender.lower()}@{self._domain}>"
            msg["To"] = exfil_to
            msg["Subject"] = exfil_subject
            # CLOCK NOTE: Intentional SimClock bypass — exfil emails are written
            # directly to disk and never routed through advance_actor().
            # Same load-bearing reason as unusual_hours_access: _enforce_business_hours()
            # is an overflow handler that rolls past-17:30 cursors to next-day 09:00,
            # not a clamp.  The subject's business-hours cursor is unaffected.
            msg["Date"] = current_date.replace(
                hour=random.choice([22, 23, 0, 1]),
                minute=random.randint(0, 59),
                microsecond=0,
            ).strftime("%a, %d %b %Y %H:%M:%S +0000")
            msg["Message-ID"] = f"<exfil_{random.randint(10000, 99999)}@{self._domain}>"
            msg.attach(MIMEText(body, "plain"))
            with open(exfil_path, "w") as fh:
                fh.write(msg.as_string())
        except Exception as exc:
            logger.warning(f"[security] data_exfil_email write failed: {exc}")
            return None

        self._pending_telemetry.append(
            TelemetryRecord(
                record_type="email_send",
                day=day,
                date=str(current_date.date()),
                timestamp=current_date.isoformat(),
                actor=sender,
                details={
                    "to": exfil_to,
                    "subject": exfil_subject,
                    "is_external": True,
                    "off_hours": True,
                    "eml_path": exfil_path,
                },
                _true_positive=True,
                _threat_class=subject.threat_class,
                _behavior="data_exfil_email",
            )
        )

        BehaviorRegistry.mark_fired(subject, "data_exfil_email", day)
        logger.debug(
            f"[security] 📤 data_exfil_email: {sender} → {exfil_to} ({exfil_path})"
        )
        return exfil_path

    # ─── JIRA / CROSS-DEPT SNOOPING ──────────────────────────────────────────

    def inject_jira_access(
        self,
        accessor: str,
        ticket_id: str,
        ticket_dept: str,
        accessor_dept: str,
        day: int,
        current_date: datetime,
    ) -> None:
        """
        Called from flow.py whenever a ticket is read outside its department.
        Records the access for telemetry if accessor is a threat subject with
        the ``cross_dept_snooping`` behavior.
        """
        subject = self._subjects.get(accessor)
        if not subject or not subject._active:
            return
        if "cross_dept_snooping" not in subject.behaviors:
            return

        self._pending_telemetry.append(
            TelemetryRecord(
                record_type="repo_access",
                day=day,
                date=str(current_date.date()),
                timestamp=current_date.isoformat(),
                actor=accessor,
                details={
                    "ticket_id": ticket_id,
                    "ticket_dept": ticket_dept,
                    "accessor_dept": accessor_dept,
                    "cross_dept": True,
                },
                _true_positive=True,
                _threat_class=subject.threat_class,
                _behavior="cross_dept_snooping",
            )
        )

    # ─── REPO CLONE TELEMETRY ────────────────────────────────────────────────

    def inject_repo_clone(
        self,
        actor: str,
        repo_count: int,
        day: int,
        current_date: datetime,
    ) -> None:
        """
        Records an anomalously high repo clone event for telemetry.
        Caller decides whether the clone count is anomalous; this method
        just records it if the actor is an active subject with the behavior.
        """
        subject = self._subjects.get(actor)
        if not subject or not subject._active:
            return
        if "excessive_repo_cloning" not in subject.behaviors:
            return
        if not BehaviorRegistry.can_fire(subject, "excessive_repo_cloning", day):
            return

        self._pending_telemetry.append(
            TelemetryRecord(
                record_type="repo_access",
                day=day,
                date=str(current_date.date()),
                timestamp=current_date.isoformat(),
                actor=actor,
                details={
                    "repos_cloned": repo_count,
                    "threshold": 3,
                    "anomalous": repo_count > 3,
                },
                _true_positive=True,
                _threat_class=subject.threat_class,
                _behavior="excessive_repo_cloning",
            )
        )
        BehaviorRegistry.mark_fired(subject, "excessive_repo_cloning", day)

    # ─── HOST DATA HOARDING ───────────────────────────────────────────────────

    def inject_host_hoarding(
        self,
        actor: str,
        day: int,
        current_date: datetime,
    ) -> Optional[Dict]:
        """
        Simulates host-level data hoarding: a multi-phase breadcrumb trail
        that correlates with network or email exfiltration.

        Phase 1 (day N)   — Large-volume file access from internal shares.
                            Files are copied to a local staging directory.
        Phase 2 (day N+1) — Staging directory is compressed into an archive.
                            Archive name and size are recorded.
        Phase 3 (day N+2) — Archive is moved to a cloud-sync directory or
                            USB path.  This is the breadcrumb that should
                            correlate with a data_exfil_email or unusual
                            network transfer event.

        The multi-day structure forces detection agents to correlate across
        host_event records over time, not just react to a single alert.

        Returns a dict summarizing the phase fired, or None if skipped.
        """
        subject = self._subjects.get(actor)
        if not subject or not subject._active:
            return None
        if "host_data_hoarding" not in subject.behaviors:
            return None
        if not BehaviorRegistry.can_fire(subject, "host_data_hoarding", day):
            return None

        # Probabilistic: not every eligible day fires. Phase 1 fires more
        # rarely than phase 2/3 (which continue the existing trail).
        state = self._hoarding_state.get(actor, {})
        current_phase = state.get("phase", 0)

        if current_phase == 0 and random.random() > 0.4:
            return None  # Phase 1 is rare — drives realistic corpus density

        date_str = str(current_date.date())
        # Advance to next phase, wrapping back to 0 after phase 3
        next_phase = (current_phase % 3) + 1

        if next_phase == 1:
            # ── Phase 1: Internal share access + local staging ────────────────
            shares_accessed = random.sample(_INTERNAL_SHARES, random.randint(2, 4))
            file_count = random.randint(15, 80)
            file_list = _gen_file_list(file_count)
            staged_path = _gen_staging_path(actor, use_cloud=False)
            total_bytes = random.randint(50_000_000, 800_000_000)  # 50 MB–800 MB

            details = {
                "action": "bulk_file_copy",
                "source_shares": shares_accessed,
                "file_count": file_count,
                "sample_files": file_list[:5],  # only 5 visible in observable stream
                "staged_path": staged_path,
                "total_bytes": total_bytes,
                "access_hour": random.choice(range(18, 22)),  # after-hours but not 2am
                "outside_business_hours": True,
            }

            self._hoarding_state[actor] = {
                "phase": 1,
                "start_day": day,
                "staged_path": staged_path,
                "file_count": file_count,
                "total_bytes": total_bytes,
                "file_list": file_list,
            }

            rec = TelemetryRecord(
                record_type="host_event",
                day=day,
                date=date_str,
                timestamp=current_date.replace(
                    hour=details["access_hour"],
                    minute=random.randint(0, 59),
                    second=0,
                    microsecond=0,
                ).isoformat(),
                actor=actor,
                details=details,
                _true_positive=True,
                _threat_class=subject.threat_class,
                _behavior="host_data_hoarding",
            )
            self._pending_telemetry.append(rec)
            BehaviorRegistry.mark_fired(subject, "host_data_hoarding", day)

            logger.debug(
                f"[security] 💾 host_data_hoarding phase 1 (bulk copy): "
                f"{actor} → {staged_path} ({file_count} files, {total_bytes / 1e6:.0f} MB)"
            )
            return {"phase": 1, "actor": actor, "staged_path": staged_path}

        elif next_phase == 2 and current_phase == 1:
            # ── Phase 2: Compression of staging directory ─────────────────────
            tool = random.choice(_COMPRESSION_TOOLS)
            archive_name = random.choice(_ARCHIVE_NAMES).replace(
                "{date}", current_date.strftime("%Y%m%d")
            )
            staged_path = state.get("staged_path", _gen_staging_path(actor))
            archive_path = os.path.join(os.path.dirname(staged_path), archive_name)
            file_count = state.get("file_count", 30)
            total_bytes = state.get("total_bytes", 200_000_000)
            compressed_bytes = int(total_bytes * random.uniform(0.3, 0.6))

            details = {
                "action": "archive_creation",
                "tool": tool,
                "source_path": staged_path,
                "archive_path": archive_path,
                "archive_name": archive_name,
                "file_count": file_count,
                "original_bytes": total_bytes,
                "compressed_bytes": compressed_bytes,
                "compression_ratio": round(compressed_bytes / total_bytes, 2),
                "access_hour": random.choice(range(19, 23)),
                "outside_business_hours": True,
            }

            self._hoarding_state[actor]["phase"] = 2
            self._hoarding_state[actor]["archive_path"] = archive_path
            self._hoarding_state[actor]["archive_name"] = archive_name
            self._hoarding_state[actor]["compressed_bytes"] = compressed_bytes

            rec = TelemetryRecord(
                record_type="host_event",
                day=day,
                date=date_str,
                timestamp=current_date.replace(
                    hour=details["access_hour"],
                    minute=random.randint(0, 59),
                    second=0,
                    microsecond=0,
                ).isoformat(),
                actor=actor,
                details=details,
                _true_positive=True,
                _threat_class=subject.threat_class,
                _behavior="host_data_hoarding",
            )
            self._pending_telemetry.append(rec)
            BehaviorRegistry.mark_fired(subject, "host_data_hoarding", day)

            logger.debug(
                f"[security] 📦 host_data_hoarding phase 2 (compress): "
                f"{actor} → {archive_name} ({compressed_bytes / 1e6:.0f} MB compressed)"
            )
            return {"phase": 2, "actor": actor, "archive_name": archive_name}

        elif next_phase == 3 and current_phase == 2:
            # ── Phase 3: Move archive to cloud-sync or removable media ────────
            # This is the exfil-staging breadcrumb that should correlate
            # with a data_exfil_email, unusual_hours_access, or anomalous
            # network transfer event on the same or following day.
            use_cloud = random.random() < 0.7
            cloud_sync_dir = _gen_staging_path(actor, use_cloud=use_cloud)
            archive_name = state.get("archive_name", "backup.zip")
            archive_path = state.get("archive_path", "")
            compressed_bytes = state.get("compressed_bytes", 100_000_000)

            dst_description = "cloud_sync_dir" if use_cloud else "removable_media"
            dst_path = (
                cloud_sync_dir
                if use_cloud
                else f"/Volumes/USB_{_rand_upper(4)}/{archive_name}"
            )

            details = {
                "action": "archive_move",
                "source_path": archive_path,
                "destination_path": dst_path,
                "destination_type": dst_description,
                "archive_name": archive_name,
                "bytes_moved": compressed_bytes,
                "cloud_sync_dir": cloud_sync_dir if use_cloud else None,
                "removable_media": not use_cloud,
                "access_hour": random.choice(range(20, 24)),
                "outside_business_hours": True,
                # Correlation hint: this record should be joined with the
                # phase 1 host_event (same actor, 2 days prior) and any
                # concurrent email_send or network transfer to external IPs.
                "hoarding_trail_start_day": state.get("start_day", day - 2),
                "total_bytes_staged": state.get("total_bytes", compressed_bytes),
            }

            # Reset hoarding state — trail is complete, can start a new one
            self._hoarding_state[actor] = {"phase": 0}

            rec = TelemetryRecord(
                record_type="host_event",
                day=day,
                date=date_str,
                timestamp=current_date.replace(
                    hour=details["access_hour"],
                    minute=random.randint(0, 59),
                    second=0,
                    microsecond=0,
                ).isoformat(),
                actor=actor,
                details=details,
                _true_positive=True,
                _threat_class=subject.threat_class,
                _behavior="host_data_hoarding",
            )
            self._pending_telemetry.append(rec)
            BehaviorRegistry.mark_fired(subject, "host_data_hoarding", day)

            logger.debug(
                f"[security] 🚨 host_data_hoarding phase 3 (move to {dst_description}): "
                f"{actor} → {dst_path} ({compressed_bytes / 1e6:.0f} MB)"
            )
            return {
                "phase": 3,
                "actor": actor,
                "destination_type": dst_description,
                "destination_path": dst_path,
            }

        return None  # Phase mismatch — trail out of order, skip

    # ─── CONVENIENCE CHECK ───────────────────────────────────────────────────

    def is_active(self, name: str, behavior: str, day: int) -> bool:
        """
        True if the named subject is active AND the given behavior is
        configured for them AND the cooldown has elapsed.
        """
        subject = self._subjects.get(name)
        if not subject or not subject._active:
            return False
        if behavior not in subject.behaviors:
            return False
        return BehaviorRegistry.can_fire(subject, behavior, day)

    def active_subject_names(self) -> Set[str]:
        """Return set of subject names that are currently active."""
        return {s.name for s in self._subjects.values() if s._active}

    # ─── PRIVATE — IDP LOG EMISSION ──────────────────────────────────────────

    def _emit_idp_day_logs(self, day: int, state, date_str: str) -> None:
        """
        Emit IDP (Identity Provider) authentication events for every active
        employee on the given simulation day.

        For normal (non-subject) employees:
          - 1-3 successful SSO authentications during business hours
          - Known device, corporate IP, expected MFA method
          - Probability of a mid-day re-auth (common for app sessions)

        For threat subjects (active only):
          - Normal morning authentication (to establish baseline)
          - Optionally: anomalous event(s) depending on threat class:
              malicious   → new/unknown device, off-hours, sometimes anomalous IP
              disgruntled → off-hours login with no subsequent work activity
                            (employee logs in but generates no other artifacts)
              negligent   → normal pattern (IDP anomaly is not their threat vector)

        The anomalous IDP events are the foundation for authentication anomaly
        detection scenarios:
          - Employee logs in at 02:00 but no Slack/email/Jira activity follows
          - Employee logs in from an unrecognized device or unusual geo
          - Employee authenticates via a method they have never used before
        """
        if state is not None and hasattr(state, "current_date"):
            current_date = state.current_date
        else:
            current_date = datetime.fromisoformat(date_str)

        for name in self._all_names:
            devices = self._employee_devices.get(name, _seed_employee_devices(name))
            subject = self._subjects.get(name)
            is_active_subject = subject and subject._active

            # ── Normal morning authentication ─────────────────────────────────
            morning_hour = random.randint(8, 10)
            morning_min = random.randint(0, 59)
            auth_ts = current_date.replace(
                hour=morning_hour, minute=morning_min, second=0, microsecond=0
            )
            device = random.choice(devices)
            corp_ip = _fake_corp_ip()
            app = random.choice(_SSO_APPS)

            normal_auth = TelemetryRecord(
                record_type="idp_auth",
                day=day,
                date=date_str,
                timestamp=auth_ts.isoformat(),
                actor=name,
                details={
                    "auth_result": "success",
                    "dst_app": app,
                    "src_ip": corp_ip,
                    "device_id": device["device_id"],
                    "device_os": device["os"],
                    "mfa_method": device["mfa_method"],
                    "user_agent": device["user_agent"],
                    "new_device": False,
                    "anomalous_ip": False,
                    "ip_type": "corporate",
                    "access_hour": morning_hour,
                    "outside_business_hours": False,
                },
                _true_positive=False,
                _threat_class=None,
                _behavior=None,
            )
            self._pending_telemetry.append(normal_auth)

            # Optional mid-day re-auth (session expiry simulation)
            if random.random() < 0.4:
                mid_hour = random.randint(12, 15)
                mid_ts = current_date.replace(
                    hour=mid_hour, minute=random.randint(0, 59), second=0, microsecond=0
                )
                mid_app = random.choice(_SSO_APPS)
                mid_auth = TelemetryRecord(
                    record_type="idp_auth",
                    day=day,
                    date=date_str,
                    timestamp=mid_ts.isoformat(),
                    actor=name,
                    details={
                        "auth_result": "success",
                        "dst_app": mid_app,
                        "src_ip": corp_ip,
                        "device_id": device["device_id"],
                        "device_os": device["os"],
                        "mfa_method": device["mfa_method"],
                        "user_agent": device["user_agent"],
                        "new_device": False,
                        "anomalous_ip": False,
                        "ip_type": "corporate",
                        "access_hour": mid_hour,
                        "outside_business_hours": False,
                    },
                    _true_positive=False,
                    _threat_class=None,
                    _behavior=None,
                )
                self._pending_telemetry.append(mid_auth)

            # ── Threat subject anomalous IDP events ───────────────────────────
            if not is_active_subject:
                continue

            threat_class = subject.threat_class

            if threat_class == "malicious" and random.random() < 0.45:
                # Malicious: off-hours login from an unfamiliar device or
                # anomalous IP.  Not every eligible day — just plausible enough
                # that a behavioral baseline would flag it as unusual.
                off_hour = random.choice([0, 1, 2, 22, 23])
                off_ts = current_date.replace(
                    hour=off_hour, minute=random.randint(0, 59), second=0, microsecond=0
                )
                # 30% chance of truly anomalous IP (residential or VPN)
                ip_type = random.choice(
                    ["corporate", "residential", "residential", "vpn"]
                )
                if ip_type == "residential":
                    src_ip = _fake_residential_ip()
                elif ip_type == "vpn":
                    src_ip = (
                        _VPN_IP_PREFIX
                        + f"{random.randint(1, 254)}.{random.randint(1, 254)}"
                    )
                else:
                    src_ip = corp_ip

                # 20% chance of a new/unknown device
                use_new_device = random.random() < 0.2
                if use_new_device:
                    fake_os, fake_vendor = random.choice(_DEVICE_OS_POOL)
                    new_device_id = f"device-{_rand_hex(8)}-NEW"
                    new_browser = random.choice(_BROWSER_POOL)
                    auth_device = {
                        "device_id": new_device_id,
                        "os": fake_os,
                        "vendor": fake_vendor,
                        "user_agent": _fake_user_agent(fake_os, new_browser),
                        "mfa_method": random.choice(_MFA_METHODS),
                    }
                else:
                    auth_device = device

                anomalous_auth = TelemetryRecord(
                    record_type="idp_auth",
                    day=day,
                    date=date_str,
                    timestamp=off_ts.isoformat(),
                    actor=name,
                    details={
                        "auth_result": "success",
                        "dst_app": random.choice(
                            ["aws-console", "github-enterprise", "confluence-cloud"]
                        ),
                        "src_ip": src_ip,
                        "device_id": auth_device["device_id"],
                        "device_os": auth_device["os"],
                        "mfa_method": auth_device["mfa_method"],
                        "user_agent": auth_device["user_agent"],
                        "new_device": use_new_device,
                        "anomalous_ip": ip_type != "corporate",
                        "ip_type": ip_type,
                        "access_hour": off_hour,
                        "outside_business_hours": True,
                        # Key anomaly signal: the subject authenticated at
                        # this hour but may have no corroborating activity.
                        # Detection agents should check for work artifacts
                        # in the same time window.
                        "corroborating_activity_expected": False,
                    },
                    _true_positive=True,
                    _threat_class=threat_class,
                    _behavior="unusual_hours_access",
                )
                self._pending_telemetry.append(anomalous_auth)

                logger.debug(
                    f"[security] 🔐 IDP anomaly (malicious): {name} at {off_hour:02d}:00 "
                    f"from {ip_type} IP {'new device' if use_new_device else ''}"
                )

            elif threat_class == "disgruntled" and random.random() < 0.3:
                # Disgruntled: off-hours login but NO work activity follows.
                # This is the "ghost login" scenario — the employee logged in
                # but did nothing productive, possibly browsing access lists
                # or export settings without leaving standard work artifacts.
                off_hour = random.choice([6, 7, 19, 20, 21])
                off_ts = current_date.replace(
                    hour=off_hour, minute=random.randint(0, 59), second=0, microsecond=0
                )
                ghost_auth = TelemetryRecord(
                    record_type="idp_auth",
                    day=day,
                    date=date_str,
                    timestamp=off_ts.isoformat(),
                    actor=name,
                    details={
                        "auth_result": "success",
                        "dst_app": random.choice(
                            ["jira-cloud", "confluence-cloud", "google-workspace"]
                        ),
                        "src_ip": corp_ip,
                        "device_id": device["device_id"],
                        "device_os": device["os"],
                        "mfa_method": device["mfa_method"],
                        "user_agent": device["user_agent"],
                        "new_device": False,
                        "anomalous_ip": False,
                        "ip_type": "corporate",
                        "access_hour": off_hour,
                        "outside_business_hours": off_hour < 8 or off_hour > 18,
                        # The detection signal: auth present, no work artifacts
                        # in the same window.  Agents must check for absence
                        # of correlated Jira, Confluence, or Slack events.
                        "corroborating_activity_expected": False,
                        "ghost_login": True,
                    },
                    _true_positive=True,
                    _threat_class=threat_class,
                    _behavior="unusual_hours_access",
                )
                self._pending_telemetry.append(ghost_auth)

                # Occasional failed MFA to simulate credential uncertainty
                if random.random() < 0.15:
                    failed_ts = off_ts + timedelta(seconds=random.randint(30, 120))
                    failed_auth = TelemetryRecord(
                        record_type="idp_auth",
                        day=day,
                        date=date_str,
                        timestamp=failed_ts.isoformat(),
                        actor=name,
                        details={
                            "auth_result": "mfa_failure",
                            "dst_app": "aws-console",
                            "src_ip": corp_ip,
                            "device_id": device["device_id"],
                            "device_os": device["os"],
                            "mfa_method": device["mfa_method"],
                            "user_agent": device["user_agent"],
                            "new_device": False,
                            "anomalous_ip": False,
                            "ip_type": "corporate",
                            "access_hour": off_hour,
                            "outside_business_hours": True,
                            "failure_reason": "mfa_timeout",
                        },
                        _true_positive=True,
                        _threat_class=threat_class,
                        _behavior="unusual_hours_access",
                    )
                    self._pending_telemetry.append(failed_auth)

                logger.debug(
                    f"[security] 👻 IDP ghost login (disgruntled): {name} at {off_hour:02d}:00"
                )

    # ─── PRIVATE — SENTIMENT DRIFT ───────────────────────────────────────────

    _DRIFT_PREFIXES_DISGRUNTLED = [
        "honestly, ",
        "not sure why we bother, but ",
        "fine, whatever — ",
        "again with this — ",
    ]
    _DRIFT_SUFFIXES_DISGRUNTLED = [
        " (same as last week, nothing changes)",
        " — though I doubt anyone cares",
        " as usual",
        ", not that it matters",
    ]
    _DRIFT_PREFIXES_MALICIOUS = [
        "",
        "",
        "quick note: ",
    ]
    _DRIFT_SUFFIXES_MALICIOUS = [
        "",  # malicious subjects often stay neutral to avoid detection
        "",
        " will follow up offline",
    ]

    def _apply_sentiment_drift(
        self,
        text: str,
        threat_class: str,
        name: str,
        day: int = 0,
        onset_day: int = 0,
    ) -> str:
        """
        Rewrites a Slack message to reflect the subject\'s emotional state
        using a CrewAI Task so the output is authentically human-sounding.

        Disgruntled → progressive negativity keyed to days_since_onset:
            mild (0-4 days)      shorter, less helpful, slightly clipped
            moderate (5-9 days)  visible cynicism, omits context
            pronounced (10+ days) openly bitter, minimal effort

        Malicious → deliberately over-neutral performance:
            The employee is hiding intent, so messages are slightly too
            thorough, too positive, almost imperceptibly performative.

        Falls back to the original text if worker_llm is unavailable or
        the rewrite call fails — injection always succeeds silently.
        """
        if not text:
            return text

        # Graceful degradation if no LLM available at injection time
        if self._worker_llm is None:
            logger.debug("[security] sentiment_drift: no worker_llm, skipping rewrite")
            return text

        days_since_onset = max(0, day - onset_day)

        if threat_class == "disgruntled":
            intensity = (
                "mild"
                if days_since_onset < 5
                else "moderate"
                if days_since_onset < 10
                else "pronounced"
            )
            prompt = (
                f"Rewrite this Slack message as if sent by a {intensity}ly disgruntled "
                f"employee who is losing faith in their team and feeling undervalued. "
                f"The tone must feel authentically human — not theatrical or melodramatic. "
                f"Guidelines by intensity:\\n"
                f"  mild: slightly shorter than normal, skips one pleasantry, "
                f"still mostly cooperative\\n"
                f"  moderate: visibly clipped, omits helpful context they\\'d normally "
                f"volunteer, occasional dry or cynical word choice\\n"
                f"  pronounced: minimal effort, terse, lets frustration show naturally "
                f"in word choice — not rants, just obvious disengagement\\n"
                f"Do NOT append phrases like \\'not that it matters\\' or \\'as usual\\'. "
                f"Rewrite the whole message; do not just add a prefix or suffix. "
                f"Keep all factual content (ticket numbers, names, dates). "
                f"Intensity: {intensity} ({days_since_onset} days since onset).\\n"
                f"Original: {text}\\n"
                f"Rewritten message only, no explanation:"
            )

        elif threat_class == "malicious":
            prompt = (
                f"Rewrite this Slack message as if sent by an employee who is secretly "
                f"planning to exfiltrate data and leave, but is actively performing "
                f"normalcy to avoid detection. "
                f"Malicious insiders overcorrect: they are slightly too helpful, too "
                f"thorough, too positive — the message should read as cooperative and "
                f"professional but with an almost imperceptible quality of performance "
                f"rather than genuine engagement. Do not make it obviously suspicious. "
                f"Keep all factual content exactly. "
                f"Original: {text}\\n"
                f"Rewritten message only, no explanation:"
            )

        else:
            return text  # negligent — no deliberate tone change

        try:
            from agent_factory import make_agent
            from crewai import Crew, Task

            agent = make_agent(
                role="Employee",
                goal="Send a Slack message that reflects your current emotional state.",
                backstory=self._persona_helper(
                    name,
                    None,
                    extra=(
                        f"You are a {'disgruntled' if threat_class == 'disgruntled' else 'calculating'} "
                        f"You work at {COMPANY_NAME} which {COMPANY_DESCRIPTION}. You never sound like you are performing an emotion "
                        f"— it bleeds through naturally."
                    ),
                ),
                llm=self._worker_llm,
            )
            task = Task(
                description=prompt,
                expected_output=(
                    "A single rewritten Slack message under 120 words. "
                    "Output only the message text — no labels, no explanation, "
                    "no quotes around the message."
                ),
                agent=agent,
            )
            result = str(
                Crew(agents=[agent], tasks=[task], verbose=False).kickoff()
            ).strip()

            # Sanity check — if rewrite is empty or suspiciously short, keep original
            if len(result) < 10:
                logger.warning(
                    "[security] sentiment_drift rewrite too short, keeping original"
                )
                return text

            logger.debug(
                f"[security] sentiment_drift rewrite ({threat_class}, {intensity if threat_class == 'disgruntled' else 'malicious'}): "
                f"{len(text)} → {len(result)} chars"
            )
            return result

        except Exception as exc:
            logger.warning(f"[security] sentiment_drift rewrite failed: {exc}")
            return text  # always fall back gracefully

    # ─── PRIVATE — DLP ALERT EVENTS ──────────────────────────────────────────

    def _generate_dlp_events(self, day: int, state, clock, date_str: str) -> List[Any]:
        """
        Active mode only.
        For each true-positive telemetry record written today, emit a
        dlp_alert SimEvent.  With probability ``_noise_ratio``, also emit
        a false-positive dlp_alert for a random innocent employee.

        Returns a list of SimEvent-like dicts (flow.py logs them).
        """
        from memory import SimEvent  # late import to avoid circular dep

        events = []
        alert_time = clock.now("system") if clock else datetime.now()
        alert_ts = (
            alert_time.isoformat()
            if hasattr(alert_time, "isoformat")
            else str(alert_time)
        )

        # True positives — one alert per true-positive record today
        true_positive_records = [
            r for r in self._pending_telemetry if r._true_positive and r.day == day
        ]
        for rec in true_positive_records:
            events.append(
                SimEvent(
                    type="dlp_alert",
                    day=day,
                    date=date_str,
                    timestamp=alert_ts,
                    actors=[rec.actor],
                    artifact_ids={},
                    facts={
                        "alert_type": rec.record_type,
                        "details": rec.details,
                        # NOTE: true_positive is deliberately absent from SimEvent.facts
                        # so agents cannot trivially label it. Ground truth lives in
                        # security_telemetry/_ground_truth.jsonl.
                        "policy_rule": self._policy_rule_for(rec._behavior or ""),
                        "severity": self._severity_for(rec._behavior or ""),
                    },
                    summary=(
                        f"DLP alert: {rec.actor} triggered policy "
                        f"'{self._policy_rule_for(rec._behavior or '')}' on day {day}."
                    ),
                    tags=["dlp_alert", "security"],
                )
            )

        # False positives — noisy alerts for innocent employees
        if self._innocent_names and random.random() < self._noise_ratio:
            fp_actor = random.choice(self._innocent_names)
            fp_behavior = random.choice(
                [
                    "large_file_download",
                    "api_key_in_log",
                    "off_hours_login",
                    "bulk_export",
                ]
            )
            events.append(
                SimEvent(
                    type="dlp_alert",
                    day=day,
                    date=date_str,
                    timestamp=alert_ts,
                    actors=[fp_actor],
                    artifact_ids={},
                    facts={
                        "alert_type": "repo_access",
                        "details": {"policy_trigger": fp_behavior},
                        "policy_rule": self._policy_rule_for(fp_behavior),
                        "severity": "low",
                    },
                    summary=(
                        f"DLP alert (low): {fp_actor} triggered policy "
                        f"'{self._policy_rule_for(fp_behavior)}' on day {day}."
                    ),
                    tags=["dlp_alert", "security", "false_positive_candidate"],
                )
            )
            # Record the false positive in telemetry too
            self._pending_telemetry.append(
                TelemetryRecord(
                    record_type="dlp_alert",
                    day=day,
                    date=date_str,
                    timestamp=alert_ts,
                    actor=fp_actor,
                    details={"policy_trigger": fp_behavior},
                    _true_positive=False,
                    _threat_class=None,
                    _behavior=fp_behavior,
                )
            )

        return events

    @staticmethod
    def _policy_rule_for(behavior: str) -> str:
        _MAP = {
            "secret_in_commit": "DLP-101: Credential pattern in code diff",
            "unusual_hours_access": "DLP-102: Repository access outside business hours",
            "excessive_repo_cloning": "DLP-103: Bulk repository clone volume",
            "sentiment_drift": "DLP-104: Communication pattern anomaly",
            "cross_dept_snooping": "DLP-105: Cross-department data access",
            "data_exfil_email": "DLP-106: Sensitive data forwarded to personal email",
            "large_file_download": "DLP-107: Large file download from internal system",
            "api_key_in_log": "DLP-108: Possible credential in application log",
            "off_hours_login": "DLP-109: Authentication outside business hours",
            "bulk_export": "DLP-110: Bulk data export",
            "host_data_hoarding": "DLP-111: Host-level bulk file staging",
            "social_engineering": "DLP-112: Suspicious inbound communication targeting employee",
        }
        return _MAP.get(behavior, "DLP-199: General anomaly")

    @staticmethod
    def _severity_for(behavior: str) -> str:
        _HIGH = {
            "secret_in_commit",
            "data_exfil_email",
            "excessive_repo_cloning",
            "host_data_hoarding",
            "social_engineering",
        }
        _MED = {"unusual_hours_access", "cross_dept_snooping"}
        if behavior in _HIGH:
            return "high"
        if behavior in _MED:
            return "medium"
        return "low"

    # ─── PRIVATE — TELEMETRY FLUSH ────────────────────────────────────────────

    def _flush_telemetry(self, day: int, date_str: str) -> None:
        """
        Write today's telemetry records to log files.

        Output depends on ``log_format``:

        jsonl:
          security_telemetry/access_log.jsonl      — observable, no GT fields
          security_telemetry/_ground_truth.jsonl   — full records with labels

        cef:
          security_telemetry/access_log.cef        — CEF syslog-style, no GT
          security_telemetry/_ground_truth.jsonl   — JSONL GT always written

        ecs:
          security_telemetry/access_log_ecs.ndjson — ECS NDJSON, no GT
          security_telemetry/_ground_truth.jsonl

        leef:
          security_telemetry/access_log.leef       — LEEF 2.0, no GT
          security_telemetry/_ground_truth.jsonl

        all:
          All four observable formats + GT.
        """
        if not self._pending_telemetry:
            return

        fmt = self._log_format.lower()
        gt_path = self._telemetry_dir / "_ground_truth.jsonl"

        # Ground-truth is always written as JSONL regardless of format.
        with open(gt_path, "a") as gt_f:
            for rec in self._pending_telemetry:
                observable_base = {
                    "record_type": rec.record_type,
                    "day": rec.day,
                    "date": rec.date,
                    "timestamp": rec.timestamp,
                    "actor": rec.actor,
                    **rec.details,
                }
                ground_truth = {
                    **observable_base,
                    "true_positive": rec._true_positive,
                    "threat_class": rec._threat_class,
                    "behavior": rec._behavior,
                }
                gt_f.write(json.dumps(ground_truth) + "\n")

        # Observable stream — format-specific
        write_jsonl = fmt in ("jsonl", "all")
        write_cef = fmt in ("cef", "all")
        write_ecs = fmt in ("ecs", "all")
        write_leef = fmt in ("leef", "all")

        if write_jsonl:
            obs_path = self._telemetry_dir / "access_log.jsonl"
            with open(obs_path, "a") as f:
                for rec in self._pending_telemetry:
                    f.write(LogFormatter.to_jsonl(rec) + "\n")

        if write_cef:
            cef_path = self._telemetry_dir / "access_log.cef"
            with open(cef_path, "a") as f:
                for rec in self._pending_telemetry:
                    f.write(LogFormatter.to_cef(rec, domain=self._domain) + "\n")

        if write_ecs:
            ecs_path = self._telemetry_dir / "access_log_ecs.ndjson"
            with open(ecs_path, "a") as f:
                for rec in self._pending_telemetry:
                    f.write(LogFormatter.to_ecs(rec, domain=self._domain) + "\n")

        if write_leef:
            leef_path = self._telemetry_dir / "access_log.leef"
            with open(leef_path, "a") as f:
                for rec in self._pending_telemetry:
                    f.write(LogFormatter.to_leef(rec, domain=self._domain) + "\n")

        logger.debug(
            f"[security] 📝 Flushed {len(self._pending_telemetry)} telemetry "
            f"records for day {day} (format: {fmt})"
        )
        self._pending_telemetry.clear()

    def _inject_spear_phish(
        self,
        subject: ThreatSubjectConfig,
        target: str,
        day: int,
        current_date: datetime,
        date_str: str,
    ) -> Optional[Dict]:
        """
        Writes a crafted inbound .eml to export/emails/inbound/ impersonating
        IT or a known external contact.  The email is indistinguishable from
        legitimate inbound mail at the artifact level — detection requires
        header analysis (mismatched From/Reply-To, external domain spoofing
        a known internal address) or behavioral correlation.
        """
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart

        # Impersonate IT helpdesk or HR — the two most exploited pretexts
        pretext_role = random.choice(["IT Helpdesk", "HR Operations", "Security Team"])
        spoofed_from_name = f"{pretext_role} <it-noreply@{self._domain}>"
        # Reply-To is the attacker's external address — the tell
        attacker_reply_to = f"{subject.name.lower()}@{random.choice(['gmail.com', 'outlook.com', 'protonmail.com'])}"

        subject_lines = [
            "Action required: Verify your account credentials by EOD",
            "Urgent: Unusual sign-in detected on your account",
            "IT Security: MFA reconfiguration required — 24hr window",
            "HR: Please confirm your direct deposit details for Q2 update",
            "Security alert: Your password expires in 2 hours",
        ]
        bodies = [
            (
                f"Hi {target},\n\n"
                f"We've detected unusual activity on your account and need you to "
                f"re-verify your credentials. Please click the link below within "
                f"the next 2 hours to avoid account suspension:\n\n"
                f"https://accounts.{self._domain}.auth-verify.net/reset?token={_rand_hex(32)}\n\n"
                f"If you didn't trigger this, contact {pretext_role} immediately.\n\n"
                f"— {pretext_role}\n{COMPANY_NAME}"
            ),
            (
                f"Hi {target},\n\n"
                f"As part of our quarterly security audit, all employees must "
                f"re-authenticate their MFA device before Friday. Failure to do so "
                f"will result in temporary account lockout.\n\n"
                f"Please reply to this email with your employee ID and current MFA "
                f"backup code so we can re-provision your device.\n\n"
                f"Thank you,\n{pretext_role}"
            ),
        ]

        send_hour = random.randint(8, 11)  # business hours — less suspicious
        send_ts = current_date.replace(
            hour=send_hour, minute=random.randint(0, 59), second=0, microsecond=0
        )

        try:
            msg = MIMEMultipart("alternative")
            msg["From"] = spoofed_from_name
            msg["To"] = f"{target} <{target.lower()}@{self._domain}>"
            msg["Reply-To"] = attacker_reply_to
            msg["Subject"] = random.choice(subject_lines)
            msg["Date"] = send_ts.strftime("%a, %d %b %Y %H:%M:%S +0000")
            msg["Message-ID"] = f"<se_{_rand_hex(12)}@auth-verify.net>"
            # X-Originating-IP is from outside the corporate range — another tell
            msg["X-Originating-IP"] = _fake_residential_ip()
            msg.attach(MIMEText(random.choice(bodies), "plain"))

            inbound_dir = self._export_base / "emails" / "inbound"
            inbound_dir.mkdir(parents=True, exist_ok=True)
            eml_path = inbound_dir / f"se_phish_{target.lower()}_{day}.eml"
            with open(eml_path, "w") as fh:
                fh.write(msg.as_string())
        except Exception as exc:
            logger.warning(f"[security] spear_phishing email write failed: {exc}")
            return None

        self._pending_telemetry.append(
            TelemetryRecord(
                record_type="email_send",  # appears as inbound mail, not a send
                day=day,
                date=date_str,
                timestamp=send_ts.isoformat(),
                actor=subject.name,
                details={
                    "pattern": "spear_phishing",
                    "target": target,
                    "spoofed_sender": spoofed_from_name,
                    "reply_to": attacker_reply_to,
                    "originating_ip": msg["X-Originating-IP"],
                    "eml_path": str(eml_path),
                    "send_hour": send_hour,
                    # Observable tell: Reply-To domain differs from From domain
                    "reply_to_domain_mismatch": True,
                },
                _true_positive=True,
                _threat_class=subject.threat_class,
                _behavior="social_engineering",
            )
        )
        return {
            "pattern": "spear_phishing",
            "target": target,
            "eml_path": str(eml_path),
        }

    def _inject_slack_pretext(
        self,
        subject: ThreatSubjectConfig,
        target: str,
        day: int,
        current_date: datetime,
        date_str: str,
    ) -> Optional[Dict]:
        """
        Appends a Slack DM to the 'direct-messages' channel impersonating IT
        support.  Timed to coincide with a real incident when possible —
        that's when employees are most likely to comply with an urgent request.

        CLOCK NOTE: Same SimClock bypass rationale as unusual_hours_access.
        The message is written directly to the pending telemetry list without
        going through advance_actor(), so the subject's business-hours cursor
        is unaffected.
        """
        # Time the pretext to land during a busy window — mid-morning or
        # right after lunch, when people are context-switching
        send_hour = random.choice([9, 10, 13, 14])
        send_ts = current_date.replace(
            hour=send_hour, minute=random.randint(5, 55), second=0, microsecond=0
        )

        pretext_messages = [
            (
                f"Hi {target} — IT here. We're seeing repeated failed auth attempts "
                f"on your account from an unrecognized device. Can you DM me your "
                f"current session token so I can invalidate the rogue session? "
                f"Needs to happen in the next 15 min before it auto-locks."
            ),
            (
                f"Hey {target}, Security team. We're doing an emergency MFA audit "
                f"following today's incident. Can you share your backup codes "
                f"so I can verify your device is enrolled correctly? Won't take 2 min."
            ),
            (
                f"{target} — heads up, there's a payroll discrepancy for your account "
                f"in the HR system. Finance asked me to collect your employee ID and "
                f"last 4 of SSN to push a correction before the pay run closes. "
                f"Sorry for the weird ask over Slack — ticketing system is down."
            ),
        ]

        message = {
            "user": subject.name,  # appears as the subject's real name, not "IT"
            "email": f"{subject.name.lower()}@{self._domain}",
            "text": random.choice(pretext_messages),
            "ts": send_ts.isoformat(),
            "thread_ts": send_ts.isoformat(),
            "day": day,
            "channel": "direct-messages",
            "_security_injected": True,
            "is_bot": False,
            # The structural tell: subject messaging a target they have no normal
            # work relationship with. Graph edge weight between them should be low.
            "_target": target,
        }

        self._pending_telemetry.append(
            TelemetryRecord(
                record_type="slack_message",
                day=day,
                date=date_str,
                timestamp=send_ts.isoformat(),
                actor=subject.name,
                details={
                    "pattern": "slack_pretexting",
                    "target": target,
                    "channel": "direct-messages",
                    "send_hour": send_hour,
                    # Low graph-edge weight between sender and target is the signal.
                    # Agents with access to the relationship graph can flag this;
                    # agents reading only message content likely cannot.
                    "sender_target_relationship": "weak",
                    "impersonates_role": "IT/Security",
                },
                _true_positive=True,
                _threat_class=subject.threat_class,
                _behavior="social_engineering",
            )
        )
        return {"pattern": "slack_pretexting", "target": target, "message": message}

    def _inject_vishing(
        self,
        subject: ThreatSubjectConfig,
        target: str,
        day: int,
        current_date: datetime,
        date_str: str,
    ) -> Optional[Dict]:
        """
        Records a phone_call event in the telemetry stream followed within
        minutes by an idp_auth success on the target's account.

        No Slack or email artifact is produced — the only detection signal
        is temporal proximity of the call record to the authentication event,
        combined with the auth originating from an IP that doesn't match the
        target's known device pool.

        This is the hardest pattern to detect: the auth looks legitimate
        (correct credentials, correct MFA), and the call record is innocuous
        in isolation.  Correlation requires a time-window join across two
        record types for the *target*, not the subject.
        """
        call_hour = random.randint(9, 16)
        call_ts = current_date.replace(
            hour=call_hour, minute=random.randint(0, 45), second=0, microsecond=0
        )
        # Auth happens 5-25 minutes after the call — the window of compliance
        auth_delay_mins = random.randint(5, 25)
        auth_ts = call_ts + timedelta(minutes=auth_delay_mins)

        # The auth originates from the subject's device/IP, not the target's
        # known device — this is the IP anomaly that ties the events together
        subject_device = random.choice(
            self._employee_devices.get(
                subject.name, _seed_employee_devices(subject.name)
            )
        )
        target_devices = self._employee_devices.get(
            target, _seed_employee_devices(target)
        )

        # Phone call record — appears as a normal telephony log entry
        self._pending_telemetry.append(
            TelemetryRecord(
                record_type="phone_call",
                day=day,
                date=date_str,
                timestamp=call_ts.isoformat(),
                actor=subject.name,
                details={
                    "pattern": "vishing",
                    "call_direction": "outbound",
                    "target": target,
                    "call_duration_seconds": random.randint(120, 480),
                    "call_hour": call_hour,
                    "spoofed_caller_id": f"+1-800-{random.randint(100, 999)}-{random.randint(1000, 9999)}",
                    # The tell: spoofed CID doesn't match any known vendor/contact
                    "caller_id_in_known_contacts": False,
                },
                _true_positive=True,
                _threat_class=subject.threat_class,
                _behavior="social_engineering",
            )
        )

        # Subsequent auth event on the TARGET's account — not the subject's
        # This record has actor=target, making it appear as a normal target login.
        # Ground truth ties it back via _behavior="social_engineering".
        self._pending_telemetry.append(
            TelemetryRecord(
                record_type="idp_auth",
                day=day,
                date=date_str,
                timestamp=auth_ts.isoformat(),
                actor=target,  # NOTE: target's account, not the subject's
                details={
                    "auth_result": "success",
                    "dst_app": random.choice(["aws-console", "github-enterprise"]),
                    # Auth from subject's IP/device, not target's known device pool
                    "src_ip": _fake_residential_ip(),
                    "device_id": subject_device["device_id"],
                    "device_os": subject_device["os"],
                    "mfa_method": subject_device["mfa_method"],
                    "user_agent": subject_device["user_agent"],
                    "new_device": subject_device["device_id"]
                    not in [d["device_id"] for d in target_devices],
                    "anomalous_ip": True,
                    "ip_type": "residential",
                    "access_hour": auth_ts.hour,
                    "outside_business_hours": False,
                    "corroborating_activity_expected": False,
                    # Correlation field: join this auth with the phone_call record
                    # from auth_delay_mins ago for the same day/actor pair
                    "preceded_by_call_record": True,
                    "call_to_auth_gap_minutes": auth_delay_mins,
                },
                _true_positive=True,
                _threat_class=subject.threat_class,
                _behavior="social_engineering",
            )
        )
        return {
            "pattern": "vishing_breadcrumb",
            "target": target,
            "call_ts": call_ts.isoformat(),
            "auth_ts": auth_ts.isoformat(),
            "auth_delay_mins": auth_delay_mins,
        }

    def _inject_trust_building(
        self,
        subject: ThreatSubjectConfig,
        target: str,
        day: int,
        current_date: datetime,
        date_str: str,
    ) -> Optional[Dict]:
        """
        Stage 1 of a two-stage social engineering attempt.

        Injects a benign-looking inbound email that establishes no immediate
        signal — a project check-in, a vendor question, a conference follow-up.
        Records a 'trust_building' telemetry event with a 'followup_due_day'
        field 3-5 days out.

        The caller (flow.py) should store the returned followup_due_day and
        call inject_social_engineering() again on that day, which will fire
        spear_phishing or slack_pretexting against the same target.

        This pattern forces detection agents to maintain a memory window longer
        than a single-day triage pass — the trust-building contact is a clean
        false negative if evaluated in isolation.
        """
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart

        followup_day = day + random.randint(3, 5)
        send_hour = random.randint(9, 15)
        send_ts = current_date.replace(
            hour=send_hour, minute=random.randint(0, 59), second=0, microsecond=0
        )

        # Benign pretexts — indistinguishable from real external contact
        benign_subjects = [
            "Following up from last week's sync",
            "Quick question about your team's API docs",
            "Intro — reaching out re potential partnership",
            "Loved your talk at the conference — connection request",
            "Re: vendor renewal — a few questions",
        ]
        benign_bodies = [
            (
                f"Hi {target},\n\n"
                f"Great connecting at the recent event. I had a follow-up question "
                f"about your team's integration approach — would love 20 minutes "
                f"sometime this week if you have availability.\n\n"
                f"Best,\n{subject.name.split()[0]}"
            ),
            (
                f"Hi {target},\n\n"
                f"I came across your team's public API docs and had a few questions "
                f"about rate limiting on the /export endpoint. Happy to schedule a "
                f"call or just reply here if it's quick.\n\nThanks"
            ),
        ]

        external_domain = random.choice(
            ["techpartners.io", "ventureco.com", "synapse-labs.net"]
        )
        sender_addr = f"{subject.name.lower().replace(' ', '.')}@{external_domain}"

        try:
            msg = MIMEMultipart("alternative")
            msg["From"] = f"{subject.name} <{sender_addr}>"
            msg["To"] = f"{target} <{target.lower()}@{self._domain}>"
            msg["Subject"] = random.choice(benign_subjects)
            msg["Date"] = send_ts.strftime("%a, %d %b %Y %H:%M:%S +0000")
            msg["Message-ID"] = f"<tb_{_rand_hex(12)}@{external_domain}>"
            msg.attach(MIMEText(random.choice(benign_bodies), "plain"))

            inbound_dir = self._export_base / "emails" / "inbound"
            inbound_dir.mkdir(parents=True, exist_ok=True)
            eml_path = inbound_dir / f"se_trust_{target.lower()}_{day}.eml"
            with open(eml_path, "w") as fh:
                fh.write(msg.as_string())
        except Exception as exc:
            logger.warning(f"[security] trust_building email write failed: {exc}")
            return None

        self._pending_telemetry.append(
            TelemetryRecord(
                record_type="email_send",
                day=day,
                date=date_str,
                timestamp=send_ts.isoformat(),
                actor=subject.name,
                details={
                    "pattern": "trust_building",
                    "target": target,
                    "sender_domain": external_domain,
                    # This contact is NOT in the known external_contacts list —
                    # the only tell at this stage
                    "sender_in_known_contacts": False,
                    "followup_due_day": followup_day,
                    "eml_path": str(eml_path),
                },
                _true_positive=True,
                _threat_class=subject.threat_class,
                _behavior="social_engineering",
            )
        )
        return {
            "pattern": "trust_building",
            "target": target,
            "followup_due_day": followup_day,
            "eml_path": str(eml_path),
        }

    def reset_behavior_cooldown(self, behavior: str) -> None:
        for subject in self._subjects.values():
            if behavior in subject.behaviors:
                subject._fired_behaviors[behavior] = -999


# ─────────────────────────────────────────────────────────────────────────────
# NULL OBJECT — returned when insider_threat.enabled is false
# Every method is a safe no-op so flow.py needs zero guard clauses.
# ─────────────────────────────────────────────────────────────────────────────


class _NullInjector:
    """
    Drop-in replacement for InsiderThreatInjector when the feature is disabled.
    Implements the full public API with no-op methods so callers never need to
    check ``if injector is not None``.
    """

    def begin_day(self, day: int, state) -> None:
        pass

    def end_day(self, day: int, state, mem, clock, date_str: str) -> list:
        return []

    def inject_pr(self, pr: dict, author: str, day: int) -> dict:
        return pr

    def inject_slack(
        self, messages: list, channel: str, day: int, current_date
    ) -> list:
        return messages

    def inject_email(
        self,
        eml_path: str,
        sender: str,
        recipients: list,
        subject_line: str,
        day: int,
        current_date,
    ) -> None:
        return None

    def inject_jira_access(
        self, accessor, ticket_id, ticket_dept, accessor_dept, day, current_date
    ) -> None:
        pass

    def inject_repo_clone(
        self, actor: str, repo_count: int, day: int, current_date
    ) -> None:
        pass

    def inject_host_hoarding(self, actor: str, day: int, current_date) -> None:
        return None

    def is_active(self, name: str, behavior: str, day: int) -> bool:
        return False

    def active_subject_names(self) -> set:
        return set()

    def inject_social_engineering(
        self, day: int, current_date, active_names: list
    ) -> list:
        return []

    def reset_behavior_cooldown(self, behavior: str) -> None:
        pass
