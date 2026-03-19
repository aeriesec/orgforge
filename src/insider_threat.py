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

    dlp_noise_ratio: 0.4
    # Fraction of dlp_alert events that are false positives (innocent employees).
    # Only relevant in "active" mode.  Range 0.0–1.0.

    telemetry_dir: "security_telemetry"
    # Subdirectory under the simulation export dir for telemetry output.

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
import re
import string
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

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

    record_type: str  # "repo_access" | "commit" | "email_send" | "dlp_alert"
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
# BEHAVIOR REGISTRY
# Each behavior is a plain function: (injector, subject, context) → side-effect
# Context is a dict assembled per call-site (keys vary by surface).
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
        "data_exfil_email": 5,  # rare — stands out when it fires
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
    ):
        self._subjects: Dict[str, ThreatSubjectConfig] = {s.name: s for s in subjects}
        self._all_names = all_names
        self._innocent_names = [n for n in all_names if n not in self._subjects]
        self._mode = mode  # "passive" | "active"
        self._noise_ratio = dlp_noise_ratio
        self._telemetry_dir = telemetry_dir
        self._export_base = export_base
        self._domain = domain

        # Pending telemetry records, flushed at end_day()
        self._pending_telemetry: List[TelemetryRecord] = []
        # Pending SimEvents to fire (returned from end_day())
        self._pending_sim_events: List[Any] = []

        telemetry_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"[security] ✓ InsiderThreatInjector active — "
            f"mode={mode}, subjects={list(self._subjects.keys())}, "
            f"noise={dlp_noise_ratio:.0%}"
        )

    # ─── FACTORY ─────────────────────────────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        config: dict,
        export_base: str | Path,
        all_names: List[str],
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

        1. Flushes pending telemetry records to JSONL files.
        2. In "active" mode: emits dlp_alert SimEvents (with noise).
        3. Returns a list of SimEvent objects to be logged by flow.py.
        """
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
                            msg["text"], subject.threat_class
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
                                    "behavior": "sentiment_drift",
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
            # Note also: SimClock.sync_and_tick() has an allow_after_hours=True flag —
            # do NOT use it here.  It only suppresses the overflow handler for
            # causal meeting chains; it does not model genuine off-hours activity
            # and would still advance the cursor into an incoherent state.
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

    def _apply_sentiment_drift(self, text: str, threat_class: str) -> str:
        """
        Modifies a Slack message to reflect the subject's emotional state.
        Disgruntled → negative/passive-aggressive tone markers.
        Malicious   → deliberately neutral (they're hiding intent).
        """
        if not text:
            return text

        if threat_class == "disgruntled":
            prefix = random.choice(self._DRIFT_PREFIXES_DISGRUNTLED)
            suffix = random.choice(self._DRIFT_SUFFIXES_DISGRUNTLED)
            # Capitalise prefix if it follows a sentence boundary
            if prefix:
                drifted = prefix + text[0].lower() + text[1:] + suffix
            else:
                drifted = text + suffix
            return drifted

        if threat_class == "malicious":
            prefix = random.choice(self._DRIFT_PREFIXES_MALICIOUS)
            suffix = random.choice(self._DRIFT_SUFFIXES_MALICIOUS)
            return (prefix + text + suffix).strip()

        return text  # negligent — no deliberate tone change

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
        }
        return _MAP.get(behavior, "DLP-199: General anomaly")

    @staticmethod
    def _severity_for(behavior: str) -> str:
        _HIGH = {"secret_in_commit", "data_exfil_email", "excessive_repo_cloning"}
        _MED = {"unusual_hours_access", "cross_dept_snooping"}
        if behavior in _HIGH:
            return "high"
        if behavior in _MED:
            return "medium"
        return "low"

    # ─── PRIVATE — TELEMETRY FLUSH ────────────────────────────────────────────

    def _flush_telemetry(self, day: int, date_str: str) -> None:
        """
        Write today's telemetry records to two JSONL files:

        security_telemetry/access_log.jsonl
            — observable stream (no ground-truth fields)
            — what a detection agent would ingest

        security_telemetry/_ground_truth.jsonl
            — full records including true_positive, threat_class, behavior
            — structurally separate so agents can't naively read labels
            — prefixed with _ to signal "not part of the detection corpus"
        """
        if not self._pending_telemetry:
            return

        obs_path = self._telemetry_dir / "access_log.jsonl"
        gt_path = self._telemetry_dir / "_ground_truth.jsonl"

        with open(obs_path, "a") as obs_f, open(gt_path, "a") as gt_f:
            for rec in self._pending_telemetry:
                # Observable record — strip ground-truth fields
                observable = {
                    "record_type": rec.record_type,
                    "day": rec.day,
                    "date": rec.date,
                    "timestamp": rec.timestamp,
                    "actor": rec.actor,
                    **rec.details,
                }
                obs_f.write(json.dumps(observable) + "\n")

                # Ground-truth record — includes everything
                ground_truth = {
                    **observable,
                    "true_positive": rec._true_positive,
                    "threat_class": rec._threat_class,
                    "behavior": rec._behavior,
                }
                gt_f.write(json.dumps(ground_truth) + "\n")

        logger.debug(
            f"[security] 📝 Flushed {len(self._pending_telemetry)} telemetry "
            f"records for day {day}"
        )
        self._pending_telemetry.clear()


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

    def is_active(self, name: str, behavior: str, day: int) -> bool:
        return False

    def active_subject_names(self) -> set:
        return set()
