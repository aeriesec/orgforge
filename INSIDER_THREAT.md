# Insider Threat Module

OrgForge includes an optional insider threat simulation layer that injects realistic
anomalous behavior into the normal simulation flow. The module is designed to produce
labeled training corpora for security detection agents.

It is **completely inert by default**. Enabling it requires an explicit config entry.
When disabled, the simulation behaves identically to a standard OrgForge run — no
performance overhead, no additional output, no code paths activated.

---

## Table of contents

- [Design principles](#design-principles)
- [Quick start](#quick-start)
- [Config reference](#config-reference)
- [Threat classes](#threat-classes)
- [Behavior reference](#behavior-reference)
- [IDP logs](#idp-logs)
- [Industry-standard log formats](#industry-standard-log-formats)
- [Output files](#output-files)
- [Detection agent training](#detection-agent-training)
- [Files changed](#files-changed)

---

## Design principles

**Behaviors, not labels.** No artifact ever contains the word "malicious" or any
annotation identifying the subject as a threat. Detection agents must earn the signal
by correlating across artifact types, just as a real analyst would.

**Surface reuse.** Every artifact the module produces — PR descriptions, Slack messages,
email files, JIRA access records — flows through the existing OrgForge artifact
pipeline. The module influences content at injection points; it never bypasses normal
event machinery.

**Temporal onset.** Subjects behave normally before their `onset_day`. Behavioral data
from day 1 to `onset_day - 1` is a clean negative example baseline. Detection agents
that over-index on early data will produce false positives.

**Noise injection.** In `active` mode, synthetic DLP alerts fire for innocent employees
at a configurable rate. Agents must learn to demand multi-signal correlation rather
than reacting to single events.

**Ground truth separation.** Observable telemetry and ground truth labels are written
to separate files. The labels file is prefixed with `_` to signal that it is not part
of the detection corpus.

---

## Quick start

1. Add the `insider_threat` block to your `config.yaml` (see [Config reference](#config-reference)).
2. Set `enabled: true` and configure at least one subject.
3. Run OrgForge normally — the module activates automatically.
4. Inspect output in `export/security_telemetry/`.

To run with the feature off (default), no config entry is needed at all.

---

## Config reference

Add this block to your existing `config.yaml`. Every field except `enabled` is optional
and has a sensible default.

```yaml
insider_threat:
  # Master toggle. false = completely inert, zero overhead.
  # Default: false
  enabled: false

  # Simulation mode.
  #   passive — behaviors injected into normal artifacts only.
  #             No extra SimEvents emitted. Use this to produce a
  #             realistic corpus without SIEM noise.
  #   active  — additionally emits dlp_alert SimEvents with false
  #             positives mixed in at dlp_noise_ratio. Use this to
  #             train agents to triage alert streams.
  # Default: "passive"
  mode: "passive"

  # Log format for the observable telemetry stream.
  #   jsonl  — custom JSONL (default, backward compatible)
  #   cef    — Common Event Format (ArcSight, Splunk Universal Forwarder)
  #   ecs    — Elastic Common Schema v8.x (Elastic SIEM, OpenSearch)
  #   leef   — Log Event Extended Format 2.0 (IBM QRadar)
  #   all    — write all four formats side-by-side
  # Ground truth is always written as JSONL regardless of this setting.
  # Default: "jsonl"
  log_format: "jsonl"

  # Emit IDP authentication events for all active employees each day.
  # When true, every employee gets realistic SSO auth records; threat subjects
  # additionally receive anomalous IDP events (off-hours, new device, ghost logins).
  # Default: true
  idp_logs: true

  subjects:
    - name: "Jordan" # must match a name in org_chart
      threat_class: "negligent"
      onset_day: 5 # clean behavior on days 1–4
      behaviors:
        - "secret_in_commit"

    - name: "Riley"
      threat_class: "disgruntled"
      onset_day: 12
      behaviors:
        - "sentiment_drift"
        - "cross_dept_snooping"
        - "unusual_hours_access"
        - "host_data_hoarding"

    - name: "Morgan"
      threat_class: "malicious"
      onset_day: 18
      behaviors:
        - "data_exfil_email"
        - "excessive_repo_cloning"
        - "unusual_hours_access"
        - "sentiment_drift"
        - "host_data_hoarding"

  # Fraction of dlp_alert events that are false positives (innocent employees).
  # Only applies in mode: "active".
  # 0.0 = every alert is real     (easy — not realistic)
  # 0.4 = 40% noise               (recommended starting point)
  # 0.8 = very noisy              (trains agents to demand multi-signal evidence)
  # Default: 0.4
  dlp_noise_ratio: 0.4

  # Subdirectory under the simulation export dir for telemetry output.
  # Default: "security_telemetry"
  telemetry_dir: "security_telemetry"
```

### Field reference

| Field             | Type   | Default                | Description                                              |
| ----------------- | ------ | ---------------------- | -------------------------------------------------------- |
| `enabled`         | bool   | `false`                | Master toggle. Must be `true` to activate the module.    |
| `mode`            | string | `"passive"`            | `"passive"` or `"active"`. See above.                    |
| `log_format`      | string | `"jsonl"`              | Output format for the observable telemetry stream.       |
| `idp_logs`        | bool   | `true`                 | Emit IDP authentication events for all active employees. |
| `subjects`        | list   | `[]`                   | One entry per threat subject. See below.                 |
| `dlp_noise_ratio` | float  | `0.4`                  | False positive rate for DLP alerts. Active mode only.    |
| `telemetry_dir`   | string | `"security_telemetry"` | Telemetry output subdirectory name.                      |

### Subject fields

| Field          | Type   | Required | Description                                                                |
| -------------- | ------ | -------- | -------------------------------------------------------------------------- |
| `name`         | string | yes      | Must match an employee name in `org_chart`.                                |
| `threat_class` | string | yes      | `"negligent"`, `"disgruntled"`, or `"malicious"`.                          |
| `onset_day`    | int    | yes      | Day number when behaviors begin. Clean baseline before this.               |
| `behaviors`    | list   | yes      | One or more behavior names. See [Behavior reference](#behavior-reference). |

---

## Threat classes

Threat class controls how behaviors are expressed, not which behaviors fire. A subject
with `sentiment_drift` and `threat_class: disgruntled` produces passive-aggressive
Slack messages. The same subject with `threat_class: malicious` produces deliberately
neutral messages — they are hiding their intent.

| Class         | Behavioral profile                                                                         | Detection difficulty                                    |
| ------------- | ------------------------------------------------------------------------------------------ | ------------------------------------------------------- |
| `negligent`   | Accidental, no concealment. Single behavior typical.                                       | Low — clear artifact signal                             |
| `disgruntled` | Emotional drift, access anomalies, ghost IDP logins, no direct exfil.                      | Medium — requires multi-day sentiment + IDP correlation |
| `malicious`   | Deliberate, active concealment, multiple surfaces including host hoarding + IDP anomalies. | High — neutral Slack, anomalous email/git/host          |

---

## Behavior reference

The following strings are valid entries under a subject's `behaviors:` list.

### `secret_in_commit`

- **Surface:** PR description (`export/git/prs/*.json`)
- **What it does:** Injects a realistic-looking fake credential into the PR body
  as a test fixture comment, inline env note, or diff snippet. Credential types
  include AWS access keys, GitHub tokens, database passwords, Stripe keys, SendGrid
  API keys, and Slack webhook URLs. All values are synthetically generated and are
  not real credentials.
- **Valid for:** `negligent`, `malicious`
- **Cooldown:** 4 days between firings

### `unusual_hours_access`

- **Surface:** Slack message threads
- **What it does:** Appends a low-key Slack message from the subject at an off-hours
  timestamp (01:00–03:00 or 22:00–23:00 sim time). The message text is innocuous.
  The signal is the timestamp. Does not advance the subject's SimClock cursor — the
  off-hours message is forensically isolated from their normal business-hours activity.
- **Valid for:** `malicious`, `disgruntled`
- **Cooldown:** 1 day (probabilistic — fires on ~35% of eligible days)

### `excessive_repo_cloning`

- **Surface:** `security_telemetry/access_log.*` only
- **What it does:** Records an anomalously high repository clone count in the
  telemetry stream. No artifact is written to the normal simulation outputs. Detection
  requires an agent that reads the telemetry stream, not just PR or Slack artifacts.
- **Valid for:** `malicious`
- **Cooldown:** 2 days

### `sentiment_drift`

- **Surface:** Slack messages (any channel)
- **What it does:** Modifies the subject's existing Slack messages before they are
  written to memory. Behavior depends on `threat_class`:
  - `disgruntled` — passive-aggressive prefixes and suffixes added to messages
  - `malicious` — messages are made deliberately neutral (subject is hiding intent)
- **Valid for:** `disgruntled`, `malicious`
- **Cooldown:** 1 day (fires on most active days once onset)

### `cross_dept_snooping`

- **Surface:** `security_telemetry/access_log.*`
- **What it does:** Records when the subject reads tickets outside their own
  department. Requires `flow.py` to call `injector.inject_jira_access()` at ticket
  read sites. The SimEvent log shows normal ticket activity — the anomaly is only
  visible in the telemetry stream.
- **Valid for:** `malicious`, `disgruntled`
- **Cooldown:** 2 days

### `data_exfil_email`

- **Surface:** `export/emails/outbound/` + `security_telemetry/access_log.*`
- **What it does:** Writes a separate `.eml` file from the subject to a personal
  email address (gmail, protonmail, outlook, yahoo) containing a vague internal data
  reference. The email is sent at off-hours (22:00–01:00 sim time). It is embedded
  in Memory as a normal email artifact, making it findable by agents that search the
  artifact corpus. Does not advance the subject's SimClock cursor.
- **Valid for:** `malicious`
- **Cooldown:** 5 days (probabilistic — fires on ~50% of eligible days)

### `host_data_hoarding`

- **Surface:** `security_telemetry/access_log.*`
- **What it does:** Simulates a **3-phase, multi-day data staging trail** on the
  employee's workstation. Detection agents must correlate across all three phases —
  and ideally against concurrent `data_exfil_email` or network transfer events — to
  build a complete exfiltration case.

  **Phase 1 — Bulk file copy:** The subject accesses 2–4 internal file shares during
  after-hours (18:00–22:00) and copies 15–80 files (50–800 MB) to a local staging
  directory under their user profile. The `source_shares`, `file_count`, `staged_path`,
  and `total_bytes` are recorded.

  **Phase 2 — Compression:** The staging directory is archived using a common tool
  (7z, WinRAR, tar, zip) into a single archive file. `archive_name`, `original_bytes`,
  `compressed_bytes`, and `compression_ratio` are recorded.

  **Phase 3 — Exfil staging:** The archive is moved to either a cloud-sync directory
  (Dropbox, Google Drive, OneDrive, iCloud) or a removable media path. The record
  includes a `hoarding_trail_start_day` field that ties this event back to Phase 1 —
  a breadcrumb for correlation.

  The phase 3 record should be joined with any concurrent `email_send` record from
  `data_exfil_email`, or any anomalous network transfer, to build the full exfil case.

- **Valid for:** `malicious`, `disgruntled`
- **Cooldown:** 3 days (Phase 1 is additionally probabilistic at 40%)

### `social_engineering`

- **Surface:** `export/emails/inbound/` + `security_telemetry/access_log.*`
- **What it does:** Simulates inbound social engineering attempts against active
  employees using one of four patterns, selected randomly per firing:
  - `spear_phishing` — crafted inbound `.eml` impersonating IT/HR with a spoofed
    From address. The tell is a mismatched Reply-To domain and an
    `X-Originating-IP` outside the corporate range.
  - `slack_pretexting` — direct Slack message from the subject impersonating IT
    support, timed to coincide with an active incident when possible. The tell is
    a low graph-edge weight between sender and target — they have no normal work
    relationship.
  - `vishing_breadcrumb` — a `phone_call` telemetry record followed 5–25 minutes
    later by an `idp_auth` success on the _target's_ account from the subject's
    device/IP. No Slack or email artifact is produced. The auth record is filed
    under the target's actor name, not the subject's — agents doing per-actor
    triage on the subject will miss it entirely.
  - `trust_building` — a benign inbound email that produces no immediate signal,
    with a `followup_due_day` field 3–5 days out. The cooldown is reset on that
    day so a follow-up attack fires against the same target. Forces agents to
    maintain a memory window longer than a single-day triage pass.
- **Valid for:** `malicious` only
- **Cooldown:** 6

---

## IDP logs

When `idp_logs: true` (the default), the module emits realistic Identity Provider
authentication events for **every active employee** at the end of each simulation day.
These serve as the authentication baseline that anomaly detection agents correlate
against other behavioral signals.

### Normal employee IDP events

Each non-subject employee receives 1–3 authentication events per day:

- Morning SSO authentication (08:00–10:00) to a random internal application
- Optional mid-day re-auth (12:00–15:00) simulating session expiry
- Known device ID, corporate IP, expected MFA method, standard user agent

These events establish the "quiet" baseline that makes anomalies detectable.

### Threat subject IDP anomalies

Anomalous IDP events are injected for active threat subjects on top of their normal
morning authentication.

**`malicious` subjects (~45% of active days):**

- Off-hours authentication (22:00–02:00)
- Source IP may be residential, VPN, or corporate (30% chance of non-corporate)
- 20% chance of an unrecognized `new_device` fingerprint
- `corroborating_activity_expected: false` — no Jira, Slack, or email should
  follow from this login. Agents that check for downstream activity will find nothing.

**`disgruntled` subjects (~30% of active days):**

- Ghost login at early morning or evening (06:00–07:00 or 19:00–21:00)
- Corporate IP and known device (disgruntled employees typically work from known
  locations — they are not hiding their origin)
- `ghost_login: true` flag in the observable record
- 15% chance of a subsequent MFA failure, simulating credential uncertainty
- Like malicious, `corroborating_activity_expected: false`

**`negligent` subjects:** No anomalous IDP events. Negligent insiders accidentally
leave credentials in code — their authentication pattern is normal.

### Authentication anomaly scenarios

IDP logs unlock detection scenarios that were not previously possible:

| Scenario                                | Signal to correlate                                                                          |
| --------------------------------------- | -------------------------------------------------------------------------------------------- |
| Ghost login — logged in but did nothing | `idp_auth` success with no Slack/Jira/email in same time window                              |
| Stealthy enumeration                    | `idp_auth` to `aws-console` + no EC2/S3 activity + `host_data_hoarding` phase 3 the next day |
| Unknown device + exfil                  | `new_device: true` + `data_exfil_email` within 48 hours                                      |
| Credential stuffing test                | Multiple `mfa_failure` records across different apps in a short window                       |
| Residential IP + off-hours              | `anomalous_ip: true` + `outside_business_hours: true` with no VPN indicator                  |

---

## Industry-standard log formats

The observable telemetry stream can be exported in four formats, configured via
`log_format` in `config.yaml`. This allows direct ingestion into common SIEM platforms
without writing a custom parser.

### JSONL (default)

```
{"record_type": "idp_auth", "day": 8, "actor": "Morgan", "auth_result": "success", ...}
```

Backward-compatible. Use with any tool that reads newline-delimited JSON.

### CEF — Common Event Format

```
CEF:0|OrgForge|InsiderThreatSim|1.0|ORGFORGE-IDP-AUTH|Identity Provider Authentication|5|rt=2026-03-12T01:14:00 suser=Morgan src=100.64.22.41 SimDay=8 ...
```

**Target platforms:** Splunk Universal Forwarder, ArcSight, any syslog-based SIEM.

Configure Splunk to monitor `access_log.cef` with `sourcetype=cef`.

### ECS — Elastic Common Schema v8.x

```json
{"@timestamp":"2026-03-12T01:14:00","event":{"kind":"event","category":["authentication"],"type":["start"],"action":"success"},"user":{"name":"Morgan","domain":"example.com"},...}
```

**Target platforms:** Elastic SIEM (Kibana), OpenSearch Security Analytics.

Index directly into Elasticsearch via Filebeat with `filebeat.inputs: [{type: filestream, paths: ["access_log_ecs.ndjson"]}]`. No ingest pipeline required — fields map to ECS natively.

### LEEF — Log Event Extended Format 2.0

```
LEEF:2.0|OrgForge|InsiderThreatSim|1.0|ORGFORGE_IDP_AUTH|	devTime=2026-03-12T01:14:00	usrName=Morgan	...
```

**Target platforms:** IBM QRadar.

Point a QRadar log source at `access_log.leef` with protocol `Syslog` and source type `LEEF`.

### Exporting all formats simultaneously

Set `log_format: "all"` to write all four files in parallel. Useful when onboarding
a new team that uses a mix of platforms, or for benchmark comparisons.

---

## Output files

All output is written under the simulation's export directory.

```
export/
├── git/prs/                       # PR JSON — may contain injected secrets
├── emails/outbound/               # outbound .eml files — may contain exfil emails
├── slack/                         # Slack JSONL — may contain off-hours messages
└── security_telemetry/
    ├── access_log.jsonl           # observable JSONL stream — NO labels (if log_format includes jsonl)
    ├── access_log.cef             # CEF syslog stream (if log_format includes cef)
    ├── access_log_ecs.ndjson      # ECS NDJSON (if log_format includes ecs)
    ├── access_log.leef            # LEEF 2.0 (if log_format includes leef)
    └── _ground_truth.jsonl        # full records WITH labels — always JSONL
```

### `access_log.jsonl` (observable stream)

One JSON record per line. Each record is an observable security event. This is the
stream a detection agent ingests. It contains no `true_positive`, `threat_class`, or
`behavior` fields.

**IDP record example:**

```json
{
  "record_type": "idp_auth",
  "day": 8,
  "date": "2026-03-12",
  "timestamp": "2026-03-12T01:14:22",
  "actor": "Morgan",
  "auth_result": "success",
  "dst_app": "aws-console",
  "src_ip": "100.64.22.41",
  "device_id": "device-a3f2b1c9-NEW",
  "device_os": "Windows 11 22H2",
  "mfa_method": "totp",
  "new_device": true,
  "anomalous_ip": true,
  "ip_type": "residential",
  "access_hour": 1,
  "outside_business_hours": true,
  "corroborating_activity_expected": false
}
```

**Host hoarding record example (Phase 3):**

```json
{
  "record_type": "host_event",
  "day": 22,
  "date": "2026-03-26",
  "timestamp": "2026-03-26T22:07:44",
  "actor": "Morgan",
  "action": "archive_move",
  "source_path": "C:\\Users\\morgan\\AppData\\Local\\Temp\\backup",
  "destination_path": "~/Dropbox/work-backup/backup_20260326.zip",
  "destination_type": "cloud_sync_dir",
  "archive_name": "backup_20260326.zip",
  "bytes_moved": 142000000,
  "cloud_sync_dir": "~/Dropbox/work-backup",
  "removable_media": false,
  "access_hour": 22,
  "outside_business_hours": true,
  "hoarding_trail_start_day": 20,
  "total_bytes_staged": 420000000
}
```

### `_ground_truth.jsonl`

Same structure as the observable stream with three additional fields appended to every
record:

```json
{
  "record_type": "idp_auth",
  "actor": "Morgan",
  "true_positive": true,
  "threat_class": "malicious",
  "behavior": "unusual_hours_access"
}
```

### `dlp_alert` SimEvents (active mode only)

When `mode: "active"`, the module emits `dlp_alert` SimEvents into the main
OrgForge event log. These appear alongside normal events (`standup`, `ticket_progress`,
etc.) with no structural distinction. Policy rule strings reference the DLP policy
table:

| Policy  | Behavior                                            |
| ------- | --------------------------------------------------- |
| DLP-101 | Credential pattern in code diff                     |
| DLP-102 | Repository access outside business hours            |
| DLP-103 | Bulk repository clone volume                        |
| DLP-104 | Communication pattern anomaly                       |
| DLP-105 | Cross-department data access                        |
| DLP-106 | Sensitive data forwarded to personal email          |
| DLP-112 | Suspicious inbound communication targeting employee |

False positive events fire for innocent employees with `severity: "low"` and
policy rules DLP-107 through DLP-110. These are structurally identical to true
positive alerts. The `true_positive` flag exists only in `_ground_truth.jsonl`.

---

## Detection agent training

### Corpus structure for agents

An agent investigating a simulation run has read access to:

- `security_telemetry/access_log.*` — telemetry stream (format depends on config)
- `export/git/prs/*.json` — PR artifacts
- `export/slack/*.json` — Slack message threads
- `export/emails/**/*.eml` — all email artifacts
- OrgForge SimEvent log — full event history via query

An agent does **not** have access to `_ground_truth.jsonl` during investigation.

### Recommended pipeline

A three-stage CrewAI pipeline maps naturally onto the output:

**Stage 1 — Triage.** A lightweight agent reads `access_log.*` in 7-day sliding
windows. Flags employees with 2+ signals in the same window for escalation. This is
the noise-reduction layer and should run on a smaller model.

**Stage 2 — Correlation.** For each flagged employee, an agent with separate tools per
artifact surface (PR reader, Slack reader, email reader, telemetry reader, IDP reader)
builds a multi-signal case. The manager agent decides which surfaces to check first
based on the triage output. The `hoarding_trail_start_day` field in host_event Phase 3
records is a deliberate cross-day correlation hint.

**Stage 3 — Verdict.** A synthesis agent produces a structured JSON verdict:

```json
{
  "name": "Morgan",
  "verdict": "likely_threat",
  "threat_class": "malicious",
  "behaviors_observed": [
    "data_exfil_email",
    "host_data_hoarding",
    "unusual_hours_access"
  ],
  "evidence": [
    {
      "source": "idp_auth",
      "day": 20,
      "description": "Authenticated from residential IP on new device at 01:14 with no downstream activity"
    },
    {
      "source": "host_event",
      "day": 20,
      "description": "Phase 1: 63 files (420 MB) copied from //fileserver01/HR/headcount to local staging"
    },
    {
      "source": "host_event",
      "day": 22,
      "description": "Phase 3: Archive moved to Dropbox sync directory — correlates with IDP anomaly 2 days prior"
    },
    {
      "source": "email",
      "day": 22,
      "description": "Off-hours email to morgan.personal@gmail.com with subject 'Backup - do not delete'"
    }
  ],
  "recommended_action": "Revoke AWS and GitHub access immediately, forensic image of workstation",
  "confidence": "high"
}
```

### IDP correlation queries

When evaluating agents, validate that they can answer:

1. **Ghost login detection:** "Find employees who authenticated via SSO on day X but
   produced zero Slack messages, Jira updates, or PRs between their login time and
   end of business."

2. **New device + exfil correlation:** "Find employees who used a previously unseen
   device fingerprint within 48 hours of sending email to an external address."

3. **Host trail reconstruction:** "Find `host_event` records with `action: archive_move`
   and trace back to the `action: bulk_file_copy` record from `hoarding_trail_start_day`
   for the same actor."

4. **Multi-surface convergence:** "Find actors with at least one signal from each of:
   IDP anomaly, host_event, and email_send in the same 5-day window."

### Evaluation

Score verdicts against `_ground_truth.jsonl`:

```python
def evaluate(agent_verdicts, ground_truth_path):
    ground_truth = {}
    with open(ground_truth_path) as f:
        for line in f:
            rec = json.loads(line)
            if rec["true_positive"]:
                ground_truth.setdefault(rec["actor"], set()).add(rec["behavior"])

    tp = fp = fn = 0
    for verdict in agent_verdicts:
        name = verdict["name"]
        is_flagged = verdict["verdict"] in ("suspicious", "likely_threat")
        is_actual  = bool(ground_truth.get(name))

        if is_flagged and is_actual:   tp += 1
        elif is_flagged:               fp += 1
        elif is_actual:                fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}
```

### Detection difficulty by design

The module is intentionally calibrated to make detection non-trivial:

- The pre-onset baseline period provides clean data that will produce false positives
  if an agent doesn't account for temporal onset
- `negligent` and early-stage `malicious` subjects produce nearly identical surface
  artifacts — differentiation requires behavioral history over multiple days
- `malicious` subjects with `sentiment_drift` maintain neutral Slack tone, removing
  the most obvious emotional signal
- `dlp_noise_ratio` fires false positive alerts for innocent employees, training
  agents not to react to single events
- `data_exfil_email` fires probabilistically and at off-hours — an agent that only
  reads business-hours artifacts will miss it
- `host_data_hoarding` spreads across 3 simulation days — an agent that only looks
  at single-day windows will never reconstruct the full trail
- IDP ghost logins are detectable only by **absence of corroborating activity** —
  agents must query for events that did _not_ happen, not just events that did
- `social_engineering` with `vishing_breadcrumb` files the post-auth event under
  the _target's_ actor name — an agent doing per-subject triage will never see it.
  Detection requires a time-window join across `phone_call` and `idp_auth` records
  for all employees, not just flagged subjects.
- `trust_building` is a clean false negative when evaluated in a single-day window.
  The first contact is indistinguishable from legitimate external mail. Only an agent
  that retains context across the `followup_due_day` gap will connect the two events.

---

## Files changed

| File                       | Change type | Description                                                                                                                               |
| -------------------------- | ----------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `insider_threat.py`        | Modified    | Added `LogFormatter` (CEF/ECS/LEEF), `host_data_hoarding` behavior (3-phase), IDP log emission, `_NullInjector.inject_host_hoarding` stub |
| `flow.py`                  | Modified    | Import, `__init__` construction, `daily_cycle` hooks, standup Slack injection, PR injection                                               |
| `external_email_ingest.py` | Modified    | `threat_injector` parameter, post-`_write_eml` inject calls                                                                               |
| `planner_models.py`        | Modified    | Added `dlp_alert`, `secret_detected`, `host_event`, `idp_auth` to `KNOWN_EVENT_TYPES`                                                     |
| `plan_validator.py`        | Modified    | Added `dlp_alert: 1`, `secret_detected: 999`, `host_event: 1`, `idp_auth: 1` to `_COOLDOWN_DAYS`                                          |

### New call site: `inject_host_hoarding`

Add one call per threat subject per day in `daily_cycle`, after `_normal_day.handle()` returns:

```python
# In daily_cycle(), after normal_day.handle() and email_ingestor calls:
for subject_name in self._threat.active_subject_names():
    result = self._threat.inject_host_hoarding(
        actor=subject_name,
        day=self.state.day,
        current_date=self.state.current_date,
    )
    if result:
        logger.debug(f"[security] host hoarding phase {result['phase']} fired for {subject_name}")
```

### Integration order for `flow.py`

`self._threat` must be constructed before `self._git` and `self._email_ingestor`
in `Flow.__init__`. The correct order:

```python
self._threat = InsiderThreatInjector.from_config(...)  # first
self._git = GitSimulator(..., threat_injector=self._threat)
self._email_ingestor = ExternalEmailIngestor(..., threat_injector=self._threat)
```

### SIEM ingestion quick-reference

| Platform           | Format     | File                    | Config hint                                   |
| ------------------ | ---------- | ----------------------- | --------------------------------------------- |
| Splunk             | CEF        | `access_log.cef`        | `sourcetype=cef` in inputs.conf               |
| Elastic SIEM       | ECS        | `access_log_ecs.ndjson` | Filebeat filestream input, no pipeline needed |
| IBM QRadar         | LEEF       | `access_log.leef`       | Log source type: LEEF, protocol: Syslog       |
| Microsoft Sentinel | CEF or ECS | either                  | CEF connector or custom table via DCR         |
| OpenSearch         | ECS        | `access_log_ecs.ndjson` | OpenSearch Dashboards Security Analytics      |
| Generic            | JSONL      | `access_log.jsonl`      | Any tool that reads NDJSON                    |
