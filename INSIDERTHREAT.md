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

  subjects:
    - name: "Jordan"          # must match a name in org_chart
      threat_class: "negligent"
      onset_day: 5            # clean behavior on days 1–4
      behaviors:
        - "secret_in_commit"

    - name: "Riley"
      threat_class: "disgruntled"
      onset_day: 12
      behaviors:
        - "sentiment_drift"
        - "cross_dept_snooping"
        - "unusual_hours_access"

    - name: "Morgan"
      threat_class: "malicious"
      onset_day: 18
      behaviors:
        - "data_exfil_email"
        - "excessive_repo_cloning"
        - "unusual_hours_access"
        - "sentiment_drift"

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

| Field | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `false` | Master toggle. Must be `true` to activate the module. |
| `mode` | string | `"passive"` | `"passive"` or `"active"`. See above. |
| `subjects` | list | `[]` | One entry per threat subject. See below. |
| `dlp_noise_ratio` | float | `0.4` | False positive rate for DLP alerts. Active mode only. |
| `telemetry_dir` | string | `"security_telemetry"` | Telemetry output subdirectory name. |

### Subject fields

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | string | yes | Must match an employee name in `org_chart`. |
| `threat_class` | string | yes | `"negligent"`, `"disgruntled"`, or `"malicious"`. |
| `onset_day` | int | yes | Day number when behaviors begin. Clean baseline before this. |
| `behaviors` | list | yes | One or more behavior names. See [Behavior reference](#behavior-reference). |

---

## Threat classes

Threat class controls how behaviors are expressed, not which behaviors fire. A subject
with `sentiment_drift` and `threat_class: disgruntled` produces passive-aggressive
Slack messages. The same subject with `threat_class: malicious` produces deliberately
neutral messages — they are hiding their intent.

| Class | Behavioral profile | Detection difficulty |
|---|---|---|
| `negligent` | Accidental, no concealment. Single behavior typical. | Low — clear artifact signal |
| `disgruntled` | Emotional drift, access anomalies, no direct exfil. | Medium — requires multi-day sentiment correlation |
| `malicious` | Deliberate, active concealment, multiple surfaces. | High — neutral Slack, anomalous email/git |

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

- **Surface:** `security_telemetry/access_log.jsonl` only
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

- **Surface:** `security_telemetry/access_log.jsonl`
- **What it does:** Records when the subject reads tickets outside their own
  department. Requires `flow.py` to call `injector.inject_jira_access()` at ticket
  read sites. The SimEvent log shows normal ticket activity — the anomaly is only
  visible in the telemetry stream.
- **Valid for:** `malicious`, `disgruntled`
- **Cooldown:** 2 days

### `data_exfil_email`

- **Surface:** `export/emails/outbound/` + `security_telemetry/access_log.jsonl`
- **What it does:** Writes a separate `.eml` file from the subject to a personal
  email address (gmail, protonmail, outlook, yahoo) containing a vague internal data
  reference. The email is sent at off-hours (22:00–01:00 sim time). It is embedded
  in Memory as a normal email artifact, making it findable by agents that search the
  artifact corpus. Does not advance the subject's SimClock cursor.
- **Valid for:** `malicious`
- **Cooldown:** 5 days (probabilistic — fires on ~50% of eligible days)

---

## Output files

All output is written under the simulation's export directory.

```
export/
├── git/prs/                     # PR JSON — may contain injected secrets
├── emails/outbound/             # outbound .eml files — may contain exfil emails
├── slack/                       # Slack JSONL — may contain off-hours messages
└── security_telemetry/
    ├── access_log.jsonl         # observable telemetry stream — NO labels
    └── _ground_truth.jsonl      # full records WITH labels — held out for eval
```

### `access_log.jsonl`

One JSON record per line. Each record is an observable security event:

```json
{
  "record_type": "commit",
  "day": 8,
  "date": "2024-01-10",
  "timestamp": "2024-01-10T14:32:11",
  "actor": "Jordan",
  "pr_id": "PR-112",
  "ticket_id": "JIRA-23",
  "secret_var": "AWS_SECRET_ACCESS_KEY",
  "commit_hour": 14
}
```

This is the stream a detection agent ingests. It contains no `true_positive`,
`threat_class`, or `behavior` fields. Actor names appear as normal employee names
with no threat annotation.

### `_ground_truth.jsonl`

Same structure as `access_log.jsonl` with three additional fields:

```json
{
  "record_type": "commit",
  "day": 8,
  "actor": "Jordan",
  "secret_var": "AWS_SECRET_ACCESS_KEY",
  "true_positive": true,
  "threat_class": "negligent",
  "behavior": "secret_in_commit"
}
```

The `_` prefix signals this file is not part of the detection corpus. Evaluation
tooling reads it; detection agents do not.

### `dlp_alert` SimEvents (active mode only)

When `mode: "active"`, the module emits `dlp_alert` SimEvents into the main
OrgForge event log. These appear alongside normal events (`standup`, `ticket_progress`,
etc.) with no structural distinction. Policy rule strings reference the DLP policy
table:

| Policy | Behavior |
|---|---|
| DLP-101 | Credential pattern in code diff |
| DLP-102 | Repository access outside business hours |
| DLP-103 | Bulk repository clone volume |
| DLP-104 | Communication pattern anomaly |
| DLP-105 | Cross-department data access |
| DLP-106 | Sensitive data forwarded to personal email |

False positive events fire for innocent employees with `severity: "low"` and
policy rules DLP-107 through DLP-110. These are structurally identical to true
positive alerts. The `true_positive` flag exists only in `_ground_truth.jsonl`.

---

## Detection agent training

### Corpus structure for agents

An agent investigating a simulation run has read access to:

- `security_telemetry/access_log.jsonl` — telemetry stream
- `export/git/prs/*.json` — PR artifacts
- `export/slack/*.json` — Slack message threads
- `export/emails/**/*.eml` — all email artifacts
- OrgForge SimEvent log — full event history via query

An agent does **not** have access to `_ground_truth.jsonl` during investigation.

### Recommended pipeline

A three-stage CrewAI pipeline maps naturally onto the output:

**Stage 1 — Triage.** A lightweight agent reads `access_log.jsonl` in 7-day sliding
windows. Flags employees with 2+ signals in the same window for escalation. This is
the noise-reduction layer and should run on a smaller model.

**Stage 2 — Correlation.** For each flagged employee, an agent with separate tools per
artifact surface (PR reader, Slack reader, email reader, telemetry reader) builds a
multi-signal case. The manager agent decides which surfaces to check first based on
the triage output.

**Stage 3 — Verdict.** A synthesis agent produces a structured JSON verdict:

```json
{
  "name": "Jordan",
  "verdict": "likely_threat",
  "threat_class": "negligent",
  "behaviors_observed": ["secret_in_commit"],
  "evidence": [
    {
      "source": "pr",
      "day": 8,
      "description": "PR-112 description contains AWS_SECRET_ACCESS_KEY pattern"
    }
  ],
  "recommended_action": "Rotate credential, schedule security training",
  "confidence": "high"
}
```

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

---

## Files changed

| File | Change type | Description |
|---|---|---|
| `insider_threat.py` | New file | Core module — injector, behaviors, telemetry |
| `flow.py` | Modified | Import, `__init__` construction, `daily_cycle` hooks, standup Slack injection, PR injection |
| `external_email_ingest.py` | Modified | `threat_injector` parameter, post-`_write_eml` inject calls in `_send_customer_reply`, `_send_vendor_ack`, `_send_hr_outbound` |
| `planner_models.py` | Modified | Added `dlp_alert` and `secret_detected` to `KNOWN_EVENT_TYPES` |
| `plan_validator.py` | Modified | Added `dlp_alert: 1` and `secret_detected: 999` to `_COOLDOWN_DAYS` |

### Integration order for `flow.py`

`self._threat` must be constructed before `self._git` and `self._email_ingestor`
in `Flow.__init__`. The correct order:

```python
self._threat = InsiderThreatInjector.from_config(...)  # first
self._git = GitSimulator(..., threat_injector=self._threat)
self._email_ingestor = ExternalEmailIngestor(..., threat_injector=self._threat)
```
