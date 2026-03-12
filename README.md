# OrgForge

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)
[![Run Tests](https://github.com/aeriesec/orgforge/actions/workflows/tests.yml/badge.svg)](https://github.com/aeriesec/orgforge/actions/workflows/tests.yml)

### A deterministic corporate simulator for generating ground-truth ecosystems and evaluating enterprise AI agents

OrgForge simulates weeks of realistic enterprise activity — Confluence pages, JIRA tickets, Slack threads, Git PRs, emails, and server logs — grounded in an event-driven state machine so LLMs can't hallucinate facts out of sequence.

The dataset is the exhaust of a living simulation. Engineers leave mid-sprint, forcing deterministic incident handoffs and ticket reassignments. Knowledge gaps surface when under-documented systems break. New hires build their internal network through simulated collaboration. Stress propagates through a live, weighted social graph. Every artifact reflects the exact state of the org at the moment it was written.

---

## Why Does This Exist?

When building AI agents that reason over institutional knowledge, you need a realistic corpus to test against. The only widely-used corporate dataset is the Enron email corpus — 25 years old, legally sensitive, and covering one company in crisis.

OrgForge generates that corpus from scratch, parameterized to any company, industry, or org structure. LLMs write the prose, but the facts — who was on-call, which ticket was open, when the incident resolved, who just left the team — are strictly controlled by the state machine.

**The central design bet:** grounding LLM output in a deterministic event log makes the dataset actually useful for evaluating retrieval systems. You have ground truth about what happened, when, who was involved, and what the org's state was — so you can measure whether an agent surfaces the right context, not just plausible-sounding context.

---

## What the Output Looks Like

Here's what a slice of a real simulation produces. An incident fires on Day 8:

**`slack/channels/engineering-incidents.json`** — the alert arrives first, timestamped to the millisecond the on-call pager fired:

```json
{
  "ts": "2026-03-10T14:23:07",
  "user": "pagerduty-bot",
  "text": "🔴 P1 ALERT: TitanDB latency spike — connection pool exhaustion under load. On-call: Jax."
}
```

**`jira/IT-108.json`** — opened seconds later, facts pulled from the same SimEvent:

```json
{
  "id": "IT-108",
  "type": "incident",
  "priority": "P1",
  "title": "TitanDB: latency spike",
  "root_cause": "connection pool exhaustion under load",
  "assignee": "Jax",
  "reporter": "system",
  "opened": "2026-03-10T14:23:19"
}
```

**`confluence/postmortems/IT-108.md`** — written the next day, linking the same root cause and PR:

> _This incident was triggered by connection pool exhaustion under sustained load, first surfaced in IT-108. The fix landed in PR #47 (merged by Sarah). A prior knowledge gap in TitanDB connection management — stemming from Jordan's departure on Day 12 — contributed to the delayed diagnosis._

The postmortem references the same root cause as the ticket. The sales email that week mentions platform instability. The sprint retro records the velocity hit. None of this is coincidence — it all traces back to one SimEvent that every downstream artifact reads from.

---

## What Gets Generated

A default 22-day simulation produces:

| Artifact                   | Description                                                                                                                 |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `confluence/archives/`     | Seed documents: technical specs, campaign briefs, OKR docs                                                                  |
| `confluence/general/`      | Ad-hoc pages written during the simulation                                                                                  |
| `confluence/postmortems/`  | Post-incident write-ups grounded in actual root causes                                                                      |
| `confluence/retros/`       | Sprint retrospectives referencing real velocity and incidents                                                               |
| `jira/`                    | Sprint tickets, P1 incident tickets with linked PRs                                                                         |
| `slack/channels/`          | Standup transcripts, incident alerts, engineering chatter, bot messages                                                     |
| `git/prs/`                 | Pull requests with reviewers, merge status, linked tickets                                                                  |
| `emails/threads/`          | Multi-turn incident escalation and knowledge gap threads                                                                    |
| `emails/sprint/`           | Sprint kickoff and mid-point check-in emails                                                                                |
| `emails/leadership/`       | Weekly leadership sync summaries                                                                                            |
| `emails/hr/`               | Welcome emails, morale interventions, remote policy                                                                         |
| `emails/sales/`            | Weekly pipeline updates referencing actual incident stability                                                               |
| `servers/logs/`            | AWS cost alerts, Snyk security findings, GitHub Actions output                                                              |
| `simulation_snapshot.json` | Full state: incidents, morale curve, system health, relationship graph, departed employees, new hires, knowledge gap events |
| `simulation.log`           | Complete chronological system and debug logs for the entire run                                                             |

---

## Architecture & Mechanics

OrgForge is not an LLM wrapper. Four interlocking systems enforce correctness.

👉 **[Read the full Architecture Deep-Dive here.](ARCHITECTURE.md)**

---

## The Departure Cascade

The most complex behaviour in the simulation. When an engineer departs mid-sprint, the following fires in order before that day's planning runs:

1. **Incident handoff** — active incidents assigned to the departing engineer are rerouted via Dijkstra escalation routing (while the node is still in the graph) to the next available person in the chain.
2. **Ticket reassignment** — orphaned JIRA tickets go to the dept lead. `In Progress` tickets without a linked PR reset to `To Do` so the new owner starts fresh; tickets with a PR keep their status so the review/merge flow closes them naturally.
3. **Graph recompute** — betweenness centrality is recalculated on the smaller graph. Engineers absorbing the departed node's bridging load receive a proportional stress hit.
4. **Knowledge gap propagation** — if the departed engineer owned undocumented domains (configured via `documented_pct`), those gaps are registered in the SimEvent log and surface in subsequent incidents as contributing factors.
5. **`employee_departed` SimEvent** — emitted with edge snapshot, centrality at departure, reassigned tickets, and incident handoffs. Full ground truth for retrieval evaluation.

So when Jordan leaves on Day 12, the postmortem on Day 9's incident doesn't mention her. But the postmortem on Day 15 might: _"A prior knowledge gap in auth-service, stemming from a recent departure, contributed to the delayed diagnosis."_ That sentence is grounded in a real SimEvent, not LLM inference.

---

## Quickstart

`flow.py` is the main simulation entry point. `config/config.yaml` is the single source of truth for org structure, personas, incident triggers, and quality presets.

### Setup Options

| Scenario                           | Command                              | Notes                                       |
| ---------------------------------- | ------------------------------------ | ------------------------------------------- |
| Everything in Docker               | `docker compose up`                  | Recommended for first run                   |
| Local Ollama + Docker for the rest | `docker compose up mongodb orgforge` | Set `OLLAMA_BASE_URL` in `.env`             |
| Cloud preset (AWS Bedrock)         | `docker compose up mongodb orgforge` | Set credentials in `.env`, skip Ollama      |
| Fully local, no Docker             | `python flow.py`                     | Requires MongoDB and Ollama running locally |

### Option 1 — Everything in Docker (Recommended)

```bash
git clone https://github.com/aeriesec/orgforge
cd orgforge
docker compose up
```

First run pulls models automatically (~5–8 min depending on your connection). Subsequent runs start in seconds — models are cached in a named volume.

When the simulation finishes, run the email generator:

```bash
python email_gen.py
```

Output lands in `./export/`.

### Option 2 — Local Ollama, Docker for MongoDB Only

Create a `.env` file:

```bash
OLLAMA_BASE_URL=http://host.docker.internal:11434
```

Then:

```bash
docker compose up mongodb orgforge
```

> **Linux note:** `host.docker.internal` requires Docker Desktop, or the `extra_hosts: host-gateway` entry in `docker-compose.yaml` (already included).

### Option 3 — Cloud Preset (AWS Bedrock + OpenAI)

Best output quality. Uses Claude Sonnet for document generation, Llama 3.1 8B on Bedrock for high-volume worker calls, and OpenAI `text-embedding-3-large` for embeddings.

Set `quality_preset: "cloud"` in `config.yaml`, then:

```bash
# .env
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=us-east-1
OPENAI_API_KEY=...
```

```bash
pip install boto3 langchain-aws openai
docker compose up mongodb orgforge
```

### Option 4 — Fully Local, No Docker

```bash
docker run -p 27017:27017 mongodb/mongodb-atlas-local

ollama pull qwen2.5:7b-instruct-q4_K_M
ollama pull qwen2.5:1.5b-instruct
ollama pull mxbai-embed-large

pip install -r requirements.txt
python src/flow.py
python src/email_gen.py
```

### Running on AWS EC2

**Cheap EC2 + Bedrock/OpenAI (no GPU required)**

A `t3.small` works fine — the cloud APIs do all the heavy lifting.

1. Launch an EC2 instance (Ubuntu or Amazon Linux) and install Docker
2. `git clone https://github.com/aeriesec/orgforge.git && cd orgforge`
3. `cp .env.example .env` and fill in your credentials
4. Set `quality_preset: "cloud"` in `config/config.yaml`
5. `docker compose up --build -d mongodb orgforge`

**GPU Instance + 70B Local Models**

For `Llama 3.3 70B` entirely locally, use a `g5.2xlarge` or `g5.12xlarge` with the Deep Learning AMI. Uncomment the GPU `deploy` block under the `ollama` service in `docker-compose.yaml`, set `quality_preset: "local_gpu"`, then `docker compose up -d`.

---

## Configuration

`config/config.yaml` is the single source of truth. No Python changes are needed for most customizations.

### Quality Presets

```yaml
quality_preset: "local_cpu" # local_cpu | local_gpu | cloud
```

| Preset      | Planner                     | Worker                | Embeddings             | Best For                 |
| ----------- | --------------------------- | --------------------- | ---------------------- | ------------------------ |
| `local_cpu` | qwen2.5:7b-instruct-q4_K_M  | qwen2.5:1.5b-instruct | mxbai-embed-large      | Local dev, no GPU        |
| `local_gpu` | llama3.3:70b-instruct-q4_KM | llama3.1:8b-instruct  | mxbai-embed-large      | High-fidelity local runs |
| `cloud`     | Claude Sonnet (Bedrock)     | llama3.1:8b (Bedrock) | text-embedding-3-large | Best output quality      |

### Key Config Fields

| Field                     | Purpose                                                                         |
| ------------------------- | ------------------------------------------------------------------------------- |
| `company_name`            | Injected into all generated prose                                               |
| `simulation_days`         | Length of the simulation (default: 22)                                          |
| `legacy_system`           | The unstable system referenced in incidents, tickets, and docs                  |
| `incident_triggers`       | Keywords in the daily theme that trigger a P1 incident                          |
| `sprint_ticket_themes`    | Pool of ticket titles drawn during sprint planning                              |
| `adhoc_confluence_topics` | Spontaneous wiki pages generated on normal days                                 |
| `knowledge_gaps`          | Static departed employees whose absence creates documentation gaps from day one |
| `org_lifecycle`           | Dynamic departures and hires that occur during the simulation (see below)       |
| `roles`                   | Maps simulation roles (on-call, incident commander, HR lead) to departments     |
| `morale`                  | Decay rate, recovery rate, intervention threshold                               |
| `org_chart` + `leads`     | Everyone in the company and who runs each department                            |
| `personas`                | Writing style, stress level, and expertise per named employee                   |
| `external_contacts`       | Vendors, customers, and cloud providers that get pulled into incidents          |

### Dynamic Org Lifecycle

Engineers can join and leave during the simulation. Departures and hires are scheduled in config and execute before the day's planning runs, so every downstream artifact that day reflects the new roster.

```yaml
org_lifecycle:
  scheduled_departures:
    - name: "Jordan"
      day: 12
      reason: "voluntary" # voluntary | layoff | performance
      role: "Senior Backend Engineer"
      knowledge_domains:
        - "auth-service"
        - "redis-cache"
      documented_pct: 0.25 # fraction written down — drives gap severity

  scheduled_hires:
    - name: "Taylor"
      day: 15
      dept: "Engineering"
      role: "Backend Engineer"
      expertise: ["Python", "FastAPI"]
      style: "methodical, asks lots of questions before writing code"
      tenure: "new"

  enable_random_attrition: false
  random_attrition_daily_prob: 0.005
```

On hire, the new engineer enters the graph with cold-start edges below the `warmup_threshold`, so the day planner naturally proposes `warmup_1on1` and `onboarding_session` events until real collaboration warms the edges.

---

## How the Event Bus Works

Every significant action emits a `SimEvent`:

```python
SimEvent(
    type="incident_opened",
    day=8,
    date="2026-03-10",
    actors=["Jax", "Sarah"],
    artifact_ids={"jira": "IT-108"},
    facts={
        "title": "TitanDB: latency spike",
        "root_cause": "connection pool exhaustion under load",
        "involves_gap": True
    },
    summary="P1 incident IT-108: connection pool exhaustion",
    tags=["incident", "P1"]
)
```

Every downstream artifact pulls its facts from the event log rather than asking an LLM to invent them. This prevents temporal drift and hallucination across a multi-week simulation.

The end-of-day `day_summary` SimEvent captures a structured snapshot of everything that happened:

```python
facts={
    "active_actors":        ["Jax", "Sarah", "Morgan"],
    "dominant_event":       "incident_opened",
    "event_type_counts":    {"incident_opened": 1, "pr_review": 2, "standup": 1},
    "departments_involved": ["Engineering"],
    "open_incidents":       ["IT-108"],
    "stress_snapshot":      {"Jax": 72, "Sarah": 55, "Morgan": 41},
    "health_trend":         "degraded",
    "morale_trend":         "moderate",
}
```

This is what makes the dataset useful for RAG evaluation: you have ground truth about what happened, when, who was involved, and what the org's state was — so you can measure whether a retrieval system actually surfaces the right context.

---

## Memory Requirements

| Preset      | RAM Required | Notes                                    |
| ----------- | ------------ | ---------------------------------------- |
| `local_cpu` | ~5 GB        | Qwen 2.5 7B q4 + MongoDB + Python        |
| `local_gpu` | ~48 GB VRAM  | Llama 3.3 70B — requires A100 or 2× A10G |
| `cloud`     | ~500 MB      | Only MongoDB + Python run locally        |

For `local_gpu` on AWS, a `g5.2xlarge` (A10G 24GB) runs 70B at q4 quantization. At ~$0.50/hour spot pricing a full 22-day simulation costs roughly $3–5.

---

## Project Structure

```
orgforge/
├── .github/workflows/    # CI/CD pipelines
├── src/
│   ├── flow.py           # State machine and simulation engine
│   ├── day_planner.py    # LLM-driven per-department daily planning
│   ├── normal_day.py     # Agenda dispatcher — produces typed artifacts per activity
│   ├── planner_models.py # Dataclasses for plans, events, and validation results
│   ├── plan_validator.py # Integrity boundary between LLM proposals and execution
│   ├── org_lifecycle.py  # Dynamic hiring, firing, and knowledge gap propagation
│   ├── graph_dynamics.py # Social graph: stress propagation, edge decay, escalation
│   ├── memory.py         # Vector DB and SimEvent bus
│   └── email_gen.py      # Reflective post-processing artifacts
├── config/               # YAML configurations
├── tests/                # Pytest suite
├── scripts/              # Entrypoint and helper scripts
├── export/               # Output directory for generated dataset
├── README.md
├── ARCHITECTURE.md
└── CONTRIBUTING.md
```

---

## Roadmap

- [ ] Plugin architecture for community artifact types (Zoom, Zendesk, PagerDuty, Salesforce)
- [ ] Domain packs — pre-configured `config.yaml` templates for healthcare, fintech, legal
- [ ] ONNX embedding support for faster CPU inference
- [ ] Export to HuggingFace dataset format
- [ ] Evaluation harness — benchmark RAG retrieval against SimEvent ground truth

---

## Adding a New Artifact Type

1. Add an event emission in `flow.py` when the triggering condition occurs
2. Write a handler that reads from the SimEvent log and generates the artifact
3. Call it from `email_gen.py`'s `run()` method or as a new post-processing script

A formal plugin architecture is on the roadmap. Open an issue before starting so we can align on the interface.

---

## Contributing

Contributions are welcome. Please read **[CONTRIBUTING.md](CONTRIBUTING.md)** before opening a PR. For new domain configs or artifact types, open an Issue first.

---

## License

MIT — see **[LICENSE](LICENSE)**.
