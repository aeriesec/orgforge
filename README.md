# OrgForge

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)
[![Run Tests](https://github.com/aeriesec/orgforge/actions/workflows/tests.yml/badge.svg)](https://github.com/aeriesec/orgforge/actions/workflows/tests.yml)

**Synthetic corporate dataset generator for AI agent evaluation.**

Echelon simulates weeks of realistic enterprise activity — Confluence pages, JIRA tickets, Slack threads, Git PRs, emails, and server logs — all grounded in an event-driven state machine so LLMs can't hallucinate facts out of sequence.

---

## Why Does This Exist?

When building AI agents that reason over institutional knowledge, you need a realistic corpus to test against. The only widely-used corporate dataset is the Enron email corpus — and it's 25 years old, legally sensitive, and covers one company in crisis.

OrgForge generates that corpus from scratch, parameterized to any company, industry, or org structure. Everything is grounded in a SimEvent log: LLMs write prose, but the facts — who was on-call, which ticket was open, when the incident resolved — are always controlled by the state machine.

The larger goal is proving that institutional knowledge capture is a solvable problem. OrgForge is the open-source testbed that builds toward that.

---

## What Gets Generated

A default 22-day simulation produces:

| Artifact                   | Description                                                             |
| -------------------------- | ----------------------------------------------------------------------- |
| `confluence/archives/`     | Seed documents: technical specs, campaign briefs, OKR docs              |
| `confluence/general/`      | Ad-hoc pages written during the simulation                              |
| `confluence/postmortems/`  | Post-incident write-ups grounded in actual root causes                  |
| `confluence/retros/`       | Sprint retrospectives referencing real velocity and incidents           |
| `jira/`                    | Sprint tickets, P1 incident tickets with linked PRs                     |
| `slack/channels/`          | Standup transcripts, incident alerts, engineering chatter, bot messages |
| `git/prs/`                 | Pull requests with reviewers, merge status, linked tickets              |
| `emails/threads/`          | Multi-turn incident escalation and knowledge gap threads                |
| `emails/sprint/`           | Sprint kickoff and mid-point check-in emails                            |
| `emails/leadership/`       | Weekly leadership sync summaries                                        |
| `emails/hr/`               | Welcome email, morale intervention, remote policy                       |
| `emails/sales/`            | Weekly pipeline updates referencing actual incident stability           |
| `servers/logs/`            | AWS cost alerts, Snyk security findings, GitHub Actions output          |
| `simulation_snapshot.json` | Full state: incidents, morale curve, system health, all artifact IDs    |
| `simulation.log`           | Complete chronological system and debug logs for the entire run         |

Every artifact references real SimEvent facts. The incident email thread cites the same root cause as the JIRA ticket. The postmortem links the correct PR. The sales email mentions platform instability on weeks that actually had incidents.

---

## Architecture & Mechanics

OrgForge is not just an LLM wrapper. It uses a strict event-driven state machine (`CrewAI Flow`), a vector database (`MongoDB Atlas Local`), and a dynamic social graph (`NetworkX`) to prevent hallucinations and maintain temporal consistency.

👉 **[Read the full Architecture Deep-Dive here.](ARCHITECTURE.md)**

---

## Quickstart

### Setup Options

| Scenario                           | Command                              | Notes                                       |
| ---------------------------------- | ------------------------------------ | ------------------------------------------- |
| Everything in Docker               | `docker compose up`                  | Recommended for first run                   |
| Local Ollama + Docker for the rest | `docker compose up mongodb orgforge` | Set `OLLAMA_BASE_URL` in `.env`             |
| Cloud preset (AWS Bedrock)         | `docker compose up mongodb orgforge` | Set credentials in `.env`, skip Ollama      |
| Fully local, no Docker             | `python flow.py`                     | Requires MongoDB and Ollama running locally |

_(See the [`config/config.yaml`](config/config.yaml) file to customize the company, industry, and hardware presets)_

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

If you already have Ollama running on your machine, create a `.env` file:

```bash
# .env
OLLAMA_BASE_URL=http://host.docker.internal:11434
```

Then start only the services you need:

```bash
docker compose up mongodb orgforge
```

> **Linux note:** `host.docker.internal` requires Docker Desktop, or the `extra_hosts: host-gateway` entry in `docker-compose.yaml` (already included). Plain Docker Engine on Linux does not resolve this automatically without it.

### Option 3 — Cloud Preset (AWS Bedrock + OpenAI)

Best output quality. Uses Claude 3.5 Sonnet for document generation, Llama 3.1 8B on Bedrock for high-volume worker calls, and OpenAI `text-embedding-3-large` for embeddings.

Set `quality_preset: "cloud"` in `config.yaml`, then create a `.env` file:

```bash
# .env
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=us-east-1
OPENAI_API_KEY=...
```

Install the cloud dependencies and start without Ollama:

```bash
pip install boto3 langchain-aws openai
docker compose up mongodb orgforge
```

### Running on AWS EC2 or Cloud VMs

If you want to run a massive dataset generation without tying up your laptop, you can easily deploy OrgForge to a cloud VM.

#### Option A: The API Route (Cheap EC2 + Bedrock/OpenAI)

You don't need a GPU for this. A cheap `t3.small` instance works perfectly because the cloud APIs do all the heavy lifting.

1. Launch an EC2 instance (Ubuntu or Amazon Linux) and install Docker.
2. Clone the repo: `git clone https://github.com/aeriesec/orgforge.git && cd orgforge`
3. Copy the env file: `cp .env.example .env`
4. Edit `.env`:
   - Set `INSTALL_CLOUD_DEPS=true`
   - Add your `OPENAI_API_KEY` and AWS Credentials.
5. Edit `config/config.yaml` and set `quality_preset: "cloud"`
6. Run it in the background:
   ```bash
   docker compose up --build -d mongodb orgforge
   ```

#### Option B: The Big Iron Route (GPU Instance + 70B Local Models)

If you want to run `Llama 3.3 70B` entirely locally for maximum privacy, you'll need a GPU instance (e.g., an AWS `g5.2xlarge` or `g5.12xlarge`).

1. Launch an instance using the **Deep Learning AMI** (which comes with NVIDIA drivers and Docker pre-installed).
2. Open `docker-compose.yaml` and **uncomment the GPU `deploy` block** under the `ollama` service so the container can access the hardware.
3. Edit `config/config.yaml` and set `quality_preset: "local_gpu"`.
4. Start the stack:

```bash
docker compose up -d

```

### Option 4 — Fully Local, No Docker

Requires MongoDB Atlas Local and Ollama running on your machine:

```bash
# Start MongoDB
docker run -p 27017:27017 mongodb/mongodb-atlas-local

# Pull models (local_cpu preset)
ollama pull qwen2.5:7b-instruct-q4_K_M
ollama pull qwen2.5:1.5b-instruct
ollama pull mxbai-embed-large

pip install -r requirements.txt
python src/flow.py
python src/email_gen.py
```

---

## Configuration

`config/config.yaml` is the single source of truth. No Python changes are needed for most customizations.

### Quality Presets

```yaml
# Switch between: "local_cpu" | "local_gpu" | "cloud"
quality_preset: "local_cpu"
```

| Preset      | Planner                     | Worker                 | Embeddings             | Best For                        |
| ----------- | --------------------------- | ---------------------- | ---------------------- | ------------------------------- |
| `local_cpu` | Qwen 2.5 7B q4              | Qwen 2.5 1.5B          | mxbai-embed-large      | Laptops, parameter iteration    |
| `local_gpu` | Llama 3.3 70B               | Llama 3.1 8B           | mxbai-embed-large      | Local GPU, high quality offline |
| `cloud`     | Claude 3.5 Sonnet (Bedrock) | Llama 3.1 8B (Bedrock) | text-embedding-3-large | Production dataset generation   |

### Simulating a Different Company

Change these sections in `config.yaml` — no Python required:

```yaml
simulation:
  company_name: "Meridian Capital"
  industry: "financial technology"
  domain: "meridiancapital.com"

legacy_system:
  name: "RiskEngine"
  project_name: "Project Mercury"
  description: "legacy risk calculation service"
  aws_alert_message: "RiskEngine batch job costs remain elevated."

incident_triggers:
  - "breach"
  - "compliance"
  - "fail"
  - "timeout"
  - "latency"

sprint_ticket_themes:
  - "Refactor {legacy_system} settlement logic"
  - "Add circuit breaker to FX feed"
  - "Fix race condition in order matching"
  - "Basel III config audit"
  - "Rate limiting on public API"

knowledge_gaps:
  - name: "Richard"
    left: "2024-09"
    role: "Head of Quant"
    knew_about: ["RiskEngine", "legacy pricing model", "Basel III configs"]
    documented_pct: 0.15

org_chart:
  Product: ["Dana", "Wei", "Priya"]
  Engineering: ["Sam", "Lena", "Omar", "Chris"]
  Risk: ["Felix", "Ingrid", "Tom"]
  Sales: ["Marcus", "Blake", "Tasha"]
  HR_Ops: ["Karen", "Dave"]

leads:
  Product: "Dana"
  Engineering: "Sam"
  Risk: "Felix"
  Sales: "Marcus"
  HR_Ops: "Karen"
```

### Key Config Sections

| Section                   | What It Controls                                                            |
| ------------------------- | --------------------------------------------------------------------------- |
| `quality_preset`          | Which model profile is active                                               |
| `simulation`              | Company name, domain, run duration, event probabilities                     |
| `legacy_system`           | The unstable legacy system referenced in incidents, tickets, and docs       |
| `incident_triggers`       | Keywords in the daily theme that trigger a P1 incident                      |
| `sprint_ticket_themes`    | Pool of ticket titles drawn during sprint planning                          |
| `adhoc_confluence_topics` | Spontaneous wiki pages generated on normal days                             |
| `knowledge_gaps`          | Departed employees whose absence creates documentation gaps                 |
| `roles`                   | Maps simulation roles (on-call, incident commander, HR lead) to departments |
| `morale`                  | Decay rate, recovery rate, intervention threshold                           |
| `org_chart` + `leads`     | Everyone in the company and who runs each department                        |
| `personas`                | Writing style, stress level, and expertise per named employee               |
| `external_contacts`       | External contacts                                                           |

---

## How the Event Bus Works

Every significant action in the simulation emits a `SimEvent`:

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
        "involves_bill": True
    },
    summary="P1 incident IT-108: connection pool exhaustion",
    tags=["incident", "P1"]
)
```

Every downstream artifact — the email thread, the postmortem, the Slack alert — pulls its facts from the event log rather than asking an LLM to invent them. This is what prevents temporal drift and hallucination across a multi-week simulation.

The SimEvent log is also what makes the dataset useful for RAG evaluation: you have ground truth about what happened, when, and who was involved, so you can measure whether a retrieval system actually surfaces the right context.

---

## Memory Requirements

| Preset      | RAM Required | Notes                                    |
| ----------- | ------------ | ---------------------------------------- |
| `local_cpu` | ~5 GB        | Qwen 2.5 7B q4 + MongoDB + Python        |
| `local_gpu` | ~48 GB VRAM  | Llama 3.3 70B — requires A100 or 2× A10G |
| `cloud`     | ~500 MB      | Only MongoDB + Python run locally        |

For `local_gpu` on AWS, a `g5.2xlarge` (A10G 24GB) runs the 70B model at q4 quantization. At ~$0.50/hour spot pricing a full 22-day simulation costs roughly $3–5.

---

## Adding a New Artifact Type

The codebase is structured for extension. To add a new artifact type (Zoom transcripts, Zendesk tickets, PagerDuty alerts):

1. Add an event emission in `flow.py` when the triggering condition occurs
2. Write a handler method that reads from the SimEvent log and generates the artifact
3. Call it from `email_gen.py`'s `run()` method, or as a new post-processing script

A formal plugin architecture is on the roadmap. If you want to add a new artifact type, open an issue first so we can align on the interface.

---

## Project Structure

```
orgforge/
├── .github/workflows/   # CI/CD pipelines
├── src/                 # Application source code
│   ├── flow.py          # State machine and simulation engine
│   ├── email_gen.py     # Reflective post-processing artifacts
│   ├── memory.py        # Vector DB and Event Bus
│   └── graph_dynamics.py# Social graph and network logic
├── config/              # YAML Configurations
├── tests/               # Pytest suite
├── scripts/             # Entrypoint and helper scripts
├── export/              # Output directory for generated dataset
├── README.md
├── ARCHITECTURE.md      # Deep dive into the engine
└── CONTRIBUTING.md      # Guidelines for opening PRs
```

---

## Roadmap

- [ ] Plugin architecture for community artifact types (Zoom, Zendesk, PagerDuty, Salesforce)
- [ ] Domain packs — pre-configured `config.yaml` templates for healthcare, fintech, legal
- [ ] ONNX embedding support for faster CPU inference
- [ ] Export to HuggingFace dataset format
- [ ] Evaluation harness — benchmark RAG retrieval against SimEvent ground truth

---

## Contributing

Contributions are welcome! Please read our **[Contributing Guidelines](CONTRIBUTING.md)** before opening a Pull Request. If you are adding a new domain config or artifact type, please open an Issue first to discuss it.

---

## License

This project is licensed under the MIT License - see the **[LICENSE](LICENSE)** file for details.
