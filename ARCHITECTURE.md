# OrgForge Architecture

OrgForge is an event-driven simulation engine designed to generate synthetic, temporally consistent corporate datasets for AI agent evaluation.

Unlike standard LLM generation scripts that prompt a model to "write a bunch of emails," OrgForge runs a strict state machine. The LLMs are used purely as "actors" to write prose, while the underlying Python engine dictates the facts, timing, and relationships.

## 1. The State Machine (`flow.py`)

At its core, OrgForge is built on the `CrewAI Flow` framework. The simulation operates on a strict day-by-day loop.

- **Genesis Phase:** Runs once at the start of the simulation to generate seed documents (technical specs, campaign briefs) using the configured `planner` LLM.
- **The Daily Loop:** Skips weekends. Every weekday, the simulation generates a daily theme.
- **Scheduled Events:** Depending on the day of the week, the engine triggers Sprint Planning, Standups, or Sprint Retrospectives.
- **Reactive Events:** If the daily theme matches configured `incident_triggers` (e.g., "crash", "timeout"), the engine initiates a P1 incident. Incidents progress through a strict lifecycle: `detected` → `investigating` → `fix_in_progress` → `resolved`.

## 2. The Ground Truth Event Bus (`memory.py`)

To prevent LLM hallucinations, every significant action emits a `SimEvent`.

A `SimEvent` is a structured dictionary containing the exact facts of what happened (e.g., `actors`, `artifact_ids`, `facts`, `summary`, and `tags`). When an LLM generates a Slack message or an email, it is provided context fetched from the `SimEvent` log via a hybrid vector search. This ensures that if a ticket was resolved on Day 4, an email generated on Day 6 will correctly reference Day 4.

**Vector Storage:**
Events and document artifacts are embedded and stored in a local MongoDB Atlas instance.

- **Embedders:** OrgForge supports local Ollama models, AWS Bedrock (Titan), or OpenAI.
- **Safety Fallback:** If the configured embedder goes offline, the system gracefully degrades to a deterministic SHA-256 hashing algorithm to create pseudo-embeddings, preventing the simulation from crashing.

## 3. Social Graph Dynamics (`graph_dynamics.py`)

OrgForge uses `NetworkX` to maintain a living, weighted social graph of all employees and external contacts.

- **Edge Weights:** Relationships (edges) decay slightly every simulated day. However, if two employees collaborate in a Slack thread or a Git PR, their edge weight receives a boost.
- **Burnout Propagation:** Incident responders take a raw stress hit. At the end of every day, the system calculates the network's "key players" using betweenness centrality. If a key player exceeds the `burnout_threshold`, their excess stress bleeds onto their immediate neighbors proportional to their edge weights.
- **Escalation Paths:** When an engineer needs to escalate an issue to leadership, the system uses Dijkstra's algorithm on an inverse-weight graph. This means escalations naturally flow through "work besties" (strong edges) rather than organizational strangers.

## 4. The Git Simulator (`flow.py`)

The system simulates a GitHub-like environment. When an incident reaches the `fix_in_progress` stage, the on-call engineer opens a PR. The simulator queries the `NetworkX` social graph to automatically select the author's closest engineering colleague as the PR reviewer.
