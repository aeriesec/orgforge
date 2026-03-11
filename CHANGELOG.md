# Changelog

All notable changes to OrgForge will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [v0.6.0] — 2026-03-11

### Added

- **CEO Persona & Role (`config.yaml`, `flow.py`)**: Added "John" as the CEO persona with a "Visionary Closer" style and "Pressure Multiplier" social role to drive high-level organizational themes.
- **Dynamic Tech Stack Generation (`confluence_writer.py`)**: The simulation now generates a canonical, industry-plausible tech stack at genesis, including legacy "warts" and technical debt, which is persisted to MongoDB for grounding all subsequent technical content.
- **Context-Aware Voice Cards (`normal_day.py`)**: Introduced a modular voice card system that adjusts persona metadata (tenure, expertise, mood, anti-patterns) based on the interaction type (e.g., 1:1 vs. design discussion) to improve LLM character consistency.
- **HyDE-Style Query Rewriting (`memory.py`)**: Implemented `recall_with_rewrite` to generate hypothetical passages before embedding, significantly improving RAG retrieval for ad-hoc documentation and design topics.
- **Automated Conversation Summarization (`normal_day.py`)**: The final turn of 1:1s and mentoring sessions now generates a JSON-based third-person summary of the discussion for future reference in the actor’s memory.
- **Social Graph Checkpointing (`flow.py`, `memory.py`)**: Daily state snapshots now include full serialization of the NetworkX social graph, allowing the simulation to resume with relationship weights and centrality metrics intact.

### Changed

- **Cloud Preset Default (`config.yaml`)**: The default `quality_preset` is now `cloud`, utilizing DeepSeek v3.2 on Bedrock for both planning and worker tasks.
- **Persona Depth Expansion (`config.yaml`)**: Significantly expanded all primary personas with detailed "anti-patterns" (behaviors the LLM must avoid) and nuanced typing quirks to prevent "corporate drift".
- **Multi-Agent Sequential Interactions (`normal_day.py`)**: Refactored Slack interactions (DMs, Async Q&A, Design) to use a dedicated Agent per participant in a sequential Crew, replacing the previous multi-turn single-agent approach.
- **Unified Role Resolution (`config_loader.py`)**: Role resolution (e.g., `ceo`, `scrum_master`) is now centralized in `config_loader.py` to ensure consistency across the planner, email generator, and flow engine.
- **Embedding Optimization (`config.yaml`)**: Default embedding dimensions reduced from 3072 to 1024 to align with updated Bedrock models.
- **Infrastructure Update (`docker-compose.yaml`)**: Upgraded local MongoDB Atlas image to version 8 and enabled `DO_NOT_TRACK`.

### Removed

- **Hardcoded Tone Modifiers (`flow.py`)**: Replaced static stress-to-tone mappings in `persona_backstory` with dynamic, graph-driven `stress_tone_hint` logic.
- **Redundant Stop Sequences (`flow.py`)**: Applied a patch to the CrewAI Bedrock provider to strip `stopSequences`, resolving compatibility issues with certain models.

---

## [v0.5.0] — 2026-03-09

### Added

- **Persona-Driven Content Generation (`confluence_writer.py`, `flow.py`)**: Ad-hoc Confluence topics and sprint ticket titles are now generated at runtime via LLM calls grounded in each author's `expertise` tags, the current daily theme, and RAG context — eliminating the static `adhoc_confluence_topics` and `sprint_ticket_themes` lists from `config.yaml`. Changing `industry` now automatically shifts what gets documented and planned without any other configuration.
- **Active-Actor Authorship (`confluence_writer.py`)**: Ad-hoc pages are now authored by someone from `state.daily_active_actors`, ensuring every page is tied to a person provably working that day.
- **Expertise-Weighted Participant Routing (`normal_day.py`, `memory.py`)**: Added `_expertise_matched_participants()` and `Memory.find_confluence_experts()` to inject subject-matter experts into Slack threads and design discussions via vector similarity over existing Confluence pages, with social-graph proximity weighting as fallback. Applied to `_handle_async_question()` and `_handle_design_discussion()`, replacing previous random sampling.
- **Publish Ledger & Same-Day Dedup (`flow.py`, `confluence_writer.py`)**: `confluence_published_today` tracks root pages per day to prevent topic duplication and enforce causal ordering in expert injection.
- **Per-Actor Multi-Agent Conversations (`normal_day.py`)**: All Slack interaction types (1:1s, async questions, design discussions, mentoring, watercooler) now use a dedicated `Agent` per participant in a sequential `Crew`, with per-person voice cards derived from personas. Replaces the previous single-agent thread writer.
- **Causal Chain Tracking (`flow.py`, `normal_day.py`)**: Incidents now carry a `CausalChainHandler` that accumulates artifact IDs (bot alerts, tickets, PRs, Confluence postmortems, Slack threads) as the incident progresses, providing full ground-truth traceability in SimEvents.
- **Recurrence Detection (`flow.py`)**: `RecurrenceDetector` identifies whether a new incident is a repeat of a prior one via vector similarity, annotating tickets and SimEvents with `recurrence_of` and `recurrence_gap_days`.
- **Sentiment-Driven Stress (`graph_dynamics.py`, `normal_day.py`)**: VADER sentiment scoring is applied to generated Slack content, nudging actor stress up or down based on message tone (capped at ±5 per artifact).
- **Simulation Checkpointing & Resume (`flow.py`, `memory.py`)**: Daily state snapshots (morale, health, stress, actor cursors) are saved to MongoDB after each day. On restart, the simulation picks up from the last checkpoint and skips genesis if artifacts already exist.
- **Dedicated MongoDB Collections (`memory.py`)**: Jira tickets, PRs, Slack messages, and checkpoints are now stored in dedicated collections (`jira_tickets`, `pull_requests`, `slack_messages`, `checkpoints`) with appropriate indexes, replacing in-memory state lists on `State`.
- **Token Usage Tracking (`memory.py`)**: Optional `DEBUG_TOKEN_TRACKING` mode logs all LLM and embed calls to a `token_usage` collection with per-caller aggregation.
- **Parallelized Sprint Planning (`flow.py`)**: Ticket generation is now done per-department in parallel via `ThreadPoolExecutor`, with an LLM-negotiated sprint theme agreed between product and engineering leads before ticket creation begins.
- **LLM-Persona Org Theme & Dept Planning (`day_planner.py`)**: The CEO agent now has a persona-grounded backstory with stress level; department planners similarly use the lead's persona. Active incident context is injected into planning prompts with a narrative constraint preventing "Day 1 startup" framing.
- **Persona `interests` Field (`config.yaml`)**: All personas now have an `interests` list used to generate contextually grounded watercooler chat topics.

### Changed

- **State Artifact Lists Removed (`flow.py`)**: `confluence_pages`, `jira_tickets`, `slack_threads`, and `pr_registry` removed from `State`; counts and snapshots now query MongoDB directly.
- **Slack Persistence Unified (`memory.py`, `normal_day.py`, `flow.py`)**: All Slack writes go through `Memory.log_slack_messages()`, which handles disk export, MongoDB upsert, and thread ID generation in one call. `_save_slack()` now returns `(path, thread_id)`.
- **Ticket & PR Persistence Unified**: All ticket and PR mutations go through `Memory.upsert_ticket()` / `Memory.upsert_pr()` instead of mutating in-memory lists.
- **`ConfluenceWriter._finalise_page()` author signature**: `authors: List[str]` replaced by `author: str` throughout; single primary author is used for metadata, embedding, and the `author_expertise` upsert.
- **Embedding Improvements (`memory.py`, `artifact_registry.py`)**: Switched Bedrock embedder from Titan G1 to Cohere Embed v4 (1024-dim dotProduct with scalar quantization). Chunk size increased from 3K to 12K chars. Event types backed by other artifacts skip embedding to avoid duplication.
- **Sprint & Retro Cadence (`flow.py`)**: Sprint planning and retro days are now driven by a configurable `sprint_length_days` rather than hardcoded weekday checks.
- **`recall()` temporal filtering (`memory.py`)**: `as_of_time` and new `since` parameters now enforce causal floor/ceiling directly inside the MongoDB `$vectorSearch` pre-filter; accepts both `datetime` objects and ISO strings via `_to_iso()`.
- **Cloud preset updated (`config.yaml`)**: Default cloud planner upgraded to `claude-sonnet-4-6`; Bedrock worker also upgraded; embedder switched to local Stella 1.5B.
- **`config.yaml` persona stress levels reduced**: Marcus and Sarah's baseline stress lowered from 70–85 to 50–55.

### Removed

- **`adhoc_confluence_topics` and `sprint_ticket_themes` (`config.yaml`)**: Both static lists deleted; content is now generated at runtime.
- **`state.jira_tickets`, `state.confluence_pages`, `state.slack_threads`, `state.pr_registry`**: Removed from `State`; all reads and writes go through MongoDB.
- **`_legacy_slack_chatter()` (`normal_day.py`)**: Fallback single-agent Slack generation removed; the planner is now always required.
- **Hardcoded `CONF-` prefix in `id_prefix` config**: Stripped from `genesis_docs` configuration.

---

## [v0.4.1] — 2026-03-06

### Fixed

- **Test Suite Mocking (`tests/test_flow.py`, `tests/test_normal_day.py`)**: Added `@patch("confluence_writer.Crew")` to `test_postmortem_artifact_timestamp_within_actor_work_block` to ensure the delegated writer is properly mocked. Fixed a `StopIteration` error in the postmortem test by asserting against the unified `confluence_created` event type containing a `postmortem` tag instead of the deprecated `postmortem_created` event. Mocked `ConfluenceWriter.write_design_doc` with a side effect in `test_design_discussion_confluence_stub_created_sometimes` to emit the expected simulation event during assertions.
- **Fixture Initialization (`tests/test_lifecycle.py`)**: Appended `_registry = MagicMock()` and `_confluence = MagicMock()` to the `mock_flow` fixture to prevent `AttributeError` crashes during test setup.

### Changed

- **Unified Confluence Generation (`flow.py`, `normal_day.py`)**: Fully delegated the creation of genesis documents, postmortems, ad-hoc pages, and design doc stubs to the centralized `ConfluenceWriter`. Instantiated `ArtifactRegistry` and `ConfluenceWriter` in the `Flow` constructor and injected them into the `NormalDayHandler`.
- **Deterministic ID Allocation (`flow.py`, `config.yaml`)**: Updated `next_jira_id`, `_handle_sprint_planning`, and `_handle_incident` to pull JIRA IDs directly from the `ArtifactRegistry`. Refactored the `genesis_docs` prompts in `config.yaml` to generate single pages with explicit IDs rather than relying on brittle `---PAGE BREAK---` splitting. Stripped the hardcoded `CONF-` from the `id_prefix` configurations for engineering and marketing.
- **Structured Ticket Context (`normal_day.py`)**: Replaced the manual string formatting of JIRA ticket states in `_handle_jira_ticket_work` with `_registry.ticket_summary(ticket, self._state.day).for_prompt()`, ensuring the LLM receives complete, structured context. Added a graceful fallback if the registry isn't wired yet.
- **Documentation Cleanup (`ticket_assigner.py`)**: Removed outdated architectural notes referencing "Options B + C" from the module docstring.

---

## [v0.4.0] — 2026-03-06

### Fixed

- **Persona Type Safety (`flow.py`)**: Added explicit type guards in `_generate_adhoc_confluence_page` to ensure the `author` variable is resolved to a `str` before being passed to `persona_backstory`, resolving a Pylance `reportArgumentType` error.
- **SimEvent Schema Integrity (`normal_day.py`)**: Updated `_handle_collision_event` to include the mandatory `artifact_ids` field, properly linking unplanned Slack interactions to their generated JSON paths in the event log.
- **Test Suite Mocking (`tests/test_flow.py`)**: Corrected `TypeError` in incident logic tests where `MagicMock` objects were compared to integers; stress lookups in `graph_dynamics` now return valid numeric values during simulation tests.

### Changed

- **Linguistic Persistence Architecture (`flow.py`, `normal_day.py`)**: Refactored `NormalDayHandler` to accept a `persona_helper` dependency, enabling all synthetic artifacts (Jira, Slack, PRs) to pull from a unified, stress-reactive linguistic fingerprint.
- **Multi-Agent Standup Simulation (`flow.py`)**: Replaced the single-agent "Tech Lead" standup with a multi-agent loop; each attendee now generates a unique update based on their specific typing quirks, technical expertise, and live stress level.
- **"Messy" Org Coordination (`day_planner.py`, `normal_day.py`)**: Enhanced the `OrgCoordinator` to trigger unplanned "collisions"—ranging from high-synergy mentorship to high-friction accountability checks—based on the current tension and morale of department leads.
- **Dynamic Persona Schema (`config.yaml`)**: Expanded persona definitions to include `typing_quirks` (e.g., lowercase-only, emoji-heavy), `social_role`, and `pet_peeves` to break the "cousin" effect and ensure diverse, human-like technical prose.
- **Character-Accurate Ad-hoc Documentation (`flow.py`, `normal_day.py`)**: Integrated persona backstories into the ad-hoc Confluence generator, ensuring background documentation matches the specific voice and stress state of the authoring employee.

---

## [v0.3.9] — 2026-03-06

### Fixed

- **Duplicate ticket work (`day_planner.py`)**: `_parse_plan()` now maintains a
  `claimed_tickets` set across all engineer plans in a single planning pass.
  Any `ticket_progress` agenda item referencing a ticket ID already claimed by
  another engineer is stripped before execution, preventing multiple agents from
  independently logging progress on the same ticket on the same day. Violations
  are logged as warnings.
- **`_validator` attribute error (`flow.py`)**: `daily_cycle` was referencing
  `self._day_planner.validator` instead of `self._day_planner._validator`,
  causing an `AttributeError` on day 11 when `patch_validator_for_lifecycle`
  was first invoked.

### Changed

- **Ticket dedup safety net (`plan_validator.py`, `normal_day.py`)**: Added a
  secondary `ticket_actors_today` guard on `state` that tracks which actors have
  executed `ticket_progress` against a given ticket ID within the current day.
  The validator checks this before approving `ProposedEvent` entries via
  `facts_hint["ticket_id"]`, and `_handle_ticket_progress` registers the actor
  on success. Resets to `{}` at the top of each `daily_cycle()`. Acts as a
  catch-all for paths that bypass `_parse_plan`.

---

## [v0.3.8] — 2026-03-06

### Added

- **Automated PR generation from ticket completion** (`normal_day.py`, `flow.py`): When an engineer's LLM output indicates `is_code_complete: true`, `_handle_ticket_progress` now automatically calls `GitSimulator.create_pr`, attaches the resulting PR ID to the ticket, and advances its status to "In Review". The ticket JSON on disk is updated atomically at the same timestamp.
- **LLM-authored PR descriptions** (`flow.py`): `GitSimulator` now accepts a `worker_llm` parameter and uses a CrewAI agent to write a contextual Markdown PR body ("What Changed" / "Why") using memory context, falling back to a plain auto-generated string on failure.
- **JIRA ticket spawning from design discussions** (`normal_day.py`): `_create_design_doc_stub` now receives the live Slack transcript and prompts the planner LLM for structured JSON containing both the Confluence markdown and 1–3 concrete follow-up `new_tickets`. Each ticket is saved to state, written to disk, and embedded in memory so the `DayPlanner` can schedule it the next day.
- **Blocker detection and memory logging** (`normal_day.py`): `_handle_ticket_progress` scans the LLM comment for blocker keywords and, when found, logs a `blocker_flagged` `SimEvent` with the relevant ticket and actor.

### Changed

- **`_handle_ticket_progress` now outputs structured JSON** (`normal_day.py`): The engineer agent is prompted to return `{"comment": "...", "is_code_complete": boolean}` rather than plain text, enabling downstream PR automation. Falls back gracefully on parse failure.
- **`DepartmentPlanner` prompt hardened** (`day_planner.py`): Added critical rules for role enforcement (non-engineering staff cannot perform engineering tasks), ticket allocation (only the explicit assignee progresses a ticket), and event deduplication (one initiator per `design_discussion` / `async_question`). The `activity_type` field is now constrained to an explicit enum.
- **`OrgCoordinator` collision prompt tightened** (`day_planner.py`): Replaced the loose "only if genuinely motivated" language with numbered rules enforcing strict role separation, real-name actor matching, and a conservative default of `{"collision": null}`. Department summaries now include member name lists to reduce hallucinated actors.
- **Cross-department channel routing unified** (`normal_day.py`): Both `_handle_async_question` and `_handle_design_discussion` now derive the target channel from the full participant set — routing to `#digital-hq` whenever participants span multiple departments, rather than always defaulting to the initiator's department channel.
- **`GitSimulator` reviewer lookup made case-insensitive** (`flow.py`): Department membership check now uses `.lower()` string matching and `.get()` with a default weight, preventing `KeyError` crashes when node attributes are missing.

### Fixed

- Incident-flow PR creation now links the new PR ID back to the originating JIRA ticket and persists the updated ticket JSON to disk (`flow.py`).
- PR review comments are now written back to the PR's JSON file on disk before the bot message is emitted (`normal_day.py`).

---

## [v0.3.7] — 2026-03-05

### Added

- **Watercooler distraction system** (`normal_day.py`): Employees now have a
  configurable probability (`simulation.watercooler_prob`, default `0.15`) of
  being pulled into an off-topic Slack conversation during their workday. The
  distraction is gated once per employee per day, targets a randomly selected
  agenda item rather than always the first, and pulls in 1–2 colleagues weighted
  by social graph edge strength. A context-switch penalty (`0.16–0.25h`) is
  applied both to the agenda item's estimated hours and to the SimClock cursor,
  so the time cost is reflected in downstream scheduling.
- **`test_normal_day.py`**: First test suite for `NormalDayHandler`. Covers
  dispatch routing, ticket progress, 1:1 handling, async questions, design
  discussions, mentoring, SimClock cursor advancement, distraction gate
  behaviour, graph dynamics integration, and utility functions (30 tests).

### Fixed

- Distraction index selection now uses `random.choice` over the full list of
  non-deferred agenda item indices rather than always targeting the first item.
- Context-switch penalty now calls `self._clock.advance_actor()` in addition to
  mutating `item.estimated_hrs`, ensuring SimClock reflects the lost time rather
  than only the plan model.

---

## [v0.3.6] — 2026-03-06

### Fixed

- Resolved `UnboundLocalError` for `random` in `normal_day.py` — removed redundant
  `import random` statements inside `_handle_design_discussion`, `_handle_1on1`,
  `_handle_qa_question`, and `_handle_blocked_engineer`. Python's function-scoping
  rules caused the conditional inner imports to shadow the top-level import,
  leaving `random` unbound at call sites earlier in the same function body.
- Updated `config/config.yaml` cloud model references to use the `bedrock/` prefix
  so model selection routes correctly to AWS Bedrock at runtime.

---

## [v0.3.5] — 2026-03-05

### Added

- Tests for `sim_clock.py` — full coverage of all public methods including
  business-hours enforcement, weekend rollover, and clock monotonicity.
- Tests for `SimClock` integration across `flow.py` and `org_lifecycle.py` —
  verifies artifact timestamps fall within actor work blocks, ceremony timestamps
  land in scheduled windows, and departure/hire events carry valid ISO-8601 times.

### Fixed

- `_embed_and_count()` — added missing `timestamp` parameter forwarded to
  `Memory.embed_artifact()`. Artifacts were previously embedded without a
  timestamp, breaking temporal filtering in `context_for_prompt()`.
- `SimClock.schedule_meeting()` — fixed `ValueError: empty range in randrange`
  crash when `min_hour == max_hour` (e.g. departure events pinned to 09:xx).

---

## [v0.3.4] — 2026-03-05

### Added

- `sim_clock.py` — Actor-local simulation clock replacing all `random.randint`
  timestamp generation across `flow.py` and `normal_day.py`. Each employee now
  has an independent time cursor, guaranteeing no individual can have two
  overlapping artifacts and allowing genuine parallel activity across the org.

- `SimClock.advance_actor()` — Ambient work primitive. Advances a single actor's
  cursor by `estimated_hrs` and returns a randomly sampled artifact timestamp
  from within that work block. Used for ticket progress, Confluence pages, and
  deep work sessions where no causal ordering exists between actors.

- `SimClock.sync_and_tick()` — Causal work primitive. Synchronizes all
  participating actors to the latest cursor among them (the thread cannot start
  until the busiest person is free), then ticks forward by a random delta.
  Used for incident response chains, escalations, and PR review threads.

- `SimClock.tick_message()` — Per-message Slack cadence ticker. Wraps
  `sync_and_tick` with cadence hints: `"incident"` (1–4 min), `"normal"`
  (3–12 min), `"async"` (10–35 min). Replaces the flat random hour assignment
  in `_parse_slack_messages()` so messages within a thread are always
  chronologically ordered and realistically spaced.

- `SimClock.tick_system()` — Independent cursor for automated bot alerts
  (Datadog, PagerDuty, GitHub Actions). Advances separately from human actors
  so bot messages are never gated by an individual's availability.

- `SimClock.sync_to_system()` — Incident response helper. Pulls on-call and
  incident lead cursors forward to the system clock when a P1 fires, ensuring
  all human response artifacts are stamped after the triggering alert.

- `SimClock.at()` — Scheduled meeting pin. Stamps an artifact at the declared
  meeting time and advances all attendee cursors to the meeting end. Used for
  standup (09:30), sprint planning, and retrospectives.

- `SimClock.schedule_meeting()` — Randomized ceremony scheduler. Picks a
  random slot within a defined window (e.g. sprint planning 09:30–11:00,
  retro 14:00–16:00) and pins all attendees via `at()`.

- `SimClock.sync_and_advance()` — Multi-actor ambient work primitive. Syncs
  participants to the latest cursor then advances all by a shared duration.
  Used for collaborative work blocks like pair programming or design sessions.

### Fixed

- Timestamps across Slack threads, JIRA comments, Confluence pages, bot alerts,
  and external contact summaries were previously generated with independent
  `random.randint(hour, ...)` calls, producing out-of-order and causally
  inconsistent artifact timelines. All timestamp generation now routes through
  `SimClock`, restoring correct forensic ordering throughout the corpus.

---

## [v0.3.3] — 2026-03-05

### Changed

- Change embedding endpoint to call `/embed` instead of `/embeddings` and
  update the handling of the response.
- Convert `event_counts` to a plain dict in flow.py

---

## [v0.3.2] — 2026-03-04

### Changed

- Use requirements.txt instead of hard-coded requirements in Dockerfile
- `docker-compose.py` should start the app from src/

---

## [v0.3.1] — 2026-03-04

### Changed

- remove doubled inc.days_active += 1 call
- remove duplicate \_day_planner.plan() call

---

## [v0.3.0] — 2026-03-04

### Added

- **`org_lifecycle.py` — `OrgLifecycleManager`** — new module that owns all
  dynamic roster mutations. The engine controls every side-effect; LLMs only
  produce narrative prose after the fact. Three public entry points called from
  `flow.py` before planning runs each day:
  - `process_departures()` — fires scheduled departures and optional random
    attrition; executes three deterministic side-effects in strict order before
    removing the node (see below)
  - `process_hires()` — adds new engineer nodes at `edge_weight_floor` with
    cold-start edges, bootstraps a persona, and emits an `employee_hired` SimEvent
  - `scan_for_knowledge_gaps()` — scans any free text (incident root cause,
    Confluence body) against all departed employees' `knowledge_domains` and
    emits a `knowledge_gap_detected` SimEvent on first hit per domain; deduplicates
    across the full simulation run
  - `get_roster_context()` — compact string injected into `DepartmentPlanner`
    prompts so the LLM naturally proposes `warmup_1on1` and `onboarding_session`
    events for new hires and references recent departures by name
    _File: `org_lifecycle.py` (new)_

- **Departure side-effect 1 — Active incident handoff** — before the departing
  node is removed, every active incident whose linked JIRA ticket is assigned to
  that engineer triggers a Dijkstra escalation chain while the node is still
  present. Ownership transfers to the first non-departing person in the chain,
  falling back to the dept lead if no path exists. The JIRA `assignee` field is
  mutated deterministically; an `escalation_chain` SimEvent with
  `trigger: "forced_handoff_on_departure"` is emitted.
  _File: `org_lifecycle.py` — `OrgLifecycleManager._handoff_active_incidents()`_

- **Departure side-effect 2 — JIRA ticket reassignment** — all non-Done tickets
  owned by the departing engineer are reassigned to the dept lead. Status logic:
  `"To Do"` tickets keep their status; `"In Progress"` tickets with no linked PR
  are reset to `"To Do"` so the new owner starts fresh; `"In Progress"` tickets
  with a linked PR retain their status so the existing PR review/merge flow closes
  them naturally. Tickets already handled by the incident handoff are not
  double-logged. Each reassignment emits a `ticket_progress` SimEvent with
  `reason: "departure_reassignment"`.
  _File: `org_lifecycle.py` — `OrgLifecycleManager._reassign_jira_tickets()`_

- **Departure side-effect 3 — Centrality vacuum stress** — after the node is
  removed, betweenness centrality is recomputed on the smaller graph and diffed
  against the pre-departure snapshot. Nodes whose score increased have absorbed
  bridging load; each receives `stress_delta = Δc × multiplier` (default `40`,
  configurable via `centrality_vacuum_stress_multiplier`, hard-capped at 20 points
  per departure). This reflects the real phenomenon where a connector's departure
  leaves adjacent nodes as sole bridges across previously-separate clusters.
  _File: `org_lifecycle.py` — `OrgLifecycleManager._apply_centrality_vacuum()`_

- **New hire cold-start edges** — hired engineers enter the graph with edges at
  `edge_weight_floor` to cross-dept nodes and `floor × 2` to same-dept peers.
  Both values sit below `warmup_threshold` (default `2.0`) so `DepartmentPlanner`
  will propose `warmup_1on1` and `onboarding_session` events organically until
  enough collaboration has occurred to warm the edges past the threshold.
  `OrgLifecycleManager.warm_up_edge()` is called from `flow.py` whenever one of
  those events fires.
  _File: `org_lifecycle.py` — `OrgLifecycleManager._execute_hire()`_

- **`patch_validator_for_lifecycle()`** — call once per day after
  `process_departures()` / `process_hires()` to prune departed names from
  `PlanValidator._valid_actors` and add new hire names. Keeps the actor integrity
  check honest without rebuilding the validator from scratch each day.
  _File: `org_lifecycle.py`_

- **`recompute_escalation_after_departure()`** — thin wrapper called from
  `flow.py._end_of_day()` that rebuilds the escalation chain from the dept's
  remaining first responder after the departed node has been removed. Logs the
  updated path as an `escalation_chain` SimEvent for ground-truth retrieval.
  _File: `org_lifecycle.py`_

- **New `KNOWN_EVENT_TYPES`** — `employee_departed`, `employee_hired`,
  `knowledge_gap_detected`, `onboarding_session`, `farewell_message`,
  `warmup_1on1` added to the validator vocabulary so the planner can propose
  them without triggering the novel event fallback path.
  _File: `planner_models.py`_

- **New `State` fields** — `departed_employees: Dict[str, Dict]` and
  `new_hires: Dict[str, Dict]` added to track dynamic roster changes across
  the simulation; both are populated by `OrgLifecycleManager` and included in
  `simulation_snapshot.json` at EOD.
  _File: `flow.py` — `State`_

- **`simulation_snapshot.json` lifecycle sections** — `departed_employees`,
  `new_hires`, and `knowledge_gap_events` arrays appended to the final snapshot
  so the full roster history is available alongside relationship and stress data.
  _File: `flow.py` — `Flow._print_final_report()`_

- **`org_lifecycle` config block** — new top-level config section supports
  `scheduled_departures`, `scheduled_hires`, `enable_random_attrition`, and
  `random_attrition_daily_prob`. Random attrition is off by default; when
  enabled it fires at most one unscheduled departure per day, skipping leads.
  _File: `config.yaml`_

### Changed

- **`DepartmentPlanner` prompt** — accepts a `lifecycle_context` string injected
  between the cross-dept signals and known event types sections. When non-empty
  it surfaces recent departures (with reassigned tickets), recent hires (with
  warm edge count), and unresolved knowledge domains so the LLM plan reflects
  actual roster state rather than a static org chart.
  _File: `day_planner.py` — `DepartmentPlanner._PLAN_PROMPT`, `DepartmentPlanner.plan()`_

- **`DayPlannerOrchestrator`** — holds a `validator` reference so
  `patch_validator_for_lifecycle()` can update `_valid_actors` before each day's
  plan is generated, without rebuilding the validator on every call.
  _File: `day_planner.py` — `DayPlannerOrchestrator.__init__()`_

- **`flow.py` module-level org state** — `ORG_CHART` and `PERSONAS` are now
  copied into `LIVE_ORG_CHART` and `LIVE_PERSONAS` at startup. All roster-sensitive
  code paths reference the live copies; the frozen originals remain available for
  config introspection. `OrgLifecycleManager` mutates the live copies in place.
  _File: `flow.py`_

---

## [v0.2.0] — 2026-03-04

### Added

- **Enriched `day_summary` SimEvent** — the ground-truth end-of-day record now
  carries structured fields that make it genuinely useful for RAG evaluation.
  Previously `active_actors` was always `[]`; all fields below are now populated
  deterministically by the engine, not inferred by an LLM.
  - `active_actors` — names of everyone who participated in at least one event
  - `dominant_event` — most frequently fired event type for the day
  - `event_type_counts` — full `{event_type: count}` frequency map
  - `departments_involved` — derived from active actors via org chart lookup
  - `open_incidents` — ticket IDs of incidents still unresolved at EOD
  - `stress_snapshot` — `{name: stress}` for active actors only
  - `health_trend` — `"critical"` / `"degraded"` / `"recovering"` / `"healthy"`
  - `morale_trend` — `"low"` / `"moderate"` / `"healthy"`

  Two new `State` fields support this: `daily_active_actors: List[str]` and
  `daily_event_type_counts: Dict[str, int]`, both reset at EOD. Two new helpers —
  `_record_daily_actor()` and `_record_daily_event()` — are sprinkled at every
  event-firing site in `flow.py`.
  _Files: `flow.py` — `State`, `Flow._end_of_day()`, and all event handlers_

- **`planner_models.py`** — pure dataclass layer, no LLM or engine dependencies.
  Defines the full planning type hierarchy used by `day_planner.py` and
  `normal_day.py`:
  - `AgendaItem` — a single planned activity for one engineer on one day
  - `EngineerDayPlan` — full-day agenda with stress level, capacity calculation,
    and `apply_incident_pressure()` to defer low-priority items when an incident fires
  - `DepartmentDayPlan` — dept-level plan: engineer agendas + proposed events +
    cross-dept signals that influenced planning
  - `OrgDayPlan` — assembled from all dept plans after `OrgCoordinator` runs;
    `all_events_by_priority()` returns a flat sorted list for the day loop executor
  - `ProposedEvent` — an LLM-proposed event pending validator approval
  - `ValidationResult` — outcome of a `PlanValidator` check
  - `KNOWN_EVENT_TYPES` — the vocabulary set the validator enforces; novel proposals
    outside this set are logged rather than silently dropped
    _File: `planner_models.py` (new)_

- **`plan_validator.py`** — integrity boundary between LLM proposals and the
  execution engine. Checks every `ProposedEvent` against five rules before the
  engine executes it:
  1. **Actor integrity** — all named actors must exist in `org_chart` or
     `external_contacts`; invented names are rejected with a clear reason
  2. **Novel event triage** — unknown event types are approved if they carry a
     known `artifact_hint` (`slack`, `jira`, `confluence`, `email`), and logged
     as `novel_event_proposed` SimEvents for the community backlog regardless
  3. **State plausibility** — tonally inappropriate events are blocked (e.g. no
     `team_celebration` when `system_health < 40`)
  4. **Cooldown windows** — configurable per-event-type minimum days between firings
  5. **Morale gating** — `morale_intervention` only fires when morale is actually low
     _File: `plan_validator.py` (new)_

- **`day_planner.py`** — LLM-driven planning layer that replaces `_generate_theme()`
  in `flow.py`. Three classes:
  - `DepartmentPlanner` — one instance per department. Receives org theme, 7-day
    dept history, cross-dept signals, current roster with live stress levels, and
    open JIRA tickets. Produces a `DepartmentDayPlan` via structured JSON prompt
    with graceful fallback for unparseable LLM output.
  - `OrgCoordinator` — reads all dept plans and identifies one cross-dept collision
    event per day. Prompt is intentionally narrow — only genuinely motivated
    interactions qualify (e.g. Sales reacting to an Engineering incident).
  - `DayPlannerOrchestrator` — top-level entry point called from `flow.py`.
    Engineering plans first as the primary driver; other depts react to
    Engineering's plan before `OrgCoordinator` looks for collision points.
    Rejected and novel events each produce their own SimEvent types so nothing
    is silently discarded.
    _File: `day_planner.py` (new)_

- **`normal_day.py` — `NormalDayHandler`** — replaces `_handle_normal_day()` in
  `flow.py` entirely. Dispatches each engineer's non-deferred agenda items to typed
  handlers that produce specific artifacts:

  | `activity_type`     | Artifacts produced                                       |
  | ------------------- | -------------------------------------------------------- |
  | `ticket_progress`   | JIRA comment + optional blocker Slack thread             |
  | `pr_review`         | GitHub bot message + optional author reply               |
  | `1on1`              | DM thread (3–5 messages)                                 |
  | `async_question`    | Slack thread (3–5 messages) in appropriate channel       |
  | `design_discussion` | Slack thread + 30% chance Confluence design doc stub     |
  | `mentoring`         | DM thread + double social graph edge boost               |
  | `deep_work`         | SimEvent only — intentionally produces no artifact       |
  | deferred (any)      | `agenda_item_deferred` SimEvent logging the interruption |

  Falls back to original random Slack chatter if `org_day_plan` is `None`,
  preserving compatibility with runs that predate the planning layer.
  _File: `normal_day.py` (new)_

---

## [v0.1.2] — 2026-03-04

### Changed

- **`_generate_theme()` switched from `PLANNER_MODEL` to `WORKER_MODEL`**
  Theme generation requires only a single sentence output and does not benefit
  from the planner's capacity. Using the 1.5b worker model reduces per-day
  overhead significantly given `_generate_theme()` fires every simulated day.
  _File: `flow.py` — `Flow._generate_theme()`_

### Added

- **Timeout parameter on `OllamaLLM`** — explicit `timeout` value added to
  `build_llm()` to prevent `litellm.Timeout` errors on slower hardware where
  CPU-bound generation can exceed the default 600s limit.
  _File: `flow.py` — `build_llm()`_

---

## [v0.1.1] — 2026-03-03

### Fixed

- **`TypeError: SimEvent.__init__() got an unexpected keyword argument '_id'`**
  MongoDB documents fetched via `.find()` in `recall_events()` include internal
  fields (`_id`, `embedding`) that are not defined on the `SimEvent` dataclass.
  These fields are now stripped before constructing `SimEvent` objects.
  _File: `memory.py` — `Memory.recall_events()`_

- **`NameError: name 'prop' is not defined` in `_print_final_report()`**
  The `prop` variable returned by `graph_dynamics.propagate_stress()` was scoped
  locally to `_end_of_day()` but referenced in `_print_final_report()`. It is now
  stored as `self._last_stress_prop` and accessed safely via `hasattr` guard to
  handle runs that exit before `_end_of_day()` is called.
  _File: `flow.py` — `Flow._end_of_day()`, `Flow._print_final_report()`_

### Added

- **`Memory.reset(export_dir=None)`** — clears MongoDB `artifacts` and `events`
  collections, resets the in-memory `_event_log`, and optionally wipes the export
  directory. Re-attaches the `FileHandler` to a fresh `simulation.log` after the
  wipe so logging continues uninterrupted.
  _File: `memory.py`_

- **`--reset` CLI flag** — passing `--reset` when invoking `flow.py` triggers
  `Memory.reset()` with the configured `BASE` export directory before the
  simulation starts, ensuring MongoDB and `/export` always represent the same run.
  _File: `flow.py`_

---

## [v0.1.0] — 2026-03-01

### Added

- Initial release of the OrgForge simulation engine
- MongoDB vector search memory layer (`memory.py`) with Ollama, OpenAI, and AWS
  Bedrock embedding providers
- CrewAI-based daily simulation loop with incident detection, sprint planning,
  standups, retrospectives, and postmortem generation (`flow.py`)
- NetworkX social graph with stress propagation and edge decay (`graph_dynamics.py`)
- Export to Confluence, JIRA, Slack, Git PR, and email artifact formats
- Multi-provider LLM support via `quality_preset` config (`local_cpu`, `local_gpu`,
  `cloud` / AWS Bedrock)
- Knowledge gap simulation for departed employees
