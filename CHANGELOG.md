# Changelog

All notable changes to OrgForge will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

  | `activity_type`    | Artifacts produced                                       |
  |--------------------|----------------------------------------------------------|
  | `ticket_progress`  | JIRA comment + optional blocker Slack thread             |
  | `pr_review`        | GitHub bot message + optional author reply               |
  | `1on1`             | DM thread (3–5 messages)                                 |
  | `async_question`   | Slack thread (3–5 messages) in appropriate channel       |
  | `design_discussion`| Slack thread + 30% chance Confluence design doc stub     |
  | `mentoring`        | DM thread + double social graph edge boost               |
  | `deep_work`        | SimEvent only — intentionally produces no artifact       |
  | deferred (any)     | `agenda_item_deferred` SimEvent logging the interruption |

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
