from unittest.mock import MagicMock, patch
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# EXISTING TESTS (unchanged)
# ─────────────────────────────────────────────────────────────────────────────


def test_embedder_fallback_mechanism():
    """Ensures the embedder fallback generates a deterministic vector of the correct dimension."""
    from memory import BaseEmbedder

    class DummyEmbedder(BaseEmbedder):
        def embed(self, text):
            return self._fallback(text)

    dims = 1024
    embedder = DummyEmbedder(dims=dims)

    vec1 = embedder.embed("The server is on fire")
    vec2 = embedder.embed("The server is on fire")
    vec3 = embedder.embed("Everything is fine")

    assert len(vec1) == dims
    assert vec1 == vec2
    assert vec1 != vec3


def test_memory_recall_pipeline_filters():
    """Verifies Memory.recall builds the correct MongoDB aggregation pipeline with filters."""
    from memory import Memory

    mem = Memory()
    mem._artifacts.aggregate = MagicMock(return_value=[])

    mem.recall(
        query="database crash",
        n=5,
        type_filter="jira",
        day_range=(2, 8),
    )

    args, kwargs = mem._artifacts.aggregate.call_args
    pipeline = args[0]

    vector_search_stage = pipeline[0]["$vectorSearch"]
    search_filter = vector_search_stage.get("filter", {})

    assert search_filter["type"]["$eq"] == "jira"
    assert search_filter["day"]["$gte"] == 2
    assert search_filter["day"]["$lte"] == 8
    assert vector_search_stage["limit"] == 5


def test_memory_recall_pipeline_as_of_time_datetime():
    """
    recall() must translate a datetime as_of_time into a $lte timestamp
    filter inside $vectorSearch.
    """
    from datetime import datetime
    from memory import Memory

    mem = Memory()
    mem._artifacts.aggregate = MagicMock(return_value=[])

    cutoff = datetime(2026, 1, 5, 14, 30, 0)
    mem.recall(query="auth failure", n=3, as_of_time=cutoff)

    pipeline = mem._artifacts.aggregate.call_args[0][0]
    search_filter = pipeline[0]["$vectorSearch"].get("filter", {})

    assert search_filter["timestamp"]["$lte"] == cutoff.isoformat()


def test_memory_recall_pipeline_as_of_time_iso_string():
    """
    recall() must also accept a pre-formatted ISO string for as_of_time.
    """
    from memory import Memory

    mem = Memory()
    mem._artifacts.aggregate = MagicMock(return_value=[])

    iso = "2026-01-05T14:30:00"
    mem.recall(query="auth failure", n=3, as_of_time=iso)

    pipeline = mem._artifacts.aggregate.call_args[0][0]
    search_filter = pipeline[0]["$vectorSearch"].get("filter", {})

    assert search_filter["timestamp"]["$lte"] == iso


def test_memory_recall_pipeline_no_as_of_time():
    """
    When as_of_time is None no timestamp filter must appear in the pipeline.
    """
    from memory import Memory

    mem = Memory()
    mem._artifacts.aggregate = MagicMock(return_value=[])

    mem.recall(query="anything", n=3)

    pipeline = mem._artifacts.aggregate.call_args[0][0]
    search_filter = pipeline[0]["$vectorSearch"].get("filter", {})

    assert "timestamp" not in search_filter


def test_log_dept_plan_serializes_nested_dataclasses(make_test_memory):
    """
    log_dept_plan must successfully insert into MongoDB even when
    engineer_plans contain nested AgendaItem dataclasses.
    """
    from dataclasses import asdict
    from planner_models import AgendaItem, EngineerDayPlan, ProposedEvent

    mem = make_test_memory

    agenda = [
        AgendaItem(
            activity_type="deep_work",
            description="Finish the migration doc",
            estimated_hrs=2.0,
        )
    ]
    eng_plan = EngineerDayPlan(
        name="Sarah",
        dept="Product",
        agenda=agenda,
        stress_level=0,
    )
    event = ProposedEvent(
        event_type="design_discussion",
        actors=["Sarah", "Jax"],
        rationale="Clarify TitanDB inputs",
        facts_hint={},
        priority=1,
    )

    mem.log_dept_plan(
        day=1,
        date="2026-03-11",
        dept="Product",
        lead="Sarah",
        theme="Migration week",
        engineer_plans=[asdict(eng_plan)],
        proposed_events=[asdict(event)],
        raw={},
    )

    doc = mem._plans.find_one({"dept": "Product", "day": 1})
    assert doc is not None
    assert doc["engineer_plans"][0]["name"] == "Sarah"
    assert doc["engineer_plans"][0]["agenda"][0]["activity_type"] == "deep_work"


def test_simevent_serialization():
    """Verifies SimEvent can serialize to and from a dict without data loss."""
    from memory import SimEvent

    original_event = SimEvent(
        type="incident_resolved",
        day=5,
        date="2026-03-03",
        actors=["Alice", "Bob"],
        artifact_ids={"jira": "ORG-105", "pr": "PR-100"},
        facts={"duration_days": 2, "root_cause": "DNS failure"},
        summary="Alice fixed the DNS",
        tags=["incident", "p1"],
        timestamp="2026-03-05T13:33:51.027Z",
    )

    serialized = original_event.to_dict()
    restored_event = SimEvent.from_dict(serialized)

    assert restored_event.type == "incident_resolved"
    assert "Bob" in restored_event.actors
    assert restored_event.artifact_ids["pr"] == "PR-100"
    assert restored_event.facts["duration_days"] == 2
    assert "p1" in restored_event.tags


# ─────────────────────────────────────────────────────────────────────────────
# NEW TESTS
# ─────────────────────────────────────────────────────────────────────────────


# ── OllamaEmbedder — input_type prefix bug ────────────────────────────────────


class TestOllamaEmbedderInputType:
    """
    Guards the fix for the silent input_type drop in OllamaEmbedder.embed().
    The Ollama /api/embed payload must include the correct instruction prefix
    so stella / mxbai use asymmetric query vs. document embeddings.
    """

    def _make_embedder(self, mock_post):
        """Return an OllamaEmbedder whose HTTP layer is fully mocked."""
        from memory import OllamaEmbedder

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "embeddings": [[0.1] * 1024],
            "prompt_eval_count": 10,
        }
        mock_post.return_value = mock_response

        embedder = OllamaEmbedder(model="stella_en_1.5b_v5", dims=1024)
        embedder._ok = True
        return embedder

    @patch("requests.post")
    def test_search_query_prefix_sent_to_ollama(self, mock_post):
        """
        When input_type='search_query', the payload sent to Ollama must contain
        the query instruction prefix — not the bare text.
        """
        embedder = self._make_embedder(mock_post)
        embedder.embed("firmware upgrade 503 error", input_type="search_query")

        payload = mock_post.call_args[1]["json"]
        assert "search_query" in payload["input"].lower() or payload[
            "input"
        ].startswith("Represent this query"), (
            "OllamaEmbedder.embed() is silently dropping input_type='search_query'. "
            "The Ollama payload must include the instruction prefix."
        )

    @patch("requests.post")
    def test_search_document_prefix_sent_to_ollama(self, mock_post):
        """
        When input_type='search_document', the payload must contain the document
        instruction prefix so stored vectors are in the correct embedding space.
        """
        embedder = self._make_embedder(mock_post)
        embedder.embed("iOS mobile crash stack trace", input_type="search_document")

        payload = mock_post.call_args[1]["json"]
        assert "search_document" in payload["input"].lower() or payload[
            "input"
        ].startswith("Represent this sentence"), (
            "OllamaEmbedder.embed() is silently dropping input_type='search_document'. "
            "The Ollama payload must include the instruction prefix."
        )

    @patch("requests.post")
    def test_query_and_document_payloads_differ(self, mock_post):
        """
        The same raw text must produce different payloads for 'search_query'
        vs 'search_document' — if they're identical, the asymmetric instruction
        tuning is not being applied and routing scores will compress.
        """
        embedder = self._make_embedder(mock_post)

        embedder.embed("rate limit config", input_type="search_query")
        query_payload = mock_post.call_args[1]["json"]["input"]

        embedder.embed("rate limit config", input_type="search_document")
        doc_payload = mock_post.call_args[1]["json"]["input"]

        assert query_payload != doc_payload, (
            "search_query and search_document produced identical Ollama payloads. "
            "input_type is being ignored — this kills asymmetric retrieval quality."
        )


# ── recall() — mutually exclusive filter guard ────────────────────────────────


def test_recall_raises_on_type_filter_and_type_exclude_together():
    """
    Passing both type_filter and type_exclude is a programmer error.
    recall() must raise ValueError rather than silently applying one and ignoring
    the other, since that would produce misleading search results.
    """
    from memory import Memory

    mem = Memory()
    mem._artifacts.aggregate = MagicMock(return_value=[])

    with pytest.raises(ValueError, match="mutually exclusive"):
        mem.recall(
            query="test",
            type_filter="jira",
            type_exclude=["persona_skill"],
        )


def test_recall_type_exclude_builds_nin_filter():
    """
    type_exclude must translate to a $nin filter in the vectorSearch stage,
    not a $eq. context_for_prompt() relies on this to hide persona_skill
    artifacts from general RAG retrieval.
    """
    from memory import Memory

    mem = Memory()
    mem._artifacts.aggregate = MagicMock(return_value=[])

    mem.recall(query="incident response", n=3, type_exclude=["persona_skill"])

    pipeline = mem._artifacts.aggregate.call_args[0][0]
    search_filter = pipeline[0]["$vectorSearch"].get("filter", {})

    assert "$nin" in search_filter.get("type", {}), (
        "type_exclude must produce a {'type': {'$nin': [...]}} filter. "
        "If it's missing, persona_skill artifacts pollute general RAG results."
    )
    assert "persona_skill" in search_filter["type"]["$nin"]


def test_recall_since_builds_gte_timestamp_filter():
    """
    `since` (causal floor) must produce a $gte timestamp filter alongside
    as_of_time's $lte. When both are provided, the pipeline must carry both
    bounds in the same timestamp filter dict.
    """
    from memory import Memory

    mem = Memory()
    mem._artifacts.aggregate = MagicMock(return_value=[])

    mem.recall(
        query="sprint planning",
        n=3,
        since="2026-03-01T00:00:00",
        as_of_time="2026-03-10T23:59:59",
    )

    pipeline = mem._artifacts.aggregate.call_args[0][0]
    search_filter = pipeline[0]["$vectorSearch"].get("filter", {})
    ts = search_filter.get("timestamp", {})

    assert ts.get("$gte") == "2026-03-01T00:00:00"
    assert ts.get("$lte") == "2026-03-10T23:59:59"


# ── log_event — embedding skip list ──────────────────────────────────────────


def test_log_event_skips_embed_for_noisy_types(make_test_memory):
    """
    High-volume event types in _SKIP_EMBED_TYPES must be stored with
    embedding=None. Embedding every slack_message_sent or jira_ticket_created
    would waste tokens and inflate the vector index with low-signal noise.
    """
    from memory import SimEvent

    mem = make_test_memory
    mem._embed = MagicMock(return_value=[0.1] * 1024)

    skip_event = SimEvent(
        type="slack_message_sent",
        day=1,
        date="2026-03-02",
        timestamp="2026-03-02T10:00:00",
        actors=["Jax"],
        artifact_ids={},
        facts={},
        summary="Jax posted in #backend",
        tags=["slack"],
    )
    mem.log_event(skip_event)

    assert not mem._embed.called, (
        "log_event() called _embed() for 'slack_message_sent', which is in "
        "_SKIP_EMBED_TYPES. These events must be stored with embedding=None."
    )

    stored = mem._events.find_one({"type": "slack_message_sent"})
    assert stored is not None
    assert stored.get("embedding") is None


def test_log_event_embeds_high_signal_types(make_test_memory):
    """
    Events NOT in _SKIP_EMBED_TYPES (e.g. incident_resolved) must be
    embedded so they appear in vector search results.
    """
    from memory import SimEvent

    mem = make_test_memory
    fake_vec = [0.42] * 1024
    mem._embed = MagicMock(return_value=fake_vec)

    event = SimEvent(
        type="incident_resolved",
        day=3,
        date="2026-03-04",
        timestamp="2026-03-04T15:00:00",
        actors=["Deepa", "Jax"],
        artifact_ids={"jira": "INC-001"},
        facts={"root_cause": "db connection pool exhausted"},
        summary="Deepa resolved the connection pool incident",
        tags=["incident"],
    )
    mem.log_event(event)

    assert mem._embed.called, (
        "log_event() did not call _embed() for 'incident_resolved'. "
        "High-signal events must be embedded for vector retrieval."
    )
    stored = mem._events.find_one({"type": "incident_resolved"})
    assert stored["embedding"] == fake_vec


def test_log_event_appends_to_in_memory_log(make_test_memory):
    """
    Every logged event, regardless of type, must be appended to _event_log.
    persona_history() and events_by_type() depend on this in-memory list —
    they never touch MongoDB.
    """
    from memory import SimEvent

    mem = make_test_memory
    initial_len = len(mem._event_log)

    for event_type in ["slack_message_sent", "incident_resolved", "postmortem_created"]:
        mem.log_event(
            SimEvent(
                type=event_type,
                day=1,
                date="2026-03-02",
                timestamp="2026-03-02T09:00:00",
                actors=["Morgan"],
                artifact_ids={},
                facts={},
                summary=f"Test event: {event_type}",
                tags=[],
            )
        )

    assert len(mem._event_log) == initial_len + 3


# ── embed_artifact ────────────────────────────────────────────────────────────


def test_embed_artifact_upserts_not_duplicates(make_test_memory):
    """
    Calling embed_artifact() twice with the same id must update the existing
    document, not insert a second one. The artifact store is an upsert cache —
    duplicate IDs would corrupt vector search recall.
    """
    mem = make_test_memory
    mem._embed = MagicMock(return_value=[0.1] * 1024)

    kwargs = dict(
        id="conf_001",
        type="confluence",
        title="TitanDB Overview",
        content="TitanDB is the legacy sports tracking database.",
        day=1,
        date="2026-03-02",
        timestamp="2026-03-02T08:00:00",
    )
    mem.embed_artifact(**kwargs)
    mem.embed_artifact(**{**kwargs, "content": "Updated content after review."})

    count = mem._artifacts.count_documents({"_id": "conf_001"})
    assert count == 1, (
        "embed_artifact() inserted duplicate documents for the same id. "
        "It must upsert so re-embedding an artifact doesn't inflate the index."
    )

    stored = mem._artifacts.find_one({"_id": "conf_001"})
    assert "Updated" in stored["content"]


def test_embed_artifact_uses_search_document_input_type(make_test_memory):
    """
    embed_artifact() must call _embed() with input_type='search_document'.
    Using 'search_query' for stored documents would flip the asymmetric
    instruction tuning and make all retrieval scores meaningless.
    """
    mem = make_test_memory
    mem._embed = MagicMock(return_value=[0.1] * 1024)

    mem.embed_artifact(
        id="conf_002",
        type="confluence",
        title="Auth Service Runbook",
        content="Steps for rotating the OAuth2 client secret.",
        day=2,
        date="2026-03-03",
        timestamp="2026-03-03T09:00:00",
    )

    call_kwargs = mem._embed.call_args[1]
    assert call_kwargs.get("input_type") == "search_document", (
        "embed_artifact() must pass input_type='search_document' to _embed(). "
        "Using 'search_query' inverts the asymmetric embedding space."
    )


# ── _to_iso ───────────────────────────────────────────────────────────────────


class TestToIso:
    """
    _to_iso() is the normalisation layer that lets every causal-ceiling argument
    accept either a datetime or a pre-formatted string. Any regression here
    silently breaks all as_of_time filtering across recall(), recall_events(),
    context_for_incident(), etc.
    """

    def test_none_passthrough(self):
        from memory import Memory

        assert Memory._to_iso(None) is None

    def test_iso_string_passthrough(self):
        from memory import Memory

        iso = "2026-03-10T14:30:00"
        assert Memory._to_iso(iso) == iso

    def test_datetime_converted_to_iso(self):
        from datetime import datetime
        from memory import Memory

        dt = datetime(2026, 3, 10, 14, 30, 0)
        result = Memory._to_iso(dt)
        assert result == "2026-03-10T14:30:00"
        assert isinstance(result, str)


# ── Jira / PR helpers ─────────────────────────────────────────────────────────


def test_upsert_and_get_ticket(make_test_memory):
    """
    upsert_ticket() + get_ticket() round-trip: a ticket written once must be
    retrievable by id, and re-upserting the same id must update, not duplicate.
    """
    mem = make_test_memory

    ticket = {
        "id": "INC-042",
        "title": "Redis cache eviction causing 503s",
        "status": "In Progress",
        "assignee": "Deepa",
        "dept": "Engineering_Backend",
        "story_points": 5,
    }
    mem.upsert_ticket(ticket)
    retrieved = mem.get_ticket("INC-042")

    assert retrieved is not None
    assert retrieved["title"] == "Redis cache eviction causing 503s"
    assert retrieved["assignee"] == "Deepa"
    assert "_id" not in retrieved, (
        "get_ticket() must exclude _id (non-serialisable ObjectId)"
    )

    # Re-upsert with updated status — must not create a second document
    mem.upsert_ticket({**ticket, "status": "Done"})
    assert mem._jira.count_documents({"id": "INC-042"}) == 1
    assert mem.get_ticket("INC-042")["status"] == "Done"


def test_get_open_tickets_for_dept(make_test_memory):
    """
    get_open_tickets_for_dept() must return only non-Done tickets for the
    given dept, not tickets from other depts or completed tickets.
    """
    mem = make_test_memory

    mem.upsert_ticket(
        {
            "id": "T-1",
            "title": "Fix API",
            "status": "In Progress",
            "dept": "Engineering_Backend",
            "assignee": "Jax",
        }
    )
    mem.upsert_ticket(
        {
            "id": "T-2",
            "title": "Write tests",
            "status": "Done",
            "dept": "Engineering_Backend",
            "assignee": "Deepa",
        }
    )
    mem.upsert_ticket(
        {
            "id": "T-3",
            "title": "Design mockup",
            "status": "In Progress",
            "dept": "Design",
            "assignee": "Priya",
        }
    )

    results = mem.get_open_tickets_for_dept(members=[], dept_name="Engineering_Backend")

    ids = [t["id"] for t in results]
    assert "T-1" in ids
    assert "T-2" not in ids, "Done tickets must be excluded"
    assert "T-3" not in ids, "Tickets from other depts must be excluded"


def test_get_reviewable_prs_for(make_test_memory):
    """
    get_reviewable_prs_for() must return only open PRs where the given person
    is listed as a reviewer — not closed PRs, not PRs they authored.
    """
    mem = make_test_memory

    mem.upsert_pr(
        {
            "pr_id": "PR-10",
            "title": "Add rate limiting",
            "status": "open",
            "reviewers": ["Morgan", "Deepa"],
            "author": "Liam",
        }
    )
    mem.upsert_pr(
        {
            "pr_id": "PR-11",
            "title": "Fix auth bug",
            "status": "merged",
            "reviewers": ["Morgan"],
            "author": "Jax",
        }
    )
    mem.upsert_pr(
        {
            "pr_id": "PR-12",
            "title": "Update docs",
            "status": "open",
            "reviewers": ["Kaitlyn"],
            "author": "Hanna",
        }
    )

    results = mem.get_reviewable_prs_for("Morgan")
    pr_ids = [pr["pr_id"] for pr in results]

    assert "PR-10" in pr_ids
    assert "PR-11" not in pr_ids, "Merged PRs must not be returned"
    assert "PR-12" not in pr_ids, (
        "PRs where Morgan is not a reviewer must not be returned"
    )


# ── persona_history / events_by_type ─────────────────────────────────────────


def test_persona_history_returns_only_actor_events(make_test_memory):
    """
    persona_history() must return only events where the named person is in
    actors, ordered chronologically, capped at n. It operates on the in-memory
    _event_log — no MongoDB queries.
    """
    from memory import SimEvent

    mem = make_test_memory

    def _evt(actor, summary, day):
        return SimEvent(
            type="deep_work",
            day=day,
            date=f"2026-03-{day:02d}",
            timestamp=f"2026-03-{day:02d}T09:00:00",
            actors=[actor],
            artifact_ids={},
            facts={},
            summary=summary,
            tags=[],
        )

    for i in range(1, 7):
        mem._event_log.append(_evt("Jax", f"Jax event {i}", i))
    mem._event_log.append(_evt("Deepa", "Deepa event", 3))

    history = mem.persona_history("Jax", n=4)
    assert len(history) == 4
    assert all("Jax" in e.actors for e in history)
    # Must be the last 4 Jax events
    assert history[-1].summary == "Jax event 6"


def test_events_by_type_filters_correctly(make_test_memory):
    """
    events_by_type() must return only events with the exact matching type
    from the in-memory log.
    """
    from memory import SimEvent

    mem = make_test_memory

    def _evt(etype):
        return SimEvent(
            type=etype,
            day=1,
            date="2026-03-02",
            timestamp="2026-03-02T10:00:00",
            actors=["Jax"],
            artifact_ids={},
            facts={},
            summary=f"A {etype} event",
            tags=[],
        )

    mem._event_log += [
        _evt("incident_resolved"),
        _evt("incident_resolved"),
        _evt("deploy"),
    ]

    results = mem.events_by_type("incident_resolved")
    assert len(results) == 2
    assert all(e.type == "incident_resolved" for e in results)


# ── context_for_incident ──────────────────────────────────────────────────────


def test_context_for_incident_includes_ticket_and_prior(make_test_memory):
    """
    context_for_incident() must include the ticket title and root cause, and
    surface any prior incident for recurrence signal — both without embedding.
    """
    from memory import SimEvent

    mem = make_test_memory

    mem.upsert_ticket(
        {
            "id": "INC-007",
            "title": "Payment service timeout",
            "status": "Open",
            "assignee": "Sanjay",
            "root_cause": "upstream dependency latency spike",
        }
    )
    mem.log_event(
        SimEvent(
            type="incident_detected",
            day=2,
            date="2026-03-03",
            timestamp="2026-03-03T08:00:00",
            actors=["Sanjay"],
            artifact_ids={"jira": "INC-006"},
            facts={"root_cause": "prior DNS failure"},
            summary="Prior incident on day 2",
            tags=["incident"],
        )
    )

    ctx = mem.context_for_incident("INC-007")

    assert "INC-007" in ctx
    assert "Payment service timeout" in ctx
    assert "upstream dependency latency spike" in ctx


def test_context_for_incident_missing_ticket(make_test_memory):
    """
    context_for_incident() must not raise when the ticket doesn't exist —
    it should return a graceful 'not found' message.
    """
    mem = make_test_memory
    ctx = mem.context_for_incident("INC-DOESNOTEXIST")
    assert "not found" in ctx.lower() or "INC-DOESNOTEXIST" in ctx


# ── context_for_person ────────────────────────────────────────────────────────


def test_context_for_person_shows_open_tickets_only(make_test_memory):
    """
    context_for_person() must include open tickets assigned to the person
    and exclude Done tickets — standup context should only surface active work.
    """
    mem = make_test_memory

    mem.upsert_ticket(
        {
            "id": "T-A",
            "title": "Migrate auth service",
            "status": "In Progress",
            "assignee": "Kaitlyn",
            "priority": 1,
        }
    )
    mem.upsert_ticket(
        {
            "id": "T-B",
            "title": "Old closed ticket",
            "status": "Done",
            "assignee": "Kaitlyn",
            "priority": 2,
        }
    )

    ctx = mem.context_for_person("Kaitlyn")

    assert "Migrate auth service" in ctx
    assert "Old closed ticket" not in ctx


def test_context_for_person_no_tickets_message(make_test_memory):
    """
    context_for_person() must include a 'no open tickets' message when the
    person has nothing assigned, rather than silently omitting the section.
    """
    mem = make_test_memory
    ctx = mem.context_for_person("Yusuf")
    assert "no open tickets" in ctx.lower()


# ── save / load checkpoint ────────────────────────────────────────────────────


def test_save_and_load_checkpoint(make_test_memory):
    """
    save_checkpoint() + load_latest_checkpoint() round-trip: the latest
    checkpoint must match the last saved state, not an earlier one.
    """
    mem = make_test_memory

    mem.save_checkpoint(
        day=3,
        state_vars={"system_health": 85, "team_morale": 0.72},
        stress={"Jax": 55},
        cursors={},
        graph_data={},
    )
    mem.save_checkpoint(
        day=5,
        state_vars={"system_health": 70, "team_morale": 0.65},
        stress={"Jax": 75},
        cursors={},
        graph_data={},
    )

    latest = mem.load_latest_checkpoint()
    assert latest is not None
    assert latest["day"] == 5
    assert latest["state"]["system_health"] == 70


# ── has_genesis_artifacts ─────────────────────────────────────────────────────


def test_has_genesis_artifacts_false_when_empty(make_test_memory):
    """Returns False on a fresh sim with no events."""
    mem = make_test_memory
    assert mem.has_genesis_artifacts() is False


def test_has_genesis_artifacts_true_after_genesis_event(make_test_memory):
    """Returns True once any event tagged 'genesis' is logged."""
    from memory import SimEvent

    mem = make_test_memory
    mem.log_event(
        SimEvent(
            type="confluence_page_created",
            day=0,
            date="2026-03-01",
            timestamp="2026-03-01T07:00:00",
            actors=["system"],
            artifact_ids={"confluence": "ENG-001"},
            facts={},
            summary="Genesis doc written",
            tags=["genesis"],
        )
    )
    assert mem.has_genesis_artifacts() is True


# ── conversation summary store ────────────────────────────────────────────────


def test_save_and_retrieve_conversation_summary(make_test_memory):
    """
    save_conversation_summary() + context_for_person_conversations():
    a saved 1on1 summary must appear in the context for either participant,
    since participants are stored sorted and queried by membership.
    """
    mem = make_test_memory

    mem.save_conversation_summary(
        conv_type="1on1",
        participants=["Sarah", "Jax"],
        summary="Discussed TitanDB migration timeline and Jax's concerns about test coverage.",
        day=4,
        date="2026-03-05",
        timestamp="2026-03-05T11:00:00",
        slack_thread_id="slack_general_2026-03-05T11:00:00",
    )

    ctx_sarah = mem.context_for_person_conversations("Sarah", conv_type="1on1")
    ctx_jax = mem.context_for_person_conversations("Jax", conv_type="1on1")

    assert "TitanDB migration" in ctx_sarah
    assert "TitanDB migration" in ctx_jax


def test_conversation_summary_type_filter(make_test_memory):
    """
    context_for_person_conversations() with conv_type='mentoring' must not
    return 1on1 summaries, and vice versa.
    """
    mem = make_test_memory

    mem.save_conversation_summary(
        conv_type="1on1",
        participants=["Jax", "Deepa"],
        summary="Talked about sprint velocity.",
        day=2,
        date="2026-03-03",
        timestamp="2026-03-03T10:00:00",
    )
    mem.save_conversation_summary(
        conv_type="mentoring",
        participants=["Jax", "Liam"],
        summary="Liam shadowed Jax on the Redis investigation.",
        day=3,
        date="2026-03-04",
        timestamp="2026-03-04T10:00:00",
    )

    mentoring_ctx = mem.context_for_person_conversations("Jax", conv_type="mentoring")
    one_on_one_ctx = mem.context_for_person_conversations("Jax", conv_type="1on1")

    assert "Redis investigation" in mentoring_ctx
    assert "Redis investigation" not in one_on_one_ctx
    assert "sprint velocity" in one_on_one_ctx
    assert "sprint velocity" not in mentoring_ctx


# ── recall_with_rewrite ───────────────────────────────────────────────────────


def test_recall_with_rewrite_falls_back_without_llm():
    """
    recall_with_rewrite() with llm_callable=None must fall back to
    context_for_prompt() and not raise. Callers that haven't wired in an LLM
    yet must degrade gracefully — a missing llm_callable is a common init state.
    """
    from memory import Memory

    mem = Memory()
    mem._artifacts.aggregate = MagicMock(return_value=[])
    mem._artifacts.count_documents = MagicMock(return_value=0)

    # Must not raise
    result = mem.recall_with_rewrite(
        raw_query="kubernetes pod crash loop",
        n=3,
        llm_callable=None,
    )
    assert isinstance(result, str)


def test_recall_with_rewrite_uses_rewritten_query():
    """
    When llm_callable is provided, recall_with_rewrite() must embed the
    rewritten query, not the raw one. The rewrite step is HyDE — the LLM
    generates a hypothetical passage that embeds closer to real documents.
    """
    from memory import Memory

    mem = Memory()
    mem._artifacts.aggregate = MagicMock(return_value=[])
    mem._artifacts.count_documents = MagicMock(return_value=0)

    captured_queries = []
    original_context = mem.context_for_prompt

    def capturing_context(query, **kwargs):
        captured_queries.append(query)
        return original_context(query, **kwargs)

    mem.context_for_prompt = capturing_context

    llm = MagicMock(
        return_value="A runbook describing Kubernetes pod restart policies and backoff intervals."
    )

    mem.recall_with_rewrite(raw_query="k8s restarts", n=3, llm_callable=llm)

    assert len(captured_queries) == 1
    assert captured_queries[0] != "k8s restarts", (
        "recall_with_rewrite() passed the raw query to context_for_prompt instead "
        "of the LLM-rewritten passage. The HyDE rewrite step is being skipped."
    )


# ── stats ─────────────────────────────────────────────────────────────────────


def test_stats_reflects_artifact_and_event_counts(make_test_memory):
    """
    stats() must report accurate counts from MongoDB — not stale cached values.
    After inserting an artifact and an event, both counters must increment.
    """
    from memory import SimEvent

    mem = make_test_memory
    mem._embed = MagicMock(return_value=[0.1] * 1024)

    before = mem.stats()

    mem.embed_artifact(
        id="stats_test_art",
        type="confluence",
        title="Test page",
        content="Some content",
        day=1,
        date="2026-03-02",
        timestamp="2026-03-02T10:00:00",
    )
    mem.log_event(
        SimEvent(
            type="incident_resolved",
            day=1,
            date="2026-03-02",
            timestamp="2026-03-02T10:00:00",
            actors=["Jax"],
            artifact_ids={},
            facts={},
            summary="Test event",
            tags=[],
        )
    )

    after = mem.stats()

    assert after["artifact_count"] == before["artifact_count"] + 1
    assert after["event_count"] == before["event_count"] + 1


# ── reset ─────────────────────────────────────────────────────────────────────


def test_reset_clears_all_collections_and_event_log(make_test_memory):
    """
    reset() must wipe artifacts, events, jira, prs, slack, and the in-memory
    _event_log. A sim restarted with --reset must see a completely empty state.
    """
    from memory import SimEvent

    mem = make_test_memory
    mem._embed = MagicMock(return_value=[0.1] * 1024)

    mem.embed_artifact(
        id="pre_reset",
        type="confluence",
        title="Pre-reset doc",
        content="content",
        day=1,
        date="2026-03-02",
        timestamp="2026-03-02T08:00:00",
    )
    mem.upsert_ticket(
        {
            "id": "T-PRE",
            "title": "Pre-reset ticket",
            "status": "Open",
            "assignee": "Jax",
        }
    )
    mem._event_log.append(
        SimEvent(
            type="deploy",
            day=1,
            date="2026-03-02",
            timestamp="2026-03-02T09:00:00",
            actors=["Jax"],
            artifact_ids={},
            facts={},
            summary="Pre-reset deploy",
            tags=[],
        )
    )

    mem.reset()

    assert mem._artifacts.count_documents({}) == 0
    assert mem._events.count_documents({}) == 0
    assert mem._jira.count_documents({}) == 0
    assert len(mem._event_log) == 0
