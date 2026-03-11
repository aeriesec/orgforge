from unittest.mock import MagicMock


def test_embedder_fallback_mechanism():
    """Ensures the embedder fallback generates a deterministic vector of the correct dimension."""
    from memory import BaseEmbedder

    # Create a dummy embedder that ONLY uses the fallback
    class DummyEmbedder(BaseEmbedder):
        def embed(self, text):
            return self._fallback(text)

    dims = 1024
    embedder = DummyEmbedder(dims=dims)

    vec1 = embedder.embed("The server is on fire")
    vec2 = embedder.embed("The server is on fire")
    vec3 = embedder.embed("Everything is fine")

    # 1. It must match the exact dimensions required by the DB
    assert len(vec1) == dims

    # 2. It must be deterministic (same text = same vector)
    assert vec1 == vec2

    # 3. Different text must yield a different vector
    assert vec1 != vec3


def test_memory_recall_pipeline_filters():
    """Verifies Memory.recall builds the correct MongoDB aggregation pipeline with filters."""
    # conftest autouse fixture already patches MongoClient, build_embedder,
    # and _init_vector_indexes — no per-test patching needed.
    from memory import Memory

    mem = Memory()
    mem._artifacts.aggregate = MagicMock(return_value=[])

    # Trigger a search with all optional filters
    mem.recall(
        query="database crash",
        n=5,
        type_filter="jira",
        day_range=(2, 8),
    )

    # Intercept the pipeline that was sent to MongoDB
    args, kwargs = mem._artifacts.aggregate.call_args
    pipeline = args[0]

    # Extract the $vectorSearch stage
    vector_search_stage = pipeline[0]["$vectorSearch"]
    search_filter = vector_search_stage.get("filter", {})

    # Assert the filters were correctly translated into MongoDB syntax
    assert search_filter["type"]["$eq"] == "jira"
    assert search_filter["day"]["$gte"] == 2
    assert search_filter["day"]["$lte"] == 8
    assert vector_search_stage["limit"] == 5


def test_memory_recall_pipeline_as_of_time_datetime():
    """
    recall() must translate a datetime as_of_time into a $lte timestamp
    filter inside $vectorSearch.  Accepts datetime objects from SimClock.
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
    recall() must also accept a pre-formatted ISO string for as_of_time —
    legacy callers that already hold a .isoformat() string must not break.
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
    When as_of_time is None no timestamp filter must appear in the pipeline —
    genesis runs and offline tools must see the full artifact set.
    """
    from memory import Memory

    mem = Memory()
    mem._artifacts.aggregate = MagicMock(return_value=[])

    mem.recall(query="anything", n=3)

    pipeline = mem._artifacts.aggregate.call_args[0][0]
    search_filter = pipeline[0]["$vectorSearch"].get("filter", {})

    assert "timestamp" not in search_filter


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

    # Simulate saving to DB and reading back
    serialized = original_event.to_dict()
    restored_event = SimEvent.from_dict(serialized)

    # Verify complex nested structures survived the round trip
    assert restored_event.type == "incident_resolved"
    assert "Bob" in restored_event.actors
    assert restored_event.artifact_ids["pr"] == "PR-100"
    assert restored_event.facts["duration_days"] == 2
    assert "p1" in restored_event.tags
