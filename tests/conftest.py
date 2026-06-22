import pytest
import mongomock
import builtins
from unittest.mock import patch, MagicMock
from memory import Memory

real_open = builtins.open


def smart_open(filename, mode="r", *args, **kwargs):
    """
    Pass-through mock: Allows third-party libraries (like CrewAI and LiteLLM)
    to read their internal files. Mocks OrgForge config reads and ALL file writes.
    """
    fname = str(filename)

    if mode in ("w", "w+", "a", "a+", "wb", "ab") or "config.yaml" in fname:
        m = MagicMock()
        m.__enter__.return_value = m
        m.read.return_value = ""
        return m

    return real_open(filename, mode, *args, **kwargs)

@pytest.fixture(autouse=True)
def patch_mongomock_search_indexes():
    """
    mongomock doesn't implement Atlas Search index methods
    (list_search_indexes, create_search_index, drop_search_index).
    This patches them to no-ops so Memory._init_text_indexes()
    runs without error during tests.
    """
    def _noop_list_search_indexes(self):
        return []

    def _noop_create_search_index(self, model=None):
        return "mock-index"

    def _noop_drop_search_index(self, name):
        pass

    with patch.object(
        mongomock.Collection, "list_search_indexes", _noop_list_search_indexes, create=True
    ), patch.object(
        mongomock.Collection, "create_search_index", _noop_create_search_index, create=True
    ), patch.object(
        mongomock.Collection, "drop_search_index", _noop_drop_search_index, create=True
    ):
        yield

@pytest.fixture(autouse=True)
def patch_search_artifacts_text():
    """
    mongomock doesn't support $search (Atlas Search aggregation stage).
    Patch search_artifacts_text to return empty results during tests.
    """
    with patch.object(Memory, "search_artifacts_text", return_value=[]):
        yield
        
@pytest.fixture
def make_test_memory():
    mem = Memory(mongo_client=mongomock.MongoClient())
    mem.log_event = MagicMock(wraps=mem.log_event)
    return mem


@pytest.fixture(autouse=True)
def mock_config_and_db():
    """
    Prevents tests from actually trying to load local files or
    connect to MongoDB during initialization.
    """
    mock_cfg = {
        "simulation": {
            "company_name": "TestCorp",
            "domain": "test.com",
            "start_date": "2026-01-01",
            "max_days": 1,
        },
        "model_presets": {"local_gpu": {"planner": "mock", "worker": "mock"}},
        "quality_preset": "local_gpu",
        "org_chart": {"Engineering": ["Alice"]},
        "leads": {"Engineering": "Alice"},
        "personas": {
            "Alice": {
                "style": "casual",
                "expertise": ["coding"],
                "tenure": "1y",
                "stress": 10,
            }
        },
        "default_persona": {
            "style": "standard",
            "expertise": [],
            "tenure": "1y",
            "stress": 10,
        },
        "legacy_system": {
            "name": "OldDB",
            "description": "Legacy",
            "project_name": "Modernize",
        },
        "morale": {"initial": 0.8, "daily_decay": 0.99, "good_day_recovery": 0.05},
        "roles": {
            "on_call_engineer": "Engineering",
            "incident_commander": "Engineering",
            "postmortem_writer": "Engineering",
        },
        "incident_triggers": ["crash", "fail", "error"],
        "external_contacts": [],
    }

    with (
        patch("builtins.open", side_effect=smart_open),
        patch("yaml.safe_load", return_value=mock_cfg),
        patch("memory.MongoClient"),
        patch("agent_factory.Agent"),
        patch("normal_day.Task"),
        patch("normal_day.Crew") as mock_crew_cls,
    ):
        mock_crew_cls.return_value.kickoff.return_value = (
            "Alice: Hello.\nBob: Hi there."
        )
        yield
