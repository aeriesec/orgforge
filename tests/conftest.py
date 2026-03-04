import pytest
import yaml
from unittest.mock import patch, MagicMock

@pytest.fixture(autouse=True)
def mock_config_and_db():
    """
    Prevents tests from actually trying to load local files or 
    connect to MongoDB/Ollama during initialization.
    """
    mock_cfg = {
        "simulation": {"company_name": "TestCorp", "domain": "test.com", "start_date": "2026-01-01", "max_days": 1},
        "model_presets": {"local_cpu": {"planner": "mock", "worker": "mock"}},
        "quality_preset": "local_cpu",
        "org_chart": {"Engineering": ["Alice"]},
        "leads": {"Engineering": "Alice"},
        "personas": {"Alice": {"style": "casual", "expertise": ["coding"], "tenure": "1y", "stress": 10}},
        "default_persona": {"style": "standard", "expertise": [], "tenure": "1y", "stress": 10},
        "legacy_system": {"name": "OldDB", "description": "Legacy", "project_name": "Modernize"},
        "morale": {"initial": 0.8, "daily_decay": 0.99, "good_day_recovery": 0.05}
    }
    
    with patch("builtins.open", MagicMock()), \
         patch("yaml.safe_load", return_value=mock_cfg), \
         patch("pymongo.MongoClient"), \
         patch("flow.build_llm"):
        yield