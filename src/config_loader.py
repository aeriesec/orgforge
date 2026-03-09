"""
config_loader.py
================
Single source of truth for all config constants derived from config.yaml.

"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

# ── Paths ─────────────────────────────────────────────────────────────────────
SRC_DIR      = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
EXPORT_DIR  = PROJECT_ROOT / "export"

EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load YAML ─────────────────────────────────────────────────────────────────
with open(CONFIG_PATH, "r") as _f:
    _data = yaml.safe_load(_f)

if _data is None:
    raise ValueError("Config file is empty")

CONFIG: Dict[str, Any] = _data

# ── Top-level constants ───────────────────────────────────────────────────────
COMPANY_NAME    = CONFIG["simulation"]["company_name"]
COMPANY_DOMAIN  = CONFIG["simulation"]["domain"]
INDUSTRY        = CONFIG["simulation"].get("industry", "technology")
BASE            = CONFIG["simulation"].get("output_dir", str(EXPORT_DIR))
ORG_CHART       = CONFIG["org_chart"]
LEADS           = CONFIG["leads"]
PERSONAS        = CONFIG["personas"]
DEFAULT_PERSONA = CONFIG["default_persona"]
LEGACY          = CONFIG["legacy_system"]
PRODUCT_PAGE    = CONFIG.get("product_page", "Product Launch")

DEPARTED_EMPLOYEES: Dict[str, Dict] = {
    gap["name"]: {
        "left":           gap["left"],
        "role":           gap["role"],
        "knew_about":     gap["knew_about"],
        "documented_pct": gap["documented_pct"],
    }
    for gap in CONFIG.get("knowledge_gaps", [])
}

ALL_NAMES      = [name for dept in ORG_CHART.values() for name in dept]
LIVE_ORG_CHART = {dept: list(members) for dept, members in ORG_CHART.items()}
LIVE_PERSONAS  = {k: dict(v) for k, v in PERSONAS.items()}

# ── Model preset ──────────────────────────────────────────────────────────────
_PRESET_NAME = CONFIG.get("quality_preset", "local_cpu")
_PRESET      = CONFIG["model_presets"][_PRESET_NAME]
_PROVIDER    = _PRESET.get("provider", "ollama")