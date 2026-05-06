"""
config_loader.py
================
Single source of truth for all constants.

"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml


SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

_CONFIG_PATH_ENV = os.environ.get("ORGFORGE_CONFIG_PATH")
CONFIG_PATH = (
    Path(_CONFIG_PATH_ENV).expanduser()
    if _CONFIG_PATH_ENV
    else PROJECT_ROOT / "config" / "config.yaml"
)
if not CONFIG_PATH.is_absolute():
    CONFIG_PATH = PROJECT_ROOT / CONFIG_PATH


with open(CONFIG_PATH, "r") as _f:
    _data = yaml.safe_load(_f)

if _data is None:
    raise ValueError("Config file is empty")

CONFIG: Dict[str, Any] = _data

_EXPORT_DIR_RAW = CONFIG["simulation"].get("output_dir", str(PROJECT_ROOT / "export"))
EXPORT_DIR = Path(_EXPORT_DIR_RAW).expanduser()
if not EXPORT_DIR.is_absolute():
    EXPORT_DIR = PROJECT_ROOT / EXPORT_DIR

EXPORT_DIR.mkdir(parents=True, exist_ok=True)


COMPANY_NAME = CONFIG["simulation"]["company_name"]
COMPANY_DOMAIN = CONFIG["simulation"]["domain"]
COMPANY_DESCRIPTION = CONFIG["simulation"]["company_description"]
INDUSTRY = CONFIG["simulation"].get("industry", "technology")
BASE = str(EXPORT_DIR)
ORG_CHART = CONFIG["org_chart"]
LEADS = CONFIG["leads"]
PERSONAS = CONFIG["personas"]
DEFAULT_PERSONA = CONFIG["default_persona"]
LEGACY = CONFIG["legacy_system"]
PRODUCT_PAGE = CONFIG.get("product_page", "Product Launch")

DEPARTED_EMPLOYEES: Dict[str, Dict] = {
    gap["name"]: {
        "left": gap["left"],
        "role": gap["role"],
        "dept": gap["dept"],
        "knew_about": gap["knew_about"],
        "documented_pct": gap["documented_pct"],
    }
    for gap in CONFIG.get("knowledge_gaps", [])
}

ALL_NAMES = [name for dept in ORG_CHART.values() for name in dept]
LIVE_ORG_CHART = {dept: list(members) for dept, members in ORG_CHART.items()}
LIVE_PERSONAS = {k: dict(v) for k, v in PERSONAS.items()}

_PRESET_NAME = CONFIG.get("quality_preset", "local_gpu")
_PRESET = CONFIG["model_presets"][_PRESET_NAME]
_PROVIDER = _PRESET.get("provider", "ollama")


def resolve_role(role_key: str) -> str:
    dept = CONFIG.get("roles", {}).get(role_key)
    if dept and dept in LEADS:
        return LEADS[dept]

    if dept and dept in CONFIG.get("org_chart", {}):
        members = CONFIG["org_chart"][dept]
        if members:
            return members[0]
    return next(iter(LEADS.values()))
