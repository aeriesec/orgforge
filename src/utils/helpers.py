from typing import Dict, List

from config_loader import LIVE_ORG_CHART


def dept_of_name(name: str) -> str:
    for dept, members in LIVE_ORG_CHART.items():
        if name in members:
            return dept
    return "Unknown"
