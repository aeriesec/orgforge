"""
Real-world structural grounding for OrgForge content rendering.

See README.md for architecture. The module is opt-in via the
ORGFORGE_GROUNDING_ENABLED environment flag; when disabled, all hooks are
no-ops and OrgForge produces its baseline pure-synthetic output.
"""
import os

GROUNDING_ENABLED = os.environ.get(
    "ORGFORGE_GROUNDING_ENABLED", ""
).lower() in {"1", "true", "yes", "on"}

__all__ = ["GROUNDING_ENABLED"]
