from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("ORGFORGE_CONFIG_PATH", str(REPO_ROOT / "config" / "velomind.yaml"))

from config_loader import BASE, CONFIG  # noqa: E402
import genesis  # noqa: E402
from flow import OrgForgeSimulation, PLANNER_MODEL  # noqa: E402
from seed_velomind_baseline import seed_baseline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed and run the VeloMind OrgForge scenario."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe the configured MongoDB database and export directory first.",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Run OrgForge without importing the worldsim VeloMind fixture.",
    )
    args = parser.parse_args()

    mem = genesis.initialize(config=CONFIG, planner_llm=PLANNER_MODEL, reset=args.reset)

    if not args.skip_baseline:
        counts = seed_baseline(mem, CONFIG, Path(BASE))
        print(f"[velomind] Baseline imported: {counts}")

    sim = OrgForgeSimulation(mem=mem)
    sim.run()


if __name__ == "__main__":
    main()
