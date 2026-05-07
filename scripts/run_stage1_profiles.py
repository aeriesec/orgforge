"""
Stage 1 — build cached genre pattern profiles for the realworld-grounding-poc.

For each genre in the taxonomy (or a filtered subset), this script:

  1. Calls dispatcher.stage1_corpus(genre, n) to fetch real-world seeds via
     Tier-1 / Tier-2 / Tier-3 sources.
  2. Computes deterministic structural stats over those seeds.
  3. (Unless --dry-run) makes one LLM call to extract higher-level patterns
     (tone, vagueness, dialog-act sequence, resolution shape, drift, threading,
     rendering hints).
  4. Saves the merged profile to src/grounding/profiles/<genre>.yaml.

Usage:
    # No-cost validation (deterministic stats only):
    python scripts/run_stage1_profiles.py --dry-run

    # Live run, all 14 genres, gpt-5-mini (the grounding_worker):
    python scripts/run_stage1_profiles.py

    # Subset:
    python scripts/run_stage1_profiles.py --genres beta_rider_safety_report \
        hardware_firmware_incident_thread

The script is idempotent: existing profiles are overwritten only when the
fetch + extraction succeed. Failed extractions leave any prior profile in place.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("ORGFORGE_CONFIG_PATH", str(REPO_ROOT / "config" / "velomind.yaml"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
)
logger = logging.getLogger("stage1")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip the LLM extraction step. Profiles will contain deterministic "
        "stats only. No LLM cost.",
    )
    parser.add_argument(
        "--genres",
        nargs="*",
        default=None,
        help="Optional subset of genres to run. Defaults to all.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Override the per-genre sample size (default = sample_n from taxonomy).",
    )
    args = parser.parse_args()

    from grounding.fetcher.dispatcher import FetcherDispatcher
    from grounding.profile_extractor import build_profile, save_profile

    disp = FetcherDispatcher()
    all_genres = list((disp.taxonomy.get("genres") or {}).keys())
    target_genres = args.genres if args.genres else all_genres

    llm = None
    if not args.dry_run:
        from grounding.llm_client import GroundingLLM
        llm = GroundingLLM()
        logger.info("[stage1] LLM client constructed (lazy; first call will resolve model).")

    summary: list[dict] = []
    for genre in target_genres:
        if genre not in all_genres:
            logger.warning("[stage1] genre %s not in taxonomy; skipping", genre)
            continue
        logger.info("[stage1] === genre: %s ===", genre)
        try:
            seeds = disp.stage1_corpus(genre, n=args.n)
        except Exception as exc:
            logger.error("[stage1] %s fetch failed: %s", genre, exc)
            summary.append({"genre": genre, "status": "fetch_failed"})
            continue

        if not seeds:
            logger.warning("[stage1] %s: 0 seeds. Skipping profile.", genre)
            summary.append({"genre": genre, "status": "no_seeds"})
            continue

        profile = build_profile(genre, seeds, llm=llm, dry_run=args.dry_run)
        path = save_profile(genre, profile)
        det = profile["deterministic"]
        has_llm = profile["llm_patterns"] is not None
        logger.info(
            "[stage1] %s: %d seeds, mean body=%d chars, llm=%s → %s",
            genre,
            det["sample_count"],
            det["body_length_chars"]["mean"],
            "yes" if has_llm else "no",
            path.name,
        )
        summary.append({
            "genre": genre,
            "status": "ok",
            "seed_count": det["sample_count"],
            "llm": has_llm,
            "path": str(path),
        })

    # Summary
    print("\n=== Stage 1 summary ===")
    ok = sum(1 for s in summary if s["status"] == "ok")
    no_seeds = sum(1 for s in summary if s["status"] == "no_seeds")
    failed = sum(1 for s in summary if s["status"] == "fetch_failed")
    print(f"  ok={ok}  no_seeds={no_seeds}  fetch_failed={failed}")
    for s in summary:
        if s["status"] == "ok":
            print(f"  ✅ {s['genre']:<40} seeds={s['seed_count']:<3} llm={s['llm']}")
        else:
            print(f"  ⚠️ {s['genre']:<40} {s['status']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
