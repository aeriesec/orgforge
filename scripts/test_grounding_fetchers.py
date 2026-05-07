"""
Smoke test for the grounding fetchers.

Runs a small Stage-1 corpus pull for two representative genres so we can
verify (a) APIs are reachable, (b) seed schema is well-formed, (c) per-source
fallback works when a source fails.

Usage:
    python scripts/test_grounding_fetchers.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from pprint import pprint

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
)

from grounding.fetcher.dispatcher import FetcherDispatcher  # noqa: E402


def _summarise(seeds, max_print=2):
    print(f"  → {len(seeds)} seeds")
    for s in seeds[:max_print]:
        print(f"    - source={s.source} tier={s.tier} id={s.seed_id}")
        print(f"      title: {s.title[:120]}")
        body_preview = s.body.replace("\n", " ")[:160] if s.body else "(empty)"
        print(f"      body : {body_preview}")
        print(f"      url  : {s.url}")


def main() -> int:
    disp = FetcherDispatcher()

    print("\n[1] Stage 1 — beta_rider_safety_report (NHTSA VOQ + CPSC + ConvoKit)")
    seeds = disp.stage1_corpus("beta_rider_safety_report", n=5)
    _summarise(seeds)

    print("\n[2] Stage 1 — hardware_firmware_incident_thread (NHTSA + ROS + ES)")
    seeds = disp.stage1_corpus("hardware_firmware_incident_thread", n=4)
    _summarise(seeds)

    print("\n[3] Stage 2 — beta_rider_safety_report topical 'cold start hill assist'")
    seeds = disp.stage2_topical(
        "beta_rider_safety_report",
        query_hint="cold start hill assist delay",
        n=2,
    )
    _summarise(seeds)

    print("\n[4] Stage 2 — supplier_eco_vendor_email topical 'cloud services agreement'")
    seeds = disp.stage2_topical(
        "supplier_eco_vendor_email",
        query_hint="cloud services agreement",
        n=2,
    )
    _summarise(seeds)

    # Per-fetcher independent reachability tests so we know the status of every
    # source even when the dispatcher's higher-priority sources fill the quota.
    print("\n=== Per-fetcher reachability ===")
    from grounding.fetcher.nhtsa import NHTSAVOQFetcher, NHTSARecallFetcher
    from grounding.fetcher.cpsc import CPSCRecallFetcher
    from grounding.fetcher.sec_edgar import SECEdgarFetcher
    from grounding.fetcher.ros_discourse import ROSDiscourseFetcher
    from grounding.fetcher.convokit_local import ConvokitFetcher
    from grounding.fetcher.forum_scraper import EndlessSphereFetcher, AdafruitForumFetcher

    direct_tests = [
        ("NHTSA VOQ",       NHTSAVOQFetcher,     "beta_rider_safety_report",       ""),
        ("NHTSA Recall",    NHTSARecallFetcher,  "hardware_firmware_incident_thread", ""),
        ("CPSC Recall",     CPSCRecallFetcher,   "beta_rider_safety_report",       "battery"),
        ("SEC EDGAR EX-10", SECEdgarFetcher,     "supplier_eco_vendor_email",      "supply agreement"),
        ("SEC EDGAR S-1",   SECEdgarFetcher,     "investor_diligence_qa",          "supply concentration"),
        ("ROS Discourse",   ROSDiscourseFetcher, "hardware_firmware_incident_thread", "firmware crash"),
        ("Endless Sphere",  EndlessSphereFetcher, "battery_lab_discussion",         ""),
        ("Adafruit Forum",  AdafruitForumFetcher, "knowledge_gap_escalation",       ""),
        ("Convokit local",  ConvokitFetcher,     "investor_diligence_qa",          ""),
    ]
    for label, cls, genre, hint in direct_tests:
        try:
            inst = cls()
            seeds = inst.fetch_topical(genre, query_hint=hint, n=1) if hint else inst.fetch_for_genre(genre, n=1)
            status = "✅" if seeds else "⚠️ empty"
            ex = seeds[0].title[:80] if seeds else "—"
            print(f"  {status} {label:<18} {genre:<35} → {ex}")
        except Exception as exc:
            print(f"  ❌ {label:<18} {genre:<35} → {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
