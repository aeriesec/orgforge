"""
backfill_escalation_artifacts.py
=================================
One-time script to create ESC-{ticket_id} artifacts in MongoDB
from existing escalation_chain SimEvents.

Run once against your existing database before re-running export_to_hf.py.
No sim re-run required.
"""

import logging
from memory import Memory

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("backfill")


def backfill_escalation_artifacts(mem: Memory) -> None:
    escalation_events = list(mem._events.find({"type": "escalation_chain"}, {"_id": 0}))
    logger.info(f"Found {len(escalation_events)} escalation_chain events")

    skipped = 0
    created = 0

    for event in escalation_events:
        ticket_id = event.get("artifact_ids", {}).get("jira", "")
        if not ticket_id:
            logger.warning(
                f"  escalation_chain event missing jira artifact_id — skipping"
            )
            skipped += 1
            continue

        embed_id = f"ESC-{ticket_id}"

        # Check if already exists — idempotent
        existing = mem._artifacts.find_one({"_id": embed_id})
        if existing:
            logger.info(f"  {embed_id} already exists — skipping")
            skipped += 1
            continue

        facts = event.get("facts", {})
        escalation_actors = facts.get("escalation_actors", [])
        escalation_narrative = facts.get("escalation_narrative", "")

        if not escalation_actors and not escalation_narrative:
            logger.warning(f"  {embed_id} has no actors or narrative — skipping")
            skipped += 1
            continue

        content = (
            f"Escalation actors: {', '.join(escalation_actors)}\n{escalation_narrative}"
        )

        mem.embed_artifact(
            id=embed_id,
            type="escalation",
            title=f"Escalation chain for {ticket_id}",
            content=content,
            day=event.get("day", 0),
            date=event.get("date", ""),
            timestamp=event.get("timestamp", ""),
            metadata={
                "ticket_id": ticket_id,
                "escalation_actors": escalation_actors,
            },
        )
        logger.info(f"  ✓ Created {embed_id} — {len(escalation_actors)} actors")
        created += 1

    logger.info(f"Backfill complete: {created} created, {skipped} skipped")


if __name__ == "__main__":
    mem = Memory()
    backfill_escalation_artifacts(mem)
    logger.info("Done — re-run export_to_hf.py to rebuild the corpus and baselines")
