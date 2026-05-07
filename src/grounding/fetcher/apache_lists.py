"""
Apache lists.apache.org — Pony Mail Foal API.

Public domain corporate-style mailing list archives. Closest available real
source for:
  - Internal status-update email (`ticket_completion_email`)
  - Cross-team coordination email
  - Outbound professional formal email patterns (when filtered to release
    announcements / customer-facing announce@ lists)

API: https://lists.apache.org/api/mbox.lua?list=<list>&domain=<domain>&d=YYYY-MM
Returns mbox content with full RFC-822 headers preserved.

We pull a small curated set of lists and parse threads from the most recent
months. Each thread (group of related messages by Message-ID + In-Reply-To)
becomes one Seed.
"""
from __future__ import annotations

import email as _email
import io
import logging
import mailbox
from email import policy
from typing import Optional

from .base import BaseFetcher
from .seed import Seed, LICENSE_PUBLIC_RECORDS

logger = logging.getLogger("orgforge.grounding.fetcher.apache_lists")


# Curated lists — mix of dev (engineering), user (customer-facing), and
# announce (formal outbound) for varied tones.
LIST_QUERIES: list[tuple[str, str]] = [
    ("dev", "kafka.apache.org"),         # eng status, vendor coordination
    ("dev", "spark.apache.org"),         # release planning, eng status
    ("user", "kafka.apache.org"),        # user-facing Q&A
    ("announce", "apache.org"),          # formal outbound announcements
    ("dev", "cassandra.apache.org"),     # eng coordination
]
# Recent months — Pony Mail returns latest by default; we hit a few back.
RECENT_MONTHS = ["2026-04", "2026-03", "2026-02"]


class ApacheListsFetcher(BaseFetcher):
    name = "apache_lists"
    tier = 1
    license_tag = LICENSE_PUBLIC_RECORDS

    BASE_URL = "https://lists.apache.org/api/mbox.lua"

    GENRE_HINTS: dict[str, str] = {
        "internal_sprint_planning": "release plan discussion",
        "ticket_completion_email": "status update",
        "outbound_sales_email": "announce release available",
        "supplier_eco_vendor_email": "engineering change request",
        "exec_launch_decision": "vote release",
    }

    def _query(self, genre: str, n: int, query_hint: str) -> list[Seed]:
        hint = (query_hint or self.GENRE_HINTS.get(genre, "")).lower()
        seeds: list[Seed] = []
        for list_name, domain in LIST_QUERIES:
            if len(seeds) >= n:
                break
            for ym in RECENT_MONTHS:
                if len(seeds) >= n:
                    break
                mbox_text = self._cached_get(
                    self.BASE_URL,
                    params={"list": list_name, "domain": domain, "d": ym},
                    json_response=False,
                )
                if not mbox_text:
                    continue
                threads = self._extract_threads_from_mbox(mbox_text, hint)
                for thread in threads:
                    if len(seeds) >= n:
                        break
                    seeds.append(self._seed_from_thread(thread, genre, list_name, domain, ym))
        logger.info(
            "[%s] genre=%s returned %d/%d seeds",
            self.name, genre, len(seeds), n,
        )
        return seeds

    def _extract_threads_from_mbox(
        self, mbox_text: str, hint: str
    ) -> list[list[dict]]:
        """Parse mbox text → list of threads (each thread = list of message
        dicts). Threading by Message-ID + In-Reply-To headers.
        Filters to threads that touch the hint string when provided."""
        # mailbox can't read from a string directly, so we use mailbox.mboxMessage
        # via email.message_from_bytes on individual chunks.
        messages: list[dict] = []
        # Split mbox by 'From ' line at start (RFC standard)
        chunks: list[str] = []
        current: list[str] = []
        for line in mbox_text.splitlines(keepends=True):
            if line.startswith("From ") and current:
                chunks.append("".join(current))
                current = [line]
            else:
                current.append(line)
        if current:
            chunks.append("".join(current))

        for chunk in chunks:
            try:
                msg = _email.message_from_string(chunk, policy=policy.default)
            except Exception:
                continue
            mid = msg.get("Message-ID", "")
            if not mid:
                continue
            in_reply = msg.get("In-Reply-To", "")
            from_ = str(msg.get("From", ""))
            subject = str(msg.get("Subject", ""))
            date = str(msg.get("Date", ""))
            body = ""
            try:
                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type() == "text/plain":
                            try:
                                body = part.get_content()
                                break
                            except Exception:
                                pass
                else:
                    body = msg.get_content()
            except Exception:
                body = msg.get_payload(decode=False) or ""
                if not isinstance(body, str):
                    body = str(body)
            messages.append({
                "id": mid,
                "in_reply_to": in_reply,
                "from": from_,
                "subject": subject,
                "date": date,
                "body": body[:2000],
            })

        # Group by root: walk in_reply_to chain to find root message ID.
        by_id = {m["id"]: m for m in messages}
        thread_by_root: dict[str, list[dict]] = {}
        for m in messages:
            cur_id, cur_msg = m["id"], m
            seen = set()
            while cur_msg.get("in_reply_to") and cur_msg["in_reply_to"] in by_id and cur_id not in seen:
                seen.add(cur_id)
                cur_msg = by_id[cur_msg["in_reply_to"]]
                cur_id = cur_msg["id"]
            thread_by_root.setdefault(cur_id, []).append(m)
        threads = list(thread_by_root.values())

        # Sort each thread chronologically + filter
        result: list[list[dict]] = []
        for t in threads:
            t.sort(key=lambda x: x.get("date", ""))
            if len(t) < 2:
                continue
            if hint:
                joined = (
                    " ".join(m.get("subject", "") + " " + m.get("body", "") for m in t)
                ).lower()
                if hint not in joined:
                    # Soft filter — drop ~half on miss but keep some so we
                    # don't end up with 0 seeds when the hint is rare.
                    if len(result) > 0:
                        continue
            result.append(t)
        return result

    def _seed_from_thread(
        self, thread: list[dict], genre: str, list_name: str, domain: str, ym: str
    ) -> Seed:
        first = thread[0]
        body = "\n\n".join(
            f"From: {m['from']}\nDate: {m['date']}\nSubject: {m['subject']}\n\n{m['body']}"
            for m in thread[:8]
        )
        actors = list({m["from"] for m in thread})
        return self._make_seed(
            seed_id=f"apache_lists/{list_name}-{domain}/{first['id'].strip('<>')[:32]}",
            url=f"https://lists.apache.org/list.html?{list_name}@{domain}",
            title=first.get("subject", f"Apache {list_name}@{domain} thread"),
            body=body,
            actors=actors,
            timestamps=[m.get("date", "") for m in thread],
            topic_tags=[list_name, domain],
            target_genre=genre,
            raw={
                "list": list_name,
                "domain": domain,
                "month": ym,
                "thread_size": len(thread),
            },
        )
