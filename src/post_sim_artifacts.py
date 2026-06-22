"""
post_sim_artifacts.py
=====================
Post-simulation artifact generator for OrgForge.

Produces three artifact sets that are fully derivable from the SimEvent log
without running the daily simulation loop again:

  export/nps/
      responses/{org_name}_{day}.json   — per-customer NPS survey response
      summary.json                      — aggregate score, detractor/promoter counts

  export/invoices/
      {invoice_id}.json                 — per-customer invoice with SLA credit line items

  export/datadog/
      metrics.jsonl                     — time-series health + latency metrics (Prometheus-compatible)
      alerts.jsonl                      — fired alert records linked to incidents

"""

from __future__ import annotations

import argparse
import json
import logging
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config_loader import (
    BASE,
    COMPANY_DESCRIPTION,
    COMPANY_DOMAIN,
    COMPANY_NAME,
    CONFIG,
    INDUSTRY,
)
from memory import Memory, SimEvent
from agent_factory import make_agent
from crewai import Task, Crew

logger = logging.getLogger("orgforge.post_sim")


SLA_BREACH_THRESHOLD_DAYS = 1


SLA_CREDIT_RATE = 0.02


DEFAULT_CONTRACT_VALUE = 50_000


INVOICE_PAYMENT_TERMS_DAYS = 30


METRIC_INTERVAL_MINS = 15


BASE_LATENCY_MS = 120


MAX_LATENCY_MULTIPLIER = 8.0


NPS_RESPONSE_RATE = 0.85


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)


def _append_jsonl(path: Path, record: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(record) + "\n")


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _sim_date(day: int, start_date: datetime) -> datetime:
    """Convert a sim day number to a real calendar date, skipping weekends."""
    current = start_date
    biz_day = 1
    while biz_day < day:
        current += timedelta(days=1)
        if current.weekday() < 5:
            biz_day += 1
    return current


def _iso(dt: datetime) -> str:
    return dt.isoformat()


class EventIndex:
    """
    Walks the SimEvent log once and builds every lookup structure the three
    artifact writers need. Keeps all downstream logic O(1) per query.
    """

    def __init__(self, events: List[SimEvent], start_date: datetime):
        self.start_date = start_date

        self.health_by_day: Dict[int, int] = {}

        self.incidents: Dict[str, Dict] = {}

        self.customer_tickets: Dict[str, List[Dict]] = {}

        self.contract_values: Dict[str, float] = {}

        self.customer_risk_flags: Dict[str, List[str]] = {}

        self.incident_root_causes: Dict[str, str] = {}

        self.daily_sentiment: Dict[int, List[float]] = {}

        self._index(events)

    def _index(self, events: List[SimEvent]) -> None:

        zd_open: Dict[str, Dict] = {}
        zd_linked: Dict[str, str] = {}

        for e in events:
            t = e.type
            f = e.facts

            if t == "day_summary":
                h = f.get("system_health")
                if h is not None:
                    self.health_by_day[e.day] = int(h)

            elif t == "incident_opened":
                iid = e.artifact_ids.get("jira", "")
                if iid:
                    self.incidents[iid] = {
                        "incident_id": iid,
                        "open_day": e.day,
                        "open_ts": e.timestamp,
                        "resolve_day": None,
                        "resolve_ts": None,
                        "root_cause": f.get("root_cause", ""),
                        "component": f.get("root_cause", "")[:60],
                        "duration_days": None,
                    }
                    self.incident_root_causes[iid] = f.get("root_cause", "")

            elif t == "incident_resolved":
                iid = e.artifact_ids.get("jira", "")
                if iid and iid in self.incidents:
                    self.incidents[iid]["resolve_day"] = e.day
                    self.incidents[iid]["resolve_ts"] = e.timestamp
                    dur = e.day - self.incidents[iid]["open_day"]
                    self.incidents[iid]["duration_days"] = dur

            # ── Zendesk ticket lifecycle ──────────────────────────────────────
            elif t == "zd_ticket_opened":
                tid = f.get("ticket_id", "")
                if tid:
                    zd_open[tid] = {
                        "ticket_id": tid,
                        "subject": f.get("subject", ""),
                        "org_name": f.get("org_name", "Unknown"),
                        "opened_day": e.day,
                        "resolved_day": None,
                        "escalated": False,
                    }

            elif t == "zd_tickets_escalated":
                for tid in f.get("ticket_ids", []):
                    if tid in zd_open:
                        zd_open[tid]["escalated"] = True
                        zd_linked[tid] = f.get("incident_id", "")

            elif t == "zd_tickets_resolved":
                for tid in f.get("ticket_ids", []):
                    if tid in zd_open:
                        zd_open[tid]["resolved_day"] = e.day

            elif t == "sf_deals_risk_flagged":
                iid = f.get("incident_id", "")
                for org in f.get("account_names", []):
                    self.customer_risk_flags.setdefault(org, [])
                    if iid and iid not in self.customer_risk_flags[org]:
                        self.customer_risk_flags[org].append(iid)

            elif t == "crm_touchpoint":
                org = f.get("account_name", "")

                if org and org not in self.contract_values:
                    self.contract_values[org] = DEFAULT_CONTRACT_VALUE

            elif t == "customer_complaint":
                sentiment = f.get("sentiment_score")
                if sentiment is not None:
                    self.daily_sentiment.setdefault(e.day, []).append(
                        _safe_float(sentiment)
                    )

        for tid, rec in zd_open.items():
            org = rec["org_name"]
            if org not in self.customer_tickets:
                self.customer_tickets[org] = []
            self.customer_tickets[org].append(rec)

        for tid, iid in zd_linked.items():
            if tid in zd_open:
                org = zd_open[tid]["org_name"]
                self.customer_risk_flags.setdefault(org, [])
                if iid and iid not in self.customer_risk_flags[org]:
                    self.customer_risk_flags[org].append(iid)


class NPSWriter:
    """
    Generates realistic NPS survey responses for each customer org.

    Score derivation (deterministic, no LLM):
      Base score: 9
      -3 if any ZD ticket was escalated to a P1 incident
      -2 per unresolved ZD ticket at sim end
      -1 per SLA breach day (incident duration > threshold)
      +1 if all tickets resolved quickly (< 1 day each)
      Clamped to [0, 10].

    NPS classification:
      9-10 → Promoter
      7-8  → Passive
      0-6  → Detractor

    Optional LLM call enriches the `verbatim_comment` field.
    """

    _RESPONSE_DELAY_DAYS = 3

    def __init__(
        self,
        index: EventIndex,
        export_dir: Path,
        start_date: datetime,
        sim_end_day: int,
    ):
        self._idx = index
        self._export = export_dir / "nps"
        self._start = start_date
        self._end_day = sim_end_day

    def _score(self, org: str) -> Tuple[int, Dict]:
        """Return (nps_score, scoring_detail) for an org."""
        tickets = self._idx.customer_tickets.get(org, [])
        risk_ids = self._idx.customer_risk_flags.get(org, [])

        score = 9
        detail: Dict[str, Any] = {
            "escalated_tickets": 0,
            "unresolved_tickets": 0,
            "sla_breach_days": 0,
            "quick_resolutions": 0,
        }

        for t in tickets:
            if t["escalated"]:
                score -= 3
                detail["escalated_tickets"] += 1

            resolved_day = t.get("resolved_day")
            if resolved_day is None:
                score -= 2
                detail["unresolved_tickets"] += 1
            else:
                age = resolved_day - t["opened_day"]
                if age <= 1:
                    score += 1
                    detail["quick_resolutions"] += 1

        for iid in risk_ids:
            inc = self._idx.incidents.get(iid, {})
            dur = inc.get("duration_days") or 0
            breach_days = max(0, dur - SLA_BREACH_THRESHOLD_DAYS)
            score -= breach_days
            detail["sla_breach_days"] += breach_days

        score = max(0, min(10, score))
        return score, detail

    def _classify(self, score: int) -> str:
        if score >= 9:
            return "promoter"
        if score >= 7:
            return "passive"
        return "detractor"

    def _response_date(self, org: str) -> str:
        """Survey response date = last ticket resolution + delay, or end of sim."""
        tickets = self._idx.customer_tickets.get(org, [])
        resolved_days = [
            t["resolved_day"] for t in tickets if t.get("resolved_day") is not None
        ]
        base_day = max(resolved_days) if resolved_days else self._end_day
        response_day = min(base_day + self._RESPONSE_DELAY_DAYS, self._end_day + 5)
        return _iso(_sim_date(response_day, self._start))

    def build_responses(self) -> List[Dict]:
        """Build one response record per customer org. No LLM call here."""
        responses = []
        orgs = list(self._idx.customer_tickets.keys())

        # Also include orgs that had risk flags but no tickets (SF-only customers)
        for org in self._idx.customer_risk_flags:
            if org not in orgs:
                orgs.append(org)

        for org in orgs:
            if random.random() > NPS_RESPONSE_RATE:
                continue  # simulate non-response

            score, detail = self._score(org)
            record = {
                "response_id": f"NPS-{uuid.uuid4().hex[:8].upper()}",
                "org_name": org,
                "respondent_email": f"feedback@{org.lower().replace(' ', '')}.com",
                "submitted_at": self._response_date(org),
                "survey_type": "NPS",
                "score": score,
                "classification": self._classify(score),
                "verbatim_comment": None,  # filled by LLM batch or placeholder
                "scoring_detail": detail,
                "metadata": {
                    "triggered_by": "post_sim_survey",
                    "sim_company": COMPANY_NAME,
                },
            }
            responses.append(record)

        return responses

    def write(self, responses: List[Dict]) -> None:
        self._export.mkdir(parents=True, exist_ok=True)
        for r in responses:
            safe_org = r["org_name"].lower().replace(" ", "_")
            path = self._export / "responses" / f"{safe_org}.json"
            _write_json(path, r)
            logger.info(
                f"  [nps] {r['org_name']}: score={r['score']} "
                f"({r['classification']}) → {path.name}"
            )

        scores = [r["score"] for r in responses]
        promoters = sum(1 for r in responses if r["classification"] == "promoter")
        passives = sum(1 for r in responses if r["classification"] == "passive")
        detractors = sum(1 for r in responses if r["classification"] == "detractor")
        n = len(scores)
        nps_score = round(((promoters - detractors) / n) * 100) if n else 0

        summary_date = _sim_date(self._end_day + 5, self._start)

        summary = {
            "generated_at": _iso(summary_date),
            "company": COMPANY_NAME,
            "response_count": n,
            "nps_score": nps_score,
            "avg_score": round(sum(scores) / n, 2) if n else 0,
            "promoters": promoters,
            "passives": passives,
            "detractors": detractors,
            "promoter_pct": round(promoters / n * 100, 1) if n else 0,
            "detractor_pct": round(detractors / n * 100, 1) if n else 0,
        }
        _write_json(self._export / "summary.json", summary)
        logger.info(
            f"  [nps] Summary: NPS={nps_score}, "
            f"promoters={promoters}, passives={passives}, detractors={detractors}"
        )


class InvoiceWriter:
    """
    Generates one invoice per customer org per billing period (monthly, derived
    from sim length). Each invoice has standard line items plus conditional
    SLA credit line items when incident duration exceeded the threshold.

    All arithmetic — no LLM calls.

    Invoice schema matches what a real SaaS billing system exports (Stripe,
    Chargebee, Zuora). Fields chosen for eval utility: an agent asked "what
    credits were issued due to the TitanDB incident?" has a deterministic answer.
    """

    def __init__(
        self,
        index: EventIndex,
        export_dir: Path,
        start_date: datetime,
        sim_end_day: int,
        mem: Memory,
    ):
        self._idx = index
        self._export = export_dir / "invoices"
        self._start = start_date
        self._end_day = sim_end_day
        self._mem = mem

        self._contract_values = dict(index.contract_values)
        self._load_contract_values_from_mongo()

    def _load_contract_values_from_mongo(self) -> None:
        """Override defaults with real opportunity amounts if SF is enabled."""
        try:
            for acc in self._mem._db["sf_accounts"].find({}, {"_id": 0}):
                org = acc.get("name", "")
                arr = acc.get("arr")

                if org and arr:
                    self._contract_values[org] = float(arr)
        except Exception:
            pass

    def _annual_value(self, org: str) -> float:
        return self._contract_values.get(org, DEFAULT_CONTRACT_VALUE)

    def _monthly_value(self, org: str) -> float:
        return round(self._annual_value(org) / 12, 2)

    def _sla_credits(self, org: str) -> List[Dict]:
        """Return credit line items for every SLA breach affecting this customer."""
        credits = []
        risk_ids = self._idx.customer_risk_flags.get(org, [])

        for iid in risk_ids:
            inc = self._idx.incidents.get(iid, {})
            dur = inc.get("duration_days") or 0
            breach_days = max(0, dur - SLA_BREACH_THRESHOLD_DAYS)
            if breach_days <= 0:
                continue

            credit_amount = round(
                self._monthly_value(org) * SLA_CREDIT_RATE * breach_days, 2
            )
            credits.append(
                {
                    "line_item_type": "sla_credit",
                    "description": (
                        f"SLA credit — incident {iid} exceeded "
                        f"{SLA_BREACH_THRESHOLD_DAYS}d SLA by {breach_days}d. "
                        f"Root cause: {inc.get('component', 'system failure')[:80]}."
                    ),
                    "incident_id": iid,
                    "breach_days": breach_days,
                    "credit_rate": SLA_CREDIT_RATE,
                    "amount": -credit_amount,  # negative = credit
                    "currency": "USD",
                }
            )

        return credits

    def build_invoices(self) -> List[Dict]:
        invoices = []

        # Billing period: one invoice covers the entire sim duration
        period_start = _iso(self._start)
        period_end = _iso(_sim_date(self._end_day, self._start))
        due_date = _iso(
            _sim_date(self._end_day, self._start)
            + timedelta(days=INVOICE_PAYMENT_TERMS_DAYS)
        )

        orgs = set(self._idx.customer_tickets.keys()) | set(
            self._idx.customer_risk_flags.keys()
        )
        # Also include any orgs known purely from contract/SF data
        for org in self._contract_values:
            orgs.add(org)

        for org in sorted(orgs):
            monthly_fee = self._monthly_value(org)
            credits = self._sla_credits(org)
            credit_total = sum(c["amount"] for c in credits)
            subtotal = round(monthly_fee + credit_total, 2)
            tax = round(subtotal * 0.08, 2)  # 8% tax — typical US SaaS
            total = round(subtotal + tax, 2)

            line_items = [
                {
                    "line_item_type": "subscription",
                    "description": f"{COMPANY_NAME} Platform — monthly subscription",
                    "quantity": 1,
                    "unit_price": monthly_fee,
                    "amount": monthly_fee,
                    "currency": "USD",
                },
                *credits,
                {
                    "line_item_type": "tax",
                    "description": "Sales tax (8%)",
                    "amount": tax,
                    "currency": "USD",
                },
            ]

            invoice = {
                "invoice_id": f"INV-{uuid.uuid4().hex[:8].upper()}",
                "invoice_date": period_end,
                "due_date": due_date,
                "status": "open",
                "billing_period": {
                    "start": period_start,
                    "end": period_end,
                },
                "customer": {
                    "org_name": org,
                    "billing_email": f"billing@{org.lower().replace(' ', '')}.com",
                },
                "vendor": {
                    "company": COMPANY_NAME,
                    "domain": COMPANY_DOMAIN,
                },
                "line_items": line_items,
                "subtotal": round(monthly_fee + credit_total, 2),
                "tax": tax,
                "total_due": total,
                "currency": "USD",
                "payment_terms": f"Net {INVOICE_PAYMENT_TERMS_DAYS}",
                "notes": (
                    f"SLA credits applied for {len(credits)} incident(s)."
                    if credits
                    else ""
                ),
                "metadata": {
                    "sla_credits_count": len(credits),
                    "credit_total": credit_total,
                    "generated_by": "orgforge_post_sim",
                },
            }
            invoices.append(invoice)

        return invoices

    def write(self, invoices: List[Dict]) -> None:
        for inv in invoices:
            fname = f"{inv['invoice_id']}.json"
            _write_json(self._export / fname, inv)
            credit_note = (
                f" | credits={inv['metadata']['credit_total']}"
                if inv["metadata"]["sla_credits_count"]
                else ""
            )
            logger.info(
                f"  [invoice] {inv['customer']['org_name']}: "
                f"total={inv['total_due']}{credit_note} → {fname}"
            )


class DatadogWriter:
    """
    Synthesises two outputs that together constitute a realistic Datadog export:

    metrics.jsonl — time-series samples at METRIC_INTERVAL_MINS resolution.
      Metrics emitted per sample:
        system.health               (gauge, 0-100)
        app.request.latency.p99     (gauge, ms)
        app.request.latency.p50     (gauge, ms)
        app.error.rate              (gauge, errors/min)
        app.throughput              (gauge, req/min)

      Interpolation rules:
        Between incident_open → incident_resolve: linear degradation from
        pre-incident health to floor value, then sharp recovery on resolve.
        Outside incidents: gentle daily drift ±2 points around checkpoint health.

    alerts.jsonl — one record per incident, formatted as a Datadog alert event.
      Fields match the Datadog Events API schema so these can be replayed
      against a real Datadog org if desired.

    Optional LLM call generates realistic monitor names (e.g.
    "High p99 latency on /api/ingest") from the root cause string.
    """

    _HEALTH_FLOOR = 15

    def __init__(
        self,
        index: EventIndex,
        export_dir: Path,
        start_date: datetime,
        sim_end_day: int,
    ):
        self._idx = index
        self._export = export_dir / "datadog"
        self._start = start_date
        self._end_day = sim_end_day

        self._monitor_names: Dict[str, str] = {}

    def _health_at(self, day: int, minute_offset: int = 0) -> float:
        """
        Return interpolated system health at a given day + intra-day minute offset.
        Incidents drag health down linearly from open to nadir, then step-recover.
        """

        base = _safe_float(
            self._idx.health_by_day.get(day)
            or self._idx.health_by_day.get(day - 1)
            or 85,
            default=85.0,
        )

        # Check if we're inside an active incident window
        for inc in self._idx.incidents.values():
            open_day = inc.get("open_day", 0)
            resolve_day = inc.get("resolve_day") or (open_day + 4)

            if not (open_day <= day <= resolve_day):
                continue

            duration_days = max(1, resolve_day - open_day)
            minutes_into_incident = (day - open_day) * 24 * 60 + minute_offset
            total_incident_minutes = duration_days * 24 * 60

            # Progress 0→1 through the incident window
            progress = min(1.0, minutes_into_incident / total_incident_minutes)

            if progress < 0.15:
                # Rapid degradation in the first 15% of the window
                drop_factor = progress / 0.15
                floor = base - (base - self._HEALTH_FLOOR) * drop_factor
            elif progress < 0.85:
                # Hold at nadir
                floor = self._HEALTH_FLOOR + random.uniform(-3, 3)
            else:
                # Recovery ramp in the final 15%
                ramp = (progress - 0.85) / 0.15
                floor = self._HEALTH_FLOOR + (base - self._HEALTH_FLOOR) * ramp

            base = max(self._HEALTH_FLOOR, min(base, floor))

        # Small intra-day jitter (±2) to avoid perfectly flat lines
        return max(0.0, min(100.0, base + random.uniform(-2, 2)))

    def _latency_p99(self, health: float) -> float:
        """p99 latency rises as health falls. Exponential to match real degradation."""
        degradation = max(0.0, 1.0 - health / 100.0)
        multiplier = 1.0 + (MAX_LATENCY_MULTIPLIER - 1.0) * (degradation**1.8)
        jitter = random.uniform(0.85, 1.15)
        return round(BASE_LATENCY_MS * multiplier * jitter, 1)

    def _latency_p50(self, p99: float) -> float:
        """p50 is roughly p99/3.5 under normal conditions, tighter during incidents."""
        ratio = random.uniform(2.8, 4.2)
        return round(p99 / ratio, 1)

    def _error_rate(self, health: float) -> float:
        """Errors per minute. Near-zero when healthy, spikes during incidents."""
        if health > 80:
            return round(random.uniform(0.0, 0.8), 2)
        if health > 50:
            return round(random.uniform(1.0, 5.0), 2)
        return round(random.uniform(8.0, 40.0), 2)

    def _throughput(self, health: float) -> float:
        """Requests per minute. Drops under incident pressure."""
        base_rps = random.uniform(180, 220)
        pressure = max(0.2, health / 100.0)
        return round(base_rps * pressure, 1)

    # ── Sample generation ─────────────────────────────────────────────────────

    def build_metrics(self) -> int:
        """Write metrics.jsonl. Returns total sample count."""
        path = self._export / "metrics.jsonl"
        sample_count = 0

        for day in range(1, self._end_day + 1):
            day_dt = _sim_date(day, self._start)
            minutes_per_day = 24 * 60
            samples_per_day = minutes_per_day // METRIC_INTERVAL_MINS

            for i in range(samples_per_day):
                minute_offset = i * METRIC_INTERVAL_MINS
                ts_dt = day_dt + timedelta(minutes=minute_offset)
                ts_unix = int(ts_dt.timestamp())
                health = self._health_at(day, minute_offset)
                p99 = self._latency_p99(health)
                p50 = self._latency_p50(p99)

                tags = [
                    "env:production",
                    f"service:{COMPANY_NAME.lower().replace(' ', '-')}",
                    f"sim_day:{day}",
                ]

                for metric, value, mtype in [
                    ("system.health", round(health, 2), "gauge"),
                    ("app.request.latency.p99", p99, "gauge"),
                    ("app.request.latency.p50", p50, "gauge"),
                    ("app.error.rate", self._error_rate(health), "gauge"),
                    ("app.throughput", self._throughput(health), "gauge"),
                ]:
                    _append_jsonl(
                        path,
                        {
                            "metric": metric,
                            "type": mtype,
                            "value": value,
                            "timestamp": ts_unix,
                            "timestamp_iso": ts_dt.isoformat(),
                            "tags": tags,
                            "host": f"prod-web-{(i % 3) + 1:02d}",
                        },
                    )
                    sample_count += 1

        logger.info(f"  [datadog] metrics.jsonl: {sample_count:,} samples written")
        return sample_count

    def build_alerts(
        self, monitor_names: Optional[Dict[str, str]] = None
    ) -> List[Dict]:
        """
        Build one alert record per incident. Returns records; caller writes them.
        monitor_names: incident_id → human-readable monitor name (from LLM batch).
        """
        alerts = []

        for iid, inc in sorted(
            self._idx.incidents.items(), key=lambda x: x[1]["open_day"]
        ):
            monitor_name = (monitor_names or {}).get(
                iid,
                f"Anomaly detected — {inc.get('component', 'system')[:60]}",
            )
            resolve_ts = inc.get("resolve_ts")

            alert = {
                "id": iid,
                "title": f"[P1] {monitor_name}",
                "text": (
                    f"## Alert\n\n"
                    f"**Monitor:** {monitor_name}\n"
                    f"**Severity:** Critical\n"
                    f"**Root cause:** {inc.get('root_cause', '')}\n\n"
                    f"System health dropped to degraded levels. "
                    f"On-call paged. See linked JIRA: {iid}."
                ),
                "alert_type": "error",
                "priority": "normal",
                "source_type_name": "Datadog",
                "date_happened": int(datetime.fromisoformat(inc["open_ts"]).timestamp())
                if inc.get("open_ts")
                else 0,
                "date_resolved": int(datetime.fromisoformat(resolve_ts).timestamp())
                if resolve_ts
                else None,
                "tags": [
                    "severity:critical",
                    f"incident:{iid}",
                    "env:production",
                    f"sim_day:{inc['open_day']}",
                ],
                "attributes": {
                    "jira_ticket": iid,
                    "root_cause": inc.get("root_cause", ""),
                    "open_day": inc["open_day"],
                    "resolve_day": inc.get("resolve_day"),
                    "duration_days": inc.get("duration_days"),
                    "monitor_name": monitor_name,
                },
            }
            alerts.append(alert)

        return alerts

    def write_alerts(self, alerts: List[Dict]) -> None:
        path = self._export / "alerts.jsonl"
        for alert in alerts:
            _append_jsonl(path, alert)
        logger.info(f"  [datadog] alerts.jsonl: {len(alerts)} alert(s) written")


def _batch_nps_comments(
    responses: List[Dict],
    worker_llm,
) -> Dict[str, str]:
    """
    Single LLM call. Returns {response_id → verbatim_comment}.

    We send all NPS responses in one prompt and ask for a JSON array back,
    then zip the results. One call regardless of customer count.
    """
    if not responses:
        return {}

    summaries = []
    for r in responses:
        detail = r.get("scoring_detail", {})
        summaries.append(
            f"- id={r['response_id']} org={r['org_name']} score={r['score']} "
            f"({r['classification']}) "
            f"escalated_tickets={detail.get('escalated_tickets', 0)} "
            f"unresolved={detail.get('unresolved_tickets', 0)} "
            f"sla_breaches={detail.get('sla_breach_days', 0)}d"
        )

    prompt_body = "\n".join(summaries)

    agent = make_agent(
        role="Customer Research Analyst",
        goal="Write realistic NPS survey verbatim comments from customer data.",
        backstory=(
            f"You work at a B2B SaaS company called {COMPANY_NAME} in the "
            f"{INDUSTRY} space. You are reviewing NPS survey data and writing "
            f"the open-ended verbatim comment each customer left."
        ),
        llm=worker_llm,
    )
    task = Task(
        description=(
            f"Each line below is a customer NPS response summary for {COMPANY_NAME}, "
            f"which {COMPANY_DESCRIPTION}.\n\n"
            f"{prompt_body}\n\n"
            f"Write one realistic verbatim comment for each customer. The comment "
            f"should sound like something a real B2B buyer would type: 1-3 sentences, "
            f"specific to their experience (support issues, uptime, sales contact), "
            f"matching the score and classification.\n\n"
            f"Respond ONLY with a JSON array in this exact order (same order as the "
            f"input list above):\n"
            f'[{{"id": "response_id", "comment": "verbatim text"}}, ...]\n'
            f"No preamble, no markdown fences. Raw JSON only."
        ),
        expected_output="JSON array of {id, comment} objects, same order as input.",
        agent=agent,
    )

    raw = str(Crew(agents=[agent], tasks=[task], verbose=False).kickoff()).strip()

    try:
        import json_repair

        parsed = json_repair.loads(raw)
        if not isinstance(parsed, list):
            parsed = []
    except Exception:
        parsed = []

    result: Dict[str, str] = {}
    for item in parsed:
        if isinstance(item, dict) and "id" in item and "comment" in item:
            result[item["id"]] = item["comment"]

    return result


def _batch_alert_names(
    incidents: Dict[str, Dict],
    worker_llm,
) -> Dict[str, str]:
    """
    Single LLM call. Returns {incident_id → monitor_name}.

    Monitor names should look like real Datadog monitor titles:
    "High p99 latency on /api/ingest", "Connection pool exhaustion — TitanDB".
    """
    if not incidents:
        return {}

    lines = [
        f"- {iid}: {inc.get('root_cause', '')[:120]}" for iid, inc in incidents.items()
    ]
    prompt_body = "\n".join(lines)

    agent = make_agent(
        role="Site Reliability Engineer",
        goal="Name Datadog monitors based on incident root causes.",
        backstory=(
            f"You are an SRE at {COMPANY_NAME} which {COMPANY_DESCRIPTION}. You maintain the Datadog "
            f"observability stack and write monitor names that are concise, "
            f"specific, and follow the convention: "
            f"'<Symptom> — <Component>' or '<Metric> on <Endpoint>'."
        ),
        llm=worker_llm,
    )
    task = Task(
        description=(
            f"Each line is an incident ID and its root cause for {COMPANY_NAME}:\n\n"
            f"{prompt_body}\n\n"
            f"Write a short Datadog monitor name for each incident (5-12 words). "
            f"Examples: 'High p99 latency — /api/search', "
            f"'Redis connection pool exhaustion under load', "
            f"'Auth token cache stampede'.\n\n"
            f"Respond ONLY with a JSON array:\n"
            f'[{{"id": "incident_id", "monitor_name": "short name"}}, ...]\n'
            f"Same order as input. No preamble, no markdown fences. Raw JSON only."
        ),
        expected_output="JSON array of {id, monitor_name} objects.",
        agent=agent,
    )

    raw = str(Crew(agents=[agent], tasks=[task], verbose=False).kickoff()).strip()

    try:
        import json_repair

        parsed = json_repair.loads(raw)
        if not isinstance(parsed, list):
            parsed = []
    except Exception:
        parsed = []

    result: Dict[str, str] = {}
    for item in parsed:
        if isinstance(item, dict) and "id" in item and "monitor_name" in item:
            result[item["id"]] = item["monitor_name"]

    return result


def run(export_dir: Path, use_llm: bool = True, only: Optional[set] = None) -> None:
    logger.info("[post_sim] Starting post-simulation artifact generation...")

    only = only or {"nps", "invoices", "datadog"}
    mem = Memory()
    events = mem.get_event_log(from_db=True)
    start_date = datetime.strptime(CONFIG["simulation"]["start_date"], "%Y-%m-%d")
    max_days = CONFIG["simulation"]["max_days"]

    logger.info(f"[post_sim] Loaded {len(events)} events from MongoDB.")

    idx = EventIndex(events, start_date)
    logger.info(
        f"[post_sim] Index built: "
        f"{len(idx.incidents)} incidents, "
        f"{len(idx.customer_tickets)} customer orgs, "
        f"{len(idx.health_by_day)} health snapshots."
    )

    responses = []
    invoices = []
    alerts = []

    if "nps" in only:
        logger.info("[post_sim] → NPS surveys")
        nps_writer = NPSWriter(idx, export_dir, start_date, max_days)
        responses = nps_writer.build_responses()

    if "invoices" in only:
        logger.info("[post_sim] → Invoices")
        inv_writer = InvoiceWriter(idx, export_dir, start_date, max_days, mem)
        invoices = inv_writer.build_invoices()

    if "datadog" in only:
        logger.info("[post_sim] → Datadog metrics")
        dd_writer = DatadogWriter(idx, export_dir, start_date, max_days)
        dd_writer.build_metrics()
        alerts = dd_writer.build_alerts()

    if use_llm:
        if "nps" in only and responses:
            logger.info("[post_sim] → LLM batch 1/2: NPS verbatim comments")
            try:
                from flow import WORKER_MODEL

                nps_comments = _batch_nps_comments(responses, WORKER_MODEL)
                for r in responses:
                    r["verbatim_comment"] = nps_comments.get(
                        r["response_id"], _nps_placeholder(r)
                    )
            except Exception as e:
                logger.warning(
                    f"[post_sim] NPS LLM call failed ({e}) — using placeholders"
                )
                for r in responses:
                    r["verbatim_comment"] = _nps_placeholder(r)

        if "datadog" in only and idx.incidents:
            logger.info("[post_sim] → LLM batch 2/2: Datadog alert monitor names")
            try:
                from flow import WORKER_MODEL

                monitor_names = _batch_alert_names(idx.incidents, WORKER_MODEL)
                alerts = dd_writer.build_alerts(monitor_names)
            except Exception as e:
                logger.warning(
                    f"[post_sim] Alert names LLM call failed ({e}) — using root causes"
                )
    else:
        if not use_llm:
            logger.info("[post_sim] LLM calls skipped (--no-llm).")
        for r in responses:
            r["verbatim_comment"] = _nps_placeholder(r)

    if "nps" in only:
        nps_writer.write(responses)
    if "invoices" in only:
        inv_writer.write(invoices)
    if "datadog" in only:
        dd_writer.write_alerts(alerts)

    logger.info(
        f"[post_sim] Done. "
        f"NPS={len(responses)}, "
        f"invoices={len(invoices)}, "
        f"alerts={len(alerts)}."
    )


def _nps_placeholder(r: Dict) -> str:
    """Deterministic fallback comment when LLM is skipped."""
    classification = r["classification"]
    detail = r.get("scoring_detail", {})
    org = r["org_name"]

    if classification == "promoter":
        return (
            f"Really happy with {COMPANY_NAME}. The team is responsive and "
            f"the platform has been rock solid for us."
        )
    if classification == "passive":
        return (
            "Generally good but there have been a couple of hiccups. "
            "Would be a 10 if the reliability was more consistent."
        )

    if detail.get("escalated_tickets", 0):
        return (
            "We had a support ticket escalate into a full incident and it "
            "took longer than expected to resolve. Impacted our team significantly."
        )
    return (
        "Some reliability issues during our contract period that affected our "
        "operations. Hoping to see improvement before renewal."
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Generate post-simulation artifacts: NPS, invoices, Datadog metrics."
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=BASE,
        help="Root export directory (default: from config.yaml)",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip the two optional LLM calls and use deterministic placeholders.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=["nps", "invoices", "datadog"],
        help="Only regenerate specific artifact types.",
    )
    args = parser.parse_args()

    run(
        export_dir=args.export_dir,
        use_llm=not args.no_llm,
        only=set(args.only) if args.only else None,
    )
