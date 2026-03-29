"""
eval_harness.py
===============
Post-simulation eval dataset generator for OrgForge.

Run after flow.py completes:
    python eval_harness.py

Produces two files in export/eval/:
    causal_threads.json   — explicit artifact graphs with actor knowledge states
    eval_questions.json   — typed Q&A pairs with deterministic ground-truth answers

Design principle
----------------
Answers are derived deterministically from the SimEvent log — the LLM only
writes question prose. This means:
  - Ground truth is always correct by construction (no hallucination risk)
  - Questions can be regenerated with different wording without changing answers
  - The eval set is reproducible across sim runs with the same config

Question types
--------------
  RETRIEVAL     "Which artifact first mentioned X?"
                Answer: specific artifact ID from SimEvent log

  CAUSAL        "What action did Y take after Z happened?"
                Answer: next SimEvent in the causal chain

  TEMPORAL      "Did person P know about X when they made decision D?"
                Answer: boolean + evidence, derived from actor knowledge snapshots
                Note: samples both involves_gap=True incidents (pool A) and all
                other incidents (pool B) to cover true-positive and true-negative cases.

  GAP_DETECTION "Was the email from X ever actioned?"
                Answer: boolean derived from email_dropped / customer_email_routed events

  ROUTING       "Who was the first internal person to see the complaint from X?"
                Answer: liaison name from inbound_external_email SimEvent

  PLAN          "What was dept X focused on during Day N?"
                Answer: theme + actors from dept_plans collection

  ESCALATION    "Who was involved in the escalation chain for ticket X?"
                Answer: escalation_actors from escalation_chain SimEvent

  KNOWLEDGE_GAP "What domain was undocumented when incident X fired?"
                Answer: gap_areas from knowledge_gap_detected SimEvent

  POSTMORTEM    "Which Confluence doc captured the postmortem for incident X?"
                Answer: confluence artifact ID from postmortem_created SimEvent

  STANDUP       "What did person X report at standup on Day N?"
                Answer: summary + slack_thread_id from standup SimEvent

  CUSTOMER_ESC  "Who handled the escalation from customer X?"
                Answer: first_handler + downstream artifacts from customer_escalation
                or customer_email_routed SimEvent

  ZD_RESOLUTION "Was Zendesk ticket X resolved and how long did it take?"
                Answer: resolved boolean + duration_days from zd_ticket_opened /
                zd_tickets_resolved SimEvents

  ZD_ESCALATION "Which Zendesk tickets escalated to incident X?"
                Answer: ticket_ids list from zd_tickets_escalated SimEvent
                (uses CAUSAL scorer — incident + ticket chain)

  SF_RISK       "Which Salesforce accounts were flagged at-risk after incident X?"
                Answer: at_risk_accounts list from sf_deals_risk_flagged SimEvent

  SF_TOUCHPOINT "What opportunity was advanced by the email from X to Y?"
                Answer: opportunity_id + stage from crm_touchpoint SimEvent
                (uses CAUSAL scorer — email → opportunity chain)

  SF_OWNERSHIP  "Which accounts lost their owner when X departed?"
                Answer: lapsed_accounts + lapsed_opportunities from
                sf_ownership_lapsed SimEvent (uses RETRIEVAL scorer)

  DATADOG_ALERT "Which Datadog alert fired for incident X?"
                Answer: incident_id inferred from incident_opened SimEvent
                (uses RETRIEVAL scorer — alert → incident link)

  NPS_SCORE     "What NPS score did customer X give and what drove it?"
                Answer: nps_score + classification derived deterministically
                from ZD ticket and incident SimEvents (mirrors NPSWriter formula)

  INVOICE_SLA   "What SLA credit appeared on customer X's invoice?"
                Answer: breach_duration_days + sla_credit_per_org derived from
                incident duration × SLA_CREDIT_RATE (mirrors InvoiceWriter logic)

Usage for RAG eval
------------------
Each question in eval_questions.json has:
  - question_text      — natural language question
  - question_type      — one of the five types above
  - ground_truth       — the answer an agent should produce
  - evidence_chain     — list of SimEvent IDs / artifact IDs that support the answer
  - difficulty         — "easy" | "medium" | "hard"
  - requires_reasoning — whether answering requires multi-hop traversal

An eval harness compares agent answers against ground_truth.
evidence_chain lets you score partial credit (did the agent find the right artifacts
even if it drew the wrong conclusion?).
"""

from __future__ import annotations

import json
import logging
import random
import yaml
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

from agent_factory import make_agent
from crewai import Crew, Task
from memory import Memory, SimEvent

logger = logging.getLogger("orgforge.eval")


with open(Path(__file__).resolve().parent.parent / "config" / "config.yaml") as f:
    _CFG = yaml.safe_load(f)

BASE = Path(_CFG["simulation"].get("output_dir", "./export"))
EVAL_DIR = BASE / "eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)


class CausalThreadBuilder:
    """
    Reconstructs explicit artifact graphs from the SimEvent log.
    Each thread has a root artifact and a directed list of nodes,
    each annotated with which actors knew about it at that timestamp.
    """

    def __init__(self, mem: Memory):
        self._mem = mem
        self._events: List[SimEvent] = mem.get_event_log(from_db=True)

    def build_all(self) -> List[dict]:
        threads = []
        threads.extend(self._incident_threads())
        threads.extend(self._customer_email_threads())
        threads.extend(self._hr_threads())
        threads.extend(self._dropped_email_threads())
        threads.extend(self._postmortem_threads())
        threads.extend(self._design_doc_threads())
        threads.extend(self._zendesk_threads())
        threads.extend(self._salesforce_threads())
        return threads

    def _incident_threads(self) -> List[dict]:
        threads = []
        opened = [e for e in self._events if e.type == "incident_opened"]
        for event in opened:
            ticket_id = event.artifact_ids.get("jira")
            if not ticket_id:
                continue
            chain = event.facts.get("causal_chain", [ticket_id])
            nodes = self._build_nodes(chain, event)
            threads.append(
                {
                    "chain_id": f"incident_{ticket_id}",
                    "chain_type": "incident",
                    "root_artifact": ticket_id,
                    "root_event_type": "incident_opened",
                    "day": event.day,
                    "date": event.date,
                    "nodes": nodes,
                    "terminal_artifact": chain[-1] if chain else ticket_id,
                    "complete": any(
                        e.type == "incident_resolved"
                        and e.artifact_ids.get("jira") == ticket_id
                        for e in self._events
                    ),
                    "involves_knowledge_gap": event.facts.get("involves_gap", False),
                    "recurrence_of": event.facts.get("recurrence_of"),
                }
            )
        return threads

    def _customer_email_threads(self) -> List[dict]:
        threads = []
        routed = [e for e in self._events if e.type == "customer_email_routed"]
        for event in routed:
            email_id = event.artifact_ids.get("email")
            if not email_id:
                continue
            chain = event.facts.get("causal_chain", [email_id])
            nodes = self._build_nodes(chain, event)
            threads.append(
                {
                    "chain_id": f"customer_email_{email_id}",
                    "chain_type": "customer_email",
                    "root_artifact": email_id,
                    "root_event_type": "inbound_external_email",
                    "day": event.day,
                    "date": event.date,
                    "nodes": nodes,
                    "terminal_artifact": chain[-1] if chain else email_id,
                    "complete": len(chain) > 1,  # False if only root = no action
                    "high_priority": event.facts.get("high_priority", False),
                    "source": event.facts.get("source"),
                    "jira_opened": any("IT-" in n or "ORG-" in n for n in chain[1:]),
                }
            )
        return threads

    def _dropped_email_threads(self) -> List[dict]:
        """Dropped emails are their own thread type — single-node chains."""
        threads = []
        dropped = [e for e in self._events if e.type == "email_dropped"]
        for event in dropped:
            email_id = event.artifact_ids.get("email")
            if not email_id:
                continue
            threads.append(
                {
                    "chain_id": f"dropped_email_{email_id}",
                    "chain_type": "dropped_email",
                    "root_artifact": email_id,
                    "root_event_type": "inbound_external_email",
                    "day": event.day,
                    "date": event.date,
                    "nodes": [
                        {
                            "artifact_id": email_id,
                            "event_type": "inbound_external_email",
                            "timestamp": event.timestamp,
                            "known_to": event.actors,
                            "caused": [],  # intentionally empty
                        }
                    ],
                    "terminal_artifact": email_id,
                    "complete": False,  # never actioned — this is the eval signal
                    "source": event.facts.get("source"),
                    "subject": event.facts.get("subject"),
                }
            )
        return threads

    def _hr_threads(self) -> List[dict]:
        threads = []
        hr_emails = [e for e in self._events if e.type == "hr_outbound_email"]
        hired = [e for e in self._events if e.type == "employee_hired"]

        for email_event in hr_emails:
            prospect = email_event.facts.get("prospect")
            embed_id = email_event.artifact_ids.get("embed_id")
            hire_day = email_event.facts.get("hire_day")

            # Find matching employee_hired event
            hire_event = next(
                (e for e in hired if prospect in e.actors and e.day == hire_day),
                None,
            )

            chain = [embed_id]
            if hire_event:
                hire_artifact = hire_event.artifact_ids.get(
                    "jira", hire_event.artifact_ids.get("slack_thread", "")
                )
                if hire_artifact:
                    chain.append(hire_artifact)

            threads.append(
                {
                    "chain_id": f"hr_{prospect}",
                    "chain_type": "hr_hire",
                    "root_artifact": embed_id,
                    "root_event_type": "hr_outbound_email",
                    "day": email_event.day,
                    "date": email_event.date,
                    "nodes": self._build_nodes(chain, email_event),
                    "terminal_artifact": chain[-1],
                    "complete": hire_event is not None,
                    "prospect": prospect,
                    "hire_day": hire_day,
                }
            )
        return threads

    def _postmortem_threads(self) -> List[dict]:
        """
        Builds incident → postmortem causal chains.
        Each thread pairs the incident_opened event with its postmortem_created
        event, enabling CAUSAL questions like 'What postmortem followed ORG-130?'
        and TEMPORAL questions about whether the postmortem existed before a
        recurrence.
        """
        threads = []
        postmortems = [e for e in self._events if e.type == "postmortem_created"]
        for event in postmortems:
            ticket_id = event.artifact_ids.get("jira")
            conf_id = event.artifact_ids.get("confluence")
            if not ticket_id or not conf_id:
                continue

            chain = event.facts.get("causal_chain", [ticket_id, conf_id])
            nodes = self._build_nodes(chain, event)
            threads.append(
                {
                    "chain_id": f"postmortem_{ticket_id}",
                    "chain_type": "postmortem",
                    "root_artifact": ticket_id,
                    "root_event_type": "incident_opened",
                    "terminal_artifact": conf_id,
                    "day": event.day,
                    "date": event.date,
                    "nodes": nodes,
                    "complete": True,
                    "confluence_id": conf_id,
                    "root_cause": event.facts.get("root_cause", ""),
                }
            )
        return threads

    def _zendesk_threads(self) -> List[dict]:
        """
        Builds Zendesk ticket lifecycle chains:
          zd_ticket_opened → (zd_tickets_escalated) → zd_tickets_resolved

        Each thread records whether the ticket was escalated to an incident,
        enabling CAUSAL questions like 'Which ZD tickets escalated to ORG-42?'
        and ZD_RESOLUTION questions like 'Was ZD-501 resolved, and how quickly?'
        """
        threads = []
        opened_events: Dict[str, SimEvent] = {}
        for e in self._events:
            if e.type == "zd_ticket_opened":
                tid = e.facts.get("ticket_id", "")
                if tid:
                    opened_events[tid] = e

        escalated_tickets: Dict[str, str] = {}
        resolved_days: Dict[str, int] = {}
        for e in self._events:
            if e.type == "zd_tickets_escalated":
                iid = e.facts.get("incident_id", "")
                for tid in e.facts.get("ticket_ids", []):
                    escalated_tickets[tid] = iid
            elif e.type == "zd_tickets_resolved":
                for tid in e.facts.get("ticket_ids", []):
                    resolved_days[tid] = e.day

        for tid, open_event in opened_events.items():
            incident_id = escalated_tickets.get(tid)
            resolve_day = resolved_days.get(tid)
            org = open_event.facts.get("org_name", "Unknown")

            chain = [tid]
            if incident_id:
                chain.append(incident_id)

            nodes = self._build_nodes(chain, open_event)

            threads.append(
                {
                    "chain_id": f"zd_{tid}",
                    "chain_type": "zendesk_ticket",
                    "root_artifact": tid,
                    "root_event_type": "zd_ticket_opened",
                    "day": open_event.day,
                    "date": open_event.date,
                    "nodes": nodes,
                    "terminal_artifact": chain[-1],
                    "complete": resolve_day is not None,
                    "escalated": bool(incident_id),
                    "incident_id": incident_id,
                    "org_name": org,
                    "subject": open_event.facts.get("subject", ""),
                    "open_day": open_event.day,
                    "resolve_day": resolve_day,
                    "duration_days": (resolve_day - open_event.day)
                    if resolve_day
                    else None,
                }
            )
        return threads

    def _salesforce_threads(self) -> List[dict]:
        """
        Builds Salesforce causal chains across three event types:
          crm_touchpoint       — a sales email created or advanced an opportunity
          sf_deals_risk_flagged — an incident caused accounts to be flagged at-risk
          sf_ownership_lapsed  — an employee departure orphaned accounts/opps

        Each thread type answers a different class of eval question:
          crm_touchpoint       → CAUSAL: 'What opportunity was created from this email?'
          sf_deals_risk_flagged → SF_RISK: 'Which accounts were at risk after incident X?'
          sf_ownership_lapsed  → RETRIEVAL: 'Which accounts lost their owner when X left?'
        """
        threads = []

        for e in self._events:
            if e.type != "crm_touchpoint":
                continue
            opp_id = e.artifact_ids.get("sf_opp", "")
            if not opp_id:
                continue
            threads.append(
                {
                    "chain_id": f"sf_touchpoint_{opp_id}",
                    "chain_type": "sf_touchpoint",
                    "root_artifact": opp_id,
                    "root_event_type": "crm_touchpoint",
                    "day": e.day,
                    "date": e.date,
                    "nodes": self._build_nodes([opp_id], e),
                    "terminal_artifact": opp_id,
                    "complete": True,
                    "account_name": e.facts.get("account_name", ""),
                    "stage": e.facts.get("stage", ""),
                    "sender": e.facts.get("sender", ""),
                    "subject": e.facts.get("subject", ""),
                }
            )

        for e in self._events:
            if e.type != "sf_deals_risk_flagged":
                continue
            incident_id = e.facts.get("incident_id", "")
            account_names = e.facts.get("account_names", [])
            if not incident_id or not account_names:
                continue
            threads.append(
                {
                    "chain_id": f"sf_risk_{incident_id}",
                    "chain_type": "sf_risk",
                    "root_artifact": incident_id,
                    "root_event_type": "sf_deals_risk_flagged",
                    "day": e.day,
                    "date": e.date,
                    "nodes": self._build_nodes([incident_id], e),
                    "terminal_artifact": incident_id,
                    "complete": True,
                    "incident_id": incident_id,
                    "at_risk_accounts": account_names,
                    "account_count": len(account_names),
                }
            )

        for e in self._events:
            if e.type != "sf_ownership_lapsed":
                continue
            departed = e.facts.get("departed_employee", "")
            accs = e.facts.get("accounts_lapsed", [])
            opps = e.facts.get("opportunities_lapsed", [])
            if not departed:
                continue
            # Chain root is the departed-employee event anchor; artifacts are
            # the lapsed account/opp IDs stored as lists in artifact_ids.
            chain = accs[:1] or opps[:1] or [f"lapsed_{departed}"]
            threads.append(
                {
                    "chain_id": f"sf_lapse_{departed.lower().replace(' ', '_')}",
                    "chain_type": "sf_ownership_lapse",
                    "root_artifact": chain[0],
                    "root_event_type": "sf_ownership_lapsed",
                    "day": e.day,
                    "date": e.date,
                    "nodes": self._build_nodes(chain, e),
                    "terminal_artifact": chain[-1],
                    "complete": True,
                    "departed_employee": departed,
                    "role": e.facts.get("role", ""),
                    "lapsed_accounts": accs,
                    "lapsed_opportunities": opps,
                }
            )

        return threads

    def _design_doc_threads(self) -> List[dict]:
        threads = []
        conf_events = [
            e
            for e in self._events
            if e.type == "confluence_created"
            and e.facts.get("type") == "design_doc"
            and e.facts.get("causal_chain")  # only after your fix
        ]
        for event in conf_events:
            conf_id = event.artifact_ids.get("confluence")
            if not conf_id:
                continue
            chain = event.facts.get("causal_chain", [conf_id])
            nodes = self._build_nodes(chain, event)
            threads.append(
                {
                    "chain_id": f"design_doc_{conf_id}",
                    "chain_type": "design_doc",
                    "root_artifact": conf_id,
                    "root_event_type": "confluence_created",
                    "day": event.day,
                    "date": event.date,
                    "nodes": nodes,
                    "terminal_artifact": chain[-1] if chain else conf_id,
                    "complete": len(chain) > 1,
                    "author": next(iter(event.actors), None),
                    "spawned_tickets": event.facts.get("spawned_tickets", []),
                }
            )
        return threads

    def _build_nodes(self, chain: List[str], root_event: SimEvent) -> List[dict]:
        """
        Build annotated node list from a causal chain.
        Each node records: artifact_id, event_type, timestamp, known_to, caused.
        known_to is derived from which actors appear in SimEvents up to that point.
        """
        nodes = []
        cumulative_known: Set[str] = set()

        for i, artifact_id in enumerate(chain):
            event = self._find_event_for_artifact(artifact_id)
            if event:
                cumulative_known.update(event.actors)
                caused = [chain[i + 1]] if i + 1 < len(chain) else []
                nodes.append(
                    {
                        "artifact_id": artifact_id,
                        "event_type": event.type,
                        "timestamp": event.timestamp,
                        "day": event.day,
                        "known_to": list(cumulative_known),
                        "caused": caused,
                    }
                )
            else:
                nodes.append(
                    {
                        "artifact_id": artifact_id,
                        "event_type": "unknown",
                        "timestamp": root_event.timestamp,
                        "day": root_event.day,
                        "known_to": list(root_event.actors),
                        "caused": [chain[i + 1]] if i + 1 < len(chain) else [],
                    }
                )
        return nodes

    def _find_event_for_artifact(self, artifact_id: str) -> Optional[SimEvent]:
        """Find the SimEvent most directly associated with an artifact ID."""
        for event in self._events:
            if artifact_id in event.artifact_ids.values():
                return event
            if artifact_id in event.facts.get("causal_chain", []):
                continue
        return None


class EvalQuestionGenerator:
    """
    Generates typed eval questions from causal threads and SimEvents.

    For each question:
      1. Extract answer deterministically from SimEvent log (no LLM)
      2. LLM wraps the answer in natural-sounding question prose
      3. Store question_text + ground_truth + evidence_chain together

    The LLM never touches the ground_truth field — only the question wording.
    """

    def __init__(self, mem: Memory, worker_llm):
        self._mem = mem
        self._worker_llm = worker_llm
        self._events: List[SimEvent] = mem.get_event_log(from_db=True)

    def generate(self, threads: List[dict]) -> List[dict]:
        questions = []
        questions.extend(self._retrieval_questions(threads))
        questions.extend(self._causal_questions(threads))
        questions.extend(self._temporal_questions())
        questions.extend(self._gap_detection_questions(threads))
        questions.extend(self._routing_questions(threads))
        questions.extend(self._escalation_questions())
        questions.extend(self._knowledge_gap_questions())
        questions.extend(self._postmortem_questions(threads))
        questions.extend(self._confluence_questions())
        questions.extend(self._standup_questions())
        questions.extend(self._customer_escalation_questions())
        questions.extend(self._zendesk_resolution_questions(threads))
        questions.extend(self._zd_escalation_questions(threads))
        questions.extend(self._sf_risk_questions(threads))
        questions.extend(self._sf_ownership_lapse_questions(threads))
        questions.extend(self._sf_touchpoint_questions(threads))
        questions.extend(self._datadog_alert_questions())
        questions.extend(self._nps_score_questions())
        questions.extend(self._invoice_sla_questions())
        questions.extend(self._pr_review_questions(threads))
        questions.extend(self._blocker_questions())
        questions.extend(self._vendor_routing_questions())
        questions.extend(self._design_discussion_questions())
        return questions

    def _retrieval_questions(self, threads: List[dict]) -> List[dict]:
        questions = []
        incident_threads = [t for t in threads if t["chain_type"] == "incident"]
        logger.info(
            f"[eval] _retrieval_questions: {len(incident_threads)} incident threads available"
        )
        for thread in random.sample(incident_threads, min(50, len(incident_threads))):
            root_event = self._find_event_by_artifact(thread["root_artifact"])
            if not root_event:
                continue
            root_cause = root_event.facts.get("root_cause", "")
            if not root_cause:
                continue

            ground_truth = {
                "artifact_id": thread["root_artifact"],
                "artifact_type": "jira",
                "timestamp": root_event.timestamp,
                "day": root_event.day,
            }
            evidence = [thread["root_artifact"]]

            q_text = self._generate_question_prose(
                template=(
                    f"Generate a retrieval question asking which artifact or ticket "
                    f'first documented this incident: "{root_cause[:80]}". '
                    f"The question should sound like a natural analyst query. "
                    f"Output only the question text."
                )
            )
            if q_text:
                questions.append(
                    {
                        "question_id": f"retrieval_{thread['root_artifact']}",
                        "question_type": "RETRIEVAL",
                        "question_text": q_text,
                        "ground_truth": ground_truth,
                        "evidence_chain": evidence,
                        "difficulty": "easy",
                        "requires_reasoning": False,
                        "chain_id": thread["chain_id"],
                    }
                )
        return questions

    def _causal_questions(self, threads: List[dict]) -> List[dict]:
        questions = []
        multi_hop = [
            t
            for t in threads
            if len(t.get("nodes", [])) >= 3
            and t["chain_type"] in ("incident", "customer_email", "design_doc")
        ]
        logger.info(
            f"[eval] _causal_questions: {len(multi_hop)} multi-hop threads available"
        )
        for thread in random.sample(multi_hop, min(50, len(multi_hop))):
            nodes = thread["nodes"]

            if len(nodes) < 2:
                continue
            trigger_node = nodes[0]
            result_node = nodes[1]

            ground_truth = {
                "artifact_id": result_node["artifact_id"],
                "event_type": result_node["event_type"],
                "actors": result_node["known_to"],
                "timestamp": result_node["timestamp"],
            }
            evidence = [trigger_node["artifact_id"], result_node["artifact_id"]]

            trigger_event = self._find_event_by_artifact(trigger_node["artifact_id"])
            trigger_desc = (
                trigger_event.summary if trigger_event else trigger_node["artifact_id"]
            )

            q_text = self._generate_question_prose(
                template=(
                    f"Generate a causal question asking what action or artifact "
                    f'was produced as a direct result of: "{trigger_desc[:100]}". '
                    f"The question should probe cause-and-effect reasoning. "
                    f"Output only the question text."
                )
            )
            if q_text:
                questions.append(
                    {
                        "question_id": f"causal_{trigger_node['artifact_id']}",
                        "question_type": "CAUSAL",
                        "question_text": q_text,
                        "ground_truth": ground_truth,
                        "evidence_chain": evidence,
                        "difficulty": "medium",
                        "requires_reasoning": True,
                        "chain_id": thread["chain_id"],
                    }
                )
        return questions

    def _temporal_questions(self) -> List[dict]:
        """
        Uses actor knowledge snapshots from day_summary SimEvents.
        Asks whether a person had relevant information before a key decision.

        Two candidate pools are combined so the eval covers both true-positive
        and true-negative temporal reasoning:

          Pool A — incidents where involves_gap=True (knowledge was definitely
                   lost before the incident).  These always yield had_knowledge=False
                   questions, exercising the "departure caused a gap" path.

          Pool B — all other incident_opened events.  For each, we check whether
                   ANY departure event preceded the incident day, regardless of
                   whether gap keywords matched.  This yields a mix of
                   had_knowledge=True and had_knowledge=False questions, preventing
                   the eval from being gamed by always answering "no knowledge".

        The two pools are sampled independently then merged, so a run with no
        involves_gap incidents still produces temporal questions from Pool B.
        """
        questions = []
        departures = [e for e in self._events if e.type == "employee_departed"]

        # ── Pool A: explicit gap incidents (involves_gap=True) ─────────────────
        pool_a = [
            e
            for e in self._events
            if e.type == "incident_opened" and e.facts.get("involves_gap")
        ]

        # ── Pool B: all other incidents — derive gap from departure overlap ────
        pool_a_ids = {e.artifact_ids.get("jira") for e in pool_a}
        pool_b = [
            e
            for e in self._events
            if e.type == "incident_opened"
            and not e.facts.get("involves_gap")
            and e.artifact_ids.get("jira") not in pool_a_ids
        ]

        # Track (person, domain) pairs already emitted to prevent the eval
        # collapsing into N copies of the same question when one departure
        # dominates the event log (e.g. only Jordan left, owns auth-service).
        seen_person_domain: Set[tuple] = set()

        def _build_temporal_question(event: SimEvent) -> Optional[dict]:
            ticket_id = event.artifact_ids.get("jira")
            assignee = next(iter(event.actors), None)
            if not ticket_id or not assignee:
                return None

            # ── Resolve gap_domain ────────────────────────────────────────────
            # Priority order:
            #   1. Explicit gap_areas on the event (Pool A / flagged incidents)
            #   2. root_cause keywords from the incident itself — this produces
            #      had_knowledge=True questions when no matching departure exists,
            #      which is the primary source of positive-class TEMPORAL examples.
            #   3. Most-recent departure's knowledge_domains as a last resort.
            gap_areas = event.facts.get("gap_areas", [])
            root_cause = event.facts.get("root_cause", "")

            if gap_areas:
                gap_domain = gap_areas[0]
            elif root_cause:
                _stop = {"the", "a", "an", "in", "of", "on", "was", "is", "due", "to"}
                tokens = [
                    t
                    for t in root_cause.split()
                    if t.lower() not in _stop and len(t) > 4
                ]
                gap_domain = tokens[0] if tokens else "system"
            else:
                prior_dep = next(
                    (
                        d
                        for d in sorted(departures, key=lambda d: d.day, reverse=True)
                        if d.day < event.day
                    ),
                    None,
                )
                if prior_dep:
                    domains = prior_dep.facts.get("knowledge_domains", [])
                    gap_domain = domains[0] if domains else "undocumented system"
                else:
                    gap_domain = "system"

            if (assignee, gap_domain) in seen_person_domain:
                return None
            seen_person_domain.add((assignee, gap_domain))

            # ── Deterministic had_knowledge ───────────────────────────────────
            # True  = no departure of a domain expert preceded this incident
            #         → the team DID have access to documentation
            # False = a departure covering this domain preceded the incident
            #         → the knowledge was gone before the incident fired
            departure = next(
                (
                    d
                    for d in self._events
                    if d.type == "employee_departed"
                    and gap_domain in str(d.facts.get("knowledge_domains", []))
                    and d.day < event.day
                ),
                None,
            )

            had_knowledge = departure is None
            ground_truth = {
                "had_knowledge": had_knowledge,
                "person": assignee,
                "domain": gap_domain,
                "incident_day": event.day,
                "departure_day": departure.day if departure else None,
                "reasoning": (
                    f"No departure of the {gap_domain} knowledge owner had occurred before Day {event.day}."
                    if had_knowledge
                    else (
                        f"{departure.actors[0]} (who owned {gap_domain}) left on "
                        f"Day {departure.day}, {event.day - departure.day} days before "
                        f"this incident."
                    )
                ),
            }
            evidence = [ticket_id]
            if departure:
                evidence.append(f"EVENT-{departure.day}-employee_departed")

            q_text = self._generate_question_prose(
                template=(
                    f"Generate a temporal knowledge question asking whether "
                    f"{assignee} had access to documentation about {gap_domain} "
                    f"when incident {ticket_id} was opened on Day {event.day}. "
                    f"The question should test whether an agent can reason about "
                    f"what information existed at a specific point in time. "
                    f"Output only the question text."
                )
            )
            if not q_text:
                return None

            return {
                "question_id": f"temporal_{ticket_id}_{gap_domain}",
                "question_type": "TEMPORAL",
                "question_text": q_text,
                "ground_truth": ground_truth,
                "evidence_chain": [e for e in evidence if e],
                "difficulty": "hard",
                "requires_reasoning": True,
                "chain_id": f"incident_{ticket_id}",
            }

        for event in random.sample(pool_a, min(50, len(pool_a))):
            q = _build_temporal_question(event)
            if q:
                questions.append(q)

        for event in random.sample(pool_b, min(50, len(pool_b))):
            q = _build_temporal_question(event)
            if q:
                questions.append(q)

        hk_dist = {True: 0, False: 0}
        for q in questions:
            hk_dist[q["ground_truth"]["had_knowledge"]] += 1
        logger.info(
            f"[eval] _temporal_questions: pool_a={len(pool_a)}, pool_b={len(pool_b)}, "
            f"generated={len(questions)}, had_knowledge={hk_dist}"
        )
        return questions

    def _gap_detection_questions(self, threads: List[dict]) -> List[dict]:
        questions = []
        dropped = [t for t in threads if t["chain_type"] == "dropped_email"]
        routed = [
            t
            for t in threads
            if t["chain_type"] == "customer_email" and t.get("complete")
        ]
        logger.info(
            f"[eval] _gap_detection_questions: {len(dropped)} dropped, "
            f"{len(routed)} routed-complete threads available"
        )

        for thread in random.sample(dropped, min(50, len(dropped))):
            subject = thread.get("subject", thread["root_artifact"])
            source = thread.get("source", "unknown sender")
            ground_truth = {
                "was_actioned": False,
                "artifact_id": thread["root_artifact"],
                "source": source,
                "downstream_artifacts": [],
                "reason": "Email received but no Slack message or JIRA was created.",
            }
            q_text = self._generate_question_prose(
                template=(
                    f"Generate a gap-detection question asking whether any action "
                    f"was taken after {source} sent an email with subject "
                    f'"{subject[:60]}". The question should require the agent to '
                    f"search for downstream artifacts and notice their absence. "
                    f"Output only the question text."
                )
            )
            if q_text:
                questions.append(
                    {
                        "question_id": f"gap_{thread['root_artifact']}",
                        "question_type": "GAP_DETECTION",
                        "question_text": q_text,
                        "ground_truth": ground_truth,
                        "evidence_chain": [thread["root_artifact"]],
                        "difficulty": "hard",
                        "requires_reasoning": True,
                        "chain_id": thread["chain_id"],
                    }
                )

        for thread in random.sample(routed, min(20, len(routed))):
            source = thread.get("source", "unknown")
            chain = [n["artifact_id"] for n in thread["nodes"]]
            ground_truth = {
                "was_actioned": True,
                "artifact_id": thread["root_artifact"],
                "source": source,
                "downstream_artifacts": chain[1:],
                "jira_opened": thread.get("jira_opened", False),
            }
            q_text = self._generate_question_prose(
                template=(
                    f"Generate a gap-detection question asking whether any action "
                    f"was taken after {source} sent a complaint email. "
                    f"Output only the question text."
                )
            )
            if q_text:
                questions.append(
                    {
                        "question_id": f"gap_{thread['root_artifact']}_actioned",
                        "question_type": "GAP_DETECTION",
                        "question_text": q_text,
                        "ground_truth": ground_truth,
                        "evidence_chain": chain,
                        "difficulty": "medium",
                        "requires_reasoning": True,
                        "chain_id": thread["chain_id"],
                    }
                )
        return questions

    def _routing_questions(self, threads: List[dict]) -> List[dict]:
        questions = []
        customer_threads = [
            t for t in threads if t["chain_type"] in ("customer_email", "dropped_email")
        ]
        logger.info(
            f"[eval] _routing_questions: {len(customer_threads)} customer threads available"
        )
        for thread in random.sample(customer_threads, min(25, len(customer_threads))):
            root_node = thread["nodes"][0] if thread["nodes"] else None
            if not root_node:
                continue

            # Build an exclusion set: the external source (may be stored as a
            # company name like "RunnersPeak Corp", not an email address) plus
            # anything that looks like an email address.  Only proper internal
            # actor names (single/dual words, no @ or Corp/LLC/Inc suffix) pass.
            external_source = thread.get("source", "")
            _external_markers = {"corp", "llc", "inc", "ltd", "gmbh", "co."}

            def _is_internal(actor: str) -> bool:
                if "@" in actor:
                    return False
                if actor == external_source:
                    return False
                # Catch "RunnersPeak Corp", "MarathonTech LLC", etc.
                if any(part.lower() in _external_markers for part in actor.split()):
                    return False
                return True

            first_internal = next(
                (a for a in root_node.get("known_to", []) if _is_internal(a)),
                None,
            )
            if not first_internal:
                logger.warning(
                    f"[eval] _routing_questions: no internal actor found in "
                    f"known_to={root_node.get('known_to')} for thread "
                    f"{thread['chain_id']} — skipping"
                )
                continue

            ground_truth = {
                "first_recipient": first_internal,
                "artifact_id": thread["root_artifact"],
                "timestamp": root_node["timestamp"],
                "was_escalated": len(thread["nodes"]) > 1,
            }
            source = thread.get("source", "a customer")
            q_text = self._generate_question_prose(
                template=(
                    f"Generate a routing question asking which internal employee "
                    f"first received the email from {source}. The question should "
                    f"test whether an agent can trace the initial delivery path. "
                    f"Output only the question text."
                )
            )
            if q_text:
                questions.append(
                    {
                        "question_id": f"routing_{thread['root_artifact']}",
                        "question_type": "ROUTING",
                        "question_text": q_text,
                        "ground_truth": ground_truth,
                        "evidence_chain": [thread["root_artifact"]],
                        "difficulty": "easy",
                        "requires_reasoning": False,
                        "chain_id": thread["chain_id"],
                    }
                )
        return questions

    # ── ESCALATION — "Who was pulled into the escalation for ticket X?" ───────────
    def _escalation_questions(self) -> List[dict]:
        questions = []
        esc_events = [e for e in self._events if e.type == "escalation_chain"]
        logger.info(
            f"[eval] _escalation_questions: {len(esc_events)} escalation_chain events available"
        )

        for event in random.sample(esc_events, min(25, len(esc_events))):
            ticket_id = event.artifact_ids.get("jira", "")
            actors = event.facts.get("escalation_actors", [])
            narrative = event.facts.get("escalation_narrative", "")
            if not ticket_id or not actors:
                continue

            ground_truth = {
                "ticket_id": ticket_id,
                "escalation_actors": actors,
                "hops": len(actors) - 1,
                "narrative": narrative,
            }

            q_text = self._generate_question_prose(
                template=(
                    f"Generate a question asking who was involved in the escalation "
                    f"chain for incident {ticket_id}. The question should test "
                    f"whether an agent can trace the escalation path. "
                    f"Output only the question text."
                )
            )
            if q_text:
                questions.append(
                    {
                        "question_id": f"escalation_{ticket_id}",
                        "question_type": "ESCALATION",
                        "question_text": q_text,
                        "ground_truth": ground_truth,
                        "evidence_chain": [ticket_id],
                        "difficulty": "easy",
                        "requires_reasoning": False,
                        "chain_id": f"incident_{ticket_id}",
                    }
                )
        return questions

    # ── KNOWLEDGE GAP — "What domain was undocumented when X fired?" ──────────────
    def _knowledge_gap_questions(self) -> List[dict]:
        questions = []
        gap_events = [
            e
            for e in self._events
            if e.type == "knowledge_gap_detected"
            and e.facts.get("trigger") != "centrality_vacuum"
        ]
        logger.info(
            f"[eval] _knowledge_gap_questions: {len(gap_events)} knowledge_gap_detected events available"
        )

        for event in random.sample(gap_events, min(25, len(gap_events))):
            ticket_id = event.artifact_ids.get("jira", "")
            gap_areas = event.facts.get("gap_areas", [])
            actors = event.actors
            if not ticket_id or not gap_areas:
                continue

            ground_truth = {
                "ticket_id": ticket_id,
                "gap_areas": gap_areas,
                "detected_by": actors,
                "artifact_id": ticket_id,
            }

            q_text = self._generate_question_prose(
                template=(
                    f"Generate a question asking which knowledge domain or system "
                    f"was found to be undocumented during incident {ticket_id}. "
                    f"The question should test whether an agent can identify "
                    f"documentation gaps from incident records. "
                    f"Output only the question text."
                )
            )
            if q_text:
                questions.append(
                    {
                        "question_id": f"gap_{ticket_id}",
                        "question_type": "KNOWLEDGE_GAP",
                        "question_text": q_text,
                        "ground_truth": ground_truth,
                        "evidence_chain": [ticket_id],
                        "difficulty": "easy",
                        "requires_reasoning": False,
                        "chain_id": f"incident_{ticket_id}",
                    }
                )
        logger.info(
            f"[eval] _knowledge_gap_questions: {len(gap_events)} gaps available, "
            f"generated={len(questions)}"
        )
        return questions

    # ── POSTMORTEM — "What postmortem documented incident X?" ─────────────────

    def _postmortem_questions(self, threads: List[dict]) -> List[dict]:
        """
        CAUSAL questions that traverse the incident → postmortem chain.
        Tests whether an agent can identify that a postmortem was written and
        find the confluence artifact that contains it.
        """
        questions = []
        pm_threads = [t for t in threads if t["chain_type"] == "postmortem"]
        logger.info(
            f"[eval] _postmortem_questions: {len(pm_threads)} postmortem threads available"
        )

        for thread in random.sample(pm_threads, min(25, len(pm_threads))):
            ticket_id = thread["root_artifact"]
            conf_id = thread["confluence_id"]
            root_cause = thread.get("root_cause", "")
            nodes = thread["nodes"]
            if len(nodes) < 2:
                continue

            ground_truth = {
                "incident_id": ticket_id,
                "postmortem_confluence_id": conf_id,
                "postmortem_day": thread["day"],
                "root_cause": root_cause,
            }
            evidence = [ticket_id, conf_id]

            q_text = self._generate_question_prose(
                template=(
                    f"Generate a causal question asking which Confluence document "
                    f"captured the postmortem for incident {ticket_id}. The question "
                    f"should test whether an agent can trace from a Jira incident to "
                    f"its postmortem artifact. "
                    f"Output only the question text."
                )
            )
            if q_text:
                questions.append(
                    {
                        "question_id": f"postmortem_{ticket_id}",
                        "question_type": "CAUSAL",
                        "question_text": q_text,
                        "ground_truth": ground_truth,
                        "evidence_chain": evidence,
                        "difficulty": "medium",
                        "requires_reasoning": True,
                        "chain_id": f"postmortem_{ticket_id}",
                    }
                )
        return questions

    # ── CONFLUENCE — "Which doc first covered topic X?" ───────────────────────

    def _confluence_questions(self) -> List[dict]:
        """
        RETRIEVAL questions over the confluence_created event log.
        Tests whether an agent can locate the right Confluence page for a topic.
        Queries the events collection directly (not artifacts) so ground truth
        is deterministic from the SimEvent, not from embedding similarity.
        """
        questions = []
        conf_events = [e for e in self._events if e.type == "confluence_created"]
        logger.info(
            f"[eval] _confluence_questions: {len(conf_events)} confluence_created events available"
        )

        for event in random.sample(conf_events, min(50, len(conf_events))):
            conf_id = event.artifact_ids.get("confluence") or event.artifact_ids.get(
                "page_id"
            )
            author = next(iter(event.actors), None)
            topic = (
                event.facts.get("topic")
                or event.facts.get("title")
                or event.facts.get("summary", "")
            )
            if not conf_id or not topic:
                continue

            ground_truth = {
                "confluence_id": conf_id,
                "author": author,
                "day": event.day,
                "topic": topic,
            }

            q_text = self._generate_question_prose(
                template=(
                    f"Generate a retrieval question asking which Confluence page "
                    f'covers the topic: "{topic[:80]}". The question should sound '
                    f"like an engineer searching internal documentation. "
                    f"Output only the question text."
                )
            )
            if q_text:
                questions.append(
                    {
                        "question_id": f"confluence_{conf_id}",
                        "question_type": "RETRIEVAL",
                        "question_text": q_text,
                        "ground_truth": ground_truth,
                        "evidence_chain": [conf_id],
                        "difficulty": "easy",
                        "requires_reasoning": False,
                        "chain_id": f"confluence_{conf_id}",
                    }
                )
        return questions

    # ── STANDUP — "What was X working on during Day N standup?" ──────────────

    def _standup_questions(self) -> List[dict]:
        """
        RETRIEVAL questions over standup SimEvents.
        Tests whether an agent can identify what a specific engineer reported
        at standup on a given day — a lightweight memory retrieval task.
        """
        questions = []
        standup_events = [e for e in self._events if e.type == "standup"]
        logger.info(
            f"[eval] _standup_questions: {len(standup_events)} standup events available"
        )

        for event in random.sample(standup_events, min(25, len(standup_events))):
            if not event.actors:
                continue
            # Pick one actor from the standup to ask about
            actor = random.choice(event.actors)
            slack_thread = event.artifact_ids.get("slack_thread", "")
            day = event.day

            ground_truth = {
                "actor": actor,
                "day": day,
                "artifact_id": slack_thread,
                "all_participants": event.actors,
                "summary": event.summary,
            }
            evidence = [slack_thread] if slack_thread else []

            q_text = self._generate_question_prose(
                template=(
                    f"Generate a retrieval question asking what {actor} reported "
                    f"during the standup on Day {day}. The question should test "
                    f"whether an agent can locate a specific person's standup "
                    f"update from a Slack thread. "
                    f"Output only the question text."
                )
            )
            if q_text:
                questions.append(
                    {
                        "question_id": f"standup_{actor}_day{day}",
                        "question_type": "RETRIEVAL",
                        "question_text": q_text,
                        "ground_truth": ground_truth,
                        "evidence_chain": evidence,
                        "difficulty": "easy",
                        "requires_reasoning": False,
                        "chain_id": f"standup_day{day}",
                    }
                )
        return questions

    # ── CUSTOMER ESCALATION — "How was the escalation from X resolved?" ───────

    def _customer_escalation_questions(self) -> List[dict]:
        """
        CAUSAL questions over customer_escalation and customer_email_routed events.
        Tests whether an agent can trace what action was taken after a customer
        escalation arrived — who handled it, and what artifacts were created.
        Uses customer_email_routed as the primary source since it's confirmed
        present in the event log; customer_escalation events are included when found.
        """
        questions = []
        esc_events = [
            e
            for e in self._events
            if e.type in ("customer_escalation", "customer_email_routed")
            and e.facts.get("source")  # need a named source for the question
        ]
        logger.info(
            f"[eval] _customer_escalation_questions: {len(esc_events)} "
            f"customer escalation/routed events available"
        )

        for event in random.sample(esc_events, min(25, len(esc_events))):
            source = event.facts.get("source", "a customer")
            ticket_id = event.artifact_ids.get("jira") or event.artifact_ids.get(
                "email", ""
            )
            handler = next(iter(event.actors), None)
            if not handler:
                continue

            causal_chain = event.facts.get(
                "causal_chain", [ticket_id] if ticket_id else []
            )
            downstream = [a for a in causal_chain if a != ticket_id]

            ground_truth = {
                "source": source,
                "first_handler": handler,
                "artifact_id": ticket_id,
                "downstream_artifacts": downstream,
                "day": event.day,
                "was_escalated": bool(downstream),
            }
            evidence = [ticket_id] if ticket_id else []

            q_text = self._generate_question_prose(
                template=(
                    f"Generate a causal question asking who handled the escalation "
                    f"from {source} and what action was taken. The question should "
                    f"require tracing from the initial contact through to any "
                    f"tickets or responses created. "
                    f"Output only the question text."
                )
            )
            if q_text:
                questions.append(
                    {
                        "question_id": f"escalation_handling_{event.artifact_ids.get('email') or ticket_id}",
                        "question_type": "CAUSAL",
                        "question_text": q_text,
                        "ground_truth": ground_truth,
                        "evidence_chain": [e for e in evidence if e],
                        "difficulty": "medium",
                        "requires_reasoning": True,
                        "chain_id": f"customer_{source}",
                    }
                )
        return questions

    # ── ZD_RESOLUTION — "Was ZD ticket X resolved and how long did it take?" ────

    def _zendesk_resolution_questions(self, threads: List[dict]) -> List[dict]:
        """
        ZD_RESOLUTION questions over zendesk_ticket threads.
        Tests whether an agent can determine ticket resolution status and SLA
        duration from the Zendesk ticket lifecycle event chain.

        Maps to existing RETRIEVAL scorer — ground truth is a boolean + day count,
        not a narrative, so exact-match is appropriate.
        """
        questions = []
        zd_threads = [t for t in threads if t["chain_type"] == "zendesk_ticket"]
        logger.info(
            f"[eval] _zendesk_resolution_questions: {len(zd_threads)} ZD threads available"
        )

        for thread in random.sample(zd_threads, min(30, len(zd_threads))):
            tid = thread["root_artifact"]
            org = thread.get("org_name", "a customer")
            subject = thread.get("subject", tid)

            ground_truth = {
                "ticket_id": tid,
                "artifact_id": tid,
                "org_name": org,
                "resolved": thread["complete"],
                "duration_days": thread.get("duration_days"),
                "escalated": thread["escalated"],
                "incident_id": thread.get("incident_id"),
            }
            evidence = [tid]
            if thread.get("incident_id"):
                evidence.append(thread["incident_id"])

            q_text = self._generate_question_prose(
                template=(
                    f"Generate a retrieval question asking whether Zendesk ticket "
                    f"{tid} from {org} was resolved, and if so, how many days it "
                    f"took. The question should test whether an agent can trace the "
                    f"full ticket lifecycle. "
                    f"Output only the question text."
                )
            )
            if q_text:
                difficulty = "medium" if thread["escalated"] else "easy"
                questions.append(
                    {
                        "question_id": f"zd_resolution_{tid}",
                        "question_type": "ZD_RESOLUTION",
                        "question_text": q_text,
                        "ground_truth": ground_truth,
                        "evidence_chain": evidence,
                        "difficulty": difficulty,
                        "requires_reasoning": thread["escalated"],
                        "chain_id": thread["chain_id"],
                    }
                )
        return questions

    # ── ZD_ESCALATION — "Which tickets escalated to incident X?" ─────────────

    def _zd_escalation_questions(self, threads: List[dict]) -> List[dict]:
        """
        CAUSAL questions over zendesk_ticket threads where escalated=True.
        Tests cross-system reasoning: given a Jira incident, can the agent
        identify which ZD tickets triggered or were linked to it?
        """
        questions = []
        escalated = [
            t for t in threads if t["chain_type"] == "zendesk_ticket" and t["escalated"]
        ]
        logger.info(
            f"[eval] _zd_escalation_questions: {len(escalated)} escalated ZD threads available"
        )

        # Group by incident so we can ask "all tickets for incident X" questions
        by_incident: Dict[str, List[dict]] = {}
        for t in escalated:
            iid = t.get("incident_id", "")
            if iid:
                by_incident.setdefault(iid, []).append(t)

        for incident_id, tickets in random.sample(
            list(by_incident.items()), min(20, len(by_incident))
        ):
            ticket_ids = [t["root_artifact"] for t in tickets]
            orgs = list({t.get("org_name", "") for t in tickets if t.get("org_name")})

            ground_truth = {
                "incident_id": incident_id,
                "artifact_id": incident_id,
                "ticket_ids": ticket_ids,
                "ticket_count": len(ticket_ids),
                "affected_orgs": orgs,
                "event_type": "zd_tickets_escalated",
            }
            evidence = [incident_id] + ticket_ids

            q_text = self._generate_question_prose(
                template=(
                    f"Generate a causal question asking which Zendesk support "
                    f"tickets were escalated as a result of incident {incident_id}. "
                    f"The question should require the agent to trace from the "
                    f"incident back to the affected customer tickets. "
                    f"Output only the question text."
                )
            )
            if q_text:
                questions.append(
                    {
                        "question_id": f"zd_escalation_{incident_id}",
                        "question_type": "CAUSAL",
                        "question_text": q_text,
                        "ground_truth": ground_truth,
                        "evidence_chain": evidence,
                        "difficulty": "medium",
                        "requires_reasoning": True,
                        "chain_id": f"zd_escalation_{incident_id}",
                    }
                )
        return questions

    # ── SF_RISK — "Which accounts were flagged at-risk after incident X?" ─────

    def _sf_risk_questions(self, threads: List[dict]) -> List[dict]:
        """
        SF_RISK questions over sf_risk threads (sf_deals_risk_flagged events).
        New question type: distinct from CAUSAL because it specifically tests
        cross-system awareness — the agent must connect an engineering incident
        to its commercial impact in Salesforce.
        """
        questions = []
        risk_threads = [t for t in threads if t["chain_type"] == "sf_risk"]
        logger.info(
            f"[eval] _sf_risk_questions: {len(risk_threads)} SF risk threads available"
        )

        for thread in random.sample(risk_threads, min(25, len(risk_threads))):
            incident_id = thread["incident_id"]
            accounts = thread["at_risk_accounts"]
            if not accounts:
                continue

            ground_truth = {
                "incident_id": incident_id,
                "artifact_id": incident_id,
                "at_risk_accounts": accounts,
                "account_count": len(accounts),
                "day": thread["day"],
            }
            evidence = [incident_id]

            q_text = self._generate_question_prose(
                template=(
                    f"Generate a question asking which Salesforce customer accounts "
                    f"were flagged as at-risk following incident {incident_id}. "
                    f"The question should require the agent to connect an engineering "
                    f"incident to its downstream commercial impact. "
                    f"Output only the question text."
                )
            )
            if q_text:
                questions.append(
                    {
                        "question_id": f"sf_risk_{incident_id}",
                        "question_type": "SF_RISK",
                        "question_text": q_text,
                        "ground_truth": ground_truth,
                        "evidence_chain": evidence,
                        "difficulty": "hard",
                        "requires_reasoning": True,
                        "chain_id": thread["chain_id"],
                    }
                )
        return questions

    def _sf_ownership_lapse_questions(self, threads: List[dict]) -> List[dict]:
        """
        RETRIEVAL questions over sf_ownership_lapse threads.
        Uses existing RETRIEVAL scorer — ground truth is a deterministic account
        list derived from sf_ownership_lapsed SimEvents.
        Tests cross-domain reasoning: departure event → CRM consequence.
        """
        questions = []
        lapse_threads = [t for t in threads if t["chain_type"] == "sf_ownership_lapse"]
        logger.info(
            f"[eval] _sf_ownership_lapse_questions: {len(lapse_threads)} lapse threads available"
        )

        for thread in lapse_threads:  # typically low-volume; use all
            departed = thread["departed_employee"]
            accs = thread["lapsed_accounts"]
            opps = thread["lapsed_opportunities"]
            if not accs and not opps:
                continue

            ground_truth = {
                "departed_employee": departed,
                "role": thread.get("role", ""),
                "artifact_id": thread["root_artifact"],
                "lapsed_accounts": accs,
                "lapsed_opportunities": opps,
                "day": thread["day"],
            }
            evidence = accs[:3] + opps[:2]  # cap evidence list length

            q_text = self._generate_question_prose(
                template=(
                    f"Generate a retrieval question asking which Salesforce accounts "
                    f"or open opportunities were left without an owner after "
                    f"{departed} departed. The question should test whether an agent "
                    f"can trace the CRM impact of an employee departure. "
                    f"Output only the question text."
                )
            )
            if q_text:
                questions.append(
                    {
                        "question_id": f"sf_lapse_{departed.lower().replace(' ', '_')}",
                        "question_type": "RETRIEVAL",
                        "question_text": q_text,
                        "ground_truth": ground_truth,
                        "evidence_chain": [e for e in evidence if e],
                        "difficulty": "medium",
                        "requires_reasoning": True,
                        "chain_id": thread["chain_id"],
                    }
                )
        return questions

    def _sf_touchpoint_questions(self, threads: List[dict]) -> List[dict]:
        """
        CAUSAL questions over sf_touchpoint threads (crm_touchpoint events).
        Tests whether an agent can link an outbound sales email to the SF
        opportunity it created or advanced.
        """
        questions = []
        tp_threads = [t for t in threads if t["chain_type"] == "sf_touchpoint"]
        logger.info(
            f"[eval] _sf_touchpoint_questions: {len(tp_threads)} SF touchpoint threads available"
        )

        for thread in random.sample(tp_threads, min(25, len(tp_threads))):
            opp_id = thread["root_artifact"]
            sender = thread.get("sender", "a sales rep")
            account = thread.get("account_name", "a customer")
            stage = thread.get("stage", "")
            subject = thread.get("subject", "")

            ground_truth = {
                "opportunity_id": opp_id,
                "artifact_id": opp_id,
                "account_name": account,
                "stage": stage,
                "sender": sender,
                "event_type": "crm_touchpoint",
            }
            evidence = [opp_id]

            q_text = self._generate_question_prose(
                template=(
                    f"Generate a causal question asking which Salesforce opportunity "
                    f"was created or advanced when {sender} sent an outbound email "
                    f'with subject "{subject[:60]}" to {account}. The question '
                    f"should test whether an agent can trace from an outbound email "
                    f"to its CRM outcome. "
                    f"Output only the question text."
                )
            )
            if q_text:
                questions.append(
                    {
                        "question_id": f"sf_touchpoint_{opp_id}",
                        "question_type": "CAUSAL",
                        "question_text": q_text,
                        "ground_truth": ground_truth,
                        "evidence_chain": evidence,
                        "difficulty": "medium",
                        "requires_reasoning": True,
                        "chain_id": thread["chain_id"],
                    }
                )
        return questions

    def _datadog_alert_questions(self) -> List[dict]:
        """
        DATADOG_ALERT questions inferred from incident_opened SimEvents.

        Datadog alert records are generated post-sim from incident events
        (see post_sim_artifacts.py DatadogWriter). Because no dedicated
        'datadog_alert_fired' SimEvent is emitted during the sim, we derive
        ground truth directly from incident_opened facts: the monitor name
        and the incident it corresponds to are deterministically linked.

        The agent must retrieve the right incident artifact and demonstrate
        awareness that a Datadog alert is the upstream signal for the incident.
        Maps to RETRIEVAL scorer — ground truth is the incident artifact_id.
        """
        questions = []
        incident_events = [e for e in self._events if e.type == "incident_opened"]
        logger.info(
            f"[eval] _datadog_alert_questions: {len(incident_events)} incidents available"
        )

        for event in random.sample(incident_events, min(20, len(incident_events))):
            ticket_id = event.artifact_ids.get("jira", "")
            root_cause = event.facts.get("root_cause", "")
            if not ticket_id or not root_cause:
                continue

            ground_truth = {
                "incident_id": ticket_id,
                "artifact_id": ticket_id,
                "root_cause": root_cause,
                "open_day": event.day,
                # Monitor name is generated post-sim via LLM batch; ground truth
                # here is the incident ID so the scorer can do an exact match
                # without depending on the LLM-enriched monitor name string.
                "monitor_source": "datadog",
            }
            evidence = [ticket_id]

            q_text = self._generate_question_prose(
                template=(
                    f"Generate a retrieval question asking which Datadog alert or "
                    f"monitor fired to trigger incident {ticket_id}, whose root "
                    f'cause was: "{root_cause[:80]}". The question should test '
                    f"whether an agent can connect an observability alert to the "
                    f"incident it caused. "
                    f"Output only the question text."
                )
            )
            if q_text:
                questions.append(
                    {
                        "question_id": f"dd_alert_{ticket_id}",
                        "question_type": "DATADOG_ALERT",
                        "question_text": q_text,
                        "ground_truth": ground_truth,
                        "evidence_chain": evidence,
                        "difficulty": "medium",
                        "requires_reasoning": False,
                        "chain_id": f"incident_{ticket_id}",
                    }
                )
        return questions

    def _nps_score_questions(self) -> List[dict]:
        """
        NPS_SCORE questions inferred from ZD ticket + incident SimEvents.

        NPS responses are generated post-sim (see post_sim_artifacts.py NPSWriter).
        Ground truth is derived deterministically from the same scoring formula
        used by NPSWriter — no disk read required:
          Base 9; -3 per escalated ZD ticket; -2 per unresolved ZD ticket;
          -1 per SLA breach day; clamped to [0, 10].

        Tests whether an agent can reason across ZD tickets, incidents, and
        the NPS artifact to explain a customer's satisfaction score.
        """
        questions = []

        org_data: Dict[str, Dict] = {}

        for e in self._events:
            if e.type == "zd_ticket_opened":
                org = e.facts.get("org_name", "")
                if not org:
                    continue
                org_data.setdefault(
                    org,
                    {
                        "tickets": [],
                        "escalated_count": 0,
                        "unresolved_count": 0,
                        "breach_days": 0,
                    },
                )
                org_data[org]["tickets"].append(e.facts.get("ticket_id", ""))

            elif e.type == "zd_tickets_escalated":
                # Each escalated ticket adds to the org's escalated count
                iid = e.facts.get("incident_id", "")
                for tid in e.facts.get("ticket_ids", []):
                    # Find org for this ticket
                    for e2 in self._events:
                        if (
                            e2.type == "zd_ticket_opened"
                            and e2.facts.get("ticket_id") == tid
                        ):
                            org = e2.facts.get("org_name", "")
                            if org in org_data:
                                org_data[org]["escalated_count"] += 1
                            break

        if not org_data:
            logger.info("[eval] _nps_score_questions: no ZD ticket data — skipping")
            return questions

        for org, data in random.sample(list(org_data.items()), min(20, len(org_data))):
            ticket_ids = data["tickets"]
            if not ticket_ids:
                continue

            score = 9
            score -= 3 * data["escalated_count"]
            score = max(0, min(10, score))

            classification = (
                "promoter" if score >= 9 else "passive" if score >= 7 else "detractor"
            )

            ground_truth = {
                "org_name": org,
                "artifact_id": ticket_ids[0],
                "nps_score": score,
                "classification": classification,
                "escalated_tickets": data["escalated_count"],
                "ticket_ids": ticket_ids,
            }
            evidence = ticket_ids[:3]

            q_text = self._generate_question_prose(
                template=(
                    f"Generate a question asking what NPS score the customer "
                    f"{org} would give based on their support experience, and "
                    f"what factors drove that score. The question should require "
                    f"the agent to reason across support tickets and incident "
                    f"history to predict customer satisfaction. "
                    f"Output only the question text."
                )
            )
            if q_text:
                questions.append(
                    {
                        "question_id": f"nps_{org.lower().replace(' ', '_')}",
                        "question_type": "NPS_SCORE",
                        "question_text": q_text,
                        "ground_truth": ground_truth,
                        "evidence_chain": evidence,
                        "difficulty": "hard",
                        "requires_reasoning": True,
                        "chain_id": f"nps_{org.lower().replace(' ', '_')}",
                    }
                )
        return questions

    def _invoice_sla_questions(self) -> List[dict]:
        """
        INVOICE_SLA questions inferred from incident + ZD SimEvents.

        Invoices are generated post-sim (see post_sim_artifacts.py InvoiceWriter).
        Ground truth is derived from SLA breach logic: any incident that lasted
        more than SLA_BREACH_THRESHOLD_DAYS (1 day) generates an SLA credit
        on the affected customers' invoices at SLA_CREDIT_RATE (2%) of contract value.

        Tests whether an agent can reason about the financial consequences of
        incidents by tracing: incident duration → SLA breach → invoice line item.
        """
        questions = []

        resolved_by_id: Dict[str, SimEvent] = {
            e.artifact_ids.get("jira", ""): e
            for e in self._events
            if e.type == "incident_resolved"
        }

        sla_breach_incidents = []
        for e in self._events:
            if e.type != "incident_opened":
                continue
            ticket_id = e.artifact_ids.get("jira", "")
            if not ticket_id:
                continue
            resolve_event = resolved_by_id.get(ticket_id)
            if not resolve_event:
                continue
            duration = resolve_event.day - e.day
            if duration > 1:
                sla_breach_incidents.append(
                    {
                        "ticket_id": ticket_id,
                        "open_day": e.day,
                        "resolve_day": resolve_event.day,
                        "duration_days": duration,
                        "root_cause": e.facts.get("root_cause", ""),
                    }
                )

        incident_to_orgs: Dict[str, List[str]] = {}
        for e in self._events:
            if e.type == "zd_tickets_escalated":
                iid = e.facts.get("incident_id", "")
                if not iid:
                    continue
                for tid in e.facts.get("ticket_ids", []):
                    for e2 in self._events:
                        if (
                            e2.type == "zd_ticket_opened"
                            and e2.facts.get("ticket_id") == tid
                        ):
                            org = e2.facts.get("org_name", "")
                            if org:
                                incident_to_orgs.setdefault(iid, []).append(org)
                            break
            elif e.type == "sf_deals_risk_flagged":
                iid = e.facts.get("incident_id", "")
                if iid:
                    for org in e.facts.get("account_names", []):
                        incident_to_orgs.setdefault(iid, []).append(org)

        logger.info(
            f"[eval] _invoice_sla_questions: {len(sla_breach_incidents)} "
            f"SLA-breaching incidents available"
        )

        for inc in random.sample(
            sla_breach_incidents, min(20, len(sla_breach_incidents))
        ):
            ticket_id = inc["ticket_id"]
            orgs = list(set(incident_to_orgs.get(ticket_id, [])))
            if not orgs:
                continue

            credit_per_org = round(50_000 * 0.02 * inc["duration_days"], 2)

            ground_truth = {
                "incident_id": ticket_id,
                "artifact_id": ticket_id,
                "breach_duration_days": inc["duration_days"],
                "affected_orgs": orgs,
                "sla_credit_per_org": credit_per_org,
                "root_cause": inc["root_cause"],
            }
            evidence = [ticket_id]

            q_text = self._generate_question_prose(
                template=(
                    f"Generate a question asking what SLA credit would appear on "
                    f"the invoice for customers affected by incident {ticket_id}, "
                    f"which remained open for {inc['duration_days']} days. The "
                    f"question should require the agent to reason about SLA breach "
                    f"thresholds and financial credit calculations. "
                    f"Output only the question text."
                )
            )
            if q_text:
                questions.append(
                    {
                        "question_id": f"invoice_sla_{ticket_id}",
                        "question_type": "INVOICE_SLA",
                        "question_text": q_text,
                        "ground_truth": ground_truth,
                        "evidence_chain": evidence,
                        "difficulty": "hard",
                        "requires_reasoning": True,
                        "chain_id": f"incident_{ticket_id}",
                    }
                )
        return questions

    def _pr_review_questions(self, threads: List[dict]) -> List[dict]:
        questions = []
        pr_events = [e for e in self._events if e.type == "pr_review"]
        logger.info(
            f"[eval] _pr_review_questions: {len(pr_events)} pr_review events available"
        )

        for event in random.sample(pr_events, min(40, len(pr_events))):
            pr_id = event.artifact_ids.get("pr", "")
            reviewer = event.facts.get("reviewer", "")
            author = event.facts.get("author", "")
            verdict = event.facts.get("verdict", "")
            pr_title = event.facts.get("pr_title", "")
            linked_ticket = event.artifact_ids.get("jira", "")

            if not pr_id or not reviewer or not verdict:
                continue

            ground_truth_review = {
                "pr_id": pr_id,
                "reviewer": reviewer,
                "author": author,
                "verdict": verdict,
                "linked_ticket": linked_ticket,
                "day": event.day,
            }
            evidence_review = [pr_id] + ([linked_ticket] if linked_ticket else [])

            q_text_review = self._generate_question_prose(
                template=(
                    f"Generate a retrieval question asking who reviewed pull request "
                    f"{pr_id} and whether it was approved or had changes requested. "
                    f"The question should test whether an agent can locate the review "
                    f"record and identify both the reviewer and their verdict. "
                    f"Output only the question text."
                )
            )
            if q_text_review:
                questions.append(
                    {
                        "question_id": f"pr_review_{pr_id}",
                        "question_type": "PR_REVIEW",
                        "question_text": q_text_review,
                        "ground_truth": ground_truth_review,
                        "evidence_chain": [e for e in evidence_review if e],
                        "difficulty": "easy",
                        "requires_reasoning": False,
                        "chain_id": f"pr_{pr_id}",
                    }
                )

            if not linked_ticket:
                continue

            ground_truth_causal = {
                "artifact_id": pr_id,
                "event_type": "pr_review",
                "actors": [reviewer, author],
                "timestamp": event.timestamp,
                "verdict": verdict,
            }
            evidence_causal = [linked_ticket, pr_id]

            q_text_causal = self._generate_question_prose(
                template=(
                    f"Generate a causal question asking what pull request was opened "
                    f"or reviewed as a result of work on ticket {linked_ticket}. "
                    f"The question should require the agent to trace from a Jira ticket "
                    f"to the GitHub PR that resolved it. "
                    f"Output only the question text."
                )
            )
            if q_text_causal:
                questions.append(
                    {
                        "question_id": f"pr_causal_{linked_ticket}_{pr_id}",
                        "question_type": "CAUSAL",
                        "question_text": q_text_causal,
                        "ground_truth": ground_truth_causal,
                        "evidence_chain": evidence_causal,
                        "difficulty": "medium",
                        "requires_reasoning": True,
                        "chain_id": f"pr_{pr_id}",
                    }
                )

        return questions

    def _blocker_questions(self) -> List[dict]:
        questions = []
        blocker_events = [e for e in self._events if e.type == "blocker_flagged"]
        logger.info(
            f"[eval] _blocker_questions: {len(blocker_events)} blocker_flagged events available"
        )

        seen_tickets: Set[str] = set()
        for event in random.sample(blocker_events, min(25, len(blocker_events))):
            ticket_id = event.artifact_ids.get("jira", "")
            slack_thread = event.artifact_ids.get("slack_thread", "")
            assignee = next(iter(event.actors), None)
            blocker_reason = event.facts.get(
                "comment", event.facts.get("blocker_reason", "")
            )

            if not ticket_id or not assignee or ticket_id in seen_tickets:
                continue
            seen_tickets.add(ticket_id)

            ground_truth = {
                "artifact_id": ticket_id,
                "was_blocked": True,
                "assignee": assignee,
                "ticket_id": ticket_id,
                "day": event.day,
                "slack_thread": slack_thread,
            }
            evidence = [ticket_id] + ([slack_thread] if slack_thread else [])

            q_text = self._generate_question_prose(
                template=(
                    f"Generate a retrieval question asking what ticket {assignee} was "
                    f"blocked on during Day {event.day}, and what the blocker was. "
                    f"The question should test whether an agent can locate a blocker "
                    f"report in the ticket or Slack history. "
                    f"Output only the question text."
                )
            )
            if q_text:
                questions.append(
                    {
                        "question_id": f"blocker_{ticket_id}_day{event.day}",
                        "question_type": "RETRIEVAL",
                        "question_text": q_text,
                        "ground_truth": ground_truth,
                        "evidence_chain": [e for e in evidence if e],
                        "difficulty": "medium",
                        "requires_reasoning": False,
                        "chain_id": f"blocker_{ticket_id}",
                    }
                )

        blocked_ticket_ids = {e.artifact_ids.get("jira", "") for e in blocker_events}
        progress_events = [
            e
            for e in self._events
            if e.type == "ticket_progress"
            and e.artifact_ids.get("jira", "") not in blocked_ticket_ids
            and e.artifact_ids.get("jira", "")
        ]
        for event in random.sample(progress_events, min(15, len(progress_events))):
            ticket_id = event.artifact_ids.get("jira", "")
            assignee = next(iter(event.actors), None)
            if not ticket_id or not assignee:
                continue

            ground_truth = {
                "artifact_id": ticket_id,
                "was_blocked": False,
                "assignee": assignee,
                "ticket_id": ticket_id,
                "day": event.day,
                "slack_thread": None,
            }

            q_text = self._generate_question_prose(
                template=(
                    f"Generate a question asking whether {assignee} reported any "
                    f"blockers while working on ticket {ticket_id} on Day {event.day}. "
                    f"The question should require the agent to check the ticket history "
                    f"and confirm whether a blocker was or wasn't reported. "
                    f"Output only the question text."
                )
            )
            if q_text:
                questions.append(
                    {
                        "question_id": f"blocker_check_{ticket_id}_day{event.day}",
                        "question_type": "RETRIEVAL",
                        "question_text": q_text,
                        "ground_truth": ground_truth,
                        "evidence_chain": [ticket_id],
                        "difficulty": "easy",
                        "requires_reasoning": False,
                        "chain_id": f"blocker_{ticket_id}",
                    }
                )

        logger.info(
            f"[eval] _blocker_questions: generated {len(questions)} questions "
            f"(pool_A={len(blocker_events)}, pool_B={len(progress_events)})"
        )
        return questions

    def _vendor_routing_questions(self) -> List[dict]:
        questions = []
        vendor_events = [e for e in self._events if e.type == "vendor_email_routed"]
        logger.info(
            f"[eval] _vendor_routing_questions: {len(vendor_events)} vendor_email_routed events available"
        )

        for event in random.sample(vendor_events, min(25, len(vendor_events))):
            email_id = event.artifact_ids.get("email", "")
            vendor = event.facts.get("vendor", "")
            topic = event.facts.get("topic", "")
            routed_to = event.facts.get("routed_to", "")
            causal_chain = event.facts.get("causal_chain", [])

            if not email_id or not vendor or not routed_to:
                continue

            jira_opened = any(
                a
                for a in causal_chain
                if isinstance(a, str) and (a.startswith("IT-") or a.startswith("ORG-"))
            )

            ground_truth = {
                "first_recipient": routed_to,
                "artifact_id": email_id,
                "vendor": vendor,
                "topic": topic,
                "jira_opened": jira_opened,
                "timestamp": event.timestamp,
            }
            evidence = [email_id] + ([c for c in causal_chain[:2] if c != email_id])

            q_text = self._generate_question_prose(
                template=(
                    f"Generate a routing question asking which internal engineer or "
                    f"team member handled the alert or email from {vendor} regarding "
                    f'"{topic[:60]}". The question should test whether an agent can '
                    f"trace inbound vendor communications to the responsible engineer. "
                    f"Output only the question text."
                )
            )
            if q_text:
                questions.append(
                    {
                        "question_id": f"vendor_routing_{email_id}",
                        "question_type": "ROUTING",
                        "question_text": q_text,
                        "ground_truth": ground_truth,
                        "evidence_chain": [e for e in evidence if e],
                        "difficulty": "medium",
                        "requires_reasoning": False,
                        "chain_id": f"vendor_{email_id}",
                    }
                )
        return questions

    def _design_discussion_questions(self) -> List[dict]:
        questions = []
        dd_events = [e for e in self._events if e.type == "design_discussion"]
        logger.info(
            f"[eval] _design_discussion_questions: {len(dd_events)} design_discussion events available"
        )

        for event in random.sample(dd_events, min(30, len(dd_events))):
            topic = event.facts.get("topic", "")
            participants = event.facts.get("participants", event.actors)
            medium = event.facts.get("medium", "slack")
            spawned_doc = event.facts.get("spawned_doc", False)
            conf_id = event.artifact_ids.get("confluence", "")

            artifact_id = (
                event.artifact_ids.get("zoom_transcript")
                or event.artifact_ids.get("slack_thread")
                or ""
            )
            linked_ticket = event.artifact_ids.get("jira", "")

            if not artifact_id or not topic or not participants:
                continue

            ground_truth_ret = {
                "artifact_id": artifact_id,
                "topic": topic,
                "participants": participants,
                "medium": medium,
                "day": event.day,
                "linked_ticket": linked_ticket,
            }
            evidence_ret = [artifact_id] + ([linked_ticket] if linked_ticket else [])

            q_text_ret = self._generate_question_prose(
                template=(
                    f"Generate a retrieval question asking who participated in the "
                    f"{'Zoom call' if medium == 'zoom' else 'Slack design discussion'} "
                    f'about "{topic[:80]}". The question should require the agent to '
                    f"locate the {'transcript' if medium == 'zoom' else 'thread'} and "
                    f"identify all participants. "
                    f"Output only the question text."
                )
            )
            if q_text_ret:
                questions.append(
                    {
                        "question_id": f"design_discussion_{artifact_id}",
                        "question_type": "RETRIEVAL",
                        "question_text": q_text_ret,
                        "ground_truth": ground_truth_ret,
                        "evidence_chain": [e for e in evidence_ret if e],
                        "difficulty": "easy",
                        "requires_reasoning": False,
                        "chain_id": f"design_{artifact_id}",
                    }
                )

            if not spawned_doc or not conf_id:
                continue

            ground_truth_causal = {
                "artifact_id": conf_id,
                "event_type": "confluence_created",
                "actors": participants,
                "timestamp": event.timestamp,
                "source_discussion": artifact_id,
            }
            evidence_causal = [artifact_id, conf_id]

            q_text_causal = self._generate_question_prose(
                template=(
                    f"Generate a causal question asking which Confluence document was "
                    f'created as a result of the design discussion on "{topic[:80]}". '
                    f"The question should require the agent to trace from the discussion "
                    f"artifact to the documentation it produced. "
                    f"Output only the question text."
                )
            )
            if q_text_causal:
                questions.append(
                    {
                        "question_id": f"design_doc_spawned_{artifact_id}",
                        "question_type": "CAUSAL",
                        "question_text": q_text_causal,
                        "ground_truth": ground_truth_causal,
                        "evidence_chain": evidence_causal,
                        "difficulty": "medium",
                        "requires_reasoning": True,
                        "chain_id": f"design_{artifact_id}",
                    }
                )

        return questions

    def _generate_question_prose(self, template: str) -> Optional[str]:
        """LLM writes question wording only. Never touches ground truth."""
        agent = make_agent(
            role="Eval Dataset Author",
            goal="Write natural-sounding evaluation questions for AI agent benchmarks.",
            backstory=(
                "You write clear, specific questions for evaluating AI retrieval "
                "and reasoning systems. Questions should be unambiguous and answerable "
                "from a corporate document corpus."
            ),
            llm=self._worker_llm,
        )
        task = Task(
            description=template,
            expected_output="One question ending with a question mark. No preamble.",
            agent=agent,
        )
        try:
            result = str(
                Crew(agents=[agent], tasks=[task], verbose=False).kickoff()
            ).strip()
            # Ensure it ends with ?
            if not result.endswith("?"):
                result = result.rstrip(".") + "?"
            return result
        except Exception as exc:
            logger.warning(f"[eval] Question prose generation failed: {exc}")
            return None

    def _find_event_by_artifact(self, artifact_id: str) -> Optional[SimEvent]:
        for event in self._events:
            if artifact_id in event.artifact_ids.values():
                return event
        return None


class EvalHarness:
    def __init__(self):
        from flow import build_llm

        self._mem = Memory()
        self._worker_llm = build_llm("worker")

    def run(self) -> None:
        logger.info("[bold cyan]🔬 Building eval dataset...[/bold cyan]")

        builder = CausalThreadBuilder(self._mem)
        threads = builder.build_all()
        logger.info(f"  {len(threads)} causal threads extracted")

        threads_path = EVAL_DIR / "causal_threads.json"
        with open(threads_path, "w") as f:
            json.dump(threads, f, indent=2, default=str)
        logger.info(f"  → {threads_path}")

        generator = EvalQuestionGenerator(self._mem, self._worker_llm)
        questions = generator.generate(threads)
        logger.info(f"  {len(questions)} eval questions generated")

        by_type: Dict[str, int] = defaultdict(int)
        by_difficulty: Dict[str, int] = defaultdict(int)
        for q in questions:
            by_type[q["question_type"]] += 1
            by_difficulty[q["difficulty"]] += 1

        questions_path = EVAL_DIR / "eval_questions.json"
        with open(questions_path, "w") as f:
            json.dump(
                {
                    "metadata": {
                        "generated_at": datetime.now().isoformat(),
                        "total_questions": len(questions),
                        "by_type": dict(by_type),
                        "by_difficulty": dict(by_difficulty),
                    },
                    "questions": questions,
                },
                f,
                indent=2,
                default=str,
            )
        logger.info(f"  → {questions_path}")
        logger.info(
            f"[green]✓ Eval dataset complete.[/green] "
            f"Types: {dict(by_type)} | Difficulty: {dict(by_difficulty)}"
        )


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    EvalHarness().run()
