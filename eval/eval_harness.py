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

# ── Config ────────────────────────────────────────────────────────────────────
with open(Path(__file__).resolve().parent.parent / "config" / "config.yaml") as f:
    _CFG = yaml.safe_load(f)

BASE = Path(_CFG["simulation"].get("output_dir", "./export"))
EVAL_DIR = BASE / "eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# CAUSAL THREAD BUILDER
# ─────────────────────────────────────────────────────────────────────────────


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

            # The causal chain is: incident ticket → postmortem confluence page
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
                    "complete": True,  # postmortem_created means it was written
                    "confluence_id": conf_id,
                    "root_cause": event.facts.get("root_cause", ""),
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
            # Find the SimEvent that created this artifact
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
                # Artifact exists but no event found — still include it
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
                continue  # don't match chain references, only direct artifact_ids
        return None


# ─────────────────────────────────────────────────────────────────────────────
# QUESTION GENERATOR
# ─────────────────────────────────────────────────────────────────────────────


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
        questions.extend(self._plan_questions())
        # questions.extend(self._plan_execution_questions()) removed — proposed_events is future feature
        questions.extend(self._escalation_questions())
        questions.extend(self._knowledge_gap_questions())
        questions.extend(self._postmortem_questions(threads))
        questions.extend(self._confluence_questions())
        questions.extend(self._standup_questions())
        questions.extend(self._customer_escalation_questions())
        return questions

    # ── RETRIEVAL — "Which artifact first mentioned X?" ───────────────────────

    def _retrieval_questions(self, threads: List[dict]) -> List[dict]:
        questions = []
        incident_threads = [t for t in threads if t["chain_type"] == "incident"]
        logger.info(
            f"[eval] _retrieval_questions: {len(incident_threads)} incident threads available"
        )
        for thread in random.sample(incident_threads, min(12, len(incident_threads))):
            root_event = self._find_event_by_artifact(thread["root_artifact"])
            if not root_event:
                continue
            root_cause = root_event.facts.get("root_cause", "")
            if not root_cause:
                continue

            # Deterministic answer: the root artifact ID
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

    # ── CAUSAL — "What happened after X?" ────────────────────────────────────

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
        for thread in random.sample(multi_hop, min(10, len(multi_hop))):
            nodes = thread["nodes"]
            # Ask about the transition from node 1 → node 2
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

    # ── TEMPORAL — "Did person P know about X when they made decision D?" ─────

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
                # Use the first meaningful token from the root cause description.
                # Strip common stop words so we don't get gap_domain="the" etc.
                _stop = {"the", "a", "an", "in", "of", "on", "was", "is", "due", "to"}
                tokens = [
                    t
                    for t in root_cause.split()
                    if t.lower() not in _stop and len(t) > 4
                ]
                gap_domain = tokens[0] if tokens else "system"
            else:
                # Last resort: most recent departure's first domain
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

            # ── Dedup guard ───────────────────────────────────────────────────
            # Skip if we already have a question for this (person, domain) pair.
            # This prevents a single dominant departure from generating N identical
            # questions that only differ in ticket number.
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

        # Sample both pools independently.
        # Caps raised to 8 per pool so short sim runs still yield meaningful n.
        # The (person, domain) dedup guard inside _build_temporal_question ensures
        # we don't burn all slots on the same departure-domain pair.
        for event in random.sample(pool_a, min(8, len(pool_a))):
            q = _build_temporal_question(event)
            if q:
                questions.append(q)

        for event in random.sample(pool_b, min(8, len(pool_b))):
            q = _build_temporal_question(event)
            if q:
                questions.append(q)

        # Log had_knowledge balance — if this is all False, the sim run has
        # only one departure and the eval will be gameable by always answering "no".
        hk_dist = {True: 0, False: 0}
        for q in questions:
            hk_dist[q["ground_truth"]["had_knowledge"]] += 1
        logger.info(
            f"[eval] _temporal_questions: pool_a={len(pool_a)}, pool_b={len(pool_b)}, "
            f"generated={len(questions)}, had_knowledge={hk_dist}"
        )
        return questions

    # ── GAP DETECTION — "Was this email ever actioned?" ──────────────────────

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

        # Questions about dropped emails (answer: no action)
        for thread in random.sample(dropped, min(5, len(dropped))):
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

        # Paired questions about routed emails (answer: yes, action was taken)
        for thread in random.sample(routed, min(2, len(routed))):
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

    # ── ROUTING — "Who first saw this?" ──────────────────────────────────────

    def _routing_questions(self, threads: List[dict]) -> List[dict]:
        questions = []
        customer_threads = [
            t for t in threads if t["chain_type"] in ("customer_email", "dropped_email")
        ]
        logger.info(
            f"[eval] _routing_questions: {len(customer_threads)} customer threads available"
        )
        for thread in random.sample(customer_threads, min(5, len(customer_threads))):
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

    # ── SPRINT PLANNING — "What did person X plan to work on during Day N?" ──────
    def _plan_questions(self) -> List[dict]:
        questions = []
        try:
            plans = list(self._mem._db["dept_plans"].find({}, {"_id": 0}))
        except Exception:
            return []

        for plan in random.sample(plans, min(15, len(plans))):
            dept = plan.get("dept", "")
            day = plan.get("day", 0)
            lead = plan.get("lead", "")
            theme = plan.get("theme", "")
            # Extract actors from nested engineer_plans
            actors = [
                ep["name"] for ep in plan.get("engineer_plans", []) if ep.get("name")
            ]
            if not dept or not theme or not actors:
                continue

            ground_truth = {
                "dept": dept,
                "day": day,
                "lead": lead,
                "theme": theme,
                "actors": actors,
                "artifact_id": f"PLAN-{day}-{dept}",
                # proposed_events intentionally excluded
            }

            q_text = self._generate_question_prose(
                template=(
                    f"Generate a question asking what the {dept} department's "
                    f"focus or theme was on Day {day} of the simulation. "
                    f"The question should sound like a planning retrospective query. "
                    f"Output only the question text."
                )
            )
            if q_text:
                questions.append(
                    {
                        "question_id": f"plan_{dept}_day{day}",
                        "question_type": "PLAN",
                        "question_text": q_text,
                        "ground_truth": ground_truth,
                        "evidence_chain": [f"PLAN-{day}-{dept}"],
                        "difficulty": "easy",
                        "requires_reasoning": False,
                        "chain_id": f"plan_{dept}_day{day}",
                    }
                )
        return questions

    def _plan_execution_questions(self) -> List[dict]:
        questions = []
        try:
            plans = list(self._mem._db["dept_plans"].find({}, {"_id": 0}))
        except Exception:
            return []

        for plan in plans:
            dept = plan.get("dept", "")
            day = plan.get("day", 0)
            for proposed in plan.get("proposed_events", []):
                event_type = proposed.get("event_type", "")
                actors = proposed.get("actors", [])
                if not event_type or not actors:
                    continue

                # Check if a matching SimEvent was actually logged
                executed = any(
                    e
                    for e in self._events
                    if e.type == event_type
                    and e.day == day
                    and any(a in e.actors for a in actors)
                )

                ground_truth = {
                    "was_executed": executed,
                    "event_type": event_type,
                    "dept": dept,
                    "day": day,
                    "proposed_actors": actors,
                }

                q_text = self._generate_question_prose(
                    template=(
                        f"Generate a question asking whether the proposed "
                        f"{event_type.replace('_', ' ')} for the {dept} department "
                        f"on Day {day} actually took place. The question should test "
                        f"whether an agent can verify planned vs actual activity. "
                        f"Output only the question text."
                    )
                )
                if q_text:
                    questions.append(
                        {
                            "question_id": f"plan_exec_{dept}_{event_type}_day{day}",
                            "question_type": "PLAN_EXECUTION",
                            "question_text": q_text,
                            "ground_truth": ground_truth,
                            "evidence_chain": [f"PLAN-{day}-{dept}"],
                            "difficulty": "medium",
                            "requires_reasoning": True,
                            "chain_id": f"plan_{dept}_day{day}",
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

        for event in random.sample(esc_events, min(6, len(esc_events))):
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

        for event in random.sample(gap_events, min(6, len(gap_events))):
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

        for thread in random.sample(pm_threads, min(6, len(pm_threads))):
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

        for event in random.sample(conf_events, min(8, len(conf_events))):
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

        for event in random.sample(standup_events, min(8, len(standup_events))):
            if not event.actors:
                continue
            # Pick one actor from the standup to ask about
            actor = random.choice(event.actors)
            slack_thread = event.artifact_ids.get("slack_thread", "")
            day = event.day

            ground_truth = {
                "actor": actor,
                "day": day,
                "slack_thread_id": slack_thread,
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

        for event in random.sample(esc_events, min(6, len(esc_events))):
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

    # ── LLM question prose ────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────────────────────


class EvalHarness:
    def __init__(self):
        from flow import build_llm

        self._mem = Memory()
        self._worker_llm = build_llm("worker")

    def run(self) -> None:
        logger.info("[bold cyan]🔬 Building eval dataset...[/bold cyan]")

        # 1. Build causal threads
        builder = CausalThreadBuilder(self._mem)
        threads = builder.build_all()
        logger.info(f"  {len(threads)} causal threads extracted")

        threads_path = EVAL_DIR / "causal_threads.json"
        with open(threads_path, "w") as f:
            json.dump(threads, f, indent=2, default=str)
        logger.info(f"  → {threads_path}")

        # 2. Generate eval questions
        generator = EvalQuestionGenerator(self._mem, self._worker_llm)
        questions = generator.generate(threads)
        logger.info(f"  {len(questions)} eval questions generated")

        # Annotate with difficulty distribution
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
