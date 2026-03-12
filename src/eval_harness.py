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

  GAP_DETECTION "Was the email from X ever actioned?"
                Answer: boolean derived from email_dropped / customer_email_routed events

  ROUTING       "Who was the first internal person to see the complaint from X?"
                Answer: liaison name from inbound_external_email SimEvent

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
import os
import random
import yaml
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from agent_factory import make_agent
from crewai import Crew, Task
from memory import Memory, SimEvent

logger = logging.getLogger("orgforge.eval")

# ── Config ────────────────────────────────────────────────────────────────────
with open("config.yaml") as f:
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
        self._events: List[SimEvent] = mem.get_event_log()

    def build_all(self) -> List[dict]:
        threads = []
        threads.extend(self._incident_threads())
        threads.extend(self._customer_email_threads())
        threads.extend(self._hr_threads())
        threads.extend(self._dropped_email_threads())
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
            threads.append({
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
            })
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
            threads.append({
                "chain_id": f"customer_email_{email_id}",
                "chain_type": "customer_email",
                "root_artifact": email_id,
                "root_event_type": "inbound_external_email",
                "day": event.day,
                "date": event.date,
                "nodes": nodes,
                "terminal_artifact": chain[-1] if chain else email_id,
                "complete": len(chain) > 1,   # False if only root = no action
                "high_priority": event.facts.get("high_priority", False),
                "source": event.facts.get("source"),
                "jira_opened": any("IT-" in n or "ORG-" in n for n in chain[1:]),
            })
        return threads

    def _dropped_email_threads(self) -> List[dict]:
        """Dropped emails are their own thread type — single-node chains."""
        threads = []
        dropped = [e for e in self._events if e.type == "email_dropped"]
        for event in dropped:
            email_id = event.artifact_ids.get("email")
            if not email_id:
                continue
            threads.append({
                "chain_id": f"dropped_email_{email_id}",
                "chain_type": "dropped_email",
                "root_artifact": email_id,
                "root_event_type": "inbound_external_email",
                "day": event.day,
                "date": event.date,
                "nodes": [{
                    "artifact_id": email_id,
                    "event_type": "inbound_external_email",
                    "timestamp": event.timestamp,
                    "known_to": event.actors,
                    "caused": [],     # intentionally empty
                }],
                "terminal_artifact": email_id,
                "complete": False,    # never actioned — this is the eval signal
                "source": event.facts.get("source"),
                "subject": event.facts.get("subject"),
            })
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
                (e for e in hired
                 if prospect in e.actors and e.day == hire_day),
                None,
            )

            chain = [embed_id]
            if hire_event:
                hire_artifact = hire_event.artifact_ids.get("jira",
                                hire_event.artifact_ids.get("slack_thread", ""))
                if hire_artifact:
                    chain.append(hire_artifact)

            threads.append({
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
            })
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
                nodes.append({
                    "artifact_id": artifact_id,
                    "event_type": event.type,
                    "timestamp": event.timestamp,
                    "day": event.day,
                    "known_to": list(cumulative_known),
                    "caused": caused,
                })
            else:
                # Artifact exists but no event found — still include it
                nodes.append({
                    "artifact_id": artifact_id,
                    "event_type": "unknown",
                    "timestamp": root_event.timestamp,
                    "day": root_event.day,
                    "known_to": list(root_event.actors),
                    "caused": [chain[i + 1]] if i + 1 < len(chain) else [],
                })
        return nodes

    def _find_event_for_artifact(self, artifact_id: str) -> Optional[SimEvent]:
        """Find the SimEvent most directly associated with an artifact ID."""
        for event in self._events:
            if artifact_id in event.artifact_ids.values():
                return event
            if artifact_id in event.facts.get("causal_chain", []):
                continue   # don't match chain references, only direct artifact_ids
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
        self._events: List[SimEvent] = mem.get_event_log()

    def generate(self, threads: List[dict]) -> List[dict]:
        questions = []
        questions.extend(self._retrieval_questions(threads))
        questions.extend(self._causal_questions(threads))
        questions.extend(self._temporal_questions())
        questions.extend(self._gap_detection_questions(threads))
        questions.extend(self._routing_questions(threads))
        return questions

    # ── RETRIEVAL — "Which artifact first mentioned X?" ───────────────────────

    def _retrieval_questions(self, threads: List[dict]) -> List[dict]:
        questions = []
        incident_threads = [t for t in threads if t["chain_type"] == "incident"]
        for thread in random.sample(incident_threads, min(3, len(incident_threads))):
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
                    f"first documented this incident: \"{root_cause[:80]}\". "
                    f"The question should sound like a natural analyst query. "
                    f"Output only the question text."
                )
            )
            if q_text:
                questions.append({
                    "question_id": f"retrieval_{thread['root_artifact']}",
                    "question_type": "RETRIEVAL",
                    "question_text": q_text,
                    "ground_truth": ground_truth,
                    "evidence_chain": evidence,
                    "difficulty": "easy",
                    "requires_reasoning": False,
                    "chain_id": thread["chain_id"],
                })
        return questions

    # ── CAUSAL — "What happened after X?" ────────────────────────────────────

    def _causal_questions(self, threads: List[dict]) -> List[dict]:
        questions = []
        multi_hop = [
            t for t in threads
            if len(t.get("nodes", [])) >= 3 and t["chain_type"] in
            ("incident", "customer_email")
        ]
        for thread in random.sample(multi_hop, min(4, len(multi_hop))):
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
                    f"was produced as a direct result of: \"{trigger_desc[:100]}\". "
                    f"The question should probe cause-and-effect reasoning. "
                    f"Output only the question text."
                )
            )
            if q_text:
                questions.append({
                    "question_id": f"causal_{trigger_node['artifact_id']}",
                    "question_type": "CAUSAL",
                    "question_text": q_text,
                    "ground_truth": ground_truth,
                    "evidence_chain": evidence,
                    "difficulty": "medium",
                    "requires_reasoning": True,
                    "chain_id": thread["chain_id"],
                })
        return questions

    # ── TEMPORAL — "Did person P know about X when they made decision D?" ─────

    def _temporal_questions(self) -> List[dict]:
        """
        Uses actor knowledge snapshots from day_summary SimEvents.
        Asks whether a person had relevant information before a key decision.
        """
        questions = []
        incidents = [e for e in self._events if e.type == "incident_opened"
                     and e.facts.get("involves_gap")]

        for event in random.sample(incidents, min(3, len(incidents))):
            ticket_id = event.artifact_ids.get("jira")
            assignee = next(iter(event.actors), None)
            gap_areas = event.facts.get("gap_areas", [])
            if not assignee or not gap_areas:
                continue

            # Deterministic: check if any departure event for this gap area
            # happened before the incident day
            gap_domain = gap_areas[0]
            departure = next(
                (e for e in self._events
                 if e.type == "employee_departed"
                 and gap_domain in str(e.facts.get("knowledge_domains", []))
                 and e.day < event.day),
                None,
            )

            had_knowledge = departure is None   # True if no prior departure
            ground_truth = {
                "had_knowledge": had_knowledge,
                "person": assignee,
                "domain": gap_domain,
                "incident_day": event.day,
                "departure_day": departure.day if departure else None,
                "reasoning": (
                    f"{'No' if not had_knowledge else 'No'} departure of the "
                    f"{gap_domain} knowledge owner had occurred before Day {event.day}."
                    if not departure else
                    f"{departure.actors[0]} (who owned {gap_domain}) left on "
                    f"Day {departure.day}, {event.day - departure.day} days before "
                    f"this incident."
                ),
            }
            evidence = [ticket_id]
            if departure:
                evidence.append(departure.artifact_ids.get("jira",
                                departure.artifact_ids.get("slack_thread", "")))

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
            if q_text:
                questions.append({
                    "question_id": f"temporal_{ticket_id}_{gap_domain}",
                    "question_type": "TEMPORAL",
                    "question_text": q_text,
                    "ground_truth": ground_truth,
                    "evidence_chain": [e for e in evidence if e],
                    "difficulty": "hard",
                    "requires_reasoning": True,
                    "chain_id": f"incident_{ticket_id}",
                })
        return questions

    # ── GAP DETECTION — "Was this email ever actioned?" ──────────────────────

    def _gap_detection_questions(self, threads: List[dict]) -> List[dict]:
        questions = []
        dropped = [t for t in threads if t["chain_type"] == "dropped_email"]
        routed = [t for t in threads if t["chain_type"] == "customer_email"
                  and t.get("complete")]

        # Questions about dropped emails (answer: no action)
        for thread in random.sample(dropped, min(3, len(dropped))):
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
                    f"\"{subject[:60]}\". The question should require the agent to "
                    f"search for downstream artifacts and notice their absence. "
                    f"Output only the question text."
                )
            )
            if q_text:
                questions.append({
                    "question_id": f"gap_{thread['root_artifact']}",
                    "question_type": "GAP_DETECTION",
                    "question_text": q_text,
                    "ground_truth": ground_truth,
                    "evidence_chain": [thread["root_artifact"]],
                    "difficulty": "hard",
                    "requires_reasoning": True,
                    "chain_id": thread["chain_id"],
                })

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
                questions.append({
                    "question_id": f"gap_{thread['root_artifact']}_actioned",
                    "question_type": "GAP_DETECTION",
                    "question_text": q_text,
                    "ground_truth": ground_truth,
                    "evidence_chain": chain,
                    "difficulty": "medium",
                    "requires_reasoning": True,
                    "chain_id": thread["chain_id"],
                })
        return questions

    # ── ROUTING — "Who first saw this?" ──────────────────────────────────────

    def _routing_questions(self, threads: List[dict]) -> List[dict]:
        questions = []
        customer_threads = [
            t for t in threads
            if t["chain_type"] in ("customer_email", "dropped_email")
        ]
        for thread in random.sample(customer_threads, min(3, len(customer_threads))):
            root_node = thread["nodes"][0] if thread["nodes"] else None
            if not root_node:
                continue
            first_internal = next(
                (a for a in root_node.get("known_to", [])
                 if "@" not in a),   # exclude email addresses
                None,
            )
            if not first_internal:
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
                questions.append({
                    "question_id": f"routing_{thread['root_artifact']}",
                    "question_type": "ROUTING",
                    "question_text": q_text,
                    "ground_truth": ground_truth,
                    "evidence_chain": [thread["root_artifact"]],
                    "difficulty": "easy",
                    "requires_reasoning": False,
                    "chain_id": thread["chain_id"],
                })
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
        from config_loader import _PRESET, _PROVIDER
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
            json.dump({
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "total_questions": len(questions),
                    "by_type": dict(by_type),
                    "by_difficulty": dict(by_difficulty),
                },
                "questions": questions,
            }, f, indent=2, default=str)
        logger.info(f"  → {questions_path}")
        logger.info(
            f"[green]✓ Eval dataset complete.[/green] "
            f"Types: {dict(by_type)} | Difficulty: {dict(by_difficulty)}"
        )


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    EvalHarness().run()
