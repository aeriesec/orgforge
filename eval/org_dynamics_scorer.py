"""
org_dynamics_scorer.py
=======================
Scoring logic for OrgForge Organizational Dynamics questions.

Drop this alongside agentic_eval_harness.py and import it.

Each category has its own answer scorer and trajectory scorer.
All scorers follow the same pattern as PerspectiveScorer / CounterfactualScorer:

  score_answer(final_answer, ground_truth) → (float, bool)
  score_trajectory(trajectory, question)   → OrgDynamicsTrajectoryScore

Answer scoring uses partial credit across structured components.
Trajectory scoring checks whether the agent retrieved the right artifacts
before answering — the evidence_search_space from the question defines
what "right" means.

Changes from v1:
- no_hallucination check is now real: flags agent-cited names not in GT or org chart
- correct_tools_used penalty for zero tool use is now 0.0 (was 0.3)
- Unified correct threshold of 0.55 across all categories
- OrgFriction root cause scoring uses token-overlap only on content words (len>4),
  with a note that semantic similarity would be more robust
- CAUSAL_PRESSURE downstream_signals linkage tightened (see question builder)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, Set, Tuple

logger = logging.getLogger("orgforge.org_dynamics_scorer")

# Shared org-chart actor allowlist — populated once at import time via
# _configure_known_actors().  Scorers will work without it (hallucination
# check is skipped if the set is empty) but calling _configure_known_actors()
# from eval harness setup gives the best signal.
_KNOWN_ACTORS: Set[str] = set()


def configure_known_actors(names: Set[str]) -> None:
    """Call once from harness setup with the full set of employee names."""
    global _KNOWN_ACTORS
    _KNOWN_ACTORS = {n.lower() for n in names}


# ─────────────────────────────────────────────────────────────────────────────
# TRAJECTORY SCORE DATACLASS
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class OrgDynamicsTrajectoryScore:
    search_coverage: float  # fraction of evidence_search_space the agent checked
    correct_tools_used: float  # did agent use appropriate tools for the category
    no_hallucination: float  # agent didn't fabricate actors/numbers not in corpus
    composite: float


# ─────────────────────────────────────────────────────────────────────────────
# ANSWER SCORERS BY CATEGORY
# ─────────────────────────────────────────────────────────────────────────────

# Unified passing threshold — all categories use this.
_CORRECT_THRESHOLD = 0.55


class AttentionCostScorer:
    """
    Scores ATTENTION_COST answers.

    Expected answer schema:
    {
        "actor": "string",
        "total_distraction_hours": float,
        "reasoning": "string"
    }
    OR for overhead questions:
    {
        "pct_non_ticket": float,
        "ticket_hours": float,
        "non_ticket_hours": float,
        "reasoning": "string"
    }
    """

    def score_answer(
        self, final_answer: Dict, ground_truth: Dict
    ) -> Tuple[float, bool]:
        if not final_answer:
            return 0.0, False

        score = 0.0

        # Actor identification (if applicable)
        gt_actor = ground_truth.get("actor", "")
        if gt_actor:
            agent_actor = str(final_answer.get("actor", "")).strip()
            if agent_actor.lower() == gt_actor.lower():
                score += 0.40
            elif gt_actor.lower() in agent_actor.lower():
                score += 0.20

        # Numeric accuracy — distraction hours or pct_non_ticket
        gt_hours = ground_truth.get("total_distraction_hours")
        gt_pct = ground_truth.get("pct_non_ticket")

        if gt_hours is not None:
            try:
                agent_hours = float(final_answer.get("total_distraction_hours", -1))
                if abs(agent_hours - gt_hours) <= 0.25:
                    score += 0.40
                elif abs(agent_hours - gt_hours) <= 0.75:
                    score += 0.20
                elif agent_hours > 0:
                    score += 0.10  # found something, wrong magnitude
            except (ValueError, TypeError):
                pass

        elif gt_pct is not None:
            try:
                agent_pct = float(final_answer.get("pct_non_ticket", -1))
                if abs(agent_pct - gt_pct) <= 5.0:
                    score += 0.40
                elif abs(agent_pct - gt_pct) <= 15.0:
                    score += 0.20
            except (ValueError, TypeError):
                pass

        # Day range awareness
        gt_range = ground_truth.get("day_range", [])
        agent_reasoning = str(final_answer.get("reasoning", "")).lower()
        if gt_range and str(gt_range[0]) in agent_reasoning:
            score += 0.20

        correct = score >= _CORRECT_THRESHOLD
        return round(min(score, 1.0), 4), correct

    def score_trajectory(
        self, trajectory, question: Dict
    ) -> OrgDynamicsTrajectoryScore:
        return _score_trajectory_generic(
            trajectory,
            question,
            required_tools={"get_events_for_day", "search_artifacts"},
        )


class ResourcePressureScorer:
    """
    Scores RESOURCE_PRESSURE answers.

    Expected answer schema:
    {
        "engineer": "string",
        "dept": "string",
        "overflow_hours": float,  OR  "cross_dept_event_count": int,
        "reasoning": "string"
    }
    """

    def score_answer(
        self, final_answer: Dict, ground_truth: Dict
    ) -> Tuple[float, bool]:
        if not final_answer:
            return 0.0, False

        score = 0.0

        gt_engineer = ground_truth.get("engineer", "")
        gt_actor = ground_truth.get("actor", gt_engineer)

        agent_engineer = str(
            final_answer.get("engineer", final_answer.get("actor", ""))
        ).strip()

        if gt_actor and agent_engineer.lower() == gt_actor.lower():
            score += 0.45
        elif gt_actor and gt_actor.lower() in agent_engineer.lower():
            score += 0.25

        # Dept identification
        gt_dept = ground_truth.get("dept", "")
        if gt_dept:
            agent_dept = str(final_answer.get("dept", "")).lower()
            if gt_dept.lower() in agent_dept or agent_dept in gt_dept.lower():
                score += 0.20

        # Numeric accuracy
        gt_overflow = ground_truth.get("overflow_hours")
        if gt_overflow is not None:
            try:
                agent_overflow = float(final_answer.get("overflow_hours", -1))
                if abs(agent_overflow - gt_overflow) <= 0.5:
                    score += 0.35
                elif abs(agent_overflow - gt_overflow) <= 1.5:
                    score += 0.15
            except (ValueError, TypeError):
                pass

        correct = score >= _CORRECT_THRESHOLD
        return round(min(score, 1.0), 4), correct

    def score_trajectory(
        self, trajectory, question: Dict
    ) -> OrgDynamicsTrajectoryScore:
        return _score_trajectory_generic(
            trajectory,
            question,
            required_tools={"get_events_for_day", "search_artifacts"},
        )


class CausalPressureScorer:
    """
    Scores CAUSAL_PRESSURE answers.

    Expected answer schema:
    {
        "trigger_source": "string",
        "downstream_depts": ["string"],
        "propagation_chain": ["artifact_id"],
        "theme_shifts": [{"dept": "string", "day": int, "theme": "string"}],
        "reasoning": "string"
    }
    """

    def score_answer(
        self, final_answer: Dict, ground_truth: Dict
    ) -> Tuple[float, bool]:
        if not final_answer:
            return 0.0, False

        score = 0.0

        # Source identification
        gt_source = str(ground_truth.get("source", "")).lower()
        gt_org = str(ground_truth.get("org", "")).lower()
        agent_reasoning = str(final_answer.get("reasoning", "")).lower()
        agent_source = str(final_answer.get("trigger_source", "")).lower()

        if gt_source and (gt_source in agent_source or gt_source in agent_reasoning):
            score += 0.25
        elif gt_org and gt_org in agent_reasoning:
            score += 0.15

        # Downstream dept identification
        gt_theme_shifts = ground_truth.get("theme_shifts", [])
        gt_depts = {t["dept"].lower() for t in gt_theme_shifts}
        agent_depts_raw = final_answer.get("downstream_depts", [])
        agent_depts = {str(d).lower() for d in agent_depts_raw}

        if gt_depts:
            overlap = len(gt_depts & agent_depts) / len(gt_depts)
            score += 0.35 * overlap

        # Propagation chain — did agent find downstream artifacts?
        gt_chain = set(ground_truth.get("propagation_chain", []))
        agent_chain = set(final_answer.get("propagation_chain", []))
        if gt_chain:
            chain_overlap = len(gt_chain & agent_chain) / len(gt_chain)
            score += 0.40 * chain_overlap

        correct = score >= _CORRECT_THRESHOLD
        return round(min(score, 1.0), 4), correct

    def score_trajectory(
        self, trajectory, question: Dict
    ) -> OrgDynamicsTrajectoryScore:
        return _score_trajectory_generic(
            trajectory,
            question,
            required_tools={"get_email", "search_artifacts", "get_events_for_day"},
        )


class AssignmentQualityScorer:
    """
    Scores ASSIGNMENT_QUALITY answers.

    Expected answer schema:
    {
        "ticket_id": "string",
        "assigned_engineer": "string",
        "assessment": "optimal | suboptimal | poor",
        "alternative_engineer": "string",  (optional)
        "reasoning": "string"
    }
    """

    def score_answer(
        self, final_answer: Dict, ground_truth: Dict
    ) -> Tuple[float, bool]:
        if not final_answer:
            return 0.0, False

        score = 0.0

        # Ticket identification
        gt_ticket = ground_truth.get("ticket_id", "")
        agent_ticket = str(final_answer.get("ticket_id", ""))
        if gt_ticket and agent_ticket == gt_ticket:
            score += 0.20

        # Engineer identification
        gt_engineer = ground_truth.get("assigned_engineer", "")
        agent_engineer = str(final_answer.get("assigned_engineer", "")).strip()
        if gt_engineer and agent_engineer.lower() == gt_engineer.lower():
            score += 0.25

        # Assessment accuracy
        gt_assessment = ground_truth.get("assessment", "")
        agent_assessment = str(final_answer.get("assessment", "")).lower()
        gt_is_poor = "poor" in gt_assessment or "mismatch" in gt_assessment
        agent_is_poor = "poor" in agent_assessment or "suboptimal" in agent_assessment

        if gt_is_poor == agent_is_poor:
            score += 0.35

        # Stress indicator awareness
        gt_stress = ground_truth.get("stress_indicator", "")
        agent_reasoning = str(final_answer.get("reasoning", "")).lower()
        if gt_stress and (gt_stress in agent_reasoning or "stress" in agent_reasoning):
            score += 0.20

        correct = score >= _CORRECT_THRESHOLD
        return round(min(score, 1.0), 4), correct

    def score_trajectory(
        self, trajectory, question: Dict
    ) -> OrgDynamicsTrajectoryScore:
        return _score_trajectory_generic(
            trajectory,
            question,
            required_tools={"get_ticket", "search_artifacts"},
        )


class OrgFrictionScorer:
    """
    Scores ORG_FRICTION answers.

    Expected answer schema:
    {
        "tension_level": "high | medium | low",
        "actors": ["string"],
        "depts_involved": ["string"],
        "root_cause": "string",
        "reasoning": "string"
    }

    Root cause scoring uses content-word overlap (words > 4 chars) between the
    ground truth rationale and the agent's root_cause + reasoning fields.
    This is a rough proxy — semantic similarity scoring would be more robust
    but requires an embedding call.  The overlap is capped at 1.0 and weighted
    generously (2× multiplier) because prose paraphrasing naturally reduces
    exact token overlap.
    """

    def score_answer(
        self, final_answer: Dict, ground_truth: Dict
    ) -> Tuple[float, bool]:
        if not final_answer:
            return 0.0, False

        score = 0.0

        # Tension level
        gt_tension = ground_truth.get("tension_level", "").lower()
        agent_tension = str(final_answer.get("tension_level", "")).lower()
        if gt_tension and agent_tension == gt_tension:
            score += 0.20
        elif gt_tension == "high" and agent_tension in ("high", "medium"):
            score += 0.10

        # Actor identification
        gt_actors = {a.lower() for a in ground_truth.get("actors", [])}
        agent_actors = {str(a).lower() for a in final_answer.get("actors", [])}
        if gt_actors:
            actor_overlap = len(gt_actors & agent_actors) / len(gt_actors)
            score += 0.30 * actor_overlap

        # Dept identification
        gt_depts = {d.lower() for d in ground_truth.get("depts_involved", [])}
        agent_depts = {str(d).lower() for d in final_answer.get("depts_involved", [])}
        if gt_depts:
            dept_overlap = len(gt_depts & agent_depts) / len(gt_depts)
            score += 0.25 * dept_overlap

        # Root cause / rationale — content-word overlap only (skip stop-words).
        # Words ≤ 4 chars are excluded to avoid matching "this", "that", "with" etc.
        gt_rationale = ground_truth.get("rationale", "").lower()
        agent_cause = str(final_answer.get("root_cause", "")).lower()
        agent_reasoning = str(final_answer.get("reasoning", "")).lower()

        if gt_rationale:
            gt_words = set(re.findall(r"[a-z]+", gt_rationale))
            agent_words = set(
                re.findall(r"[a-z]+", agent_cause + " " + agent_reasoning)
            )
            content_words = {w for w in gt_words if len(w) > 4}
            if content_words:
                word_overlap = len(content_words & agent_words) / len(content_words)
                score += 0.25 * min(word_overlap * 2, 1.0)

        correct = score >= _CORRECT_THRESHOLD
        return round(min(score, 1.0), 4), correct

    def score_trajectory(
        self, trajectory, question: Dict
    ) -> OrgDynamicsTrajectoryScore:
        return _score_trajectory_generic(
            trajectory,
            question,
            required_tools={
                "get_slack_thread",
                "get_events_for_day",
                "search_artifacts",
            },
        )


# ─────────────────────────────────────────────────────────────────────────────
# SHARED TRAJECTORY SCORER
# ─────────────────────────────────────────────────────────────────────────────


def _score_trajectory_generic(
    trajectory,
    question: Dict,
    required_tools: set,
) -> OrgDynamicsTrajectoryScore:
    """
    Generic trajectory scorer used by all ORG_DYNAMICS categories.

    Three components:
    1. search_coverage    — fraction of evidence_search_space the agent retrieved
    2. correct_tools_used — 1.0 if at least one required tool was used, 0.0 otherwise
                            (was 0.3 for zero-tool case in v1 — fixed)
    3. no_hallucination   — penalises agent for citing names that are not in the
                            ground truth actor list and not in the known org roster.
                            Returns 1.0 when _KNOWN_ACTORS is not configured (safe
                            default) or when no suspicious names are found.
    """
    calls = trajectory.tool_calls or []

    if not calls:
        return OrgDynamicsTrajectoryScore(
            search_coverage=0.0,
            correct_tools_used=0.0,  # no tools → no credit (was 0.3)
            no_hallucination=1.0,  # no claims → no hallucinations
            composite=0.0,
        )

    # ── 1. Search coverage ───────────────────────────────────────────────────
    expected_space = set(question.get("evidence_search_space", []))
    retrieved_ids: set = set()
    for call in calls:
        retrieved_ids.update(call.result_ids or [])
        for arg_val in (call.arguments or {}).values():
            if isinstance(arg_val, str) and len(arg_val) > 3:
                retrieved_ids.add(arg_val)

    coverage = (
        len(expected_space & retrieved_ids) / len(expected_space)
        if expected_space
        else (1.0 if calls else 0.0)
    )

    # ── 2. Correct tools used ────────────────────────────────────────────────
    used_tools = {c.tool_name for c in calls}
    tools_score = 1.0 if required_tools & used_tools else 0.0  # was 0.3

    # ── 3. Hallucination check ───────────────────────────────────────────────
    # Collect the ground-truth actor allowlist for this question.
    gt = question.get("ground_truth", {})
    gt_actors: Set[str] = set()
    for field in ("actors", "depts_involved", "trigger_actors"):
        for name in gt.get(field, []):
            gt_actors.add(str(name).lower())
    for field in ("actor", "engineer", "assigned_engineer"):
        val = gt.get(field, "")
        if val:
            gt_actors.add(str(val).lower())

    hallucination_score = 1.0  # default: clean

    if _KNOWN_ACTORS and gt_actors:
        # Extract all capitalised tokens from the agent's final answer that look
        # like proper nouns (first-letter uppercase, ≥ 2 chars).  If any appear
        # in the corpus-wide actor list but NOT in the GT allowlist for this
        # question, they are suspicious but not conclusive (e.g. a mentioned
        # bystander).  We only penalise if they are wholly absent from the known
        # org roster — those are genuine fabrications.
        final_str = str(trajectory.final_answer or "")
        cited_names = {
            tok.lower() for tok in re.findall(r"\b[A-Z][a-z]{1,}\b", final_str)
        }
        # Remove names that are legitimately in the ground truth or org
        suspicious = cited_names - gt_actors - _KNOWN_ACTORS
        if suspicious:
            # Each fabricated name deducts 0.15, floor 0.0
            penalty = min(len(suspicious) * 0.15, 1.0)
            hallucination_score = round(max(0.0, 1.0 - penalty), 4)
            logger.debug(
                f"[org_dynamics] Hallucination flag: suspicious names={suspicious} "
                f"→ score {hallucination_score}"
            )

    # ── Composite ────────────────────────────────────────────────────────────
    composite = round(
        0.50 * coverage + 0.30 * tools_score + 0.20 * hallucination_score,
        4,
    )

    return OrgDynamicsTrajectoryScore(
        search_coverage=round(coverage, 4),
        correct_tools_used=tools_score,
        no_hallucination=hallucination_score,
        composite=composite,
    )


# ─────────────────────────────────────────────────────────────────────────────
# DISPATCHER — maps category → scorer instance
# ─────────────────────────────────────────────────────────────────────────────

_SCORERS = {
    "ATTENTION_COST": AttentionCostScorer(),
    "RESOURCE_PRESSURE": ResourcePressureScorer(),
    "CAUSAL_PRESSURE": CausalPressureScorer(),
    "ASSIGNMENT_QUALITY": AssignmentQualityScorer(),
    "ORG_FRICTION": OrgFrictionScorer(),
}


def get_scorer(category: str):
    """Returns the appropriate scorer for a given category."""
    scorer = _SCORERS.get(category)
    if not scorer:
        raise ValueError(f"Unknown ORG_DYNAMICS category: {category}")
    return scorer
