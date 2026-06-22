"""
graph_eval_track.py
===================
TRACK 4 — GRAPH

Eval questions sourced from the social graph and stress system.
This module is designed to be imported by eval_harness.py and
agentic_eval_harness.py — it does not run standalone.

Ground truth sources (all from MongoDB `checkpoints` collection):
  - checkpoint["stress"]          → {name: int} stress score per actor per day
  - checkpoint["graph"]           → nx.node_link_data() — full edge-weight graph
  - checkpoint["state"]["morale"] → org morale float
  - checkpoint["state"]["health"] → system health int

Three question subtypes:

  GRAPH/BURNOUT_PROPAGATION
    "Given the stress scores and graph weights on Day N, which actor
     absorbed the most stress bleed from a burnt-out key player?"
    Ground truth: propagation_result derived from PropagationResult-equivalent
    logic run against the checkpoint graph. No LLM inference needed.

  GRAPH/CENTRALITY_SHIFT
    "Between Day A and Day B, which actor's betweenness centrality changed
     the most, and what event explains the shift?"
    Ground truth: delta = |centrality_B[actor] - centrality_A[actor]|, actor
    with max delta. Corroborating event from SimEvent log (departure, hire,
    incident that boosted edges).

  GRAPH/ESTRANGEMENT
    "Which pair of actors had the lowest relationship weight by Day N, and
     were they ever on the same incident?"
    Ground truth: estranged pair = argmin(edge weight), corroborated by
    checking incident_opened events for shared actors.

  GRAPH/ESCALATION_PATH
    "On Day N, if a P1 incident had been assigned to actor X, who would
     have been the next node in the Dijkstra escalation chain?"
    Ground truth: re-run Dijkstra on checkpoint graph, return second node
    in chain (the first hop from X to the nearest Lead).

Scoring weights (registered in agentic_eval_harness.py):
  answer      0.50  — the graph math is exact; a wrong answer is wrong
  trajectory  0.50  — agent must retrieve checkpoint/day data, not guess

Agent tool available for this track:
  get_graph_snapshot(day: int) → checkpoint["graph"] + checkpoint["stress"]
  This is added to GatedTools in agentic_eval_harness.py via the
  _register_graph_tool() hook below.
"""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("orgforge.eval.graph")

# ── burnout threshold mirrors graph_dynamics.py DEFAULT_CFG ──────────────────
_BURNOUT_THRESHOLD = 72
_STRESS_BLEED_RATE = 0.25
_KEY_PLAYER_MULTIPLIER = 2.0
_EDGE_WEIGHT_FLOOR = 0.5

_MAX_QUESTIONS_PER_SUBTYPE = 4


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class GraphSnapshot:
    """
    One checkpoint reconstructed as a queryable in-memory graph.
    We import networkx lazily so this module works even if nx is not
    installed at eval-generation time (the evaluator doesn't need it).
    """

    day: int
    stress: Dict[str, int]  # {name: score}
    edges: List[Tuple[str, str, float]]  # (u, v, weight)
    nodes: List[str]
    morale: float
    health: int

    @classmethod
    def from_checkpoint(cls, doc: dict) -> "GraphSnapshot":
        graph_data = doc.get("graph", {})
        stress = doc.get("stress", {})
        state = doc.get("state", {})

        nodes: List[str] = []
        for n in graph_data.get("nodes", []):
            nid = n.get("id") or n.get("name")
            if nid:
                nodes.append(str(nid))

        edges: List[Tuple[str, str, float]] = []
        for e in graph_data.get("links", []) or graph_data.get("edges", []):
            # node_link_data stores source/target as node-list indices or ids
            src = e.get("source")
            tgt = e.get("target")
            if src is None or tgt is None:
                continue
            # If stored as int indices, resolve to names
            if isinstance(src, int) and src < len(nodes):
                src = nodes[src]
            if isinstance(tgt, int) and tgt < len(nodes):
                tgt = nodes[tgt]
            w = float(e.get("weight", 1.0))
            edges.append((str(src), str(tgt), w))

        return cls(
            day=doc.get("day", 0),
            stress={str(k): int(v) for k, v in stress.items()},
            edges=edges,
            nodes=nodes,
            morale=float(state.get("morale", 0.5)),
            health=int(state.get("health", 80)),
        )

    def centrality(self) -> Dict[str, float]:
        """
        Betweenness centrality computed from stored edge weights.
        Falls back to degree-based approximation if networkx is unavailable.
        """
        try:
            import networkx as nx

            G = self._to_nx()
            return nx.betweenness_centrality(G, weight="weight", normalized=True)
        except ImportError:
            # Degree-based proxy: (sum of neighbour weights) / total weight
            neighbour_weight: Dict[str, float] = defaultdict(float)
            total_weight = 0.0
            for u, v, w in self.edges:
                neighbour_weight[u] += w
                neighbour_weight[v] += w
                total_weight += w
            if total_weight == 0:
                return {n: 0.0 for n in self.nodes}
            return {n: neighbour_weight[n] / total_weight for n in self.nodes}

    def dijkstra_next_hop(
        self,
        source: str,
        leads: List[str],
    ) -> Optional[str]:
        """
        Return the first hop from source toward the nearest lead.
        Returns None if no path exists.
        """
        try:
            import networkx as nx

            G = self._to_nx()
            best_hop: Optional[str] = None
            best_cost = float("inf")
            for lead in leads:
                if lead == source or lead not in G:
                    continue
                if source not in G:
                    continue
                try:
                    path = nx.dijkstra_path(
                        G,
                        source,
                        lead,
                        weight=lambda u, v, d: 1.0 / max(d.get("weight", 1.0), 0.01),
                    )
                    cost = nx.dijkstra_path_length(
                        G,
                        source,
                        lead,
                        weight=lambda u, v, d: 1.0 / max(d.get("weight", 1.0), 0.01),
                    )
                    if len(path) >= 2 and cost < best_cost:
                        best_cost = cost
                        best_hop = path[1]
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
            return best_hop
        except ImportError:
            # Fallback: return highest-weight neighbour of source that is a lead
            neighbour_weights = {v: w for u, v, w in self.edges if u == source}
            neighbour_weights.update({u: w for u, v, w in self.edges if v == source})
            lead_neighbours = {n: w for n, w in neighbour_weights.items() if n in leads}
            if lead_neighbours:
                return max(lead_neighbours, key=lead_neighbours.get)
            # No direct lead neighbour — return highest-weight neighbour
            if neighbour_weights:
                return max(neighbour_weights, key=neighbour_weights.get)
            return None

    def estranged_pairs(self, top_n: int = 5) -> List[Tuple[str, str, float]]:
        """Pairs with the lowest edge weight (decay near floor)."""
        sorted_edges = sorted(self.edges, key=lambda x: x[2])
        # Exclude self-loops and external nodes with no internal-only edges
        internal = [
            (u, v, w)
            for u, v, w in sorted_edges
            if u != v and u in self.stress and v in self.stress
        ]
        return internal[:top_n]

    def simulate_stress_propagation(
        self,
    ) -> Dict[str, Any]:
        """
        One tick of PropagationResult-equivalent logic derived from
        graph_dynamics.py propagate_stress(). Returns a dict with:
          key_players       — actors above the centrality cutoff
          burnt_out         — actors whose stress >= _BURNOUT_THRESHOLD
          bleed_targets     — {actor: stress_absorbed_from_key_players}
          max_bleed_target  — actor that absorbed the most stress
        """
        centrality = self.centrality()
        scores = sorted(centrality.values())
        if not scores:
            return {
                "key_players": [],
                "burnt_out": [],
                "bleed_targets": {},
                "max_bleed_target": None,
            }

        median = scores[len(scores) // 2]
        cutoff = median * _KEY_PLAYER_MULTIPLIER
        key_players = [n for n, c in centrality.items() if c >= cutoff]
        burnt_out = [n for n, s in self.stress.items() if s >= _BURNOUT_THRESHOLD]

        # Build adjacency for bleed computation
        adj: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        for u, v, w in self.edges:
            adj[u].append((v, w))
            adj[v].append((u, w))

        bleed_targets: Dict[str, float] = defaultdict(float)

        for kp in key_players:
            kp_stress = self.stress.get(kp, 0)
            if kp_stress < _BURNOUT_THRESHOLD:
                continue
            excess = kp_stress - _BURNOUT_THRESHOLD
            neighbours = adj[kp]
            total_w = sum(w for _, w in neighbours)
            if total_w == 0:
                continue
            for nb, w in neighbours:
                bleed = excess * _STRESS_BLEED_RATE * (w / total_w)
                if bleed > 0:
                    bleed_targets[nb] += bleed

        max_bleed_target = (
            max(bleed_targets, key=bleed_targets.get) if bleed_targets else None
        )

        return {
            "key_players": sorted(key_players),
            "burnt_out": sorted(burnt_out),
            "bleed_targets": dict(bleed_targets),
            "max_bleed_target": max_bleed_target,
            "max_bleed_amount": round(bleed_targets.get(max_bleed_target, 0), 2)
            if max_bleed_target
            else 0.0,
        }

    def _to_nx(self):
        import networkx as nx

        G = nx.Graph()
        G.add_nodes_from(self.nodes)
        for u, v, w in self.edges:
            G.add_edge(u, v, weight=w)
        return G

    def to_export_dict(self) -> dict:
        """Serializable form for export/eval/graph_snapshots.json."""
        return {
            "day": self.day,
            "stress": self.stress,
            "edges": [{"u": u, "v": v, "weight": w} for u, v, w in self.edges],
            "nodes": self.nodes,
            "morale": self.morale,
            "health": self.health,
        }


@dataclass
class GraphQuestion:
    question_id: str
    question_type: str  # always "GRAPH"
    graph_subtype: (
        str  # BURNOUT_PROPAGATION | CENTRALITY_SHIFT | ESTRANGEMENT | ESCALATION_PATH
    )
    difficulty: str  # "medium" | "hard"
    question_prose: str
    ground_truth: Dict[str, Any]
    expected_tool_calls: List[str]  # tools the agent must use
    as_of_day: int  # the day whose snapshot is the primary source
    secondary_day: Optional[int]  # for CENTRALITY_SHIFT questions
    actors_involved: List[str]
    corroborating_event_id: Optional[str]  # SimEvent mongo_id that explains the answer
    cross_subsystem: bool

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# SNAPSHOT LOADER
# ─────────────────────────────────────────────────────────────────────────────


class GraphSnapshotBuilder:
    """
    Reads all checkpoints from MongoDB and reconstructs GraphSnapshot objects.
    Snapshots are written to export/eval/graph_snapshots.json for agent retrieval.
    """

    def __init__(self, mem):
        self._mem = mem

    def build_all(self) -> List[GraphSnapshot]:
        checkpoints = list(self._mem._db["checkpoints"].find({}).sort("day", 1))
        if not checkpoints:
            logger.warning(
                "[graph_eval] No checkpoints found — GRAPH track will be empty."
            )
            return []

        snapshots = []
        for doc in checkpoints:
            if not doc.get("graph") or not doc.get("stress"):
                logger.debug(
                    f"[graph_eval] Checkpoint day={doc.get('day')} missing graph or stress — skipping."
                )
                continue
            try:
                snap = GraphSnapshot.from_checkpoint(doc)
                if snap.nodes:
                    snapshots.append(snap)
            except Exception as e:
                logger.warning(
                    f"[graph_eval] Failed to parse checkpoint day={doc.get('day')}: {e}"
                )

        logger.info(f"[graph_eval] Loaded {len(snapshots)} graph snapshots.")
        return snapshots


# ─────────────────────────────────────────────────────────────────────────────
# QUESTION GENERATORS (one per subtype)
# ─────────────────────────────────────────────────────────────────────────────


class GraphQuestionGenerator:
    """
    Generates GRAPH-track eval questions from checkpoints and the SimEvent log.
    All ground truth is derived deterministically — no LLM inference.

    The LLM is used only to write the prose question (same pattern as the
    three existing tracks in eval_harness.py).
    """

    def __init__(
        self,
        mem,
        snapshots: List[GraphSnapshot],
        leads: List[str],
        worker_llm,
    ):
        self._mem = mem
        self._snapshots = snapshots
        self._leads = leads
        self._worker_llm = worker_llm

        # Pre-load events once for corroboration lookups
        self._events = list(
            self._mem._db["events"]
            .find(
                {},
                {
                    "_id": 1,
                    "type": 1,
                    "day": 1,
                    "actors": 1,
                    "facts": 1,
                    "artifact_ids": 1,
                    "summary": 1,
                },
            )
            .sort("day", 1)
        )

    # ── public entry point ────────────────────────────────────────────────────

    def generate(self) -> List[GraphQuestion]:
        questions: List[GraphQuestion] = []

        questions.extend(self._gen_burnout_propagation())
        questions.extend(self._gen_centrality_shift())
        questions.extend(self._gen_estrangement())
        questions.extend(self._gen_escalation_path())

        logger.info(
            f"[graph_eval] Generated {len(questions)} GRAPH questions "
            f"({sum(1 for q in questions if q.difficulty == 'hard')} hard)."
        )
        return questions

    # ── BURNOUT_PROPAGATION ───────────────────────────────────────────────────

    def _gen_burnout_propagation(self) -> List[GraphQuestion]:
        """
        Find snapshots where at least one key player was burnt out and
        stress bleed was non-trivial. Ask which actor absorbed the most.
        """
        results: List[GraphQuestion] = []

        candidates = []
        for snap in self._snapshots:
            prop = snap.simulate_stress_propagation()
            if prop["max_bleed_target"] and prop["max_bleed_amount"] >= 2.0:
                candidates.append((snap, prop))

        random.shuffle(candidates)
        for snap, prop in candidates[:_MAX_QUESTIONS_PER_SUBTYPE]:
            max_target = prop["max_bleed_target"]
            max_amount = prop["max_bleed_amount"]
            key_players = prop["key_players"]
            burnt_out = prop["burnt_out"]

            if not key_players or not max_target:
                continue

            # Find the burnt-out key player that contributed the most bleed
            # (the one with the highest stress above the threshold)
            kp_stresses = {kp: snap.stress.get(kp, 0) for kp in key_players}
            primary_kp = (
                max(kp_stresses, key=kp_stresses.get) if kp_stresses else key_players[0]
            )

            # Corroborate with an incident or gap event involving the key player
            corroborating = self._find_corroborating_event(primary_kp, snap.day)

            qid = f"GRAPH-BURNOUT-day{snap.day}-{primary_kp.replace(' ', '_')}"
            prose = self._write_prose(
                subtype="BURNOUT_PROPAGATION",
                template=(
                    f"Write a natural-language question asking: "
                    f"On Day {snap.day}, given the stress levels and social graph, "
                    f"which team member absorbed the most stress from burnt-out key players? "
                    f"Do not name the answer ({max_target}) in the question. "
                    f"Reference the key player(s) by role or context, not name. "
                    f"The question must mention Day {snap.day} and reference stress propagation. "
                    f"End with a question mark. 15-100 words."
                ),
                ground_truth_str=max_target,
            )
            if not prose:
                continue

            results.append(
                GraphQuestion(
                    question_id=qid,
                    question_type="GRAPH",
                    graph_subtype="BURNOUT_PROPAGATION",
                    difficulty="hard" if len(key_players) > 1 else "medium",
                    question_prose=prose,
                    ground_truth={
                        "max_bleed_target": max_target,
                        "max_bleed_amount": max_amount,
                        "key_players": key_players,
                        "burnt_out": burnt_out,
                        "primary_source": primary_kp,
                        "stress_snapshot": {
                            k: v
                            for k, v in snap.stress.items()
                            if k in key_players + [max_target]
                        },
                    },
                    expected_tool_calls=["get_graph_snapshot", "get_stress_snapshot"],
                    as_of_day=snap.day,
                    secondary_day=None,
                    actors_involved=[primary_kp, max_target],
                    corroborating_event_id=corroborating,
                    cross_subsystem=False,
                )
            )

        return results

    # ── CENTRALITY_SHIFT ──────────────────────────────────────────────────────

    def _gen_centrality_shift(self) -> List[GraphQuestion]:
        """
        Find day-pairs where one actor's betweenness centrality changed
        significantly. Ask which actor and why.
        """
        results: List[GraphQuestion] = []

        if len(self._snapshots) < 2:
            return results

        # Compare consecutive snapshots
        shift_candidates = []
        for i in range(len(self._snapshots) - 1):
            snap_a = self._snapshots[i]
            snap_b = self._snapshots[i + 1]

            cent_a = snap_a.centrality()
            cent_b = snap_b.centrality()

            # Only consider actors present in both
            common = set(cent_a) & set(cent_b)
            if not common:
                continue

            deltas = {actor: abs(cent_b[actor] - cent_a[actor]) for actor in common}
            max_actor = max(deltas, key=deltas.get)
            max_delta = deltas[max_actor]

            if max_delta >= 0.05:  # meaningful shift
                shift_candidates.append(
                    (snap_a, snap_b, max_actor, max_delta, cent_a, cent_b)
                )

        random.shuffle(shift_candidates)
        for snap_a, snap_b, actor, delta, cent_a, cent_b in shift_candidates[
            :_MAX_QUESTIONS_PER_SUBTYPE
        ]:
            direction = "increased" if cent_b[actor] > cent_a[actor] else "decreased"

            # Find the event that explains the shift (departure, hire, incident)
            corroborating = self._find_centrality_event(actor, snap_a.day, snap_b.day)

            qid = f"GRAPH-CENTRALITY-day{snap_a.day}-{snap_b.day}-{actor.replace(' ', '_')}"
            prose = self._write_prose(
                subtype="CENTRALITY_SHIFT",
                template=(
                    f"Write a natural-language question asking: "
                    f"Between Day {snap_a.day} and Day {snap_b.day}, which team member "
                    f"experienced the largest change in their betweenness centrality "
                    f"in the collaboration graph, and what organisational event explains it? "
                    f"Do not name the answer ({actor}) or the direction ({direction}). "
                    f"Reference the day range and the concept of centrality or influence. "
                    f"End with a question mark. 15-100 words."
                ),
                ground_truth_str=actor,
            )
            if not prose:
                continue

            results.append(
                GraphQuestion(
                    question_id=qid,
                    question_type="GRAPH",
                    graph_subtype="CENTRALITY_SHIFT",
                    difficulty="hard",
                    question_prose=prose,
                    ground_truth={
                        "actor": actor,
                        "centrality_day_a": round(cent_a[actor], 4),
                        "centrality_day_b": round(cent_b[actor], 4),
                        "delta": round(delta, 4),
                        "direction": direction,
                        "day_a": snap_a.day,
                        "day_b": snap_b.day,
                    },
                    expected_tool_calls=["get_graph_snapshot", "get_events_for_day"],
                    as_of_day=snap_b.day,
                    secondary_day=snap_a.day,
                    actors_involved=[actor],
                    corroborating_event_id=corroborating,
                    cross_subsystem=True,
                )
            )

        return results

    # ── ESTRANGEMENT ──────────────────────────────────────────────────────────

    def _gen_estrangement(self) -> List[GraphQuestion]:
        """
        Find the most estranged internal pair on a given day.
        Ask whether they ever collaborated on the same incident.
        """
        results: List[GraphQuestion] = []

        # Use later snapshots where decay has had time to act
        late_snaps = self._snapshots[len(self._snapshots) // 2 :]
        if not late_snaps:
            late_snaps = self._snapshots

        random.shuffle(late_snaps)
        seen_pairs: Set[Tuple[str, str]] = set()

        for snap in late_snaps[: _MAX_QUESTIONS_PER_SUBTYPE * 2]:
            pairs = snap.estranged_pairs(top_n=3)
            if not pairs:
                continue

            for u, v, w in pairs:
                pair_key = (min(u, v), max(u, v))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                shared_incident = self._shared_incident(u, v, up_to_day=snap.day)

                qid = f"GRAPH-ESTRANGE-day{snap.day}-{u.replace(' ', '_')}-{v.replace(' ', '_')}"
                answer_str = f"{u} and {v}"
                prose = self._write_prose(
                    subtype="ESTRANGEMENT",
                    template=(
                        f"Write a natural-language question asking: "
                        f"By Day {snap.day}, which pair of team members had the "
                        f"weakest collaboration relationship based on their interaction "
                        f"history, and had they ever worked together on the same incident? "
                        f"Do not name the pair ({u} and {v}) in the question. "
                        f"Reference the day and the concept of relationship decay or "
                        f"collaboration frequency. End with a question mark. 15-100 words."
                    ),
                    ground_truth_str=answer_str,
                )
                if not prose:
                    continue

                results.append(
                    GraphQuestion(
                        question_id=qid,
                        question_type="GRAPH",
                        graph_subtype="ESTRANGEMENT",
                        difficulty="medium",
                        question_prose=prose,
                        ground_truth={
                            "estranged_pair": [u, v],
                            "edge_weight": round(w, 4),
                            "shared_incident": shared_incident,
                            "answer": answer_str,
                        },
                        expected_tool_calls=[
                            "get_graph_snapshot",
                            "get_events_for_day",
                            "get_ticket",
                        ],
                        as_of_day=snap.day,
                        secondary_day=None,
                        actors_involved=[u, v],
                        corroborating_event_id=None,
                        cross_subsystem=False,
                    )
                )

                if len(results) >= _MAX_QUESTIONS_PER_SUBTYPE:
                    return results

        return results

    # ── ESCALATION_PATH ───────────────────────────────────────────────────────

    def _gen_escalation_path(self) -> List[GraphQuestion]:
        """
        Hypothetical: on Day N, if actor X were the first responder,
        who would be the next hop in the Dijkstra escalation chain?
        """
        results: List[GraphQuestion] = []

        if not self._leads:
            logger.warning(
                "[graph_eval] No leads configured — skipping ESCALATION_PATH."
            )
            return results

        for snap in random.sample(
            self._snapshots, min(len(self._snapshots), _MAX_QUESTIONS_PER_SUBTYPE * 3)
        ):
            internal_actors = [
                n for n in snap.nodes if n in snap.stress and n not in self._leads
            ]
            if not internal_actors:
                continue

            actor = random.choice(internal_actors)
            next_hop = snap.dijkstra_next_hop(actor, self._leads)

            if not next_hop:
                continue

            # Find if there was an actual incident on this day involving actor
            corroborating = self._find_corroborating_event(actor, snap.day)

            qid = f"GRAPH-ESCALATION-day{snap.day}-{actor.replace(' ', '_')}"
            prose = self._write_prose(
                subtype="ESCALATION_PATH",
                template=(
                    f"Write a natural-language question asking: "
                    f"On Day {snap.day}, if {actor} had been the first responder "
                    f"to a P1 incident, who would have been the next person in their "
                    f"escalation chain based on the collaboration graph at that time? "
                    f"Do not name the answer ({next_hop}) in the question. "
                    f"Reference the day and the concept of escalation or incident response. "
                    f"End with a question mark. 15-100 words."
                ),
                ground_truth_str=next_hop,
            )
            if not prose:
                continue

            results.append(
                GraphQuestion(
                    question_id=qid,
                    question_type="GRAPH",
                    graph_subtype="ESCALATION_PATH",
                    difficulty="medium",
                    question_prose=prose,
                    ground_truth={
                        "first_responder": actor,
                        "next_hop": next_hop,
                        "leads": self._leads,
                        "day": snap.day,
                    },
                    expected_tool_calls=["get_graph_snapshot"],
                    as_of_day=snap.day,
                    secondary_day=None,
                    actors_involved=[actor, next_hop],
                    corroborating_event_id=corroborating,
                    cross_subsystem=False,
                )
            )

            if len(results) >= _MAX_QUESTIONS_PER_SUBTYPE:
                break

        return results

    # ── helpers ───────────────────────────────────────────────────────────────

    def _find_corroborating_event(self, actor: str, day: int) -> Optional[str]:
        """Find a SimEvent on or just before `day` involving `actor`."""
        _CORROBORATING_TYPES = {
            "incident_opened",
            "incident_resolved",
            "knowledge_gap_detected",
            "employee_departed",
            "employee_hired",
            "postmortem_created",
        }
        candidates = [
            e
            for e in self._events
            if actor in (e.get("actors") or [])
            and abs(e.get("day", 0) - day) <= 2
            and e.get("type") in _CORROBORATING_TYPES
        ]
        if candidates:
            # Prefer same-day, then nearest
            candidates.sort(key=lambda e: abs(e.get("day", 0) - day))
            return str(candidates[0].get("_id", ""))
        return None

    def _find_centrality_event(
        self, actor: str, day_a: int, day_b: int
    ) -> Optional[str]:
        """Find the org event between day_a and day_b that explains the centrality shift."""
        _CENTRALITY_TYPES = {
            "employee_departed",
            "employee_hired",
            "incident_opened",
            "centrality_vacuum",
            "knowledge_gap_detected",
        }
        candidates = [
            e
            for e in self._events
            if day_a <= e.get("day", 0) <= day_b
            and e.get("type") in _CENTRALITY_TYPES
            and (
                actor in (e.get("actors") or [])
                or actor == e.get("facts", {}).get("name")
                or actor == e.get("facts", {}).get("departed")
            )
        ]
        if candidates:
            return str(candidates[0].get("_id", ""))

        # Also accept any departure/hire in the window even if not directly involving the actor
        any_org_event = next(
            (
                e
                for e in self._events
                if day_a <= e.get("day", 0) <= day_b
                and e.get("type") in {"employee_departed", "employee_hired"}
            ),
            None,
        )
        return str(any_org_event["_id"]) if any_org_event else None

    def _shared_incident(self, u: str, v: str, up_to_day: int) -> Optional[str]:
        """Return the Jira ID of a shared incident, or None."""
        for e in self._events:
            if e.get("type") != "incident_opened":
                continue
            if e.get("day", 0) > up_to_day:
                continue
            actors = set(e.get("actors") or [])
            if u in actors and v in actors:
                return e.get("artifact_ids", {}).get("jira")
        return None

    def _write_prose(
        self, subtype: str, template: str, ground_truth_str: str
    ) -> Optional[str]:
        """
        Ask the worker LLM to write the question prose.
        Validates and retries up to 3 times (same pattern as eval_harness.py).
        """
        try:
            from agent_factory import make_agent
            from crewai import Crew, Task
        except ImportError:
            # If crewai is not available (e.g. unit test), return a template placeholder
            return f"[{subtype} question about {ground_truth_str} — prose generation unavailable]"

        agent = make_agent(
            role="Eval Dataset Author",
            goal="Write natural-sounding evaluation questions for AI agent benchmarks.",
            backstory=(
                "You write clear, specific questions for evaluating AI agents on reasoning tasks. "
                "Questions must reference the social graph, stress levels, or collaboration patterns "
                "of the simulated organisation. Questions must be unambiguous and answerable only "
                "through careful analysis of graph snapshots and event data."
            ),
            llm=self._worker_llm,
        )

        for attempt in range(3):
            retry_note = (
                " Previous attempt failed validation. Make sure the question: "
                "ends with '?', does not reveal the answer, and is 15-100 words long."
                if attempt > 0
                else ""
            )
            task = Task(
                description=template + retry_note,
                expected_output="One question ending with a question mark. No preamble or explanation.",
                agent=agent,
            )
            try:
                result = str(
                    Crew(agents=[agent], tasks=[task], verbose=False).kickoff()
                ).strip()
                if self._validate_prose(result, ground_truth_str):
                    return result
                logger.debug(
                    f"[graph_eval] Prose validation failed (attempt {attempt + 1}): {result[:80]}"
                )
            except Exception as exc:
                logger.warning(
                    f"[graph_eval] Prose generation error (attempt {attempt + 1}): {exc}"
                )

        return None

    def _validate_prose(self, text: str, ground_truth_str: str) -> bool:
        import re

        normalized = text.replace("\u2013", "-").replace("\u2014", "-")

        if not normalized.endswith("?"):
            return False

        words = normalized.split()
        if len(words) < 10 or len(words) > 150:
            return False

        gt_lower = ground_truth_str.lower()
        if gt_lower in normalized.lower() and len(gt_lower) > 4:
            return False

        if re.search(r"\b[A-Z]{1,4}-\d{2,6}\b", normalized):
            return False

        if not re.search(r"day\s+\d+", normalized, re.IGNORECASE):
            return False

        return True


def register_graph_tool(gated_tools_instance, mem, question: dict):
    """
    Monkey-patches a get_graph_snapshot() and get_stress_snapshot() method
    onto an existing GatedTools instance for GRAPH-track questions.

    Call this in AgentaticEvalHarness._run_question() when question_type == "GRAPH":

        from graph_eval_track import register_graph_tool
        register_graph_tool(tools, self._mem, question)

    The agent then has access to:
        tools.get_graph_snapshot(day: int) -> dict
        tools.get_stress_snapshot(day: int) -> dict
    """
    import time

    _mem = mem
    _as_of_day = question.get("as_of_day", 9999)

    def get_graph_snapshot(day: int) -> dict:
        t0 = time.time()
        horizon_violation = day > _as_of_day
        effective_day = min(day, _as_of_day)
        doc = _mem._db["checkpoints"].find_one(
            {"day": effective_day},
            {"_id": 0, "graph": 1, "stress": 1, "state": 1, "day": 1},
        )
        if not doc:
            nearest = _mem._db["checkpoints"].find_one(
                {"day": {"$lte": effective_day}},
                {"_id": 0, "graph": 1, "stress": 1, "state": 1, "day": 1},
                sort=[("day", -1)],
            )
            doc = nearest or {}

        g = doc.get("graph", {})
        node_list = [
            n.get("id") or n.get("name")
            for n in g.get("nodes", [])
            if n.get("id") or n.get("name")
        ]
        edge_list = [
            {
                "source": e.get("source"),
                "target": e.get("target"),
                "weight": round(float(e.get("weight", 1.0)), 4),
            }
            for e in (g.get("links") or g.get("edges", []))
        ]
        result = {
            "day": doc.get("day", effective_day),
            "nodes": node_list,
            "edges": edge_list,
            "stress": doc.get("stress", {}),
            "state": doc.get("state", {}),
        }
        gated_tools_instance._record(
            tool_name="get_graph_snapshot",
            arguments={"day": day},
            results=[result],
            t0=t0,
            horizon_violation=horizon_violation,
            timestamp_applied=str(effective_day),
        )
        return result

    def get_stress_snapshot(day: int) -> dict:
        t0 = time.time()
        horizon_violation = day > _as_of_day
        effective_day = min(day, _as_of_day)
        doc = _mem._db["checkpoints"].find_one(
            {"day": effective_day},
            {"_id": 0, "stress": 1, "day": 1},
        )
        if not doc:
            nearest = _mem._db["checkpoints"].find_one(
                {"day": {"$lte": effective_day}},
                {"_id": 0, "stress": 1, "day": 1},
                sort=[("day", -1)],
            )
            doc = nearest or {}

        result = {
            "day": doc.get("day", effective_day),
            "stress": doc.get("stress", {}),
        }
        gated_tools_instance._record(
            tool_name="get_stress_snapshot",
            arguments={"day": day},
            results=[result],
            t0=t0,
            horizon_violation=horizon_violation,
            timestamp_applied=str(effective_day),
        )
        return result

    import types

    gated_tools_instance.get_graph_snapshot = types.MethodType(
        lambda self, day: get_graph_snapshot(day), gated_tools_instance
    )
    gated_tools_instance.get_stress_snapshot = types.MethodType(
        lambda self, day: get_stress_snapshot(day), gated_tools_instance
    )


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH TRAJECTORY SCORER
# ─────────────────────────────────────────────────────────────────────────────


def score_graph_trajectory(
    trajectory,  # AgentTrajectory from agentic_eval_harness.py
    question: dict,
    ground_truth: dict,
) -> float:
    """
    Trajectory score for GRAPH questions (weight: 0.50).

    Rules:
    1. Agent must have called get_graph_snapshot (or get_stress_snapshot)
       at least once — otherwise score = 0.0 regardless of answer.
    2. Agent must have called it for the correct day (as_of_day ± 1).
    3. For CENTRALITY_SHIFT: agent must also query the secondary_day.
    4. Penalty for horizon violations (checking future checkpoints).

    Returns a float in [0.0, 1.0].
    """
    required_tools = set(question.get("expected_tool_calls", []))
    as_of_day = question.get("as_of_day", 0)
    secondary_day = question.get("secondary_day")

    tool_names_called = {tc.tool_name for tc in trajectory.tool_calls}
    graph_calls = [
        tc
        for tc in trajectory.tool_calls
        if tc.tool_name in ("get_graph_snapshot", "get_stress_snapshot")
    ]

    # Rule 1: must have called a graph tool
    if not graph_calls:
        return 0.0

    # Rule 2: must have queried the correct day
    def _day_from_args(tc) -> Optional[int]:
        return tc.arguments.get("day")

    days_queried = {
        _day_from_args(tc) for tc in graph_calls if _day_from_args(tc) is not None
    }
    correct_day_queried = any(abs(d - as_of_day) <= 1 for d in days_queried)

    if not correct_day_queried:
        return 0.2  # queried graph but wrong day

    score = 0.7

    # Rule 3: CENTRALITY_SHIFT requires both days
    if secondary_day is not None:
        secondary_queried = any(abs(d - secondary_day) <= 1 for d in days_queried)
        if secondary_queried:
            score += 0.15
        else:
            score -= 0.2

    # Horizon violations (future checkpoints queried)
    horizon_violations = sum(1 for tc in graph_calls if tc.horizon_violation)
    if horizon_violations:
        score -= 0.1 * min(horizon_violations, 3)

    # Bonus: called all expected tools
    if required_tools.issubset(tool_names_called):
        score += 0.15

    return round(max(0.0, min(1.0, score)), 4)


def score_graph_answer(
    agent_answer: dict,
    ground_truth: dict,
    graph_subtype: str,
) -> Tuple[float, bool]:
    """
    Answer score for GRAPH questions (weight: 0.50).
    Returns (score: float, correct: bool).

    Accepts fuzzy name matching (case-insensitive, partial name match)
    because agents often abbreviate names.
    """

    def _name_match(a: str, b: str) -> bool:
        a_tok = set(a.lower().split())
        b_tok = set(b.lower().split())
        if not a_tok or not b_tok:
            return False
        jaccard = len(a_tok & b_tok) / len(a_tok | b_tok)
        return jaccard >= 0.5

    extracted = agent_answer.get("answer", "") or agent_answer.get("actor", "") or ""

    if graph_subtype == "BURNOUT_PROPAGATION":
        gt_actor = ground_truth.get("max_bleed_target", "")
        if _name_match(str(extracted), gt_actor):
            return 1.0, True
        # Partial credit: named someone who was in bleed_targets at all
        bleed_targets = ground_truth.get("bleed_targets", {})
        if any(_name_match(str(extracted), n) for n in bleed_targets):
            return 0.4, False
        return 0.0, False

    elif graph_subtype == "CENTRALITY_SHIFT":
        gt_actor = ground_truth.get("actor", "")
        if _name_match(str(extracted), gt_actor):
            gt_dir = ground_truth.get("direction", "")
            agent_dir = agent_answer.get("direction", "")
            if gt_dir and agent_dir and gt_dir.lower() in agent_dir.lower():
                return 1.0, True
            return 0.8, True
        return 0.0, False

    elif graph_subtype == "ESTRANGEMENT":
        pair = ground_truth.get("estranged_pair", [])
        if not pair:
            return 0.0, False
        full_answer = json.dumps(agent_answer).lower()
        both_mentioned = all(p.lower() in full_answer for p in pair)
        if both_mentioned:
            gt_incident = ground_truth.get("shared_incident")
            agent_incident = agent_answer.get("shared_incident")
            if gt_incident is not None and agent_incident is not None:
                incident_match = str(gt_incident) in str(agent_incident)
                return (1.0 if incident_match else 0.7), True
            return 0.8, True
        return 0.0, False

    elif graph_subtype == "ESCALATION_PATH":
        gt_hop = ground_truth.get("next_hop", "")
        if _name_match(str(extracted), gt_hop):
            return 1.0, True
        return 0.0, False

    return 0.0, False


def build_graph_track(
    mem,
    worker_llm,
    leads: List[str],
    eval_dir: Path,
) -> Tuple[List[GraphSnapshot], List[dict]]:
    """
    Full pipeline: load snapshots → generate questions → write outputs.

    Returns (snapshots, questions_as_dicts) so EvalHarness can merge them
    into the main eval_questions.json.

    Writes two side-car files:
      eval_dir/graph_snapshots.json   — all snapshots (agent retrieval corpus)
      eval_dir/graph_questions.json   — GRAPH questions only
    """
    logger.info("[graph_eval] Building GRAPH track...")

    snapshot_builder = GraphSnapshotBuilder(mem)
    snapshots = snapshot_builder.build_all()

    if not snapshots:
        logger.warning("[graph_eval] No snapshots — GRAPH track skipped.")
        return [], []

    snap_path = eval_dir / "graph_snapshots.json"
    with open(snap_path, "w") as f:
        json.dump(
            [s.to_export_dict() for s in snapshots],
            f,
            indent=2,
            default=str,
        )
    logger.info(f"  → {snap_path} ({len(snapshots)} snapshots)")

    generator = GraphQuestionGenerator(
        mem=mem,
        snapshots=snapshots,
        leads=leads,
        worker_llm=worker_llm,
    )
    questions = generator.generate()

    q_dicts = [q.to_dict() for q in questions]
    graph_q_path = eval_dir / "graph_questions.json"
    with open(graph_q_path, "w") as f:
        json.dump(q_dicts, f, indent=2, default=str)
    logger.info(f"  → {graph_q_path} ({len(questions)} questions)")

    return snapshots, q_dicts
