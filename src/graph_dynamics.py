"""
graph_dynamics.py
=================
Drop-in NetworkX enhancement layer for OrgForge.

Three capabilities:
  1. Influence / Burnout Propagation  (betweenness centrality + stress bleed)
  2. Temporal Edge-Weight Decay & Reinforcement  (interaction boosts + daily decay)
  3. Shortest-Path Escalation  (Dijkstra on inverse-weight graph)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import networkx as nx


# ── Defaults (override via config["graph_dynamics"]) ─────────────────────────

DEFAULT_CFG = {
    # Propagation
    "stress_bleed_rate":       0.25,   # fraction of key-player excess that bleeds
    "burnout_threshold":       65,     # stress score that triggers propagation
    "incident_stress_hit":     20,     # raw stress added per P1 involvement
    "stress_daily_recovery":   3,      # flat recovery applied to everyone EOD
    "key_player_multiplier":   2.0,    # top-N% by betweenness = key players
    # Edge dynamics
    "edge_decay_rate":         0.97,   # multiplicative daily decay
    "slack_boost":             1.5,    # weight added per shared Slack thread
    "pr_review_boost":         3.0,    # weight added per PR review pair
    "incident_boost":          4.0,    # weight added per shared incident
    "edge_weight_floor":       0.5,    # decay never goes below this
    # Escalation
    "escalation_max_hops":     6,
}


# ── Return types ──────────────────────────────────────────────────────────────

@dataclass
class PropagationResult:
    affected:        List[str]         # names whose stress changed due to bleed
    stress_snapshot: Dict[str, int]    # full {name: stress} map after tick
    burnt_out:       List[str]         # names at or above burnout_threshold
    key_players:     List[str]         # high-centrality nodes this tick


@dataclass
class EscalationChain:
    chain:        List[Tuple[str, str]]  # [(name, role_label), ...]
    path_length:  int
    reached_lead: bool
    raw_path:     List[str]              # raw nx path for logging


# ── Main class ────────────────────────────────────────────────────────────────

class GraphDynamics:
    """
    Wraps a NetworkX Graph and adds temporal, influence, and pathfinding
    behaviour. All mutations happen in-place on self.G so existing code
    that holds a reference to the graph sees live data automatically.
    """

    def __init__(self, G: nx.Graph, config: dict):
        self.G = G
        self.cfg: Dict = {**DEFAULT_CFG, **config.get("graph_dynamics", {})}

        personas = config.get("personas", {})
        self._stress: Dict[str, int] = {
            node: int(personas.get(node, {}).get("stress", 30))
            for node in self.G.nodes()
        }

        self._centrality_cache: Optional[Dict[str, float]] = None
        self._centrality_dirty: bool = True

        self._org_chart: Dict[str, List[str]] = config.get("org_chart", {})
        self._leads: Dict[str, str] = config.get("leads", {})

    # ── 1. STRESS / BURNOUT PROPAGATION ──────────────────────────────────────

    def apply_incident_stress(self, actors: List[str], hit: Optional[int] = None) -> None:
        """Apply a raw stress hit to each actor directly involved in a P1."""
        hit = hit if hit is not None else self.cfg["incident_stress_hit"]
        for name in actors:
            if name in self._stress:
                self._stress[name] = min(100, self._stress[name] + hit)

    def propagate_stress(self) -> PropagationResult:
        """
        One propagation tick. Call once per day, inside/after _end_of_day().

        Steps:
          1. Find key players (top percentile by betweenness centrality).
          2. Burnt-out key players bleed stress to neighbours, proportional
             to edge weight (close colleagues absorb more than distant ones).
          3. Apply flat daily recovery to everyone.
        """
        centrality  = self._get_centrality()
        scores      = sorted(centrality.values())
        median = scores[len(scores) // 2]
        cutoff = median * self.cfg.get("key_player_multiplier", 2.0)
        key_players = [n for n, c in centrality.items() if c >= cutoff]

        burn_thresh = self.cfg["burnout_threshold"]
        bleed_rate  = self.cfg["stress_bleed_rate"]
        affected: set = set()

        for kp in key_players:
            kp_stress = self._stress.get(kp, 0)
            if kp_stress < burn_thresh:
                continue
            excess      = kp_stress - burn_thresh
            neighbours  = list(self.G.neighbors(kp))
            total_w     = sum(self.G[kp][nb].get("weight", 1.0) for nb in neighbours)
            if total_w == 0:
                continue
            for nb in neighbours:
                w     = self.G[kp][nb].get("weight", 1.0)
                bleed = int(excess * bleed_rate * (w / total_w))
                if bleed > 0:
                    self._stress[nb] = min(100, self._stress.get(nb, 30) + bleed)
                    affected.add(nb)

        # Daily recovery
        recovery = self.cfg["stress_daily_recovery"]
        for name in self._stress:
            self._stress[name] = max(0, self._stress[name] - recovery)

        burnt_out = [n for n, s in self._stress.items() if s >= burn_thresh]
        return PropagationResult(
            affected=sorted(affected),
            stress_snapshot=dict(self._stress),
            burnt_out=burnt_out,
            key_players=key_players,
        )

    def stress_label(self, name: str) -> str:
        """Drop-in replacement for the static stress lookup in persona_backstory()."""
        s = self._stress.get(name, 30)
        if s < 35:  return "low"
        if s < 60:  return "moderate"
        if s < 80:  return "high"
        return "critically high"

    def stress_tone_hint(self, name: str) -> str:
        """One-sentence LLM directive that colours a person's Slack/email voice."""
        s = self._stress.get(name, 30)
        if s < 35:
            return f"{name} is in a good headspace today -- helpful and upbeat."
        if s < 60:
            return f"{name} is a little stretched but holding it together."
        if s < 80:
            return (f"{name} is visibly stressed -- terse messages, short replies, "
                    "occasionally snapping at teammates.")
        return (f"{name} is burnt out -- messages are clipped and passive-aggressive; "
                "they are running on fumes and feel unsupported.")

    # ── 2. TEMPORAL EDGE-WEIGHT DYNAMICS ─────────────────────────────────────

    def record_slack_interaction(self, participants: List[str]) -> None:
        """Boost edges for all pairs in a Slack thread. Call end of _handle_normal_day()."""
        self._boost_pairs(participants, self.cfg["slack_boost"])

    def record_pr_review(self, author: str, reviewers: List[str]) -> None:
        """Boost edges between PR author and reviewers. Call in GitSimulator.create_pr()."""
        self._boost_pairs([author] + reviewers, self.cfg["pr_review_boost"])

    def record_incident_collaboration(self, actors: List[str]) -> None:
        """Boost edges among incident co-responders. Call in _handle_incident()."""
        self._boost_pairs(actors, self.cfg["incident_boost"])

    def decay_edges(self) -> None:
        """
        Apply multiplicative daily decay, with a hard floor.
        Call once per day inside _end_of_day().
        """
        decay = self.cfg["edge_decay_rate"]
        floor = self.cfg["edge_weight_floor"]
        for u, v, data in self.G.edges(data=True):
            self.G[u][v]["weight"] = round(max(floor, data.get("weight", 1.0) * decay), 4)
        self._centrality_dirty = True

    def relationship_summary(self, top_n: int = 5) -> List[Tuple[str, str, float]]:
        """Top N strongest relationships. Add to simulation_snapshot.json."""
        edges = [(u, v, d["weight"]) for u, v, d in self.G.edges(data=True)
                 if d.get("weight", 0) > self.cfg["edge_weight_floor"]]
        return sorted(edges, key=lambda x: x[2], reverse=True)[:top_n]

    def estranged_pairs(self, threshold_multiplier: float = 1.2) -> List[Tuple[str, str, float]]:
        """Pairs whose weight has decayed to near the floor -- 'estranged teams'."""
        floor  = self.cfg["edge_weight_floor"]
        cutoff = floor * threshold_multiplier
        pairs  = [(u, v, d["weight"]) for u, v, d in self.G.edges(data=True)
                  if d.get("weight", floor) <= cutoff]
        return sorted(pairs, key=lambda x: x[2])

    # ── 3. SHORTEST-PATH ESCALATION ───────────────────────────────────────────

    def build_escalation_chain(
        self,
        first_responder: str,
        domain_keywords: Optional[List[str]] = None,
    ) -> EscalationChain:
        """
        Dijkstra on an inverse-weight graph from first_responder to the
        nearest Lead (or domain expert). Strong relationships are cheap
        to traverse, so escalation naturally flows through 'work besties'.

        Pass domain_keywords (e.g. a departed employee's known systems) to
        prefer a Lead with matching expertise as the target.
        """
        target = self._find_escalation_target(first_responder, domain_keywords)
        if target is None:
            return EscalationChain(
                chain=[(first_responder, self._role_label(first_responder))],
                path_length=0, reached_lead=False, raw_path=[first_responder],
            )

        cost_graph = nx.Graph()
        for u, v, data in self.G.edges(data=True):
            w = max(data.get("weight", 1.0), 0.01)
            cost_graph.add_edge(u, v, weight=1.0 / w)

        try:
            raw_path = nx.dijkstra_path(cost_graph, first_responder, target, weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            raw_path = [first_responder, target]

        raw_path     = raw_path[: self.cfg["escalation_max_hops"] + 1]
        chain        = [(n, self._role_label(n)) for n in raw_path]
        reached_lead = any(n in self._leads.values() for n in raw_path[1:])

        return EscalationChain(
            chain=chain, path_length=len(raw_path) - 1,
            reached_lead=reached_lead, raw_path=raw_path,
        )

    def escalation_narrative(self, chain: EscalationChain) -> str:
        """
        Human-readable escalation path for LLM prompts and SimEvent summaries.
        Example: "Jordan (On-Call Engineer) -> Sam (Engineering Lead) -- 1 hop."
        """
        hops   = " -> ".join(f"{n} ({r})" for n, r in chain.chain)
        suffix = (f" -- {chain.path_length} hop(s) to reach leadership."
                  if chain.reached_lead else " -- escalation did not reach a Lead.")
        return hops + suffix

    def relevant_external_contacts(
        self,
        event_type: str,
        system_health: int,
        config: dict,
    ) -> List[dict]:
        """
        Returns external contact config entries that should be triggered
        given the current event type and system health.
        Called from _advance_incidents() to decide whether to generate
        an external contact summary.
        """
        triggered = []
        for contact in config.get("external_contacts", []):
            if event_type not in contact.get("trigger_events", []):
                continue
            threshold = contact.get("trigger_health_threshold", 100)
            if system_health <= threshold:
                triggered.append(contact)
        return triggered

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_centrality(self) -> Dict[str, float]:
        if self._centrality_dirty or self._centrality_cache is None:
            self._centrality_cache = nx.betweenness_centrality(
                self.G, weight="weight", normalized=True
            )
            self._centrality_dirty = False
        return self._centrality_cache

    def _boost_pairs(self, people: List[str], boost: float) -> None:
        seen: set = set()
        for i, a in enumerate(people):
            for b in people[i + 1:]:
                key = (min(a, b), max(a, b))
                if key in seen:
                    continue
                seen.add(key)
                if self.G.has_edge(a, b):
                    self.G[a][b]["weight"] = round(
                        self.G[a][b].get("weight", 1.0) + boost, 4
                    )
                    self._centrality_dirty = True

    def _find_escalation_target(
        self, source: str, domain_keywords: Optional[List[str]] = None
    ) -> Optional[str]:
        leads_set = set(self._leads.values())
        if domain_keywords:
            expert_leads = [
                n for n in leads_set
                if n != source and self.G.has_node(n)
                and any(kw.lower() in n.lower() for kw in domain_keywords)
            ]
            if expert_leads:
                return expert_leads[0]
        other_leads = [n for n in leads_set if n != source and self.G.has_node(n)]
        if other_leads:
            return max(other_leads,
                       key=lambda lead: self.G[source][lead].get("weight", 0.0)
                       if self.G.has_edge(source, lead) else 0.0)
        centrality = self._get_centrality()
        ranked = sorted([(n, c) for n, c in centrality.items() if n != source],
                        key=lambda x: x[1], reverse=True)
        return ranked[0][0] if ranked else None

    def _role_label(self, name: str) -> str:
        if name in self._leads.values():
            dept = next((d for d, lead in self._leads.items() if lead == name), "")
            return f"{dept} Lead" if dept else "Lead"
        dept = next((d for d, members in self._org_chart.items() if name in members), "")
        return f"{dept} Engineer" if dept else "Engineer"