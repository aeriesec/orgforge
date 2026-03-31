"""
retrieval_extensions.py
=======================
Drop-in retrieval extensions for eval_e2e.py.

Provides two new Retriever subclasses that slot directly into
build_retriever() and the existing eval loop:

  RRFRetriever
  ------------
  Reciprocal Rank Fusion over any 2-N sub-retrievers.
  Fuses BM25 (lexical) with a dense retriever (Cohere / OpenAI / Bedrock)
  by default, producing a ranked list whose score is:

      RRF(d) = Σ_r  1 / (k + rank_r(d))

  where k=60 is the standard smoothing constant.

  Usage (eval_e2e.py CLI addition):
      python eval_e2e.py --retriever rrf --generator claude
      python eval_e2e.py --retriever rrf-openai --generator claude
      python eval_e2e.py --retriever rrf-bedrock --generator claude

  GraphAugmentedRetriever
  -----------------------
  Wraps any base Retriever and expands results by walking artifact
  relationship edges that are embedded in the corpus itself.

  The graph expander:
    1. Indexes all edges at index() time  →  O(|corpus|) build
    2. At retrieve() time, takes the base retriever's top-K results,
       adds 1-hop neighbors from the edge graph, re-ranks the combined
       pool by (base_score + neighbour_boost), and returns top-K.

  Neighbor boost decays with hop distance:
      boost(d, hop) = NEIGHBOUR_BOOST_BASE ** hop   (default: 0.5 per hop)

  Usage:
      python eval_e2e.py --retriever graph-bm25 --generator claude
      python eval_e2e.py --retriever graph-cohere --generator claude
      python eval_e2e.py --retriever graph-rrf --generator claude
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple


logger = logging.getLogger("orgforge.retrieval_extensions")

# ── Artifact-ID pattern: covers ORG-42, CONF-ENG-007, EMAIL-003,
#    SLACK-THREAD-9, PR-12, ZD-456, etc.
_ARTIFACT_ID_RE = re.compile(
    r"\b(?:ORG|CONF|EMAIL|SLACK(?:-THREAD)?|PR|ZD|SF|DD|JIRA)[-_][\w-]+",
    re.IGNORECASE,
)

# Fields in corpus docs that may carry related artifact IDs (JSON-encoded or plain list).
_RELATION_FIELDS = (
    "related_ids",
    "causal_chain",
    "artifact_ids",
    "evidence_chain",
    "downstream_artifacts",
    "linked_artifacts",
)

# RRF smoothing constant (Cormack et al. 2009 recommend k=60).
RRF_K: int = 60

# Graph expansion: score boost applied to 1-hop neighbors.
# Each additional hop multiplies by this factor  (geometric decay).
NEIGHBOUR_BOOST_BASE: float = 0.5

# Maximum graph hops to expand.  Keep at 1-2 to avoid noise amplification.
MAX_HOPS: int = 2


# ─────────────────────────────────────────────────────────────────────────────
# RECIPROCAL RANK FUSION
# ─────────────────────────────────────────────────────────────────────────────


class RRFRetriever:
    """
    Fuse ranked lists from two or more Retriever instances using
    Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, SIGIR 2009).

    RRF score for document d across ranker set R:

        rrf(d) = Σ_{r ∈ R}  1 / (k + rank_r(d))

    Documents not ranked by a given retriever are assigned rank = infinity
    (contributing 0 to the sum), which naturally deprioritises them without
    discarding them entirely.

    Parameters
    ----------
    retrievers : list of Retriever
        At least two Retriever instances.  All must be indexable with the same
        corpus.  Mixing BM25 + dense gives the best lexical/semantic coverage.
    k : int
        RRF smoothing constant.  Default 60 matches the canonical paper.
    candidate_k : int
        How many candidates each sub-retriever fetches before fusion.
        Should be ≥ final top_k; larger values improve recall at the cost of
        extra embedding lookups on dense retrievers.
    """

    name = "rrf"

    def __init__(
        self,
        retrievers: List,
        k: int = RRF_K,
        candidate_k: int = 50,
    ) -> None:
        if len(retrievers) < 2:
            raise ValueError("RRFRetriever requires at least two sub-retrievers.")
        self._retrievers = retrievers
        self._k = k
        self._candidate_k = candidate_k
        # Build a human-readable name from the sub-retriever names.
        sub_names = "+".join(r.name for r in retrievers)
        self.name = f"rrf({sub_names})"

    # ------------------------------------------------------------------
    # Retriever protocol
    # ------------------------------------------------------------------

    def index(self, corpus: List[dict]) -> None:
        """Index every sub-retriever with the same corpus."""
        for r in self._retrievers:
            logger.info(f"  [RRF] Indexing sub-retriever: {r.name}")
            r.index(corpus)
        logger.info(f"  [RRF] All {len(self._retrievers)} sub-retrievers indexed.")

    def retrieve(self, query: str, top_k: int = 10) -> List[str]:
        """
        Fetch candidates from each sub-retriever, apply RRF scoring,
        and return the top_k doc_ids ordered by descending RRF score.
        """
        candidate_k = max(self._candidate_k, top_k * 3)

        # Collect per-retriever ranked lists.
        ranked_lists: List[List[str]] = []
        for r in self._retrievers:
            try:
                ranked = r.retrieve(query, top_k=candidate_k)
            except Exception as exc:
                logger.warning(f"  [RRF] Sub-retriever {r.name} failed: {exc}")
                ranked = []
            ranked_lists.append(ranked)

        # Compute RRF scores.
        rrf_scores: Dict[str, float] = defaultdict(float)
        for ranked in ranked_lists:
            for rank, doc_id in enumerate(ranked, start=1):
                rrf_scores[doc_id] += 1.0 / (self._k + rank)

        # Sort by descending RRF score.
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in sorted_docs[:top_k]]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def per_retriever_ranks(
        self, query: str, candidate_k: int = 50
    ) -> Dict[str, Dict[str, int]]:
        """
        Diagnostic helper: returns {retriever_name: {doc_id: rank}} for a query.
        Useful for understanding which sub-retriever contributed each result.
        """
        out: Dict[str, Dict[str, int]] = {}
        for r in self._retrievers:
            ranked = r.retrieve(query, top_k=candidate_k)
            out[r.name] = {doc_id: rank + 1 for rank, doc_id in enumerate(ranked)}
        return out


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH-AUGMENTED RETRIEVAL
# ─────────────────────────────────────────────────────────────────────────────


class _ArtifactGraph:
    """
    Bidirectional adjacency list extracted from corpus metadata.

    Edges are collected from:
      1. Structured relation fields (RELATION_FIELDS) — parsed as JSON or
         plain lists of artifact ID strings.
      2. Artifact-ID tokens embedded in the body / title text.

    All edges are bidirectional: if doc A references doc B, we add both
    A → B and B → A so that graph traversal works in both directions.
    """

    def __init__(self) -> None:
        self._adj: Dict[str, Set[str]] = defaultdict(set)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, corpus: List[dict]) -> None:
        """Populate the adjacency list from the corpus."""
        doc_ids: Set[str] = {r["doc_id"] for r in corpus}

        for doc in corpus:
            src = doc["doc_id"]
            neighbors: Set[str] = set()

            # 1. Structured relation fields.
            for field in _RELATION_FIELDS:
                val = doc.get(field)
                if val is None:
                    continue
                # Might be a JSON-encoded string (e.g. stored as text in parquet).
                if isinstance(val, str):
                    try:
                        val = json.loads(val)
                    except (json.JSONDecodeError, ValueError):
                        # Try to extract IDs inline from the raw string.
                        neighbors.update(self._extract_ids_from_text(val, doc_ids))
                        continue
                # Might be a dict (artifact_ids maps type → id).
                if isinstance(val, dict):
                    for v in val.values():
                        if isinstance(v, str) and v in doc_ids:
                            neighbors.add(v)
                        elif isinstance(v, list):
                            neighbors.update(x for x in v if x in doc_ids)
                elif isinstance(val, list):
                    for item in val:
                        if isinstance(item, str) and item in doc_ids:
                            neighbors.add(item)

            # 2. Inline artifact-ID tokens in body / title.
            for text_field in ("body", "content", "title"):
                text = doc.get(text_field) or ""
                neighbors.update(self._extract_ids_from_text(text, doc_ids))

            # Remove self-loops.
            neighbors.discard(src)

            # Register bidirectional edges.
            for tgt in neighbors:
                self._adj[src].add(tgt)
                self._adj[tgt].add(src)

        total_edges = sum(len(v) for v in self._adj.values()) // 2
        logger.info(
            f"  [Graph] Artifact graph built: "
            f"{len(self._adj)} nodes, ~{total_edges} undirected edges"
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def neighbors(self, doc_id: str) -> Set[str]:
        """Return all direct neighbors of doc_id."""
        return set(self._adj.get(doc_id, set()))

    def expand(
        self,
        seed_ids: List[str],
        max_hops: int = MAX_HOPS,
    ) -> Dict[str, int]:
        """
        BFS from seed_ids up to max_hops away.

        Returns {doc_id: hop_distance} for every reachable node,
        excluding the seeds themselves (hop 0).
        """
        visited: Dict[str, int] = {}
        frontier: Set[str] = set(seed_ids)
        current_hop = 0

        while frontier and current_hop < max_hops:
            current_hop += 1
            next_frontier: Set[str] = set()
            for node in frontier:
                for nbr in self.neighbors(node):
                    if nbr not in visited and nbr not in set(seed_ids):
                        visited[nbr] = current_hop
                        next_frontier.add(nbr)
            frontier = next_frontier

        return visited

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_ids_from_text(text: str, doc_ids: Set[str]) -> Set[str]:
        """Extract all artifact-ID tokens from free text that exist in the corpus."""
        tokens = _ARTIFACT_ID_RE.findall(text)
        return {t.upper() for t in tokens if t.upper() in doc_ids}


class GraphAugmentedRetriever:
    """
    Wraps any base Retriever and expands its results by one or more hops
    along an artifact relationship graph built from corpus metadata.

    Algorithm
    ---------
    retrieve(query, top_k):
      1. Ask base retriever for `candidate_k` docs → seeded set S.
      2. Expand S up to `max_hops` hops in the artifact graph.
         Each hop-N neighbour receives a boost of:
             boost = base_score(nearest_seed) * NEIGHBOUR_BOOST_BASE^N
         where base_score is approximated as 1 / rank for the nearest
         seed in S that reaches this neighbour.
      3. Merge seed scores and neighbour boosts, normalise, return top_k.

    This is particularly valuable for OrgForge's CAUSAL and TEMPORAL
    question types, which require multi-artifact evidence chains that a
    single-document retriever may miss.

    Parameters
    ----------
    base_retriever : Retriever
        Any indexable retriever (BM25, Cohere, OpenAI, RRFRetriever, …).
    max_hops : int
        Graph expansion depth.  1–2 recommended; ≥3 adds noise.
    neighbour_boost_base : float
        Multiplicative decay per hop.  0.5 means 1-hop neighbors get 50%
        of the connecting seed's score, 2-hop get 25%, etc.
    candidate_k : int
        Seeds fetched from base retriever before graph expansion.
        Should be larger than final top_k so the graph has richer seeds.
    """

    def __init__(
        self,
        base_retriever,
        max_hops: int = MAX_HOPS,
        neighbour_boost_base: float = NEIGHBOUR_BOOST_BASE,
        candidate_k: int = 30,
    ) -> None:
        self._base = base_retriever
        self._max_hops = max_hops
        self._neighbour_boost_base = neighbour_boost_base
        self._candidate_k = candidate_k
        self._graph = _ArtifactGraph()
        self.name = f"graph({base_retriever.name},hops={max_hops})"

    # ------------------------------------------------------------------
    # Retriever protocol
    # ------------------------------------------------------------------

    def index(self, corpus: List[dict]) -> None:
        """Index the base retriever and build the artifact graph."""
        logger.info(f"  [Graph] Indexing base retriever: {self._base.name}")
        self._base.index(corpus)
        logger.info("  [Graph] Building artifact relationship graph …")
        self._graph.build(corpus)

    def retrieve(self, query: str, top_k: int = 10) -> List[str]:
        """
        Retrieve top_k documents by combining base retriever scores with
        graph-neighbour boost scores.
        """
        candidate_k = max(self._candidate_k, top_k * 2)

        # Step 1: seed retrieval.
        seeds: List[str] = self._base.retrieve(query, top_k=candidate_k)
        if not seeds:
            return []

        # Approximate base score as 1/(rank) — monotone proxy for relevance.
        seed_scores: Dict[str, float] = {
            doc_id: 1.0 / (rank + 1) for rank, doc_id in enumerate(seeds)
        }

        # Step 2: graph expansion.
        # For each neighbour, find its minimum hop distance across all seeds,
        # and use the highest-scoring seed that reaches it for the boost.
        neighbour_scores: Dict[str, float] = {}
        for seed_id, seed_score in seed_scores.items():
            reachable = self._graph.expand([seed_id], max_hops=self._max_hops)
            for nbr_id, hop in reachable.items():
                boost = seed_score * (self._neighbour_boost_base**hop)
                if boost > neighbour_scores.get(nbr_id, 0.0):
                    neighbour_scores[nbr_id] = boost

        # Step 3: merge seed + neighbour scores.
        combined: Dict[str, float] = dict(seed_scores)
        for doc_id, boost in neighbour_scores.items():
            if doc_id in combined:
                # Already in seed set — add boost on top.
                combined[doc_id] += boost
            else:
                combined[doc_id] = boost

        # Step 4: sort by descending combined score and return top_k.
        sorted_docs = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in sorted_docs[:top_k]]

    # ------------------------------------------------------------------
    # Diagnostic helpers
    # ------------------------------------------------------------------

    def explain(self, query: str, top_k: int = 10) -> List[dict]:
        """
        Returns a list of dicts with retrieval provenance for each result:
            {
              "doc_id": str,
              "combined_score": float,
              "from_base": bool,        # was it in the seed set?
              "hop_distance": int,      # 0 = seed, N = N hops away
              "base_rank": int | None,  # rank in base retriever (1-indexed)
            }
        Useful for offline debugging of graph expansion.
        """
        candidate_k = max(self._candidate_k, top_k * 2)
        seeds = self._base.retrieve(query, top_k=candidate_k)
        seed_scores = {doc_id: 1.0 / (rank + 1) for rank, doc_id in enumerate(seeds)}
        seed_rank = {doc_id: rank + 1 for rank, doc_id in enumerate(seeds)}

        neighbour_info: Dict[str, Tuple[float, int]] = {}  # doc_id → (boost, hop)
        for seed_id, seed_score in seed_scores.items():
            reachable = self._graph.expand([seed_id], max_hops=self._max_hops)
            for nbr_id, hop in reachable.items():
                boost = seed_score * (self._neighbour_boost_base**hop)
                if boost > neighbour_info.get(nbr_id, (0.0, 999))[0]:
                    neighbour_info[nbr_id] = (boost, hop)

        combined: Dict[str, float] = dict(seed_scores)
        for doc_id, (boost, _) in neighbour_info.items():
            combined[doc_id] = combined.get(doc_id, 0.0) + boost

        sorted_docs = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for doc_id, score in sorted_docs:
            hop = 0 if doc_id in seed_scores else neighbour_info.get(doc_id, (0, -1))[1]
            results.append(
                {
                    "doc_id": doc_id,
                    "combined_score": round(score, 6),
                    "from_base": doc_id in seed_scores,
                    "hop_distance": hop,
                    "base_rank": seed_rank.get(doc_id),
                }
            )
        return results
