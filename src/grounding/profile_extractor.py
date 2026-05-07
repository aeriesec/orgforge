"""
Stage 1 — extract a structural pattern profile per genre, one time.

Architecture:
  - compute_deterministic_stats(seeds): pure-Python length / actor / timestamp
    statistics. No LLM. Always runs.
  - extract_llm_patterns(seeds, llm): single LLM call summarising tone register,
    vagueness markers, dialog-act sequence pattern, resolution shape. Optional;
    skipped in --dry-run.
  - build_profile(genre, seeds, llm=None): combines both into a single dict.

Profiles are cached at profiles/<genre>.yaml.
"""
from __future__ import annotations

import json
import logging
import statistics
from pathlib import Path
from typing import Optional

import yaml

from .fetcher.seed import Seed

logger = logging.getLogger("orgforge.grounding.profile_extractor")

PROFILES_DIR = Path(__file__).parent / "profiles"


def load_profile(genre: str) -> Optional[dict]:
    p = PROFILES_DIR / f"{genre}.yaml"
    if not p.exists():
        return None
    return yaml.safe_load(p.read_text())


def save_profile(genre: str, profile: dict) -> Path:
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    p = PROFILES_DIR / f"{genre}.yaml"
    p.write_text(yaml.safe_dump(profile, sort_keys=False, allow_unicode=True))
    return p


def compute_deterministic_stats(seeds: list[Seed]) -> dict:
    """Pure-Python statistics over seed bodies. No LLM, no I/O."""
    if not seeds:
        return {"sample_count": 0}

    lengths = [len(s.body or "") for s in seeds]
    word_counts = [len((s.body or "").split()) for s in seeds]
    actor_counts = [len(s.actors) for s in seeds]
    title_lengths = [len(s.title or "") for s in seeds]

    def _percentile(xs: list[int], p: float) -> int:
        if not xs:
            return 0
        xs_sorted = sorted(xs)
        idx = int(round((len(xs_sorted) - 1) * p))
        return xs_sorted[idx]

    sources_seen = {s.source for s in seeds}
    return {
        "sample_count": len(seeds),
        "sources": sorted(sources_seen),
        "body_length_chars": {
            "mean": int(statistics.mean(lengths)) if lengths else 0,
            "median": int(statistics.median(lengths)) if lengths else 0,
            "p95": _percentile(lengths, 0.95),
            "max": max(lengths) if lengths else 0,
            "min": min(lengths) if lengths else 0,
        },
        "body_length_words": {
            "mean": int(statistics.mean(word_counts)) if word_counts else 0,
            "median": int(statistics.median(word_counts)) if word_counts else 0,
            "p95": _percentile(word_counts, 0.95),
        },
        "title_length_chars": {
            "mean": int(statistics.mean(title_lengths)) if title_lengths else 0,
            "median": int(statistics.median(title_lengths)) if title_lengths else 0,
        },
        "actor_count": {
            "mean": (
                round(statistics.mean(actor_counts), 2) if actor_counts else 0
            ),
            "max": max(actor_counts) if actor_counts else 0,
            "distribution_bins": {
                "1": sum(1 for c in actor_counts if c <= 1),
                "2-3": sum(1 for c in actor_counts if 2 <= c <= 3),
                "4-6": sum(1 for c in actor_counts if 4 <= c <= 6),
                "7+": sum(1 for c in actor_counts if c >= 7),
            },
        },
    }


_LLM_SYSTEM = (
    "You are a corpus structural analyst. You receive several real-world "
    "workplace artifacts (chat threads, emails, complaints, regulatory text). "
    "Your job is to extract structural patterns that an LLM rendering "
    "fictional artifacts of the same genre should imitate. "
    "You DO NOT summarize content. You characterise SHAPE: tone register, "
    "vagueness markers, dialog-act sequence, resolution shape, off-topic rate, "
    "threading style. Output strict JSON only."
)

_LLM_USER_TMPL = """\
Genre: {genre}

I will paste {n_artifacts} real-world artifacts of this genre below. Each is
delimited with === ARTIFACT k ===.

{artifacts}

Output a JSON object with exactly these keys:
  "tone_register":          one of ["formal", "semi-formal", "casual", "mixed"]
  "tone_examples":          list of 3-5 short tone snippets (verbatim or paraphrased) that exemplify the tone
  "vagueness_rate":         "rare" | "moderate" | "frequent" — how often hedges/uncertainty markers appear
  "vagueness_markers":      list of 5-10 example phrases ("let me check", "I think", "not sure", etc.)
  "dialog_act_sequence":    list[str] of typical acts in order, e.g. ["request_status", "give_partial_info", "disclose_blocker", "defer"]
  "resolution_shape":       one of ["resolved_clean", "resolved_partial", "trailing_off", "unresolved", "deferred"]
  "drift_rate":             "none" | "low" | "medium" | "high"  — frequency of off-topic chatter
  "threading_style":        one of ["linear", "branching", "quote_heavy", "mention_heavy", "broadcast"]
  "rendering_hints":        list of 3-7 short imperative tips for rendering THIS genre realistically
                            (e.g. "messages frequently end without resolution",
                             "speakers switch topics mid-thread",
                             "common abbreviations: TS, ECO, OTA")
  "avoid":                  list of 2-5 things a rendering LLM should NOT do (e.g. "do not produce a clean executive summary at the end")
"""


def _format_artifacts_for_prompt(seeds: list[Seed], char_budget: int = 18000) -> str:
    """Concatenate seed bodies with delimiters, capping total chars to keep
    the prompt under context-window limits."""
    parts: list[str] = []
    used = 0
    per_seed_budget = max(800, char_budget // max(1, len(seeds)))
    for i, s in enumerate(seeds, start=1):
        head = f"=== ARTIFACT {i} (source={s.source}, title={s.title[:80]}) ==="
        body = s.body or ""
        if len(body) > per_seed_budget:
            body = body[:per_seed_budget] + " ...[truncated]"
        block = head + "\n" + body
        if used + len(block) > char_budget:
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)


def extract_llm_patterns(
    genre: str,
    seeds: list[Seed],
    llm,
) -> dict:
    """Single LLM call. Returns the dict described in the schema above."""
    if not seeds:
        return {}
    artifacts_block = _format_artifacts_for_prompt(seeds)
    user_msg = _LLM_USER_TMPL.format(
        genre=genre,
        n_artifacts=len(seeds),
        artifacts=artifacts_block,
    )
    return llm.complete_json(system=_LLM_SYSTEM, user=user_msg, max_tokens=1500)


def build_profile(
    genre: str,
    seeds: list[Seed],
    llm=None,
    *,
    dry_run: bool = False,
) -> dict:
    """Build the cached profile for a genre. If `dry_run` or `llm is None`, the
    returned profile contains deterministic stats only and a placeholder
    `llm_patterns: null`. The renderer can still use it."""
    deterministic = compute_deterministic_stats(seeds)

    llm_patterns: Optional[dict] = None
    if seeds and llm is not None and not dry_run:
        try:
            llm_patterns = extract_llm_patterns(genre, seeds, llm)
        except Exception as exc:
            logger.warning("[profile_extractor] %s LLM extraction failed: %s", genre, exc)
            llm_patterns = None

    profile = {
        "genre": genre,
        "schema_version": 1,
        "seeds_used": [s.seed_id for s in seeds],
        "deterministic": deterministic,
        "llm_patterns": llm_patterns,
    }
    return profile
