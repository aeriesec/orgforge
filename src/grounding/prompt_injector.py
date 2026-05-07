"""
Runtime hook called by OrgForge renderers immediately before the worker LLM
is invoked. Idempotent and side-effect-free when grounding is disabled.

Two public entry points:
  - augment_backstory(backstory: str, event_ctx) -> str
        Wraps a CrewAI Agent backstory with genre profile + topical examples.
        Use this in normal_day.py handlers right after `backstory = persona_utils.get_voice_card(...)`.
  - inject(prompt: str, event_ctx) -> str
        Generic prompt-string augmenter for any prompt-shaped string.

When ORGFORGE_GROUNDING_ENABLED is unset (default), both functions are
no-ops — the original string is returned unchanged. This guarantees the
baseline run is byte-identical to vanilla OrgForge.
"""
from __future__ import annotations

import logging
from typing import Optional

from . import GROUNDING_ENABLED
from .event_context import EventContext
from .profile_extractor import load_profile

logger = logging.getLogger("orgforge.grounding.injector")

_GENRE_LOOKUP_CACHE: dict[str, Optional[dict]] = {}


def _resolve_genre(event_ctx: EventContext) -> Optional[str]:
    """Resolve activity_type → genre name using the taxonomy. Lazy-imports the
    dispatcher so that disabling grounding skips all imports."""
    try:
        from .fetcher.dispatcher import FetcherDispatcher
    except Exception:
        return None
    disp = FetcherDispatcher()
    routing = (disp.taxonomy.get("activity_routing") or {})
    rule = routing.get(event_ctx.activity_type)
    if isinstance(rule, dict):
        # Component-aware routing: pick by_component when component matches a key
        if event_ctx.component:
            comp_lc = event_ctx.component.lower()
            for key, genre in (rule.get("by_component") or {}).items():
                if key in comp_lc:
                    return genre
        return rule.get("default")
    if isinstance(rule, str):
        return rule
    # Channel override
    chan_routing = (disp.taxonomy.get("channel_routing") or {})
    if event_ctx.channel_kind:
        for chan_name, genre in chan_routing.items():
            if chan_name in event_ctx.channel_kind:
                return genre
    return None


def _profile_for_genre(genre: str) -> Optional[dict]:
    if genre in _GENRE_LOOKUP_CACHE:
        return _GENRE_LOOKUP_CACHE[genre]
    p = load_profile(genre)
    _GENRE_LOOKUP_CACHE[genre] = p
    return p


def _render_grounding_block(genre: str, profile: dict) -> str:
    """Render a profile into a short rendering-hints block for the agent's
    backstory. Compact (under ~700 chars) so it doesn't dominate the prompt.

    Includes deterministic stats always; LLM-extracted patterns when present.
    """
    det = profile.get("deterministic", {}) or {}
    body_chars = det.get("body_length_chars", {}) or {}
    body_words = det.get("body_length_words", {}) or {}
    actor = det.get("actor_count", {}) or {}
    sources = ", ".join(det.get("sources", [])[:3])
    sample = det.get("sample_count", 0)

    lines = [
        "## REAL-WORLD STRUCTURAL GROUNDING (this genre)",
        f"- Genre: {genre}",
        f"- Reference corpus: {sample} real artifacts from {sources or 'mixed sources'}.",
        f"- Target body length: ~{body_words.get('mean', 0)} words "
        f"(median {body_words.get('median', 0)}, p95 {body_words.get('p95', 0)}).",
        f"- Typical actor count: {actor.get('mean', 0)} "
        f"(do not exceed {actor.get('max', 5)}).",
    ]

    llm_patterns = profile.get("llm_patterns")
    if llm_patterns and isinstance(llm_patterns, dict):
        tone = llm_patterns.get("tone_register")
        if tone:
            lines.append(f"- Tone register: {tone}.")
        vagueness = llm_patterns.get("vagueness_rate")
        if vagueness:
            lines.append(f"- Vagueness/hedging rate: {vagueness}.")
        markers = llm_patterns.get("vagueness_markers") or []
        if markers:
            lines.append(
                "- Sample hedging phrases that fit this genre: "
                + ", ".join(f'"{m}"' for m in markers[:5])
            )
        resolution = llm_patterns.get("resolution_shape")
        if resolution:
            lines.append(f"- Resolution shape: {resolution}.")
        hints = llm_patterns.get("rendering_hints") or []
        for h in hints[:5]:
            lines.append(f"- Hint: {h}")
        avoid = llm_patterns.get("avoid") or []
        for a in avoid[:3]:
            lines.append(f"- AVOID: {a}")
    else:
        # Without LLM-extracted patterns, give general structural guidance.
        lines.append(
            "- Match real-world realism: prefer plain factual sentences, "
            "explicit components/IDs/dates, occasional uncertainty rather "
            "than tidy executive summaries."
        )

    return "\n".join(lines)


def _build_topical_block(
    genre: str, event_ctx: EventContext, max_examples: int = 2
) -> str:
    """Stage 2: per-event topical example fetch. Cached at the dispatcher
    level. Falls back silently if no seeds match."""
    try:
        from .fetcher.dispatcher import FetcherDispatcher
    except Exception:
        return ""
    query_terms: list[str] = []
    if event_ctx.component:
        query_terms.append(event_ctx.component)
    if event_ctx.symptom:
        query_terms.append(event_ctx.symptom)
    if not query_terms:
        return ""
    hint = " ".join(query_terms)
    try:
        seeds = FetcherDispatcher().stage2_topical(genre, query_hint=hint, n=max_examples)
    except Exception as exc:
        logger.debug("[injector] topical fetch failed: %s", exc)
        return ""
    if not seeds:
        return ""
    out_lines = [
        "",
        "## REAL-WORLD TOPICAL EXAMPLES (style only — do not copy facts)",
    ]
    for i, s in enumerate(seeds, start=1):
        snippet = (s.body or "").replace("\n", " ").strip()
        snippet = snippet[:600] + ("…" if len(snippet) > 600 else "")
        out_lines.append(f"### Example {i} ({s.source}): {s.title[:100]}")
        out_lines.append(snippet)
    return "\n".join(out_lines)


def augment_backstory(backstory: str, event_ctx: Optional[EventContext]) -> str:
    """Wrap a CrewAI Agent backstory with grounding context. No-op if grounding
    disabled or context insufficient."""
    if not GROUNDING_ENABLED:
        return backstory
    if event_ctx is None:
        return backstory
    try:
        genre = _resolve_genre(event_ctx)
        if not genre:
            return backstory
        profile = _profile_for_genre(genre)
        if not profile:
            return backstory
        grounding_block = _render_grounding_block(genre, profile)
        topical_block = _build_topical_block(genre, event_ctx)
        return (
            backstory.rstrip()
            + "\n\n"
            + grounding_block
            + ("\n\n" + topical_block if topical_block else "")
            + "\n"
        )
    except Exception as exc:
        # Never crash the simulation because of grounding. Log + return original.
        logger.warning("[injector] augment_backstory failed (returning original): %s", exc)
        return backstory


def inject(prompt: str, event_ctx: Optional[EventContext]) -> str:
    """Generic alias of augment_backstory for non-CrewAI prompt chunks."""
    return augment_backstory(prompt, event_ctx)
