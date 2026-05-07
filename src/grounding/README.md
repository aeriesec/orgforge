# Real-World Grounding Layer

This module adds **real-world structural grounding** to OrgForge's content
rendering, without changing the SimEvent state machine.

## Where this fits

OrgForge already produces deterministic events (`SimEvent`) and uses an LLM to
render those events into Slack messages, emails, Confluence pages, etc.

This grounding layer sits **between the event and the renderer**. It does not
invent facts. It only changes the *shape and texture* of the rendered prose so
the output statistically resembles real-world workplace communication
(realistic length distributions, vagueness rate, off-topic chatter,
unresolved-thread frequency, threading topology, etc.) instead of the smoother,
"LLM-cleaned" feel of pure-synthetic generation.

## Pipeline

```
Event from velomind.yaml (or any company config)
    ↓
1. event_context.py        : extract {component, symptom, actors, channel,
                                       arc_state} from the event
    ↓
2. genre_taxonomy.yaml     : map event → genre (e.g.
                                       hardware_firmware_incident_thread)
    ↓
3. profile lookup          : profiles/<genre>.yaml — cached one-time profile
                                       (length distribution, threading shape,
                                       vagueness rate, dialog-act sequence)
    ↓
4. fetcher/                : event-specific topical fetch via Tier-1 API,
                                       Tier-2 community archive, or Tier-3
                                       disciplined scrape
    ↓
5. prompt_injector.py      : extends OrgForge's existing renderer prompt with
                                       genre profile + topical real examples
    ↓
OrgForge's existing renderer LLM emits prose that respects both
the SimEvent facts and the grounded structural shape.
```

## Stage 1 vs Stage 2

- **Stage 1 (one-time per genre):** profile_extractor.py pulls N real artifacts
  for a genre, computes the pattern profile, caches to profiles/<genre>.yaml.
  Run once per company config.
- **Stage 2 (per individual event):** for each event during simulation, run a
  small topical fetch (0–5 examples matching this event's specific
  component/symptom). Genre profile is already cached.

## Source tiers

- **Tier 1 — Public-domain government APIs:** NHTSA, CPSC, SEC EDGAR, FCC, EPA.
  Cleanest legal posture, structured APIs. Used first.
- **Tier 2 — Open-license community archives:** ROS Discourse, ConvoKit (Reddit
  mirror, AfD, FOMC), IRC-Disentanglement.
- **Tier 3 — Disciplined scrapers:** Endless Sphere, Adafruit, BikeForums.
  robots.txt + 2s throttle + identified User-Agent + per-fetch caching.

## Output coupling to OrgForge

This module is non-invasive. Nothing in OrgForge's SimEvent bus, day_planner,
or normal_day handlers needs to change conceptually. The single integration
point is `prompt_injector.inject(prompt, event_ctx)` — called immediately
before the worker LLM is invoked.

When grounding is disabled (env flag), the injector is a no-op and OrgForge
renders pure-synthetic output identical to its baseline behaviour. This makes
A/B comparison clean: same SimEvent log, same personas, same arc state, only
the prompt-injection layer toggles on/off.

## Files

- `event_context.py` — extract event context from agenda items / SimEvents.
- `genre_taxonomy.yaml` — event → genre mapping (per-company override allowed).
- `profile_extractor.py` — Stage 1 LLM-driven profile builder.
- `prompt_injector.py` — Stage 2 runtime injector hook.
- `fetcher/` — Tier-1/2/3 dispatchers.
- `profiles/` — cached genre profile YAMLs (Stage 1 output).
- `_cache/` — raw artifact cache (gitignored, per-source subdirectories).

## Reuse for other companies

Swap `config/velomind.yaml` for any other company config. The genre taxonomy
includes per-industry overrides; the fetchers route based on the company's
industry to the right sources (e.g. fintech config routes to CFPB Consumer
Complaints + SEC enforcement actions instead of NHTSA).
