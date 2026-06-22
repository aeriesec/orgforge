#!/usr/bin/env python3
"""
build_leaderboard.py
====================
Reads OrgForge agentic eval results from export/eval/ and produces:

  leaderboard/
    README.md          ← HuggingFace dataset card (paste into your HF repo)
    leaderboard.csv    ← Raw numbers for programmatic use / Space rendering
    per_model/
      <model>.json     ← Full merged stats per model (gated + zero-shot + ungated)

Usage
-----
  # From your project root (same level as export/):
  python build_leaderboard.py

  # Point at a different eval dir or output dir:
  python build_leaderboard.py --eval-dir path/to/eval --out-dir path/to/leaderboard

  # Only include specific models:
  python build_leaderboard.py --models claude-3-5-sonnet gpt-4o

  # Regenerate after adding new model runs without rewriting existing per_model jsons:
  python build_leaderboard.py --incremental
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


TRACKS = ["PERSPECTIVE", "COUNTERFACTUAL", "SILENCE"]


COLUMNS: List[tuple] = [
    ("rank", "Rank", "{}"),
    ("display_name", "Model", "{}"),
    ("model", "Model ID", "{}"),
    ("gated.overall.violation_adjusted_combined_score", "⭐ Score (gated)", "{:.3f}"),
    ("gated.overall.combined_score", "Combined", "{:.3f}"),
    ("gated.overall.answer_score", "Answer", "{:.3f}"),
    ("gated.overall.trajectory_score", "Trajectory", "{:.3f}"),
    ("gated.overall.accuracy", "Accuracy", "{:.1%}"),
    ("gated.overall.global_violation_rate", "Viol. Rate", "{:.1%}"),
    ("gated.overall.global_compliance_tier", "Compliance", "{}"),
    ("epistemic_tax", "Epistemic Tax", "{:.3f}"),
    ("zero_shot.overall.combined_score", "Zero-Shot", "{:.3f}"),
    ("gated.by_type.PERSPECTIVE.violation_adjusted_combined_score", "PERSP.", "{:.3f}"),
    ("gated.by_type.COUNTERFACTUAL.combined_score", "COUNTF.", "{:.3f}"),
    ("gated.by_type.SILENCE.combined_score", "SILENCE", "{:.3f}"),
    ("gated.overall.avg_tool_calls", "Avg Tools", "{:.1f}"),
    ("gated.overall.n", "N", "{}"),
]


MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "us.anthropic.claude-opus-4-6-v1": "Claude Opus 4.6",
    "us.anthropic.claude-sonnet-4-6": "Claude Sonnet 4.6",
    "claude-haiku-4-5": "Claude Haiku 4.5",
    "claude-opus-4": "Claude Opus 4",
    "claude-sonnet-4": "Claude Sonnet 4",
    "claude-3-7-sonnet": "Claude 3.7 Sonnet",
    "claude-3-5-sonnet-20241022": "Claude 3.5 Sonnet (Oct 24)",
    "claude-3-5-sonnet-20240620": "Claude 3.5 Sonnet (Jun 24)",
    "claude-3-5-sonnet": "Claude 3.5 Sonnet",
    "claude-3-5-haiku": "Claude 3.5 Haiku",
    "claude-3-opus": "Claude 3 Opus",
    "claude-3-sonnet": "Claude 3 Sonnet",
    "claude-3-haiku": "Claude 3 Haiku",
    "gpt-4o-mini": "GPT-4o Mini",
    "gpt-4o": "GPT-4o",
    "gpt-4-turbo": "GPT-4 Turbo",
    "gpt-4": "GPT-4",
    "o3-mini": "o3-mini",
    "o3": "o3",
    "o1-mini": "o1-mini",
    "o1": "o1",
    "deepseek.v3.2": "DeepSeek v3.2",
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "gemini-2.0-flash": "Gemini 2.0 Flash",
    "gemini-1.5-pro": "Gemini 1.5 Pro",
    "gemini-1.5-flash": "Gemini 1.5 Flash",
    "llama-3.3-70b": "Llama 3.3 70B",
    "llama-3.1-405b": "Llama 3.1 405B",
    "llama-3.1-70b": "Llama 3.1 70B",
    "llama-3.1-8b": "Llama 3.1 8B",
    "amazon.nova-pro": "Amazon Nova Pro",
    "amazon.nova-lite": "Amazon Nova Lite",
    "amazon.nova-micro": "Amazon Nova Micro",
    "mistral.mistral-large-3-675b-instruct": "Mistral Large 3",
    "mistral-small": "Mistral Small",
    "mixtral-8x7b": "Mixtral 8×7B",
    "command-r-plus": "Cohere Command R+",
    "moonshotai.kimi-k2.5": "Kimi 2.5",
    "qwen.qwen3-235b-a22b-2507-v1_0": "Qwen3 235B",
    "qwen.qwen3-235b-a22b-2507-v1/0": "Qwen3 235B",
}

_DISPLAY_NAME_KEYS = sorted(MODEL_DISPLAY_NAMES, key=len, reverse=True)


def _pretty_model_name(model_id: str) -> str:
    """Return a human-readable name for a model ID, or the ID itself if unknown."""
    lower = model_id.lower()
    for key in _DISPLAY_NAME_KEYS:
        if key.lower() in lower:
            return MODEL_DISPLAY_NAMES[key]
    return model_id


def _deep_get(d: Dict, dotpath: str, default: Any = None) -> Any:
    """Safely traverse a nested dict with a dot-separated path."""
    parts = dotpath.split(".")
    cur = d
    for p in parts:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(p, default)
        if cur is default:
            return default
    return cur


def _fmt(val: Any, fmt: str) -> str:
    if val is None:
        return "—"
    try:
        return fmt.format(val)
    except (ValueError, TypeError):
        return str(val)


def _stem_to_model(stem: str) -> str:
    """Reverse the filename sanitisation applied by the harness."""
    return stem.replace("_", "/").replace(":", ":")


def _discover_models(eval_dir: Path, only: Optional[List[str]] = None) -> List[str]:
    """Return sorted list of model stems that have at least a gated result file."""
    stems = set()
    for f in eval_dir.glob("gated_*.json"):
        stem = f.stem[len("gated_") :]
        stems.add(stem)
    if only:
        keep = set()
        for s in stems:
            model = _stem_to_model(s)
            if any(o in model or o in s for o in only):
                keep.add(s)
        stems = keep
    return sorted(stems)


def _load_run(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get("summary", data)


def _compute_epistemic_tax(
    gated: Optional[Dict], ungated: Optional[Dict]
) -> Optional[float]:
    """
    Epistemic Tax = ungated_combined - gated_combined.
    Positive value means the model pays a cost for being gated (normal).
    A negative value means gating somehow helped (unusual, flag it).
    """
    if not gated or not ungated:
        return None
    g = _deep_get(gated, "overall.combined_score")
    u = _deep_get(ungated, "overall.combined_score")
    if g is None or u is None:
        return None
    return round(u - g, 4)


def _merge_model(stem: str, eval_dir: Path) -> Dict:
    gated = _load_run(eval_dir / f"gated_{stem}.json")
    zero = _load_run(eval_dir / f"zero_shot_{stem}.json")
    ungated = _load_run(eval_dir / f"ungated_{stem}.json")

    raw_model_id = _stem_to_model(stem)
    merged: Dict[str, Any] = {
        "model": raw_model_id,
        "display_name": _pretty_model_name(raw_model_id),
        "stem": stem,
        "generated": datetime.now(timezone.utc).isoformat(),
        "gated": gated,
        "zero_shot": zero,
        "ungated": ungated,
        "epistemic_tax": _compute_epistemic_tax(gated, ungated),
    }

    merged["primary_score"] = (
        _deep_get(gated, "overall.violation_adjusted_combined_score") if gated else None
    )
    return merged


def _build_csv_rows(ranked: List[Dict]) -> List[Dict]:
    rows = []
    for i, m in enumerate(ranked, 1):
        row: Dict[str, Any] = {
            "rank": i,
            "model": m["model"],
            "display_name": m["display_name"],
        }
        for key, _label, _fmt in COLUMNS:
            if key in ("rank", "model", "display_name"):
                continue
            if key == "epistemic_tax":
                row[key] = m.get("epistemic_tax")
            else:
                row[key] = _deep_get(m, key)
        rows.append(row)
    return rows


def write_csv(ranked: List[Dict], out_path: Path) -> None:
    rows = _build_csv_rows(ranked)
    headers = [col[0] for col in COLUMNS]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  ✓ CSV  → {out_path}")


_COMPLIANCE_EMOJI = {
    "compliant": "🟢",
    "borderline": "🟡",
    "non_compliant": "🔴",
}


def _compliance_badge(tier: Optional[str]) -> str:
    if tier is None:
        return "—"
    return f"{_COMPLIANCE_EMOJI.get(tier, '')} {tier}"


def _md_table(ranked: List[Dict]) -> str:
    """Render the main leaderboard markdown table."""
    # Header row
    header_labels = [col[1] for col in COLUMNS]
    sep = ["---"] * len(header_labels)
    lines = [
        "| " + " | ".join(header_labels) + " |",
        "| " + " | ".join(sep) + " |",
    ]

    for i, m in enumerate(ranked, 1):
        cells = []
        for key, _label, fmt in COLUMNS:
            if key == "rank":
                cells.append(str(i))
            elif key == "display_name":
                cells.append(m.get("display_name") or m["model"])
            elif key == "model":
                cells.append(f"`{m['model']}`")
            elif key == "epistemic_tax":
                val = m.get("epistemic_tax")
                cells.append(_fmt(val, fmt))
            elif key == "gated.overall.global_compliance_tier":
                val = _deep_get(m, key)
                cells.append(_compliance_badge(val))
            else:
                val = _deep_get(m, key)
                cells.append(_fmt(val, fmt))
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def _per_track_table(ranked: List[Dict]) -> str:
    """Secondary table: per-track breakdown for gated run."""
    track_cols = [
        ("PERSPECTIVE", "violation_adjusted_combined_score", "Score"),
        ("PERSPECTIVE", "accuracy", "Acc"),
        ("PERSPECTIVE", "violation_rate", "Viol%"),
        ("COUNTERFACTUAL", "combined_score", "Score"),
        ("COUNTERFACTUAL", "accuracy", "Acc"),
        ("SILENCE", "combined_score", "Score"),
        ("SILENCE", "search_space_coverage", "Search%"),
    ]
    headers = ["Model"] + [f"{t[:5]}/{k}" for t, k, _ in track_cols]
    sep = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for m in ranked:
        row = [f"`{m['model']}`"]
        for track, field, _ in track_cols:
            val = _deep_get(m, f"gated.by_type.{track}.{field}")
            if val is None:
                row.append("—")
            elif isinstance(val, float):
                row.append(f"{val:.3f}")
            else:
                row.append(str(val))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def write_readme(ranked: List[Dict], out_path: Path, dataset_name: str) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    n_models = len(ranked)

    readme = f"""---
tags:
  - leaderboard
  - agentic-eval
  - orgforge
license: apache-2.0
pretty_name: {dataset_name}
size_categories:
  - n<1K
---

# {dataset_name}

> Last updated: **{now}** · {n_models} model(s) evaluated

## What is this?

This leaderboard benchmarks language models on the **OrgForge Agentic Evaluation
Harness** — a simulator-grounded benchmark for three distinct reasoning tracks:

| Track | What it tests | Answer weight | Trajectory weight |
|---|---|---|---|
| **PERSPECTIVE** | Epistemic discipline — stay within an actor's visibility cone | 40% | 60% |
| **COUNTERFACTUAL** | Causal tracing — identify mechanism and actors | 50% | 50% |
| **SILENCE** | Absence detection — prove something doesn't exist by searching | 30% | 70% |

Each model is evaluated under three conditions:

| Condition | File prefix | Description |
|---|---|---|
| **Gated** | `gated_` | Actor/subsystem gates enforced — normal operating conditions |
| **Ungated** | `ungated_` | God-mode corpus access — sets the *Epistemic Tax* ceiling |
| **Zero-Shot** | `zero_shot_` | No tools provided — establishes the hallucination floor |

**Epistemic Tax** = `ungated_combined − gated_combined`. The higher the tax, the
more a model is penalised by operating within realistic permission boundaries.

**Primary ranking metric**: `violation_adjusted_combined_score` on the gated run.
This is `combined_score × (1 − violation_rate)²`, so a model cannot buy ranking
through high answer accuracy while ignoring epistemic discipline.

---

## Leaderboard

{_md_table(ranked)}

### Per-track breakdown (gated run)

{_per_track_table(ranked)}

---

## Score definitions

| Column | Definition |
|---|---|
| ⭐ Score | `combined × (1 − violation_rate)²` — primary ranking axis |
| Combined | Weighted average of answer + trajectory per track weights |
| Answer | Correctness of the final answer |
| Trajectory | Quality of the tool-call path taken |
| Accuracy | Fraction of questions answered correctly (binary) |
| Viol. Rate | Actor-gate violations / total tool calls (PERSPECTIVE track) |
| Compliance | `compliant` < 5% · `borderline` < 20% · `non_compliant` ≥ 20% |
| Epistemic Tax | ungated_combined − gated_combined |
| Zero-Shot | combined_score with no tools — hallucination floor |
| Avg Tools | Mean tool calls per question |

---

## Files

| File | Description |
|---|---|
| `leaderboard.csv` | Machine-readable leaderboard with all metrics |
| `per_model/<model>.json` | Full merged stats per model |

---

## How to reproduce

```bash
# Gated (standard) run
python agentic_eval_harness.py --model <model_id>

# Zero-shot floor
python agentic_eval_harness.py --model <model_id> --zero-shot

# Ungated ceiling
python agentic_eval_harness.py --model <model_id> --ungated

# Build leaderboard from eval/ directory
python build_leaderboard.py
```

---

## Citation

```bibtex
@misc{{orgforge-agentic-eval,
  title  = {{{dataset_name}}},
  year   = {{{now[:4]}}},
  note   = {{OrgForge Agentic Evaluation Harness v2}},
  url    = {{https://huggingface.co/datasets/YOUR_ORG/{dataset_name.lower().replace(" ", "-")}}}
}}
```
"""
    out_path.write_text(readme, encoding="utf-8")
    print(f"  ✓ README → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build OrgForge leaderboard for HuggingFace"
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=Path("export/eval"),
        help="Directory containing gated_*.json / zero_shot_*.json / ungated_*.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("leaderboard"),
        help="Output directory (created if missing)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Filter to specific model name substrings",
    )
    parser.add_argument(
        "--dataset-name",
        default="OrgForge Agentic Eval Leaderboard",
        help="Pretty name shown in the HuggingFace dataset card",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Skip per_model/ JSON generation for models that already have one",
    )
    args = parser.parse_args()

    eval_dir: Path = args.eval_dir
    out_dir: Path = args.out_dir

    if not eval_dir.exists():
        raise SystemExit(f"eval-dir not found: {eval_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    per_model_dir = out_dir / "per_model"
    per_model_dir.mkdir(exist_ok=True)

    print(f"\n🔍  Scanning {eval_dir} …")
    stems = _discover_models(eval_dir, only=args.models)

    if not stems:
        raise SystemExit("No gated_*.json files found. Have you run the harness yet?")

    print(
        f"   Found {len(stems)} model(s): {', '.join(_stem_to_model(s) for s in stems)}\n"
    )

    merged_models: List[Dict] = []
    for stem in stems:
        per_model_path = per_model_dir / f"{stem}.json"
        if args.incremental and per_model_path.exists():
            print(f"  ↩  {stem} — loading cached per_model JSON")
            with open(per_model_path) as f:
                m = json.load(f)
        else:
            print(f"  ⚙  {stem} — merging gated / zero_shot / ungated …")
            m = _merge_model(stem, eval_dir)
            with open(per_model_path, "w") as f:
                json.dump(m, f, indent=2)
            print(f"     saved → {per_model_path}")
        merged_models.append(m)

    def _sort_key(m: Dict):
        s = m.get("primary_score")
        return (-s if s is not None else float("inf"), m["model"])

    ranked = sorted(merged_models, key=_sort_key)

    print(f"\n📊  Writing leaderboard to {out_dir} …")
    write_csv(ranked, out_dir / "leaderboard.csv")
    write_readme(ranked, out_dir / "README.md", args.dataset_name)

    print("""
✅  Done!

Next steps to publish on HuggingFace
─────────────────────────────────────
1. Create a new Dataset repo on huggingface.co
   (Datasets → New Dataset → set it to Public or Private)

2. Push the leaderboard/ folder:

     pip install huggingface_hub
     huggingface-cli upload YOUR_ORG/YOUR_REPO leaderboard/ .

   Or with the Python SDK:

     from huggingface_hub import HfApi
     api = HfApi()
     api.upload_folder(
         folder_path="leaderboard",
         repo_id="YOUR_ORG/YOUR_REPO",
         repo_type="dataset",
     )

3. (Optional) Create a Gradio Space that reads leaderboard.csv for
   an interactive table — the standard HF leaderboard template works
   out of the box with the CSV columns this script generates.

   Template: https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard
""")


if __name__ == "__main__":
    main()
