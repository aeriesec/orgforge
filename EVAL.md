# 🔬 Evaluating Epistemic Discipline with OrgForge v2

OrgForge provides a deterministic framework to measure not just if an AI agent can find information, but whether it has the **discipline** to respect organizational boundaries, temporal horizons, and causal logic.

In OrgForge v2, we move away from "Waldo-style" retrieval benchmarks. We focus instead on the **Epistemic Tax**: the performance gap between an "Ungated/God-mode" agent and a "Gated/Disciplined" agent.

---

## The Evaluation Workflow

The evaluation process follows a three-stage pipeline after your simulation (`flow.py`) completes:

| Phase             | Script                    | Purpose                                                                             |
| :---------------- | :------------------------ | :---------------------------------------------------------------------------------- |
| **1. Generation** | `eval_harness.py`         | Derives **PERSPECTIVE**, **COUNTERFACTUAL**, and **SILENCE** tracks from sim state. |
| **2. Baselines**  | `export_to_hf.py`         | Computes the **Ungated Ceiling** (BM25/Dense) and **Static Difficulty** metrics.    |
| **3. Execution**  | `agentic_eval_harness.py` | Runs the agentic tool-use loop and calculates the **Epistemic Tax**.                |

---

## 1. Establishing the Baselines (Tier 1 & 2)

Before running an agent, we establish the "Floor" and "Ceiling" of the dataset using `export_to_hf.py`. This script requires **zero LLM calls** and runs locally.

```bash
python eval/export_to_hf.py
```

### Tier 1: The Ungated Ceiling

We run BM25 and Dense Retrieval (Qwen3-4B) with **all gates removed**. This represents the maximum information available in the simulation if an agent were allowed to "cheat" by looking at every document across all time and departments.

### Tier 2: Static Reasoning Difficulty

We calculate metrics that define how "hard" the reasoning task is, independent of the model:

- **Contamination Rate:** % of top-tier search results that are "out-of-cone" (forbidden) for the actor.
- **Multi-hop Rate:** % of questions unreachable by a single retrieval pass.
- **Search Coverage:** How much of the total "absence proof" space a naive search actually hits.

---

## 2. Executing the Agentic Eval

The `agentic_eval_harness.py` runs the agent through a tool-use loop. To get a full picture of a model's performance, you should run it in three modes:

### A. The Gated Run (The Real Test)

The agent must answer questions while the harness strictly enforces visibility cones and temporal horizons.

```bash
python eval/agentic_eval_harness.py --model claude-3-5-sonnet --max-steps 15
```

### B. The Ungated Run (The Ceiling)

The same agent, but with all security gates disabled. This defines the model's personal "best case" scenario.

```bash
python eval/agentic_eval_harness.py --model claude-3-5-sonnet --ungated
```

### C. The Zero-Shot Run (The Floor)

The agent is given the question with **no tools**. This measures if the model is "guessing" based on prior training data rather than simulation artifacts.

```bash
python eval/agentic_eval_harness.py --model claude-3-5-sonnet --zero-shot
```

---

## 🎯 Scoring & The Epistemic Tax

The core metric of OrgForge v2 is the **Epistemic Tax**. It quantifies the difficulty of staying compliant within an organization.

$$\text{Epistemic Tax} = \text{Score}_{\text{ungated}} - \text{Score}_{\text{gated}}$$

### Track-Specific Scoring Logic

| Track              | Success Criteria                            | Failure Penalty                                                                                |
| :----------------- | :------------------------------------------ | :--------------------------------------------------------------------------------------------- |
| **PERSPECTIVE**    | Answer correctly using _only_ visible docs. | **Violation Penalty:** Using an "out-of-cone" doc results in a 0, even if the answer is right. |
| **COUNTERFACTUAL** | Identify the correct `causal_mechanism`.    | **Logic Gap:** Identifying the outcome but missing the "Why" (e.g. missing a Jira link).       |
| **SILENCE**        | Prove an artifact does not exist.           | **Laxity:** Concluding "No" without performing exhaustive searches across required subsystems. |

---

## 📊 Interpreting the Leaderboard

A high-performing agent in OrgForge isn't just accurate; it is **verifiably disciplined**.

- **The Cheater:** High accuracy, but high `violation_count`. (Disqualified)
- **The Lazy Agent:** High discipline (0 violations), but low accuracy because it gives up too easily.
- **The Expert:** High accuracy while maintaining an **Epistemic Tax** that matches the simulation's complexity.

---

## Environment Variables

Ensure your `.env` is configured for the providers you wish to test:

- `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` (for Bedrock)
