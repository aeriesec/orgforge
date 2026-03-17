# 🔬 Evaluating Enterprise RAG with OrgForge

OrgForge provides a deterministic framework to measure exactly how well an AI agent retrieves and reasons over institutional knowledge. Unlike traditional benchmarks, OrgForge ground truth is derived directly from the simulation state machine, eliminating "hallucinated" answers in the evaluation set.

## The Evaluation Workflow

The evaluation process follows a three-stage pipeline after your simulation (`flow.py`) completes:

| Phase                | Script            | Purpose                                                                     |
| -------------------- | ----------------- | --------------------------------------------------------------------------- |
| **1. Generation**    | `eval_harness.py` | Transforms SimEvents into a typed Q&A dataset with deterministic answers.   |
| **2. Normalization** | `export_to_hf.py` | Flattens artifacts into a corpus and runs BM25/Dense retrieval baselines.   |
| **3. Execution**     | `eval_e2e.py`     | Runs the full Retrieve → Generate → Score pipeline against your chosen LLM. |

---

## 1. Generating the Eval Dataset

Once a simulation finishes, the harness extracts causal threads and generates questions. While an LLM is used to make the question prose sound natural, the **ground truth** and **evidence chain** are pulled directly from the event log.

```bash
python src/eval_harness.py

```

**Key Outputs in `export/eval/`:**

- **`eval_questions.json`**: The core benchmark containing typed questions (Temporal, Causal, etc.), difficulty levels, and the required evidence IDs.
- **`causal_threads.json`**: Maps of how artifacts (Tickets -> PRs -> Docs) are linked.

---

## 2. Running End-to-End Benchmarks

The `eval_e2e.py` script is the primary tool for testing your RAG agents. It supports various retrievers (BM25, Cohere, OpenAI) and generation models (Claude, GPT-4, etc.).

### Common Commands

**Standard RAG Test (BM25 + GPT-4o):**

```bash
python eval_e2e.py --retriever bm25 --generator openai --model gpt-4o

```

**High-Fidelity Test (Cohere Embed + Claude 3.5 via Bedrock):**

```bash
python eval_e2e.py --retriever cohere --generator bedrock --model anthropic.claude-3-5-sonnet-20241022-v2:0

```

**Retrieval-Only (Smoke test for MRR/Recall without LLM costs):**

```bash
python eval_e2e.py --retriever cohere --generator none

```

Since you're adding these specific flags to handle AWS infrastructure, local development, and rate limiting, it's best to group them under an **"Advanced Configuration"** or **"Advanced Execution"** section. This helps users who are moving beyond a simple local smoke test.

Here is a snippet you can drop into your `EVAL.md`:

---

### ⚙️ Advanced Execution & Infrastructure

For production-grade evaluations or restricted environments, `eval_e2e.py` provides granular control over infrastructure and rate limits.

| Flag           | Purpose                                                         | Recommended Use                                                                                             |
| -------------- | --------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `--region`     | Sets the **AWS Region** for Bedrock calls.                      | Use `us-east-1` or `us-west-2` for the widest model availability.                                           |
| `--local`      | Points to a **local directory** for HF-formatted Parquet files. | Use this to bypass HuggingFace downloads and test private simulation runs.                                  |
| `--scorer`     | Explicitly defines the path to **`scorer.py`**.                 | Necessary if running the eval script from a different directory than the source.                            |
| `--call-delay` | Introduces a **sleep timer** (seconds) between LLM calls.       | **Crucial for Bedrock.** Set to `2.0` or higher if you encounter `ThrottlingException` on high-tier models. |

#### Example: Running a Cloud-Hybrid Eval

If you have exported your dataset to `./export/my_run/` and want to test against **Claude 3.7** on AWS without hitting rate limits:

```bash
python eval_e2e.py \
  --retriever cohere-bedrock \
  --generator bedrock \
  --model anthropic.claude-3-7-sonnet-20250219-v1:0 \
  --region us-east-1 \
  --local ./export/hf_dataset \
  --call-delay 1.5 \
  --scorer ./src/scorer.py

```

## 3. Understanding Question Types & Scoring

OrgForge uses `scorer.py` to provide deterministic, per-type scoring on a scale of **0.0 to 1.0**.

### Reasoning Categories

- **RETRIEVAL**: Locating the specific artifact that first documented a fact.
- **CAUSAL**: Identifying the specific action or document that followed an event (e.g., "Which PR resolved IT-108?").
- **TEMPORAL**: Reasoning about an actor's knowledge state at a specific point in time (e.g., "Did Jax know about the domain gap _before_ the incident?").
- **GAP_DETECTION**: Identifying the absence of action, such as an email that was never responded to.
- **ROUTING**: Tracing the first internal recipient of external communications.

### Partial Credit Logic

The scorer separates **retrieval quality** from **reasoning quality**. An agent can receive partial credit (~0.2–0.8) if it finds the correct documents in the `evidence_chain` even if it draws the wrong conclusion. A score of **≥ 0.9** is considered a full "correct" answer.

---

## 4. Viewing Results

Each evaluation run generates a unique `run_id` and saves data to `results/<run_id>/`:

- **`summary.json`**: Aggregate metrics by question type and difficulty.
- **`per_question.json`**: A deep dive into every individual query, the context retrieved, and the specific failure reason if it was incorrect.
- **`leaderboard.json`**: An append-only file used to compare different model/retriever combinations over time.

---

### 🎯 Scoring Methodology & Math

The `OrgForgeScorer` uses a weighted formula to calculate the final `score` [0.0 - 1.0] for each question:

$$Score = (PrimaryScore \times 0.8) + (EvidenceScore \times 0.2)$$

- **Primary Score (80%):** Measures the accuracy of the final answer (e.g., correct `artifact_id` or boolean state).
- **Evidence Score (20%):** Measures retrieval recall. This ensures agents get partial credit for "finding the right docs" even if the reasoning fails.

#### Temporal "Off-by-One" Logic

In **TEMPORAL** questions (e.g., "Did Jax know X?"), the scorer is aware of the simulation's daily rhythm.

- **Full Credit:** If the agent correctly identifies the boolean state **and** identifies the employee departure day within **±1 day** of the ground truth.
- **Partial Credit (0.6):** If the agent gets the boolean right but misses the departure day or provides an incorrect date.

---

## Environment Variables

Ensure your `.env` is configured for the providers you wish to test:

- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY`
- `COHERE_API_KEY`
- `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` (for Bedrock)
