"""
eval_e2e.py
===========
End-to-end evaluation harness for the OrgForge Enterprise RAG Benchmark.

Runs a full retrieve → generate → score pipeline against any combination
of retriever and generation model, then writes results to
results/<run_id>/  and appends a row to leaderboard.json.

Supports:
  Retrievers : bm25 | cohere | openai | sentence-transformers
  Generators : claude | openai | cohere (Command R+)

Usage
-----
# BM25 retrieval + Claude generation (uses HF dataset by default)
python eval_e2e.py --retriever bm25 --generator claude --model claude-sonnet-4-20250514

# Cohere Embed v4 retrieval + Claude generation
python eval_e2e.py --retriever cohere --generator claude --model claude-sonnet-4-20250514

# BM25 + GPT-4o
python eval_e2e.py --retriever bm25 --generator openai --model gpt-4o

# Load from local parquets instead of HF
python eval_e2e.py --retriever bm25 --generator claude --local ./export/hf_dataset

# Limit to N questions (useful for smoke-testing)
python eval_e2e.py --retriever bm25 --generator claude --limit 10

# Dry-run: retrieval only, no generation (just MRR@10 / Recall@10)
python eval_e2e.py --retriever cohere --generator none

Environment variables
---------------------
  ANTHROPIC_API_KEY   required for --generator claude
  OPENAI_API_KEY      required for --generator openai or --retriever openai
  COHERE_API_KEY      required for --retriever cohere or --generator cohere

Output
------
results/<run_id>/
  per_question.json   — full per-question results (retrieval + generation + score)
  summary.json        — aggregate metrics by question type and difficulty
leaderboard.json      — append-only leaderboard table (one row per run)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger("orgforge.eval_e2e")

# ── Constants ─────────────────────────────────────────────────────────────────

HF_DATASET_ID = os.environ.get("HF_DATASET_ID", "INSERT_ID_HERE")
TOP_K = 10
RESULTS_DIR = Path("results")
LEADERBOARD_PATH = Path("leaderboard.json")


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────


def load_dataset(local_path: Optional[str] = None) -> Tuple[List[dict], List[dict]]:
    """
    Returns (corpus, questions).
    Loads from local Parquet files if local_path is given,
    otherwise downloads from HuggingFace.
    """
    if local_path:
        return _load_local(Path(local_path))
    return _load_hf()


def _load_hf() -> Tuple[List[dict], List[dict]]:
    try:
        from datasets import load_dataset as hf_load
    except ImportError:
        raise SystemExit("pip install datasets  (or pass --local path/to/hf_dataset)")
    logger.info(f"Loading corpus from HuggingFace: {HF_DATASET_ID}")
    corpus_ds = hf_load(
        HF_DATASET_ID, data_files="corpus/corpus-00000.parquet", split="train"
    )
    questions_ds = hf_load(
        HF_DATASET_ID, data_files="questions/questions-00000.parquet", split="train"
    )
    corpus = [dict(r) for r in corpus_ds]
    questions = [dict(r) for r in questions_ds]
    logger.info(
        f"  {len(corpus)} corpus docs, {len(questions)} questions loaded from HF"
    )
    return corpus, questions


def _load_local(base: Path) -> Tuple[List[dict], List[dict]]:
    try:
        import pandas as pd
    except ImportError:
        raise SystemExit("pip install pandas pyarrow")

    corpus_path = base / "corpus" / "corpus-00000.parquet"
    questions_path = base / "questions" / "questions-00000.parquet"

    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")
    if not questions_path.exists():
        raise FileNotFoundError(f"Questions not found: {questions_path}")

    corpus = pd.read_parquet(corpus_path).to_dict("records")
    questions = pd.read_parquet(questions_path).to_dict("records")
    logger.info(
        f"  {len(corpus)} corpus docs, {len(questions)} questions loaded from {base}"
    )
    return corpus, questions


# ─────────────────────────────────────────────────────────────────────────────
# RETRIEVERS
# ─────────────────────────────────────────────────────────────────────────────


class Retriever:
    """Base class — subclasses implement index() and retrieve()."""

    name: str = "base"

    def index(self, corpus: List[dict]) -> None:
        raise NotImplementedError

    def retrieve(self, query: str, top_k: int = TOP_K) -> List[str]:
        """Returns ordered list of doc_ids."""
        raise NotImplementedError


class BM25Retriever(Retriever):
    name = "bm25"

    def index(self, corpus: List[dict]) -> None:
        from rank_bm25 import BM25Okapi

        self._doc_ids = [r["doc_id"] for r in corpus]
        tokenised = [
            self._tokenize(r.get("body") or r.get("content") or "") for r in corpus
        ]
        self._bm25 = BM25Okapi(tokenised)
        logger.info(f"  BM25 index built ({len(self._doc_ids)} docs)")

    def retrieve(self, query: str, top_k: int = TOP_K) -> List[str]:
        scores = self._bm25.get_scores(self._tokenize(query))
        indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self._doc_ids[i] for i in indices[:top_k]]

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.sub(r"[^\w\s]", " ", (text or "").lower()).split()


class CohereRetriever(Retriever):
    """
    Cohere Embed v4 (embed-v4.0) with cosine similarity.
    Uses the 'search_document' / 'search_query' input types.
    Set COHERE_API_KEY in your environment.
    """

    name = "cohere-embed-v4"

    def __init__(self, model: str = "embed-v4.0", batch_size: int = 96):
        self._model = model
        self._batch_size = batch_size

    def index(self, corpus: List[dict]) -> None:
        import cohere

        api_key = os.environ.get("COHERE_API_KEY")
        if not api_key:
            raise SystemExit("Set COHERE_API_KEY to use Cohere retriever")

        self._co = cohere.ClientV2(api_key=api_key)
        self._doc_ids = [r["doc_id"] for r in corpus]
        bodies = [r.get("body") or r.get("content") or "" for r in corpus]

        logger.info(f"  Embedding {len(bodies)} docs with {self._model} ...")
        embeddings = []
        for i in range(0, len(bodies), self._batch_size):
            batch = bodies[i : i + self._batch_size]
            resp = self._co.embed(
                texts=batch,
                model=self._model,
                input_type="search_document",
                embedding_types=["float"],
            )
            embeddings.extend(resp.embeddings.float_)
            logger.info(
                f"    embedded {min(i + self._batch_size, len(bodies))}/{len(bodies)}"
            )

        mat = np.array(embeddings, dtype=np.float32)
        # Normalise for cosine similarity via dot product
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        self._matrix = mat / np.where(norms == 0, 1, norms)
        logger.info("  Cohere index ready")

    def retrieve(self, query: str, top_k: int = TOP_K) -> List[str]:

        resp = self._co.embed(
            texts=[query],
            model=self._model,
            input_type="search_query",
            embedding_types=["float"],
        )
        q_vec = np.array(resp.embeddings.float_[0], dtype=np.float32)
        q_vec /= max(np.linalg.norm(q_vec), 1e-9)
        scores = self._matrix @ q_vec
        indices = scores.argsort()[::-1][:top_k]
        return [self._doc_ids[int(i)] for i in indices]


class OpenAIRetriever(Retriever):
    """
    OpenAI text-embedding-3-large.
    Set OPENAI_API_KEY in your environment.
    """

    name = "openai-text-embedding-3-large"

    def __init__(self, model: str = "text-embedding-3-large", batch_size: int = 512):
        self._model = model
        self._batch_size = batch_size

    def index(self, corpus: List[dict]) -> None:

        from openai import OpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("Set OPENAI_API_KEY to use OpenAI retriever")

        self._client = OpenAI(api_key=api_key)
        self._doc_ids = [r["doc_id"] for r in corpus]
        bodies = [r.get("body", "") or "" for r in corpus]

        logger.info(f"  Embedding {len(bodies)} docs with {self._model} ...")
        embeddings = []
        for i in range(0, len(bodies), self._batch_size):
            batch = bodies[i : i + self._batch_size]
            resp = self._client.embeddings.create(model=self._model, input=batch)
            embeddings.extend([e.embedding for e in resp.data])
            logger.info(
                f"    embedded {min(i + self._batch_size, len(bodies))}/{len(bodies)}"
            )

        mat = np.array(embeddings, dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        self._matrix = mat / np.where(norms == 0, 1, norms)
        logger.info("  OpenAI index ready")

    def retrieve(self, query: str, top_k: int = TOP_K) -> List[str]:

        resp = self._client.embeddings.create(model=self._model, input=[query])
        q_vec = np.array(resp.data[0].embedding, dtype=np.float32)
        q_vec /= max(np.linalg.norm(q_vec), 1e-9)
        scores = self._matrix @ q_vec
        indices = scores.argsort()[::-1][:top_k]
        return [self._doc_ids[int(i)] for i in indices]


class BedrockCohereRetriever(Retriever):
    """
    Cohere Embed v4 via Amazon Bedrock (invoke_model).

    Uses the same AWS credential chain as BedrockGenerator — no separate
    API key needed if you're already authenticated to Bedrock.

    Model ID : cohere.embed-v4:0
    Regions  : us-east-1, eu-west-1, ap-northeast-1
               (cross-region inference also supported)
    """

    name = "bedrock-cohere-embed-v4"

    def __init__(
        self,
        model: str = "us.cohere.embed-v4:0",
        region: str = "us-east-1",
        batch_size: int = 96,
    ):
        import boto3

        self._model = model
        self._batch_size = batch_size
        self._client = boto3.client("bedrock-runtime", region_name=region)
        logger.info(f"  BedrockCohereRetriever — model: {model}, region: {region}")

    def _embed(self, texts: List[str], input_type: str) -> "np.ndarray":
        import json

        all_embeddings = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            body = json.dumps(
                {
                    "texts": batch,
                    "input_type": input_type,  # "search_document" or "search_query"
                    "embedding_types": ["float"],
                }
            )
            resp = self._client.invoke_model(
                modelId=self._model,
                body=body,
                accept="*/*",
                contentType="application/json",
            )
            result = json.loads(resp["body"].read())
            # Bedrock Cohere v4 returns {"embeddings": {"float": [[...], ...]}}
            batch_vecs = result["embeddings"]["float"]
            all_embeddings.extend(batch_vecs)
            logger.info(
                f"    embedded {min(i + self._batch_size, len(texts))}/{len(texts)}"
            )
        return np.array(all_embeddings, dtype=np.float32)

    def index(self, corpus: List[dict]) -> None:

        self._doc_ids = [r["doc_id"] for r in corpus]
        bodies = [r.get("body", "") or "" for r in corpus]

        logger.info(f"  Embedding {len(bodies)} docs via Bedrock Cohere Embed v4 ...")
        mat = self._embed(bodies, input_type="search_document")
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        self._matrix = mat / np.where(norms == 0, 1, norms)
        logger.info("  Bedrock Cohere index ready")

    def retrieve(self, query: str, top_k: int = TOP_K) -> List[str]:

        q_mat = self._embed([query], input_type="search_query")
        q_vec = q_mat[0]
        q_vec /= max(float(np.linalg.norm(q_vec)), 1e-9)
        scores = self._matrix @ q_vec
        indices = scores.argsort()[::-1][:top_k]
        return [self._doc_ids[int(i)] for i in indices]


def build_retriever(name: str, region: str = "us-east-1") -> Retriever:
    if name == "bm25":
        return BM25Retriever()
    if name == "cohere":
        return CohereRetriever()
    if name == "cohere-bedrock":
        return BedrockCohereRetriever(region=region)
    if name == "openai":
        return OpenAIRetriever()
    raise ValueError(
        f"Unknown retriever: {name!r}. Choose bm25 | cohere | cohere-bedrock | openai"
    )


# ─────────────────────────────────────────────────────────────────────────────
# GENERATORS
# ─────────────────────────────────────────────────────────────────────────────


SYSTEM_PROMPT = """\
You are evaluating an enterprise knowledge base. You will be given a question
type, a question, and retrieved document excerpts. Answer using ONLY the
provided documents. Always respond with valid JSON matching the schema for
the question type — no markdown fences, no extra keys.

─── RETRIEVAL ───────────────────────────────────────────────────────────────
Which artifact first documented a specific fact?
{
  "artifact_id": "<exact doc_id e.g. ORG-42 or CONF-ENG-007>",
  "artifact_type": "<jira|confluence|slack_thread|email|pr>",
  "timestamp": "<ISO datetime if available, else omit>",
  "retrieved_artifact_ids": ["<id1>", "<id2>"]
}

─── CAUSAL ──────────────────────────────────────────────────────────────────
What artifact or action directly followed event X?
{
  "artifact_id": "<doc_id of the resulting artifact>",
  "event_type": "<event type string e.g. confluence_created>",
  "actors": ["<name1>", "<name2>"],
  "retrieved_artifact_ids": ["<id1>", "<id2>"]
}

─── TEMPORAL ────────────────────────────────────────────────────────────────
Did person P have access/knowledge of domain D before incident I?
{
  "had_knowledge": true,
  "person": "<name>",
  "domain": "<domain string>",
  "departure_day": null,
  "reasoning": "<one sentence>"
}

─── GAP_DETECTION ───────────────────────────────────────────────────────────
Was email/artifact E ever actioned?
{
  "was_actioned": false,
  "artifact_id": "<the email or artifact doc_id>",
  "downstream_artifacts": [],
  "retrieved_artifact_ids": ["<id1>"]
}

─── ROUTING ─────────────────────────────────────────────────────────────────
Who was the first internal person to receive/see inbound artifact X?
{
  "first_recipient": "<person name>",
  "was_escalated": true,
  "retrieved_artifact_ids": ["<id1>"]
}

─── PLAN ────────────────────────────────────────────────────────────────────
What was department X focused on during Day N?
{
  "dept": "<exact department name e.g. Engineering_Backend>",
  "theme": "<short theme description>",
  "retrieved_artifact_ids": ["<id1>"]
}

─── ESCALATION ──────────────────────────────────────────────────────────────
Who was involved in the escalation chain for incident X?
{
  "escalation_actors": ["<name1>", "<name2>"],
  "retrieved_artifact_ids": ["<id1>", "<id2>"]
}

─── KNOWLEDGE_GAP ───────────────────────────────────────────────────────────
What domain was undocumented when incident X fired?
{
  "gap_areas": ["<domain1>", "<domain2>"],
  "retrieved_artifact_ids": ["<id1>"]
}

If the documents contain insufficient evidence, still return the correct
schema with your best guess and add "insufficient_evidence": true.
"""


def _build_context(corpus_map: Dict[str, dict], doc_ids: List[str]) -> str:
    parts = []
    for doc_id in doc_ids:
        doc = corpus_map.get(doc_id)
        if not doc:
            continue
        parts.append(
            f"--- [{doc_id}] {doc.get('title', '')} ({doc.get('doc_type', '')}) ---\n"
            f"{(doc.get('body', '') or '')[:1500]}"
        )
    return "\n\n".join(parts)


class Generator:
    name: str = "base"

    def generate(self, question: str, question_type: str, context: str) -> dict:
        raise NotImplementedError


class NullGenerator(Generator):
    """Used for retrieval-only runs (--generator none)."""

    name = "none"

    def generate(self, question: str, question_type: str, context: str) -> dict:
        return {"answer": None, "artifact_ids": [], "reasoning": "retrieval-only run"}


class ClaudeGenerator(Generator):
    def __init__(self, model: str = "claude-sonnet-4-20250514", max_tokens: int = 512):
        import anthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise SystemExit("Set ANTHROPIC_API_KEY to use Claude generator")
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens
        self.name = f"claude/{model}"

    def generate(self, question: str, question_type: str, context: str) -> dict:
        user_msg = (
            f"Question type: {question_type}\n\n"
            f"Question: {question}\n\n"
            f"Retrieved documents:\n{context}"
        )
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        return _parse_json_response(resp.content[0].text)


class OpenAIGenerator(Generator):
    def __init__(self, model: str = "gpt-4o", max_tokens: int = 512):
        from openai import OpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("Set OPENAI_API_KEY to use OpenAI generator")
        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens
        self.name = f"openai/{model}"

    def generate(self, question: str, question_type: str, context: str) -> dict:
        user_msg = (
            f"Question type: {question_type}\n\n"
            f"Question: {question}\n\n"
            f"Retrieved documents:\n{context}"
        )
        resp = self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},
        )
        return _parse_json_response(resp.choices[0].message.content)


class CohereGenerator(Generator):
    def __init__(self, model: str = "command-r-plus", max_tokens: int = 512):
        import cohere

        api_key = os.environ.get("COHERE_API_KEY")
        if not api_key:
            raise SystemExit("Set COHERE_API_KEY to use Cohere generator")
        self._co = cohere.ClientV2(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens
        self.name = f"cohere/{model}"

    def generate(self, question: str, question_type: str, context: str) -> dict:
        user_msg = (
            f"Question type: {question_type}\n\n"
            f"Question: {question}\n\n"
            f"Retrieved documents:\n{context}"
        )
        resp = self._co.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=self._max_tokens,
        )
        return _parse_json_response(resp.message.content[0].text)


class BedrockGenerator(Generator):
    """
    Amazon Bedrock via boto3 converse() API.

    Works with any model Bedrock exposes through the Converse API:
      Claude   : anthropic.claude-3-5-sonnet-20241022-v2:0
                 anthropic.claude-3-7-sonnet-20250219-v1:0
      Llama    : meta.llama3-3-70b-instruct-v1:0
      Mistral  : mistral.mistral-large-2402-v1:0
      Nova     : amazon.nova-pro-v1:0  /  amazon.nova-lite-v1:0
      Titan    : amazon.titan-text-premier-v1:0

    Authentication uses your standard AWS credential chain:
      - environment variables   AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
      - ~/.aws/credentials profile
      - IAM role (EC2 / ECS / Lambda)

    Pass --region to target a specific Bedrock region (default: us-east-1).
    Cross-region inference profile IDs (e.g. us.anthropic.claude-3-7-...) are
    supported — just pass the full profile ARN or ID as --model.
    """

    def __init__(
        self,
        model: str = "anthropic.claude-3-5-sonnet-20241022-v2:0",
        region: str = "us-east-1",
        max_tokens: int = 512,
        call_delay: float = 1.0,  # seconds to sleep between every call
        max_retries: int = 6,  # retries on ThrottlingException
        retry_base_delay: float = 5.0,  # initial backoff seconds (doubles each retry)
    ):
        import boto3

        if not region or len(region.split("-")) < 3:
            raise ValueError(
                f"Invalid AWS region: {region!r}. "
                "Expected format like 'us-east-1' or 'us-west-2'. "
                "Pass --region explicitly to override ~/.aws/config."
            )

        self._model = model
        self._max_tokens = max_tokens
        self._call_delay = call_delay
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay
        self._client = boto3.client("bedrock-runtime", region_name=region)
        self.name = f"bedrock/{model}"
        logger.info(f"  Bedrock client initialised — model: {model}, region: {region}")

    def generate(self, question: str, question_type: str, context: str) -> dict:
        import random

        user_msg = (
            f"Question type: {question_type}\n\n"
            f"Question: {question}\n\n"
            f"Retrieved documents:\n{context}"
        )

        # Polite inter-call delay to stay under TPM limits
        if self._call_delay > 0:
            time.sleep(self._call_delay)

        last_exc = None
        for attempt in range(self._max_retries + 1):
            try:
                resp = self._client.converse(
                    modelId=self._model,
                    system=[{"text": SYSTEM_PROMPT}],
                    messages=[{"role": "user", "content": [{"text": user_msg}]}],
                    inferenceConfig={"maxTokens": self._max_tokens},
                )
                content_blocks = resp["output"]["message"]["content"]
                text = ""
                for block in content_blocks:
                    if "text" in block:
                        text = block["text"]
                        break
                if not text:
                    logger.warning(f"  Unexpected content blocks: {content_blocks}")
                    return {
                        "answer": str(content_blocks),
                        "artifact_ids": [],
                        "reasoning": "parse error",
                    }
                return _parse_json_response(text)

            except Exception as exc:
                error_code = (
                    getattr(exc, "response", {}).get("Error", {}).get("Code", "")
                )
                if error_code == "ThrottlingException":
                    if attempt >= self._max_retries:
                        logger.error(
                            f"  Throttled after {self._max_retries} retries — giving up"
                        )
                        raise
                    # Exponential backoff with ±20% jitter
                    delay = self._retry_base_delay * (2**attempt)
                    delay *= 0.8 + 0.4 * random.random()
                    logger.warning(
                        f"  Throttled (attempt {attempt + 1}/{self._max_retries}), "
                        f"retrying in {delay:.1f}s ..."
                    )
                    time.sleep(delay)
                    last_exc = exc
                else:
                    raise

        raise last_exc  # should never reach here


def _parse_json_response(text: str) -> dict:
    """
    Extract JSON from model response, tolerating:
      - <think>...</think> reasoning blocks (DeepSeek R1 / chain-of-thought models)
      - markdown fences (```json ... ```)
      - prose before/after the JSON block
      - missing fences (bare JSON)
    """
    # 1. Strip <think>...</think> blocks (DeepSeek R1 and similar CoT models)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # 2. Try to extract a fenced JSON block first
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    else:
        # 3. Find the first { and last } to extract bare JSON
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
        else:
            candidate = text.strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return {"answer": text, "artifact_ids": [], "reasoning": "parse error"}


def build_generator(
    name: str, model: Optional[str], region: str = "us-east-1", call_delay: float = 1.0
) -> Generator:
    if name == "none":
        return NullGenerator()
    if name == "claude":
        return ClaudeGenerator(model=model or "claude-sonnet-4-20250514")
    if name == "openai":
        return OpenAIGenerator(model=model or "gpt-4o")
    if name == "cohere":
        return CohereGenerator(model=model or "command-r-plus")
    if name == "bedrock":
        return BedrockGenerator(
            model=model or "anthropic.claude-3-5-sonnet-20241022-v2:0",
            region=region,
            call_delay=call_delay,
        )
    raise ValueError(
        f"Unknown generator: {name!r}. Choose none | claude | openai | cohere | bedrock"
    )


# ─────────────────────────────────────────────────────────────────────────────
# SCORING  (wraps scorer.py if present, falls back to retrieval-only metrics)
# ─────────────────────────────────────────────────────────────────────────────


def _load_scorer(scorer_path: Optional[str] = None):
    """
    Try to import OrgForgeScorer from scorer.py.
    Search order:
      1. --scorer CLI arg (explicit path)
      2. Same directory as eval_e2e.py
      3. Parent directory of eval_e2e.py
      4. Current working directory
    """
    from importlib.machinery import SourceFileLoader
    import types

    candidates = []
    if scorer_path:
        candidates.append(Path(scorer_path))

    this_dir = Path(__file__).resolve().parent
    candidates += [
        this_dir / "scorer.py",
        this_dir.parent / "scorer.py",
        Path.cwd() / "scorer.py",
    ]

    for p in candidates:
        if not p.exists():
            continue
        try:
            import sys

            mod = types.ModuleType("orgforge_scorer")
            mod.__file__ = str(p)
            sys.modules["orgforge_scorer"] = (
                mod  # must be registered before exec for @dataclass
            )
            SourceFileLoader("orgforge_scorer", str(p)).exec_module(mod)
            scorer = mod.OrgForgeScorer()
            logger.info(f"  scorer.py loaded from {p}")
            return scorer
        except Exception as exc:
            logger.warning(f"  scorer.py found at {p} but failed to load ({exc})")

    logger.warning(
        "scorer.py not found — using retrieval-only metrics. "
        "Pass --scorer /path/to/scorer.py to fix this."
    )
    return None


def score_answer(
    scorer,
    question: dict,
    agent_answer: dict,
    top_k_ids: List[str],
) -> dict:
    """
    Returns a scoring dict:
      retrieval_mrr    — MRR@10 based on evidence_chain
      retrieval_recall — Recall@10 based on evidence_chain
      answer_score     — 0.0–1.0 from scorer.py, or None if unavailable
      correct          — bool (score >= 0.9), or None
    """
    evidence = question.get("evidence_chain", [])
    if isinstance(evidence, str):
        try:
            evidence = json.loads(evidence)
        except Exception:
            evidence = []

    relevant = set(evidence)
    mrr = next(
        (1.0 / (i + 1) for i, d in enumerate(top_k_ids) if d in relevant),
        0.0,
    )
    recall = (
        sum(1 for d in top_k_ids if d in relevant) / len(relevant) if relevant else 1.0
    )

    answer_score = None
    if (
        scorer is not None
        and agent_answer.get("answer") is not None
        or any(
            k in agent_answer
            for k in (
                "artifact_id",
                "had_knowledge",
                "was_actioned",
                "first_recipient",
                "dept",
                "escalation_actors",
                "gap_areas",
            )
        )
    ):
        try:
            # Inject retrieved IDs so evidence scoring works even if LLM omits them
            enriched = {**agent_answer}
            if (
                "retrieved_artifact_ids" not in enriched
                or not enriched["retrieved_artifact_ids"]
            ):
                enriched["retrieved_artifact_ids"] = top_k_ids
            result = scorer.score(question, enriched)
            answer_score = result.score  # ScorerResult.score is always a float
        except Exception as exc:
            logger.debug(f"Scorer error on {question.get('question_id')}: {exc}")

    return {
        "retrieval_mrr": round(mrr, 4),
        "retrieval_recall": round(recall, 4),
        "answer_score": round(answer_score, 4) if answer_score is not None else None,
        "correct": (answer_score >= 0.9) if answer_score is not None else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# AGGREGATE METRICS
# ─────────────────────────────────────────────────────────────────────────────


def _mean(vals: List[float]) -> float:
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def aggregate(per_question: List[dict]) -> dict:
    by_type: Dict[str, list] = defaultdict(list)
    by_diff: Dict[str, list] = defaultdict(list)

    for r in per_question:
        qtype = r.get("question_type", "UNKNOWN")
        diff = r.get("difficulty", "unknown")
        by_type[qtype].append(r)
        by_diff[diff].append(r)

    def _agg_group(rows):
        mrr_vals = [r["scores"]["retrieval_mrr"] for r in rows]
        rec_vals = [r["scores"]["retrieval_recall"] for r in rows]
        score_vals = [
            r["scores"]["answer_score"]
            for r in rows
            if r["scores"]["answer_score"] is not None
        ]
        correct_vals = [
            r["scores"]["correct"] for r in rows if r["scores"]["correct"] is not None
        ]
        return {
            "n": len(rows),
            "mrr_at_10": _mean(mrr_vals),
            "recall_at_10": _mean(rec_vals),
            "answer_score": _mean(score_vals) if score_vals else None,
            "accuracy": _mean([float(v) for v in correct_vals])
            if correct_vals
            else None,
        }

    return {
        "overall": _agg_group(per_question),
        "by_type": {k: _agg_group(v) for k, v in sorted(by_type.items())},
        "by_difficulty": {k: _agg_group(v) for k, v in sorted(by_diff.items())},
    }


# ─────────────────────────────────────────────────────────────────────────────
# LEADERBOARD
# ─────────────────────────────────────────────────────────────────────────────

LEADERBOARD_CSV_PATH = Path("leaderboard.csv")

# All question types — used to produce stable CSV columns across runs even
# when a given run hasn't seen every type yet (cells will be empty).
_ALL_QTYPES = [
    "CAUSAL",
    "ESCALATION",
    "GAP_DETECTION",
    "PLAN",
    "RETRIEVAL",
    "ROUTING",
    "TEMPORAL",
]


def _flatten_row(row: dict) -> dict:
    """Flatten a leaderboard JSON row into a CSV-friendly dict.

    Per-type metrics become columns: mrr_CAUSAL, score_CAUSAL, etc.
    Tier 1 = mrr_at_10 / recall_at_10  (always present)
    Tier 2 = answer_score / accuracy    (None for retrieval-only runs)
    """
    flat = {
        "run_id": row.get("run_id", ""),
        "timestamp": row.get("timestamp", ""),
        "tier": row.get("tier", ""),
        "retriever": row.get("retriever", ""),
        "generator": row.get("generator", ""),
        "n": row.get("n", ""),
        # Tier 1 overall
        "mrr_at_10": row.get("mrr_at_10", ""),
        "recall_at_10": row.get("recall_at_10", ""),
        # Tier 2 overall (empty string for Tier 1-only runs)
        "answer_score": row.get("answer_score", ""),
        "accuracy": row.get("accuracy", ""),
    }
    by_type = row.get("by_type", {})
    for qtype in _ALL_QTYPES:
        m = by_type.get(qtype, {})
        flat[f"mrr_{qtype}"] = m.get("mrr_at_10", "")
        flat[f"score_{qtype}"] = m.get("answer_score", "")
    return flat


def _write_leaderboard_csv(leaderboard: List[dict]) -> None:
    import csv

    if not leaderboard:
        return

    # Union all keys across rows so older rows without newer qtypes still render
    all_keys: list = []
    seen_keys: set = set()
    for row in leaderboard:
        for k in _flatten_row(row):
            if k not in seen_keys:
                all_keys.append(k)
                seen_keys.add(k)

    with open(LEADERBOARD_CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for row in leaderboard:
            writer.writerow(_flatten_row(row))

    logger.info(f"  leaderboard CSV updated: {LEADERBOARD_CSV_PATH}")


def update_leaderboard(
    run_id: str, retriever: str, generator: str, summary: dict
) -> None:
    leaderboard = []
    if LEADERBOARD_PATH.exists():
        leaderboard = json.loads(LEADERBOARD_PATH.read_text())

    overall = summary.get("overall", {})
    tier = "1" if generator == "none" else "1+2"
    row = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tier": tier,
        "retriever": retriever,
        "generator": generator,
        "n": overall.get("n"),
        "mrr_at_10": overall.get("mrr_at_10"),
        "recall_at_10": overall.get("recall_at_10"),
        "answer_score": overall.get("answer_score"),  # None for Tier 1 runs
        "accuracy": overall.get("accuracy"),  # None for Tier 1 runs
        "by_type": {
            qtype: {
                "mrr_at_10": m.get("mrr_at_10"),
                "answer_score": m.get("answer_score"),
            }
            for qtype, m in summary.get("by_type", {}).items()
        },
    }

    # Replace existing run with same id, else append
    leaderboard = [r for r in leaderboard if r.get("run_id") != run_id]
    leaderboard.append(row)

    # Tier 1+2 rows rank above Tier 1; within each tier sort by primary metric desc
    leaderboard.sort(
        key=lambda r: (
            0 if r.get("tier") == "1+2" else 1,
            -(r.get("answer_score") or 0.0),
            -(r.get("mrr_at_10") or 0.0),
        ),
    )

    LEADERBOARD_PATH.write_text(json.dumps(leaderboard, indent=2))
    logger.info(f"  leaderboard JSON updated: {LEADERBOARD_PATH}")

    _write_leaderboard_csv(leaderboard)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EVAL LOOP
# ─────────────────────────────────────────────────────────────────────────────


def run_eval(args: argparse.Namespace) -> None:
    run_dir_tmp = RESULTS_DIR / f"_tmp_{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    run_dir_tmp.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    corpus, questions = load_dataset(args.local)

    if args.limit:
        questions = questions[: args.limit]
        logger.info(f"  Limited to {args.limit} questions")

    corpus_map = {r["doc_id"]: r for r in corpus}

    # 2. Build retriever + index
    retriever = build_retriever(args.retriever, region=args.region)
    logger.info(f"Indexing with {retriever.name} ...")
    t0 = time.time()
    retriever.index(corpus)
    logger.info(f"  Index built in {time.time() - t0:.1f}s")

    # 3. Build generator — do this before constructing run_id so we can use generator.name
    generator = build_generator(
        args.generator, args.model, region=args.region, call_delay=args.call_delay
    )
    logger.info(f"Generator: {generator.name}")

    # run_id uses the full generator name (e.g. bedrock/claude-opus-4-6) not just the flag
    safe_gen = generator.name.replace("/", "-").replace(":", "-")
    run_id = f"{retriever.name}__{safe_gen}__{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    run_dir = RESULTS_DIR / run_id
    run_dir_tmp.rename(run_dir)
    logger.info(f"Run ID: {run_id}")

    # 4. Load scorer
    scorer = _load_scorer(getattr(args, "scorer", None))

    # 5. Eval loop — deserialise JSON string fields from parquet before scoring
    for q in questions:
        for field in ("ground_truth", "evidence_chain"):
            val = q.get(field)
            if isinstance(val, str):
                try:
                    q[field] = json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    pass

    per_question = []
    for i, q in enumerate(questions):
        qid = q.get("question_id", f"q{i}")
        qtype = q.get("question_type", "")
        qtext = q.get("question_text", "")

        # Retrieve
        top_k_ids = retriever.retrieve(qtext, top_k=TOP_K)

        # Generate
        context = _build_context(corpus_map, top_k_ids)
        agent_answer = generator.generate(qtext, qtype, context)

        # Score
        scores = score_answer(scorer, q, agent_answer, top_k_ids)

        per_question.append(
            {
                "question_id": qid,
                "question_type": qtype,
                "difficulty": q.get("difficulty"),
                "question_text": qtext,
                "top_k_ids": top_k_ids,
                "agent_answer": agent_answer,
                "scores": scores,
            }
        )

        status = (
            f"✓ {scores['answer_score']:.2f}"
            if scores["answer_score"] is not None
            else f"MRR {scores['retrieval_mrr']:.2f}"
        )
        logger.info(f"  [{i + 1}/{len(questions)}] {qid} ({qtype}) — {status}")

    # 6. Aggregate
    summary = aggregate(per_question)

    # 7. Write results
    with open(run_dir / "per_question.json", "w") as f:
        json.dump(per_question, f, indent=2, default=str)
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Results written to {run_dir}")

    # 8. Update leaderboard
    update_leaderboard(run_id, retriever.name, generator.name, summary)

    # 9. Print summary table
    _print_summary(summary, retriever.name, generator.name)


def _print_summary(summary: dict, retriever: str, generator: str) -> None:
    print(f"\n{'=' * 64}")
    print(f"  Retriever : {retriever}")
    print(f"  Generator : {generator}")
    print(f"{'=' * 64}")
    print(
        f"  {'Type':<16} {'MRR@10':>8} {'Recall@10':>10} {'Score':>8} {'Acc':>6} {'N':>4}"
    )
    print(f"  {'-' * 56}")

    def _fmt(v):
        return f"{v:.4f}" if v is not None else "  n/a "

    overall_row = summary.get("overall", {})
    print(
        f"  {'OVERALL':<16} {_fmt(overall_row.get('mrr_at_10')):>8} "
        f"{_fmt(overall_row.get('recall_at_10')):>10} "
        f"{_fmt(overall_row.get('answer_score')):>8} "
        f"{_fmt(overall_row.get('accuracy')):>6} "
        f"{overall_row.get('n', 0):>4}"
    )
    print(f"  {'-' * 56}")
    for qtype, m in sorted(summary.get("by_type", {}).items()):
        print(
            f"  {qtype:<16} {_fmt(m.get('mrr_at_10')):>8} "
            f"{_fmt(m.get('recall_at_10')):>10} "
            f"{_fmt(m.get('answer_score')):>8} "
            f"{_fmt(m.get('accuracy')):>6} "
            f"{m.get('n', 0):>4}"
        )
    print(f"{'=' * 64}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OrgForge end-to-end RAG evaluation harness"
    )
    p.add_argument(
        "--retriever",
        choices=["bm25", "cohere", "cohere-bedrock", "openai"],
        default="bm25",
        help="Retriever to use (default: bm25). cohere-bedrock uses Bedrock credentials.",
    )
    p.add_argument(
        "--generator",
        choices=["none", "claude", "openai", "cohere", "bedrock"],
        default="claude",
        help="Generation model to use (default: claude). Use 'none' for retrieval-only.",
    )
    p.add_argument(
        "--model",
        default=None,
        help=(
            "Specific model string for the generator. Examples:\n"
            "  claude   : claude-sonnet-4-20250514\n"
            "  openai   : gpt-4o\n"
            "  cohere   : command-r-plus\n"
            "  bedrock  : anthropic.claude-3-5-sonnet-20241022-v2:0\n"
            "             anthropic.claude-3-7-sonnet-20250219-v1:0\n"
            "             meta.llama3-3-70b-instruct-v1:0\n"
            "             mistral.mistral-large-2402-v1:0\n"
            "             amazon.nova-pro-v1:0"
        ),
    )
    p.add_argument(
        "--region",
        default="us-east-1",
        help="AWS region for Bedrock (default: us-east-1)",
    )
    p.add_argument(
        "--local",
        default=None,
        metavar="PATH",
        help="Path to local hf_dataset directory (skips HuggingFace download)",
    )
    p.add_argument(
        "--scorer",
        default=None,
        metavar="PATH",
        help="Explicit path to scorer.py (e.g. ../scorer.py). Auto-discovered if omitted.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Evaluate only the first N questions (useful for smoke-testing)",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help=f"Number of documents to retrieve per question (default: {TOP_K})",
    )
    p.add_argument(
        "--call-delay",
        type=float,
        default=1.0,
        metavar="SECONDS",
        help="Sleep between LLM calls to avoid throttling (default: 1.0s). "
        "Increase to 2-3 for Opus or if you keep hitting ThrottlingException.",
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Debug logging",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    run_eval(args)
