"""
embed_worker.py
===============
Background embedding queue for OrgForge.

Decouples artifact embedding from LLM generation so Infinity/Ollama inference
runs while the next Bedrock call is in flight, rather than blocking between
each generation step.

Architecture
------------
- A ThreadPoolExecutor processes embed tasks concurrently from a Queue.
- Concurrency is tuned via EMBED_WORKER_CONCURRENCY (default: 8 for Infinity,
  1 for Ollama and gateway-backed OpenAI providers).
- The main sim loop calls enqueue() instead of mem.embed_artifact() directly.
- Before any vector search (context_for_prompt, recall, search_events) the
  caller must call drain() to flush pending embeds — this ensures causal
  consistency so searches never miss artifacts that were logically prior.
- At end-of-day, daily_cycle() calls drain() once before the checkpoint write.

Usage in flow.py
----------------
    # __init__
    from embed_worker import EmbedWorker
    self._embed_worker = EmbedWorker(self._mem)
    self._embed_worker.start()

    # replacing _embed_and_count
    def _embed_and_count(self, **kwargs):
        self._embed_worker.enqueue(**kwargs)
        self.state.daily_artifacts_created += 1

    # before any vector search or at end-of-day
    self._embed_worker.drain()

    # after simulation completes
    self._embed_worker.stop()

Thread safety
-------------
- Queue is thread-safe by design.
- mem.embed_artifact() writes to MongoDB via PyMongo, which is thread-safe.
- daily_artifacts_created is incremented on the main thread (in enqueue),
  so counts remain accurate without locking.
- _errors is guarded by _errors_lock for concurrent appends from the pool.
"""

from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from queue import Queue, Empty
from typing import Any, Dict, List

logger = logging.getLogger("orgforge.embed_worker")

_SENTINEL = None

# How many embed HTTP calls to run in parallel.
# - Infinity:  8-16 is a good starting point on a Xeon 6975P; Infinity's
#              dynamic batching coalesces concurrent requests server-side.
# - Ollama and gateway-backed OpenAI providers: keep low to avoid provider
#              retry storms during end-of-day drains.
_PROVIDER_DEFAULT_CONCURRENCY = {
    "ollama": 1,
    "openai_labelbox": 1,
}
_DEFAULT_CONCURRENCY = int(
    os.environ.get(
        "EMBED_WORKER_CONCURRENCY",
        str(_PROVIDER_DEFAULT_CONCURRENCY.get(os.environ.get("EMBED_PROVIDER"), 8)),
    )
)


class EmbedWorker:
    """
    Concurrent background worker that drains an embed task queue using a
    thread pool. Works with both Ollama (concurrency=1) and Infinity
    (concurrency=8+).

    Parameters
    ----------
    mem : Memory
        The shared Memory instance. _embed() is called on it from the worker
        threads — PyMongo handles connection pooling safely.
    concurrency : int
        Number of concurrent embed calls. Set via EMBED_WORKER_CONCURRENCY
        env var or passed directly. Default: 8.
    maxsize : int
        Maximum queue depth before enqueue() blocks the caller. Default 0
        (unbounded) is correct for OrgForge since the LLM is always slower
        than embedding.
    """

    def __init__(self, mem, concurrency: int = _DEFAULT_CONCURRENCY, maxsize: int = 0):
        self._mem = mem
        self._concurrency = concurrency
        self._queue: Queue[Dict[str, Any] | None] = Queue(maxsize=maxsize)
        self._executor = ThreadPoolExecutor(
            max_workers=concurrency,
            thread_name_prefix="embed-pool",
        )
        self._dispatcher = threading.Thread(
            target=self._dispatch_loop,
            name="embed-dispatcher",
            daemon=True,
        )
        self._futures: List[Future] = []
        self._futures_lock = threading.Lock()
        self._errors: list[Exception] = []
        self._errors_lock = threading.Lock()

    def start(self) -> None:
        """Start the background dispatcher thread. Call once from Flow.__init__."""
        self._dispatcher.start()
        logger.info(
            f"[embed_worker] Background embed queue started "
            f"(concurrency={self._concurrency})."
        )

    def stop(self) -> None:
        """
        Flush remaining tasks then shut down cleanly.
        Call after the simulation loop exits.
        """
        self.drain()
        self._queue.put(_SENTINEL)
        self._dispatcher.join(timeout=60)
        self._executor.shutdown(wait=True, cancel_futures=False)
        if self._dispatcher.is_alive():
            logger.warning("[embed_worker] Dispatcher thread did not exit within 60s.")
        else:
            logger.info("[embed_worker] Background embed queue stopped cleanly.")

    def enqueue(self, **kwargs) -> None:
        """
        Non-blocking enqueue of an embed task.

        Accepts the same keyword arguments as Memory.embed_artifact():
            id, type, title, content, day, date, timestamp, metadata
        Plus the internal routing key:
            _target: "artifacts" (default) or "events"
        """
        self._queue.put(kwargs)

    def drain(self) -> None:
        """
        Block until all currently queued and in-flight embed tasks are complete.

        Call this:
          - Before any vector search (recall, context_for_prompt, search_events)
          - At end-of-day before the checkpoint write
          - Before the simulation's final report

        After drain() returns, MongoDB is consistent with all enqueued artifacts.
        Any errors accumulated during background processing are logged here.
        """
        # Wait for queue to be fully dispatched to the thread pool
        self._queue.join()

        # Wait for all in-flight futures (tasks running in the pool right now)
        with self._futures_lock:
            futures_snapshot = list(self._futures)

        for fut in futures_snapshot:
            try:
                fut.result()
            except Exception as exc:
                with self._errors_lock:
                    self._errors.append(exc)

        with self._futures_lock:
            self._futures.clear()

        if self._errors:
            with self._errors_lock:
                for err in self._errors:
                    logger.error(f"[embed_worker] Background embed error: {err}")
                self._errors.clear()

    def _dispatch_loop(self) -> None:
        """
        Dispatcher thread body. Pulls tasks off the queue and submits them to
        the thread pool. Runs until it receives the sentinel value.
        """
        while True:
            try:
                task = self._queue.get(block=True, timeout=5)
            except Empty:
                continue

            if task is _SENTINEL:
                self._queue.task_done()
                break

            future = self._executor.submit(self._process_task, task)
            with self._futures_lock:
                # Prune completed futures to avoid unbounded list growth
                self._futures = [f for f in self._futures if not f.done()]
                self._futures.append(future)

            # Mark the queue slot as done immediately after dispatch —
            # drain() waits on futures directly for in-flight completion.
            self._queue.task_done()

    def _process_task(self, task: Dict[str, Any]) -> None:
        """
        Executed in a pool thread. Calls the embedder and writes to MongoDB.
        This is where actual HTTP calls to Infinity/Ollama happen.
        """
        try:
            target = task.pop("_target", "artifacts")

            if target == "events":
                text = task["content"]
                vector = self._mem._embed(
                    text,
                    input_type="search_document",
                    caller="log_event_async",
                    doc_id=task["id"],
                    doc_type=task["type"],
                )
                if vector:
                    self._mem._events.update_one(
                        {"_id": task["id"]},
                        {"$set": {"embedding": vector}},
                    )
            else:
                embed_text = task["content"]
                vector = self._mem._embed(
                    embed_text,
                    input_type="search_document",
                    caller="embed_artifact_async",
                    doc_id=task["id"],
                    doc_type=task["type"],
                )
                if vector:
                    self._mem._artifacts.update_one(
                        {"_id": task["id"]},
                        {"$set": {"embedding": vector}},
                    )
        except Exception as exc:
            with self._errors_lock:
                self._errors.append(exc)
            logger.warning(
                f"[embed_worker] embed failed for id={task.get('id')}: {exc}"
            )
