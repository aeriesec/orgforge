"""
embed_worker.py
===============
Background embedding queue for OrgForge.

Decouples artifact embedding from LLM generation so Stella/Ollama inference
runs while the next Bedrock call is in flight, rather than blocking between
each generation step.

Architecture
------------
- A single daemon thread consumes embed tasks from a Queue.
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
"""

from __future__ import annotations

import logging
import threading
from queue import Queue, Empty
from typing import Any, Dict

logger = logging.getLogger("orgforge.embed_worker")

_SENTINEL = None  # signals the consumer thread to exit cleanly


class EmbedWorker:
    """
    Single background thread that drains an embed task queue.

    Parameters
    ----------
    mem : Memory
        The shared Memory instance. embed_artifact() is called on it from the
        worker thread — PyMongo handles connection pooling safely.
    maxsize : int
        Maximum queue depth before enqueue() blocks the caller. Default 0
        (unbounded) is correct for OrgForge since the LLM is always slower
        than embedding and we never want the sim to stall on the queue.
    """

    def __init__(self, mem, maxsize: int = 0):
        self._mem = mem
        self._queue: Queue[Dict[str, Any] | None] = Queue(maxsize=maxsize)
        self._thread = threading.Thread(
            target=self._consume,
            name="embed-worker",
            daemon=True,  # exits automatically if main thread dies
        )
        self._errors: list[Exception] = []  # accumulated — surfaced at drain()

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background consumer thread. Call once from Flow.__init__."""
        self._thread.start()
        logger.info("[embed_worker] Background embed queue started.")

    def stop(self) -> None:
        """
        Flush remaining tasks then shut down the consumer thread cleanly.
        Call after the simulation loop exits.
        """
        self.drain()
        self._queue.put(_SENTINEL)
        self._thread.join(timeout=60)
        if self._thread.is_alive():
            logger.warning("[embed_worker] Consumer thread did not exit within 60s.")
        else:
            logger.info("[embed_worker] Background embed queue stopped cleanly.")

    # ── Public API ─────────────────────────────────────────────────────────────

    def enqueue(self, **kwargs) -> None:
        """
        Non-blocking enqueue of an embed task.

        Accepts the same keyword arguments as Memory.embed_artifact():
            id, type, title, content, day, date, timestamp, metadata
        """
        self._queue.put(kwargs)

    def drain(self) -> None:
        """
        Block until all currently queued embed tasks are complete.

        Call this:
          - Before any vector search (recall, context_for_prompt, search_events)
          - At end-of-day before the checkpoint write
          - Before the simulation's final report

        After drain() returns, MongoDB is consistent with all enqueued artifacts.
        Any errors accumulated during background processing are logged here.
        """
        self._queue.join()  # blocks until task_done() called for every item

        if self._errors:
            for err in self._errors:
                logger.error(f"[embed_worker] Background embed error: {err}")
            self._errors.clear()

    # ── Consumer ───────────────────────────────────────────────────────────────

    def _consume(self) -> None:
        """
        Worker thread body. Runs until it receives the sentinel value.
        Each task is a kwargs dict for mem.embed_artifact().
        """
        while True:
            try:
                task = self._queue.get(block=True, timeout=5)
            except Empty:
                continue

            if task is _SENTINEL:
                self._queue.task_done()
                break

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
                self._errors.append(exc)
                logger.warning(
                    f"[embed_worker] embed failed for id={task.get('id')}: {exc}"
                )
            finally:
                self._queue.task_done()
