"""
Lightweight LLM client for grounding-internal work (profile extraction, topical
relevance scoring).

Uses the `grounding_worker` model from the active preset (defaults to a small,
cheap model) so we don't burn the main `worker` budget on short structural
summarisation tasks.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

logger = logging.getLogger("orgforge.grounding.llm_client")


class GroundingLLM:
    """Thin wrapper around LiteLLM. Constructed lazily so importing this module
    doesn't require the LiteLLM/CrewAI deps if grounding is disabled."""

    def __init__(self, model: Optional[str] = None):
        self._model_override = model
        self._litellm = None
        self._provider_params: dict[str, Any] = {}
        self._model: Optional[str] = None
        self._reasoning_effort: Optional[str] = None

    def _ensure_ready(self) -> None:
        if self._litellm is not None:
            return
        # Resolve model from active preset
        try:
            from config_loader import _PRESET, _PROVIDER  # type: ignore
            from provider_config import (
                openai_labelbox_litellm_model,
                openai_labelbox_litellm_params,
            )
        except Exception as exc:
            raise RuntimeError(
                "GroundingLLM requires OrgForge's config_loader to be importable"
            ) from exc

        raw_model = (
            self._model_override
            or _PRESET.get("grounding_worker")
            or _PRESET.get("worker")
        )
        if not raw_model:
            raise RuntimeError("No grounding_worker or worker model in preset")

        # OpenAI/Labelbox needs the LiteLLM provider prefix added
        if _PROVIDER == "openai_labelbox":
            self._model = openai_labelbox_litellm_model(raw_model)
            self._provider_params = openai_labelbox_litellm_params()
        elif _PROVIDER == "bedrock":
            self._model = raw_model
            self._provider_params = {}
        else:
            self._model = raw_model
            self._provider_params = {}

        # Reasoning models (gpt-5+) accept reasoning_effort and reject temperature
        model_lc = (raw_model or "").lower()
        if "gpt-5" in model_lc or model_lc.startswith("o1") or model_lc.startswith("o3"):
            # Defaults to "high" so all grounding LLM calls use the same
            # reasoning level as the main worker. Override per-preset with
            # `grounding_reasoning_effort` if you want it cheaper.
            self._reasoning_effort = _PRESET.get(
                "grounding_reasoning_effort", "high"
            )
        else:
            self._reasoning_effort = None

        import litellm  # type: ignore
        self._litellm = litellm

    def complete_json(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1500,
        temperature: float = 0.2,
    ) -> dict:
        """Make a single completion call expecting JSON output. Returns the
        parsed dict, or raises on parse failure."""
        self._ensure_ready()
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **self._provider_params,
        }
        if self._reasoning_effort:
            kwargs["reasoning_effort"] = self._reasoning_effort
            kwargs["max_completion_tokens"] = max_tokens
            # LiteLLM's static OpenAI schema lags behind the API for gpt-5
            # reasoning params; whitelist explicitly so the param flows
            # through.
            kwargs["allowed_openai_params"] = ["reasoning_effort"]
        else:
            kwargs["temperature"] = temperature
            kwargs["max_tokens"] = max_tokens

        # Many providers honour response_format; safe to pass on OpenAI route.
        kwargs["response_format"] = {"type": "json_object"}

        resp = self._litellm.completion(**kwargs)
        text = resp["choices"][0]["message"]["content"]
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            logger.warning("[grounding_llm] JSON parse failed; retrying naive extract")
            # Last-ditch: find the first {...} block
            start, end = text.find("{"), text.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(text[start : end + 1])
            raise RuntimeError(f"GroundingLLM JSON parse failed: {text[:200]}") from exc
