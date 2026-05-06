"""
Shared model-provider configuration helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Dict


OPENAI_LABELBOX_API_KEY_ENV = "OPENAI_LABELBOX_API_KEY"
OPENAI_LABELBOX_BASE_URL_ENV = "OPENAI_LABELBOX_BASE_URL"
OPENAI_LABELBOX_DEFAULT_HEADERS_ENV = "OPENAI_LABELBOX_DEFAULT_HEADERS"

OPENAI_LABELBOX_DEFAULT_BASE_URL = (
    "https://models.labelbox.com/api/v1/models/litellm/v1"
)


@dataclass(frozen=True)
class OpenAILabelboxConfig:
    api_key: str
    base_url: str
    default_headers: Dict[str, str]


def _parse_json_headers(raw: str, env_name: str) -> Dict[str, str]:
    if not raw:
        return {}

    raw = raw.strip()
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {"'", '"'}:
        raw = raw[1:-1]

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{env_name} must be a JSON object") from exc

    if not isinstance(parsed, dict):
        raise ValueError(f"{env_name} must be a JSON object")

    return {str(key): str(value) for key, value in parsed.items()}


def openai_labelbox_config() -> OpenAILabelboxConfig:
    """
    Return the OpenAI-compatible Labelbox gateway settings.

    The default base URL matches the public Labelbox LiteLLM-compatible endpoint,
    while the API key and context headers are expected to come from the runtime
    environment.
    """
    return OpenAILabelboxConfig(
        api_key=os.environ.get(OPENAI_LABELBOX_API_KEY_ENV, ""),
        base_url=os.environ.get(
            OPENAI_LABELBOX_BASE_URL_ENV, OPENAI_LABELBOX_DEFAULT_BASE_URL
        ),
        default_headers=_parse_json_headers(
            os.environ.get(OPENAI_LABELBOX_DEFAULT_HEADERS_ENV, ""),
            OPENAI_LABELBOX_DEFAULT_HEADERS_ENV,
        ),
    )


def openai_labelbox_litellm_model(labelbox_model: str) -> str:
    """
    LiteLLM consumes the first path component as the provider prefix.

    Labelbox model names already include their upstream provider, e.g.
    ``openai/gpt-4o``. Prefixing once more makes LiteLLM route via OpenAI while
    preserving the original Labelbox model name in the request body.
    """
    return f"openai/{labelbox_model.strip()}"


def openai_labelbox_litellm_params() -> dict:
    cfg = openai_labelbox_config()
    return {
        "api_key": cfg.api_key,
        "base_url": cfg.base_url,
        "extra_headers": cfg.default_headers,
    }
