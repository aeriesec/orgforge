from types import SimpleNamespace
from unittest.mock import MagicMock


def test_openai_labelbox_config_parses_quoted_default_headers(monkeypatch):
    from provider_config import (
        openai_labelbox_config,
        openai_labelbox_litellm_model,
        openai_labelbox_litellm_params,
    )

    headers = (
        '\'{"x-labelbox-context":"{\\"tag\\":\\"foundry_litellm_test_openai\\",'
        '\\"project_id\\":\\"test_cuid\\"}"}\''
    )
    monkeypatch.setenv("OPENAI_LABELBOX_API_KEY", "lb-key")
    monkeypatch.setenv(
        "OPENAI_LABELBOX_BASE_URL",
        "https://models.labelbox.com/api/v1/models/litellm/v1",
    )
    monkeypatch.setenv("OPENAI_LABELBOX_DEFAULT_HEADERS", headers)

    cfg = openai_labelbox_config()

    assert cfg.api_key == "lb-key"
    assert cfg.base_url == "https://models.labelbox.com/api/v1/models/litellm/v1"
    assert cfg.default_headers == {
        "x-labelbox-context": '{"tag":"foundry_litellm_test_openai","project_id":"test_cuid"}'
    }
    assert openai_labelbox_litellm_model("openai/gpt-4o") == "openai/openai/gpt-4o"
    assert openai_labelbox_litellm_params()["extra_headers"] == cfg.default_headers


def test_openai_labelbox_embedder_uses_api_key_base_url_and_default_headers(
    monkeypatch,
):
    import sys
    import memory

    fake_response = SimpleNamespace(
        data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3, 0.4])]
    )
    fake_client = MagicMock()
    fake_client.embeddings.create.return_value = fake_response
    fake_openai_cls = MagicMock(return_value=fake_client)

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=fake_openai_cls))
    monkeypatch.setenv("OPENAI_LABELBOX_API_KEY", "lb-key")
    monkeypatch.setenv(
        "OPENAI_LABELBOX_BASE_URL",
        "https://models.labelbox.com/api/v1/models/litellm/v1",
    )
    monkeypatch.setenv(
        "OPENAI_LABELBOX_DEFAULT_HEADERS",
        '{"x-labelbox-context":"{\\"tag\\":\\"t\\",\\"project_id\\":\\"p\\"}"}',
    )

    embedder = memory.OpenAILabelboxEmbedder(
        model="openai/text-embedding-3-large",
        dims=4,
    )
    vector = embedder.embed("incident response notes", input_type="search_query")

    fake_openai_cls.assert_called_once_with(
        api_key="lb-key",
        base_url="https://models.labelbox.com/api/v1/models/litellm/v1",
        default_headers={
            "x-labelbox-context": '{"tag":"t","project_id":"p"}',
        },
        max_retries=0,
        timeout=30.0,
    )
    fake_client.embeddings.create.assert_called_once_with(
        model="openai/text-embedding-3-large",
        input="incident response notes",
        dimensions=4,
        timeout=30.0,
    )
    assert vector == [0.1, 0.2, 0.3, 0.4]
