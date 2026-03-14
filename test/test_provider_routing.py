"""Tests for _resolve_endpoint() provider routing logic."""

from nerif.utils.utils import _resolve_endpoint


def test_openai_model_routing():
    base_url, api_key, model, provider = _resolve_endpoint("gpt-4o")
    assert provider == "openai"
    assert model == "gpt-4o"
    assert base_url == "https://api.openai.com/v1"


def test_anthropic_prefix():
    base_url, api_key, model, provider = _resolve_endpoint("anthropic/claude-3-opus")
    assert provider == "anthropic"
    assert model == "claude-3-opus"
    assert base_url == "https://api.anthropic.com"


def test_gemini_prefix():
    base_url, api_key, model, provider = _resolve_endpoint("gemini/gemini-pro")
    assert provider == "gemini"
    assert model == "gemini-pro"
    assert base_url == "https://generativelanguage.googleapis.com"


def test_ollama_prefix():
    base_url, api_key, model, provider = _resolve_endpoint("ollama/llama3")
    assert provider == "openai_compat"
    assert model == "llama3"
    assert "11434" in base_url


def test_vllm_prefix():
    base_url, api_key, model, provider = _resolve_endpoint("vllm/mistral-7b")
    assert provider == "openai_compat"
    assert model == "mistral-7b"
    assert "8000" in base_url


def test_sllm_prefix():
    base_url, api_key, model, provider = _resolve_endpoint("sllm/phi-3")
    assert provider == "openai_compat"
    assert model == "phi-3"
    assert "8343" in base_url


def test_openrouter_prefix():
    base_url, api_key, model, provider = _resolve_endpoint("openrouter/meta/llama-3")
    assert provider == "openai_compat"
    assert model == "meta/llama-3"
    assert base_url == "https://openrouter.ai/api/v1"


def test_custom_openai_prefix():
    base_url, api_key, model, provider = _resolve_endpoint("custom_openai/my-model")
    assert provider == "openai_compat"
    assert model == "my-model"


def test_unknown_model_defaults_to_openai():
    base_url, api_key, model, provider = _resolve_endpoint("some-unknown-model")
    assert provider == "openai"
    assert model == "some-unknown-model"
    assert base_url == "https://api.openai.com/v1"


def test_custom_api_key_and_base_url_overrides():
    base_url, api_key, model, provider = _resolve_endpoint(
        "gpt-4o",
        api_key="sk-custom-key",
        base_url="https://my-proxy.example.com/v1",
    )
    assert api_key == "sk-custom-key"
    assert base_url == "https://my-proxy.example.com/v1"
    assert model == "gpt-4o"
    assert provider == "openai"

    # Also verify overrides work for prefixed providers
    base_url, api_key, model, provider = _resolve_endpoint(
        "anthropic/claude-3-opus",
        api_key="sk-ant-custom",
        base_url="https://my-anthropic-proxy.example.com",
    )
    assert api_key == "sk-ant-custom"
    assert base_url == "https://my-anthropic-proxy.example.com"
    assert provider == "anthropic"
