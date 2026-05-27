
import pytest
from unittest.mock import MagicMock, patch
from src.llm.router import LLMRouter, Provider, PROVIDER_DEFAULTS


# ── fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def mock_anthropic_client():
    client = MagicMock()
    msg = MagicMock()
    msg.content = [MagicMock(text="mocked anthropic response")]
    client.messages.create.return_value = msg
    return client


@pytest.fixture
def mock_groq_client():
    client = MagicMock()
    choice = MagicMock()
    choice.message.content = "mocked groq response"
    client.chat.completions.create.return_value = MagicMock(choices=[choice])
    return client


@pytest.fixture
def mock_gemini_client():
    client = MagicMock()
    client.generate_content.return_value = MagicMock(text="mocked gemini response")
    return client


@pytest.fixture
def router(mock_anthropic_client, mock_groq_client, mock_gemini_client):
    r = LLMRouter.__new__(LLMRouter)
    r.providers = list(Provider)
    import itertools
    r._cycle = itertools.cycle(r.providers)
    r._clients = {
        Provider.ANTHROPIC: mock_anthropic_client,
        Provider.GROQ:      mock_groq_client,
        Provider.GEMINI:    mock_gemini_client,
    }
    return r


# ── provider defaults ─────────────────────────────────────────────────

def test_all_providers_have_defaults():
    for provider in Provider:
        assert provider in PROVIDER_DEFAULTS
        cfg = PROVIDER_DEFAULTS[provider]
        assert cfg.model
        assert cfg.max_tokens > 0


# ── routing ───────────────────────────────────────────────────────────

def test_anthropic_call(router, mock_anthropic_client):
    result = router.complete("hello", provider=Provider.ANTHROPIC)
    assert result == "mocked anthropic response"
    mock_anthropic_client.messages.create.assert_called_once()


def test_groq_call(router, mock_groq_client):
    result = router.complete("hello", provider=Provider.GROQ)
    assert result == "mocked groq response"
    mock_groq_client.chat.completions.create.assert_called_once()


def test_gemini_call(router, mock_gemini_client):
    result = router.complete("hello", provider=Provider.GEMINI)
    assert result == "mocked gemini response"
    mock_gemini_client.generate_content.assert_called_once()


# ── fallback ──────────────────────────────────────────────────────────

def test_fallback_on_failure(router, mock_anthropic_client, mock_groq_client):
    mock_anthropic_client.messages.create.side_effect = Exception("rate limited")
    result = router.complete("hello", provider=Provider.ANTHROPIC, retries=3)
    # should have fallen back and succeeded
    assert result in ("mocked groq response", "mocked gemini response")


def test_all_fail_raises(router, mock_anthropic_client, mock_groq_client, mock_gemini_client):
    mock_anthropic_client.messages.create.side_effect = Exception("fail")
    mock_groq_client.chat.completions.create.side_effect = Exception("fail")
    mock_gemini_client.generate_content.side_effect = Exception("fail")
    with pytest.raises(RuntimeError, match="All providers failed"):
        router.complete("hello", retries=3)


# ── available providers ───────────────────────────────────────────────

def test_available_providers(router):
    available = router.available_providers()
    assert Provider.ANTHROPIC in available
    assert Provider.GROQ in available
    assert Provider.GEMINI in available


def test_no_keys_raises():
    with patch.dict("os.environ", {
        "ANTHROPIC_API_KEY": "",
        "GROQ_API_KEY": "",
        "GEMINI_API_KEY": "",
    }):
        with pytest.raises(RuntimeError, match="No API keys found"):
            LLMRouter()
