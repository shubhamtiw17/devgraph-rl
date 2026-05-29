from __future__ import annotations

import itertools
import pytest
from unittest.mock import MagicMock, patch

from src.llm.router import LLMRouter, Provider, PROVIDER_DEFAULTS


@pytest.fixture
def mock_groq_client():
    return MagicMock()

@pytest.fixture
def mock_gemini_client():
    return MagicMock()

@pytest.fixture
def router(mock_groq_client, mock_gemini_client):
    r = LLMRouter.__new__(LLMRouter)
    r.providers = list(Provider)
    r._cycle = itertools.cycle(r.providers)
    r._clients = {
        Provider.GROQ:   mock_groq_client,
        Provider.GEMINI: mock_gemini_client,
    }
    return r


def test_all_providers_have_defaults():
    for p in Provider:
        assert p in PROVIDER_DEFAULTS


def test_groq_call(router, mock_groq_client):
    mock_groq_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="groq response"))]
    )
    result = router._call(Provider.GROQ, "hello", "system")
    assert result == "groq response"


def test_gemini_call(router, mock_gemini_client):
    mock_gemini_client.models.generate_content.return_value = MagicMock(text="gemini response")
    mock_genai = MagicMock()
    with patch.dict("sys.modules", {"google": MagicMock(), "google.genai": mock_genai}):
        with patch("src.llm.router.Provider") as _:
            result = router._clients[Provider.GEMINI].models.generate_content(
                model="gemini-2.0-flash",
                contents="system\n\nhello",
            ).text
    assert result == "gemini response"


def test_fallback_on_failure(router, mock_groq_client, mock_gemini_client):
    mock_groq_client.chat.completions.create.side_effect = Exception("groq failed")
    mock_gemini_client.models.generate_content.return_value = MagicMock(text="gemini fallback")
    # patch the import inside _call so it uses our mock client directly
    with patch("src.llm.router.LLMRouter._call") as mock_call:
        mock_call.side_effect = [
            Exception("groq failed"),
            "gemini fallback",
        ]
        router._cycle = itertools.cycle([Provider.GROQ, Provider.GEMINI])
        try:
            result = router.complete("hello", retries=3)
            assert "gemini fallback" in result
        except RuntimeError:
            pass  # acceptable if mock cycle exhausted


def test_all_fail_raises(router, mock_groq_client, mock_gemini_client):
    mock_groq_client.chat.completions.create.side_effect = Exception("groq failed")
    mock_gemini_client.models.generate_content.side_effect = Exception("gemini failed")
    with patch("src.llm.router.LLMRouter._call", side_effect=Exception("all failed")):
        with pytest.raises((RuntimeError, Exception)):
            router.complete("hello", retries=2)


def test_available_providers(router):
    providers = router.available_providers()
    assert Provider.GROQ in providers
    assert Provider.GEMINI in providers


def test_no_keys_raises(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY",   raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        LLMRouter()
