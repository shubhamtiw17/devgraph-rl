from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from src.memory.embedder import (
    MiniLMEmbedder,
    GeminiEmbedder,
    CohereEmbedder,
    get_embedder,
    available_embedders,
    EMBEDDER_REGISTRY,
)
from src.memory.vector_store import VectorStore, SearchResult, CompareResult
from src.memory.memory_manager import MemoryManager
from src.agents.base_agent import AgentResult


# ── Fixtures ─────────────────────────────────────────────────────────────────

def make_fake_embedder(name: str, dim: int) -> MagicMock:
    emb = MagicMock()
    emb.name = name
    emb.dim = dim
    emb.is_available.return_value = True

    rng = np.random.default_rng(seed=abs(hash(name)) % (2**32))

    def fake_encode(texts):
        vecs = rng.random((len(texts), dim)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms

    def fake_encode_one(text):
        return fake_encode([text])[0]

    emb.encode.side_effect = fake_encode
    emb.encode_one.side_effect = fake_encode_one
    return emb


@pytest.fixture()
def fake_store(tmp_path):
    store = VectorStore.__new__(VectorStore)
    from src.memory.vector_store import EmbedderIndex
    store._indexes = {
        "minilm": EmbedderIndex(make_fake_embedder("minilm", 384), tmp_path),
        "gemini": EmbedderIndex(make_fake_embedder("gemini", 768), tmp_path),
        "cohere": EmbedderIndex(make_fake_embedder("cohere", 384), tmp_path),
    }
    return store


@pytest.fixture()
def fake_manager(fake_store):
    return MemoryManager(store=fake_store)


def make_result(output: str = "ok", success: bool = True) -> AgentResult:
    return AgentResult(agent_name="test_agent", output=output, success=success)


# ── Embedder unit tests ───────────────────────────────────────────────────────

class TestMiniLMEmbedder:
    def test_name_and_dim(self):
        emb = MiniLMEmbedder()
        assert emb.name == "minilm"
        assert emb.dim == 384

    def test_is_available(self):
        emb = MiniLMEmbedder()
        assert emb.is_available() is True

    def test_encode_empty(self):
        emb = MiniLMEmbedder()
        result = emb.encode([])
        assert result.shape == (0, 384)

    def test_encode_returns_correct_shape(self):
        emb = MiniLMEmbedder()
        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_model.encode.return_value = np.ones((2, 384), dtype=np.float32)
            mock_st.return_value = mock_model
            emb._model = mock_model
            result = emb.encode(["hello", "world"])
            assert result.shape == (2, 384)
            assert result.dtype == np.float32

    def test_encode_one_returns_1d(self):
        emb = MiniLMEmbedder()
        with patch.object(emb, "encode", return_value=np.ones((1, 384), dtype=np.float32)):
            result = emb.encode_one("hello")
            assert result.shape == (384,)


class TestGeminiEmbedder:
    def test_name_and_dim(self):
        emb = GeminiEmbedder()
        assert emb.name == "gemini"
        assert emb.dim == 768

    def test_is_available_false_without_key(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        emb = GeminiEmbedder()
        assert emb.is_available() is False

    def test_is_available_true_with_key(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        emb = GeminiEmbedder()
        assert emb.is_available() is True

    def test_encode_empty(self):
        emb = GeminiEmbedder()
        result = emb.encode([])
        assert result.shape == (0, 768)

    def test_encode_calls_api(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        emb = GeminiEmbedder()
        mock_genai = MagicMock()
        mock_genai.embed_content.return_value = {"embedding": [0.1] * 768}
        emb._client = mock_genai
        result = emb.encode(["test text"])
        assert result.shape == (1, 768)
        assert result.dtype == np.float32


class TestCohereEmbedder:
    def test_name_and_dim(self):
        emb = CohereEmbedder()
        assert emb.name == "cohere"
        assert emb.dim == 384

    def test_is_available_false_without_key(self, monkeypatch):
        monkeypatch.delenv("COHERE_API_KEY", raising=False)
        emb = CohereEmbedder()
        assert emb.is_available() is False

    def test_is_available_true_with_key(self, monkeypatch):
        monkeypatch.setenv("COHERE_API_KEY", "fake-key")
        emb = CohereEmbedder()
        assert emb.is_available() is True

    def test_encode_empty(self):
        emb = CohereEmbedder()
        result = emb.encode([])
        assert result.shape == (0, 384)

    def test_encode_calls_api(self, monkeypatch):
        monkeypatch.setenv("COHERE_API_KEY", "fake-key")
        emb = CohereEmbedder()
        mock_client = MagicMock()
        mock_client.embed.return_value = MagicMock(embeddings=[[0.1] * 384])
        emb._client = mock_client
        result = emb.encode(["test text"])
        assert result.shape == (1, 384)
        assert result.dtype == np.float32


class TestEmbedderRegistry:
    def test_get_embedder_minilm(self):
        emb = get_embedder("minilm")
        assert emb.name == "minilm"

    def test_get_embedder_gemini(self):
        emb = get_embedder("gemini")
        assert emb.name == "gemini"

    def test_get_embedder_cohere(self):
        emb = get_embedder("cohere")
        assert emb.name == "cohere"

    def test_get_embedder_invalid(self):
        with pytest.raises(ValueError, match="Unknown embedder"):
            get_embedder("unknown")

    def test_available_embedders_returns_list(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "fake")
        monkeypatch.setenv("COHERE_API_KEY", "fake")
        result = available_embedders()
        assert isinstance(result, list)
        assert "minilm" in result


# ── VectorStore tests ─────────────────────────────────────────────────────────

class TestVectorStore:
    def test_initial_size_zero(self, fake_store):
        assert fake_store.size("minilm") == 0
        assert fake_store.size("gemini") == 0
        assert fake_store.size("cohere") == 0

    def test_add_increases_size(self, fake_store):
        fake_store.add("hello world", "minilm")
        assert fake_store.size("minilm") == 1
        assert fake_store.size("gemini") == 0

    def test_add_batch(self, fake_store):
        fake_store.add_batch(["text one", "text two", "text three"], "cohere")
        assert fake_store.size("cohere") == 3

    def test_search_returns_results(self, fake_store):
        fake_store.add("add error handling to payment", "minilm")
        fake_store.add("refactor database layer", "minilm")
        results = fake_store.search("handle exceptions", "minilm", top_k=2)
        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)

    def test_search_empty_index_returns_empty(self, fake_store):
        results = fake_store.search("anything", "minilm", top_k=5)
        assert results == []

    def test_search_result_has_embedder_name(self, fake_store):
        fake_store.add("test text", "gemini")
        results = fake_store.search("test", "gemini", top_k=1)
        assert results[0].embedder_name == "gemini"

    def test_search_result_has_text(self, fake_store):
        fake_store.add("unique text content", "minilm")
        results = fake_store.search("unique text content", "minilm", top_k=1)
        assert results[0].text == "unique text content"

    def test_search_respects_top_k(self, fake_store):
        for i in range(10):
            fake_store.add(f"text number {i}", "minilm")
        results = fake_store.search("text", "minilm", top_k=3)
        assert len(results) == 3

    def test_search_all_returns_compare_result(self, fake_store):
        fake_store.add("test memory", "minilm")
        fake_store.add("test memory", "gemini")
        fake_store.add("test memory", "cohere")
        result = fake_store.search_all("test", top_k=1)
        assert isinstance(result, CompareResult)
        assert set(result.results.keys()) == {"minilm", "gemini", "cohere"}

    def test_search_all_empty_indexes(self, fake_store):
        result = fake_store.search_all("anything", top_k=5)
        assert result.results["minilm"] == []
        assert result.results["gemini"] == []
        assert result.results["cohere"] == []

    def test_clear_resets_size(self, fake_store):
        fake_store.add("text", "minilm")
        assert fake_store.size("minilm") == 1
        fake_store.clear("minilm")
        assert fake_store.size("minilm") == 0

    def test_status_returns_all_embedders(self, fake_store):
        status = fake_store.status()
        assert set(status.keys()) == {"minilm", "gemini", "cohere"}
        assert "size" in status["minilm"]
        assert "dim" in status["minilm"]
        assert "available" in status["minilm"]

    def test_sync_to_all(self, fake_store):
        fake_store.add_batch(["text a", "text b"], "minilm")
        counts = fake_store.sync_to_all("minilm")
        assert counts["gemini"] == 2
        assert counts["cohere"] == 2
        assert fake_store.size("gemini") == 2
        assert fake_store.size("cohere") == 2

    def test_add_with_metadata(self, fake_store):
        fake_store.add("text", "minilm", metadata={"tag": "test"})
        results = fake_store.search("text", "minilm", top_k=1)
        assert results[0].metadata["tag"] == "test"


# ── MemoryManager tests ───────────────────────────────────────────────────────

class TestMemoryManager:
    def test_store_increases_size(self, fake_manager):
        fake_manager.store(
            task="fix the login bug",
            result=make_result("patched auth module"),
            agent_type="coding",
            repo_path="/tmp/repo",
            embedder_name="minilm",
        )
        assert fake_manager.size("minilm") == 1

    def test_store_to_all_populates_all_indexes(self, fake_manager):
        fake_manager.store_to_all(
            task="refactor payment service",
            result=make_result("extracted PaymentProcessor class"),
            agent_type="coding",
            repo_path="/tmp/repo",
        )
        assert fake_manager.size("minilm") == 1
        assert fake_manager.size("gemini") == 1
        assert fake_manager.size("cohere") == 1

    def test_retrieve_returns_results(self, fake_manager):
        fake_manager.store(
            task="add error handling to payment",
            result=make_result("wrapped in try/except"),
            agent_type="coding",
            repo_path="/tmp/repo",
            embedder_name="minilm",
        )
        results = fake_manager.retrieve("handle exceptions", embedder_name="minilm", top_k=1)
        assert len(results) == 1
        assert "add error handling" in results[0].text

    def test_retrieve_empty_returns_empty(self, fake_manager):
        results = fake_manager.retrieve("anything", embedder_name="minilm")
        assert results == []

    def test_retrieve_filters_by_agent_type(self, fake_manager):
        fake_manager.store(
            task="plan the refactor",
            result=make_result("step 1, step 2"),
            agent_type="planner",
            repo_path="/tmp/repo",
            embedder_name="minilm",
        )
        fake_manager.store(
            task="write the refactor code",
            result=make_result("def refactor():"),
            agent_type="coding",
            repo_path="/tmp/repo",
            embedder_name="minilm",
        )
        results = fake_manager.retrieve(
            "refactor", embedder_name="minilm",
            top_k=5, agent_type="planner"
        )
        assert all(r.metadata["agent_type"] == "planner" for r in results)

    def test_retrieve_filters_by_repo_path(self, fake_manager):
        fake_manager.store(
            task="fix repo A bug",
            result=make_result("fixed"),
            agent_type="coding",
            repo_path="/tmp/repo_a",
            embedder_name="minilm",
        )
        fake_manager.store(
            task="fix repo B bug",
            result=make_result("fixed"),
            agent_type="coding",
            repo_path="/tmp/repo_b",
            embedder_name="minilm",
        )
        results = fake_manager.retrieve(
            "fix bug", embedder_name="minilm",
            top_k=5, repo_path="/tmp/repo_a"
        )
        assert all(r.metadata["repo_path"] == "/tmp/repo_a" for r in results)

    def test_retrieve_all_returns_compare_result(self, fake_manager):
        fake_manager.store_to_all(
            task="add logging",
            result=make_result("added logger"),
            agent_type="coding",
            repo_path="/tmp/repo",
        )
        result = fake_manager.retrieve_all("logging", top_k=1)
        assert isinstance(result, CompareResult)
        assert len(result.results["minilm"]) == 1
        assert len(result.results["gemini"]) == 1
        assert len(result.results["cohere"]) == 1

    def test_metadata_contains_required_fields(self, fake_manager):
        fake_manager.store(
            task="test task",
            result=make_result(),
            agent_type="planner",
            repo_path="/tmp/repo",
            embedder_name="minilm",
        )
        results = fake_manager.retrieve("test", embedder_name="minilm", top_k=1)
        meta = results[0].metadata
        assert "agent_type" in meta
        assert "repo_path" in meta
        assert "timestamp" in meta
        assert "related_files" in meta

    def test_metadata_related_files(self, fake_manager):
        fake_manager.store(
            task="refactor payment",
            result=make_result(),
            agent_type="coding",
            repo_path="/tmp/repo",
            embedder_name="minilm",
            related_files=["payment.py", "checkout.py"],
        )
        results = fake_manager.retrieve("payment", embedder_name="minilm", top_k=1)
        assert results[0].metadata["related_files"] == ["payment.py", "checkout.py"]

    def test_status_returns_dict(self, fake_manager):
        status = fake_manager.status()
        assert isinstance(status, dict)
        assert "minilm" in status

    def test_sync_to_all(self, fake_manager):
        fake_manager.store(
            task="task a",
            result=make_result(),
            agent_type="coding",
            repo_path="/tmp/repo",
            embedder_name="minilm",
        )
        counts = fake_manager.sync_to_all("minilm")
        assert counts["gemini"] == 1
        assert counts["cohere"] == 1