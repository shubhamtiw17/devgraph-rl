from __future__ import annotations

import os
import numpy as np
from abc import ABC, abstractmethod
from typing import List


# ── Base ────────────────────────────────────────────────────────────────────

class BaseEmbedder(ABC):
    @property
    @abstractmethod
    def name(self) -> str:

    @property
    @abstractmethod
    def dim(self) -> int:

    @abstractmethod
    def is_available(self) -> bool:

    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

# ── MiniLM (local) ──────────────────────────────────────────────────────────

class MiniLMEmbedder(BaseEmbedder):
    MODEL_NAME = "all-MiniLM-L6-v2"
    _DIM = 384

    def __init__(self, batch_size: int = 32) -> None:
        self._batch_size = batch_size
        self._model = None  # lazy load

    @property
    def name(self) -> str:
        return "minilm"

    @property
    def dim(self) -> int:
        return self._DIM

    def is_available(self) -> bool:
        try:
            from sentence_transformers import SentenceTransformer  # noqa: F401
            return True
        except ImportError:
            return False

    @property
    def _loaded_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.MODEL_NAME)
        return self._model

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._DIM), dtype=np.float32)
        embeddings = self._loaded_model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.array(embeddings, dtype=np.float32)


# ── Gemini ───────────────────────────────────────────────────────────────────

class GeminiEmbedder(BaseEmbedder):

    MODEL_NAME = "models/text-embedding-004"
    _DIM = 768

    def __init__(self) -> None:
        self._client = None  # lazy

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def dim(self) -> int:
        return self._DIM

    def is_available(self) -> bool:
        return bool(os.getenv("GEMINI_API_KEY"))

    @property
    def _loaded_client(self):
        if self._client is None:
            import google.generativeai as genai
            genai.configure(api_key=os.environ["GEMINI_API_KEY"])
            self._client = genai
        return self._client

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._DIM), dtype=np.float32)

        client = self._loaded_client
        embeddings = []
        for text in texts:
            result = client.embed_content(
                model=self.MODEL_NAME,
                content=text,
                task_type="retrieval_document",
            )
            embeddings.append(result["embedding"])

        return np.array(embeddings, dtype=np.float32)


# ── Cohere ───────────────────────────────────────────────────────────────────

class CohereEmbedder(BaseEmbedder):

    MODEL_NAME = "embed-english-light-v3.0"
    _DIM = 384

    def __init__(self) -> None:
        self._client = None  # lazy

    @property
    def name(self) -> str:
        return "cohere"

    @property
    def dim(self) -> int:
        return self._DIM

    def is_available(self) -> bool:
        return bool(os.getenv("COHERE_API_KEY"))

    @property
    def _loaded_client(self):
        if self._client is None:
            import cohere
            self._client = cohere.Client(api_key=os.environ["COHERE_API_KEY"])
        return self._client

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._DIM), dtype=np.float32)

        response = self._loaded_client.embed(
            texts=texts,
            model=self.MODEL_NAME,
            input_type="search_document",
        )
        return np.array(response.embeddings, dtype=np.float32)


# ── Registry ─────────────────────────────────────────────────────────────────

EMBEDDER_REGISTRY: dict[str, BaseEmbedder] = {
    "minilm": MiniLMEmbedder(),
    "gemini": GeminiEmbedder(),
    "cohere": CohereEmbedder(),
}


def get_embedder(name: str) -> BaseEmbedder:
    if name not in EMBEDDER_REGISTRY:
        raise ValueError(f"Unknown embedder '{name}'. Choose from: {list(EMBEDDER_REGISTRY)}")
    return EMBEDDER_REGISTRY[name]


def available_embedders() -> list[str]:
    return [name for name, emb in EMBEDDER_REGISTRY.items() if emb.is_available()]