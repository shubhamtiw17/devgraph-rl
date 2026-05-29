from __future__ import annotations

import os
import json
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from src.memory.embedder import BaseEmbedder, get_embedder, EMBEDDER_REGISTRY


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    text: str
    score: float          # cosine similarity (0-1, higher = more similar)
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedder_name: str = ""


@dataclass
class CompareResult:
    query: str
    results: Dict[str, List[SearchResult]]   # embedder_name -> results


# ── Single index wrapper ─────────────────────────────────────────────────────

class EmbedderIndex:
    def __init__(self, embedder: BaseEmbedder, store_path: Path) -> None:
        self._embedder = embedder
        self._store_path = store_path
        self._store_path.mkdir(parents=True, exist_ok=True)

        self._index_path = store_path / f"faiss_{embedder.name}.bin"
        self._meta_path  = store_path / f"meta_{embedder.name}.json"

        # parallel lists — index i in _texts matches row i in FAISS
        self._texts: List[str] = []
        self._metadata: List[Dict[str, Any]] = []

        self._index: Optional[faiss.Index] = None
        self._load()

    # ── persistence ─────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._index_path.exists() and self._meta_path.exists():
            self._index = faiss.read_index(str(self._index_path))
            with open(self._meta_path) as f:
                saved = json.load(f)
                self._texts    = saved.get("texts", [])
                self._metadata = saved.get("metadata", [])
        else:
            self._index = faiss.IndexFlatIP(self._embedder.dim)  # inner product = cosine on normalised vecs

    def _save(self) -> None:
        faiss.write_index(self._index, str(self._index_path))
        with open(self._meta_path, "w") as f:
            json.dump({"texts": self._texts, "metadata": self._metadata}, f, indent=2)

    # ── public API ───────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return len(self._texts)

    @property
    def embedder_name(self) -> str:
        return self._embedder.name

    def add(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        vec = self._embedder.encode_one(text).reshape(1, -1)
        self._index.add(vec)
        self._texts.append(text)
        self._metadata.append(metadata or {})
        self._save()

    def add_batch(self, texts: List[str], metadatas: Optional[List[Dict]] = None) -> None:
        if not texts:
            return
        vecs = self._embedder.encode(texts)
        self._index.add(vecs)
        self._texts.extend(texts)
        self._metadata.extend(metadatas or [{} for _ in texts])
        self._save()

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        if self.size == 0:
            return []

        top_k = min(top_k, self.size)
        vec = self._embedder.encode_one(query).reshape(1, -1)
        scores, indices = self._index.search(vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append(SearchResult(
                text=self._texts[idx],
                score=float(score),
                metadata=self._metadata[idx],
                embedder_name=self._embedder.name,
            ))
        return results

    def clear(self) -> None:
        self._index = faiss.IndexFlatIP(self._embedder.dim)
        self._texts = []
        self._metadata = []
        self._save()


# ── Multi-index store ────────────────────────────────────────────────────────

class VectorStore:
    def __init__(self, store_path: Optional[str] = None) -> None:
        path = Path(store_path or os.getenv("VECTOR_STORE_PATH", "./data/vector_store"))
        self._indexes: Dict[str, EmbedderIndex] = {
            name: EmbedderIndex(emb, path)
            for name, emb in EMBEDDER_REGISTRY.items()
        }

    # ── per-embedder ─────────────────────────────────────────────────────────

    def add(self, text: str, embedder_name: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        self._indexes[embedder_name].add(text, metadata)

    def add_batch(self, texts: List[str], embedder_name: str,
                  metadatas: Optional[List[Dict]] = None) -> None:
        self._indexes[embedder_name].add_batch(texts, metadatas)

    def search(self, query: str, embedder_name: str, top_k: int = 5) -> List[SearchResult]:
        return self._indexes[embedder_name].search(query, top_k)

    def size(self, embedder_name: str) -> int:
        return self._indexes[embedder_name].size

    def clear(self, embedder_name: str) -> None:
        self._indexes[embedder_name].clear()

    # ── compare search ───────────────────────────────────────────────────────

    def search_all(self, query: str, top_k: int = 5) -> CompareResult:
        results: Dict[str, List[SearchResult]] = {}
        for name, index in self._indexes.items():
            results[name] = index.search(query, top_k)
        return CompareResult(query=query, results=results)

    # ── sync ─────────────────────────────────────────────────────────────────

    def sync_to_all(self, embedder_name: str) -> Dict[str, int]:
        source = self._indexes[embedder_name]
        if source.size == 0:
            return {}

        counts: Dict[str, int] = {}
        for name, index in self._indexes.items():
            if name == embedder_name:
                continue
            index.add_batch(source._texts, source._metadata)
            counts[name] = len(source._texts)
        return counts

    # ── status ───────────────────────────────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        return {
            name: {
                "size": idx.size,
                "dim": idx._embedder.dim,
                "available": idx._embedder.is_available(),
            }
            for name, idx in self._indexes.items()
        }
