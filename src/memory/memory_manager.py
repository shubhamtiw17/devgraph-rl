from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Dict, Any

from src.memory.vector_store import VectorStore, SearchResult, CompareResult
from src.agents.base_agent import AgentResult


def _build_metadata(
    agent_type: str,
    repo_path: str,
    related_files: Optional[List[str]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "agent_type":    agent_type,
        "repo_path":     repo_path,
        "related_files": related_files or [],
        "timestamp":     datetime.utcnow().isoformat(),
        **(extra or {}),
    }


def _result_to_text(task: str, result: AgentResult) -> str:
    parts = [f"TASK: {task}"]
    if result.output:
        output_preview = result.output[:500]
        parts.append(f"OUTPUT: {output_preview}")
    if result.error:
        parts.append(f"ERROR: {result.error}")
    return "\n".join(parts)


class MemoryManager:
    def __init__(self, store: Optional[VectorStore] = None) -> None:
        self._store = store or VectorStore()

    def store(
        self,
        task: str,
        result: AgentResult,
        agent_type: str,
        repo_path: str,
        embedder_name: str = "minilm",
        related_files: Optional[List[str]] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        text = _result_to_text(task, result)
        metadata = _build_metadata(
            agent_type=agent_type,
            repo_path=repo_path,
            related_files=related_files,
            extra=extra_metadata,
        )
        self._store.add(text=text, embedder_name=embedder_name, metadata=metadata)

    def store_to_all(
        self,
        task: str,
        result: AgentResult,
        agent_type: str,
        repo_path: str,
        related_files: Optional[List[str]] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        from src.memory.embedder import EMBEDDER_REGISTRY
        for name in EMBEDDER_REGISTRY:
            self.store(
                task=task,
                result=result,
                agent_type=agent_type,
                repo_path=repo_path,
                embedder_name=name,
                related_files=related_files,
                extra_metadata=extra_metadata,
            )

    def retrieve(
        self,
        query: str,
        embedder_name: str = "minilm",
        top_k: int = 5,
        agent_type: Optional[str] = None,
        repo_path: Optional[str] = None,
    ) -> List[SearchResult]:
        results = self._store.search(
            query=query,
            embedder_name=embedder_name,
            top_k=top_k * 3 if (agent_type or repo_path) else top_k,
        )
        if agent_type or repo_path:
            results = [
                r for r in results
                if (not agent_type or r.metadata.get("agent_type") == agent_type)
                and (not repo_path or r.metadata.get("repo_path") == repo_path)
            ]
            results = results[:top_k]
        return results

    def retrieve_all(self, query: str, top_k: int = 5) -> CompareResult:
        return self._store.search_all(query=query, top_k=top_k)

    def sync_to_all(self, source_embedder: str = "minilm") -> Dict[str, int]:
        return self._store.sync_to_all(source_embedder)

    def status(self) -> Dict[str, Any]:
        return self._store.status()

    def size(self, embedder_name: str) -> int:
        return self._store.size(embedder_name)
