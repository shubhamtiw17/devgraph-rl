from __future__ import annotations

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.memory.memory_manager import MemoryManager
from src.memory.vector_store import SearchResult, CompareResult
from src.agents.base_agent import AgentResult

router = APIRouter(prefix="/api/memory", tags=["memory"])

# singleton manager — shared across all requests
_manager: Optional[MemoryManager] = None


def get_manager() -> MemoryManager:
    global _manager
    if _manager is None:
        _manager = MemoryManager()
    return _manager


# ── Request / Response models ─────────────────────────────────────────────────

class StoreRequest(BaseModel):
    text: str
    embedder: str = "minilm"
    agent_type: str = "manual"
    repo_path: str = "/tmp/devgraph"
    related_files: List[str] = []
    extra_metadata: Dict[str, Any] = {}


class SearchRequest(BaseModel):
    query: str
    embedder: str = "minilm"
    top_k: int = 5
    agent_type: Optional[str] = None
    repo_path: Optional[str] = None


class CompareRequest(BaseModel):
    query: str
    top_k: int = 5


class SyncRequest(BaseModel):
    source_embedder: str = "minilm"


class ClearRequest(BaseModel):
    embedder: str


class SearchResultOut(BaseModel):
    text: str
    score: float
    metadata: Dict[str, Any]
    embedder_name: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/status")
def memory_status() -> Dict[str, Any]:
    return get_manager().status()


@router.post("/store")
def store_memory(req: StoreRequest) -> Dict[str, Any]:
    try:
        result = AgentResult(
            agent_name=req.agent_type,
            output=req.text,
            success=True,
        )
        get_manager().store(
            task=req.text,
            result=result,
            agent_type=req.agent_type,
            repo_path=req.repo_path,
            embedder_name=req.embedder,
            related_files=req.related_files,
            extra_metadata=req.extra_metadata,
        )
        return {
            "stored": True,
            "embedder": req.embedder,
            "size": get_manager().size(req.embedder),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search")
def search_memory(req: SearchRequest) -> List[SearchResultOut]:
    try:
        results = get_manager().retrieve(
            query=req.query,
            embedder_name=req.embedder,
            top_k=req.top_k,
            agent_type=req.agent_type,
            repo_path=req.repo_path,
        )
        return [
            SearchResultOut(
                text=r.text,
                score=round(r.score, 4),
                metadata=r.metadata,
                embedder_name=r.embedder_name,
            )
            for r in results
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search/compare")
def compare_search(req: CompareRequest) -> Dict[str, Any]:
    try:
        compare = get_manager().retrieve_all(query=req.query, top_k=req.top_k)
        return {
            "query": compare.query,
            "results": {
                embedder: [
                    SearchResultOut(
                        text=r.text,
                        score=round(r.score, 4),
                        metadata=r.metadata,
                        embedder_name=r.embedder_name,
                    ).model_dump()
                    for r in results
                ]
                for embedder, results in compare.results.items()
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sync")
def sync_memories(req: SyncRequest) -> Dict[str, Any]:
    try:
        counts = get_manager().sync_to_all(req.source_embedder)
        return {"synced": True, "source": req.source_embedder, "counts": counts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/clear")
def clear_index(req: ClearRequest) -> Dict[str, Any]:
    try:
        get_manager()._store.clear(req.embedder)
        return {"cleared": True, "embedder": req.embedder}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
