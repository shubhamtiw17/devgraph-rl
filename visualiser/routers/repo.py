from __future__ import annotations

import time
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from visualiser.services.repo_manager import load_repo, RepoInfo, list_cached_repos
from visualiser.services.query_engine import query_repo, QueryResult

router = APIRouter(prefix="/api/repo", tags=["repo"])

# ── In-memory state ───────────────────────────────────────────────────────────

_current_repo: Optional[RepoInfo] = None


# ── Request / Response models ─────────────────────────────────────────────────

class LoadRequest(BaseModel):
    url: str
    language: Optional[str] = None   # override auto-detection if provided


class QueryRequest(BaseModel):
    query: str
    language: Optional[str] = None   # override detected language


class LoadResponse(BaseModel):
    name:        str
    url:         str
    language:    str
    file_counts: Dict[str, int]
    graphs:      Dict[str, Any]
    load_time_ms: int


class QueryResponse(BaseModel):
    query:          str
    answer:         str
    relevant_nodes: List[str]
    memory_hits:    List[str]
    graph_context:  Dict[str, Any]
    error:          Optional[str] = None


# ── Graph builder helper ──────────────────────────────────────────────────────

def _build_graphs_for_repo(repo_path: str, language: str) -> Dict[str, Any]:
    from visualiser.services.graph_builder import build_graphs
    return build_graphs(repo_path, language)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/load", response_model=LoadResponse)
def load_repository(req: LoadRequest):
    global _current_repo

    t0 = time.time()

    try:
        info = load_repo(req.url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    language = req.language or info.language
    _current_repo = info

    try:
        graphs = _build_graphs_for_repo(info.local_path, language)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Graphs built but serialisation failed: {e}"
        )

    load_time_ms = int((time.time() - t0) * 1000)

    return LoadResponse(
        name=info.name,
        url=info.url,
        language=language,
        file_counts=info.file_counts,
        graphs=graphs,
        load_time_ms=load_time_ms,
    )


@router.get("/status")
def repo_status() -> Dict[str, Any]:
    if _current_repo is None:
        return {"loaded": False, "cached": list_cached_repos()}
    return {
        "loaded":      True,
        "name":        _current_repo.name,
        "url":         _current_repo.url,
        "language":    _current_repo.language,
        "local_path":  _current_repo.local_path,
        "file_counts": _current_repo.file_counts,
        "cached":      list_cached_repos(),
    }


@router.post("/query", response_model=QueryResponse)
def query_repository(req: QueryRequest):
    if _current_repo is None:
        raise HTTPException(
            status_code=400,
            detail="No repo loaded. POST to /api/repo/load first."
        )

    language = req.language or _current_repo.language

    result = query_repo(
        query=req.query,
        repo_path=_current_repo.local_path,
        repo_name=_current_repo.name,
        language=language,
    )

    return QueryResponse(
        query=result.query,
        answer=result.answer,
        relevant_nodes=result.relevant_nodes,
        memory_hits=result.memory_hits,
        graph_context=result.graph_context,
        error=result.error,
    )


@router.delete("/clear")
def clear_current_repo() -> Dict[str, Any]:
    global _current_repo
    name = _current_repo.name if _current_repo else None
    _current_repo = None
    return {"cleared": True, "was": name}