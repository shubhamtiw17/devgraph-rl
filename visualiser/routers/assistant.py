from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from visualiser.services.assistant_engine import (
    AssistantEngine,
    AssistantMode,
    SystemContext,
    detect_mode,
    get_engine,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/assistant", tags=["assistant"])

# In-memory conversation history — one session per server process
# Keyed by session_id (default "default" for single-user mode)
_histories: dict[str, list[dict]] = {}


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message:    str
    language:   str  = "python"
    session_id: str  = "default"
    # Optional context overrides from the UI
    repo_name:    Optional[str] = None
    repo_language: Optional[str] = None
    loaded_file:  Optional[str] = None
    file_content: Optional[str] = None


class ResetRequest(BaseModel):
    session_id: str = "default"


# ---------------------------------------------------------------------------
# Context builder — reads live system state
# ---------------------------------------------------------------------------

def _build_context(req: ChatRequest) -> SystemContext:
    ctx = SystemContext(
        repo_name=req.repo_name,
        repo_language=req.repo_language,
        loaded_file=req.loaded_file,
        file_content=req.file_content,
    )

    # Pull live repo state if not overridden
    if not ctx.repo_name:
        try:
            from visualiser.services.repo_manager import get_repo_manager
            rm = get_repo_manager()
            status = rm.get_status()
            if status.get("loaded"):
                ctx.repo_name     = status.get("name")
                ctx.repo_language = status.get("language")
        except Exception as e:
            logger.debug("Repo manager unavailable: %s", e)

    # Pull reward stats
    try:
        from src.rewards.reward_store import get_reward_store
        store = get_reward_store()
        stats = store.stats()
        ctx.reward_stats = {
            "total":   stats.total,
            "average": stats.average,
            "best":    stats.best,
        }
    except Exception as e:
        logger.debug("Reward store unavailable: %s", e)

    # Pull recent memories
    try:
        from src.memory.memory_manager import get_memory_manager
        mm = get_memory_manager()
        status = mm.status()
        ctx.n_memories = sum(
            v.get("size", 0) for v in status.values()
            if isinstance(v, dict)
        )
        # Fetch a few recent memories as context
        results = mm.retrieve(
            query=req.message,
            embedder="minilm",
            top_k=3,
        )
        ctx.recent_memories = [r.text[:200] for r in results if hasattr(r, "text")]
    except Exception as e:
        logger.debug("Memory manager unavailable: %s", e)

    return ctx


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/chat")
async def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    engine  = get_engine()
    history = _histories.setdefault(req.session_id, [])
    ctx     = _build_context(req)

    try:
        response = engine.chat(
            message=req.message,
            context=ctx,
            history=history,
            language=req.language,
        )
    except Exception as exc:
        logger.error("Assistant chat error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

    # Append to history
    history.append({"role": "user",      "content": req.message})
    history.append({"role": "assistant", "content": response.message})

    # Cap history at 20 turns to avoid context window blowout
    if len(history) > 20:
        _histories[req.session_id] = history[-20:]

    return {
        "session_id":   req.session_id,
        "mode":         response.mode.value,
        "message":      response.message,
        "code":         response.code,
        "language":     response.language,
        "sandbox":      response.sandbox.__dict__ if response.sandbox else None,
        "score":        {
            "score":      response.score.score,
            "dimensions": response.score.dimensions,
            "summary":    response.score.summary,
            "delta":      response.score.delta,
        } if response.score else None,
        "suggestions":  response.suggestions,
        "stored":       response.stored,
        "duration_ms":  response.duration_ms,
        "context_used": response.context_used,
        "context": {
            "repo_name":    ctx.repo_name,
            "repo_language": ctx.repo_language,
            "loaded_file":  ctx.loaded_file,
            "n_memories":   ctx.n_memories,
        },
    }


@router.post("/reset")
async def reset(req: ResetRequest):
    _histories.pop(req.session_id, None)
    return {"reset": True, "session_id": req.session_id}


@router.get("/context")
async def get_context():
    ctx = _build_context(ChatRequest(message=""))

    return {
        "repo_name":    ctx.repo_name,
        "repo_language": ctx.repo_language,
        "n_memories":   ctx.n_memories,
        "reward_stats": ctx.reward_stats,
        "has_file":     bool(ctx.loaded_file),
    }


@router.get("/history")
async def get_history(session_id: str = "default"):
    history = _histories.get(session_id, [])
    return {
        "session_id": session_id,
        "turns":      len(history) // 2,
        "history":    history,
    }