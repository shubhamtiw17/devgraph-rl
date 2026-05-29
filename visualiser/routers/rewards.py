from __future__ import annotations

from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.rewards.reward_model import get_reward_model
from src.rewards.reward_store import get_reward_store

router = APIRouter(prefix="/api/rewards", tags=["rewards"])


# ── Request / Response models ─────────────────────────────────────────────────

class ScoreRequest(BaseModel):
    task:               str
    output:             str
    agent_type:         str   = "coding"
    language:           str   = "python"
    repo_path:          Optional[str] = None
    test_pass_rate:     float = 0.0
    execution_success:  bool  = False


class DimensionOut(BaseModel):
    name:     str
    score:    float
    weight:   float
    feedback: str


class ScoreResponse(BaseModel):
    final_score: float
    dimensions:  List[DimensionOut]
    feedback:    List[str]
    summary:     str
    saved_id:    str


class RecordOut(BaseModel):
    id:          str
    timestamp:   str
    task:        str
    agent_type:  str
    final_score: float
    dimensions:  Dict[str, float]
    feedback:    List[str]


class StatsOut(BaseModel):
    total:    int
    average:  float
    best:     float
    worst:    float
    by_agent: Dict[str, float]
    trend:    List[float]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/score", response_model=ScoreResponse)
def score_output(req: ScoreRequest) -> ScoreResponse:
    """Score an agent output across all 5 dimensions."""
    try:
        rm     = get_reward_model()
        rs     = get_reward_store()
        result = rm.score(
            task=req.task,
            output=req.output,
            agent_type=req.agent_type,
            language=req.language,
            repo_path=req.repo_path,
            test_pass_rate=req.test_pass_rate,
            execution_success=req.execution_success,
        )
        record = rs.save(result)
        return ScoreResponse(
            final_score=result.final_score,
            dimensions=[
                DimensionOut(
                    name=d.name,
                    score=d.score,
                    weight=d.weight,
                    feedback=d.feedback,
                )
                for d in result.dimensions
            ],
            feedback=result.feedback,
            summary=result.summary,
            saved_id=record.id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history", response_model=List[RecordOut])
def get_history(n: int = 20) -> List[RecordOut]:
    """Return n most recent scored records."""
    rs      = get_reward_store()
    records = rs.load_recent(n)
    return [
        RecordOut(
            id=r.id,
            timestamp=r.timestamp,
            task=r.task,
            agent_type=r.agent_type,
            final_score=r.final_score,
            dimensions=r.dimensions,
            feedback=r.feedback,
        )
        for r in records
    ]


@router.get("/top", response_model=List[RecordOut])
def get_top(n: int = 10) -> List[RecordOut]:
    """Return top n records by final score."""
    rs      = get_reward_store()
    records = rs.top_scores(n)
    return [
        RecordOut(
            id=r.id,
            timestamp=r.timestamp,
            task=r.task,
            agent_type=r.agent_type,
            final_score=r.final_score,
            dimensions=r.dimensions,
            feedback=r.feedback,
        )
        for r in records
    ]


@router.get("/stats", response_model=StatsOut)
def get_stats() -> StatsOut:
    """Return summary statistics across all scored outputs."""
    rs = get_reward_store()
    s  = rs.stats()
    return StatsOut(
        total=s.total,
        average=s.average,
        best=s.best,
        worst=s.worst,
        by_agent=s.by_agent,
        trend=s.trend,
    )


@router.delete("/clear")
def clear_history() -> Dict[str, Any]:
    """Wipe all reward history."""
    get_reward_store().clear()
    return {"cleared": True}
