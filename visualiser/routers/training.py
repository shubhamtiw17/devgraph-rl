from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/training", tags=["training"])

# Lazy — only constructed on first request so imports don't fail at startup
_trainer = None


def _get_trainer():
    global _trainer
    if _trainer is not None:
        return _trainer

    try:
        from src.rewards.reward_store import get_reward_store
        from src.training.trainer import Trainer
        _trainer = Trainer(reward_store=get_reward_store())
        return _trainer
    except Exception as exc:
        logger.error("Could not build Trainer: %s", exc)
        raise HTTPException(status_code=500, detail=f"Trainer init failed: {exc}")


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    min_score: float = 0.0
    limit: Optional[int] = None


class SweepRequest(BaseModel):
    min_score: float = 0.0
    n_epochs: int = 5
    max_configs: Optional[int] = 6
    limit: Optional[int] = None


class RunRequest(BaseModel):
    min_score: float = 0.0
    n_sweep_epochs: int = 5
    max_sweep_configs: Optional[int] = 6
    n_train_epochs: int = 3
    base_model: str = "Qwen/Qwen2.5-0.5B"
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    hub_token: Optional[str] = None
    skip_sweep: bool = False
    limit: Optional[int] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/analyze")
async def analyze(req: AnalyzeRequest):
    trainer = _get_trainer()
    try:
        result = trainer.run_analysis(
            min_score=req.min_score,
            limit=req.limit,
        )
        analyzed = result.analyzed

        response = {
            "success": result.success,
            "error": result.error,
            "duration_seconds": result.total_duration_seconds,
            "status": result.status.summary(),
        }

        if analyzed:
            response["distribution"] = {
                "total": analyzed.distribution.total,
                "high": analyzed.distribution.high,
                "medium": analyzed.distribution.medium,
                "low": analyzed.distribution.low,
                "mean": analyzed.distribution.mean,
                "std": analyzed.distribution.std,
                "min": analyzed.distribution.min,
                "max": analyzed.distribution.max,
                "quality_ratio": analyzed.distribution.quality_ratio,
                "outlier_count": len(analyzed.distribution.outlier_indices),
            }
            response["clusters"] = [
                {
                    "cluster_id": c.cluster_id,
                    "label": c.label,
                    "size": c.size,
                    "mean_score": c.mean_score,
                }
                for c in analyzed.clusters.clusters
            ]
            response["pairs"] = {
                "total_records": analyzed.pairs.total_records,
                "pairs_selected": analyzed.pairs.pairs_selected,
                "min_delta": analyzed.pairs.min_delta,
                "mean_delta": analyzed.pairs.mean_delta,
            }
            response["feature_importance"] = {
                "features": analyzed.feature_importance.features,
                "importances": analyzed.feature_importance.importances,
                "top_feature": analyzed.feature_importance.top_feature,
                "bottom_feature": analyzed.feature_importance.bottom_feature,
            }
            response["ready_for_training"] = analyzed.ready_for_training

        return response

    except Exception as exc:
        logger.error("Analysis error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/sweep")
async def sweep(req: SweepRequest):
    trainer = _get_trainer()
    try:
        result = trainer.run_sweep(
            min_score=req.min_score,
            n_epochs=req.n_epochs,
            max_configs=req.max_configs,
            limit=req.limit,
        )

        response = {
            "success": result.success,
            "error": result.error,
            "duration_seconds": result.total_duration_seconds,
            "status": result.status.summary(),
        }

        if result.sweep:
            s = result.sweep
            response["sweep"] = {
                "n_runs": s.n_runs,
                "best_eval_loss": s.best_eval_loss,
                "best_config": s.best_config.to_dict(),
                "runs": [
                    {
                        "experiment_id": r.config.experiment_id,
                        "eval_loss": r.eval_loss,
                        "train_loss": r.train_loss,
                        "train_losses": r.train_losses,
                        "eval_losses": r.eval_losses,
                        "overfit_gap": r.overfit_gap,
                        "success": r.success,
                        "config": r.config.to_dict(),
                    }
                    for r in s.ranked_runs
                ],
            }

        return response

    except Exception as exc:
        logger.error("Sweep error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/run")
async def run(req: RunRequest):
    trainer = _get_trainer()
    try:
        result = trainer.run(
            min_score=req.min_score,
            n_sweep_epochs=req.n_sweep_epochs,
            max_sweep_configs=req.max_sweep_configs,
            n_train_epochs=req.n_train_epochs,
            base_model=req.base_model,
            push_to_hub=req.push_to_hub,
            hub_model_id=req.hub_model_id,
            hub_token=req.hub_token,
            skip_sweep=req.skip_sweep,
            limit=req.limit,
        )

        response = {
            "success": result.success,
            "error": result.error,
            "duration_seconds": result.total_duration_seconds,
            "status": result.status.summary(),
        }

        if result.training:
            t = result.training
            response["training"] = {
                "base_model": t.base_model,
                "n_train": t.n_train,
                "n_eval": t.n_eval,
                "n_epochs": t.n_epochs,
                "best_loss": t.best_loss,
                "final_reward_delta": t.final_reward_delta,
                "improved": t.improved,
                "hub_model_id": t.hub_model_id,
                "checkpoint_path": t.checkpoint_path,
                "epoch_metrics": [m.to_dict() for m in t.epoch_metrics],
            }

        if result.sweep:
            response["best_config"] = result.sweep.best_config.to_dict()

        return response

    except Exception as exc:
        logger.error("Run error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/status")
async def get_status():
    trainer = _get_trainer()
    s = trainer.status()
    return {
        "stage": s.stage.value,
        "records_collected": s.records_collected,
        "pairs_selected": s.pairs_selected,
        "n_train": s.n_train,
        "n_eval": s.n_eval,
        "sweep_done": s.sweep_done,
        "best_eval_loss": s.best_eval_loss,
        "training_done": s.training_done,
        "best_training_loss": s.best_training_loss,
        "hub_model_id": s.hub_model_id,
        "ready_to_sweep": s.ready_to_sweep,
        "ready_to_train": s.ready_to_train,
        "error": s.error,
    }


@router.get("/history")
async def get_history():
    result_path = Path("./data/checkpoints/training_result.json")
    if not result_path.exists():
        raise HTTPException(
            status_code=404,
            detail="No training history found. Run the pipeline first.",
        )
    try:
        data = json.loads(result_path.read_text())
        return data
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not read history: {exc}")


@router.delete("/reset")
async def reset():
    global _trainer
    _trainer = None
    return {"reset": True, "message": "Pipeline reset. Next request will reinitialise."}