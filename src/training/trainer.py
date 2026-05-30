from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from src.training.data_collector import AnalyzedCollection, DataCollector
from src.training.dataset_builder import DatasetBuilder, SplitResult
from src.training.keras_experiments import (
    ExperimentConfig,
    KerasExperiments,
    SweepResult,
)
from src.training.sklearn_analyzer import SklearnAnalyzer
from src.training.torch_trainer import TrainerConfig, TrainingResult, TorchTrainer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline status
# ---------------------------------------------------------------------------

class PipelineStage(str, Enum):
    IDLE        = "idle"
    COLLECTING  = "collecting"
    ANALYZING   = "analyzing"
    BUILDING    = "building"
    SWEEPING    = "sweeping"
    TRAINING    = "training"
    DONE        = "done"
    FAILED      = "failed"


@dataclass
class PipelineStatus:
    stage: PipelineStage = PipelineStage.IDLE
    records_collected: int = 0
    pairs_selected: int = 0
    n_train: int = 0
    n_eval: int = 0
    sweep_done: bool = False
    best_eval_loss: Optional[float] = None
    training_done: bool = False
    best_training_loss: Optional[float] = None
    hub_model_id: Optional[str] = None
    error: Optional[str] = None

    @property
    def ready_to_sweep(self) -> bool:
        return self.pairs_selected >= 5

    @property
    def ready_to_train(self) -> bool:
        return self.n_train >= 5

    def summary(self) -> str:
        lines = [
            f"Stage: {self.stage.value}",
            f"Records: {self.records_collected} | Pairs: {self.pairs_selected}",
            f"Train: {self.n_train} | Eval: {self.n_eval}",
            f"Sweep done: {self.sweep_done}",
            f"Training done: {self.training_done}",
        ]
        if self.best_eval_loss is not None:
            lines.append(f"Best sweep loss: {self.best_eval_loss:.4f}")
        if self.best_training_loss is not None:
            lines.append(f"Best train loss: {self.best_training_loss:.4f}")
        if self.hub_model_id:
            lines.append(f"Hub model: {self.hub_model_id}")
        if self.error:
            lines.append(f"Error: {self.error}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Full pipeline result
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    analyzed: Optional[AnalyzedCollection] = None

    split: Optional[SplitResult] = None

    sweep: Optional[SweepResult] = None

    training: Optional[TrainingResult] = None

    status: PipelineStatus = field(default_factory=PipelineStatus)
    total_duration_seconds: float = 0.0
    success: bool = False
    error: Optional[str] = None

    def summary(self) -> str:
        lines = ["=== PipelineResult ===", self.status.summary()]
        if self.analyzed:
            lines.append(f"Distribution: {self.analyzed.distribution.summary()}")
            lines.append(f"Pairs: {self.analyzed.pairs.summary()}")
        if self.sweep:
            lines.append(f"Sweep: {self.sweep.summary()}")
        if self.training:
            lines.append(f"Training: {self.training.summary()}")
        lines.append(f"Total time: {self.total_duration_seconds:.1f}s")
        lines.append(f"Success: {self.success}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Trainer (orchestrator)
# ---------------------------------------------------------------------------

class Trainer:

    def __init__(
        self,
        reward_store,
        min_score: float = 0.0,
        eval_ratio: float = 0.10,
        experiment_dir: Path = Path("./data/experiments"),
        checkpoint_dir: Path = Path("./data/checkpoints"),
        embedding_dim: int = 384,
    ) -> None:
        self.reward_store = reward_store
        self.min_score = min_score
        self.eval_ratio = eval_ratio
        self.experiment_dir = experiment_dir
        self.checkpoint_dir = checkpoint_dir
        self.embedding_dim = embedding_dim

        # Sub-components — constructed lazily or at init (all CI safe)
        self._collector: Optional[DataCollector] = None
        self._builder: Optional[DatasetBuilder] = None
        self._keras: Optional[KerasExperiments] = None

        # Mutable pipeline status
        self._status = PipelineStatus()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def status(self) -> PipelineStatus:
        return self._status

    def run_analysis(
        self,
        min_score: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> PipelineResult:

        start = time.time()
        result = PipelineResult()

        try:
            self._status.stage = PipelineStage.COLLECTING
            collector = self._get_collector()
            analyzed = collector.collect_and_analyze(
                min_score=min_score if min_score is not None else self.min_score,
                limit=limit,
            )
            result.analyzed = analyzed
            self._status.records_collected = analyzed.collection.count
            self._status.pairs_selected = analyzed.pairs.pairs_selected

            self._status.stage = PipelineStage.ANALYZING
            logger.info("Analysis complete: %s", analyzed.collection.summary())

            self._status.stage = PipelineStage.DONE
            result.success = True

        except Exception as exc:
            logger.error("Analysis failed: %s", exc, exc_info=True)
            self._status.stage = PipelineStage.FAILED
            self._status.error = str(exc)
            result.error = str(exc)

        result.status = self._status
        result.total_duration_seconds = time.time() - start
        return result

    def run_sweep(
        self,
        min_score: Optional[float] = None,
        sweep_space: Optional[dict] = None,
        n_epochs: int = 5,
        max_configs: Optional[int] = None,
        limit: Optional[int] = None,
        embedding_vectors: Optional[list[list[float]]] = None,
    ) -> PipelineResult:

        start = time.time()
        result = PipelineResult()

        try:
            # Step 1 — collect + analyze
            self._status.stage = PipelineStage.COLLECTING
            collector = self._get_collector()
            analyzed = collector.collect_and_analyze(
                min_score=min_score if min_score is not None else self.min_score,
                limit=limit,
                embedding_vectors=embedding_vectors,
            )
            result.analyzed = analyzed
            self._status.records_collected = analyzed.collection.count
            self._status.pairs_selected = analyzed.pairs.pairs_selected

            if not analyzed.ready_for_training:
                msg = (
                    f"Not enough training pairs: {analyzed.pairs.pairs_selected} "
                    f"(need >= 5). Collect more reward data first."
                )
                logger.warning(msg)
                result.error = msg
                result.status = self._status
                result.total_duration_seconds = time.time() - start
                return result

            # Step 2 — build dataset
            self._status.stage = PipelineStage.BUILDING
            builder = self._get_builder()
            split = builder.build_split(analyzed, eval_ratio=self.eval_ratio)
            result.split = split
            self._status.n_train = split.n_train
            self._status.n_eval = split.n_eval

            # Step 3 — keras sweep (uses embedding format for baseline model)
            self._status.stage = PipelineStage.SWEEPING
            keras = self._get_keras()
            train_emb = self._pairs_to_embeddings(analyzed.pairs.pairs)
            eval_emb  = train_emb[:max(1, len(train_emb) // 10)]  # 10% for eval

            sweep = keras.run_sweep(
                train_data=train_emb,
                eval_data=eval_emb,
                sweep_space=sweep_space,
                n_epochs=n_epochs,
                max_configs=max_configs,
            )
            result.sweep = sweep
            self._status.sweep_done = True
            self._status.best_eval_loss = sweep.best_eval_loss

            self._status.stage = PipelineStage.DONE
            result.success = True
            logger.info("Sweep complete: %s", sweep.summary())

        except Exception as exc:
            logger.error("Sweep failed: %s", exc, exc_info=True)
            self._status.stage = PipelineStage.FAILED
            self._status.error = str(exc)
            result.error = str(exc)

        result.status = self._status
        result.total_duration_seconds = time.time() - start
        return result

    def run(
        self,
        min_score: Optional[float] = None,
        sweep_space: Optional[dict] = None,
        n_sweep_epochs: int = 5,
        max_sweep_configs: Optional[int] = None,
        n_train_epochs: int = 3,
        push_to_hub: bool = False,
        hub_model_id: Optional[str] = None,
        hub_token: Optional[str] = None,
        base_model: str = "Qwen/Qwen2.5-0.5B",
        limit: Optional[int] = None,
        skip_sweep: bool = False,
    ) -> PipelineResult:

        start = time.time()
        result = PipelineResult()

        try:
            # ── Step 1: collect + analyze ──────────────────────────────
            self._status.stage = PipelineStage.COLLECTING
            collector = self._get_collector()
            analyzed = collector.collect_and_analyze(
                min_score=min_score if min_score is not None else self.min_score,
                limit=limit,
            )
            result.analyzed = analyzed
            self._status.records_collected = analyzed.collection.count
            self._status.pairs_selected = analyzed.pairs.pairs_selected

            if not analyzed.ready_for_training:
                msg = (
                    f"Not enough training pairs: {analyzed.pairs.pairs_selected}. "
                    "Collect more reward data first."
                )
                logger.warning(msg)
                result.error = msg
                result.status = self._status
                result.total_duration_seconds = time.time() - start
                return result

            # ── Step 2: build dataset ──────────────────────────────────
            self._status.stage = PipelineStage.BUILDING
            builder = self._get_builder()
            split = builder.build_split(analyzed, eval_ratio=self.eval_ratio)
            result.split = split
            self._status.n_train = split.n_train
            self._status.n_eval = split.n_eval

            # ── Step 3: keras sweep (optional) ────────────────────────
            torch_config = TrainerConfig(
                base_model=base_model,
                output_dir=str(self.checkpoint_dir),
                n_epochs=n_train_epochs,
                push_to_hub=push_to_hub,
                hub_model_id=hub_model_id,
                hub_token=hub_token,
            )

            if not skip_sweep:
                self._status.stage = PipelineStage.SWEEPING
                keras = self._get_keras()
                train_emb = self._pairs_to_embeddings(analyzed.pairs.pairs)
                eval_emb  = train_emb[:max(1, len(train_emb) // 10)]

                sweep = keras.run_sweep(
                    train_data=train_emb,
                    eval_data=eval_emb,
                    sweep_space=sweep_space,
                    n_epochs=n_sweep_epochs,
                    max_configs=max_sweep_configs,
                )
                result.sweep = sweep
                self._status.sweep_done = True
                self._status.best_eval_loss = sweep.best_eval_loss
                logger.info("Sweep complete: %s", sweep.summary())

                # Apply best Keras config to TorchTrainer
                torch_config = self._apply_sweep_to_torch_config(
                    torch_config, sweep.best_config
                )

            # ── Step 4: DPO training ───────────────────────────────────
            self._status.stage = PipelineStage.TRAINING
            torch_trainer = TorchTrainer(torch_config)
            training = torch_trainer.train(split.dataset_dict)
            result.training = training
            self._status.training_done = training.success
            self._status.best_training_loss = training.best_loss
            self._status.hub_model_id = training.hub_model_id

            self._status.stage = (
                PipelineStage.DONE if training.success else PipelineStage.FAILED
            )
            result.success = training.success
            if not training.success:
                result.error = training.error

            logger.info("Training complete: %s", training.summary())

        except Exception as exc:
            logger.error("Pipeline failed: %s", exc, exc_info=True)
            self._status.stage = PipelineStage.FAILED
            self._status.error = str(exc)
            result.error = str(exc)

        result.status = self._status
        result.total_duration_seconds = time.time() - start
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_collector(self) -> DataCollector:
        if self._collector is None:
            self._collector = DataCollector(
                reward_store=self.reward_store,
                analyzer=SklearnAnalyzer(),
                min_score=self.min_score,
            )
        return self._collector

    def _get_builder(self) -> DatasetBuilder:
        if self._builder is None:
            self._builder = DatasetBuilder()
        return self._builder

    def _get_keras(self) -> KerasExperiments:
        if self._keras is None:
            self._keras = KerasExperiments(
                experiment_dir=self.experiment_dir,
                embedding_dim=self.embedding_dim,
            )
        return self._keras

    @staticmethod
    def _pairs_to_embeddings(pairs) -> list[dict]:
        import hashlib
        import struct

        result = []
        for pair in pairs:
            def text_to_vec(text: str, dim: int = 384) -> list[float]:
                vec = []
                seed = text.encode()
                for i in range(dim):
                    h = hashlib.md5(seed + i.to_bytes(4, "little")).digest()
                    val = struct.unpack("f", h[:4])[0]
                    vec.append(float(val))
                # Normalise
                norm = sum(x * x for x in vec) ** 0.5 or 1.0
                return [x / norm for x in vec]

            result.append({
                "chosen_embedding":   text_to_vec(pair.chosen_output,  dim=384),
                "rejected_embedding": text_to_vec(pair.rejected_output, dim=384),
            })
        return result

    @staticmethod
    def _apply_sweep_to_torch_config(
        torch_config: TrainerConfig,
        best_keras: ExperimentConfig,
    ) -> TrainerConfig:

        torch_config.learning_rate = best_keras.learning_rate
        torch_config.per_device_train_batch_size = best_keras.batch_size
        torch_config.per_device_eval_batch_size = best_keras.batch_size
        return torch_config