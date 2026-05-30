from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.training.trainer import (
    PipelineResult,
    PipelineStage,
    PipelineStatus,
    Trainer,
)
from src.training.data_collector import AnalyzedCollection, CollectionResult
from src.training.dataset_builder import SplitResult
from src.training.keras_experiments import (
    ExperimentConfig,
    ExperimentRun,
    SweepResult,
)
from src.training.sklearn_analyzer import (
    ClusterResult,
    FeatureImportanceResult,
    PairSelectionResult,
    ScoreDistribution,
    TrainingPair,
)
from src.training.torch_trainer import EpochMetrics, TrainingResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_store(records=None) -> MagicMock:
    store = MagicMock()
    store.load_all.return_value = records or []
    return store


def make_pair(i: int = 0) -> TrainingPair:
    return TrainingPair(
        task=f"task_{i}",
        chosen_output=f"good_{i}",
        chosen_score=0.9,
        rejected_output=f"bad_{i}",
        rejected_score=0.2,
        score_delta=0.7,
    )


def make_pair_selection(n: int) -> PairSelectionResult:
    pairs = [make_pair(i) for i in range(n)]
    return PairSelectionResult(
        pairs=pairs,
        total_records=n * 2,
        pairs_selected=n,
        min_delta=0.5,
        mean_delta=0.6,
    )


def make_distribution() -> ScoreDistribution:
    return ScoreDistribution(
        total=20, high=10, medium=5, low=5,
        mean=0.7, std=0.15, min=0.1, max=0.95,
    )


def make_analyzed(n_pairs: int = 6, ready: bool = True) -> AnalyzedCollection:
    pair_result = make_pair_selection(n_pairs)
    collection = CollectionResult(
        records=[{}] * (n_pairs * 2),
        total_in_store=n_pairs * 2,
        filtered_out=0,
        duplicates_removed=0,
        min_score_used=0.6,
    )
    dist = make_distribution()
    clusters = MagicMock()
    clusters.n_clusters = 3
    importance = MagicMock()
    importance.top_feature = "correctness"

    analyzed = AnalyzedCollection(
        collection=collection,
        distribution=dist,
        clusters=clusters,
        pairs=pair_result,
        feature_importance=importance,
    )
    # Override ready_for_training
    type(analyzed).ready_for_training = property(lambda self: ready)
    return analyzed


def make_split(n_train: int = 8, n_eval: int = 2) -> SplitResult:
    from datasets import Dataset, DatasetDict
    empty = Dataset.from_dict({
        "prompt": [], "chosen": [], "rejected": [],
        "score_delta": [], "chosen_score": [], "rejected_score": [],
    })
    split = SplitResult(
        dataset_dict=DatasetDict({"train": empty, "eval": empty}),
        n_train=n_train,
        n_eval=n_eval,
        eval_ratio=0.2,
        mean_delta=0.6,
    )
    return split


def make_sweep(best_eval_loss: float = 0.3) -> SweepResult:
    cfg = ExperimentConfig(learning_rate=2e-4, batch_size=16)
    run = ExperimentRun(
        config=cfg,
        train_loss=0.25,
        eval_loss=best_eval_loss,
        success=True,
    )
    return SweepResult(
        runs=[run],
        best_config=cfg,
        best_eval_loss=best_eval_loss,
        total_duration_seconds=60.0,
    )


def make_training_result(success: bool = True) -> TrainingResult:
    return TrainingResult(
        base_model="test-model",
        n_train=8,
        n_eval=2,
        n_epochs=2,
        success=success,
        best_loss=0.3 if success else float("inf"),
        error=None if success else "CUDA OOM",
        epoch_metrics=[
            EpochMetrics(1, 0.5, 0.7, 0.3, 0.4, 10.0),
            EpochMetrics(2, 0.3, 0.85, 0.2, 0.65, 10.0),
        ],
    )


def make_trainer(**kwargs) -> Trainer:
    defaults = dict(
        reward_store=make_store(),
        min_score=0.6,
        eval_ratio=0.1,
        experiment_dir=Path("/tmp/test_exp"),
        checkpoint_dir=Path("/tmp/test_ckpt"),
    )
    defaults.update(kwargs)
    return Trainer(**defaults)


# ---------------------------------------------------------------------------
# PipelineStatus
# ---------------------------------------------------------------------------

class TestPipelineStatus:
    def test_ready_to_sweep_true(self):
        s = PipelineStatus(pairs_selected=5)
        assert s.ready_to_sweep is True

    def test_ready_to_sweep_false(self):
        s = PipelineStatus(pairs_selected=4)
        assert s.ready_to_sweep is False

    def test_ready_to_train_true(self):
        s = PipelineStatus(n_train=5)
        assert s.ready_to_train is True

    def test_ready_to_train_false(self):
        s = PipelineStatus(n_train=4)
        assert s.ready_to_train is False

    def test_summary_contains_stage(self):
        s = PipelineStatus(stage=PipelineStage.ANALYZING)
        assert "analyzing" in s.summary()

    def test_summary_contains_records(self):
        s = PipelineStatus(records_collected=42)
        assert "42" in s.summary()

    def test_summary_contains_hub_model(self):
        s = PipelineStatus(hub_model_id="user/model")
        assert "user/model" in s.summary()


# ---------------------------------------------------------------------------
# PipelineResult
# ---------------------------------------------------------------------------

class TestPipelineResult:
    def test_summary_contains_stage(self):
        r = PipelineResult(status=PipelineStatus(stage=PipelineStage.DONE))
        s = r.summary()
        assert "done" in s

    def test_summary_with_analyzed(self):
        r = PipelineResult(
            analyzed=make_analyzed(),
            status=PipelineStatus(),
        )
        s = r.summary()
        assert "Distribution" in s or "Pairs" in s

    def test_summary_with_training(self):
        r = PipelineResult(
            training=make_training_result(),
            status=PipelineStatus(),
        )
        s = r.summary()
        assert "Training" in s


# ---------------------------------------------------------------------------
# Trainer.status()
# ---------------------------------------------------------------------------

class TestTrainerStatus:
    def test_initial_stage_idle(self):
        trainer = make_trainer()
        assert trainer.status().stage == PipelineStage.IDLE

    def test_returns_pipeline_status(self):
        trainer = make_trainer()
        assert isinstance(trainer.status(), PipelineStatus)


# ---------------------------------------------------------------------------
# Trainer.run_analysis()
# ---------------------------------------------------------------------------

class TestRunAnalysis:
    def test_returns_pipeline_result(self):
        trainer = make_trainer()
        with patch.object(trainer, "_get_collector") as mock_gc:
            mock_gc.return_value.collect_and_analyze.return_value = make_analyzed()
            result = trainer.run_analysis()
        assert isinstance(result, PipelineResult)

    def test_success_flag(self):
        trainer = make_trainer()
        with patch.object(trainer, "_get_collector") as mock_gc:
            mock_gc.return_value.collect_and_analyze.return_value = make_analyzed()
            result = trainer.run_analysis()
        assert result.success is True

    def test_analyzed_populated(self):
        trainer = make_trainer()
        analyzed = make_analyzed()
        with patch.object(trainer, "_get_collector") as mock_gc:
            mock_gc.return_value.collect_and_analyze.return_value = analyzed
            result = trainer.run_analysis()
        assert result.analyzed is analyzed

    def test_stage_done_on_success(self):
        trainer = make_trainer()
        with patch.object(trainer, "_get_collector") as mock_gc:
            mock_gc.return_value.collect_and_analyze.return_value = make_analyzed()
            result = trainer.run_analysis()
        assert result.status.stage == PipelineStage.DONE

    def test_status_records_collected(self):
        trainer = make_trainer()
        analyzed = make_analyzed(n_pairs=6)
        with patch.object(trainer, "_get_collector") as mock_gc:
            mock_gc.return_value.collect_and_analyze.return_value = analyzed
            trainer.run_analysis()
        assert trainer.status().records_collected == analyzed.collection.count

    def test_status_pairs_selected(self):
        trainer = make_trainer()
        analyzed = make_analyzed(n_pairs=6)
        with patch.object(trainer, "_get_collector") as mock_gc:
            mock_gc.return_value.collect_and_analyze.return_value = analyzed
            trainer.run_analysis()
        assert trainer.status().pairs_selected == 6

    def test_exception_sets_failed_stage(self):
        trainer = make_trainer()
        with patch.object(trainer, "_get_collector") as mock_gc:
            mock_gc.return_value.collect_and_analyze.side_effect = RuntimeError("boom")
            result = trainer.run_analysis()
        assert result.success is False
        assert result.status.stage == PipelineStage.FAILED
        assert "boom" in result.error

    def test_min_score_forwarded(self):
        trainer = make_trainer()
        with patch.object(trainer, "_get_collector") as mock_gc:
            mock_gc.return_value.collect_and_analyze.return_value = make_analyzed()
            trainer.run_analysis(min_score=0.9)
        mock_gc.return_value.collect_and_analyze.assert_called_once_with(
            min_score=0.9, limit=None,
        )

    def test_duration_recorded(self):
        trainer = make_trainer()
        with patch.object(trainer, "_get_collector") as mock_gc:
            mock_gc.return_value.collect_and_analyze.return_value = make_analyzed()
            result = trainer.run_analysis()
        assert result.total_duration_seconds >= 0.0


# ---------------------------------------------------------------------------
# Trainer.run_sweep()
# ---------------------------------------------------------------------------

class TestRunSweep:
    def test_returns_pipeline_result(self):
        trainer = make_trainer()
        with patch.object(trainer, "_get_collector") as mock_gc, \
             patch.object(trainer, "_get_builder") as mock_gb, \
             patch.object(trainer, "_get_keras") as mock_gk:
            mock_gc.return_value.collect_and_analyze.return_value = make_analyzed()
            mock_gb.return_value.build_split.return_value = make_split()
            mock_gk.return_value.run_sweep.return_value = make_sweep()
            result = trainer.run_sweep()
        assert isinstance(result, PipelineResult)

    def test_sweep_populated(self):
        trainer = make_trainer()
        sweep = make_sweep(best_eval_loss=0.22)
        with patch.object(trainer, "_get_collector") as mock_gc, \
             patch.object(trainer, "_get_builder") as mock_gb, \
             patch.object(trainer, "_get_keras") as mock_gk:
            mock_gc.return_value.collect_and_analyze.return_value = make_analyzed()
            mock_gb.return_value.build_split.return_value = make_split()
            mock_gk.return_value.run_sweep.return_value = sweep
            result = trainer.run_sweep()
        assert result.sweep is sweep

    def test_not_enough_pairs_returns_early(self):
        trainer = make_trainer()
        with patch.object(trainer, "_get_collector") as mock_gc:
            mock_gc.return_value.collect_and_analyze.return_value = make_analyzed(
                n_pairs=2, ready=False
            )
            result = trainer.run_sweep()
        assert result.success is False
        assert result.error is not None
        assert result.sweep is None

    def test_status_sweep_done(self):
        trainer = make_trainer()
        with patch.object(trainer, "_get_collector") as mock_gc, \
             patch.object(trainer, "_get_builder") as mock_gb, \
             patch.object(trainer, "_get_keras") as mock_gk:
            mock_gc.return_value.collect_and_analyze.return_value = make_analyzed()
            mock_gb.return_value.build_split.return_value = make_split()
            mock_gk.return_value.run_sweep.return_value = make_sweep()
            trainer.run_sweep()
        assert trainer.status().sweep_done is True

    def test_best_eval_loss_stored(self):
        trainer = make_trainer()
        with patch.object(trainer, "_get_collector") as mock_gc, \
             patch.object(trainer, "_get_builder") as mock_gb, \
             patch.object(trainer, "_get_keras") as mock_gk:
            mock_gc.return_value.collect_and_analyze.return_value = make_analyzed()
            mock_gb.return_value.build_split.return_value = make_split()
            mock_gk.return_value.run_sweep.return_value = make_sweep(0.18)
            trainer.run_sweep()
        assert abs(trainer.status().best_eval_loss - 0.18) < 1e-9

    def test_exception_caught(self):
        trainer = make_trainer()
        with patch.object(trainer, "_get_collector") as mock_gc:
            mock_gc.return_value.collect_and_analyze.side_effect = RuntimeError("disk full")
            result = trainer.run_sweep()
        assert result.success is False
        assert "disk full" in result.error


# ---------------------------------------------------------------------------
# Trainer.run() — full pipeline
# ---------------------------------------------------------------------------

class TestRun:
    def _patch_all(self, trainer, sweep=True, training_success=True):
        """Return context manager patching all sub-components."""
        from contextlib import ExitStack
        stack = ExitStack()
        mock_gc = stack.enter_context(patch.object(trainer, "_get_collector"))
        mock_gb = stack.enter_context(patch.object(trainer, "_get_builder"))
        mock_gk = stack.enter_context(patch.object(trainer, "_get_keras"))

        mock_gc.return_value.collect_and_analyze.return_value = make_analyzed()
        mock_gb.return_value.build_split.return_value = make_split()
        mock_gk.return_value.run_sweep.return_value = make_sweep()

        mock_torch = stack.enter_context(
            patch("src.training.trainer.TorchTrainer")
        )
        mock_torch.return_value.train.return_value = make_training_result(
            success=training_success
        )
        return stack, mock_gc, mock_gb, mock_gk, mock_torch

    def test_returns_pipeline_result(self):
        trainer = make_trainer()
        with self._patch_all(trainer)[0]:
            result = trainer.run()
        assert isinstance(result, PipelineResult)

    def test_success_when_training_succeeds(self):
        trainer = make_trainer()
        with self._patch_all(trainer)[0]:
            result = trainer.run()
        assert result.success is True

    def test_training_result_populated(self):
        trainer = make_trainer()
        with self._patch_all(trainer)[0]:
            result = trainer.run()
        assert isinstance(result.training, TrainingResult)

    def test_analyzed_populated(self):
        trainer = make_trainer()
        with self._patch_all(trainer)[0]:
            result = trainer.run()
        assert result.analyzed is not None

    def test_split_populated(self):
        trainer = make_trainer()
        with self._patch_all(trainer)[0]:
            result = trainer.run()
        assert result.split is not None

    def test_sweep_populated_by_default(self):
        trainer = make_trainer()
        with self._patch_all(trainer)[0]:
            result = trainer.run(skip_sweep=False)
        assert result.sweep is not None

    def test_sweep_skipped(self):
        trainer = make_trainer()
        with self._patch_all(trainer)[0]:
            result = trainer.run(skip_sweep=True)
        assert result.sweep is None

    def test_stage_done_on_success(self):
        trainer = make_trainer()
        with self._patch_all(trainer)[0]:
            trainer.run()
        assert trainer.status().stage == PipelineStage.DONE

    def test_stage_failed_when_training_fails(self):
        trainer = make_trainer()
        with self._patch_all(trainer, training_success=False)[0]:
            trainer.run()
        assert trainer.status().stage == PipelineStage.FAILED

    def test_not_enough_pairs_returns_early(self):
        trainer = make_trainer()
        with patch.object(trainer, "_get_collector") as mock_gc:
            mock_gc.return_value.collect_and_analyze.return_value = make_analyzed(
                n_pairs=2, ready=False
            )
            result = trainer.run()
        assert result.success is False
        assert result.training is None

    def test_exception_caught(self):
        trainer = make_trainer()
        with patch.object(trainer, "_get_collector") as mock_gc:
            mock_gc.return_value.collect_and_analyze.side_effect = ValueError("bad data")
            result = trainer.run()
        assert result.success is False
        assert "bad data" in result.error

    def test_duration_recorded(self):
        trainer = make_trainer()
        with self._patch_all(trainer)[0]:
            result = trainer.run()
        assert result.total_duration_seconds >= 0.0

    def test_training_done_status(self):
        trainer = make_trainer()
        with self._patch_all(trainer)[0]:
            trainer.run()
        assert trainer.status().training_done is True


# ---------------------------------------------------------------------------
# Trainer._apply_sweep_to_torch_config
# ---------------------------------------------------------------------------

class TestApplySweepToTorchConfig:
    def test_learning_rate_transferred(self):
        from src.training.torch_trainer import TrainerConfig
        torch_cfg = TrainerConfig(learning_rate=1e-5)
        keras_cfg = ExperimentConfig(learning_rate=2e-4, batch_size=16)
        result = Trainer._apply_sweep_to_torch_config(torch_cfg, keras_cfg)
        assert abs(result.learning_rate - 2e-4) < 1e-10

    def test_batch_size_transferred(self):
        from src.training.torch_trainer import TrainerConfig
        torch_cfg = TrainerConfig()
        keras_cfg = ExperimentConfig(learning_rate=2e-4, batch_size=32)
        result = Trainer._apply_sweep_to_torch_config(torch_cfg, keras_cfg)
        assert result.per_device_train_batch_size == 32
        assert result.per_device_eval_batch_size == 32


# ---------------------------------------------------------------------------
# Trainer._pairs_to_embeddings
# ---------------------------------------------------------------------------

class TestPairsToEmbeddings:
    def test_returns_list_of_dicts(self):
        pairs = [make_pair(i) for i in range(3)]
        result = Trainer._pairs_to_embeddings(pairs)
        assert isinstance(result, list)
        assert all(isinstance(r, dict) for r in result)

    def test_correct_length(self):
        pairs = [make_pair(i) for i in range(4)]
        result = Trainer._pairs_to_embeddings(pairs)
        assert len(result) == 4

    def test_has_chosen_and_rejected_keys(self):
        pairs = [make_pair(0)]
        result = Trainer._pairs_to_embeddings(pairs)
        assert "chosen_embedding" in result[0]
        assert "rejected_embedding" in result[0]

    def test_embedding_dim_384(self):
        pairs = [make_pair(0)]
        result = Trainer._pairs_to_embeddings(pairs)
        assert len(result[0]["chosen_embedding"]) == 384
        assert len(result[0]["rejected_embedding"]) == 384

    def test_empty_pairs_returns_empty(self):
        result = Trainer._pairs_to_embeddings([])
        assert result == []