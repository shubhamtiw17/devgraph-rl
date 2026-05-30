from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.training.keras_experiments import (
    DEFAULT_SWEEP_SPACE,
    ExperimentConfig,
    ExperimentRun,
    KerasExperiments,
    SweepResult,
    build_configs_from_space,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_config(**kwargs) -> ExperimentConfig:
    defaults = dict(
        learning_rate=2e-4,
        batch_size=16,
        hidden_dim=128,
        dropout=0.1,
        n_epochs=3,
        embedding_dim=8,   # tiny dim for tests
        experiment_id="test_exp",
    )
    defaults.update(kwargs)
    return ExperimentConfig(**defaults)


def make_run(
    eval_loss: float = 0.3,
    train_loss: float = 0.25,
    success: bool = True,
    config: ExperimentConfig = None,
) -> ExperimentRun:
    return ExperimentRun(
        config=config or make_config(),
        train_loss=train_loss,
        eval_loss=eval_loss,
        train_losses=[train_loss + 0.1, train_loss],
        eval_losses=[eval_loss + 0.1, eval_loss],
        duration_seconds=5.0,
        success=success,
        error=None if success else "something went wrong",
    )


def make_embedding_data(n: int, dim: int = 8) -> list[dict]:
    """Fake embedding data for _prepare_data tests."""
    rng = np.random.default_rng(42)
    return [
        {
            "chosen_embedding":   rng.standard_normal(dim).tolist(),
            "rejected_embedding": rng.standard_normal(dim).tolist(),
        }
        for _ in range(n)
    ]


def make_experiments(dim: int = 8) -> KerasExperiments:
    return KerasExperiments(
        experiment_dir=Path("/tmp/test_keras_experiments"),
        embedding_dim=dim,
    )


def make_fake_history(n_epochs: int = 3) -> dict:
    """Fake Keras history.history dict."""
    return {
        "loss":     [0.5 - i * 0.05 for i in range(n_epochs)],
        "val_loss": [0.55 - i * 0.04 for i in range(n_epochs)],
        "accuracy": [0.6 + i * 0.05 for i in range(n_epochs)],
    }


# ---------------------------------------------------------------------------
# ExperimentConfig
# ---------------------------------------------------------------------------

class TestExperimentConfig:
    def test_defaults(self):
        c = ExperimentConfig()
        assert c.learning_rate == 2e-4
        assert c.batch_size == 16
        assert c.hidden_dim == 128
        assert c.dropout == 0.1
        assert c.n_epochs == 5
        assert c.embedding_dim == 384

    def test_to_dict_keys(self):
        c = make_config()
        d = c.to_dict()
        assert set(d.keys()) == {
            "learning_rate", "batch_size", "hidden_dim",
            "dropout", "n_epochs", "embedding_dim", "experiment_id",
        }

    def test_to_dict_values(self):
        c = make_config(learning_rate=1e-3, batch_size=8)
        d = c.to_dict()
        assert d["learning_rate"] == 1e-3
        assert d["batch_size"] == 8

    def test_from_dict_roundtrip(self):
        c = make_config(learning_rate=5e-5, hidden_dim=64)
        c2 = ExperimentConfig.from_dict(c.to_dict())
        assert c2.learning_rate == 5e-5
        assert c2.hidden_dim == 64
        assert c2.experiment_id == c.experiment_id


# ---------------------------------------------------------------------------
# ExperimentRun
# ---------------------------------------------------------------------------

class TestExperimentRun:
    def test_overfit_gap_positive(self):
        run = make_run(train_loss=0.2, eval_loss=0.4)
        assert abs(run.overfit_gap - 0.2) < 1e-9

    def test_overfit_gap_negative(self):
        # eval better than train — unusual but valid
        run = make_run(train_loss=0.4, eval_loss=0.2)
        assert abs(run.overfit_gap - (-0.2)) < 1e-9

    def test_to_dict_keys(self):
        run = make_run()
        d = run.to_dict()
        assert "config" in d
        assert "train_loss" in d
        assert "eval_loss" in d
        assert "train_losses" in d
        assert "eval_losses" in d
        assert "overfit_gap" in d
        assert "success" in d

    def test_from_dict_roundtrip(self):
        run = make_run(eval_loss=0.42, train_loss=0.31)
        run2 = ExperimentRun.from_dict(run.to_dict())
        assert abs(run2.eval_loss - 0.42) < 1e-9
        assert abs(run2.train_loss - 0.31) < 1e-9
        assert run2.success is True

    def test_from_dict_failed_run(self):
        run = make_run(success=False)
        run2 = ExperimentRun.from_dict(run.to_dict())
        assert run2.success is False
        assert run2.error is not None


# ---------------------------------------------------------------------------
# SweepResult
# ---------------------------------------------------------------------------

class TestSweepResult:
    def _make_sweep(self, n_runs: int = 3) -> SweepResult:
        runs = [make_run(eval_loss=0.5 - i * 0.1, config=make_config(
            experiment_id=f"exp_{i}")) for i in range(n_runs)]
        return SweepResult(
            runs=runs,
            best_config=runs[-1].config,
            best_eval_loss=runs[-1].eval_loss,
            total_duration_seconds=60.0,
            sweep_space={"learning_rate": [1e-3, 2e-4]},
        )

    def test_n_runs(self):
        sweep = self._make_sweep(3)
        assert sweep.n_runs == 3

    def test_successful_runs_all_pass(self):
        sweep = self._make_sweep(3)
        assert len(sweep.successful_runs) == 3

    def test_successful_runs_filters_failures(self):
        runs = [make_run(success=True), make_run(success=False)]
        sweep = SweepResult(
            runs=runs,
            best_config=runs[0].config,
            best_eval_loss=runs[0].eval_loss,
            total_duration_seconds=10.0,
        )
        assert len(sweep.successful_runs) == 1

    def test_ranked_runs_ascending_eval_loss(self):
        runs = [
            make_run(eval_loss=0.5, config=make_config(experiment_id="a")),
            make_run(eval_loss=0.2, config=make_config(experiment_id="b")),
            make_run(eval_loss=0.8, config=make_config(experiment_id="c")),
        ]
        sweep = SweepResult(
            runs=runs,
            best_config=runs[1].config,
            best_eval_loss=0.2,
            total_duration_seconds=30.0,
        )
        ranked = sweep.ranked_runs
        losses = [r.eval_loss for r in ranked]
        assert losses == sorted(losses)

    def test_summary_string(self):
        sweep = self._make_sweep(3)
        s = sweep.summary()
        assert "Runs" in s
        assert "Best eval loss" in s

    def test_save_and_load(self, tmp_path):
        sweep = self._make_sweep(2)
        path = tmp_path / "sweep.json"
        sweep.save(path)
        assert path.exists()

        loaded = SweepResult.load(path)
        assert loaded.n_runs == 2
        assert abs(loaded.best_eval_loss - sweep.best_eval_loss) < 1e-9

    def test_save_creates_parent_dirs(self, tmp_path):
        sweep = self._make_sweep(1)
        nested = tmp_path / "a" / "b" / "sweep.json"
        sweep.save(nested)
        assert nested.exists()

    def test_load_runs_parsed_as_experiment_runs(self, tmp_path):
        sweep = self._make_sweep(2)
        path = tmp_path / "sweep.json"
        sweep.save(path)
        loaded = SweepResult.load(path)
        for run in loaded.runs:
            assert isinstance(run, ExperimentRun)

    def test_load_best_config_parsed(self, tmp_path):
        sweep = self._make_sweep(2)
        path = tmp_path / "sweep.json"
        sweep.save(path)
        loaded = SweepResult.load(path)
        assert isinstance(loaded.best_config, ExperimentConfig)


# ---------------------------------------------------------------------------
# build_configs_from_space
# ---------------------------------------------------------------------------

class TestBuildConfigsFromSpace:
    def test_returns_list_of_experiment_configs(self):
        space = {"learning_rate": [1e-3, 2e-4], "batch_size": [8, 16]}
        configs = build_configs_from_space(space)
        assert all(isinstance(c, ExperimentConfig) for c in configs)

    def test_correct_count(self):
        space = {"learning_rate": [1e-3, 2e-4], "batch_size": [8, 16]}
        configs = build_configs_from_space(space)
        assert len(configs) == 4   # 2 * 2

    def test_unique_experiment_ids(self):
        space = {"learning_rate": [1e-3, 2e-4, 5e-5], "batch_size": [8, 16]}
        configs = build_configs_from_space(space)
        ids = [c.experiment_id for c in configs]
        assert len(ids) == len(set(ids))

    def test_all_lr_values_covered(self):
        space = {"learning_rate": [1e-3, 2e-4, 5e-5]}
        configs = build_configs_from_space(space)
        lrs = {c.learning_rate for c in configs}
        assert lrs == {1e-3, 2e-4, 5e-5}

    def test_n_epochs_applied(self):
        space = {"learning_rate": [1e-3]}
        configs = build_configs_from_space(space, n_epochs=10)
        assert all(c.n_epochs == 10 for c in configs)

    def test_embedding_dim_applied(self):
        space = {"learning_rate": [1e-3]}
        configs = build_configs_from_space(space, embedding_dim=768)
        assert all(c.embedding_dim == 768 for c in configs)

    def test_default_sweep_space_non_empty(self):
        configs = build_configs_from_space(DEFAULT_SWEEP_SPACE)
        assert len(configs) > 0


# ---------------------------------------------------------------------------
# KerasExperiments._prepare_data
# ---------------------------------------------------------------------------

class TestPrepareData:
    def test_output_shape(self):
        ke = make_experiments(dim=8)
        data = make_embedding_data(5, dim=8)
        X, y = ke._prepare_data(data)
        assert X.shape == (10, 16)   # 5 pairs * 2, dim*2
        assert y.shape == (10,)

    def test_labels_alternating(self):
        ke = make_experiments(dim=8)
        data = make_embedding_data(3, dim=8)
        _, y = ke._prepare_data(data)
        # Pattern: 1, 0, 1, 0, 1, 0
        assert list(y) == [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]

    def test_empty_data_returns_empty(self):
        ke = make_experiments(dim=8)
        X, y = ke._prepare_data([])
        assert X.shape[0] == 0
        assert y.shape[0] == 0

    def test_float32_dtype(self):
        ke = make_experiments(dim=8)
        data = make_embedding_data(4, dim=8)
        X, y = ke._prepare_data(data)
        assert X.dtype == np.float32
        assert y.dtype == np.float32

    def test_missing_embeddings_use_zeros(self):
        ke = make_experiments(dim=4)
        data = [{}]   # no embeddings provided
        X, y = ke._prepare_data(data)
        assert X.shape == (2, 8)
        # Both rows should contain zeros
        assert np.all(X == 0.0)


# ---------------------------------------------------------------------------
# KerasExperiments.run_baseline() — mocked
# ---------------------------------------------------------------------------

class TestRunBaseline:
    def test_returns_experiment_run(self):
        ke = make_experiments()
        data = make_embedding_data(5, dim=8)
        fake_history = make_fake_history(3)

        with patch.object(ke, "_build_model", return_value=MagicMock()), \
             patch.object(ke, "_fit_model", return_value=fake_history):
            result = ke.run_baseline(data, data)

        assert isinstance(result, ExperimentRun)

    def test_baseline_success(self):
        ke = make_experiments()
        data = make_embedding_data(5, dim=8)

        with patch.object(ke, "_build_model", return_value=MagicMock()), \
             patch.object(ke, "_fit_model", return_value=make_fake_history(3)):
            result = ke.run_baseline(data, data)

        assert result.success is True
        assert result.error is None

    def test_baseline_experiment_id(self):
        ke = make_experiments()
        data = make_embedding_data(5, dim=8)

        with patch.object(ke, "_build_model", return_value=MagicMock()), \
             patch.object(ke, "_fit_model", return_value=make_fake_history(3)):
            result = ke.run_baseline(data, data)

        assert result.config.experiment_id == "baseline"

    def test_baseline_uses_provided_config(self):
        ke = make_experiments()
        data = make_embedding_data(5, dim=8)
        cfg = make_config(learning_rate=1e-3, experiment_id="custom")

        with patch.object(ke, "_build_model", return_value=MagicMock()), \
             patch.object(ke, "_fit_model", return_value=make_fake_history(3)):
            result = ke.run_baseline(data, data, config=cfg)

        assert result.config.learning_rate == 1e-3
        assert result.config.experiment_id == "custom"

    def test_baseline_losses_recorded(self):
        ke = make_experiments()
        data = make_embedding_data(5, dim=8)
        history = make_fake_history(3)

        with patch.object(ke, "_build_model", return_value=MagicMock()), \
             patch.object(ke, "_fit_model", return_value=history):
            result = ke.run_baseline(data, data)

        assert len(result.train_losses) == 3
        assert len(result.eval_losses) == 3
        assert abs(result.train_loss - history["loss"][-1]) < 1e-6
        assert abs(result.eval_loss - history["val_loss"][-1]) < 1e-6

    def test_baseline_import_error_caught(self):
        ke = make_experiments()
        data = make_embedding_data(5, dim=8)

        with patch.object(ke, "_build_model",
                          side_effect=ImportError("No module named keras")):
            result = ke.run_baseline(data, data)

        assert result.success is False
        assert result.error is not None

    def test_baseline_runtime_error_caught(self):
        ke = make_experiments()
        data = make_embedding_data(5, dim=8)

        with patch.object(ke, "_build_model",
                          side_effect=RuntimeError("OOM")):
            result = ke.run_baseline(data, data)

        assert result.success is False
        assert "OOM" in result.error


# ---------------------------------------------------------------------------
# KerasExperiments.run_sweep() — mocked
# ---------------------------------------------------------------------------

class TestRunSweep:
    def _patched_run_single(self, ke, histories: list[dict]):
        """
        Patch _run_single to return ExperimentRuns with given histories
        in sequence.
        """
        calls = iter(histories)

        def fake_run_single(train_data, eval_data, config):
            h = next(calls, make_fake_history(2))
            return ExperimentRun(
                config=config,
                train_loss=h["loss"][-1],
                eval_loss=h["val_loss"][-1],
                train_losses=h["loss"],
                eval_losses=h["val_loss"],
                duration_seconds=1.0,
                success=True,
            )
        return fake_run_single

    def test_returns_sweep_result(self):
        ke = make_experiments()
        data = make_embedding_data(5, dim=8)
        small_space = {"learning_rate": [1e-3, 2e-4]}

        with patch.object(ke, "_run_single", side_effect=lambda td, ed, cfg:
                ExperimentRun(config=cfg, train_loss=0.3, eval_loss=0.4,
                              duration_seconds=1.0, success=True)):
            result = ke.run_sweep(data, data, sweep_space=small_space)

        assert isinstance(result, SweepResult)

    def test_n_runs_matches_space(self):
        ke = make_experiments()
        data = make_embedding_data(5, dim=8)
        small_space = {"learning_rate": [1e-3, 2e-4], "batch_size": [8, 16]}

        with patch.object(ke, "_run_single", side_effect=lambda td, ed, cfg:
                ExperimentRun(config=cfg, train_loss=0.3, eval_loss=0.4,
                              duration_seconds=1.0, success=True)):
            result = ke.run_sweep(data, data, sweep_space=small_space)

        assert result.n_runs == 4

    def test_max_configs_respected(self):
        ke = make_experiments()
        data = make_embedding_data(5, dim=8)

        with patch.object(ke, "_run_single", side_effect=lambda td, ed, cfg:
                ExperimentRun(config=cfg, train_loss=0.3, eval_loss=0.4,
                              duration_seconds=1.0, success=True)):
            result = ke.run_sweep(data, data, max_configs=2)

        assert result.n_runs == 2

    def test_best_config_has_lowest_eval_loss(self):
        ke = make_experiments()
        data = make_embedding_data(5, dim=8)
        small_space = {"learning_rate": [1e-3, 2e-4, 5e-5]}
        eval_losses = iter([0.5, 0.2, 0.8])

        with patch.object(ke, "_run_single", side_effect=lambda td, ed, cfg:
                ExperimentRun(config=cfg, train_loss=0.3,
                              eval_loss=next(eval_losses),
                              duration_seconds=1.0, success=True)):
            result = ke.run_sweep(data, data, sweep_space=small_space)

        assert abs(result.best_eval_loss - 0.2) < 1e-9

    def test_all_failed_runs_returns_sweep(self):
        ke = make_experiments()
        data = make_embedding_data(5, dim=8)
        small_space = {"learning_rate": [1e-3]}

        with patch.object(ke, "_run_single", side_effect=lambda td, ed, cfg:
                ExperimentRun(config=cfg, train_loss=0.0, eval_loss=float("inf"),
                              duration_seconds=1.0, success=False,
                              error="crash")):
            result = ke.run_sweep(data, data, sweep_space=small_space)

        assert isinstance(result, SweepResult)
        assert result.best_eval_loss == float("inf")

    def test_sweep_saves_json(self, tmp_path):
        ke = KerasExperiments(experiment_dir=tmp_path, embedding_dim=8)
        data = make_embedding_data(5, dim=8)
        small_space = {"learning_rate": [1e-3]}

        with patch.object(ke, "_run_single", side_effect=lambda td, ed, cfg:
                ExperimentRun(config=cfg, train_loss=0.3, eval_loss=0.4,
                              duration_seconds=1.0, success=True)):
            ke.run_sweep(data, data, sweep_space=small_space)

        assert (tmp_path / "sweep_result.json").exists()

    def test_sweep_space_stored_in_result(self):
        ke = make_experiments()
        data = make_embedding_data(5, dim=8)
        small_space = {"learning_rate": [1e-3, 2e-4]}

        with patch.object(ke, "_run_single", side_effect=lambda td, ed, cfg:
                ExperimentRun(config=cfg, train_loss=0.3, eval_loss=0.4,
                              duration_seconds=1.0, success=True)):
            result = ke.run_sweep(data, data, sweep_space=small_space)

        assert "learning_rate" in result.sweep_space

    def test_ranked_runs_best_first(self):
        ke = make_experiments()
        data = make_embedding_data(5, dim=8)
        small_space = {"learning_rate": [1e-3, 2e-4, 5e-5]}
        eval_losses = iter([0.5, 0.2, 0.8])

        with patch.object(ke, "_run_single", side_effect=lambda td, ed, cfg:
                ExperimentRun(config=cfg, train_loss=0.3,
                              eval_loss=next(eval_losses),
                              duration_seconds=1.0, success=True)):
            result = ke.run_sweep(data, data, sweep_space=small_space)

        ranked_losses = [r.eval_loss for r in result.ranked_runs]
        assert ranked_losses == sorted(ranked_losses)