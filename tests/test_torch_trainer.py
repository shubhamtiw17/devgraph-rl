from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from src.training.torch_trainer import (
    EpochMetrics,
    TrainerConfig,
    TrainingResult,
    TorchTrainer,
    DEFAULT_BASE_MODEL,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_config(**kwargs) -> TrainerConfig:
    defaults = dict(
        base_model="test-model",
        output_dir="/tmp/test_trainer_output",
        n_epochs=2,
        push_to_hub=False,
    )
    defaults.update(kwargs)
    return TrainerConfig(**defaults)


def make_epoch_metrics(epoch: int = 1, loss: float = 0.5) -> EpochMetrics:
    return EpochMetrics(
        epoch=epoch,
        loss=loss,
        reward_chosen=0.8,
        reward_rejected=0.3,
        reward_delta=0.5,
        duration_seconds=10.0,
    )


def make_dataset_dict(n_train: int = 10, n_eval: int = 2):
    """Mock DatasetDict with __len__ and .get()."""
    train_ds = MagicMock()
    train_ds.__len__ = lambda self: n_train
    eval_ds = MagicMock()
    eval_ds.__len__ = lambda self: n_eval

    dd = MagicMock()
    dd.get = lambda key, default=None: (
        train_ds if key == "train" else eval_ds if key == "eval" else default
    )
    dd.__getitem__ = lambda self, key: (
        train_ds if key == "train" else eval_ds
    )
    return dd


# ---------------------------------------------------------------------------
# EpochMetrics
# ---------------------------------------------------------------------------

class TestEpochMetrics:
    def test_to_dict_keys(self):
        m = make_epoch_metrics()
        d = m.to_dict()
        assert set(d.keys()) == {
            "epoch", "loss", "reward_chosen",
            "reward_rejected", "reward_delta", "duration_seconds",
        }

    def test_to_dict_values(self):
        m = EpochMetrics(
            epoch=2, loss=0.3, reward_chosen=0.9,
            reward_rejected=0.4, reward_delta=0.5, duration_seconds=20.0,
        )
        d = m.to_dict()
        assert d["epoch"] == 2
        assert d["loss"] == 0.3
        assert d["reward_delta"] == 0.5


# ---------------------------------------------------------------------------
# TrainerConfig
# ---------------------------------------------------------------------------

class TestTrainerConfig:
    def test_defaults(self):
        c = TrainerConfig()
        assert c.base_model == DEFAULT_BASE_MODEL
        assert c.n_epochs == 3
        assert c.lora_r == 16
        assert c.push_to_hub is False

    def test_custom_values(self):
        c = make_config(n_epochs=5, learning_rate=1e-4)
        assert c.n_epochs == 5
        assert c.learning_rate == 1e-4

    def test_to_dict_keys(self):
        c = make_config()
        d = c.to_dict()
        assert "base_model" in d
        assert "n_epochs" in d
        assert "lora_r" in d
        assert "push_to_hub" in d

    def test_lora_target_modules_default(self):
        c = TrainerConfig()
        assert "q_proj" in c.lora_target_modules
        assert "v_proj" in c.lora_target_modules


# ---------------------------------------------------------------------------
# TrainingResult
# ---------------------------------------------------------------------------

class TestTrainingResult:
    def test_final_reward_delta_no_epochs(self):
        r = TrainingResult(base_model="m", n_train=0, n_eval=0, n_epochs=0)
        assert r.final_reward_delta == 0.0

    def test_final_reward_delta_with_epochs(self):
        r = TrainingResult(
            base_model="m", n_train=10, n_eval=2, n_epochs=2,
            epoch_metrics=[
                make_epoch_metrics(epoch=1, loss=0.5),
                EpochMetrics(epoch=2, loss=0.3,
                             reward_chosen=0.9, reward_rejected=0.2,
                             reward_delta=0.7, duration_seconds=10.0),
            ],
        )
        assert abs(r.final_reward_delta - 0.7) < 1e-9

    def test_improved_true(self):
        r = TrainingResult(
            base_model="m", n_train=10, n_eval=2, n_epochs=2,
            epoch_metrics=[
                EpochMetrics(1, 0.5, 0.6, 0.4, 0.2, 10.0),
                EpochMetrics(2, 0.3, 0.9, 0.2, 0.7, 10.0),
            ],
        )
        assert r.improved is True

    def test_improved_false_single_epoch(self):
        r = TrainingResult(
            base_model="m", n_train=10, n_eval=2, n_epochs=1,
            epoch_metrics=[EpochMetrics(1, 0.5, 0.6, 0.4, 0.2, 10.0)],
        )
        assert r.improved is False

    def test_improved_false_no_growth(self):
        r = TrainingResult(
            base_model="m", n_train=10, n_eval=2, n_epochs=2,
            epoch_metrics=[
                EpochMetrics(1, 0.5, 0.6, 0.4, 0.2, 10.0),
                EpochMetrics(2, 0.4, 0.5, 0.4, 0.1, 10.0),  # delta shrank
            ],
        )
        assert r.improved is False

    def test_summary_string(self):
        r = TrainingResult(
            base_model="test-model", n_train=10, n_eval=2,
            n_epochs=2, success=True,
            epoch_metrics=[make_epoch_metrics()],
        )
        s = r.summary()
        assert "test-model" in s
        assert "Success" in s

    def test_save_and_load(self, tmp_path):
        r = TrainingResult(
            base_model="test-model", n_train=10, n_eval=2,
            n_epochs=2, success=True, best_loss=0.3,
            epoch_metrics=[make_epoch_metrics(epoch=1, loss=0.3)],
        )
        path = tmp_path / "result.json"
        r.save(path)
        assert path.exists()

        loaded = TrainingResult.load(path)
        assert loaded.base_model == "test-model"
        assert loaded.success is True
        assert loaded.best_loss == 0.3
        assert len(loaded.epoch_metrics) == 1
        assert loaded.epoch_metrics[0].epoch == 1

    def test_save_creates_parent_dirs(self, tmp_path):
        r = TrainingResult(
            base_model="m", n_train=0, n_eval=0, n_epochs=0,
        )
        nested = tmp_path / "a" / "b" / "c" / "result.json"
        r.save(nested)
        assert nested.exists()

    def test_load_epoch_metrics_parsed(self, tmp_path):
        r = TrainingResult(
            base_model="m", n_train=5, n_eval=1, n_epochs=2,
            epoch_metrics=[
                make_epoch_metrics(epoch=1, loss=0.5),
                make_epoch_metrics(epoch=2, loss=0.3),
            ],
        )
        path = tmp_path / "r.json"
        r.save(path)
        loaded = TrainingResult.load(path)
        assert len(loaded.epoch_metrics) == 2
        assert isinstance(loaded.epoch_metrics[0], EpochMetrics)


# ---------------------------------------------------------------------------
# TorchTrainer.validate_config()
# ---------------------------------------------------------------------------

class TestValidateConfig:
    def test_valid_config_no_warnings(self):
        config = make_config(
            n_epochs=3, learning_rate=2e-4,
            lora_r=16, push_to_hub=False,
            bf16=False, fp16=False,
            max_length=512, max_prompt_length=256,
        )
        trainer = TorchTrainer(config)
        warnings = trainer.validate_config()
        assert warnings == []

    def test_zero_epochs_warns(self):
        trainer = TorchTrainer(make_config(n_epochs=0))
        assert any("n_epochs" in w for w in trainer.validate_config())

    def test_negative_lr_warns(self):
        trainer = TorchTrainer(make_config(learning_rate=-0.1))
        assert any("learning_rate" in w for w in trainer.validate_config())

    def test_unusual_lora_r_warns(self):
        trainer = TorchTrainer(make_config(lora_r=7))
        assert any("lora_r" in w for w in trainer.validate_config())

    def test_standard_lora_r_no_warn(self):
        for r in (4, 8, 16, 32, 64):
            trainer = TorchTrainer(make_config(lora_r=r))
            warns = trainer.validate_config()
            assert not any("lora_r" in w for w in warns), f"Unexpected warn for lora_r={r}"

    def test_push_to_hub_without_model_id_warns(self):
        trainer = TorchTrainer(make_config(push_to_hub=True, hub_model_id=None))
        assert any("hub_model_id" in w for w in trainer.validate_config())

    def test_push_to_hub_without_token_warns(self, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        trainer = TorchTrainer(make_config(
            push_to_hub=True, hub_model_id="user/model", hub_token=None,
        ))
        assert any("hub_token" in w or "HF_TOKEN" in w
                   for w in trainer.validate_config())

    def test_push_to_hub_with_token_no_warn(self):
        trainer = TorchTrainer(make_config(
            push_to_hub=True, hub_model_id="user/model", hub_token="hf_abc",
        ))
        warns = trainer.validate_config()
        assert not any("hub_token" in w for w in warns)

    def test_bf16_and_fp16_both_true_warns(self):
        trainer = TorchTrainer(make_config(bf16=True, fp16=True))
        assert any("bf16" in w or "fp16" in w for w in trainer.validate_config())

    def test_prompt_length_gte_max_length_warns(self):
        trainer = TorchTrainer(make_config(max_length=256, max_prompt_length=256))
        assert any("max_prompt_length" in w for w in trainer.validate_config())

    def test_prompt_length_lt_max_length_no_warn(self):
        trainer = TorchTrainer(make_config(max_length=512, max_prompt_length=256))
        warns = trainer.validate_config()
        assert not any("max_prompt_length" in w for w in warns)


# ---------------------------------------------------------------------------
# TorchTrainer.train() — fully mocked
# ---------------------------------------------------------------------------

class TestTrain:
    """
    All ML deps are mocked. We test orchestration logic only:
    - helpers are called in the right order
    - metrics are recorded correctly
    - checkpoint is saved
    - errors are caught and returned in result
    - result JSON is written to disk
    """

    def _make_mock_dpo_trainer(self, log_history=None):
        """Build a mock DPOTrainer with configurable log_history."""
        dpo = MagicMock()
        dpo.train.return_value = MagicMock()
        dpo.state = MagicMock()
        dpo.state.log_history = log_history or [
            {"epoch": 1, "loss": 0.5, "rewards/chosen": 0.7, "rewards/rejected": 0.3},
            {"epoch": 2, "loss": 0.3, "rewards/chosen": 0.85, "rewards/rejected": 0.25},
        ]
        return dpo

    def _mock_imports(self):
        """Patch heavy ML imports so tests run without peft/trl/transformers."""
        import sys
        mocks = {}
        for mod in ["peft", "trl", "transformers"]:
            if mod not in sys.modules:
                mocks[mod] = MagicMock()
                sys.modules[mod] = mocks[mod]
        # Sub-modules needed by lazy imports in train()
        for sub in ["peft.LoraConfig", "peft.get_peft_model", "peft.TaskType",
                    "trl.DPOTrainer", "trl.DPOConfig",
                    "transformers.AutoModelForCausalLM",
                    "transformers.AutoTokenizer"]:
            parts = sub.split(".")
            if parts[0] in sys.modules:
                pass  # already mocked
        return mocks

    def test_train_returns_training_result(self, tmp_path):
        config = make_config(output_dir=str(tmp_path))
        trainer = TorchTrainer(config)
        fake_metrics = [make_epoch_metrics(1, 0.5), make_epoch_metrics(2, 0.3)]

        with patch.object(trainer, "_load_tokenizer", return_value=MagicMock()), \
             patch.object(trainer, "_load_model", return_value=MagicMock()), \
             patch.object(trainer, "_apply_lora", return_value=MagicMock()), \
             patch.object(trainer, "_build_dpo_trainer",
                          return_value=self._make_mock_dpo_trainer()), \
             patch.object(trainer, "_run_training", return_value=fake_metrics):
            result = trainer.train(make_dataset_dict())

        assert isinstance(result, TrainingResult)

    def test_train_success_flag(self, tmp_path):
        config = make_config(output_dir=str(tmp_path))
        trainer = TorchTrainer(config)
        fake_metrics = [make_epoch_metrics()]

        with patch.object(trainer, "_load_tokenizer", return_value=MagicMock()), \
             patch.object(trainer, "_load_model", return_value=MagicMock()), \
             patch.object(trainer, "_apply_lora", return_value=MagicMock()), \
             patch.object(trainer, "_build_dpo_trainer",
                          return_value=self._make_mock_dpo_trainer()), \
             patch.object(trainer, "_run_training", return_value=fake_metrics):
            result = trainer.train(make_dataset_dict())

        assert result.success is True
        assert result.error is None

    def test_train_epoch_metrics_recorded(self, tmp_path):
        config = make_config(output_dir=str(tmp_path), n_epochs=2)
        trainer = TorchTrainer(config)
        fake_metrics = [make_epoch_metrics(1, 0.5), make_epoch_metrics(2, 0.3)]

        with patch.object(trainer, "_load_tokenizer", return_value=MagicMock()), \
             patch.object(trainer, "_load_model", return_value=MagicMock()), \
             patch.object(trainer, "_apply_lora", return_value=MagicMock()), \
             patch.object(trainer, "_build_dpo_trainer",
                          return_value=self._make_mock_dpo_trainer()), \
             patch.object(trainer, "_run_training", return_value=fake_metrics):
            result = trainer.train(make_dataset_dict())

        assert len(result.epoch_metrics) == 2
        assert result.epoch_metrics[0].loss == 0.5
        assert result.epoch_metrics[1].loss == 0.3

    def test_train_best_loss_set(self, tmp_path):
        config = make_config(output_dir=str(tmp_path))
        trainer = TorchTrainer(config)
        fake_metrics = [make_epoch_metrics(1, 0.5), make_epoch_metrics(2, 0.3)]

        with patch.object(trainer, "_load_tokenizer", return_value=MagicMock()), \
             patch.object(trainer, "_load_model", return_value=MagicMock()), \
             patch.object(trainer, "_apply_lora", return_value=MagicMock()), \
             patch.object(trainer, "_build_dpo_trainer",
                          return_value=self._make_mock_dpo_trainer()), \
             patch.object(trainer, "_run_training", return_value=fake_metrics):
            result = trainer.train(make_dataset_dict())

        assert abs(result.best_loss - 0.3) < 1e-9

    def test_train_checkpoint_path_set(self, tmp_path):
        config = make_config(output_dir=str(tmp_path))
        trainer = TorchTrainer(config)

        with patch.object(trainer, "_load_tokenizer", return_value=MagicMock()), \
             patch.object(trainer, "_load_model", return_value=MagicMock()), \
             patch.object(trainer, "_apply_lora", return_value=MagicMock()), \
             patch.object(trainer, "_build_dpo_trainer",
                          return_value=self._make_mock_dpo_trainer()), \
             patch.object(trainer, "_run_training",
                          return_value=[make_epoch_metrics()]):
            result = trainer.train(make_dataset_dict())

        assert result.checkpoint_path == str(tmp_path)

    def test_train_result_json_written(self, tmp_path):
        config = make_config(output_dir=str(tmp_path))
        trainer = TorchTrainer(config)

        with patch.object(trainer, "_load_tokenizer", return_value=MagicMock()), \
             patch.object(trainer, "_load_model", return_value=MagicMock()), \
             patch.object(trainer, "_apply_lora", return_value=MagicMock()), \
             patch.object(trainer, "_build_dpo_trainer",
                          return_value=self._make_mock_dpo_trainer()), \
             patch.object(trainer, "_run_training",
                          return_value=[make_epoch_metrics()]):
            trainer.train(make_dataset_dict())

        result_file = tmp_path / "training_result.json"
        assert result_file.exists()
        data = json.loads(result_file.read_text())
        assert data["success"] is True

    def test_train_n_train_n_eval_set(self, tmp_path):
        config = make_config(output_dir=str(tmp_path))
        trainer = TorchTrainer(config)

        with patch.object(trainer, "_load_tokenizer", return_value=MagicMock()), \
             patch.object(trainer, "_load_model", return_value=MagicMock()), \
             patch.object(trainer, "_apply_lora", return_value=MagicMock()), \
             patch.object(trainer, "_build_dpo_trainer",
                          return_value=self._make_mock_dpo_trainer()), \
             patch.object(trainer, "_run_training",
                          return_value=[make_epoch_metrics()]):
            result = trainer.train(make_dataset_dict(n_train=10, n_eval=2))

        assert result.n_train == 10
        assert result.n_eval == 2

    def test_import_error_caught(self, tmp_path):
        config = make_config(output_dir=str(tmp_path))
        trainer = TorchTrainer(config)

        with patch.object(trainer, "_load_tokenizer",
                          side_effect=ImportError("torch not found")):
            result = trainer.train(make_dataset_dict())

        assert result.success is False
        assert result.error is not None
        assert "torch" in result.error.lower() or "not" in result.error.lower()

    def test_runtime_error_caught(self, tmp_path):
        config = make_config(output_dir=str(tmp_path))
        trainer = TorchTrainer(config)

        with patch.object(trainer, "_load_tokenizer", return_value=MagicMock()), \
             patch.object(trainer, "_load_model",
                          side_effect=RuntimeError("CUDA out of memory")):
            result = trainer.train(make_dataset_dict())

        assert result.success is False
        assert "CUDA out of memory" in result.error

    def test_hub_push_called_when_configured(self, tmp_path):
        config = make_config(
            output_dir=str(tmp_path),
            push_to_hub=True,
            hub_model_id="user/test-model",
            hub_token="hf_test",
        )
        trainer = TorchTrainer(config)
        mock_dpo = self._make_mock_dpo_trainer()

        with patch.object(trainer, "_load_tokenizer", return_value=MagicMock()), \
             patch.object(trainer, "_load_model", return_value=MagicMock()), \
             patch.object(trainer, "_apply_lora", return_value=MagicMock()), \
             patch.object(trainer, "_build_dpo_trainer", return_value=mock_dpo), \
             patch.object(trainer, "_run_training",
                          return_value=[make_epoch_metrics()]):
            result = trainer.train(make_dataset_dict())

        mock_dpo.push_to_hub.assert_called_once_with("user/test-model")
        assert result.hub_model_id == "user/test-model"

    def test_hub_push_not_called_when_disabled(self, tmp_path):
        config = make_config(output_dir=str(tmp_path), push_to_hub=False)
        trainer = TorchTrainer(config)
        mock_dpo = self._make_mock_dpo_trainer()

        with patch.object(trainer, "_load_tokenizer", return_value=MagicMock()), \
             patch.object(trainer, "_load_model", return_value=MagicMock()), \
             patch.object(trainer, "_apply_lora", return_value=MagicMock()), \
             patch.object(trainer, "_build_dpo_trainer", return_value=mock_dpo), \
             patch.object(trainer, "_run_training",
                          return_value=[make_epoch_metrics()]):
            trainer.train(make_dataset_dict())

        mock_dpo.push_to_hub.assert_not_called()

    def test_duration_recorded(self, tmp_path):
        config = make_config(output_dir=str(tmp_path))
        trainer = TorchTrainer(config)

        with patch.object(trainer, "_load_tokenizer", return_value=MagicMock()), \
             patch.object(trainer, "_load_model", return_value=MagicMock()), \
             patch.object(trainer, "_apply_lora", return_value=MagicMock()), \
             patch.object(trainer, "_build_dpo_trainer",
                          return_value=self._make_mock_dpo_trainer()), \
             patch.object(trainer, "_run_training",
                          return_value=[make_epoch_metrics()]):
            result = trainer.train(make_dataset_dict())

        assert result.total_duration_seconds >= 0.0


# ---------------------------------------------------------------------------
# Static helpers
# ---------------------------------------------------------------------------

class TestStaticHelpers:
    def test_extract_loss_from_log(self):
        log = [
            {"epoch": 1, "loss": 0.5},
            {"epoch": 1, "loss": 0.45},
        ]
        loss = TorchTrainer._extract_loss(log, epoch=0)
        assert abs(loss - 0.45) < 1e-9

    def test_extract_loss_fallback(self):
        log = [{"loss": 0.6}]
        loss = TorchTrainer._extract_loss(log, epoch=99)
        assert abs(loss - 0.6) < 1e-9

    def test_extract_loss_empty_log(self):
        loss = TorchTrainer._extract_loss([], epoch=0)
        assert loss == 0.0

    def test_extract_metric_found(self):
        log = [{"rewards/chosen": 0.8}, {"rewards/chosen": 0.9}]
        val = TorchTrainer._extract_metric(log, "rewards/chosen", 0.0)
        assert abs(val - 0.9) < 1e-9

    def test_extract_metric_missing(self):
        log = [{"loss": 0.5}]
        val = TorchTrainer._extract_metric(log, "rewards/chosen", 0.42)
        assert abs(val - 0.42) < 1e-9

    def test_extract_metric_empty_log(self):
        val = TorchTrainer._extract_metric([], "rewards/chosen", 0.0)
        assert val == 0.0