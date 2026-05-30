from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    # These are only imported for type hints in IDEs — never at runtime in CI
    from datasets import DatasetDict
    from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)

# Default base model — small enough for free Colab T4 GPU
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-0.5B"

# Local checkpoint directory
DEFAULT_OUTPUT_DIR = Path("./data/checkpoints")


# ---------------------------------------------------------------------------
# Result dataclasses — importable with zero ML deps
# ---------------------------------------------------------------------------

@dataclass
class EpochMetrics:
    epoch: int
    loss: float
    reward_chosen: float      # mean reward for chosen outputs
    reward_rejected: float    # mean reward for rejected outputs
    reward_delta: float       # chosen - rejected (should grow each epoch)
    duration_seconds: float

    def to_dict(self) -> dict:
        return {
            "epoch": self.epoch,
            "loss": self.loss,
            "reward_chosen": self.reward_chosen,
            "reward_rejected": self.reward_rejected,
            "reward_delta": self.reward_delta,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class TrainingResult:
    base_model: str
    n_train: int
    n_eval: int
    n_epochs: int
    epoch_metrics: list[EpochMetrics] = field(default_factory=list)
    best_loss: float = float("inf")
    hub_model_id: Optional[str] = None
    checkpoint_path: Optional[str] = None
    total_duration_seconds: float = 0.0
    success: bool = False
    error: Optional[str] = None

    @property
    def final_reward_delta(self) -> float:
        if not self.epoch_metrics:
            return 0.0
        return self.epoch_metrics[-1].reward_delta

    @property
    def improved(self) -> bool:
        if len(self.epoch_metrics) < 2:
            return False
        return (
            self.epoch_metrics[-1].reward_delta
            > self.epoch_metrics[0].reward_delta
        )

    def summary(self) -> str:
        return (
            f"Model: {self.base_model} | "
            f"Epochs: {self.n_epochs} | "
            f"Best loss: {self.best_loss:.4f} | "
            f"Final delta: {self.final_reward_delta:.3f} | "
            f"Improved: {self.improved} | "
            f"Success: {self.success}"
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "base_model": self.base_model,
            "n_train": self.n_train,
            "n_eval": self.n_eval,
            "n_epochs": self.n_epochs,
            "epoch_metrics": [m.to_dict() for m in self.epoch_metrics],
            "best_loss": self.best_loss,
            "hub_model_id": self.hub_model_id,
            "checkpoint_path": self.checkpoint_path,
            "total_duration_seconds": self.total_duration_seconds,
            "success": self.success,
            "error": self.error,
        }
        path.write_text(json.dumps(payload, indent=2))
        logger.info("Training result saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> "TrainingResult":
        data = json.loads(path.read_text())
        metrics = [EpochMetrics(**m) for m in data.pop("epoch_metrics", [])]
        return cls(epoch_metrics=metrics, **data)


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainerConfig:
    base_model: str = DEFAULT_BASE_MODEL
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    n_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    learning_rate: float = 2e-4
    max_length: int = 512
    max_prompt_length: int = 256

    # LoRA
    lora_r: int = 16           # rank — higher = more capacity, more VRAM
    lora_alpha: int = 32       # scaling factor
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )

    # HuggingFace Hub
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    hub_token: Optional[str] = None

    # Misc
    seed: int = 42
    logging_steps: int = 10
    save_steps: int = 100
    bf16: bool = False          # use bf16 if GPU supports it
    fp16: bool = False          # use fp16 as fallback

    def to_dict(self) -> dict:
        return {
            "base_model": self.base_model,
            "output_dir": self.output_dir,
            "n_epochs": self.n_epochs,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "learning_rate": self.learning_rate,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "push_to_hub": self.push_to_hub,
            "hub_model_id": self.hub_model_id,
        }


# ---------------------------------------------------------------------------
# TorchTrainer
# ---------------------------------------------------------------------------

class TorchTrainer:

    def __init__(self, config: Optional[TrainerConfig] = None) -> None:
        self.config = config or TrainerConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, dataset_dict: "DatasetDict") -> TrainingResult:
        start = time.time()
        result = TrainingResult(
            base_model=self.config.base_model,
            n_train=len(dataset_dict.get("train", [])),
            n_eval=len(dataset_dict.get("eval", [])),
            n_epochs=self.config.n_epochs,
        )

        try:
            # Lazy imports — only executed when train() is actually called
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import LoraConfig, get_peft_model, TaskType
            from trl import DPOTrainer, DPOConfig

            logger.info("Loading tokenizer: %s", self.config.base_model)
            tokenizer = self._load_tokenizer()

            logger.info("Loading base model: %s", self.config.base_model)
            model = self._load_model()

            logger.info("Applying LoRA adapters.")
            model = self._apply_lora(model)

            logger.info("Building DPO trainer.")
            dpo_trainer = self._build_dpo_trainer(model, tokenizer, dataset_dict)

            logger.info("Starting DPO training for %d epochs.", self.config.n_epochs)
            epoch_metrics = self._run_training(dpo_trainer)
            result.epoch_metrics = epoch_metrics

            # Best loss across epochs
            if epoch_metrics:
                result.best_loss = min(m.loss for m in epoch_metrics)

            # Save checkpoint locally
            output_path = Path(self.config.output_dir)
            dpo_trainer.save_model(str(output_path))
            result.checkpoint_path = str(output_path)
            logger.info("Checkpoint saved to %s", output_path)

            # Push to HuggingFace Hub if configured
            if self.config.push_to_hub and self.config.hub_model_id:
                logger.info("Pushing to Hub: %s", self.config.hub_model_id)
                dpo_trainer.push_to_hub(self.config.hub_model_id)
                result.hub_model_id = self.config.hub_model_id

            result.success = True

        except ImportError as exc:
            msg = (
                f"Training dependencies not installed: {exc}. "
                "Run: pip install torch transformers trl peft"
            )
            logger.error(msg)
            result.error = msg

        except Exception as exc:
            logger.error("Training failed: %s", exc, exc_info=True)
            result.error = str(exc)

        result.total_duration_seconds = time.time() - start

        # Always save result JSON regardless of success
        result_path = Path(self.config.output_dir) / "training_result.json"
        try:
            result.save(result_path)
        except Exception as exc:
            logger.warning("Could not save training result: %s", exc)

        return result

    def validate_config(self) -> list[str]:
        warnings: list[str] = []

        if self.config.n_epochs < 1:
            warnings.append("n_epochs must be >= 1")

        if self.config.learning_rate <= 0:
            warnings.append("learning_rate must be positive")

        if self.config.lora_r not in (4, 8, 16, 32, 64):
            warnings.append(
                f"lora_r={self.config.lora_r} is unusual; "
                "typical values: 4, 8, 16, 32, 64"
            )

        if self.config.push_to_hub and not self.config.hub_model_id:
            warnings.append("push_to_hub=True but hub_model_id is not set")

        if self.config.push_to_hub and not (
            self.config.hub_token or os.environ.get("HF_TOKEN")
        ):
            warnings.append(
                "push_to_hub=True but no hub_token or HF_TOKEN env var found"
            )

        if self.config.bf16 and self.config.fp16:
            warnings.append("bf16 and fp16 cannot both be True")

        if self.config.max_prompt_length >= self.config.max_length:
            warnings.append(
                "max_prompt_length should be less than max_length "
                f"({self.config.max_prompt_length} >= {self.config.max_length})"
            )

        return warnings

    def _load_tokenizer(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _load_model(self):
        import torch
        from transformers import AutoModelForCausalLM
        dtype = (
            torch.bfloat16 if self.config.bf16
            else torch.float16 if self.config.fp16
            else torch.float32
        )
        return AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            torch_dtype=dtype,
            trust_remote_code=True,
        )

    def _apply_lora(self, model):
        from peft import LoraConfig, get_peft_model, TaskType
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
        return get_peft_model(model, lora_config)

    def _build_dpo_trainer(self, model, tokenizer, dataset_dict):
        from trl import DPOTrainer, DPOConfig
        dpo_config = DPOConfig(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.n_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            learning_rate=self.config.learning_rate,
            max_length=self.config.max_length,
            max_prompt_length=self.config.max_prompt_length,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            seed=self.config.seed,
            bf16=self.config.bf16,
            fp16=self.config.fp16,
            report_to="none",     # disable wandb/tensorboard in base config
        )
        return DPOTrainer(
            model=model,
            args=dpo_config,
            train_dataset=dataset_dict.get("train"),
            eval_dataset=dataset_dict.get("eval"),
            tokenizer=tokenizer,
        )

    def _run_training(self, dpo_trainer) -> list[EpochMetrics]:
        metrics: list[EpochMetrics] = []

        for epoch in range(self.config.n_epochs):
            epoch_start = time.time()
            train_output = dpo_trainer.train()
            log_history = getattr(dpo_trainer.state, "log_history", [])

            # Extract loss from the last log entry for this epoch
            loss = self._extract_loss(log_history, epoch)

            # Extract reward metrics if present
            reward_chosen   = self._extract_metric(log_history, "rewards/chosen", 0.0)
            reward_rejected = self._extract_metric(log_history, "rewards/rejected", 0.0)

            metrics.append(EpochMetrics(
                epoch=epoch + 1,
                loss=loss,
                reward_chosen=reward_chosen,
                reward_rejected=reward_rejected,
                reward_delta=reward_chosen - reward_rejected,
                duration_seconds=time.time() - epoch_start,
            ))

        return metrics

    @staticmethod
    def _extract_loss(log_history: list[dict], epoch: int) -> float:
        epoch_logs = [
            entry for entry in log_history
            if entry.get("epoch", -1) == epoch + 1 and "loss" in entry
        ]
        if epoch_logs:
            return float(epoch_logs[-1]["loss"])
        # Fallback: last entry with a loss key
        loss_entries = [e for e in log_history if "loss" in e]
        return float(loss_entries[-1]["loss"]) if loss_entries else 0.0

    @staticmethod
    def _extract_metric(
        log_history: list[dict],
        key: str,
        default: float,
    ) -> float:
        entries = [e for e in log_history if key in e]
        return float(entries[-1][key]) if entries else default