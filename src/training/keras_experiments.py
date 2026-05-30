from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_EXPERIMENT_DIR = Path("./data/experiments")


# ---------------------------------------------------------------------------
# Config and result dataclasses — importable with zero ML deps
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    learning_rate: float = 2e-4
    batch_size: int = 16
    hidden_dim: int = 128
    dropout: float = 0.1
    n_epochs: int = 5
    embedding_dim: int = 384      # matches MiniLM output dim
    experiment_id: str = "exp_0"

    def to_dict(self) -> dict:
        return {
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "n_epochs": self.n_epochs,
            "embedding_dim": self.embedding_dim,
            "experiment_id": self.experiment_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ExperimentRun:
    config: ExperimentConfig
    train_loss: float
    eval_loss: float
    train_losses: list[float] = field(default_factory=list)  # per epoch
    eval_losses: list[float] = field(default_factory=list)   # per epoch
    duration_seconds: float = 0.0
    success: bool = True
    error: Optional[str] = None

    @property
    def overfit_gap(self) -> float:
        return self.eval_loss - self.train_loss

    def to_dict(self) -> dict:
        return {
            "config": self.config.to_dict(),
            "train_loss": self.train_loss,
            "eval_loss": self.eval_loss,
            "train_losses": self.train_losses,
            "eval_losses": self.eval_losses,
            "duration_seconds": self.duration_seconds,
            "success": self.success,
            "error": self.error,
            "overfit_gap": self.overfit_gap,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentRun":
        config = ExperimentConfig.from_dict(d["config"])
        return cls(
            config=config,
            train_loss=d["train_loss"],
            eval_loss=d["eval_loss"],
            train_losses=d.get("train_losses", []),
            eval_losses=d.get("eval_losses", []),
            duration_seconds=d.get("duration_seconds", 0.0),
            success=d.get("success", True),
            error=d.get("error"),
        )


@dataclass
class SweepResult:
    runs: list[ExperimentRun]
    best_config: ExperimentConfig
    best_eval_loss: float
    total_duration_seconds: float
    sweep_space: dict = field(default_factory=dict)

    @property
    def n_runs(self) -> int:
        return len(self.runs)

    @property
    def successful_runs(self) -> list[ExperimentRun]:
        return [r for r in self.runs if r.success]

    @property
    def ranked_runs(self) -> list[ExperimentRun]:
        return sorted(
            self.successful_runs,
            key=lambda r: r.eval_loss,
        )

    def summary(self) -> str:
        return (
            f"Runs: {self.n_runs} | "
            f"Successful: {len(self.successful_runs)} | "
            f"Best eval loss: {self.best_eval_loss:.4f} | "
            f"Best lr: {self.best_config.learning_rate} | "
            f"Best batch: {self.best_config.batch_size}"
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "best_config": self.best_config.to_dict(),
            "best_eval_loss": self.best_eval_loss,
            "total_duration_seconds": self.total_duration_seconds,
            "sweep_space": self.sweep_space,
            "runs": [r.to_dict() for r in self.runs],
        }
        path.write_text(json.dumps(payload, indent=2))
        logger.info("Sweep result saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> "SweepResult":
        data = json.loads(path.read_text())
        runs = [ExperimentRun.from_dict(r) for r in data["runs"]]
        best_config = ExperimentConfig.from_dict(data["best_config"])
        return cls(
            runs=runs,
            best_config=best_config,
            best_eval_loss=data["best_eval_loss"],
            total_duration_seconds=data["total_duration_seconds"],
            sweep_space=data.get("sweep_space", {}),
        )


# ---------------------------------------------------------------------------
# Default sweep space
# ---------------------------------------------------------------------------

DEFAULT_SWEEP_SPACE: dict[str, list] = {
    "learning_rate": [1e-3, 2e-4, 5e-5],
    "batch_size":    [8, 16, 32],
    "hidden_dim":    [64, 128, 256],
    "dropout":       [0.1, 0.2],
}


def build_configs_from_space(
    sweep_space: dict[str, list],
    n_epochs: int = 5,
    embedding_dim: int = 384,
) -> list[ExperimentConfig]:

    keys = list(sweep_space.keys())
    values = list(sweep_space.values())
    configs = []
    for i, combo in enumerate(product(*values)):
        params = dict(zip(keys, combo))
        configs.append(ExperimentConfig(
            learning_rate=params.get("learning_rate", 2e-4),
            batch_size=params.get("batch_size", 16),
            hidden_dim=params.get("hidden_dim", 128),
            dropout=params.get("dropout", 0.1),
            n_epochs=n_epochs,
            embedding_dim=embedding_dim,
            experiment_id=f"exp_{i}",
        ))
    return configs


# ---------------------------------------------------------------------------
# KerasExperiments
# ---------------------------------------------------------------------------

class KerasExperiments:

    def __init__(
        self,
        experiment_dir: Path = DEFAULT_EXPERIMENT_DIR,
        embedding_dim: int = 384,
    ) -> None:
        self.experiment_dir = experiment_dir
        self.embedding_dim = embedding_dim

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_baseline(
        self,
        train_data: list[dict],
        eval_data: list[dict],
        config: Optional[ExperimentConfig] = None,
    ) -> ExperimentRun:

        cfg = config or ExperimentConfig(
            embedding_dim=self.embedding_dim,
            experiment_id="baseline",
        )
        return self._run_single(train_data, eval_data, cfg)

    def run_sweep(
        self,
        train_data: list[dict],
        eval_data: list[dict],
        sweep_space: Optional[dict[str, list]] = None,
        n_epochs: int = 5,
        max_configs: Optional[int] = None,
    ) -> SweepResult:

        space = sweep_space or DEFAULT_SWEEP_SPACE
        configs = build_configs_from_space(
            space,
            n_epochs=n_epochs,
            embedding_dim=self.embedding_dim,
        )

        if max_configs is not None:
            configs = configs[:max_configs]

        logger.info("Starting sweep: %d configs", len(configs))
        sweep_start = time.time()
        runs: list[ExperimentRun] = []

        for cfg in configs:
            logger.info("Running %s (lr=%s, batch=%s, hidden=%s)",
                        cfg.experiment_id, cfg.learning_rate,
                        cfg.batch_size, cfg.hidden_dim)
            run = self._run_single(train_data, eval_data, cfg)
            runs.append(run)

        # Rank successful runs by eval_loss
        successful = [r for r in runs if r.success]
        if not successful:
            # All failed — return a failed sweep with dummy best
            dummy_config = configs[0] if configs else ExperimentConfig()
            return SweepResult(
                runs=runs,
                best_config=dummy_config,
                best_eval_loss=float("inf"),
                total_duration_seconds=time.time() - sweep_start,
                sweep_space=space,
            )

        best_run = min(successful, key=lambda r: r.eval_loss)

        result = SweepResult(
            runs=runs,
            best_config=best_run.config,
            best_eval_loss=best_run.eval_loss,
            total_duration_seconds=time.time() - sweep_start,
            sweep_space=space,
        )

        # Save to disk
        save_path = self.experiment_dir / "sweep_result.json"
        try:
            result.save(save_path)
        except Exception as exc:
            logger.warning("Could not save sweep result: %s", exc)

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_single(
        self,
        train_data: list[dict],
        eval_data: list[dict],
        config: ExperimentConfig,
    ) -> ExperimentRun:
        start = time.time()
        try:
            import numpy as np
            X_train, y_train = self._prepare_data(train_data)
            X_eval,  y_eval  = self._prepare_data(eval_data)

            model = self._build_model(config)
            history = self._fit_model(model, X_train, y_train,
                                      X_eval, y_eval, config)

            train_losses = history.get("loss", [])
            eval_losses  = history.get("val_loss", [])

            return ExperimentRun(
                config=config,
                train_loss=float(train_losses[-1]) if train_losses else 0.0,
                eval_loss=float(eval_losses[-1])   if eval_losses  else 0.0,
                train_losses=[float(x) for x in train_losses],
                eval_losses=[float(x) for x in eval_losses],
                duration_seconds=time.time() - start,
                success=True,
            )

        except ImportError as exc:
            msg = f"Keras not installed: {exc}. Run: pip install keras tensorflow"
            logger.error(msg)
            return ExperimentRun(
                config=config,
                train_loss=0.0,
                eval_loss=float("inf"),
                duration_seconds=time.time() - start,
                success=False,
                error=msg,
            )

        except Exception as exc:
            logger.error("Experiment %s failed: %s", config.experiment_id, exc)
            return ExperimentRun(
                config=config,
                train_loss=0.0,
                eval_loss=float("inf"),
                duration_seconds=time.time() - start,
                success=False,
                error=str(exc),
            )

    def _prepare_data(
        self,
        data: list[dict],
    ):

        import numpy as np

        X_rows, y_rows = [], []
        for item in data:
            chosen   = item.get("chosen_embedding", [0.0] * self.embedding_dim)
            rejected = item.get("rejected_embedding", [0.0] * self.embedding_dim)
            # Chosen example: [chosen, rejected] → label 1
            X_rows.append(chosen + rejected)
            y_rows.append(1.0)
            # Rejected example: [rejected, chosen] → label 0
            X_rows.append(rejected + chosen)
            y_rows.append(0.0)

        X = np.array(X_rows, dtype=np.float32)
        y = np.array(y_rows, dtype=np.float32)
        return X, y

    def _build_model(self, config: ExperimentConfig):
        """Build the Keras reward predictor model."""
        import keras
        from keras import layers

        input_dim = self.embedding_dim * 2   # chosen + rejected concatenated

        model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(config.hidden_dim, activation="relu"),
            layers.Dropout(config.dropout),
            layers.Dense(config.hidden_dim // 2, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def _fit_model(
        self,
        model,
        X_train,
        y_train,
        X_eval,
        y_eval,
        config: ExperimentConfig,
    ) -> dict:
        import keras

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=2,
                restore_best_weights=True,
            )
        ]

        history = model.fit(
            X_train, y_train,
            validation_data=(X_eval, y_eval),
            epochs=config.n_epochs,
            batch_size=config.batch_size,
            callbacks=callbacks,
            verbose=0,
        )
        return history.history