from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from datasets import Dataset, DatasetDict

from src.training.data_collector import AnalyzedCollection
from src.training.sklearn_analyzer import PairSelectionResult, TrainingPair

logger = logging.getLogger(__name__)

# Minimum pairs needed to produce a usable dataset
MIN_PAIRS = 5


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BuildResult:
    dataset: Dataset
    n_pairs: int
    mean_delta: float
    min_delta: float
    max_delta: float
    columns: list[str]

    @property
    def ready(self) -> bool:
        return self.n_pairs >= MIN_PAIRS

    def summary(self) -> str:
        return (
            f"Pairs: {self.n_pairs} | "
            f"Delta — min: {self.min_delta:.3f} "
            f"mean: {self.mean_delta:.3f} "
            f"max: {self.max_delta:.3f} | "
            f"Ready: {self.ready}"
        )


@dataclass
class SplitResult:
    dataset_dict: DatasetDict
    n_train: int
    n_eval: int
    eval_ratio: float
    mean_delta: float

    @property
    def ready(self) -> bool:
        return self.n_train >= MIN_PAIRS

    def summary(self) -> str:
        return (
            f"Train: {self.n_train} | Eval: {self.n_eval} | "
            f"Eval ratio: {self.eval_ratio:.0%} | "
            f"Mean delta: {self.mean_delta:.3f} | "
            f"Ready: {self.ready}"
        )


# ---------------------------------------------------------------------------
# DatasetBuilder
# ---------------------------------------------------------------------------

class DatasetBuilder:

    def __init__(self, min_delta: float = 0.0) -> None:
        self.min_delta = min_delta

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        source: AnalyzedCollection | PairSelectionResult,
    ) -> BuildResult:
        pairs = self._extract_pairs(source)
        pairs = self._apply_min_delta(pairs)

        if not pairs:
            logger.warning("No pairs available after filtering — returning empty dataset.")
            empty = Dataset.from_dict(self._empty_schema())
            return BuildResult(
                dataset=empty,
                n_pairs=0,
                mean_delta=0.0,
                min_delta=0.0,
                max_delta=0.0,
                columns=list(self._empty_schema().keys()),
            )

        rows = self._pairs_to_rows(pairs)
        dataset = Dataset.from_dict(rows)
        deltas = [p.score_delta for p in pairs]

        logger.info("Built dataset with %d pairs.", len(pairs))

        return BuildResult(
            dataset=dataset,
            n_pairs=len(pairs),
            mean_delta=float(sum(deltas) / len(deltas)),
            min_delta=float(min(deltas)),
            max_delta=float(max(deltas)),
            columns=dataset.column_names,
        )

    def build_split(
        self,
        source: AnalyzedCollection | PairSelectionResult,
        eval_ratio: float = 0.1,
        seed: int = 42,
    ) -> SplitResult:
        if not 0.0 < eval_ratio < 1.0:
            raise ValueError(f"eval_ratio must be between 0 and 1, got {eval_ratio}")

        build = self.build(source)

        if build.n_pairs == 0:
            empty = Dataset.from_dict(self._empty_schema())
            return SplitResult(
                dataset_dict=DatasetDict({"train": empty, "eval": empty}),
                n_train=0,
                n_eval=0,
                eval_ratio=eval_ratio,
                mean_delta=0.0,
            )

        # Need at least 2 rows to split
        if build.n_pairs < 2:
            return SplitResult(
                dataset_dict=DatasetDict({
                    "train": build.dataset,
                    "eval": Dataset.from_dict(self._empty_schema()),
                }),
                n_train=build.n_pairs,
                n_eval=0,
                eval_ratio=eval_ratio,
                mean_delta=build.mean_delta,
            )

        split = build.dataset.train_test_split(
            test_size=eval_ratio,
            seed=seed,
        )
        dataset_dict = DatasetDict({
            "train": split["train"],
            "eval": split["test"],
        })

        logger.info(
            "Split dataset: train=%d eval=%d",
            len(split["train"]),
            len(split["test"]),
        )

        return SplitResult(
            dataset_dict=dataset_dict,
            n_train=len(split["train"]),
            n_eval=len(split["test"]),
            eval_ratio=eval_ratio,
            mean_delta=build.mean_delta,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_pairs(
        source: AnalyzedCollection | PairSelectionResult,
    ) -> list[TrainingPair]:
        if isinstance(source, AnalyzedCollection):
            return source.pairs.pairs
        if isinstance(source, PairSelectionResult):
            return source.pairs
        raise TypeError(f"Unsupported source type: {type(source)}")

    def _apply_min_delta(self, pairs: list[TrainingPair]) -> list[TrainingPair]:
        if self.min_delta <= 0.0:
            return pairs
        return [p for p in pairs if p.score_delta >= self.min_delta]

    @staticmethod
    def _pairs_to_rows(pairs: list[TrainingPair]) -> dict[str, list]:
        return {
            "prompt":         [p.task for p in pairs],
            "chosen":         [p.chosen_output for p in pairs],
            "rejected":       [p.rejected_output for p in pairs],
            "score_delta":    [p.score_delta for p in pairs],
            "chosen_score":   [p.chosen_score for p in pairs],
            "rejected_score": [p.rejected_score for p in pairs],
        }

    @staticmethod
    def _empty_schema() -> dict[str, list]:
        return {
            "prompt": [],
            "chosen": [],
            "rejected": [],
            "score_delta": [],
            "chosen_score": [],
            "rejected_score": [],
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_builder: Optional[DatasetBuilder] = None


def get_builder(min_delta: float = 0.0) -> DatasetBuilder:
    global _builder
    if _builder is None:
        _builder = DatasetBuilder(min_delta=min_delta)
    return _builder