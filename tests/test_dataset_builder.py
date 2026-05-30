from __future__ import annotations

import pytest
from datasets import Dataset, DatasetDict
from unittest.mock import MagicMock

from src.training.dataset_builder import (
    BuildResult,
    DatasetBuilder,
    SplitResult,
    get_builder,
    MIN_PAIRS,
)
from src.training.sklearn_analyzer import (
    PairSelectionResult,
    TrainingPair,
)
from src.training.data_collector import AnalyzedCollection, CollectionResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_pair(
    task: str = "sort a list",
    chosen: str = "def sort(x): return sorted(x)",
    rejected: str = "def sort(x): return x",
    chosen_score: float = 0.9,
    rejected_score: float = 0.3,
) -> TrainingPair:
    return TrainingPair(
        task=task,
        chosen_output=chosen,
        chosen_score=chosen_score,
        rejected_output=rejected,
        rejected_score=rejected_score,
        score_delta=chosen_score - rejected_score,
    )


def make_pair_selection(n: int, delta: float = 0.5) -> PairSelectionResult:
    """Build a PairSelectionResult with n pairs."""
    pairs = [
        make_pair(
            task=f"task_{i}",
            chosen=f"good_{i}",
            rejected=f"bad_{i}",
            chosen_score=0.8,
            rejected_score=0.8 - delta,
        )
        for i in range(n)
    ]
    return PairSelectionResult(
        pairs=pairs,
        total_records=n * 2,
        pairs_selected=n,
        min_delta=delta,
        mean_delta=delta,
    )


def make_analyzed(n_pairs: int = 6) -> AnalyzedCollection:
    """Build a minimal AnalyzedCollection wrapping a PairSelectionResult."""
    pair_result = make_pair_selection(n_pairs)

    collection = CollectionResult(
        records=[{}] * (n_pairs * 2),
        total_in_store=n_pairs * 2,
        filtered_out=0,
        duplicates_removed=0,
        min_score_used=0.0,
    )

    # Minimal mocks for the sklearn fields
    dist = MagicMock()
    dist.total = n_pairs * 2

    clusters = MagicMock()
    clusters.n_clusters = 3

    importance = MagicMock()
    importance.top_feature = "correctness"

    return AnalyzedCollection(
        collection=collection,
        distribution=dist,
        clusters=clusters,
        pairs=pair_result,
        feature_importance=importance,
        embedding_audit=None,
    )


@pytest.fixture
def builder():
    return DatasetBuilder()


# ---------------------------------------------------------------------------
# BuildResult properties
# ---------------------------------------------------------------------------

class TestBuildResult:
    def test_ready_true(self):
        ds = Dataset.from_dict({"prompt": ["t"] * MIN_PAIRS})
        r = BuildResult(
            dataset=ds, n_pairs=MIN_PAIRS,
            mean_delta=0.5, min_delta=0.3, max_delta=0.7,
            columns=["prompt"],
        )
        assert r.ready is True

    def test_ready_false(self):
        ds = Dataset.from_dict({"prompt": []})
        r = BuildResult(
            dataset=ds, n_pairs=MIN_PAIRS - 1,
            mean_delta=0.0, min_delta=0.0, max_delta=0.0,
            columns=["prompt"],
        )
        assert r.ready is False

    def test_summary_contains_key_fields(self):
        ds = Dataset.from_dict({"prompt": ["t"] * 6})
        r = BuildResult(
            dataset=ds, n_pairs=6,
            mean_delta=0.5, min_delta=0.3, max_delta=0.7,
            columns=["prompt"],
        )
        s = r.summary()
        assert "Pairs" in s
        assert "Ready" in s


# ---------------------------------------------------------------------------
# SplitResult properties
# ---------------------------------------------------------------------------

class TestSplitResult:
    def test_ready_true(self):
        empty = Dataset.from_dict({"prompt": []})
        r = SplitResult(
            dataset_dict=DatasetDict({"train": empty, "eval": empty}),
            n_train=MIN_PAIRS,
            n_eval=1,
            eval_ratio=0.1,
            mean_delta=0.5,
        )
        assert r.ready is True

    def test_ready_false(self):
        empty = Dataset.from_dict({"prompt": []})
        r = SplitResult(
            dataset_dict=DatasetDict({"train": empty, "eval": empty}),
            n_train=MIN_PAIRS - 1,
            n_eval=0,
            eval_ratio=0.1,
            mean_delta=0.0,
        )
        assert r.ready is False

    def test_summary_contains_key_fields(self):
        empty = Dataset.from_dict({"prompt": []})
        r = SplitResult(
            dataset_dict=DatasetDict({"train": empty, "eval": empty}),
            n_train=9, n_eval=1,
            eval_ratio=0.1, mean_delta=0.5,
        )
        s = r.summary()
        assert "Train" in s
        assert "Eval" in s


# ---------------------------------------------------------------------------
# DatasetBuilder.build() — from PairSelectionResult
# ---------------------------------------------------------------------------

class TestBuildFromPairSelection:
    def test_returns_build_result(self, builder):
        source = make_pair_selection(6)
        result = builder.build(source)
        assert isinstance(result, BuildResult)

    def test_dataset_is_hf_dataset(self, builder):
        source = make_pair_selection(6)
        result = builder.build(source)
        assert isinstance(result.dataset, Dataset)

    def test_n_pairs_correct(self, builder):
        source = make_pair_selection(6)
        result = builder.build(source)
        assert result.n_pairs == 6

    def test_dataset_length_matches_pairs(self, builder):
        source = make_pair_selection(6)
        result = builder.build(source)
        assert len(result.dataset) == 6

    def test_required_columns_present(self, builder):
        source = make_pair_selection(6)
        result = builder.build(source)
        for col in ["prompt", "chosen", "rejected", "score_delta",
                    "chosen_score", "rejected_score"]:
            assert col in result.dataset.column_names

    def test_prompt_values_correct(self, builder):
        source = make_pair_selection(3)
        result = builder.build(source)
        prompts = result.dataset["prompt"]
        assert prompts == ["task_0", "task_1", "task_2"]

    def test_chosen_values_correct(self, builder):
        source = make_pair_selection(3)
        result = builder.build(source)
        assert result.dataset["chosen"] == ["good_0", "good_1", "good_2"]

    def test_rejected_values_correct(self, builder):
        source = make_pair_selection(3)
        result = builder.build(source)
        assert result.dataset["rejected"] == ["bad_0", "bad_1", "bad_2"]

    def test_score_delta_values(self, builder):
        source = make_pair_selection(3, delta=0.5)
        result = builder.build(source)
        for delta in result.dataset["score_delta"]:
            assert abs(delta - 0.5) < 1e-6

    def test_mean_delta_correct(self, builder):
        source = make_pair_selection(4, delta=0.6)
        result = builder.build(source)
        assert abs(result.mean_delta - 0.6) < 1e-6

    def test_min_delta_correct(self, builder):
        source = make_pair_selection(4, delta=0.4)
        result = builder.build(source)
        assert abs(result.min_delta - 0.4) < 1e-6

    def test_max_delta_correct(self, builder):
        source = make_pair_selection(4, delta=0.7)
        result = builder.build(source)
        assert abs(result.max_delta - 0.7) < 1e-6

    def test_ready_true_enough_pairs(self, builder):
        source = make_pair_selection(MIN_PAIRS)
        result = builder.build(source)
        assert result.ready is True

    def test_ready_false_too_few_pairs(self, builder):
        source = make_pair_selection(MIN_PAIRS - 1)
        result = builder.build(source)
        assert result.ready is False

    def test_empty_pairs_returns_empty_dataset(self, builder):
        source = make_pair_selection(0)
        result = builder.build(source)
        assert result.n_pairs == 0
        assert len(result.dataset) == 0
        assert result.ready is False

    def test_columns_in_result_match_dataset(self, builder):
        source = make_pair_selection(6)
        result = builder.build(source)
        assert set(result.columns) == set(result.dataset.column_names)


# ---------------------------------------------------------------------------
# DatasetBuilder.build() — from AnalyzedCollection
# ---------------------------------------------------------------------------

class TestBuildFromAnalyzedCollection:
    def test_accepts_analyzed_collection(self, builder):
        source = make_analyzed(n_pairs=6)
        result = builder.build(source)
        assert isinstance(result, BuildResult)

    def test_pairs_extracted_correctly(self, builder):
        source = make_analyzed(n_pairs=6)
        result = builder.build(source)
        assert result.n_pairs == 6

    def test_columns_present(self, builder):
        source = make_analyzed(n_pairs=6)
        result = builder.build(source)
        assert "prompt" in result.dataset.column_names
        assert "chosen" in result.dataset.column_names
        assert "rejected" in result.dataset.column_names

    def test_unsupported_source_raises(self, builder):
        with pytest.raises(TypeError):
            builder.build("not a valid source")


# ---------------------------------------------------------------------------
# DatasetBuilder min_delta filtering
# ---------------------------------------------------------------------------

class TestMinDeltaFiltering:
    def test_min_delta_filters_weak_pairs(self):
        builder = DatasetBuilder(min_delta=0.6)
        # Mix of strong and weak pairs
        strong = make_pair(chosen_score=0.9, rejected_score=0.2)  # delta=0.7
        weak   = make_pair(chosen_score=0.7, rejected_score=0.5)  # delta=0.2
        source = PairSelectionResult(
            pairs=[strong, weak],
            total_records=4,
            pairs_selected=2,
            min_delta=0.2,
            mean_delta=0.45,
        )
        result = builder.build(source)
        assert result.n_pairs == 1

    def test_min_delta_zero_keeps_all(self):
        builder = DatasetBuilder(min_delta=0.0)
        source = make_pair_selection(4, delta=0.1)
        result = builder.build(source)
        assert result.n_pairs == 4

    def test_min_delta_above_all_returns_empty(self):
        builder = DatasetBuilder(min_delta=0.99)
        source = make_pair_selection(4, delta=0.5)
        result = builder.build(source)
        assert result.n_pairs == 0


# ---------------------------------------------------------------------------
# DatasetBuilder.build_split()
# ---------------------------------------------------------------------------

class TestBuildSplit:
    def test_returns_split_result(self, builder):
        source = make_pair_selection(10)
        result = builder.build_split(source)
        assert isinstance(result, SplitResult)

    def test_dataset_dict_has_train_and_eval(self, builder):
        source = make_pair_selection(10)
        result = builder.build_split(source)
        assert "train" in result.dataset_dict
        assert "eval" in result.dataset_dict

    def test_train_and_eval_are_datasets(self, builder):
        source = make_pair_selection(10)
        result = builder.build_split(source)
        assert isinstance(result.dataset_dict["train"], Dataset)
        assert isinstance(result.dataset_dict["eval"], Dataset)

    def test_total_rows_preserved(self, builder):
        source = make_pair_selection(10)
        result = builder.build_split(source, eval_ratio=0.2)
        assert result.n_train + result.n_eval == 10

    def test_eval_ratio_respected(self, builder):
        source = make_pair_selection(10)
        result = builder.build_split(source, eval_ratio=0.2)
        assert result.n_eval == 2
        assert result.n_train == 8

    def test_eval_ratio_stored(self, builder):
        source = make_pair_selection(10)
        result = builder.build_split(source, eval_ratio=0.1)
        assert result.eval_ratio == 0.1

    def test_seed_reproducible(self, builder):
        source_a = make_pair_selection(10)
        source_b = make_pair_selection(10)
        r1 = builder.build_split(source_a, seed=42)
        r2 = builder.build_split(source_b, seed=42)
        assert r1.dataset_dict["train"]["prompt"] == r2.dataset_dict["train"]["prompt"]

    def test_invalid_eval_ratio_raises(self, builder):
        source = make_pair_selection(10)
        with pytest.raises(ValueError):
            builder.build_split(source, eval_ratio=0.0)
        with pytest.raises(ValueError):
            builder.build_split(source, eval_ratio=1.0)

    def test_empty_source_returns_empty_split(self, builder):
        source = make_pair_selection(0)
        result = builder.build_split(source)
        assert result.n_train == 0
        assert result.n_eval == 0

    def test_single_pair_goes_to_train(self, builder):
        source = make_pair_selection(1)
        result = builder.build_split(source, eval_ratio=0.2)
        assert result.n_train == 1
        assert result.n_eval == 0

    def test_mean_delta_forwarded(self, builder):
        source = make_pair_selection(10, delta=0.6)
        result = builder.build_split(source)
        assert abs(result.mean_delta - 0.6) < 1e-6

    def test_ready_true(self, builder):
        source = make_pair_selection(10)
        result = builder.build_split(source)
        assert result.ready is True

    def test_ready_false_small_dataset(self, builder):
        source = make_pair_selection(MIN_PAIRS - 1)
        result = builder.build_split(source)
        assert result.ready is False


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

class TestGetBuilder:
    def test_returns_dataset_builder(self):
        b = get_builder()
        assert isinstance(b, DatasetBuilder)

    def test_singleton_same_instance(self):
        b1 = get_builder()
        b2 = get_builder()
        assert b1 is b2