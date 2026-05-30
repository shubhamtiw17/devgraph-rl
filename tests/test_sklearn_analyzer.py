import pytest
from src.training.sklearn_analyzer import (
    SklearnAnalyzer,
    ScoreDistribution,
    ClusterResult,
    PairSelectionResult,
    TrainingPair,
    EmbeddingAudit,
    FeatureImportanceResult,
    get_analyzer,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def analyzer():
    return SklearnAnalyzer(
        high_threshold=0.75,
        low_threshold=0.40,
        min_score_delta=0.15,
        random_state=42,
    )


def make_records(scores: list[float], task: str = "write a sort function") -> list[dict]:
    return [
        {
            "task": task,
            "output": f"def solution_{i}(): pass",
            "score": s,
            "dimensions": {
                "correctness": s * 1.1,
                "code_quality": s * 0.9,
                "task_completion": s,
                "graph_alignment": s * 0.8,
                "memory_relevance": s * 0.7,
            },
        }
        for i, s in enumerate(scores)
    ]


def make_multi_task_records() -> list[dict]:
    tasks = [
        ("parse CSV file", [0.9, 0.8, 0.3, 0.2]),
        ("sort a list", [0.85, 0.75, 0.25, 0.15]),
        ("read JSON", [0.95, 0.6, 0.4, 0.1]),
    ]
    records = []
    for task, scores in tasks:
        records.extend(make_records(scores, task=task))
    return records


# ---------------------------------------------------------------------------
# analyze_reward_distribution
# ---------------------------------------------------------------------------

class TestAnalyzeRewardDistribution:
    def test_empty_records(self, analyzer):
        result = analyzer.analyze_reward_distribution([])
        assert isinstance(result, ScoreDistribution)
        assert result.total == 0
        assert result.high == 0
        assert result.medium == 0
        assert result.low == 0

    def test_counts_correctly(self, analyzer):
        records = make_records([0.9, 0.8, 0.6, 0.3, 0.2])
        result = analyzer.analyze_reward_distribution(records)
        assert result.total == 5
        assert result.high == 2    # 0.9, 0.8 >= 0.75
        assert result.medium == 1  # 0.6 in [0.40, 0.75)
        assert result.low == 2     # 0.3, 0.2 < 0.40

    def test_mean_and_std(self, analyzer):
        scores = [0.2, 0.4, 0.6, 0.8]
        records = make_records(scores)
        result = analyzer.analyze_reward_distribution(records)
        assert abs(result.mean - 0.5) < 0.01
        assert result.std > 0

    def test_min_max(self, analyzer):
        records = make_records([0.1, 0.5, 0.95])
        result = analyzer.analyze_reward_distribution(records)
        assert abs(result.min - 0.1) < 0.001
        assert abs(result.max - 0.95) < 0.001

    def test_quality_ratio(self, analyzer):
        records = make_records([0.9, 0.8, 0.3, 0.2])
        result = analyzer.analyze_reward_distribution(records)
        assert result.quality_ratio == 0.5   # 2/4

    def test_quality_ratio_zero_total(self, analyzer):
        result = analyzer.analyze_reward_distribution([])
        assert result.quality_ratio == 0.0

    def test_outlier_detection_skipped_small_dataset(self, analyzer):
        # Fewer than 10 records — outlier detection is skipped gracefully
        records = make_records([0.5, 0.6, 0.7])
        result = analyzer.analyze_reward_distribution(records)
        assert result.outlier_indices == []

    def test_outlier_detection_large_dataset(self, analyzer):
        # 20 records with one obvious outlier at -9.0 (using raw dict)
        records = [{"task": "t", "output": "o", "score": 0.7 + i * 0.01}
                   for i in range(19)]
        records.append({"task": "t", "output": "o", "score": -9.0})
        result = analyzer.analyze_reward_distribution(records)
        # The extreme outlier should be detected
        assert len(result.outlier_indices) >= 1

    def test_summary_string(self, analyzer):
        records = make_records([0.8, 0.5, 0.2])
        result = analyzer.analyze_reward_distribution(records)
        s = result.summary()
        assert "Total" in s
        assert "High" in s
        assert "Mean" in s


# ---------------------------------------------------------------------------
# cluster_by_quality
# ---------------------------------------------------------------------------

class TestClusterByQuality:
    def test_returns_cluster_result(self, analyzer):
        records = make_records([0.9, 0.85, 0.5, 0.45, 0.2, 0.15])
        result = analyzer.cluster_by_quality(records)
        assert isinstance(result, ClusterResult)

    def test_three_clusters_by_default(self, analyzer):
        records = make_records([0.9, 0.85, 0.5, 0.45, 0.2, 0.15])
        result = analyzer.cluster_by_quality(records)
        assert result.n_clusters == 3

    def test_labels_are_valid(self, analyzer):
        records = make_records([0.9, 0.85, 0.5, 0.45, 0.2, 0.15])
        result = analyzer.cluster_by_quality(records)
        valid_labels = {"high", "medium", "low"}
        for c in result.clusters:
            assert c.label in valid_labels

    def test_high_and_low_exist(self, analyzer):
        records = make_records([0.9, 0.85, 0.5, 0.45, 0.2, 0.15])
        result = analyzer.cluster_by_quality(records)
        labels = {c.label for c in result.clusters}
        assert "high" in labels
        assert "low" in labels

    def test_high_quality_indices(self, analyzer):
        records = make_records([0.9, 0.85, 0.5, 0.45, 0.2, 0.15])
        result = analyzer.cluster_by_quality(records)
        # high_quality must contain at least some indices
        assert len(result.high_quality) > 0

    def test_low_quality_indices(self, analyzer):
        records = make_records([0.9, 0.85, 0.5, 0.45, 0.2, 0.15])
        result = analyzer.cluster_by_quality(records)
        assert len(result.low_quality) > 0

    def test_too_few_records_falls_back(self, analyzer):
        records = make_records([0.5, 0.6])   # fewer than n_clusters=3
        result = analyzer.cluster_by_quality(records)
        assert result.n_clusters == 1
        assert len(result.clusters) == 1

    def test_inertia_positive(self, analyzer):
        records = make_records([0.9, 0.85, 0.5, 0.45, 0.2, 0.15])
        result = analyzer.cluster_by_quality(records)
        assert result.inertia >= 0.0

    def test_all_indices_covered(self, analyzer):
        records = make_records([0.9, 0.85, 0.5, 0.45, 0.2, 0.15])
        result = analyzer.cluster_by_quality(records)
        all_indices = sorted(
            idx for c in result.clusters for idx in c.record_indices
        )
        assert all_indices == list(range(len(records)))


# ---------------------------------------------------------------------------
# select_training_pairs
# ---------------------------------------------------------------------------

class TestSelectTrainingPairs:
    def test_returns_pair_selection_result(self, analyzer):
        records = make_multi_task_records()
        result = analyzer.select_training_pairs(records)
        assert isinstance(result, PairSelectionResult)

    def test_pairs_are_training_pairs(self, analyzer):
        records = make_multi_task_records()
        result = analyzer.select_training_pairs(records)
        for p in result.pairs:
            assert isinstance(p, TrainingPair)

    def test_chosen_score_greater_than_rejected(self, analyzer):
        records = make_multi_task_records()
        result = analyzer.select_training_pairs(records)
        for p in result.pairs:
            assert p.chosen_score > p.rejected_score

    def test_score_delta_matches(self, analyzer):
        records = make_multi_task_records()
        result = analyzer.select_training_pairs(records)
        for p in result.pairs:
            assert abs(p.score_delta - (p.chosen_score - p.rejected_score)) < 1e-9

    def test_delta_above_minimum(self, analyzer):
        records = make_multi_task_records()
        result = analyzer.select_training_pairs(records)
        for p in result.pairs:
            assert p.score_delta >= analyzer.min_score_delta

    def test_pairs_sorted_by_delta_descending(self, analyzer):
        records = make_multi_task_records()
        result = analyzer.select_training_pairs(records)
        deltas = [p.score_delta for p in result.pairs]
        assert deltas == sorted(deltas, reverse=True)

    def test_max_pairs_respected(self, analyzer):
        records = make_multi_task_records()
        result = analyzer.select_training_pairs(records, max_pairs=2)
        assert result.pairs_selected <= 2

    def test_too_few_records(self, analyzer):
        result = analyzer.select_training_pairs([])
        assert result.pairs_selected == 0
        assert result.pairs == []

    def test_single_record_per_task_no_pairs(self, analyzer):
        # Each task only has one record — can't form pairs
        records = [
            {"task": "task_a", "output": "o1", "score": 0.9},
            {"task": "task_b", "output": "o2", "score": 0.2},
        ]
        result = analyzer.select_training_pairs(records)
        assert result.pairs_selected == 0

    def test_to_dict_keys(self, analyzer):
        records = make_multi_task_records()
        result = analyzer.select_training_pairs(records)
        if result.pairs:
            d = result.pairs[0].to_dict()
            assert "prompt" in d
            assert "chosen" in d
            assert "rejected" in d
            assert "score_delta" in d

    def test_summary_string(self, analyzer):
        records = make_multi_task_records()
        result = analyzer.select_training_pairs(records)
        s = result.summary()
        assert "Pairs" in s


# ---------------------------------------------------------------------------
# audit_embedding_quality
# ---------------------------------------------------------------------------

class TestAuditEmbeddingQuality:
    def _make_vectors(self, n: int, dim: int = 16) -> list[list[float]]:
        import numpy as np
        rng = np.random.default_rng(42)
        return rng.standard_normal((n, dim)).tolist()

    def test_returns_embedding_audit(self, analyzer):
        vectors = self._make_vectors(20)
        result = analyzer.audit_embedding_quality(vectors)
        assert isinstance(result, EmbeddingAudit)

    def test_n_vectors_correct(self, analyzer):
        vectors = self._make_vectors(20)
        result = analyzer.audit_embedding_quality(vectors)
        assert result.n_vectors == 20

    def test_n_dimensions_correct(self, analyzer):
        vectors = self._make_vectors(20, dim=16)
        result = analyzer.audit_embedding_quality(vectors)
        assert result.n_dimensions == 16

    def test_quality_label_valid(self, analyzer):
        vectors = self._make_vectors(20)
        result = analyzer.audit_embedding_quality(vectors)
        assert result.quality in {"good", "fair", "poor"}

    def test_explained_variance_sums_to_at_most_one(self, analyzer):
        vectors = self._make_vectors(20)
        result = analyzer.audit_embedding_quality(vectors)
        assert sum(result.explained_variance) <= 1.01   # float tolerance

    def test_too_few_vectors_returns_poor(self, analyzer):
        vectors = self._make_vectors(2)   # below threshold
        result = analyzer.audit_embedding_quality(vectors)
        assert result.quality == "poor"
        assert result.silhouette == 0.0

    def test_silhouette_in_valid_range(self, analyzer):
        vectors = self._make_vectors(30)
        result = analyzer.audit_embedding_quality(vectors)
        assert -1.0 <= result.silhouette <= 1.0

    def test_summary_string(self, analyzer):
        vectors = self._make_vectors(20)
        result = analyzer.audit_embedding_quality(vectors)
        s = result.summary()
        assert "Vectors" in s
        assert "Silhouette" in s


# ---------------------------------------------------------------------------
# feature_importance
# ---------------------------------------------------------------------------

class TestFeatureImportance:
    def test_returns_feature_importance_result(self, analyzer):
        records = make_multi_task_records()
        result = analyzer.feature_importance(records)
        assert isinstance(result, FeatureImportanceResult)

    def test_features_match_defaults(self, analyzer):
        records = make_multi_task_records()
        result = analyzer.feature_importance(records)
        expected = {
            "correctness", "code_quality", "task_completion",
            "graph_alignment", "memory_relevance",
        }
        assert set(result.features) == expected

    def test_importances_sum_to_one(self, analyzer):
        records = make_multi_task_records()
        result = analyzer.feature_importance(records)
        assert abs(sum(result.importances) - 1.0) < 0.01

    def test_importances_non_negative(self, analyzer):
        records = make_multi_task_records()
        result = analyzer.feature_importance(records)
        for imp in result.importances:
            assert imp >= 0.0

    def test_top_feature_is_string(self, analyzer):
        records = make_multi_task_records()
        result = analyzer.feature_importance(records)
        assert isinstance(result.top_feature, str)
        assert result.top_feature in result.features

    def test_bottom_feature_is_string(self, analyzer):
        records = make_multi_task_records()
        result = analyzer.feature_importance(records)
        assert isinstance(result.bottom_feature, str)
        assert result.bottom_feature in result.features

    def test_custom_dimension_keys(self, analyzer):
        records = [
            {"task": "t", "output": "o", "score": 0.8,
             "dimensions": {"speed": 0.9, "safety": 0.7}},
            {"task": "t", "output": "o2", "score": 0.3,
             "dimensions": {"speed": 0.4, "safety": 0.2}},
            {"task": "t2", "output": "o3", "score": 0.6,
             "dimensions": {"speed": 0.6, "safety": 0.5}},
            {"task": "t2", "output": "o4", "score": 0.9,
             "dimensions": {"speed": 0.95, "safety": 0.8}},
            {"task": "t3", "output": "o5", "score": 0.1,
             "dimensions": {"speed": 0.1, "safety": 0.15}},
        ]
        result = analyzer.feature_importance(records, dimension_keys=["speed", "safety"])
        assert set(result.features) == {"speed", "safety"}

    def test_too_few_records_returns_equal_importances(self, analyzer):
        records = make_records([0.5, 0.8])   # only 2 records
        result = analyzer.feature_importance(records)
        # Should return equal weights, not crash
        assert len(result.importances) == 5
        for imp in result.importances:
            assert imp > 0.0

    def test_summary_string(self, analyzer):
        records = make_multi_task_records()
        result = analyzer.feature_importance(records)
        s = result.summary()
        assert "correctness" in s


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

class TestGetAnalyzer:
    def test_returns_sklearn_analyzer(self):
        a = get_analyzer()
        assert isinstance(a, SklearnAnalyzer)

    def test_singleton_same_instance(self):
        a1 = get_analyzer()
        a2 = get_analyzer()
        assert a1 is a2