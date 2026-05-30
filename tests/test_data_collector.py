from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.training.data_collector import (
    AnalyzedCollection,
    CollectionResult,
    DataCollector,
)
from src.training.sklearn_analyzer import SklearnAnalyzer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_store(records: list[dict]) -> MagicMock:
    store = MagicMock()
    store.load_all.return_value = records
    return store


_record_counter = 0


def make_record(
    task: str = "sort a list",
    output: str | None = None,
    score: float = 0.8,
    days_ago: int = 0,
) -> dict:
    """Each call produces a unique (task, output) pair by default."""
    global _record_counter
    _record_counter += 1
    rec_id = f"rec_{_record_counter}"
    unique_output = output if output is not None else f"def solution_{_record_counter}(): pass"
    ts = datetime.now(tz=timezone.utc) - timedelta(days=days_ago)
    return {
        "id": rec_id,
        "task": task,
        "output": unique_output,
        "score": score,
        "timestamp": ts.isoformat(),
        "dimensions": {
            "correctness": score,
            "code_quality": score * 0.9,
            "task_completion": score,
            "graph_alignment": score * 0.8,
            "memory_relevance": score * 0.7,
        },
    }


def make_collector(records: list[dict], min_score: float = 0.0) -> DataCollector:
    return DataCollector(
        reward_store=make_store(records),
        analyzer=SklearnAnalyzer(random_state=42),
        min_score=min_score,
    )


# ---------------------------------------------------------------------------
# CollectionResult properties
# ---------------------------------------------------------------------------

class TestCollectionResult:
    def test_count(self):
        r = CollectionResult(
            records=[{}, {}], total_in_store=5,
            filtered_out=2, duplicates_removed=1, min_score_used=0.5,
        )
        assert r.count == 2


    def test_ready_for_training_false(self):
        r = CollectionResult(
            records=[{}] * 9, total_in_store=9,
            filtered_out=0, duplicates_removed=0, min_score_used=0.5,
        )
        assert r.ready_for_training is False

    def test_summary_contains_key_fields(self):
        r = CollectionResult(
            records=[{}, {}], total_in_store=5,
            filtered_out=2, duplicates_removed=1, min_score_used=0.5,
        )
        s = r.summary()
        assert "Collected" in s
        assert "Ready" in s


# ---------------------------------------------------------------------------
# DataCollector.collect()
# ---------------------------------------------------------------------------

class TestCollect:
    def test_returns_collection_result(self):
        collector = make_collector([make_record(score=0.8)])
        result = collector.collect()
        assert isinstance(result, CollectionResult)

    def test_all_records_pass_when_no_threshold(self):
        records = [make_record(score=0.9), make_record(score=0.5), make_record(score=0.2)]
        collector = make_collector(records, min_score=0.0)
        result = collector.collect()
        assert result.count == 3



    def test_total_in_store(self):
        records = [make_record(score=0.9), make_record(score=0.3)]
        collector = make_collector(records, min_score=0.6)
        result = collector.collect()
        assert result.total_in_store == 2


    def test_limit_respected(self):
        records = [make_record(score=0.9) for _ in range(10)]
        collector = make_collector(records, min_score=0.0)
        result = collector.collect(limit=3)
        assert result.count == 3

    def test_deduplication(self):
        # Same task + output twice
        rec = make_record(task="sort", output="def f(): pass", score=0.8)
        records = [rec, rec]
        collector = make_collector(records, min_score=0.0)
        result = collector.collect()
        assert result.count == 1
        assert result.duplicates_removed == 1

    def test_no_dedup_when_disabled(self):
        rec = make_record(task="sort", output="def f(): pass", score=0.8)
        store = make_store([rec, rec])
        collector = DataCollector(
            reward_store=store,
            analyzer=SklearnAnalyzer(random_state=42),
            min_score=0.0,
            dedup=False,
        )
        result = collector.collect()
        assert result.count == 2
        assert result.duplicates_removed == 0

    def test_empty_store(self):
        collector = make_collector([], min_score=0.0)
        result = collector.collect()
        assert result.count == 0
        assert result.total_in_store == 0

    def test_store_exception_returns_empty(self):
        store = MagicMock()
        store.load_all.side_effect = RuntimeError("disk error")
        collector = DataCollector(
            reward_store=store,
            analyzer=SklearnAnalyzer(random_state=42),
        )
        result = collector.collect()
        assert result.count == 0


class TestCollectRecent:
    def test_returns_collection_result(self):
        records = [make_record(score=0.8, days_ago=1)]
        collector = make_collector(records, min_score=0.0)
        result = collector.collect_recent(days=7)
        assert isinstance(result, CollectionResult)

    def test_recent_records_included(self):
        records = [make_record(score=0.8, days_ago=3)]
        collector = make_collector(records, min_score=0.0)
        result = collector.collect_recent(days=7)
        assert result.count == 1

    def test_old_records_excluded(self):
        records = [make_record(score=0.8, days_ago=30)]
        collector = make_collector(records, min_score=0.0)
        result = collector.collect_recent(days=7)
        assert result.count == 0

    def test_mix_recent_and_old(self):
        records = [
            make_record(score=0.8, days_ago=2),   # recent
            make_record(score=0.8, days_ago=10),  # old
        ]
        collector = make_collector(records, min_score=0.0)
        result = collector.collect_recent(days=7)
        assert result.count == 1


    def test_no_timestamp_always_included(self):
        rec = {"task": "t", "output": "o", "score": 0.8}  # no timestamp
        collector = make_collector([rec], min_score=0.0)
        result = collector.collect_recent(days=7)
        assert result.count == 1

    def test_unix_timestamp_recent(self):
        import time
        rec = {
            "task": "t", "output": "o", "score": 0.8,
            "timestamp": time.time() - 3600,   # 1 hour ago
        }
        collector = make_collector([rec], min_score=0.0)
        result = collector.collect_recent(days=1)
        assert result.count == 1

    def test_unix_timestamp_old(self):
        import time
        rec = {
            "task": "t", "output": "o", "score": 0.8,
            "timestamp": time.time() - 86400 * 30,  # 30 days ago
        }
        collector = make_collector([rec], min_score=0.0)
        result = collector.collect_recent(days=7)
        assert result.count == 0

    def test_limit_respected(self):
        records = [make_record(score=0.8, days_ago=1) for _ in range(5)]
        collector = make_collector(records, min_score=0.0)
        result = collector.collect_recent(days=7, limit=2)
        assert result.count == 2


# ---------------------------------------------------------------------------
# DataCollector.collect_and_analyze()
# ---------------------------------------------------------------------------

class TestCollectAndAnalyze:
    def _multi_task_records(self) -> list[dict]:
        tasks = [
            ("parse CSV",    [0.9, 0.3]),
            ("sort list",    [0.88, 0.2]),
            ("read JSON",    [0.91, 0.1]),
            ("write tests",  [0.85, 0.25]),
            ("refactor fn",  [0.82, 0.3]),
            ("fix bug",      [0.95, 0.15]),
        ]
        records = []
        for task, scores in tasks:
            for i, s in enumerate(scores):
                records.append(make_record(
                    task=task,
                    output=f"{task.replace(' ', '_')}_impl_{i}_{s}",
                    score=s,
                ))
        return records

    def test_returns_analyzed_collection(self):
        collector = make_collector(self._multi_task_records(), min_score=0.0)
        result = collector.collect_and_analyze()
        assert isinstance(result, AnalyzedCollection)

    def test_collection_attached(self):
        collector = make_collector(self._multi_task_records(), min_score=0.0)
        result = collector.collect_and_analyze()
        assert isinstance(result.collection, CollectionResult)

    def test_records_passthrough(self):
        records = self._multi_task_records()
        collector = make_collector(records, min_score=0.0)
        result = collector.collect_and_analyze()
        assert len(result.records) == len(records)

    def test_distribution_present(self):
        collector = make_collector(self._multi_task_records(), min_score=0.0)
        result = collector.collect_and_analyze()
        assert result.distribution.total > 0

    def test_clusters_present(self):
        collector = make_collector(self._multi_task_records(), min_score=0.0)
        result = collector.collect_and_analyze()
        assert result.clusters.n_clusters > 0


    def test_feature_importance_present(self):
        collector = make_collector(self._multi_task_records(), min_score=0.0)
        result = collector.collect_and_analyze()
        assert result.feature_importance.top_feature != ""

    def test_embedding_audit_none_by_default(self):
        collector = make_collector(self._multi_task_records(), min_score=0.0)
        result = collector.collect_and_analyze()
        assert result.embedding_audit is None

    def test_embedding_audit_when_vectors_provided(self):
        import numpy as np
        vectors = np.random.default_rng(42).standard_normal((20, 16)).tolist()
        collector = make_collector(self._multi_task_records(), min_score=0.0)
        result = collector.collect_and_analyze(embedding_vectors=vectors)
        assert result.embedding_audit is not None


    def test_ready_for_training_false_on_empty(self):
        collector = make_collector([], min_score=0.0)
        result = collector.collect_and_analyze()
        assert result.ready_for_training is False

    def test_min_score_forwarded(self):
        records = self._multi_task_records()
        collector = make_collector(records, min_score=0.0)
        # Force high threshold — most records excluded
        result = collector.collect_and_analyze(min_score=0.95)
        assert result.collection.min_score_used == 0.95

    def test_summary_string(self):
        collector = make_collector(self._multi_task_records(), min_score=0.0)
        result = collector.collect_and_analyze()
        s = result.summary()
        assert "AnalyzedCollection" in s
        assert "Ready" in s


# ---------------------------------------------------------------------------
# Private helpers (tested via public API edge cases)
# ---------------------------------------------------------------------------

class TestDeduplication:
    def test_different_outputs_same_task_not_deduped(self):
        records = [
            make_record(task="sort", output="def f(): return sorted(x)", score=0.8),
            make_record(task="sort", output="def f(): return x[::-1]", score=0.7),
        ]
        collector = make_collector(records, min_score=0.0)
        result = collector.collect()
        assert result.count == 2

    def test_same_id_deduped(self):
        # Two records with the same id are duplicates regardless of task/output
        rec = make_record(task="sort", output="def f(): pass", score=0.8)
        duplicate = dict(rec)  # same id
        collector = make_collector([rec, duplicate], min_score=0.0)
        result = collector.collect()
        assert result.count == 1