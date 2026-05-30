from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

from src.training.sklearn_analyzer import (
    ClusterResult,
    EmbeddingAudit,
    FeatureImportanceResult,
    PairSelectionResult,
    ScoreDistribution,
    SklearnAnalyzer,
    get_analyzer,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CollectionResult:
    records: list[dict]
    total_in_store: int
    filtered_out: int
    duplicates_removed: int
    min_score_used: float

    @property
    def count(self) -> int:
        return len(self.records)

    @property
    def ready_for_training(self) -> bool:
        return self.count >= 10

    def summary(self) -> str:
        return (
            f"Collected: {self.count} | "
            f"Store total: {self.total_in_store} | "
            f"Filtered: {self.filtered_out} | "
            f"Deduped: {self.duplicates_removed} | "
            f"Min score: {self.min_score_used:.2f} | "
            f"Ready: {self.ready_for_training}"
        )


@dataclass
class AnalyzedCollection:
    collection: CollectionResult
    distribution: ScoreDistribution
    clusters: ClusterResult
    pairs: PairSelectionResult
    feature_importance: FeatureImportanceResult
    embedding_audit: Optional[EmbeddingAudit] = None

    @property
    def records(self) -> list[dict]:
        return self.collection.records

    @property
    def ready_for_training(self) -> bool:
        return self.pairs.pairs_selected >= 5

    def summary(self) -> str:
        lines = [
            "=== AnalyzedCollection ===",
            self.collection.summary(),
            self.distribution.summary(),
            self.pairs.summary(),
            f"Top predictor: {self.feature_importance.top_feature}",
            f"Ready for training: {self.ready_for_training}",
        ]
        if self.embedding_audit:
            lines.append(self.embedding_audit.summary())
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# DataCollector
# ---------------------------------------------------------------------------

class DataCollector:

    def __init__(
        self,
        reward_store,                          # RewardStore instance
        analyzer: Optional[SklearnAnalyzer] = None,
        min_score: float = 0.60,
        dedup: bool = True,
    ) -> None:
        self.reward_store = reward_store
        self.analyzer = analyzer or get_analyzer()
        self.min_score = min_score
        self.dedup = dedup

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect(
        self,
        min_score: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> CollectionResult:
        threshold = min_score if min_score is not None else self.min_score
        all_records = self._load_all()
        total = len(all_records)

        filtered = self._filter_by_score(all_records, threshold)
        filtered_out = total - len(filtered)

        deduped, dupes_removed = self._deduplicate(filtered) if self.dedup else (filtered, 0)

        if limit is not None:
            deduped = deduped[:limit]

        logger.info(
            "Collected %d records (total=%d, filtered=%d, deduped=%d)",
            len(deduped), total, filtered_out, dupes_removed,
        )

        return CollectionResult(
            records=deduped,
            total_in_store=total,
            filtered_out=filtered_out,
            duplicates_removed=dupes_removed,
            min_score_used=threshold,
        )

    def collect_recent(
        self,
        days: int = 7,
        min_score: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> CollectionResult:
        threshold = min_score if min_score is not None else self.min_score
        all_records = self._load_all()
        total = len(all_records)

        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=days)
        recent = self._filter_by_time(all_records, cutoff)
        time_filtered_out = total - len(recent)

        score_filtered = self._filter_by_score(recent, threshold)
        score_filtered_out = len(recent) - len(score_filtered)

        deduped, dupes_removed = (
            self._deduplicate(score_filtered) if self.dedup
            else (score_filtered, 0)
        )

        if limit is not None:
            deduped = deduped[:limit]

        logger.info(
            "Collected %d recent records (days=%d, time_filtered=%d, score_filtered=%d)",
            len(deduped), days, time_filtered_out, score_filtered_out,
        )

        return CollectionResult(
            records=deduped,
            total_in_store=total,
            filtered_out=time_filtered_out + score_filtered_out,
            duplicates_removed=dupes_removed,
            min_score_used=threshold,
        )

    def collect_and_analyze(
        self,
        min_score: Optional[float] = None,
        limit: Optional[int] = None,
        embedding_vectors: Optional[list[list[float]]] = None,
    ) -> AnalyzedCollection:
        collection = self.collect(min_score=min_score, limit=limit)
        records = collection.records

        distribution = self.analyzer.analyze_reward_distribution(records)
        clusters = self.analyzer.cluster_by_quality(records)
        pairs = self.analyzer.select_training_pairs(records)
        importance = self.analyzer.feature_importance(records)

        audit: Optional[EmbeddingAudit] = None
        if embedding_vectors:
            audit = self.analyzer.audit_embedding_quality(embedding_vectors)

        return AnalyzedCollection(
            collection=collection,
            distribution=distribution,
            clusters=clusters,
            pairs=pairs,
            feature_importance=importance,
            embedding_audit=audit,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_all(self) -> list[dict]:
        try:
            raw = self.reward_store.load_all() or []
            result = []
            for r in raw:
                d = r.to_dict() if hasattr(r, "to_dict") else dict(r)
                d["score"] = d.pop("final_score", 0.0)
                # Use the stored output from metadata if present,
                # otherwise fall back to the record id so dedup
                # never collapses two distinct records into one.
                stored_output = (d.get("metadata") or {}).get("output", "")
                d["output"] = stored_output if stored_output else d.get("id", "")
                result.append(d)
            return result
        except Exception as exc:
            logger.error("Failed to load from RewardStore: %s", exc)
            return []

    @staticmethod
    def _filter_by_score(records: list[dict], threshold: float) -> list[dict]:
        return [r for r in records if r.get("score", 0.0) >= threshold]

    @staticmethod
    def _filter_by_time(
        records: list[dict],
        cutoff: datetime,
    ) -> list[dict]:
        result = []
        for rec in records:
            ts_raw = rec.get("timestamp")
            if ts_raw is None:
                result.append(rec)
                continue
            try:
                if isinstance(ts_raw, str):
                    ts = datetime.fromisoformat(ts_raw)
                    # Make timezone-aware if naive
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                elif isinstance(ts_raw, (int, float)):
                    ts = datetime.fromtimestamp(ts_raw, tz=timezone.utc)
                else:
                    result.append(rec)
                    continue
                if ts >= cutoff:
                    result.append(rec)
            except (ValueError, OSError):
                # Unparseable timestamp — include it
                result.append(rec)
        return result

    @staticmethod
    def _deduplicate(records: list[dict]) -> tuple[list[dict], int]:
        """
        Remove duplicate records by id.
        Falls back to (task, output) if id is missing.
        """
        seen: set[str] = set()
        unique: list[dict] = []
        for rec in records:
            key = rec.get("id") or f"{rec.get('task','').strip()}||{rec.get('output','').strip()}"
            if key in seen:
                continue
            seen.add(key)
            unique.append(rec)
        return unique, len(records) - len(unique)