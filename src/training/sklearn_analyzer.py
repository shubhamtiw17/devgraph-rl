from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses — plain data, no business logic
# ---------------------------------------------------------------------------

@dataclass
class ScoreDistribution:
    total: int
    high: int          # score >= high_threshold
    medium: int        # low_threshold <= score < high_threshold
    low: int           # score < low_threshold
    mean: float
    std: float
    min: float
    max: float
    outlier_indices: list[int] = field(default_factory=list)
    high_threshold: float = 0.75
    low_threshold: float = 0.40

    @property
    def quality_ratio(self) -> float:
        return self.high / self.total if self.total > 0 else 0.0

    def summary(self) -> str:
        return (
            f"Total: {self.total} | High: {self.high} | "
            f"Medium: {self.medium} | Low: {self.low} | "
            f"Mean: {self.mean:.3f} ± {self.std:.3f} | "
            f"Outliers: {len(self.outlier_indices)}"
        )


@dataclass
class QualityCluster:
    cluster_id: int
    label: str          # "high" | "medium" | "low"
    size: int
    mean_score: float
    record_indices: list[int] = field(default_factory=list)


@dataclass
class ClusterResult:
    n_clusters: int
    clusters: list[QualityCluster]
    inertia: float

    @property
    def high_quality(self) -> list[int]:
        return [
            idx
            for c in self.clusters
            if c.label == "high"
            for idx in c.record_indices
        ]

    @property
    def low_quality(self) -> list[int]:
        return [
            idx
            for c in self.clusters
            if c.label == "low"
            for idx in c.record_indices
        ]


@dataclass
class TrainingPair:
    task: str
    chosen_output: str
    chosen_score: float
    rejected_output: str
    rejected_score: float
    score_delta: float   # chosen - rejected; higher = clearer signal

    def to_dict(self) -> dict:
        return {
            "prompt": self.task,
            "chosen": self.chosen_output,
            "rejected": self.rejected_output,
            "score_delta": self.score_delta,
        }


@dataclass
class PairSelectionResult:
    pairs: list[TrainingPair]
    total_records: int
    pairs_selected: int
    min_delta: float
    mean_delta: float

    def summary(self) -> str:
        return (
            f"Pairs: {self.pairs_selected} / {self.total_records} records | "
            f"Min delta: {self.min_delta:.3f} | "
            f"Mean delta: {self.mean_delta:.3f}"
        )


@dataclass
class EmbeddingAudit:
    n_vectors: int
    n_dimensions: int
    silhouette: float        # -1 to 1; >0.5 is good
    explained_variance: list[float]   # top-3 PCA components
    quality: str             # "good" | "fair" | "poor"

    def summary(self) -> str:
        ev = [f"{v:.1%}" for v in self.explained_variance[:3]]
        return (
            f"Vectors: {self.n_vectors} | Dims: {self.n_dimensions} | "
            f"Silhouette: {self.silhouette:.3f} ({self.quality}) | "
            f"PCA top-3: {ev}"
        )


@dataclass
class FeatureImportanceResult:
    features: list[str]
    importances: list[float]   # normalised 0-1, same order as features
    top_feature: str
    bottom_feature: str

    def summary(self) -> str:
        ranked = sorted(
            zip(self.features, self.importances),
            key=lambda x: x[1],
            reverse=True,
        )
        parts = [f"{f}: {i:.3f}" for f, i in ranked]
        return " | ".join(parts)


# ---------------------------------------------------------------------------
# Main analyser
# ---------------------------------------------------------------------------

class SklearnAnalyzer:

    def __init__(
        self,
        high_threshold: float = 0.75,
        low_threshold: float = 0.40,
        min_score_delta: float = 0.15,
        random_state: int = 42,
    ) -> None:
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.min_score_delta = min_score_delta
        self.random_state = random_state

    # ------------------------------------------------------------------
    # 1. Reward score distribution analysis
    # ------------------------------------------------------------------

    def analyze_reward_distribution(
        self,
        records: list[dict],
        contamination: float = 0.05,
    ) -> ScoreDistribution:
        if not records:
            return ScoreDistribution(
                total=0, high=0, medium=0, low=0,
                mean=0.0, std=0.0, min=0.0, max=0.0,
                high_threshold=self.high_threshold,
                low_threshold=self.low_threshold,
            )

        scores = np.array([r.get("score", 0.0) for r in records], dtype=float)

        high = int(np.sum(scores >= self.high_threshold))
        low = int(np.sum(scores < self.low_threshold))
        medium = len(records) - high - low

        # Outlier detection — needs at least 10 samples
        outlier_indices: list[int] = []
        if len(scores) >= 10:
            try:
                detector = EllipticEnvelope(
                    contamination=contamination,
                    random_state=self.random_state,
                )
                preds = detector.fit_predict(scores.reshape(-1, 1))
                outlier_indices = [int(i) for i, p in enumerate(preds) if p == -1]
            except Exception as exc:
                logger.warning("Outlier detection skipped: %s", exc)

        return ScoreDistribution(
            total=len(records),
            high=high,
            medium=medium,
            low=low,
            mean=float(np.mean(scores)),
            std=float(np.std(scores)),
            min=float(np.min(scores)),
            max=float(np.max(scores)),
            outlier_indices=outlier_indices,
            high_threshold=self.high_threshold,
            low_threshold=self.low_threshold,
        )

    # ------------------------------------------------------------------
    # 2. Quality clustering
    # ------------------------------------------------------------------

    def cluster_by_quality(
        self,
        records: list[dict],
        n_clusters: int = 3,
    ) -> ClusterResult:
        if len(records) < n_clusters:
            # Not enough data — put everything in one cluster
            cluster = QualityCluster(
                cluster_id=0,
                label="medium",
                size=len(records),
                mean_score=float(np.mean([r.get("score", 0.0) for r in records])) if records else 0.0,
                record_indices=list(range(len(records))),
            )
            return ClusterResult(n_clusters=1, clusters=[cluster], inertia=0.0)

        scores = np.array([r.get("score", 0.0) for r in records], dtype=float).reshape(-1, 1)

        km = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init="auto")
        labels = km.fit_predict(scores)
        centers = km.cluster_centers_.flatten()

        # Sort cluster ids by their centroid score so we can label them
        sorted_ids = np.argsort(centers)   # ascending: [low_id, med_id, high_id]
        tier_names = {
            sorted_ids[0]: "low",
            sorted_ids[-1]: "high",
        }
        # Everything in between is medium (handles n_clusters > 3 gracefully)
        for cid in range(n_clusters):
            if cid not in tier_names:
                tier_names[cid] = "medium"

        clusters: list[QualityCluster] = []
        for cid in range(n_clusters):
            indices = [i for i, lbl in enumerate(labels) if lbl == cid]
            cluster_scores = scores[indices].flatten()
            clusters.append(QualityCluster(
                cluster_id=cid,
                label=tier_names[cid],
                size=len(indices),
                mean_score=float(np.mean(cluster_scores)) if len(cluster_scores) > 0 else 0.0,
                record_indices=indices,
            ))

        return ClusterResult(
            n_clusters=n_clusters,
            clusters=clusters,
            inertia=float(km.inertia_),
        )

    # ------------------------------------------------------------------
    # 3. Training pair selection
    # ------------------------------------------------------------------

    def select_training_pairs(
        self,
        records: list[dict],
        max_pairs: int = 500,
    ) -> PairSelectionResult:
        if len(records) < 2:
            return PairSelectionResult(
                pairs=[], total_records=len(records),
                pairs_selected=0, min_delta=0.0, mean_delta=0.0,
            )

        # Group by task
        task_groups: dict[str, list[dict]] = {}
        for rec in records:
            task = rec.get("task", "").strip()
            if not task:
                continue
            task_groups.setdefault(task, []).append(rec)

        pairs: list[TrainingPair] = []
        for task, group in task_groups.items():
            if len(group) < 2:
                continue
            sorted_group = sorted(group, key=lambda r: r.get("score", 0.0))
            best = sorted_group[-1]
            worst = sorted_group[0]
            delta = best.get("score", 0.0) - worst.get("score", 0.0)
            if delta < self.min_score_delta:
                continue
            pairs.append(TrainingPair(
                task=task,
                chosen_output=best.get("output", ""),
                chosen_score=float(best.get("score", 0.0)),
                rejected_output=worst.get("output", ""),
                rejected_score=float(worst.get("score", 0.0)),
                score_delta=float(delta),
            ))

        # Sort by delta — strongest signal first
        pairs.sort(key=lambda p: p.score_delta, reverse=True)
        pairs = pairs[:max_pairs]

        deltas = [p.score_delta for p in pairs]
        return PairSelectionResult(
            pairs=pairs,
            total_records=len(records),
            pairs_selected=len(pairs),
            min_delta=float(min(deltas)) if deltas else 0.0,
            mean_delta=float(np.mean(deltas)) if deltas else 0.0,
        )

    # ------------------------------------------------------------------
    # 4. Embedding quality audit
    # ------------------------------------------------------------------

    def audit_embedding_quality(
        self,
        vectors: list[list[float]],
        n_clusters: int = 3,
    ) -> EmbeddingAudit:
        if len(vectors) < max(n_clusters + 1, 4):
            return EmbeddingAudit(
                n_vectors=len(vectors),
                n_dimensions=len(vectors[0]) if vectors else 0,
                silhouette=0.0,
                explained_variance=[],
                quality="poor",
            )

        X = np.array(vectors, dtype=float)
        n_dims = X.shape[1]

        # PCA — top 3 components (or fewer if dim < 3)
        n_components = min(3, n_dims, len(vectors))
        pca = PCA(n_components=n_components, random_state=self.random_state)
        X_pca = pca.fit_transform(X)
        explained = pca.explained_variance_ratio_.tolist()

        # Silhouette on PCA-reduced space
        sil = 0.0
        try:
            km = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init="auto")
            cluster_labels = km.fit_predict(X_pca)
            if len(set(cluster_labels)) > 1:
                sil = float(silhouette_score(X_pca, cluster_labels))
        except Exception as exc:
            logger.warning("Silhouette scoring skipped: %s", exc)

        quality = "good" if sil >= 0.5 else ("fair" if sil >= 0.2 else "poor")

        return EmbeddingAudit(
            n_vectors=len(vectors),
            n_dimensions=n_dims,
            silhouette=sil,
            explained_variance=explained,
            quality=quality,
        )

    # ------------------------------------------------------------------
    # 5. Feature importance
    # ------------------------------------------------------------------

    def feature_importance(
        self,
        records: list[dict],
        dimension_keys: Optional[list[str]] = None,
    ) -> FeatureImportanceResult:
        if dimension_keys is None:
            dimension_keys = [
                "correctness",
                "code_quality",
                "task_completion",
                "graph_alignment",
                "memory_relevance",
            ]

        # Build feature matrix — use 0.0 for missing dimension scores
        X_rows, y_rows = [], []
        for rec in records:
            dims = rec.get("dimensions", {})
            row = [dims.get(k, 0.0) for k in dimension_keys]
            X_rows.append(row)
            y_rows.append(rec.get("score", 0.0))

        if len(X_rows) < 5:
            # Not enough data for meaningful importance
            equal = 1.0 / len(dimension_keys)
            return FeatureImportanceResult(
                features=dimension_keys,
                importances=[equal] * len(dimension_keys),
                top_feature=dimension_keys[0],
                bottom_feature=dimension_keys[-1],
            )

        X = StandardScaler().fit_transform(np.array(X_rows, dtype=float))
        y = np.array(y_rows, dtype=float)

        model = Ridge(alpha=1.0)
        model.fit(X, y)

        pi = permutation_importance(
            model, X, y,
            n_repeats=10,
            random_state=self.random_state,
        )
        raw = pi.importances_mean

        # Normalise to 0-1
        total = np.sum(np.abs(raw))
        if total > 0:
            normalised = (np.abs(raw) / total).tolist()
        else:
            normalised = [1.0 / len(dimension_keys)] * len(dimension_keys)

        ranked = sorted(
            zip(dimension_keys, normalised),
            key=lambda x: x[1],
            reverse=True,
        )

        return FeatureImportanceResult(
            features=dimension_keys,
            importances=normalised,
            top_feature=ranked[0][0],
            bottom_feature=ranked[-1][0],
        )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_analyzer: Optional[SklearnAnalyzer] = None


def get_analyzer(
    high_threshold: float = 0.75,
    low_threshold: float = 0.40,
    min_score_delta: float = 0.15,
) -> SklearnAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = SklearnAnalyzer(
            high_threshold=high_threshold,
            low_threshold=low_threshold,
            min_score_delta=min_score_delta,
        )
    return _analyzer