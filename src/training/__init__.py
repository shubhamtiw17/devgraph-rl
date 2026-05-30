"""
src/training/__init__.py

Training package — lazy exports.
Heavy deps (datasets, torch, keras) are only imported when actually used.
"""

from src.training.sklearn_analyzer import (
    SklearnAnalyzer,
    ScoreDistribution,
    ClusterResult,
    TrainingPair,
    PairSelectionResult,
    EmbeddingAudit,
    FeatureImportanceResult,
    get_analyzer,
)

from src.training.data_collector import (
    DataCollector,
    CollectionResult,
    AnalyzedCollection,
)

from src.training.keras_experiments import (
    KerasExperiments,
    ExperimentConfig,
    ExperimentRun,
    SweepResult,
    build_configs_from_space,
)

from src.training.torch_trainer import (
    TorchTrainer,
    TrainerConfig,
    TrainingResult,
    EpochMetrics,
)

from src.training.trainer import (
    Trainer,
    PipelineResult,
    PipelineStatus,
    PipelineStage,
)

def get_dataset_builder(*args, **kwargs):
    """Lazy import — only loads datasets library when called."""
    from src.training.dataset_builder import DatasetBuilder, BuildResult, SplitResult, get_builder
    return DatasetBuilder(*args, **kwargs)

__all__ = [
    "SklearnAnalyzer", "ScoreDistribution", "ClusterResult",
    "TrainingPair", "PairSelectionResult", "EmbeddingAudit",
    "FeatureImportanceResult", "get_analyzer",
    "DataCollector", "CollectionResult", "AnalyzedCollection",
    "KerasExperiments", "ExperimentConfig", "ExperimentRun",
    "SweepResult", "build_configs_from_space",
    "TorchTrainer", "TrainerConfig", "TrainingResult", "EpochMetrics",
    "Trainer", "PipelineResult", "PipelineStatus", "PipelineStage",
    "get_dataset_builder",
]
