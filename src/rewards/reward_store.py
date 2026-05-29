from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

from src.rewards.reward_model import RewardResult


# ── Stored record ─────────────────────────────────────────────────────────────

@dataclass
class RewardRecord:
    id:          str
    timestamp:   str
    task:        str
    agent_type:  str
    final_score: float
    dimensions:  Dict[str, float]
    feedback:    List[str]
    metadata:    Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id":          self.id,
            "timestamp":   self.timestamp,
            "task":        self.task,
            "agent_type":  self.agent_type,
            "final_score": self.final_score,
            "dimensions":  self.dimensions,
            "feedback":    self.feedback,
            "metadata":    self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RewardRecord":
        return cls(
            id=d["id"],
            timestamp=d["timestamp"],
            task=d["task"],
            agent_type=d["agent_type"],
            final_score=d["final_score"],
            dimensions=d["dimensions"],
            feedback=d["feedback"],
            metadata=d.get("metadata", {}),
        )


# ── Stats ─────────────────────────────────────────────────────────────────────

@dataclass
class RewardStats:
    total:        int
    average:      float
    best:         float
    worst:        float
    by_agent:     Dict[str, float]   # agent_type -> average score
    trend:        List[float]        # last 20 scores chronologically


# ── Store ─────────────────────────────────────────────────────────────────────

class RewardStore:
    """
    Persists reward scores to disk as newline-delimited JSON.
    One record per line — easy to append, easy to read.
    """

    def __init__(self, store_path: Optional[str] = None) -> None:
        path = store_path or os.getenv("VECTOR_STORE_PATH", "./data/vector_store")
        self._dir  = Path(path)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._file = self._dir / "reward_history.jsonl"

    # ── write ─────────────────────────────────────────────────────────

    def save(self, result: RewardResult) -> RewardRecord:
        """Persist a RewardResult and return the stored record."""
        record = RewardRecord(
            id=datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f"),
            timestamp=datetime.utcnow().isoformat(),
            task=result.task,
            agent_type=result.agent_type,
            final_score=result.final_score,
            dimensions=result.dimension_map,
            feedback=result.feedback,
            metadata=result.metadata,
        )
        with open(self._file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict()) + "\n")
        return record

    # ── read ──────────────────────────────────────────────────────────

    def load_all(self) -> List[RewardRecord]:
        """Load all records chronologically."""
        if not self._file.exists():
            return []
        records = []
        with open(self._file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(RewardRecord.from_dict(json.loads(line)))
                    except Exception:
                        continue
        return records

    def load_recent(self, n: int = 20) -> List[RewardRecord]:
        """Load the n most recent records."""
        return self.load_all()[-n:]

    def load_by_agent(self, agent_type: str) -> List[RewardRecord]:
        """Load all records for a specific agent type."""
        return [r for r in self.load_all() if r.agent_type == agent_type]

    def top_scores(self, n: int = 10) -> List[RewardRecord]:
        """Return top n records by final score."""
        records = self.load_all()
        return sorted(records, key=lambda r: r.final_score, reverse=True)[:n]

    # ── stats ─────────────────────────────────────────────────────────

    def stats(self) -> RewardStats:
        """Compute summary statistics across all records."""
        records = self.load_all()

        if not records:
            return RewardStats(
                total=0, average=0.0, best=0.0, worst=0.0,
                by_agent={}, trend=[],
            )

        scores = [r.final_score for r in records]

        by_agent: Dict[str, List[float]] = {}
        for r in records:
            by_agent.setdefault(r.agent_type, []).append(r.final_score)

        return RewardStats(
            total=len(records),
            average=round(sum(scores) / len(scores), 3),
            best=round(max(scores), 3),
            worst=round(min(scores), 3),
            by_agent={
                agent: round(sum(s) / len(s), 3)
                for agent, s in by_agent.items()
            },
            trend=[r.final_score for r in records[-20:]],
        )

    def clear(self) -> None:
        """Wipe all stored records."""
        if self._file.exists():
            self._file.unlink()


# ── Singleton ─────────────────────────────────────────────────────────────────

_store: Optional[RewardStore] = None


def get_reward_store() -> RewardStore:
    global _store
    if _store is None:
        _store = RewardStore()
    return _store
