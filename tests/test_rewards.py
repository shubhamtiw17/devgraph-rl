from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.rewards.code_quality import score_code_quality, QualityResult
from src.rewards.reward_model import RewardModel, RewardResult, DimensionScore, DIMENSION_WEIGHTS
from src.rewards.reward_store import RewardStore, RewardRecord, RewardStats


# ════════════════════════════════════════════════
# CODE QUALITY TESTS
# ════════════════════════════════════════════════

class TestCodeQuality:
    GOOD_CODE = '''
def calculate_total(items: list, tax_rate: float) -> float:
    """Calculate total price including tax."""
    try:
        subtotal = sum(item.price for item in items)
        return subtotal * (1 + tax_rate)
    except AttributeError as e:
        raise ValueError(f"Invalid item: {e}")

def apply_discount(price: float, discount: float) -> float:
    """Apply discount to price."""
    try:
        return price * (1 - discount)
    except TypeError as e:
        raise ValueError(f"Invalid input: {e}")
'''

    BAD_CODE = "def f(x): return x"

    def test_good_code_scores_high(self):
        r = score_code_quality(self.GOOD_CODE)
        assert r.overall >= 0.7

    def test_bad_code_scores_lower(self):
        r = score_code_quality(self.BAD_CODE)
        assert r.overall < score_code_quality(self.GOOD_CODE).overall

    def test_empty_code_scores_zero(self):
        r = score_code_quality("")
        assert r.overall == 0.0
        assert r.error is not None

    def test_syntax_error_scores_zero(self):
        r = score_code_quality("def broken(:")
        assert r.overall == 0.0

    def test_returns_all_signals(self):
        r = score_code_quality(self.GOOD_CODE)
        assert "length"         in r.signals
        assert "complexity"     in r.signals
        assert "naming"         in r.signals
        assert "documentation"  in r.signals
        assert "error_handling" in r.signals

    def test_signals_between_zero_and_one(self):
        r = score_code_quality(self.GOOD_CODE)
        for k, v in r.signals.items():
            assert 0.0 <= v <= 1.0, f"{k} signal out of range: {v}"

    def test_overall_between_zero_and_one(self):
        r = score_code_quality(self.GOOD_CODE)
        assert 0.0 <= r.overall <= 1.0

    def test_feedback_is_list(self):
        r = score_code_quality(self.GOOD_CODE)
        assert isinstance(r.feedback, list)
        assert len(r.feedback) > 0

    def test_non_python_language(self):
        r = score_code_quality("console.log('hi')", language="javascript")
        assert r.overall == 0.7
        assert r.language == "javascript"

    def test_complex_code_penalized(self):
        complex_code = """
def process(x):
    if x > 0:
        if x > 10:
            if x > 100:
                for i in range(x):
                    while i > 0:
                        if i % 2:
                            i -= 1
                        else:
                            i -= 2
    return x
"""
        r = score_code_quality(complex_code)
        assert r.signals["complexity"] < 0.8

    def test_undocumented_code_penalized(self):
        code = "def add(a, b):\n    return a + b\n\ndef mul(a, b):\n    return a * b"
        r = score_code_quality(code)
        assert r.signals["documentation"] < 0.5

    def test_error_handling_rewarded(self):
        code = """
def safe_divide(a, b):
    \"\"\"Divide safely.\"\"\"
    try:
        return a / b
    except ZeroDivisionError:
        return None
"""
        r = score_code_quality(code)
        assert r.signals["error_handling"] >= 0.7

    def test_summary_string(self):
        r = score_code_quality(self.GOOD_CODE)
        assert isinstance(r.summary, str)
        assert "Quality:" in r.summary


# ════════════════════════════════════════════════
# REWARD MODEL TESTS
# ════════════════════════════════════════════════

class TestRewardModel:
    GOOD_CODE = '''
def add(a, b):
    """Add two numbers."""
    try:
        return a + b
    except TypeError as e:
        raise ValueError(f"Invalid: {e}")
'''

    def _make_model(self) -> RewardModel:
        return RewardModel()

    def test_returns_reward_result(self):
        rm = self._make_model()
        with patch("src.rewards.reward_model._score_task_completion", return_value=(0.8, "mocked")):
            r = rm.score(task="test task", output=self.GOOD_CODE, agent_type="coding")
        assert isinstance(r, RewardResult)

    def test_final_score_between_zero_and_one(self):
        rm = self._make_model()
        with patch("src.rewards.reward_model._score_task_completion", return_value=(0.8, "mocked")):
            r = rm.score(task="test task", output=self.GOOD_CODE, agent_type="coding")
        assert 0.0 <= r.final_score <= 1.0

    def test_has_five_dimensions(self):
        rm = self._make_model()
        with patch("src.rewards.reward_model._score_task_completion", return_value=(0.8, "mocked")):
            r = rm.score(task="test task", output=self.GOOD_CODE, agent_type="coding")
        assert len(r.dimensions) == 5

    def test_dimension_names(self):
        rm = self._make_model()
        with patch("src.rewards.reward_model._score_task_completion", return_value=(0.8, "mocked")):
            r = rm.score(task="test task", output=self.GOOD_CODE, agent_type="coding")
        names = {d.name for d in r.dimensions}
        assert names == {"correctness", "code_quality", "task_completion", "graph_alignment", "memory_relevance"}

    def test_dimension_scores_in_range(self):
        rm = self._make_model()
        with patch("src.rewards.reward_model._score_task_completion", return_value=(0.8, "mocked")):
            r = rm.score(task="test task", output=self.GOOD_CODE, agent_type="coding")
        for d in r.dimensions:
            assert 0.0 <= d.score <= 1.0, f"{d.name} out of range: {d.score}"

    def test_high_pass_rate_increases_correctness(self):
        rm = self._make_model()
        with patch("src.rewards.reward_model._score_task_completion", return_value=(0.8, "mocked")):
            r_high = rm.score(task="t", output=self.GOOD_CODE, agent_type="coding", test_pass_rate=1.0)
            r_low  = rm.score(task="t", output=self.GOOD_CODE, agent_type="coding", test_pass_rate=0.0)
        corr_high = next(d.score for d in r_high.dimensions if d.name == "correctness")
        corr_low  = next(d.score for d in r_low.dimensions  if d.name == "correctness")
        assert corr_high > corr_low

    def test_planner_skips_code_quality(self):
        rm = self._make_model()
        with patch("src.rewards.reward_model._score_task_completion", return_value=(0.8, "mocked")):
            r = rm.score(task="plan refactor", output="Step 1: extract", agent_type="planner")
        cq = next(d for d in r.dimensions if d.name == "code_quality")
        assert cq.score == 0.7

    def test_dimension_map(self):
        rm = self._make_model()
        with patch("src.rewards.reward_model._score_task_completion", return_value=(0.8, "mocked")):
            r = rm.score(task="t", output=self.GOOD_CODE, agent_type="coding")
        assert isinstance(r.dimension_map, dict)
        assert "correctness" in r.dimension_map

    def test_feedback_list(self):
        rm = self._make_model()
        with patch("src.rewards.reward_model._score_task_completion", return_value=(0.8, "mocked")):
            r = rm.score(task="t", output=self.GOOD_CODE, agent_type="coding")
        assert isinstance(r.feedback, list)
        assert len(r.feedback) == 5

    def test_summary_string(self):
        rm = self._make_model()
        with patch("src.rewards.reward_model._score_task_completion", return_value=(0.8, "mocked")):
            r = rm.score(task="t", output=self.GOOD_CODE, agent_type="coding")
        assert isinstance(r.summary, str)
        assert "Final:" in r.summary

    def test_weights_sum_to_one(self):
        total = sum(DIMENSION_WEIGHTS.values())
        assert abs(total - 1.0) < 0.001

    def test_final_score_is_weighted_sum(self):
        rm = self._make_model()
        with patch("src.rewards.reward_model._score_task_completion", return_value=(0.8, "mocked")):
            r = rm.score(task="t", output=self.GOOD_CODE, agent_type="coding")
        expected = sum(d.weighted for d in r.dimensions)
        assert abs(r.final_score - expected) < 0.001


# ════════════════════════════════════════════════
# REWARD STORE TESTS
# ════════════════════════════════════════════════

class TestRewardStore:
    def _make_result(self, task="test task", agent="coding", score=0.8) -> "RewardResult":
        rm = RewardModel()
        with patch("src.rewards.reward_model._score_task_completion", return_value=(score, "mocked")):
            return rm.score(
                task=task,
                output="def add(a,b): return a+b",
                agent_type=agent,
                test_pass_rate=score,
                execution_success=True,
            )

    def test_save_returns_record(self, tmp_path):
        rs = RewardStore(store_path=str(tmp_path))
        result = self._make_result()
        record = rs.save(result)
        assert isinstance(record, RewardRecord)
        assert record.final_score == result.final_score

    def test_load_all_empty(self, tmp_path):
        rs = RewardStore(store_path=str(tmp_path))
        assert rs.load_all() == []

    def test_load_all_after_save(self, tmp_path):
        rs = RewardStore(store_path=str(tmp_path))
        rs.save(self._make_result())
        rs.save(self._make_result())
        records = rs.load_all()
        assert len(records) == 2

    def test_load_recent(self, tmp_path):
        rs = RewardStore(store_path=str(tmp_path))
        for i in range(5):
            rs.save(self._make_result(task=f"task {i}"))
        recent = rs.load_recent(n=3)
        assert len(recent) == 3

    def test_load_by_agent(self, tmp_path):
        rs = RewardStore(store_path=str(tmp_path))
        rs.save(self._make_result(agent="coding"))
        rs.save(self._make_result(agent="planner"))
        rs.save(self._make_result(agent="coding"))
        coding = rs.load_by_agent("coding")
        assert len(coding) == 2
        assert all(r.agent_type == "coding" for r in coding)

    def test_top_scores(self, tmp_path):
        rs = RewardStore(store_path=str(tmp_path))
        rs.save(self._make_result(score=0.9))
        rs.save(self._make_result(score=0.5))
        rs.save(self._make_result(score=0.7))
        top = rs.top_scores(n=2)
        assert len(top) == 2
        assert top[0].final_score >= top[1].final_score

    def test_stats_empty(self, tmp_path):
        rs = RewardStore(store_path=str(tmp_path))
        s  = rs.stats()
        assert s.total == 0
        assert s.average == 0.0

    def test_stats_after_saves(self, tmp_path):
        rs = RewardStore(store_path=str(tmp_path))
        rs.save(self._make_result(agent="coding",  score=1.0))
        rs.save(self._make_result(agent="planner", score=0.5))
        s = rs.stats()
        assert s.total == 2
        assert s.best  >= s.worst
        assert "coding"  in s.by_agent
        assert "planner" in s.by_agent

    def test_trend_length(self, tmp_path):
        rs = RewardStore(store_path=str(tmp_path))
        for i in range(25):
            rs.save(self._make_result())
        s = rs.stats()
        assert len(s.trend) == 20

    def test_clear(self, tmp_path):
        rs = RewardStore(store_path=str(tmp_path))
        rs.save(self._make_result())
        rs.clear()
        assert rs.load_all() == []

    def test_record_has_timestamp(self, tmp_path):
        rs = RewardStore(store_path=str(tmp_path))
        record = rs.save(self._make_result())
        assert isinstance(record.timestamp, str)
        assert len(record.timestamp) > 0

    def test_record_has_dimensions(self, tmp_path):
        rs = RewardStore(store_path=str(tmp_path))
        record = rs.save(self._make_result())
        assert isinstance(record.dimensions, dict)
        assert "correctness" in record.dimensions

    def test_persistence_across_instances(self, tmp_path):
        rs1 = RewardStore(store_path=str(tmp_path))
        rs1.save(self._make_result())
        rs2 = RewardStore(store_path=str(tmp_path))
        assert len(rs2.load_all()) == 1

    def test_to_dict_roundtrip(self, tmp_path):
        rs     = RewardStore(store_path=str(tmp_path))
        record = rs.save(self._make_result())
        d      = record.to_dict()
        record2 = RewardRecord.from_dict(d)
        assert record2.final_score == record.final_score
        assert record2.task        == record.task
