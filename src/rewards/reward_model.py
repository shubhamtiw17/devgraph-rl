from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
load_dotenv()

from src.rewards.code_quality import score_code_quality, QualityResult


# ── Dimension weights ─────────────────────────────────────────────────────────

DIMENSION_WEIGHTS = {
    "correctness":        0.35,
    "code_quality":       0.25,
    "task_completion":    0.20,
    "graph_alignment":    0.10,
    "memory_relevance":   0.10,
}


# ── Result ────────────────────────────────────────────────────────────────────

@dataclass
class DimensionScore:
    name:     str
    score:    float
    weight:   float
    feedback: str = ""

    @property
    def weighted(self) -> float:
        return self.score * self.weight


@dataclass
class RewardResult:
    final_score:  float
    dimensions:   List[DimensionScore]
    task:         str
    agent_type:   str
    output:       str
    feedback:     List[str]          = field(default_factory=list)
    metadata:     Dict[str, Any]     = field(default_factory=dict)
    error:        Optional[str]      = None

    @property
    def dimension_map(self) -> Dict[str, float]:
        return {d.name: d.score for d in self.dimensions}

    @property
    def summary(self) -> str:
        scores = " | ".join(f"{d.name[:4]}: {d.score:.2f}" for d in self.dimensions)
        return f"Final: {self.final_score:.3f} | {scores}"


# ── Dimension scorers ─────────────────────────────────────────────────────────

def _score_correctness(
    test_pass_rate: float,
    execution_success: bool,
) -> tuple[float, str]:
    """
    Score based on sandbox test results.
    If no tests provided, use execution success as signal.
    """
    if test_pass_rate > 0:
        score = test_pass_rate
        fb = f"Test pass rate: {test_pass_rate:.0%}"
    elif execution_success:
        score = 0.6   # ran without error but no tests
        fb = "Code executed successfully (no tests provided)"
    else:
        score = 0.1
        fb = "Code failed to execute"
    return round(score, 3), fb


def _score_task_completion(
    task: str,
    output: str,
    agent_type: str,
) -> tuple[float, str]:
    """
    Use LLM to judge if output addresses the task.
    Falls back to keyword overlap if LLM unavailable.
    """
    try:
        from src.llm.router import get_router
        router = get_router()

        prompt = f"""You are evaluating whether an agent's output addresses a given task.

AGENT TYPE: {agent_type}
TASK: {task}
OUTPUT (first 500 chars): {output[:500]}

Rate how well the output addresses the task on a scale of 0.0 to 1.0:
- 1.0 = perfectly addresses the task
- 0.7 = mostly addresses it with minor gaps
- 0.5 = partially addresses it
- 0.3 = barely addresses it
- 0.0 = completely off-task

Respond with ONLY a float number between 0.0 and 1.0. Nothing else."""

        response = router.complete(prompt, system="You are a precise code evaluation system. Respond only with a float.")
        score = float(response.strip())
        score = max(0.0, min(1.0, score))
        return round(score, 3), f"LLM task completion score: {score:.2f}"

    except Exception:
        # fallback — keyword overlap between task and output
        task_words  = set(task.lower().split())
        output_words = set(output.lower().split())
        overlap = len(task_words & output_words) / max(len(task_words), 1)
        score   = min(overlap * 2, 1.0)
        return round(score, 3), f"Keyword overlap score: {score:.2f} (LLM unavailable)"


def _score_graph_alignment(
    output: str,
    repo_path: Optional[str],
    language: str,
) -> tuple[float, str]:
    """
    Check if output references files/functions that exist in the repo graph.
    Higher score if agent's output mentions real code entities.
    """
    if not repo_path:
        return 0.7, "No repo context — neutral graph alignment score"

    try:
        from src.graphs.ast_graph import ASTGraph
        ast_graph = ASTGraph().build(repo_path, language=language)

        real_nodes = set()
        for node_id in ast_graph.graph.nodes():
            parts = str(node_id).replace("\\", "/").split("/")
            real_nodes.update(p.lower() for p in parts if p)

        output_lower = output.lower()
        matches = sum(1 for node in real_nodes if len(node) > 3 and node in output_lower)

        if matches == 0:
            return 0.5, "Output doesn't reference known repo entities"
        if matches >= 5:
            return 1.0, f"Output references {matches} known repo entities"
        score = 0.5 + (matches / 10)
        return round(min(score, 1.0), 3), f"Output references {matches} known entities"

    except Exception as e:
        return 0.6, f"Graph alignment check failed: {e}"


def _score_memory_relevance(
    task: str,
    output: str,
    repo_path: Optional[str],
) -> tuple[float, str]:
    """
    Check if output reuses patterns from memory.
    High score if similar past work exists and output resembles it.
    """
    try:
        from src.memory import MemoryManager
        mm = MemoryManager()

        results = mm.retrieve(
            query=task,
            embedder_name="minilm",
            top_k=3,
            repo_path=repo_path,
        )

        if not results:
            return 0.5, "No past memory found for this task"

        top_score = results[0].score
        if top_score > 0.85:
            return 0.9, f"High memory relevance — similar past work found (score: {top_score:.2f})"
        if top_score > 0.65:
            return 0.7, f"Moderate memory relevance (score: {top_score:.2f})"
        return 0.5, f"Low memory relevance (score: {top_score:.2f})"

    except Exception as e:
        return 0.5, f"Memory check failed: {e}"


# ── Main reward model ─────────────────────────────────────────────────────────

class RewardModel:
    """
    Scores agent outputs across 5 dimensions.
    Produces a final weighted score 0.0 - 1.0.
    """

    def score(
        self,
        task:               str,
        output:             str,
        agent_type:         str         = "coding",
        language:           str         = "python",
        repo_path:          Optional[str] = None,
        test_pass_rate:     float       = 0.0,
        execution_success:  bool        = False,
    ) -> RewardResult:
        """
        Score an agent output.

        Args:
            task:              The task given to the agent.
            output:            The agent's output (code or text).
            agent_type:        planner | coding | reviewer.
            language:          Programming language.
            repo_path:         Path to repo for graph alignment.
            test_pass_rate:    From sandbox test runner (0.0-1.0).
            execution_success: From sandbox executor.

        Returns:
            RewardResult with final score and per-dimension breakdown.
        """
        dimensions = []
        feedback   = []

        # 1 — Correctness
        c_score, c_fb = _score_correctness(test_pass_rate, execution_success)
        dimensions.append(DimensionScore("correctness", c_score, DIMENSION_WEIGHTS["correctness"], c_fb))
        feedback.append(c_fb)

        # 2 — Code quality
        if agent_type in ("coding", "reviewer"):
            quality = score_code_quality(output, language)
            q_score = quality.overall
            q_fb    = "; ".join(quality.feedback[:2])
        else:
            q_score = 0.7   # planners produce text not code
            q_fb    = "Code quality not applicable for planner output"
        dimensions.append(DimensionScore("code_quality", q_score, DIMENSION_WEIGHTS["code_quality"], q_fb))
        feedback.append(q_fb)

        # 3 — Task completion
        t_score, t_fb = _score_task_completion(task, output, agent_type)
        dimensions.append(DimensionScore("task_completion", t_score, DIMENSION_WEIGHTS["task_completion"], t_fb))
        feedback.append(t_fb)

        # 4 — Graph alignment
        g_score, g_fb = _score_graph_alignment(output, repo_path, language)
        dimensions.append(DimensionScore("graph_alignment", g_score, DIMENSION_WEIGHTS["graph_alignment"], g_fb))
        feedback.append(g_fb)

        # 5 — Memory relevance
        m_score, m_fb = _score_memory_relevance(task, output, repo_path)
        dimensions.append(DimensionScore("memory_relevance", m_score, DIMENSION_WEIGHTS["memory_relevance"], m_fb))
        feedback.append(m_fb)

        final_score = sum(d.weighted for d in dimensions)

        return RewardResult(
            final_score=round(final_score, 3),
            dimensions=dimensions,
            task=task,
            agent_type=agent_type,
            output=output,
            feedback=feedback,
            metadata={
                "language":          language,
                "repo_path":         repo_path,
                "test_pass_rate":    test_pass_rate,
                "execution_success": execution_success,
            },
        )


# ── Singleton ─────────────────────────────────────────────────────────────────

_model: Optional[RewardModel] = None


def get_reward_model() -> RewardModel:
    global _model
    if _model is None:
        _model = RewardModel()
    return _model
