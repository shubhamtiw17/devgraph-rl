"""
Tests for the agent layer.
All LLM calls are mocked — no API keys needed.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from src.agents.base_agent import AgentContext, AgentResult
from src.agents.planner import PlannerAgent
from src.agents.coding import CodingAgent


# ── fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def context():
    return AgentContext(
        repo_path="/tmp/fake_repo",
        task="Add input validation to the login function",
    )


@pytest.fixture
def mock_router():
    return MagicMock()


# ── AgentContext ──────────────────────────────────────────────────────

def test_context_defaults():
    ctx = AgentContext(repo_path="/repo", task="fix bug")
    assert ctx.history     == []
    assert ctx.metadata    == {}
    assert ctx.language    == "python"
    assert ctx.target_file is None
    assert ctx.constraints == []


def test_context_with_history():
    ctx = AgentContext(
        repo_path="/repo",
        task="fix bug",
        history=[{"agent": "planner", "output": "step 1"}],
    )
    assert len(ctx.history) == 1


def test_context_with_language():
    ctx = AgentContext(repo_path="/repo", task="write script", language="shell")
    assert ctx.language == "shell"


def test_context_with_constraints():
    ctx = AgentContext(
        repo_path="/repo",
        task="fix bug",
        constraints=["no third party libs", "handle all exceptions"],
    )
    assert len(ctx.constraints) == 2


def test_context_with_target_file():
    ctx = AgentContext(
        repo_path="/repo",
        task="fix bug",
        target_file="src/auth/login.py",
    )
    assert ctx.target_file == "src/auth/login.py"


# ── AgentResult ───────────────────────────────────────────────────────

def test_result_success():
    result = AgentResult(agent_name="planner", output="ok", success=True)
    assert result.success is True
    assert result.error is None


def test_result_failure():
    result = AgentResult(
        agent_name="planner", output="", success=False, error="timeout"
    )
    assert result.success is False
    assert result.error == "timeout"


# ── PlannerAgent ──────────────────────────────────────────────────────

def test_planner_parses_valid_json(context, mock_router):
    mock_router.complete.return_value = (
        '{"subtasks": ["validate inputs", "write tests", "update docs"]}'
    )
    agent = PlannerAgent()
    agent.router = mock_router

    result = agent.run(context)

    assert result.success is True
    assert result.agent_name == "planner"
    assert len(result.artifacts["subtasks"]) == 3
    assert "validate inputs" in result.artifacts["subtasks"]


def test_planner_handles_markdown_fences(context, mock_router):
    mock_router.complete.return_value = (
        '```json\n{"subtasks": ["step 1", "step 2"]}\n```'
    )
    agent = PlannerAgent()
    agent.router = mock_router

    result = agent.run(context)

    assert result.success is True
    assert len(result.artifacts["subtasks"]) == 2


def test_planner_fallback_on_bad_json(context, mock_router):
    mock_router.complete.return_value = "This is not JSON at all"
    agent = PlannerAgent()
    agent.router = mock_router

    result = agent.run(context)

    assert result.success is True
    assert len(result.artifacts["subtasks"]) >= 1


def test_planner_filters_irrelevant_history(mock_router):
    """Planner should only use architect/reviewer history, not coder history."""
    mock_router.complete.return_value = '{"subtasks": ["do thing"]}'
    ctx = AgentContext(
        repo_path="/repo",
        task="refactor auth",
        history=[
            {"agent": "coding",    "output": "wrote some code"},   # irrelevant
            {"agent": "architect", "output": "use clean arch"},    # relevant
        ],
    )
    agent = PlannerAgent()
    agent.router = mock_router

    agent.run(ctx)

    prompt = mock_router.complete.call_args[1]["prompt"]
    assert "use clean arch" in prompt       # relevant — included
    assert "wrote some code" not in prompt  # irrelevant — excluded


def test_planner_includes_constraints(mock_router):
    mock_router.complete.return_value = '{"subtasks": ["do thing"]}'
    ctx = AgentContext(
        repo_path="/repo",
        task="fix login",
        constraints=["no third party libs", "raise ValueError on bad input"],
    )
    agent = PlannerAgent()
    agent.router = mock_router

    agent.run(ctx)

    prompt = mock_router.complete.call_args[1]["prompt"]
    assert "no third party libs" in prompt
    assert "raise ValueError on bad input" in prompt


def test_planner_includes_target_file(mock_router):
    mock_router.complete.return_value = '{"subtasks": ["do thing"]}'
    ctx = AgentContext(
        repo_path="/repo",
        task="fix login",
        target_file="src/auth/login.py",
    )
    agent = PlannerAgent()
    agent.router = mock_router

    agent.run(ctx)

    prompt = mock_router.complete.call_args[1]["prompt"]
    assert "src/auth/login.py" in prompt


# ── CodingAgent ───────────────────────────────────────────────────────

def test_coding_extracts_python_block(context, mock_router):
    mock_router.complete.return_value = (
        "Here is the code:\n"
        "```python\n"
        "def validate(x: str) -> bool:\n"
        "    return x is not None\n"
        "```"
    )
    agent = CodingAgent()
    agent.router = mock_router

    result = agent.run(context)

    assert result.success is True
    assert "def validate" in result.artifacts["code"]
    assert result.artifacts["language"] == "python"


def test_coding_extracts_shell_block(mock_router):
    ctx = AgentContext(
        repo_path="/repo",
        task="write deploy script",
        language="shell",
    )
    mock_router.complete.return_value = (
        "```shell\n"
        "#!/bin/bash\necho 'deploying'\n"
        "```"
    )
    agent = CodingAgent()
    agent.router = mock_router

    result = agent.run(ctx)

    assert result.success is True
    assert "deploying" in result.artifacts["code"]
    assert result.artifacts["language"] == "shell"


def test_coding_extracts_java_block(mock_router):
    ctx = AgentContext(
        repo_path="/repo",
        task="write User class",
        language="java",
    )
    mock_router.complete.return_value = (
        "```java\n"
        "public class User { private String name; }\n"
        "```"
    )
    agent = CodingAgent()
    agent.router = mock_router

    result = agent.run(ctx)

    assert "User" in result.artifacts["code"]
    assert result.artifacts["language"] == "java"


def test_coding_fallback_any_fence(mock_router):
    """Falls back to any fenced block if exact language not found."""
    ctx = AgentContext(repo_path="/repo", task="fix bug", language="python")
    mock_router.complete.return_value = (
        "```\ndef fix(): pass\n```"
    )
    agent = CodingAgent()
    agent.router = mock_router

    result = agent.run(ctx)
    assert "def fix" in result.artifacts["code"]


def test_coding_fallback_no_fences(context, mock_router):
    """Falls back to raw response when no fences present."""
    mock_router.complete.return_value = "def validate(x):\n    return x is not None"
    agent = CodingAgent()
    agent.router = mock_router

    result = agent.run(context)
    assert "def validate" in result.artifacts["code"]


def test_coding_filters_irrelevant_history(mock_router):
    """Coder should only use planner/architect history, not debug/testing."""
    mock_router.complete.return_value = "```python\npass\n```"
    ctx = AgentContext(
        repo_path="/repo",
        task="write validator",
        history=[
            {"agent": "planner",  "output": "validate inputs first"},  # relevant
            {"agent": "testing",  "output": "tests failed"},           # irrelevant
        ],
    )
    agent = CodingAgent()
    agent.router = mock_router

    agent.run(ctx)

    prompt = mock_router.complete.call_args[1]["prompt"]
    assert "validate inputs first" in prompt   # relevant — included
    assert "tests failed" not in prompt        # irrelevant — excluded


def test_coding_includes_constraints(mock_router):
    mock_router.complete.return_value = "```python\npass\n```"
    ctx = AgentContext(
        repo_path="/repo",
        task="write auth",
        constraints=["no global state", "raise ValueError on bad input"],
    )
    agent = CodingAgent()
    agent.router = mock_router

    agent.run(ctx)

    prompt = mock_router.complete.call_args[1]["prompt"]
    assert "no global state" in prompt
    assert "raise ValueError on bad input" in prompt


def test_coding_includes_target_file(mock_router):
    mock_router.complete.return_value = "```python\npass\n```"
    ctx = AgentContext(
        repo_path="/repo",
        task="fix login",
        target_file="src/auth/login.py",
    )
    agent = CodingAgent()
    agent.router = mock_router

    agent.run(ctx)

    prompt = mock_router.complete.call_args[1]["prompt"]
    assert "src/auth/login.py" in prompt


def test_agent_error_handling(context, mock_router):
    """Router throws — agent returns failure result, never crashes."""
    mock_router.complete.side_effect = Exception("network error")
    agent = CodingAgent()
    agent.router = mock_router

    result = agent.run(context)

    assert result.success is False
    assert "network error" in result.error
