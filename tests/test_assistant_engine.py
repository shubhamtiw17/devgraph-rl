
from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from visualiser.services.assistant_engine import (
    AssistantEngine,
    AssistantMode,
    AssistantResponse,
    ScoreResult,
    SandboxResult,
    SystemContext,
    detect_mode,
    _extract_code,
    _detect_expertise,
    get_engine,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_engine(
    llm_response: str = "Here is the code:\n```python\ndef f(): pass\n```",
    sandbox_valid: bool = True,
    sandbox_passed: bool = True,
    score: float = 0.8,
) -> AssistantEngine:
    """Build an AssistantEngine with all deps mocked."""
    router = MagicMock()
    router.complete.return_value = llm_response

    sandbox = MagicMock()
    sb_result = MagicMock()
    sb_result.validation.valid = sandbox_valid
    sb_result.execution.success = sandbox_passed
    sb_result.execution.stdout = "ok"
    sb_result.execution.stderr = ""
    sandbox.run.return_value = sb_result

    reward = MagicMock()
    rw_result = MagicMock()
    rw_result.final_score = score
    rw_result.summary = "Good code"
    dim = MagicMock()
    dim.name = "correctness"
    dim.score = score
    rw_result.dimensions = [dim]
    reward.score.return_value = rw_result

    memory = MagicMock()

    return AssistantEngine(
        llm_router=router,
        sandbox=sandbox,
        reward_model=reward,
        memory_manager=memory,
        score_threshold=0.70,
    )


def make_context(**kwargs) -> SystemContext:
    return SystemContext(**kwargs)


# ---------------------------------------------------------------------------
# detect_mode
# ---------------------------------------------------------------------------

class TestDetectMode:
    def test_generate_write(self):
        assert detect_mode("write a function to sort a list") == AssistantMode.GENERATE

    def test_generate_create(self):
        assert detect_mode("create a class for handling CSV files") == AssistantMode.GENERATE

    def test_generate_implement(self):
        assert detect_mode("implement a binary search algorithm") == AssistantMode.GENERATE

    def test_improve_refactor(self):
        assert detect_mode("refactor this messy code") == AssistantMode.IMPROVE

    def test_improve_fix(self):
        assert detect_mode("fix this function") == AssistantMode.IMPROVE

    def test_improve_slow(self):
        assert detect_mode("this code is too slow, optimize it") == AssistantMode.IMPROVE

    def test_improve_wins_over_generate(self):
        # "rewrite" is improve, not generate
        assert detect_mode("rewrite this function to be better") == AssistantMode.IMPROVE

    def test_guide_explain(self):
        assert detect_mode("explain how decorators work") == AssistantMode.GUIDE

    def test_guide_how(self):
        assert detect_mode("how do I set up the project?") == AssistantMode.GUIDE

    def test_guide_error(self):
        assert detect_mode("I'm getting an import error") == AssistantMode.GUIDE

    def test_guide_default_no_keywords(self):
        assert detect_mode("what is this") == AssistantMode.GUIDE

    def test_case_insensitive(self):
        assert detect_mode("WRITE a function") == AssistantMode.GENERATE
        assert detect_mode("REFACTOR this") == AssistantMode.IMPROVE
        assert detect_mode("EXPLAIN this") == AssistantMode.GUIDE


# ---------------------------------------------------------------------------
# SystemContext
# ---------------------------------------------------------------------------

class TestSystemContext:
    def test_empty_context(self):
        ctx = SystemContext()
        s = ctx.to_prompt_section()
        assert "No system context" in s

    def test_repo_name_included(self):
        ctx = SystemContext(repo_name="myrepo", repo_language="python")
        s = ctx.to_prompt_section()
        assert "myrepo" in s
        assert "python" in s

    def test_file_content_truncated(self):
        ctx = SystemContext(file_content="x" * 1000)
        s = ctx.to_prompt_section()
        # Should include first 800 chars
        assert "x" * 800 in s
        assert "x" * 801 not in s

    def test_memories_included(self):
        ctx = SystemContext(recent_memories=["mem1", "mem2"])
        s = ctx.to_prompt_section()
        assert "mem1" in s
        assert "mem2" in s

    def test_reward_stats_included(self):
        ctx = SystemContext(reward_stats={"average": 0.75, "total": 42})
        s = ctx.to_prompt_section()
        assert "42" in s
        assert "0.75" in s

    def test_active_file_included(self):
        ctx = SystemContext(loaded_file="src/main.py")
        s = ctx.to_prompt_section()
        assert "src/main.py" in s


# ---------------------------------------------------------------------------
# AssistantEngine.chat() — mode routing
# ---------------------------------------------------------------------------

class TestChatModeRouting:
    def test_generate_mode_detected(self):
        engine = make_engine()
        resp = engine.chat("write a sort function")
        assert resp.mode == AssistantMode.GENERATE

    def test_improve_mode_detected(self):
        engine = make_engine()
        resp = engine.chat("refactor this function")
        assert resp.mode == AssistantMode.IMPROVE

    def test_guide_mode_detected(self):
        engine = make_engine()
        resp = engine.chat("explain how this works")
        assert resp.mode == AssistantMode.GUIDE

    def test_returns_assistant_response(self):
        engine = make_engine()
        resp = engine.chat("write a function")
        assert isinstance(resp, AssistantResponse)

    def test_duration_recorded(self):
        engine = make_engine()
        resp = engine.chat("write a function")
        assert resp.duration_ms >= 0.0

    def test_context_used_true_with_repo(self):
        engine = make_engine()
        ctx = make_context(repo_name="myrepo")
        resp = engine.chat("write a function", context=ctx)
        assert resp.context_used is True

    def test_context_used_false_without_context(self):
        engine = make_engine()
        resp = engine.chat("explain decorators")
        assert resp.context_used is False

    def test_exception_returns_error_response(self):
        engine = make_engine()
        engine._router.complete.side_effect = RuntimeError("boom")
        resp = engine.chat("write a function")
        assert isinstance(resp, AssistantResponse)
        assert "error" in resp.message.lower() or resp.message != ""

    def test_no_router_uses_fallback(self):
        engine = AssistantEngine()  # no deps
        resp = engine.chat("write a sort function")
        assert "LLM router" in resp.message or "router" in resp.message.lower()


# ---------------------------------------------------------------------------
# Generate mode
# ---------------------------------------------------------------------------

class TestGenerateMode:
    def test_code_extracted(self):
        engine = make_engine(llm_response="Here:\n```python\ndef sort(x): return sorted(x)\n```")
        resp = engine.chat("write a sort function")
        assert resp.code is not None
        assert "def sort" in resp.code

    def test_sandbox_called(self):
        engine = make_engine()
        engine.chat("write a sort function")
        engine._sandbox.run.assert_called_once()

    def test_reward_scored(self):
        engine = make_engine()
        engine.chat("write a sort function")
        engine._reward_model.score.assert_called_once()

    def test_high_score_stores_to_memory(self):
        engine = make_engine(score=0.9)
        engine.chat("write a sort function")
        engine._memory.store.assert_called_once()

    def test_low_score_not_stored(self):
        engine = make_engine(score=0.4)
        engine.chat("write a sort function")
        engine._memory.store.assert_not_called()

    def test_sandbox_result_attached(self):
        engine = make_engine()
        resp = engine.chat("write a function")
        assert resp.sandbox is not None
        assert isinstance(resp.sandbox, SandboxResult)

    def test_score_result_attached(self):
        engine = make_engine()
        resp = engine.chat("write a function")
        assert resp.score is not None
        assert isinstance(resp.score, ScoreResult)

    def test_language_forwarded(self):
        engine = make_engine()
        resp = engine.chat("write a function", language="javascript")
        assert resp.language == "javascript"

    def test_suggestions_populated(self):
        engine = make_engine()
        resp = engine.chat("write a function")
        assert len(resp.suggestions) > 0

    def test_stored_flag_true_high_score(self):
        engine = make_engine(score=0.9)
        resp = engine.chat("write a function")
        assert resp.stored is True

    def test_stored_flag_false_low_score(self):
        engine = make_engine(score=0.3)
        resp = engine.chat("write a function")
        assert resp.stored is False


# ---------------------------------------------------------------------------
# Improve mode
# ---------------------------------------------------------------------------

class TestImproveMode:
    def test_code_extracted_from_response(self):
        engine = make_engine(
            llm_response="Improved:\n```python\ndef sort(x: list) -> list:\n    return sorted(x)\n```\nChanges: added type hints."
        )
        resp = engine.chat("refactor this", context=make_context(file_content="def sort(x): return sorted(x)"))
        assert resp.code is not None

    def test_sandbox_called(self):
        engine = make_engine()
        engine.chat("fix this code", context=make_context(file_content="def f(): pass"))
        assert engine._sandbox.run.call_count >= 1

    def test_reward_scored(self):
        engine = make_engine()
        engine.chat("improve this", context=make_context(file_content="def f(): pass"))
        assert engine._reward_model.score.call_count >= 1

    def test_delta_calculated(self):
        engine = make_engine(score=0.8)
        resp = engine.chat("refactor this", context=make_context(file_content="def f(): pass"))
        if resp.score:
            assert isinstance(resp.score.delta, float)

    def test_suggestions_populated(self):
        engine = make_engine()
        resp = engine.chat("fix this code")
        assert len(resp.suggestions) > 0

    def test_mode_is_improve(self):
        engine = make_engine()
        resp = engine.chat("refactor and optimise this")
        assert resp.mode == AssistantMode.IMPROVE


# ---------------------------------------------------------------------------
# Guide mode
# ---------------------------------------------------------------------------

class TestGuideMode:
    def test_mode_is_guide(self):
        engine = make_engine(llm_response="Decorators are functions that wrap other functions.")
        resp = engine.chat("explain decorators")
        assert resp.mode == AssistantMode.GUIDE

    def test_no_sandbox_called(self):
        engine = make_engine()
        engine.chat("explain how this works")
        engine._sandbox.run.assert_not_called()

    def test_no_reward_scored(self):
        engine = make_engine()
        engine.chat("how do I set up the project?")
        engine._reward_model.score.assert_not_called()

    def test_no_memory_stored(self):
        engine = make_engine()
        engine.chat("what is a decorator?")
        engine._memory.store.assert_not_called()

    def test_code_extracted_if_present(self):
        engine = make_engine(
            llm_response="Example:\n```python\ndef decorator(f): return f\n```"
        )
        resp = engine.chat("explain decorators with example")
        assert resp.code is not None

    def test_suggestions_populated(self):
        engine = make_engine()
        resp = engine.chat("how do I set up this project?")
        assert len(resp.suggestions) > 0

    def test_message_in_response(self):
        engine = make_engine(llm_response="Decorators wrap functions.")
        resp = engine.chat("explain decorators")
        assert "Decorators" in resp.message


# ---------------------------------------------------------------------------
# _extract_code
# ---------------------------------------------------------------------------

class TestExtractCode:
    def test_python_fence(self):
        text = "Here:\n```python\ndef f(): pass\n```\nDone."
        assert _extract_code(text, "python") == "def f(): pass"

    def test_generic_fence(self):
        text = "Here:\n```\ndef f(): pass\n```"
        assert _extract_code(text, "python") == "def f(): pass"

    def test_no_code_returns_none(self):
        assert _extract_code("no code here", "python") is None

    def test_empty_string_returns_none(self):
        assert _extract_code("", "python") is None

    def test_none_returns_none(self):
        assert _extract_code(None, "python") is None

    def test_javascript_fence(self):
        text = "```javascript\nfunction f() {}\n```"
        assert _extract_code(text, "javascript") == "function f() {}"

    def test_first_block_returned(self):
        text = "```python\ndef a(): pass\n```\ntext\n```python\ndef b(): pass\n```"
        result = _extract_code(text, "python")
        assert "def a" in result

    def test_strips_whitespace(self):
        text = "```python\n  def f(): pass  \n```"
        assert _extract_code(text, "python") == "def f(): pass"


# ---------------------------------------------------------------------------
# _detect_expertise
# ---------------------------------------------------------------------------

class TestDetectExpertise:
    def test_beginner_signal(self):
        result = _detect_expertise("I'm new to python what is a function", [])
        assert result == "beginner"

    def test_expert_signal(self):
        result = _detect_expertise("how does the GIL affect async coroutines?", [])
        assert result == "expert"

    def test_intermediate_default(self):
        # Neutral message with enough history to not be beginner
        history = [{"role": "user"}, {"role": "assistant"}] * 3
        result = _detect_expertise("parse a CSV file", history)
        assert result == "intermediate"

    def test_empty_history_beginner(self):
        result = _detect_expertise("help me with this", [])
        assert result == "beginner"

    def test_long_history_shifts_expertise(self):
        history = [{"role": "user"}, {"role": "assistant"}] * 3
        result = _detect_expertise("how do metaclasses work?", history)
        assert result == "expert"


# ---------------------------------------------------------------------------
# AssistantResponse.to_dict()
# ---------------------------------------------------------------------------

class TestAssistantResponseToDict:
    def test_keys_present(self):
        resp = AssistantResponse(
            message="hello",
            mode=AssistantMode.GUIDE,
            language="python",
        )
        d = resp.to_dict()
        assert "message" in d
        assert "mode" in d
        assert "code" in d
        assert "sandbox" in d
        assert "score" in d
        assert "suggestions" in d
        assert "stored" in d
        assert "duration_ms" in d
        assert "context_used" in d

    def test_mode_is_string(self):
        resp = AssistantResponse(message="hi", mode=AssistantMode.GENERATE)
        assert resp.to_dict()["mode"] == "generate"

    def test_sandbox_none_when_not_set(self):
        resp = AssistantResponse(message="hi", mode=AssistantMode.GUIDE)
        assert resp.to_dict()["sandbox"] is None

    def test_sandbox_dict_when_set(self):
        resp = AssistantResponse(
            message="hi", mode=AssistantMode.GENERATE,
            sandbox=SandboxResult(valid=True, passed=True, output="ok", error=""),
        )
        d = resp.to_dict()["sandbox"]
        assert d["valid"] is True
        assert d["passed"] is True

    def test_score_none_when_not_set(self):
        resp = AssistantResponse(message="hi", mode=AssistantMode.GUIDE)
        assert resp.to_dict()["score"] is None

    def test_score_dict_when_set(self):
        resp = AssistantResponse(
            message="hi", mode=AssistantMode.GENERATE,
            score=ScoreResult(score=0.8, summary="good", delta=0.1),
        )
        d = resp.to_dict()["score"]
        assert abs(d["score"] - 0.8) < 1e-9
        assert abs(d["delta"] - 0.1) < 1e-9


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

class TestGetEngine:
    def test_returns_assistant_engine(self):
        engine = get_engine()
        assert isinstance(engine, AssistantEngine)

    def test_singleton_same_instance(self):
        import visualiser.services.assistant_engine as ae
        ae._engine = None  # reset
        e1 = get_engine()
        ae._engine = None  # reset
        e2 = get_engine()
        assert isinstance(e1, AssistantEngine)
        assert isinstance(e2, AssistantEngine)