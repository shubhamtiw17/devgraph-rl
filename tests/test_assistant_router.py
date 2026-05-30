
from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI

from visualiser.routers.assistant import router
from visualiser.services.assistant_engine import (
    AssistantResponse,
    AssistantMode,
    SandboxResult,
    ScoreResult,
)


# ---------------------------------------------------------------------------
# Test app + client
# ---------------------------------------------------------------------------

app = FastAPI()
app.include_router(router)
client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_response(
    mode: AssistantMode = AssistantMode.GUIDE,
    message: str = "Here is my answer.",
    code: str = None,
    score: float = None,
    sandbox_passed: bool = None,
) -> AssistantResponse:
    resp = AssistantResponse(
        message=message,
        mode=mode,
        code=code,
        language="python",
        suggestions=["Do X", "Try Y"],
        stored=False,
        duration_ms=42.0,
        context_used=False,
    )
    if score is not None:
        resp.score = ScoreResult(score=score, summary="ok", delta=0.1)
    if sandbox_passed is not None:
        resp.sandbox = SandboxResult(valid=True, passed=sandbox_passed, output="ok")
    return resp


def mock_engine(response: AssistantResponse):
    engine = MagicMock()
    engine.chat.return_value = response
    return engine


# ---------------------------------------------------------------------------
# POST /api/assistant/chat
# ---------------------------------------------------------------------------

class TestChatEndpoint:
    def test_returns_200(self):
        with patch("visualiser.routers.assistant.get_engine",
                   return_value=mock_engine(make_response())), \
             patch("visualiser.routers.assistant._build_context",
                   return_value=MagicMock(
                       repo_name=None, repo_language=None,
                       loaded_file=None, n_memories=0,
                       to_prompt_section=lambda: ""
                   )):
            r = client.post("/api/assistant/chat",
                            json={"message": "explain decorators"})
        assert r.status_code == 200

    def test_response_has_required_keys(self):
        with patch("visualiser.routers.assistant.get_engine",
                   return_value=mock_engine(make_response())), \
             patch("visualiser.routers.assistant._build_context",
                   return_value=MagicMock(
                       repo_name=None, repo_language=None,
                       loaded_file=None, n_memories=0,
                   )):
            r = client.post("/api/assistant/chat",
                            json={"message": "explain decorators"})
        data = r.json()
        for key in ["session_id", "mode", "message", "code",
                    "sandbox", "score", "suggestions", "stored",
                    "duration_ms", "context_used", "context"]:
            assert key in data, f"Missing key: {key}"

    def test_mode_in_response(self):
        with patch("visualiser.routers.assistant.get_engine",
                   return_value=mock_engine(make_response(mode=AssistantMode.GENERATE))), \
             patch("visualiser.routers.assistant._build_context",
                   return_value=MagicMock(
                       repo_name=None, repo_language=None,
                       loaded_file=None, n_memories=0,
                   )):
            r = client.post("/api/assistant/chat",
                            json={"message": "write a sort function"})
        assert r.json()["mode"] == "generate"

    def test_empty_message_returns_400(self):
        r = client.post("/api/assistant/chat", json={"message": ""})
        assert r.status_code == 400

    def test_whitespace_message_returns_400(self):
        r = client.post("/api/assistant/chat", json={"message": "   "})
        assert r.status_code == 400

    def test_session_id_default(self):
        with patch("visualiser.routers.assistant.get_engine",
                   return_value=mock_engine(make_response())), \
             patch("visualiser.routers.assistant._build_context",
                   return_value=MagicMock(
                       repo_name=None, repo_language=None,
                       loaded_file=None, n_memories=0,
                   )):
            r = client.post("/api/assistant/chat",
                            json={"message": "hello"})
        assert r.json()["session_id"] == "default"

    def test_custom_session_id(self):
        with patch("visualiser.routers.assistant.get_engine",
                   return_value=mock_engine(make_response())), \
             patch("visualiser.routers.assistant._build_context",
                   return_value=MagicMock(
                       repo_name=None, repo_language=None,
                       loaded_file=None, n_memories=0,
                   )):
            r = client.post("/api/assistant/chat",
                            json={"message": "hello", "session_id": "user_42"})
        assert r.json()["session_id"] == "user_42"

    def test_code_in_response(self):
        resp = make_response(code="def f(): pass")
        with patch("visualiser.routers.assistant.get_engine",
                   return_value=mock_engine(resp)), \
             patch("visualiser.routers.assistant._build_context",
                   return_value=MagicMock(
                       repo_name=None, repo_language=None,
                       loaded_file=None, n_memories=0,
                   )):
            r = client.post("/api/assistant/chat",
                            json={"message": "write a function"})
        assert r.json()["code"] == "def f(): pass"

    def test_score_in_response(self):
        resp = make_response(score=0.85)
        with patch("visualiser.routers.assistant.get_engine",
                   return_value=mock_engine(resp)), \
             patch("visualiser.routers.assistant._build_context",
                   return_value=MagicMock(
                       repo_name=None, repo_language=None,
                       loaded_file=None, n_memories=0,
                   )):
            r = client.post("/api/assistant/chat",
                            json={"message": "write a function"})
        score = r.json()["score"]
        assert score is not None
        assert abs(score["score"] - 0.85) < 1e-6

    def test_sandbox_in_response(self):
        resp = make_response(sandbox_passed=True)
        with patch("visualiser.routers.assistant.get_engine",
                   return_value=mock_engine(resp)), \
             patch("visualiser.routers.assistant._build_context",
                   return_value=MagicMock(
                       repo_name=None, repo_language=None,
                       loaded_file=None, n_memories=0,
                   )):
            r = client.post("/api/assistant/chat",
                            json={"message": "write a function"})
        sandbox = r.json()["sandbox"]
        assert sandbox is not None
        assert sandbox["passed"] is True

    def test_suggestions_in_response(self):
        with patch("visualiser.routers.assistant.get_engine",
                   return_value=mock_engine(make_response())), \
             patch("visualiser.routers.assistant._build_context",
                   return_value=MagicMock(
                       repo_name=None, repo_language=None,
                       loaded_file=None, n_memories=0,
                   )):
            r = client.post("/api/assistant/chat",
                            json={"message": "explain decorators"})
        assert isinstance(r.json()["suggestions"], list)

    def test_language_forwarded(self):
        engine = mock_engine(make_response())
        with patch("visualiser.routers.assistant.get_engine",
                   return_value=engine), \
             patch("visualiser.routers.assistant._build_context",
                   return_value=MagicMock(
                       repo_name=None, repo_language=None,
                       loaded_file=None, n_memories=0,
                   )):
            client.post("/api/assistant/chat",
                        json={"message": "write a function", "language": "javascript"})
        call_kwargs = engine.chat.call_args
        assert call_kwargs.kwargs.get("language") == "javascript" or \
               (call_kwargs.args and "javascript" in str(call_kwargs.args))

    def test_context_section_in_response(self):
        with patch("visualiser.routers.assistant.get_engine",
                   return_value=mock_engine(make_response())), \
             patch("visualiser.routers.assistant._build_context",
                   return_value=MagicMock(
                       repo_name="myrepo", repo_language="python",
                       loaded_file=None, n_memories=5,
                   )):
            r = client.post("/api/assistant/chat",
                            json={"message": "explain decorators"})
        ctx = r.json()["context"]
        assert "repo_name" in ctx
        assert "n_memories" in ctx


# ---------------------------------------------------------------------------
# POST /api/assistant/reset
# ---------------------------------------------------------------------------

class TestResetEndpoint:
    def test_reset_returns_200(self):
        r = client.post("/api/assistant/reset", json={"session_id": "test_session"})
        assert r.status_code == 200

    def test_reset_response(self):
        r = client.post("/api/assistant/reset", json={"session_id": "test_session"})
        data = r.json()
        assert data["reset"] is True
        assert data["session_id"] == "test_session"

    def test_reset_clears_history(self):
        # Add some history
        with patch("visualiser.routers.assistant.get_engine",
                   return_value=mock_engine(make_response())), \
             patch("visualiser.routers.assistant._build_context",
                   return_value=MagicMock(
                       repo_name=None, repo_language=None,
                       loaded_file=None, n_memories=0,
                   )):
            client.post("/api/assistant/chat",
                        json={"message": "hello", "session_id": "clear_test"})

        # Reset
        client.post("/api/assistant/reset", json={"session_id": "clear_test"})

        # History should be empty
        r = client.get("/api/assistant/history?session_id=clear_test")
        assert r.json()["turns"] == 0

    def test_reset_nonexistent_session_ok(self):
        r = client.post("/api/assistant/reset",
                        json={"session_id": "nonexistent_xyz"})
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# GET /api/assistant/context
# ---------------------------------------------------------------------------

class TestContextEndpoint:
    def test_returns_200(self):
        with patch("visualiser.routers.assistant._build_context",
                   return_value=MagicMock(
                       repo_name=None, repo_language=None,
                       loaded_file=None, n_memories=0,
                       reward_stats=None,
                   )):
            r = client.get("/api/assistant/context")
        assert r.status_code == 200

    def test_response_has_required_keys(self):
        with patch("visualiser.routers.assistant._build_context",
                   return_value=MagicMock(
                       repo_name="myrepo", repo_language="python",
                       loaded_file=None, n_memories=3,
                       reward_stats={"total": 10, "average": 0.7},
                   )):
            r = client.get("/api/assistant/context")
        data = r.json()
        assert "repo_name" in data
        assert "n_memories" in data
        assert "reward_stats" in data
        assert "has_file" in data


# ---------------------------------------------------------------------------
# GET /api/assistant/history
# ---------------------------------------------------------------------------

class TestHistoryEndpoint:
    def test_empty_history(self):
        r = client.get("/api/assistant/history?session_id=empty_xyz")
        assert r.status_code == 200
        data = r.json()
        assert data["turns"] == 0
        assert data["history"] == []

    def test_history_grows_with_messages(self):
        sid = "history_test_abc"
        with patch("visualiser.routers.assistant.get_engine",
                   return_value=mock_engine(make_response())), \
             patch("visualiser.routers.assistant._build_context",
                   return_value=MagicMock(
                       repo_name=None, repo_language=None,
                       loaded_file=None, n_memories=0,
                   )):
            client.post("/api/assistant/chat",
                        json={"message": "msg1", "session_id": sid})
            client.post("/api/assistant/chat",
                        json={"message": "msg2", "session_id": sid})

        r = client.get(f"/api/assistant/history?session_id={sid}")
        assert r.json()["turns"] == 2

    def test_session_id_in_response(self):
        r = client.get("/api/assistant/history?session_id=my_session")
        assert r.json()["session_id"] == "my_session"

    def test_default_session_id(self):
        r = client.get("/api/assistant/history")
        assert r.json()["session_id"] == "default"