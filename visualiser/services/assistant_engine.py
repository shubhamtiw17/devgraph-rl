from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mode detection
# ---------------------------------------------------------------------------

class AssistantMode(str, Enum):
    GENERATE = "generate"
    IMPROVE  = "improve"
    GUIDE    = "guide"


# Keywords that signal each mode
_GENERATE_SIGNALS = [
    "write", "create", "build", "implement", "add", "generate",
    "make", "code", "function", "class", "script", "module",
]

_IMPROVE_SIGNALS = [
    "refactor", "improve", "fix", "optimise", "optimize", "clean",
    "review", "better", "rewrite", "simplify", "speed up", "slow",
    "long", "messy", "bad", "broken", "wrong", "debug",
]

_GUIDE_SIGNALS = [
    "explain", "how", "why", "what", "when", "where", "help",
    "understand", "setup", "install", "configure", "error",
    "failing", "doesn't work", "not working", "stuck",
]


def detect_mode(message: str) -> AssistantMode:
    msg = message.lower()

    improve_score  = sum(1 for s in _IMPROVE_SIGNALS  if s in msg)
    generate_score = sum(1 for s in _GENERATE_SIGNALS if s in msg)
    guide_score    = sum(1 for s in _GUIDE_SIGNALS    if s in msg)

    # Improve wins ties with Generate (safer — don't overwrite user code)
    if improve_score >= generate_score and improve_score > 0:
        return AssistantMode.IMPROVE
    if generate_score > guide_score:
        return AssistantMode.GENERATE
    return AssistantMode.GUIDE


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------

@dataclass
class SystemContext:

    repo_name:        Optional[str]       = None
    repo_language:    Optional[str]       = None
    loaded_file:      Optional[str]       = None
    file_content:     Optional[str]       = None
    graph_summary:    Optional[str]       = None   # brief text from graph
    recent_memories:  list[str]           = field(default_factory=list)
    reward_stats:     Optional[dict]      = None
    n_memories:       int                 = 0

    def to_prompt_section(self) -> str:
        parts = []
        if self.repo_name:
            lang = f" ({self.repo_language})" if self.repo_language else ""
            parts.append(f"Loaded repo: {self.repo_name}{lang}")
        if self.loaded_file:
            parts.append(f"Active file: {self.loaded_file}")
        if self.file_content:
            preview = self.file_content[:800]
            parts.append(f"File content (first 800 chars):\n```\n{preview}\n```")
        if self.graph_summary:
            parts.append(f"Graph summary: {self.graph_summary}")
        if self.recent_memories:
            mem_text = "\n".join(f"  - {m}" for m in self.recent_memories[:5])
            parts.append(f"Recent memory context:\n{mem_text}")
        if self.reward_stats:
            avg = self.reward_stats.get("average", 0)
            total = self.reward_stats.get("total", 0)
            parts.append(f"Reward history: {total} scored outputs, avg score {avg:.2f}")
        return "\n".join(parts) if parts else "No system context available."


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------

@dataclass
class SandboxResult:
    valid:   bool = True
    passed:  bool = True
    output:  str  = ""
    error:   str  = ""


@dataclass
class ScoreResult:
    score:      float      = 0.0
    dimensions: dict       = field(default_factory=dict)
    summary:    str        = ""
    delta:      float      = 0.0   # score - previous_score (Improve mode)


@dataclass
class AssistantResponse:

    message:      str               # main text response
    mode:         AssistantMode
    code:         Optional[str]     = None   # extracted code block if present
    language:     str               = "python"
    sandbox:      Optional[SandboxResult] = None
    score:        Optional[ScoreResult]   = None
    suggestions:  list[str]         = field(default_factory=list)
    stored:       bool              = False   # True if stored to memory
    duration_ms:  float             = 0.0
    context_used: bool              = False

    def to_dict(self) -> dict:
        return {
            "message":     self.message,
            "mode":        self.mode.value,
            "code":        self.code,
            "language":    self.language,
            "sandbox":     {
                "valid":  self.sandbox.valid,
                "passed": self.sandbox.passed,
                "output": self.sandbox.output,
                "error":  self.sandbox.error,
            } if self.sandbox else None,
            "score": {
                "score":      self.score.score,
                "dimensions": self.score.dimensions,
                "summary":    self.score.summary,
                "delta":      self.score.delta,
            } if self.score else None,
            "suggestions":  self.suggestions,
            "stored":       self.stored,
            "duration_ms":  self.duration_ms,
            "context_used": self.context_used,
        }


# ---------------------------------------------------------------------------
# Assistant Engine
# ---------------------------------------------------------------------------

class AssistantEngine:

    def __init__(
        self,
        llm_router=None,
        sandbox=None,
        reward_model=None,
        memory_manager=None,
        score_threshold: float = 0.70,
    ) -> None:

        self._router         = llm_router
        self._sandbox        = sandbox
        self._reward_model   = reward_model
        self._memory         = memory_manager
        self.score_threshold = score_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        message: str,
        context: Optional[SystemContext] = None,
        history: Optional[list[dict]]   = None,
        language: str = "python",
    ) -> AssistantResponse:

        start = time.time()
        ctx   = context or SystemContext()
        hist  = history or []
        mode  = detect_mode(message)

        logger.info("Assistant mode=%s message=%r", mode.value, message[:60])

        try:
            if mode == AssistantMode.GENERATE:
                response = self._generate(message, ctx, hist, language)
            elif mode == AssistantMode.IMPROVE:
                response = self._improve(message, ctx, hist, language)
            else:
                response = self._guide(message, ctx, hist, language)
        except Exception as exc:
            logger.error("Assistant engine error: %s", exc, exc_info=True)
            response = AssistantResponse(
                message=f"I encountered an error: {exc}. Please try again.",
                mode=mode,
                language=language,
            )

        response.duration_ms  = (time.time() - start) * 1000
        response.context_used = bool(
            ctx.repo_name or ctx.file_content or ctx.recent_memories
        )
        return response

    # ------------------------------------------------------------------
    # Mode handlers
    # ------------------------------------------------------------------

    def _generate(
        self,
        message: str,
        ctx: SystemContext,
        history: list[dict],
        language: str,
    ) -> AssistantResponse:

        system_prompt = self._build_system_prompt(
            mode=AssistantMode.GENERATE,
            ctx=ctx,
            language=language,
        )
        user_prompt = (
            f"Task: {message}\n\n"
            f"Generate complete, production-quality {language} code. "
            f"Include type hints, docstrings, and error handling. "
            f"Return ONLY the code in a single code block."
        )

        raw = self._llm_call(system_prompt, user_prompt, history)
        code = _extract_code(raw, language)

        # Validate through sandbox
        sandbox_result = self._sandbox_check(code, language)

        # Score the output
        score_result = self._score_output(
            task=message,
            output=code or raw,
            language=language,
            passed=sandbox_result.passed if sandbox_result else False,
        )

        # Auto-store to memory if score is high enough
        stored = self._maybe_store(
            text=f"Generated: {message}\n{code or raw}",
            score=score_result.score if score_result else 0.0,
            agent_type="coding",
        )

        suggestions = self._generate_suggestions(mode=AssistantMode.GENERATE, code=code)

        return AssistantResponse(
            message=raw,
            mode=AssistantMode.GENERATE,
            code=code,
            language=language,
            sandbox=sandbox_result,
            score=score_result,
            suggestions=suggestions,
            stored=stored,
        )

    def _improve(
        self,
        message: str,
        ctx: SystemContext,
        history: list[dict],
        language: str,
    ) -> AssistantResponse:

        # Extract code from context or from the message itself
        original_code = ctx.file_content or _extract_code(message, language) or ""
        original_score: Optional[ScoreResult] = None

        # Score original if we have code
        if original_code:
            original_score = self._score_output(
                task=message,
                output=original_code,
                language=language,
                passed=False,
            )

        system_prompt = self._build_system_prompt(
            mode=AssistantMode.IMPROVE,
            ctx=ctx,
            language=language,
        )
        user_prompt = (
            f"Request: {message}\n\n"
            + (f"Original code:\n```{language}\n{original_code}\n```\n\n"
               if original_code else "")
            + f"Improve this code: fix issues, add type hints, docstrings, "
              f"error handling, and follow best practices. "
              f"Return the improved version in a single code block, "
              f"then explain the key changes made."
        )

        raw = self._llm_call(system_prompt, user_prompt, history)
        improved_code = _extract_code(raw, language)

        # Validate improved code
        sandbox_result = self._sandbox_check(improved_code, language)

        # Score improved code
        new_score = self._score_output(
            task=message,
            output=improved_code or raw,
            language=language,
            passed=sandbox_result.passed if sandbox_result else False,
        )

        # Calculate delta vs original
        if new_score and original_score:
            new_score.delta = new_score.score - original_score.score

        # Store if improved
        stored = self._maybe_store(
            text=f"Improved: {message}\n{improved_code or raw}",
            score=new_score.score if new_score else 0.0,
            agent_type="coding",
        )

        suggestions = self._generate_suggestions(
            mode=AssistantMode.IMPROVE,
            code=improved_code,
            score=new_score,
        )

        return AssistantResponse(
            message=raw,
            mode=AssistantMode.IMPROVE,
            code=improved_code,
            language=language,
            sandbox=sandbox_result,
            score=new_score,
            suggestions=suggestions,
            stored=stored,
        )

    def _guide(
        self,
        message: str,
        ctx: SystemContext,
        history: list[dict],
        language: str,
    ) -> AssistantResponse:

        expertise = _detect_expertise(message, history)
        system_prompt = self._build_system_prompt(
            mode=AssistantMode.GUIDE,
            ctx=ctx,
            language=language,
            expertise=expertise,
        )
        user_prompt = message

        raw = self._llm_call(system_prompt, user_prompt, history)
        code = _extract_code(raw, language)

        suggestions = self._generate_suggestions(
            mode=AssistantMode.GUIDE,
            message=message,
        )

        return AssistantResponse(
            message=raw,
            mode=AssistantMode.GUIDE,
            code=code,
            language=language,
            suggestions=suggestions,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_system_prompt(
        self,
        mode: AssistantMode,
        ctx: SystemContext,
        language: str,
        expertise: str = "intermediate",
    ) -> str:

        base = (
            "You are DevGraph-RL Assistant — an expert software engineering AI "
            "built into a graph-augmented RLHF system. "
            "You have access to the user's loaded repository, its graph structure, "
            "reward history, and semantic memory.\n\n"
        )

        context_section = ctx.to_prompt_section()

        mode_instructions = {
            AssistantMode.GENERATE: (
                f"Mode: GENERATE\n"
                f"Language: {language}\n"
                f"Write complete, production-quality code. Always include:\n"
                f"- Type hints on all functions\n"
                f"- Docstrings (Google style)\n"
                f"- Error handling for edge cases\n"
                f"- Follow conventions visible in the loaded repo if present.\n"
                f"Return code in a fenced code block. Be concise after the code."
            ),
            AssistantMode.IMPROVE: (
                f"Mode: IMPROVE\n"
                f"Language: {language}\n"
                f"Analyse the provided code and improve it. Focus on:\n"
                f"- Reducing complexity and length\n"
                f"- Adding missing type hints and docstrings\n"
                f"- Fixing error handling gaps\n"
                f"- Following best practices for {language}\n"
                f"Return the improved code first, then explain changes briefly."
            ),
            AssistantMode.GUIDE: (
                f"Mode: GUIDE\n"
                f"Expertise level: {expertise}\n"
                f"{'Use simple language, avoid jargon, explain concepts from scratch.' if expertise == 'beginner' else ''}"
                f"{'Use technical language, skip basics, focus on specifics.' if expertise == 'expert' else ''}"
                f"Be direct and practical. If debugging, identify the root cause first. "
                f"If explaining, use concrete examples from the loaded repo when possible."
            ),
        }

        return (
            base
            + f"System context:\n{context_section}\n\n"
            + mode_instructions[mode]
        )

    def _llm_call(
        self,
        system: str,
        user: str,
        history: list[dict],
    ) -> str:
        if self._router is None:
            return self._fallback_response(user)

        try:
            # Build prompt from history + current message
            prompt_parts = []
            for turn in history[-6:]:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                prefix = "User" if role == "user" else "Assistant"
                prompt_parts.append(f"{prefix}: {content}")
            prompt_parts.append(f"User: {user}")
            prompt = "\n".join(prompt_parts)

            result = self._router.complete(
                prompt=prompt,
                system=system,
            )
            return result if isinstance(result, str) else str(result)

        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            return self._fallback_response(user)

    def _sandbox_check(
        self,
        code: Optional[str],
        language: str,
    ) -> Optional[SandboxResult]:
        if not code or self._sandbox is None:
            return None
        try:
            result = self._sandbox.run(code=code, language=language)
            return SandboxResult(
                valid=result.validation.valid,
                passed=result.execution.success if result.execution else False,
                output=result.execution.stdout[:500] if result.execution else "",
                error=result.execution.stderr[:300] if result.execution else "",
            )
        except Exception as exc:
            logger.warning("Sandbox check failed: %s", exc)
            return SandboxResult(valid=True, passed=False, error=str(exc))

    def _score_output(
        self,
        task: str,
        output: str,
        language: str,
        passed: bool,
    ) -> Optional[ScoreResult]:
        if not output or self._reward_model is None:
            return None
        try:
            result = self._reward_model.score(
                task=task,
                output=output,
                agent_type="coding",
                language=language,
                execution_success=passed,
                test_pass_rate=1.0 if passed else 0.0,
            )
            return ScoreResult(
                score=result.final_score,
                dimensions={d.name: d.score for d in result.dimensions},
                summary=result.summary,
            )
        except Exception as exc:
            logger.warning("Scoring failed: %s", exc)
            return None

    def _maybe_store(
        self,
        text: str,
        score: float,
        agent_type: str,
    ) -> bool:
        if score < self.score_threshold or self._memory is None:
            return False
        try:
            self._memory.store(
                text=text,
                embedder="minilm",
                metadata={"agent_type": agent_type, "score": score},
            )
            return True
        except Exception as exc:
            logger.warning("Memory store failed: %s", exc)
            return False

    @staticmethod
    def _generate_suggestions(
        mode: AssistantMode,
        code: Optional[str] = None,
        score: Optional[ScoreResult] = None,
        message: str = "",
    ) -> list[str]:
        suggestions: list[str] = []

        if mode == AssistantMode.GENERATE:
            if code:
                suggestions.append("Run this in the Sandbox tab to verify it works")
                suggestions.append("Score this output in the Rewards tab")
            suggestions.append("Ask me to add tests for this code")
            suggestions.append("Ask me to improve this code")

        elif mode == AssistantMode.IMPROVE:
            if score and score.score > 0:
                suggestions.append(f"Score is {score.score:.2f} — store to memory if happy")
            if score and score.delta > 0:
                suggestions.append(f"Improvement: +{score.delta:.2f} over original")
            suggestions.append("Ask me to write tests for the improved version")
            suggestions.append("Apply this to the repo and reload the graph")

        elif mode == AssistantMode.GUIDE:
            if "setup" in message.lower() or "install" in message.lower():
                suggestions.append("Check the README for full setup instructions")
            suggestions.append("Load a repo to get context-aware answers")
            suggestions.append("Ask me to generate or improve specific code")

        return suggestions[:3]

    @staticmethod
    def _fallback_response(message: str) -> str:
        return (
            f"I received your message: '{message[:100]}'\n\n"
            "However, the LLM router is not available right now. "
            "Please check that your GROQ_API_KEY or GEMINI_API_KEY "
            "is set in the .env file and restart the server."
        )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _extract_code(text: str, language: str = "python") -> Optional[str]:
    if not text:
        return None

    # Try language-specific fence: ```python ... ```
    pattern = rf"```{re.escape(language)}\s*\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Try generic fence: ``` ... ```
    match = re.search(r"```\w*\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    return None


def _detect_expertise(message: str, history: list[dict]) -> str:
    expert_terms = [
        "async", "coroutine", "metaclass", "decorator", "generator",
        "comprehension", "context manager", "descriptor", "closure",
        "mro", "gc", "gil", "bytecode", "cython", "ctypes",
    ]
    beginner_terms = [
        "what is", "how do i", "i don't understand", "confused",
        "new to", "beginner", "just started", "first time",
    ]

    msg_lower = message.lower()

    expert_hits   = sum(1 for t in expert_terms   if t in msg_lower)
    beginner_hits = sum(1 for t in beginner_terms if t in msg_lower)

    # Long history suggests comfortable user
    history_depth = len(history)

    if expert_hits >= 2 or (expert_hits >= 1 and history_depth > 4):
        return "expert"
    if beginner_hits >= 1 or history_depth == 0:
        return "beginner"
    return "intermediate"


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_engine: Optional[AssistantEngine] = None


def get_engine() -> AssistantEngine:
    global _engine

    llm_router     = None
    sandbox        = None
    reward_model   = None
    memory_manager = None

    try:
        from src.llm.router import get_router
        llm_router = get_router()
        logger.info("Assistant: LLM router connected")
    except Exception as e:
        logger.warning("Assistant: LLM router unavailable: %s", e)

    try:
        from src.sandbox.sandbox import get_sandbox
        sandbox = get_sandbox()
        logger.info("Assistant: Sandbox connected")
    except Exception as e:
        logger.warning("Assistant: Sandbox unavailable: %s", e)

    try:
        from src.rewards.reward_model import get_reward_model
        reward_model = get_reward_model()
        logger.info("Assistant: Reward model connected")
    except Exception as e:
        logger.warning("Assistant: Reward model unavailable: %s", e)

    try:
        from src.memory.memory_manager import get_memory_manager
        memory_manager = get_memory_manager()
        logger.info("Assistant: Memory manager connected")
    except Exception as e:
        logger.warning("Assistant: Memory manager unavailable: %s", e)

    _engine = AssistantEngine(
        llm_router=llm_router,
        sandbox=sandbox,
        reward_model=reward_model,
        memory_manager=memory_manager,
    )
    return _engine