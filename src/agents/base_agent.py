from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from src.llm.router import Provider


@dataclass
class AgentContext:
    repo_path:   str
    task:        str
    history:     list[dict] = field(default_factory=list)
    metadata:    dict       = field(default_factory=dict)
    language:    str        = "python"
    target_file: str | None = None
    constraints: list[str]  = field(default_factory=list)


@dataclass
class AgentResult:
    agent_name: str
    output:     str
    success:    bool
    artifacts:  dict[str, Any] = field(default_factory=dict)
    error:      str | None     = None


class BaseAgent(ABC):
    name:          str = "base"
    system_prompt: str = "You are an expert software engineering agent."

    def __init__(self, provider: Provider | None = None):
        self._router  = None
        self.provider = provider

    @property
    def router(self):
        if self._router is None:
            from src.llm.router import get_router
            self._router = get_router()
        return self._router

    @router.setter
    def router(self, value):
        self._router = value

    def run(self, context: AgentContext) -> AgentResult:
        try:
            prompt   = self.build_prompt(context)
            response = self.router.complete(
                prompt=prompt,
                system=self.system_prompt,
                provider=self.provider,
            )
            return self.parse_response(response, context)
        except Exception as exc:
            return AgentResult(
                agent_name=self.name,
                output="",
                success=False,
                error=str(exc),
            )

    @abstractmethod
    def build_prompt(self, context: AgentContext) -> str:
        raise NotImplementedError

    @abstractmethod
    def parse_response(self, response: str, context: AgentContext) -> AgentResult:
        raise NotImplementedError
