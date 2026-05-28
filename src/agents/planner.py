"""
Planner Agent — decomposes a high-level task into ordered subtasks.
Uses Pydantic for strict JSON validation of LLM output.
"""
from __future__ import annotations

import json
from pydantic import BaseModel, ValidationError

from src.agents.base_agent import BaseAgent, AgentContext, AgentResult


class PlannerOutput(BaseModel):
    """Strict schema for planner JSON output."""
    subtasks: list[str]


# history entries the planner cares about
RELEVANT_HISTORY = ["architect", "reviewer"]


class PlannerAgent(BaseAgent):
    name = "planner"
    system_prompt = """
You are a senior software architect with 15 years of experience in
large-scale Python systems.

Your job is to decompose engineering tasks into ordered, concrete subtasks
that a junior coding agent can execute independently.

Rules:
- Each subtask must be self-contained and independently executable
- Order subtasks by dependency — earlier subtasks must not depend on later ones
- Be specific: "Add null check to login() in auth/login.py" not "fix login"
- Maximum 6 subtasks — if more are needed, group related work
- Always consider: tests, error handling, and documentation as subtasks

Output ONLY valid JSON — no explanation, no markdown:
{"subtasks": ["subtask 1", "subtask 2", "subtask 3"]}
"""

    def build_prompt(self, context: AgentContext) -> str:
        # only include history from relevant agents
        relevant = [
            h for h in context.history
            if h["agent"] in RELEVANT_HISTORY
        ][-3:]

        history_str = ""
        if relevant:
            history_str = "\n\nPrevious steps:\n" + "\n".join(
                f"[{h['agent']}]: {h['output'][:300]}"
                f"{'...' if len(h['output']) > 300 else ''}"
                for h in relevant
            )

        constraints_str = ""
        if context.constraints:
            constraints_str = "\n\nConstraints:\n" + "\n".join(
                f"- {c}" for c in context.constraints
            )

        target_str = ""
        if context.target_file:
            target_str = f"\nTarget file: {context.target_file}"

        return (
            f"Repository: {context.repo_path}\n"
            f"Task: {context.task}"
            f"{target_str}"
            f"{constraints_str}"
            f"{history_str}\n\n"
            "Decompose this into 3 to 6 concrete, actionable subtasks. "
            "Respond ONLY with JSON."
        )

    def parse_response(
        self, response: str, context: AgentContext
    ) -> AgentResult:
        # strip markdown fences
        clean = (
            response.strip()
            .removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )

        try:
            data     = PlannerOutput.model_validate_json(clean)
            subtasks = data.subtasks

            if len(subtasks) == 0:
                raise ValueError("subtasks list is empty")

            return AgentResult(
                agent_name=self.name,
                output="\n".join(f"- {t}" for t in subtasks),
                success=True,
                artifacts={"subtasks": subtasks},
            )

        except (ValidationError, ValueError, json.JSONDecodeError):
            # graceful fallback — treat whole task as single subtask
            return AgentResult(
                agent_name=self.name,
                output=context.task,
                success=True,
                artifacts={"subtasks": [context.task]},
            )
