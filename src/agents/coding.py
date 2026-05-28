from __future__ import annotations

import re

from src.agents.base_agent import BaseAgent, AgentContext, AgentResult

RELEVANT_HISTORY = ["planner", "architect"]


class CodingAgent(BaseAgent):
    name = "coding"
    system_prompt = """
You are a senior software engineer. You write clean, production-grade code.

Standards you always follow:
- Type hints on all function signatures
- Docstrings on all classes and public methods
- No magic numbers — use named constants
- Handle errors explicitly — never silent failures
- No global state
- Functions do one thing only

Output format:
- Output the COMPLETE file — never truncate with comments like "# rest of code"
- Use a single fenced code block with the correct language tag
- Include all imports at the top of the file
"""

    def build_prompt(self, context: AgentContext) -> str:
        lang = context.language

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
            f"Language: {lang}\n"
            f"Task: {context.task}"
            f"{target_str}"
            f"{constraints_str}"
            f"{history_str}\n\n"
            f"Generate the required code inside a ```{lang} block. "
            "Output the complete file — do not truncate."
        )

    def parse_response(self, response: str, context: AgentContext) -> AgentResult:
        lang = context.language

        # 1. exact language tag
        pattern_exact = rf"```{re.escape(lang)}\n(.*?)```"
        code_blocks = re.findall(pattern_exact, response, re.DOTALL)

        # 2. any fenced block
        if not code_blocks:
            code_blocks = re.findall(
                r"```(?:\w+)?\n(.*?)```", response, re.DOTALL
            )

        # 3. raw response fallback
        code = code_blocks[0].strip() if code_blocks else response.strip()

        return AgentResult(
            agent_name=self.name,
            output=response,
            success=True,
            artifacts={
                "code": code,
                "language": lang,
            },
        )
