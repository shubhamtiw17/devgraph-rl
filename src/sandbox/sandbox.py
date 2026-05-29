from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Dict

from src.sandbox.validator import validate_code, ValidationResult
from src.sandbox.executor import Executor, ExecutionResult
from src.sandbox.test_runner import TestRunner, TestResult


# ── Unified result ────────────────────────────────────────────────────────────

@dataclass
class SandboxResult:
    success:          bool
    validation:       ValidationResult  = None
    execution:        Optional[ExecutionResult] = None
    tests:            Optional[TestResult]      = None
    skipped_reason:   Optional[str]             = None

    @property
    def stdout(self) -> str:
        return self.execution.stdout if self.execution else ""

    @property
    def stderr(self) -> str:
        return self.execution.stderr if self.execution else ""

    @property
    def pass_rate(self) -> float:
        return self.tests.pass_rate if self.tests else 0.0

    @property
    def summary(self) -> str:
        parts = []
        if self.validation and not self.validation.valid:
            parts.append(f"Validation failed: {'; '.join(self.validation.errors)}")
        if self.execution:
            status = "✓ ran" if self.execution.success else "✗ crashed"
            parts.append(f"{status} in {self.execution.duration_ms:.0f}ms")
        if self.tests:
            parts.append(f"Tests: {self.tests.passed}/{self.tests.total} passed ({self.tests.pass_rate:.0%})")
        return " | ".join(parts) if parts else "No execution"


# ── Sandbox ───────────────────────────────────────────────────────────────────

class Sandbox:
    """
    Main sandbox API. Validates, executes, and optionally tests code.
    Single entry point for all agent code execution.
    """

    def __init__(
        self,
        timeout_seconds: int = 30,
        max_output_bytes: int = 65536,
    ) -> None:
        self._executor   = Executor(timeout_seconds, max_output_bytes)
        self._runner     = TestRunner(timeout_seconds)
        self._timeout    = timeout_seconds

    def run(
        self,
        code:       str,
        language:   str = "python",
        test_code:  Optional[str] = None,
        extra_files: Optional[Dict[str, str]] = None,
    ) -> SandboxResult:
        """
        Validate and execute code. Optionally run tests.

        Args:
            code:        Source code to execute.
            language:    Programming language.
            test_code:   Optional pytest test file content.
            extra_files: Optional helper files written alongside code.

        Returns:
            SandboxResult with validation, execution, and test results.
        """
        # ── Step 1: Validate ──────────────────────────────────────────────────
        validation = validate_code(code, language)

        if not validation.valid:
            return SandboxResult(
                success=False,
                validation=validation,
                skipped_reason="Validation failed — code not executed",
            )

        # ── Step 2: Execute ───────────────────────────────────────────────────
        execution = self._executor.run(code, extra_files=extra_files)

        # ── Step 3: Run tests (optional) ──────────────────────────────────────
        tests = None
        if test_code and language == "python":
            tests = self._runner.run(
                source_code=code,
                test_code=test_code,
            )

        success = execution.success and (tests.success if tests else True)

        return SandboxResult(
            success=success,
            validation=validation,
            execution=execution,
            tests=tests,
        )

    def run_repo_tests(self, repo_path: str) -> TestResult:
        """
        Run existing pytest suite in a repo.
        Used to check if agent changes break existing tests.
        """
        return self._runner.run_existing_tests(repo_path)

    def validate_only(self, code: str, language: str = "python") -> ValidationResult:
        """Quick validation without execution."""
        return validate_code(code, language)


# ── Module-level singleton ────────────────────────────────────────────────────

_sandbox: Optional[Sandbox] = None


def get_sandbox() -> Sandbox:
    global _sandbox
    if _sandbox is None:
        timeout = int(os.getenv("SANDBOX_TIMEOUT_SECONDS", "30"))
        _sandbox = Sandbox(timeout_seconds=timeout)
    return _sandbox
