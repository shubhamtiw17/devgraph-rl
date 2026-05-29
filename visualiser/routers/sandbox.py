from __future__ import annotations

from typing import Optional, Dict
from fastapi import APIRouter
from pydantic import BaseModel

from src.sandbox.sandbox import get_sandbox

router = APIRouter(prefix="/api/sandbox", tags=["sandbox"])


# ── Request / Response models ─────────────────────────────────────────────────

class RunRequest(BaseModel):
    code:        str
    language:    str                     = "python"
    test_code:   Optional[str]           = None
    extra_files: Optional[Dict[str, str]] = None


class ValidateRequest(BaseModel):
    code:     str
    language: str = "python"


class ValidationResponse(BaseModel):
    valid:    bool
    errors:   list[str]
    warnings: list[str]


class ExecutionResponse(BaseModel):
    success:       bool
    stdout:        str
    stderr:        str
    exit_code:     int
    timed_out:     bool
    duration_ms:   float


class TestResponse(BaseModel):
    success:     bool
    passed:      int
    failed:      int
    errors:      int
    total:       int
    pass_rate:   float
    output:      str
    timed_out:   bool
    duration_ms: float


class SandboxResponse(BaseModel):
    success:        bool
    summary:        str
    validation:     ValidationResponse
    execution:      Optional[ExecutionResponse] = None
    tests:          Optional[TestResponse]      = None
    skipped_reason: Optional[str]              = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/validate", response_model=ValidationResponse)
def validate_code(req: ValidateRequest) -> ValidationResponse:
    """Validate code syntax and safety without executing."""
    sb = get_sandbox()
    r  = sb.validate_only(req.code, req.language)
    return ValidationResponse(
        valid=r.valid,
        errors=r.errors,
        warnings=r.warnings,
    )


@router.post("/run", response_model=SandboxResponse)
def run_code(req: RunRequest) -> SandboxResponse:
    """Validate and execute code. Optionally run tests."""
    sb = get_sandbox()
    r  = sb.run(
        code=req.code,
        language=req.language,
        test_code=req.test_code,
        extra_files=req.extra_files,
    )

    execution = None
    if r.execution:
        execution = ExecutionResponse(
            success=r.execution.success,
            stdout=r.execution.stdout,
            stderr=r.execution.stderr,
            exit_code=r.execution.exit_code,
            timed_out=r.execution.timed_out,
            duration_ms=r.execution.duration_ms,
        )

    tests = None
    if r.tests:
        tests = TestResponse(
            success=r.tests.success,
            passed=r.tests.passed,
            failed=r.tests.failed,
            errors=r.tests.errors,
            total=r.tests.total,
            pass_rate=r.tests.pass_rate,
            output=r.tests.output,
            timed_out=r.tests.timed_out,
            duration_ms=r.tests.duration_ms,
        )

    return SandboxResponse(
        success=r.success,
        summary=r.summary,
        validation=ValidationResponse(
            valid=r.validation.valid,
            errors=r.validation.errors,
            warnings=r.validation.warnings,
        ),
        execution=execution,
        tests=tests,
        skipped_reason=r.skipped_reason,
    )
