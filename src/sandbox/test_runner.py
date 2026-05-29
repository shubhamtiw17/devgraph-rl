from __future__ import annotations

import sys
import re
import tempfile
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ── Result ────────────────────────────────────────────────────────────────────

@dataclass
class TestResult:
    success:      bool
    passed:       int           = 0
    failed:       int           = 0
    errors:       int           = 0
    total:        int           = 0
    pass_rate:    float         = 0.0
    output:       str           = ""
    timed_out:    bool          = False
    error:        Optional[str] = None
    duration_ms:  float         = 0.0

    def __post_init__(self):
        self.total     = self.passed + self.failed + self.errors
        self.pass_rate = self.passed / self.total if self.total > 0 else 0.0


# ── Parser ────────────────────────────────────────────────────────────────────

def _parse_pytest_output(output: str) -> dict:
    passed = failed = errors = 0
    passed_m = re.search(r"(\d+) passed", output)
    failed_m = re.search(r"(\d+) failed", output)
    error_m  = re.search(r"(\d+) error",  output)
    if passed_m: passed = int(passed_m.group(1))
    if failed_m: failed = int(failed_m.group(1))
    if error_m:  errors = int(error_m.group(1))
    return {"passed": passed, "failed": failed, "errors": errors}


# ── TestRunner ────────────────────────────────────────────────────────────────

class TestRunner:
    def __init__(self, timeout_seconds: int = 30) -> None:
        self._timeout = timeout_seconds

    def run(
        self,
        source_code:     str,
        test_code:       str,
        source_filename: str = "solution.py",
        test_filename:   str = "test_solution.py",
    ) -> TestResult:
        import time

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            # write conftest.py to add tmpdir to sys.path
            (tmp / "conftest.py").write_text(
                f"import sys\nsys.path.insert(0, r'{tmpdir}')\n",
                encoding="utf-8",
            )
            (tmp / source_filename).write_text(source_code, encoding="utf-8")
            (tmp / test_filename).write_text(test_code,     encoding="utf-8")

            t0 = time.perf_counter()
            try:
                proc = subprocess.run(
                    [
                        sys.executable, "-m", "pytest",
                        str(tmp / test_filename),
                        "-v", "--tb=short", "--no-header",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=self._timeout,
                    cwd=str(tmp),
                )
                duration_ms = (time.perf_counter() - t0) * 1000
                output = proc.stdout + proc.stderr
                counts = _parse_pytest_output(output)

                return TestResult(
                    success=proc.returncode == 0,
                    passed=counts["passed"],
                    failed=counts["failed"],
                    errors=counts["errors"],
                    output=output[:4096],
                    duration_ms=round(duration_ms, 2),
                )

            except subprocess.TimeoutExpired:
                return TestResult(
                    success=False,
                    output=f"Timed out after {self._timeout}s",
                    timed_out=True,
                    duration_ms=self._timeout * 1000,
                )

            except Exception as e:
                return TestResult(
                    success=False,
                    error=str(e),
                    output=str(e),
                )

    def run_existing_tests(
        self,
        repo_path: str,
        timeout_seconds: Optional[int] = None,
    ) -> TestResult:
        import time

        timeout = timeout_seconds or self._timeout
        t0 = time.perf_counter()

        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short", "-q"],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=repo_path,
            )
            duration_ms = (time.perf_counter() - t0) * 1000
            output = proc.stdout + proc.stderr
            counts = _parse_pytest_output(output)

            return TestResult(
                success=proc.returncode == 0,
                passed=counts["passed"],
                failed=counts["failed"],
                errors=counts["errors"],
                output=output[:4096],
                duration_ms=round(duration_ms, 2),
            )

        except subprocess.TimeoutExpired:
            return TestResult(
                success=False,
                output=f"Timed out after {timeout}s",
                timed_out=True,
            )

        except Exception as e:
            return TestResult(success=False, error=str(e), output=str(e))
