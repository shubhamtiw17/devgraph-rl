from __future__ import annotations

import sys
import tempfile
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List


# ── Result ────────────────────────────────────────────────────────────────────

@dataclass
class ExecutionResult:
    success:    bool
    stdout:     str               = ""
    stderr:     str               = ""
    exit_code:  int               = 0
    timed_out:  bool              = False
    error:      Optional[str]     = None
    duration_ms: float            = 0.0


# ── Executor ──────────────────────────────────────────────────────────────────

class Executor:
    """
    Runs Python code in an isolated subprocess with timeout.
    Never runs code directly in the main process.
    """

    def __init__(
        self,
        timeout_seconds: int = 30,
        max_output_bytes: int = 65536,   # 64KB output cap
    ) -> None:
        self._timeout   = timeout_seconds
        self._max_output = max_output_bytes

    def run(
        self,
        code: str,
        extra_files: Optional[dict[str, str]] = None,
    ) -> ExecutionResult:
        """
        Execute Python code in a temporary directory.

        Args:
            code:        Python source code to run.
            extra_files: Optional dict of {filename: content} written
                         alongside the main script (e.g. helper modules).

        Returns:
            ExecutionResult with stdout, stderr, exit_code, duration.
        """
        import time

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            # write extra files first
            if extra_files:
                for fname, content in extra_files.items():
                    (tmp / fname).write_text(content, encoding="utf-8")

            # write main script
            script = tmp / "main.py"
            script.write_text(code, encoding="utf-8")

            t0 = time.perf_counter()
            try:
                proc = subprocess.run(
                    [sys.executable, str(script)],
                    capture_output=True,
                    text=True,
                    timeout=self._timeout,
                    cwd=str(tmp),
                )
                duration_ms = (time.perf_counter() - t0) * 1000

                stdout = proc.stdout[:self._max_output]
                stderr = proc.stderr[:self._max_output]

                return ExecutionResult(
                    success=proc.returncode == 0,
                    stdout=stdout,
                    stderr=stderr,
                    exit_code=proc.returncode,
                    timed_out=False,
                    duration_ms=round(duration_ms, 2),
                )

            except subprocess.TimeoutExpired:
                return ExecutionResult(
                    success=False,
                    stderr=f"Execution timed out after {self._timeout}s",
                    exit_code=-1,
                    timed_out=True,
                    duration_ms=self._timeout * 1000,
                )

            except Exception as e:
                return ExecutionResult(
                    success=False,
                    stderr=str(e),
                    exit_code=-1,
                    error=str(e),
                )