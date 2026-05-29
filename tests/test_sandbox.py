from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from src.sandbox.validator import validate_code, ValidationResult
from src.sandbox.executor import Executor, ExecutionResult
from src.sandbox.test_runner import TestRunner, TestResult
from src.sandbox.sandbox import Sandbox, SandboxResult, get_sandbox


# ════════════════════════════════════════════════
# VALIDATOR TESTS
# ════════════════════════════════════════════════

class TestValidator:
    def test_valid_simple_code(self):
        r = validate_code("def add(a, b): return a + b")
        assert r.valid is True
        assert r.errors == []

    def test_valid_class(self):
        r = validate_code("class Foo:\n    def bar(self): return 1")
        assert r.valid is True

    def test_syntax_error(self):
        r = validate_code("def broken(:")
        assert r.valid is False
        assert any("SyntaxError" in e for e in r.errors)

    def test_empty_code(self):
        r = validate_code("")
        assert r.valid is False
        assert any("Empty" in e for e in r.errors)

    def test_whitespace_only(self):
        r = validate_code("   \n\n  ")
        assert r.valid is False

    def test_banned_import_subprocess(self):
        r = validate_code("import subprocess\nsubprocess.run(['ls'])")
        assert r.valid is False
        assert any("subprocess" in e for e in r.errors)

    def test_banned_import_socket(self):
        r = validate_code("import socket")
        assert r.valid is False
        assert any("socket" in e for e in r.errors)

    def test_banned_os_system(self):
        r = validate_code("import os\nos.system('ls')")
        assert r.valid is False
        assert any("os.system" in e for e in r.errors)

    def test_banned_shutil_rmtree(self):
        r = validate_code("import shutil\nshutil.rmtree('/tmp')")
        assert r.valid is False
        assert any("shutil.rmtree" in e for e in r.errors)

    def test_banned_eval(self):
        r = validate_code("eval('print(1)')")
        assert r.valid is False
        assert any("eval" in e for e in r.errors)

    def test_banned_exec(self):
        r = validate_code("exec('x = 1')")
        assert r.valid is False
        assert any("exec" in e for e in r.errors)

    def test_non_python_language(self):
        r = validate_code("console.log('hi')", language="javascript")
        assert r.valid is True
        assert any("not supported" in w for w in r.warnings)

    def test_valid_imports_allowed(self):
        r = validate_code("import math\nimport json\nimport re\nprint(math.pi)")
        assert r.valid is True

    def test_multiline_valid(self):
        code = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

result = factorial(5)
print(result)
"""
        r = validate_code(code)
        assert r.valid is True


# ════════════════════════════════════════════════
# EXECUTOR TESTS
# ════════════════════════════════════════════════

class TestExecutor:
    def test_simple_execution(self):
        ex = Executor(timeout_seconds=10)
        r  = ex.run("print('hello')")
        assert r.success is True
        assert "hello" in r.stdout

    def test_exit_code_zero_on_success(self):
        ex = Executor(timeout_seconds=10)
        r  = ex.run("x = 1 + 1")
        assert r.exit_code == 0

    def test_runtime_error(self):
        ex = Executor(timeout_seconds=10)
        r  = ex.run("x = 1 / 0")
        assert r.success is False
        assert r.exit_code != 0
        assert "ZeroDivisionError" in r.stderr

    def test_stdout_captured(self):
        ex = Executor(timeout_seconds=10)
        r  = ex.run("print('test output')")
        assert "test output" in r.stdout

    def test_stderr_captured(self):
        ex = Executor(timeout_seconds=10)
        r  = ex.run("import sys\nsys.stderr.write('err msg')")
        assert "err msg" in r.stderr

    def test_duration_recorded(self):
        ex = Executor(timeout_seconds=10)
        r  = ex.run("x = 1 + 1")
        assert r.duration_ms >= 0

    def test_extra_files(self):
        ex = Executor(timeout_seconds=10)
        helper = "def greet(): return 'hi'"
        r = ex.run(
            "from helper import greet\nprint(greet())",
            extra_files={"helper.py": helper},
        )
        assert r.success is True
        assert "hi" in r.stdout

    def test_timeout(self):
        ex = Executor(timeout_seconds=2)
        r  = ex.run("while True: pass")
        assert r.success is False
        assert r.timed_out is True

    def test_multiline_output(self):
        ex = Executor(timeout_seconds=10)
        r  = ex.run("for i in range(3): print(i)")
        assert r.success is True
        assert "0" in r.stdout
        assert "2" in r.stdout


# ════════════════════════════════════════════════
# TEST RUNNER TESTS
# ════════════════════════════════════════════════

class TestTestRunner:
    SOURCE = """
def add(a, b):
    return a + b

def multiply(a, b):
    return a * b
"""
    TESTS_PASSING = """
from solution import add, multiply

def test_add():
    assert add(2, 3) == 5

def test_multiply():
    assert multiply(3, 4) == 12
"""
    TESTS_FAILING = """
from solution import add

def test_add_wrong():
    assert add(2, 3) == 999
"""

    def test_all_passing(self):
        tr = TestRunner(timeout_seconds=30)
        r  = tr.run(self.SOURCE, self.TESTS_PASSING)
        assert r.success is True
        assert r.passed == 2
        assert r.failed == 0
        assert r.pass_rate == 1.0

    def test_failing_test(self):
        tr = TestRunner(timeout_seconds=30)
        r  = tr.run(self.SOURCE, self.TESTS_FAILING)
        assert r.success is False
        assert r.failed == 1
        assert r.pass_rate == 0.0

    def test_total_count(self):
        tr = TestRunner(timeout_seconds=30)
        r  = tr.run(self.SOURCE, self.TESTS_PASSING)
        assert r.total == 2

    def test_pass_rate_partial(self):
        mixed_tests = """
from solution import add

def test_pass():
    assert add(1, 1) == 2

def test_fail():
    assert add(1, 1) == 999
"""
        tr = TestRunner(timeout_seconds=30)
        r  = tr.run(self.SOURCE, mixed_tests)
        assert r.passed == 1
        assert r.failed == 1
        assert r.pass_rate == 0.5

    def test_output_captured(self):
        tr = TestRunner(timeout_seconds=30)
        r  = tr.run(self.SOURCE, self.TESTS_PASSING)
        assert len(r.output) > 0

    def test_duration_recorded(self):
        tr = TestRunner(timeout_seconds=30)
        r  = tr.run(self.SOURCE, self.TESTS_PASSING)
        assert r.duration_ms > 0

    def test_import_error_in_tests(self):
        bad_tests = "from nonexistent_module import foo\ndef test_x(): assert foo() == 1"
        tr = TestRunner(timeout_seconds=30)
        r  = tr.run(self.SOURCE, bad_tests)
        assert r.success is False


# ════════════════════════════════════════════════
# SANDBOX TESTS
# ════════════════════════════════════════════════

class TestSandbox:
    def test_valid_code_runs(self):
        sb = Sandbox(timeout_seconds=10)
        r  = sb.run("print('ok')")
        assert r.success is True
        assert "ok" in r.stdout

    def test_dangerous_code_blocked(self):
        sb = Sandbox(timeout_seconds=10)
        r  = sb.run("import os\nos.system('ls')")
        assert r.success is False
        assert r.skipped_reason is not None
        assert r.execution is None

    def test_runtime_error(self):
        sb = Sandbox(timeout_seconds=10)
        r  = sb.run("raise ValueError('test error')")
        assert r.success is False
        assert r.execution is not None

    def test_with_tests_passing(self):
        sb     = Sandbox(timeout_seconds=30)
        source = "def add(a, b): return a + b"
        tests  = "from solution import add\ndef test_add(): assert add(1,2)==3"
        r      = sb.run(source, test_code=tests)
        assert r.success is True
        assert r.pass_rate == 1.0

    def test_with_tests_failing(self):
        sb     = Sandbox(timeout_seconds=30)
        source = "def add(a, b): return a + b"
        tests  = "from solution import add\ndef test_add(): assert add(1,2)==999"
        r      = sb.run(source, test_code=tests)
        assert r.success is False
        assert r.pass_rate == 0.0

    def test_summary_property(self):
        sb = Sandbox(timeout_seconds=10)
        r  = sb.run("print('hi')")
        assert isinstance(r.summary, str)
        assert len(r.summary) > 0

    def test_stdout_property(self):
        sb = Sandbox(timeout_seconds=10)
        r  = sb.run("print('hello')")
        assert "hello" in r.stdout

    def test_stderr_property_on_blocked(self):
        sb = Sandbox(timeout_seconds=10)
        r  = sb.run("eval('x')")
        assert r.stderr == ""   # blocked before execution

    def test_validate_only(self):
        sb = Sandbox(timeout_seconds=10)
        r  = sb.validate_only("def foo(): return 1")
        assert r.valid is True

    def test_validate_only_catches_syntax(self):
        sb = Sandbox(timeout_seconds=10)
        r  = sb.validate_only("def broken(:")
        assert r.valid is False

    def test_singleton(self):
        sb1 = get_sandbox()
        sb2 = get_sandbox()
        assert sb1 is sb2

    def test_non_python_skips_safety(self):
        sb = Sandbox(timeout_seconds=10)
        r  = sb.validate_only("console.log('hi')", language="javascript")
        assert r.valid is True
