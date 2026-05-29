from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import List, Set


# ── Banned patterns ───────────────────────────────────────────────────────────

BANNED_MODULES = {
    "subprocess", "socket", "urllib",
    "requests", "httpx", "pickle", "marshal",
    "ftplib", "telnetlib", "smtplib",
}

BANNED_CALLS = {
    "eval", "exec", "compile", "__import__", "input",
}

BANNED_OS_ATTRS = {
    "system", "popen", "remove", "unlink", "rmdir",
    "removedirs", "rename", "renames",
}

BANNED_SHUTIL_ATTRS = {
    "rmtree", "move",
}

BANNED_BUILTINS_OPEN = {"open"}


# ── Result ────────────────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    valid:    bool
    errors:   List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.valid = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)


# ── Visitor ───────────────────────────────────────────────────────────────────

class _SafetyVisitor(ast.NodeVisitor):
    def __init__(self, result: ValidationResult) -> None:
        self._result = result
        self._imported_os     = False
        self._imported_shutil = False

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            name = alias.name.split(".")[0]
            if name in BANNED_MODULES:
                self._result.add_error(f"Banned module import: {alias.name}")
            if alias.name == "os" or alias.name.startswith("os."):
                self._imported_os = True
            if alias.name == "shutil":
                self._imported_shutil = True
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        base   = module.split(".")[0]
        if base in BANNED_MODULES:
            self._result.add_error(f"Banned module import: from {module}")
        if base == "os":
            self._imported_os = True
        if base == "shutil":
            self._imported_shutil = True
        # direct imports of dangerous functions
        for alias in node.names:
            if module == "os" and alias.name in BANNED_OS_ATTRS:
                self._result.add_error(f"Banned import: from os import {alias.name}")
            if module == "shutil" and alias.name in BANNED_SHUTIL_ATTRS:
                self._result.add_error(f"Banned import: from shutil import {alias.name}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # bare calls: eval(), exec(), open()
        if isinstance(node.func, ast.Name):
            if node.func.id in BANNED_CALLS:
                self._result.add_error(f"Banned call: {node.func.id}()")
            if node.func.id in BANNED_BUILTINS_OPEN:
                self._result.add_warning("Use of open() detected — file I/O not allowed in sandbox")

        # attribute calls: os.system(), shutil.rmtree()
        if isinstance(node.func, ast.Attribute):
            attr = node.func.attr
            if isinstance(node.func.value, ast.Name):
                obj = node.func.value.id
                if obj == "os" and attr in BANNED_OS_ATTRS:
                    self._result.add_error(f"Banned call: os.{attr}()")
                if obj == "shutil" and attr in BANNED_SHUTIL_ATTRS:
                    self._result.add_error(f"Banned call: shutil.{attr}()")

        self.generic_visit(node)


# ── Public API ────────────────────────────────────────────────────────────────

def validate_code(code: str, language: str = "python") -> ValidationResult:
    """
    Validate code for syntax errors and dangerous patterns.
    Only full Python safety analysis supported.
    Other languages get syntax-only check with a warning.
    """
    result = ValidationResult(valid=True)

    if not code or not code.strip():
        result.add_error("Empty code — nothing to validate.")
        return result

    if language != "python":
        result.add_warning(f"Safety validation not supported for {language} — proceeding with caution.")
        return result

    # ── Syntax check ──────────────────────────────────────────────────────────
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        result.add_error(f"SyntaxError at line {e.lineno}: {e.msg}")
        return result

    # ── Safety check ─────────────────────────────────────────────────────────
    visitor = _SafetyVisitor(result)
    visitor.visit(tree)

    return result
