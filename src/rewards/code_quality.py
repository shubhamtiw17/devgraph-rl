from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional


# ── Result ────────────────────────────────────────────────────────────────────

@dataclass
class QualityResult:
    overall:          float                  # 0.0 - 1.0
    signals:          Dict[str, float]       # per-signal scores
    feedback:         List[str]              # human readable notes
    language:         str      = "python"
    error:            Optional[str] = None

    @property
    def summary(self) -> str:
        return f"Quality: {self.overall:.2f} | " + " | ".join(
            f"{k}: {v:.2f}" for k, v in self.signals.items()
        )


# ── Signal weights ────────────────────────────────────────────────────────────

SIGNAL_WEIGHTS = {
    "length":         0.15,   # not too short, not too long
    "complexity":     0.25,   # cyclomatic complexity
    "naming":         0.20,   # meaningful names
    "documentation":  0.20,   # docstrings present
    "error_handling": 0.20,   # try/except present where needed
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _count_lines(code: str) -> int:
    return len([l for l in code.splitlines() if l.strip()])


def _score_length(lines: int) -> float:
    """
    Ideal range: 10-200 lines.
    Too short → probably incomplete.
    Too long → probably needs refactoring.
    """
    if lines < 3:   return 0.2
    if lines < 10:  return 0.6
    if lines <= 200: return 1.0
    if lines <= 400: return 0.8
    if lines <= 600: return 0.6
    return 0.4


def _cyclomatic_complexity(tree: ast.AST) -> int:
    """Count branching points — if/for/while/except/with each add 1."""
    count = 1
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.For, ast.While, ast.ExceptHandler,
                              ast.With, ast.Assert, ast.comprehension)):
            count += 1
    return count


def _score_complexity(complexity: int, num_functions: int) -> float:
    """
    Score based on average complexity per function.
    Low complexity = clean code.
    """
    if num_functions == 0:
        avg = complexity
    else:
        avg = complexity / num_functions

    if avg <= 3:  return 1.0
    if avg <= 5:  return 0.85
    if avg <= 8:  return 0.65
    if avg <= 12: return 0.45
    return 0.25


def _score_naming(tree: ast.AST) -> tuple[float, list[str]]:
    """
    Check function/variable names:
    - Not single characters (except loop vars i, j, k, x, y)
    - Not generic names (foo, bar, temp, tmp, data, result)
    - snake_case for functions/vars
    """
    ALLOWED_SHORT = {"i", "j", "k", "x", "y", "n", "e", "f"}
    GENERIC_NAMES = {"foo", "bar", "baz", "temp", "tmp", "data",
                     "result", "val", "value", "obj", "item", "thing"}

    bad_names = []
    total = 0
    bad   = 0

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = node.name
            if name.startswith("_"):
                name = name.lstrip("_")
            total += 1
            if len(name) == 1 and name not in ALLOWED_SHORT:
                bad += 1
                bad_names.append(f"single-char function: {node.name}")
            elif name.lower() in GENERIC_NAMES:
                bad += 1
                bad_names.append(f"generic name: {node.name}")

        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            name = node.id
            total += 1
            if len(name) == 1 and name not in ALLOWED_SHORT:
                bad += 1
                bad_names.append(f"single-char variable: {name}")

    if total == 0:
        return 0.8, []

    score = 1.0 - min(bad / total, 0.8)
    return round(score, 3), bad_names[:3]


def _score_documentation(tree: ast.AST, num_functions: int) -> float:
    """Count functions with docstrings vs total functions."""
    if num_functions == 0:
        return 0.8  # no functions — not penalized

    documented = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if (ast.get_docstring(node)):
                documented += 1

    return min(documented / num_functions, 1.0)


def _score_error_handling(tree: ast.AST, num_functions: int) -> float:
    """
    Check for try/except blocks.
    Not required for every function but presence is a good signal.
    """
    try_blocks  = sum(1 for n in ast.walk(tree) if isinstance(n, ast.Try))
    except_blocks = sum(1 for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler))

    if num_functions == 0:
        return 0.7

    # reward presence of error handling
    if try_blocks == 0:
        return 0.5   # no error handling at all
    if try_blocks >= num_functions * 0.3:
        return 1.0   # good coverage
    return 0.75      # some error handling


# ── Public API ────────────────────────────────────────────────────────────────

def score_code_quality(code: str, language: str = "python") -> QualityResult:
    """
    Score code quality using AST analysis.
    Only full analysis for Python — other languages get a neutral score.
    """
    if language != "python":
        return QualityResult(
            overall=0.7,
            signals={k: 0.7 for k in SIGNAL_WEIGHTS},
            feedback=[f"Full quality analysis not supported for {language}"],
            language=language,
        )

    if not code or not code.strip():
        return QualityResult(
            overall=0.0,
            signals={k: 0.0 for k in SIGNAL_WEIGHTS},
            feedback=["Empty code"],
            language=language,
            error="Empty code",
        )

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return QualityResult(
            overall=0.0,
            signals={k: 0.0 for k in SIGNAL_WEIGHTS},
            feedback=[f"SyntaxError: {e}"],
            language=language,
            error=str(e),
        )

    lines        = _count_lines(code)
    num_functions = sum(
        1 for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    )
    complexity   = _cyclomatic_complexity(tree)

    length_score  = _score_length(lines)
    complex_score = _score_complexity(complexity, num_functions)
    naming_score, bad_names = _score_naming(tree)
    doc_score     = _score_documentation(tree, num_functions)
    err_score     = _score_error_handling(tree, num_functions)

    signals = {
        "length":         length_score,
        "complexity":     complex_score,
        "naming":         naming_score,
        "documentation":  doc_score,
        "error_handling": err_score,
    }

    overall = sum(SIGNAL_WEIGHTS[k] * v for k, v in signals.items())

    feedback = []
    if length_score < 0.7:
        feedback.append(f"Code length ({lines} lines) outside ideal range (10-200)")
    if complex_score < 0.7:
        feedback.append(f"High cyclomatic complexity ({complexity}) — consider refactoring")
    if naming_score < 0.7:
        feedback.append(f"Poor naming: {', '.join(bad_names)}")
    if doc_score < 0.5:
        feedback.append(f"Low documentation coverage ({doc_score:.0%} of functions)")
    if err_score < 0.6:
        feedback.append("No error handling detected")
    if not feedback:
        feedback.append("Code quality looks good")

    return QualityResult(
        overall=round(overall, 3),
        signals={k: round(v, 3) for k, v in signals.items()},
        feedback=feedback,
        language=language,
    )
