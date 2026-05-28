from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import networkx as nx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hard-fail import
# ---------------------------------------------------------------------------

try:
    from tree_sitter_languages import get_language
    from tree_sitter import Parser as _TSParser
except ImportError as _e:
    raise ImportError(
        "tree-sitter-languages is required for DependencyGraph.\n"
        "Install it with:\n"
        "    pip install -e '.[graphs]'\n"
        "or directly:\n"
        "    pip install tree-sitter-languages tree-sitter==0.21.3"
    ) from _e


def _make_parser(ts_language) -> "_TSParser":
    """Create a Parser and bind it to ts_language. Works with tree-sitter 0.21.x."""
    parser = _TSParser()
    parser.set_language(ts_language)
    return parser

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_LANGUAGES = ("python", "javascript", "java", "cpp")

FILE_EXTENSIONS: dict[str, list[str]] = {
    "python":     [".py"],
    "javascript": [".js", ".mjs", ".ts"],
    "java":       [".java"],
    "cpp":        [".cpp", ".cc", ".cxx", ".h", ".hpp"],
}

E_IMPORTS = "imports"
N_FILE    = "file"


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class GraphBuildError(RuntimeError):
    """Raised when a source file cannot be parsed during graph construction."""


# ---------------------------------------------------------------------------
# Tree-sitter query strings per language
# Captures the module/path strings from import statements only.
# ---------------------------------------------------------------------------

# Python: `import foo`, `from foo.bar import baz`
_PY_IMPORT_QUERY = """
(import_statement
  name: (dotted_name) @module)

(import_from_statement
  module_name: (dotted_name) @module
  name: (_) @symbol)

(import_from_statement
  module_name: (relative_import) @module)
"""

# JavaScript/TypeScript: import ... from '...', require('...')
_JS_IMPORT_QUERY = """
(import_statement
  source: (string) @path)

(call_expression
  function: (identifier) @fn
  arguments: (arguments (string) @path)
  (#eq? @fn "require"))
"""

# Java: import com.example.Foo;
_JAVA_IMPORT_QUERY = """
(import_declaration
  (scoped_identifier) @module)
"""

# C++: #include "local.h"  (quoted = local; angle = stdlib, skipped)
_CPP_INCLUDE_QUERY = """
(preproc_include
  path: (string_literal) @path)
"""

QUERIES: dict[str, str] = {
    "python":     _PY_IMPORT_QUERY,
    "javascript": _JS_IMPORT_QUERY,
    "java":       _JAVA_IMPORT_QUERY,
    "cpp":        _CPP_INCLUDE_QUERY,
}


# ---------------------------------------------------------------------------
# Per-language import extractors
# ---------------------------------------------------------------------------

def _extract_python(
    source: bytes,
    rel: str,
    language,
    query,
) -> list[tuple[str, int]]:

    parser = _make_parser(language)
    tree   = _parse_or_raise(parser, source, rel)
    captures = query.captures(tree.root_node)

    # Group captures by module, count symbols
    modules: dict[str, int] = {}
    for node, capture_name in captures:
        text = node.text.decode("utf-8", errors="replace")
        if capture_name == "module":
            if text not in modules:
                modules[text] = 0
        elif capture_name == "symbol":
            # find the most recently seen module key
            if modules:
                last = list(modules)[-1]
                modules[last] += 1

    # Any module with 0 symbols was a plain `import x` → weight 1
    return [(mod, max(count, 1)) for mod, count in modules.items()]


def _extract_javascript(
    source: bytes,
    rel: str,
    language,
    query,
) -> list[tuple[str, int]]:
    parser = _make_parser(language)
    tree   = _parse_or_raise(parser, source, rel)
    captures = query.captures(tree.root_node)

    paths: list[str] = []
    for node, capture_name in captures:
        if capture_name == "path":
            # Strip surrounding quotes
            raw = node.text.decode("utf-8", errors="replace").strip("'\"` ")
            if raw:
                paths.append(raw)

    # weight = 1 per import statement (symbol count not extractable without
    # deeper analysis of import specifiers — kept simple intentionally)
    return [(p, 1) for p in paths]


def _extract_java(
    source: bytes,
    rel: str,
    language,
    query,
) -> list[tuple[str, int]]:
    parser = _make_parser(language)
    tree   = _parse_or_raise(parser, source, rel)
    captures = query.captures(tree.root_node)

    modules: list[str] = []
    for node, capture_name in captures:
        if capture_name == "module":
            modules.append(node.text.decode("utf-8", errors="replace"))
    return [(m, 1) for m in modules]


def _extract_cpp(
    source: bytes,
    rel: str,
    language,
    query,
) -> list[tuple[str, int]]:
    parser = _make_parser(language)
    tree   = _parse_or_raise(parser, source, rel)
    captures = query.captures(tree.root_node)

    paths: list[str] = []
    for node, capture_name in captures:
        if capture_name == "path":
            # Strip quotes — only local includes reach here (angle brackets
            # are a different node type and not matched by string_literal)
            raw = node.text.decode("utf-8", errors="replace").strip('"')
            if raw:
                paths.append(raw)
    return [(p, 1) for p in paths]


EXTRACTORS = {
    "python":     _extract_python,
    "javascript": _extract_javascript,
    "java":       _extract_java,
    "cpp":        _extract_cpp,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_or_raise(parser, source: bytes, rel: str):
    """Parse source bytes; raise GraphBuildError on any tree-sitter error."""
    tree = parser.parse(source)
    if tree.root_node.has_error:
        raise GraphBuildError(
            f"tree-sitter reported a parse error in: {rel}\n"
            "Fix the syntax error in that file and retry."
        )
    return tree


def _file_id(rel: str) -> str:
    return f"file:{rel}"


def _resolve_import(
    raw: str,
    language: str,
    known_files: set[str],
) -> Optional[str]:
    if language == "python":
        parts = raw.split(".")
        for start in range(len(parts)):
            as_path = "/".join(parts[start:])
            for ext in FILE_EXTENSIONS["python"]:
                candidate = f"{as_path}{ext}"
                if candidate in known_files:
                    return candidate
            init = f"{as_path}/__init__.py"
            if init in known_files:
                return init

    elif language == "javascript":
        # Normalise: strip leading ./ but keep ../
        stripped = raw.lstrip("./")
        # Try with and without extensions
        candidates = [stripped] + [f"{stripped}{ext}" for ext in FILE_EXTENSIONS["javascript"]]
        for c in candidates:
            if c in known_files:
                return c

    elif language == "java":
        # Last dotted component is the class name = filename
        parts = raw.split(".")
        class_name = parts[-1]
        for rel in known_files:
            if Path(rel).stem == class_name:
                return rel

    elif language == "cpp":
        # raw is already a filename like "shapes.h"
        for rel in known_files:
            if Path(rel).name == raw:
                return rel

    return None


# ---------------------------------------------------------------------------
# Main public class
# ---------------------------------------------------------------------------

class DependencyGraph:

    def __init__(self) -> None:
        self.graph: nx.DiGraph = nx.DiGraph()
        self.build_time_ms: float = 0.0
        self._language: str = ""

    def build(self, repo_path: str | Path, language: str) -> "DependencyGraph":
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language: {language!r}. "
                f"Choose one of: {SUPPORTED_LANGUAGES}"
            )

        root = Path(repo_path).resolve()
        if not root.exists():
            raise FileNotFoundError(f"repo_path does not exist: {root}")

        t0 = time.perf_counter()

        self.graph.clear()
        self._language = language

        extensions   = FILE_EXTENSIONS[language]
        ts_language  = get_language(language)
        query        = ts_language.query(QUERIES[language])
        extractor    = EXTRACTORS[language]

        # Collect all source files
        source_files: list[Path] = []
        for ext in extensions:
            source_files.extend(sorted(root.rglob(f"*{ext}")))

        if not source_files:
            logger.warning("No %s source files found under %s", language, root)
            self.build_time_ms = (time.perf_counter() - t0) * 1000
            return self

        # Pass 1: add all file nodes
        known_files: set[str] = set()
        for src_file in source_files:
            rel = str(src_file.relative_to(root))
            known_files.add(rel)
            self.graph.add_node(
                _file_id(rel),
                kind=N_FILE,
                path=rel,
                language=language,
            )

        # Pass 2: extract imports and build edges
        for src_file in source_files:
            rel    = str(src_file.relative_to(root))
            source = src_file.read_bytes()

            imports = extractor(source, rel, ts_language, query)

            for raw_import, symbol_count in imports:
                target_rel = _resolve_import(raw_import, language, known_files)
                if target_rel and target_rel != rel:
                    src_id = _file_id(rel)
                    tgt_id = _file_id(target_rel)
                    # Accumulate weight if edge already exists
                    if self.graph.has_edge(src_id, tgt_id):
                        self.graph[src_id][tgt_id]["weight"] += symbol_count
                    else:
                        self.graph.add_edge(
                            src_id, tgt_id,
                            kind=E_IMPORTS,
                            weight=symbol_count,
                        )

        self.build_time_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "DependencyGraph[%s] built: %d nodes, %d edges in %.1f ms",
            language,
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
            self.build_time_ms,
        )
        return self

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_dependencies(self, file_node: str) -> list[tuple[str, int]]:
        return sorted(
            [
                (tgt, data["weight"])
                for _, tgt, data in self.graph.out_edges(file_node, data=True)
            ],
            key=lambda x: x[1],
            reverse=True,
        )

    def get_dependents(self, file_node: str) -> list[tuple[str, int]]:
        return sorted(
            [
                (src, data["weight"])
                for src, _, data in self.graph.in_edges(file_node, data=True)
            ],
            key=lambda x: x[1],
            reverse=True,
        )

    def most_depended_upon(self, top_n: int = 5) -> list[tuple[str, int]]:
        return sorted(
            self.graph.in_degree(),
            key=lambda x: x[1],
            reverse=True,
        )[:top_n]