from __future__ import annotations

import logging
import time
from pathlib import Path

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
        "tree-sitter-languages is required for CallGraph.\n"
        "Install it with:\n"
        "    pip install -e '.[graphs]'\n"
        "or directly:\n"
        "    pip install tree-sitter-languages tree-sitter==0.21.3"
    ) from _e

from src.graphs.dependency_graph import (
    FILE_EXTENSIONS,
    SUPPORTED_LANGUAGES,
    GraphBuildError,
    _parse_or_raise,
    _make_parser,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

E_CALLS = "calls"
N_FUNC  = "func"


# ---------------------------------------------------------------------------
# Node ID helper
# ---------------------------------------------------------------------------

def _func_id(rel: str, qualname: str) -> str:
    return f"func:{rel}:{qualname}"


# ---------------------------------------------------------------------------
# Tree-sitter query strings per language
# Two captures per language: @def (function definitions) and @call (call sites)
# ---------------------------------------------------------------------------

_PY_QUERY = """
(function_definition
  name: (identifier) @def)

(call
  function: (identifier) @call)

(call
  function: (attribute
    attribute: (identifier) @call))
"""

_JS_QUERY = """
(function_declaration
  name: (identifier) @def)

(method_definition
  name: (property_identifier) @def)

(arrow_function) @def

(call_expression
  function: (identifier) @call)

(call_expression
  function: (member_expression
    property: (property_identifier) @call))
"""

_JAVA_QUERY = """
(method_declaration
  name: (identifier) @def)

(method_invocation
  name: (identifier) @call)
"""

_CPP_QUERY = """
(function_definition
  declarator: (function_declarator
    declarator: (identifier) @def))

(function_definition
  declarator: (function_declarator
    declarator: (qualified_identifier
      name: (identifier) @def)))

(function_definition
  declarator: (pointer_declarator
    declarator: (function_declarator
      declarator: (qualified_identifier
        name: (identifier) @def))))

(call_expression
  function: (identifier) @call)

(call_expression
  function: (field_expression
    field: (field_identifier) @call))

(call_expression
  function: (qualified_identifier
    name: (identifier) @call))
"""

QUERIES: dict[str, str] = {
    "python":     _PY_QUERY,
    "javascript": _JS_QUERY,
    "java":       _JAVA_QUERY,
    "cpp":        _CPP_QUERY,
}


# ---------------------------------------------------------------------------
# Per-file parser: collects definitions and call sites
# ---------------------------------------------------------------------------

class _FileCallParser:

    def __init__(self, language: str, source: bytes, rel: str) -> None:
        self._language = language
        self._source   = source
        self._rel      = rel

    def extract(self, ts_language, query) -> dict[str, list[str]]:
        parser = _make_parser(ts_language)
        tree   = _parse_or_raise(parser, self._source, self._rel)

        captures = query.captures(tree.root_node)

        # Sort captures by their start byte so we process them in source order
        captures_sorted = sorted(captures, key=lambda x: x[0].start_byte)

        defs: dict[str, list[str]] = {}   # qualname → [called, ...]
        current_def: str | None = None

        for node, capture_name in captures_sorted:
            name = node.text.decode("utf-8", errors="replace")
            if capture_name == "def":
                current_def = name
                if current_def not in defs:
                    defs[current_def] = []
            elif capture_name == "call" and current_def is not None:
                defs[current_def].append(name)

        return defs


# ---------------------------------------------------------------------------
# Main public class
# ---------------------------------------------------------------------------

class CallGraph:

    def __init__(self) -> None:
        self.graph: nx.DiGraph = nx.DiGraph()
        self.build_time_ms: float = 0.0
        self._language: str = ""

    def build(self, repo_path: str | Path, language: str) -> "CallGraph":
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

        extensions  = FILE_EXTENSIONS[language]
        ts_language = get_language(language)
        query       = ts_language.query(QUERIES[language])

        # Collect all source files
        source_files: list[Path] = []
        for ext in extensions:
            source_files.extend(sorted(root.rglob(f"*{ext}")))

        if not source_files:
            logger.warning("No %s source files found under %s", language, root)
            self.build_time_ms = (time.perf_counter() - t0) * 1000
            return self

        # Pass 1: extract all function definitions → add nodes
        # file_defs: rel → {func_name → [called_names]}
        file_defs: dict[str, dict[str, list[str]]] = {}

        for src_file in source_files:
            rel    = str(src_file.relative_to(root))
            source = src_file.read_bytes()
            parser = _FileCallParser(language, source, rel)
            defs   = parser.extract(ts_language, query)
            file_defs[rel] = defs

            for func_name in defs:
                node_id = _func_id(rel, func_name)
                self.graph.add_node(
                    node_id,
                    kind=N_FUNC,
                    name=func_name,
                    path=rel,
                    language=language,
                )

        # Build a lookup: func_name → [node_id, ...]  (multiple files may
        # define a function with the same name)
        name_to_nodes: dict[str, list[str]] = {}
        for node_id, data in self.graph.nodes(data=True):
            name = data["name"]
            name_to_nodes.setdefault(name, []).append(node_id)

        # Pass 2: build call edges
        for rel, defs in file_defs.items():
            for caller_name, called_names in defs.items():
                caller_id = _func_id(rel, caller_name)
                call_counts: dict[str, int] = {}

                for called_name in called_names:
                    # Resolve: prefer same-file definition, then any other file
                    same_file = _func_id(rel, called_name)
                    if same_file in self.graph and same_file != caller_id:
                        call_counts[same_file] = call_counts.get(same_file, 0) + 1
                    elif called_name in name_to_nodes:
                        for candidate in name_to_nodes[called_name]:
                            if candidate != caller_id:
                                call_counts[candidate] = call_counts.get(candidate, 0) + 1
                                break  # take first match only

                for callee_id, count in call_counts.items():
                    if self.graph.has_edge(caller_id, callee_id):
                        self.graph[caller_id][callee_id]["weight"] += count
                    else:
                        self.graph.add_edge(
                            caller_id, callee_id,
                            kind=E_CALLS,
                            weight=count,
                        )

        self.build_time_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "CallGraph[%s] built: %d nodes, %d edges in %.1f ms",
            language,
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
            self.build_time_ms,
        )
        return self

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_callees(self, func_node: str) -> list[tuple[str, int]]:
        return sorted(
            [
                (tgt, data["weight"])
                for _, tgt, data in self.graph.out_edges(func_node, data=True)
            ],
            key=lambda x: x[1],
            reverse=True,
        )

    def get_callers(self, func_node: str) -> list[tuple[str, int]]:
        return sorted(
            [
                (src, data["weight"])
                for src, _, data in self.graph.in_edges(func_node, data=True)
            ],
            key=lambda x: x[1],
            reverse=True,
        )

    def most_called(self, top_n: int = 5) -> list[tuple[str, int]]:
        return sorted(
            self.graph.in_degree(),
            key=lambda x: x[1],
            reverse=True,
        )[:top_n]