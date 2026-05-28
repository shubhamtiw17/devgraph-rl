"""
src/graphs/ast_graph.py

Parses a source repository into a NetworkX DiGraph capturing structural
relationships: file containment, class/function nesting, import edges,
and class inheritance.

Language support
----------------
Python     : stdlib ast module (no extra deps)
JavaScript : tree-sitter (tree-sitter-languages required)
Java       : tree-sitter
C++        : tree-sitter

Schema is identical across all languages:
  Nodes : file | class | func
  Edges : contains | imports | inherits
"""

from __future__ import annotations

import ast
import logging
import warnings
from pathlib import Path
from typing import Optional

import networkx as nx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Node / edge type constants
# ---------------------------------------------------------------------------
N_FILE  = "file"
N_CLASS = "class"
N_FUNC  = "func"

E_CONTAINS = "contains"
E_IMPORTS  = "imports"
E_INHERITS = "inherits"

# ---------------------------------------------------------------------------
# File extensions per language
# ---------------------------------------------------------------------------
_EXTENSIONS: dict[str, list[str]] = {
    "python":     [".py"],
    "javascript": [".js", ".mjs", ".ts"],
    "java":       [".java"],
    "cpp":        [".cpp", ".cc", ".cxx", ".h", ".hpp"],
}

# ---------------------------------------------------------------------------
# ID helpers
# ---------------------------------------------------------------------------

def _file_id(rel: str) -> str:
    return f"file:{rel}"

def _class_id(rel: str, name: str) -> str:
    return f"class:{rel}:{name}"

def _func_id(rel: str, qualname: str) -> str:
    return f"func:{rel}:{qualname}"


# ===========================================================================
# PYTHON PATH  —  stdlib ast (unchanged from Phase 4a)
# ===========================================================================

class _FileParser(ast.NodeVisitor):
    def __init__(self, graph: nx.DiGraph, rel: str, tree: ast.AST) -> None:
        self._g    = graph
        self._rel  = rel
        self._tree = tree
        self._class_stack: list[str] = []
        self.imports: list[str] = []
        self.class_bases: dict[str, list[str]] = {}

    def collect(self) -> None:
        self._g.add_node(_file_id(self._rel), kind=N_FILE, path=self._rel)
        self.visit(self._tree)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        class_node = _class_id(self._rel, node.name)
        self._g.add_node(class_node, kind=N_CLASS, name=node.name,
                         path=self._rel, lineno=node.lineno)
        parent = (_class_id(self._rel, self._class_stack[-1])
                  if self._class_stack else _file_id(self._rel))
        self._g.add_edge(parent, class_node, kind=E_CONTAINS)

        bases = [n for b in node.bases for n in [_name_from_expr(b)] if n]
        if bases:
            self.class_bases[node.name] = bases

        self._class_stack.append(node.name)
        self.generic_visit(node)
        self._class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_func(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_func(node)

    def _visit_func(self, node) -> None:
        qualname = (f"{self._class_stack[-1]}.{node.name}"
                    if self._class_stack else node.name)
        func_node = _func_id(self._rel, qualname)
        self._g.add_node(func_node, kind=N_FUNC, name=node.name,
                         qualname=qualname, path=self._rel, lineno=node.lineno)
        parent = (_class_id(self._rel, self._class_stack[-1])
                  if self._class_stack else _file_id(self._rel))
        self._g.add_edge(parent, func_node, kind=E_CONTAINS)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.append(alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            self.imports.append(node.module)


def _name_from_expr(expr: ast.expr) -> Optional[str]:
    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Attribute):
        v = _name_from_expr(expr.value)
        return f"{v}.{expr.attr}" if v else None
    return None


def _module_to_rel(module_str: str, known_files: set[str]) -> Optional[str]:
    parts = module_str.split(".")
    for start in range(len(parts)):
        as_path = "/".join(parts[start:])
        for candidate in [f"{as_path}.py", f"{as_path}/__init__.py"]:
            if candidate in known_files:
                return candidate
    return None


def _resolve_class(base_name: str, current_rel: str,
                   rel_to_parser: dict) -> Optional[str]:
    simple = base_name.split(".")[-1]
    candidate = _class_id(current_rel, simple)
    if candidate in rel_to_parser[current_rel]._g:
        return candidate
    for rel, parser in rel_to_parser.items():
        if rel == current_rel:
            continue
        candidate = _class_id(rel, simple)
        if candidate in parser._g:
            return candidate
    return None


# ===========================================================================
# TREE-SITTER PATH  —  JavaScript / Java / C++
# ===========================================================================

# ---------------------------------------------------------------------------
# tree-sitter queries per language
# Each query uses two capture names:
#   @class_name  — identifier that names a class definition
#   @func_name   — identifier that names a function/method definition
#   @func_anon   — arrow function assigned to a variable (JS only)
#   @import_path — module/path string in an import statement
#   @base_name   — identifier of a base class in an inheritance clause
# ---------------------------------------------------------------------------

_JS_QUERY = """
; ── Classes ──────────────────────────────────────────────────────────
(class_declaration
  name: (identifier) @class_name)

; ── Named functions and methods ──────────────────────────────────────
(function_declaration
  name: (identifier) @func_name)

(method_definition
  name: (property_identifier) @func_name)

; ── Arrow functions assigned to variables: const area = (r) => ...
(lexical_declaration
  (variable_declarator
    name: (identifier) @func_name
    value: (arrow_function)))

; ── Imports ───────────────────────────────────────────────────────────
(import_statement
  source: (string) @import_path)

; ── Inheritance: class Circle extends Shape ───────────────────────────
(class_heritage
  (identifier) @base_name)
"""

_JAVA_QUERY = """
; ── Classes ──────────────────────────────────────────────────────────
(class_declaration
  name: (identifier) @class_name)

; ── Methods ──────────────────────────────────────────────────────────
(method_declaration
  name: (identifier) @func_name)

; ── Imports ───────────────────────────────────────────────────────────
(import_declaration
  (scoped_identifier) @import_path)

; ── Inheritance: class Circle extends Shape ───────────────────────────
(superclass
  (type_identifier) @base_name)
"""

_CPP_QUERY = """
; ── Classes / structs ─────────────────────────────────────────────────
(class_specifier
  name: (type_identifier) @class_name)

(struct_specifier
  name: (type_identifier) @class_name)

; ── Free functions: void foo() { } ────────────────────────────────────
(function_definition
  declarator: (function_declarator
    declarator: (identifier) @func_name))

; ── Qualified methods: Circle::area() { } ─────────────────────────────
(function_definition
  declarator: (function_declarator
    declarator: (qualified_identifier
      name: (identifier) @func_name)))

; ── Includes (local only — string_literal, not system_lib_string) ─────
(preproc_include
  path: (string_literal) @import_path)

; ── Inheritance: class Circle : public Shape ──────────────────────────
(base_class_clause
  (type_identifier) @base_name)
"""

_TS_QUERIES = {
    "javascript": _JS_QUERY,
    "java":       _JAVA_QUERY,
    "cpp":        _CPP_QUERY,
}


class _TSFileData:
    def __init__(self) -> None:
        self.classes:      list[tuple[str, int]] = []  # (name, lineno)
        self.functions:    list[tuple[str, int]] = []  # (qualname, lineno)
        self.imports:      list[str]             = []  # raw import strings
        self.class_bases:  dict[str, list[str]]  = {}  # class_name → [base, ...]


class _TreeSitterASTBuilder:
    def __init__(self, graph: nx.DiGraph, language: str) -> None:
        self._g        = graph
        self._language = language
        self._file_data: dict[str, _TSFileData] = {}

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                from tree_sitter_languages import get_language as _get_lang
                from tree_sitter import Parser as _Parser
            ts_lang          = _get_lang(language)
            self._query      = ts_lang.query(_TS_QUERIES[language])
            self._ts_lang    = ts_lang
            self._Parser     = _Parser
        except ImportError:
            raise ImportError(
                "tree-sitter-languages is required for non-Python AST graphs.\n"
                "Install with: pip install tree-sitter-languages tree-sitter==0.21.3"
            )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def build(self, root: Path) -> None:
        extensions = _EXTENSIONS[self._language]
        source_files: list[Path] = []
        for ext in extensions:
            source_files.extend(sorted(root.rglob(f"*{ext}")))

        known_files: set[str] = set()

        # Pass 1: nodes
        for src_file in source_files:
            rel = str(src_file.relative_to(root))
            known_files.add(rel)
            source = src_file.read_bytes()
            data   = self._collect_file(rel, source)
            self._file_data[rel] = data

            # Add file node
            self._g.add_node(_file_id(rel), kind=N_FILE, path=rel)

            # Add class nodes + file→class edges
            for class_name, lineno in data.classes:
                cnode = _class_id(rel, class_name)
                self._g.add_node(cnode, kind=N_CLASS, name=class_name,
                                 path=rel, lineno=lineno)
                self._g.add_edge(_file_id(rel), cnode, kind=E_CONTAINS)

            # Add function nodes + parent→func edges
            for qualname, lineno in data.functions:
                # qualname is "ClassName.method" or bare "funcName"
                parts = qualname.split(".", 1)
                simple_name = parts[-1]
                fnode  = _func_id(rel, qualname)
                self._g.add_node(fnode, kind=N_FUNC, name=simple_name,
                                 qualname=qualname, path=rel, lineno=lineno)
                # parent: class if qualname contains ".", else file
                if len(parts) == 2:
                    parent = _class_id(rel, parts[0])
                    if parent in self._g:
                        self._g.add_edge(parent, fnode, kind=E_CONTAINS)
                    else:
                        self._g.add_edge(_file_id(rel), fnode, kind=E_CONTAINS)
                else:
                    self._g.add_edge(_file_id(rel), fnode, kind=E_CONTAINS)

        # Pass 2: cross-file edges
        self._build_cross_file_edges(known_files)

    # ------------------------------------------------------------------
    # Per-file collection
    # ------------------------------------------------------------------

    def _collect_file(self, rel: str, source: bytes) -> _TSFileData:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            parser = self._Parser()
            parser.set_language(self._ts_lang)

        tree     = parser.parse(source)
        captures = self._query.captures(tree.root_node)

        captures_sorted = sorted(captures, key=lambda x: x[0].start_byte)

        data          = _TSFileData()
        current_class: Optional[str] = None
        class_end_bytes: dict[str, int] = {}  # class_name → end byte

        for node, cap in captures_sorted:
            if cap == "class_name":
                # The class body ends at the parent class node's end byte
                class_node_end = node.parent.end_byte if node.parent else node.end_byte
                class_end_bytes[node.text.decode("utf-8", errors="replace")] = class_node_end

        for node, cap in captures_sorted:
            text   = node.text.decode("utf-8", errors="replace").strip("'\"` ")
            lineno = node.start_point[0] + 1 

            if cap == "class_name":
                current_class = text
                data.classes.append((text, lineno))

            elif cap == "func_name":
                # Update current_class based on byte position
                current_class = self._current_class_at(
                    node.start_byte, class_end_bytes, captures_sorted
                )
                qualname = (f"{current_class}.{text}"
                            if current_class else text)
                data.functions.append((qualname, lineno))

            elif cap == "import_path":
                data.imports.append(text)

            elif cap == "base_name":
                if current_class:
                    data.class_bases.setdefault(current_class, []).append(text)

        return data

    def _current_class_at(
        self,
        byte_pos: int,
        class_end_bytes: dict[str, int],
        captures_sorted: list,
    ) -> Optional[str]:
        candidates = [
            (end, name)
            for name, end in class_end_bytes.items()
            if end >= byte_pos
        ]
        if not candidates:
            return None
        _, class_name = min(candidates)
        return class_name

    # ------------------------------------------------------------------
    # Cross-file edges
    # ------------------------------------------------------------------

    def _build_cross_file_edges(self, known_files: set[str]) -> None:
        from src.graphs.dependency_graph import _resolve_import

        for rel, data in self._file_data.items():
            src_file = _file_id(rel)

            # Import edges
            for raw in data.imports:
                # Strip quotes left over from C++ string_literal captures
                raw = raw.strip('"\'')
                target = _resolve_import(raw, self._language, known_files)
                if target and target != rel:
                    self._g.add_edge(src_file, _file_id(target), kind=E_IMPORTS)

            # Inheritance edges
            for class_name, bases in data.class_bases.items():
                src_class = _class_id(rel, class_name)
                if src_class not in self._g:
                    continue
                for base in bases:
                    tgt = self._resolve_base(base, rel)
                    if tgt and tgt != src_class:
                        self._g.add_edge(src_class, tgt, kind=E_INHERITS)

    def _resolve_base(self, base_name: str, current_rel: str) -> Optional[str]:
        simple = base_name.split(".")[-1]
        candidate = _class_id(current_rel, simple)
        if candidate in self._g:
            return candidate
        for rel in self._file_data:
            if rel == current_rel:
                continue
            candidate = _class_id(rel, simple)
            if candidate in self._g:
                return candidate
        return None


# ===========================================================================
# PUBLIC CLASS
# ===========================================================================

class ASTGraph:
    def __init__(self) -> None:
        self.graph: nx.DiGraph = nx.DiGraph()
        self._rel_to_parser: dict[str, _FileParser] = {}  # Python path only

    def build(self, repo_path: str | Path,
              language: str = "python") -> "ASTGraph":
        root = Path(repo_path).resolve()
        if not root.exists():
            raise FileNotFoundError(f"repo_path does not exist: {root}")

        self.graph.clear()
        self._rel_to_parser.clear()

        if language == "python":
            self._build_python(root)
        elif language in ("javascript", "java", "cpp"):
            self._build_treesitter(root, language)
        else:
            raise ValueError(
                f"Unsupported language: {language!r}. "
                f"Choose from: python, javascript, java, cpp"
            )

        logger.info(
            "ASTGraph[%s] built: %d nodes, %d edges",
            language,
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )
        return self

    # ------------------------------------------------------------------
    # Python build (stdlib ast)
    # ------------------------------------------------------------------

    def _build_python(self, root: Path) -> None:
        for py_file in sorted(root.rglob("*.py")):
            rel = str(py_file.relative_to(root))
            try:
                source = py_file.read_text(encoding="utf-8", errors="replace")
                tree   = ast.parse(source, filename=str(py_file))
            except SyntaxError as exc:
                logger.warning("Skipping %s — SyntaxError: %s", rel, exc)
                continue
            parser = _FileParser(self.graph, rel, tree)
            parser.collect()
            self._rel_to_parser[rel] = parser

        self._build_python_cross_file_edges(root)

    def _build_python_cross_file_edges(self, root: Path) -> None:
        known_files = set(self._rel_to_parser)
        for rel, parser in self._rel_to_parser.items():
            src_file_node = _file_id(rel)
            for module_str in parser.imports:
                target_rel = _module_to_rel(module_str, known_files)
                if target_rel:
                    tgt = _file_id(target_rel)
                    if src_file_node != tgt:
                        self.graph.add_edge(src_file_node, tgt, kind=E_IMPORTS)
            for class_name, bases in parser.class_bases.items():
                src_class = _class_id(rel, class_name)
                for base in bases:
                    tgt = _resolve_class(base, rel, self._rel_to_parser)
                    if tgt and tgt != src_class:
                        self.graph.add_edge(src_class, tgt, kind=E_INHERITS)

    # ------------------------------------------------------------------
    # tree-sitter build (JS / Java / C++)
    # ------------------------------------------------------------------

    def _build_treesitter(self, root: Path, language: str) -> None:
        builder = _TreeSitterASTBuilder(self.graph, language)
        builder.build(root)

    # ------------------------------------------------------------------
    # Query helpers (unchanged)
    # ------------------------------------------------------------------

    def get_dependencies(self, file_node: str) -> list[str]:
        return [tgt for _, tgt, d in self.graph.out_edges(file_node, data=True)
                if d.get("kind") == E_IMPORTS]

    def get_class_hierarchy(self, class_node: str) -> list[str]:
        return [tgt for _, tgt, d in self.graph.out_edges(class_node, data=True)
                if d.get("kind") == E_INHERITS]

    def get_file_nodes(self) -> list[str]:
        return [n for n, d in self.graph.nodes(data=True) if d.get("kind") == N_FILE]

    def get_class_nodes(self) -> list[str]:
        return [n for n, d in self.graph.nodes(data=True) if d.get("kind") == N_CLASS]

    def get_function_nodes(self) -> list[str]:
        return [n for n, d in self.graph.nodes(data=True) if d.get("kind") == N_FUNC]


# ---------------------------------------------------------------------------
# Module-level resolution helpers (used by Python path + tests)
# ---------------------------------------------------------------------------

def _module_to_rel(module_str: str, known_files: set[str]) -> Optional[str]:
    parts = module_str.split(".")
    for start in range(len(parts)):
        as_path = "/".join(parts[start:])
        for candidate in [f"{as_path}.py", f"{as_path}/__init__.py"]:
            if candidate in known_files:
                return candidate
    return None


def _resolve_class(base_name: str, current_rel: str,
                   rel_to_parser: dict) -> Optional[str]:
    simple = base_name.split(".")[-1]
    candidate = _class_id(current_rel, simple)
    if candidate in rel_to_parser[current_rel]._g:
        return candidate
    for rel, parser in rel_to_parser.items():
        if rel == current_rel:
            continue
        candidate = _class_id(rel, simple)
        if candidate in parser._g:
            return candidate
    return None


def _class_exists(name: str, rel: str, rel_to_parser: dict) -> bool:
    return _class_id(rel, name) in rel_to_parser[rel]._g