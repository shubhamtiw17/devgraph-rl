from __future__ import annotations

import time
from pathlib import Path

import networkx as nx

from src.graphs.ast_graph import ASTGraph
from src.graphs.dependency_graph import DependencyGraph
from src.graphs.call_graph import CallGraph

# ---------------------------------------------------------------------------
# Architecture metadata — shown in the UI info cards
# ---------------------------------------------------------------------------

GRAPH_META = {
    "ast": {
        "title":       "AST Graph",
        "subtitle":    "Structural Anatomy",
        "description": (
            "Parses every source file into an Abstract Syntax Tree and lifts "
            "structural entities — files, classes, and functions — into graph nodes. "
            "Edges encode containment (file→class, class→method), import relationships "
            "between files, and inheritance between classes."
        ),
        "how_it_works": (
            "Two-pass build: Pass 1 walks every file's AST with Python's stdlib `ast` "
            "module, collecting all node types. Pass 2 resolves cross-file import and "
            "inheritance edges now that all nodes are known. This ordering eliminates "
            "forward-reference problems."
        ),
        "advantages": [
            "Answers 'what is the shape of this codebase?' in O(1) after build",
            "No runtime execution — safe on untrusted or broken repos",
            "Inheritance chains expose architectural patterns at a glance",
            "Foundation for context-aware agent planning",
        ],
        "complexity": {
            "build":     "O(N) where N = total AST nodes across all files",
            "traversal": "O(V + E) standard DiGraph BFS/DFS",
        },
        "node_types": ["file", "class", "func"],
        "edge_types": ["contains", "imports", "inherits"],
    },
    "dependency": {
        "title":       "Dependency Graph",
        "subtitle":    "Module Coupling",
        "description": (
            "A file-level graph where edges represent import relationships, weighted "
            "by the number of symbols imported. Reveals coupling strength between modules "
            "and identifies the most-depended-upon files (high in-degree = high blast radius)."
        ),
        "how_it_works": (
            "tree-sitter parses real import syntax for each language — no regex. "
            "Python uses dotted module paths with src-prefix stripping. JavaScript "
            "resolves ES module paths and require() calls. Java matches scoped identifiers. "
            "C++ matches quoted #include (angle-bracket stdlib includes are ignored). "
            "Edge weight = number of symbols imported from the target file."
        ),
        "advantages": [
            "Weighted edges quantify coupling strength, not just existence",
            "High in-degree nodes are change-risk hotspots",
            "Transitive closure reveals indirect dependencies",
            "Guides refactoring: high-weight edges are tightest couplings",
        ],
        "complexity": {
            "build":     "O(F × I) where F = files, I = imports per file",
            "traversal": "O(V + E); PageRank-style analysis is O(V²) worst case",
        },
        "node_types": ["file"],
        "edge_types": ["imports (weighted)"],
    },
    "call": {
        "title":       "Call Graph",
        "subtitle":    "Execution Flow",
        "description": (
            "A function-level graph where edges represent invocation relationships "
            "between functions and methods. Edge weight is the number of call sites. "
            "Reveals execution paths, hot functions, and dead code candidates."
        ),
        "how_it_works": (
            "tree-sitter extracts function definitions (@def captures) and call-site "
            "identifiers (@call captures) per language. Captures are sorted by source "
            "byte offset and assigned to their enclosing function scope. "
            "Resolution prefers same-file definitions, then cross-file by name match. "
            "Edge weight accumulates across multiple call sites."
        ),
        "advantages": [
            "Traces execution paths from any entry point",
            "High in-degree functions are performance-critical hotspots",
            "Zero-in-degree non-entry functions are dead code candidates",
            "Essential for impact analysis: what breaks if I change function X?",
        ],
        "complexity": {
            "build":     "O(F × D) where F = files, D = definitions + calls per file",
            "traversal": "O(V + E); reachability from entry point is O(V + E) DFS",
        },
        "node_types": ["func"],
        "edge_types": ["calls (weighted)"],
    },
}

# ---------------------------------------------------------------------------
# Colour maps for D3
# ---------------------------------------------------------------------------

NODE_COLOURS = {
    "file":  "#4A9EE8",   # blue
    "class": "#F4913A",   # orange
    "func":  "#4DB87A",   # green
}

EDGE_COLOURS = {
    "contains": "#94A3B8",  # slate
    "imports":  "#4A9EE8",  # blue
    "inherits": "#EF4444",  # red
    "calls":    "#A855F7",  # purple
}


# ---------------------------------------------------------------------------
# Serialiser
# ---------------------------------------------------------------------------

def _serialise_graph(g: nx.DiGraph) -> dict:
    nodes = []
    for node_id, data in g.nodes(data=True):
        kind = data.get("kind", "file")
        nodes.append({
            "id":     node_id,
            "label":  data.get("name") or data.get("path") or node_id,
            "kind":   kind,
            "path":   data.get("path", ""),
            "lineno": data.get("lineno"),
            "color":  NODE_COLOURS.get(kind, "#94A3B8"),
            **{k: v for k, v in data.items()
               if k not in ("kind", "name", "path", "lineno")},
        })

    edges = []
    for u, v, data in g.edges(data=True):
        kind = data.get("kind", "")
        edges.append({
            "source": u,
            "target": v,
            "kind":   kind,
            "weight": data.get("weight", 1),
            "color":  EDGE_COLOURS.get(kind, "#94A3B8"),
        })

    return {"nodes": nodes, "edges": edges}


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_graphs(repo_path: str, language: str) -> dict:
    results = {}

    # AST graph — Python uses stdlib ast, other languages use tree-sitter.
    # ASTGraph.build() routes internally based on language.
    t0 = time.perf_counter()
    ag = ASTGraph().build(repo_path, language=language)
    ast_ms = (time.perf_counter() - t0) * 1000

    # Dependency graph (tree-sitter, all languages)
    dg = DependencyGraph().build(repo_path, language=language)

    # Call graph (tree-sitter, all languages)
    cg = CallGraph().build(repo_path, language=language)

    for key, graph_obj, ms in [
        ("ast",        ag,  ast_ms),
        ("dependency", dg,  dg.build_time_ms),
        ("call",       cg,  cg.build_time_ms),
    ]:
        g    = graph_obj.graph
        meta = GRAPH_META[key]

        results[key] = {
            **meta,
            "graph": _serialise_graph(g),
            "note": None,
            "stats": {
                "nodes":        g.number_of_nodes(),
                "edges":        g.number_of_edges(),
                "build_time_ms": round(ms, 2),
            },
        }

    return results