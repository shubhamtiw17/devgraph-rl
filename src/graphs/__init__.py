from src.graphs.ast_graph import ASTGraph
from src.graphs.dependency_graph import DependencyGraph, GraphBuildError
from src.graphs.call_graph import CallGraph

__all__ = [
    "ASTGraph",
    "DependencyGraph",
    "CallGraph",
    "GraphBuildError",
    "build_full_graph",
]


def build_full_graph(repo_path: str, language: str = "python") -> dict:
    return {
        "ast":        ASTGraph().build(repo_path),
        "dependency": DependencyGraph().build(repo_path, language=language),
        "call":       CallGraph().build(repo_path, language=language),
    }