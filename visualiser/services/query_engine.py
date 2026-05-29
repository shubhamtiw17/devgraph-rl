from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()


@dataclass
class QueryResult:
    query:          str
    answer:         str
    relevant_nodes: List[str]        = field(default_factory=list)
    memory_hits:    List[str]        = field(default_factory=list)
    graph_context:  Dict[str, Any]   = field(default_factory=dict)
    embedder_used:  str              = "minilm"
    error:          Optional[str]    = None


def _search_graphs(
    query: str,
    repo_path: str,
    language: str,
) -> tuple[List[str], Dict[str, Any]]:
    try:
        from src.graphs.ast_graph import ASTGraph
        from src.graphs.dependency_graph import DependencyGraph
        from src.graphs.call_graph import CallGraph

        ast  = ASTGraph().build(repo_path, language=language)
        dep  = DependencyGraph().build(repo_path, language=language)
        call = CallGraph().build(repo_path, language=language)

        query_lower = query.lower()
        query_words = set(query_lower.split())

        relevant_nodes: List[str] = []
        for node_id, data in ast.graph.nodes(data=True):
            node_label = str(node_id).lower()
            overlap = sum(1 for w in query_words if len(w) > 3 and w in node_label)
            if overlap > 0:
                relevant_nodes.append(node_id)

        graph_context = {
            "ast_nodes":       ast.graph.number_of_nodes(),
            "ast_edges":       ast.graph.number_of_edges(),
            "dep_nodes":       dep.graph.number_of_nodes(),
            "dep_edges":       dep.graph.number_of_edges(),
            "call_nodes":      call.graph.number_of_nodes(),
            "call_edges":      call.graph.number_of_edges(),
            "language":        language,
            "top_files":       _top_files(ast),
            "top_functions":   _top_functions(call),
            "all_files":       _all_files(ast),
        }

        return relevant_nodes[:20], graph_context

    except Exception as e:
        return [], {"error": str(e), "ast_nodes": 0, "dep_nodes": 0, "call_nodes": 0}


def _top_files(ast) -> List[str]:
    files = [
        (n, ast.graph.degree(n))
        for n, d in ast.graph.nodes(data=True)
        if d.get("kind") == "file"
    ]
    files.sort(key=lambda x: x[1], reverse=True)
    return [f[0] for f in files[:8]]


def _all_files(ast) -> List[str]:
    return [
        n for n, d in ast.graph.nodes(data=True)
        if d.get("kind") == "file"
    ]


def _top_functions(call) -> List[str]:
    funcs = [(n, call.graph.in_degree(n)) for n in call.graph.nodes()]
    funcs.sort(key=lambda x: x[1], reverse=True)
    return [f[0] for f in funcs[:8] if f[1] > 0]


def _search_memory(query: str, repo_path: str, top_k: int = 3) -> List[str]:
    try:
        from src.memory import MemoryManager
        mm = MemoryManager()
        results = mm.retrieve(
            query=query,
            embedder_name="minilm",
            top_k=top_k,
            repo_path=repo_path,
        )
        return [r.text for r in results if r.score > 0.5]
    except Exception:
        return []


def _build_prompt(
    query: str,
    repo_name: str,
    language: str,
    graph_context: Dict[str, Any],
    relevant_nodes: List[str],
    memory_hits: List[str],
) -> str:
    nodes_str  = "\n".join(f"  - {n}" for n in relevant_nodes[:15]) or "  (none matched query keywords)"
    memory_str = "\n".join(f"  - {m[:200]}" for m in memory_hits) or "  (no past memory for this repo)"
    top_files  = "\n".join(f"  - {f}" for f in graph_context.get("top_files", []))
    top_funcs  = "\n".join(f"  - {f}" for f in graph_context.get("top_functions", []))
    all_files  = "\n".join(f"  - {f}" for f in graph_context.get("all_files", [])[:30])

    return f"""You are an expert software engineering assistant analyzing a real codebase.

REPOSITORY: {repo_name}
LANGUAGE: {language}

GRAPH STATISTICS:
  AST Graph:        {graph_context.get('ast_nodes', 0)} nodes, {graph_context.get('ast_edges', 0)} edges
  Dependency Graph: {graph_context.get('dep_nodes', 0)} nodes, {graph_context.get('dep_edges', 0)} edges
  Call Graph:       {graph_context.get('call_nodes', 0)} nodes, {graph_context.get('call_edges', 0)} edges

ALL FILES IN REPO:
{all_files}

TOP FILES BY CONNECTIVITY (most imported/depended on):
{top_files}

TOP FUNCTIONS BY CALL COUNT:
{top_funcs}

NODES MATCHING QUERY KEYWORDS:
{nodes_str}

PAST MEMORY (similar work done on this repo before):
{memory_str}

USER QUESTION: {query}

Answer based on the graph data above. Be specific — name actual files and functions you can see.
If a file name contains keywords related to the question, mention it.
Keep the answer concise, technical, and under 200 words."""


def _call_llm(prompt: str) -> str:
    try:
        from src.llm.router import get_router
        router = get_router()
        return router.complete(prompt)
    except Exception as e:
        return f"LLM unavailable: {e}"


def query_repo(
    query: str,
    repo_path: str,
    repo_name: str,
    language: str,
) -> QueryResult:
    if not query.strip():
        return QueryResult(query=query, answer="Please enter a question.", error="empty query")

    relevant_nodes, graph_context = _search_graphs(query, repo_path, language)
    memory_hits = _search_memory(query, repo_path)

    prompt = _build_prompt(
        query=query,
        repo_name=repo_name,
        language=language,
        graph_context=graph_context,
        relevant_nodes=relevant_nodes,
        memory_hits=memory_hits,
    )

    answer = _call_llm(prompt)

    return QueryResult(
        query=query,
        answer=answer,
        relevant_nodes=relevant_nodes,
        memory_hits=memory_hits,
        graph_context=graph_context,
    )
