import pytest
import networkx as nx

from src.graphs.ast_graph import (
    ASTGraph,
    _module_to_rel,
    _file_id,
    _class_id,
    _func_id,
    N_FILE, N_CLASS, N_FUNC,
    E_CONTAINS, E_IMPORTS, E_INHERITS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write(tmp_path, rel: str, source: str):
    """Write *source* to tmp_path / rel, creating parent dirs as needed."""
    target = tmp_path / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source, encoding="utf-8")


def build(tmp_path) -> ASTGraph:
    return ASTGraph().build(tmp_path)


# ---------------------------------------------------------------------------
# 1. Node ID helpers
# ---------------------------------------------------------------------------

class TestNodeIdHelpers:
    def test_file_id(self):
        assert _file_id("src/foo.py") == "file:src/foo.py"

    def test_class_id(self):
        assert _class_id("src/foo.py", "Bar") == "class:src/foo.py:Bar"

    def test_func_id(self):
        assert _func_id("src/foo.py", "Bar.baz") == "func:src/foo.py:Bar.baz"


# ---------------------------------------------------------------------------
# 2. _module_to_rel
# ---------------------------------------------------------------------------

class TestModuleToRel:
    KNOWN = {
        "agents/base_agent.py",
        "agents/__init__.py",
        "llm/router.py",
    }

    def test_plain_module(self):
        assert _module_to_rel("agents.base_agent", self.KNOWN) == "agents/base_agent.py"

    def test_src_prefixed_module(self):
        # src.agents.base_agent should resolve by stripping the src. prefix
        assert _module_to_rel("src.agents.base_agent", self.KNOWN) == "agents/base_agent.py"

    def test_package_resolves_to_init(self):
        assert _module_to_rel("agents", self.KNOWN) == "agents/__init__.py"

    def test_src_prefixed_package(self):
        assert _module_to_rel("src.agents", self.KNOWN) == "agents/__init__.py"

    def test_external_library_returns_none(self):
        assert _module_to_rel("networkx", self.KNOWN) is None

    def test_unknown_module_returns_none(self):
        assert _module_to_rel("totally.unknown.module", self.KNOWN) is None


# ---------------------------------------------------------------------------
# 3. Empty / trivial repo
# ---------------------------------------------------------------------------

class TestEmptyRepo:
    def test_empty_directory(self, tmp_path):
        ag = build(tmp_path)
        assert ag.graph.number_of_nodes() == 0
        assert ag.graph.number_of_edges() == 0

    def test_single_empty_file(self, tmp_path):
        write(tmp_path, "module.py", "")
        ag = build(tmp_path)
        assert "file:module.py" in ag.graph

    def test_nonexistent_path_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ASTGraph().build(tmp_path / "does_not_exist")


# ---------------------------------------------------------------------------
# 4. File nodes
# ---------------------------------------------------------------------------

class TestFileNodes:
    def test_file_node_added(self, tmp_path):
        write(tmp_path, "foo.py", "x = 1")
        ag = build(tmp_path)
        assert _file_id("foo.py") in ag.graph

    def test_file_node_kind_attribute(self, tmp_path):
        write(tmp_path, "foo.py", "")
        ag = build(tmp_path)
        assert ag.graph.nodes[_file_id("foo.py")]["kind"] == N_FILE

    def test_file_node_path_attribute(self, tmp_path):
        write(tmp_path, "sub/bar.py", "")
        ag = build(tmp_path)
        node = ag.graph.nodes[_file_id("sub/bar.py")]
        assert node["path"] == "sub/bar.py"

    def test_get_file_nodes_returns_all(self, tmp_path):
        write(tmp_path, "a.py", "")
        write(tmp_path, "b.py", "")
        ag = build(tmp_path)
        file_nodes = ag.get_file_nodes()
        assert _file_id("a.py") in file_nodes
        assert _file_id("b.py") in file_nodes

    def test_non_py_files_ignored(self, tmp_path):
        write(tmp_path, "readme.md", "# hello")
        (tmp_path / "data.json").write_text("{}", encoding="utf-8")
        ag = build(tmp_path)
        assert ag.graph.number_of_nodes() == 0


# ---------------------------------------------------------------------------
# 5. Class nodes
# ---------------------------------------------------------------------------

class TestClassNodes:
    SOURCE = """\
class Animal:
    pass

class Dog(Animal):
    pass
"""

    def test_class_nodes_added(self, tmp_path):
        write(tmp_path, "pets.py", self.SOURCE)
        ag = build(tmp_path)
        assert _class_id("pets.py", "Animal") in ag.graph
        assert _class_id("pets.py", "Dog") in ag.graph

    def test_class_node_kind(self, tmp_path):
        write(tmp_path, "pets.py", self.SOURCE)
        ag = build(tmp_path)
        assert ag.graph.nodes[_class_id("pets.py", "Animal")]["kind"] == N_CLASS

    def test_class_node_name_attribute(self, tmp_path):
        write(tmp_path, "pets.py", self.SOURCE)
        ag = build(tmp_path)
        assert ag.graph.nodes[_class_id("pets.py", "Dog")]["name"] == "Dog"

    def test_class_node_lineno(self, tmp_path):
        write(tmp_path, "pets.py", self.SOURCE)
        ag = build(tmp_path)
        assert ag.graph.nodes[_class_id("pets.py", "Animal")]["lineno"] == 1
        assert ag.graph.nodes[_class_id("pets.py", "Dog")]["lineno"] == 4

    def test_get_class_nodes(self, tmp_path):
        write(tmp_path, "pets.py", self.SOURCE)
        ag = build(tmp_path)
        class_nodes = ag.get_class_nodes()
        assert _class_id("pets.py", "Animal") in class_nodes
        assert _class_id("pets.py", "Dog") in class_nodes


# ---------------------------------------------------------------------------
# 6. Function nodes
# ---------------------------------------------------------------------------

class TestFunctionNodes:
    SOURCE = """\
def standalone():
    pass

class MyClass:
    def method(self):
        pass

    async def async_method(self):
        pass
"""

    def test_top_level_function_node(self, tmp_path):
        write(tmp_path, "funcs.py", self.SOURCE)
        ag = build(tmp_path)
        assert _func_id("funcs.py", "standalone") in ag.graph

    def test_method_qualified_name(self, tmp_path):
        write(tmp_path, "funcs.py", self.SOURCE)
        ag = build(tmp_path)
        assert _func_id("funcs.py", "MyClass.method") in ag.graph

    def test_async_method_qualified_name(self, tmp_path):
        write(tmp_path, "funcs.py", self.SOURCE)
        ag = build(tmp_path)
        assert _func_id("funcs.py", "MyClass.async_method") in ag.graph

    def test_function_node_kind(self, tmp_path):
        write(tmp_path, "funcs.py", self.SOURCE)
        ag = build(tmp_path)
        assert ag.graph.nodes[_func_id("funcs.py", "standalone")]["kind"] == N_FUNC

    def test_function_node_name_attribute(self, tmp_path):
        write(tmp_path, "funcs.py", self.SOURCE)
        ag = build(tmp_path)
        node = ag.graph.nodes[_func_id("funcs.py", "MyClass.method")]
        assert node["name"] == "method"
        assert node["qualname"] == "MyClass.method"

    def test_get_function_nodes(self, tmp_path):
        write(tmp_path, "funcs.py", self.SOURCE)
        ag = build(tmp_path)
        func_nodes = ag.get_function_nodes()
        assert _func_id("funcs.py", "standalone") in func_nodes
        assert _func_id("funcs.py", "MyClass.method") in func_nodes


# ---------------------------------------------------------------------------
# 7. Contains edges
# ---------------------------------------------------------------------------

class TestContainsEdges:
    SOURCE = """\
def top_func():
    pass

class Top:
    def top_method(self):
        pass

    class Inner:
        def inner_method(self):
            pass
"""

    def _edges_of_kind(self, ag, kind):
        return {(u, v) for u, v, d in ag.graph.edges(data=True) if d["kind"] == kind}

    def test_file_contains_top_level_function(self, tmp_path):
        write(tmp_path, "t.py", self.SOURCE)
        ag = build(tmp_path)
        assert (_file_id("t.py"), _func_id("t.py", "top_func")) in self._edges_of_kind(ag, E_CONTAINS)

    def test_file_contains_top_level_class(self, tmp_path):
        write(tmp_path, "t.py", self.SOURCE)
        ag = build(tmp_path)
        assert (_file_id("t.py"), _class_id("t.py", "Top")) in self._edges_of_kind(ag, E_CONTAINS)

    def test_class_contains_method(self, tmp_path):
        write(tmp_path, "t.py", self.SOURCE)
        ag = build(tmp_path)
        assert (_class_id("t.py", "Top"), _func_id("t.py", "Top.top_method")) in self._edges_of_kind(ag, E_CONTAINS)

    def test_outer_class_contains_inner_class(self, tmp_path):
        write(tmp_path, "t.py", self.SOURCE)
        ag = build(tmp_path)
        assert (_class_id("t.py", "Top"), _class_id("t.py", "Inner")) in self._edges_of_kind(ag, E_CONTAINS)

    def test_inner_class_contains_its_method(self, tmp_path):
        write(tmp_path, "t.py", self.SOURCE)
        ag = build(tmp_path)
        assert (_class_id("t.py", "Inner"), _func_id("t.py", "Inner.inner_method")) in self._edges_of_kind(ag, E_CONTAINS)


# ---------------------------------------------------------------------------
# 8. Import edges
# ---------------------------------------------------------------------------

class TestImportEdges:
    def test_plain_import_edge(self, tmp_path):
        write(tmp_path, "utils.py", "def helper(): pass")
        write(tmp_path, "main.py", "import utils")
        ag = build(tmp_path)
        assert ag.graph.has_edge(_file_id("main.py"), _file_id("utils.py"))
        kind = ag.graph[_file_id("main.py")][_file_id("utils.py")]["kind"]
        assert kind == E_IMPORTS

    def test_from_import_edge(self, tmp_path):
        write(tmp_path, "utils.py", "def helper(): pass")
        write(tmp_path, "main.py", "from utils import helper")
        ag = build(tmp_path)
        assert ag.graph.has_edge(_file_id("main.py"), _file_id("utils.py"))

    def test_no_self_loop_on_import(self, tmp_path):
        write(tmp_path, "utils.py", "from utils import something")
        ag = build(tmp_path)
        assert not ag.graph.has_edge(_file_id("utils.py"), _file_id("utils.py"))

    def test_external_import_produces_no_edge(self, tmp_path):
        write(tmp_path, "main.py", "import os\nimport networkx")
        ag = build(tmp_path)
        # Only one node (the file itself), no import edges
        assert ag.graph.number_of_edges() == 0

    def test_get_dependencies_returns_imported_files(self, tmp_path):
        write(tmp_path, "utils.py", "")
        write(tmp_path, "main.py", "from utils import something")
        ag = build(tmp_path)
        deps = ag.get_dependencies(_file_id("main.py"))
        assert _file_id("utils.py") in deps

    def test_src_prefixed_import_resolves(self, tmp_path):
        # Mirrors DevGraph-RL's own import style: src.agents.base_agent
        write(tmp_path, "agents/base_agent.py", "class BaseAgent: pass")
        write(tmp_path, "agents/coding.py", "from src.agents.base_agent import BaseAgent")
        ag = build(tmp_path)
        assert ag.graph.has_edge(
            _file_id("agents/coding.py"),
            _file_id("agents/base_agent.py"),
        )


# ---------------------------------------------------------------------------
# 9. Inheritance edges
# ---------------------------------------------------------------------------

class TestInheritanceEdges:
    def test_same_file_inheritance(self, tmp_path):
        write(tmp_path, "pets.py", "class Animal: pass\nclass Dog(Animal): pass")
        ag = build(tmp_path)
        assert ag.graph.has_edge(
            _class_id("pets.py", "Dog"),
            _class_id("pets.py", "Animal"),
        )
        kind = ag.graph[_class_id("pets.py", "Dog")][_class_id("pets.py", "Animal")]["kind"]
        assert kind == E_INHERITS

    def test_cross_file_inheritance(self, tmp_path):
        write(tmp_path, "base.py", "class Base: pass")
        write(tmp_path, "child.py", "from base import Base\nclass Child(Base): pass")
        ag = build(tmp_path)
        assert ag.graph.has_edge(
            _class_id("child.py", "Child"),
            _class_id("base.py", "Base"),
        )

    def test_unresolvable_base_produces_no_edge(self, tmp_path):
        # Inheriting from an external class (e.g. pydantic.BaseModel) should
        # not crash and should not add a spurious node or edge.
        write(tmp_path, "model.py", "class MyModel(BaseModel): pass")
        ag = build(tmp_path)
        assert ag.graph.number_of_nodes() == 2   # file + class
        inherit_edges = [
            (u, v) for u, v, d in ag.graph.edges(data=True)
            if d["kind"] == E_INHERITS
        ]
        assert inherit_edges == []

    def test_get_class_hierarchy(self, tmp_path):
        write(tmp_path, "pets.py", "class Animal: pass\nclass Dog(Animal): pass")
        ag = build(tmp_path)
        parents = ag.get_class_hierarchy(_class_id("pets.py", "Dog"))
        assert _class_id("pets.py", "Animal") in parents

    def test_multiple_bases(self, tmp_path):
        src = "class A: pass\nclass B: pass\nclass C(A, B): pass"
        write(tmp_path, "multi.py", src)
        ag = build(tmp_path)
        parents = ag.get_class_hierarchy(_class_id("multi.py", "C"))
        assert _class_id("multi.py", "A") in parents
        assert _class_id("multi.py", "B") in parents


# ---------------------------------------------------------------------------
# 10. build() is idempotent (calling twice resets the graph)
# ---------------------------------------------------------------------------

class TestBuildIdempotent:
    def test_rebuild_clears_previous_graph(self, tmp_path):
        write(tmp_path, "a.py", "class A: pass")
        ag = ASTGraph()
        ag.build(tmp_path)
        first_count = ag.graph.number_of_nodes()

        # Add a second file and rebuild
        write(tmp_path, "b.py", "class B: pass")
        ag.build(tmp_path)
        second_count = ag.graph.number_of_nodes()

        assert second_count > first_count

    def test_method_chaining(self, tmp_path):
        write(tmp_path, "a.py", "")
        result = ASTGraph().build(tmp_path)
        assert isinstance(result, ASTGraph)


# ---------------------------------------------------------------------------
# 11. Syntax error resilience
# ---------------------------------------------------------------------------

class TestSyntaxErrorResilience:
    def test_bad_file_skipped_good_file_parsed(self, tmp_path):
        write(tmp_path, "good.py", "class Good: pass")
        write(tmp_path, "bad.py", "def (broken syntax!!!")
        ag = build(tmp_path)
        # good.py should still be in the graph
        assert _file_id("good.py") in ag.graph
        # bad.py should be absent (skipped, not crashed)
        assert _file_id("bad.py") not in ag.graph


# ---------------------------------------------------------------------------
# 12. Integration: structure matching DevGraph-RL's own src/ tree
# ---------------------------------------------------------------------------

class TestIntegrationShape:
    """
    Writes a minimal replica of DevGraph-RL's agent layer and asserts the
    exact graph structure we verified manually against the real src/ tree.
    """

    def setup_repo(self, tmp_path):
        write(tmp_path, "llm/__init__.py", "")
        write(tmp_path, "llm/router.py", """\
from enum import Enum
class Provider(Enum):
    ANTHROPIC = "anthropic"
class LLMRouter:
    def get_completion(self): pass
""")
        write(tmp_path, "agents/__init__.py", "from src.agents.base_agent import BaseAgent")
        write(tmp_path, "agents/base_agent.py", """\
from src.llm.router import LLMRouter
class BaseAgent:
    def build_prompt(self): pass
    def parse_response(self): pass
""")
        write(tmp_path, "agents/coding.py", """\
from src.agents.base_agent import BaseAgent
class CodingAgent(BaseAgent):
    def build_prompt(self): pass
    def parse_response(self): pass
""")

    def test_all_file_nodes_present(self, tmp_path):
        self.setup_repo(tmp_path)
        ag = build(tmp_path)
        for rel in ["llm/__init__.py", "llm/router.py", "agents/__init__.py",
                    "agents/base_agent.py", "agents/coding.py"]:
            assert _file_id(rel) in ag.graph

    def test_inheritance_resolved(self, tmp_path):
        self.setup_repo(tmp_path)
        ag = build(tmp_path)
        assert ag.graph.has_edge(
            _class_id("agents/coding.py", "CodingAgent"),
            _class_id("agents/base_agent.py", "BaseAgent"),
        )

    def test_import_chain(self, tmp_path):
        self.setup_repo(tmp_path)
        ag = build(tmp_path)
        # agents/coding.py -> agents/base_agent.py
        assert ag.graph.has_edge(
            _file_id("agents/coding.py"),
            _file_id("agents/base_agent.py"),
        )
        # agents/base_agent.py -> llm/router.py
        assert ag.graph.has_edge(
            _file_id("agents/base_agent.py"),
            _file_id("llm/router.py"),
        )

    def test_method_qualnames_in_integration(self, tmp_path):
        self.setup_repo(tmp_path)
        ag = build(tmp_path)
        func_nodes = ag.get_function_nodes()
        assert _func_id("agents/base_agent.py", "BaseAgent.build_prompt") in func_nodes
        assert _func_id("agents/coding.py", "CodingAgent.parse_response") in func_nodes

    def test_graph_is_directed(self, tmp_path):
        self.setup_repo(tmp_path)
        ag = build(tmp_path)
        assert isinstance(ag.graph, nx.DiGraph)