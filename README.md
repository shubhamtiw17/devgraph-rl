# DevGraph-RL

> **Graph-Augmented RLHF Multi-Agent Autonomous Software Engineering System**

An autonomous AI engineering platform that understands codebases, plans software
modifications, writes and refactors code, runs tests, and self-improves via
reinforcement learning coordinated across specialist agents.

---

## What is this?

DevGraph-RL is a research and portfolio project combining four areas of modern AI engineering:

- **Multi-agent LLMs** — specialist agents (planner, coder, reviewer, architect) that each handle one concern and hand off structured results
- **Graph intelligence** — the codebase is parsed into three directed graphs so agents can reason about structure, coupling, and execution flow rather than reading raw text
- **Reward modelling + RLHF** — agents are scored on their outputs; scores feed a training loop that improves models over time
- **Heuristic search** — a search layer explores the space of possible code modifications and picks the most promising path before committing

---

## Environment

| Concern | Choice |
|---|---|
| OS | Windows 11 + WSL2 (Ubuntu) — all work done inside WSL2 |
| Python | 3.11.15 inside a venv at `~/devgraph-rl/.venv` |
| Editor | VS Code connected to WSL2 |
| GPU | Intel UHD 630 (no CUDA) — API for agents, free cloud GPU (Colab/Kaggle) for training |
| LLM providers | Claude (Anthropic), Groq (llama3-70b), Gemini (gemini-1.5-flash) — free tier, auto-rotating |
| CI | GitHub Actions — pytest runs on every push |

---

## Project structure

```
devgraph-rl/
├── src/
│   ├── agents/                  # Specialist agent layer
│   │   ├── base_agent.py            # ABC + shared dataclasses
│   │   ├── planner.py               # Decomposes tasks into subtasks
│   │   └── coding.py                # Generates / refactors code
│   ├── graphs/                  # Graph intelligence layer
│   │   ├── ast_graph.py             # Structural graph (file/class/func nodes)
│   │   ├── dependency_graph.py      # Module coupling graph (weighted imports)
│   │   ├── call_graph.py            # Execution flow graph (function calls)
│   │   └── __init__.py              # Clean exports + build_full_graph()
│   ├── llm/                     # LLM router
│   │   └── router.py                # Round-robin + fallback across providers
│   ├── memory/                  # (upcoming)
│   ├── rewards/                 # (upcoming)
│   ├── sandbox/                 # (upcoming)
│   └── search/                  # (upcoming)
├── visualiser/                  # FastAPI graph visualiser
│   ├── main.py                      # FastAPI app entry point
│   ├── routers/graphs.py            # GET /api/graphs?language=
│   ├── services/graph_builder.py    # Builds + serialises all 3 graphs
│   ├── static/index.html            # D3 three-panel SPA
│   ├── samples/                     # Synthetic repos per language
│   │   ├── python/                  # Shape hierarchy in Python
│   │   ├── javascript/              # Same in JavaScript (ES modules)
│   │   ├── java/                    # Same in Java (packages)
│   │   └── cpp/                     # Same in C++ (headers + source)
│   └── requirements.txt
├── tests/
│   ├── test_router.py               # 8 tests
│   ├── test_agents.py               # 22 tests
│   └── test_graphs.py               # 52 tests
├── pyproject.toml
└── .github/workflows/ci.yml
```

---

## Shipped phases

### Phase 1 — Repo scaffold

Project skeleton: `pyproject.toml` with `setuptools.build_meta`, optional dependency groups (`dev`, `graphs`, `training`), `.env.example`, `.gitignore`, and GitHub Actions CI running pytest on every push.

The `training` extra (`torch`, `transformers`, `trl`) is excluded from local install — runs on free cloud GPU to avoid the CUDA dependency.

---

### Phase 2 — LLM router

**File:** `src/llm/router.py`

Provider-agnostic LLM client in front of Claude, Groq, and Gemini. All agents go through this — never a provider SDK directly.

```
┌─────────────────────────────────────────────┐
│                  LLMRouter                  │
│  providers: [ANTHROPIC, GROQ, GEMINI]       │
│  strategy:  round-robin cycle               │
│  on failure → next provider (up to 3 tries) │
└──────────┬──────────────────────────────────┘
           │
    ┌──────▼──────┐   ┌──────────────┐   ┌─────────────────┐
    │  Anthropic  │   │     Groq     │   │     Gemini      │
    │  Claude 3.5 │   │ llama3-70b   │   │ gemini-1.5-flash│
    └─────────────┘   └──────────────┘   └─────────────────┘
```

**Key decisions:**
- `Provider` is an `Enum`, `LLMConfig` is a `dataclass` — strongly typed, no magic strings
- `get_router()` returns a module-level singleton
- Fallback is silent to callers — agents receive a result or exception, never a provider detail
- All 8 tests mock the provider SDKs — CI needs no API keys

---

### Phase 3 — Agent layer

**Files:** `src/agents/base_agent.py`, `src/agents/planner.py`, `src/agents/coding.py`

```
                    ┌──────────────────────┐
                    │      BaseAgent (ABC) │
                    │  build_prompt()      │ ← abstract
                    │  parse_response()    │ ← abstract
                    │  run()               │
                    │  router  (lazy init) │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
    ┌─────────▼──────┐ ┌───────▼──────┐  (more agents
    │  PlannerAgent  │ │  CodingAgent │   in future phases)
    │ → PlannerOutput│ │ → code + lang│
    │   (Pydantic)   │ │              │
    └────────────────┘ └──────────────┘
```

`AgentContext` carries everything an agent needs:

| Field | Purpose |
|---|---|
| `repo_path` | Root of the repo being modified |
| `task` | Natural language task description |
| `history` | Prior agent outputs (filtered per agent) |
| `language` | Target language: python, java, shell, cpp, html, markdown |
| `target_file` | Precise file to modify (optional) |
| `constraints` | Hard constraints the output must satisfy |
| `metadata` | Arbitrary key-value bag |

**Key decisions:**
- **Lazy router init** — router initialises on first LLM call; mock injectable for tests via `agent.router = mock`
- **Filtered history** — each agent declares `RELEVANT_HISTORY`; history capped at 3 entries, truncated at 300 chars
- **Pydantic on PlannerAgent** — `PlannerOutput` model gives field-level validation errors instead of silent dict failures
- **Graceful degradation** — agents never raise; always return `AgentResult(success=False, error=...)` on failure

---

### Phase 4 — Graph intelligence + visualiser

**Files:** `src/graphs/`, `visualiser/`

Three graphs, each answering a different question about the codebase:

```
  Source files
       │
       ├──► AST Graph          "What exists and how is it structured?"
       │     Nodes: file, class, func
       │     Edges: contains, imports, inherits
       │
       ├──► Dependency Graph   "What depends on what, and how tightly?"
       │     Nodes: file
       │     Edges: imports (weighted by symbol count)
       │
       └──► Call Graph         "What calls what at runtime?"
             Nodes: func
             Edges: calls (weighted by call-site count)
```

#### AST Graph (`ast_graph.py`)

Parses every source file into structural nodes. Python uses stdlib `ast`. JavaScript, Java, and C++ use tree-sitter — same output schema across all four languages.

```
  Pass 1 — collect nodes (all files first, no cross-file edges yet)
  ┌──────────────────────────────────────────────────────┐
  │  file:shapes/base.py                  [kind=file]   │
  │  class:shapes/base.py:Shape           [kind=class]  │
  │  func:shapes/base.py:Shape.area       [kind=func]   │
  │  func:shapes/base.py:Shape.describe   [kind=func]   │
  └──────────────────────────────────────────────────────┘
       │
  Pass 2 — resolve cross-file edges
  ┌──────────────────────────────────────────────────────┐
  │  contains  : file→class, class→method               │
  │  imports   : file→file  (intra-repo only)           │
  │  inherits  : class→class                            │
  └──────────────────────────────────────────────────────┘
```

Node ID format:

| Type | Format | Example |
|---|---|---|
| File | `file:<rel_path>` | `file:agents/planner.py` |
| Class | `class:<rel_path>:<Name>` | `class:agents/planner.py:PlannerAgent` |
| Function | `func:<rel_path>:<qualname>` | `func:agents/base_agent.py:BaseAgent.run` |

Qualified names (`ClassName.method`) prevent collisions between same-named methods in different classes.

Results on the real `src/` tree:
```
  58 nodes  ·  59 edges
  Import chain:  agents/__init__.py → base_agent.py → llm/router.py
  Inheritance:   CodingAgent → BaseAgent,  PlannerAgent → BaseAgent
```

#### Dependency Graph (`dependency_graph.py`)

File-level coupling graph. Edge weight = symbol count imported.

```
  main.py ──(weight=1)──► calculator.py
  shapes.py ──(weight=2)──► utils.py        ← imports 2 symbols
  calculator.py ──(weight=3)──► shapes.py   ← imports 3 classes
```

High in-degree = high blast radius. `most_depended_upon()` surfaces hotspots instantly.

Language | Import syntax captured
---|---
Python | `import x`, `from x import a, b, c` (weight = symbol count)
JavaScript | `import ... from '...'`, `require('...')`
Java | `import com.example.Foo`
C++ | `#include "local.h"` (quoted only; angle-bracket stdlib skipped)

#### Call Graph (`call_graph.py`)

Function-level execution graph. Edge weight = number of call sites.

```
  main → AreaCalculator.__init__
  report → describe → area → round_result
  add_circle → Circle.__init__ → validate_positive
```

High in-degree functions are hotspots. Zero in-degree non-entry functions are dead code candidates.

C++ captures both bare definitions (`void foo()`) and qualified definitions (`Circle::area()`) via an extended tree-sitter query.

JavaScript captures arrow functions assigned to variables (`const area = (r) => ...`) using the variable name as the function name.

#### Visualiser

FastAPI backend + D3 frontend. Three panels side by side, independent per panel.

```
  GET /api/graphs?language=python
  →  {
       "ast":        { nodes, edges, stats, architecture card data },
       "dependency": { nodes, edges, stats, architecture card data },
       "call":       { nodes, edges, stats, architecture card data }
     }
```

```
┌─────────────────────────────────────────────────────────┐
│  DevGraph-RL Visualiser    [Python] [JavaScript] [Java] [C++] │
├──────────────────┬──────────────────┬────────────────────┤
│  AST Graph       │ Dependency Graph │ Call Graph         │
│  31n · 35e       │ 5n · 6e          │ 17n · 17e          │
│  3.4ms build     │ 4.7ms build      │ 9.9ms build        │
│                  │                  │                    │
│  [D3 canvas]     │ [D3 canvas]      │ [D3 canvas]        │
│                  │                  │                    │
│  ▼ Architecture card (expandable)                        │
│    What it is · How it works · Advantages · Complexity   │
└──────────────────┴──────────────────┴────────────────────┘
  ● File  ● Class  ● Function  — Contains  — Imports  — Inherits  — Calls
```

Node colours: 🔵 file · 🟠 class · 🟢 function  
Edge colours: grey=contains · blue=imports · red=inherits · purple=calls

Run the visualiser:

```bash
pip install -e ".[graphs]"
uvicorn visualiser.main:app --reload --port 8000
# open http://localhost:8000
```

**Key decisions:**
- **tree-sitter pinned to 0.21.3** — `tree-sitter-languages 1.10.2` was compiled against 0.21.x; 0.22+ breaks the Language constructor
- **Parser() + set_language()** — avoids the deprecated `Language(path, name)` constructor
- **Hard-fail on parse error** — sample repos are hand-written and must parse cleanly; no silent skipping unlike the Python AST path
- **Two-pass build on all three graphs** — eliminates forward-reference ordering problems
- **`src.` prefix stripping** — import resolver drops leading path components until a known file matches
- **D3 force-directed layout** — highly connected nodes cluster naturally toward the centre; no manual positioning needed

**Graph stats across sample repos (same Shape hierarchy in all 4 languages):**

| Language | AST | Dependency | Call |
|---|---|---|---|
| Python | 31n / 35e | 5n / 6e | 17n / 17e |
| JavaScript | 31n / 35e | 5n / 6e | 20n / 14e |
| Java | 20n / 23e | 5n / 7e | 10n / 8e |
| C++ | 20n / 23e | 7n / 8e | 9n / 5e |

---

## Test strategy

All tests mock the LLM router — no real API calls, no keys needed in CI.

| Suite | Tests | What it covers |
|---|---|---|
| `test_router.py` | 8 | Provider enum, round-robin, fallback, singleton |
| `test_agents.py` | 22 | Context dataclasses, prompt construction, output parsing, resilience |
| `test_graphs.py` | 52 | Node IDs, edge types, attributes, resolution logic, resilience, integration |
| **Total** | **82** | |

---

## Installation

```bash
git clone https://github.com/<you>/devgraph-rl
cd devgraph-rl
python -m venv .venv && source .venv/bin/activate

# Core + dev tools
pip install -e ".[dev]"

# Graph intelligence + visualiser (adds tree-sitter-languages, fastapi, uvicorn)
pip install -e ".[graphs]"

cp .env.example .env   # add your API keys
pytest                 # all 82 tests, no keys needed
```

---

## Roadmap

| Phase | Status | Description |
|---|---|---|
| 1 — Scaffold | ✅ Done | Repo structure, CI, dependency management |
| 2 — LLM router | ✅ Done | Multi-provider routing with fallback |
| 3 — Agent layer | ✅ Done | Planner + Coder agents on shared base |
| 4 — Graph intelligence | ✅ Done | Three graphs (AST, dependency, call) across Python/JS/Java/C++ + FastAPI visualiser |
| 5 — Memory layer | 🔜 Next | Vector store for agent recall across runs |
| 6 — Sandbox | ⬜ Planned | Safe code execution and test running |
| 7 — Reward modelling | ⬜ Planned | Scoring agent outputs for RLHF |
| 8 — Search | ⬜ Planned | Heuristic search over modification space |
| 9 — RLHF training | ⬜ Planned | Fine-tuning loop on cloud GPU |