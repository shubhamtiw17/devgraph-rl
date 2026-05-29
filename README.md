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

## Architecture
GitHub Repo URL
↓
Graph Intelligence (AST + Dependency + Call Graph)
↓
Multi-Agent Layer (Planner → Coder → Reviewer)
↓
Sandbox (Safe execution + pytest runner)
↓
Memory Layer (FAISS + 3 embedders)
↓
Reward Model (5-dimension scoring)
↓
RLHF Training Loop (Phase 8 — cloud GPU)

---

## Live Visualiser

Four interactive tabs running on FastAPI + D3.js:

| Tab | What it shows |
|---|---|
| **Graphs** | AST, Dependency, Call Graph — load any GitHub repo by URL |
| **Memory** | Store/search/compare across MiniLM, Gemini, Cohere embedders |
| **Sandbox** | Live code editor with safety validation + pytest runner |
| **Rewards** | Score agent outputs across 5 dimensions + D3 trend chart |

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM Router | Groq (llama-3.3-70b) + Gemini (gemini-2.0-flash) — auto-rotating |
| Graph Intelligence | NetworkX + tree-sitter (Python/JS/Java/C++) |
| Memory | FAISS + sentence-transformers (MiniLM) + Gemini + Cohere embedders |
| Sandbox | AST-based safety validator + isolated subprocess execution |
| Reward Model | 5-dimension scorer (correctness, quality, completion, graph alignment, memory) |
| Visualiser | FastAPI + D3.js force-directed graphs |
| Testing | pytest — 208+ tests, all mocked, CI green |
| Environment | WSL2 (Ubuntu) + Python 3.11 + free-tier APIs only |

---

## Completed Phases

| Phase | What was built | Tests |
|---|---|---|
| 1 | Repo scaffold — pyproject.toml, CI, src layout | — |
| 2 | LLM Router — Groq + Gemini, round-robin, auto-fallback | 8 |
| 3 | Agent layer — PlannerAgent, CodingAgent, BaseAgent ABC | 22 |
| 4 | Graph intelligence — AST, Dependency, Call graphs + D3 visualiser | 52 |
| 5 | Memory layer — FAISS, 3 embedders, semantic search, compare view | 45 |
| 6 | Sandbox — safety validator, subprocess executor, pytest runner | 42 |
| 7 | Reward model — 5-dimension scorer, history store, D3 trend chart | 39 |
| 8 | RLHF training loop | coming |

---

## Quick Start

### Prerequisites
- WSL2 (Ubuntu) or Linux
- Python 3.11
- API keys: Groq (free) + Gemini (free) + Cohere (free)

### Setup

```bash
git clone https://github.com/shubhamtiw17/devgraph-rl
cd devgraph-rl

python -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"
pip install -e ".[graphs]"

cp .env.example .env
# Add your API keys to .env
```

### Run the visualiser

```bash
uvicorn visualiser.main:app --reload --port 8000
```

Open **http://localhost:8000**

### Run tests

```bash
pytest tests/ -v
```

---

## Environment Variables

```dotenv
GROQ_API_KEY=gsk_...          # console.groq.com — free
GEMINI_API_KEY=AIza...        # aistudio.google.com — free
COHERE_API_KEY=...            # dashboard.cohere.com — free

DEFAULT_LLM_PROVIDER=auto
MAX_AGENT_ITERATIONS=10
AGENT_TIMEOUT_SECONDS=120
SANDBOX_TIMEOUT_SECONDS=30
REPO_WORKSPACE=/tmp/devgraph_repos
VECTOR_STORE_PATH=./data/vector_store
```

---

## Project Structure
devgraph-rl/
├── src/
│   ├── agents/          # PlannerAgent, CodingAgent, BaseAgent
│   ├── graphs/          # ASTGraph, DependencyGraph, CallGraph
│   ├── llm/             # LLMRouter — Groq + Gemini auto-rotation
│   ├── memory/          # FAISS vector store, 3 embedders, MemoryManager
│   ├── rewards/         # RewardModel, CodeQuality scorer, RewardStore
│   └── sandbox/         # Validator, Executor, TestRunner, Sandbox
├── visualiser/
│   ├── routers/         # FastAPI routers — graphs, memory, repo, sandbox, rewards
│   ├── services/        # GraphBuilder, RepoManager, QueryEngine
│   └── static/          # index.html — D3 visualiser (4 tabs)
├── tests/               # 208+ tests — all mocked, CI green
├── .github/workflows/   # CI — pytest on every push
└── pyproject.toml
---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Lazy router initialisation | Prevents API key check at import time |
| Three independent FAISS indexes | No dimension conflict between embedders |
| AST-based safety validation | No subprocess needed for validation step |
| Shallow git clone (depth=1) | 10x faster than full clone for large repos |
| Flat FAISS index (IndexFlatIP) | Exact search — correct at portfolio scale |
| ABC pattern for embedders | Swap MiniLM → CodeBERT without touching downstream code |
| Subprocess isolation for sandbox | Main server unaffected by crashes or timeouts |
| Newline-delimited JSON for reward store | Append-only, no file locking, easy to parse |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/graphs?language=python` | Build + return all 3 graphs |
| POST | `/api/repo/load` | Clone GitHub repo + build graphs |
| POST | `/api/repo/query` | Natural language query on repo |
| POST | `/api/memory/store` | Store memory in embedder index |
| POST | `/api/memory/search` | Semantic search in one index |
| POST | `/api/memory/search/compare` | Search all 3 embedders side by side |
| POST | `/api/memory/sync` | Re-encode memories across embedders |
| POST | `/api/sandbox/validate` | Validate code safety + syntax |
| POST | `/api/sandbox/run` | Execute code + optional tests |
| POST | `/api/rewards/score` | Score agent output (5 dimensions) |
| GET | `/api/rewards/stats` | Reward history statistics |
| GET | `/api/rewards/history` | Recent scored outputs |
| GET | `/api/rewards/top` | Top scoring outputs |

---

## Research Value

- **Embedder comparison** — same query across MiniLM (local), Gemini (768-dim), Cohere (384-dim) side by side
- **Graph-augmented retrieval** — memory search informed by real code structure
- **Multi-language graph analysis** — Python, JavaScript, Java, C++ with identical output schema
- **Reward signal design** — 5-dimension scoring combining execution results, static analysis, and LLM judgment

---

## License

MIT
