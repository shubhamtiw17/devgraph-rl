# Design Decisions

**Why we built DevGraph-RL the way we did**

This document covers the architectural rationale behind every major component — what we chose, why, what we gave up, and what we could have done differently.

---

## Table of Contents

1. [LLM Router](#1-llm-router)
2. [Graph Intelligence Layer](#2-graph-intelligence-layer)
3. [Memory Layer](#3-memory-layer)
4. [Sandbox](#4-sandbox)
5. [Reward Model](#5-reward-model)
6. [RLHF Training Pipeline](#6-rlhf-training-pipeline)
7. [Assistant Engine](#7-assistant-engine)
8. [Frontend Architecture](#8-frontend-architecture)
9. [Overall System Architecture](#9-overall-system-architecture)

---

## 1. LLM Router

### What we chose
A custom router that auto-rotates between **Groq** (llama-3.3-70b-versatile) and **Google Gemini** (gemini-2.0-flash), with configurable retry logic and automatic fallback.

### Why
- **Groq** offers the fastest inference of any free-tier provider (~500 tokens/sec on llama-3.3-70b) — critical for the assistant tab feeling responsive
- **Gemini** gives a capable fallback with a generous free quota and strong code understanding
- Dual-provider rotation means the system keeps working when one provider rate-limits or goes down
- A custom router costs ~80 lines of Python vs paying for an orchestration layer

### Advantages
- Zero vendor lock-in — swap providers by changing one line
- Automatic retry with exponential backoff on rate limit errors
- Free at any usage level that a portfolio project would hit
- Response latency under 2 seconds for most queries on Groq

### Disadvantages
- Two API keys to manage instead of one
- Groq's free tier has a daily token limit — heavy use burns through it
- No streaming support in the current implementation (responses arrive all at once)
- Manual provider rotation means no intelligent load balancing

### Alternatives considered

| Alternative | Why we didn't choose it |
|---|---|
| **OpenAI GPT-4o** | Costs money; not suitable for an open portfolio project |
| **Anthropic Claude API** | Same cost concern; also overkill for code generation tasks |
| **LiteLLM** | Adds a dependency for something we can do in 80 lines; hides the logic |
| **Ollama (local)** | No GPU on the dev machine; inference too slow for interactive use |
| **LangChain router** | Heavy dependency, adds complexity, we only need routing not orchestration |
| **Single provider** | Single point of failure; rate limits break the whole system |

---

## 2. Graph Intelligence Layer

### What we chose
**tree-sitter** for parsing, producing three graph types (AST, Dependency, Call) rendered with **D3.js force simulation**.

### Why
- tree-sitter is the de-facto standard for language-agnostic parsing — the same library used in Neovim, GitHub Copilot, and VS Code
- It supports Python, JavaScript, Java, and C++ out of the box with the same API
- Force-directed graphs are the right choice for code graphs: nodes cluster naturally by module, edges reveal coupling without manual layout
- Three graph types give orthogonal views of the same codebase — structure (AST), coupling (dependency), behaviour (call)

### Advantages
- Parsing is incremental and fast — sub-millisecond for most files
- The same parser handles all four supported languages
- D3 force simulation is interactive: draggable nodes, zoomable, tooltips on hover
- Three simultaneous views reveal things one graph can't: a class might look fine in the AST but show excessive coupling in the dependency graph

### Disadvantages
- tree-sitter grammars must be compiled for each language — adds to setup complexity
- The call graph is a static approximation: it cannot resolve dynamic dispatch, monkey-patching, or runtime-generated calls
- Large repos (1000+ files) make the force simulation slow and the graphs unreadable
- No incremental graph updates — reloading the repo rebuilds all three graphs from scratch

### Alternatives considered

| Alternative | Why we didn't choose it |
|---|---|
| **ast module (Python only)** | Only works for Python; we needed multi-language support |
| **Joern / code-property graphs** | Excellent but complex to set up; overkill for a portfolio visualiser |
| **NetworkX + matplotlib** | Static images, not interactive; matplotlib in a browser is a poor experience |
| **Cytoscape.js** | Good graph library but heavier than D3; less flexibility for custom force layouts |
| **Graphviz** | Generates static SVGs; no interactivity; layout algorithm not designed for large dynamic graphs |
| **Language Server Protocol** | Would give richer semantic info but requires per-language server setup |

---

## 3. Memory Layer

### What we chose
**FAISS** as the vector index, with three parallel embedders: **sentence-transformers MiniLM** (local), **Google Gemini embeddings** (API), and **Cohere embeddings** (API).

### Why
- FAISS is the fastest open-source approximate nearest-neighbour library — Facebook's production vector search, runs in-process with no server
- Three embedders lets the system demonstrate that different embedding models find different relevant memories for the same query — a genuine research insight built into the UI
- MiniLM runs locally with no API key, making the system functional even offline
- The comparison tab is a teaching tool: users see concretely which embedder retrieves more relevant results for their query

### Advantages
- FAISS searches millions of vectors in milliseconds on CPU
- No separate vector database to deploy or maintain
- Three-embedder comparison is a unique feature — most systems use one embedder
- MiniLM means the system works without any API keys at all
- Semantic search gives dramatically better retrieval than keyword search for code tasks

### Disadvantages
- FAISS indexes are in-memory — a server restart loses all stored memories (mitigated by FAISS's save/load, but not currently auto-saved)
- Three embedders means three times the storage and three times the embedding cost for any store operation
- MiniLM (384-dim) is significantly less capable than Gemini (768-dim) for nuanced code semantics
- No metadata filtering — can't search "only memories from planner agent in the last 7 days"

### Alternatives considered

| Alternative | Why we didn't choose it |
|---|---|
| **ChromaDB** | Good embedded vector DB but adds a dependency and a server process |
| **Pinecone** | Managed service with cost; not suitable for offline or free use |
| **Weaviate** | Requires Docker; too heavy for a single-machine portfolio project |
| **SQLite with vector extension** | Interesting but sqlite-vec is experimental and slow vs FAISS |
| **OpenAI embeddings only** | Single vendor, costs money, hides the multi-embedder insight |
| **BM25 (keyword search)** | Fast and interpretable but misses semantic similarity — "fix payment" won't find "handle checkout error" |

---

## 4. Sandbox

### What we chose
A two-stage pipeline: **static security validation** (pattern matching against a blocklist) followed by **subprocess execution** with a 30-second timeout and output capture.

### Why
- Static validation catches the obvious dangers (eval, exec, os.system, subprocess, __import__) before any code runs
- Subprocess isolation means a crashing or hanging script doesn't kill the server
- The 30-second timeout prevents infinite loops from hanging indefinitely
- Integrated pytest runner means users can write and run tests in the same UI without switching tools

### Advantages
- Stateless — each execution is independent, no shared state between runs
- Fast for simple scripts — subprocess overhead is negligible vs execution time
- Pytest integration is genuinely useful: write code, write tests, run both in one click
- Validation errors are readable and specific ("blocked pattern: os.system at line 4")

### Disadvantages
- Static validation is bypassable by a determined user (base64-encoded eval, getattr tricks, etc.) — this is a dev tool, not a security boundary
- Subprocess isolation is lighter than a container — the subprocess still has filesystem access
- No resource limits beyond timeout: a script can allocate all available RAM
- JavaScript/Java/C++ execution requires the runtime to be installed on the host machine
- No persistent state between runs — can't build on a previous execution's output

### Alternatives considered

| Alternative | Why we didn't choose it |
|---|---|
| **Docker containers** | Strong isolation but requires Docker daemon; too heavy for a dev tool |
| **WebAssembly (Pyodide)** | True sandboxing in-browser but limited to Python, no filesystem, slow startup |
| **Firecracker microVMs** | Production-grade isolation (used by AWS Lambda) but massive operational complexity |
| **RestrictedPython** | Python-only; incomplete — sophisticated scripts can still escape |
| **E2B / Modal** | Managed sandboxes with cost; not suitable for a local dev tool |
| **No sandbox at all** | Viable for a local tool but removes the educational value of showing safe execution |

---

## 5. Reward Model

### What we chose
A **rule-based multi-dimensional scorer** with five weighted dimensions, storing results in a **JSONL flat file**.

### Why
- Rule-based scoring is interpretable — every score has an explanation, unlike a neural reward model
- Five dimensions capture orthogonal aspects of code quality: a script can be correct but insecure, or complete but inefficient
- JSONL is the simplest possible store: human-readable, appendable, no database required, trivially portable
- Weighted aggregation (not a neural network) means the scoring is deterministic and debuggable

### Advantages
- Fully explainable — every dimension score comes with a feedback string
- Zero latency — pure Python, no model inference
- JSONL store is portable: copy one file to transfer all reward history
- Weights are configurable without retraining
- Works offline with no API keys

### Disadvantages
- Rule-based scoring misses nuance: a heuristic for "code quality" can't catch all bad code
- The five dimensions reflect one team's opinion about what matters in code
- No inter-rater reliability — two different reward models trained on the same data would score differently
- JSONL has no indexing — loading history for 10,000+ records is slow
- Scores are not calibrated against human preferences — they measure proxy signals (test pass rate, pattern counts) not true quality

### Alternatives considered

| Alternative | Why we didn't choose it |
|---|---|
| **Neural reward model (trained on human preferences)** | Requires a large preference dataset we don't have; overkill for a portfolio project |
| **GPT-4 as judge** | Good quality but expensive and slow; would break the real-time scoring UX |
| **Constitutional AI approach** | Interesting but requires multiple LLM calls per score and a defined constitution |
| **SQLite** | Better for querying but more complex setup; JSONL is portable and sufficient |
| **PostgreSQL** | Production-grade but requires a database server for what is essentially a log |
| **Single composite score** | Loses the interpretability that dimension breakdown provides |

---

## 6. RLHF Training Pipeline

### What we chose
A **three-library pipeline**: scikit-learn for analysis and pair selection, Keras (JAX backend) for hyperparameter sweep, PyTorch + trl + peft for DPO training. GPU training runs on **Google Colab T4** via a dedicated notebook.

### Why
- Three libraries demonstrate mastery of the full ML stack — not just one framework
- sklearn is the right tool for tabular analysis (clustering, feature importance) — no need for a neural network to analyse reward distributions
- Keras sweep is fast and interpretable — a reward head model is simple enough that Keras adds no overhead
- DPO (Direct Preference Optimisation) is the state-of-the-art RLHF alternative — it avoids training a separate reward model and is more stable than PPO
- Colab T4 is free and adequate for fine-tuning a 0.5B parameter model

### Advantages
- DPO is significantly more stable than PPO-based RLHF — no reward hacking, no actor-critic complexity
- LoRA fine-tuning means the full model never needs to fit in VRAM — only the adapters are trained
- Keras sweep provides a quantitative basis for the DPO learning rate and batch size — not guesswork
- The pipeline is modular: each stage can be run independently or skipped
- The Colab notebook is self-contained — anyone can run it with a free Google account

### Disadvantages
- The three-library approach has dependency conflicts (torchao version, trl API changes between releases)
- 4–5 training pairs is not enough for meaningful generalisation — the model memorises rather than learns
- DPO requires explicit (chosen, rejected) pairs — if all reward scores are similar, no pairs are found
- Colab free tier disconnects after ~3 hours — training a large dataset risks losing progress
- The Keras sweep trains a reward head, not the language model — the correlation between sweep results and DPO performance is indirect

### Alternatives considered

| Alternative | Why we didn't choose it |
|---|---|
| **PPO-based RLHF** | Much more complex — requires separate actor, critic, and reward models; training is unstable |
| **REINFORCE** | Simpler than PPO but high variance; worse than DPO for language model fine-tuning |
| **Full fine-tuning (no LoRA)** | Requires 10x+ more VRAM; impossible on a free T4 |
| **Kaggle GPU instead of Colab** | Viable alternative — 30 hours/week free; notebook is compatible |
| **Single training library** | Would demonstrate less breadth; the three-library approach is a deliberate portfolio choice |
| **Reward model training (separate neural network)** | Adds complexity without clear benefit for a 4-pair dataset |
| **RLVR (reinforcement learning from verifiable rewards)** | Interesting but requires verifiable tasks (math, code execution) — our reward is multi-dimensional and not verifiable |

---

## 7. Assistant Engine

### What we chose
A **mode-detection pipeline** (Generate/Improve/Guide) with keyword scoring, chained through LLM → sandbox → reward model → memory. The assistant reads live system state (repo, memory, rewards) as context for every call.

### Why
- Three modes cover the three primary use cases of a coding assistant without requiring the user to manually select a mode
- Auto-detection from keywords is fast, interpretable, and requires no additional model call
- Chaining sandbox and reward model after LLM output creates a feedback loop: the assistant generates code, validates it, scores it, and stores it — all in one interaction
- Reading live system state (not a static profile) means the assistant's context is always current

### Advantages
- Mode detection happens in microseconds — no latency added
- The full chain (generate → validate → score → store) creates genuine data for future training
- Context awareness means the assistant gives answers specific to the loaded repo, not generic advice
- Expertise detection (beginner/intermediate/expert) adapts tone without asking the user

### Disadvantages
- Keyword-based mode detection is brittle — "how do I write a sort function" could be Generate or Guide depending on intent
- The LLM call uses a concatenated history string rather than a proper messages array — some context nuance is lost
- No streaming — the full response arrives at once, making long generations feel slow
- The singleton engine pattern means a server restart loses the session history (history is in-memory only)
- Expertise detection from a single message is unreliable — a beginner can use expert terminology

### Alternatives considered

| Alternative | Why we didn't choose it |
|---|---|
| **Ask user to select mode manually** | Worse UX; users want to just type, not configure |
| **Use a classifier model for mode detection** | Adds latency and a dependency for marginal accuracy improvement |
| **LangChain agent with tools** | Good architecture but heavy dependency; tool-calling adds 1-2 extra LLM calls per interaction |
| **OpenAI function calling / tool use** | Elegant but vendor-specific; our router supports plain text completion only |
| **Redis for session storage** | Production-correct but requires a Redis server; overkill for a single-user dev tool |
| **Streaming responses** | Would require SSE endpoint changes and frontend EventSource handling — deferred to future work |

---

## 8. Frontend Architecture

### What we chose
A **single-file vanilla JS + D3.js + Chart.js** application. No framework, no build step, no bundler.

### Why
- A single HTML file is trivially deployable — copy one file, done
- No build step means no webpack, no node_modules, no compilation — the visualiser works immediately after cloning
- D3.js is the gold standard for data-driven interactive graphics — nothing else comes close for force-directed graphs
- Chart.js handles the ML training curves with minimal code
- Vanilla JS forces deliberate, readable code — no framework magic hiding what's happening

### Advantages
- Zero build tooling — open the file or point a server at it, it works
- No framework version conflicts — the file will render correctly in any modern browser indefinitely
- D3 gives pixel-level control over every graph element
- Fast initial load — no framework bundle to parse (D3 + Chart.js via CDN is ~200KB)
- Easy to read and modify — no JSX, no TypeScript compilation, no module resolution

### Disadvantages
- A 3000-line single HTML file is hard to navigate without a project-wide search
- No component reuse — similar UI patterns (cards, buttons, pills) are duplicated across tabs
- No TypeScript — type errors surface at runtime, not at development time
- Global state (`currentLang`, `currentRepo`, `simulations`) is error-prone at scale
- No hot module replacement — a code change requires a full page refresh
- Testing the frontend requires a running server — no unit tests for JS logic

### Alternatives considered

| Alternative | Why we didn't choose it |
|---|---|
| **React + Vite** | Better component model but adds build step, node_modules, and framework churn |
| **Vue.js** | Same trade-off as React; adds complexity for a single-developer project |
| **Svelte** | Excellent for this use case but compilation step is still required |
| **HTMX** | Interesting for server-driven UI but D3 requires client-side control over DOM |
| **Plotly Dash** | Python-native dashboard but limited graph customisation; D3 is far more flexible |
| **Streamlit** | Fast to prototype but too constrained for custom graph layouts and tab architecture |
| **Multiple HTML files** | Simpler per-file but requires navigation between pages — breaks the single-app feel |

---

## 9. Overall System Architecture

### What we chose
A **monolithic FastAPI application** with a single-file frontend, running in a Python virtual environment on WSL2. All state is file-based (JSONL, FAISS indexes on disk).

### Why
- A monolith is the right choice for a portfolio project — it's deployable in one command, understandable end-to-end, and has no distributed systems complexity
- FastAPI gives async endpoints, automatic OpenAPI docs, and Pydantic validation with minimal boilerplate
- File-based state (JSONL + FAISS) means the system works without a database or message queue
- WSL2 on Windows is the most common developer environment for the target audience

### Advantages
- Single `uvicorn visualiser.main:app --reload` to run everything
- No Docker, no Kubernetes, no message queue, no separate database process
- FastAPI's automatic `/docs` endpoint gives a free interactive API explorer
- Pydantic models catch malformed requests at the boundary before they reach business logic
- Async endpoints mean file I/O and subprocess calls don't block other requests

### Disadvantages
- A monolith doesn't scale horizontally — two server instances would have separate in-memory state and separate FAISS indexes
- File-based state is not concurrent-safe — two simultaneous reward scores writing to the same JSONL will interleave
- No authentication — the visualiser is open to anyone on the local network
- No background task queue — long-running operations (repo clone, training) block the request
- WSL2 file system performance is slower than native Linux for I/O-heavy operations

### Alternatives considered

| Alternative | Why we didn't choose it |
|---|---|
| **Microservices (separate services for memory, sandbox, rewards)** | Operational complexity far exceeds the benefit for a single-user tool |
| **Django** | More batteries included but heavier; FastAPI is better for an API-first design |
| **Flask** | Good but lacks async support and automatic schema validation |
| **PostgreSQL + pgvector** | Would replace both JSONL and FAISS; better at scale but requires a running database |
| **Celery + Redis for background tasks** | Correct architecture for long-running tasks but major operational overhead |
| **Docker Compose** | Good for reproducible deployment but adds friction for the target developer audience |