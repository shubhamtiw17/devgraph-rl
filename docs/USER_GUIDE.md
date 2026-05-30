# User Guide

**DevGraph-RL: Deployment and Walkthrough**

This guide covers everything from first-time setup to a full walkthrough of every tab in the visualiser. It is written for two audiences: technical users who want the precise details, and non-technical users who want to understand what they are looking at.

---

## Table of Contents

1. [Deployment](#1-deployment)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
   - [Starting the Server](#starting-the-server)
   - [Stopping the Server](#stopping-the-server)
2. [The Visualiser](#2-the-visualiser)
   - [Header Bar](#header-bar)
3. [Tab Walkthroughs](#3-tab-walkthroughs)
   - [Graphs Tab](#graphs-tab)
   - [Memory Tab](#memory-tab)
   - [Sandbox Tab](#sandbox-tab)
   - [Rewards Tab](#rewards-tab)
   - [ML Lab Tab](#ml-lab-tab)
   - [Assistant Tab](#assistant-tab)
4. [End-to-End Workflow](#4-end-to-end-workflow)
5. [Troubleshooting](#5-troubleshooting)

---

## 1. Deployment

### Prerequisites

**For technical users:**
- Python 3.11 (exactly 3.10 and 3.12 have dependency conflicts with tree-sitter grammars)
- Git
- WSL2 (Ubuntu 22.04+) if on Windows, or any Linux distro
- A Groq API key (free at [console.groq.com](https://console.groq.com)) — takes 30 seconds to create

**For non-technical users:**
You need a computer running Windows (with WSL2) or Linux, and a free account at [console.groq.com](https://console.groq.com). Everything else is installed automatically.

---

### Installation

Open a terminal (on Windows: search for "Ubuntu" in the Start Menu after installing WSL2).

**Step 1 : Clone the repository**
```bash
git clone https://github.com/shubhamtiw17/devgraph-rl.git
cd devgraph-rl
```

**Step 2 : Create a Python virtual environment**

A virtual environment is an isolated Python installation so DevGraph-RL's dependencies don't conflict with anything else on your machine.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

You will see `(.venv)` at the start of your terminal prompt. This means the environment is active.

**Step 3 : Install dependencies**
```bash
pip install -e ".[dev,graphs]"
```

This installs around 60 packages. It takes 2–5 minutes on first run. You will only need to do this once.

**Step 4 : Configure API keys**
```bash
cp .env.example .env
nano .env
```

Replace the placeholder values with your real keys:

```env
GROQ_API_KEY=gsk_your_actual_key_here
GEMINI_API_KEY=AIzaSy_your_actual_key_here    # optional
COHERE_API_KEY=your_cohere_key_here           # optional
```

Save the file (`Ctrl+X`, then `Y`, then `Enter` in nano).

> **Non-technical note:** Only `GROQ_API_KEY` is required. The other two are optional the system works without them, just with fewer embedder options in the Memory tab.

---

### Starting the Server

Every time you want to use DevGraph-RL:

```bash
# Navigate to the project folder
cd ~/devgraph-rl

# Activate the virtual environment
source .venv/bin/activate

# Load the API keys
set -a && source .env && set +a

# Start the server
uvicorn visualiser.main:app --reload
```

You will see output like:
```
✓ Groq ready
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

Open your browser and go to **http://localhost:8000**

The `--reload` flag means the server automatically restarts when you edit any Python file useful during development.

> **Non-technical note:** You need to run these four commands every time you start a new terminal session. Consider saving them as a shell alias or a small script called `start.sh`.

---

### Stopping the Server

Press `Ctrl+C` in the terminal where uvicorn is running.

---

## 2. The Visualiser

When you open `http://localhost:8000` you will see the main interface. It has three fixed areas:

**Header bar**  always visible at the top. Contains the logo, repo input, language selector, and tab buttons.

**Status bar** a thin strip below the header showing the current status (loading, ready, error) and the name of the loaded repo.

**Main area**  changes based on which tab is selected.

---

### Header Bar

**Logo** : DevGraph-RL, top left.

**Repo input + Load button**  paste any public GitHub URL and click Load. The system will clone the repo, parse it, and build all three graphs. Example: `https://github.com/pallets/flask`

**Language selector** (Python / JavaScript / Java / C++)  only visible on the Graphs tab. Switches the sample graph language when no repo is loaded.

**Tab buttons**  six tabs: Graphs, Memory, Sandbox, Rewards, ML Lab, Assistant. Each tab is a different tool. You can switch between them freely at any time — your work in one tab is not lost when you switch to another.

---

## 3. Tab Walkthroughs

---

### Graphs Tab

**What it is:**
Three interactive graphs that show the structure of your code. They are rendered side by side: AST Graph (left), Dependency Graph (centre), Call Graph (right).

**For non-technical users:**
Think of these as three different maps of the same city. The AST graph shows the buildings (files, classes, functions). The dependency graph shows which roads connect which districts (which files import which other files). The call graph shows the traffic, which functions call which other functions when the code runs.

**For technical users:**
- AST graph is built from tree-sitter's concrete syntax tree, flattened to file/class/function nodes with contains/inherits edges
- Dependency graph uses import statement analysis — each `import` becomes a directed edge
- Call graph is static analysis of function call expressions — dynamic dispatch and runtime-generated calls are not captured
- All three use D3.js force simulation with configurable link distance and charge strength

**How to use it:**

1. Without a repo loaded, the tab shows a sample Python graph (shapes library). Use the language buttons to switch to JavaScript, Java, or C++ samples.

2. To analyse your own code:
   - Paste a GitHub URL into the repo input at the top
   - Click **Load**
   - The status bar shows "Cloning..." then the repo name and node count when done
   - All three graphs rebuild automatically

3. **Interacting with graphs:**
   - **Drag** any node to reposition it, the simulation adjusts
   - **Scroll** to zoom in and out
   - **Hover** over any node to see a tooltip with its full path and line number
   - **Click** the "Architecture & How It Works" bar at the bottom of each panel to expand an explanation of that graph type

4. **Ask a question** about the loaded repo using the query bar (below the status bar). Type a natural language question like "where is the authentication logic?" or "which functions call the database?" and click Ask. Relevant nodes will highlight in all three graphs simultaneously.

**What the colours mean:**
- Blue nodes: files
- Orange nodes: classes
- Green nodes: functions
- Grey edges: contains (a file contains a class)
- Blue edges: imports
- Red edges: inherits
- Purple edges: calls

---

### Memory Tab

**What it is:**
A semantic search system. You store text snippets (descriptions of code tasks, solutions, or insights) and later retrieve them by meaning, not by exact keyword match.

**For non-technical users:**
Think of this as a smart notebook. You can write "I fixed the payment bug by adding a try-except around the database call" and save it. Later, when you ask "how do I handle database errors?" the system finds that note even though you didn't use the word "database" in the exact same way. It understands meaning, not just keywords.

**For technical users:**
- FAISS flat L2 index per embedder
- MiniLM: all-MiniLM-L6-v2, 384-dim, runs locally via sentence-transformers
- Gemini: text-embedding-004, 768-dim, requires GEMINI_API_KEY
- Cohere: embed-english-v3.0, 384-dim, requires COHERE_API_KEY
- Cosine similarity search, top-K configurable

**How to use it:**

**Embedder Status** (top card) shows which embedders are available (green dot) and how many vectors each index holds.

**Store Memory** (left card):
1. Type a description of a task or solution in the Text/Task field
2. Select an embedder (MiniLM if you want no API key; Gemini or Cohere if you have those keys and want higher-dimensional embeddings)
3. Select the agent type (manual for things you write yourself; coding/planner/reviewer for agent-generated content)
4. Optionally set the repo path if this memory relates to a specific codebase
5. Click **Store Memory** to save to one embedder, or **Store to All** to save to all three simultaneously

**Semantic Search** (right card):
1. Type a query in natural language e.g. "handle errors in the checkout flow"
2. Select which embedder to search
3. Set Top-K (how many results to return)
4. Click **Search**
5. Results appear ranked by similarity score (0 to 1, higher is more similar)

**Compare Embedders** (bottom card):
- Run the same query through all three embedders simultaneously
- Results are shown in three columns side by side
- This reveals which embedder finds more relevant memories for your specific type of query

**Sync Indexes**:
- If you have stored many memories in one embedder and want to copy them to another, use Sync to All
- This re-encodes all memories from the selected source embedder into the other two

---

### Sandbox Tab

**What it is:**
A safe code editor and runner built into the browser. Write code, run it, see the output — without leaving the visualiser.

**For non-technical users:**
Think of this as a safe scratch pad where you can try out code. The system checks your code for dangerous operations before running it, then runs it in an isolated environment so it can't damage your computer. You also have a test area on the right — you can write automatic checks (tests) to verify your code works correctly.

**For technical users:**
- Two-stage pipeline: static validation (regex/AST pattern matching against a blocklist) then subprocess execution with 30-second timeout and stdout/stderr capture
- pytest integration: test code is written to a temp directory alongside `solution.py` (the source code) and pytest is invoked on it
- Blocked patterns: `os.system`, `subprocess`, `eval`, `exec`, `__import__`, writes outside `/tmp`
- Supports Python (and JavaScript/Java/C++ if the runtime is installed on the host)

**How to use it:**

**Code Editor** (left):
1. Select a language from the dropdown
2. Write your code in the source code area
3. Click **Validate** to check for dangerous patterns without running you'll see a green "Validation Passed" or a list of blocked operations
4. Click **Run** to execute the code stdout and stderr appear in the Results section at the bottom

**Test Runner** (right):
1. Write pytest tests in the test code area, import from `solution` (your source code is automatically saved as `solution.py`)
2. Click **Run with Tests** — runs both your code and your tests together
3. Results show passed/failed counts, a progress bar, and the full pytest output

**Results** (bottom):
- Shows execution status (success/failed), timing, stdout, stderr
- For test runs: shows passed count, failed count, pass rate percentage

---

### Rewards Tab

**What it is:**
A scoring system for agent-generated code. Paste a task and its output, click Score It, and get a detailed breakdown across five quality dimensions.

**For non-technical users:**
Think of this as a grading system for code. You tell it what you asked an AI to do (the task) and what it produced (the output), and it gives you a score from 0 to 100% across five categories: like a report card. Over time you build a history of scores that shows how the AI is performing.

**For technical users:**
- Five weighted dimensions: Correctness (0.35), Code Quality (0.25), Task Completion (0.20), Efficiency (0.10), Security (0.10)
- Correctness: test pass rate + execution success signal
- Code Quality: AST-based metrics (function count, docstring presence, type hint density, nesting depth)
- Task Completion: keyword overlap between task description and output
- Efficiency: cyclomatic complexity heuristic
- Security: pattern matching against dangerous operations
- Results stored to `data/vector_store/reward_history.jsonl`

**How to use it:**

**Score an Output** (top card):
1. Type a task description in the Task field. e.g. "write a function that removes duplicates from a list"
2. Paste the agent's code or text in the Agent Output area
3. Select the agent type (coding, planner, or reviewer)
4. Select the language
5. Set Test Pass Rate if you ran tests (0 if no tests, 1.0 if all passed, 0.5 if half passed)
6. Click **Score It**
7. The score breakdown appears on the right a large percentage number, a summary sentence, and five dimension bars

**Statistics** (left of lower section):
- Total scored count
- Average, best, and worst scores
- Average score by agent type
- Score trend chart (last 20 outputs) shows whether quality is improving or declining over time

**Top Scoring Outputs** (right of lower section):
- The N highest-scoring outputs stored so far
- Click any result to see the task description

**Score History** (bottom):
- Full table of all scored outputs, sorted newest first
- Columns: time, task, agent type, overall score, correctness, quality, completion

**Clear All** — removes all reward history. Use this to start fresh if you want to generate clean training data.

> **Important:** The scores you collect here become the training data for the ML Lab tab. Score at least 10 outputs across different quality levels (some good, some bad) for the same tasks to generate useful training pairs.

---

### ML Lab Tab

**What it is:**
The RLHF (Reinforcement Learning from Human Feedback) training pipeline. Takes the reward scores you collected, analyses them, finds (good, bad) pairs, and uses them to fine-tune a language model to produce better code.

**For non-technical users:**
Think of this as a school for the AI. The Rewards tab collects report cards. This tab looks at all the report cards, finds examples of good work and bad work on the same assignment, and teaches the AI to do more of the good and less of the bad. The result is an AI that is better at the specific types of tasks you scored.

**For technical users:**
Three sequential panels, each corresponding to one stage of the training pipeline:

**Panel 1 — Sklearn Analysis (blue dot):**

Analyses the reward history using scikit-learn.

1. Set **Min score threshold** records below this score are excluded from analysis (use 0.0 to include all records for pair finding)
2. Set **Record limit** leave blank to process all records
3. Click **Run Analysis**

Results show:
- **Score Distribution**: total records, high/medium/low counts, mean ± std
- **Pairs selected**: how many (chosen, rejected) pairs were found, pairs require the same task with different quality outputs
- **Feature Importance**: which scoring dimensions matter most for quality (bar chart)
- **Quality Clusters**: KMeans clustering of records into high/medium/low quality groups

> You need at least 5 pairs to proceed to training. If you see 0 pairs, score more outputs for the same tasks at different quality levels, or lower the min score threshold.

**Panel 2 — Keras Sweep (yellow dot):**

Finds the best learning rate and batch size for training.

1. Set **Epochs per config** how many training epochs to run for each hyperparameter combination (3-5 is sufficient)
2. Set **Max configs** how many combinations to try (6 covers the grid well)
3. Click **Run Sweep**

Results show:
- Best config (learning rate, batch size, hidden dim)
- Loss curve for the best run
- Ranked list of all runs

> The Keras sweep requires GPU for reasonable speed. Run it in the Colab notebook (see below) rather than locally.

**Panel 3 : PyTorch DPO Training (red dot):**

The actual fine-tuning step.

1. Select a **Base model**  Qwen2.5-0.5B is recommended (smallest model that fits in free T4 VRAM)
2. Set **Training epochs** 3 is a good default
3. Optionally set a **HuggingFace repo** and **HF token** to push the trained model to the Hub
4. Check **Skip Keras sweep** if you want to use default hyperparameters
5. Click **Full Pipeline** to run all three stages sequentially

> Full training requires a GPU and takes 1-3 hours. Use the Colab notebook for this step.

**Load History** — after downloading `training_result.json` from Colab and copying it to `data/checkpoints/`, click this to display the training results in the ML Lab tab without re-running.

**Running on Colab (recommended for training):**

1. Open `notebooks/train_colab.ipynb` in Google Colab
2. Set Runtime → T4 GPU
3. Run Cell 1 (installs deps), then Runtime → Restart runtime
4. Run Cell 2 (config) — set your GitHub repo URL
5. Run Cell 3 (clone repo)
6. Run Cell 4 (upload `reward_history.jsonl`)
7. Run Cell 5 (analysis) — check pairs are found
8. Run Cell 6 (build dataset)
9. Run Cell 7 (Keras sweep) — optional
10. Run Cell 8 (DPO training) — takes 1-3 hours
11. Run Cell 9 (download result) — saves `training_result.json`
12. Copy result to `~/devgraph-rl/data/checkpoints/training_result.json`
13. Click **Load History** in the ML Lab tab

---

### Assistant Tab

**What it is:**
A context-aware AI assistant embedded in the visualiser. It knows about your loaded repo, your stored memories, and your reward history — and uses all of that to give more relevant answers than a generic chatbot.

**For non-technical users:**
Think of this as an AI coding helper that has been watching everything you've done in the other tabs. It knows which repo you loaded, what code patterns you've stored in memory, and how well previous code attempts scored. When you ask it a question, it uses all of that context to give you a more relevant answer than a general-purpose AI would.

**For technical users:**
- Mode detection: keyword scoring over improve/generate/guide signal words: O(1), no model call
- Context assembly: live reads from repo_manager (name, language), reward_store (stats), memory_manager (top-3 semantic results for the current message)
- LLM call: single `router.complete(prompt, system)` with history concatenated into the prompt string
- Post-processing: code extraction (regex), sandbox validation, reward scoring, memory storage if score ≥ 0.70

**Context bar** (top strip):
Shows what the assistant currently knows:
- **Repo pill** : name and language of the loaded repo (orange if loaded)
- **Memories pill** : how many vectors are in the memory store
- **Scored pill** : how many reward records have been collected
- **Mode badge** : GENERATE / IMPROVE / GUIDE, updates live as you type
- **Language selector** : hint to the assistant about which language to use for code generation
- **Clear chat** : resets the conversation history

**How to use it:**

**Starter buttons** : four pre-written prompts appear when the chat is empty. Click any of them to fill the input and send. Good for a first test.

**Typing your own message:**
- Press **Enter** to send
- Press **Shift+Enter** for a newline within your message
- The mode badge shows which mode your message will trigger before you send

**What happens after you send:**
1. A typing indicator (three animated dots) appears while the LLM processes
2. The response arrives as a text bubble
3. If the response contains code, it appears in a separate code block with a **Copy** button
4. Below the response you may see chips:
   - **Sandbox passed/failed** — the generated code was automatically run
   - **Score %** — the reward model's score for the generated code
   - **+X% improved** — in Improve mode, how much better the refactored code scored vs the original
   - **Stored to memory** — the output scored above 70% and was automatically stored to the memory index
5. **Suggestion buttons** — clickable next steps. Click any to fill the input with that suggestion.

**The three modes in practice:**

*Generate mode* (triggered by: write, create, build, implement, generate, make, function, class, script)
> "Write a function to parse a CSV file with error handling"

The assistant writes complete production-quality code, validates it through the sandbox, scores it, and stores it to memory if the score is above 70%.

*Improve mode* (triggered by: refactor, improve, fix, optimise, optimize, clean, review, rewrite, debug)
> "Refactor the loaded file to follow best practices"

The assistant reads the loaded file content, scores the original code, rewrites it, scores the improved version, and reports the delta. Works best when a repo is loaded so the assistant has the actual file content.

*Guide mode* (triggered by: explain, how, why, what, help, understand, error, not working)
> "Explain how the graph visualiser works"

The assistant explains, debugs, or onboards. It adapts its language based on detected expertise level — simpler explanations for beginners, technical depth for experts.

---

## 4. End-to-End Workflow

Here is the recommended sequence for getting the most out of DevGraph-RL:

**Step 1 — Explore a codebase (Graphs tab)**
Load a GitHub repo. Explore the three graphs. Use the Ask bar to query the repo. Get familiar with the structure.

**Step 2 — Store key insights (Memory tab)**
Write down what you learned — key patterns, important functions, architectural decisions. Store them with MiniLM (no API key needed). These memories will appear as context in the Assistant tab.

**Step 3 — Generate and test code (Sandbox tab)**
Write code with the help of the Assistant tab (see step 5), then validate and run it here. Use the test runner to verify correctness.

**Step 4 — Score outputs (Rewards tab)**
For each meaningful piece of code you generate, score it. Do this for both good and bad attempts at the same task — the contrast is what creates training data. Aim for at least 10 scored pairs across 3-5 distinct tasks.

**Step 5 — Use the assistant (Assistant tab)**
Ask the assistant to generate, improve, or explain code. The assistant reads your loaded repo and stored memories as context. High-scoring outputs are automatically added to memory for future context.

**Step 6 — Train (ML Lab tab)**
After scoring 10+ outputs, run the analysis to find training pairs. Run DPO training on Colab. Download the result and load it in the ML Lab tab. The model has now been fine-tuned on your preferences.

**Step 7 — Iterate**
Score more outputs from the fine-tuned model (via the Assistant tab). Run training again. The reward delta should grow with each iteration.

---

## 5. Troubleshooting

**"LLM router is not available"**

The API keys aren't loaded. Run:
```bash
set -a && source .env && set +a
uvicorn visualiser.main:app --reload
```

**"No pairs found" in ML Lab analysis**

You need the same task scored at different quality levels. Score 3-5 different attempts at the same task (e.g. "write an add function") — some deliberately bad (no error handling, no type hints), some good. Set Min score threshold to 0.0.

**Graphs don't load / tree-sitter error**

The tree-sitter grammars need to be compiled:
```bash
pip install -e ".[graphs]" --force-reinstall
```

**Server starts but browser shows blank page**

Hard refresh: `Ctrl+Shift+R`. If that doesn't work, check the terminal for Python errors.

**"Permission denied" when cloning a repo**

The repo is private. DevGraph-RL only supports public GitHub repos in the current version.

**Colab disconnects during training**

Colab free tier disconnects after ~90 minutes of inactivity. The checkpoint is saved every 100 steps — re-run Cell 8 and it will resume from the last checkpoint. Alternatively, keep the Colab tab focused to prevent idle timeout.

**Memory tab shows "Failed" for Gemini or Cohere**

Those embedders require API keys. Add `GEMINI_API_KEY` or `COHERE_API_KEY` to your `.env` file. MiniLM works without any API keys and is the default.

**Port 8000 already in use**

Another process is using the port. Either kill it:
```bash
pkill -f uvicorn
```
Or start on a different port:
```bash
uvicorn visualiser.main:app --reload --port 8001
```
Then open `http://localhost:8001`.