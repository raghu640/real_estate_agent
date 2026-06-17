# CLAUDE.md — Property Pricing Agent

This file tells Claude Code how to work with this project.
**Before making any change: read this file, then read the relevant source files.
Never rely on values written here — always derive them from the actual code.**

---

## What this project does

A LangGraph agent that estimates residential property prices for the Bengaluru market.
The user provides a natural-language property description; the agent calls tools,
reasons over the combined results, and returns a structured price estimate with range,
PSF, key factors, and confidence level.

---

## Project structure

Run this to see the live layout before making any changes:

```bash
find . -not -path './.venv/*' -not -path './__pycache__/*' | sort
```

Key files:
- `agent.py` — LangGraph graph definition, system prompt, entry point (`price_property()`)
- `tools/` — one file per tool; each exports a single `@tool`-decorated function
- `tools/__init__.py` — the `TOOLS` list; this is the canonical list of active tools
- `requirements.txt` — all dependencies and pinned versions
- `data/` — place historical CSVs and SQLite DB here

---

## Architecture

### Graph topology

Read `agent.py` directly for the authoritative graph definition. The pattern is:

```
entry point → AgentState → [agent node] → should_continue()
                                ↑               ↓ tool_calls?
                           [ToolNode] ←─────── yes
                                               ↓ no
                                              END
```

Do not hardcode tool call sequences — Claude decides which tools to call and in what order.

### AgentState

Read the `AgentState` TypedDict in `agent.py` for field names, types, and annotations.
Do not copy field definitions here — they will drift.

### Model

The model name is defined once in `agent.py`. Read it from there. Do not duplicate it here.

---

## Tools

### Discovering tools

The canonical list of active tools is the `TOOLS` list in `tools/__init__.py`.
Read that file to know what tools exist. Do not maintain a duplicate list here.

To see each tool's signature, inputs, and outputs:

```bash
grep -n "^def \|^@tool\|Args:\|Returns:" tools/*.py
```

### Tool categories

Tools fall into two categories — check the file header comment in each tool file:

- **MCP tools**: call external Apify scrapers for live listing data. Require `APIFY_API_TOKEN`.
- **Local tools**: pure Python computation, no external calls.

The Apify MCP URL and actor slugs are defined in the tool files themselves. Read them there.

---

## Environment variables

The required env vars are documented in each file that uses them via `os.environ["KEY"]`.
To find all required vars:

```bash
grep -rn 'os.environ\[' tools/ agent.py
```

Store values in `.env` at project root. Never commit `.env`. Add `.env` to `.gitignore`.

---

## How to run

```bash
# Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run (example query is in the __main__ block of agent.py)
python agent.py

# Or import
from agent import price_property
print(price_property("<your query here>"))
```

---

## Coding conventions

- **One tool per file** in `tools/`. File name = tool function name.
- **Docstrings are contracts**: every `@tool` function must have a Google-style docstring
  with `Args:` and `Returns:`. LangGraph passes the docstring to the LLM as the tool
  description — vague docstrings produce bad tool calls.
- **Tools always return `dict`**. Never return plain strings. The agent needs structured
  data to reason over.
- **Tools never raise**. Catch all exceptions and return `{"error": str(e), "data": []}`.
  The agent must be able to handle tool failures gracefully without crashing the graph.
- **No hardcoded secrets**. Use `os.environ["KEY"]` — fail loudly with a clear `KeyError`
  if a required variable is missing.
- **No hardcoded data in tool logic**. Lookup tables (locality profiles, adjustment factors,
  actor slugs) must live in named constants at the top of the file, not inline in functions.
- **Type hints required** on all function signatures.

---

## Data

All lookup tables (locality profiles, PSF benchmarks, adjustment factors) must live in
`data/` as JSON or CSV files — not hardcoded in source files.

Tools load them at runtime:

```python
import json, pathlib
DATA = json.loads((pathlib.Path(__file__).parent.parent / "data" / "filename.json").read_text())
```

To inspect current data files:

```bash
ls data/
```

To validate a data file loads correctly:

```bash
python -c "import json; d=json.load(open('data/<file>.json')); print(list(d.keys()))"
```

If you find hardcoded lookup tables inside `tools/*.py`, move them to `data/` and update
the tool to load from file. This keeps data changes out of code review and separately auditable.

---

## Extending the agent

### Add a new local tool
1. Create `tools/my_tool.py` — one `@tool` function, full docstring, returns `dict`
2. Import and add to `TOOLS` in `tools/__init__.py`
3. Update the system prompt in `agent.py` only if the new tool changes the reasoning flow

### Add a new Apify MCP scraper
1. Find the actor slug on `apify.com/store`
2. Add the slug to the `mcp_servers` URL in the relevant tool file (or create a new tool file)
3. Follow the existing MCP tool pattern in `tools/web_search.py`
4. Import and add to `TOOLS` in `tools/__init__.py`

### Add LangSmith tracing
No code changes needed. Set these env vars:

```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=<your key>
LANGCHAIN_PROJECT=property-agent
```

### Add memory / checkpointing
In `agent.py`, pass a checkpointer to `graph_builder.compile()` and a `thread_id` to
`graph.invoke()`. See LangGraph docs: https://langchain-ai.github.io/langgraph/concepts/persistence/

### Wrap in a FastAPI endpoint
Create `api.py` with a `POST /estimate` route calling `price_property(query)`.
Add `fastapi` and `uvicorn` to `requirements.txt`.

---

## Debugging

```bash
# See what tools are registered
python -c "from tools import TOOLS; [print(t.name, '-', t.description[:80]) for t in TOOLS]"

# Check env vars are set
python -c "import os; [print(k) for k in ['ANTHROPIC_API_KEY','APIFY_API_TOKEN'] if not os.environ.get(k)]"

# Run with LangSmith tracing
LANGCHAIN_TRACING_V2=true python agent.py

# Inspect graph structure
python -c "from agent import graph; print(graph.get_graph().draw_ascii())"
```

---

## Known limitations

Read `agent.py` and tool files for inline `# TODO` and `# LIMITATION` comments —
those are the authoritative record of known gaps. Do not duplicate them here.