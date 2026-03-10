# Examples for akgentic-tool

This directory contains example scripts demonstrating the usage of this module.

## Running Examples

```bash
# From workspace root with uv
uv run python packages/akgentic-tool/examples/<example>.py
```

## Available Examples

### knowledge_agent.py

Demonstrates end-to-end usage of `KnowledgeGraphTool` in a realistic scenario.

**What it demonstrates:**
- `KnowledgeGraphTool` configuration with `GetGraph`, `update_graph`, and `search` options
- Entity and relation creation via `update_graph` using `ManageGraph`, `EntityCreate`, `RelationCreate`
- Full graph retrieval and subgraph BFS queries via `get_graph`
- Root entity filtering (`roots_only=True`)
- Keyword search (no API key required) and hybrid/vector search (requires `OPENAI_API_KEY`)
- Compact system prompt summary — how the tool injects graph context into LLM system prompts
- Graceful degradation: hybrid search falls back to keyword mode without an OpenAI API key
- Production-quality patterns: Pydantic models throughout, no `dict[str, Any]`

**How to run:**

```bash
# Keyword-only mode — no API key required
uv run python packages/akgentic-tool/examples/knowledge_agent.py

# With hybrid/vector search (requires openai package and API key)
OPENAI_API_KEY=sk-... uv run python packages/akgentic-tool/examples/knowledge_agent.py
```

**Prerequisites:**

```bash
# Install all packages including knowledge graph extras
uv sync --all-packages --extra kg
```

The `[kg]` extra installs `openai` and `numpy` required by the vector index.
Keyword search and basic graph operations work without an OpenAI API key;
only hybrid/vector search requires a valid `OPENAI_API_KEY`.

**Expected output:**

```
============================================================
Knowledge Graph Tool — End-to-End Example
============================================================

--- Setup: configuring KnowledgeGraphTool ---
Tool: KnowledgeGraph — Knowledge graph tool for structured knowledge with semantic search

--- Step 1: Building tech stack knowledge graph ---
Graph built: Done
  Entities: 6
  Relations: 5

--- Step 2: Querying the knowledge graph ---

Full graph: 6 entities, 5 relations
  FastAPI (Framework) [ROOT]: Modern async Python web framework...
  ...

--- Step 3: System prompt summary (compact graph context) ---
**Knowledge Graph Summary:**
Entities: 6 | Relations: 5
Entity types: Cache, CI-CD, Container, Database, Framework, Language
...

--- Step 4: Searching the knowledge graph ---

Keyword search for 'database':
Search Results:
  - [entity] PostgreSQL (Database): ...
  ...
```

## Example Structure

Each example:

1. Imports necessary dependencies with `from __future__ import annotations`
2. Wires the tool using a minimal observer (same pattern as integration tests)
3. Demonstrates realistic domain use cases with meaningful data
4. Shows both simple and advanced usage patterns
5. Cleans up resources on exit
