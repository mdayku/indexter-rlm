# Indexter-RLM Product Requirements Document

## Vision

**Indexter-RLM** is an RLM-style (Recursive Language Model) context environment for coding agents. It treats the codebase as an **interactive environment to explore**, not a static document to retrieve from.

> *The future of code-aware LLMs is not bigger context windows, but treating the codebase as an environment the model must explore, not a document it must memorize.*

---

## Problem Statement

Modern coding agents struggle with large repositories because they treat codebases as:
- Static documents
- One-shot retrieval targets

Even advanced RAG systems:
- Retrieve top-K chunks once
- Stuff them into a prompt
- Hope the answer is locally contained

This fails for:
- **Multi-hop logic** - understanding flows across files
- **Misleading naming** - function names don't match behavior
- **Distributed invariants** - constraints spread across modules
- **Refactors and debugging** - requires exploration, not guessing

---

## Solution: RLM-Style Context Management

### The Key Insight (from RLM paper arXiv:2512.24601)

> **Context is not text to be loaded - it is an environment to be explored.**

RLMs interact with context stored externally, recursively inspecting, decomposing, and summarizing parts of the input. They avoid loading the full context into the prompt at once.

In practice, RLMs behave like:
> *A developer with a search bar, file tree, and scratchpad*

### Why Indexter Is the Right Foundation

Indexter already provides:
- **Semantic code units** (functions, classes, sections via tree-sitter)
- **Externalized context** (vector index + filesystem)
- **Tool-based access** (FastMCP server for LLMs)
- **Incremental indexing** (no runtime embedding cost)

This satisfies the core RLM requirement:
> *Context exists outside the model and is accessed intentionally.*

Indexter is already a **context environment** - we're adding recursive control.

---

## Architecture Overview

```
Agent (Cursor / Claude / Codex)
        |
        v
+-------------------+
| RLM Controller    |  <-- Enforced via .cursorrules + MCP prompts
| (search -> open   |
|  -> compress ->   |
|  -> refine)       |
+-------------------+
        |
        v
+-------------------+
| Indexter-RLM      |
| Context Environment|
+-------------------+
| - semantic search |
| - file access     |
| - symbol listing  |
| - scratchpad      |
+-------------------+
        |
        v
+-------------------+
| Vector Store      |
| (Qdrant local)    |
+-------------------+
```

---

## MVP: Minimum Viable RLM Loop

The smallest "wow" is:

1. **Search** - Semantic query via Indexter
2. **Open** - Read actual code nodes (not just summaries)
3. **Compress** - Store key observations (scratchpad)
4. **Refine** - Generate next query based on gaps
5. **Stop** - When evidence converges or budget reached

This is recursion over **context**, not tokens.

---

## Feature Roadmap

### Phase 1: Navigation Tools (Current)
- [x] `search_repository` - Semantic search
- [x] `list_repositories` - Show indexed repos
- [ ] `get_file` - Read actual file content
- [ ] `list_symbols` - List functions/classes in a file
- [ ] `find_references` - Find usages of a symbol

### Phase 2: Stateful Context (Scratchpad)
- [ ] `store_note` - Persist observations during exploration
- [ ] `get_notes` - Retrieve accumulated context
- [ ] Session-based note persistence

### Phase 3: Recursive Controller
- [ ] Enforce search-before-reasoning via `.cursorrules`
- [ ] Add stopping rules and confidence heuristics
- [ ] Debug logging of exploration path

### Phase 4: Evaluation
- [ ] Compare against naive large-context prompting
- [ ] Compare against single-shot RAG
- [ ] Metrics: correctness, hallucination rate, tool calls, tokens

---

## Success Metrics

1. **Correctness** - Agent answers match actual code behavior
2. **Grounding** - Agent cites retrieved code in reasoning
3. **Efficiency** - Fewer tokens used than context-stuffing
4. **Hallucination rate** - Lower than baseline RAG

---

## Technical Decisions

### Embedding Model
- **Current**: `BAAI/bge-small-en-v1.5` (384 dims, fast, local)
- **Rationale**: Good enough for semantic search; quality gains come from better chunking (tree-sitter), not larger embeddings

### Vector Store
- **Current**: Qdrant (local mode)
- **Rationale**: No server setup required, file-based persistence

### Parser
- **Current**: tree-sitter via `tree-sitter-language-pack`
- **Rationale**: Language-aware parsing extracts meaningful code units (functions, classes, methods)

### MCP Framework
- **Current**: FastMCP
- **Rationale**: Simple Python decorator-based tool definition, stdio/http transport

---

## Changelog

### 2026-01-06
- Forked from Indexter v0.1.0
- Renamed to indexter-rlm
- Updated Python requirement to >=3.10
- Updated CLI entry point to `indexter-rlm`
- Updated MCP server name to `indexter-rlm`

