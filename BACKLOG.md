# Indexter-RLM Development Backlog

## Current Sprint

### Epic 1: Navigation Tools (Phase 1)

**Goal**: Transform Indexter from "search only" to "explore" - add tools that let agents navigate the codebase.

#### Story 1.1: Get File Tool
Add MCP tool to read actual file content.

- [ ] **Task 1.1.1**: Add `get_file(path, start_line?, end_line?)` to `mcp/tools.py`
- [ ] **Task 1.1.2**: Register tool in `mcp/server.py` with `@mcp.tool()`
- [ ] **Task 1.1.3**: Add tests in `mcp/tests/test_tools.py`
- [ ] **Task 1.1.4**: Update `mcp/prompts.py` workflow guide

**Acceptance Criteria**:
- Agent can read file content by path
- Optional line range filtering
- Returns file content with line numbers
- Graceful error handling for missing files

#### Story 1.2: List Symbols Tool
Add MCP tool to list functions/classes in a file.

- [ ] **Task 1.2.1**: Add `list_symbols(repo_name, file_path)` to `mcp/tools.py`
- [ ] **Task 1.2.2**: Leverage existing tree-sitter parsing in `parsers/`
- [ ] **Task 1.2.3**: Return structured list: `[{name, type, line, signature}]`
- [ ] **Task 1.2.4**: Add tests

**Acceptance Criteria**:
- Agent can list all functions/classes in a file
- Returns name, type (function/class/method), line number
- Optionally includes signature

---

## Next Sprint

### Epic 2: Stateful Context (Scratchpad)

**Goal**: Let agents accumulate observations across exploration steps.

#### Story 2.1: Store Note Tool
- [ ] Add `store_note(key, content)` to MCP server
- [ ] In-memory storage per session
- [ ] Overwrite on duplicate key

#### Story 2.2: Get Notes Tool
- [ ] Add `get_notes()` to return all stored notes
- [ ] Add `get_note(key)` for single retrieval

#### Story 2.3: Note Persistence (Optional)
- [ ] Persist notes to disk per repo
- [ ] Clear notes on session end or explicit request

---

## Future Sprints

### Epic 3: Recursive Controller

**Goal**: Enforce the RLM loop via prompts and rules.

#### Story 3.1: Enhanced .cursorrules
- [ ] Add explicit search-before-reasoning enforcement
- [ ] Add note-taking requirements for multi-step tasks

#### Story 3.2: Search Workflow Prompt
- [ ] Update `mcp/prompts.py` with RLM workflow
- [ ] Add examples of recursive exploration

#### Story 3.3: Exploration Logging
- [ ] Log tool calls during exploration
- [ ] Track search queries and results
- [ ] Output exploration summary

---

### Epic 4: Embedding Model Options (Optional)

**Goal**: Support alternative embedding models for quality/cost tradeoffs.

#### Story 4.1: Upgrade to bge-base
- [ ] Test `BAAI/bge-base-en-v1.5` (768 dims vs 384)
- [ ] Document re-indexing requirements
- [ ] Benchmark quality difference

#### Story 4.2: OpenAI Embeddings Support
- [ ] Add `embedding_provider` config option (`local` | `openai`)
- [ ] Implement OpenAI embedding generation in `store.py`
- [ ] Use `OPENAI_API_KEY` from environment
- [ ] Fall back to local if no key
- [ ] Add `text-embedding-3-small` and `text-embedding-3-large` options
- [ ] Update documentation

**Note**: Current model `bge-small-en-v1.5` is good. This is optimization, not critical path.

---

### Epic 5: Find References (Advanced)

**Goal**: Enable cross-file navigation by finding symbol usages.

#### Story 4.1: Symbol Index
- [ ] Index symbol definitions with location
- [ ] Track import relationships

#### Story 4.2: Find References Tool
- [ ] Add `find_references(repo_name, symbol_name)` tool
- [ ] Return list of files/lines where symbol is used

---

### Epic 6: Evaluation Framework

**Goal**: Measure RLM approach vs baselines.

#### Story 5.1: Task Suite
- [ ] Multi-hop tracing tasks
- [ ] Refactor understanding tasks
- [ ] Invariant discovery tasks

#### Story 5.2: Metrics Collection
- [ ] Correctness scoring
- [ ] Hallucination detection
- [ ] Token usage tracking
- [ ] Tool call counting

#### Story 5.3: Baseline Comparison
- [ ] Naive large-context prompting
- [ ] Single-shot RAG
- [ ] No-retrieval baseline

---

## Completed

### Initial Fork (2026-01-06)
- [x] Fork from Indexter v0.1.0
- [x] Rename package to `indexter_rlm`
- [x] Update Python requirement to >=3.10
- [x] Update CLI to `indexter-rlm`
- [x] Update MCP server name
- [x] Initialize git repo
- [x] Verify installation works

