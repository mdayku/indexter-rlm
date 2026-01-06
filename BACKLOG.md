# Indexter-RLM Development Backlog

## Next Sprint

### Epic 6: Cross-Language Symbol Support

**Goal**: Extend find_references to work across multiple languages.

#### Story 6.1: Language-Specific Symbol Extractors
- [ ] JavaScript/TypeScript symbol extraction
- [ ] Rust symbol extraction
- [ ] Go symbol extraction

#### Story 6.2: Cross-Language Imports
- [ ] Track cross-language references (e.g., Python calling Rust via PyO3)
- [ ] Unified import graph across languages

---

### Epic 7: Evaluation Framework

**Goal**: Measure RLM approach vs baselines.

#### Story 7.1: Task Suite
- [ ] Multi-hop tracing tasks
- [ ] Refactor understanding tasks
- [ ] Invariant discovery tasks

#### Story 7.2: Metrics Collection
- [ ] Correctness scoring
- [ ] Hallucination detection
- [ ] Token usage tracking
- [ ] Tool call counting

#### Story 7.3: Baseline Comparison
- [ ] Naive large-context prompting
- [ ] Single-shot RAG
- [ ] No-retrieval baseline

---

## Completed

### Epic 5: Find References (2026-01-06)
- [x] Created `symbols.py` with SymbolIndex, SymbolDefinition, SymbolReference, ImportRelation
- [x] Created `symbol_extractor.py` for Python symbol extraction
- [x] Semantic reference tracking (not just string matching)
- [x] Import chain tracking with `get_import_chain()`
- [x] Added `find_references` MCP tool
- [x] Added `find_definition` MCP tool
- [x] Added `list_definitions` MCP tool
- [x] Per-repo JSON persistence at `~/.config/indexter/symbols/{repo}.json`
- [x] 28 tests for symbol index and extractor
- [x] 7 tests for MCP tools

### Git Hooks Feature (2026-01-06)
- [x] Added `hook install` CLI command (post-commit, pre-push, pre-commit)
- [x] Added `hook uninstall` CLI command
- [x] Added `hook status` CLI command
- [x] 18 tests for hook functionality
- [x] Documentation in README.md

### Epic 4: Embedding Model Options (2026-01-06)
- [x] Added `embedding.provider` config (`local` | `openai`)
- [x] Added `embedding.model` config for model selection
- [x] Created `embeddings.py` with `LocalEmbedder` and `OpenAIEmbedder`
- [x] Support bge-small/base/large (local) and text-embedding-3-small/large (OpenAI)
- [x] Fallback to local if no OpenAI API key
- [x] Added 12 tests for embedding providers

### Epic 3: Recursive Controller (2026-01-06)
- [x] Enhanced `.cursorrules` with RLM enforcement
- [x] Added DEBUG_WORKFLOW_PROMPT and REFACTOR_WORKFLOW_PROMPT
- [x] Added MCP resource `cursorrules://indexter-rlm`
- [x] Created ExplorationLogger with JSON Lines logging
- [x] Added `exploration_summary` tool

### Epic 2: Stateful Context (2026-01-06)
- [x] Added note tools: save_note, retrieve_note, list_notes, remove_note, remove_all_notes
- [x] Per-repo JSON persistence at `~/.config/indexter/notes/{repo}.json`
- [x] Added 10 tests for note functionality

### Epic 1: Navigation Tools (2026-01-06)
- [x] Added `read_file` tool with line range filtering
- [x] Added `get_symbols` tool leveraging tree-sitter parsers

### Initial Fork (2026-01-06)
- [x] Fork from Indexter v0.1.0
- [x] Rename package to `indexter_rlm`
- [x] Update Python requirement to >=3.10
- [x] Update CLI to `indexter-rlm`
- [x] Update MCP server name
- [x] Initialize git repo
- [x] Verify installation works

