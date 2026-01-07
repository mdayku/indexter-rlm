# Indexter-RLM

**RLM-Style Context Environment for Coding Agents**

Indexter-RLM is a semantic code indexing and navigation tool that treats your codebase as an **interactive environment** for AI agents to explore, not a static document to retrieve from.

> *The future of code-aware LLMs is not bigger context windows, but treating the codebase as an environment the model must explore, not a document it must memorize.*

## What Makes This Different

Traditional code RAG:
```
files -> chunk text -> embed -> retrieve top-K -> hope for the best
```

Indexter-RLM:
```
parse semantically -> embed code units -> search -> open -> explore -> accumulate context
```

Key differences:
- **Semantic parsing** via tree-sitter (functions, classes, not text chunks)
- **Incremental indexing** (only changed files)
- **Navigation tools** (read files, list symbols, not just search)
- **Scratchpad** for accumulating observations (planned)
- **MCP integration** for Cursor, Claude, and other AI agents

---

## Features

- **Semantic parsing** using tree-sitter for Python, JavaScript, TypeScript, Rust, HTML, CSS, JSON, YAML, TOML, Markdown
- **Respects .gitignore** and configurable ignore patterns
- **Incremental updates** via content hash comparison
- **Vector search** powered by Qdrant with FastEmbed
- **CLI** for repository management
- **MCP server** for AI agent integration
- **Multi-repo support** with separate collections
- **XDG-compliant** configuration

---

## Installation

> **New to Indexter-RLM?** See [QUICKSTART.md](QUICKSTART.md) for a step-by-step guide to setting up Cursor integration.

### From Source (recommended for now)

```bash
git clone https://github.com/mdayku/indexter-rlm.git
cd indexter-rlm
pip install -e ".[full]"
```

### Requirements
- Python >= 3.10

---

## Quickstart

```bash
# 1. Register a repository
indexter-rlm init /path/to/your/repo

# 2. Index it
indexter-rlm index your-repo

# 3. Search semantically
indexter-rlm search "where is authentication handled?" your-repo

# 4. Check status
indexter-rlm status
```

---

## CLI Reference

| Command | Description |
|---------|-------------|
| `indexter-rlm init <path>` | Register a git repository |
| `indexter-rlm index <name> [--full]` | Index/re-index a repository |
| `indexter-rlm search "<query>" <name>` | Semantic search |
| `indexter-rlm status` | Show all indexed repositories |
| `indexter-rlm forget <name>` | Remove repository from index |
| `indexter-rlm config show` | Show current configuration |
| `indexter-rlm hook install` | Install git hook for auto-indexing |
| `indexter-rlm hook uninstall` | Remove indexter-rlm git hooks |
| `indexter-rlm hook status` | Show git hook status |

---

## MCP Server

Indexter-RLM provides an MCP server for AI agent integration.

### Available Tools

| Tool | Description |
|------|-------------|
| `list_repositories()` | List all indexed repos with status |
| `search_repository(name, query, ...)` | Semantic search with filters |
| `read_file(name, path, start?, end?)` | Read file content with line numbers |
| `get_symbols(name, path)` | List functions/classes in a file |

### Symbol Navigation Tools

| Tool | Description |
|------|-------------|
| `find_references(name, symbol)` | Find all usages of a symbol with import chains |
| `find_definition(name, symbol)` | Jump to where a symbol is defined |
| `list_definitions(name, path)` | List all symbols defined in a file |

### Scratchpad Tools

| Tool | Description |
|------|-------------|
| `save_note(name, key, content)` | Save observations |
| `retrieve_note(name, key)` | Get a single note |
| `list_notes(name, tag?)` | List all notes |
| `remove_note(name, key)` | Delete a note |
| `remove_all_notes(name)` | Clear all notes |
| `exploration_summary(name)` | Get session statistics |

### Cursor Integration

Add to your Cursor MCP settings:

```json
{
  "mcpServers": {
    "indexter-rlm": {
      "command": "rlm-mcp"
    }
  }
}
```

Add to your `.cursorrules`:

```markdown
## Indexter Integration

For any task involving:
- Understanding existing code
- Modifying non-trivial logic
- Debugging behavior
- Refactoring

You MUST:
1. Query Indexter via MCP with a semantic search
2. Review the returned code snippets before proposing changes
3. Cite the retrieved code in your reasoning
```

---

## Configuration

### Global Settings

Location: `~/.config/indexter/indexter.toml`

```toml
max_file_size = 1048576  # 1 MB
max_files = 1000
top_k = 10

[embedding]
provider = "local"  # or "openai"
model = "BAAI/bge-small-en-v1.5"

[store]
mode = "local"  # or "remote" or "memory"

[mcp]
transport = "stdio"  # or "http"
```

### Embedding Models

| Provider | Model | Dimensions | Notes |
|----------|-------|------------|-------|
| local | `BAAI/bge-small-en-v1.5` | 384 | Default, fast |
| local | `BAAI/bge-base-en-v1.5` | 768 | Better quality |
| local | `BAAI/bge-large-en-v1.5` | 1024 | Best quality |
| openai | `text-embedding-3-small` | 1536 | Fast, requires API key |
| openai | `text-embedding-3-large` | 3072 | Best quality |

To use OpenAI embeddings, set `OPENAI_API_KEY` environment variable:

```toml
[embedding]
provider = "openai"
model = "text-embedding-3-small"
```

**Note**: Changing the embedding model requires re-indexing (`indexter-rlm index <name> --full`).

### Per-Repository Settings

Create `indexter.toml` in your repo root, or add `[tool.indexter]` to `pyproject.toml`:

```toml
[tool.indexter]
ignore_patterns = ["generated/", "vendor/"]
max_files = 500
```

### Git Hooks (Auto-Indexing)

Keep the semantic index in sync with your repository automatically:

```bash
# Install post-commit hook (recommended - runs in background)
indexter-rlm hook install /path/to/repo

# Or use pre-push hook (batches multiple commits)
indexter-rlm hook install /path/to/repo --type pre-push

# Check hook status
indexter-rlm hook status

# Remove hooks
indexter-rlm hook uninstall
```

| Hook Type | When It Runs | Blocking? | Best For |
|-----------|--------------|-----------|----------|
| `post-commit` | After each commit | No (background) | Daily development |
| `pre-push` | Before pushing | Yes | Ensuring sync before sharing |
| `pre-commit` | Before each commit | Yes | Strict sync (can slow commits) |

---

## Supported Languages

| Language | Extensions | Semantic Units |
|----------|------------|----------------|
| Python | `.py` | Functions, classes, methods + docstrings |
| JavaScript | `.js`, `.jsx` | Functions, classes + JSDoc |
| TypeScript | `.ts`, `.tsx` | Functions, classes, interfaces + TSDoc |
| Rust | `.rs` | Functions, structs, enums, traits + doc comments |
| HTML | `.html` | Semantic elements |
| CSS | `.css` | Rule sets, media queries |
| JSON | `.json` | Objects, arrays |
| YAML | `.yaml`, `.yml` | Block mappings |
| TOML | `.toml` | Tables, arrays |
| Markdown | `.md` | Headings with sections |
| *Fallback* | `*` | Fixed-size chunks |

---

## Roadmap

See [PRD.md](PRD.md) for vision and [BACKLOG.md](BACKLOG.md) for development tasks.

### Phase 1: Navigation Tools (Current)
- [x] Semantic search
- [ ] File reading
- [ ] Symbol listing

### Phase 2: Stateful Context
- [ ] Scratchpad (notes)
- [ ] Session persistence

### Phase 3: Recursive Controller
- [ ] Enforced exploration workflow
- [ ] Exploration logging

---

## Development

```bash
# Install with dev dependencies
pip install -e ".[full]"

# Run tests
pytest

# Lint
ruff check . --fix
ruff format .
```

---

## License

MIT - see [LICENSE](LICENSE)

---

## Credits

Forked from [Indexter](https://github.com/jdbadger/indexter) by Joe Badger.
