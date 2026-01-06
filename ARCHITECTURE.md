# Indexter-RLM Architecture

## Overview

Indexter-RLM is an RLM-style context environment for coding agents. It provides semantic code search and navigation tools via the Model Context Protocol (MCP).

```
+------------------+     +------------------+     +------------------+
|  AI Agent        |     |  Indexter-RLM    |     |  Codebase        |
|  (Cursor/Claude) | <-> |  MCP Server      | <-> |  (Git Repos)     |
+------------------+     +------------------+     +------------------+
                               |
                               v
                         +------------------+
                         |  Vector Store    |
                         |  (Qdrant local)  |
                         +------------------+
```

---

## System Components

### 1. CLI (`src/indexter_rlm/cli/`)

Command-line interface for repository management.

| Command | Description |
|---------|-------------|
| `init <path>` | Register a git repository |
| `index <name>` | Parse and embed code into vector store |
| `search "<query>" <name>` | Semantic search |
| `status` | Show all indexed repositories |
| `forget <name>` | Remove repository from index |
| `config` | Manage configuration |

**Entry point**: `indexter-rlm` (defined in `pyproject.toml`)

### 2. MCP Server (`src/indexter_rlm/mcp/`)

FastMCP server exposing tools for AI agents.

| Tool | Description |
|------|-------------|
| `list_repositories()` | List all indexed repos with status |
| `search_repository(name, query, ...)` | Semantic search with filters |
| *(planned)* `get_file(path, ...)` | Read file content |
| *(planned)* `list_symbols(file)` | List functions/classes |
| *(planned)* `store_note(key, content)` | Scratchpad storage |

**Entry point**: `rlm-mcp` (defined in `pyproject.toml`)

### 3. Parsers (`src/indexter_rlm/parsers/`)

Tree-sitter based parsers for semantic code chunking.

| Parser | Languages |
|--------|-----------|
| `python.py` | Python |
| `javascript.py` | JavaScript |
| `typescript.py` | TypeScript |
| `rust.py` | Rust |
| `html.py` | HTML |
| `css.py` | CSS |
| `json.py` | JSON |
| `yaml.py` | YAML |
| `toml.py` | TOML |
| `markdown.py` | Markdown |
| `chunk.py` | Fallback text chunking |

**Node types extracted**: functions, classes, methods, interfaces, sections, docstrings

### 4. Store (`src/indexter_rlm/store.py`)

Vector store operations using Qdrant.

| Mode | Description |
|------|-------------|
| `local` | File-based storage (default) |
| `remote` | Docker/cloud Qdrant server |
| `memory` | In-memory for testing |

**Location**: `~/.local/share/indexter/store/`

### 5. Models (`src/indexter_rlm/models.py`)

Core domain models.

| Model | Purpose |
|-------|---------|
| `Repo` | Repository management (init, index, search) |
| `IndexResult` | Indexing operation result |
| `NodeInfo` | Parsed code node metadata |

### 6. Config (`src/indexter_rlm/config.py`)

Configuration management following XDG spec.

| Setting | Default |
|---------|---------|
| `embedding_model` | `BAAI/bge-small-en-v1.5` |
| `max_file_size` | 1 MB |
| `max_files` | 1000 |
| `top_k` | 10 |

**Locations**:
- Global: `~/.config/indexter/indexter.toml`
- Per-repo: `<repo>/indexter.toml` or `pyproject.toml [tool.indexter]`
- Repos registry: `~/.config/indexter/repos.json`

---

## Data Flow

### Indexing Flow

```
1. User runs: indexter-rlm index <repo>
              |
              v
2. Walker scans files (respecting .gitignore + ignore_patterns)
              |
              v
3. Parser extracts semantic nodes via tree-sitter
   - Functions, classes, methods, docstrings
              |
              v
4. Embeddings generated via FastEmbed (bge-small-en-v1.5)
              |
              v
5. Vectors stored in Qdrant collection (indexter_<repo_name>)
```

### Search Flow

```
1. Agent calls: search_repository(name, query, ...)
              |
              v
2. Query embedded using same model
              |
              v
3. Qdrant performs similarity search
              |
              v
4. Top-K results returned with:
   - Score
   - Content (code snippet)
   - File path
   - Metadata (node type, language, line numbers)
```

### RLM Exploration Flow (Target State)

```
1. Agent receives task
              |
              v
2. Agent calls: search_repository("how does X work?")
              |
              v
3. Agent reviews results, calls: get_file(path) for details
              |
              v
4. Agent stores observations: store_note("key", "insight")
              |
              v
5. Agent refines query based on gaps
              |
              v
6. Repeat until confident or budget exhausted
              |
              v
7. Agent synthesizes answer from accumulated notes
```

---

## Directory Structure

```
indexter-rlm/
+-- pyproject.toml          # Package config, dependencies, entry points
+-- src/indexter_rlm/
|   +-- __init__.py         # Package init, version
|   +-- config.py           # Settings, XDG paths
|   +-- exceptions.py       # Custom exceptions
|   +-- models.py           # Repo, IndexResult, NodeInfo
|   +-- store.py            # Qdrant operations
|   +-- utils.py            # Utility functions
|   +-- walker.py           # File system walker
|   +-- cli/
|   |   +-- cli.py          # Main CLI commands
|   |   +-- config.py       # Config subcommands
|   +-- mcp/
|   |   +-- server.py       # MCP server + tool definitions
|   |   +-- tools.py        # Tool implementations
|   |   +-- prompts.py      # Prompt templates
|   +-- parsers/
|       +-- base.py         # Base parser class
|       +-- python.py       # Python parser
|       +-- ...             # Other language parsers
+-- scripts/
|   +-- build_analysis_doc.py  # Generate codebase analysis doc
+-- .cursorrules            # Agent behavior rules
+-- PRD.md                  # Product requirements
+-- BACKLOG.md              # Development backlog
+-- ARCHITECTURE.md         # This file
```

---

## Key Decisions

### Why Tree-Sitter?
- Language-aware parsing (not just text splitting)
- Extracts meaningful units (functions, classes, methods)
- Consistent structure across languages
- Fast and battle-tested

### Why Qdrant?
- Local mode requires no server setup
- Good Python client
- Supports filtering on metadata
- Scales to remote/cloud when needed

### Why FastMCP?
- Simple decorator-based tool definition
- Handles stdio/http transport
- Clean integration with async Python

### Why bge-small-en-v1.5?
- Small enough for local use (~130 MB)
- Good quality for code search
- Fast embedding generation
- Quality gains come from better chunking, not larger models

---

## Extension Points

### Adding a New Parser
1. Create `parsers/<language>.py` inheriting from `BaseParser`
2. Implement `parse(content: str) -> list[NodeInfo]`
3. Register in `parsers/__init__.py`
4. Add tests in `parsers/tests/`

### Adding a New MCP Tool
1. Implement function in `mcp/tools.py`
2. Add `@mcp.tool()` wrapper in `mcp/server.py`
3. Update `mcp/prompts.py` workflow guide
4. Add tests in `mcp/tests/`

### Adding a New CLI Command
1. Add function with `@app.command()` in `cli/cli.py`
2. Follow existing patterns for error handling + output
3. Add tests in `cli/tests/`

