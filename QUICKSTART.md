# Indexter-RLM Quickstart

Get semantic code navigation for your AI agent in under 5 minutes.

---

## 1. Install Indexter-RLM

```bash
# Clone and install
git clone https://github.com/mdayku/indexter-rlm.git
cd indexter-rlm
pip install -e ".[full]"
```

Verify installation:

```bash
indexter-rlm --version
```

---

## 2. Index Your Codebase

```bash
# Navigate to your project
cd /path/to/your/project

# Register the repository
indexter-rlm init .

# Index it (first run downloads embedding model ~30MB)
indexter-rlm index your-project-name
```

Check status:

```bash
indexter-rlm status
```

You should see something like:

```
Repositories:
  your-project-name
    Path: /path/to/your/project
    Indexed: 142 nodes, 87 documents
    Stale: 0
```

### Optional: Enable Auto-Indexing

Keep the index in sync automatically:

```bash
# Install a post-commit hook (runs in background after each commit)
indexter-rlm hook install /path/to/your/project
```

---

## 3. Configure Cursor MCP

Create or edit your Cursor MCP settings file:

| OS | Location |
|----|----------|
| **Windows** | `%APPDATA%\Cursor\User\globalStorage\saoudrizwan.claude-dev\settings\cline_mcp_settings.json` |
| **macOS** | `~/Library/Application Support/Cursor/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json` |
| **Linux** | `~/.config/Cursor/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json` |

**Alternative**: Use Cursor Settings UI → Extensions → Claude Dev → MCP Servers

Add the indexter-rlm server:

```json
{
  "mcpServers": {
    "indexter-rlm": {
      "command": "rlm-mcp"
    }
  }
}
```

> **Note**: If you installed in a virtual environment, use the full path to `rlm-mcp`:
> ```json
> {
>   "mcpServers": {
>     "indexter-rlm": {
>       "command": "/path/to/venv/bin/rlm-mcp"
>     }
>   }
> }
> ```

Restart Cursor to load the MCP server.

---

## 4. Add .cursorrules to Your Project

Create a `.cursorrules` file in your project root to guide the AI agent:

```markdown
## Indexter Integration

Before reasoning about code, follow this loop:

1. **SEARCH** - Find relevant code via `search_repository()`
2. **OPEN** - Read files with `read_file()` or `get_symbols()`
3. **RECORD** - Save observations with `save_note()`
4. **REFINE** - Search again based on findings
5. **REPEAT** - Until you have sufficient context

### Required Behavior

For ANY task involving:
- Understanding existing code
- Modifying non-trivial logic
- Debugging behavior
- Refactoring

You MUST:
1. Query Indexter via `search_repository()` with a semantic search
2. Use `get_symbols()` to understand file structure before reading
3. Use `read_file()` to inspect actual implementation
4. Cite the retrieved code (file paths and line numbers) in your reasoning

**DO NOT** rely on assumptions or memory as a substitute for search results.

### Note-Taking (Multi-Step Tasks)

For complex tasks involving multiple files:

1. Create a plan: `save_note("repo", "plan", "1. Find X 2. Trace Y 3. Check Z")`
2. Record findings: `save_note("repo", "location", "Found in src/auth.py:45-80")`
3. Review before concluding: `list_notes("repo")`
4. Clean up: `remove_all_notes("repo")`
```

---

## 5. Verify It Works

Open your project in Cursor and ask the AI agent:

> "Search for where user authentication is handled"

The agent should:
1. Call `search_repository()` with your query
2. Return semantically relevant code chunks
3. Offer to read specific files for more context

---

## Available MCP Tools

| Tool | Description |
|------|-------------|
| `list_repositories()` | List all indexed repos |
| `search_repository(name, query)` | Semantic code search |
| `read_file(name, path)` | Read file with line numbers |
| `get_symbols(name, path)` | List functions/classes in a file |
| `find_references(name, symbol)` | Find all usages of a symbol |
| `find_definition(name, symbol)` | Jump to symbol definition |
| `save_note(name, key, content)` | Save an observation |
| `list_notes(name)` | List all saved notes |
| `exploration_summary(name)` | Get session statistics |

---

## Troubleshooting

### MCP server not found

Make sure `rlm-mcp` is in your PATH:

```bash
which rlm-mcp  # macOS/Linux
where rlm-mcp  # Windows
```

If not found, use the full path in your MCP config.

### No search results

1. Check that the repo is indexed: `indexter-rlm status`
2. Re-index if needed: `indexter-rlm index your-repo --full`
3. Try broader search terms

### Index out of date

Run incremental update:

```bash
indexter-rlm index your-repo
```

Or install auto-indexing:

```bash
indexter-rlm hook install /path/to/repo
```

---

## Next Steps

- See [README.md](README.md) for full documentation
- Configure embedding models in `~/.config/indexter/indexter.toml`
- Add per-repo settings in `indexter.toml` or `pyproject.toml`

