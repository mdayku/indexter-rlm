"""
MCP prompt implementations for Indexter-RLM.

Prompts provide reusable templates for RLM-style exploration workflows.
"""

SEARCH_WORKFLOW_PROMPT = """\
# Indexter-RLM Exploration Workflow

Use Indexter-RLM as an **environment to explore**, not just a retriever.
Follow this recursive loop for understanding code:

## The RLM Loop

1. **Search** - Find relevant code by meaning
2. **Open** - Read the actual source code
3. **Understand** - List symbols to see file structure
4. **Refine** - Search again based on what you learned
5. **Repeat** - Until you have enough context

## Available Tools

### `list_repositories()`
List all indexed repos with status (nodes, files, stale count).

### `search_repository(name, query, ...)`
Semantic search for code by meaning. Filters:
- `file_path`: Limit to directory (use trailing `/` for prefix)
- `language`: Filter by language ('python', 'javascript', etc.)
- `node_type`: Filter by structure ('function', 'class', 'method')
- `node_name`: Filter by symbol name
- `has_documentation`: Find documented/undocumented code
- `limit`: Max results (default: 10)

### `read_file(name, file_path, start_line?, end_line?)`
Read actual file content after finding it via search.
Use line ranges for large files.

### `get_symbols(name, file_path)`
List all functions, classes, methods in a file.
Use to understand structure before reading.

## Example: Understanding a Feature

```
# 1. Search for the feature
results = search_repository("my-repo", "user authentication")

# 2. See what symbols are in a relevant file
symbols = get_symbols("my-repo", "src/auth/handler.py")

# 3. Read the specific function
content = read_file("my-repo", "src/auth/handler.py", 45, 80)

# 4. Search for related code
more = search_repository("my-repo", "password validation")

# 5. Continue exploring until you understand
```

## Key Principles

1. **Search before reasoning** - Don't guess, find the code
2. **Open before concluding** - Read the actual implementation
3. **Explore iteratively** - One search leads to another
4. **Cite your sources** - Reference the files/lines you found
"""


def get_search_workflow() -> str:
    """Get the RLM exploration workflow prompt template."""
    return SEARCH_WORKFLOW_PROMPT
