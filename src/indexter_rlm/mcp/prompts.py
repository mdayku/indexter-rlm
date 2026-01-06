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

### Symbol Navigation Tools

### `find_references(name, symbol_name, include_imports?)`
Find all usages of a symbol across the repository.
Returns definitions, references, and import chains.

### `find_definition(name, symbol_name)`
Jump to where a symbol is defined.
Returns file path, line number, and documentation.

### `list_definitions(name, file_path)`
List all symbol definitions in a file.
More detailed than get_symbols - includes qualified names.

### Scratchpad Tools (Notes)

These tools let you accumulate observations across exploration steps:

### `save_note(name, key, content, tags?)`
Store an observation or finding. Notes persist across sessions.
- `key`: Unique identifier (overwrites if exists)
- `content`: Your observation
- `tags`: Optional list for categorization

### `retrieve_note(name, key)`
Get a single note by key.

### `list_notes(name, tag?)`
List all notes, optionally filtered by tag.

### `remove_note(name, key)` / `remove_all_notes(name)`
Clean up notes when done.

## Example: Understanding a Feature

```
# 1. Search for the feature
results = search_repository("my-repo", "user authentication")

# 2. See what symbols are in a relevant file
symbols = get_symbols("my-repo", "src/auth/handler.py")

# 3. Read the specific function
content = read_file("my-repo", "src/auth/handler.py", 45, 80)

# 4. Record your finding
save_note("my-repo", "auth_flow", "Uses JWT with 24h expiry", ["auth"])

# 5. Search for related code
more = search_repository("my-repo", "password validation")

# 6. Review accumulated knowledge
notes = list_notes("my-repo")
```

## Key Principles

1. **Search before reasoning** - Don't guess, find the code
2. **Open before concluding** - Read the actual implementation
3. **Take notes** - Record observations as you explore
4. **Explore iteratively** - One search leads to another
5. **Cite your sources** - Reference the files/lines you found
"""


DEBUG_WORKFLOW_PROMPT = """\
# Debugging Workflow with Indexter-RLM

Use this workflow when tracking down bugs or unexpected behavior.

## The Debug Loop

1. **Identify** - Search for the symptom or error
2. **Trace** - Follow the execution path
3. **Narrow** - Isolate the problematic code
4. **Verify** - Confirm the root cause
5. **Document** - Record findings before fixing

## Step-by-Step Process

### Step 1: Find the Error Origin
```
# Search for the error message or symptom
search_repository("repo", "TypeError: cannot read property")

# Or search for the failing functionality
search_repository("repo", "user login validation")
```

### Step 2: Understand the Call Chain
```
# Get symbols in the relevant file
symbols = get_symbols("repo", "src/auth/login.py")

# Read the function that throws
content = read_file("repo", "src/auth/login.py", 45, 80)

# Note the call chain
save_note("repo", "call_chain", "login() -> validate_user() -> check_token()")
```

### Step 3: Trace Data Flow
```
# Search for where the problematic value originates
search_repository("repo", "user token generation")

# Search for where it's consumed
search_repository("repo", "token validation check")
```

### Step 4: Check Edge Cases
```
# Search for error handling
search_repository("repo", "token expired handling", node_type="function")

# Look for related tests
search_repository("repo", "test login invalid token")
```

### Step 5: Document Before Fixing
```
# Record root cause
save_note("repo", "root_cause", "Token expiry check uses wrong timezone", ["bug"])

# Record affected files
save_note("repo", "affected_files", "src/auth/token.py:45, src/auth/login.py:78")

# Review all findings
list_notes("repo", tag="bug")
```

## Common Debug Queries

| Symptom | Query |
|---------|-------|
| Null/undefined errors | `"null check" OR "undefined handling"` |
| Type errors | `"type validation" OR "type conversion"` |
| Race conditions | `"async" AND "await" AND "concurrent"` |
| Memory issues | `"cache" OR "cleanup" OR "dispose"` |
| Auth failures | `"authentication" OR "authorization" OR "permission"` |

## Key Principles

1. **Don't assume** - Search for actual behavior, not expected behavior
2. **Follow the data** - Trace values from origin to error
3. **Check the tests** - Existing tests reveal expected behavior
4. **Note everything** - What you rule out is as important as what you find
"""


REFACTOR_WORKFLOW_PROMPT = """\
# Safe Refactoring Workflow with Indexter-RLM

Use this workflow when restructuring code to ensure nothing breaks.

## The Refactor Loop

1. **Map** - Understand all usages of the target code
2. **Document** - Record current behavior and contracts
3. **Plan** - Design changes with minimal blast radius
4. **Verify** - Check all call sites and dependencies
5. **Execute** - Make changes with full context

## Step-by-Step Process

### Step 1: Map All Usages
```
# Find the symbol you want to refactor
search_repository("repo", "UserService class")

# Find everywhere it's used
search_repository("repo", "UserService", node_type="function")
search_repository("repo", "import UserService")
search_repository("repo", "from user_service import")

# Note all usage locations
save_note("repo", "usages", "Found in: api/routes.py, services/auth.py, tests/")
```

### Step 2: Document Current Contracts
```
# Read the implementation
content = read_file("repo", "src/services/user_service.py")

# Get the full interface
symbols = get_symbols("repo", "src/services/user_service.py")

# Document public API
save_note("repo", "public_api", "Methods: create_user(), get_user(), delete_user()")

# Document return types and side effects
save_note("repo", "contracts", "create_user returns User, writes to DB and cache")
```

### Step 3: Find Tests
```
# Search for existing tests
search_repository("repo", "test UserService")
search_repository("repo", "UserService", file_path="tests/")

# Note test coverage
save_note("repo", "test_coverage", "Unit tests in test_user_service.py, integration in test_api.py")
```

### Step 4: Identify Dependencies
```
# What does this code depend on?
search_repository("repo", "UserService", node_type="class")
read_file("repo", "src/services/user_service.py", 1, 30)  # Check imports

# What depends on this code?
search_repository("repo", "user_service")

# Note the dependency graph
save_note("repo", "dependencies", "Depends: Database, Cache. Dependents: API, Auth")
```

### Step 5: Plan Changes
```
# Review all notes before planning
notes = list_notes("repo")

# Document your refactor plan
save_note("repo", "refactor_plan", '''
1. Extract interface IUserService
2. Update UserService to implement interface
3. Update all imports to use interface
4. Add new CachedUserService implementation
''')
```

## Safety Checklist

Before refactoring, ensure you have:
- [ ] Mapped ALL usages (not just obvious ones)
- [ ] Documented the current public API
- [ ] Found all related tests
- [ ] Identified upstream and downstream dependencies
- [ ] Created a reversible plan

## Key Principles

1. **Map before moving** - Know every usage before changing anything
2. **Preserve contracts** - Public API changes require updating all callers
3. **Tests are documentation** - They show expected behavior
4. **Small steps** - Prefer multiple small refactors over one big one
5. **Verify at each step** - Run tests after each change
"""


# Cursorrules content for MCP resource
CURSORRULES_CONTENT = """\
# Indexter-RLM Agent Rules

## The RLM Loop (MUST FOLLOW)

Before reasoning about code, follow this loop:

1. **SEARCH** - Find relevant code via semantic search
2. **OPEN** - Read the actual source files
3. **RECORD** - Save observations as notes
4. **REFINE** - Search again based on findings
5. **REPEAT** - Until you have sufficient context

## Search-Before-Reasoning

For ANY task involving code understanding, modification, debugging, or refactoring:

1. Query Indexter via `search_repository()` with semantic search
2. Use `get_symbols()` to understand file structure
3. Use `read_file()` to inspect actual implementation
4. Cite retrieved code (file paths, line numbers) in reasoning

**DO NOT** rely on assumptions or memory as a substitute for search results.

## Note-Taking (Multi-Step Tasks)

For tasks with 3+ steps or multiple files:

1. **Before**: Note your exploration plan
2. **During**: Record key findings as you discover them
3. **Before concluding**: Review accumulated notes
4. **After**: Clean up temporary notes

## Available Tools

- `list_repositories()` - List all indexed repos
- `search_repository(name, query, ...)` - Semantic code search
- `read_file(name, path, start?, end?)` - Read file content
- `get_symbols(name, path)` - List functions/classes in file
- `save_note(name, key, content, tags?)` - Store observation
- `retrieve_note(name, key)` - Get single note
- `list_notes(name, tag?)` - List all notes
- `remove_note(name, key)` - Delete note
- `remove_all_notes(name)` - Clear all notes
"""


def get_search_workflow() -> str:
    """Get the RLM exploration workflow prompt template."""
    return SEARCH_WORKFLOW_PROMPT


def get_debug_workflow() -> str:
    """Get the debugging workflow prompt template."""
    return DEBUG_WORKFLOW_PROMPT


def get_refactor_workflow() -> str:
    """Get the refactoring workflow prompt template."""
    return REFACTOR_WORKFLOW_PROMPT


def get_cursorrules() -> str:
    """Get the cursorrules content for agents."""
    return CURSORRULES_CONTENT
