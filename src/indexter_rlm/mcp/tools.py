"""
MCP tool implementations for Indexter-RLM.

Tools perform actions and can mutate state.
Provides semantic search, file navigation, and context management.
"""

from pathlib import Path

from anyio import create_task_group

from indexter_rlm.models import Repo
from indexter_rlm.parsers import get_parser

from .exploration_log import get_exploration_logger
from .notes import get_note_store


async def list_repos() -> list[dict]:
    """
    List all Indexter-configured repositories.

    Returns:
        A list of dictionaries, each containing status information for
        a configured repository. Each dict includes keys: 'name', 'path',
        number of nodes indexed, number of documents indexed, and number of stale
        documents in the index.
    """
    repos = await Repo.list()
    statuses = []

    async def _add_status(repo):
        status = await repo.status()
        statuses.append(status)

    async with create_task_group() as tg:
        for repo in repos:
            tg.start_soon(_add_status, repo)

    return statuses


async def search_repo(
    name: str,
    query: str,
    file_path: str | None = None,
    language: str | None = None,
    node_type: str | None = None,
    node_name: str | None = None,
    has_documentation: bool | None = None,
    limit: int | None = None,
) -> dict:
    """
    Perform semantic search across an Indexter-configured repository's indexed code.

    Search uses vector embeddings to find semantically similar code
    chunks.

    Args:
        name: The repository name.
        query: Natural language search query.
        file_path: Filter by file path (exact match or prefix with trailing /).
        language: Filter by programming language (e.g., 'python', 'javascript').
        node_type: Filter by node type (e.g., 'function', 'class', 'method').
        node_name: Filter by node name.
        has_documentation: Filter by documentation presence.
        limit: Maximum number of results to return (defaults to 10).
    Returns:
        Dict with results list containing matched code chunks with scores.

    Raises:
        RepoNotFoundError: If the specified repository is not found.
    """
    repo = await Repo.get(name)

    # Ensure the index is up to date before searching
    await repo.index()

    # Use repo settings top_k if available, otherwise default to 10
    default_limit = repo.settings.top_k if repo.settings else 10
    limit = limit if limit is not None else default_limit

    results = await repo.search(
        query=query,
        file_path=file_path,
        language=language,
        node_type=node_type,
        node_name=node_name,
        has_documentation=has_documentation,
        limit=limit,
    )

    # Log the search
    exploration_logger = get_exploration_logger(name)
    exploration_logger.log(
        tool="search_repository",
        args={
            "query": query,
            "file_path": file_path,
            "language": language,
            "node_type": node_type,
            "limit": limit,
        },
        result_summary={"count": len(results)},
    )

    return {"results": results, "count": len(results)}


async def get_file(
    name: str,
    file_path: str,
    start_line: int | None = None,
    end_line: int | None = None,
) -> dict:
    """
    Read file content from an Indexter-configured repository.

    Use this tool to inspect actual source code after finding relevant files
    via search. Supports optional line range filtering for large files.

    Args:
        name: The repository name.
        file_path: Path to the file relative to repository root.
        start_line: Optional starting line number (1-based, inclusive).
        end_line: Optional ending line number (1-based, inclusive).

    Returns:
        Dict with file content, path, line range, and total line count.

    Raises:
        RepoNotFoundError: If the specified repository is not found.
        FileNotFoundError: If the file does not exist.
    """
    repo = await Repo.get(name)
    repo_path = Path(repo.path)
    full_path = repo_path / file_path

    if not full_path.exists():
        return {
            "error": f"File not found: {file_path}",
            "repository": name,
            "file_path": file_path,
        }

    if not full_path.is_file():
        return {
            "error": f"Path is not a file: {file_path}",
            "repository": name,
            "file_path": file_path,
        }

    try:
        content = full_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return {
            "error": f"Cannot read file (binary or encoding issue): {file_path}",
            "repository": name,
            "file_path": file_path,
        }

    lines = content.splitlines()
    total_lines = len(lines)

    # Apply line range filtering
    if start_line is not None or end_line is not None:
        start_idx = (start_line - 1) if start_line else 0
        end_idx = end_line if end_line else total_lines
        start_idx = max(0, start_idx)
        end_idx = min(total_lines, end_idx)
        lines = lines[start_idx:end_idx]
        actual_start = start_idx + 1
        actual_end = end_idx
    else:
        actual_start = 1
        actual_end = total_lines

    # Format with line numbers
    numbered_lines = [f"{i:4d} | {line}" for i, line in enumerate(lines, start=actual_start)]
    formatted_content = "\n".join(numbered_lines)

    # Log the file read
    exploration_logger = get_exploration_logger(name)
    exploration_logger.log(
        tool="read_file",
        args={
            "file_path": file_path,
            "start_line": start_line,
            "end_line": end_line,
        },
        result_summary={"lines_read": actual_end - actual_start + 1, "total_lines": total_lines},
    )

    return {
        "repository": name,
        "file_path": file_path,
        "content": formatted_content,
        "start_line": actual_start,
        "end_line": actual_end,
        "total_lines": total_lines,
    }


async def list_symbols(
    name: str,
    file_path: str,
) -> dict:
    """
    List all symbols (functions, classes, methods) in a file.

    Use this tool to get an overview of a file's structure before
    reading specific sections with get_file.

    Args:
        name: The repository name.
        file_path: Path to the file relative to repository root.

    Returns:
        Dict with list of symbols, each containing name, type, line, and signature.

    Raises:
        RepoNotFoundError: If the specified repository is not found.
        FileNotFoundError: If the file does not exist.
    """
    repo = await Repo.get(name)
    repo_path = Path(repo.path)
    full_path = repo_path / file_path

    if not full_path.exists():
        return {
            "error": f"File not found: {file_path}",
            "repository": name,
            "file_path": file_path,
        }

    if not full_path.is_file():
        return {
            "error": f"Path is not a file: {file_path}",
            "repository": name,
            "file_path": file_path,
        }

    try:
        content = full_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return {
            "error": f"Cannot read file (binary or encoding issue): {file_path}",
            "repository": name,
            "file_path": file_path,
        }

    # Get parser for this file type
    parser = get_parser(file_path)
    if parser is None:
        return {
            "error": f"No parser available for file type: {file_path}",
            "repository": name,
            "file_path": file_path,
            "symbols": [],
        }

    # Parse and extract symbols
    symbols = []
    try:
        for _content, metadata in parser.parse(content):
            symbol = {
                "name": metadata.get("node_name", "unknown"),
                "type": metadata.get("node_type", "unknown"),
                "line": metadata.get("start_line", 0),
                "end_line": metadata.get("end_line", 0),
            }
            # Add signature if available
            if signature := metadata.get("signature"):
                symbol["signature"] = signature
            # Add parent scope if available (e.g., class for methods)
            if parent := metadata.get("parent_scope"):
                symbol["parent"] = parent
            symbols.append(symbol)
    except Exception as e:
        return {
            "error": f"Failed to parse file: {e}",
            "repository": name,
            "file_path": file_path,
            "symbols": [],
        }

    # Sort by line number
    symbols.sort(key=lambda s: s["line"])

    return {
        "repository": name,
        "file_path": file_path,
        "symbols": symbols,
        "count": len(symbols),
    }


# =============================================================================
# Note Tools (Scratchpad)
# =============================================================================


async def store_note(
    name: str,
    key: str,
    content: str,
    tags: list[str] | None = None,
) -> dict:
    """
    Store a note in the repository's scratchpad.

    Use this to record observations, hypotheses, and findings during
    code exploration. Notes persist across sessions.

    Args:
        name: The repository name.
        key: Unique identifier for the note (overwrites if exists).
        content: The note content.
        tags: Optional list of tags for categorization.

    Returns:
        Dict with the stored note details.
    """
    # Validate repo exists
    await Repo.get(name)

    note_store = get_note_store(name)
    note = note_store.store(key=key, content=content, tags=tags)

    # Log the note creation
    exploration_logger = get_exploration_logger(name)
    exploration_logger.log(
        tool="save_note",
        args={"key": key, "tags": tags},
        result_summary={"content_length": len(content)},
    )

    return {
        "repository": name,
        "action": "stored",
        "note": note.model_dump_for_display(),
    }


async def get_note(
    name: str,
    key: str,
) -> dict:
    """
    Retrieve a single note by key.

    Args:
        name: The repository name.
        key: The note key to retrieve.

    Returns:
        Dict with the note details, or error if not found.
    """
    # Validate repo exists
    await Repo.get(name)

    note_store = get_note_store(name)
    note = note_store.get(key)

    if note is None:
        return {
            "repository": name,
            "error": f"Note not found: {key}",
        }

    return {
        "repository": name,
        "note": note.model_dump_for_display(),
    }


async def get_notes(
    name: str,
    tag: str | None = None,
) -> dict:
    """
    List all notes for a repository, optionally filtered by tag.

    Args:
        name: The repository name.
        tag: Optional tag to filter notes by.

    Returns:
        Dict with list of notes and count.
    """
    # Validate repo exists
    await Repo.get(name)

    note_store = get_note_store(name)
    notes = note_store.list(tag=tag)

    return {
        "repository": name,
        "filter_tag": tag,
        "notes": [n.model_dump_for_display() for n in notes],
        "count": len(notes),
    }


async def delete_note(
    name: str,
    key: str,
) -> dict:
    """
    Delete a note by key.

    Args:
        name: The repository name.
        key: The note key to delete.

    Returns:
        Dict indicating success or failure.
    """
    # Validate repo exists
    await Repo.get(name)

    note_store = get_note_store(name)
    deleted = note_store.delete(key)

    if deleted:
        return {
            "repository": name,
            "action": "deleted",
            "key": key,
        }
    else:
        return {
            "repository": name,
            "error": f"Note not found: {key}",
        }


async def clear_notes(
    name: str,
) -> dict:
    """
    Delete all notes for a repository.

    Args:
        name: The repository name.

    Returns:
        Dict with count of deleted notes.
    """
    # Validate repo exists
    await Repo.get(name)

    note_store = get_note_store(name)
    count = note_store.clear()

    return {
        "repository": name,
        "action": "cleared",
        "notes_deleted": count,
    }


# =============================================================================
# Exploration Logging
# =============================================================================


async def get_exploration_summary(
    name: str,
) -> dict:
    """
    Get a summary of the current exploration session.

    Returns statistics about tool calls, searches, and files read
    during the current session.

    Args:
        name: The repository name.

    Returns:
        Dict with session statistics.
    """
    # Validate repo exists
    await Repo.get(name)

    exploration_logger = get_exploration_logger(name)
    summary = exploration_logger.get_summary()

    return summary


# =============================================================================
# Symbol Navigation Tools
# =============================================================================


async def find_symbol_references(
    name: str,
    symbol_name: str,
    include_imports: bool = True,
) -> dict:
    """
    Find all references to a symbol across the repository.

    Args:
        name: The repository name.
        symbol_name: Name of the symbol to find references for.
        include_imports: Include import chains (default True).

    Returns:
        Dict with definitions, references, and import chains.
    """
    from indexter_rlm.symbol_extractor import build_symbol_index
    from indexter_rlm.walker import Walker

    repo = await Repo.get(name)
    repo_path = Path(repo.path)

    # Get list of Python files
    walker = Walker(repo)
    documents = [doc async for doc in walker.walk()]
    py_files = [doc["path"] for doc in documents if doc["path"].endswith(".py")]

    # Build/update symbol index
    index = await build_symbol_index(repo.name, repo_path, py_files)

    # Find definitions
    definitions = index.find_definitions(symbol_name)
    definition_results = [
        {
            "name": d.name,
            "qualified_name": d.qualified_name,
            "type": d.symbol_type,
            "file_path": d.file_path,
            "line": d.line,
            "end_line": d.end_line,
            "signature": d.signature,
            "documentation": d.documentation[:500] if d.documentation else "",
        }
        for d in definitions
    ]

    # Find references
    references = index.find_references(symbol_name)
    reference_results = [
        {
            "file_path": r.file_path,
            "line": r.line,
            "column": r.column,
            "context": r.context,
            "ref_type": r.ref_type,
        }
        for r in references
    ]

    result = {
        "symbol_name": symbol_name,
        "repository": name,
        "definitions": definition_results,
        "definitions_count": len(definition_results),
        "references": reference_results,
        "references_count": len(reference_results),
    }

    # Include import chains if requested
    if include_imports:
        import_chains = index.get_import_chain(symbol_name)
        result["import_chains"] = import_chains
        result["import_chains_count"] = len(import_chains)

    # Log the exploration
    exploration_logger = get_exploration_logger(name)
    exploration_logger.log(
        tool="find_references",
        args={"symbol_name": symbol_name},
        result_summary={
            "definitions": len(definition_results),
            "references": len(reference_results),
        },
    )

    return result


async def find_symbol_definition(
    name: str,
    symbol_name: str,
) -> dict:
    """
    Find where a symbol is defined.

    Args:
        name: The repository name.
        symbol_name: Name of the symbol to find.

    Returns:
        Dict with definition locations.
    """
    from indexter_rlm.symbol_extractor import build_symbol_index
    from indexter_rlm.walker import Walker

    repo = await Repo.get(name)
    repo_path = Path(repo.path)

    # Get list of Python files
    walker = Walker(repo)
    documents = [doc async for doc in walker.walk()]
    py_files = [doc["path"] for doc in documents if doc["path"].endswith(".py")]

    # Build/update symbol index
    index = await build_symbol_index(repo.name, repo_path, py_files)

    # Find definitions
    definitions = index.find_definitions(symbol_name)

    if not definitions:
        return {
            "symbol_name": symbol_name,
            "repository": name,
            "error": f"No definition found for '{symbol_name}'",
            "definitions": [],
        }

    return {
        "symbol_name": symbol_name,
        "repository": name,
        "definitions": [
            {
                "name": d.name,
                "qualified_name": d.qualified_name,
                "type": d.symbol_type,
                "file_path": d.file_path,
                "line": d.line,
                "end_line": d.end_line,
                "signature": d.signature,
                "documentation": d.documentation[:500] if d.documentation else "",
            }
            for d in definitions
        ],
        "count": len(definitions),
    }


async def list_file_symbols(
    name: str,
    file_path: str,
) -> dict:
    """
    List all symbols defined in a file.

    Args:
        name: The repository name.
        file_path: Path to the file.

    Returns:
        Dict with list of symbols in the file.
    """
    from indexter_rlm.symbol_extractor import build_symbol_index

    repo = await Repo.get(name)
    repo_path = Path(repo.path)

    # Build/update symbol index for this file
    index = await build_symbol_index(repo.name, repo_path, [file_path])

    # Get symbols for this file
    symbol_names = index.file_symbols.get(file_path, [])

    symbols = []
    for sym_name in symbol_names:
        for defn in index.find_definitions(sym_name):
            if defn.file_path == file_path:
                symbols.append(
                    {
                        "name": defn.name,
                        "qualified_name": defn.qualified_name,
                        "type": defn.symbol_type,
                        "line": defn.line,
                        "signature": defn.signature,
                    }
                )

    # Sort by line number
    symbols.sort(key=lambda s: s["line"])

    return {
        "repository": name,
        "file_path": file_path,
        "symbols": symbols,
        "count": len(symbols),
    }
