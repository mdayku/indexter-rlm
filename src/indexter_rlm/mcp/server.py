"""
Indexter-RLM MCP Server.

A FastMCP server exposing RLM-style context environment for coding agents.
Provides semantic search, file navigation, and context management tools.
"""

from __future__ import annotations

from fastmcp import FastMCP

from indexter_rlm.config import settings

from .prompts import (
    get_cursorrules,
    get_debug_workflow,
    get_refactor_workflow,
    get_search_workflow,
)
from .tools import (
    clear_notes,
    delete_note,
    find_symbol_definition,
    find_symbol_references,
    get_exploration_summary,
    get_file,
    get_note,
    get_notes,
    list_file_symbols,
    list_repos,
    list_symbols,
    search_repo,
    store_note,
)

# Create the MCP server
mcp = FastMCP(
    "indexter-rlm",
    instructions=(
        "RLM-style context environment for coding agents - recursive navigation over semantic code"
    ),
)


@mcp.tool()
async def list_repositories() -> list[dict]:
    """
    List all Indexter-configured repositories.

    Returns a list of repository objects with name, path, and
    indexing status (i.e., number of nodes indexed, number of documents indexed,
    number of stale documents in the index).
    """
    return await list_repos()


@mcp.tool()
async def search_repository(
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
    Semantic search across an Indexter-configured repository's indexed code.

    Supports filtering by file path, language, node type, node name,
    and documentation presence.

    Args:
        name: The repository name.
        query: Natural language search query.
        file_path: Filter by file path (exact match or prefix with trailing /).
        language: Filter by programming language (e.g., 'python', 'javascript').
        node_type: Filter by node type (e.g., 'function', 'class', 'method').
        node_name: Filter by node name.
        has_documentation: Filter by documentation presence.
        limit: Maximum number of results to return (defaults to 10).

    Returns code chunks ranked by semantic similarity to the query.
    """
    return await search_repo(
        name=name,
        query=query,
        file_path=file_path,
        language=language,
        node_type=node_type,
        node_name=node_name,
        has_documentation=has_documentation,
        limit=limit,
    )


@mcp.tool()
async def read_file(
    name: str,
    file_path: str,
    start_line: int | None = None,
    end_line: int | None = None,
) -> dict:
    """
    Read file content from an Indexter-configured repository.

    Use this tool after search_repository to inspect actual source code.
    Supports optional line range filtering for large files.

    Args:
        name: The repository name.
        file_path: Path to the file relative to repository root.
        start_line: Optional starting line number (1-based, inclusive).
        end_line: Optional ending line number (1-based, inclusive).

    Returns file content with line numbers, path, and metadata.
    """
    return await get_file(
        name=name,
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
    )


@mcp.tool()
async def get_symbols(
    name: str,
    file_path: str,
) -> dict:
    """
    List all symbols (functions, classes, methods) in a file.

    Use this tool to understand a file's structure before reading
    specific sections with read_file.

    Args:
        name: The repository name.
        file_path: Path to the file relative to repository root.

    Returns list of symbols with name, type, line number, and signature.
    """
    return await list_symbols(name=name, file_path=file_path)


# =============================================================================
# Note Tools (Scratchpad)
# =============================================================================


@mcp.tool()
async def save_note(
    name: str,
    key: str,
    content: str,
    tags: list[str] | None = None,
) -> dict:
    """
    Store a note in the repository's scratchpad.

    Use this to record observations, hypotheses, and findings during
    code exploration. Notes persist across sessions and can be tagged
    for organization.

    Args:
        name: The repository name.
        key: Unique identifier for the note (overwrites if exists).
        content: The note content.
        tags: Optional list of tags for categorization.

    Returns the stored note with metadata.
    """
    return await store_note(name=name, key=key, content=content, tags=tags)


@mcp.tool()
async def retrieve_note(
    name: str,
    key: str,
) -> dict:
    """
    Retrieve a single note by key.

    Args:
        name: The repository name.
        key: The note key to retrieve.

    Returns the note content and metadata.
    """
    return await get_note(name=name, key=key)


@mcp.tool()
async def list_notes(
    name: str,
    tag: str | None = None,
) -> dict:
    """
    List all notes for a repository.

    Use this to review accumulated observations and findings.
    Notes are sorted by most recently updated.

    Args:
        name: The repository name.
        tag: Optional tag to filter notes by.

    Returns list of notes with metadata.
    """
    return await get_notes(name=name, tag=tag)


@mcp.tool()
async def remove_note(
    name: str,
    key: str,
) -> dict:
    """
    Delete a note by key.

    Args:
        name: The repository name.
        key: The note key to delete.

    Returns confirmation of deletion.
    """
    return await delete_note(name=name, key=key)


@mcp.tool()
async def remove_all_notes(
    name: str,
) -> dict:
    """
    Delete all notes for a repository.

    Use this to clear the scratchpad and start fresh.

    Args:
        name: The repository name.

    Returns count of deleted notes.
    """
    return await clear_notes(name=name)


# =============================================================================
# Exploration Session
# =============================================================================


@mcp.tool()
async def exploration_summary(
    name: str,
) -> dict:
    """
    Get a summary of the current exploration session.

    Returns statistics about tool calls, searches performed, and files read
    during the current session. Use this to review your exploration progress.

    Args:
        name: The repository name.

    Returns session statistics including tool counts, search queries, and files read.
    """
    return await get_exploration_summary(name=name)


# =============================================================================
# Symbol Navigation Tools
# =============================================================================


@mcp.tool()
async def find_references(
    name: str,
    symbol_name: str,
    include_imports: bool = True,
) -> dict:
    """
    Find all references to a symbol across the repository.

    Use this tool to understand how a symbol (function, class, constant) is used
    throughout the codebase. Returns definitions, usages, and import chains.

    Args:
        name: The repository name.
        symbol_name: Name of the symbol to find references for.
        include_imports: Include transitive import chains (default True).

    Returns definitions, references, and import chains for the symbol.
    """
    return await find_symbol_references(
        name=name,
        symbol_name=symbol_name,
        include_imports=include_imports,
    )


@mcp.tool()
async def find_definition(
    name: str,
    symbol_name: str,
) -> dict:
    """
    Find where a symbol is defined.

    Use this tool to jump to the definition of a function, class, or constant.

    Args:
        name: The repository name.
        symbol_name: Name of the symbol to find.

    Returns the definition location(s) with file path, line, and documentation.
    """
    return await find_symbol_definition(name=name, symbol_name=symbol_name)


@mcp.tool()
async def list_definitions(
    name: str,
    file_path: str,
) -> dict:
    """
    List all symbol definitions in a file.

    Use this tool to get an overview of what's defined in a file.
    More detailed than get_symbols - includes qualified names.

    Args:
        name: The repository name.
        file_path: Path to the file relative to repository root.

    Returns list of symbols with names, types, lines, and signatures.
    """
    return await list_file_symbols(name=name, file_path=file_path)


# =============================================================================
# Prompts
# =============================================================================


@mcp.prompt()
def search_workflow() -> str:
    """Guide for effectively searching Indexter-configured code repositories."""
    return get_search_workflow()


@mcp.prompt()
def debug_workflow() -> str:
    """Guide for debugging code using the RLM exploration loop."""
    return get_debug_workflow()


@mcp.prompt()
def refactor_workflow() -> str:
    """Guide for safely refactoring code with full context awareness."""
    return get_refactor_workflow()


# =============================================================================
# Resources
# =============================================================================


@mcp.resource("cursorrules://indexter-rlm")
def cursorrules_resource() -> str:
    """
    Agent rules for RLM-style code exploration.

    Copy this content to your .cursorrules file to enforce the RLM loop
    in your AI coding assistant.
    """
    return get_cursorrules()


def run_server() -> None:
    """Run the MCP server based on configuration settings."""
    if settings.mcp.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport="http", host=settings.mcp.host, port=settings.mcp.port)


if __name__ == "__main__":
    run_server()
