"""
Indexter-RLM MCP Server.

A FastMCP server exposing RLM-style context environment for coding agents.
Provides semantic search, file navigation, and context management tools.
"""

from __future__ import annotations

from fastmcp import FastMCP

from indexter_rlm.config import settings

from .prompts import get_search_workflow
from .tools import get_file, list_repos, list_symbols, search_repo

# Create the MCP server
mcp = FastMCP(
    "indexter-rlm",
    instructions="RLM-style context environment for coding agents - recursive navigation over semantic code",
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


@mcp.prompt()
def search_workflow() -> str:
    """Guide for effectively searching Indexter-configured code repositories."""
    return get_search_workflow()


def run_server() -> None:
    """Run the MCP server based on configuration settings."""
    if settings.mcp.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport="http", host=settings.mcp.host, port=settings.mcp.port)


if __name__ == "__main__":
    run_server()
