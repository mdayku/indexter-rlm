"""
MCP tool implementations for Indexter-RLM.

Tools perform actions and can mutate state.
Provides semantic search, file navigation, and context management.
"""

from pathlib import Path

from anyio import create_task_group

from indexter_rlm.models import Repo
from indexter_rlm.parsers import get_parser


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
    numbered_lines = [
        f"{i:4d} | {line}"
        for i, line in enumerate(lines, start=actual_start)
    ]
    formatted_content = "\n".join(numbered_lines)

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
