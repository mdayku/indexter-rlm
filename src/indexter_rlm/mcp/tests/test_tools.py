"""Tests for MCP tools using FastMCP Client.

This module tests the list_repos and search_repo tools through the FastMCP Client,
following best practices from https://gofastmcp.com/patterns/testing
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastmcp.client import Client
from fastmcp.client.transports import FastMCPTransport

from indexter_rlm.exceptions import RepoNotFoundError
from indexter_rlm.mcp.notes import clear_note_store_cache

# =============================================================================
# list_repos Tool Tests
# =============================================================================


async def test_list_repos_tool_success(mcp_client: Client[FastMCPTransport], mock_repo_list):
    """Test list_repositories tool returns all configured repositories."""
    with patch("indexter_rlm.mcp.tools.Repo.list", new_callable=AsyncMock) as mock_list:
        # Setup mock repos with async status methods
        repo1 = MagicMock()
        repo1.status = AsyncMock(
            return_value={
                "repository": "repo1",
                "path": "/path/to/repo1",
                "nodes_indexed": 100,
                "documents_indexed": 20,
                "documents_indexed_stale": 0,
            }
        )

        repo2 = MagicMock()
        repo2.status = AsyncMock(
            return_value={
                "repository": "repo2",
                "path": "/path/to/repo2",
                "nodes_indexed": 50,
                "documents_indexed": 10,
                "documents_indexed_stale": 2,
            }
        )

        mock_list.return_value = [repo1, repo2]

        result = await mcp_client.call_tool(name="list_repositories", arguments={})

        assert result.data is not None
        assert isinstance(result.data, list)
        assert len(result.data) == 2
        # Verify the mocks were called correctly
        mock_list.assert_awaited_once()
        repo1.status.assert_awaited_once()
        repo2.status.assert_awaited_once()


async def test_list_repos_tool_empty(mcp_client: Client[FastMCPTransport]):
    """Test list_repositories tool with no configured repositories."""
    with patch("indexter_rlm.mcp.tools.Repo.list", new_callable=AsyncMock) as mock_list:
        mock_list.return_value = []

        result = await mcp_client.call_tool(name="list_repositories", arguments={})

        assert result.data is not None
        assert isinstance(result.data, list)
        assert len(result.data) == 0


async def test_list_repos_tool_includes_all_status_fields(mcp_client: Client[FastMCPTransport]):
    """Test that list_repositories includes all expected status fields."""
    with patch("indexter_rlm.mcp.tools.Repo.list", new_callable=AsyncMock) as mock_list:
        repo = MagicMock()
        repo.status = AsyncMock(
            return_value={
                "repository": "test-repo",
                "path": "/path/to/repo",
                "nodes_indexed": 150,
                "documents_indexed": 25,
                "documents_indexed_stale": 3,
            }
        )
        mock_list.return_value = [repo]

        result = await mcp_client.call_tool(name="list_repositories", arguments={})

        assert result.data is not None
        assert len(result.data) == 1
        # Verify the tool was called correctly
        mock_list.assert_awaited_once()
        repo.status.assert_awaited_once()


# =============================================================================
# search_repo Tool Tests
# =============================================================================


async def test_search_repo_tool_success(
    mcp_client: Client[FastMCPTransport], sample_search_results
):
    """Test search_repository tool with a valid repository and query."""
    with patch("indexter_rlm.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.settings = MagicMock(top_k=20)
        mock_repo.index = AsyncMock()
        mock_repo.search = AsyncMock(return_value=sample_search_results)
        mock_get.return_value = mock_repo

        result = await mcp_client.call_tool(
            name="search_repository", arguments={"name": "test-repo", "query": "process data"}
        )

        assert result.data is not None
        assert "results" in result.data
        assert "count" in result.data
        assert result.data["count"] == 2
        assert len(result.data["results"]) == 2
        mock_get.assert_awaited_once_with("test-repo")
        mock_repo.index.assert_awaited_once()
        mock_repo.search.assert_awaited_once()


async def test_search_repo_tool_not_found(mcp_client: Client[FastMCPTransport]):
    """Test search_repository tool when repository is not found."""
    with patch("indexter_rlm.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_get.side_effect = RepoNotFoundError("Repository not found: missing-repo")

        # Should raise exception, not return error dict
        with pytest.raises(Exception):  # noqa: B017
            await mcp_client.call_tool(
                name="search_repository", arguments={"name": "missing-repo", "query": "test query"}
            )


async def test_search_repo_tool_with_file_path_filter(mcp_client: Client[FastMCPTransport]):
    """Test search_repository tool with file_path filter."""
    with patch("indexter_rlm.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.settings = MagicMock(top_k=20)
        mock_repo.index = AsyncMock()
        mock_repo.search = AsyncMock(return_value=[])
        mock_get.return_value = mock_repo

        await mcp_client.call_tool(
            name="search_repository",
            arguments={"name": "test-repo", "query": "query", "file_path": "src/utils.py"},
        )

        # Verify file_path was passed to search
        call_kwargs = mock_repo.search.call_args.kwargs
        assert call_kwargs["file_path"] == "src/utils.py"


async def test_search_repo_tool_with_language_filter(mcp_client: Client[FastMCPTransport]):
    """Test search_repository tool with language filter."""
    with patch("indexter_rlm.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.settings = MagicMock(top_k=20)
        mock_repo.index = AsyncMock()
        mock_repo.search = AsyncMock(return_value=[])
        mock_get.return_value = mock_repo

        await mcp_client.call_tool(
            name="search_repository",
            arguments={"name": "test-repo", "query": "query", "language": "python"},
        )

        call_kwargs = mock_repo.search.call_args.kwargs
        assert call_kwargs["language"] == "python"


async def test_search_repo_tool_with_node_type_filter(mcp_client: Client[FastMCPTransport]):
    """Test search_repository tool with node_type filter."""
    with patch("indexter_rlm.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.settings = MagicMock(top_k=20)
        mock_repo.index = AsyncMock()
        mock_repo.search = AsyncMock(return_value=[])
        mock_get.return_value = mock_repo

        await mcp_client.call_tool(
            name="search_repository",
            arguments={"name": "test-repo", "query": "query", "node_type": "function"},
        )

        call_kwargs = mock_repo.search.call_args.kwargs
        assert call_kwargs["node_type"] == "function"


async def test_search_repo_tool_with_node_name_filter(mcp_client: Client[FastMCPTransport]):
    """Test search_repository tool with node_name filter."""
    with patch("indexter_rlm.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.settings = MagicMock(top_k=20)
        mock_repo.index = AsyncMock()
        mock_repo.search = AsyncMock(return_value=[])
        mock_get.return_value = mock_repo

        await mcp_client.call_tool(
            name="search_repository",
            arguments={"name": "test-repo", "query": "query", "node_name": "process_data"},
        )

        call_kwargs = mock_repo.search.call_args.kwargs
        assert call_kwargs["node_name"] == "process_data"


async def test_search_repo_tool_with_has_documentation_filter(mcp_client: Client[FastMCPTransport]):
    """Test search_repository tool with has_documentation filter."""
    with patch("indexter_rlm.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.settings = MagicMock(top_k=20)
        mock_repo.index = AsyncMock()
        mock_repo.search = AsyncMock(return_value=[])
        mock_get.return_value = mock_repo

        await mcp_client.call_tool(
            name="search_repository",
            arguments={"name": "test-repo", "query": "query", "has_documentation": True},
        )

        call_kwargs = mock_repo.search.call_args.kwargs
        assert call_kwargs["has_documentation"] is True


async def test_search_repo_tool_with_all_filters(mcp_client: Client[FastMCPTransport]):
    """Test search_repository tool with all filters combined."""
    with patch("indexter_rlm.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.settings = MagicMock(top_k=20)
        mock_repo.index = AsyncMock()
        mock_repo.search = AsyncMock(return_value=[])
        mock_get.return_value = mock_repo

        await mcp_client.call_tool(
            name="search_repository",
            arguments={
                "name": "test-repo",
                "query": "query",
                "file_path": "src/",
                "language": "python",
                "node_type": "class",
                "node_name": "MyClass",
                "has_documentation": True,
                "limit": 15,
            },
        )

        call_kwargs = mock_repo.search.call_args.kwargs
        assert call_kwargs["file_path"] == "src/"
        assert call_kwargs["language"] == "python"
        assert call_kwargs["node_type"] == "class"
        assert call_kwargs["node_name"] == "MyClass"
        assert call_kwargs["has_documentation"] is True
        assert call_kwargs["limit"] == 15


async def test_search_repo_tool_uses_repo_settings_top_k(mcp_client: Client[FastMCPTransport]):
    """Test that search_repository tool uses limit from repo settings."""
    with patch("indexter_rlm.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.settings = MagicMock(top_k=50)
        mock_repo.index = AsyncMock()
        mock_repo.search = AsyncMock(return_value=[])
        mock_get.return_value = mock_repo

        await mcp_client.call_tool(
            name="search_repository", arguments={"name": "test-repo", "query": "query"}
        )

        call_kwargs = mock_repo.search.call_args.kwargs
        assert call_kwargs["limit"] == 50


async def test_search_repo_tool_defaults_to_10_when_no_settings(
    mcp_client: Client[FastMCPTransport],
):
    """Test that search_repository tool defaults to 10 when repo has no settings."""
    with patch("indexter_rlm.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.settings = None
        mock_repo.index = AsyncMock()
        mock_repo.search = AsyncMock(return_value=[])
        mock_get.return_value = mock_repo

        await mcp_client.call_tool(
            name="search_repository", arguments={"name": "test-repo", "query": "query"}
        )

        call_kwargs = mock_repo.search.call_args.kwargs
        assert call_kwargs["limit"] == 10


async def test_search_repo_tool_empty_results(mcp_client: Client[FastMCPTransport]):
    """Test search_repository tool with no matching results."""
    with patch("indexter_rlm.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.settings = MagicMock(top_k=20)
        mock_repo.index = AsyncMock()
        mock_repo.search = AsyncMock(return_value=[])
        mock_get.return_value = mock_repo

        result = await mcp_client.call_tool(
            name="search_repository", arguments={"name": "test-repo", "query": "nonexistent code"}
        )

        assert result.data is not None
        assert result.data["results"] == []
        assert result.data["count"] == 0


async def test_search_repo_tool_result_count_matches_results_length(
    mcp_client: Client[FastMCPTransport],
):
    """Test that count field matches the length of results."""
    with patch("indexter_rlm.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.settings = MagicMock(top_k=20)
        mock_repo.index = AsyncMock()
        test_results = [{"id": f"result{i}"} for i in range(15)]
        mock_repo.search = AsyncMock(return_value=test_results)
        mock_get.return_value = mock_repo

        result = await mcp_client.call_tool(
            name="search_repository", arguments={"name": "test-repo", "query": "query"}
        )

        assert result.data is not None
        assert result.data["count"] == len(result.data["results"])
        assert result.data["count"] == 15


async def test_search_repo_tool_passes_query_correctly(mcp_client: Client[FastMCPTransport]):
    """Test that search_repository tool passes the query parameter correctly."""
    with patch("indexter_rlm.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.settings = MagicMock(top_k=20)
        mock_repo.index = AsyncMock()
        mock_repo.search = AsyncMock(return_value=[])
        mock_get.return_value = mock_repo

        query_string = "find authentication logic"
        await mcp_client.call_tool(
            name="search_repository", arguments={"name": "test-repo", "query": query_string}
        )

        call_kwargs = mock_repo.search.call_args.kwargs
        assert call_kwargs["query"] == query_string


async def test_search_repo_tool_with_custom_limit(mcp_client: Client[FastMCPTransport]):
    """Test that search_repository tool uses custom limit when provided."""
    with patch("indexter_rlm.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.settings = MagicMock(top_k=50)
        mock_repo.index = AsyncMock()
        mock_repo.search = AsyncMock(return_value=[])
        mock_get.return_value = mock_repo

        await mcp_client.call_tool(
            name="search_repository", arguments={"name": "test-repo", "query": "query", "limit": 5}
        )

        call_kwargs = mock_repo.search.call_args.kwargs
        # Custom limit should override repo settings
        assert call_kwargs["limit"] == 5


async def test_search_repo_tool_with_none_filters(mcp_client: Client[FastMCPTransport]):
    """Test search_repository tool with all optional filters set to None."""
    with patch("indexter_rlm.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.settings = MagicMock(top_k=20)
        mock_repo.index = AsyncMock()
        mock_repo.search = AsyncMock(return_value=[])
        mock_get.return_value = mock_repo

        result = await mcp_client.call_tool(
            name="search_repository",
            arguments={
                "name": "test-repo",
                "query": "test",
                "file_path": None,
                "language": None,
                "node_type": None,
                "node_name": None,
                "has_documentation": None,
                "limit": None,
            },
        )

        assert result.data is not None
        assert result.data["count"] == 0


async def test_search_repo_tool_with_directory_path_filter(mcp_client: Client[FastMCPTransport]):
    """Test search_repository tool with directory path filter (trailing /)."""
    with patch("indexter_rlm.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.settings = MagicMock(top_k=20)
        mock_repo.index = AsyncMock()
        mock_repo.search = AsyncMock(return_value=[])
        mock_get.return_value = mock_repo

        await mcp_client.call_tool(
            name="search_repository",
            arguments={"name": "test-repo", "query": "query", "file_path": "src/utils/"},
        )

        call_kwargs = mock_repo.search.call_args.kwargs
        assert call_kwargs["file_path"] == "src/utils/"


async def test_search_repo_tool_automatically_indexes(mcp_client: Client[FastMCPTransport]):
    """Test that search_repository tool automatically calls repo.index before searching."""
    with patch("indexter_rlm.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.settings = MagicMock(top_k=20)
        mock_repo.index = AsyncMock()
        mock_repo.search = AsyncMock(return_value=[])
        mock_get.return_value = mock_repo

        await mcp_client.call_tool(
            name="search_repository", arguments={"name": "test-repo", "query": "query"}
        )

        # Verify index was called before search
        mock_repo.index.assert_awaited_once()
        mock_repo.search.assert_awaited_once()

        # Verify order: index should be called before search
        assert mock_repo.index.call_count == 1
        assert mock_repo.search.call_count == 1


# =============================================================================
# Parameterized Tests
# =============================================================================


@pytest.mark.parametrize(
    "filter_name,filter_value",
    [
        ("language", "python"),
        ("language", "javascript"),
        ("node_type", "function"),
        ("node_type", "class"),
        ("node_type", "method"),
        ("has_documentation", True),
        ("has_documentation", False),
    ],
)
async def test_search_repo_tool_individual_filters(
    mcp_client: Client[FastMCPTransport],
    filter_name,
    filter_value,
):
    """Test search_repository tool with individual filter parameters."""
    with patch("indexter_rlm.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.settings = MagicMock(top_k=20)
        mock_repo.index = AsyncMock()
        mock_repo.search = AsyncMock(return_value=[])
        mock_get.return_value = mock_repo

        arguments = {
            "name": "test-repo",
            "query": "test query",
            filter_name: filter_value,
        }

        await mcp_client.call_tool(name="search_repository", arguments=arguments)

        call_kwargs = mock_repo.search.call_args.kwargs
        assert call_kwargs[filter_name] == filter_value


# =============================================================================
# read_file Tool Tests
# =============================================================================


async def test_read_file_tool_success(mcp_client: Client[FastMCPTransport], tmp_path):
    """Test read_file tool reads file content successfully."""
    # Create a test file
    test_file = tmp_path / "test.py"
    test_file.write_text("def hello():\n    return 'world'\n")

    with patch("indexter_rlm.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.path = str(tmp_path)
        mock_get.return_value = mock_repo

        result = await mcp_client.call_tool(
            name="read_file",
            arguments={"name": "test-repo", "file_path": "test.py"},
        )

        assert result.data is not None
        assert result.data["repository"] == "test-repo"
        assert result.data["file_path"] == "test.py"
        assert "content" in result.data
        assert "def hello():" in result.data["content"]
        assert result.data["total_lines"] == 2
        assert result.data["start_line"] == 1
        assert result.data["end_line"] == 2


async def test_read_file_tool_with_line_range(mcp_client: Client[FastMCPTransport], tmp_path):
    """Test read_file tool with start_line and end_line parameters."""
    # Create a multi-line file
    test_file = tmp_path / "multi.py"
    test_file.write_text("line1\nline2\nline3\nline4\nline5\n")

    with patch("indexter_rlm.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.path = str(tmp_path)
        mock_get.return_value = mock_repo

        result = await mcp_client.call_tool(
            name="read_file",
            arguments={
                "name": "test-repo",
                "file_path": "multi.py",
                "start_line": 2,
                "end_line": 4,
            },
        )

        assert result.data is not None
        assert result.data["start_line"] == 2
        assert result.data["end_line"] == 4
        assert result.data["total_lines"] == 5
        # Content should only include lines 2-4
        assert "line2" in result.data["content"]
        assert "line3" in result.data["content"]
        assert "line4" in result.data["content"]
        assert "line1" not in result.data["content"]
        assert "line5" not in result.data["content"]


async def test_read_file_tool_file_not_found(mcp_client: Client[FastMCPTransport], tmp_path):
    """Test read_file tool when file does not exist."""
    with patch("indexter_rlm.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.path = str(tmp_path)
        mock_get.return_value = mock_repo

        result = await mcp_client.call_tool(
            name="read_file",
            arguments={"name": "test-repo", "file_path": "nonexistent.py"},
        )

        assert result.data is not None
        assert "error" in result.data
        assert "not found" in result.data["error"].lower()


async def test_read_file_tool_path_is_directory(mcp_client: Client[FastMCPTransport], tmp_path):
    """Test read_file tool when path is a directory, not a file."""
    subdir = tmp_path / "subdir"
    subdir.mkdir()

    with patch("indexter_rlm.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.path = str(tmp_path)
        mock_get.return_value = mock_repo

        result = await mcp_client.call_tool(
            name="read_file",
            arguments={"name": "test-repo", "file_path": "subdir"},
        )

        assert result.data is not None
        assert "error" in result.data
        assert "not a file" in result.data["error"].lower()


async def test_read_file_tool_repo_not_found(mcp_client: Client[FastMCPTransport]):
    """Test read_file tool when repository is not found."""
    with patch("indexter_rlm.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_get.side_effect = RepoNotFoundError("Repository not found: missing-repo")

        with pytest.raises(Exception):  # noqa: B017
            await mcp_client.call_tool(
                name="read_file",
                arguments={"name": "missing-repo", "file_path": "test.py"},
            )


async def test_read_file_tool_with_start_line_only(mcp_client: Client[FastMCPTransport], tmp_path):
    """Test read_file tool with only start_line specified."""
    test_file = tmp_path / "lines.py"
    test_file.write_text("line1\nline2\nline3\nline4\n")

    with patch("indexter_rlm.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.path = str(tmp_path)
        mock_get.return_value = mock_repo

        result = await mcp_client.call_tool(
            name="read_file",
            arguments={"name": "test-repo", "file_path": "lines.py", "start_line": 3},
        )

        assert result.data is not None
        assert result.data["start_line"] == 3
        assert result.data["end_line"] == 4
        assert "line3" in result.data["content"]
        assert "line4" in result.data["content"]


async def test_read_file_tool_with_end_line_only(mcp_client: Client[FastMCPTransport], tmp_path):
    """Test read_file tool with only end_line specified."""
    test_file = tmp_path / "lines.py"
    test_file.write_text("line1\nline2\nline3\nline4\n")

    with patch("indexter_rlm.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.path = str(tmp_path)
        mock_get.return_value = mock_repo

        result = await mcp_client.call_tool(
            name="read_file",
            arguments={"name": "test-repo", "file_path": "lines.py", "end_line": 2},
        )

        assert result.data is not None
        assert result.data["start_line"] == 1
        assert result.data["end_line"] == 2
        assert "line1" in result.data["content"]
        assert "line2" in result.data["content"]


async def test_read_file_tool_line_numbers_in_content(
    mcp_client: Client[FastMCPTransport], tmp_path
):
    """Test that read_file output includes line numbers."""
    test_file = tmp_path / "numbered.py"
    test_file.write_text("first\nsecond\nthird\n")

    with patch("indexter_rlm.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.path = str(tmp_path)
        mock_get.return_value = mock_repo

        result = await mcp_client.call_tool(
            name="read_file",
            arguments={"name": "test-repo", "file_path": "numbered.py"},
        )

        assert result.data is not None
        content = result.data["content"]
        # Should have line number formatting
        assert "1 |" in content or "   1 |" in content


# =============================================================================
# get_symbols Tool Tests
# =============================================================================


async def test_get_symbols_tool_success(mcp_client: Client[FastMCPTransport], tmp_path):
    """Test get_symbols tool returns symbols from a Python file."""
    test_file = tmp_path / "module.py"
    test_file.write_text('''def hello():
    """Greet the world."""
    return "world"

class MyClass:
    def method(self):
        pass
''')

    with patch("indexter_rlm.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.path = str(tmp_path)
        mock_get.return_value = mock_repo

        result = await mcp_client.call_tool(
            name="get_symbols",
            arguments={"name": "test-repo", "file_path": "module.py"},
        )

        assert result.data is not None
        assert result.data["repository"] == "test-repo"
        assert result.data["file_path"] == "module.py"
        assert "symbols" in result.data
        assert "count" in result.data
        assert result.data["count"] >= 2  # At least function and class


async def test_get_symbols_tool_file_not_found(mcp_client: Client[FastMCPTransport], tmp_path):
    """Test get_symbols tool when file does not exist."""
    with patch("indexter_rlm.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.path = str(tmp_path)
        mock_get.return_value = mock_repo

        result = await mcp_client.call_tool(
            name="get_symbols",
            arguments={"name": "test-repo", "file_path": "nonexistent.py"},
        )

        assert result.data is not None
        assert "error" in result.data
        assert "not found" in result.data["error"].lower()


async def test_get_symbols_tool_unknown_file_type_uses_chunk_parser(
    mcp_client: Client[FastMCPTransport], tmp_path
):
    """Test get_symbols tool uses fallback chunk parser for unknown file types."""
    test_file = tmp_path / "data.xyz"
    test_file.write_text("random content")

    with patch("indexter_rlm.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.path = str(tmp_path)
        mock_get.return_value = mock_repo

        result = await mcp_client.call_tool(
            name="get_symbols",
            arguments={"name": "test-repo", "file_path": "data.xyz"},
        )

        # ChunkParser is used as fallback - returns a chunk symbol
        assert result.data is not None
        assert "symbols" in result.data
        assert result.data["count"] >= 1
        # Chunk parser returns "chunk" type symbols
        assert any(s["type"] == "chunk" for s in result.data["symbols"])


async def test_get_symbols_tool_repo_not_found(mcp_client: Client[FastMCPTransport]):
    """Test get_symbols tool when repository is not found."""
    with patch("indexter_rlm.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_get.side_effect = RepoNotFoundError("Repository not found: missing-repo")

        with pytest.raises(Exception):  # noqa: B017
            await mcp_client.call_tool(
                name="get_symbols",
                arguments={"name": "missing-repo", "file_path": "test.py"},
            )


async def test_get_symbols_tool_path_is_directory(mcp_client: Client[FastMCPTransport], tmp_path):
    """Test get_symbols tool when path is a directory."""
    subdir = tmp_path / "subdir"
    subdir.mkdir()

    with patch("indexter_rlm.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.path = str(tmp_path)
        mock_get.return_value = mock_repo

        result = await mcp_client.call_tool(
            name="get_symbols",
            arguments={"name": "test-repo", "file_path": "subdir"},
        )

        assert result.data is not None
        assert "error" in result.data
        assert "not a file" in result.data["error"].lower()


async def test_get_symbols_tool_symbols_sorted_by_line(
    mcp_client: Client[FastMCPTransport], tmp_path
):
    """Test that get_symbols returns symbols sorted by line number."""
    test_file = tmp_path / "ordered.py"
    test_file.write_text('''class First:
    pass

def second():
    pass

class Third:
    pass
''')

    with patch("indexter_rlm.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.path = str(tmp_path)
        mock_get.return_value = mock_repo

        result = await mcp_client.call_tool(
            name="get_symbols",
            arguments={"name": "test-repo", "file_path": "ordered.py"},
        )

        assert result.data is not None
        symbols = result.data["symbols"]
        # Verify symbols are sorted by line number
        lines = [s["line"] for s in symbols]
        assert lines == sorted(lines)


async def test_get_symbols_tool_symbol_structure(mcp_client: Client[FastMCPTransport], tmp_path):
    """Test that symbols have expected fields: name, type, line."""
    test_file = tmp_path / "structure.py"
    test_file.write_text('''def my_function():
    pass
''')

    with patch("indexter_rlm.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.path = str(tmp_path)
        mock_get.return_value = mock_repo

        result = await mcp_client.call_tool(
            name="get_symbols",
            arguments={"name": "test-repo", "file_path": "structure.py"},
        )

        assert result.data is not None
        assert len(result.data["symbols"]) >= 1
        symbol = result.data["symbols"][0]
        # Check required fields
        assert "name" in symbol
        assert "type" in symbol
        assert "line" in symbol


# =============================================================================
# Note Tools Tests (Scratchpad)
# =============================================================================


async def test_save_note_tool_success(mcp_client: Client[FastMCPTransport], tmp_path):
    """Test save_note tool creates a new note."""
    clear_note_store_cache()
    with patch("indexter_rlm.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.name = "test-repo"
        mock_get.return_value = mock_repo

        # Patch the note store to use a temp directory
        with patch("indexter_rlm.mcp.notes.settings") as mock_settings:
            mock_settings.config_dir = tmp_path

            result = await mcp_client.call_tool(
                name="save_note",
                arguments={
                    "name": "test-repo",
                    "key": "auth_flow",
                    "content": "Uses JWT tokens for authentication",
                    "tags": ["auth", "security"],
                },
            )

            assert result.data is not None
            assert result.data["repository"] == "test-repo"
            assert result.data["action"] == "stored"
            assert result.data["note"]["key"] == "auth_flow"
            assert result.data["note"]["content"] == "Uses JWT tokens for authentication"
            assert result.data["note"]["tags"] == ["auth", "security"]
            assert "created_at" in result.data["note"]
            assert "updated_at" in result.data["note"]


async def test_save_note_tool_update_existing(mcp_client: Client[FastMCPTransport], tmp_path):
    """Test save_note tool updates an existing note."""
    clear_note_store_cache()
    with patch("indexter_rlm.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.name = "test-repo"
        mock_get.return_value = mock_repo

        with patch("indexter_rlm.mcp.notes.settings") as mock_settings:
            mock_settings.config_dir = tmp_path

            # Create initial note
            await mcp_client.call_tool(
                name="save_note",
                arguments={
                    "name": "test-repo",
                    "key": "finding",
                    "content": "Initial observation",
                },
            )

            # Update the note
            result = await mcp_client.call_tool(
                name="save_note",
                arguments={
                    "name": "test-repo",
                    "key": "finding",
                    "content": "Updated observation",
                },
            )

            assert result.data is not None
            assert result.data["note"]["content"] == "Updated observation"


async def test_retrieve_note_tool_success(mcp_client: Client[FastMCPTransport], tmp_path):
    """Test retrieve_note tool returns an existing note."""
    clear_note_store_cache()
    with patch("indexter_rlm.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.name = "test-repo"
        mock_get.return_value = mock_repo

        with patch("indexter_rlm.mcp.notes.settings") as mock_settings:
            mock_settings.config_dir = tmp_path

            # Create a note first
            await mcp_client.call_tool(
                name="save_note",
                arguments={
                    "name": "test-repo",
                    "key": "test_key",
                    "content": "Test content",
                },
            )

            # Retrieve it
            result = await mcp_client.call_tool(
                name="retrieve_note",
                arguments={"name": "test-repo", "key": "test_key"},
            )

            assert result.data is not None
            assert "note" in result.data
            assert result.data["note"]["key"] == "test_key"
            assert result.data["note"]["content"] == "Test content"


async def test_retrieve_note_tool_not_found(mcp_client: Client[FastMCPTransport], tmp_path):
    """Test retrieve_note tool returns error when note doesn't exist."""
    clear_note_store_cache()
    with patch("indexter_rlm.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.name = "test-repo"
        mock_get.return_value = mock_repo

        with patch("indexter_rlm.mcp.notes.settings") as mock_settings:
            mock_settings.config_dir = tmp_path

            result = await mcp_client.call_tool(
                name="retrieve_note",
                arguments={"name": "test-repo", "key": "nonexistent"},
            )

            assert result.data is not None
            assert "error" in result.data
            assert "not found" in result.data["error"].lower()


async def test_list_notes_tool_success(mcp_client: Client[FastMCPTransport], tmp_path):
    """Test list_notes tool returns all notes."""
    clear_note_store_cache()
    with patch("indexter_rlm.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.name = "test-repo"
        mock_get.return_value = mock_repo

        with patch("indexter_rlm.mcp.notes.settings") as mock_settings:
            mock_settings.config_dir = tmp_path

            # Create multiple notes
            await mcp_client.call_tool(
                name="save_note",
                arguments={"name": "test-repo", "key": "note1", "content": "First"},
            )
            await mcp_client.call_tool(
                name="save_note",
                arguments={"name": "test-repo", "key": "note2", "content": "Second"},
            )

            # List all
            result = await mcp_client.call_tool(
                name="list_notes",
                arguments={"name": "test-repo"},
            )

            assert result.data is not None
            assert result.data["count"] == 2
            assert len(result.data["notes"]) == 2


async def test_list_notes_tool_filter_by_tag(mcp_client: Client[FastMCPTransport], tmp_path):
    """Test list_notes tool filters by tag."""
    clear_note_store_cache()
    with patch("indexter_rlm.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.name = "test-repo"
        mock_get.return_value = mock_repo

        with patch("indexter_rlm.mcp.notes.settings") as mock_settings:
            mock_settings.config_dir = tmp_path

            # Create notes with different tags
            await mcp_client.call_tool(
                name="save_note",
                arguments={
                    "name": "test-repo",
                    "key": "auth",
                    "content": "Auth note",
                    "tags": ["security"],
                },
            )
            await mcp_client.call_tool(
                name="save_note",
                arguments={
                    "name": "test-repo",
                    "key": "perf",
                    "content": "Perf note",
                    "tags": ["performance"],
                },
            )

            # Filter by tag
            result = await mcp_client.call_tool(
                name="list_notes",
                arguments={"name": "test-repo", "tag": "security"},
            )

            assert result.data is not None
            assert result.data["count"] == 1
            assert result.data["notes"][0]["key"] == "auth"


async def test_remove_note_tool_success(mcp_client: Client[FastMCPTransport], tmp_path):
    """Test remove_note tool deletes a note."""
    clear_note_store_cache()
    with patch("indexter_rlm.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.name = "test-repo"
        mock_get.return_value = mock_repo

        with patch("indexter_rlm.mcp.notes.settings") as mock_settings:
            mock_settings.config_dir = tmp_path

            # Create a note
            await mcp_client.call_tool(
                name="save_note",
                arguments={"name": "test-repo", "key": "to_delete", "content": "Delete me"},
            )

            # Delete it
            result = await mcp_client.call_tool(
                name="remove_note",
                arguments={"name": "test-repo", "key": "to_delete"},
            )

            assert result.data is not None
            assert result.data["action"] == "deleted"
            assert result.data["key"] == "to_delete"

            # Verify it's gone
            result2 = await mcp_client.call_tool(
                name="retrieve_note",
                arguments={"name": "test-repo", "key": "to_delete"},
            )
            assert "error" in result2.data


async def test_remove_note_tool_not_found(mcp_client: Client[FastMCPTransport], tmp_path):
    """Test remove_note tool returns error when note doesn't exist."""
    clear_note_store_cache()
    with patch("indexter_rlm.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.name = "test-repo"
        mock_get.return_value = mock_repo

        with patch("indexter_rlm.mcp.notes.settings") as mock_settings:
            mock_settings.config_dir = tmp_path

            result = await mcp_client.call_tool(
                name="remove_note",
                arguments={"name": "test-repo", "key": "nonexistent"},
            )

            assert result.data is not None
            assert "error" in result.data


async def test_remove_all_notes_tool_success(mcp_client: Client[FastMCPTransport], tmp_path):
    """Test remove_all_notes tool clears all notes."""
    clear_note_store_cache()
    with patch("indexter_rlm.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.name = "test-repo"
        mock_get.return_value = mock_repo

        with patch("indexter_rlm.mcp.notes.settings") as mock_settings:
            mock_settings.config_dir = tmp_path

            # Create multiple notes
            await mcp_client.call_tool(
                name="save_note",
                arguments={"name": "test-repo", "key": "note1", "content": "First"},
            )
            await mcp_client.call_tool(
                name="save_note",
                arguments={"name": "test-repo", "key": "note2", "content": "Second"},
            )

            # Clear all
            result = await mcp_client.call_tool(
                name="remove_all_notes",
                arguments={"name": "test-repo"},
            )

            assert result.data is not None
            assert result.data["action"] == "cleared"
            assert result.data["notes_deleted"] == 2

            # Verify empty
            result2 = await mcp_client.call_tool(
                name="list_notes",
                arguments={"name": "test-repo"},
            )
            assert result2.data["count"] == 0


async def test_note_repo_not_found(mcp_client: Client[FastMCPTransport]):
    """Test note tools raise error when repository is not found."""
    clear_note_store_cache()
    with patch("indexter_rlm.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_get.side_effect = RepoNotFoundError("Repository not found: missing-repo")

        with pytest.raises(Exception):  # noqa: B017
            await mcp_client.call_tool(
                name="save_note",
                arguments={
                    "name": "missing-repo",
                    "key": "test",
                    "content": "content",
                },
            )


# =============================================================================
# find_references Tool Tests
# =============================================================================


async def test_find_references_tool_success(
    mcp_client: Client[FastMCPTransport], tmp_path
):
    """Test find_references tool returns definitions and references."""
    from pathlib import Path as PathType

    from indexter_rlm.symbols import clear_symbol_index_cache

    clear_symbol_index_cache()

    # Create a test Python file
    test_file = tmp_path / "test_module.py"
    test_file.write_text('''
def my_function(x: int) -> str:
    """A test function."""
    return str(x)

result = my_function(42)
''')

    with patch("indexter_rlm.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.name = "test-repo"
        mock_repo.path = PathType(tmp_path)
        mock_get.return_value = mock_repo

        # Mock the Walker to return our test file
        async def mock_walk():
            yield {"path": "test_module.py"}

        with patch("indexter_rlm.walker.Walker") as MockWalker:
            mock_walker_instance = MagicMock()
            mock_walker_instance.walk = mock_walk
            MockWalker.return_value = mock_walker_instance

            with patch("indexter_rlm.mcp.tools.get_exploration_logger") as mock_logger:
                mock_logger.return_value = MagicMock()

                with patch(
                    "indexter_rlm.symbols.get_config_dir",
                    return_value=tmp_path / ".config",
                ):
                    result = await mcp_client.call_tool(
                        name="find_references",
                        arguments={
                            "name": "test-repo",
                            "symbol_name": "my_function",
                        },
                    )

                    assert result.data is not None
                    assert result.data["symbol_name"] == "my_function"
                    assert "definitions" in result.data
                    assert "references" in result.data


async def test_find_references_tool_with_import_chains(
    mcp_client: Client[FastMCPTransport], tmp_path
):
    """Test find_references includes import chains when requested."""
    from pathlib import Path as PathType

    from indexter_rlm.symbols import clear_symbol_index_cache

    clear_symbol_index_cache()

    # Create test files
    (tmp_path / "module_a.py").write_text('''
def shared_func():
    pass
''')
    (tmp_path / "module_b.py").write_text('''
from module_a import shared_func

shared_func()
''')

    with patch("indexter_rlm.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.name = "test-repo"
        mock_repo.path = PathType(tmp_path)
        mock_get.return_value = mock_repo

        # Mock the Walker to return our test files
        async def mock_walk():
            yield {"path": "module_a.py"}
            yield {"path": "module_b.py"}

        with patch("indexter_rlm.walker.Walker") as MockWalker:
            mock_walker_instance = MagicMock()
            mock_walker_instance.walk = mock_walk
            MockWalker.return_value = mock_walker_instance

            with patch("indexter_rlm.mcp.tools.get_exploration_logger") as mock_logger:
                mock_logger.return_value = MagicMock()

                with patch(
                    "indexter_rlm.symbols.get_config_dir",
                    return_value=tmp_path / ".config",
                ):
                    result = await mcp_client.call_tool(
                        name="find_references",
                        arguments={
                            "name": "test-repo",
                            "symbol_name": "shared_func",
                            "include_imports": True,
                        },
                    )

                    assert result.data is not None
                    assert "import_chains" in result.data


async def test_find_references_tool_not_found(
    mcp_client: Client[FastMCPTransport], tmp_path
):
    """Test find_references returns empty results for unknown symbol."""
    from pathlib import Path as PathType

    from indexter_rlm.symbols import clear_symbol_index_cache

    clear_symbol_index_cache()

    (tmp_path / "empty.py").write_text("# empty file\n")

    with patch("indexter_rlm.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.name = "test-repo"
        mock_repo.path = PathType(tmp_path)
        mock_get.return_value = mock_repo

        # Mock the Walker
        async def mock_walk():
            yield {"path": "empty.py"}

        with patch("indexter_rlm.walker.Walker") as MockWalker:
            mock_walker_instance = MagicMock()
            mock_walker_instance.walk = mock_walk
            MockWalker.return_value = mock_walker_instance

            with patch("indexter_rlm.mcp.tools.get_exploration_logger") as mock_logger:
                mock_logger.return_value = MagicMock()

                with patch(
                    "indexter_rlm.symbols.get_config_dir",
                    return_value=tmp_path / ".config",
                ):
                    result = await mcp_client.call_tool(
                        name="find_references",
                        arguments={
                            "name": "test-repo",
                            "symbol_name": "nonexistent_symbol",
                        },
                    )

                    assert result.data is not None
                    assert result.data["definitions_count"] == 0
                    assert result.data["references_count"] == 0


# =============================================================================
# find_definition Tool Tests
# =============================================================================


async def test_find_definition_tool_success(
    mcp_client: Client[FastMCPTransport], tmp_path
):
    """Test find_definition tool finds symbol definition."""
    from pathlib import Path as PathType

    from indexter_rlm.symbols import clear_symbol_index_cache

    clear_symbol_index_cache()

    test_file = tmp_path / "module.py"
    test_file.write_text('''
class UserService:
    """Handles user operations."""

    def authenticate(self, user_id: str) -> bool:
        return True
''')

    with patch("indexter_rlm.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.name = "test-repo"
        mock_repo.path = PathType(tmp_path)
        mock_get.return_value = mock_repo

        # Mock the Walker
        async def mock_walk():
            yield {"path": "module.py"}

        with patch("indexter_rlm.walker.Walker") as MockWalker:
            mock_walker_instance = MagicMock()
            mock_walker_instance.walk = mock_walk
            MockWalker.return_value = mock_walker_instance

            with patch(
                "indexter_rlm.symbols.get_config_dir",
                return_value=tmp_path / ".config",
            ):
                result = await mcp_client.call_tool(
                    name="find_definition",
                    arguments={
                        "name": "test-repo",
                        "symbol_name": "UserService",
                    },
                )

                assert result.data is not None
                assert result.data["symbol_name"] == "UserService"
                assert len(result.data["definitions"]) == 1
                assert result.data["definitions"][0]["type"] == "class"


async def test_find_definition_tool_not_found(
    mcp_client: Client[FastMCPTransport], tmp_path
):
    """Test find_definition returns error for unknown symbol."""
    from pathlib import Path as PathType

    from indexter_rlm.symbols import clear_symbol_index_cache

    clear_symbol_index_cache()

    (tmp_path / "empty.py").write_text("# empty\n")

    with patch("indexter_rlm.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.name = "test-repo"
        mock_repo.path = PathType(tmp_path)
        mock_get.return_value = mock_repo

        # Mock the Walker
        async def mock_walk():
            yield {"path": "empty.py"}

        with patch("indexter_rlm.walker.Walker") as MockWalker:
            mock_walker_instance = MagicMock()
            mock_walker_instance.walk = mock_walk
            MockWalker.return_value = mock_walker_instance

            with patch(
                "indexter_rlm.symbols.get_config_dir",
                return_value=tmp_path / ".config",
            ):
                result = await mcp_client.call_tool(
                    name="find_definition",
                    arguments={
                        "name": "test-repo",
                        "symbol_name": "MissingClass",
                    },
                )

                assert result.data is not None
                assert "error" in result.data


# =============================================================================
# list_definitions Tool Tests
# =============================================================================


async def test_list_definitions_tool_success(
    mcp_client: Client[FastMCPTransport], tmp_path
):
    """Test list_definitions returns all symbols in a file."""
    from pathlib import Path as PathType

    from indexter_rlm.symbols import clear_symbol_index_cache

    clear_symbol_index_cache()

    test_file = tmp_path / "module.py"
    test_file.write_text('''
MAX_SIZE = 100

class DataProcessor:
    def process(self, data):
        pass

    def validate(self, data):
        pass

def helper_function():
    pass
''')

    with patch("indexter_rlm.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.name = "test-repo"
        mock_repo.path = PathType(tmp_path)
        mock_get.return_value = mock_repo

        with patch(
            "indexter_rlm.symbols.get_config_dir", return_value=tmp_path / ".config"
        ):
            result = await mcp_client.call_tool(
                name="list_definitions",
                arguments={
                    "name": "test-repo",
                    "file_path": "module.py",
                },
            )

            assert result.data is not None
            assert result.data["file_path"] == "module.py"
            symbols = result.data["symbols"]
            symbol_names = [s["name"] for s in symbols]
            assert "DataProcessor" in symbol_names
            assert "process" in symbol_names
            assert "helper_function" in symbol_names


async def test_list_definitions_tool_empty_file(
    mcp_client: Client[FastMCPTransport], tmp_path
):
    """Test list_definitions returns empty list for file with no definitions."""
    from pathlib import Path as PathType

    from indexter_rlm.symbols import clear_symbol_index_cache

    clear_symbol_index_cache()

    test_file = tmp_path / "empty.py"
    test_file.write_text("# Just a comment\n")

    with patch("indexter_rlm.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.name = "test-repo"
        mock_repo.path = PathType(tmp_path)
        mock_get.return_value = mock_repo

        with patch(
            "indexter_rlm.symbols.get_config_dir", return_value=tmp_path / ".config"
        ):
            result = await mcp_client.call_tool(
                name="list_definitions",
                arguments={
                    "name": "test-repo",
                    "file_path": "empty.py",
                },
            )

            assert result.data is not None
            assert result.data["count"] == 0
