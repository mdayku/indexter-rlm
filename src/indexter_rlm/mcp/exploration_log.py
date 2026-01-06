"""
Exploration logging for Indexter-RLM.

Provides structured JSON logging of tool calls during code exploration.
Logs are stored per-repository at ~/.config/indexter/logs/{repo}/exploration_{timestamp}.jsonl
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from indexter_rlm.config import settings

logger = logging.getLogger(__name__)


class ExplorationLogger:
    """
    Logger for tracking tool calls during code exploration.

    Writes JSON Lines format to enable easy parsing and analysis.
    Each exploration session gets its own log file.
    """

    def __init__(self, repo_name: str) -> None:
        """
        Initialize an exploration logger for a repository.

        Args:
            repo_name: Name of the repository being explored.
        """
        self.repo_name = repo_name
        self._session_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        self._log_file: Path | None = None
        self._enabled = True

    @property
    def logs_dir(self) -> Path:
        """Directory where logs are stored for this repo."""
        return settings.config_dir / "logs" / self.repo_name

    @property
    def log_file(self) -> Path:
        """Path to the current session's log file."""
        if self._log_file is None:
            self._log_file = self.logs_dir / f"exploration_{self._session_id}.jsonl"
        return self._log_file

    def _ensure_dir(self) -> None:
        """Ensure the logs directory exists."""
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        tool: str,
        args: dict[str, Any],
        result_summary: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        """
        Log a tool call.

        Args:
            tool: Name of the tool called.
            args: Arguments passed to the tool.
            result_summary: Summary of the result (not full content).
            error: Error message if the call failed.
        """
        if not self._enabled:
            return

        try:
            self._ensure_dir()

            entry = {
                "timestamp": datetime.now(UTC).isoformat(),
                "session_id": self._session_id,
                "tool": tool,
                "args": args,
            }

            if result_summary:
                entry["result"] = result_summary

            if error:
                entry["error"] = error

            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")

        except Exception as e:
            logger.warning(f"Failed to write exploration log: {e}")

    def disable(self) -> None:
        """Disable logging for this session."""
        self._enabled = False

    def enable(self) -> None:
        """Enable logging for this session."""
        self._enabled = True

    def get_session_log(self) -> list[dict]:
        """
        Read the current session's log entries.

        Returns:
            List of log entries from the current session.
        """
        if not self.log_file.exists():
            return []

        entries = []
        try:
            with open(self.log_file, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        entries.append(json.loads(line))
        except Exception as e:
            logger.warning(f"Failed to read exploration log: {e}")

        return entries

    def get_summary(self) -> dict:
        """
        Get a summary of the current exploration session.

        Returns:
            Dict with session statistics.
        """
        entries = self.get_session_log()

        tool_counts: dict[str, int] = {}
        search_queries: list[str] = []
        files_read: list[str] = []
        notes_created: list[str] = []
        errors: list[str] = []

        for entry in entries:
            tool = entry.get("tool", "unknown")
            tool_counts[tool] = tool_counts.get(tool, 0) + 1

            args = entry.get("args", {})

            if tool == "search_repository":
                if query := args.get("query"):
                    search_queries.append(query)

            elif tool == "read_file":
                if file_path := args.get("file_path"):
                    files_read.append(file_path)

            elif tool == "save_note":
                if key := args.get("key"):
                    notes_created.append(key)

            if err := entry.get("error"):
                errors.append(err)

        return {
            "session_id": self._session_id,
            "repository": self.repo_name,
            "total_calls": len(entries),
            "tool_counts": tool_counts,
            "search_queries": search_queries,
            "files_read": list(set(files_read)),  # Dedupe
            "notes_created": notes_created,
            "errors": errors,
        }


# Global cache of loggers per repository
_exploration_loggers: dict[str, ExplorationLogger] = {}


def get_exploration_logger(repo_name: str) -> ExplorationLogger:
    """
    Get or create an ExplorationLogger for a repository.

    Args:
        repo_name: Name of the repository.

    Returns:
        ExplorationLogger instance for the repository.
    """
    if repo_name not in _exploration_loggers:
        _exploration_loggers[repo_name] = ExplorationLogger(repo_name)
    return _exploration_loggers[repo_name]


def clear_exploration_logger_cache() -> None:
    """
    Clear the global exploration logger cache.

    Used for testing to ensure a clean state between tests.
    """
    _exploration_loggers.clear()

