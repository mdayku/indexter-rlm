"""
Note storage for agent scratchpad functionality.

Provides per-repository note storage with in-memory caching and file persistence.
Notes are stored at ~/.indexter-rlm/notes/{repo_name}.json.
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from indexter_rlm.config import settings
from indexter_rlm.models import Note

logger = logging.getLogger(__name__)


class NoteStore:
    """
    Per-repository note storage with file persistence.

    Manages notes for a single repository, keeping them in memory for fast
    access while persisting to disk for durability across sessions.

    Attributes:
        repo_name: Name of the repository this store manages notes for.
    """

    def __init__(self, repo_name: str) -> None:
        """
        Initialize a note store for a repository.

        Args:
            repo_name: Name of the repository.
        """
        self.repo_name = repo_name
        self._notes: dict[str, Note] = {}
        self._loaded = False

    @property
    def notes_dir(self) -> Path:
        """Directory where notes are stored."""
        return settings.config_dir / "notes"

    @property
    def notes_file(self) -> Path:
        """Path to the JSON file for this repository's notes."""
        return self.notes_dir / f"{self.repo_name}.json"

    def _ensure_dir(self) -> None:
        """Ensure the notes directory exists."""
        self.notes_dir.mkdir(parents=True, exist_ok=True)

    def _load(self) -> None:
        """Load notes from disk if not already loaded."""
        if self._loaded:
            return

        self._ensure_dir()

        if self.notes_file.exists():
            try:
                data = json.loads(self.notes_file.read_text(encoding="utf-8"))
                for note_data in data.get("notes", []):
                    # Parse datetime strings back to datetime objects
                    if "created_at" in note_data and isinstance(note_data["created_at"], str):
                        note_data["created_at"] = datetime.fromisoformat(note_data["created_at"])
                    if "updated_at" in note_data and isinstance(note_data["updated_at"], str):
                        note_data["updated_at"] = datetime.fromisoformat(note_data["updated_at"])
                    note = Note(**note_data)
                    self._notes[note.key] = note
                logger.debug(f"Loaded {len(self._notes)} notes for {self.repo_name}")
            except Exception as e:
                logger.warning(f"Failed to load notes for {self.repo_name}: {e}")
                self._notes = {}

        self._loaded = True

    def _save(self) -> None:
        """Persist notes to disk."""
        self._ensure_dir()

        try:
            data = {
                "repo": self.repo_name,
                "notes": [note.model_dump_for_display() for note in self._notes.values()],
            }
            self.notes_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
            logger.debug(f"Saved {len(self._notes)} notes for {self.repo_name}")
        except Exception as e:
            logger.error(f"Failed to save notes for {self.repo_name}: {e}")

    def store(self, key: str, content: str, tags: list[str] | None = None) -> Note:
        """
        Store or update a note.

        If a note with the given key exists, it will be updated.
        Otherwise, a new note will be created.

        Args:
            key: Unique identifier for the note.
            content: The note content.
            tags: Optional list of tags for categorization.

        Returns:
            The stored or updated Note.
        """
        self._load()

        now = datetime.now(UTC)
        existing = self._notes.get(key)

        if existing:
            # Update existing note
            note = Note(
                key=key,
                content=content,
                tags=tags if tags is not None else existing.tags,
                created_at=existing.created_at,
                updated_at=now,
            )
        else:
            # Create new note
            note = Note(
                key=key,
                content=content,
                tags=tags or [],
                created_at=now,
                updated_at=now,
            )

        self._notes[key] = note
        self._save()
        return note

    def get(self, key: str) -> Note | None:
        """
        Retrieve a single note by key.

        Args:
            key: The note key to retrieve.

        Returns:
            The Note if found, None otherwise.
        """
        self._load()
        return self._notes.get(key)

    def list(self, tag: str | None = None) -> list[Note]:
        """
        List all notes, optionally filtered by tag.

        Args:
            tag: Optional tag to filter notes by.

        Returns:
            List of notes, sorted by updated_at (most recent first).
        """
        self._load()

        notes = list(self._notes.values())

        if tag:
            notes = [n for n in notes if tag in n.tags]

        # Sort by updated_at, most recent first
        notes.sort(key=lambda n: n.updated_at, reverse=True)
        return notes

    def delete(self, key: str) -> bool:
        """
        Delete a note by key.

        Args:
            key: The note key to delete.

        Returns:
            True if the note was deleted, False if it didn't exist.
        """
        self._load()

        if key in self._notes:
            del self._notes[key]
            self._save()
            return True
        return False

    def clear(self) -> int:
        """
        Delete all notes for this repository.

        Returns:
            Number of notes deleted.
        """
        self._load()

        count = len(self._notes)
        self._notes = {}
        self._save()
        return count


# Global cache of note stores per repository
_note_stores: dict[str, NoteStore] = {}


def get_note_store(repo_name: str) -> NoteStore:
    """
    Get or create a NoteStore for a repository.

    Uses a global cache to avoid creating multiple stores for the same repo.

    Args:
        repo_name: Name of the repository.

    Returns:
        NoteStore instance for the repository.
    """
    if repo_name not in _note_stores:
        _note_stores[repo_name] = NoteStore(repo_name)
    return _note_stores[repo_name]


def clear_note_store_cache() -> None:
    """
    Clear the global note store cache.

    Used for testing to ensure a clean state between tests.
    """
    _note_stores.clear()

