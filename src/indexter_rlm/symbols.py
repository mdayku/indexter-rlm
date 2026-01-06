"""
Symbol index for cross-file reference tracking.

This module provides a symbol index that tracks:
- Symbol definitions (functions, classes, methods, constants)
- Symbol usages/references across files
- Import relationships for transitive dependency tracking

The index is built during parsing and stored per-repository in JSON format.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from .config import get_config_dir

logger = logging.getLogger(__name__)


class SymbolDefinition(BaseModel):
    """A symbol definition (function, class, method, constant)."""

    name: str
    """The symbol name (e.g., 'UserService', 'authenticate')."""

    qualified_name: str
    """Fully qualified name including parent scope (e.g., 'UserService.authenticate')."""

    symbol_type: Literal["function", "class", "method", "constant", "variable"]
    """The type of symbol."""

    file_path: str
    """Path to the file containing this definition (relative to repo root)."""

    line: int
    """Line number where the symbol is defined (1-based)."""

    end_line: int
    """End line number of the definition."""

    signature: str = ""
    """Function/method signature if applicable."""

    documentation: str = ""
    """Docstring or documentation comment."""


class SymbolReference(BaseModel):
    """A reference to a symbol (usage site)."""

    symbol_name: str
    """The name being referenced."""

    file_path: str
    """Path to the file containing this reference."""

    line: int
    """Line number of the reference (1-based)."""

    column: int = 0
    """Column number of the reference (0-based)."""

    context: str = ""
    """The line of code containing the reference."""

    ref_type: Literal["usage", "import", "call", "attribute"] = "usage"
    """Type of reference."""


class ImportRelation(BaseModel):
    """An import relationship between files."""

    importing_file: str
    """The file that contains the import statement."""

    imported_module: str
    """The module being imported (e.g., 'auth.service')."""

    imported_names: list[str] = Field(default_factory=list)
    """Specific names imported (empty for 'import module')."""

    line: int
    """Line number of the import statement."""

    is_from_import: bool = False
    """True if 'from X import Y' style."""


class SymbolIndex(BaseModel):
    """Symbol index for a repository."""

    repo_name: str
    """Repository name."""

    definitions: dict[str, list[SymbolDefinition]] = Field(default_factory=dict)
    """Map from symbol name to list of definitions (can have multiple with same name)."""

    references: dict[str, list[SymbolReference]] = Field(default_factory=dict)
    """Map from symbol name to list of references."""

    imports: list[ImportRelation] = Field(default_factory=list)
    """All import relationships in the repository."""

    file_symbols: dict[str, list[str]] = Field(default_factory=dict)
    """Map from file path to list of symbol names defined in that file."""

    def add_definition(self, definition: SymbolDefinition) -> None:
        """Add a symbol definition to the index."""
        name = definition.name
        if name not in self.definitions:
            self.definitions[name] = []
        self.definitions[name].append(definition)

        # Track which file defines this symbol
        if definition.file_path not in self.file_symbols:
            self.file_symbols[definition.file_path] = []
        if name not in self.file_symbols[definition.file_path]:
            self.file_symbols[definition.file_path].append(name)

    def add_reference(self, reference: SymbolReference) -> None:
        """Add a symbol reference to the index."""
        name = reference.symbol_name
        if name not in self.references:
            self.references[name] = []
        self.references[name].append(reference)

    def add_import(self, import_rel: ImportRelation) -> None:
        """Add an import relationship to the index."""
        self.imports.append(import_rel)

    def find_definitions(self, symbol_name: str) -> list[SymbolDefinition]:
        """Find all definitions of a symbol."""
        return self.definitions.get(symbol_name, [])

    def find_references(self, symbol_name: str) -> list[SymbolReference]:
        """Find all references to a symbol."""
        return self.references.get(symbol_name, [])

    def get_importers(self, module_path: str) -> list[ImportRelation]:
        """Find all files that import from a given module."""
        return [
            imp
            for imp in self.imports
            if imp.imported_module == module_path or imp.imported_module.endswith(f".{module_path}")
        ]

    def get_import_chain(self, symbol_name: str, max_depth: int = 10) -> list[list[str]]:
        """
        Get transitive import chains for a symbol.

        Returns a list of import paths, each showing how the symbol
        reaches different files through imports.
        """
        # Find where the symbol is defined
        definitions = self.find_definitions(symbol_name)
        if not definitions:
            return []

        chains: list[list[str]] = []

        for defn in definitions:
            # Convert file path to module name (remove .py, replace / with .)
            source_module = self._file_to_module(defn.file_path)

            # BFS to find all import chains
            visited: set[str] = set()
            queue: list[tuple[str, list[str]]] = [(source_module, [defn.file_path])]

            while queue and len(chains) < 100:  # Limit results
                current_module, path = queue.pop(0)

                if current_module in visited:
                    continue
                visited.add(current_module)

                # Find files that import this module
                for imp in self.imports:
                    if self._modules_match(imp.imported_module, current_module):
                        # Check if specific name is imported (for 'from X import Y')
                        if imp.is_from_import and imp.imported_names:
                            names = imp.imported_names
                            if symbol_name not in names and "*" not in names:
                                continue

                        new_path = path + [imp.importing_file]
                        if len(new_path) <= max_depth:
                            chains.append(new_path)
                            importer_module = self._file_to_module(imp.importing_file)
                            queue.append((importer_module, new_path))

        return chains

    def clear_file(self, file_path: str) -> None:
        """Remove all symbols and references from a specific file."""
        # Remove definitions from this file
        for symbol_name in list(self.definitions.keys()):
            self.definitions[symbol_name] = [
                d for d in self.definitions[symbol_name] if d.file_path != file_path
            ]
            if not self.definitions[symbol_name]:
                del self.definitions[symbol_name]

        # Remove references from this file
        for symbol_name in list(self.references.keys()):
            self.references[symbol_name] = [
                r for r in self.references[symbol_name] if r.file_path != file_path
            ]
            if not self.references[symbol_name]:
                del self.references[symbol_name]

        # Remove imports from this file
        self.imports = [i for i in self.imports if i.importing_file != file_path]

        # Remove file from file_symbols
        if file_path in self.file_symbols:
            del self.file_symbols[file_path]

    def _file_to_module(self, file_path: str) -> str:
        """Convert a file path to a module name."""
        # Remove .py extension and convert path separators
        module = file_path.replace("\\", "/")
        if module.endswith(".py"):
            module = module[:-3]
        return module.replace("/", ".")

    def _modules_match(self, import_module: str, source_module: str) -> bool:
        """Check if an import matches a source module."""
        # Exact match
        if import_module == source_module:
            return True
        # Suffix match (import might be relative)
        if source_module.endswith(import_module):
            return True
        if import_module.endswith(source_module):
            return True
        return False


# Storage for symbol indices
_symbol_indices: dict[str, SymbolIndex] = {}


def get_symbol_index_path(repo_name: str) -> Path:
    """Get the path to the symbol index file for a repository."""
    symbols_dir = get_config_dir() / "symbols"
    symbols_dir.mkdir(parents=True, exist_ok=True)
    return symbols_dir / f"{repo_name}.json"


def load_symbol_index(repo_name: str) -> SymbolIndex:
    """Load or create a symbol index for a repository."""
    if repo_name in _symbol_indices:
        return _symbol_indices[repo_name]

    index_path = get_symbol_index_path(repo_name)
    if index_path.exists():
        try:
            data = json.loads(index_path.read_text(encoding="utf-8"))
            index = SymbolIndex(**data)
        except Exception as e:
            logger.warning(f"Failed to load symbol index for {repo_name}: {e}")
            index = SymbolIndex(repo_name=repo_name)
    else:
        index = SymbolIndex(repo_name=repo_name)

    _symbol_indices[repo_name] = index
    return index


def save_symbol_index(index: SymbolIndex) -> None:
    """Save a symbol index to disk."""
    index_path = get_symbol_index_path(index.repo_name)
    try:
        index_path.write_text(json.dumps(index.model_dump(), indent=2), encoding="utf-8")
    except Exception as e:
        logger.error(f"Failed to save symbol index: {e}")


def clear_symbol_index_cache() -> None:
    """Clear the in-memory cache of symbol indices."""
    _symbol_indices.clear()
