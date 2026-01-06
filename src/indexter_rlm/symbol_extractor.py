"""
Symbol extraction for building the symbol index.

This module provides functions to extract symbols, references, and imports
from source files using tree-sitter parsing.
"""

from __future__ import annotations

import logging
from pathlib import Path

from tree_sitter_language_pack import get_parser

from .symbols import (
    ImportRelation,
    SymbolDefinition,
    SymbolIndex,
    SymbolReference,
    load_symbol_index,
    save_symbol_index,
)

logger = logging.getLogger(__name__)


def extract_python_symbols(
    file_path: str,
    content: str,
    index: SymbolIndex,
) -> None:
    """
    Extract symbols, references, and imports from a Python file.

    Args:
        file_path: Path to the file (relative to repo root).
        content: File content.
        index: SymbolIndex to populate.
    """
    try:
        parser = get_parser("python")
        tree = parser.parse(content.encode())
    except Exception as e:
        logger.warning(f"Failed to parse {file_path}: {e}")
        return

    source_bytes = content.encode()

    # Clear any existing entries for this file
    index.clear_file(file_path)

    # Track defined symbols in this file for reference detection
    defined_symbols: set[str] = set()

    # Walk the tree and extract symbols
    _extract_definitions(tree.root_node, file_path, source_bytes, index, defined_symbols)
    _extract_imports(tree.root_node, file_path, source_bytes, index)
    _extract_references(tree.root_node, file_path, source_bytes, index, defined_symbols)


def _extract_definitions(
    node,
    file_path: str,
    source_bytes: bytes,
    index: SymbolIndex,
    defined_symbols: set[str],
    parent_scope: str = "",
) -> None:
    """Extract symbol definitions recursively."""
    if node.type == "function_definition":
        name_node = node.child_by_field_name("name")
        if name_node:
            name = name_node.text.decode()
            qualified = f"{parent_scope}.{name}" if parent_scope else name
            symbol_type = "method" if parent_scope else "function"

            # Get signature
            body = node.child_by_field_name("body")
            sig_end = body.start_byte if body else node.end_byte
            signature = source_bytes[node.start_byte : sig_end].decode().rstrip().rstrip(":")

            # Get docstring
            documentation = _get_docstring(node, source_bytes)

            index.add_definition(
                SymbolDefinition(
                    name=name,
                    qualified_name=qualified,
                    symbol_type=symbol_type,
                    file_path=file_path,
                    line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    signature=signature,
                    documentation=documentation or "",
                )
            )
            defined_symbols.add(name)

    elif node.type == "class_definition":
        name_node = node.child_by_field_name("name")
        if name_node:
            name = name_node.text.decode()
            qualified = f"{parent_scope}.{name}" if parent_scope else name

            index.add_definition(
                SymbolDefinition(
                    name=name,
                    qualified_name=qualified,
                    symbol_type="class",
                    file_path=file_path,
                    line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    documentation=_get_docstring(node, source_bytes) or "",
                )
            )
            defined_symbols.add(name)

            # Recurse into class body with new scope
            body = node.child_by_field_name("body")
            if body:
                for child in body.children:
                    _extract_definitions(
                        child, file_path, source_bytes, index, defined_symbols, name
                    )
            return  # Don't recurse normally, we handled the body

    elif node.type == "assignment":
        # Module-level constant assignments
        if node.parent and node.parent.type == "module":
            left = None
            for child in node.children:
                if child.type == "identifier":
                    left = child
                    break
            if left:
                name = left.text.decode()
                # Only track UPPER_CASE constants
                if name.isupper() or (name.replace("_", "").isupper() and "_" in name):
                    index.add_definition(
                        SymbolDefinition(
                            name=name,
                            qualified_name=name,
                            symbol_type="constant",
                            file_path=file_path,
                            line=node.start_point[0] + 1,
                            end_line=node.end_point[0] + 1,
                        )
                    )
                    defined_symbols.add(name)

    # Recurse into children
    for child in node.children:
        _extract_definitions(child, file_path, source_bytes, index, defined_symbols, parent_scope)


def _extract_imports(
    node,
    file_path: str,
    source_bytes: bytes,
    index: SymbolIndex,
) -> None:
    """Extract import relationships."""
    if node.type == "import_statement":
        # import module.submodule
        for child in node.children:
            if child.type == "dotted_name":
                module_name = child.text.decode()
                index.add_import(
                    ImportRelation(
                        importing_file=file_path,
                        imported_module=module_name,
                        imported_names=[],
                        line=node.start_point[0] + 1,
                        is_from_import=False,
                    )
                )
            elif child.type == "aliased_import":
                # import X as Y
                name_node = child.child_by_field_name("name")
                if name_node:
                    module_name = name_node.text.decode()
                    index.add_import(
                        ImportRelation(
                            importing_file=file_path,
                            imported_module=module_name,
                            imported_names=[],
                            line=node.start_point[0] + 1,
                            is_from_import=False,
                        )
                    )

    elif node.type == "import_from_statement":
        # from module import name1, name2
        module_node = node.child_by_field_name("module_name")
        module_name = module_node.text.decode() if module_node else ""

        imported_names: list[str] = []
        for child in node.children:
            if child.type == "dotted_name" and child != module_node:
                imported_names.append(child.text.decode())
            elif child.type == "aliased_import":
                name_node = child.child_by_field_name("name")
                if name_node:
                    imported_names.append(name_node.text.decode())
            elif child.type == "wildcard_import":
                imported_names.append("*")

        if module_name or imported_names:
            index.add_import(
                ImportRelation(
                    importing_file=file_path,
                    imported_module=module_name,
                    imported_names=imported_names,
                    line=node.start_point[0] + 1,
                    is_from_import=True,
                )
            )

    # Recurse into children
    for child in node.children:
        _extract_imports(child, file_path, source_bytes, index)


def _extract_references(
    node,
    file_path: str,
    source_bytes: bytes,
    index: SymbolIndex,
    defined_symbols: set[str],
    in_import: bool = False,
) -> None:
    """Extract symbol references (usages)."""
    # Skip import statements - they're tracked separately
    if node.type in ("import_statement", "import_from_statement"):
        return

    if node.type == "identifier":
        name = node.text.decode()

        # Skip if this is a definition site (part of function/class/assignment)
        parent = node.parent
        if parent:
            if parent.type in ("function_definition", "class_definition"):
                if parent.child_by_field_name("name") == node:
                    return  # This is a definition, not a reference
            if parent.type == "assignment" and parent.children[0] == node:
                return  # Left side of assignment

        # Skip common Python builtins to reduce noise
        builtins = {
            "True",
            "False",
            "None",
            "self",
            "cls",
            "print",
            "len",
            "range",
            "str",
            "int",
            "float",
            "bool",
            "list",
            "dict",
            "set",
            "tuple",
            "open",
            "type",
            "isinstance",
            "hasattr",
            "getattr",
            "setattr",
            "super",
            "staticmethod",
            "classmethod",
            "property",
            "Exception",
        }
        if name in builtins:
            return

        # Get context line
        line_start = source_bytes.rfind(b"\n", 0, node.start_byte) + 1
        line_end = source_bytes.find(b"\n", node.end_byte)
        if line_end == -1:
            line_end = len(source_bytes)
        context = source_bytes[line_start:line_end].decode().strip()

        # Determine reference type
        ref_type = "usage"
        if parent and parent.type == "call":
            ref_type = "call"
        elif parent and parent.type == "attribute":
            ref_type = "attribute"

        index.add_reference(
            SymbolReference(
                symbol_name=name,
                file_path=file_path,
                line=node.start_point[0] + 1,
                column=node.start_point[1],
                context=context[:200],  # Limit context length
                ref_type=ref_type,
            )
        )

    # Recurse into children
    for child in node.children:
        _extract_references(child, file_path, source_bytes, index, defined_symbols, in_import)


def _get_docstring(node, source_bytes: bytes) -> str | None:
    """Extract docstring from a function or class definition."""
    if node.type not in ("function_definition", "class_definition"):
        return None

    body = node.child_by_field_name("body")
    if not body or not body.children:
        return None

    first_stmt = body.children[0]

    # Direct string literal
    if first_stmt.type == "string":
        return _strip_docstring(first_stmt.text.decode())

    # String wrapped in expression_statement
    if first_stmt.type == "expression_statement" and first_stmt.children:
        expr = first_stmt.children[0]
        if expr.type == "string":
            return _strip_docstring(expr.text.decode())

    return None


def _strip_docstring(text: str) -> str:
    """Remove docstring quotes and normalize whitespace."""
    for quote in ('"""', "'''", '"', "'"):
        if text.startswith(quote) and text.endswith(quote):
            text = text[len(quote) : -len(quote)]
            break
    return text.strip()


async def build_symbol_index(
    repo_name: str,
    repo_path: Path,
    files: list[str],
) -> SymbolIndex:
    """
    Build or update the symbol index for a repository.

    Args:
        repo_name: Name of the repository.
        repo_path: Path to the repository root.
        files: List of file paths to index (relative to repo root).

    Returns:
        The updated SymbolIndex.
    """
    index = load_symbol_index(repo_name)

    for file_path in files:
        if not file_path.endswith(".py"):
            continue  # Only Python for now

        full_path = repo_path / file_path
        if not full_path.exists():
            index.clear_file(file_path)
            continue

        try:
            content = full_path.read_text(encoding="utf-8")
            extract_python_symbols(file_path, content, index)
        except Exception as e:
            logger.warning(f"Failed to extract symbols from {file_path}: {e}")

    save_symbol_index(index)
    return index
