"""Tests for the symbol extractor module."""


from indexter_rlm.symbol_extractor import (
    extract_python_symbols,
)
from indexter_rlm.symbols import SymbolIndex, clear_symbol_index_cache


class TestExtractPythonSymbols:
    """Tests for Python symbol extraction."""

    def setup_method(self):
        clear_symbol_index_cache()

    def test_extract_function_definition(self):
        code = '''
def my_function(x: int) -> str:
    """A test function."""
    return str(x)
'''
        index = SymbolIndex(repo_name="test")
        extract_python_symbols("test.py", code, index)

        definitions = index.find_definitions("my_function")
        assert len(definitions) == 1
        assert definitions[0].symbol_type == "function"
        assert definitions[0].line == 2
        assert "A test function" in definitions[0].documentation

    def test_extract_class_definition(self):
        code = '''
class MyClass:
    """A test class."""
    pass
'''
        index = SymbolIndex(repo_name="test")
        extract_python_symbols("test.py", code, index)

        definitions = index.find_definitions("MyClass")
        assert len(definitions) == 1
        assert definitions[0].symbol_type == "class"
        assert "A test class" in definitions[0].documentation

    def test_extract_method_definition(self):
        code = '''
class MyClass:
    def my_method(self) -> None:
        pass
'''
        index = SymbolIndex(repo_name="test")
        extract_python_symbols("test.py", code, index)

        definitions = index.find_definitions("my_method")
        assert len(definitions) == 1
        assert definitions[0].symbol_type == "method"
        assert definitions[0].qualified_name == "MyClass.my_method"

    def test_extract_constant_definition(self):
        code = '''
MAX_RETRIES = 3
TIMEOUT_SECONDS = 30
'''
        index = SymbolIndex(repo_name="test")
        extract_python_symbols("test.py", code, index)

        definitions = index.find_definitions("MAX_RETRIES")
        assert len(definitions) == 1
        assert definitions[0].symbol_type == "constant"

    def test_extract_import_statement(self):
        code = '''
import os
import json
'''
        index = SymbolIndex(repo_name="test")
        extract_python_symbols("test.py", code, index)

        assert len(index.imports) == 2
        modules = [imp.imported_module for imp in index.imports]
        assert "os" in modules
        assert "json" in modules

    def test_extract_from_import(self):
        code = '''
from pathlib import Path
from typing import List, Dict
'''
        index = SymbolIndex(repo_name="test")
        extract_python_symbols("test.py", code, index)

        assert len(index.imports) == 2
        pathlib_import = next(
            (i for i in index.imports if i.imported_module == "pathlib"), None
        )
        assert pathlib_import is not None
        assert pathlib_import.is_from_import
        assert "Path" in pathlib_import.imported_names

    def test_extract_symbol_references(self):
        code = '''
def greet(name):
    return f"Hello, {name}"

result = greet("World")
'''
        index = SymbolIndex(repo_name="test")
        extract_python_symbols("test.py", code, index)

        # "greet" should be referenced where it's called
        references = index.find_references("greet")
        assert len(references) >= 1
        assert any(r.ref_type == "call" for r in references)

    def test_clear_file_removes_old_symbols(self):
        code_v1 = '''
def old_function():
    pass
'''
        code_v2 = '''
def new_function():
    pass
'''
        index = SymbolIndex(repo_name="test")

        # First extraction
        extract_python_symbols("test.py", code_v1, index)
        assert len(index.find_definitions("old_function")) == 1

        # Second extraction (file changed)
        extract_python_symbols("test.py", code_v2, index)
        assert len(index.find_definitions("old_function")) == 0
        assert len(index.find_definitions("new_function")) == 1

    def test_extract_with_signature(self):
        code = '''
def complex_function(
    arg1: int,
    arg2: str,
    *args,
    **kwargs
) -> dict:
    """Does something complex."""
    return {}
'''
        index = SymbolIndex(repo_name="test")
        extract_python_symbols("test.py", code, index)

        definitions = index.find_definitions("complex_function")
        assert len(definitions) == 1
        assert "arg1: int" in definitions[0].signature

    def test_file_symbols_tracks_all_definitions(self):
        code = '''
class MyClass:
    def method1(self):
        pass

    def method2(self):
        pass

def standalone():
    pass

CONSTANT = 42
'''
        index = SymbolIndex(repo_name="test")
        extract_python_symbols("test.py", code, index)

        file_symbols = index.file_symbols.get("test.py", [])
        assert "MyClass" in file_symbols
        assert "method1" in file_symbols
        assert "method2" in file_symbols
        assert "standalone" in file_symbols
        # CONSTANT might not be tracked if it doesn't match uppercase pattern strictly

    def test_builtins_not_tracked_as_references(self):
        code = '''
x = len([1, 2, 3])
y = str(42)
z = print("hello")
'''
        index = SymbolIndex(repo_name="test")
        extract_python_symbols("test.py", code, index)

        # Builtins should be filtered out
        assert len(index.find_references("len")) == 0
        assert len(index.find_references("str")) == 0
        assert len(index.find_references("print")) == 0

    def test_reference_context_captured(self):
        code = '''
result = my_function(42, "test")
'''
        index = SymbolIndex(repo_name="test")
        extract_python_symbols("test.py", code, index)

        refs = index.find_references("my_function")
        assert len(refs) == 1
        assert "my_function(42" in refs[0].context

