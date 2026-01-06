"""Tests for the symbol index module."""

from indexter_rlm.symbols import (
    ImportRelation,
    SymbolDefinition,
    SymbolIndex,
    SymbolReference,
    clear_symbol_index_cache,
    get_symbol_index_path,
    load_symbol_index,
    save_symbol_index,
)


class TestSymbolDefinition:
    """Tests for SymbolDefinition model."""

    def test_create_function_definition(self):
        defn = SymbolDefinition(
            name="my_function",
            qualified_name="my_function",
            symbol_type="function",
            file_path="src/module.py",
            line=10,
            end_line=20,
            signature="def my_function(x: int) -> str",
            documentation="A test function.",
        )
        assert defn.name == "my_function"
        assert defn.symbol_type == "function"
        assert defn.line == 10

    def test_create_method_definition(self):
        defn = SymbolDefinition(
            name="my_method",
            qualified_name="MyClass.my_method",
            symbol_type="method",
            file_path="src/module.py",
            line=15,
            end_line=25,
        )
        assert defn.qualified_name == "MyClass.my_method"
        assert defn.symbol_type == "method"


class TestSymbolReference:
    """Tests for SymbolReference model."""

    def test_create_usage_reference(self):
        ref = SymbolReference(
            symbol_name="my_function",
            file_path="src/other.py",
            line=5,
            column=10,
            context="result = my_function(42)",
            ref_type="call",
        )
        assert ref.symbol_name == "my_function"
        assert ref.ref_type == "call"

    def test_default_ref_type(self):
        ref = SymbolReference(
            symbol_name="var",
            file_path="src/module.py",
            line=1,
        )
        assert ref.ref_type == "usage"


class TestImportRelation:
    """Tests for ImportRelation model."""

    def test_import_statement(self):
        imp = ImportRelation(
            importing_file="src/main.py",
            imported_module="os.path",
            imported_names=[],
            line=1,
            is_from_import=False,
        )
        assert imp.imported_module == "os.path"
        assert not imp.is_from_import

    def test_from_import_statement(self):
        imp = ImportRelation(
            importing_file="src/main.py",
            imported_module="module",
            imported_names=["func1", "func2"],
            line=2,
            is_from_import=True,
        )
        assert imp.is_from_import
        assert "func1" in imp.imported_names


class TestSymbolIndex:
    """Tests for SymbolIndex."""

    def test_add_and_find_definition(self):
        index = SymbolIndex(repo_name="test-repo")
        defn = SymbolDefinition(
            name="TestClass",
            qualified_name="TestClass",
            symbol_type="class",
            file_path="src/module.py",
            line=1,
            end_line=50,
        )
        index.add_definition(defn)

        results = index.find_definitions("TestClass")
        assert len(results) == 1
        assert results[0].name == "TestClass"

    def test_add_and_find_reference(self):
        index = SymbolIndex(repo_name="test-repo")
        ref = SymbolReference(
            symbol_name="TestClass",
            file_path="src/other.py",
            line=10,
        )
        index.add_reference(ref)

        results = index.find_references("TestClass")
        assert len(results) == 1
        assert results[0].file_path == "src/other.py"

    def test_add_import(self):
        index = SymbolIndex(repo_name="test-repo")
        imp = ImportRelation(
            importing_file="src/main.py",
            imported_module="module",
            imported_names=["func"],
            line=1,
            is_from_import=True,
        )
        index.add_import(imp)

        assert len(index.imports) == 1

    def test_file_symbols_tracking(self):
        index = SymbolIndex(repo_name="test-repo")
        defn = SymbolDefinition(
            name="MyFunc",
            qualified_name="MyFunc",
            symbol_type="function",
            file_path="src/module.py",
            line=1,
            end_line=10,
        )
        index.add_definition(defn)

        assert "MyFunc" in index.file_symbols["src/module.py"]

    def test_clear_file(self):
        index = SymbolIndex(repo_name="test-repo")
        defn = SymbolDefinition(
            name="MyFunc",
            qualified_name="MyFunc",
            symbol_type="function",
            file_path="src/module.py",
            line=1,
            end_line=10,
        )
        ref = SymbolReference(
            symbol_name="MyFunc",
            file_path="src/module.py",
            line=5,
        )
        imp = ImportRelation(
            importing_file="src/module.py",
            imported_module="other",
            line=1,
        )
        index.add_definition(defn)
        index.add_reference(ref)
        index.add_import(imp)

        index.clear_file("src/module.py")

        assert index.find_definitions("MyFunc") == []
        assert index.find_references("MyFunc") == []
        assert len(index.imports) == 0
        assert "src/module.py" not in index.file_symbols

    def test_get_importers(self):
        index = SymbolIndex(repo_name="test-repo")
        imp = ImportRelation(
            importing_file="src/main.py",
            imported_module="auth.service",
            imported_names=["AuthService"],
            line=1,
            is_from_import=True,
        )
        index.add_import(imp)

        importers = index.get_importers("auth.service")
        assert len(importers) == 1

    def test_multiple_definitions_same_name(self):
        """Test that symbols with same name in different files are tracked."""
        index = SymbolIndex(repo_name="test-repo")
        defn1 = SymbolDefinition(
            name="helper",
            qualified_name="helper",
            symbol_type="function",
            file_path="src/a.py",
            line=1,
            end_line=5,
        )
        defn2 = SymbolDefinition(
            name="helper",
            qualified_name="helper",
            symbol_type="function",
            file_path="src/b.py",
            line=1,
            end_line=5,
        )
        index.add_definition(defn1)
        index.add_definition(defn2)

        results = index.find_definitions("helper")
        assert len(results) == 2


class TestSymbolIndexPersistence:
    """Tests for symbol index persistence."""

    def setup_method(self):
        clear_symbol_index_cache()

    def test_get_symbol_index_path(self):
        path = get_symbol_index_path("my-repo")
        assert path.name == "my-repo.json"
        assert "symbols" in str(path)

    def test_load_creates_new_index(self):
        index = load_symbol_index("new-test-repo-xyz")
        assert index.repo_name == "new-test-repo-xyz"
        assert len(index.definitions) == 0

    def test_save_and_load(self, tmp_path, monkeypatch):
        # Monkeypatch config dir to use tmp_path
        from indexter_rlm import symbols

        def mock_config_dir():
            return tmp_path / ".config" / "indexter"

        monkeypatch.setattr(symbols, "get_config_dir", mock_config_dir)
        clear_symbol_index_cache()

        # Create and save index
        index = SymbolIndex(repo_name="persist-test")
        defn = SymbolDefinition(
            name="TestFunc",
            qualified_name="TestFunc",
            symbol_type="function",
            file_path="src/test.py",
            line=1,
            end_line=10,
        )
        index.add_definition(defn)
        save_symbol_index(index)

        # Clear cache and reload
        clear_symbol_index_cache()
        loaded = load_symbol_index("persist-test")

        assert loaded.repo_name == "persist-test"
        assert len(loaded.find_definitions("TestFunc")) == 1
