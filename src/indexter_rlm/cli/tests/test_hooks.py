"""Tests for git hook management."""

from typer.testing import CliRunner

from indexter_rlm.cli.hooks import (
    HOOK_MARKER,
    HOOK_TEMPLATES,
    get_git_hooks_dir,
    hooks_app,
    is_indexter_hook,
)

runner = CliRunner()


class TestGetGitHooksDir:
    """Tests for get_git_hooks_dir function."""

    def test_regular_git_repo(self, tmp_path):
        """Test finding hooks dir in a regular git repo."""
        git_dir = tmp_path / ".git"
        git_dir.mkdir()

        result = get_git_hooks_dir(tmp_path)

        assert result == git_dir / "hooks"

    def test_not_a_git_repo(self, tmp_path):
        """Test returns None for non-git directory."""
        result = get_git_hooks_dir(tmp_path)

        assert result is None

    def test_git_worktree(self, tmp_path):
        """Test finding hooks dir in a git worktree."""
        # Create a git file pointing to actual git dir
        actual_git_dir = tmp_path / "actual_git"
        actual_git_dir.mkdir()

        git_file = tmp_path / "worktree" / ".git"
        git_file.parent.mkdir()
        git_file.write_text(f"gitdir: {actual_git_dir}")

        result = get_git_hooks_dir(git_file.parent)

        assert result == actual_git_dir / "hooks"


class TestIsIndexterHook:
    """Tests for is_indexter_hook function."""

    def test_indexter_hook(self, tmp_path):
        """Test detecting an indexter-rlm hook."""
        hook_file = tmp_path / "post-commit"
        hook_file.write_text(f"#!/bin/sh\n{HOOK_MARKER}\necho 'test'")

        assert is_indexter_hook(hook_file) is True

    def test_other_hook(self, tmp_path):
        """Test detecting a non-indexter hook."""
        hook_file = tmp_path / "post-commit"
        hook_file.write_text("#!/bin/sh\necho 'other hook'")

        assert is_indexter_hook(hook_file) is False

    def test_nonexistent_hook(self, tmp_path):
        """Test handling nonexistent hook file."""
        hook_file = tmp_path / "nonexistent"

        assert is_indexter_hook(hook_file) is False


class TestHookInstall:
    """Tests for hook install command."""

    def test_install_post_commit_hook(self, tmp_path):
        """Test installing post-commit hook."""
        # Create a git repo
        git_dir = tmp_path / ".git"
        git_dir.mkdir()

        result = runner.invoke(hooks_app, ["install", str(tmp_path)])

        assert result.exit_code == 0
        assert "Installed post-commit hook" in result.output

        hook_path = git_dir / "hooks" / "post-commit"
        assert hook_path.exists()
        assert is_indexter_hook(hook_path)

    def test_install_pre_push_hook(self, tmp_path):
        """Test installing pre-push hook."""
        git_dir = tmp_path / ".git"
        git_dir.mkdir()

        result = runner.invoke(hooks_app, ["install", str(tmp_path), "--type", "pre-push"])

        assert result.exit_code == 0
        assert "Installed pre-push hook" in result.output

        hook_path = git_dir / "hooks" / "pre-push"
        assert hook_path.exists()

    def test_install_pre_commit_hook_warning(self, tmp_path):
        """Test that pre-commit hook shows warning."""
        git_dir = tmp_path / ".git"
        git_dir.mkdir()

        result = runner.invoke(hooks_app, ["install", str(tmp_path), "--type", "pre-commit"])

        assert result.exit_code == 0
        assert "Warning" in result.output or "pre-commit" in result.output

    def test_install_not_git_repo(self, tmp_path):
        """Test error when not a git repo."""
        result = runner.invoke(hooks_app, ["install", str(tmp_path)])

        assert result.exit_code == 1
        assert "is not a git" in result.output

    def test_install_hook_exists_no_force(self, tmp_path):
        """Test error when hook exists and not using force."""
        git_dir = tmp_path / ".git"
        hooks_dir = git_dir / "hooks"
        hooks_dir.mkdir(parents=True)

        # Create existing non-indexter hook
        hook_path = hooks_dir / "post-commit"
        hook_path.write_text("#!/bin/sh\necho 'existing'")

        result = runner.invoke(hooks_app, ["install", str(tmp_path)])

        assert result.exit_code == 1
        assert "already exists" in result.output

    def test_install_hook_force_overwrites(self, tmp_path):
        """Test force flag overwrites existing hook."""
        git_dir = tmp_path / ".git"
        hooks_dir = git_dir / "hooks"
        hooks_dir.mkdir(parents=True)

        # Create existing hook
        hook_path = hooks_dir / "post-commit"
        hook_path.write_text("#!/bin/sh\necho 'existing'")

        result = runner.invoke(hooks_app, ["install", str(tmp_path), "--force"])

        assert result.exit_code == 0
        assert "Installed" in result.output

        # Check backup was created
        backup_path = hooks_dir / "post-commit.backup"
        assert backup_path.exists()


class TestHookUninstall:
    """Tests for hook uninstall command."""

    def test_uninstall_single_hook(self, tmp_path):
        """Test uninstalling a single hook."""
        git_dir = tmp_path / ".git"
        hooks_dir = git_dir / "hooks"
        hooks_dir.mkdir(parents=True)

        # Install hook first
        hook_path = hooks_dir / "post-commit"
        hook_path.write_text(HOOK_TEMPLATES["post-commit"])

        result = runner.invoke(hooks_app, ["uninstall", str(tmp_path), "--type", "post-commit"])

        assert result.exit_code == 0
        assert "Uninstalled" in result.output
        assert not hook_path.exists()

    def test_uninstall_all_hooks(self, tmp_path):
        """Test uninstalling all hooks."""
        git_dir = tmp_path / ".git"
        hooks_dir = git_dir / "hooks"
        hooks_dir.mkdir(parents=True)

        # Install multiple hooks
        for hook_type, content in HOOK_TEMPLATES.items():
            (hooks_dir / hook_type).write_text(content)

        result = runner.invoke(hooks_app, ["uninstall", str(tmp_path)])

        assert result.exit_code == 0
        assert "Uninstalled" in result.output

        # All hooks should be removed
        for hook_type in HOOK_TEMPLATES:
            assert not (hooks_dir / hook_type).exists()

    def test_uninstall_preserves_other_hooks(self, tmp_path):
        """Test that non-indexter hooks are preserved."""
        git_dir = tmp_path / ".git"
        hooks_dir = git_dir / "hooks"
        hooks_dir.mkdir(parents=True)

        # Create a non-indexter hook
        hook_path = hooks_dir / "post-commit"
        hook_path.write_text("#!/bin/sh\necho 'other'")

        result = runner.invoke(hooks_app, ["uninstall", str(tmp_path)])

        assert result.exit_code == 0
        # Hook should still exist since it's not an indexter hook
        assert hook_path.exists()


class TestHookStatus:
    """Tests for hook status command."""

    def test_status_no_hooks(self, tmp_path):
        """Test status with no hooks installed."""
        git_dir = tmp_path / ".git"
        git_dir.mkdir()

        result = runner.invoke(hooks_app, ["status", str(tmp_path)])

        assert result.exit_code == 0
        assert "not installed" in result.output

    def test_status_with_indexter_hook(self, tmp_path):
        """Test status with indexter hook installed."""
        git_dir = tmp_path / ".git"
        hooks_dir = git_dir / "hooks"
        hooks_dir.mkdir(parents=True)

        # Install hook
        hook_path = hooks_dir / "post-commit"
        hook_path.write_text(HOOK_TEMPLATES["post-commit"])

        result = runner.invoke(hooks_app, ["status", str(tmp_path)])

        assert result.exit_code == 0
        assert "indexter-rlm" in result.output

    def test_status_not_git_repo(self, tmp_path):
        """Test status error for non-git directory."""
        result = runner.invoke(hooks_app, ["status", str(tmp_path)])

        assert result.exit_code == 1
        assert "is not a git" in result.output
