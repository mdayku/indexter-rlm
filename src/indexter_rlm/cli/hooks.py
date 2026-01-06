"""
Git hook management for Indexter-RLM.

Provides commands to install and manage git hooks that automatically
keep the semantic index in sync with repository changes.

Available hooks:
- post-commit: Re-index after each commit (recommended)
- pre-push: Re-index before pushing (batches multiple commits)
"""

import stat
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

console = Console()

hooks_app = typer.Typer(
    name="hook",
    help="Manage git hooks for automatic re-indexing.",
    no_args_is_help=True,
)


# Hook script templates
POST_COMMIT_HOOK = """\
#!/bin/sh
# Indexter-RLM post-commit hook
# Automatically re-indexes the repository after each commit
# Installed by: indexter-rlm hook install

# Get the repository name from the directory name
REPO_NAME=$(basename "$(git rev-parse --show-toplevel)")

# Run indexter-rlm index in the background
# This won't block the commit and keeps the index up to date
echo "[indexter-rlm] Updating index for $REPO_NAME..."
indexter-rlm index "$REPO_NAME" 2>/dev/null &

exit 0
"""

PRE_PUSH_HOOK = """\
#!/bin/sh
# Indexter-RLM pre-push hook
# Ensures the semantic index is up to date before pushing
# Installed by: indexter-rlm hook install

# Get the repository name from the directory name
REPO_NAME=$(basename "$(git rev-parse --show-toplevel)")

echo "[indexter-rlm] Ensuring index is up to date for $REPO_NAME..."
indexter-rlm index "$REPO_NAME"

exit $?
"""

PRE_COMMIT_HOOK = """\
#!/bin/sh
# Indexter-RLM pre-commit hook
# Re-indexes the repository before each commit
# WARNING: This can slow down commits, especially with OpenAI embeddings
# Installed by: indexter-rlm hook install

# Get the repository name from the directory name
REPO_NAME=$(basename "$(git rev-parse --show-toplevel)")

echo "[indexter-rlm] Updating index for $REPO_NAME..."
indexter-rlm index "$REPO_NAME"

exit $?
"""

HOOK_TEMPLATES = {
    "post-commit": POST_COMMIT_HOOK,
    "pre-push": PRE_PUSH_HOOK,
    "pre-commit": PRE_COMMIT_HOOK,
}

HOOK_MARKER = "# Installed by: indexter-rlm hook install"


def get_git_hooks_dir(repo_path: Path) -> Path | None:
    """Find the .git/hooks directory for a repository.

    Args:
        repo_path: Path to the repository.

    Returns:
        Path to the hooks directory, or None if not a git repo.
    """
    git_dir = repo_path / ".git"
    if git_dir.is_dir():
        return git_dir / "hooks"

    # Check for worktree (git file pointing to actual git dir)
    git_file = repo_path / ".git"
    if git_file.is_file():
        content = git_file.read_text().strip()
        if content.startswith("gitdir:"):
            actual_git_dir = Path(content.split(":", 1)[1].strip())
            if not actual_git_dir.is_absolute():
                actual_git_dir = repo_path / actual_git_dir
            return actual_git_dir / "hooks"

    return None


def is_indexter_hook(hook_path: Path) -> bool:
    """Check if a hook was installed by indexter-rlm.

    Args:
        hook_path: Path to the hook file.

    Returns:
        True if the hook contains our marker.
    """
    if not hook_path.exists():
        return False
    content = hook_path.read_text()
    return HOOK_MARKER in content


def make_executable(path: Path) -> None:
    """Make a file executable (Unix-like systems)."""
    current_mode = path.stat().st_mode
    path.chmod(current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


@hooks_app.command("install")
def install_hook(
    repo_path: Annotated[
        Path,
        typer.Argument(
            help="Path to the git repository",
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = Path("."),
    hook_type: Annotated[
        str,
        typer.Option(
            "--type",
            "-t",
            help="Hook type: post-commit (default), pre-push, or pre-commit",
        ),
    ] = "post-commit",
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="Overwrite existing hook (even if not installed by indexter-rlm)",
        ),
    ] = False,
) -> None:
    """Install a git hook for automatic re-indexing.

    By default, installs a post-commit hook that updates the index
    in the background after each commit.

    Examples:
        indexter-rlm hook install
        indexter-rlm hook install /path/to/repo --type pre-push
        indexter-rlm hook install --type pre-commit --force
    """
    if hook_type not in HOOK_TEMPLATES:
        console.print(
            f"[red]Error:[/red] Unknown hook type '{hook_type}'. "
            f"Valid types: {', '.join(HOOK_TEMPLATES.keys())}"
        )
        raise typer.Exit(1)

    hooks_dir = get_git_hooks_dir(repo_path)
    if hooks_dir is None:
        console.print(f"[red]Error:[/red] {repo_path} is not a git repository")
        raise typer.Exit(1)

    # Ensure hooks directory exists
    hooks_dir.mkdir(parents=True, exist_ok=True)

    hook_path = hooks_dir / hook_type

    # Check for existing hook
    if hook_path.exists():
        if is_indexter_hook(hook_path):
            console.print(
                f"[yellow]Note:[/yellow] {hook_type} hook already installed by indexter-rlm. "
                "Use --force to reinstall."
            )
            if not force:
                return
        elif not force:
            console.print(
                f"[red]Error:[/red] A {hook_type} hook already exists at {hook_path}\n"
                "Use --force to overwrite, or manually integrate the indexter-rlm command."
            )
            raise typer.Exit(1)
        else:
            # Backup existing hook
            backup_path = hook_path.with_suffix(".backup")
            hook_path.rename(backup_path)
            console.print(f"[dim]Backed up existing hook to {backup_path}[/dim]")

    # Write the hook
    hook_content = HOOK_TEMPLATES[hook_type]
    hook_path.write_text(hook_content)
    make_executable(hook_path)

    console.print(f"[green]✓[/green] Installed {hook_type} hook at {hook_path}")

    if hook_type == "post-commit":
        console.print("[dim]The index will be updated in the background after each commit.[/dim]")
    elif hook_type == "pre-push":
        console.print("[dim]The index will be updated before each push.[/dim]")
    elif hook_type == "pre-commit":
        console.print(
            "[yellow]Warning:[/yellow] pre-commit hooks can slow down commits. "
            "Consider using post-commit instead."
        )


@hooks_app.command("uninstall")
def uninstall_hook(
    repo_path: Annotated[
        Path,
        typer.Argument(
            help="Path to the git repository",
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = Path("."),
    hook_type: Annotated[
        str,
        typer.Option(
            "--type",
            "-t",
            help="Hook type to uninstall (or 'all' for all hooks)",
        ),
    ] = "all",
) -> None:
    """Uninstall indexter-rlm git hooks.

    Only removes hooks that were installed by indexter-rlm.

    Examples:
        indexter-rlm hook uninstall
        indexter-rlm hook uninstall --type post-commit
    """
    hooks_dir = get_git_hooks_dir(repo_path)
    if hooks_dir is None:
        console.print(f"[red]Error:[/red] {repo_path} is not a git repository")
        raise typer.Exit(1)

    if hook_type == "all":
        types_to_check = list(HOOK_TEMPLATES.keys())
    elif hook_type in HOOK_TEMPLATES:
        types_to_check = [hook_type]
    else:
        console.print(
            f"[red]Error:[/red] Unknown hook type '{hook_type}'. "
            f"Valid types: all, {', '.join(HOOK_TEMPLATES.keys())}"
        )
        raise typer.Exit(1)

    removed = []
    for ht in types_to_check:
        hook_path = hooks_dir / ht
        if hook_path.exists() and is_indexter_hook(hook_path):
            hook_path.unlink()
            removed.append(ht)

            # Restore backup if exists
            backup_path = hook_path.with_suffix(".backup")
            if backup_path.exists():
                backup_path.rename(hook_path)
                console.print(f"[dim]Restored original {ht} hook from backup[/dim]")

    if removed:
        console.print(f"[green]✓[/green] Uninstalled hooks: {', '.join(removed)}")
    else:
        console.print("[dim]No indexter-rlm hooks found to uninstall.[/dim]")


@hooks_app.command("status")
def hook_status(
    repo_path: Annotated[
        Path,
        typer.Argument(
            help="Path to the git repository",
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = Path("."),
) -> None:
    """Show the status of indexter-rlm git hooks.

    Examples:
        indexter-rlm hook status
        indexter-rlm hook status /path/to/repo
    """
    hooks_dir = get_git_hooks_dir(repo_path)
    if hooks_dir is None:
        console.print(f"[red]Error:[/red] {repo_path} is not a git repository")
        raise typer.Exit(1)

    console.print(f"[bold]Hook status for {repo_path}[/bold]\n")

    for hook_type in HOOK_TEMPLATES:
        hook_path = hooks_dir / hook_type
        if not hook_path.exists():
            console.print(f"  {hook_type}: [dim]not installed[/dim]")
        elif is_indexter_hook(hook_path):
            console.print(f"  {hook_type}: [green]installed (indexter-rlm)[/green]")
        else:
            console.print(f"  {hook_type}: [yellow]exists (other)[/yellow]")
