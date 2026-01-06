"""Shared fixtures for CLI tests."""

import pytest
from typer.testing import CliRunner

from indexter_rlm.config import StoreMode, settings


def pytest_configure(config):
    """Configure pytest to use in-memory store mode for faster tests."""
    settings.store.mode = StoreMode.memory


@pytest.fixture
def cli_runner():
    """Create a CliRunner for testing."""
    return CliRunner()
