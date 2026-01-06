"""Indexter-RLM: RLM-style context environment for coding agents."""

from importlib.metadata import version

from .models import Repo

__all__ = ["Repo"]

__version__ = version("indexter-rlm")
