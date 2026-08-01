"""
semantic_retrieval/__init__.py — Habeas Corpus Legal Search Engine
===================================================================
Package marker for the Semantic Retrieval Engine.

Public API (import from here for convenience):

    from semantic_retrieval import build_index, search_cases

All heavy imports are deferred — importing this package is instant.
"""

from semantic_retrieval.pipeline import build_index, search_cases

__all__ = ["build_index", "search_cases"]
