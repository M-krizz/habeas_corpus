"""
conversation/merger.py — Nyaya-Setu Conversation Brain
=======================================================
Pure function: takes current ConversationMemory + a JSON patch
from the extractor and returns an updated memory.

This is intentionally kept as a thin wrapper around
ConversationMemory.merge() — separated for testability and
to keep the controller (conversation.py) clean.
"""

from __future__ import annotations

from conversation.memory import ConversationMemory


def merge_facts(memory: ConversationMemory, patch: dict) -> ConversationMemory:
    """
    Merge extracted facts into the conversation memory.

    Parameters
    ----------
    memory : current ConversationMemory instance
    patch  : dict of new facts from the extractor

    Returns
    -------
    The same ConversationMemory instance, mutated in-place.
    (Returned for convenience in chaining.)
    """
    if not patch:
        return memory

    memory.merge(patch)
    print(f"[merger] Memory updated. Filled slots: {list(memory.filled_slots().keys())}")
    return memory
