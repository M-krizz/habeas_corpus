"""
conversation/conversation.py — Nyaya-Setu Conversation Brain Controller
========================================================================
The master orchestrator for the Conversation Brain.

Every user message flows through this single entry point:

    process_message(conversation_id, user_message)
        → load/create memory
        → extract facts (extractor)
        → merge into memory (merger)
        → check slots (slot_checker)
        → if not ready: return follow-up question
        → if ready: rewrite → Knowledge Brain → Reasoning Brain → answer

This module also manages the in-memory session store so that
multiple conversations can run concurrently (e.g., via the web UI).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from conversation.memory import ConversationMemory
from conversation.extractor import extract_facts
from conversation.merger import merge_facts
from conversation.slot_checker import check_slots, SlotCheckResult
from conversation.rewriter import rewrite_for_retrieval, build_faiss_query


# ---------------------------------------------------------------------------
# In-memory session store (conversation_id → ConversationMemory)
# ---------------------------------------------------------------------------

_sessions: dict[str, ConversationMemory] = {}


def get_or_create_session(conversation_id: str | None = None) -> ConversationMemory:
    """
    Retrieve an existing session or create a new one.
    If conversation_id is None, a new session is created.
    """
    if conversation_id and conversation_id in _sessions:
        return _sessions[conversation_id]

    memory = ConversationMemory()
    if conversation_id:
        memory.conversation_id = conversation_id
    _sessions[memory.conversation_id] = memory
    print(f"[conversation] New session: {memory.conversation_id}")
    return memory


def get_session(conversation_id: str) -> ConversationMemory | None:
    """Retrieve a session by ID. Returns None if not found."""
    return _sessions.get(conversation_id)


def list_sessions() -> list[str]:
    """List all active session IDs."""
    return list(_sessions.keys())


# ---------------------------------------------------------------------------
# Response types
# ---------------------------------------------------------------------------

@dataclass
class ConversationResponse:
    """
    The result of processing one user message.

    Either:
    - type="question" → system is asking a follow-up question
    - type="answer"   → system has produced a full legal analysis
    """
    type: str                                # "question" | "answer"
    conversation_id: str
    message: str                             # The text to display to the user
    memory_snapshot: dict[str, Any]          # Current state of conversation memory
    slot_status: dict[str, Any] | None = None  # Slot-filling progress
    full_response: dict[str, Any] | None = None  # Full ReasoningResponse (when type="answer")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def process_message(
    conversation_id: str | None,
    user_message: str,
) -> ConversationResponse:
    """
    Process a single user message through the Conversation Brain.

    This is the ONLY entry point for multi-turn conversation.

    Parameters
    ----------
    conversation_id : session ID (None to create new session)
    user_message    : the user's raw message text

    Returns
    -------
    ConversationResponse — either a follow-up question or a full answer
    """
    # ── Step 1: Load or create session ──────────────────────────────
    memory = get_or_create_session(conversation_id)
    memory.add_message("user", user_message)

    print(f"\n{'='*60}")
    print(f"[conversation] Session {memory.conversation_id} | "
          f"Turn {memory.turn_count} | Message: '{user_message[:80]}...'")
    print(f"{'='*60}")

    # ── Step 2: Extract facts from the message ──────────────────────
    patch = extract_facts(user_message, memory.to_dict())

    # ── Step 3: Merge into memory ───────────────────────────────────
    merge_facts(memory, patch)

    # ── Step 4: Check slot readiness ────────────────────────────────
    slot_result = check_slots(memory)

    if not slot_result.is_ready:
        # ── Not ready: ask a follow-up question ────────────────────
        question = slot_result.next_question or \
            "Could you provide more details about your situation?"
        memory.add_message("assistant", question)

        return ConversationResponse(
            type="question",
            conversation_id=memory.conversation_id,
            message=question,
            memory_snapshot=memory.filled_slots(),
            slot_status={
                "filled": slot_result.filled_count,
                "required": slot_result.required_count,
                "missing_required": slot_result.missing_required,
                "missing_helpful": slot_result.missing_helpful[:3],
            },
        )

    # ── Step 5: Ready! Trigger Knowledge Brain + Reasoning Brain ───
    print(f"[conversation] Slots filled. Triggering Knowledge + Reasoning Brains...")
    memory.ready_for_retrieval = True

    # Rewrite memory into a LegalQuery for the pipeline
    legal_query = rewrite_for_retrieval(memory)

    # Run the existing pipeline (Knowledge Brain + Reasoning Brain)
    from api.pipeline import run_query_with_legal_query
    response, new_hashes = await run_query_with_legal_query(legal_query, memory)

    # Build the answer message
    answer_text = response.summary
    if hasattr(response, "what_to_do") and response.what_to_do:
        answer_text += f"\n\n**What you should do:**\n{response.what_to_do}"

    memory.add_message("assistant", answer_text)

    # Schedule background indexing if new cases were staged
    if new_hashes:
        print(f"[conversation] Staged {len(new_hashes)} new cases for background indexing.")

    return ConversationResponse(
        type="answer",
        conversation_id=memory.conversation_id,
        message=answer_text,
        memory_snapshot=memory.filled_slots(),
        slot_status={
            "filled": slot_result.filled_count,
            "required": slot_result.required_count,
            "missing_required": [],
            "missing_helpful": slot_result.missing_helpful[:3],
        },
        full_response=response.model_dump(),
    )
