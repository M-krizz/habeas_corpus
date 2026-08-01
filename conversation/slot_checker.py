"""
conversation/slot_checker.py — Nyaya-Setu Conversation Brain
=============================================================
Domain-specific slot definitions and readiness logic.

Every legal domain has a set of REQUIRED and HELPFUL slots.
The slot checker determines:
    1. Whether we have enough information to trigger retrieval.
    2. Which slot is missing next.
    3. A natural follow-up question for the missing slot.

The system asks at most one question per turn (never a barrage).
"""

from __future__ import annotations

from dataclasses import dataclass

from conversation.memory import ConversationMemory


# ---------------------------------------------------------------------------
# Slot registry — one entry per legal domain
# ---------------------------------------------------------------------------

@dataclass
class SlotDefinition:
    """Defines what information is needed for a legal domain."""
    required: list[str]       # Must have these before retrieval
    helpful: list[str]        # Nice to have, improve retrieval accuracy
    min_required: int         # Minimum required slots that must be filled


# The slot registry maps incident types → what information we need.
SLOT_REGISTRY: dict[str, SlotDefinition] = {

    "Road Accident": SlotDefinition(
        required=["vehicle", "injury"],
        helpful=["state", "fir_filed", "sections", "city"],
        min_required=2,
    ),

    "Property Dispute": SlotDefinition(
        required=["property_type", "dispute_type"],
        helpful=["state", "agreement_exists"],
        min_required=2,
    ),

    "Tenancy / Eviction Dispute": SlotDefinition(
        required=["property_type", "dispute_type"],
        helpful=["state", "agreement_exists"],
        min_required=2,
    ),

    "Cheque Bounce": SlotDefinition(
        required=["cheque_amount", "bank"],
        helpful=["notice_sent"],
        min_required=1,
    ),

    "Domestic Violence": SlotDefinition(
        required=["injury"],
        helpful=["fir_filed", "state"],
        min_required=1,
    ),

    "Consumer Complaint": SlotDefinition(
        required=["incident"],
        helpful=["state", "cheque_amount"],
        min_required=1,
    ),

    "Animal Attack": SlotDefinition(
        required=["injury"],
        helpful=["state", "fir_filed"],
        min_required=1,
    ),

    "Labour Dispute": SlotDefinition(
        required=["incident"],
        helpful=["state"],
        min_required=1,
    ),

    "Fraud / Forgery": SlotDefinition(
        required=["incident"],
        helpful=["fir_filed", "state"],
        min_required=1,
    ),

    "Criminal Offence": SlotDefinition(
        required=["incident"],
        helpful=["fir_filed", "state", "injury"],
        min_required=1,
    ),

    "Family Dispute": SlotDefinition(
        required=["incident"],
        helpful=["state"],
        min_required=1,
    ),

    "Homicide": SlotDefinition(
        required=["incident"],
        helpful=["fir_filed", "state"],
        min_required=1,
    ),

    "Theft / Robbery": SlotDefinition(
        required=["incident"],
        helpful=["fir_filed", "state"],
        min_required=1,
    ),

    "IP Infringement": SlotDefinition(
        required=["incident"],
        helpful=["state"],
        min_required=1,
    ),
}

# Default for unknown domains
_DEFAULT_SLOTS = SlotDefinition(
    required=["incident"],
    helpful=["state", "sections"],
    min_required=1,
)


# ---------------------------------------------------------------------------
# Natural-language follow-up questions for each slot
# ---------------------------------------------------------------------------

_SLOT_QUESTIONS: dict[str, str] = {
    "incident":          "Could you describe what happened? For example, was it an accident, a property issue, a fraud case, or something else?",
    "vehicle":           "What type of vehicle was involved — bike, car, truck, auto, or something else?",
    "injury":            "Was anyone injured? If so, was it minor, serious, or fatal?",
    "state":             "Which Indian state did this happen in?",
    "city":              "Which city did this occur in?",
    "fir_filed":         "Have you filed an FIR (First Information Report) with the police?",
    "sections":          "Are there any specific legal sections mentioned (e.g., Section 134, Section 498A)?",
    "property_type":     "What type of property is involved — land, flat, house, or commercial?",
    "dispute_type":      "What is the nature of the dispute — eviction, encroachment, ownership, or something else?",
    "agreement_exists":  "Is there a written agreement or contract related to this matter?",
    "cheque_amount":     "What was the amount on the cheque?",
    "bank":              "Which bank was the cheque drawn on?",
    "notice_sent":       "Has a legal notice been sent to the other party?",
    "current_goal":      "What would you like help with — understanding your rights, getting compensation, defending yourself, or something else?",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class SlotCheckResult:
    """Result of checking whether enough slots are filled."""
    is_ready: bool
    missing_required: list[str]
    missing_helpful: list[str]
    next_question: str | None
    filled_count: int
    required_count: int


def check_slots(memory: ConversationMemory) -> SlotCheckResult:
    """
    Check if the conversation memory has enough information
    to trigger the Knowledge Brain.

    Parameters
    ----------
    memory : current ConversationMemory

    Returns
    -------
    SlotCheckResult with readiness flag, missing slots, and next question.
    """
    # If we don't even know the incident type, ask about that first
    if not memory.incident:
        return SlotCheckResult(
            is_ready=False,
            missing_required=["incident"],
            missing_helpful=[],
            next_question=_SLOT_QUESTIONS["incident"],
            filled_count=0,
            required_count=1,
        )

    # Look up domain-specific slot requirements
    slot_def = SLOT_REGISTRY.get(memory.incident, _DEFAULT_SLOTS)
    filled = memory.filled_slots()

    # Count how many required slots are filled
    filled_required = [s for s in slot_def.required if s in filled]
    missing_required = [s for s in slot_def.required if s not in filled]
    missing_helpful = [s for s in slot_def.helpful if s not in filled]

    is_ready = len(filled_required) >= slot_def.min_required

    # After 4 turns, stop asking and proceed with what we have
    # (don't frustrate the user with too many questions)
    if memory.turn_count >= 4 and memory.incident:
        is_ready = True

    # Find the next question to ask
    next_question = None
    if not is_ready:
        # Prioritize required slots, then helpful
        for slot in missing_required + missing_helpful[:1]:
            if slot in _SLOT_QUESTIONS:
                next_question = _SLOT_QUESTIONS[slot]
                break

    result = SlotCheckResult(
        is_ready=is_ready,
        missing_required=missing_required,
        missing_helpful=missing_helpful,
        next_question=next_question,
        filled_count=len(filled_required),
        required_count=slot_def.min_required,
    )

    status = "READY" if is_ready else f"WAITING ({len(filled_required)}/{slot_def.min_required} required)"
    print(f"[slot_checker] {status} | missing_required={missing_required} "
          f"| missing_helpful={missing_helpful[:3]}")

    return result
