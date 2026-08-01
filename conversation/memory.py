"""
conversation/memory.py — Nyaya-Setu Conversation Brain
=======================================================
ConversationMemory is the structured JSON state for one
conversation session.  It stores *legal facts*, not chat
history.  Every downstream module (Knowledge Brain,
Reasoning Brain) consumes this object — they never see
raw user messages.

Example memory after 3 turns:
    {
        "conversation_id": "abc-123",
        "incident": "Road Accident",
        "legal_domain": "Motor Vehicles",
        "vehicle": "Bike",
        "state": "Tamil Nadu",
        "city": "Chennai",
        "sections": ["134"],
        "injury": "Minor",
        "current_goal": "Find Liability",
        "ready_for_retrieval": true
    }
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any


@dataclass
class ConversationMemory:
    """Structured legal state for a single conversation session."""

    conversation_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    # ── Core legal facts (populated incrementally via slot-filling) ──
    incident: str | None = None              # "Road Accident"
    legal_domain: str | None = None          # "Motor Vehicles"
    vehicle: str | None = None               # "Bike", "Car", "Truck"
    state: str | None = None                 # "Tamil Nadu"
    city: str | None = None                  # "Chennai"
    court: str | None = None                 # "Madras High Court"
    sections: list[str] = field(default_factory=list)   # ["134", "166"]
    acts: list[str] = field(default_factory=list)       # ["Motor Vehicles Act"]
    injury: str | None = None                # "Minor", "Serious", "Fatal"
    fir_filed: bool | None = None            # True / False / None (unknown)
    current_goal: str | None = None          # "Find Liability"

    # ── Property-specific slots ──
    property_type: str | None = None         # "Land", "Flat", "House"
    dispute_type: str | None = None          # "Eviction", "Encroachment"
    agreement_exists: bool | None = None

    # ── Cheque bounce slots ──
    cheque_amount: str | None = None
    bank: str | None = None
    notice_sent: bool | None = None

    # ── Overflow for domain-specific fields ──
    custom_facts: dict[str, Any] = field(default_factory=dict)

    # ── Conversation history (for context, not for retrieval) ──
    messages: list[dict[str, str]] = field(default_factory=list)

    # ── State flags ──
    ready_for_retrieval: bool = False
    turn_count: int = 0

    # ── Timestamps ──
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # ------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------

    def add_message(self, role: str, content: str) -> None:
        """Append a message to the conversation history."""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        if role == "user":
            self.turn_count += 1
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def merge(self, patch: dict[str, Any]) -> None:
        """
        Apply a JSON patch of extracted facts into memory.

        Rules:
        - New keys are added.
        - Existing scalar values are overwritten (user corrected themselves).
        - List fields (sections, acts) are unioned, not replaced.
        - 'messages' is never overwritten via merge.
        - Unknown keys go into custom_facts.
        """
        list_fields = {"sections", "acts"}
        reserved = {"messages", "conversation_id", "created_at",
                     "turn_count", "ready_for_retrieval"}

        for key, value in patch.items():
            if key in reserved or value is None:
                continue

            if key in list_fields:
                existing = getattr(self, key, [])
                if isinstance(value, list):
                    merged = list(dict.fromkeys(existing + value))
                else:
                    merged = list(dict.fromkeys(existing + [str(value)]))
                setattr(self, key, merged)

            elif hasattr(self, key):
                setattr(self, key, value)

            else:
                # Unknown fields → custom_facts overflow
                self.custom_facts[key] = value

        self.updated_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Serialize memory to a plain dict (JSON-safe)."""
        return asdict(self)

    def filled_slots(self) -> dict[str, Any]:
        """Return only the non-None legal fact fields."""
        skip = {"messages", "custom_facts", "conversation_id",
                "created_at", "updated_at", "turn_count",
                "ready_for_retrieval"}
        result = {}
        for key, value in asdict(self).items():
            if key in skip:
                continue
            if value is not None and value != [] and value != {}:
                result[key] = value
        return result

    def summary_text(self) -> str:
        """
        Produce a human-readable summary of filled slots.
        Used by the Reasoning Brain for context.
        """
        filled = self.filled_slots()
        if not filled:
            return "No facts collected yet."
        parts = []
        for k, v in filled.items():
            label = k.replace("_", " ").title()
            if isinstance(v, list):
                parts.append(f"{label}: {', '.join(str(x) for x in v)}")
            elif isinstance(v, bool):
                parts.append(f"{label}: {'Yes' if v else 'No'}")
            else:
                parts.append(f"{label}: {v}")
        return "; ".join(parts)
