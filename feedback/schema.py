"""
feedback/schema.py — Habeas Corpus
====================================
Pydantic models for user feedback records.
"""
from __future__ import annotations
from pydantic import BaseModel, Field
from datetime import datetime, timezone


class FeedbackRecord(BaseModel):
    query:      str   = Field(description="The original user query.")
    case_id:    str   = Field(description="The case the user rated.")
    rating:     int   = Field(description="1=helpful, -1=not helpful.")
    timestamp:  str   = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    session_id: str   = Field(default="anonymous")
