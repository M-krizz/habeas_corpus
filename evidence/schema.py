"""
evidence/schema.py — Habeas Corpus
====================================
Pydantic model for a fully-assembled case evidence object.

The Evidence Aggregator produces CaseEvidence objects by combining:
  - FAISS chunk hits (relevant text passages)
  - Neo4j structured metadata (judges, acts, sections, parties)

The LLM Reasoner consumes a list of CaseEvidence objects and
produces the final answer.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class CaseEvidence(BaseModel):
    """
    Complete, structured evidence package for a single legal case.
    This is the unit of information passed to the LLM prompt builder.
    """

    # --- Identity ---
    case_id: str = Field(description="Stable identifier (filename stem or case name).")
    case_name: str = Field(default="", description="Full case name from the KG.")
    decision_date: str = Field(default="")

    # --- Court & Judges ---
    court: str = Field(default="")
    judges: list[str] = Field(default_factory=list)

    # --- Legal framework ---
    acts: list[str] = Field(default_factory=list)
    sections: list[str] = Field(default_factory=list)

    # --- Parties ---
    petitioners: list[str] = Field(default_factory=list)
    respondents: list[str] = Field(default_factory=list)

    # --- Scoring ---
    semantic_score: float = Field(default=0.0)
    graph_score:    float = Field(default=0.0)
    final_score:    float = Field(default=0.0)

    # --- Text evidence ---
    relevant_chunks: list[str] = Field(
        default_factory=list,
        description="Top semantic chunk texts from FAISS (up to 3)."
    )
    top_chunk_text: str = Field(
        default="",
        description="The single best-matching passage (for quick display)."
    )

    # --- Provenance ---
    source: str = Field(
        default="permanent_kg",
        description="'permanent_kg' | 'temporary_pool' | 'web_retrieved'."
    )
    source_url: str = Field(
        default="",
        description="Indian Kanoon URL if retrieved from the web."
    )
