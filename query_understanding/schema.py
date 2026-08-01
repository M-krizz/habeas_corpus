"""
query_understanding/schema.py — Habeas Corpus
==============================================
Pydantic models for the structured legal query output produced by the
Legal Concept Mapper.  These models are the contract between the LLM
output and the rest of the retrieval pipeline.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class LegalQuery(BaseModel):
    """
    Structured representation of a raw user query after legal concept mapping.

    Every field is populated by the LLM-based concept mapper. Downstream
    modules (graph_retriever, hybrid_ranker, prompt_builder) consume this
    object directly — they never see the raw user string.
    """
    original_query: str = Field(description="The raw user query, unchanged.")
    detected_language: str = Field(
        default="en",
        description="ISO language code of user query, e.g. 'en', 'ta' (Tamil), 'hi' (Hindi)."
    )
    query_in_english: str = Field(
        default="",
        description="English translation or legal summary of user query if non-English."
    )

    legal_domain: str = Field(
        description="High-level area of law. E.g. 'Motor Vehicles', 'Criminal', "
                    "'Property', 'Family', 'Consumer', 'Labour', 'Contract'."
    )
    incident_type: str = Field(
        description="Specific nature of the incident. E.g. 'Road Accident', "
                    "'Wrongful Termination', 'Domestic Violence', 'Software Piracy'."
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="Key legal and factual terms extracted for FAISS and graph search."
    )
    suggested_acts: list[str] = Field(
        default_factory=list,
        description="Indian statutes likely applicable. Full names preferred, "
                    "e.g. 'Motor Vehicles Act', 'Indian Penal Code'."
    )
    suggested_sections: list[str] = Field(
        default_factory=list,
        description="Section numbers explicitly mentioned or strongly implied. "
                    "Bare numbers only: '134', '279', '304A'."
    )
    key_entities: list[str] = Field(
        default_factory=list,
        description="Physical entities, roles, or objects central to the incident. "
                    "E.g. ['truck', 'pedestrian', 'insurer', 'employer']."
    )
    expanded_concepts: list[str] = Field(
        default_factory=list,
        description="Deeper legal doctrines behind the incident. "
                    "E.g. ['tort liability', 'contributory negligence', 'mens rea']."
    )

    @property
    def search_text(self) -> str:
        """
        Produce a single enriched search string for FAISS embedding.

        Combines original query (if non-English), English translation, incident type,
        keywords, acts, and expanded concepts so BGE-M3 captures cross-lingual
        semantic meaning.
        """
        query_parts = []
        if self.query_in_english and self.query_in_english != self.original_query:
            query_parts.append(self.query_in_english)
        else:
            query_parts.append(self.original_query)

        parts = (
            query_parts
            + [self.incident_type]
            + self.keywords
            + self.suggested_acts
            + [f"Section {s}" for s in self.suggested_sections]
            + self.expanded_concepts
        )
        return " ".join(parts)


class ReasoningResponse(BaseModel):
    """
    Structured answer produced by the LLM Legal Reasoner.
    Returned to the UI and API caller.
    """
    summary: str = Field(description="Plain-English explanation of the legal situation.")
    applicable_acts: list[str] = Field(default_factory=list)
    applicable_sections: list[str] = Field(default_factory=list)
    precedents: list[dict] = Field(
        default_factory=list,
        description="[{case, court, held, relevance}] — one entry per cited case."
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Retrieval confidence score (0–1)."
    )
    knowledge_source: str = Field(
        default="permanent_kg",
        description="'permanent_kg' | 'temporary_pool' | 'hybrid'."
    )
    disclaimer: str = Field(
        default="This is AI-assisted legal research, not legal advice. "
                "Consult a qualified advocate for legal counsel."
    )
    legal_query: LegalQuery | None = Field(
        default=None,
        description="The structured query that produced this response."
    )
