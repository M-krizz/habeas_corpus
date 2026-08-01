"""query_understanding/__init__.py — Habeas Corpus"""


def map_legal_concepts(query: str):
    """Lazy import wrapper to avoid circular imports at startup."""
    from query_understanding.legal_concept_mapper import map_legal_concepts as _fn
    return _fn(query)


__all__ = ["map_legal_concepts"]
