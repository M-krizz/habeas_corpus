"""
reasoning/prompt_builder.py — Habeas Corpus
=============================================
Constructs the structured prompt that is sent to the Gemini LLM.

The prompt is engineered so that:
1. The LLM ONLY uses the provided judicial evidence (no hallucination)
2. The output is always valid JSON (parseable by the response module)
3. The answer is grounded in real case precedents, not general knowledge
4. The response format is consistent for every query

This prompt design is the key reason hallucination is minimised —
the LLM is given exhaustive context and told to say "insufficient evidence"
rather than invent law.
"""

from __future__ import annotations

from evidence.schema import CaseEvidence
from query_understanding.schema import LegalQuery


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_HEADER = """\
You are Habeas Corpus, an AI legal research assistant specialising in Indian law.

CRITICAL RULES:
1. Answer ONLY using the judicial evidence provided below. Do not use your general knowledge.
2. If the evidence is insufficient to answer, say so in the summary field.
3. Never fabricate case names, section numbers, or legal outcomes.
4. You are a research tool, not a lawyer. Always include the disclaimer.
5. Explain in plain English that a non-lawyer can understand.
6. Always cite the specific case name when making a legal point.

Return your answer as valid JSON matching EXACTLY this structure (no other text):
{
  "summary": "<3-5 sentence plain-English explanation>",
  "applicable_acts": ["<Act name>", ...],
  "applicable_sections": ["<Section number>", ...],
  "precedents": [
    {
      "case": "<case name>",
      "court": "<court name>",
      "date": "<decision date>",
      "held": "<what the court decided, 1-2 sentences>",
      "relevance": "<why this case is relevant to the user's query, 1 sentence>"
    }
  ],
  "what_to_do": "<3-4 actionable steps the user can take>",
  "disclaimer": "This is AI-assisted legal research based on verified judicial precedents. It is not legal advice. Please consult a qualified advocate."
}
"""

_QUERY_SECTION = """\

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USER QUESTION:
{original_query}

LEGAL CONTEXT IDENTIFIED:
• Domain       : {legal_domain}
• Incident     : {incident_type}
• Relevant Acts: {acts}
• Sections     : {sections}
• Key Concepts : {concepts}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

_CASE_TEMPLATE = """\
CASE {rank}: {case_name}
  Court         : {court}
  Date          : {date}
  Acts involved : {acts}
  Sections      : {sections}
  Judges        : {judges}
  Petitioner(s) : {petitioners}
  Respondent(s) : {respondents}
  Relevance score: {score:.2f}

  Key passages from the judgment:
{passages}
"""

_FOOTER = """\
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Answer the user's question using only the above {n_cases} case(s) as evidence.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_prompt(
    legal_query: LegalQuery,
    evidence:    list[CaseEvidence],
) -> str:
    """
    Construct the full LLM prompt from structured legal query + case evidence.

    Parameters
    ----------
    legal_query : LegalQuery from the concept mapper
    evidence    : list[CaseEvidence] from the evidence aggregator

    Returns
    -------
    str — the complete prompt to send to the LLM
    """
    parts: list[str] = [_SYSTEM_HEADER]

    # Query context section
    parts.append(_QUERY_SECTION.format(
        original_query = legal_query.original_query,
        legal_domain   = legal_query.legal_domain,
        incident_type  = legal_query.incident_type,
        acts           = ", ".join(legal_query.suggested_acts) or "Not identified",
        sections       = ", ".join(legal_query.suggested_sections) or "Not identified",
        concepts       = ", ".join(legal_query.expanded_concepts[:5]) or "Not identified",
    ))

    if not evidence:
        parts.append("\nNO JUDICIAL EVIDENCE FOUND IN THE KNOWLEDGE BASE.\n"
                     "State that no relevant precedents were found and "
                     "suggest the user consult a legal professional.\n")
    else:
        parts.append(f"\nJUDICIAL EVIDENCE ({len(evidence)} case(s)):\n")
        for i, ev in enumerate(evidence, 1):
            passages = _format_passages(ev.relevant_chunks)
            block = _CASE_TEMPLATE.format(
                rank        = i,
                case_name   = ev.case_name or ev.case_id,
                court       = ev.court or "Supreme Court of India",
                date        = ev.decision_date or "Unknown",
                acts        = ", ".join(ev.acts) or "Not specified",
                sections    = ", ".join(str(s) for s in ev.sections) or "Not specified",
                judges      = ", ".join(ev.judges[:3]) or "Not specified",
                petitioners = ", ".join(ev.petitioners[:2]) or "Not specified",
                respondents = ", ".join(ev.respondents[:2]) or "Not specified",
                score       = ev.final_score,
                passages    = passages,
            )
            parts.append(block)

    parts.append(_FOOTER.format(n_cases=len(evidence)))

    return "\n".join(parts)


def _format_passages(chunks: list[str], max_chars: int = 400) -> str:
    """
    Format chunk texts as indented bullet points, truncated to max_chars each.
    """
    if not chunks:
        return "    (No text passages available)"
    lines = []
    for chunk in chunks[:3]:
        excerpt = chunk[:max_chars].strip()
        if len(chunk) > max_chars:
            excerpt += "..."
        lines.append(f"    • {excerpt}")
    return "\n".join(lines)
