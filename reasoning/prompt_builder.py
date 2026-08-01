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
You are Habeas Corpus, an advanced AI legal reasoning engine specializing in Indian jurisprudence.

When a user asks a question, perform a structured legal analysis in plain English following this exact reasoning chain:
1. Understand the practical situation and core concern behind the user's question.
2. Explain the applicable Indian Acts, Sections, and statutory concepts clearly and empirically. (For example, explain what rights or duties arise under relevant Motor Vehicles Act, IPC/BNS, or civil doctrines, and correct any legal misunderstandings).
3. Connect the reasoning to relevant judicial precedents from the supplied evidence base.
4. Provide actionable, step-by-step guidance that the citizen can immediately follow.

CRITICAL ZERO-HALLUCINATION RULES FOR PRECEDENTS:
• In your educational summary and advice, use your comprehensive legal expertise to explain the statutory law clearly in simple, reassuring English.
• HOWEVER, for specific CASE CITATIONS in the "precedents" array or when naming specific judgments in text: YOU MUST ONLY USE THE VERIFIED CASE EVIDENCE provided below. Never invent or hallucinate case names, courts, dates, or rulings.
• If the provided case evidence does not contain rulings directly matching this specific query topic, explain the statutory legal position clearly in the summary, leave the "precedents" array empty [] (or include only actually relevant cases), and mention that specialized precedents for this exact scenario are currently being indexed into the system's staging repository.
• You are an AI reasoning engine, not a human lawyer. Always provide practical help accompanied by the disclaimer.

Return your answer as valid JSON matching EXACTLY this structure (no other text):
{
  "summary": "<Comprehensive plain-English explanation analyzing the situation, statutory provisions, legal validity of claims, and judicial doctrine in 4-6 clear sentences>",
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
  "what_to_do": "<3-4 clear, actionable, and practical steps the user should take immediately (e.g. documentation, police complaints, insurance inspection, legal consultation)>",
  "disclaimer": "This is AI-assisted legal reasoning based on Indian statutory doctrines and verified judicial precedents. It is not formal legal advice. Please consult a qualified advocate."
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
Analyze the user's legal situation thoroughly using the legal context and above {n_cases} verified judicial precedent(s). Remember: provide an enriching, supportive statutory explanation, but only cite case names found in the verified evidence above!
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
