"""
query_understanding/legal_concept_mapper.py — Habeas Corpus
============================================================
The Legal Concept Mapper is the first module in the pipeline.

Its ONLY job is to translate a plain-English user query into a
structured LegalQuery object.  It NEVER answers the legal question —
that is the LLM Reasoner's job downstream.

Architecture decision:
    LLM-based (Option 2 from the plan) — Gemini Flash via google-generativeai.
    Falls back to a rule-based extractor if the API is unavailable or
    the key is not configured.

Usage:
    from query_understanding.legal_concept_mapper import map_legal_concepts

    lq = map_legal_concepts("My neighbour's dog bit my child.")
    # → LegalQuery(legal_domain='Criminal/Tort', incident_type='Animal Bite', ...)

CLI smoke-test:
    python -m query_understanding.legal_concept_mapper "Someone forged my signature"
"""

from __future__ import annotations

import json
import os
import re
import sys
from functools import lru_cache

from dotenv import load_dotenv

from query_understanding.schema import LegalQuery

load_dotenv()


# ---------------------------------------------------------------------------
# Gemini client — lazy singleton
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_gemini():
    """Load the Gemini Flash client once. Returns None if key is missing."""
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key or api_key == "YOUR_GEMINI_KEY_HERE":
        print("[legal_concept_mapper] WARNING — GEMINI_API_KEY not set. "
              "Using rule-based fallback mapper.")
        return None
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        print("[legal_concept_mapper] Gemini 2.5 Flash client loaded.")
        return client
    except Exception as exc:
        print(f"[legal_concept_mapper] WARNING — Could not load Gemini: {exc}. "
              "Falling back to rule-based mapper.")
        return None


# ---------------------------------------------------------------------------
# LLM prompt
def detect_language(text: str) -> str:
    """
    Detect language of input query.
    Returns: 'ta' (Tamil script), 'ta_roman' (Tanglish / Romanized Tamil),
             'hi' (Hindi script), 'hi_roman' (Hinglish), or 'en' (English default).
    """
    if re.search(r"[\u0B80-\u0BFF]", text):
        return "ta"
    if re.search(r"[\u0900-\u097F]", text):
        return "hi"

    # Tanglish / Romanized Tamil markers
    tanglish_words = {
        "vandi", "wandi", "vaandi", "vandiye", "oruthan", "modhitan", "kidaikuma",
        "kedaikuma", "epdi", "iruka", "pudichu", "panna", "pannitan", "pannitanga",
        "pannalam", "casela", "policela", "enaku", "enakku", "nalla", "solunga",
        "solungha", "engalukku", "aachu", "aayiduchu", "varuma", "kuduka", "varum"
    }
    tokens = set(re.findall(r"[a-z]+", text.lower()))
    if tokens & tanglish_words:
        return "ta_roman"

    return "en"


# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a multilingual legal concept extraction engine for the Habeas Corpus Indian Legal Research System.

Your task is to analyse a user's legal query (which may be in English, native Tamil script, native Hindi script, OR Romanized code-mixed languages like Tanglish e.g. "en vandi la oruthan hit pannitan") and return ONLY a JSON object with the following structure.

DO NOT answer the legal question.
DO NOT give legal advice.
ONLY extract and map concepts.

Required JSON structure:
{
  "detected_language": "<string: 'en' | 'ta' | 'ta_roman' (Tanglish) | 'hi' | 'hi_roman'>",
  "query_in_english": "<string: formal English legal translation & research statement of user query>",
  "query_in_tamil_script": "<string: native Tamil script transliteration/translation if query is in Tanglish/Tamil>",
  "legal_domain": "<string: primary area of Indian law>",
  "incident_type": "<string: specific nature of incident in English>",
  "keywords": ["<term1>", "<term2>", ...],
  "suggested_acts": ["<Full Act Name>", ...],
  "suggested_sections": ["<number only>", ...],
  "key_entities": ["<entity1>", ...],
  "expanded_concepts": ["<doctrine1>", ...]
}

Guidelines:
- detected_language: 'ta_roman' for Tanglish (Tamil in English alphabet e.g. "en vandi la oruthan hit pannitan"), 'ta' for Tamil script, 'en' for English.
- query_in_english: Convert Tanglish or any non-English query into formal English legal research statement. E.g. "en vandi la oruthan hit pannitan" -> "Motor vehicle collision where another driver struck petitioner's vehicle causing property damage and compensation claim under Motor Vehicles Act."
- query_in_tamil_script: If query is Tanglish or Tamil, provide the exact sentence in native Tamil script: "என் வண்டியை ஒருவர் மோதிவிட்டார். எனக்கு இழப்பீடு கிடைக்குமா?"
- legal_domain: Choose from: Motor Vehicles, Criminal, Property, Family, Consumer,
  Labour, Contract, Constitutional, Intellectual Property, Environmental, Tax, Other
- suggested_acts: Use full Indian statute names in English (e.g. "Motor Vehicles Act",
  "Indian Penal Code", "Transfer of Property Act", "Consumer Protection Act")
- suggested_sections: Bare numbers only ("134", "166", "304A"). Include only if
  explicitly mentioned or unambiguously implied.
- expanded_concepts: Legal doctrines in English (e.g. "tort liability", "res ipsa loquitur",
  "contributory negligence", "mens rea", "promissory estoppel")
- keywords: Mix of legal and factual terms in English, native Tamil script, and Tanglish useful for document retrieval
- Return ONLY valid JSON. No markdown, no explanation, no code fences.
"""


def _llm_map(query: str) -> LegalQuery | None:
    """
    Send query to Gemini Flash and parse the structured response.
    Returns None on any failure so the caller can use the rule-based fallback.
    """
    client = _get_gemini()
    if client is None:
        return None

    prompt = f"{_SYSTEM_PROMPT}\n\nUser query:\n{query.strip()}"
    try:
        model_name = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )
        raw = response.text.strip()

        # Strip accidental markdown fences
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)

        data = json.loads(raw)
        if "detected_language" not in data or not data["detected_language"]:
            data["detected_language"] = detect_language(query)
        if "query_in_english" not in data or not data["query_in_english"]:
            data["query_in_english"] = query

        return LegalQuery(original_query=query, **data)

    except Exception as exc:
        print(f"[legal_concept_mapper] LLM mapping failed: {exc}. "
              "Falling back to rule-based mapper.")
        return None


# ---------------------------------------------------------------------------
# Rule-based fallback
# ---------------------------------------------------------------------------

# Keyword → (legal_domain, incident_type, acts, sections, concepts)
_RULES: list[tuple[list[str], dict]] = [
    (
        ["accident", "vehicle", "bike", "truck", "car", "collision", "hit",
         "motor", "road", "driving", "motorcycle", "pedestrian",
         "விபத்து", "வண்டி", "மோதி", "இழப்பீடு", "சேதம்", "காரை",
         "दुर्घटना", "वाहन", "मुआवजा"],
        {
            "legal_domain": "Motor Vehicles",
            "incident_type": "Road Accident",
            "suggested_acts": ["Motor Vehicles Act"],
            "suggested_sections": ["134", "166", "279", "304A"],
            "expanded_concepts": ["tort liability", "negligence", "compensation",
                                   "contributory negligence", "motor accident claim"],
        },
    ),
    (
        ["dog", "animal", "bite", "attack", "cruelty", "stray", "நாய்", "கடித்தது", "விலங்கு"],
        {
            "legal_domain": "Criminal/Tort",
            "incident_type": "Animal Attack",
            "suggested_acts": ["Indian Penal Code",
                                "Prevention of Cruelty to Animals Act"],
            "suggested_sections": ["289", "304A", "428"],
            "expanded_concepts": ["strict liability", "nuisance", "negligence",
                                   "animal owner liability"],
        },
    ),
    (
        ["landlord", "tenant", "rent", "eviction", "evict", "lease",
         "property", "house", "flat", "apartment", "வாடகை", "வீடு", "நிலம்"],
        {
            "legal_domain": "Property",
            "incident_type": "Tenancy / Eviction Dispute",
            "suggested_acts": ["Transfer of Property Act",
                                "State Rent Control Act"],
            "suggested_sections": ["106", "111"],
            "expanded_concepts": ["wrongful eviction", "tenancy rights",
                                   "notice to quit", "mesne profits"],
        },
    ),
    (
        ["forge", "forged", "forgery", "signature", "fraud", "cheating",
         "document", "fake", "impersonate", "போலி", "கையெழுத்து", "ஏமாற்று"],
        {
            "legal_domain": "Criminal",
            "incident_type": "Forgery / Document Fraud",
            "suggested_acts": ["Indian Penal Code"],
            "suggested_sections": ["463", "465", "468", "471", "420"],
            "expanded_concepts": ["forgery", "fraud", "cheating",
                                   "document tampering", "mens rea"],
        },
    ),
    (
        ["copyright", "software", "piracy", "intellectual property",
         "trademark", "patent", "infringement", "plagiarism"],
        {
            "legal_domain": "Intellectual Property",
            "incident_type": "IP Infringement",
            "suggested_acts": ["Copyright Act", "Trade Marks Act",
                                "Patents Act", "Information Technology Act"],
            "suggested_sections": ["51", "63"],
            "expanded_concepts": ["copyright infringement", "software piracy",
                                   "fair use", "moral rights"],
        },
    ),
    (
        ["employment", "fired", "sacked", "termination", "dismiss",
         "salary", "wage", "labour", "worker", "retrenchment", "வேலை", "சம்பளம்"],
        {
            "legal_domain": "Labour",
            "incident_type": "Wrongful Termination / Labour Dispute",
            "suggested_acts": ["Industrial Disputes Act",
                                "Payment of Wages Act", "Factories Act"],
            "suggested_sections": ["25F", "25N", "33"],
            "expanded_concepts": ["wrongful termination", "retrenchment",
                                   "reinstatement", "back wages"],
        },
    ),
    (
        ["consumer", "product", "defect", "defective", "warranty",
         "refund", "service", "cheated", "company", "பொருள்", "நுகர்வோர்"],
        {
            "legal_domain": "Consumer",
            "incident_type": "Consumer Dispute",
            "suggested_acts": ["Consumer Protection Act"],
            "suggested_sections": ["2", "35", "58"],
            "expanded_concepts": ["deficiency in service", "unfair trade practice",
                                   "product liability", "consumer rights"],
        },
    ),
    (
        ["murder", "assault", "rape", "robbery", "theft", "kidnap",
         "abduction", "dowry", "harassment", "criminal", "கொலை", "திருட்டு"],
        {
            "legal_domain": "Criminal",
            "incident_type": "Criminal Offence",
            "suggested_acts": ["Indian Penal Code", "Code of Criminal Procedure"],
            "suggested_sections": ["302", "307", "376", "392", "498A"],
            "expanded_concepts": ["cognizable offence", "mens rea", "actus reus",
                                   "burden of proof", "FIR"],
        },
    ),
]


def _rule_based_map(query: str) -> LegalQuery:
    """
    Keyword-matching fallback that fires when the LLM is unavailable.
    Scans rules in order and returns the first match; defaults to Generic
    if nothing matches.
    """
    q_lower = query.lower()
    lang = detect_language(query)
    for keywords, fields in _RULES:
        if any(kw in q_lower for kw in keywords):
            mentioned_sections = re.findall(r"\bsection\s+(\d+[A-Za-z]?)\b",
                                             q_lower, re.IGNORECASE)
            all_sections = list(dict.fromkeys(
                fields["suggested_sections"] + mentioned_sections
            ))
            return LegalQuery(
                original_query=query,
                detected_language=lang,
                query_in_english=query if lang == "en" else fields["incident_type"],
                keywords=_extract_keywords(query),
                key_entities=_extract_keywords(query),
                **{**fields, "suggested_sections": all_sections},
            )

    # Generic fallback
    return LegalQuery(
        original_query=query,
        detected_language=lang,
        query_in_english=query,
        legal_domain="General",
        incident_type="Legal Dispute",
        keywords=_extract_keywords(query),
        suggested_acts=[],
        suggested_sections=re.findall(r"\bsection\s+(\d+[A-Za-z]?)\b",
                                       query, re.IGNORECASE),
        key_entities=_extract_keywords(query),
        expanded_concepts=["legal rights", "judicial remedy"],
    )


def _extract_keywords(text: str) -> list[str]:
    """
    Extract meaningful words from the user query (stop-word filter).
    Simple but effective for short queries.
    """
    _STOP = {
        "i", "me", "my", "we", "our", "you", "your", "he", "she", "it",
        "they", "their", "the", "a", "an", "is", "are", "was", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "will", "would", "could", "should", "may", "might", "shall", "can",
        "to", "of", "in", "on", "at", "by", "for", "with", "from", "into",
        "and", "or", "but", "not", "no", "so", "as", "if", "then", "than",
        "this", "that", "these", "those", "what", "which", "who", "how",
        "when", "where", "why", "got", "get", "says", "said", "say",
        "someone", "someone's", "other", "another", "him", "her", "his",
    }
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return list(dict.fromkeys(w for w in words if w not in _STOP and len(w) > 2))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def map_legal_concepts(query: str) -> LegalQuery:
    """
    Map a plain-English user query to a structured LegalQuery object.

    Tries the Gemini LLM first; falls back to the rule-based extractor
    if the API is unavailable.

    Parameters
    ----------
    query : raw user query string

    Returns
    -------
    LegalQuery  — structured legal concept object ready for retrieval
    """
    if not query or not query.strip():
        raise ValueError("[legal_concept_mapper] Empty query.")

    # Try LLM first
    result = _llm_map(query)
    if result is not None:
        print(f"[legal_concept_mapper] LLM mapped -> domain={result.legal_domain}, "
          f"incident={result.incident_type}")
        return result

    # Rule-based fallback
    result = _rule_based_map(query)
    print(f"[legal_concept_mapper] Rule-based -> domain={result.legal_domain}, "
          f"incident={result.incident_type}")
    return result


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else \
        "I got into a road accident. The other party says Section 134 protects him."

    print(f"\n[legal_concept_mapper] Query: '{query}'")
    print("=" * 60)
    lq = map_legal_concepts(query)
    print(f"  Domain          : {lq.legal_domain}")
    print(f"  Incident        : {lq.incident_type}")
    print(f"  Keywords        : {lq.keywords}")
    print(f"  Acts            : {lq.suggested_acts}")
    print(f"  Sections        : {lq.suggested_sections}")
    print(f"  Entities        : {lq.key_entities}")
    print(f"  Concepts        : {lq.expanded_concepts}")
    print(f"\n  Search text     : {lq.search_text}")
