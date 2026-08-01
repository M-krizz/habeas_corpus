"""
entity_extractor.py — Habeas Corpus Legal Search Engine
=========================================================
Module: extractor/entity_extractor.py

Responsibility:
    Extract raw structured metadata from Supreme Court judgment text and
    return it as a graph-ready dict (nodes + relationships).

    Nodes extracted:
        - case      : name (str), decision_date (ISO-8601 str)
        - court     : name (str)
        - judges    : list of {name}
        - acts      : list of {name, year}          — structured
        - sections  : list of {type, number}        — normalised
        - parties   : list of {name, role}          — role drives relationship type
        - citations : list of raw citation strings  — Phase-2 resolution

    Relationships returned (as strings; loader uses these to drive MERGE):
        ("Case", "HEARD_IN",         "Court")
        ("Case", "DECIDED_BY",       "Judge")
        ("Case", "UNDER_ACT",        "Act")
        ("Case", "INVOLVES_SECTION", "Section")
        ("Case", "HAS_PETITIONER",   "Party")   ← role on dict, not on node
        ("Case", "HAS_RESPONDENT",   "Party")

Strategy:
    - Regex is used for all structured / patterned fields (dates, citations,
      case numbers, section references, judge blocks).
    - spaCy (en_core_web_sm) is used only as a fallback for court name when
      the regex heuristic cannot find a known court keyword.
    - All extraction targets the first page / headnote section of the document
      (≈ first 3 000 characters) where SCR format guarantees consistent layout.

Dependencies:
    - spacy  (uv add spacy && uv run python -m spacy download en_core_web_sm)

Usage:
    from extractor.entity_extractor import extract_legal_graph

    with open("output/judgment.txt") as f:
        raw_text = f.read()

    graph = extract_legal_graph(raw_text)

    Or run directly:
        python extractor/entity_extractor.py output/2022_1_1_17_EN.txt
"""

import re
import sys
from datetime import datetime
from functools import lru_cache

import spacy


# ---------------------------------------------------------------------------
# spaCy — lazy-load once and reuse
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _nlp():
    """Load the small English model once; disable unused pipes for speed."""
    return spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])


# ---------------------------------------------------------------------------
# Constants / known values
# ---------------------------------------------------------------------------

KNOWN_COURTS = [
    "Supreme Court of India",
    "High Court",
    "District Court",
    "Sessions Court",
    "National Consumer Disputes Redressal Commission",
    "National Company Law Tribunal",
    "National Company Law Appellate Tribunal",
    "Central Administrative Tribunal",
    "Armed Forces Tribunal",
    "Income Tax Appellate Tribunal",
]

# These appear as column-margin single-letter lines in SCR PDFs — noise only
_SINGLE_LETTER_LINE = re.compile(r"^\s*[A-H]\s*$")

# Regex patterns
_DATE_PATTERN = re.compile(
    r"\b(?:JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|"
    r"SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)"
    r"\s+\d{1,2},\s+\d{4}\b",
    re.IGNORECASE,
)

_JUDGE_BLOCK_PATTERN = re.compile(
    # Matches:  [N. V. RAMANA, CJI, A. S. BOPANNA AND\nHIMA KOHLI, JJ.]
    # or:       [R. SUBHASH REDDY AND HRISHIKESH ROY, JJ.]
    r"\[([A-Z][A-Z\s.,\n]+?(?:CJI|J|JJ)\.?\])",
    re.DOTALL,
)

_JUDGE_SUFFIX = re.compile(r"\b(?:CJI|J\.?|JJ\.?)\b", re.IGNORECASE)

_CASE_NUMBER_PATTERN = re.compile(
    r"\((?:Civil Appeal|Criminal Appeal|Writ Petition|Special Leave Petition|"
    r"Transfer Petition|Contempt Petition|SLP|WP|CA|Crl\.?\s*A\.?)"
    r"[\w\s\-.,()]+?\d{4}\)",
    re.IGNORECASE,
)

_CITATION_PATTERN = re.compile(
    r"(?:"
    r"\[\d{4}\]\s+\d+\s+S\.?C\.?R\.?\s+\d+"          # [2022] 1 S.C.R. 1
    r"|\(\d{4}\)\s+\d+\s+SCC\s+\d+"                   # (2019) 20 SCC 1
    r"|\d{4}\s+AIR\s+\d+"                              # 2001 AIR 123
    r"|AIR\s+\d{4}\s+SC\s+\d+"                         # AIR 2001 SC 123
    r"|\(\d{4}\)\s+\d+\s+SCR\s+\d+"                   # (2014) 14 SCR 1029
    r"|\[\d{4}\]\s+\d+\s+SCR\s+\d+"                   # [2014] 14 SCR 1029
    r")",
    re.IGNORECASE,
)

_ACT_PATTERN = re.compile(
    r"[A-Z][A-Za-z\s&,()]+?(?:Act|Code|Rules|Regulations?|Constitution|"
    r"Ordinance|Order|Statute)"
    r"(?:[,\s]+(?:of\s+)?\d{4})?",
)

_SECTION_PATTERN = re.compile(
    r"\b(?:"
    r"s\.\s*\d+[\w\(\)\.,/]*"               # s.34, s.40(a)(iib)
    r"|[Ss]ections?\s+\d+[\w\(\)\.,/]*"     # Section 34, Sections 34
    r"|[Aa]rts?\.?\s+\d+[\w\(\)\.,/]*"      # Art. 289, Art.14
    r"|[Cc]lauses?\s+\d+[\w\(\)\.,/]*"      # Clause 2.2
    r"|[Rr]ules?\s+\d+[\w\(\)\.,/]*"        # Rule 4
    r")",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_text_for_extraction(raw_text: str) -> str:
    """
    Strip margin noise (single-letter column markers A-H in SCR PDFs) and
    page annotation markers from the raw text before extraction.
    """
    lines = raw_text.splitlines()
    cleaned = [
        line for line in lines
        if not _SINGLE_LETTER_LINE.match(line)
        and not re.match(r"^---\s*Page\s+\d+", line.strip())
    ]
    return "\n".join(cleaned)


def _headnote_window(text: str, chars: int = 3_000) -> str:
    """Return the first ``chars`` characters — the headnote zone in SCR format."""
    return text[:chars]


def _normalise_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def normalize_section(sec: str) -> dict:
    """
    Convert a raw section/clause/article/rule string into a normalised dict.

    This is the key deduplication step: stylistic variants that refer to the
    same provision collapse to a single canonical form, so Neo4j MERGE will
    not create duplicate nodes when the same section is cited differently
    across documents.

    Examples
    --------
    "s.34"              → {"type": "Section", "number": "34"}
    "s.34(2)(b)(ii)"    → {"type": "Section", "number": "34(2)(b)(ii)"}
    "Section 34"        → {"type": "Section", "number": "34"}
    "SECTION 34"        → {"type": "Section", "number": "34"}
    "Sections 34"       → {"type": "Section", "number": "34"}
    "Clause 2.2"        → {"type": "Clause",  "number": "2.2"}
    "Art. 289"          → {"type": "Article", "number": "289"}
    "Art.14"            → {"type": "Article", "number": "14"}
    "Rule 4"            → {"type": "Rule",    "number": "4"}

    The *number* field keeps sub-clause suffixes (e.g. ``34(2)(b)(ii)``) so
    that distinct provisions remain distinct nodes in the graph.
    """
    sec = _normalise_whitespace(sec)

    # Article / Art.
    m = re.match(r"[Aa]rts?\.?\s*(\d+[\w\(\)\.,/]*)", sec)
    if m:
        return {"type": "Article", "number": m.group(1)}

    # Clause
    m = re.match(r"[Cc]lauses?\s+(\d+[\w\(\)\.,/]*)", sec)
    if m:
        return {"type": "Clause", "number": m.group(1)}

    # Rule
    m = re.match(r"[Rr]ules?\s+(\d+[\w\(\)\.,/]*)", sec)
    if m:
        return {"type": "Rule", "number": m.group(1)}

    # Section (longhand: Section 34, Sections 34)
    m = re.match(r"[Ss]ections?\s+(\d+[\w\(\)\.,/]*)", sec)
    if m:
        return {"type": "Section", "number": m.group(1)}

    # Section (shorthand: s.34, s. 34, s.34(2)(b))
    m = re.match(r"s\.\s*(\d+[\w\(\)\.,/]*)", sec, re.IGNORECASE)
    if m:
        return {"type": "Section", "number": m.group(1)}

    # Fallback — return as-is with a generic type
    return {"type": "Section", "number": sec}


def _parse_act(raw: str) -> dict | None:
    """
    Split e.g. 'Arbitration and Conciliation Act, 1996' into
    {"name": "Arbitration and Conciliation Act", "year": 1996}.

    Returns None if no year is found (keeps output quality high).
    """
    raw = _normalise_whitespace(raw)
    m = re.search(r",?\s*(?:of\s+)?(\d{4})\s*$", raw)
    if not m:
        return None
    year = int(m.group(1))
    name = raw[: m.start()].rstrip(" ,")
    if len(name) < 5:
        return None
    return {"name": name, "year": year}


def _parse_date(raw: str) -> str:
    """
    Convert a raw date string ('JANUARY 07, 2022') to ISO-8601 ('2022-01-07').
    Falls back to the original normalised string if parsing fails.
    """
    raw = _normalise_whitespace(raw).upper()
    for fmt in ("%B %d, %Y", "%B %d %Y"):
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return raw  # better than silently dropping the date


# ---------------------------------------------------------------------------
# Individual extractors
# ---------------------------------------------------------------------------

def extract_parties(headnote: str) -> list[dict]:
    """
    Extract petitioner and respondent as a list of party dicts.

    SCR headnotes always have:
        PARTY NAME
        v.
        PARTY NAME

    We look for lines surrounding a standalone "v." or "v.s." line.

    Returns
    -------
    list of {"name": str, "role": "Petitioner" | "Respondent"}

    The ``role`` key is kept on the dict here so the Neo4j loader can choose
    the correct relationship type (HAS_PETITIONER vs HAS_RESPONDENT) without
    storing role on the Party node itself.
    """
    lines = [ln.strip() for ln in headnote.splitlines() if ln.strip()]

    for i, line in enumerate(lines):
        if re.match(r"^v\.?s?\.?$", line, re.IGNORECASE):
            # Walk backward for petitioner (skip noise like page numbers)
            petitioner_parts = []
            for j in range(i - 1, max(i - 5, -1), -1):
                candidate = lines[j]
                if re.match(r"^\[?\d{4}\]", candidate):   # citation line
                    break
                if re.match(r"^\d+$", candidate):          # bare page number
                    break
                petitioner_parts.insert(0, candidate)

            # Walk forward for respondent
            respondent_parts = []
            for j in range(i + 1, min(i + 5, len(lines))):
                candidate = lines[j]
                if re.match(r"^\(", candidate):             # case number
                    break
                respondent_parts.append(candidate)

            petitioner = _normalise_whitespace(" ".join(petitioner_parts))
            respondent  = _normalise_whitespace(" ".join(respondent_parts))

            parties: list[dict] = []
            if petitioner:
                parties.append({"name": petitioner, "role": "Petitioner"})
            if respondent:
                parties.append({"name": respondent, "role": "Respondent"})
            return parties

    return []


def extract_date(headnote: str) -> str:
    """
    Extract and ISO-format the decision date.

    Returns
    -------
    str  ISO-8601 date string ("2022-01-07"), or "" if not found.
    """
    match = _DATE_PATTERN.search(headnote)
    if match:
        return _parse_date(match.group())
    return ""


def extract_judges(headnote: str) -> list[dict]:
    """
    Extract judge names from the bracketed judge block, e.g.
    [N. V. RAMANA, CJI, A. S. BOPANNA AND HIMA KOHLI, JJ.]

    Returns
    -------
    list of {"name": str}
    """
    match = _JUDGE_BLOCK_PATTERN.search(headnote)
    if not match:
        return []

    raw_block = match.group(1)
    raw_block = _normalise_whitespace(raw_block)
    raw_block = raw_block.rstrip("]")
    raw_block = _JUDGE_SUFFIX.sub("", raw_block)

    parts = re.split(r"\s+AND\s+|,", raw_block, flags=re.IGNORECASE)
    judges = [_normalise_whitespace(p) for p in parts]
    judges = [j for j in judges if len(j) > 2]
    return [{"name": j} for j in judges]


def extract_court(headnote: str, full_text: str) -> str:
    """
    Identify the court name.

    Strategy:
      1. Look for reporter / header keywords (highest priority, unambiguous).
      2. Scan headnote for a known court name verbatim.
      3. Fall back to spaCy ORG entities in the first 500 chars.
    """
    search_zone = headnote + full_text[3_000:6_000]

    # Step 1 — reporter / header inference
    if re.search(r"\[?\d{4}\]?\s+\d+\s+S\.C\.R\.", search_zone):
        return "Supreme Court of India"
    if re.search(r"SUPREME COURT REPORTS", search_zone, re.IGNORECASE):
        return "Supreme Court of India"

    # Step 2 — verbatim known court in headnote only
    for known in KNOWN_COURTS:
        if known.lower() in headnote.lower():
            return known

    # Step 3 — spaCy fallback
    doc = _nlp()(headnote[:500])
    for ent in doc.ents:
        if ent.label_ == "ORG" and len(ent.text.split()) > 1:
            return ent.text.strip()

    return ""


def extract_acts(headnote: str) -> list[dict]:
    """
    Extract statute names from the headnote and return structured dicts.

    Returns
    -------
    list of {"name": str, "year": int}
    """
    matches = _ACT_PATTERN.findall(headnote)
    seen: set[str] = set()
    acts: list[dict] = []
    for m in matches:
        m = _normalise_whitespace(m)
        parsed = _parse_act(m)
        if parsed is None:
            continue
        key = f"{parsed['name']}|{parsed['year']}"
        if key not in seen:
            seen.add(key)
            acts.append(parsed)
    return sorted(acts, key=lambda a: a["name"])


def extract_sections(headnote: str) -> list[dict]:
    """
    Extract section / article / clause / rule references and return
    normalised structured dicts.

    Critically, 's.34' and 'Section 34' both produce
    ``{"type": "Section", "number": "34"}`` so Neo4j MERGE deduplicates
    them automatically — even across different documents.

    Returns
    -------
    list of {"type": str, "number": str}
    """
    matches = _SECTION_PATTERN.findall(headnote)
    seen: set[str] = set()
    sections: list[dict] = []
    for m in matches:
        m = _normalise_whitespace(m)
        if not m:
            continue
        normed = normalize_section(m)
        key = f"{normed['type']}|{normed['number']}"
        if key not in seen:
            seen.add(key)
            sections.append(normed)
    return sorted(sections, key=lambda s: (s["type"], s["number"]))


def extract_citations(full_text: str) -> list[str]:
    """
    Extract all case law citations from the full document text.
    Kept as raw strings — Phase 2 will resolve these to actual Case nodes.
    """
    matches = _CITATION_PATTERN.findall(full_text)
    normalised = [_normalise_whitespace(m) for m in matches]
    seen: set[str] = set()
    unique: list[str] = []
    for c in normalised:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


# ---------------------------------------------------------------------------
# Top-level graph extractor
# ---------------------------------------------------------------------------

def extract_concepts(headnote: str, full_text: str) -> list[dict]:
    """
    Extract canonical legal concepts with multilingual aliases from document text.
    """
    combined = (headnote + " " + full_text[:4000]).lower()
    concepts = []

    # Road Accident / Motor Vehicles
    if any(w in combined for w in ["motor vehicle", "accident", "mcop", "collision", "vehicle", "விபத்து", "வண்டி", "மோதி", "இழப்பீடு"]):
        concepts.append({
            "id": "CONCEPT_ROAD_ACCIDENT",
            "name": "Road Accident",
            "domain": "Motor Vehicles",
            "aliases": ["Motor Accident", "Road Accident", "Vehicle Collision", "MCOP", "விபத்து", "வண்டி விபத்து", "வாகன விபத்து", "சேதம்", "இழப்பீடு"],
        })

    # Forgery / Document Fraud
    if any(w in combined for w in ["forgery", "forged", "cheating", "fraud", "போலி", "கையெழுத்து"]):
        concepts.append({
            "id": "CONCEPT_FORGERY",
            "name": "Forgery / Document Fraud",
            "domain": "Criminal",
            "aliases": ["Forgery", "Fake Document", "Fraud", "Cheating", "போலி", "கையெழுத்து ஏமாற்று"],
        })

    # Tenancy / Eviction
    if any(w in combined for w in ["tenant", "landlord", "eviction", "rent control", "வாடகை", "வீடு", "நிலம்"]):
        concepts.append({
            "id": "CONCEPT_TENANCY",
            "name": "Tenancy / Eviction Dispute",
            "domain": "Property",
            "aliases": ["Tenancy", "Eviction", "Rent Control", "Lease Dispute", "வாடகை", "வீடு eviction"],
        })

    # Criminal Offence
    if any(w in combined for w in ["ipc", "penal code", "offence", "assault", "murder", "கொலை", "திருட்டு"]):
        concepts.append({
            "id": "CONCEPT_CRIMINAL_OFFENCE",
            "name": "Criminal Offence",
            "domain": "Criminal",
            "aliases": ["Criminal Offence", "IPC", "Penal Code", "FIR", "கொலை", "குற்றம்"],
        })

    return concepts


def extract_legal_graph(raw_text: str) -> dict:
    """
    Run the full extraction pipeline on a single judgment text and return
    a graph-ready dict with separate ``nodes`` and ``relationships`` keys.
    """
    # Pre-clean: remove column markers and page headers
    text = _clean_text_for_extraction(raw_text)
    headnote = _headnote_window(text)

    parties = extract_parties(headnote)

    # Derive case name from normalised party list for consistency
    petitioner_name = next(
        (p["name"] for p in parties if p["role"] == "Petitioner"), ""
    )
    respondent_name = next(
        (p["name"] for p in parties if p["role"] == "Respondent"), ""
    )
    case_name = (
        f"{petitioner_name} v. {respondent_name}"
        if petitioner_name and respondent_name
        else ""
    )

    # Language detection
    doc_lang = "Tamil" if re.search(r"[\u0B80-\u0BFF]", text) else "English"
    extracted_concepts = extract_concepts(headnote, text)

    nodes = {
        "case": {
            "name":          case_name,
            "decision_date": extract_date(headnote),
            "language":      doc_lang,
        },
        "court":     {"name": extract_court(headnote, text)},
        "judges":    extract_judges(headnote),
        "acts":      extract_acts(headnote),
        "sections":  extract_sections(headnote),
        "parties":   parties,
        "citations": extract_citations(text),
        "concepts":  extracted_concepts,
    }

    relationships = [
        ("Case", "HEARD_IN",         "Court"),
        ("Case", "DECIDED_BY",       "Judge"),
        ("Case", "UNDER_ACT",        "Act"),
        ("Case", "INVOLVES_SECTION", "Section"),
        ("Case", "HAS_PETITIONER",   "Party"),
        ("Case", "HAS_RESPONDENT",   "Party"),
    ]
    for _c in extracted_concepts:
        relationships.append(("Case", "INVOLVES_CONCEPT", "LegalConcept"))

    return {"nodes": nodes, "relationships": relationships}


# ---------------------------------------------------------------------------
# Backward-compatibility alias
# ---------------------------------------------------------------------------

#: Alias so any existing call-sites continue to work while the rest of the
#: codebase migrates to the new name.
extract_entities = extract_legal_graph


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    if len(sys.argv) < 2:
        print("Usage: python extractor/entity_extractor.py <path_to_txt_file>")
        sys.exit(0)

    path = sys.argv[1]
    try:
        with open(path, encoding="utf-8") as fh:
            raw = fh.read()
    except FileNotFoundError:
        print(f"[ERROR] File not found: {path}")
        sys.exit(1)

    result = extract_legal_graph(raw)

    print(f"\n[entity_extractor] Results for: {path}")
    print("=" * 60)
    print(json.dumps(result, indent=2, ensure_ascii=False))
