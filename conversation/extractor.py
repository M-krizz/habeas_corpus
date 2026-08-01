"""
conversation/extractor.py — Nyaya-Setu Conversation Brain
==========================================================
Extracts structured legal facts from a user message.

Primary path: Gemini LLM extracts a JSON patch from the message.
Fallback path: Regex-based extraction for common legal entities
               (vehicle types, section numbers, state names, cities).

The extractor NEVER answers the legal question — it only extracts facts.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Gemini client — lazy singleton (shared with legal_concept_mapper)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_gemini():
    """Load Gemini client once. Returns None if key is missing."""
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key or api_key == "YOUR_GEMINI_KEY_HERE":
        return None
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        return client
    except Exception:
        return None


# ---------------------------------------------------------------------------
# LLM extraction prompt
# ---------------------------------------------------------------------------

_EXTRACTION_PROMPT = """\
You are a legal fact extraction engine for the Nyaya-Setu Indian Legal AI.

You are given:
1. The user's latest message.
2. The current conversation memory (what we already know).

Your job: Extract NEW legal facts from the user's message and return them
as a flat JSON object. Only include facts that are NEW or CORRECTED —
do not repeat what is already in memory.

Extractable fields (use these exact keys):
- "incident": type of incident (e.g. "Road Accident", "Property Dispute", "Cheque Bounce", "Domestic Violence", "Consumer Complaint")
- "legal_domain": area of law (e.g. "Motor Vehicles", "Criminal", "Property", "Family", "Consumer", "Labour")
- "vehicle": vehicle type (e.g. "Bike", "Car", "Truck", "Auto", "Bus")
- "state": Indian state name (e.g. "Tamil Nadu", "Karnataka", "Maharashtra")
- "city": city name (e.g. "Chennai", "Bangalore", "Mumbai")
- "sections": list of section numbers mentioned (e.g. ["134", "166"])
- "acts": list of Act names mentioned (e.g. ["Motor Vehicles Act", "IPC"])
- "injury": severity ("Minor", "Serious", "Fatal", "None")
- "fir_filed": whether FIR was filed (true/false)
- "current_goal": what the user wants to know (e.g. "Find Liability", "Get Compensation", "Defend Themselves")
- "property_type": type of property ("Land", "Flat", "House", "Commercial")
- "dispute_type": type of property dispute ("Eviction", "Encroachment", "Ownership")
- "cheque_amount": amount on cheque
- "bank": bank name
- "notice_sent": whether legal notice was sent (true/false)
- "agreement_exists": whether there is a written agreement (true/false)

Rules:
- Return ONLY a JSON object. No markdown, no explanation.
- If the message contains no new extractable facts, return an empty JSON object.
- Do NOT guess or infer facts not stated by the user.
- Section numbers: bare numbers only ("134", not "Section 134").
- For "current_goal", infer from context what the user wants help with.

CURRENT MEMORY:
{memory_json}

USER MESSAGE:
{user_message}
"""


# ---------------------------------------------------------------------------
# LLM-based extraction
# ---------------------------------------------------------------------------

def _llm_extract(user_message: str, memory_json: str) -> dict | None:
    """
    Use Gemini to extract structured facts from a user message.
    Returns a dict of new facts, or None on failure.
    """
    client = _get_gemini()
    if client is None:
        return None

    prompt = _EXTRACTION_PROMPT.format(
        memory_json=memory_json,
        user_message=user_message.strip(),
    )

    try:
        from google.genai import types
        model_name = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=512,
                response_mime_type="application/json",
            ),
        )
        raw = response.text.strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw.strip())

        data = json.loads(raw)
        if not isinstance(data, dict):
            return None

        # Remove empty values
        return {k: v for k, v in data.items()
                if v is not None and v != "" and v != []}

    except Exception as exc:
        print(f"[extractor] LLM extraction failed: {exc}. Using regex fallback.")
        return None


# ---------------------------------------------------------------------------
# Regex-based fallback extraction
# ---------------------------------------------------------------------------

# Indian states for matching
_STATES = {
    "andhra pradesh", "arunachal pradesh", "assam", "bihar",
    "chhattisgarh", "goa", "gujarat", "haryana", "himachal pradesh",
    "jharkhand", "karnataka", "kerala", "madhya pradesh", "maharashtra",
    "manipur", "meghalaya", "mizoram", "nagaland", "odisha", "punjab",
    "rajasthan", "sikkim", "tamil nadu", "telangana", "tripura",
    "uttar pradesh", "uttarakhand", "west bengal", "delhi",
}

# Major cities → state mapping
_CITIES = {
    "mumbai": "Maharashtra", "delhi": "Delhi", "bangalore": "Karnataka",
    "bengaluru": "Karnataka", "chennai": "Tamil Nadu",
    "hyderabad": "Telangana", "kolkata": "West Bengal",
    "pune": "Maharashtra", "ahmedabad": "Gujarat", "jaipur": "Rajasthan",
    "lucknow": "Uttar Pradesh", "bhopal": "Madhya Pradesh",
    "chandigarh": "Punjab", "kochi": "Kerala", "coimbatore": "Tamil Nadu",
    "nagpur": "Maharashtra", "patna": "Bihar", "indore": "Madhya Pradesh",
    "thiruvananthapuram": "Kerala", "visakhapatnam": "Andhra Pradesh",
    "guwahati": "Assam", "ranchi": "Jharkhand", "raipur": "Chhattisgarh",
    "surat": "Gujarat", "vadodara": "Gujarat", "noida": "Uttar Pradesh",
    "gurgaon": "Haryana", "gurugram": "Haryana",
}

# Vehicle keywords
_VEHICLES = {
    "bike": "Bike", "motorcycle": "Bike", "two-wheeler": "Bike",
    "scooter": "Bike", "scooty": "Bike", "motorbike": "Bike",
    "car": "Car", "sedan": "Car", "hatchback": "Car",
    "truck": "Truck", "lorry": "Truck",
    "auto": "Auto", "auto-rickshaw": "Auto", "autorickshaw": "Auto",
    "bus": "Bus", "van": "Van", "tempo": "Tempo",
    "cycle": "Bicycle", "bicycle": "Bicycle",
    "pedestrian": "Pedestrian",
}

# Incident keyword patterns
_INCIDENT_PATTERNS = [
    (["accident", "collision", "hit", "crash", "road"], "Road Accident", "Motor Vehicles"),
    (["property", "land", "flat", "house", "tenant", "landlord", "evict"],
     "Property Dispute", "Property"),
    (["cheque", "bounce", "dishonour"], "Cheque Bounce", "Criminal"),
    (["dowry", "domestic", "violence", "harassment", "498"],
     "Domestic Violence", "Family"),
    (["consumer", "product", "defect", "refund", "warranty"],
     "Consumer Complaint", "Consumer"),
    (["theft", "robbery", "stolen", "steal"], "Theft / Robbery", "Criminal"),
    (["murder", "kill", "homicide"], "Homicide", "Criminal"),
    (["divorce", "custody", "maintenance", "alimony"], "Family Dispute", "Family"),
    (["employment", "fired", "sacked", "salary", "termination"],
     "Labour Dispute", "Labour"),
    (["dog", "animal", "bite", "stray"], "Animal Attack", "Criminal/Tort"),
    (["fraud", "forge", "forgery", "cheating", "fake"], "Fraud / Forgery", "Criminal"),
]


def _regex_extract(user_message: str, current_memory: dict) -> dict:
    """
    Regex and keyword-based fact extraction fallback.
    Returns a dict of newly extracted facts.
    """
    msg_lower = user_message.lower()
    patch: dict = {}

    # Extract section numbers
    section_matches = re.findall(
        r"\bsection\s+(\d+[A-Za-z]?)\b", msg_lower, re.IGNORECASE
    )
    if section_matches:
        patch["sections"] = [s.upper() if s[-1].isalpha() else s
                             for s in section_matches]

    # Extract vehicle type
    if not current_memory.get("vehicle"):
        for keyword, vehicle_type in _VEHICLES.items():
            if keyword in msg_lower:
                patch["vehicle"] = vehicle_type
                break

    # Extract city and infer state
    if not current_memory.get("city"):
        for city_key, state_val in _CITIES.items():
            if city_key in msg_lower:
                patch["city"] = city_key.title()
                if not current_memory.get("state"):
                    patch["state"] = state_val
                break

    # Extract state directly
    if not current_memory.get("state") and "state" not in patch:
        for state_name in _STATES:
            if state_name in msg_lower:
                patch["state"] = state_name.title()
                break

    # Extract incident type
    if not current_memory.get("incident"):
        for keywords, incident, domain in _INCIDENT_PATTERNS:
            if any(kw in msg_lower for kw in keywords):
                patch["incident"] = incident
                patch["legal_domain"] = domain
                break

    # Extract injury severity
    if not current_memory.get("injury"):
        if any(w in msg_lower for w in ["fatal", "death", "died", "dead", "killed"]):
            patch["injury"] = "Fatal"
        elif any(w in msg_lower for w in ["serious", "severe", "hospital", "fracture", "critical"]):
            patch["injury"] = "Serious"
        elif any(w in msg_lower for w in ["minor", "bruise", "scratch", "small"]):
            patch["injury"] = "Minor"
        elif any(w in msg_lower for w in ["no injury", "not injured", "no one was hurt", "unhurt"]):
            patch["injury"] = "None"

    # Extract FIR status
    if current_memory.get("fir_filed") is None:
        if any(p in msg_lower for p in ["filed fir", "fir filed", "police complaint",
                                         "lodged fir", "registered fir", "filed a complaint"]):
            patch["fir_filed"] = True
        elif any(p in msg_lower for p in ["no fir", "didn't file", "did not file",
                                           "haven't filed", "not filed"]):
            patch["fir_filed"] = False

    # Extract cheque-specific details
    cheque_match = re.search(r"(?:rs\.?|inr|rupees?)\s*(\d[\d,]*)", msg_lower)
    if cheque_match and not current_memory.get("cheque_amount"):
        patch["cheque_amount"] = cheque_match.group(1).replace(",", "")

    # Extract boolean facts
    if "notice" in msg_lower and current_memory.get("notice_sent") is None:
        if any(w in msg_lower for w in ["sent notice", "gave notice", "served notice"]):
            patch["notice_sent"] = True
        elif any(w in msg_lower for w in ["no notice", "didn't send", "without notice"]):
            patch["notice_sent"] = False

    return patch


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_facts(user_message: str, current_memory_dict: dict) -> dict:
    """
    Extract structured legal facts from a user message.

    Tries LLM first, falls back to regex if LLM is unavailable.
    Returns a dict of new/updated facts (a "patch" to merge into memory).

    Parameters
    ----------
    user_message       : the raw message from the user
    current_memory_dict: the current memory as a dict (for LLM context)

    Returns
    -------
    dict — patch of new facts to merge into ConversationMemory
    """
    memory_json = json.dumps(
        {k: v for k, v in current_memory_dict.items()
         if k not in ("messages", "custom_facts") and v is not None
         and v != [] and v != {}},
        indent=2,
    )

    # Try LLM first
    patch = _llm_extract(user_message, memory_json)
    if patch is not None:
        print(f"[extractor] LLM extracted: {patch}")
        return patch

    # Regex fallback
    patch = _regex_extract(user_message, current_memory_dict)
    print(f"[extractor] Regex extracted: {patch}")
    return patch
