"""
reasoning/llm_client.py — Habeas Corpus
=========================================
Thin wrapper around Google Gemini Flash.

Design:
  - Singleton client loaded once
  - Returns raw text from the LLM
  - All error handling here — callers never see API exceptions
  - If GEMINI_API_KEY is missing, returns a structured "no LLM" response
    so the rest of the pipeline degrades gracefully
"""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


@lru_cache(maxsize=1)
def _get_client():
    """Load Gemini GenAI client once. Returns None if key not set."""
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key or api_key == "YOUR_GEMINI_KEY_HERE":
        print("[llm_client] WARNING — GEMINI_API_KEY not configured. "
              "LLM reasoning will return a structured fallback response.")
        return None
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        print("[llm_client] Gemini 2.5 Flash client loaded.")
        return client
    except Exception as exc:
        print(f"[llm_client] WARNING — Gemini load failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# No-LLM fallback response
# ---------------------------------------------------------------------------

_NO_LLM_RESPONSE = """\
{{
  "summary": "The Habeas Corpus AI reasoning engine found {n_cases} relevant judicial precedent(s) for your query about '{incident}'. However, the LLM reasoning layer (Gemini API) is not currently configured. Please add GEMINI_API_KEY to your .env file for full AI explanations. The relevant cases have been identified and are listed below.",
  "applicable_acts": {acts},
  "applicable_sections": {sections},
  "precedents": {precedents},
  "what_to_do": "1. Review the relevant cases listed. 2. Note the applicable Acts and Sections. 3. Configure GEMINI_API_KEY for full AI-generated explanations. 4. Consult a qualified advocate for legal advice.",
  "disclaimer": "This is AI-assisted legal research based on verified judicial precedents. It is not legal advice. Please consult a qualified advocate."
}}"""


def generate(prompt: str, fallback_context: dict | None = None) -> str:
    """
    Send a prompt to Gemini Flash and return the raw text response.

    Parameters
    ----------
    prompt           : the full prompt string from prompt_builder
    fallback_context : dict with keys: n_cases, incident, acts, sections,
                       precedents — used to build a meaningful fallback
                       if the LLM is unavailable.

    Returns
    -------
    str — raw LLM output (should be valid JSON per the prompt format)
    """
    client = _get_client()

    if client is None:
        return _build_fallback(fallback_context or {})

    try:
        from google.genai import types
        model_name = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=2048,
                response_mime_type="application/json",
            ),
        )
        raw = response.text.strip()
        print(f"[llm_client] Generated {len(raw)} chars.")
        return raw
    except Exception as exc:
        print(f"[llm_client] Generation failed: {exc}")
        return _build_fallback(fallback_context or {})


def _build_fallback(ctx: dict) -> str:
    """Build a structured JSON fallback when LLM is unavailable."""
    import json
    return _NO_LLM_RESPONSE.format(
        n_cases    = ctx.get("n_cases", 0),
        incident   = ctx.get("incident", "your legal query"),
        acts       = json.dumps(ctx.get("acts", [])),
        sections   = json.dumps(ctx.get("sections", [])),
        precedents = json.dumps(ctx.get("precedents", [])),
    )
