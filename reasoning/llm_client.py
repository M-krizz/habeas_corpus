"""
reasoning/llm_client.py — Habeas Corpus
=========================================
Unified LLM client supporting OpenRouter and Google Gemini AI Studio.

Design:
  - Supports OPENROUTER_API_KEY (default model: google/gemini-2.5-flash)
  - Supports GEMINI_API_KEY as secondary alternative
  - Singleton Gemini client loaded once (when applicable)
  - Returns raw text from the LLM
  - All error handling here — callers never see API exceptions
  - If no keys are available, returns a structured "no LLM" response
    so the rest of the pipeline degrades gracefully
"""

from __future__ import annotations

import os
import re
import json
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()


@lru_cache(maxsize=1)
def _get_client():
    """Load Gemini GenAI client once. Returns None if key not set."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "YOUR_GEMINI_KEY_HERE":
        return None
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        print("[llm_client] Gemini GenAI client loaded.")
        return client
    except Exception as exc:
        print(f"[llm_client] WARNING — Gemini GenAI load failed: {exc}")
        return None


def call_llm(
    prompt: str,
    temperature: float = 0.1,
    max_tokens: int = 2048,
    json_mode: bool = True,
) -> str | None:
    """
    Unified LLM completion function supporting OpenRouter and Google AI Studio.
    Prioritizes OPENROUTER_API_KEY if present in .env, otherwise GEMINI_API_KEY.
    Returns raw text output (or None on failure/unconfigured).
    """
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()

    if openrouter_key:
        try:
            import httpx
            model_name = os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash")
            payload: dict = {
                "model": model_name,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
            }
            if json_mode:
                payload["response_format"] = {"type": "json_object"}

            headers = {
                "Authorization": f"Bearer {openrouter_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/M-krizz/habeas_corpus",
                "X-Title": "Habeas Corpus Legal Engine",
            }
            res = httpx.post(
                "https://openrouter.ai/api/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=60.0,
            )
            res.raise_for_status()
            data = res.json()
            raw = data["choices"][0]["message"]["content"].strip()

            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = re.sub(r"^```[a-z]*\n?", "", raw)
                raw = re.sub(r"\n?```$", "", raw.strip())
            print(f"[llm_client] OpenRouter ({model_name}) generated {len(raw)} chars.")
            return raw
        except Exception as exc:
            print(f"[llm_client] OpenRouter generation failed: {exc}")
            if not gemini_key or gemini_key == "YOUR_GEMINI_KEY_HERE":
                return None

    if gemini_key and gemini_key != "YOUR_GEMINI_KEY_HERE":
        client = _get_client()
        if client is not None:
            try:
                from google.genai import types
                model_name = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
                cfg = types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
                if json_mode:
                    cfg.response_mime_type = "application/json"
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=cfg,
                )
                raw = response.text.strip()
                if raw.startswith("```"):
                    raw = re.sub(r"^```[a-z]*\n?", "", raw)
                    raw = re.sub(r"\n?```$", "", raw.strip())
                print(f"[llm_client] Gemini ({model_name}) generated {len(raw)} chars.")
                return raw
            except Exception as exc:
                print(f"[llm_client] Gemini generation failed: {exc}")
                return None

    print("[llm_client] WARNING — Neither OPENROUTER_API_KEY nor GEMINI_API_KEY is properly configured.")
    return None


# ---------------------------------------------------------------------------
# No-LLM fallback response
# ---------------------------------------------------------------------------

_NO_LLM_RESPONSE = """\
{{
  "summary": "The Habeas Corpus AI reasoning engine found {n_cases} relevant judicial precedent(s) for your query about '{incident}'. However, neither OPENROUTER_API_KEY nor GEMINI_API_KEY could be connected. Please ensure OPENROUTER_API_KEY is active in your .env file for full AI explanations. The relevant cases have been identified and are listed below.",
  "applicable_acts": {acts},
  "applicable_sections": {sections},
  "precedents": {precedents},
  "what_to_do": "1. Review the relevant cases listed. 2. Note the applicable Acts and Sections. 3. Verify your OPENROUTER_API_KEY in .env for full AI-generated explanations. 4. Consult a qualified advocate for legal advice.",
  "disclaimer": "This is AI-assisted legal research based on verified judicial precedents. It is not legal advice. Please consult a qualified advocate."
}}"""


def generate(prompt: str, fallback_context: dict | None = None) -> str:
    """
    Send a prompt to the configured LLM and return the raw text response.
    """
    res = call_llm(prompt, temperature=0.1, max_tokens=2048, json_mode=True)
    if res is not None:
        return res
    return _build_fallback(fallback_context or {})


def _build_fallback(ctx: dict) -> str:
    """Build a structured JSON fallback when LLM is unavailable."""
    return _NO_LLM_RESPONSE.format(
        n_cases    = ctx.get("n_cases", 0),
        incident   = ctx.get("incident", "your legal query"),
        acts       = json.dumps(ctx.get("acts", [])),
        sections   = json.dumps(ctx.get("sections", [])),
        precedents = json.dumps(ctx.get("precedents", [])),
    )
