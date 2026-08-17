from __future__ import annotations

import os
import random
import time
from typing import Optional

import google.generativeai as genai

from backend.config import BACKOFF_BASE_SECONDS, GEMINI_MODEL_NAME, MAX_RETRIES, MOCK_LLM, require_google_api_key, get_logger

logger = get_logger(__name__)


def _real_generate(system_prompt: str, user_prompt: str, temperature: float = 0.1) -> str:
    """Call Google Gemini via google-generativeai SDK.

    Raises RuntimeError if the key is missing or if calls fail after retries.
    """
    api_key = require_google_api_key()
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL_NAME,
        system_instruction=system_prompt,
    )

    last_exc: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = model.generate_content(
                user_prompt,
                generation_config=genai.types.GenerationConfig(temperature=temperature),
            )
            text = getattr(response, "text", None)
            if not text:
                raise RuntimeError("Gemini returned an empty response")
            return text.strip()
        except Exception as exc:  # broad on purpose for retry wrapper
            last_exc = exc
            if attempt == MAX_RETRIES:
                break
            sleep_s = BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
            logger.warning(
                "Gemini call failed (attempt %d/%d): %s; retrying in %.1fs",
                attempt,
                MAX_RETRIES,
                exc,
                sleep_s,
            )
            time.sleep(sleep_s)

    logger.error("Gemini generation failed after %d attempts", MAX_RETRIES, exc_info=True)
    raise RuntimeError(f"LLM generation failed after {MAX_RETRIES} attempts: {last_exc}") from last_exc


def _mock_generate(system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
    """Produce a deterministic, human-readable mock response suitable for demos.

    This intentionally does not call any external API and produces safe, repeatable
    outputs that mimic the shape of real responses (including [paper_id] citations
    when appropriate).
    """
    # Try to extract a short question from the user prompt to echo back
    q = None
    for line in reversed(user_prompt.splitlines()):
        line = line.strip()
        if line.lower().startswith("question:"):
            q = line.split("Question:", 1)[1].strip()
            break
    if not q:
        # fallback: first non-empty line after the context block
        parts = [l.strip() for l in user_prompt.splitlines() if l.strip()]
        q = parts[-1] if parts else "(no question)"

    # If this is the QA system prompt, return an example grounded answer
    if "Answer ONLY using information" in system_prompt:
        return (
            f"Mock answer to: {q}\n\n"
            "This is a mocked demo answer that cites retrieved excerpts only. "
            "Key claim example [2001.00001].\n\n"
            "### Notes\n"
            "- This response was produced by the demo-mode mock LLM.\n"
            "- Replace MOCK_LLM with MOCK_LLM=false and set GOOGLE_API_KEY to use real Gemini."
        )

    # If summarization prompt
    if "You are a research assistant summarizing a single academic paper" in system_prompt:
        return (
            "## Methodology\n"
            "Not stated in the retrieved excerpts [2001.00001].\n\n"
            "## Dataset\n"
            "Not stated in the retrieved excerpts [2001.00001].\n\n"
            "## Key Results\n"
            "Not stated in the retrieved excerpts [2001.00001].\n\n"
            "## Limitations\n"
            "Not stated in the retrieved excerpts [2001.00001]."
        )

    # If compare prompt
    if "You are a research assistant comparing multiple academic papers" in system_prompt:
        return (
            "| Paper | Method | Dataset | Model | Results | Limitations |\n"
            "|---|---|---|---|---|---|\n"
            "| 2001.00001 | Not stated | Not stated | Not stated | Not stated | Not stated |\n\n"
            "- Mock comparison: differences are illustrative only."
        )

    # Generic fallback
    return f"Mock output: {q}"


def generate(system_prompt: str, user_prompt: str, temperature: float = 0.1) -> str:
    """Generate text using Gemini when configured, otherwise fall back to a mock.

    The decision logic:
      - If MOCK_LLM is truthy (env var MOCK_LLM=true), always use mock.
      - Else, attempt to call the real Gemini API using GOOGLE_API_KEY from env via
        require_google_api_key().

    This ensures the repo never contains a hardcoded secret.
    """
    if MOCK_LLM:
        logger.info("MOCK_LLM is enabled — returning a mocked LLM response")
        return _mock_generate(system_prompt, user_prompt, temperature)

    # Otherwise call the real model (require_google_api_key will raise if missing)
    return _real_generate(system_prompt, user_prompt, temperature)
