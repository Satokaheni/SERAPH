"""
SERAPH Stage 1 — Emotion Recognizer
=====================================
Raw emotional signal extraction from input text.

This stage deliberately does NOT label emotions — it surfaces raw cues
(lexical, syntactic, pragmatic) that Stage 2 will classify.

Uses: claude-haiku (cheap, fast) for cost efficiency.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import anthropic
from pydantic import BaseModel, Field

from config import (
    ANTHROPIC_API_KEY,
    HAIKU_MODEL,
    MAX_TOKENS,
    TEMPERATURE,
    USE_PROMPT_CACHING,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class EmotionalSignal(BaseModel):
    """A single detected emotional cue in the input text."""

    cue_text: str = Field(description="The exact phrase or token that carries the signal")
    cue_type: str = Field(
        description=(
            "Type of cue: 'lexical' (word choice), 'syntactic' (sentence structure), "
            "'pragmatic' (implied meaning), 'prosodic_text' (punctuation/caps conveying tone), "
            "or 'contextual' (situational reference)"
        )
    )
    intensity_hint: float = Field(
        ge=0.0, le=1.0,
        description="Rough signal strength — 0.0 = very weak, 1.0 = very strong"
    )
    valence_hint: str = Field(
        description="Rough polarity: 'positive', 'negative', 'neutral', or 'ambiguous'"
    )
    notes: Optional[str] = Field(
        default=None,
        description="Optional annotation explaining the signal"
    )


class Stage1Output(BaseModel):
    """
    Output of the Emotion Recognizer (Stage 1).
    
    A structured representation of raw emotional signals present in the text.
    Deliberately avoids final emotion labels — those are Stage 2's job.
    """

    original_text: str = Field(description="The verbatim input text")
    signals: list[EmotionalSignal] = Field(
        description="All detected emotional signals, ordered by intensity (descending)"
    )
    overall_intensity_hint: float = Field(
        ge=0.0, le=1.0,
        description="Aggregate emotional intensity across all signals"
    )
    context_type: str = Field(
        description=(
            "Inferred social/communicative context: "
            "'personal_distress', 'interpersonal_conflict', 'seeking_support', "
            "'sharing_positive_news', 'venting', 'neutral_information', "
            "'crisis', 'ambiguous', or 'other'"
        )
    )
    has_negation: bool = Field(
        description="Whether negation patterns are present (e.g. 'I don't feel...')"
    )
    has_hedging: bool = Field(
        description="Whether hedging patterns are present (e.g. 'kind of', 'maybe', 'sort of')"
    )
    multimodal_text_cues: list[str] = Field(
        default_factory=list,
        description=(
            "Text-based prosodic cues: all-caps words, ellipsis, "
            "repeated punctuation, exclamation marks, etc."
        )
    )
    raw_notes: Optional[str] = Field(
        default=None,
        description="Free-form analyst notes about unusual or ambiguous signals"
    )


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

def _build_system_prompt() -> str:
    schema = json.dumps(Stage1Output.model_json_schema(), indent=2)
    return f"""You are an expert affective computing analyst specializing in \
emotional signal detection in natural language.

Your task is SIGNAL EXTRACTION ONLY — you are NOT labeling emotions yet. You are \
acting as a sensitive instrument that detects the presence and shape of emotional \
content in text, the way a seismograph detects vibrations before we classify the \
earthquake type.

Look for:
1. LEXICAL signals — emotionally charged words, intensifiers, diminishers
2. SYNTACTIC signals — sentence fragments, run-ons, rhetorical questions, \
   imperative mood
3. PRAGMATIC signals — implied meanings, what is NOT said, indirect expressions
4. PROSODIC-text signals — capitalization, punctuation patterns, ellipsis, \
   repetition conveying tone
5. CONTEXTUAL signals — situational references that imply emotional states

Important rules:
- Do NOT assign final emotion labels (e.g., do not say "this is sadness")
- DO note intensity hints (weak / moderate / strong as 0.0–1.0)
- DO note valence hints (positive / negative / neutral / ambiguous)
- Be sensitive to negation and hedging — these flip or soften signals
- Extract ALL signals, even subtle ones — Stage 2 will filter

You MUST respond ONLY with a valid JSON object that exactly matches this schema \
(use the exact field names shown — no substitutions):

{schema}

No preamble, no explanation outside the JSON."""


SYSTEM_PROMPT = _build_system_prompt()


# ---------------------------------------------------------------------------
# Recognizer
# ---------------------------------------------------------------------------


class EmotionRecognizer:
    """
    Stage 1: Emotion Recognizer.
    
    Extracts raw emotional signals from input text using the Haiku model.
    Outputs a Stage1Output Pydantic object consumed by Stage 2.
    """

    def __init__(self) -> None:
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.model = HAIKU_MODEL

    def _build_messages(self, text: str) -> list[dict]:
        """Construct the messages array for the API call."""
        return [
            {
                "role": "user",
                "content": (
                    f"Extract all emotional signals from the following text.\n\n"
                    f"TEXT:\n{text}\n\n"
                    f"Return a JSON object matching the Stage1Output schema."
                ),
            }
        ]

    def _build_system(self) -> list[dict] | str:
        """Return system prompt, optionally with cache_control header."""
        if USE_PROMPT_CACHING:
            return [
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        return SYSTEM_PROMPT

    def run(self, text: str) -> Stage1Output:
        """
        Run Stage 1 on a text input.

        Args:
            text: Raw user message or dialogue turn.

        Returns:
            Stage1Output with all detected emotional signals.

        Raises:
            ValueError: If the model returns unparseable JSON.
        """
        logger.debug("Stage 1 | input length=%d chars", len(text))

        response = self.client.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS["stage1"],
            temperature=TEMPERATURE["stage1"],
            system=self._build_system(),
            messages=self._build_messages(text),
        )

        raw = response.content[0].text.strip()

        # Strip markdown fences if model wraps JSON (shouldn't with prompt, but safe)
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        try:
            data = json.loads(raw)
            # Ensure original_text is set even if model omitted it
            data.setdefault("original_text", text)
            output = Stage1Output(**data)
        except Exception as exc:
            logger.error("Stage 1 JSON parse failed: %s\nRaw: %s", exc, raw[:500])
            raise ValueError(f"Stage 1 failed to parse model output: {exc}") from exc

        logger.debug(
            "Stage 1 | signals=%d intensity=%.2f context=%s",
            len(output.signals),
            output.overall_intensity_hint,
            output.context_type,
        )
        return output
