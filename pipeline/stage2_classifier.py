"""
SERAPH Stage 2 — Emotion Classifier
=====================================
Takes Stage 1 signal output and produces a structured emotion classification.

Classification uses:
  - Plutchik's Wheel of Emotions (8 primaries + blends)
  - Dimensional model: Valence / Arousal / Dominance (VAD)
  - Intensity scaling (0.0 – 1.0)

Uses: claude-haiku (cheap, fast, well-calibrated on classification tasks).
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
from .stage1_recognizer import Stage1Output

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Plutchik's 8 Primary Emotions + Common Dyads
# ---------------------------------------------------------------------------

PLUTCHIK_PRIMARIES = [
    "joy", "trust", "fear", "surprise",
    "sadness", "disgust", "anger", "anticipation",
]

# Plutchik dyads (adjacent pairs on the wheel)
PLUTCHIK_DYADS = {
    "love":          ("joy", "trust"),
    "submission":    ("trust", "fear"),
    "awe":           ("fear", "surprise"),
    "disapproval":   ("surprise", "sadness"),
    "remorse":       ("sadness", "disgust"),
    "contempt":      ("disgust", "anger"),
    "aggressiveness":("anger", "anticipation"),
    "optimism":      ("anticipation", "joy"),
}

ALL_EMOTION_LABELS = PLUTCHIK_PRIMARIES + list(PLUTCHIK_DYADS.keys()) + ["neutral", "mixed"]


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class VADScores(BaseModel):
    """
    Dimensional emotion representation.
    
    Valence:   -1.0 (very negative) to +1.0 (very positive)
    Arousal:    0.0 (calm/low energy) to +1.0 (excited/high energy)
    Dominance:  0.0 (submissive/powerless) to +1.0 (dominant/in control)
    """
    valence: float = Field(ge=-1.0, le=1.0)
    arousal: float = Field(ge=0.0, le=1.0)
    dominance: float = Field(ge=0.0, le=1.0)


class EmotionLabel(BaseModel):
    """A single classified emotion with confidence and intensity."""
    emotion: str = Field(description=f"One of: {', '.join(ALL_EMOTION_LABELS)}")
    confidence: float = Field(ge=0.0, le=1.0, description="Classifier confidence in this label")
    intensity: float = Field(ge=0.0, le=1.0, description="Perceived intensity of this emotion in the text")
    is_primary: bool = Field(description="True if this is the dominant emotion, False if secondary")


class Stage2Output(BaseModel):
    """
    Output of the Emotion Classifier (Stage 2).
    
    Structured emotion classification consumed by Stage 3 (Self-Simulation Core).
    """

    # ---- Primary classification ----
    primary_emotion: str = Field(
        description="The single most prominent emotion label from Plutchik taxonomy"
    )
    primary_intensity: float = Field(
        ge=0.0, le=1.0,
        description="Intensity of the primary emotion (0 = trace, 1 = overwhelming)"
    )
    primary_confidence: float = Field(
        ge=0.0, le=1.0,
        description="Classifier confidence in the primary emotion label"
    )

    # ---- Secondary / blended emotions ----
    secondary_emotions: list[EmotionLabel] = Field(
        default_factory=list,
        description="Other emotions present alongside the primary (e.g., anger mixed with sadness)"
    )

    # ---- Dimensional model ----
    vad: VADScores = Field(description="Valence-Arousal-Dominance dimensional scores")

    # ---- Contextual metadata ----
    context_type: str = Field(
        description=(
            "Refined context label from Stage 1: "
            "'personal_distress', 'interpersonal_conflict', 'seeking_support', "
            "'sharing_positive_news', 'venting', 'neutral_information', "
            "'crisis', 'ambiguous', or 'other'"
        )
    )
    emotion_regulation_state: str = Field(
        description=(
            "Gross Emotion Regulation inference: "
            "'unregulated' (raw expression), "
            "'suppressing' (hiding emotion), "
            "'reappraising' (reframing), "
            "'seeking_regulation' (asking for help managing), "
            "'regulated' (emotion is controlled)"
        )
    )
    bdi_sketch: dict = Field(
        description=(
            "Brief Belief-Desire-Intention sketch inferred from text: "
            "{'belief': str, 'desire': str, 'intention': str}"
        )
    )

    # ---- Meta ----
    is_ambiguous: bool = Field(
        description="True if classification is uncertain due to mixed or subtle signals"
    )
    classifier_notes: Optional[str] = Field(
        default=None,
        description="Optional free-form classifier reasoning for ambiguous cases"
    )


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

def _build_system_prompt() -> str:
    schema = json.dumps(Stage2Output.model_json_schema(), indent=2)
    return f"""You are an expert affective computing classifier trained in \
Plutchik's Wheel of Emotions and the dimensional VAD (Valence-Arousal-Dominance) model.

You will receive a structured emotional signal report (from a prior analysis stage) \
and must classify the emotions it describes.

CLASSIFICATION FRAMEWORK:

1. PLUTCHIK'S TAXONOMY
   Primary emotions (use one): {", ".join(PLUTCHIK_PRIMARIES)}
   Dyadic blends (use when two primaries co-occur): {", ".join(PLUTCHIK_DYADS.keys())}
   Also valid: 'neutral', 'mixed'

2. INTENSITY SCALE
   0.0 = trace presence   0.25 = mild   0.5 = moderate
   0.75 = strong          1.0 = overwhelming / flooded

3. VAD DIMENSIONS
   Valence:    -1.0 (very negative) → +1.0 (very positive)
   Arousal:     0.0 (calm)         → +1.0 (highly activated)
   Dominance:   0.0 (powerless)    → +1.0 (in control)

4. EMOTION REGULATION (Gross, 2015)
   Identify whether the speaker appears to be:
   - unregulated: expressing raw emotion directly
   - suppressing: hiding or minimising the felt emotion
   - reappraising: cognitively reframing the situation
   - seeking_regulation: asking others to help manage the emotion
   - regulated: emotion is present but well-managed

5. BDI SKETCH (Bratman, 1987)
   From the text infer a brief sketch:
   - belief: what does the speaker believe about their situation?
   - desire: what do they want?
   - intention: what are they trying to do right now?

You MUST respond ONLY with a valid JSON object that exactly matches this schema \
(use the exact field names shown — no substitutions):

{schema}

No preamble, no explanation outside the JSON."""


SYSTEM_PROMPT = _build_system_prompt()


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


class EmotionClassifier:
    """
    Stage 2: Emotion Classifier.
    
    Consumes Stage 1 signals and produces a rich, structured classification
    using Plutchik's wheel, VAD scores, and Gross/BDI contextual models.
    """

    def __init__(self) -> None:
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.model = HAIKU_MODEL

    def _build_messages(self, stage1: Stage1Output) -> list[dict]:
        """Serialize Stage 1 output as input to Stage 2."""
        signal_summary = stage1.model_dump_json(indent=2)
        return [
            {
                "role": "user",
                "content": (
                    f"Classify the emotions described in this signal report.\n\n"
                    f"SIGNAL REPORT (Stage 1 output):\n{signal_summary}\n\n"
                    f"Return a JSON object matching the Stage2Output schema."
                ),
            }
        ]

    def _build_system(self) -> list[dict] | str:
        if USE_PROMPT_CACHING:
            return [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]
        return SYSTEM_PROMPT

    def run(self, stage1_output: Stage1Output) -> Stage2Output:
        """
        Run Stage 2 on a Stage1Output.

        Args:
            stage1_output: Output from EmotionRecognizer.

        Returns:
            Stage2Output with full emotion classification.
        """
        logger.debug(
            "Stage 2 | primary_hint=%.2f context=%s",
            stage1_output.overall_intensity_hint,
            stage1_output.context_type,
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS["stage2"],
            temperature=TEMPERATURE["stage2"],
            system=self._build_system(),
            messages=self._build_messages(stage1_output),
        )

        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        try:
            data = json.loads(raw)
            output = Stage2Output(**data)
        except Exception as exc:
            logger.error("Stage 2 JSON parse failed: %s\nRaw: %s", exc, raw[:500])
            raise ValueError(f"Stage 2 failed to parse model output: {exc}") from exc

        logger.debug(
            "Stage 2 | emotion=%s intensity=%.2f vad=(v=%.2f a=%.2f d=%.2f)",
            output.primary_emotion,
            output.primary_intensity,
            output.vad.valence,
            output.vad.arousal,
            output.vad.dominance,
        )
        return output
