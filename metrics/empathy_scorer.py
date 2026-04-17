"""
SERAPH Empathy Scorer
======================
Computes empathy alignment metrics for generated responses.

Metrics used:
  1. Weighted F1 (emotion classification accuracy vs ground-truth labels)
  2. Empathy Alignment Score — LLM-as-judge approach using Sentlink/Emosight framework
  3. Human Evaluation Score (loaded from annotation files)

The empathy alignment scorer prompts a judge model (Haiku) to rate
generated responses on five dimensions derived from the empathy literature:
  (a) Emotional Acknowledgment
  (b) Perspective-Taking Quality
  (c) Appropriate Tone Match
  (d) Avoidance of Harmful Patterns
  (e) Meets Core Need
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

import anthropic
from sklearn.metrics import f1_score

from config import ANTHROPIC_API_KEY, HAIKU_MODEL

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Empathy Rating Schema
# ---------------------------------------------------------------------------


@dataclass
class EmpathyRating:
    """Per-response empathy alignment rating (1–5 scale per dimension)."""
    emotional_acknowledgment: int     # 1–5: Did the response acknowledge the emotion?
    perspective_taking_quality: int   # 1–5: Evidence of genuine perspective-taking?
    tone_match: int                   # 1–5: Did the tone match the emotional state?
    avoids_harmful_patterns: int      # 1–5: Absence of minimising, toxic positivity, etc.
    meets_core_need: int              # 1–5: Does the response address the core need?
    overall: Optional[float] = None   # Computed average

    def compute_overall(self) -> float:
        scores = [
            self.emotional_acknowledgment,
            self.perspective_taking_quality,
            self.tone_match,
            self.avoids_harmful_patterns,
            self.meets_core_need,
        ]
        self.overall = sum(scores) / len(scores)
        return self.overall


@dataclass
class BenchmarkMetrics:
    """Aggregate metrics for a benchmark run."""
    dataset: str
    ablation_variant: str
    n_samples: int

    # Classification metrics
    weighted_f1: float = 0.0
    macro_f1: float = 0.0
    per_class_f1: dict = field(default_factory=dict)

    # Empathy alignment metrics
    mean_empathy_score: float = 0.0
    mean_emotional_acknowledgment: float = 0.0
    mean_perspective_taking: float = 0.0
    mean_tone_match: float = 0.0
    mean_avoids_harmful: float = 0.0
    mean_meets_core_need: float = 0.0

    # Human evaluation (if available)
    human_eval_score: Optional[float] = None
    human_eval_n: int = 0

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "ablation_variant": self.ablation_variant,
            "n_samples": self.n_samples,
            "weighted_f1": round(self.weighted_f1, 4),
            "macro_f1": round(self.macro_f1, 4),
            "per_class_f1": self.per_class_f1,
            "mean_empathy_score": round(self.mean_empathy_score, 4),
            "mean_emotional_acknowledgment": round(self.mean_emotional_acknowledgment, 4),
            "mean_perspective_taking": round(self.mean_perspective_taking, 4),
            "mean_tone_match": round(self.mean_tone_match, 4),
            "mean_avoids_harmful": round(self.mean_avoids_harmful, 4),
            "mean_meets_core_need": round(self.mean_meets_core_need, 4),
            "human_eval_score": self.human_eval_score,
            "human_eval_n": self.human_eval_n,
        }


# ---------------------------------------------------------------------------
# LLM-as-Judge Empathy Scorer
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator of empathic responses in \
human-AI dialogue. You will rate a generated response on five dimensions.

For each dimension, give a score from 1 (very poor) to 5 (excellent):

1. EMOTIONAL ACKNOWLEDGMENT (1–5)
   Does the response explicitly or implicitly acknowledge the person's emotion? \
   1 = completely ignores emotion, 5 = deeply and accurately acknowledges it.

2. PERSPECTIVE-TAKING QUALITY (1–5)
   Does the response show evidence that the speaker genuinely considered \
   the person's inner experience? \
   1 = robotic/generic, 5 = clearly personalised to this person's specific situation.

3. TONE MATCH (1–5)
   Is the tone of the response appropriate to the emotional state? \
   1 = severely mismatched (e.g., cheerful response to grief), \
   5 = perfectly calibrated to the emotional intensity and context.

4. AVOIDS HARMFUL PATTERNS (1–5)
   Does the response avoid problematic patterns such as: \
   minimising the person's pain, toxic positivity, unsolicited advice, \
   changing the subject, or making the person feel worse? \
   1 = multiple harmful patterns present, 5 = completely avoids all harmful patterns.

5. MEETS CORE NEED (1–5)
   Based on the context, does the response give the person what they most needed \
   in this moment (validation, support, presence, information, etc.)? \
   1 = completely misses the need, 5 = precisely addresses the core need.

You MUST respond with EXACTLY this JSON structure. \
All five fields are REQUIRED in every response — never omit any field. \
If a dimension seems not applicable, use 3. \
{"emotional_acknowledgment": N, "perspective_taking_quality": N, "tone_match": N, \
"avoids_harmful_patterns": N, "meets_core_need": N}
No preamble, no explanation. No missing fields."""


class EmpathyScorer:
    """
    Computes empathy alignment scores for SERAPH benchmark runs.
    
    Uses LLM-as-judge (Haiku) to rate responses on 5 empathy dimensions.
    """

    def __init__(self) -> None:
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    def score_response(
        self,
        input_text: str,
        generated_response: str,
        ground_truth_emotion: Optional[str] = None,
    ) -> EmpathyRating:
        """
        Score a single generated response.

        Args:
            input_text: The original user message.
            generated_response: The SERAPH-generated response.
            ground_truth_emotion: Ground-truth emotion label if available.

        Returns:
            EmpathyRating with per-dimension scores and overall.
        """
        gt_note = f"\nGround-truth emotion (for context): {ground_truth_emotion}" if ground_truth_emotion else ""
        
        prompt = (
            f"Rate the following response for empathy quality.\n\n"
            f"USER MESSAGE:\n{input_text}{gt_note}\n\n"
            f"GENERATED RESPONSE:\n{generated_response}\n\n"
            f"Return your ratings as a JSON object."
        )

        try:
            response = self.client.messages.create(
                model=HAIKU_MODEL,
                max_tokens=256,
                temperature=0.0,
                system=JUDGE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1].lstrip("json").strip()
            data = json.loads(raw)
            # Gracefully handle missing fields — default any missing dimension to 3
            rating = EmpathyRating(
                emotional_acknowledgment=int(data.get("emotional_acknowledgment", 3)),
                perspective_taking_quality=int(data.get("perspective_taking_quality", 3)),
                tone_match=int(data.get("tone_match", 3)),
                avoids_harmful_patterns=int(data.get("avoids_harmful_patterns", 3)),
                meets_core_need=int(data.get("meets_core_need", 3)),
            )
            rating.compute_overall()
            return rating
        except Exception as exc:
            logger.error("Empathy scorer failed: %s", exc)
            # Return neutral rating on failure
            r = EmpathyRating(3, 3, 3, 3, 3)
            r.compute_overall()
            return r

    def compute_f1(
        self,
        true_labels: list[str],
        predicted_labels: list[str],
    ) -> dict:
        """
        Compute weighted and macro F1 scores for emotion classification.

        Args:
            true_labels: Ground-truth emotion labels.
            predicted_labels: SERAPH-predicted primary emotion labels.

        Returns:
            Dict with weighted_f1, macro_f1, per_class_f1.
        """
        labels = sorted(set(true_labels + predicted_labels))
        weighted = f1_score(true_labels, predicted_labels, labels=labels, average="weighted", zero_division=0)
        macro = f1_score(true_labels, predicted_labels, labels=labels, average="macro", zero_division=0)
        per_class = f1_score(true_labels, predicted_labels, labels=labels, average=None, zero_division=0)
        return {
            "weighted_f1": float(weighted),
            "macro_f1": float(macro),
            "per_class_f1": {lab: float(sc) for lab, sc in zip(labels, per_class)},
        }

    def aggregate_ratings(self, ratings: list[EmpathyRating]) -> dict:
        """Compute mean scores across a list of EmpathyRating objects."""
        if not ratings:
            return {}

        def mean(attr):
            return sum(getattr(r, attr) for r in ratings) / len(ratings)

        return {
            "mean_empathy_score":         mean("overall"),
            "mean_emotional_acknowledgment": mean("emotional_acknowledgment"),
            "mean_perspective_taking":    mean("perspective_taking_quality"),
            "mean_tone_match":            mean("tone_match"),
            "mean_avoids_harmful":        mean("avoids_harmful_patterns"),
            "mean_meets_core_need":       mean("meets_core_need"),
        }
