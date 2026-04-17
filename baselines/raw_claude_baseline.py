"""
SERAPH Baseline — Raw Claude (No Pipeline)
============================================
Single-prompt baseline: the same Sonnet model used in SERAPH stages 3-5,
but with NO pipeline stages — just a direct empathic response prompt.

This is the most important baseline. It answers the question:
"Does the SERAPH pipeline actually add value over just prompting Claude
to respond empathically?"

The baseline uses a well-engineered single prompt (not a naive one) to
ensure a fair comparison — we are testing pipeline architecture value,
not prompt quality.

Same input/output interface as SERAPHPipeline in main.py for fair
benchmark integration.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import anthropic

from config import ANTHROPIC_API_KEY, SONNET_MODEL

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Baseline system prompt
# ---------------------------------------------------------------------------
# This is intentionally strong — a well-written empathy prompt — to ensure
# that any performance gap reflects architectural advantage, not prompt sloppiness.

RAW_CLAUDE_SYSTEM = """You are a highly empathic conversational AI assistant.

When someone shares something with you, respond with genuine empathy and care.

Guidelines:
- First, understand what the person is feeling before responding
- Acknowledge their emotion before offering any advice or information
- Match your tone to the emotional intensity of the message
- Avoid minimising, toxic positivity, or jumping straight to solutions
- Ask yourself what this person most needs right now — and lead with that
- Keep responses appropriately sized to the emotional content
- If someone appears to be in crisis, prioritise their safety above all else

Respond as a warm, thoughtful human would — not as a generic AI assistant."""


@dataclass
class RawClaudeResult:
    """Result from the raw Claude baseline — mirrors PipelineResult interface."""
    input_text: str
    final_response: str = ""
    error: Optional[str] = None
    latency_ms: dict = field(default_factory=dict)
    ablation_variant: str = "raw_claude"

    # Stub fields to match PipelineResult interface used by benchmark evaluators
    stage1: None = None
    stage2: None = None
    stage3: None = None
    stage4: None = None
    stage5: None = None

    @property
    def success(self) -> bool:
        return self.error is None and bool(self.final_response)

    def to_dict(self) -> dict:
        return {
            "input_text":     self.input_text,
            "final_response": self.final_response,
            "ablation_variant": self.ablation_variant,
            "success":        self.success,
            "error":          self.error,
            "latency_ms":     self.latency_ms,
        }


class RawClaudeBaseline:
    """
    Baseline: single Sonnet prompt, no pipeline stages.

    Designed to be a drop-in replacement for SERAPHPipeline in benchmark
    evaluators so all baselines run through the same evaluation harness.
    """

    def __init__(self) -> None:
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.model  = SONNET_MODEL

    def run(self, text: str) -> RawClaudeResult:
        result = RawClaudeResult(input_text=text)
        t0 = time.perf_counter() * 1000

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                temperature=0.6,
                system=RAW_CLAUDE_SYSTEM,
                messages=[{"role": "user", "content": text}],
            )
            result.final_response = response.content[0].text.strip()
            result.latency_ms["total"] = time.perf_counter() * 1000 - t0
            logger.debug("Raw Claude baseline | len=%d", len(result.final_response))
        except Exception as exc:
            result.error = str(exc)
            logger.error("Raw Claude baseline error: %s", exc)

        return result


# ---------------------------------------------------------------------------
# Benchmark runner (mirrors benchmark evaluator interface)
# ---------------------------------------------------------------------------

def run_raw_claude_eval(
    dataset: str,
    sample_limit: Optional[int] = None,
    output_dir: str = "results",
) -> dict:
    """
    Run the raw Claude baseline on a dataset.

    Args:
        dataset: One of 'iemocap', 'meld', 'empathetic_dialogues'
        sample_limit: Max samples
        output_dir: Results directory

    Returns:
        BenchmarkMetrics dict
    """
    import random
    from pathlib import Path
    from metrics.empathy_scorer import EmpathyScorer, BenchmarkMetrics

    # Reuse dataset loaders from benchmark evaluators
    if dataset == "iemocap":
        from benchmarks.iemocap_eval import load_iemocap, IEMOCAP_TO_PLUTCHIK as label_map
        samples = load_iemocap(sample_limit)
    elif dataset == "meld":
        from benchmarks.meld_eval import load_meld, MELD_TO_PLUTCHIK as label_map
        samples = load_meld(sample_limit)
    elif dataset == "empathetic_dialogues":
        from benchmarks.empathetic_dialogues_eval import load_ed, ED_TO_PLUTCHIK as label_map
        samples = load_ed(sample_limit)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    if not samples:
        logger.error("No samples loaded for dataset: %s", dataset)
        return {}

    baseline = RawClaudeBaseline()
    scorer   = EmpathyScorer()

    true_labels, empathy_ratings, results = [], [], []

    for i, sample in enumerate(samples):
        utterance  = sample["utterance"]
        gt_emotion = label_map.get(sample["emotion"], "neutral")
        true_labels.append(gt_emotion)

        logger.info("Raw Claude [%d/%d] dataset=%s", i + 1, len(samples), dataset)

        result = baseline.run(utterance)

        if result.final_response:
            rating = scorer.score_response(utterance, result.final_response, gt_emotion)
            empathy_ratings.append(rating)

        results.append({
            "utterance":    utterance,
            "gt_emotion":   gt_emotion,
            "response":     result.final_response,
            "success":      result.success,
            "error":        result.error,
        })

    empathy_agg = scorer.aggregate_ratings(empathy_ratings)

    # Note: raw Claude has no emotion classification output,
    # so weighted_f1 is 0.0 for this baseline (classification not attempted).
    metrics = BenchmarkMetrics(
        dataset=dataset,
        ablation_variant="raw_claude",
        n_samples=len(samples),
        weighted_f1=0.0,
        macro_f1=0.0,
        **empathy_agg,
    )

    out_dir  = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset}_raw_claude.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"metrics": metrics.to_dict(), "results": results}, f, indent=2)

    logger.info(
        "Raw Claude | dataset=%s | empathy=%.4f | saved to %s",
        dataset, metrics.mean_empathy_score, out_path,
    )
    return metrics.to_dict()
