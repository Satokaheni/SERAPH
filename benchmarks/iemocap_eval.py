"""
SERAPH Benchmark — IEMOCAP Evaluation
========================================
Evaluates SERAPH on the IEMOCAP (Interactive Emotional Dyadic Motion Capture)
dataset — a widely-used benchmark for conversational emotion recognition.

Dataset: https://sail.usc.edu/iemocap/
Labels: anger, happiness, sadness, neutral, frustrated, excited, fearful, surprised, disgusted
We map to Plutchik primaries for compatibility.

Expected data format (after download and preprocessing):
    data/iemocap/iemocap_processed.json

Execution modes:
    Sequential (USE_BATCH_API=False in config.py): ~23 hrs for full dataset
    Batch API  (USE_BATCH_API=True  in config.py): ~2-4 hrs for full dataset
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Optional

from config import USE_BATCH_API
from metrics.empathy_scorer import EmpathyScorer, BenchmarkMetrics

logger = logging.getLogger(__name__)

IEMOCAP_TO_PLUTCHIK = {
    "anger":      "anger",
    "happiness":  "joy",
    "excited":    "joy",
    "sadness":    "sadness",
    "neutral":    "neutral",
    "frustrated": "anger",
    "fearful":    "fear",
    "surprised":  "surprise",
    "disgusted":  "disgust",
}

from paths import PATHS
DATA_PATH = PATHS.iemocap_data


def load_iemocap(sample_limit: Optional[int] = None, seed: int = 42) -> list[dict]:
    if not DATA_PATH.exists():
        logger.warning("IEMOCAP data not found at %s.", DATA_PATH)
        return []
    with DATA_PATH.open(encoding="utf-8") as f:
        data = json.load(f)
    if sample_limit:
        random.seed(seed)
        data = random.sample(data, min(sample_limit, len(data)))
    logger.info("Loaded %d IEMOCAP samples", len(data))
    return data


def run_iemocap_eval(
    ablation: str = "full",
    sample_limit: Optional[int] = None,
    output_dir: str = "results",
) -> BenchmarkMetrics:
    samples = load_iemocap(sample_limit)
    if not samples:
        logger.error("No IEMOCAP samples loaded — aborting eval.")
        return BenchmarkMetrics("iemocap", ablation, 0)

    scorer  = EmpathyScorer()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if USE_BATCH_API:
        true_labels, pred_labels, empathy_ratings, results = _run_batch(samples, ablation, out_dir)
    else:
        true_labels, pred_labels, empathy_ratings, results = _run_sequential(samples, ablation, scorer)

    f1_scores   = scorer.compute_f1(true_labels, pred_labels)
    empathy_agg = scorer.aggregate_ratings(empathy_ratings)

    metrics = BenchmarkMetrics(
        dataset="iemocap",
        ablation_variant=ablation,
        n_samples=len(samples),
        weighted_f1=f1_scores["weighted_f1"],
        macro_f1=f1_scores["macro_f1"],
        per_class_f1=f1_scores["per_class_f1"],
        **empathy_agg,
    )

    out_path = out_dir / f"iemocap_{ablation}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"metrics": metrics.to_dict(), "results": results}, f, indent=2)

    logger.info("IEMOCAP | ablation=%s | weighted_f1=%.4f | empathy=%.4f",
                ablation, metrics.weighted_f1, metrics.mean_empathy_score)
    return metrics


def _run_batch(samples, ablation, out_dir):
    from benchmarks.batch_runner import run_batch_pipeline, assemble_results
    checkpoint_dir = out_dir / f"checkpoints_iemocap_{ablation}"
    logger.info("IEMOCAP | BATCH MODE | %d samples | ablation=%s", len(samples), ablation)
    states = run_batch_pipeline(samples, IEMOCAP_TO_PLUTCHIK, ablation, checkpoint_dir)
    return assemble_results(states, EmpathyScorer())


def _run_sequential(samples, ablation, scorer):
    from main import SERAPHPipeline
    pipeline = SERAPHPipeline(ablation=ablation)
    true_labels, pred_labels, empathy_ratings, results = [], [], [], []
    for i, sample in enumerate(samples):
        utterance  = sample["utterance"]
        gt_emotion = IEMOCAP_TO_PLUTCHIK.get(sample["emotion"], "neutral")
        logger.info("IEMOCAP sequential [%d/%d]", i + 1, len(samples))
        try:
            result       = pipeline.run(utterance)
            pred_emotion = result.stage2.primary_emotion if result.stage2 else "neutral"
            true_labels.append(gt_emotion)
            pred_labels.append(pred_emotion)
            if result.final_response:
                empathy_ratings.append(scorer.score_response(utterance, result.final_response, gt_emotion))
            results.append({"utterance": utterance, "gt_emotion": gt_emotion,
                            "pred_emotion": pred_emotion, "response": result.final_response,
                            "success": result.success, "error": result.error})
        except Exception as exc:
            logger.error("Error on sample %d: %s", i, exc)
            true_labels.append(gt_emotion)
            pred_labels.append("neutral")
    return true_labels, pred_labels, empathy_ratings, results
