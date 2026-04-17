"""
SERAPH Benchmark — MELD Evaluation
======================================
Evaluates SERAPH on MELD (Multimodal EmotionLines Dataset).

Execution modes:
    Sequential (USE_BATCH_API=False): ~8 hrs for full test set
    Batch API  (USE_BATCH_API=True):  ~1-2 hrs for full test set
"""

from __future__ import annotations

import csv
import json
import logging
import random
from pathlib import Path
from typing import Optional

from config import USE_BATCH_API
from metrics.empathy_scorer import EmpathyScorer, BenchmarkMetrics

logger = logging.getLogger(__name__)

MELD_TO_PLUTCHIK = {
    "anger":    "anger",
    "disgust":  "disgust",
    "fear":     "fear",
    "joy":      "joy",
    "neutral":  "neutral",
    "sadness":  "sadness",
    "surprise": "surprise",
}

from paths import PATHS
DATA_PATH = PATHS.meld_test


def load_meld(sample_limit: Optional[int] = None, seed: int = 42) -> list[dict]:
    if not DATA_PATH.exists():
        logger.warning("MELD data not found at %s.", DATA_PATH)
        return []
    rows = []
    with DATA_PATH.open(encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "utterance":   row.get("Utterance", ""),
                "emotion":     row.get("Emotion", "neutral").lower(),
                "speaker":     row.get("Speaker", ""),
                "dialogue_id": row.get("Dialogue_ID", ""),
            })
    if sample_limit:
        random.seed(seed)
        rows = random.sample(rows, min(sample_limit, len(rows)))
    logger.info("Loaded %d MELD samples", len(rows))
    return rows


def run_meld_eval(
    ablation: str = "full",
    sample_limit: Optional[int] = None,
    output_dir: str = "results",
) -> BenchmarkMetrics:
    samples = load_meld(sample_limit)
    if not samples:
        return BenchmarkMetrics("meld", ablation, 0)

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
        dataset="meld",
        ablation_variant=ablation,
        n_samples=len(samples),
        weighted_f1=f1_scores["weighted_f1"],
        macro_f1=f1_scores["macro_f1"],
        per_class_f1=f1_scores["per_class_f1"],
        **empathy_agg,
    )

    out_path = out_dir / f"meld_{ablation}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"metrics": metrics.to_dict(), "results": results}, f, indent=2)

    logger.info("MELD | ablation=%s | weighted_f1=%.4f | empathy=%.4f",
                ablation, metrics.weighted_f1, metrics.mean_empathy_score)
    return metrics


def _run_batch(samples, ablation, out_dir):
    from benchmarks.batch_runner import run_batch_pipeline, assemble_results
    checkpoint_dir = out_dir / f"checkpoints_meld_{ablation}"
    logger.info("MELD | BATCH MODE | %d samples | ablation=%s", len(samples), ablation)
    states = run_batch_pipeline(samples, MELD_TO_PLUTCHIK, ablation, checkpoint_dir)
    return assemble_results(states, EmpathyScorer())


def _run_sequential(samples, ablation, scorer):
    from main import SERAPHPipeline
    pipeline = SERAPHPipeline(ablation=ablation)
    true_labels, pred_labels, empathy_ratings, results = [], [], [], []
    for i, sample in enumerate(samples):
        utterance  = sample["utterance"]
        gt_emotion = MELD_TO_PLUTCHIK.get(sample["emotion"], "neutral")
        logger.info("MELD sequential [%d/%d]", i + 1, len(samples))
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
            logger.error("MELD error on sample %d: %s", i, exc)
            true_labels.append(gt_emotion)
            pred_labels.append("neutral")
    return true_labels, pred_labels, empathy_ratings, results
