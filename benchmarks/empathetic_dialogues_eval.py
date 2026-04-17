"""
SERAPH Benchmark — EmpatheticDialogues Evaluation
====================================================
Evaluates SERAPH on EmpatheticDialogues (Rashkin et al., 2019).

Execution modes:
    Sequential (USE_BATCH_API=False): ~9 hrs for full valid set
    Batch API  (USE_BATCH_API=True):  ~1-2 hrs for full valid set
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

ED_TO_PLUTCHIK = {
    "afraid":       "fear",    "angry":        "anger",
    "annoyed":      "anger",   "anticipating": "anticipation",
    "anxious":      "fear",    "apprehensive": "fear",
    "ashamed":      "sadness", "caring":       "trust",
    "confident":    "joy",     "content":      "joy",
    "devastated":   "sadness", "disappointed": "sadness",
    "disgusted":    "disgust", "embarrassed":  "sadness",
    "excited":      "joy",     "faithful":     "trust",
    "furious":      "anger",   "grateful":     "joy",
    "guilty":       "sadness", "hopeful":      "anticipation",
    "impressed":    "surprise","jealous":      "anger",
    "joyful":       "joy",     "lonely":       "sadness",
    "nostalgic":    "sadness", "prepared":     "anticipation",
    "proud":        "joy",     "sad":          "sadness",
    "sentimental":  "sadness", "surprised":    "surprise",
    "terrified":    "fear",    "trusting":     "trust",
}

from paths import PATHS
DATA_PATH = PATHS.ed_valid


def load_ed(sample_limit: Optional[int] = None, seed: int = 42) -> list[dict]:
    if not DATA_PATH.exists():
        logger.warning("EmpatheticDialogues data not found at %s.", DATA_PATH)
        return []
    rows, current_conv = [], {}
    with DATA_PATH.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            conv_id  = row.get("conv_id", "")
            utt_idx  = int(row.get("utterance_idx", 0))
            utt      = row.get("utterance", "").replace("_comma_", ",")
            context  = row.get("context", "neutral").lower()
            if utt_idx == 1:
                current_conv[conv_id] = {"situation": utt, "emotion": context, "responses": []}
            elif conv_id in current_conv:
                current_conv[conv_id]["responses"].append(utt)
    for conv in current_conv.values():
        if conv["responses"]:
            rows.append({"utterance": conv["situation"], "emotion": conv["emotion"],
                         "human_response": conv["responses"][0]})
    if sample_limit:
        random.seed(seed)
        rows = random.sample(rows, min(sample_limit, len(rows)))
    logger.info("Loaded %d EmpatheticDialogues samples", len(rows))
    return rows


def run_ed_eval(
    ablation: str = "full",
    sample_limit: Optional[int] = None,
    output_dir: str = "results",
) -> BenchmarkMetrics:
    samples = load_ed(sample_limit)
    if not samples:
        return BenchmarkMetrics("empathetic_dialogues", ablation, 0)

    scorer  = EmpathyScorer()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if USE_BATCH_API:
        true_labels, pred_labels, empathy_ratings, results = _run_batch(samples, ablation, out_dir)
    else:
        true_labels, pred_labels, empathy_ratings, results = _run_sequential(samples, ablation, scorer)

    # Score human reference responses as ceiling baseline
    human_ratings = []
    for r in results:
        hr = r.get("human_response", "")
        if hr:
            human_ratings.append(scorer.score_response(r["utterance"], hr, r["gt_emotion"]))

    f1_scores   = scorer.compute_f1(true_labels, pred_labels)
    empathy_agg = scorer.aggregate_ratings(empathy_ratings)
    human_agg   = scorer.aggregate_ratings(human_ratings)

    metrics = BenchmarkMetrics(
        dataset="empathetic_dialogues",
        ablation_variant=ablation,
        n_samples=len(samples),
        weighted_f1=f1_scores["weighted_f1"],
        macro_f1=f1_scores["macro_f1"],
        per_class_f1=f1_scores["per_class_f1"],
        **empathy_agg,
    )

    out_path = out_dir / f"empathetic_dialogues_{ablation}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"metrics": metrics.to_dict(), "human_baseline": human_agg,
                   "results": results}, f, indent=2)

    logger.info("ED | ablation=%s | seraph_empathy=%.4f | human_empathy=%.4f",
                ablation, metrics.mean_empathy_score,
                human_agg.get("mean_empathy_score", 0))
    return metrics


def _run_batch(samples, ablation, out_dir):
    from benchmarks.batch_runner import run_batch_pipeline, assemble_results
    checkpoint_dir = out_dir / f"checkpoints_ed_{ablation}"
    logger.info("ED | BATCH MODE | %d samples | ablation=%s", len(samples), ablation)
    states = run_batch_pipeline(samples, ED_TO_PLUTCHIK, ablation, checkpoint_dir)
    true_labels, pred_labels, empathy_ratings, results = assemble_results(states, EmpathyScorer())
    # Reattach human responses from original samples
    sample_map = {s["utterance"]: s.get("human_response", "") for s in samples}
    for r in results:
        r["human_response"] = sample_map.get(r["utterance"], "")
    return true_labels, pred_labels, empathy_ratings, results


def _run_sequential(samples, ablation, scorer):
    from main import SERAPHPipeline
    pipeline = SERAPHPipeline(ablation=ablation)
    true_labels, pred_labels, empathy_ratings, results = [], [], [], []
    for i, sample in enumerate(samples):
        utterance      = sample["utterance"]
        gt_emotion     = ED_TO_PLUTCHIK.get(sample["emotion"], "neutral")
        human_response = sample.get("human_response", "")
        logger.info("ED sequential [%d/%d]", i + 1, len(samples))
        try:
            result       = pipeline.run(utterance)
            pred_emotion = result.stage2.primary_emotion if result.stage2 else "neutral"
            true_labels.append(gt_emotion)
            pred_labels.append(pred_emotion)
            if result.final_response:
                empathy_ratings.append(scorer.score_response(utterance, result.final_response, gt_emotion))
            results.append({"utterance": utterance, "gt_emotion": gt_emotion,
                            "pred_emotion": pred_emotion, "response": result.final_response,
                            "human_response": human_response, "success": result.success,
                            "error": result.error})
        except Exception as exc:
            logger.error("ED error on sample %d: %s", i, exc)
            true_labels.append(gt_emotion)
            pred_labels.append("neutral")
    return true_labels, pred_labels, empathy_ratings, results
