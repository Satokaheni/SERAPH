"""
SERAPH Baseline Runner
=======================
Runs all baseline systems across all specified datasets.

Baselines:
  - raw_claude    — single-prompt Sonnet, no pipeline (most important baseline)
  - dialogue_rnn  — NRC lexicon + logistic regression (free, no API calls)
  - cem           — CEM commonsense-aware baseline (LLM approximation)
  - mime          — MIME emotion-mirroring baseline (LLM approximation)

Usage:
    # Run all baselines on all datasets
    python run_baselines.py

    # Run specific baseline
    python run_baselines.py --baselines raw_claude dialogue_rnn

    # Run on specific datasets only
    python run_baselines.py --datasets meld empathetic_dialogues

    # Run with sample limit (for testing)
    python run_baselines.py --sample-limit 10
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
import logging
import traceback
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

AVAILABLE_BASELINES = {
    "raw_claude":   "Single-prompt Sonnet — no pipeline (key baseline)",
    "dialogue_rnn": "NRC lexicon + LogReg classifier (free, no API calls)",
    "cem":          "CEM commonsense-aware baseline (LLM approximation)",
    "mime":         "MIME emotion-mirroring baseline (LLM approximation)",
}

AVAILABLE_DATASETS = ["meld", "empathetic_dialogues", "iemocap"]


def run_baselines(
    baselines: list[str] | None = None,
    datasets: list[str] | None = None,
    sample_limit: Optional[int] = None,
    output_dir: str = "results",
) -> dict:
    """
    Run all specified baselines across all specified datasets.

    Args:
        baselines: List of baseline names. Defaults to all four.
        datasets: List of dataset names. Defaults to meld + empathetic_dialogues.
        sample_limit: Max samples per dataset per baseline.
        output_dir: Directory to save result JSON files.

    Returns:
        Nested dict: {baseline: {dataset: metrics_dict}}
    """
    if baselines is None:
        baselines = list(AVAILABLE_BASELINES.keys())
    if datasets is None:
        datasets = ["meld", "empathetic_dialogues"]

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict = {}

    for baseline in baselines:
        if baseline not in AVAILABLE_BASELINES:
            logger.warning("Unknown baseline: %s — skipping", baseline)
            continue

        logger.info(
            "\n%s\nBaseline: %s — %s\n%s",
            "=" * 60, baseline, AVAILABLE_BASELINES[baseline], "=" * 60
        )
        all_results[baseline] = {}

        for dataset in datasets:
            if dataset not in AVAILABLE_DATASETS:
                logger.warning("Unknown dataset: %s — skipping", dataset)
                continue

            logger.info("Running %s on %s…", baseline, dataset)
            try:
                result = _run_single(baseline, dataset, sample_limit, str(out_dir))
                all_results[baseline][dataset] = result
                logger.info(
                    "%s | %s | empathy=%.4f | weighted_f1=%.4f",
                    baseline, dataset,
                    result.get("mean_empathy_score", 0),
                    result.get("weighted_f1", 0),
                )
            except Exception as exc:
                logger.error("Failed %s/%s: %s", baseline, dataset, exc)
                logger.error("Full traceback: %s", traceback.format_exc())
                all_results[baseline][dataset] = {"error": str(exc)}

    _print_summary(all_results)
    return all_results


def _run_single(
    baseline: str,
    dataset: str,
    sample_limit: Optional[int],
    output_dir: str,
) -> dict:
    """Run a single baseline on a single dataset."""

    if baseline == "raw_claude":
        from baselines.raw_claude_baseline import run_raw_claude_eval
        return run_raw_claude_eval(
            dataset=dataset,
            sample_limit=sample_limit,
            output_dir=output_dir,
        )

    elif baseline == "dialogue_rnn":
        from baselines.dialogue_rnn_baseline import run_dialogue_rnn_eval
        return run_dialogue_rnn_eval(
            dataset=dataset,
            sample_limit=sample_limit,
            output_dir=output_dir,
        )

    elif baseline in ("cem", "mime"):
        from baselines.cem_mime_baseline import run_baseline_eval
        return run_baseline_eval(
            baseline_name=baseline,
            dataset=dataset,
            sample_limit=sample_limit,
            output_dir=output_dir,
        )

    else:
        raise ValueError(f"Unknown baseline: {baseline}")


def _print_summary(results: dict) -> None:
    """Print a readable summary table."""
    print("\n" + "=" * 70)
    print("SERAPH BASELINE RESULTS — SUMMARY")
    print("=" * 70)

    for baseline, datasets in results.items():
        print(f"\n  Baseline: {baseline.upper()}")
        print(f"  {'Dataset':<30} {'Empathy':>10} {'Weighted F1':>12}")
        print(f"  {'-' * 55}")
        for dataset, metrics in datasets.items():
            if "error" in metrics:
                print(f"  {dataset:<30} ERROR: {metrics['error']}")
                continue
            print(
                f"  {dataset:<30} "
                f"{metrics.get('mean_empathy_score', 0):>10.4f} "
                f"{metrics.get('weighted_f1', 0):>12.4f}"
            )

    print("\n" + "=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="SERAPH Baseline Runner")
    parser.add_argument(
        "--baselines",
        nargs="+",
        choices=list(AVAILABLE_BASELINES.keys()),
        default=None,
        help="Baselines to run (default: all)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=AVAILABLE_DATASETS,
        default=None,
        help="Datasets to evaluate on (default: meld + empathetic_dialogues)",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Max samples per dataset per baseline",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Output directory for result JSON files",
    )

    args = parser.parse_args()

    run_baselines(
        baselines=args.baselines,
        datasets=args.datasets,
        sample_limit=args.sample_limit,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        level=logging.INFO,
    )
    main()
