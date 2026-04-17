"""
SERAPH Ablation Study Runner
==============================
Runs all 6 ablation variants across all 3 datasets and produces
a consolidated comparison table.

Ablation Variants:
  1. full            — Full pipeline (all 5 stages)
  2. no_stage3       — Remove Stage 3 (no self-simulation) ← KEY comparison
  3. no_stage4       — Remove Stage 4 (no ethical gate)
  4. merged_1_2      — Merge Stages 1+2 into single classification
  5. stage3_only     — Self-simulation only (skip Stages 1+2)
  6. random_emotion  — Random emotion label at Stage 2

Usage:
    python -m ablations.run_ablations
    python -m ablations.run_ablations --dataset empathetic_dialogues --sample-limit 50
    python -m ablations.run_ablations --variants full --datasets meld
"""

from __future__ import annotations

import argparse
import json
import logging
import traceback
from pathlib import Path
from typing import Optional

from config import ABLATION_VARIANTS

logger = logging.getLogger(__name__)


def run_all_ablations(
    datasets: list[str] | None = None,
    sample_limit: Optional[int] = None,
    output_dir: str = "results/ablations",
    variants: list[str] | None = None,
) -> dict:
    """
    Run ablation variants across all specified datasets.

    Args:
        datasets: List of dataset names. Defaults to all three.
        sample_limit: Max samples per dataset per variant.
                      Recommend 200–500 for ablation speed.
        output_dir: Directory to save per-variant result JSON files.
        variants: Specific variant keys to run. Defaults to all six.

    Returns:
        Nested dict: {dataset: {variant: BenchmarkMetrics.to_dict()}}
    """
    from benchmarks.iemocap_eval import run_iemocap_eval
    from benchmarks.meld_eval import run_meld_eval
    from benchmarks.empathetic_dialogues_eval import run_ed_eval

    if datasets is None:
        datasets = ["iemocap", "meld", "empathetic_dialogues"]

    eval_fn_map = {
        "iemocap": run_iemocap_eval,
        "meld": run_meld_eval,
        "empathetic_dialogues": run_ed_eval,
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict = {ds: {} for ds in datasets}

    # Filter to requested variants only, preserving original order
    active_variants = {
        k: v for k, v in ABLATION_VARIANTS.items()
        if variants is None or k in variants
    }

    logger.info("Running %d variant(s): %s", len(active_variants), list(active_variants.keys()))

    for variant, description in active_variants.items():
        logger.info("\n%s\nAblation: %s — %s\n%s", "=" * 60, variant, description, "=" * 60)

        for dataset in datasets:
            if dataset not in eval_fn_map:
                logger.warning("Unknown dataset: %s — skipping", dataset)
                continue

            logger.info("Running %s on %s…", variant, dataset)
            try:
                metrics = eval_fn_map[dataset](
                    ablation=variant,
                    sample_limit=sample_limit,
                    output_dir=str(out_dir),
                )
                all_results[dataset][variant] = metrics.to_dict()
            except Exception as exc:
                logger.error("Failed %s/%s: %s", variant, dataset, exc)
                logger.error("Full traceback: %s", traceback.format_exc())
                all_results[dataset][variant] = {"error": str(exc)}

    # Save consolidated table
    summary_path = out_dir / "ablation_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    logger.info("Ablation summary saved to %s", summary_path)
    _print_summary_table(all_results)
    return all_results


def _print_summary_table(results: dict) -> None:
    """Print a readable comparison table to stdout."""
    print("\n" + "=" * 80)
    print("SERAPH ABLATION STUDY — SUMMARY TABLE")
    print("=" * 80)

    for dataset, variants in results.items():
        print(f"\n  Dataset: {dataset.upper()}")
        print(f"  {'Variant':<20} {'Weighted F1':>12} {'Empathy Score':>14} {'Tone Match':>11}")
        print(f"  {'-' * 60}")
        for variant, metrics in variants.items():
            if "error" in metrics:
                print(f"  {variant:<20} ERROR: {metrics['error']}")
                continue
            print(
                f"  {variant:<20} "
                f"{metrics.get('weighted_f1', 0):>12.4f} "
                f"{metrics.get('mean_empathy_score', 0):>14.4f} "
                f"{metrics.get('mean_tone_match', 0):>11.4f}"
            )

    print("\n" + "=" * 80)
    print("KEY COMPARISON: 'full' vs 'no_stage3' shows the self-simulation effect.")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="SERAPH Ablation Study Runner")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["iemocap", "meld", "empathetic_dialogues"],
        default=None,
        help="Datasets to evaluate on (default: all)",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Max samples per dataset per variant",
    )
    parser.add_argument(
        "--output-dir",
        default="results/ablations",
        help="Output directory for results",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=list(ABLATION_VARIANTS.keys()),
        default=None,
        help="Specific variants to run (default: all)",
    )

    args = parser.parse_args()

    run_all_ablations(
        datasets=args.datasets,
        sample_limit=args.sample_limit,
        output_dir=args.output_dir,
        variants=args.variants,
    )


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        level=logging.INFO,
    )
    main()
