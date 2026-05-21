"""
SERAPH Dataset Preparation
============================
Downloads and preprocesses the benchmark datasets into the format
expected by the SERAPH benchmark evaluators.

Usage:
    python prepare_datasets.py --all
    python prepare_datasets.py --dataset meld
    python prepare_datasets.py --dataset empathetic_dialogues

MELD and EmpatheticDialogues are freely downloadable — this script handles
both automatically.

Output locations:
    data/meld/train_sent_emo.csv
    data/meld/dev_sent_emo.csv
    data/meld/test_sent_emo.csv
    data/empathetic_dialogues/train.csv
    data/empathetic_dialogues/valid.csv
    data/empathetic_dialogues/test.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
import urllib.request
from pathlib import Path
from typing import Optional

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("seraph.data_prep")

from paths import PATHS


# ============================================================
# MELD
# ============================================================

MELD_URLS = {
    "train": "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/train_sent_emo.csv",
    "dev":   "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/dev_sent_emo.csv",
    "test":  "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/test_sent_emo.csv",
}


def prepare_meld() -> bool:
    """
    Download MELD CSV files directly from the declare-lab GitHub repository.

    Returns:
        True if all splits are available after this call.
    """
    out_dir = PATHS.meld_data
    out_dir.mkdir(parents=True, exist_ok=True)
    success = True

    for split, url in MELD_URLS.items():
        out_path = out_dir / f"{split}_sent_emo.csv"
        if out_path.exists():
            logger.info("MELD %s already present at %s — skipping.", split, out_path)
            continue

        logger.info("Downloading MELD %s …", split)
        try:
            urllib.request.urlretrieve(url, out_path)
            with out_path.open(encoding="utf-8-sig") as f:
                n = sum(1 for _ in csv.DictReader(f))
            logger.info("  ✓ MELD %s: %d utterances → %s", split, n, out_path)
        except Exception as exc:
            logger.error("  ✗ Failed to download MELD %s: %s", split, exc)
            success = False

    if success:
        _validate_meld(out_dir / "test_sent_emo.csv")
    return success


def _validate_meld(path: Path) -> None:
    required = {"Utterance", "Emotion", "Dialogue_ID", "Speaker"}
    try:
        with path.open(encoding="utf-8-sig") as f:
            cols = set(csv.DictReader(f).fieldnames or [])
        missing = required - cols
        if missing:
            logger.warning("MELD validation: missing columns %s", missing)
        else:
            logger.info("MELD validation: ✓ all required columns present")
    except Exception as exc:
        logger.warning("MELD validation failed: %s", exc)


# ============================================================
# EmpatheticDialogues
# ============================================================

ED_GITHUB_URLS = {
    "train": "https://raw.githubusercontent.com/facebookresearch/EmpatheticDialogues/master/empatheticdialogues/train.csv",
    "valid": "https://raw.githubusercontent.com/facebookresearch/EmpatheticDialogues/master/empatheticdialogues/valid.csv",
    "test":  "https://raw.githubusercontent.com/facebookresearch/EmpatheticDialogues/master/empatheticdialogues/test.csv",
}

# HuggingFace auto-converts datasets to Parquet — load those directly since
# the legacy loading script (which fetched from ParlAI CDN) is no longer supported.
ED_HF_PARQUET_URLS = {
    "train":      "hf://datasets/facebook/empathetic_dialogues@refs/convert/parquet/default/train/0000.parquet",
    "validation": "hf://datasets/facebook/empathetic_dialogues@refs/convert/parquet/default/validation/0000.parquet",
    "test":       "hf://datasets/facebook/empathetic_dialogues@refs/convert/parquet/default/test/0000.parquet",
}


def prepare_empathetic_dialogues() -> bool:
    """
    Download EmpatheticDialogues via HuggingFace datasets (preferred)
    or fall back to direct ParlAI CDN download.

    Returns:
        True if valid.csv is available after this call.
    """
    out_dir = PATHS.ed_data
    out_dir.mkdir(parents=True, exist_ok=True)

    all_present = all((out_dir / f"{s}.csv").exists() for s in ["train", "valid", "test"])
    if all_present:
        logger.info("EmpatheticDialogues already present — skipping download.")
        return True

    # Try HuggingFace first
    try:
        success = _ed_via_huggingface(out_dir)
        if success:
            return True
    except ImportError:
        logger.warning("`datasets` library not installed — trying GitHub fallback.")
    except Exception as exc:
        logger.warning("HuggingFace download failed (%s) — trying GitHub fallback.", exc)

    # Fallback: GitHub raw files
    return _ed_via_github(out_dir)


def _ed_via_huggingface(out_dir: Path) -> bool:
    from datasets import load_dataset  # type: ignore

    logger.info("Downloading EmpatheticDialogues via HuggingFace datasets …")
    # The facebook/empathetic_dialogues dataset uses a legacy loading script that
    # is no longer supported and fetches from a dead CDN. Load the Parquet files
    # that HuggingFace auto-generates instead.
    ds = load_dataset("parquet", data_files=ED_HF_PARQUET_URLS)

    split_map = {"train": "train", "valid": "validation", "test": "test"}
    fieldnames = ["conv_id", "utterance_idx", "context", "prompt",
                  "speaker_idx", "utterance", "selfeval", "tags"]

    for file_split, hf_split in split_map.items():
        out_path = out_dir / f"{file_split}.csv"
        if out_path.exists():
            continue
        rows = []
        for item in ds[hf_split]:
            rows.append({
                "conv_id":       item.get("conv_id", ""),
                "utterance_idx": item.get("utterance_idx", 0),
                "context":       item.get("context", ""),
                "prompt":        item.get("prompt", ""),
                "speaker_idx":   item.get("speaker_idx", 0),
                "utterance":     item.get("utterance", "").replace("_comma_", ","),
                "selfeval":      item.get("selfeval", ""),
                "tags":          item.get("tags", ""),
            })
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        logger.info("  ✓ ED %s: %d rows → %s", file_split, len(rows), out_path)

    return True


def _ed_via_github(out_dir: Path) -> bool:
    logger.info("Downloading EmpatheticDialogues from GitHub …")
    success = True
    for split, url in ED_GITHUB_URLS.items():
        out_path = out_dir / f"{split}.csv"
        if out_path.exists():
            continue
        try:
            urllib.request.urlretrieve(url, out_path)
            with out_path.open(encoding="utf-8") as f:
                n = sum(1 for _ in f) - 1
            logger.info("  ✓ ED %s: ~%d rows → %s", split, n, out_path)
        except Exception as exc:
            logger.error("  ✗ ED GitHub download failed for %s: %s", split, exc)
            success = False
    return success


# ============================================================
# Dataset integrity check
# ============================================================

def verify_all() -> None:
    """
    Print a summary of what's present and what's missing.
    Run this at any time to check data readiness before benchmarking.
    """
    checks = {
        "MELD train":        PATHS.meld_train,
        "MELD dev":          PATHS.meld_dev,
        "MELD test":         PATHS.meld_test,
        "ED train":          PATHS.ed_train,
        "ED valid":          PATHS.ed_valid,
        "ED test":           PATHS.ed_test,
    }
    print("\nDataset readiness check:")
    print("─" * 45)
    all_ok = True
    for label, path in checks.items():
        status = "✓" if path.exists() else "✗ MISSING"
        if not path.exists():
            all_ok = False
        size = f"({path.stat().st_size // 1024}KB)" if path.exists() else ""
        print(f"  {status:<12} {label:<25} {size}")
    print("─" * 45)
    if all_ok:
        print("  All datasets ready for benchmarking.\n")
    else:
        print("  Run `python prepare_datasets.py --all` to download missing datasets.\n")


# ============================================================
# CLI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and preprocess SERAPH benchmark datasets."
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Prepare all datasets",
    )
    parser.add_argument(
        "--dataset",
        choices=["meld", "empathetic_dialogues"],
        help="Prepare a specific dataset",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Check which datasets are present without downloading",
    )
    args = parser.parse_args()

    if args.verify:
        verify_all()
        return

    if not args.all and not args.dataset:
        parser.print_help()
        print("\nTip: run with --verify to check current dataset status.")
        sys.exit(1)

    results = {}

    if args.all or args.dataset == "meld":
        results["meld"] = prepare_meld()

    if args.all or args.dataset == "empathetic_dialogues":
        results["empathetic_dialogues"] = prepare_empathetic_dialogues()

    print()
    verify_all()

    failed = [k for k, v in results.items() if not v]
    if failed:
        logger.warning("Some datasets failed to prepare: %s", failed)
        sys.exit(1)


if __name__ == "__main__":
    main()
