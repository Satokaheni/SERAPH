"""
SERAPH Dataset Preparation
============================
Downloads (where possible) and preprocesses all three benchmark datasets
into the format expected by the SERAPH benchmark evaluators.

Usage:
    python prepare_datasets.py --all
    python prepare_datasets.py --dataset meld
    python prepare_datasets.py --dataset empathetic_dialogues
    python prepare_datasets.py --dataset iemocap --iemocap-raw data/raw/IEMOCAP_full_release

IEMOCAP NOTE:
    IEMOCAP requires a manual license request from USC SAIL.
    Request access at: https://sail.usc.edu/iemocap/
    Once downloaded, place IEMOCAP_full_release/ at data/raw/IEMOCAP_full_release/
    then run with --dataset iemocap.

MELD and EmpatheticDialogues are freely downloadable — this script handles
both automatically.

Output locations:
    data/iemocap/iemocap_processed.json
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
# IEMOCAP
# ============================================================

IEMOCAP_EMOTION_MAP = {
    "ang": "anger",
    "hap": "happiness",
    "exc": "excited",
    "sad": "sadness",
    "neu": "neutral",
    "fru": "frustrated",
    "fea": "fearful",
    "sur": "surprised",
    "dis": "disgusted",
    "oth": None,   # skip
    "xxx": None,   # skip
}


def prepare_iemocap(raw_dir: Optional[Path] = None) -> bool:
    """
    Preprocess IEMOCAP from the IEMOCAP_full_release directory.

    Parses per-session EmoEvaluation label files and transcription files,
    producing a flat JSON list at data/iemocap/iemocap_processed.json.

    Args:
        raw_dir: Path to IEMOCAP_full_release/ (defaults to data/raw/IEMOCAP_full_release/).

    Returns:
        True if processing succeeded.
    """
    raw_dir = raw_dir or (PATHS.data_raw / "IEMOCAP_full_release")
    out_dir = PATHS.iemocap_data.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = PATHS.iemocap_data

    if out_path.exists():
        with out_path.open(encoding="utf-8") as f:
            existing = json.load(f)
        logger.info(
            "IEMOCAP already processed (%d utterances) at %s — skipping.",
            len(existing), out_path,
        )
        return True

    if not raw_dir.exists():
        logger.error(
            "IEMOCAP raw directory not found at %s.\n"
            "  1. Request access at https://sail.usc.edu/iemocap/\n"
            "  2. Download and extract IEMOCAP_full_release/\n"
            "  3. Place it at %s\n"
            "  4. Re-run: python prepare_datasets.py --dataset iemocap",
            raw_dir, raw_dir,
        )
        return False

    logger.info("Processing IEMOCAP from %s …", raw_dir)
    records = []
    skipped_no_text = 0
    skipped_other_label = 0

    for session_dir in sorted(raw_dir.glob("Session*")):
        session_id = int(re.search(r"\d+", session_dir.name).group())

        label_dir = session_dir / "dialog" / "EmoEvaluation"
        trans_dir = session_dir / "dialog" / "transcriptions"

        if not label_dir.exists():
            logger.warning("No EmoEvaluation dir in %s — skipping session.", session_dir)
            continue

        for label_file in sorted(label_dir.glob("*.txt")):
            dialogue_id = label_file.stem

            # ---- Load transcriptions ----
            transcriptions: dict[str, str] = {}
            trans_file = trans_dir / f"{dialogue_id}.txt"
            if trans_file.exists():
                for line in trans_file.read_text(encoding="utf-8", errors="ignore").splitlines():
                    # Format: Ses01F_impro01_F000 [start-end]: transcription text
                    m = re.match(r"^(\S+)\s+\[\d+\.\d+-\d+\.\d+\]:\s+(.+)$", line)
                    if m:
                        transcriptions[m.group(1)] = m.group(2).strip()

            # ---- Parse emotion label lines ----
            # Format: [start - end]\tutterance_id\temotion\t[V, A, D]; ...
            for line in label_file.read_text(encoding="utf-8", errors="ignore").splitlines():
                if not line.startswith("["):
                    continue
                parts = line.split("\t")
                if len(parts) < 3:
                    continue

                utt_id      = parts[1].strip()
                emotion_raw = parts[2].strip().lower()
                emotion     = IEMOCAP_EMOTION_MAP.get(emotion_raw)

                if emotion is None:
                    skipped_other_label += 1
                    continue

                utterance = transcriptions.get(utt_id, "")
                if not utterance:
                    skipped_no_text += 1
                    continue

                # Parse VAD scores if present: [V, A, D]
                vad = {}
                vad_match = re.search(r"\[([0-9.\-]+),\s*([0-9.\-]+),\s*([0-9.\-]+)\]", line)
                if vad_match:
                    vad = {
                        "valence":   float(vad_match.group(1)),
                        "activation": float(vad_match.group(2)),
                        "dominance": float(vad_match.group(3)),
                    }

                # Determine speaker from utterance ID suffix
                speaker = "M" if re.search(r"_M\d+$", utt_id) else "F"

                records.append({
                    "utterance":    utterance,
                    "emotion":      emotion,
                    "speaker":      speaker,
                    "session":      session_id,
                    "dialogue_id":  dialogue_id,
                    "utterance_id": utt_id,
                    "vad":          vad,
                })

    logger.info(
        "IEMOCAP: extracted %d utterances "
        "(skipped %d other-label, %d no-transcription)",
        len(records), skipped_other_label, skipped_no_text,
    )

    if not records:
        logger.error("No records extracted — check raw directory structure.")
        return False

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    logger.info("IEMOCAP saved to %s", out_path)
    _print_iemocap_stats(records)
    return True


def _print_iemocap_stats(records: list[dict]) -> None:
    from collections import Counter
    counts = Counter(r["emotion"] for r in records)
    logger.info("IEMOCAP label distribution:")
    for emotion, count in sorted(counts.items(), key=lambda x: -x[1]):
        logger.info("  %-15s %d", emotion, count)


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
        "IEMOCAP processed": PATHS.iemocap_data,
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
        print("  NOTE: IEMOCAP requires manual download — see script header for instructions.\n")


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
        choices=["iemocap", "meld", "empathetic_dialogues"],
        help="Prepare a specific dataset",
    )
    parser.add_argument(
        "--iemocap-raw",
        type=Path,
        default=None,
        help="Path to IEMOCAP_full_release/ directory (default: data/raw/IEMOCAP_full_release/)",
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

    if args.all or args.dataset == "iemocap":
        results["iemocap"] = prepare_iemocap(raw_dir=args.iemocap_raw)

    print()
    verify_all()

    failed = [k for k, v in results.items() if not v]
    if failed:
        logger.warning("Some datasets failed to prepare: %s", failed)
        sys.exit(1)


if __name__ == "__main__":
    main()
