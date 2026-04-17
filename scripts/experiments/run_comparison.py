# run_comparison.py
"""
3-way comparison runner: SERAPH vs MoEL vs MIME
Feeds 400 contexts from comparison_sample.json through the batch wave engine,
then writes aligned_data.jsonl ready for score_empathy.py --jsonl

Usage:
    python run_comparison.py
    python run_comparison.py --input comparison_sample.json --output aligned_data.jsonl
    python run_comparison.py --checkpoint-dir .checkpoints/comparison  # resume after crash
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
import logging
import time
from pathlib import Path

import anthropic

from config import ANTHROPIC_API_KEY
from benchmarks.batch_runner import (
    WaveState,
    _run_wave1, _run_wave2, _run_wave3, _run_wave4, _run_wave5,
)

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("seraph.comparison")


def run_comparison(
    input_path: str = "comparison_sample.json",
    output_path: str = "aligned_data.jsonl",
    checkpoint_dir: str = ".checkpoints/comparison",
    ablation: str = "full",
) -> None:
    # ------------------------------------------------------------------ load
    sample = json.loads(Path(input_path).read_text(encoding="utf-8"))
    logger.info("Loaded %d records from %s", len(sample), input_path)

    # ------------------------------------------------------------------ build WaveStates
    # WaveState needs: sample_id, utterance, gt_emotion
    # gt_emotion isn't used for scoring here, so we use a placeholder.
    states: dict[str, WaveState] = {}
    moel_map: dict[str, str] = {}
    mime_map: dict[str, str] = {}

    for i, rec in enumerate(sample):
        sid = f"cmp_{i:04d}"
        states[sid] = WaveState(
            sample_id=sid,
            utterance=rec["context"],
            gt_emotion="n/a",   # not needed — no benchmark scoring
        )
        moel_map[sid] = rec["moel"]
        mime_map[sid] = rec["mime"]

    # ------------------------------------------------------------------ run waves
    cp_dir = Path(checkpoint_dir)
    cp_dir.mkdir(parents=True, exist_ok=True)
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    t0 = time.time()
    logger.info("=== Wave 1: Stage 1 (Emotion Recognizer) ===")
    _run_wave1(states, client, ablation, cp_dir)

    logger.info("=== Wave 2: Stage 2 (Classifier) ===")
    _run_wave2(states, client, ablation, cp_dir)

    logger.info("=== Wave 3: Stage 3 (Self-Simulation) ===")
    _run_wave3(states, client, ablation, cp_dir)

    logger.info("=== Wave 4: Stage 4 (Ethical Gate) ===")
    _run_wave4(states, client, ablation, cp_dir)

    logger.info("=== Wave 5: Stage 5 (Response Generator) ===")
    _run_wave5(states, client, ablation, cp_dir)

    elapsed = (time.time() - t0) / 60
    logger.info("All waves complete in %.1f minutes", elapsed)

    # ------------------------------------------------------------------ write output
    errors = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for i, (sid, state) in enumerate(states.items()):
            seraph = ""
            if state.stage5_data:
                seraph = state.stage5_data.get("response", "")
            if not seraph:
                seraph = f"ERROR: {state.error or 'no response'}"
                errors += 1

            out.write(json.dumps({
                "idx":     i,
                "context": state.utterance,
                "seraph":  seraph,
                "moel":    moel_map[sid],
                "mime":    mime_map[sid],
            }, ensure_ascii=False) + "\n")

    total = len(states)
    logger.info("Wrote %d records to %s (%d errors)", total, output_path, errors)

    # ------------------------------------------------------------------ sanity check
    print(f"\n{'='*60}")
    print(f"  Comparison run complete")
    print(f"  Total records : {total}")
    print(f"  Errors        : {errors}")
    print(f"  Clean records : {total - errors}")
    print(f"  Output        : {output_path}")
    print(f"  Checkpoints   : {checkpoint_dir}")
    print(f"{'='*60}")
    with open(output_path, encoding="utf-8") as f:
        first = json.loads(f.readline())
    print(f"\n  Sample output (record 0):")
    print(f"  context : {first['context'][:80]}")
    print(f"  seraph  : {first['seraph'][:80]}")
    print(f"  moel    : {first['moel'][:80]}")
    print(f"  mime    : {first['mime'][:80]}")
    print(f"\nNext step: python score_empathy.py --jsonl {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SERAPH 3-way comparison runner")
    parser.add_argument("--input",          default="comparison_sample.json")
    parser.add_argument("--output",         default="aligned_data.jsonl")
    parser.add_argument("--checkpoint-dir", default=".checkpoints/comparison")
    parser.add_argument("--ablation",       default="full",
                        help="Ablation variant (default: full)")
    args = parser.parse_args()

    run_comparison(
        input_path=args.input,
        output_path=args.output,
        checkpoint_dir=args.checkpoint_dir,
        ablation=args.ablation,
    )


if __name__ == "__main__":
    main()