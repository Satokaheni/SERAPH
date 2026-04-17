"""
prepare_human_eval_inputs.py
=============================
Converts existing JSONL files into the three per-system JSONL files
that build_human_eval.py expects:

  results/seraph_full.jsonl    — SERAPH Full responses
  results/cot_empathy.jsonl    — CoT Empathy responses
  results/moel.jsonl           — MoEL responses (strongest supervised baseline)

Source: aligned_data_4way.jsonl (400 shared contexts with all four systems)

Each output file has one record per line:
  {"context": "...", "response": "..."}

Systems chosen for human eval:
  - SERAPH Full   — the full pipeline (primary system)
  - CoT Empathy   — the matched zero-shot baseline (key comparison)
  - MoEL          — strongest supervised baseline (situates in prior work)

MoEL is preferred over MIME because MoEL has a slightly higher overall
empathy score (2.21 vs 2.15) and is more widely cited.

Usage:
    python prepare_human_eval_inputs.py
    python prepare_human_eval_inputs.py --input aligned_data_4way.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare human eval input files")
    parser.add_argument("--input",  default="aligned_data_4way.jsonl")
    parser.add_argument("--outdir", default="results")
    args = parser.parse_args()

    records = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Loaded {len(records)} records from {args.input}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    systems = {
        "seraph_full.jsonl":  "seraph",
        "cot_empathy.jsonl":  "cot",
        "moel.jsonl":         "moel",
    }

    for filename, field in systems.items():
        out_path = outdir / filename
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in records:
                response = rec.get(field, "")
                if not response:
                    continue
                f.write(json.dumps({
                    "context":  rec["context"],
                    "response": response,
                    "emotion":  rec.get("emotion", "unknown"),
                }, ensure_ascii=False) + "\n")
        print(f"  Wrote {len(records)} records → {out_path}")

    print(f"\nNow run:")
    print(f"  python build_human_eval.py \\")
    print(f"      --seraph results/seraph_full.jsonl \\")
    print(f"      --cot    results/cot_empathy.jsonl \\")
    print(f"      --raw    results/moel.jsonl \\")
    print(f"      --out    human_eval/")
    print()
    print("Note: --raw is used for the third system slot.")
    print("The system will be labeled 'Raw Claude' in the booklet.")
    print("Rename it in build_human_eval.py if you want 'MoEL' instead.")


if __name__ == "__main__":
    main()
