"""
CoT Empathy Baseline — Batch API
==================================
Single-wave batch run of the CoT Empathy condition against the 400
shared contexts from comparison_sample.json.

Reuses submit_batch / poll_until_complete / checkpoint helpers from
benchmarks/batch_runner.py. One wave, 400 requests, ~1-4 hrs, 50% discount.

Prompt:
  "Before responding, take a moment to think about what this person is
   feeling, what they need, and how your response will land. Then respond."

No Plutchik, no BDI, no Gross, no simulation framing — just generic CoT.

Usage:
    python run_cot_baseline.py
    python run_cot_baseline.py --input comparison_sample.json
    python run_cot_baseline.py --output aligned_data_4way.jsonl

Next step after completion:
    python score_empathy.py --jsonl aligned_data_4way.jsonl
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import anthropic

from config import ANTHROPIC_API_KEY, SONNET_MODEL, MAX_TOKENS, TEMPERATURE, USE_PROMPT_CACHING
from benchmarks.batch_runner import (
    submit_batch,
    poll_until_complete,
    _checkpoint_path,
    _load_checkpoint,
    _save_checkpoint,
)

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("cot_baseline")

# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a warm, empathic conversational assistant. "
    "Your goal is to respond to people in a way that makes them feel heard and understood. "
    "You must always respond with a JSON object in this exact format, with no markdown fences: "
    '{"response": "<your empathic response here>"}'
)

COT_INSTRUCTION = (
    "Before responding, take a moment to think about what this person is feeling, "
    "what they need, and how your response will land. Then respond. "
    'Return only a JSON object: {"response": "<your response>"}'
)

# Wave number — 99 avoids colliding with SERAPH's waves 1-5
COT_WAVE = 99


# ── Request builder ───────────────────────────────────────────────────────────

def _build_cot_request(idx: int, context: str) -> dict:
    system = (
        [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]
        if USE_PROMPT_CACHING else SYSTEM_PROMPT
    )
    return {
        "custom_id": f"cot_{idx:04d}",
        "params": {
            "model": SONNET_MODEL,
            "max_tokens": MAX_TOKENS.get("stage5", 512),
            "temperature": TEMPERATURE.get("stage5", 0.7),
            "system": system,
            "messages": [{
                "role": "user",
                "content": f"{COT_INSTRUCTION}\n\nPerson: {context}",
            }],
        },
    }


# ── Wave runner ───────────────────────────────────────────────────────────────

def run_cot_wave(
    sample: list[dict],
    client: anthropic.Anthropic,
    checkpoint_dir: Optional[Path],
) -> dict[int, str]:
    """
    Submit all CoT requests as a single batch wave.
    Returns dict of {idx: response_text}.

    The model is instructed to return {"response": "..."} JSON so that
    poll_until_complete parses it cleanly — no raw text handling needed.
    """
    cp = _checkpoint_path(checkpoint_dir, COT_WAVE)
    cached = _load_checkpoint(cp)

    if cached:
        logger.info("Using cached CoT wave results")
        results = cached
    else:
        requests = [_build_cot_request(i, rec["context"]) for i, rec in enumerate(sample)]
        batch_id = submit_batch(requests, client)
        results = poll_until_complete(batch_id, client)
        _save_checkpoint(cp, results)

    cot_map: dict[int, str] = {}
    errors = 0
    for custom_id, result in results.items():
        idx = int(custom_id.replace("cot_", ""))
        if not result.get("ok"):
            logger.warning("Record %d failed: %s", idx, result.get("error"))
            cot_map[idx] = f"ERROR: {result.get('error', 'unknown')}"
            errors += 1
            continue

        # JSON parsed cleanly — extract response field
        data = result.get("data", {})
        response = data.get("response", "").strip()
        if not response:
            logger.warning("Record %d: empty response field, data=%s", idx, str(data)[:100])
            cot_map[idx] = f"ERROR: empty response"
            errors += 1
        else:
            cot_map[idx] = response

    logger.info("CoT wave complete — %d responses, %d errors", len(cot_map), errors)
    return cot_map


# ── Save intermediate CoT outputs ─────────────────────────────────────────────

def save_cot_outputs(sample: list[dict], cot_map: dict[int, str], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for i, rec in enumerate(sample):
            f.write(json.dumps({
                "idx":     i,
                "context": rec["context"],
                "cot":     cot_map.get(i, "ERROR: missing"),
            }, ensure_ascii=False) + "\n")
    logger.info("CoT outputs saved → %s", path)


# ── Build 4-way aligned JSONL ─────────────────────────────────────────────────

def build_4way_jsonl(
    sample: list[dict],
    seraph_path: str,
    cot_map: dict[int, str],
    output_path: str,
) -> None:
    """
    Join CoT with existing SERAPH/MoEL/MIME aligned_data.jsonl
    into a 4-way file for score_empathy.py.
    """
    seraph_map: dict[int, str] = {}
    with open(seraph_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            seraph_map[rec["idx"]] = rec["seraph"]

    skipped = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for i, rec in enumerate(sample):
            seraph = seraph_map.get(i, "")
            cot    = cot_map.get(i, "")

            if not seraph or seraph.startswith("ERROR"):
                skipped += 1
                continue
            if not cot or cot.startswith("ERROR"):
                skipped += 1
                continue

            out.write(json.dumps({
                "idx":     i,
                "context": rec["context"],
                "seraph":  seraph,
                "cot":     cot,
                "moel":    rec["moel"],
                "mime":    rec["mime"],
            }, ensure_ascii=False) + "\n")

    total = len(sample) - skipped
    logger.info("4-way JSONL: %d records (%d skipped) → %s", total, skipped, output_path)


# ── Sanity check ──────────────────────────────────────────────────────────────

def print_summary(output_path: str) -> None:
    with open(output_path, encoding="utf-8") as f:
        lines = [json.loads(l) for l in f if l.strip()]
    rec = lines[0]
    print(f"\n{'='*60}")
    print(f"  4-way comparison complete — {len(lines)} records")
    print(f"{'='*60}")
    print(f"  context : {rec['context'][:80]}")
    print(f"  seraph  : {rec['seraph'][:80]}")
    print(f"  cot     : {rec['cot'][:80]}")
    print(f"  moel    : {rec['moel'][:80]}")
    print(f"  mime    : {rec['mime'][:80]}")
    print(f"{'='*60}")
    print(f"\nNext step: python score_empathy.py --jsonl {output_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="CoT Empathy baseline — batch API")
    parser.add_argument("--input",          default="comparison_sample.json")
    parser.add_argument("--seraph",         default="aligned_data.jsonl",
                        help="Existing SERAPH/MoEL/MIME aligned file")
    parser.add_argument("--cot-output",     default="cot_outputs.jsonl",
                        help="Intermediate CoT outputs")
    parser.add_argument("--output",         default="aligned_data_4way.jsonl",
                        help="4-way aligned file for scoring")
    parser.add_argument("--checkpoint-dir", default=".checkpoints/cot",
                        help="Checkpoint directory")
    args = parser.parse_args()

    sample = json.loads(Path(args.input).read_text(encoding="utf-8"))
    logger.info("Loaded %d records from %s", len(sample), args.input)

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    cp_dir = Path(args.checkpoint_dir)
    cp_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== CoT Empathy wave (%d requests) ===", len(sample))
    cot_map = run_cot_wave(sample, client, cp_dir)

    save_cot_outputs(sample, cot_map, args.cot_output)

    logger.info("=== Building 4-way aligned file ===")
    build_4way_jsonl(sample, args.seraph, cot_map, args.output)

    print_summary(args.output)


if __name__ == "__main__":
    main()
