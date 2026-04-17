"""
Experiment 2 — CoT Empathy on Full EmpatheticDialogues Test Set
================================================================
Runs CoT Empathy on the full EmpatheticDialogues test set so it can
be added as a proper row in the main results table alongside SERAPH
Full and w/o Stage 3.

Uses the Batch API (single wave) — same approach as run_cot_baseline.py
but reads from the full ED test set rather than the 400-context
comparison sample.

Expects the full test set in the same format used by the ED benchmark
evaluator — a JSONL or JSON file with 'utterance' fields, or a plain
.txt file with one context per line.

Output:
  cot_ed_full_outputs.jsonl     — CoT responses for full test set
  results/cot_ed_full_scores.csv — scored results (via score_empathy judge)

Usage:
    python run_cot_ed_full.py
    python run_cot_ed_full.py --test-set data/empathetic_dialogues_test.jsonl
    python run_cot_ed_full.py --test-set results/empathetic_dialogues_full.json
    (reads SERAPH's own full results file to extract contexts if needed)
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
import logging
import os
import time
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
logger = logging.getLogger("cot_ed_full")

# ── Prompt (identical to run_cot_baseline.py) ─────────────────────────────────

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

COT_WAVE = 98  # distinct from run_cot_baseline.py's wave 99


# ── Context loader ────────────────────────────────────────────────────────────

def load_contexts(test_set_path: str) -> list[dict]:
    """
    Load contexts from the full ED test set.
    Supports:
      - SERAPH full results JSON: {metrics, human_baseline, results: [{utterance, ...}]}
      - JSONL: one record per line with 'utterance', 'context', or 'input_text'
      - Plain .txt: one context per line
    Returns list of dicts with 'idx' and 'context' keys.
    Skips records with errors (success=False) to match SERAPH's evaluated set.
    """
    path = Path(test_set_path)
    if not path.exists():
        raise FileNotFoundError(f"Test set not found: {test_set_path}")

    records = []

    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))

        # Handle {metrics, results: [...]} structure from empathetic_dialogues_full.json
        if isinstance(data, dict) and "results" in data:
            items = data["results"]
        elif isinstance(data, list):
            items = data
        else:
            items = []

        for i, item in enumerate(items):
            # Skip errored records to stay consistent with SERAPH's evaluated set
            if not item.get("success", True) or item.get("error"):
                continue
            ctx = (item.get("utterance") or item.get("input_text")
                   or item.get("context") or "")
            if ctx:
                records.append({"idx": i, "context": ctx.strip()})

    elif path.suffix == ".jsonl":
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if not item.get("success", True) or item.get("error"):
                    continue
                ctx = (item.get("utterance") or item.get("context")
                       or item.get("input_text") or "")
                if ctx:
                    records.append({"idx": i, "context": ctx.strip()})

    else:
        # Plain text — one context per line
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line:
                    records.append({"idx": i, "context": line})

    logger.info("Loaded %d contexts from %s", len(records), test_set_path)
    return records


# ── Request builder ───────────────────────────────────────────────────────────

def _build_cot_request(idx: int, context: str) -> dict:
    system = (
        [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]
        if USE_PROMPT_CACHING else SYSTEM_PROMPT
    )
    return {
        "custom_id": f"coted_{idx:05d}",
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


# ── Batch wave ────────────────────────────────────────────────────────────────

def run_cot_wave(
    records: list[dict],
    client: anthropic.Anthropic,
    checkpoint_dir: Path,
) -> dict[int, str]:
    """Single batch wave over all contexts. Returns {idx: response_text}."""
    cp = _checkpoint_path(checkpoint_dir, COT_WAVE)
    cached = _load_checkpoint(cp)

    if cached:
        logger.info("Using cached wave results (%d records)", len(cached))
        results = cached
    else:
        requests = [_build_cot_request(r["idx"], r["context"]) for r in records]
        logger.info("Submitting %d requests to Batch API…", len(requests))
        batch_id = submit_batch(requests, client)
        results  = poll_until_complete(batch_id, client)
        _save_checkpoint(cp, results)

    cot_map: dict[int, str] = {}
    errors = 0
    for custom_id, result in results.items():
        idx = int(custom_id.replace("coted_", ""))
        if not result.get("ok"):
            logger.warning("Record %d failed: %s", idx, result.get("error"))
            cot_map[idx] = f"ERROR: {result.get('error', 'unknown')}"
            errors += 1
            continue
        data     = result.get("data", {})
        raw      = result.get("raw", "")
        response = data.get("response", raw).strip() if data else raw.strip()
        if not response:
            cot_map[idx] = "ERROR: empty response"
            errors += 1
        else:
            cot_map[idx] = response

    logger.info("Wave complete — %d responses, %d errors", len(cot_map), errors)
    return cot_map


# ── Save outputs ──────────────────────────────────────────────────────────────

def save_outputs(records: list[dict], cot_map: dict[int, str], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            i = rec["idx"]
            f.write(json.dumps({
                "idx":     i,
                "context": rec["context"],
                "cot":     cot_map.get(i, "ERROR: missing"),
            }, ensure_ascii=False) + "\n")
    logger.info("Outputs saved → %s", path)


# ── Score with LLM-as-judge ───────────────────────────────────────────────────

JUDGE_TEMPLATE = """Rate this empathic dialogue response on 5 dimensions (1-5 scale).

Scoring guide:
1 = Poor  2 = Below average  3 = Adequate  4 = Good  5 = Excellent

Definitions:
- emotional_acknowledgment: Does the response explicitly acknowledge the speaker's emotional state?
- perspective_taking: Does it demonstrate genuine first-person understanding of how the speaker feels?
- tone_match: Does the tone match the speaker's emotional register?
- avoids_harm: Does it avoid minimising, victim-blaming, or dysregulating content?
- meets_core_need: Does it address what the speaker most needs right now?

Return ONLY a JSON object with these exact keys (no markdown, no explanation):
{"emotional_acknowledgment": X, "perspective_taking": X, "tone_match": X, "avoids_harm": X, "meets_core_need": X}

where X is an integer 1-5.

Dialogue context:
CONTEXT_PLACEHOLDER

System response:
RESPONSE_PLACEHOLDER"""


def score_outputs(
    records: list[dict],
    cot_map: dict[int, str],
    output_csv: str,
    api_key: str,
) -> None:
    """Score all CoT outputs with the same LLM-as-judge used in score_empathy.py."""
    import pandas as pd
    import re

    client = anthropic.Anthropic(api_key=api_key)
    dims   = ["emotional_acknowledgment", "perspective_taking",
              "tone_match", "avoids_harm", "meets_core_need"]
    rows   = []
    errors = 0

    logger.info("Scoring %d records with LLM-as-judge…", len(records))
    for i, rec in enumerate(records):
        ctx      = rec["context"]
        response = cot_map.get(rec["idx"], "")

        if not response or response.startswith("ERROR"):
            logger.warning("Skipping record %d — no valid CoT response", rec["idx"])
            continue

        prompt = JUDGE_TEMPLATE.replace("CONTEXT_PLACEHOLDER", ctx.strip()) \
                               .replace("RESPONSE_PLACEHOLDER", response.strip())
        scores = None
        for attempt in range(3):
            try:
                result = client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=256,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = result.content[0].text.strip()
                text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
                text = re.sub(r'\s*```$', '', text, flags=re.MULTILINE)
                scores = json.loads(text.strip())
                assert all(k in scores for k in dims)
                assert all(1 <= scores[k] <= 5 for k in dims)
                scores["overall"] = round(sum(scores[k] for k in dims) / len(dims), 4)
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    logger.warning("Record %d scoring failed: %s", rec["idx"], e)
                    errors += 1
                    scores = {d: float("nan") for d in dims + ["overall"]}

        rows.append({
            "system":     "CoT Empathy",
            "sample_idx": rec["idx"],
            "context":    ctx[:200],
            "response":   response[:200],
            **scores,
        })

        time.sleep(0.5)  # rate limit

        if (i + 1) % 50 == 0 or (i + 1) == len(records):
            logger.info("Scored %d/%d (%d errors)", i + 1, len(records), errors)

    df = pd.DataFrame(rows)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    means = df[dims + ["overall"]].mean().round(4)
    print("\n" + "=" * 60)
    print("  CoT Empathy — Full ED Test Set Results")
    print("=" * 60)
    for dim, val in means.items():
        print(f"  {dim:<30} {val:.4f}")
    print(f"\n  Saved {len(df)} rows → {output_csv}")
    print(f"  Errors: {errors}")
    print("\n  Add this row to ed_main_results.tex:")
    vals = " & ".join(f"{means[d]:.4f}" for d in dims + ["overall"])
    print(f"  CoT Empathy & {vals} \\\\")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CoT Empathy on full ED test set — batch API"
    )
    parser.add_argument(
        "--test-set",
        default="results/empathetic_dialogues_full.json",
        help="Full ED test set (SERAPH full results JSON, JSONL, or plain .txt)",
    )
    parser.add_argument("--cot-output",     default="cot_ed_full_outputs.jsonl")
    parser.add_argument("--scores-output",  default="results/cot_ed_full_scores.csv")
    parser.add_argument("--checkpoint-dir", default=".checkpoints/cot_ed_full")
    parser.add_argument("--skip-scoring",   action="store_true",
                        help="Only run generation, skip LLM-as-judge scoring")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv(".env")
    api_key = os.environ.get("ANTHROPIC_API_KEY") or ANTHROPIC_API_KEY
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    records    = load_contexts(args.test_set)
    client     = anthropic.Anthropic(api_key=api_key)
    cp_dir     = Path(args.checkpoint_dir)
    cp_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== CoT Empathy wave (%d contexts) ===", len(records))
    cot_map = run_cot_wave(records, client, cp_dir)

    save_outputs(records, cot_map, args.cot_output)

    if not args.skip_scoring:
        logger.info("=== Scoring with LLM-as-judge ===")
        score_outputs(records, cot_map, args.scores_output, api_key)
    else:
        logger.info("Skipping scoring (--skip-scoring set). "
                    "Run score_empathy.py manually on %s", args.cot_output)


if __name__ == "__main__":
    main()
